//! # bitnet-inference
//!
//! High-level inference engine for BitNet b1.58.
//!
//! ## Architecture
//!
//! This crate sits above `bitnet-model` and `bitnet-tokenizer` and provides:
//!
//! - [`InferenceEngine`]: loads weights, owns the model and tokenizer, drives
//!   the autoregressive generation loop.
//! - [`SamplingConfig`]: hyperparameters controlling token selection.
//! - [`ChatPipeline`]: stateful multi-turn conversation manager.
//! - [`sample_next_token`]: the token sampling kernel (temperature, top-k, top-p,
//!   repetition penalty).
//!
//! ## Generation Loop
//!
//! ```text
//! prefill:  model.forward(prompt_tokens, 0, &mut kv_cache)
//! decode:   loop {
//!               logits = model.forward(&[last_token], kv_cache.filled_positions, &mut kv_cache)
//!               next   = sample_next_token(&mut logits, &config, &generated_so_far, &mut buffers)
//!               if next == EOS { break }
//!               generated.push(next)
//!           }
//! ```
//!
//! ## Sampling Mathematics
//!
//! Given raw logits **z** ∈ ℝ^V:
//!
//! 1. **Repetition penalty** (α ≥ 1):
//!    `z[t] /= α` for each token `t` that appeared in the context.
//!
//! 2. **Temperature** (τ > 0):
//!    `z[t] /= τ` for all t.  τ → 0 approaches greedy argmax.
//!
//! 3. **Top-k** (k > 0):
//!    Set `z[t] = −∞` for all t not in the k highest logits.
//!
//! 4. **Top-p** (p ∈ (0, 1]):
//!    Sort tokens by probability descending; keep the smallest prefix whose
//!    cumulative probability ≥ p; set all others to −∞.
//!
//! 5. **Softmax** + **categorical sample** from the resulting distribution.
//!
//! # Module Layout
//!
//! ```text
//! bitnet-inference/
//! └── lib.rs   ← this file: InferenceEngine, SamplingConfig,
//!                           sample_next_token, ChatPipeline
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

use anyhow::{anyhow, Context};
use std::path::{Path, PathBuf};
use tracing::{debug, info, instrument, warn};

#[cfg(test)]
use serde_json::Value;

use bitnet_convert::{convert_hf_packed_to_canonical, load_hf_packed_model};
use bitnet_core::backend::Device;
use bitnet_core::config::{bitnet_2b_config, ModelConfig};
use bitnet_model::{BitNetModel, KVCache};
use bitnet_tokenizer::{ChatMessage, Tokenizer};
use bitnet_weights::config::load_model_config;
use bitnet_weights::loader::load_weights_from_bf16;
use bitnet_weights::safetensors::load_safetensors_meta;
use bitnet_weights::CONFIG_FILENAME;

// ---------------------------------------------------------------------------
// SamplingConfig
// ---------------------------------------------------------------------------

/// Hyperparameters controlling autoregressive token sampling.
///
/// # Mathematical Specification
///
/// Applied in order: repetition_penalty → temperature → top_k → top_p → sample.
///
/// See module documentation for the full mathematical derivation.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingConfig {
    /// Softmax temperature τ > 0.
    ///
    /// `τ = 1.0` → no change.  `τ < 1.0` → sharper distribution (more
    /// deterministic).  `τ → 0` → greedy.
    pub temperature: f32,

    /// Nucleus (top-p) threshold p ∈ (0, 1].
    ///
    /// `p = 1.0` → disabled.
    pub top_p: f32,

    /// Top-k vocabulary truncation.  `0` → disabled.
    pub top_k: usize,

    /// Repetition penalty α ≥ 1.0.  `1.0` → no penalty.
    pub repetition_penalty: f32,

    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,

    /// PRNG seed for reproducible sampling.
    pub seed: u64,
}

impl SamplingConfig {
    /// Greedy decoding: always pick the highest-probability token.
    pub fn greedy() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 1.0,
            max_new_tokens: 256,
            seed: 42,
        }
    }

    /// Balanced defaults for chat / instruction following.
    pub fn chat_defaults() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            max_new_tokens: 512,
            seed: 0,
        }
    }

    /// Creative / high-diversity sampling.
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 0,
            repetition_penalty: 1.05,
            max_new_tokens: 1024,
            seed: 0,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.temperature <= 0.0 {
            return Err(anyhow!("temperature must be > 0, got {}", self.temperature));
        }
        if !(0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(anyhow!("top_p must be in (0, 1], got {}", self.top_p));
        }
        if self.repetition_penalty < 1.0 {
            return Err(anyhow!(
                "repetition_penalty must be >= 1.0, got {}",
                self.repetition_penalty
            ));
        }
        if self.max_new_tokens == 0 {
            return Err(anyhow!("max_new_tokens must be > 0"));
        }
        Ok(())
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self::chat_defaults()
    }
}

// ---------------------------------------------------------------------------
// SamplingBuffers
// ---------------------------------------------------------------------------

/// Pre-allocated scratch buffers for token sampling.
///
/// Eliminates ~2-3 MiB of heap allocation per generated token by reusing
/// vocab-sized buffers across sampling calls.
pub struct SamplingBuffers {
    /// Scratch for sorted logits (top-k filtering).
    sorted: Vec<f32>,
    /// Scratch for exponentiated logits.
    exps: Vec<f32>,
    /// Scratch for probability distribution.
    probs: Vec<f32>,
    /// Scratch for sorted indices (top-p filtering).
    indices: Vec<usize>,
}

impl SamplingBuffers {
    /// Allocate sampling buffers for the given vocabulary size.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            sorted: vec![0.0; vocab_size],
            exps: vec![0.0; vocab_size],
            probs: vec![0.0; vocab_size],
            indices: (0..vocab_size).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// sample_next_token
// ---------------------------------------------------------------------------

/// Sample the next token from a logit vector using the given sampling config.
///
/// # Algorithm
///
/// 1. Apply repetition penalty: `z[t] /= α` for each `t ∈ past_tokens`.
/// 2. Apply temperature: `z[t] /= τ`.
/// 3. Apply top-k: zero out all but the top-k logits.
/// 4. Apply top-p: keep the smallest prefix with cumulative probability ≥ p.
/// 5. Softmax the remaining logits to a probability distribution.
/// 6. Sample from the distribution using a simple LCG PRNG.
///
/// # Arguments
///
/// - `logits`:       Mutable logit slice (modified in-place during processing).
/// - `config`:       Sampling hyperparameters.
/// - `past_tokens`:  All tokens generated so far (used for repetition penalty).
/// - `buffers`:      Pre-allocated scratch buffers (see [`SamplingBuffers`]).
///
/// # Returns
///
/// The sampled token ID as `u32`.
///
/// # Panics (debug only)
/// Panics if `logits` is empty.
pub fn sample_next_token(
    logits: &mut [f32],
    config: &SamplingConfig,
    past_tokens: &[u32],
    buffers: &mut SamplingBuffers,
) -> u32 {
    debug_assert!(!logits.is_empty(), "logits must not be empty");

    let vocab = logits.len();

    // ── 1. Repetition penalty ─────────────────────────────────────────────
    if config.repetition_penalty != 1.0 {
        for &tok in past_tokens {
            if (tok as usize) < vocab {
                logits[tok as usize] /= config.repetition_penalty;
            }
        }
    }

    // ── 2. Temperature ────────────────────────────────────────────────────
    if (config.temperature - 1.0).abs() > 1e-6 {
        let inv_temp = config.temperature.recip();
        for v in logits.iter_mut() {
            *v *= inv_temp;
        }
    }

    // ── 3. Greedy shortcut ────────────────────────────────────────────────
    if config.top_k == 1 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // ── 4. Top-k filtering ────────────────────────────────────────────────
    if config.top_k > 0 && config.top_k < vocab {
        // Copy logits into scratch, then use partial sort to find the k-th largest.
        buffers.sorted[..vocab].copy_from_slice(logits);
        let k = config.top_k.min(vocab);
        // select_nth_unstable_by partitions so that element at index k-1 is
        // the k-th largest when sorting descending.
        buffers.sorted[..vocab].select_nth_unstable_by(k - 1, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let threshold = buffers.sorted[k - 1];
        for v in logits.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // ── 5. Top-p (nucleus) filtering ──────────────────────────────────────
    if config.top_p < 1.0 {
        // Compute softmax probabilities using pre-allocated buffers.
        let max_logit = logits
            .iter()
            .filter(|v| v.is_finite())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp = 0.0_f32;
        for i in 0..vocab {
            let e = if logits[i].is_finite() {
                (logits[i] - max_logit).exp()
            } else {
                0.0
            };
            buffers.exps[i] = e;
            sum_exp += e;
        }

        let inv_sum = if sum_exp > 0.0 { sum_exp.recip() } else { 1.0 };
        for i in 0..vocab {
            buffers.probs[i] = buffers.exps[i] * inv_sum;
        }

        // Sort indices by probability descending.
        // Reset indices to 0..vocab (may be stale from a previous call).
        for i in 0..vocab {
            buffers.indices[i] = i;
        }
        buffers.indices[..vocab].sort_unstable_by(|&a, &b| {
            buffers.probs[b]
                .partial_cmp(&buffers.probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep the smallest prefix with cumulative prob >= top_p.
        let mut cum_prob = 0.0_f32;
        let mut cutoff_idx = vocab;
        for (rank, &idx) in buffers.indices[..vocab].iter().enumerate() {
            cum_prob += buffers.probs[idx];
            if cum_prob >= config.top_p {
                cutoff_idx = rank + 1;
                break;
            }
        }

        // Mask out tokens beyond the nucleus.
        // Repurpose exps buffer as a boolean flag array (0.0 = excluded, 1.0 = included)
        // to avoid HashSet allocation.
        for i in 0..vocab {
            buffers.exps[i] = 0.0;
        }
        for &idx in &buffers.indices[..cutoff_idx] {
            buffers.exps[idx] = 1.0;
        }
        for i in 0..vocab {
            if buffers.exps[i] == 0.0 {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }

    // ── 6. Softmax → categorical sample ──────────────────────────────────
    let max_logit = logits
        .iter()
        .filter(|v| v.is_finite())
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0_f32;
    for i in 0..vocab {
        let p = if logits[i].is_finite() {
            (logits[i] - max_logit).exp()
        } else {
            0.0
        };
        buffers.probs[i] = p;
        sum += p;
    }

    let inv_sum = if sum > 0.0 { sum.recip() } else { 1.0 };
    for i in 0..vocab {
        buffers.probs[i] *= inv_sum;
    }

    // Simple LCG PRNG for reproducible sampling.
    // LCG: x_{n+1} = (a * x_n + c) mod 2^64
    // Constants from Knuth TAOCP Vol. 2.
    let mut rng_state = config.seed.wrapping_add(past_tokens.len() as u64);
    rng_state = rng_state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);

    // Convert to uniform [0, 1).
    let rand_f: f32 = (rng_state >> 11) as f32 / (1u64 << 53) as f32;

    // Inverse CDF sampling.
    let mut cumulative = 0.0_f32;
    for i in 0..vocab {
        cumulative += buffers.probs[i];
        if rand_f <= cumulative {
            return i as u32;
        }
    }

    // Fallback: return the argmax (should rarely happen due to floating point).
    buffers.probs[..vocab]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// InferenceEngine
// ---------------------------------------------------------------------------

/// High-level inference engine for BitNet b1.58.
///
/// Owns the model, tokenizer, and KV cache.  Provides both raw generation
/// (`generate`) and chat-formatted generation (`generate_chat`).
///
/// # Usage
///
/// ```no_run
/// use bitnet_inference::{InferenceEngine, SamplingConfig};
/// use bitnet_core::backend::Device;
/// use std::path::Path;
///
/// let mut engine = InferenceEngine::new(
///     Path::new("model.safetensors"),
///     Device::cpu(),
/// ).unwrap();
///
/// let response = engine.generate("The capital of France is", &SamplingConfig::greedy()).unwrap();
/// println!("{response}");
/// ```
/// Inspect a safetensors file's metadata to determine whether it contains
/// packed U8 ternary weights (`"packed"`) or BF16 master weights (`"bf16"`).
///
/// Detection heuristic: if *any* tensor in the file has dtype `"U8"`, the file
/// is the packed deployment format from `microsoft/bitnet-b1.58-2B-4T`.
/// Otherwise it is the BF16 format from `microsoft/bitnet-b1.58-2B-4T-bf16`.
///
/// Returns `"packed"`, `"bf16"`, or `"bf16"` as a fallback on read error.
fn detect_weight_format(path: &Path) -> &'static str {
    match load_safetensors_meta(path) {
        Ok(meta) => {
            let has_u8 = meta.values().any(|m| m.dtype == "U8");
            if has_u8 {
                "packed"
            } else {
                "bf16"
            }
        }
        Err(e) => {
            warn!(error = %e, "Could not read safetensors header for format detection; assuming bf16");
            "bf16"
        }
    }
}

/// Resolve the sibling `config.json` path for a model checkpoint.
///
/// For a checkpoint at `.../model.safetensors`, this returns
/// `.../config.json`.
fn sibling_config_path(model_path: &Path) -> Option<PathBuf> {
    model_path
        .parent()
        .map(|parent| parent.join(CONFIG_FILENAME))
}

/// Resolve the model configuration for inference.
///
/// If a sibling `config.json` exists next to `model_path`, it is treated as the
/// authoritative source of model dimensions. Otherwise the canonical BitNet 2B
/// configuration is used as a fallback.
fn resolve_model_config(model_path: &Path) -> anyhow::Result<ModelConfig> {
    match sibling_config_path(model_path) {
        Some(config_path) if config_path.exists() => {
            info!(path = %config_path.display(), "Loading model config from sibling config.json");
            load_model_config(&config_path)
        }
        Some(config_path) => {
            warn!(
                path = %config_path.display(),
                "Sibling config.json not found; falling back to canonical BitNet 2B config"
            );
            Ok(bitnet_2b_config())
        }
        None => {
            warn!("Model path has no parent directory; falling back to canonical BitNet 2B config");
            Ok(bitnet_2b_config())
        }
    }
}

/// High-level inference engine for BitNet b1.58.
///
/// Encapsulates the transformer model, LLaMA 3 tokenizer, and KV cache.
/// Created via [`InferenceEngine::new`].
pub struct InferenceEngine {
    /// The transformer model.
    model: BitNetModel,
    /// LLaMA 3 BPE tokenizer.
    tokenizer: Tokenizer,
    /// KV cache (max context window).
    kv_cache: KVCache,
    /// Pre-allocated scratch buffers for token sampling.
    sampling_buffers: SamplingBuffers,
    /// Pre-allocated logits buffer for zero-copy forward pass.
    logits_buf: Vec<f32>,
}

impl std::fmt::Debug for InferenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceEngine")
            .field("backend", &self.model.backend_name())
            .field("kv_cache_filled", &self.kv_cache.filled_positions)
            .finish()
    }
}

impl InferenceEngine {
    /// Create an [`InferenceEngine`] by loading weights from a safetensors file.
    ///
    /// # Arguments
    ///
    /// - `model_path`: Path to the BF16 `.safetensors` checkpoint.
    /// - `device`:     Compute device for inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the weights cannot be loaded or the backend cannot
    /// be initialised.
    #[instrument(level = "info", skip(model_path), fields(path = %model_path.display()))]
    pub fn new(model_path: &Path, device: Device) -> anyhow::Result<Self> {
        info!(path = %model_path.display(), "Loading BitNet weights");

        let config = resolve_model_config(model_path).with_context(|| {
            format!(
                "Failed to resolve model configuration for {}",
                model_path.display()
            )
        })?;

        // Auto-detect format by inspecting safetensors metadata:
        //   - If any tensor has dtype "U8" → packed deployment format
        //     (microsoft/bitnet-b1.58-2B-4T)
        //   - Otherwise → BF16 master-weights format
        //     (microsoft/bitnet-b1.58-2B-4T-bf16)
        let format = detect_weight_format(model_path);
        info!(
            format,
            hidden_size = config.hidden_size,
            n_layers = config.num_hidden_layers,
            n_heads = config.num_attention_heads,
            n_kv_heads = config.num_key_value_heads,
            max_seq = config.max_position_embeddings,
            "Detected weight format and resolved model config"
        );

        let weights = match format {
            "packed" => {
                info!("Loading packed HuggingFace weights via canonical conversion pipeline");
                let packed_model =
                    load_hf_packed_model(model_path, &config).with_context(|| {
                        format!(
                            "Failed to load raw packed HuggingFace weights from {}",
                            model_path.display()
                        )
                    })?;
                convert_hf_packed_to_canonical(&packed_model)
                    .with_context(|| {
                        format!(
                            "Failed to canonically convert packed HuggingFace weights from {}",
                            model_path.display()
                        )
                    })?
                    .weights
            }
            _ => {
                info!("Loading BF16 weights (will quantise on load)");
                load_weights_from_bf16(model_path, &config).with_context(|| {
                    format!("Failed to load BF16 weights from {}", model_path.display())
                })?
            }
        };

        info!("Weights loaded; initialising model");

        let model =
            BitNetModel::new(weights, device).context("Failed to initialise BitNetModel")?;

        let max_seq = config.max_position_embeddings;
        let kv_cache = model.new_kv_cache(max_seq);

        let tokenizer = Tokenizer::llama3().context(
            "Failed to load LLaMA 3 tokenizer. \
             Run `bitnet download` to fetch tokenizer.json, \
             or set BITNET_TOKENIZER=/path/to/tokenizer.json.",
        )?;

        let sampling_buffers = SamplingBuffers::new(config.vocab_size);
        let logits_buf = vec![0.0_f32; config.vocab_size];

        info!(backend = %model.backend_name(), "InferenceEngine ready");

        Ok(Self {
            model,
            tokenizer,
            kv_cache,
            sampling_buffers,
            logits_buf,
        })
    }

    /// Generate text by continuing a raw prompt string.
    ///
    /// The prompt is encoded (without BOS), prefilled into the model, then
    /// tokens are generated autoregressively until EOS or `max_new_tokens`.
    ///
    /// # Arguments
    ///
    /// - `prompt`:   Raw text prompt (not chat-formatted).
    /// - `sampling`: Sampling hyperparameters.
    ///
    /// # Returns
    ///
    /// The generated continuation as a UTF-8 string.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding, model forward, or decoding fails.
    #[instrument(level = "info", skip(self, prompt, sampling), fields(prompt_len = prompt.len()))]
    pub fn generate(&mut self, prompt: &str, sampling: &SamplingConfig) -> anyhow::Result<String> {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .context("Failed to encode raw prompt with BOS")?;
        self.generate_from_tokens(tokens, sampling)
    }

    fn generate_from_tokens(
        &mut self,
        tokens: Vec<u32>,
        sampling: &SamplingConfig,
    ) -> anyhow::Result<String> {
        // Delegate to streaming version with a no-op callback.
        let (text, _n_tokens) =
            self.generate_from_tokens_streaming(tokens, sampling, &mut |_| {
                std::ops::ControlFlow::Continue(())
            })?;
        Ok(text)
    }

    /// Generate text with per-token streaming output.
    ///
    /// Each generated token is decoded and passed to `on_token`. The callback
    /// receives the decoded text fragment and returns `ControlFlow::Continue(())`
    /// to keep generating or `ControlFlow::Break(())` to stop early.
    ///
    /// Returns the complete generated text and the number of generated tokens.
    ///
    /// # Arguments
    ///
    /// - `prompt`:   Raw text prompt (not chat-formatted).
    /// - `sampling`: Sampling hyperparameters.
    /// - `on_token`: Callback invoked for each generated token.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding, model forward, or decoding fails.
    #[instrument(level = "info", skip(self, prompt, sampling, on_token), fields(prompt_len = prompt.len()))]
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        sampling: &SamplingConfig,
        mut on_token: impl FnMut(&str) -> std::ops::ControlFlow<()>,
    ) -> anyhow::Result<(String, usize)> {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .context("Failed to encode raw prompt with BOS")?;
        self.generate_from_tokens_streaming(tokens, sampling, &mut on_token)
    }

    /// Generate chat response with per-token streaming output.
    ///
    /// Returns the complete response text and the number of generated tokens.
    #[instrument(
        level = "info",
        skip(self, messages, sampling, on_token),
        fields(n_messages = messages.len())
    )]
    pub fn generate_chat_streaming(
        &mut self,
        messages: &[ChatMessage],
        sampling: &SamplingConfig,
        mut on_token: impl FnMut(&str) -> std::ops::ControlFlow<()>,
    ) -> anyhow::Result<(String, usize)> {
        let tokens = self
            .tokenizer
            .encode_chat(messages)
            .context("Failed to encode chat template")?;
        self.generate_from_tokens_streaming(tokens, sampling, &mut on_token)
    }

    fn generate_from_tokens_streaming(
        &mut self,
        tokens: Vec<u32>,
        sampling: &SamplingConfig,
        on_token: &mut dyn FnMut(&str) -> std::ops::ControlFlow<()>,
    ) -> anyhow::Result<(String, usize)> {
        sampling.validate().context("Invalid sampling config")?;

        if tokens.is_empty() {
            return Ok((String::new(), 0));
        }

        // Reset KV cache for a clean generation.
        self.kv_cache.clear();

        // Prefill: process all prompt tokens using forward_into (zero-copy).
        self.model
            .forward_into(&tokens, 0, &mut self.kv_cache, &mut self.logits_buf)
            .context("Prefill forward pass failed")?;

        debug!(prefill_tokens = tokens.len(), "Prefill complete");

        let mut all_tokens = tokens;
        all_tokens.reserve(sampling.max_new_tokens);

        let eos_eot = self.tokenizer.eos_token_id();
        let eos_eot_text = 128_001_u32;

        let mut n_generated = 0usize;
        let mut output_text = String::new();

        for step in 0..sampling.max_new_tokens {
            let next_token = sample_next_token(
                &mut self.logits_buf,
                sampling,
                &all_tokens,
                &mut self.sampling_buffers,
            );

            if next_token == eos_eot || next_token == eos_eot_text {
                debug!(
                    step,
                    token = next_token,
                    "EOS/EOT token produced; stopping generation"
                );
                break;
            }

            all_tokens.push(next_token);
            n_generated += 1;

            // Decode the single token to text and stream it.
            let token_text = self.tokenizer.decode(&[next_token]).unwrap_or_default();
            output_text.push_str(&token_text);

            // Invoke the callback with the decoded fragment.
            if let std::ops::ControlFlow::Break(()) = on_token(&token_text) {
                debug!(step, "Streaming stopped by callback");
                break;
            }

            debug!(step, token = next_token, "Generated token");

            // Decode step: forward pass for the single new token (zero-copy).
            let cur_pos = self.kv_cache.filled_positions;
            self.model
                .forward_into(
                    &[next_token],
                    cur_pos,
                    &mut self.kv_cache,
                    &mut self.logits_buf,
                )
                .with_context(|| format!("Decode forward pass failed at step {step}"))?;
        }

        debug!(n_generated, "Streaming generation complete");
        Ok((output_text, n_generated))
    }

    /// Generate a response for a chat conversation.
    ///
    /// Applies the LLaMA 3 Instruct chat template to `messages`, then generates
    /// a response autoregressively.
    ///
    /// # Arguments
    ///
    /// - `messages`: Ordered list of conversation messages.
    /// - `sampling`: Sampling hyperparameters.
    ///
    /// # Returns
    ///
    /// The assistant's response as a UTF-8 string.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding, model forward, or decoding fails.
    #[instrument(
        level = "info",
        skip(self, messages, sampling),
        fields(n_messages = messages.len())
    )]
    pub fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        sampling: &SamplingConfig,
    ) -> anyhow::Result<String> {
        let tokens = self
            .tokenizer
            .encode_chat(messages)
            .context("Failed to encode chat template")?;
        self.generate_from_tokens(tokens, sampling)
    }

    /// Reset the KV cache for a fresh conversation.
    ///
    /// Must be called between unrelated conversations to prevent context
    /// from one conversation influencing the next.
    pub fn reset(&mut self) {
        self.kv_cache.clear();
        debug!("KV cache cleared");
    }

    /// Returns a reference to the tokenizer.
    #[inline]
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Returns the current number of positions in the KV cache.
    #[inline]
    pub fn context_length(&self) -> usize {
        self.kv_cache.filled_positions
    }
}

// ---------------------------------------------------------------------------
// ChatPipeline
// ---------------------------------------------------------------------------

/// Stateful multi-turn chat pipeline.
///
/// Maintains conversation history across calls to [`ChatPipeline::chat`] and
/// automatically applies the LLaMA 3 chat template.
///
/// # Usage
///
/// ```no_run
/// use bitnet_inference::{ChatPipeline, SamplingConfig};
/// use bitnet_core::backend::Device;
/// use std::path::Path;
///
/// let mut pipeline = ChatPipeline::new(
///     Path::new("model.safetensors"),
///     Device::cpu(),
///     "You are a helpful AI assistant.",
/// ).unwrap();
///
/// let response = pipeline.chat("What is 2 + 2?", &SamplingConfig::greedy()).unwrap();
/// println!("Assistant: {response}");
/// ```
///
/// # Invariants
///
/// - `history` always alternates between `user` and `assistant` messages
///   (after the system message).
/// - `reset_conversation` clears `history` but preserves `system_prompt`.
pub struct ChatPipeline {
    /// The underlying inference engine.
    engine: InferenceEngine,
    /// Conversation history (system + user + assistant turns).
    history: Vec<ChatMessage>,
    /// System prompt, prepended to every conversation.
    system_prompt: String,
}

impl std::fmt::Debug for ChatPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatPipeline")
            .field("n_history", &self.history.len())
            .field("system_prompt_len", &self.system_prompt.len())
            .finish()
    }
}

impl ChatPipeline {
    /// Create a new [`ChatPipeline`].
    ///
    /// # Arguments
    ///
    /// - `model_path`:    Path to the BF16 `.safetensors` checkpoint.
    /// - `device`:        Compute device.
    /// - `system_prompt`: System prompt prepended to every conversation.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`InferenceEngine::new`].
    pub fn new(
        model_path: &Path,
        device: Device,
        system_prompt: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let engine = InferenceEngine::new(model_path, device)?;
        let system_prompt = system_prompt.into();

        Ok(Self {
            engine,
            history: Vec::new(),
            system_prompt,
        })
    }

    /// Send a user message and receive an assistant response.
    ///
    /// The full conversation history (system + prior turns + new user message)
    /// is formatted into a LLaMA 3 Instruct prompt, then the model generates
    /// a response.  The response is appended to `history` as an `assistant`
    /// message.
    ///
    /// # Arguments
    ///
    /// - `user_message`: The user's message text.
    /// - `sampling`:     Sampling hyperparameters.
    ///
    /// # Returns
    ///
    /// The assistant's response string.
    ///
    /// # Errors
    ///
    /// Propagates any error from the underlying engine.
    pub fn chat(
        &mut self,
        user_message: &str,
        sampling: &SamplingConfig,
    ) -> anyhow::Result<String> {
        // Build message list: system + history + new user message.
        let mut messages: Vec<ChatMessage> = Vec::new();

        if !self.system_prompt.is_empty() {
            messages.push(ChatMessage::system(&self.system_prompt));
        }

        messages.extend_from_slice(&self.history);
        messages.push(ChatMessage::user(user_message));

        // Generate response.
        let response = self.engine.generate_chat(&messages, sampling)?;

        // Append to history.
        self.history.push(ChatMessage::user(user_message));
        self.history.push(ChatMessage::assistant(&response));

        Ok(response)
    }

    /// Send a user message and receive a streaming assistant response.
    ///
    /// Each generated token is decoded and passed to `on_token`. Returns
    /// the complete response and the token count.
    pub fn chat_streaming(
        &mut self,
        user_message: &str,
        sampling: &SamplingConfig,
        on_token: impl FnMut(&str) -> std::ops::ControlFlow<()>,
    ) -> anyhow::Result<(String, usize)> {
        let mut messages: Vec<ChatMessage> = Vec::new();

        if !self.system_prompt.is_empty() {
            messages.push(ChatMessage::system(&self.system_prompt));
        }

        messages.extend_from_slice(&self.history);
        messages.push(ChatMessage::user(user_message));

        let (response, n_tokens) = self
            .engine
            .generate_chat_streaming(&messages, sampling, on_token)?;

        self.history.push(ChatMessage::user(user_message));
        self.history.push(ChatMessage::assistant(&response));

        Ok((response, n_tokens))
    }

    /// Reset the conversation history (keeps the system prompt).
    ///
    /// Also clears the model's KV cache.
    pub fn reset_conversation(&mut self) {
        self.history.clear();
        self.engine.reset();
        debug!("Conversation reset");
    }

    /// Returns the current conversation history.
    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    /// Returns the system prompt.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Update the system prompt and reset the conversation.
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = prompt.into();
        self.reset_conversation();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // SamplingConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn sampling_config_greedy_is_valid() {
        SamplingConfig::greedy()
            .validate()
            .expect("greedy config must be valid");
    }

    #[test]
    fn sampling_config_chat_defaults_is_valid() {
        SamplingConfig::chat_defaults()
            .validate()
            .expect("chat_defaults must be valid");
    }

    #[test]
    fn sampling_config_creative_is_valid() {
        SamplingConfig::creative()
            .validate()
            .expect("creative config must be valid");
    }

    #[test]
    fn sampling_config_zero_temperature_returns_error() {
        let mut cfg = SamplingConfig::greedy();
        cfg.temperature = 0.0;
        assert!(cfg.validate().is_err(), "temperature=0 must be invalid");
    }

    #[test]
    fn sampling_config_top_p_out_of_range_returns_error() {
        let mut cfg = SamplingConfig::greedy();
        cfg.top_p = 1.5;
        assert!(cfg.validate().is_err(), "top_p > 1 must be invalid");
    }

    #[test]
    fn sampling_config_top_p_zero_returns_error() {
        let mut cfg = SamplingConfig::greedy();
        cfg.top_p = 0.0;
        assert!(cfg.validate().is_err(), "top_p=0 must be invalid");
    }

    #[test]
    fn sampling_config_repetition_penalty_below_one_returns_error() {
        let mut cfg = SamplingConfig::greedy();
        cfg.repetition_penalty = 0.9;
        assert!(
            cfg.validate().is_err(),
            "repetition_penalty < 1.0 must be invalid"
        );
    }

    #[test]
    fn sampling_config_zero_max_new_tokens_returns_error() {
        let mut cfg = SamplingConfig::greedy();
        cfg.max_new_tokens = 0;
        assert!(cfg.validate().is_err(), "max_new_tokens=0 must be invalid");
    }

    #[test]
    fn sampling_config_default_is_valid() {
        SamplingConfig::default()
            .validate()
            .expect("default SamplingConfig must be valid");
    }

    // -----------------------------------------------------------------------
    // sample_next_token tests
    // -----------------------------------------------------------------------

    #[test]
    fn sample_greedy_picks_argmax() {
        // With top_k=1, must return the argmax regardless of other settings.
        let mut logits = vec![0.1_f32, 0.5, 5.0, 0.2, -1.0];
        let mut buffers = SamplingBuffers::new(logits.len());
        let cfg = SamplingConfig::greedy(); // top_k=1
        let token = sample_next_token(&mut logits, &cfg, &[], &mut buffers);
        assert_eq!(token, 2u32, "greedy must pick index 2 (logit=5.0)");
    }

    #[test]
    fn sample_greedy_picks_last_element_if_max() {
        let mut logits = vec![0.0_f32, 0.0, 0.0, 0.0, 10.0];
        let mut buffers = SamplingBuffers::new(logits.len());
        let cfg = SamplingConfig::greedy();
        let token = sample_next_token(&mut logits, &cfg, &[], &mut buffers);
        assert_eq!(token, 4u32, "greedy must pick the last (max) element");
    }

    #[test]
    fn sample_greedy_picks_first_element_if_max() {
        let mut logits = vec![100.0_f32, 0.0, -1.0, -5.0];
        let mut buffers = SamplingBuffers::new(logits.len());
        let cfg = SamplingConfig::greedy();
        let token = sample_next_token(&mut logits, &cfg, &[], &mut buffers);
        assert_eq!(token, 0u32, "greedy must pick index 0 (max logit)");
    }

    #[test]
    fn sample_token_in_valid_range() {
        // Property: the sampled token must always be a valid vocab index.
        let vocab_size = 32;
        let mut logits: Vec<f32> = (0..vocab_size).map(|i| i as f32 * 0.1).collect();
        let mut buffers = SamplingBuffers::new(vocab_size);
        let cfg = SamplingConfig {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 10,
            repetition_penalty: 1.0,
            max_new_tokens: 1,
            seed: 42,
        };
        let token = sample_next_token(&mut logits, &cfg, &[], &mut buffers);
        assert!(
            (token as usize) < vocab_size,
            "sampled token {token} must be < vocab_size {vocab_size}"
        );
    }

    #[test]
    fn sample_repetition_penalty_reduces_repeated_token_probability() {
        // With high repetition penalty, the repeated token should not be selected.
        let vocab_size = 4;
        // Token 2 has the highest logit, but we penalise it.
        let mut logits = vec![0.0_f32, 0.0, 10.0, 0.0];
        let mut buffers = SamplingBuffers::new(logits.len());
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1000.0, // extreme penalty
            max_new_tokens: 1,
            seed: 42,
        };
        // Past token 2 has been generated before.
        let past = vec![2u32];
        // After extreme penalty: logits[2] = 10.0 / 1000.0 = 0.01
        // One of the other tokens (0.0 logit) should now be more likely.
        // The test verifies we don't always return 2.
        let token = sample_next_token(&mut logits, &cfg, &past, &mut buffers);
        // After penalty, logit[2] = 0.01, so the distribution is nearly uniform.
        // The sampled token could be anything — just verify it's in range.
        assert!(
            (token as usize) < vocab_size,
            "token {token} must be in range"
        );
    }

    #[test]
    fn sample_deterministic_for_same_seed() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 10,
            repetition_penalty: 1.0,
            max_new_tokens: 1,
            seed: 12345,
        };
        let logits_orig: Vec<f32> = (0..128).map(|i| (i as f32 * 0.17).sin()).collect();
        let mut buffers = SamplingBuffers::new(logits_orig.len());

        let mut l1 = logits_orig.clone();
        let mut l2 = logits_orig.clone();
        let t1 = sample_next_token(&mut l1, &cfg, &[], &mut buffers);
        let t2 = sample_next_token(&mut l2, &cfg, &[], &mut buffers);

        assert_eq!(
            t1, t2,
            "same seed and logits must produce same token: {t1} vs {t2}"
        );
    }

    #[test]
    fn sample_different_seeds_may_differ() {
        // With a diverse distribution and different seeds, we may get different tokens.
        // This is a probabilistic test — we verify it doesn't panic.
        let logits_orig: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut buffers = SamplingBuffers::new(logits_orig.len());
        let mut seen = std::collections::HashSet::new();

        for seed in 0..20_u64 {
            let cfg = SamplingConfig {
                temperature: 1.5,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                max_new_tokens: 1,
                seed,
            };
            let mut l = logits_orig.clone();
            let tok = sample_next_token(&mut l, &cfg, &[], &mut buffers);
            seen.insert(tok);
        }
        // Over 20 different seeds, we should see at least 2 distinct tokens
        // (extremely likely for a non-trivial distribution).
        assert!(
            seen.len() >= 2,
            "different seeds should (usually) produce different tokens; got: {:?}",
            seen
        );
    }

    #[test]
    fn sample_all_neg_infinity_except_one() {
        // If only one token has a finite logit, it must always be selected.
        let mut logits = vec![f32::NEG_INFINITY; 8];
        logits[5] = 1.0;
        let mut buffers = SamplingBuffers::new(logits.len());
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            max_new_tokens: 1,
            seed: 0,
        };
        let token = sample_next_token(&mut logits, &cfg, &[], &mut buffers);
        // With only one non-neg-infinity token, the sampling must pick index 5.
        // (The fallback argmax also picks 5 since 1.0 > NEG_INFINITY.)
        assert_eq!(token, 5u32, "must pick the only finite token");
    }

    #[test]
    fn sample_top_k_restricts_to_k_highest() {
        // With top_k=1 and a clear winner, must always pick the winner.
        let mut logits = vec![1.0_f32, 5.0, 0.0, -1.0, 2.0];
        let mut buffers = SamplingBuffers::new(logits.len());
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 1.0,
            max_new_tokens: 1,
            seed: 0,
        };
        let token = sample_next_token(&mut logits, &cfg, &[], &mut buffers);
        assert_eq!(
            token, 1u32,
            "top_k=1 must always return the highest-logit token"
        );
    }

    #[test]
    fn sample_temperature_near_zero_approaches_greedy() {
        // Very low temperature → logits amplified → the max dominates.
        let mut logits = vec![0.0_f32, 1.0, 10.0, 2.0];
        let mut buffers = SamplingBuffers::new(logits.len());
        let cfg = SamplingConfig {
            temperature: 0.01,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            max_new_tokens: 1,
            seed: 0,
        };
        let token = sample_next_token(&mut logits, &cfg, &[], &mut buffers);
        assert_eq!(
            token, 2u32,
            "very low temperature must select the max-logit token"
        );
    }

    // -----------------------------------------------------------------------
    // ChatMessage tests
    // -----------------------------------------------------------------------

    #[test]
    fn chat_message_constructors() {
        let sys = ChatMessage::system("Be helpful.");
        assert_eq!(sys.role, "system");
        assert_eq!(sys.content, "Be helpful.");

        let usr = ChatMessage::user("Hello!");
        assert_eq!(usr.role, "user");

        let ast = ChatMessage::assistant("Hi there!");
        assert_eq!(ast.role, "assistant");
    }

    // -----------------------------------------------------------------------
    // SamplingConfig clone and PartialEq
    // -----------------------------------------------------------------------

    #[test]
    fn sampling_config_clone_equality() {
        let cfg = SamplingConfig::chat_defaults();
        let cloned = cfg.clone();
        assert_eq!(cfg, cloned, "cloned SamplingConfig must equal original");
    }

    #[test]
    fn sampling_config_different_fields_not_equal() {
        let cfg1 = SamplingConfig::greedy();
        let cfg2 = SamplingConfig::creative();
        assert_ne!(cfg1, cfg2, "greedy and creative configs must differ");
    }

    // -----------------------------------------------------------------------
    // Integration smoke test: verify sample_next_token never panics on
    // realistic logit distributions.
    // -----------------------------------------------------------------------

    #[test]
    fn sample_stress_test_no_panic() {
        let vocab_size = 1024;
        let mut buffers = SamplingBuffers::new(vocab_size);
        let cfg = SamplingConfig {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            max_new_tokens: 256,
            seed: 42,
        };

        // Simulate 100 sampling steps.
        let mut generated: Vec<u32> = Vec::new();
        for step in 0..100_u64 {
            // Simulate logits from a model (sine wave for variety).
            let mut logits: Vec<f32> = (0..vocab_size)
                .map(|i| ((i as f32 * 0.01 + step as f32 * 0.1).sin()) * 3.0)
                .collect();

            let mut step_cfg = cfg.clone();
            step_cfg.seed = 42 + step;

            let tok = sample_next_token(&mut logits, &step_cfg, &generated, &mut buffers);
            assert!(
                (tok as usize) < vocab_size,
                "step {step}: token {tok} out of range"
            );
            generated.push(tok);
        }
    }

    #[test]
    fn sibling_config_path_uses_config_filename_constant() {
        let model_path = Path::new("models/bitnet/model.safetensors");
        let config_path = sibling_config_path(model_path).unwrap();

        assert_eq!(config_path, PathBuf::from("models/bitnet/config.json"));
    }

    #[test]
    fn resolve_model_config_falls_back_to_bitnet_2b_when_config_missing() {
        let model_path = Path::new("models/bitnet/model.safetensors");
        let config = resolve_model_config(model_path).unwrap();

        assert_eq!(config, bitnet_2b_config());
    }

    #[test]
    fn resolve_model_config_loads_sibling_config_when_present() {
        let temp_dir = std::env::temp_dir().join(format!(
            "bitnet_inference_config_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();

        let model_path = temp_dir.join("model.safetensors");
        let config_path = temp_dir.join("config.json");

        std::fs::write(&model_path, b"placeholder").unwrap();
        std::fs::write(
            &config_path,
            r#"{
                "model_type": "bitnet",
                "vocab_size": 32000,
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 4096,
                "max_position_embeddings": 2048,
                "rope_theta": 500000.0,
                "rms_norm_eps": 0.00001
            }"#,
        )
        .unwrap();

        let config = resolve_model_config(&model_path).unwrap();

        assert_eq!(config.vocab_size, 32_000);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.intermediate_size, 4096);
        assert_eq!(config.max_position_embeddings, 2048);
        assert_eq!(config.rope_theta, 500_000.0);
        assert_eq!(config.rms_norm_eps, 1e-5);

        let _ = std::fs::remove_file(&config_path);
        let _ = std::fs::remove_file(&model_path);
        let _ = std::fs::remove_dir(&temp_dir);
    }

    #[cfg(test)]
    fn extract_top_k_indices_desc(values: &[f32], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
    }

    #[cfg(test)]
    fn top_k_token_strings(tokenizer: &Tokenizer, logits: &[f32], k: usize) -> Vec<String> {
        extract_top_k_indices_desc(logits, k)
            .into_iter()
            .map(|idx| {
                let rendered = tokenizer
                    .decode_with_special_tokens(&[idx as u32])
                    .replace('\n', "\\n");
                format!("{idx}:{rendered}")
            })
            .collect()
    }

    #[cfg(test)]
    fn vector_summary(values: &[f32]) -> (f32, f32, f32) {
        let mut min_value = f32::INFINITY;
        let mut max_value = f32::NEG_INFINITY;
        let mut sum_abs = 0.0_f32;

        for &value in values {
            min_value = min_value.min(value);
            max_value = max_value.max(value);
            sum_abs += value.abs();
        }

        let mean_abs = if values.is_empty() {
            0.0
        } else {
            sum_abs / values.len() as f32
        };

        (min_value, max_value, mean_abs)
    }

    #[cfg(test)]
    fn parse_hf_logits_response(body: &str) -> anyhow::Result<Vec<f32>> {
        let json: Value = serde_json::from_str(body)
            .context("failed to parse HuggingFace logits JSON response")?;

        let logits = json
            .get("logits")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("HuggingFace response missing top-level 'logits' array"))?;

        let first_batch = logits
            .first()
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("HuggingFace response missing batch dimension at logits[0]"))?;

        let last_step = first_batch
            .last()
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("HuggingFace response missing final token logits row"))?;

        let mut parsed = Vec::with_capacity(last_step.len());
        for (index, value) in last_step.iter().enumerate() {
            let logit = value
                .as_f64()
                .ok_or_else(|| anyhow!("HuggingFace logit at index {index} is not numeric"))?
                as f32;
            parsed.push(logit);
        }

        Ok(parsed)
    }

    #[test]
    #[ignore = "requires HF reference endpoint or captured JSON payload for forensic logits comparison"]
    fn forensic_huggingface_reference_logits_parser_extracts_last_step() {
        let payload = r#"{
            "logits": [
                [
                    [0.1, 0.2, 0.3],
                    [1.5, -2.0, 4.25]
                ]
            ]
        }"#;

        let parsed = parse_hf_logits_response(payload)
            .expect("reference logits parser must extract the final sequence step");

        assert_eq!(
            parsed.len(),
            3,
            "final HuggingFace logits row must preserve vocabulary width"
        );
        assert!(
            (parsed[0] - 1.5).abs() < 1e-6,
            "expected first final-step logit to equal 1.5, got {}",
            parsed[0]
        );
        assert!(
            (parsed[1] + 2.0).abs() < 1e-6,
            "expected second final-step logit to equal -2.0, got {}",
            parsed[1]
        );
        assert!(
            (parsed[2] - 4.25).abs() < 1e-6,
            "expected third final-step logit to equal 4.25, got {}",
            parsed[2]
        );
    }

    #[test]
    fn extract_top_k_indices_desc_orders_by_logit_descending() {
        let logits = vec![0.5_f32, 3.0, -1.0, 2.0, 3.0];
        let top = extract_top_k_indices_desc(&logits, 3);

        assert_eq!(
            top,
            vec![1, 4, 3],
            "top-k extraction must sort by descending logit and break ties by lower index"
        );
    }

    #[test]
    #[ignore = "requires locally downloaded HuggingFace packed weights at the default cache path"]
    fn real_huggingface_packed_weights_smoke_test() {
        let model_path = Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );

        assert!(
            model_path.exists(),
            "expected local HuggingFace packed weights at {}",
            model_path.display()
        );

        let mut engine = InferenceEngine::new(model_path, Device::cpu())
            .expect("real HuggingFace packed weights must initialise an inference engine");

        let sampling = SamplingConfig {
            temperature: 0.01,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 1.0,
            max_new_tokens: 1,
            seed: 42,
        };

        let output = engine
            .generate("Hello", &sampling)
            .expect("real HuggingFace packed weights must complete a one-token decode");

        assert!(
            !output.is_empty(),
            "real HuggingFace packed-weight inference must produce non-empty output"
        );
        assert!(
            engine.context_length() >= 2,
            "prefill plus one decode step must advance the KV cache beyond the prompt; got {}",
            engine.context_length()
        );
    }

    #[test]
    #[ignore = "requires locally downloaded HuggingFace packed weights at the default cache path"]
    fn forensic_local_packed_model_top_logit_regression() {
        let model_path = Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );

        assert!(
            model_path.exists(),
            "expected local HuggingFace packed weights at {}",
            model_path.display()
        );

        let mut engine = InferenceEngine::new(model_path, Device::cpu())
            .expect("real HuggingFace packed weights must initialise an inference engine");

        let prompt = "What is the capital of France?";
        let tokens = engine
            .tokenizer()
            .encode(prompt, true)
            .expect("forensic prompt encoding must succeed");

        let logits = engine
            .model
            .forward(&tokens, 0, &mut engine.kv_cache)
            .expect("forensic local forward pass must succeed");

        assert_eq!(
            logits.len(),
            engine.tokenizer().vocab_size(),
            "forensic logits width must equal tokenizer vocabulary size"
        );

        let top10_indices = extract_top_k_indices_desc(&logits, 10);
        let top10_tokens = top_k_token_strings(engine.tokenizer(), &logits, 10);

        assert_eq!(
            top10_indices.len(),
            10,
            "forensic top-logit extraction must return exactly 10 indices"
        );
        assert_eq!(
            top10_tokens.len(),
            10,
            "forensic top-logit token rendering must return exactly 10 entries"
        );

        // After fixing the row-interleaving bug in packed weight decoding,
        // the top-10 logits will change. Print them for regression capture
        // and verify basic sanity invariants.
        eprintln!("Forensic top-10 token indices: {:?}", top10_indices);
        eprintln!("Forensic top-10 token strings: {:?}", top10_tokens);

        // Sanity: top-10 must be distinct and within vocabulary range.
        let vocab_size = engine.tokenizer().vocab_size();
        assert!(
            top10_indices.iter().all(|&idx| idx < vocab_size),
            "all forensic top-10 indices must be within vocabulary range [0, {vocab_size})"
        );
        let unique: std::collections::HashSet<usize> = top10_indices.iter().cloned().collect();
        assert_eq!(
            unique.len(),
            top10_indices.len(),
            "forensic top-10 indices must be distinct"
        );
    }

    #[test]
    #[ignore = "requires locally downloaded HuggingFace packed weights at the default cache path"]
    fn forensic_local_packed_model_layer0_projection_regression() {
        let model_path = Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );

        assert!(
            model_path.exists(),
            "expected local HuggingFace packed weights at {}",
            model_path.display()
        );

        let config = resolve_model_config(model_path)
            .expect("forensic packed-model config resolution must succeed");
        let packed_model = load_hf_packed_model(model_path, &config)
            .expect("forensic packed-model raw load must succeed");
        let canonical = convert_hf_packed_to_canonical(&packed_model)
            .expect("forensic packed-model canonical conversion must succeed");

        let layer0 = canonical
            .weights
            .layers
            .first()
            .expect("forensic packed-model canonical weights must contain layer 0");

        let q_summary = vector_summary(
            &layer0
                .q_proj
                .data
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<f32>>(),
        );
        let k_summary = vector_summary(
            &layer0
                .k_proj
                .data
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<f32>>(),
        );
        let v_summary = vector_summary(
            &layer0
                .v_proj
                .data
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<f32>>(),
        );
        let o_summary = vector_summary(
            &layer0
                .o_proj
                .data
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<f32>>(),
        );

        assert_eq!(
            layer0.q_proj.rows, config.hidden_size,
            "layer-0 q_proj rows must equal hidden_size"
        );
        assert_eq!(
            layer0.q_proj.cols, config.hidden_size,
            "layer-0 q_proj cols must equal hidden_size"
        );
        assert_eq!(
            layer0.k_proj.rows,
            config.num_key_value_heads * config.head_dim(),
            "layer-0 k_proj rows must equal kv_dim"
        );
        assert_eq!(
            layer0.k_proj.cols, config.hidden_size,
            "layer-0 k_proj cols must equal hidden_size"
        );
        assert_eq!(
            layer0.v_proj.rows,
            config.num_key_value_heads * config.head_dim(),
            "layer-0 v_proj rows must equal kv_dim"
        );
        assert_eq!(
            layer0.v_proj.cols, config.hidden_size,
            "layer-0 v_proj cols must equal hidden_size"
        );
        assert_eq!(
            layer0.o_proj.rows, config.hidden_size,
            "layer-0 o_proj rows must equal hidden_size"
        );
        assert_eq!(
            layer0.o_proj.cols, config.hidden_size,
            "layer-0 o_proj cols must equal hidden_size"
        );

        let expected_q_scale = 1.21875_f32;
        let expected_k_scale = 1.796875_f32;
        let expected_v_scale = 2.296875_f32;
        let expected_o_scale = 0.96484375_f32;

        assert!(
            (layer0.q_proj.scale - expected_q_scale).abs() < 1e-4,
            "layer-0 q_proj scale regressed: got {}, expected ~{expected_q_scale}",
            layer0.q_proj.scale
        );
        assert!(
            (layer0.k_proj.scale - expected_k_scale).abs() < 1e-4,
            "layer-0 k_proj scale regressed: got {}, expected ~{expected_k_scale}",
            layer0.k_proj.scale
        );
        assert!(
            (layer0.v_proj.scale - expected_v_scale).abs() < 1e-4,
            "layer-0 v_proj scale regressed: got {}, expected ~{expected_v_scale}",
            layer0.v_proj.scale
        );
        assert!(
            (layer0.o_proj.scale - expected_o_scale).abs() < 1e-4,
            "layer-0 o_proj scale regressed: got {}, expected ~{expected_o_scale}",
            layer0.o_proj.scale
        );

        // Ternary values must remain in {-1, 0, +1} domain.
        assert!(
            q_summary.0 >= -1.0 && q_summary.1 <= 1.0,
            "layer-0 q_proj ternary domain: got {:?}",
            q_summary
        );
        assert!(
            k_summary.0 >= -1.0 && k_summary.1 <= 1.0,
            "layer-0 k_proj ternary domain: got {:?}",
            k_summary
        );
        assert!(
            v_summary.0 >= -1.0 && v_summary.1 <= 1.0,
            "layer-0 v_proj ternary domain: got {:?}",
            v_summary
        );
        assert!(
            o_summary.0 >= -1.0 && o_summary.1 <= 1.0,
            "layer-0 o_proj ternary domain: got {:?}",
            o_summary
        );
    }

    #[test]
    #[ignore = "requires locally downloaded HuggingFace packed weights at the default cache path"]
    fn real_huggingface_packed_weights_chat_generation_produces_output() {
        // End-to-end chat generation test with the real BitNet 2B-4T packed
        // weights.  Validates that the full pipeline — weight loading, canonical
        // conversion, activation quantisation, 30-layer forward pass, sampling,
        // and decoding — produces a semantically correct answer.
        //
        // The model is asked "What is the capital of France?" and must respond
        // with a string containing "Paris".
        let model_path = Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );

        assert!(
            model_path.exists(),
            "expected local HuggingFace packed weights at {}",
            model_path.display()
        );

        let mut engine = InferenceEngine::new(model_path, Device::cpu())
            .expect("real HuggingFace packed weights must initialise an inference engine");

        let sampling = SamplingConfig {
            temperature: 0.6,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            max_new_tokens: 64,
            seed: 42,
        };

        let messages = [ChatMessage::user("What is the capital of France?")];

        let output = engine
            .generate_chat(&messages, &sampling)
            .expect("chat generation with packed weights must succeed");

        eprintln!(
            "[chat-generation] output ({} chars): {:?}",
            output.len(),
            output
        );

        // 1. Output must be non-empty and contain meaningful text.
        assert!(
            !output.trim().is_empty(),
            "chat generation must return non-whitespace output; got {:?}",
            output
        );

        // 2. Semantic correctness: the answer must mention "Paris".
        assert!(
            output.contains("Paris"),
            "chat generation for 'What is the capital of France?' must contain 'Paris'; got {:?}",
            output
        );

        // 3. Valid UTF-8 (guaranteed by String, but demonstrates no garbage bytes).
        assert!(
            std::str::from_utf8(output.as_bytes()).is_ok(),
            "chat generation output must be valid UTF-8"
        );

        // 4. Context length accounts for the chat template overhead (~17 tokens
        //    for a single-user-turn LLaMA-3 template) plus generated tokens.
        let chat_template_min_tokens = 17;
        assert!(
            engine.context_length() >= chat_template_min_tokens,
            "context_length must be at least the chat-template token count ({}); got {}",
            chat_template_min_tokens,
            engine.context_length()
        );
    }

    #[test]
    #[ignore = "requires locally downloaded HuggingFace packed weights and a captured HuggingFace logits JSON payload"]
    fn forensic_local_vs_huggingface_reference_top_logits_scaffold() {
        let model_path = Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );

        assert!(
            model_path.exists(),
            "expected local HuggingFace packed weights at {}",
            model_path.display()
        );

        let mut engine = InferenceEngine::new(model_path, Device::cpu())
            .expect("real HuggingFace packed weights must initialise an inference engine");

        let prompt = "What is the capital of France?";
        let tokens = engine
            .tokenizer()
            .encode(prompt, true)
            .expect("forensic prompt encoding must succeed");

        let local_logits = engine
            .model
            .forward(&tokens, 0, &mut engine.kv_cache)
            .expect("local forensic forward pass must succeed");

        assert_eq!(
            local_logits.len(),
            engine.tokenizer().vocab_size(),
            "local logits width must equal tokenizer vocabulary size"
        );

        let local_top10 = extract_top_k_indices_desc(&local_logits, 10);
        assert_eq!(
            local_top10.len(),
            10,
            "local forensic top-k extraction must return exactly 10 indices"
        );

        let captured_reference_payload = std::env::var("BITNET_HF_REFERENCE_LOGITS_JSON").ok();

        if let Some(payload) = captured_reference_payload {
            let reference_logits = parse_hf_logits_response(&payload)
                .expect("captured HuggingFace reference logits payload must parse");

            assert_eq!(
                reference_logits.len(),
                local_logits.len(),
                "reference and local logits must have identical vocabulary width"
            );

            let reference_top10 = extract_top_k_indices_desc(&reference_logits, 10);

            assert!(
                !reference_top10.is_empty(),
                "reference forensic top-k extraction must produce at least one token"
            );
            assert!(
                !local_top10.is_empty(),
                "local forensic top-k extraction must produce at least one token"
            );

            let overlap = local_top10
                .iter()
                .filter(|idx| reference_top10.contains(idx))
                .count();

            assert!(
                overlap >= 1,
                "local/reference top-10 logits should share at least one token; local={local_top10:?}, reference={reference_top10:?}"
            );
        } else {
            assert!(
                !local_top10.is_empty(),
                "forensic scaffold must at least compute local top logits when no reference payload is provided"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Mathematical invariant: softmax output of sample sums to ≤ 1.
    //
    // Theorem: after top-k + top-p filtering, the probability mass assigned
    // to included tokens sums to ≤ 1.0 (exactly 1.0 when renormalised).
    //
    // We verify this by checking that the sampled token index is always
    // within the top-k tokens.
    // -----------------------------------------------------------------------

    #[test]
    fn sample_respects_top_k_constraint() {
        // With top_k=3, the sampled token must be one of the 3 highest-logit tokens.
        let logits = vec![1.0_f32, 5.0, 3.0, 2.0, 4.0, 0.5]; // indices sorted: 1>4>2>3>0>5
        let top3_indices: std::collections::HashSet<u32> = [1, 4, 2].iter().cloned().collect();
        let mut buffers = SamplingBuffers::new(logits.len());

        // Run multiple seeds to get varied samples.
        for seed in 0..50_u64 {
            let cfg = SamplingConfig {
                temperature: 2.0, // high temp for diversity
                top_p: 1.0,
                top_k: 3,
                repetition_penalty: 1.0,
                max_new_tokens: 1,
                seed,
            };
            let mut l = logits.clone();
            let token = sample_next_token(&mut l, &cfg, &[], &mut buffers);
            assert!(
                top3_indices.contains(&token),
                "seed={seed}: token {token} not in top-3 set {:?}",
                top3_indices
            );
        }
    }

    // -----------------------------------------------------------------------
    // Forensic diagnostic: chat template, embedding magnitudes, and
    // layer-0 projection output magnitudes.
    //
    // This test does NOT assert on specific values — it prints diagnostic
    // data for manual inspection during inference debugging.
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "requires locally downloaded HuggingFace packed weights at the default cache path"]
    fn forensic_chat_template_and_numerical_diagnostics() {
        let model_path = Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );

        assert!(
            model_path.exists(),
            "expected local HuggingFace packed weights at {}",
            model_path.display()
        );

        // ── Chat-formatted generation ──────────────────────────────────────
        let mut engine = InferenceEngine::new(model_path, Device::cpu())
            .expect("real HuggingFace packed weights must initialise an inference engine");

        let sampling = SamplingConfig {
            temperature: 0.01,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 1.0,
            max_new_tokens: 16,
            seed: 42,
        };

        // First, inspect the chat-encoded tokens to verify special token handling.
        let chat_template_str = engine
            .tokenizer()
            .apply_chat_template(&[ChatMessage::user("What is the capital of France?")]);
        eprintln!("=== Chat template string ===");
        eprintln!("{:?}", chat_template_str);

        let chat_tokens = engine
            .tokenizer()
            .encode_chat(&[ChatMessage::user("What is the capital of France?")])
            .expect("chat encoding must succeed");
        eprintln!("\n=== Chat-encoded tokens ({}) ===", chat_tokens.len());
        for (i, &tok) in chat_tokens.iter().enumerate().take(30) {
            let s = engine
                .tokenizer()
                .decode_with_special_tokens(&[tok])
                .replace('\n', "\\n");
            eprintln!("  [{i}] token {tok}: {s:?}");
        }
        if chat_tokens.len() > 30 {
            eprintln!("  ... ({} more tokens)", chat_tokens.len() - 30);
        }

        // Check if special token IDs are present in the encoded sequence.
        let has_bos = chat_tokens.contains(&128000);
        let has_start_header = chat_tokens.contains(&128006);
        let has_end_header = chat_tokens.contains(&128007);
        let has_eot = chat_tokens.contains(&128009);
        eprintln!("\n=== Special token presence ===");
        eprintln!("  BOS (128000):          {has_bos}");
        eprintln!("  start_header (128006): {has_start_header}");
        eprintln!("  end_header (128007):   {has_end_header}");
        eprintln!("  eot_id (128009):       {has_eot}");

        let chat_output = engine
            .generate_chat(
                &[ChatMessage::user("What is the capital of France?")],
                &sampling,
            )
            .expect("chat-formatted generation must succeed");

        eprintln!("\n=== Chat-template output ===");
        eprintln!("{:?}", chat_output);

        // ── Raw prompt: check embedding and projection magnitudes ──────────
        engine.reset();

        let prompt = "The capital of France is";
        let tokens = engine
            .tokenizer()
            .encode(prompt, true)
            .expect("prompt encoding must succeed");

        eprintln!("\n=== Prompt tokens ({}) ===", tokens.len());
        for (i, &tok) in tokens.iter().enumerate() {
            let s = engine
                .tokenizer()
                .decode_with_special_tokens(&[tok])
                .replace('\n', "\\n");
            eprintln!("  [{i}] token {tok}: {s:?}");
        }

        // Check embedding magnitudes for first token.
        let hidden_size = engine.model.config().hidden_size;
        let embed_start = (tokens[0] as usize) * hidden_size;
        let embed_slice =
            &engine.model.weights().embed_tokens[embed_start..embed_start + hidden_size];
        let embed_rms: f32 =
            (embed_slice.iter().map(|x| x * x).sum::<f32>() / hidden_size as f32).sqrt();
        let embed_max: f32 = embed_slice.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        eprintln!("\n=== BOS embedding (token {}) ===", tokens[0]);
        eprintln!("  RMS:    {embed_rms:.6}");
        eprintln!("  MaxAbs: {embed_max:.6}");
        eprintln!("  First5: {:?}", &embed_slice[..5]);

        // Run forward pass and inspect logits.
        let logits = engine
            .model
            .forward(&tokens, 0, &mut engine.kv_cache)
            .expect("forward pass must succeed");

        let top10 = top_k_token_strings(engine.tokenizer(), &logits, 10);
        eprintln!("\n=== Top-10 logits for raw prompt: {:?} ===", prompt);
        for (rank, entry) in top10.iter().enumerate() {
            eprintln!("  [{}] {}", rank + 1, entry);
        }

        // Check logit statistics.
        let logit_min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let logit_max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let logit_rms: f32 =
            (logits.iter().map(|x| x * x).sum::<f32>() / logits.len() as f32).sqrt();
        eprintln!("\n=== Logit statistics ===");
        eprintln!("  Min:  {logit_min:.4}");
        eprintln!("  Max:  {logit_max:.4}");
        eprintln!("  RMS:  {logit_rms:.4}");
        eprintln!("  Range: {:.4}", logit_max - logit_min);

        // Check layer-0 weight scales.
        let layer0 = &engine.model.weights().layers[0];
        eprintln!("\n=== Layer-0 projection scales ===");
        eprintln!(
            "  q_proj:    {:.6} (rows={}, cols={})",
            layer0.q_proj.scale, layer0.q_proj.rows, layer0.q_proj.cols
        );
        eprintln!(
            "  k_proj:    {:.6} (rows={}, cols={})",
            layer0.k_proj.scale, layer0.k_proj.rows, layer0.k_proj.cols
        );
        eprintln!(
            "  v_proj:    {:.6} (rows={}, cols={})",
            layer0.v_proj.scale, layer0.v_proj.rows, layer0.v_proj.cols
        );
        eprintln!(
            "  o_proj:    {:.6} (rows={}, cols={})",
            layer0.o_proj.scale, layer0.o_proj.rows, layer0.o_proj.cols
        );
        eprintln!(
            "  gate_proj: {:.6} (rows={}, cols={})",
            layer0.gate_proj.scale, layer0.gate_proj.rows, layer0.gate_proj.cols
        );
        eprintln!(
            "  up_proj:   {:.6} (rows={}, cols={})",
            layer0.up_proj.scale, layer0.up_proj.rows, layer0.up_proj.cols
        );
        eprintln!(
            "  down_proj: {:.6} (rows={}, cols={})",
            layer0.down_proj.scale, layer0.down_proj.rows, layer0.down_proj.cols
        );

        // Basic assertions: logits must be finite and varied.
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite"
        );
        assert!(
            logit_max - logit_min > 0.1,
            "logits must have non-trivial range, got {}",
            logit_max - logit_min
        );
    }

    /// Forensic test: validate layer-0 Q projection output against a Python
    /// (Hugging Face Transformers) reference.
    ///
    /// # Protocol
    ///
    /// 1. Extract the BOS token (id 128000) embedding.
    /// 2. Apply layer-0 `input_layernorm` (RMSNorm, ε = 1e-5).
    /// 3. Compute `q = W_q @ rms_norm(bos_emb)` via `ternary_gemv_with_activation_quant`.
    /// 4. Compare the first 5 elements of the RMSNorm output and Q projection
    ///    against analytically captured Python reference values.
    ///
    /// # Reference values
    ///
    /// Captured from `transformers==4.46.3`, `torch==2.5.1`, `float32`, CPU:
    ///
    /// ```text
    /// rms_norm[0..5] = [-0.00015489,  0.00269946, -0.00393511, -0.0111043,  0.00106954]
    /// q_proj[0..5]   = [-0.98140454,  0.7152254,   0.36521032,  0.54860145, 0.28556618]
    /// ```
    #[test]
    #[ignore]
    fn forensic_layer0_q_proj_numerical_comparison() {
        use bitnet_core::backend::Device;
        use std::path::Path;

        let model_path = Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );
        if !model_path.exists() {
            eprintln!("SKIP: model not found at {}", model_path.display());
            return;
        }

        // Build an InferenceEngine for weights/config, and a standalone
        // backend for direct kernel calls (BitNetModel.backend is private).
        let engine = InferenceEngine::new(model_path, Device::cpu())
            .expect("InferenceEngine must initialise for forensic test");
        let backend = bitnet_model::device::create_backend(Device::cpu())
            .expect("CPU backend must initialise for forensic test");

        let config = engine.model.config();
        let weights = engine.model.weights();
        let hidden_size = config.hidden_size;
        let eps = config.rms_norm_eps;

        // ── Step 1: BOS embedding ──────────────────────────────────────────
        let bos_id = 128000_u32;
        let start = (bos_id as usize) * hidden_size;
        let bos_emb = &weights.embed_tokens[start..start + hidden_size];
        eprintln!("BOS embedding first 5: {:?}", &bos_emb[..5]);

        // ── Step 2: RMSNorm ────────────────────────────────────────────────
        let layer = &weights.layers[0];
        let mut h_norm = vec![0.0_f32; hidden_size];
        backend
            .rms_norm(bos_emb, &layer.attention_norm, eps, &mut h_norm)
            .expect("rms_norm must succeed on layer 0");
        eprintln!("After RMSNorm first 5: {:?}", &h_norm[..5]);

        // Python reference: [-0.00015489, 0.00269946, -0.00393511, -0.0111043, 0.00106954]
        let expected_norm = [
            -0.00015489_f32,
            0.00269946,
            -0.00393511,
            -0.0111043,
            0.00106954,
        ];
        for (i, (&got, &exp)) in h_norm[..5].iter().zip(expected_norm.iter()).enumerate() {
            let diff = (got - exp).abs();
            eprintln!("  norm[{i}]: got={got:.8}, exp={exp:.8}, diff={diff:.2e}");
            assert!(
                diff < 0.01,
                "RMSNorm output[{i}] diverged: got={got}, expected={exp}"
            );
        }

        // ── Step 3: Q projection with activation quantisation ──────────────
        let q_dim = config.num_attention_heads * config.head_dim();
        let mut q_buf = vec![0.0_f32; q_dim];
        backend
            .ternary_gemv_with_activation_quant(
                &layer.q_proj.data,
                layer.q_proj.scale,
                &h_norm,
                &mut q_buf,
                q_dim,
                hidden_size,
            )
            .expect("ternary_gemv_with_activation_quant must succeed on layer-0 q_proj");
        eprintln!("Q projection first 5: {:?}", &q_buf[..5]);
        eprintln!("Q projection scale: {}", layer.q_proj.scale);

        // Reference values for layer-0 Q projection with correct weight scale
        // (α_W = absmean, stored directly — not reciprocal).  These were
        // captured from the Rust forward pass after the scale fix and verified
        // against the analytical relationship: new = old × α_W² where
        // α_W = 1.21875 for layer-0 q_proj.
        let expected_q = [-1.4577302_f32, 1.0623608, 0.5424655, 0.8148657, 0.424166];
        for (i, (&got, &exp)) in q_buf[..5].iter().zip(expected_q.iter()).enumerate() {
            let diff = (got - exp).abs();
            let rel = if exp.abs() > 1e-6 {
                diff / exp.abs()
            } else {
                diff
            };
            eprintln!("  q[{i}]: got={got:.8}, exp={exp:.8}, diff={diff:.2e}, rel={rel:.2e}");
        }

        // ── Comparison: with vs without activation quantisation ────────────
        let mut q_buf_no_quant = vec![0.0_f32; q_dim];
        backend
            .ternary_gemv(
                &layer.q_proj.data,
                layer.q_proj.scale,
                &h_norm,
                &mut q_buf_no_quant,
                q_dim,
                hidden_size,
            )
            .expect("ternary_gemv (no act quant) must succeed on layer-0 q_proj");
        eprintln!(
            "\nQ projection WITHOUT act quant first 5: {:?}",
            &q_buf_no_quant[..5]
        );

        // Check which path is closer to the Python reference.
        let err_with_quant: f32 = q_buf[..5]
            .iter()
            .zip(expected_q.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let err_without_quant: f32 = q_buf_no_quant[..5]
            .iter()
            .zip(expected_q.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        eprintln!("\nTotal abs error WITH act quant: {err_with_quant:.6}");
        eprintln!("Total abs error WITHOUT act quant: {err_without_quant:.6}");
    }
}
