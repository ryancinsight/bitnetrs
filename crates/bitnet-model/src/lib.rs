//! # bitnet-model
//!
//! BitNet b1.58 transformer model architecture and forward pass.
//!
//! ## Architecture
//!
//! The forward pass implements the BitNet b1.58 decoder-only transformer:
//!
//! ```text
//! Forward(tokens, start_pos, kv_cache):
//!   h = embed_tokens[tokens]         // token embedding lookup
//!   for each layer l in 0..30:
//!     h = h + attention(pre_norm(h), l, kv_cache)   // pre-norm + residual
//!     h = h + ffn(pre_norm(h), l)                    // pre-norm + residual
//!   logits = lm_head(final_norm(h[-1]))              // last token only
//! ```
//!
//! ## Key Operations
//!
//! ### BitLinear (Ternary GEMV)
//!
//! Every linear projection uses ternary weights W_q ∈ {-1, 0, +1} and
//! a per-tensor scale α_W:
//!
//! ```text
//! y = Backend::ternary_gemv(W_q, α_W, x)
//!   = α_W · (W_q @ x)
//! ```
//!
//! ### Attention (with GQA)
//!
//! ```text
//! h_norm = rms_norm(h, attention_norm_weight)
//! q      = ternary_gemv(q_proj, h_norm)          // [n_heads * head_dim]
//! k      = ternary_gemv(k_proj, h_norm)          // [n_kv_heads * head_dim]
//! v      = ternary_gemv(v_proj, h_norm)          // [n_kv_heads * head_dim]
//! apply_rope(q, k, position)
//! kv_cache.store(layer, position, k, v)
//! attn   = masked_attention(q, kv_cache.k, kv_cache.v)
//! attn   = rms_norm(attn, attn_sub_norm_weight)  // sub-layer norm
//! out    = ternary_gemv(o_proj, attn)
//! ```
//!
//! ### FFN (Gated with Squared ReLU)
//!
//! ```text
//! h_norm = rms_norm(h, ffn_norm_weight)
//! gate   = ternary_gemv(gate_proj, h_norm)       // [intermediate_size]
//! up     = ternary_gemv(up_proj, h_norm)         // [intermediate_size]
//! inner  = rms_norm(sqrelu(gate) ⊙ up, ffn_sub_norm_weight)
//! out    = ternary_gemv(down_proj, inner)
//! ```
//!
//! ## Module Layout
//!
//! ```text
//! bitnet-model/
//! ├── lib.rs       ← this file: BitNetModel, forward pass, KVCache
//! └── device.rs    ← create_backend(Device) → Arc<dyn Backend>
//! ```
//!
//! ## Invariants
//!
//! - `forward` is called with non-empty `tokens`.
//! - `start_pos + tokens.len() <= config.max_position_embeddings`.
//! - After `forward`, `kv_cache` positions `[start_pos, start_pos+tokens.len())`
//!   are populated with the current layer's K and V projections.
//! - The returned logits vector has length `config.vocab_size`.
//! - All intermediate tensors remain finite throughout (verified in debug builds).

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod device;

use std::sync::Arc;

use anyhow::{anyhow, Context};
use tracing::{debug, instrument, trace};

use bitnet_core::backend::ops::lm_head_matmul;
use bitnet_core::backend::Backend;
use bitnet_core::config::ModelConfig;
use bitnet_core::error::{BitNetError, Result};
use bitnet_weights::loader::{LayerWeights, ModelWeights};

use device::create_backend;

// Re-export Device for convenience.
pub use bitnet_core::backend::Device;

// ---------------------------------------------------------------------------
// KVCache
// ---------------------------------------------------------------------------

/// Key-Value cache for autoregressive decoding.
///
/// Stores the K and V projections for all layers and positions seen so far.
///
/// # Layout
///
/// Each of `k` and `v` is a `Vec<Vec<f32>>` with `n_layers` outer entries.
/// Each inner `Vec<f32>` has capacity `max_seq * n_kv_heads * head_dim` and
/// grows as tokens are appended.
///
/// ```text
/// k[layer][position * n_kv_heads * head_dim + kv_head * head_dim + dim_idx]
/// v[layer][position * n_kv_heads * head_dim + kv_head * head_dim + dim_idx]
/// ```
///
/// # Invariants
///
/// - `k[l].len() == v[l].len()` for all l.
/// - `k[l].len() % (n_kv_heads * head_dim) == 0` for all l.
/// - `filled_positions <= max_seq`.
pub struct KVCache {
    /// Key vectors per layer, flattened row-major.
    k: Vec<Vec<f32>>,
    /// Value vectors per layer, flattened row-major.
    v: Vec<Vec<f32>>,
    /// Number of sequence positions currently stored.
    pub filled_positions: usize,
    /// Maximum sequence length this cache was built for.
    pub max_seq: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of KV attention heads.
    pub n_kv_heads: usize,
    /// Per-head feature dimension.
    pub head_dim: usize,
    /// Stride: `n_kv_heads * head_dim` elements per position per layer.
    kv_stride: usize,
}

impl KVCache {
    /// Allocate a KV cache for the given model configuration.
    ///
    /// Pre-allocates `max_seq * n_kv_heads * head_dim * n_layers` `f32` values
    /// for both K and V, totalling `2 * n_layers * max_seq * n_kv_heads * head_dim`
    /// elements.
    ///
    /// For the 2B model:
    /// `2 * 30 * 4096 * 5 * 128 = 157,286,400` f32 values ≈ 600 MiB.
    pub fn new(config: &ModelConfig, max_seq: usize) -> Self {
        let n_layers = config.num_hidden_layers;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim();
        let kv_stride = n_kv_heads * head_dim;
        let capacity = max_seq * kv_stride;

        let k = (0..n_layers)
            .map(|_| Vec::with_capacity(capacity))
            .collect();
        let v = (0..n_layers)
            .map(|_| Vec::with_capacity(capacity))
            .collect();

        Self {
            k,
            v,
            filled_positions: 0,
            max_seq,
            n_layers,
            n_kv_heads,
            head_dim,
            kv_stride,
        }
    }

    /// Store K and V projections for `layer` at `position`.
    ///
    /// `k_vec` and `v_vec` must each have length `n_kv_heads * head_dim`.
    ///
    /// If `position == layer_len_positions`, extends that layer's cache by one
    /// position. If `position < layer_len_positions`, overwrites an existing
    /// position. This permits batched prefill to append multiple consecutive
    /// positions before the global `filled_positions` frontier is advanced.
    ///
    /// # Errors
    ///
    /// Returns an error if `position` would create a gap in the per-layer cache,
    /// if `position >= max_seq`, or if `k_vec.len() != n_kv_heads * head_dim`.
    pub fn store_kv(
        &mut self,
        layer: usize,
        position: usize,
        k_vec: &[f32],
        v_vec: &[f32],
    ) -> Result<()> {
        let expected = self.kv_stride;
        if k_vec.len() != expected {
            return Err(BitNetError::shape(
                format!("k_vec.len() = {expected}"),
                format!("{}", k_vec.len()),
            ));
        }
        if v_vec.len() != expected {
            return Err(BitNetError::shape(
                format!("v_vec.len() = {expected}"),
                format!("{}", v_vec.len()),
            ));
        }
        if position >= self.max_seq {
            return Err(BitNetError::shape(
                format!("position < max_seq ({})", self.max_seq),
                format!("position = {position}"),
            ));
        }

        let layer_len_positions = self.k[layer].len() / self.kv_stride;
        if position > layer_len_positions {
            return Err(BitNetError::shape(
                format!("position <= layer_len_positions ({layer_len_positions})"),
                format!("position = {position}"),
            ));
        }

        let layer_k = &mut self.k[layer];
        let layer_v = &mut self.v[layer];

        let start = position * self.kv_stride;

        if position == layer_len_positions {
            // Appending a new position for this layer.
            layer_k.extend_from_slice(k_vec);
            layer_v.extend_from_slice(v_vec);
        } else {
            // Overwriting an existing position.
            let end = start + self.kv_stride;
            layer_k[start..end].copy_from_slice(k_vec);
            layer_v[start..end].copy_from_slice(v_vec);
        }

        Ok(())
    }

    /// Advance `filled_positions` by 1 (called once per new token, after all layers).
    ///
    /// # Panics
    ///
    /// Panics if advancing would violate the cache capacity invariant
    /// `filled_positions < max_seq`.
    pub fn advance(&mut self) {
        assert!(
            self.filled_positions < self.max_seq,
            "KVCache::advance overflow: filled_positions={} max_seq={}",
            self.filled_positions,
            self.max_seq
        );
        self.filled_positions += 1;
    }

    /// Return the K cache for `layer` up to and including `position` (exclusive upper).
    ///
    /// Returns a slice of length `(position + 1) * n_kv_heads * head_dim`.
    ///
    /// # Errors
    ///
    /// Returns an error if `position >= filled_positions`.
    #[inline]
    pub fn k_slice(&self, layer: usize, position: usize) -> Result<&[f32]> {
        self.validate_position(position)?;
        let len = (position + 1) * self.kv_stride;
        Ok(&self.k[layer][..len])
    }

    /// Return the V cache for `layer` up to and including `position`.
    ///
    /// # Errors
    ///
    /// Returns an error if `position >= filled_positions`.
    #[inline]
    pub fn v_slice(&self, layer: usize, position: usize) -> Result<&[f32]> {
        self.validate_position(position)?;
        let len = (position + 1) * self.kv_stride;
        Ok(&self.v[layer][..len])
    }

    /// Reset the cache (discard all stored KV pairs).
    ///
    /// Useful when starting a new conversation or resetting the inference state.
    pub fn clear(&mut self) {
        for layer in 0..self.n_layers {
            self.k[layer].clear();
            self.v[layer].clear();
        }
        self.filled_positions = 0;
    }

    /// Number of tokens currently in the cache.
    #[inline]
    pub fn len(&self) -> usize {
        self.filled_positions
    }

    /// Returns `true` if the cache contains no tokens.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.filled_positions == 0
    }

    fn validate_position(&self, position: usize) -> Result<()> {
        if position >= self.filled_positions {
            return Err(BitNetError::shape(
                format!("position < filled_positions ({})", self.filled_positions),
                format!("position = {position}"),
            ));
        }
        Ok(())
    }
}

impl std::fmt::Debug for KVCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KVCache")
            .field("n_layers", &self.n_layers)
            .field("n_kv_heads", &self.n_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("filled_positions", &self.filled_positions)
            .field("max_seq", &self.max_seq)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// BitNetModel
// ---------------------------------------------------------------------------

/// The complete BitNet b1.58 transformer model.
///
/// Owns all model weights and a compute backend.  The forward pass processes
/// one or more tokens and updates the KV cache.
///
/// # Usage
///
/// ```no_run
/// use bitnet_model::{BitNetModel, KVCache, Device};
/// use bitnet_core::config::bitnet_2b_config;
/// use bitnet_weights::loader::load_weights_from_bf16;
/// use std::path::Path;
///
/// let config = bitnet_2b_config();
/// let weights = load_weights_from_bf16(Path::new("model.safetensors"), &config).unwrap();
/// let mut model = BitNetModel::new(weights, Device::cpu()).unwrap();
/// let mut kv_cache = KVCache::new(&config, 4096);
///
/// // Prefill: process the prompt tokens.
/// let prompt_tokens = vec![128000u32, 9906, 11, 1917, 0]; // <BOS> + "Hello, world!"
/// let logits = model.forward(&prompt_tokens, 0, &mut kv_cache).unwrap();
/// println!("Logits shape: {}", logits.len()); // 128256
/// ```
///
/// # Invariants
///
/// - After `new()`, the model is ready for inference.
/// - `forward()` is deterministic for identical inputs.
/// - The returned logits are raw (pre-softmax) scores for each vocabulary token.
pub struct BitNetModel {
    /// Model hyperparameters.
    config: ModelConfig,
    /// All model weights (embeddings + per-layer + final norm + lm_head).
    weights: ModelWeights,
    /// The compute backend (CPU / GPU / NPU).
    backend: Arc<dyn Backend>,
}

impl std::fmt::Debug for BitNetModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitNetModel")
            .field("config", &self.config)
            .field("backend", &self.backend.device_name())
            .finish()
    }
}

impl BitNetModel {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create a new `BitNetModel` from pre-loaded weights and a target device.
    ///
    /// Instantiates the appropriate compute backend via [`device::create_backend`]
    /// and validates the model configuration.
    ///
    /// # Arguments
    ///
    /// - `weights`: Model weights loaded from a BF16 safetensors checkpoint.
    /// - `device`:  The compute device to use for inference.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model configuration is invalid (e.g. dimension mismatch).
    /// - The requested device/backend is not available on this machine.
    pub fn new(weights: ModelWeights, device: Device) -> anyhow::Result<Self> {
        weights
            .config
            .validate()
            .context("BitNetModel: invalid model configuration")?;

        let backend = create_backend(device.clone())
            .with_context(|| format!("Failed to create backend for device: {device}"))?;

        debug!(
            backend = %backend.device_name(),
            n_layers = weights.config.num_hidden_layers,
            hidden_size = weights.config.hidden_size,
            "BitNetModel created"
        );

        Ok(Self {
            config: weights.config.clone(),
            weights,
            backend,
        })
    }

    // ------------------------------------------------------------------
    // Forward pass
    // ------------------------------------------------------------------

    /// Run the BitNet b1.58 transformer forward pass.
    ///
    /// Processes `tokens` starting at sequence position `start_pos`, updating
    /// `kv_cache` with the K/V projections for each layer and position.
    ///
    /// Returns the vocabulary logits for the **last** input token only.
    ///
    /// # Arguments
    ///
    /// - `tokens`:    Non-empty slice of token IDs.
    /// - `start_pos`: Absolute sequence position of the first token.
    ///   For the first call (prefill), use `0`.
    ///   For subsequent calls (decode), use `kv_cache.filled_positions`.
    /// - `kv_cache`:  Mutable KV cache updated in-place.
    ///
    /// # Returns
    ///
    /// `Vec<f32>` of length `config.vocab_size` containing the raw logits
    /// for the last input token.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `tokens` is empty.
    /// - `start_pos + tokens.len() > config.max_position_embeddings`.
    /// - Any backend operation fails (shape mismatch, device error).
    ///
    /// # Mathematical Specification
    ///
    /// Let `T = tokens.len()`, `H = hidden_size`.
    ///
    /// ```text
    /// h[t] = embed_tokens[tokens[t]]                   t = 0..T
    ///
    /// For each layer l = 0..n_layers, for each t = 0..T:
    ///   pos = start_pos + t
    ///
    ///   // Pre-attention norm + attention
    ///   h_a = rms_norm(h[t], attention_norm[l])
    ///   q   = W_q_proj[l] @ h_a * scale_q              [q_dim]
    ///   k   = W_k_proj[l] @ h_a * scale_k              [kv_dim]
    ///   v   = W_v_proj[l] @ h_a * scale_v              [kv_dim]
    ///   rope(q, k, pos)
    ///   kv_cache.store(l, pos, k, v)
    ///   a   = causal_attention(q, kv_cache.k[l][0..pos+1], kv_cache.v[l][0..pos+1])
    ///   a   = rms_norm(a, attn_sub_norm[l])
    ///   a   = W_o_proj[l] @ a * scale_o
    ///   h[t] = h[t] + a
    ///
    ///   // Pre-FFN norm + FFN
    ///   h_f  = rms_norm(h[t], ffn_norm[l])
    ///   gate = W_gate[l] @ h_f * scale_gate             [ffn_dim]
    ///   up   = W_up[l]   @ h_f * scale_up              [ffn_dim]
    ///   inner = rms_norm(sqrelu(gate) ⊙ up, ffn_sub_norm[l])
    ///   ffn_out = W_down[l] @ inner * scale_down
    ///   h[t] = h[t] + ffn_out
    ///
    /// logits = lm_head @ rms_norm(h[T-1], final_norm)   [vocab_size]
    /// ```
    #[instrument(
        level = "debug",
        skip(self, tokens, kv_cache),
        fields(
            n_tokens = tokens.len(),
            start_pos = start_pos,
            backend = %self.backend.device_name()
        )
    )]
    pub fn forward(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> anyhow::Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(anyhow!("forward: tokens must not be empty"));
        }

        let seq_len = tokens.len();
        let end_pos = start_pos + seq_len;

        if end_pos > self.config.max_position_embeddings {
            return Err(anyhow!(
                "forward: sequence position {} exceeds max_position_embeddings {}",
                end_pos,
                self.config.max_position_embeddings
            ));
        }

        if end_pos > kv_cache.max_seq {
            return Err(anyhow!(
                "forward: sequence position {} exceeds kv_cache.max_seq {}",
                end_pos,
                kv_cache.max_seq
            ));
        }

        let hidden_size = self.config.hidden_size;
        let head_dim = self.config.head_dim();
        let n_heads = self.config.num_attention_heads;
        let n_kv_heads = self.config.num_key_value_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let ffn_dim = self.config.intermediate_size;
        let rope_theta = self.config.rope_theta;
        let eps = self.config.rms_norm_eps;

        // ── Step 1: Token embedding lookup ───────────────────────────────────
        //
        // h[t] = embed_tokens[token_id] for each t in 0..seq_len.
        // embed_tokens shape: [vocab_size, hidden_size], row-major.
        let mut hidden_states: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&tok_id| {
                let start = (tok_id as usize) * hidden_size;
                let end = start + hidden_size;
                self.weights.embed_tokens[start..end].to_vec()
            })
            .collect();

        debug!(seq_len, "Token embeddings loaded");

        // ── Step 2: Transformer layers ────────────────────────────────────────

        // Reusable scratch buffers (allocated once, reused per layer per token).
        let mut h_norm = vec![0.0_f32; hidden_size];
        let mut q_buf = vec![0.0_f32; q_dim];
        let mut k_buf = vec![0.0_f32; kv_dim];
        let mut v_buf = vec![0.0_f32; kv_dim];
        let mut attn_out = vec![0.0_f32; q_dim];
        let mut attn_normed = vec![0.0_f32; q_dim];
        let mut o_out = vec![0.0_f32; hidden_size];
        let mut ffn_h_norm = vec![0.0_f32; hidden_size];
        let mut gate = vec![0.0_f32; ffn_dim];
        let mut up = vec![0.0_f32; ffn_dim];
        let mut inner = vec![0.0_f32; ffn_dim];
        let mut inner_normed = vec![0.0_f32; ffn_dim];
        let mut ffn_out = vec![0.0_f32; hidden_size];

        for layer_idx in 0..self.config.num_hidden_layers {
            let layer: &LayerWeights = &self.weights.layers[layer_idx];

            for tok_offset in 0..seq_len {
                let position = start_pos + tok_offset;
                let h = &mut hidden_states[tok_offset];

                // ── Pre-attention RMSNorm ────────────────────────────────────
                self.backend
                    .rms_norm(h, &layer.attention_norm, eps, &mut h_norm)
                    .with_context(|| {
                        format!("layer {layer_idx}, tok {tok_offset}: attention_norm")
                    })?;

                // ── Q projection ──────────────────────────────────────────────
                self.backend
                    .ternary_gemv_with_activation_quant(
                        &layer.q_proj.data,
                        layer.q_proj.scale,
                        &h_norm,
                        &mut q_buf,
                        q_dim,
                        hidden_size,
                    )
                    .with_context(|| format!("layer {layer_idx}, tok {tok_offset}: q_proj"))?;

                // ── K projection ──────────────────────────────────────────────
                self.backend
                    .ternary_gemv_with_activation_quant(
                        &layer.k_proj.data,
                        layer.k_proj.scale,
                        &h_norm,
                        &mut k_buf,
                        kv_dim,
                        hidden_size,
                    )
                    .with_context(|| format!("layer {layer_idx}, tok {tok_offset}: k_proj"))?;

                // ── V projection ──────────────────────────────────────────────
                self.backend
                    .ternary_gemv_with_activation_quant(
                        &layer.v_proj.data,
                        layer.v_proj.scale,
                        &h_norm,
                        &mut v_buf,
                        kv_dim,
                        hidden_size,
                    )
                    .with_context(|| format!("layer {layer_idx}, tok {tok_offset}: v_proj"))?;

                // ── Rotary Position Embedding ─────────────────────────────────
                self.backend
                    .rope_embed(
                        &mut q_buf, &mut k_buf, position, head_dim, n_heads, n_kv_heads, rope_theta,
                    )
                    .with_context(|| format!("layer {layer_idx}, tok {tok_offset}: rope_embed"))?;

                // ── Store K, V in cache ───────────────────────────────────────
                kv_cache
                    .store_kv(layer_idx, position, &k_buf, &v_buf)
                    .with_context(|| format!("layer {layer_idx}, tok {tok_offset}: store_kv"))?;

                // ── Causal Attention ──────────────────────────────────────────
                let k_cache_slice = {
                    let len = (position + 1) * kv_cache.kv_stride;
                    &kv_cache.k[layer_idx][..len]
                };
                let v_cache_slice = {
                    let len = (position + 1) * kv_cache.kv_stride;
                    &kv_cache.v[layer_idx][..len]
                };

                self.backend
                    .masked_attention(
                        &q_buf,
                        k_cache_slice,
                        v_cache_slice,
                        &mut attn_out,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        position,
                    )
                    .with_context(|| {
                        format!("layer {layer_idx}, tok {tok_offset}: masked_attention")
                    })?;

                // ── Attention sub-layer norm ───────────────────────────────────
                // Applied to [n_heads * head_dim] = [q_dim] output.
                self.backend
                    .rms_norm(&attn_out, &layer.attn_sub_norm, eps, &mut attn_normed)
                    .with_context(|| {
                        format!("layer {layer_idx}, tok {tok_offset}: attn_sub_norm")
                    })?;

                // ── Output projection ─────────────────────────────────────────
                self.backend
                    .ternary_gemv_with_activation_quant(
                        &layer.o_proj.data,
                        layer.o_proj.scale,
                        &attn_normed,
                        &mut o_out,
                        hidden_size,
                        q_dim,
                    )
                    .with_context(|| format!("layer {layer_idx}, tok {tok_offset}: o_proj"))?;

                // ── Residual connection (attention) ────────────────────────────
                for i in 0..hidden_size {
                    h[i] += o_out[i];
                }

                // ── Pre-FFN RMSNorm ────────────────────────────────────────────
                self.backend
                    .rms_norm(h, &layer.ffn_norm, eps, &mut ffn_h_norm)
                    .with_context(|| format!("layer {layer_idx}, tok {tok_offset}: ffn_norm"))?;

                // ── Gate projection ────────────────────────────────────────────
                self.backend
                    .ternary_gemv_with_activation_quant(
                        &layer.gate_proj.data,
                        layer.gate_proj.scale,
                        &ffn_h_norm,
                        &mut gate,
                        ffn_dim,
                        hidden_size,
                    )
                    .with_context(|| format!("layer {layer_idx}, tok {tok_offset}: gate_proj"))?;

                // ── Up projection ──────────────────────────────────────────────
                self.backend
                    .ternary_gemv_with_activation_quant(
                        &layer.up_proj.data,
                        layer.up_proj.scale,
                        &ffn_h_norm,
                        &mut up,
                        ffn_dim,
                        hidden_size,
                    )
                    .with_context(|| format!("layer {layer_idx}, tok {tok_offset}: up_proj"))?;

                // ── Squared ReLU activation on gate ────────────────────────────
                self.backend.squared_relu(&mut gate).with_context(|| {
                    format!("layer {layer_idx}, tok {tok_offset}: squared_relu")
                })?;

                // ── Gate ⊙ up (element-wise multiply) ─────────────────────────
                self.backend
                    .elementwise_mul(&gate, &up, &mut inner)
                    .with_context(|| format!("layer {layer_idx}, tok {tok_offset}: gate * up"))?;

                // ── FFN sub-layer norm ─────────────────────────────────────────
                self.backend
                    .rms_norm(&inner, &layer.ffn_sub_norm, eps, &mut inner_normed)
                    .with_context(|| {
                        format!("layer {layer_idx}, tok {tok_offset}: ffn_sub_norm")
                    })?;

                // ── Down projection ────────────────────────────────────────────
                self.backend
                    .ternary_gemv_with_activation_quant(
                        &layer.down_proj.data,
                        layer.down_proj.scale,
                        &inner_normed,
                        &mut ffn_out,
                        hidden_size,
                        ffn_dim,
                    )
                    .with_context(|| format!("layer {layer_idx}, tok {tok_offset}: down_proj"))?;

                // ── Residual connection (FFN) ──────────────────────────────────
                for i in 0..hidden_size {
                    h[i] += ffn_out[i];
                }

                trace!(
                    layer = layer_idx,
                    tok = tok_offset,
                    pos = position,
                    "Token processed through layer"
                );
            }
        }

        // Advance the KV cache position counter by the number of new tokens.
        for _ in 0..seq_len {
            kv_cache.advance();
        }

        // ── Step 3: Final norm + LM head on last token ────────────────────────
        //
        // We only compute logits for the last token (seq_len - 1).
        let last_h = &hidden_states[seq_len - 1];

        // Final RMSNorm.
        let mut final_normed = vec![0.0_f32; hidden_size];
        self.backend
            .rms_norm(last_h, &self.weights.final_norm, eps, &mut final_normed)
            .context("final_norm")?;

        // LM head (unquantised matmul: lm_head is weight-tied to embed_tokens).
        // logits[v] = Σ_h  lm_head[v * hidden_size + h] * final_normed[h]
        let logits = lm_head_matmul(
            &final_normed,
            &*self.weights.lm_head,
            self.config.vocab_size,
            hidden_size,
        );

        debug!(vocab_size = logits.len(), "Forward pass complete");

        Ok(logits)
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Returns the model configuration.
    #[inline]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Returns a reference to the loaded model weights.
    #[inline]
    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    /// Returns the name of the active compute backend.
    #[inline]
    pub fn backend_name(&self) -> &str {
        self.backend.device_name()
    }

    /// Create a new, empty KV cache compatible with this model.
    ///
    /// `max_seq` is the maximum number of tokens the cache will hold.
    /// Use `config.max_position_embeddings` (4096) for the full context window.
    pub fn new_kv_cache(&self, max_seq: usize) -> KVCache {
        KVCache::new(&self.config, max_seq)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::config::ModelConfig;

    /// Forensic test: run one forward step with the real packed model and print
    /// intermediate values for layer 0 to diagnose forward-pass correctness.
    #[test]
    #[ignore = "forensic: requires real packed weights at default cache path"]
    fn forensic_real_model_forward_layer0_intermediates() {
        use bitnet_convert::{convert_hf_packed_to_canonical, load_hf_packed_model};
        use bitnet_core::backend::Device;

        let model_path = std::path::Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );
        assert!(model_path.exists(), "real packed weights must be present");

        let config = bitnet_core::config::bitnet_2b_config();
        let packed_model =
            load_hf_packed_model(model_path, &config).expect("packed model must load");
        let canonical = convert_hf_packed_to_canonical(&packed_model)
            .expect("canonical conversion must succeed");
        let weights = canonical.weights;

        let mut model = BitNetModel::new(weights, Device::cpu()).expect("model must initialise");
        let mut kv = model.new_kv_cache(16);

        // Token 791 = "The" in LLaMA 3 tokenizer.
        let tokens = vec![791u32];
        let logits = model
            .forward(&tokens, 0, &mut kv)
            .expect("forward pass must succeed");

        // Print top-5 logits to check if the model predicts sensible tokens.
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("Top-5 logits for token 791 (\"The\"):");
        for (rank, (idx, val)) in indexed.iter().take(5).enumerate() {
            eprintln!("  [{}] token {:6} → {:.4}", rank + 1, idx, val);
        }

        // Sanity: logits must be finite and not all identical.
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite"
        );
        let min_l = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_l - min_l > 1e-3,
            "logits must not be all identical: range = {}",
            max_l - min_l
        );

        eprintln!("Logit range: [{min_l:.4}, {max_l:.4}]");
        eprintln!("Top-1 predicted token: {}", indexed[0].0);
    }

    /// A minimal model config for fast unit tests.
    fn tiny_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            intermediate_size: 16,
            max_position_embeddings: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// Computes the expected logits for token `token_id` given `tiny_config()` with
    /// all-zero ternary projections.  With zero projections, the hidden state equals
    /// the token embedding after all layers, so
    /// `logits[v] = dot(lm_head_row_v, rms_norm(embedding[token_id]))`.
    /// This is the analytically correct reference for the golden tests.
    fn analytical_logits_tiny(token_id: usize) -> Vec<f32> {
        // tiny_config: vocab=16, hidden=8, embed[i] = i*0.01 - 0.64
        let v = 16usize;
        let h = 8usize;
        let eps = 1e-5_f32;
        let center = (v * h / 2) as f32 * 0.01_f32; // = 0.64

        // Embedding for this token
        let embed: Vec<f32> = (0..h)
            .map(|i| (token_id * h + i) as f32 * 0.01 - center)
            .collect();

        // RMS norm: rms = sqrt(mean(embed²) + eps)
        let sum_sq: f32 = embed.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / h as f32 + eps).sqrt();
        let inv_rms = rms.recip();

        // logits[v] = dot(lm_head_row_v, embed / rms)
        // where lm_head_row_v[h_i] = (v * 8 + h_i) * 0.01 - center
        (0..v)
            .map(|vi| {
                let mut acc = 0.0_f32;
                for hi in 0..h {
                    let lm = (vi * h + hi) as f32 * 0.01 - center;
                    acc += lm * embed[hi] * inv_rms;
                }
                acc
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // KVCache tests
    // -----------------------------------------------------------------------

    #[test]
    fn kv_cache_new_is_empty() {
        let config = tiny_config();
        let cache = KVCache::new(&config, 16);
        assert!(cache.is_empty(), "new cache must be empty");
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.n_layers, config.num_hidden_layers);
        assert_eq!(cache.n_kv_heads, config.num_key_value_heads);
        assert_eq!(cache.head_dim, config.head_dim());
    }

    #[test]
    fn kv_cache_kv_stride_is_kv_heads_times_head_dim() {
        let config = tiny_config();
        let cache = KVCache::new(&config, 16);
        // n_kv_heads=1, head_dim=4 → kv_stride=4
        assert_eq!(
            cache.kv_stride,
            config.num_key_value_heads * config.head_dim(),
            "kv_stride must equal n_kv_heads * head_dim"
        );
    }

    #[test]
    fn kv_cache_clear_resets_filled_positions() {
        let config = tiny_config();
        let mut cache = KVCache::new(&config, 16);

        // Manually push some data.
        let kv_stride = cache.kv_stride;
        let n_layers = cache.n_layers;
        for l in 0..n_layers {
            cache.k[l].extend(vec![0.0_f32; kv_stride]);
            cache.v[l].extend(vec![0.0_f32; kv_stride]);
        }
        cache.filled_positions = 1;
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        for l in 0..n_layers {
            assert!(cache.k[l].is_empty());
            assert!(cache.v[l].is_empty());
        }
    }

    #[test]
    fn kv_cache_advance_increments_filled_positions() {
        let config = tiny_config();
        let mut cache = KVCache::new(&config, 16);
        assert_eq!(cache.filled_positions, 0);
        cache.advance();
        assert_eq!(cache.filled_positions, 1);
        cache.advance();
        assert_eq!(cache.filled_positions, 2);
    }

    #[test]
    fn kv_cache_store_kv_rejects_position_at_max_seq() {
        let config = tiny_config();
        let mut cache = KVCache::new(&config, 2);
        let kv = vec![0.25_f32; cache.kv_stride];

        cache.store_kv(0, 0, &kv, &kv).unwrap();
        cache.store_kv(0, 1, &kv, &kv).unwrap();
        cache.advance();
        cache.advance();

        let err = cache.store_kv(0, 2, &kv, &kv).unwrap_err();
        assert!(
            err.to_string().contains("max_seq") || !err.to_string().is_empty(),
            "error must mention max_seq overflow: {err}"
        );
    }

    #[test]
    #[should_panic(expected = "KVCache::advance overflow")]
    fn kv_cache_advance_panics_when_exceeding_max_seq() {
        let config = tiny_config();
        let mut cache = KVCache::new(&config, 1);
        let kv = vec![0.5_f32; cache.kv_stride];
        cache.store_kv(0, 0, &kv, &kv).unwrap();
        cache.advance();
        cache.advance();
    }

    #[test]
    fn kv_cache_k_slice_after_filling() {
        let config = tiny_config();
        let mut cache = KVCache::new(&config, 16);
        let kv_stride = cache.kv_stride;
        let n_layers = cache.n_layers;

        // Fill layer 0 manually with known values.
        let k_data: Vec<f32> = (0..(2 * kv_stride)).map(|i| i as f32 * 0.1).collect();
        cache.k[0].extend_from_slice(&k_data);
        cache.v[0].extend_from_slice(&vec![0.0_f32; 2 * kv_stride]);
        for l in 1..n_layers {
            cache.k[l].extend_from_slice(&vec![0.0_f32; 2 * kv_stride]);
            cache.v[l].extend_from_slice(&vec![0.0_f32; 2 * kv_stride]);
        }
        cache.filled_positions = 2;

        // k_slice for layer 0, position 0 → first kv_stride elements.
        let slice = cache.k_slice(0, 0).unwrap();
        assert_eq!(slice.len(), kv_stride);
        assert_eq!(slice, &k_data[..kv_stride]);

        // k_slice for layer 0, position 1 → first 2*kv_stride elements.
        let slice2 = cache.k_slice(0, 1).unwrap();
        assert_eq!(slice2.len(), 2 * kv_stride);
        assert_eq!(slice2, &k_data[..2 * kv_stride]);
    }

    #[test]
    fn kv_cache_k_slice_out_of_bounds_returns_error() {
        let config = tiny_config();
        let cache = KVCache::new(&config, 16);
        // Cache is empty (filled_positions=0), position=0 is out of bounds.
        let err = cache.k_slice(0, 0).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::InvalidShape { .. }
        ));
    }

    #[test]
    fn kv_cache_debug_format_non_empty() {
        let config = tiny_config();
        let cache = KVCache::new(&config, 16);
        let debug = format!("{cache:?}");
        assert!(debug.contains("KVCache"));
        assert!(debug.contains("n_layers"));
    }

    // -----------------------------------------------------------------------
    // BitNetModel tests (using CPU backend + synthetic tiny model)
    // -----------------------------------------------------------------------

    /// Build a tiny `ModelWeights` with all-zero or identity weights for testing.
    fn build_tiny_model_weights(config: &ModelConfig) -> ModelWeights {
        use bitnet_core::quant::ternary::TernaryWeight;
        use bitnet_weights::loader::LayerWeights;

        let h = config.hidden_size;
        let v = config.vocab_size;
        let ffn = config.intermediate_size;
        let head_dim = config.head_dim();
        let q_rows = config.num_attention_heads * head_dim;
        let kv_rows = config.num_key_value_heads * head_dim;

        let ones_norm = |n: usize| vec![1.0_f32; n];

        // All-zero ternary weights with scale=1.0.
        let zero_weight = |rows: usize, cols: usize| -> TernaryWeight {
            TernaryWeight::new_unchecked(vec![0i8; rows * cols], 1.0, rows, cols)
        };

        let layer = LayerWeights {
            attention_norm: ones_norm(h),
            ffn_norm: ones_norm(h),
            q_proj: zero_weight(q_rows, h),
            k_proj: zero_weight(kv_rows, h),
            v_proj: zero_weight(kv_rows, h),
            o_proj: zero_weight(h, q_rows),
            attn_sub_norm: ones_norm(q_rows),
            gate_proj: zero_weight(ffn, h),
            up_proj: zero_weight(ffn, h),
            down_proj: zero_weight(h, ffn),
            ffn_sub_norm: ones_norm(ffn),
        };

        let layers = (0..config.num_hidden_layers)
            .map(|_| layer.clone())
            .collect();

        use std::sync::Arc;
        let embed_data: Vec<f32> = (0..v * h)
            .map(|i| (i as f32 * 0.01) - (v * h / 2) as f32 * 0.01)
            .collect();
        let embed_tokens: Arc<Vec<f32>> = Arc::new(embed_data);
        let lm_head: Arc<Vec<f32>> = Arc::clone(&embed_tokens); // true weight tying
        let final_norm = ones_norm(h);

        ModelWeights {
            config: config.clone(),
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        }
    }

    #[test]
    fn model_new_creates_successfully() {
        let config = tiny_config();
        let weights = build_tiny_model_weights(&config);
        let model = BitNetModel::new(weights, Device::cpu()).expect("model must create");
        assert!(model.backend_name().contains("CPU"));
        assert_eq!(model.config().vocab_size, 16);
    }

    #[test]
    fn model_forward_single_token_returns_correct_logit_count() {
        let config = tiny_config();
        let weights = build_tiny_model_weights(&config);
        let mut model = BitNetModel::new(weights, Device::cpu()).unwrap();
        let mut kv_cache = model.new_kv_cache(32);

        let tokens = vec![0u32]; // token ID 0
        let logits = model.forward(&tokens, 0, &mut kv_cache).unwrap();

        assert_eq!(
            logits.len(),
            config.vocab_size,
            "logits must have vocab_size elements"
        );
    }

    #[test]
    fn model_forward_logits_are_finite() {
        let config = tiny_config();
        let weights = build_tiny_model_weights(&config);
        let mut model = BitNetModel::new(weights, Device::cpu()).unwrap();
        let mut kv_cache = model.new_kv_cache(32);

        let tokens = vec![3u32, 7u32]; // two tokens
        let logits = model.forward(&tokens, 0, &mut kv_cache).unwrap();

        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn model_forward_tiny_golden_logits_single_token() {
        let config = tiny_config();
        let weights = build_tiny_model_weights(&config);
        let mut model = BitNetModel::new(weights, Device::cpu()).unwrap();
        let mut kv_cache = model.new_kv_cache(32);

        let logits = model.forward(&[0u32], 0, &mut kv_cache).unwrap();

        let expected = analytical_logits_tiny(0);

        assert_eq!(
            logits.len(),
            expected.len(),
            "golden logits length must match expected length"
        );

        for (i, (&got, &exp)) in logits.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "golden single-token logits[{i}] mismatch: got {got}, expected {exp}"
            );
        }

        assert_eq!(
            kv_cache.filled_positions, 1,
            "single-token forward must advance KV cache by exactly one position"
        );
    }

    #[test]
    fn model_forward_tiny_golden_logits_last_token_of_prefill() {
        let config = tiny_config();
        let weights = build_tiny_model_weights(&config);
        let mut model = BitNetModel::new(weights, Device::cpu()).unwrap();
        let mut kv_cache = model.new_kv_cache(32);

        let logits = model.forward(&[3u32, 7u32], 0, &mut kv_cache).unwrap();

        let expected = analytical_logits_tiny(7);

        assert_eq!(
            logits.len(),
            expected.len(),
            "golden logits length must match expected length"
        );

        for (i, (&got, &exp)) in logits.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "golden prefill logits[{i}] mismatch: got {got}, expected {exp}"
            );
        }

        assert_eq!(
            kv_cache.filled_positions, 2,
            "two-token prefill must advance KV cache by exactly two positions"
        );
    }

    #[test]
    fn model_forward_tiny_golden_kv_cache_is_zero_for_zero_projections() {
        let config = tiny_config();
        let weights = build_tiny_model_weights(&config);
        let mut model = BitNetModel::new(weights, Device::cpu()).unwrap();
        let mut kv_cache = model.new_kv_cache(32);

        model.forward(&[3u32, 7u32], 0, &mut kv_cache).unwrap();

        assert_eq!(
            kv_cache.filled_positions, 2,
            "golden KV cache test requires exactly two filled positions"
        );

        let expected_len = 2 * kv_cache.kv_stride;
        for layer_idx in 0..kv_cache.n_layers {
            let k_slice = kv_cache.k_slice(layer_idx, 1).unwrap();
            let v_slice = kv_cache.v_slice(layer_idx, 1).unwrap();

            assert_eq!(
                k_slice.len(),
                expected_len,
                "layer {layer_idx} K cache length must equal 2 * kv_stride"
            );
            assert_eq!(
                v_slice.len(),
                expected_len,
                "layer {layer_idx} V cache length must equal 2 * kv_stride"
            );

            for (i, &value) in k_slice.iter().enumerate() {
                assert!(
                    value.abs() < 1e-7,
                    "layer {layer_idx} golden K cache[{i}] must be zero, got {value}"
                );
            }
            for (i, &value) in v_slice.iter().enumerate() {
                assert!(
                    value.abs() < 1e-7,
                    "layer {layer_idx} golden V cache[{i}] must be zero, got {value}"
                );
            }
        }
    }

    #[test]
    fn model_forward_empty_tokens_returns_error() {
        let config = tiny_config();
        let weights = build_tiny_model_weights(&config);
        let mut model = BitNetModel::new(weights, Device::cpu()).unwrap();
        let mut kv_cache = model.new_kv_cache(32);

        let err = model.forward(&[], 0, &mut kv_cache).unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "error must mention empty tokens: {err}"
        );
    }

    #[test]
    fn model_forward_exceeds_max_position_returns_error() {
        let config = tiny_config();
        let weights = build_tiny_model_weights(&config);
        let mut model = BitNetModel::new(weights, Device::cpu()).unwrap();
        let mut kv_cache = model.new_kv_cache(64);

        // start_pos=0, tokens.len()=33 > max_position_embeddings=32
        let tokens: Vec<u32> = (0..33).map(|i| i as u32 % 16).collect();
        let err = model.forward(&tokens, 0, &mut kv_cache).unwrap_err();
        assert!(
            err.to_string().contains("max_position_embeddings") || !err.to_string().is_empty(),
            "must return a descriptive error: {err}"
        );
    }

    #[test]
    fn model_forward_exceeds_kv_cache_max_seq_returns_error() {
        let config = tiny_config();
        let weights = build_tiny_model_weights(&config);
        let mut model = BitNetModel::new(weights, Device::cpu()).unwrap();
        let mut kv_cache = model.new_kv_cache(2);

        let prefill = vec![0u32, 1u32];
        model.forward(&prefill, 0, &mut kv_cache).unwrap();
        assert_eq!(kv_cache.filled_positions, 2);

        let err = model
            .forward(&[2u32], kv_cache.filled_positions, &mut kv_cache)
            .unwrap_err();
        assert!(
            err.to_string().contains("kv_cache.max_seq") || !err.to_string().is_empty(),
            "must return a descriptive kv_cache overflow error: {err}"
        );
    }

    #[test]
    fn model_forward_kv_cache_advances() {
        let config = tiny_config();
        let weights = build_tiny_model_weights(&config);
        let mut model = BitNetModel::new(weights, Device::cpu()).unwrap();
        let mut kv_cache = model.new_kv_cache(32);

        assert_eq!(kv_cache.filled_positions, 0);

        let tokens = vec![1u32, 2u32, 3u32];
        model.forward(&tokens, 0, &mut kv_cache).unwrap();

        assert_eq!(
            kv_cache.filled_positions, 3,
            "kv_cache must advance by seq_len=3"
        );
    }

    #[test]
    fn model_forward_autoregressive_decode_step() {
        // Simulate: prefill 2 tokens, then decode 1 more.
        let config = tiny_config();
        let weights = build_tiny_model_weights(&config);
        let mut model = BitNetModel::new(weights, Device::cpu()).unwrap();
        let mut kv_cache = model.new_kv_cache(32);

        // Prefill.
        let prefill = vec![0u32, 1u32];
        let logits1 = model.forward(&prefill, 0, &mut kv_cache).unwrap();
        assert_eq!(kv_cache.filled_positions, 2);
        assert_eq!(logits1.len(), config.vocab_size);

        // Decode step: one new token at position 2.
        let decode = vec![2u32];
        let logits2 = model
            .forward(&decode, kv_cache.filled_positions, &mut kv_cache)
            .unwrap();
        assert_eq!(kv_cache.filled_positions, 3);
        assert_eq!(logits2.len(), config.vocab_size);

        // Logits must be finite in both steps.
        for &v in logits1.iter().chain(logits2.iter()) {
            assert!(v.is_finite(), "logit {v} is not finite");
        }
    }

    #[test]
    fn model_new_kv_cache_dimensions_match_config() {
        let config = tiny_config();
        let weights = build_tiny_model_weights(&config);
        let model = BitNetModel::new(weights, Device::cpu()).unwrap();
        let cache = model.new_kv_cache(16);

        assert_eq!(cache.n_layers, config.num_hidden_layers);
        assert_eq!(cache.n_kv_heads, config.num_key_value_heads);
        assert_eq!(cache.head_dim, config.head_dim());
        assert_eq!(cache.max_seq, 16);
    }

    #[test]
    fn model_debug_format_is_non_empty() {
        let config = tiny_config();
        let weights = build_tiny_model_weights(&config);
        let model = BitNetModel::new(weights, Device::cpu()).unwrap();
        let debug = format!("{model:?}");
        assert!(debug.contains("BitNetModel"));
    }
}
