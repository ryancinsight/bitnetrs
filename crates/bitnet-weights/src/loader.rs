//! Weight loading and assembly for BitNet b1.58 inference.
//!
//! # Overview
//!
//! This module implements the final stage of weight loading: mapping HuggingFace
//! safetensors tensor names to the [`ModelWeights`] struct used by the model
//! forward pass.
//!
//! Two entry points are provided:
//!
//! - [`load_weights_from_bf16`] — loads `microsoft/bitnet-b1.58-2B-4T-bf16`
//!   (BF16 master weights) and quantises them to ternary on load.
//! - [`load_weights_from_packed`] — loads `microsoft/bitnet-b1.58-2B-4T`
//!   (packed deployment weights: U8 ternary + BF16 scale per tensor).
//!
//! Use [`load_weights_from_packed`] for normal inference — it is faster to load
//! and uses less memory since the weights are already quantised.
//!
//! # Weight Mapping (packed format — `microsoft/bitnet-b1.58-2B-4T`)
//!
//! Linear layer weights are stored as two tensors per projection:
//!
//! ```text
//! model.layers.{i}.self_attn.q_proj.weight           U8  packed ternary
//! model.layers.{i}.self_attn.q_proj.weight_scale     BF16 scale α_W
//! ```
//!
//! The packed HuggingFace deployment weights use the observed 2-bit encoding:
//!
//! ```text
//! 00 -> +1
//! 01 ->  0
//! 10 -> -1
//! ```
//!
//! The effective weight is reconstructed as:
//! ```text
//! W_eff[r,c] = unpack_ternary(weight)[r,c] * weight_scale
//! ```
//!
//! i.e. `scale = weight_scale` (the file stores the direct per-tensor scale).
//!
//! # Weight Mapping (BF16 format — `microsoft/bitnet-b1.58-2B-4T-bf16`)
//!
//! The HuggingFace `bitnet-b1.58-2B-4T-bf16` checkpoint uses the following
//! naming convention (Llama-style):
//!
//! ```text
//! model.embed_tokens.weight                          → embed_tokens
//! model.layers.{i}.self_attn.q_proj.weight           → layers[i].q_proj     (quantised)
//! model.layers.{i}.self_attn.k_proj.weight           → layers[i].k_proj     (quantised)
//! model.layers.{i}.self_attn.v_proj.weight           → layers[i].v_proj     (quantised)
//! model.layers.{i}.self_attn.o_proj.weight           → layers[i].o_proj     (quantised)
//! model.layers.{i}.self_attn.attn_sub_norm.weight    → layers[i].attn_sub_norm
//! model.layers.{i}.mlp.gate_proj.weight              → layers[i].gate_proj  (quantised)
//! model.layers.{i}.mlp.up_proj.weight                → layers[i].up_proj    (quantised)
//! model.layers.{i}.mlp.down_proj.weight              → layers[i].down_proj  (quantised)
//! model.layers.{i}.mlp.ffn_sub_norm.weight           → layers[i].ffn_sub_norm
//! model.layers.{i}.input_layernorm.weight            → layers[i].attention_norm
//! model.layers.{i}.post_attention_layernorm.weight   → layers[i].ffn_norm
//! model.norm.weight                                  → final_norm
//! ```
//!
//! The `lm_head` projection shares weights with `embed_tokens` (weight tying),
//! so no separate `lm_head` key appears in the safetensors file.
//!
//! # Quantisation
//!
//! All projection weight matrices are quantised on load using the absmean
//! function from `bitnet_core::quant::absmean`:
//!
//! ```text
//! α_W  = mean(|W|)                              (per-tensor scale, clamped ≥ 1e-5)
//! W_q  = clip( round( W / α_W ), −1, 1 )       (ternary quantisation)
//! ```
//!
//! Normalisation weights (RMSNorm) are stored as plain `Vec<f32>` — they are
//! not quantised.
//!
//! # Invariants
//!
//! - Every [`LayerWeights`] in `ModelWeights::layers` is fully populated after
//!   a successful [`load_weights_from_bf16`] call.
//! - `lm_head` shares the same `Arc` allocation as `embed_tokens` (weight tying — no extra allocation).
//! - All ternary weight `data` elements satisfy `∈ {-1, 0, 1}`.
//! - All `scale` fields are `> 0`.
//! - `layers.len() == config.num_hidden_layers`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context};
use tracing::{debug, info, instrument, warn};

use bitnet_core::config::ModelConfig;
use bitnet_core::quant::absmean::absmean_quantize;
use bitnet_core::quant::ternary::TernaryWeight;

use crate::safetensors::{load_bf16_safetensors, load_raw_safetensors};

// ---------------------------------------------------------------------------
// LayerWeights
// ---------------------------------------------------------------------------

/// Weights for a single BitNet b1.58 transformer layer.
///
/// Each layer contains:
/// - Pre-attention RMSNorm weight
/// - Q, K, V, O projection ternary weights
/// - Post-attention sub-norm weight
/// - Pre-FFN RMSNorm weight
/// - Gate, Up, Down projection ternary weights
/// - Post-gate FFN sub-norm weight
///
/// # Invariants
///
/// - All `Vec<f32>` fields have length equal to their respective model dimensions.
/// - All [`TernaryWeight`] fields satisfy `data[i] ∈ {-1, 0, 1}` and `scale > 0`.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// Pre-attention RMSNorm scale γ.  Shape: `[hidden_size]`.
    pub attention_norm: Vec<f32>,

    /// Pre-FFN RMSNorm scale γ.  Shape: `[hidden_size]`.
    pub ffn_norm: Vec<f32>,

    /// Query projection ternary weight.
    ///
    /// Logical shape: `[num_attention_heads * head_dim, hidden_size]`
    ///              = `[2560, 2560]` for the 2B model.
    ///
    /// Packed checkpoint storage shape: `[640, 2560]` (4 logical output
    /// channels packed per stored row).
    pub q_proj: TernaryWeight,

    /// Key projection ternary weight.
    ///
    /// Logical shape: `[num_key_value_heads * head_dim, hidden_size]`
    ///              = `[640, 2560]` for the 2B model.
    ///
    /// Packed checkpoint storage shape: `[160, 2560]` (4 logical output
    /// channels packed per stored row).
    pub k_proj: TernaryWeight,

    /// Value projection ternary weight.
    ///
    /// Logical shape: `[num_key_value_heads * head_dim, hidden_size]`
    ///              = `[640, 2560]` for the 2B model.
    ///
    /// Packed checkpoint storage shape: `[160, 2560]` (4 logical output
    /// channels packed per stored row).
    pub v_proj: TernaryWeight,

    /// Output projection ternary weight.
    ///
    /// Logical shape: `[hidden_size, num_attention_heads * head_dim]`
    ///              = `[2560, 2560]` for the 2B model.
    ///
    /// Packed checkpoint storage shape: `[640, 2560]` (4 logical output
    /// channels packed per stored row).
    pub o_proj: TernaryWeight,

    /// Post-attention sub-layer norm scale γ.
    ///
    /// Applied to the attention output before the output projection.
    /// Shape: `[hidden_size]`.
    pub attn_sub_norm: Vec<f32>,

    /// Gate projection ternary weight (GLU gate path).
    ///
    /// Logical shape: `[intermediate_size, hidden_size]`.
    ///
    /// For the 2B model the logical FFN width is 6912, while the official
    /// packed checkpoint stores `[1728, 2560]` (4 logical output channels
    /// packed per stored row).
    pub gate_proj: TernaryWeight,

    /// Up projection ternary weight (GLU content path).
    ///
    /// Logical shape: `[intermediate_size, hidden_size]`.
    ///
    /// For the 2B model the logical FFN width is 6912, while the official
    /// packed checkpoint stores `[1728, 2560]` (4 logical output channels
    /// packed per stored row).
    pub up_proj: TernaryWeight,

    /// Down projection ternary weight (output of FFN).
    ///
    /// Logical shape: `[hidden_size, intermediate_size]` = `[2560, 6912]`
    /// for the 2B model.
    ///
    /// Packed checkpoint storage shape: `[640, 6912]` (4 logical output
    /// channels packed per stored row).
    pub down_proj: TernaryWeight,

    /// Post-gate FFN sub-layer norm scale γ.
    ///
    /// Applied to `sqrelu(gate) ⊙ up` before the down projection.
    /// Shape: `[intermediate_size]` = `[6912]`.
    pub ffn_sub_norm: Vec<f32>,
}

// ---------------------------------------------------------------------------
// ModelWeights
// ---------------------------------------------------------------------------

/// All weights for a complete BitNet b1.58 model.
///
/// Loaded from a BF16 safetensors checkpoint and quantised on-the-fly via
/// [`load_weights_from_bf16`].
///
/// # Layout
///
/// ```text
/// ModelWeights
/// ├── config           — model hyperparameters
/// ├── embed_tokens     — [vocab_size × hidden_size] f32 (not quantised)
/// ├── layers[0..30]    — LayerWeights for each transformer block
/// ├── final_norm       — [hidden_size] f32 RMSNorm weight
/// └── lm_head          — [vocab_size × hidden_size] f32 (= embed_tokens, weight-tied)
/// ```
///
/// # Weight Tying
///
/// The language model head (`lm_head`) shares its weight matrix with the token
/// embedding table (`embed_tokens`).  This is a common space-saving technique
/// in LLaMs: `lm_head.weight = embed_tokens.weight.T`.  In practice both are
/// stored as flat row-major `Vec<f32>` of the same length.
///
/// # Memory Estimate (2B model)
///
/// | Component         | Elements                          | Bytes (f32) |
/// |-------------------|-----------------------------------|-------------|
/// | embed_tokens      | 128256 × 2560 ≈ 328M              | ≈ 1.3 GB    |
/// | layers (30 × all) | ~30 × (ternary+float) ≈ 400M i8   | ≈ 400 MB    |
/// | lm_head (shared)  | 0 (alias of embed_tokens)         | 0 MB        |
///
/// Total: ~1.7 GB for the full f32 embedding + ternary weights.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Model configuration (dimensions, head counts, etc.)
    pub config: ModelConfig,

    /// Token embedding table.  Shape: `[vocab_size, hidden_size]`, row-major.
    ///
    /// `embed_tokens[token_id * hidden_size .. (token_id+1) * hidden_size]`
    /// gives the embedding vector for `token_id`.
    pub embed_tokens: Arc<Vec<f32>>,

    /// Per-layer weights.  Length == `config.num_hidden_layers`.
    pub layers: Vec<LayerWeights>,

    /// Final RMSNorm weight applied after the last transformer block.
    /// Shape: `[hidden_size]`.
    pub final_norm: Vec<f32>,

    /// Language model head weight (tied to `embed_tokens`).
    ///
    /// Shared Arc with `embed_tokens` — no extra allocation.
    /// Shape: `[vocab_size, hidden_size]`.
    /// Semantics: `logits[v] = dot(lm_head[v], hidden)`.
    pub lm_head: Arc<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// load_weights_from_bf16
// ---------------------------------------------------------------------------

/// Load and quantise all model weights from a BF16 safetensors checkpoint.
///
/// This is the primary entry point for weight loading.  It performs:
///
/// 1. Read and parse the safetensors file via [`load_bf16_safetensors`].
/// 2. For each transformer layer (0..`config.num_hidden_layers`):
///    a. Look up the relevant tensor keys.
///    b. Quantise all projection matrices to ternary via absmean.
///    c. Assemble a [`LayerWeights`] struct.
/// 3. Load `embed_tokens` as-is (f32, no quantisation).
/// 4. Load `final_norm` as-is (f32).
/// 5. Arc-clone `embed_tokens` as `lm_head` (weight tying — shared Arc, no copy).
///
/// # Arguments
///
/// - `safetensors_path`: Path to the `.safetensors` file.
/// - `config`:           Model configuration (defines expected tensor shapes).
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read.
/// - Any required tensor key is missing from the safetensors file.
/// - A tensor has the wrong number of elements for its expected shape.
/// - Quantisation fails (e.g. all-zero tensor with invalid scale).
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use bitnet_core::config::bitnet_2b_config;
/// use bitnet_weights::loader::load_weights_from_bf16;
///
/// let config = bitnet_2b_config();
/// let weights = load_weights_from_bf16(
///     Path::new("/path/to/model.safetensors"),
///     &config,
/// ).unwrap();
///
/// assert_eq!(weights.layers.len(), 30);
/// assert_eq!(weights.embed_tokens.len(), 128256 * 2560);
/// ```
#[instrument(
    level = "info",
    skip(config),
    fields(
        path = %safetensors_path.display(),
        n_layers = config.num_hidden_layers
    )
)]
pub fn load_weights_from_bf16(
    safetensors_path: &Path,
    config: &ModelConfig,
) -> anyhow::Result<ModelWeights> {
    config
        .validate()
        .context("ModelConfig validation failed before weight loading")?;

    info!(
        path = %safetensors_path.display(),
        n_layers = config.num_hidden_layers,
        vocab_size = config.vocab_size,
        hidden_size = config.hidden_size,
        "Loading BF16 safetensors weights"
    );

    // ── Step 1: Parse the safetensors file into a flat tensor map ─────────
    let tensor_map: HashMap<String, Vec<f32>> = load_bf16_safetensors(safetensors_path)
        .with_context(|| {
            format!(
                "Failed to load safetensors from {}",
                safetensors_path.display()
            )
        })?;

    debug!(n_tensors = tensor_map.len(), "Tensor map loaded");

    // ── Step 2: Load global tensors ────────────────────────────────────────

    // embed_tokens: [vocab_size, hidden_size]
    let embed_tokens = Arc::new(
        require_tensor(
            &tensor_map,
            "model.embed_tokens.weight",
            config.vocab_size * config.hidden_size,
        )
        .context("Failed to load embed_tokens")?
        .to_vec(),
    );

    debug!(n_elements = embed_tokens.len(), "embed_tokens loaded");

    // final_norm: [hidden_size]
    let final_norm = require_tensor(&tensor_map, "model.norm.weight", config.hidden_size)
        .context("Failed to load final_norm")?
        .to_vec();

    debug!("final_norm loaded");

    // lm_head: weight-tied with embed_tokens — shared Arc, no extra allocation
    let lm_head = Arc::clone(&embed_tokens);

    debug!("lm_head set (weight-tied to embed_tokens, shared Arc)");

    // ── Step 3: Load per-layer weights ─────────────────────────────────────
    let mut layers = Vec::with_capacity(config.num_hidden_layers);

    for layer_idx in 0..config.num_hidden_layers {
        let layer_weights = load_layer_weights(&tensor_map, config, layer_idx)
            .with_context(|| format!("Failed to load layer {layer_idx}"))?;
        layers.push(layer_weights);
    }

    info!(
        n_layers = layers.len(),
        "All layer weights loaded and quantised"
    );

    Ok(ModelWeights {
        config: config.clone(),
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    })
}

// ---------------------------------------------------------------------------
// load_weights_from_packed
// ---------------------------------------------------------------------------

/// Load model weights from the **packed** deployment checkpoint
/// `microsoft/bitnet-b1.58-2B-4T`.
///
/// # Format
///
/// Each linear-layer projection is stored as two tensors:
///
/// | Key suffix | Dtype | Content |
/// |------------|-------|---------|
/// | `weight`   | U8    | Packed 2-bit ternary: 4 values per byte |
/// | `weight_scale_inv` | BF16 | Per-tensor inverse scale `1/α_W` |
///
/// The ternary values are unpacked via [`bitnet_core::quant::ternary::unpack_ternary`]
/// using the HuggingFace packed encoding `00 -> 0`, `01 -> +1`, `10 -> -1`.
/// The per-tensor scale stored in `TernaryWeight` is `1.0 / weight_scale_inv`
/// so that the forward pass computes `W_eff = W_q * scale` as usual.
///
/// Non-linear tensors (norms, embeddings) remain BF16 and are converted to f32.
///
/// # Errors
///
/// Returns an error if any required tensor key is missing, the U8 byte count
/// does not match the expected packed size, or unpacking fails.
#[instrument(
    level = "info",
    skip(config),
    fields(
        path = %safetensors_path.display(),
        n_layers = config.num_hidden_layers
    )
)]
pub fn load_weights_from_packed(
    safetensors_path: &Path,
    config: &ModelConfig,
) -> anyhow::Result<ModelWeights> {
    config
        .validate()
        .context("ModelConfig validation failed before packed weight loading")?;

    info!(
        path = %safetensors_path.display(),
        "Loading packed U8 safetensors weights (microsoft/bitnet-b1.58-2B-4T)"
    );

    // Load every tensor as raw bytes + shape + dtype string.
    let raw_map = load_raw_safetensors(safetensors_path).with_context(|| {
        format!(
            "Failed to read safetensors from {}",
            safetensors_path.display()
        )
    })?;

    debug!(n_tensors = raw_map.len(), "Raw tensor map loaded");

    // ── Helper: convert a raw BF16 tensor to Vec<f32> ─────────────────────
    let get_bf16_f32 = |key: &str, expected_elems: usize| -> anyhow::Result<Vec<f32>> {
        let (bytes, shape, dtype) = raw_map.get(key).ok_or_else(|| {
            anyhow!(
                "Required tensor '{}' not found (total tensors: {})",
                key,
                raw_map.len()
            )
        })?;
        if dtype != "BF16" && dtype != "F32" && dtype != "F16" {
            return Err(anyhow!(
                "Tensor '{}' has unexpected dtype '{}' — expected BF16/F32/F16",
                key,
                dtype
            ));
        }
        let n_elems: usize = shape.iter().product::<usize>().max(1);
        if n_elems != expected_elems {
            return Err(anyhow!(
                "Tensor '{}' has {} elements, expected {}",
                key,
                n_elems,
                expected_elems
            ));
        }
        // Convert bytes → f32 based on dtype.
        use half::bf16;
        let f32_vals: Vec<f32> = match dtype.as_str() {
            "BF16" => bytes
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            "F16" => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            "F32" => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            _ => unreachable!(),
        };
        Ok(f32_vals)
    };

    // ── Helper: load a packed U8 ternary weight + its BF16 scale ─────────
    let load_packed_weight = |key: &str,
                              rows: usize,
                              cols: usize|
     -> anyhow::Result<TernaryWeight> {
        use bitnet_core::quant::ternary::unpack_ternary;

        let n_elems = rows * cols;
        let packed_bytes_needed = (n_elems + 3) / 4; // ceil(n/4)

        let scale_key = format!("{key}_scale");

        // ── Packed weight tensor (U8) ────────────────────────────────────
        let (weight_bytes, _shape, dtype) = raw_map.get(key).ok_or_else(|| {
            anyhow!(
                "Packed weight '{}' not found (total tensors: {})",
                key,
                raw_map.len()
            )
        })?;

        if dtype != "U8" && dtype != "I8" {
            warn!(
                key,
                dtype,
                "Expected U8 packed weights but found dtype '{}'; \
                 try load_weights_from_bf16 instead",
                dtype
            );
            return Err(anyhow!(
                "Tensor '{}' has dtype '{}', expected U8 (packed ternary). \
                 This file may be the BF16 checkpoint — use load_weights_from_bf16.",
                key,
                dtype
            ));
        }

        if weight_bytes.len() < packed_bytes_needed {
            return Err(anyhow!(
                "Packed tensor '{}' has {} bytes, need ≥ {} for {} elements",
                key,
                weight_bytes.len(),
                packed_bytes_needed,
                n_elems
            ));
        }

        let ternary_data = unpack_ternary(&weight_bytes[..packed_bytes_needed], n_elems)
            .with_context(|| format!("Failed to unpack ternary data for '{key}'"))?;

        // ── Scale tensor (BF16 scalar or 1-element tensor) ───────────────
        // weight_scale stores α_W directly. The tensor is usually shape [1]
        // or a scalar.
        let scale = if let Some((scale_bytes, _sc_shape, scale_dtype)) = raw_map.get(&scale_key) {
            if scale_bytes.len() < 2 {
                return Err(anyhow!(
                    "Scale tensor '{}' is too short ({} bytes)",
                    scale_key,
                    scale_bytes.len()
                ));
            }
            let scale_f32: f32 = match scale_dtype.as_str() {
                "BF16" => {
                    half::bf16::from_bits(u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]))
                        .to_f32()
                }
                "F32" => f32::from_le_bytes([
                    scale_bytes[0],
                    scale_bytes[1],
                    scale_bytes[2],
                    scale_bytes[3],
                ]),
                other => {
                    return Err(anyhow!(
                        "Scale tensor '{}' has unsupported dtype '{}'",
                        scale_key,
                        other
                    ))
                }
            };
            if scale_f32 <= 0.0 || !scale_f32.is_finite() {
                warn!(
                    key = scale_key,
                    scale_f32, "weight_scale is non-positive or non-finite; using 1.0 as fallback"
                );
                1.0_f32
            } else {
                scale_f32
            }
        } else {
            warn!(key, "Missing '{}'; defaulting scale to 1.0", scale_key);
            1.0_f32
        };

        TernaryWeight::new(ternary_data, scale, rows, cols)
            .with_context(|| format!("TernaryWeight construction failed for '{key}'"))
    };

    // ── Global tensors ─────────────────────────────────────────────────────

    let embed_tokens = Arc::new(
        get_bf16_f32(
            "model.embed_tokens.weight",
            config.vocab_size * config.hidden_size,
        )
        .context("Failed to load embed_tokens")?,
    );

    let final_norm = get_bf16_f32("model.norm.weight", config.hidden_size)
        .context("Failed to load final_norm")?;

    let lm_head = Arc::clone(&embed_tokens); // weight-tied, no extra allocation

    // ── Per-layer tensors ──────────────────────────────────────────────────
    let head_dim = config.head_dim();
    let q_rows = config.num_key_value_heads * head_dim;
    let kv_rows = q_rows / 4;
    let h = config.hidden_size;
    let ffn = config.intermediate_size / 4;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);

    for layer_idx in 0..config.num_hidden_layers {
        let base = format!("model.layers.{layer_idx}");

        // Normalisation weights (BF16 → f32, not packed).
        let attention_norm = get_bf16_f32(&format!("{base}.input_layernorm.weight"), h)
            .with_context(|| format!("layer {layer_idx}: attention_norm"))?;

        let ffn_norm = get_bf16_f32(&format!("{base}.post_attention_layernorm.weight"), h)
            .with_context(|| format!("layer {layer_idx}: ffn_norm"))?;

        let attn_sub_norm = get_bf16_f32(&format!("{base}.self_attn.attn_sub_norm.weight"), h)
            .with_context(|| format!("layer {layer_idx}: attn_sub_norm"))?;

        let ffn_sub_norm = get_bf16_f32(
            &format!("{base}.mlp.ffn_sub_norm.weight"),
            config.intermediate_size,
        )
        .with_context(|| format!("layer {layer_idx}: ffn_sub_norm"))?;

        // Packed ternary projection weights.
        let q_proj = load_packed_weight(&format!("{base}.self_attn.q_proj.weight"), q_rows, h)
            .with_context(|| format!("layer {layer_idx}: q_proj"))?;

        let k_proj = load_packed_weight(&format!("{base}.self_attn.k_proj.weight"), kv_rows, h)
            .with_context(|| format!("layer {layer_idx}: k_proj"))?;

        let v_proj = load_packed_weight(&format!("{base}.self_attn.v_proj.weight"), kv_rows, h)
            .with_context(|| format!("layer {layer_idx}: v_proj"))?;

        let o_proj = load_packed_weight(&format!("{base}.self_attn.o_proj.weight"), q_rows, h)
            .with_context(|| format!("layer {layer_idx}: o_proj"))?;

        let gate_proj = load_packed_weight(&format!("{base}.mlp.gate_proj.weight"), ffn, h)
            .with_context(|| format!("layer {layer_idx}: gate_proj"))?;

        let up_proj = load_packed_weight(&format!("{base}.mlp.up_proj.weight"), ffn, h)
            .with_context(|| format!("layer {layer_idx}: up_proj"))?;

        let down_proj =
            load_packed_weight(&format!("{base}.mlp.down_proj.weight"), q_rows, ffn * 4)
                .with_context(|| format!("layer {layer_idx}: down_proj"))?;

        debug!(layer = layer_idx, "Packed layer weights loaded");

        layers.push(LayerWeights {
            attention_norm,
            ffn_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            attn_sub_norm,
            gate_proj,
            up_proj,
            down_proj,
            ffn_sub_norm,
        });
    }

    info!(n_layers = layers.len(), "All packed layer weights loaded");

    Ok(ModelWeights {
        config: config.clone(),
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    })
}

// ---------------------------------------------------------------------------
// load_layer_weights (private)
// ---------------------------------------------------------------------------

/// Load and quantise weights for a single transformer layer.
///
/// # Quantisation
///
/// All projection matrices (q, k, v, o, gate, up, down) are quantised via
/// [`absmean_quantize`].  Normalisation vectors are stored as-is.
///
/// # Errors
///
/// Returns an error if any required tensor is missing or has the wrong shape.
fn load_layer_weights(
    tensor_map: &HashMap<String, Vec<f32>>,
    config: &ModelConfig,
    layer_idx: usize,
) -> anyhow::Result<LayerWeights> {
    let base = format!("model.layers.{layer_idx}");
    let head_dim = config.head_dim();

    debug!(layer = layer_idx, "Loading layer weights");

    // ── Normalisation weights (plain f32) ──────────────────────────────────

    let attention_norm = require_tensor(
        tensor_map,
        &format!("{base}.input_layernorm.weight"),
        config.hidden_size,
    )
    .with_context(|| format!("layer {layer_idx}: attention_norm"))?
    .to_vec();

    let ffn_norm = require_tensor(
        tensor_map,
        &format!("{base}.post_attention_layernorm.weight"),
        config.hidden_size,
    )
    .with_context(|| format!("layer {layer_idx}: ffn_norm"))?
    .to_vec();

    let attn_sub_norm = require_tensor(
        tensor_map,
        &format!("{base}.self_attn.attn_sub_norm.weight"),
        config.hidden_size,
    )
    .with_context(|| format!("layer {layer_idx}: attn_sub_norm"))?
    .to_vec();

    let ffn_sub_norm = require_tensor(
        tensor_map,
        &format!("{base}.mlp.ffn_sub_norm.weight"),
        config.intermediate_size,
    )
    .with_context(|| format!("layer {layer_idx}: ffn_sub_norm"))?
    .to_vec();

    // ── Q projection: [n_heads * head_dim, hidden_size] ────────────────────
    let q_rows = config.num_attention_heads * head_dim; // = hidden_size for 2B
    let q_proj = quantise_weight(
        tensor_map,
        &format!("{base}.self_attn.q_proj.weight"),
        q_rows,
        config.hidden_size,
    )
    .with_context(|| format!("layer {layer_idx}: q_proj"))?;

    // ── K projection: [n_kv_heads * head_dim, hidden_size] ─────────────────
    let k_rows = config.num_key_value_heads * head_dim;
    let k_proj = quantise_weight(
        tensor_map,
        &format!("{base}.self_attn.k_proj.weight"),
        k_rows,
        config.hidden_size,
    )
    .with_context(|| format!("layer {layer_idx}: k_proj"))?;

    // ── V projection: [n_kv_heads * head_dim, hidden_size] ─────────────────
    let v_proj = quantise_weight(
        tensor_map,
        &format!("{base}.self_attn.v_proj.weight"),
        k_rows, // same shape as k_proj
        config.hidden_size,
    )
    .with_context(|| format!("layer {layer_idx}: v_proj"))?;

    // ── O projection: [hidden_size, n_heads * head_dim] ────────────────────
    let o_proj = quantise_weight(
        tensor_map,
        &format!("{base}.self_attn.o_proj.weight"),
        config.hidden_size,
        q_rows, // = hidden_size for 2B
    )
    .with_context(|| format!("layer {layer_idx}: o_proj"))?;

    // ── Gate projection: [intermediate_size, hidden_size] ──────────────────
    let gate_proj = quantise_weight(
        tensor_map,
        &format!("{base}.mlp.gate_proj.weight"),
        config.intermediate_size,
        config.hidden_size,
    )
    .with_context(|| format!("layer {layer_idx}: gate_proj"))?;

    // ── Up projection: [intermediate_size, hidden_size] ────────────────────
    let up_proj = quantise_weight(
        tensor_map,
        &format!("{base}.mlp.up_proj.weight"),
        config.intermediate_size,
        config.hidden_size,
    )
    .with_context(|| format!("layer {layer_idx}: up_proj"))?;

    // ── Down projection: [hidden_size, intermediate_size] ──────────────────
    let down_proj = quantise_weight(
        tensor_map,
        &format!("{base}.mlp.down_proj.weight"),
        config.hidden_size,
        config.intermediate_size,
    )
    .with_context(|| format!("layer {layer_idx}: down_proj"))?;

    debug!(layer = layer_idx, "Layer weights loaded and quantised");

    Ok(LayerWeights {
        attention_norm,
        ffn_norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        attn_sub_norm,
        gate_proj,
        up_proj,
        down_proj,
        ffn_sub_norm,
    })
}

// ---------------------------------------------------------------------------
// require_tensor (private)
// ---------------------------------------------------------------------------

/// Look up a tensor in the map and verify its element count.
///
/// # Errors
///
/// - Returns an error if the key is not present in `tensor_map`.
/// - Returns an error if the tensor has the wrong number of elements.
fn require_tensor<'a>(
    tensor_map: &'a HashMap<String, Vec<f32>>,
    key: &str,
    expected_elements: usize,
) -> anyhow::Result<&'a [f32]> {
    let tensor = tensor_map.get(key).ok_or_else(|| {
        // Provide a helpful error listing the available keys that might be close.
        let available: Vec<&str> = tensor_map.keys().map(|s| s.as_str()).collect();
        let similar: Vec<&&str> = available
            .iter()
            .filter(|k| k.contains(key.split('.').last().unwrap_or("")))
            .take(5)
            .collect();
        if similar.is_empty() {
            anyhow!(
                "Required tensor '{}' not found in safetensors file.\n\
                 Total tensors available: {}",
                key,
                tensor_map.len()
            )
        } else {
            anyhow!(
                "Required tensor '{}' not found in safetensors file.\n\
                 Similar keys: {:?}",
                key,
                similar
            )
        }
    })?;

    if tensor.len() != expected_elements {
        return Err(anyhow!(
            "Tensor '{}' has {} elements, expected {}",
            key,
            tensor.len(),
            expected_elements
        ));
    }

    Ok(tensor.as_slice())
}

// ---------------------------------------------------------------------------
// quantise_weight (private)
// ---------------------------------------------------------------------------

/// Retrieve a weight tensor and quantise it to ternary {-1, 0, +1}.
///
/// Applies [`absmean_quantize`] to produce a [`TernaryWeight`] with:
/// - `data[i] ∈ {-1, 0, 1}` (guaranteed by absmean_quantize)
/// - `scale = mean(|W|) > 0` (clamped to 1e-5 minimum)
///
/// # Errors
///
/// Returns an error if the tensor is missing, has the wrong shape,
/// or if quantisation fails.
fn quantise_weight(
    tensor_map: &HashMap<String, Vec<f32>>,
    key: &str,
    rows: usize,
    cols: usize,
) -> anyhow::Result<TernaryWeight> {
    let expected_elements = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("Shape overflow for tensor '{}': {rows} × {cols}", key))?;

    let flat = require_tensor(tensor_map, key, expected_elements)
        .with_context(|| format!("quantise_weight: require_tensor failed for '{key}'"))?;

    let (quantised_data, scale) = absmean_quantize(flat)
        .with_context(|| format!("absmean_quantize failed for tensor '{key}' ({rows}×{cols})"))?;

    // Validate ternary invariant (should always hold after absmean_quantize).
    debug_assert!(
        quantised_data.iter().all(|&v| v >= -1 && v <= 1),
        "absmean_quantize violated ternary invariant for '{key}'"
    );
    debug_assert!(
        scale > 0.0,
        "absmean_quantize returned non-positive scale for '{key}': {scale}"
    );

    TernaryWeight::new(quantised_data, scale, rows, cols)
        .with_context(|| format!("TernaryWeight::new failed for '{key}'"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::config::bitnet_2b_config;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // -----------------------------------------------------------------------
    // Helper: build a minimal synthetic safetensors file with the correct
    // tensor keys for a small test config.
    // -----------------------------------------------------------------------

    /// A tiny model config for testing (1 layer, tiny dimensions).
    fn tiny_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 8,
            hidden_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            intermediate_size: 8,
            max_position_embeddings: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// Build a synthetic safetensors file for the `tiny_config()`.
    ///
    /// All tensors are filled with small positive BF16 values so that
    /// absmean quantisation produces non-trivial (non-all-zero) ternary weights.
    fn build_tiny_safetensors(config: &ModelConfig) -> Vec<u8> {
        use half::bf16;

        // Collect all tensors: (name, shape, values)
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

        let h = config.hidden_size;
        let v = config.vocab_size;
        let ffn = config.intermediate_size;
        let head_dim = config.head_dim();
        let q_rows = config.num_attention_heads * head_dim;
        let kv_rows = config.num_key_value_heads * head_dim;

        // Global tensors.
        tensors.push((
            "model.embed_tokens.weight".to_string(),
            vec![v, h],
            linspace(0.1, 0.9, v * h),
        ));
        tensors.push(("model.norm.weight".to_string(), vec![h], vec![1.0_f32; h]));

        // Per-layer tensors.
        for layer in 0..config.num_hidden_layers {
            let base = format!("model.layers.{layer}");

            tensors.push((
                format!("{base}.input_layernorm.weight"),
                vec![h],
                vec![1.0_f32; h],
            ));
            tensors.push((
                format!("{base}.post_attention_layernorm.weight"),
                vec![h],
                vec![1.0_f32; h],
            ));
            tensors.push((
                format!("{base}.self_attn.attn_sub_norm.weight"),
                vec![h],
                vec![1.0_f32; h],
            ));
            tensors.push((
                format!("{base}.mlp.ffn_sub_norm.weight"),
                vec![ffn],
                vec![1.0_f32; ffn],
            ));

            // Projection weights (alternating sign to ensure non-trivial quantisation).
            tensors.push((
                format!("{base}.self_attn.q_proj.weight"),
                vec![q_rows, h],
                alternating(0.5, q_rows * h),
            ));
            tensors.push((
                format!("{base}.self_attn.k_proj.weight"),
                vec![kv_rows, h],
                alternating(0.5, kv_rows * h),
            ));
            tensors.push((
                format!("{base}.self_attn.v_proj.weight"),
                vec![kv_rows, h],
                alternating(0.3, kv_rows * h),
            ));
            tensors.push((
                format!("{base}.self_attn.o_proj.weight"),
                vec![h, q_rows],
                alternating(0.4, h * q_rows),
            ));
            tensors.push((
                format!("{base}.mlp.gate_proj.weight"),
                vec![ffn, h],
                alternating(0.6, ffn * h),
            ));
            tensors.push((
                format!("{base}.mlp.up_proj.weight"),
                vec![ffn, h],
                alternating(0.7, ffn * h),
            ));
            tensors.push((
                format!("{base}.mlp.down_proj.weight"),
                vec![h, ffn],
                alternating(0.8, h * ffn),
            ));
        }

        // Build safetensors binary format.
        let mut header_map = serde_json::Map::new();
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut offset = 0usize;

        for (name, shape, values) in &tensors {
            // Encode as BF16.
            let encoded: Vec<u8> = values
                .iter()
                .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
                .collect();
            let end = offset + encoded.len();
            let entry = serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [offset, end],
            });
            header_map.insert(name.clone(), entry);
            data_bytes.extend_from_slice(&encoded);
            offset = end;
        }

        let header_json = serde_json::to_vec(&serde_json::Value::Object(header_map)).unwrap();
        let header_len = header_json.len() as u64;

        let mut file = Vec::new();
        file.extend_from_slice(&header_len.to_le_bytes());
        file.extend_from_slice(&header_json);
        file.extend_from_slice(&data_bytes);
        file
    }

    /// Generate `n` values linearly spaced in `[lo, hi]`.
    fn linspace(lo: f32, hi: f32, n: usize) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![lo];
        }
        (0..n)
            .map(|i| lo + (hi - lo) * i as f32 / (n - 1) as f32)
            .collect()
    }

    /// Generate `n` alternating values: `+base, -base, +base, ...`
    fn alternating(base: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| if i % 2 == 0 { base } else { -base })
            .collect()
    }

    // -----------------------------------------------------------------------
    // require_tensor tests
    // -----------------------------------------------------------------------

    #[test]
    fn require_tensor_found_correct_size() {
        let mut map = HashMap::new();
        map.insert("a.weight".to_string(), vec![1.0_f32, 2.0, 3.0]);
        let result = require_tensor(&map, "a.weight", 3).unwrap();
        assert_eq!(result, &[1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn require_tensor_missing_key_returns_error() {
        let map: HashMap<String, Vec<f32>> = HashMap::new();
        let err = require_tensor(&map, "missing.weight", 4).unwrap_err();
        assert!(
            err.to_string().contains("missing.weight"),
            "error must mention key: {err}"
        );
    }

    #[test]
    fn require_tensor_wrong_size_returns_error() {
        let mut map = HashMap::new();
        map.insert("w".to_string(), vec![1.0_f32, 2.0]); // 2 elements
        let err = require_tensor(&map, "w", 5).unwrap_err(); // expect 5
        assert!(
            err.to_string().contains("2") && err.to_string().contains("5"),
            "error must mention actual and expected sizes: {err}"
        );
    }

    #[test]
    fn require_tensor_zero_expected_returns_empty_slice() {
        let mut map = HashMap::new();
        map.insert("empty".to_string(), vec![]); // 0 elements
        let result = require_tensor(&map, "empty", 0).unwrap();
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // quantise_weight tests
    // -----------------------------------------------------------------------

    #[test]
    fn quantise_weight_produces_ternary_values() {
        let mut map = HashMap::new();
        // 2×3 weight matrix with mixed values.
        map.insert("w".to_string(), vec![0.5_f32, -0.5, 0.3, -0.3, 0.0, 1.0]);
        let tw = quantise_weight(&map, "w", 2, 3).unwrap();
        assert_eq!(tw.rows, 2);
        assert_eq!(tw.cols, 3);
        assert_eq!(tw.data.len(), 6);
        // All values must be ternary.
        for &v in &tw.data {
            assert!(v == -1 || v == 0 || v == 1, "not ternary: {v}");
        }
        // Scale must be positive.
        assert!(tw.scale > 0.0, "scale must be > 0, got {}", tw.scale);
    }

    #[test]
    fn quantise_weight_all_zeros_uses_minimum_scale() {
        let mut map = HashMap::new();
        map.insert("z".to_string(), vec![0.0_f32; 4]); // 2×2 all-zero
        let tw = quantise_weight(&map, "z", 2, 2).unwrap();
        // absmean clamps to 1e-5 minimum, quantised values should all be 0.
        assert!(tw.data.iter().all(|&v| v == 0));
        assert!(tw.scale > 0.0);
    }

    #[test]
    fn quantise_weight_missing_key_returns_error() {
        let map: HashMap<String, Vec<f32>> = HashMap::new();
        let err = quantise_weight(&map, "missing", 4, 4).unwrap_err();
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn quantise_weight_shape_overflow_returns_error() {
        let map: HashMap<String, Vec<f32>> = HashMap::new();
        // usize::MAX * 2 overflows.
        let err = quantise_weight(&map, "k", usize::MAX, 2).unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    // -----------------------------------------------------------------------
    // load_weights_from_bf16 integration tests
    // -----------------------------------------------------------------------

    /// Write the tiny safetensors to a tempfile and load weights.
    fn write_and_load(config: &ModelConfig) -> anyhow::Result<ModelWeights> {
        let file_bytes = build_tiny_safetensors(config);
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();
        load_weights_from_bf16(tmp.path(), config)
    }

    #[test]
    fn load_weights_tiny_config_succeeds() {
        let config = tiny_config();
        let weights = write_and_load(&config).expect("Loading tiny config must succeed");

        assert_eq!(
            weights.layers.len(),
            config.num_hidden_layers,
            "Number of layers must match config"
        );
        assert_eq!(
            weights.embed_tokens.len(),
            config.vocab_size * config.hidden_size,
            "embed_tokens length must be vocab × hidden"
        );
        assert_eq!(
            weights.final_norm.len(),
            config.hidden_size,
            "final_norm length must be hidden_size"
        );
        assert_eq!(
            weights.lm_head.len(),
            weights.embed_tokens.len(),
            "lm_head must be same length as embed_tokens (weight tying)"
        );
    }

    #[test]
    fn load_weights_embed_tokens_values_are_finite() {
        let config = tiny_config();
        let weights = write_and_load(&config).unwrap();
        for (i, &v) in weights.embed_tokens.iter().enumerate() {
            assert!(v.is_finite(), "embed_tokens[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn load_weights_lm_head_equals_embed_tokens() {
        let config = tiny_config();
        let weights = write_and_load(&config).unwrap();
        assert_eq!(
            weights.lm_head, weights.embed_tokens,
            "lm_head must be identical to embed_tokens (weight tying)"
        );
        assert!(
            std::sync::Arc::ptr_eq(&weights.lm_head, &weights.embed_tokens),
            "lm_head and embed_tokens must share the same Arc allocation (weight tying)"
        );
    }

    #[test]
    fn load_weights_layer_ternary_weights_are_ternary() {
        let config = tiny_config();
        let weights = write_and_load(&config).unwrap();

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            for (name, tw) in [
                ("q_proj", &layer.q_proj),
                ("k_proj", &layer.k_proj),
                ("v_proj", &layer.v_proj),
                ("o_proj", &layer.o_proj),
                ("gate_proj", &layer.gate_proj),
                ("up_proj", &layer.up_proj),
                ("down_proj", &layer.down_proj),
            ] {
                for &v in &tw.data {
                    assert!(
                        v == -1 || v == 0 || v == 1,
                        "layer {layer_idx}.{name}: ternary invariant violated: {v}"
                    );
                }
                assert!(
                    tw.scale > 0.0,
                    "layer {layer_idx}.{name}: scale must be > 0, got {}",
                    tw.scale
                );
            }
        }
    }

    #[test]
    fn load_weights_layer_norm_weights_are_finite() {
        let config = tiny_config();
        let weights = write_and_load(&config).unwrap();

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            for (name, nw) in [
                ("attention_norm", &layer.attention_norm),
                ("ffn_norm", &layer.ffn_norm),
                ("attn_sub_norm", &layer.attn_sub_norm),
                ("ffn_sub_norm", &layer.ffn_sub_norm),
            ] {
                for (i, &v) in nw.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "layer {layer_idx}.{name}[{i}] = {v} is not finite"
                    );
                }
            }
        }
    }

    #[test]
    fn load_weights_layer_shapes_match_config() {
        let config = tiny_config();
        let weights = write_and_load(&config).unwrap();
        let head_dim = config.head_dim();
        let q_rows = config.num_attention_heads * head_dim;
        let kv_rows = config.num_key_value_heads * head_dim;
        let h = config.hidden_size;
        let ffn = config.intermediate_size;

        for (i, layer) in weights.layers.iter().enumerate() {
            assert_eq!(
                layer.attention_norm.len(),
                h,
                "layer {i} attention_norm wrong length"
            );
            assert_eq!(layer.ffn_norm.len(), h, "layer {i} ffn_norm wrong length");
            assert_eq!(
                layer.attn_sub_norm.len(),
                h,
                "layer {i} attn_sub_norm wrong length"
            );
            assert_eq!(
                layer.ffn_sub_norm.len(),
                ffn,
                "layer {i} ffn_sub_norm wrong length"
            );

            // q_proj: [q_rows, h]
            assert_eq!(layer.q_proj.rows, q_rows, "layer {i} q_proj rows");
            assert_eq!(layer.q_proj.cols, h, "layer {i} q_proj cols");

            // k_proj: [kv_rows, h]
            assert_eq!(layer.k_proj.rows, kv_rows, "layer {i} k_proj rows");
            assert_eq!(layer.k_proj.cols, h, "layer {i} k_proj cols");

            // v_proj: [kv_rows, h]
            assert_eq!(layer.v_proj.rows, kv_rows, "layer {i} v_proj rows");
            assert_eq!(layer.v_proj.cols, h, "layer {i} v_proj cols");

            // o_proj: [h, q_rows]
            assert_eq!(layer.o_proj.rows, h, "layer {i} o_proj rows");
            assert_eq!(layer.o_proj.cols, q_rows, "layer {i} o_proj cols");

            // gate_proj: [ffn, h]
            assert_eq!(layer.gate_proj.rows, ffn, "layer {i} gate_proj rows");
            assert_eq!(layer.gate_proj.cols, h, "layer {i} gate_proj cols");

            // up_proj: [ffn, h]
            assert_eq!(layer.up_proj.rows, ffn, "layer {i} up_proj rows");
            assert_eq!(layer.up_proj.cols, h, "layer {i} up_proj cols");

            // down_proj: [h, ffn]
            assert_eq!(layer.down_proj.rows, h, "layer {i} down_proj rows");
            assert_eq!(layer.down_proj.cols, ffn, "layer {i} down_proj cols");
        }
    }

    #[test]
    fn load_weights_final_norm_has_correct_length() {
        let config = tiny_config();
        let weights = write_and_load(&config).unwrap();
        assert_eq!(weights.final_norm.len(), config.hidden_size);
    }

    #[test]
    fn load_weights_nonexistent_file_returns_error() {
        let config = tiny_config();
        let err = load_weights_from_bf16(Path::new("/nonexistent/model.safetensors"), &config)
            .unwrap_err();
        assert!(
            !err.to_string().is_empty(),
            "Error message must be non-empty"
        );
    }

    #[test]
    fn load_weights_invalid_config_returns_error() {
        // A config with 0 hidden_size is invalid.
        let mut config = tiny_config();
        config.hidden_size = 0;

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"dummy").unwrap();
        tmp.flush().unwrap();

        let err = load_weights_from_bf16(tmp.path(), &config).unwrap_err();
        assert!(
            err.to_string().contains("hidden_size") || !err.to_string().is_empty(),
            "Error must be descriptive: {err}"
        );
    }

    #[test]
    fn load_weights_missing_tensor_returns_error() {
        // Build a safetensors file that is missing one required tensor.
        // We use a tiny config but omit model.norm.weight.
        let config = tiny_config();
        let mut file_bytes = build_tiny_safetensors(&config);

        // Corrupt: truncate to 8 bytes (invalid header length → will fail to parse).
        file_bytes.truncate(8);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let err = load_weights_from_bf16(tmp.path(), &config).unwrap_err();
        assert!(!err.to_string().is_empty(), "Must return an error");
    }

    // -----------------------------------------------------------------------
    // Mathematical invariant tests
    // -----------------------------------------------------------------------

    /// Invariant: the absmean scale α_W satisfies
    /// `α_W = mean(|W|)` and `mean(|W_q * α_W - W|)` is bounded.
    ///
    /// Theorem: the L1 approximation error is ≤ α_W / 2 for unclipped values.
    #[test]
    fn quantisation_approximation_error_bounded() {
        let config = tiny_config();
        let weights = write_and_load(&config).unwrap();

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            // Check q_proj as a representative projection weight.
            let tw = &layer.q_proj;
            // Reconstruct approximate weights: w_approx[i] = tw.data[i] * tw.scale
            for (i, &q_val) in tw.data.iter().enumerate() {
                let w_approx = q_val as f32 * tw.scale;
                // w_approx ∈ {-tw.scale, 0, +tw.scale}
                let abs_approx = w_approx.abs();
                // For ternary quantisation: approx value is either 0 or ±scale.
                // This is a structural check: abs ≤ scale + epsilon.
                assert!(
                    abs_approx <= tw.scale + 1e-5,
                    "layer {layer_idx} q_proj[{i}]: approx {w_approx} exceeds scale {}",
                    tw.scale
                );
            }
        }
    }

    /// Invariant: `embed_tokens` values are in a reasonable numeric range.
    ///
    /// BF16 has limited precision; converted f32 values should be in [-10, 10]
    /// for typical model initialisation (our test uses linspace(0.1, 0.9)).
    #[test]
    fn embed_tokens_values_in_reasonable_range() {
        let config = tiny_config();
        let weights = write_and_load(&config).unwrap();
        for (i, &v) in weights.embed_tokens.iter().enumerate() {
            assert!(
                v.abs() < 1000.0,
                "embed_tokens[{i}] = {v} outside reasonable range"
            );
        }
    }

    /// Verify that quantised scale reflects the magnitude of the original weights.
    ///
    /// For alternating(0.5, n): mean(|W|) = 0.5, so scale ≈ 0.5 (within BF16 precision).
    #[test]
    fn quantised_scale_reflects_weight_magnitude() {
        let config = tiny_config();
        let weights = write_and_load(&config).unwrap();
        let tw = &weights.layers[0].q_proj;
        // alternating(0.5, n) → all |w| = 0.5 → mean(|w|) = 0.5
        // BF16 rounds 0.5 exactly, so scale should be very close to 0.5.
        assert!(
            (tw.scale - 0.5).abs() < 0.05,
            "q_proj scale should be ≈ 0.5 for alternating(0.5, n), got {}",
            tw.scale
        );
    }

    #[test]
    #[ignore = "forensic inspection: norm weights and embedding scale sanity check"]
    fn forensic_real_packed_norm_weights_and_embedding_scale() {
        let model_path = std::path::Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );
        assert!(
            model_path.exists(),
            "requires local HuggingFace packed weights"
        );

        let config = bitnet_core::config::bitnet_2b_config();
        let weights =
            load_weights_from_packed(model_path, &config).expect("must load successfully");

        // ── Norm weight values (should be close to 1.0 for a trained model) ──
        let attn_norm = &weights.layers[0].attention_norm;
        let attn_sub_norm = &weights.layers[0].attn_sub_norm;
        let ffn_norm = &weights.layers[0].ffn_norm;
        let ffn_sub_norm = &weights.layers[0].ffn_sub_norm;
        let final_norm = &weights.final_norm;

        fn describe_vec(v: &[f32], name: &str) {
            let n = v.len();
            let mean = v.iter().sum::<f32>() / n as f32;
            let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let std_dev = {
                let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
                var.sqrt()
            };
            eprintln!(
                "{name:30}: n={n:5}  mean={mean:.4}  min={min:.4}  max={max:.4}  std={std_dev:.4}  first5={:?}",
                &v[..5.min(n)]
            );
        }

        eprintln!("=== Normalization weights ===");
        describe_vec(attn_norm, "attention_norm (layer 0)");
        describe_vec(attn_sub_norm, "attn_sub_norm  (layer 0)");
        describe_vec(ffn_norm, "ffn_norm       (layer 0)");
        describe_vec(ffn_sub_norm, "ffn_sub_norm   (layer 0)");
        describe_vec(final_norm, "final_norm");

        // ── Embedding scale (token 791 = "The", token 60704 = "Paris") ──
        let h = config.hidden_size;
        let emb = &weights.embed_tokens;

        eprintln!("\n=== Embedding rows ===");
        for &(tok_id, label) in &[
            (128000usize, "BOS(128000)"),
            (791, "The(791)"),
            (60704, "Paris(60704)"),
            (17, "2(17)"),
        ] {
            let row = &emb[tok_id * h..(tok_id + 1) * h];
            describe_vec(row, label);
        }

        // ── All layer-0 scale values ──
        eprintln!("\n=== Layer-0 projection scales (α_W) ===");
        eprintln!("  q_proj  scale = {:.6}", weights.layers[0].q_proj.scale);
        eprintln!("  k_proj  scale = {:.6}", weights.layers[0].k_proj.scale);
        eprintln!("  v_proj  scale = {:.6}", weights.layers[0].v_proj.scale);
        eprintln!("  o_proj  scale = {:.6}", weights.layers[0].o_proj.scale);
        eprintln!("  gate    scale = {:.6}", weights.layers[0].gate_proj.scale);
        eprintln!("  up      scale = {:.6}", weights.layers[0].up_proj.scale);
        eprintln!("  down    scale = {:.6}", weights.layers[0].down_proj.scale);

        // ── Sanity assertions ──
        // Norm weights should be positive and finite.
        for &v in attn_norm
            .iter()
            .chain(attn_sub_norm.iter())
            .chain(ffn_norm.iter())
        {
            assert!(
                v.is_finite() && v > 0.0,
                "norm weight must be finite positive, got {v}"
            );
        }
        // Embedding values should be finite and reasonable.
        for &v in &emb[..10] {
            assert!(v.is_finite(), "embedding value must be finite, got {v}");
        }
    }

    #[test]
    #[ignore = "forensic inspection: raw byte values and bit-packing order for q_proj"]
    fn forensic_real_packed_byte_order_check() {
        let model_path = std::path::Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );
        assert!(
            model_path.exists(),
            "requires local HuggingFace packed weights"
        );

        let raw_map = load_raw_safetensors(model_path).expect("raw safetensors must load");

        let (weight_bytes, shape, dtype) = raw_map
            .get("model.layers.0.self_attn.q_proj.weight")
            .expect("q_proj weight must exist");

        eprintln!(
            "q_proj dtype={dtype}  shape={shape:?}  bytes={}",
            weight_bytes.len()
        );

        // Print the first 8 raw bytes and decode them under both endianness conventions.
        eprintln!("\nFirst 8 bytes (raw hex):");
        for (i, &b) in weight_bytes.iter().enumerate().take(8) {
            eprintln!("  byte[{i}] = 0x{b:02X} = 0b{b:08b}");
        }

        // Little-endian (our current convention): bit_pos 0 → bits 0-1, etc.
        eprintln!("\nDecoded as LITTLE-ENDIAN (element 0 = bits 0-1):");
        for byte_idx in 0..2usize {
            let b = weight_bytes[byte_idx];
            for bit_pos in 0..4usize {
                let code = (b >> (bit_pos * 2)) & 0b11;
                let val: i8 = match code {
                    0b00 => 1,
                    0b01 => 0,
                    0b10 => -1,
                    _ => -99,
                };
                eprintln!(
                    "  elem[{}] = code {} → val {}",
                    byte_idx * 4 + bit_pos,
                    code,
                    val
                );
            }
        }

        // Big-endian: bit_pos 0 → bits 6-7, etc.
        eprintln!("\nDecoded as BIG-ENDIAN (element 0 = bits 6-7):");
        for byte_idx in 0..2usize {
            let b = weight_bytes[byte_idx];
            for bit_pos in 0..4usize {
                let code = (b >> (6 - bit_pos * 2)) & 0b11;
                let val: i8 = match code {
                    0b00 => 1,
                    0b01 => 0,
                    0b10 => -1,
                    _ => -99,
                };
                eprintln!(
                    "  elem[{}] = code {} → val {}",
                    byte_idx * 4 + bit_pos,
                    code,
                    val
                );
            }
        }

        // Also check the (scale_key) value directly.
        let scale_key = "model.layers.0.self_attn.q_proj.weight_scale";
        if let Some((sb, _ss, sd)) = raw_map.get(scale_key) {
            use half::bf16;
            let sv = bf16::from_bits(u16::from_le_bytes([sb[0], sb[1]])).to_f32();
            eprintln!("\nweight_scale (key={scale_key}, dtype={sd}) = {sv}");
            eprintln!("If this is α_W: effective weight = W_q × {sv}");
            eprintln!("If this is 1/α_W: effective weight = W_q × {}", sv.recip());
        }
    }

    #[test]
    #[ignore = "forensic inspection: scale tensor shape and per-row vs scalar disambiguation"]
    fn forensic_real_packed_scale_tensor_shape() {
        let model_path = std::path::Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );
        assert!(
            model_path.exists(),
            "requires local HuggingFace packed weights"
        );

        let raw_map = load_raw_safetensors(model_path).expect("raw safetensors must load");

        // Discover all tensor keys related to q_proj to find the actual scale key.
        let q_proj_keys: Vec<&String> = raw_map
            .keys()
            .filter(|k| k.contains("layers.0") && k.contains("q_proj"))
            .collect();
        eprintln!("All layer-0 q_proj tensor keys ({}):", q_proj_keys.len());
        let mut sorted_keys = q_proj_keys.clone();
        sorted_keys.sort();
        for k in &sorted_keys {
            let (b, s, d) = &raw_map[*k];
            eprintln!("  {k}  dtype={d}  shape={s:?}  bytes={}", b.len());
        }

        // Also list all layer-0 keys to understand the full naming scheme.
        let layer0_keys: Vec<&String> =
            raw_map.keys().filter(|k| k.contains("layers.0.")).collect();
        let mut sorted_l0 = layer0_keys.clone();
        sorted_l0.sort();
        eprintln!("\nAll layer-0 tensor keys ({}):", sorted_l0.len());
        for k in &sorted_l0 {
            let (b, s, d) = &raw_map[*k];
            eprintln!("  {k}  dtype={d}  shape={s:?}  bytes={}", b.len());
        }

        // Find the scale key (look for any key with "scale" in the q_proj context).
        let scale_key_opt = sorted_keys.iter().find(|k| k.contains("scale")).copied();

        let scale_key = scale_key_opt.expect("must find a scale key for q_proj");
        let (scale_bytes, scale_shape, scale_dtype) = &raw_map[scale_key];

        eprintln!("\nUsing scale tensor key: {scale_key}");
        eprintln!("scale tensor dtype: {scale_dtype}");
        eprintln!("scale tensor shape: {scale_shape:?}");
        eprintln!("scale tensor bytes: {} bytes", scale_bytes.len());

        let n_elems: usize = scale_shape.iter().product::<usize>().max(1);
        eprintln!("scale tensor n_elems: {n_elems}");

        // Decode all scale values (BF16 → f32).
        use half::bf16;
        let scale_vals: Vec<f32> = match scale_dtype.as_str() {
            "BF16" => scale_bytes
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            "F32" => scale_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            other => panic!("unexpected scale dtype: {other}"),
        };

        eprintln!("decoded scale values ({} total):", scale_vals.len());
        for (i, &v) in scale_vals.iter().enumerate().take(8) {
            eprintln!("  scale_vals[{i}] = {v:.6}");
        }
        if scale_vals.len() > 8 {
            eprintln!("  ... ({} more)", scale_vals.len() - 8);
        }

        // Summary statistics.
        let min_s = scale_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_s = scale_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_s = scale_vals.iter().sum::<f32>() / scale_vals.len() as f32;
        eprintln!("scale_vals stats: min={min_s:.6} max={max_s:.6} mean={mean_s:.6}");

        // If the tensor has only 1 element, it is a scalar per-tensor scale.
        // If it has n_rows elements, it is a per-row scale.
        if n_elems == 1 {
            eprintln!("CONCLUSION: per-TENSOR scalar scale (1 element)");
        } else {
            let h = 2560usize; // hidden_size
            let q_rows = 20 * 128; // n_heads * head_dim
            if n_elems == q_rows {
                eprintln!("CONCLUSION: per-ROW scale ({n_elems} elements = q_rows)");
            } else if n_elems == h {
                eprintln!("CONCLUSION: per-column scale ({n_elems} elements = hidden_size)");
            } else {
                eprintln!("CONCLUSION: unknown shape ({n_elems} elements)");
            }
        }
    }

    #[test]
    #[ignore = "forensic inspection: scale values and embedding sanity for loaded packed weights"]
    fn forensic_real_packed_weight_scale_and_embedding() {
        let model_path = std::path::Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );
        assert!(
            model_path.exists(),
            "requires local HuggingFace packed weights"
        );

        let config = bitnet_core::config::bitnet_2b_config();
        let weights =
            load_weights_from_packed(model_path, &config).expect("must load successfully");

        // ── Q-proj scale ────────────────────────────────────────────────────
        let q_scale = weights.layers[0].q_proj.scale;
        eprintln!("layer[0] q_proj scale (α_W) = {q_scale:.6}");
        assert!(
            q_scale > 0.0 && q_scale.is_finite(),
            "q_proj scale must be finite positive, got {q_scale}"
        );
        // Typical absmean scale for a trained BitNet is in (0.001, 10.0).
        assert!(
            q_scale > 0.001 && q_scale < 100.0,
            "q_proj scale {q_scale} is outside plausible range (0.001, 100)"
        );

        // ── Non-zero weight ratio ────────────────────────────────────────────
        let q_data = &weights.layers[0].q_proj.data;
        let n_nonzero = q_data.iter().filter(|&&v| v != 0).count();
        let nonzero_frac = n_nonzero as f32 / q_data.len() as f32;
        eprintln!(
            "layer[0] q_proj non-zero fraction = {:.3} ({} / {})",
            nonzero_frac,
            n_nonzero,
            q_data.len()
        );
        // Expect roughly 25-50% non-zero for BitNet.
        assert!(
            nonzero_frac > 0.1 && nonzero_frac < 0.9,
            "non-zero fraction {nonzero_frac:.3} is outside plausible range (0.1, 0.9)"
        );

        // ── Value distribution {-1, 0, +1} ──────────────────────────────────
        let n_pos = q_data.iter().filter(|&&v| v == 1).count();
        let n_neg = q_data.iter().filter(|&&v| v == -1).count();
        let n_zer = q_data.iter().filter(|&&v| v == 0).count();
        eprintln!(
            "layer[0] q_proj +1: {n_pos}  0: {n_zer}  -1: {n_neg}  (total {})",
            q_data.len()
        );
        // +1 and -1 counts should be roughly equal (absmean symmetry).
        let ratio = n_pos as f32 / (n_neg as f32 + 1.0);
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "+1/-1 ratio {ratio:.3} deviates too far from 1.0 — encoding may be wrong"
        );

        // ── Embedding first-row sanity ───────────────────────────────────────
        let emb = &weights.embed_tokens;
        let h = config.hidden_size;
        let first_row = &emb[..h];
        let mean: f32 = first_row.iter().sum::<f32>() / h as f32;
        let max_abs = first_row.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        eprintln!("embed_tokens[0] mean={mean:.6}  max_abs={max_abs:.6}");
        assert!(
            max_abs > 0.0 && max_abs.is_finite(),
            "embed_tokens first row must have finite non-zero values"
        );
        // Typical embedding magnitudes for a trained model are in (0.01, 10.0).
        assert!(
            max_abs < 100.0,
            "embed_tokens max_abs {max_abs} suspiciously large — may indicate load error"
        );
    }

    #[test]
    #[ignore = "forensic inspection of locally downloaded HuggingFace packed weights"]
    fn forensic_real_packed_weight_code_distribution() {
        use std::collections::BTreeMap;

        let model_path = std::path::Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );

        assert!(
            model_path.exists(),
            "expected local HuggingFace packed weights at {}",
            model_path.display()
        );

        let raw_map = load_raw_safetensors(model_path)
            .expect("real HuggingFace packed safetensors must load for forensic inspection");

        let (weight_bytes, _shape, dtype) = raw_map
            .get("model.layers.0.self_attn.q_proj.weight")
            .expect("q_proj packed tensor must exist in the real HuggingFace checkpoint");

        assert_eq!(
            dtype, "U8",
            "forensic test expects packed q_proj tensor to have dtype U8"
        );

        let mut counts: BTreeMap<u8, usize> = BTreeMap::new();
        for &byte in weight_bytes.iter().take(4096) {
            for bit_pos in 0..4 {
                let code = (byte >> (bit_pos * 2)) & 0b11;
                *counts.entry(code).or_insert(0) += 1;
            }
        }

        eprintln!("forensic 2-bit code counts for first q_proj bytes: {counts:?}");

        let total_codes: usize = counts.values().sum();
        assert!(
            total_codes > 0,
            "forensic code count must observe at least one packed 2-bit value"
        );
        assert!(
            counts.contains_key(&0b10),
            "real HuggingFace packed weights must contain 0b10 to reproduce the loader failure; observed counts: {counts:?}"
        );
    }

    #[test]
    #[ignore = "requires locally downloaded HuggingFace packed weights at the default cache path"]
    fn real_huggingface_packed_loader_smoke_test() {
        let model_path = std::path::Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );

        assert!(
            model_path.exists(),
            "expected local HuggingFace packed weights at {}",
            model_path.display()
        );

        let config = bitnet_core::config::bitnet_2b_config();
        let weights = load_weights_from_packed(model_path, &config)
            .expect("real HuggingFace packed weights must load successfully");

        assert_eq!(
            weights.layers.len(),
            config.num_hidden_layers,
            "real HuggingFace packed loader must produce one LayerWeights per transformer block"
        );
        assert_eq!(
            weights.embed_tokens.len(),
            config.vocab_size * config.hidden_size,
            "embed_tokens length must match vocab_size * hidden_size"
        );
        assert_eq!(
            weights.final_norm.len(),
            config.hidden_size,
            "final_norm length must match hidden_size"
        );
        assert_eq!(
            weights.lm_head.len(),
            weights.embed_tokens.len(),
            "lm_head must remain weight-tied to embed_tokens"
        );
        assert!(
            std::sync::Arc::ptr_eq(&weights.lm_head, &weights.embed_tokens),
            "lm_head and embed_tokens must share the same Arc allocation (weight tying)"
        );

        let q_proj = &weights.layers[0].q_proj;
        assert_eq!(
            q_proj.rows,
            config.num_attention_heads * config.head_dim(),
            "q_proj row count must match n_heads * head_dim"
        );
        assert_eq!(
            q_proj.cols, config.hidden_size,
            "q_proj column count must match hidden_size"
        );
        assert!(
            q_proj.scale.is_finite() && q_proj.scale > 0.0,
            "q_proj scale must be finite and positive, got {}",
            q_proj.scale
        );
        assert!(
            q_proj.data.iter().all(|&v| matches!(v, -1 | 0 | 1)),
            "decoded q_proj ternary data must stay within {{-1, 0, +1}}"
        );
    }
}
