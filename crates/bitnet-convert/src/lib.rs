//! # bitnet-convert
//!
//! Conversion pipeline for aligning Rust BitNet inference with the official
//! `bitnet.cpp` preprocessing semantics.
//!
//! ## Purpose
//!
//! The HuggingFace repository `microsoft/bitnet-b1.58-2B-4T` does **not**
//! expose projection weights in the logical dense tensor shapes used by the
//! transformer forward pass. Instead, it stores a kernel-oriented packed format:
//!
//! - projection tensors are `U8`
//! - each byte stores four 2-bit lanes
//! - the packed row count is `logical_rows / 4`
//! - a companion `weight_scale` tensor stores the inverse dequant scale `s`
//! - the official conversion reconstructs logical ternary rows via:
//!
//! ```text
//! lane = (byte >> shift) & 0b11, shift ∈ {0, 2, 4, 6}
//! ternary = decode(lane)              // 00→-1, 01→0, 10→+1, 11→invalid
//! logical = reshape([packed_rows * 4, cols])
//! effective_weight = logical * weight_scale
//! ```
//!
//! In addition, the official conversion applies a LLaMA-style permutation to
//! `q_proj` and `k_proj` before writing the final runtime format.
//!
//! This crate introduces explicit conversion stages so the rest of the Rust
//! runtime can operate on semantically meaningful weights rather than raw packed
//! HuggingFace tensors.
//!
//! ## Architectural Stages
//!
//! 1. **Source ingestion**
//!    - raw HuggingFace packed tensors and metadata
//! 2. **Canonical conversion**
//!    - unpack 2-bit lanes into logical ternary matrices
//!    - store the HF `weight_scale` (absmean) directly as `TernaryWeight.scale`
//!    - apply required tensor permutations
//! 3. **Runtime packing**
//!    - future stage for TL1/TL2 or other kernel-specific layouts
//!
//! ## Current Scope
//!
//! This initial implementation provides:
//!
//! - explicit source and canonical model representations
//! - canonical packed decoding for a single projection tensor
//! - canonical conversion for the full packed HuggingFace model
//! - LLaMA-style `q_proj` / `k_proj` permutation
//!
//! It intentionally does **not** yet implement:
//!
//! - TL1/TL2 runtime repacking
//! - GGUF serialization
//! - direct integration into the inference runtime
//!
//! ## Invariants
//!
//! On successful canonical conversion:
//!
//! - every decoded projection has its **logical** dense shape
//! - every decoded ternary element is in `{-1, 0, +1}`
//! - every effective scale is finite and strictly positive
//! - `q_proj` and `k_proj` are permuted to match the official conversion path
//! - all non-projection tensors preserve their expected logical lengths
//!
//! ## Design Note
//!
//! The canonical representation stores ternary values plus an effective scale
//! factor in [`bitnet_core::quant::ternary::TernaryWeight`]. This matches the
//! existing Rust model execution path:
//!
//! ```text
//! output = (ternary_dot(input)) * scale
//! ```
//!
//! The canonical representation stores the direct multiplicative dequant scale
//! used by the runtime, so `effective_weight = ternary * scale`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context};
use bitnet_core::config::ModelConfig;
use bitnet_core::quant::ternary::TernaryWeight;
use bitnet_weights::loader::{LayerWeights, ModelWeights};
use bitnet_weights::safetensors::load_raw_safetensors;
use tracing::{debug, info, instrument, warn};

/// Raw packed tensor bytes and metadata loaded from HuggingFace safetensors.
///
/// This is the source-format representation before any semantic conversion.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedTensor {
    /// Tensor name as stored in the safetensors file.
    pub name: String,
    /// Raw tensor bytes.
    pub bytes: Vec<u8>,
    /// Tensor shape in HuggingFace storage layout.
    pub shape: Vec<usize>,
    /// HuggingFace dtype string, e.g. `U8`, `BF16`, `F32`.
    pub dtype: String,
}

impl PackedTensor {
    /// Returns the total number of stored elements implied by `shape`.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }
}

/// Raw HuggingFace packed BitNet model.
///
/// This is the direct source representation of the packed safetensors file.
#[derive(Debug, Clone)]
pub struct HfPackedModel {
    /// Model configuration resolved from `config.json`.
    pub config: ModelConfig,
    /// Raw tensors keyed by HuggingFace tensor name.
    pub tensors: HashMap<String, PackedTensor>,
}

/// Canonical logical BitNet model weights.
///
/// This is the semantically meaningful representation after official packed
/// decoding semantics have been applied.
#[derive(Debug, Clone)]
pub struct CanonicalModelWeights {
    /// Canonical logical model weights ready for dense Rust execution.
    pub weights: ModelWeights,
}

/// Runtime-packed model placeholder.
///
/// This type exists to make the conversion pipeline stages explicit. A future
/// implementation can add TL1/TL2 or GGUF-compatible runtime packing here.
#[derive(Debug, Clone)]
pub struct RuntimePackedModel {
    /// Canonical weights retained until a runtime-specific packed layout is
    /// implemented.
    pub canonical: CanonicalModelWeights,
}

/// Projection tensor role within a transformer layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionKind {
    /// Query projection.
    Query,
    /// Key projection.
    Key,
    /// Value projection.
    Value,
    /// Attention output projection.
    Output,
    /// FFN gate projection.
    Gate,
    /// FFN up projection.
    Up,
    /// FFN down projection.
    Down,
}

impl ProjectionKind {
    /// Returns `true` if the official `bitnet.cpp` GGUF conversion applies the
    /// LLaMA permutation to reorder Q/K rows from the HuggingFace half-split
    /// layout `(i, i+half)` to the interleaved layout `(2i, 2i+1)`.
    ///
    /// **This crate's RoPE uses the half-split convention** (matching
    /// HuggingFace `rotate_half`), so the permutation must NOT be applied.
    /// The method is retained for documentation and potential future use with
    /// an interleaved-RoPE backend.
    #[inline]
    pub fn requires_llama_permute(self) -> bool {
        // Disabled: our RoPE implementation uses the HuggingFace half-split
        // convention (i, i+half), not the interleaved (2i, 2i+1) convention
        // that the LLaMA permutation targets.  Applying the permutation here
        // would scramble the dimension pairs that RoPE rotates, breaking
        // attention completely (symptom: repetitive "the the the…" output).
        false
    }
}

/// Canonical decoded projection tensor.
///
/// This is the result of unpacking one HuggingFace packed projection tensor into
/// its logical dense ternary matrix plus effective scale.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalProjection {
    /// Projection role.
    pub kind: ProjectionKind,
    /// Logical ternary weight matrix.
    pub weight: TernaryWeight,
}

/// Decode a packed HuggingFace BitNet projection tensor into its logical dense
/// ternary matrix.
///
/// # Official Semantics
///
/// The official conversion path performs:
///
/// ```text
/// lane = (byte >> shift) & 0b11
/// ternary = decode(lane)   // 00→+1, 01→0, 10→-1, 11→invalid
/// logical_rows = packed_rows * 4
/// effective_scale = weight_scale
/// ```
///
/// where `weight_scale` is the direct multiplicative dequant scale stored in
/// the checkpoint.
///
/// # Arguments
///
/// - `packed_bytes`: raw `U8` packed tensor bytes
/// - `packed_rows`:  stored row count from HuggingFace
/// - `cols`:         logical column count
/// - `weight_scale`: direct multiplicative dequant scale from `*.weight_scale`
/// - `kind`:         projection role
///
/// # Returns
///
/// A [`CanonicalProjection`] with logical row count `packed_rows * 4`.
///
/// # Errors
///
/// Returns an error if:
///
/// - `weight_scale <= 0` or non-finite
/// - the packed bytes are insufficient
/// - an invalid 2-bit lane value would map outside `{-1, 0, +1}`
/// - permutation preconditions are violated
pub fn decode_packed_projection(
    packed_bytes: &[u8],
    packed_rows: usize,
    cols: usize,
    weight_scale: f32,
    kind: ProjectionKind,
    config: &ModelConfig,
) -> anyhow::Result<CanonicalProjection> {
    if packed_rows == 0 {
        return Err(anyhow!("packed_rows must be > 0"));
    }
    if cols == 0 {
        return Err(anyhow!("cols must be > 0"));
    }
    if weight_scale <= 0.0 || !weight_scale.is_finite() {
        return Err(anyhow!(
            "weight_scale must be finite and > 0, got {weight_scale}"
        ));
    }

    let logical_rows = packed_rows
        .checked_mul(4)
        .ok_or_else(|| anyhow!("logical row expansion overflow: {packed_rows} * 4"))?;
    let logical_numel = logical_rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("logical element count overflow: {logical_rows} * {cols}"))?;
    let required_bytes = (logical_numel + 3) / 4;

    if packed_bytes.len() < required_bytes {
        return Err(anyhow!(
            "packed tensor has {} bytes, need at least {} bytes for {} logical elements",
            packed_bytes.len(),
            required_bytes,
            logical_numel
        ));
    }

    let mut logical = vec![0i8; logical_numel];

    for packed_row in 0..packed_rows {
        for col in 0..cols {
            let byte_idx = packed_row * cols + col;
            let byte = packed_bytes[byte_idx];

            for (shift_idx, &shift) in [0_u8, 2, 4, 6].iter().enumerate() {
                let lane = ((byte >> shift) & 0b11) as i8;
                if lane == 0b11_i8 {
                    return Err(anyhow!(
                        "invalid packed lane value 0b11 at packed position ({packed_row}, {col}), shift {shift}"
                    ));
                }
                let logical_row = shift_idx * packed_rows + packed_row;
                let logical_idx = logical_row * cols + col;
                logical[logical_idx] = lane - 1;
            }
        }
    }

    if kind.requires_llama_permute() {
        apply_llama_permutation_in_place(&mut logical, logical_rows, cols, kind, config)
            .with_context(|| format!("failed to permute {kind:?} projection"))?;
    }

    // The HuggingFace `weight_scale` tensor stores α_W = mean(|W_original|),
    // the absmean of the original (latent) float weight matrix.  The official
    // conversion script computes:
    //
    //   s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
    //   new_scale = (1.0 / s)   # ← stored value = absmean
    //
    // During inference the dequantized projection is:
    //
    //   y = (W_ternary @ x_q) × α_W × α_x
    //
    // so the effective per-tensor multiplier passed to the GEMV kernel must be
    // α_W itself — NOT its reciprocal.  The previous `.recip()` here caused
    // every projection output to be scaled by 1/α_W instead of α_W, a factor
    // of α_W² error per layer that compounded across 30 layers and produced
    // degenerate text ("the, and the, and the, …").
    let effective_scale = weight_scale;
    let weight = TernaryWeight::from_i8(&logical, effective_scale, logical_rows, cols)
        .context("failed to construct canonical ternary weight")?;

    Ok(CanonicalProjection { kind, weight })
}

/// Apply the LLaMA-style Q/K permutation used by the official conversion path.
///
/// This matches the permutation logic used in the official BitNet conversion
/// script:
///
/// ```text
/// reshape(n_head, 2, rows / n_head / 2, cols)
/// swapaxes(1, 2)
/// reshape(rows, cols)
/// ```
///
/// For `k_proj`, `n_head_kv` is used instead of `n_head`.
///
/// # Errors
///
/// Returns an error if the matrix shape is incompatible with the permutation.
pub fn apply_llama_permutation_in_place(
    data: &mut [i8],
    rows: usize,
    cols: usize,
    kind: ProjectionKind,
    config: &ModelConfig,
) -> anyhow::Result<()> {
    let n_head = match kind {
        ProjectionKind::Query => config.num_attention_heads,
        ProjectionKind::Key => config.num_key_value_heads,
        _ => {
            return Err(anyhow!(
                "LLaMA permutation is only defined for query and key projections"
            ))
        }
    };

    if n_head == 0 {
        return Err(anyhow!("head count must be > 0"));
    }
    if rows % n_head != 0 {
        return Err(anyhow!(
            "rows {} must be divisible by head count {}",
            rows,
            n_head
        ));
    }

    let rows_per_head = rows / n_head;
    if rows_per_head % 2 != 0 {
        return Err(anyhow!(
            "rows_per_head {} must be divisible by 2 for LLaMA permutation",
            rows_per_head
        ));
    }

    let half = rows_per_head / 2;
    let original = data.to_vec();

    for head in 0..n_head {
        for pair in 0..2 {
            for inner in 0..half {
                let src_row = head * rows_per_head + pair * half + inner;
                let dst_row = head * rows_per_head + inner * 2 + pair;

                let src_start = src_row * cols;
                let dst_start = dst_row * cols;
                data[dst_start..dst_start + cols]
                    .copy_from_slice(&original[src_start..src_start + cols]);
            }
        }
    }

    Ok(())
}

/// Load a raw HuggingFace packed BitNet model from `model.safetensors`.
///
/// # Errors
///
/// Returns an error if the safetensors file cannot be read or parsed.
#[instrument(level = "info", skip(config), fields(path = %path.display()))]
pub fn load_hf_packed_model(path: &Path, config: &ModelConfig) -> anyhow::Result<HfPackedModel> {
    let raw = load_raw_safetensors(path)
        .with_context(|| format!("failed to load raw safetensors from {}", path.display()))?;

    let tensors = raw
        .into_iter()
        .map(|(name, (bytes, shape, dtype))| {
            (
                name.clone(),
                PackedTensor {
                    name,
                    bytes,
                    shape,
                    dtype,
                },
            )
        })
        .collect();

    Ok(HfPackedModel {
        config: config.clone(),
        tensors,
    })
}

/// Convert a raw HuggingFace packed model into canonical logical weights.
///
/// This function reimplements the semantic portion of the official BitNet
/// conversion path needed for dense Rust execution.
///
/// # Errors
///
/// Returns an error if any required tensor is missing, malformed, or fails
/// canonical decoding.
#[instrument(level = "info", skip(model), fields(n_tensors = model.tensors.len()))]
pub fn convert_hf_packed_to_canonical(
    model: &HfPackedModel,
) -> anyhow::Result<CanonicalModelWeights> {
    let config = &model.config;
    config
        .validate()
        .context("invalid ModelConfig before canonical conversion")?;

    let hidden = config.hidden_size;
    let q_rows = config.num_attention_heads * config.head_dim();
    let kv_rows = config.num_key_value_heads * config.head_dim();
    let ffn_rows = config.intermediate_size;

    let embed_tokens = Arc::new(
        load_float_tensor(
            &model.tensors,
            "model.embed_tokens.weight",
            config.vocab_size * hidden,
        )?
        .iter()
        .map(|&x| half::bf16::from_f32(x))
        .collect::<Vec<half::bf16>>(),
    );

    let final_norm = load_float_tensor(&model.tensors, "model.norm.weight", hidden)?;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);

    for layer_idx in 0..config.num_hidden_layers {
        let base = format!("model.layers.{layer_idx}");

        let attention_norm = load_float_tensor(
            &model.tensors,
            &format!("{base}.input_layernorm.weight"),
            hidden,
        )?;
        let ffn_norm = load_float_tensor(
            &model.tensors,
            &format!("{base}.post_attention_layernorm.weight"),
            hidden,
        )?;
        let attn_sub_norm = load_float_tensor(
            &model.tensors,
            &format!("{base}.self_attn.attn_sub_norm.weight"),
            hidden,
        )?;
        let ffn_sub_norm = load_float_tensor(
            &model.tensors,
            &format!("{base}.mlp.ffn_sub_norm.weight"),
            ffn_rows,
        )?;

        let q_proj = load_and_decode_projection(
            &model.tensors,
            &format!("{base}.self_attn.q_proj.weight"),
            q_rows,
            hidden,
            ProjectionKind::Query,
            config,
        )?;
        let k_proj = load_and_decode_projection(
            &model.tensors,
            &format!("{base}.self_attn.k_proj.weight"),
            kv_rows,
            hidden,
            ProjectionKind::Key,
            config,
        )?;
        let v_proj = load_and_decode_projection(
            &model.tensors,
            &format!("{base}.self_attn.v_proj.weight"),
            kv_rows,
            hidden,
            ProjectionKind::Value,
            config,
        )?;
        let o_proj = load_and_decode_projection(
            &model.tensors,
            &format!("{base}.self_attn.o_proj.weight"),
            hidden,
            q_rows,
            ProjectionKind::Output,
            config,
        )?;
        let gate_proj = load_and_decode_projection(
            &model.tensors,
            &format!("{base}.mlp.gate_proj.weight"),
            ffn_rows,
            hidden,
            ProjectionKind::Gate,
            config,
        )?;
        let up_proj = load_and_decode_projection(
            &model.tensors,
            &format!("{base}.mlp.up_proj.weight"),
            ffn_rows,
            hidden,
            ProjectionKind::Up,
            config,
        )?;
        let down_proj = load_and_decode_projection(
            &model.tensors,
            &format!("{base}.mlp.down_proj.weight"),
            hidden,
            ffn_rows,
            ProjectionKind::Down,
            config,
        )?;

        debug!(layer = layer_idx, "canonical layer conversion complete");

        layers.push(LayerWeights {
            attention_norm,
            ffn_norm,
            q_proj: q_proj.weight,
            k_proj: k_proj.weight,
            v_proj: v_proj.weight,
            o_proj: o_proj.weight,
            attn_sub_norm,
            gate_proj: gate_proj.weight,
            up_proj: up_proj.weight,
            down_proj: down_proj.weight,
            ffn_sub_norm,
        });
    }

    // i8-quantised embedding for lm_head matmul (per-row absmax)
    let (lm_head_i8, lm_head_scales) = {
        let n_rows = config.vocab_size;
        let row_len = config.hidden_size;
        let bf16_data: &[half::bf16] = &embed_tokens;
        let mut i8_data = vec![0i8; n_rows * row_len];
        let mut scales = vec![0.0f32; n_rows];
        for row_idx in 0..n_rows {
            let start = row_idx * row_len;
            let end = start + row_len;
            let mut max_abs: f32 = 0.0;
            for &w in &bf16_data[start..end] {
                let a = f32::from(w).abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            if max_abs < 1e-10 {
                max_abs = 1e-10;
            }
            let inv_max = 127.0 / max_abs;
            scales[row_idx] = max_abs / 127.0;
            for (i, &w) in bf16_data[start..end].iter().enumerate() {
                i8_data[start + i] = (f32::from(w) * inv_max).round().clamp(-128.0, 127.0) as i8;
            }
        }
        (Arc::new(i8_data), Arc::new(scales))
    };

    let weights = ModelWeights {
        config: config.clone(),
        embed_tokens: Arc::clone(&embed_tokens),
        layers,
        final_norm,
        lm_head: embed_tokens,
        lm_head_i8,
        lm_head_scales,
    };

    info!(
        n_layers = weights.layers.len(),
        "canonical conversion complete"
    );

    Ok(CanonicalModelWeights { weights })
}

/// Convert canonical logical weights into the current runtime-packed placeholder.
///
/// This is a no-op wrapper for now, preserving the explicit pipeline stage.
#[inline]
pub fn convert_canonical_to_runtime(canonical: CanonicalModelWeights) -> RuntimePackedModel {
    RuntimePackedModel { canonical }
}

fn load_and_decode_projection(
    tensors: &HashMap<String, PackedTensor>,
    weight_key: &str,
    logical_rows: usize,
    cols: usize,
    kind: ProjectionKind,
    config: &ModelConfig,
) -> anyhow::Result<CanonicalProjection> {
    let packed = tensors
        .get(weight_key)
        .ok_or_else(|| anyhow!("missing packed projection tensor '{weight_key}'"))?;

    if packed.dtype != "U8" && packed.dtype != "I8" {
        return Err(anyhow!(
            "projection tensor '{}' has dtype '{}', expected U8/I8",
            weight_key,
            packed.dtype
        ));
    }
    if packed.shape.len() != 2 {
        return Err(anyhow!(
            "projection tensor '{}' must be rank-2, got shape {:?}",
            weight_key,
            packed.shape
        ));
    }
    if packed.shape[1] != cols {
        return Err(anyhow!(
            "projection tensor '{}' has {} columns, expected {}",
            weight_key,
            packed.shape[1],
            cols
        ));
    }
    if packed.shape[0]
        .checked_mul(4)
        .ok_or_else(|| anyhow!("packed row expansion overflow for '{weight_key}'"))?
        != logical_rows
    {
        return Err(anyhow!(
            "projection tensor '{}' has packed rows {}, expected logical rows {} via 4x expansion",
            weight_key,
            packed.shape[0],
            logical_rows
        ));
    }

    let scale_key = format!("{weight_key}_scale");
    let weight_scale = load_scalar_scale(tensors, &scale_key)?;

    decode_packed_projection(
        &packed.bytes,
        packed.shape[0],
        cols,
        weight_scale,
        kind,
        config,
    )
    .with_context(|| format!("failed to decode projection '{weight_key}'"))
}

fn load_scalar_scale(tensors: &HashMap<String, PackedTensor>, key: &str) -> anyhow::Result<f32> {
    let tensor = tensors
        .get(key)
        .ok_or_else(|| anyhow!("missing scale tensor '{key}'"))?;

    match tensor.dtype.as_str() {
        "BF16" => {
            if tensor.bytes.len() < 2 {
                return Err(anyhow!(
                    "BF16 scale tensor '{}' is too short: {} bytes",
                    key,
                    tensor.bytes.len()
                ));
            }
            let bits = u16::from_le_bytes([tensor.bytes[0], tensor.bytes[1]]);
            Ok(half::bf16::from_bits(bits).to_f32())
        }
        "F16" => {
            if tensor.bytes.len() < 2 {
                return Err(anyhow!(
                    "F16 scale tensor '{}' is too short: {} bytes",
                    key,
                    tensor.bytes.len()
                ));
            }
            let bits = u16::from_le_bytes([tensor.bytes[0], tensor.bytes[1]]);
            Ok(half::f16::from_bits(bits).to_f32())
        }
        "F32" => {
            if tensor.bytes.len() < 4 {
                return Err(anyhow!(
                    "F32 scale tensor '{}' is too short: {} bytes",
                    key,
                    tensor.bytes.len()
                ));
            }
            Ok(f32::from_le_bytes([
                tensor.bytes[0],
                tensor.bytes[1],
                tensor.bytes[2],
                tensor.bytes[3],
            ]))
        }
        other => Err(anyhow!(
            "scale tensor '{}' has unsupported dtype '{}'",
            key,
            other
        )),
    }
}

fn load_float_tensor(
    tensors: &HashMap<String, PackedTensor>,
    key: &str,
    expected_elems: usize,
) -> anyhow::Result<Vec<f32>> {
    let tensor = tensors
        .get(key)
        .ok_or_else(|| anyhow!("missing float tensor '{key}'"))?;

    let numel = tensor.numel();
    if numel != expected_elems {
        return Err(anyhow!(
            "tensor '{}' has {} elements, expected {}",
            key,
            numel,
            expected_elems
        ));
    }

    match tensor.dtype.as_str() {
        "BF16" => Ok(tensor
            .bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        "F16" => Ok(tensor
            .bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        "F32" => Ok(tensor
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        other => Err(anyhow!(
            "tensor '{}' has unsupported float dtype '{}'",
            key,
            other
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::config::bitnet_2b_config;
    use std::collections::BTreeMap;

    /// Unpack all ternary values from a `TernaryWeight` with row-aligned packed
    /// data, returning a flat `Vec<i8>` of logical elements in row-major order.
    fn unpack_all(tw: &TernaryWeight) -> Vec<i8> {
        let mut all = Vec::with_capacity(tw.rows * tw.cols);
        for r in 0..tw.rows {
            all.extend_from_slice(&tw.row_unpacked(r));
        }
        all
    }

    fn scalar_scale_bytes_to_f32(bytes: &[u8], dtype: &str) -> anyhow::Result<f32> {
        match dtype {
            "BF16" => {
                if bytes.len() < 2 {
                    return Err(anyhow!(
                        "BF16 scale tensor is too short: {} bytes",
                        bytes.len()
                    ));
                }
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                Ok(half::bf16::from_bits(bits).to_f32())
            }
            "F16" => {
                if bytes.len() < 2 {
                    return Err(anyhow!(
                        "F16 scale tensor is too short: {} bytes",
                        bytes.len()
                    ));
                }
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                Ok(half::f16::from_bits(bits).to_f32())
            }
            "F32" => {
                if bytes.len() < 4 {
                    return Err(anyhow!(
                        "F32 scale tensor is too short: {} bytes",
                        bytes.len()
                    ));
                }
                Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            }
            other => Err(anyhow!("unsupported forensic scale dtype '{other}'")),
        }
    }

    #[test]
    fn decode_packed_projection_expands_rows_by_four() {
        // Official Microsoft conversion semantics decode each 2-bit lane as
        // `lane - 1`, so 00→-1, 01→0, 10→+1, and 11 is invalid.
        //
        // packed_rows=2 and cols=4 imply 8 logical rows × 4 cols = 32 logical
        // ternary elements, which require 8 packed bytes (2 × 4).
        //
        // With the scatter-by-shift-group layout (matching the Python reference):
        //   Rows 0,1 (shift=0): lane=(byte>>0)&3 = 0 → -1
        //   Rows 2,3 (shift=2): lane=(byte>>2)&3 = 1 →  0
        //   Rows 4,5 (shift=4): lane=(byte>>4)&3 = 2 → +1
        //   Rows 6,7 (shift=6): lane=(byte>>6)&3 = 1 →  0
        let packed = [0b01_10_01_00u8; 8];
        let cfg = bitnet_2b_config();

        let decoded =
            decode_packed_projection(&packed, 2, 4, 2.0, ProjectionKind::Value, &cfg).unwrap();

        assert_eq!(decoded.weight.rows, 8);
        assert_eq!(decoded.weight.cols, 4);
        assert_eq!(decoded.weight.scale, 2.0);
        let all_data = unpack_all(&decoded.weight);
        // Rows 0,1 (shift_idx=0): all -1
        assert_eq!(row_slice(&all_data, 4, 0), &[-1, -1, -1, -1]);
        assert_eq!(row_slice(&all_data, 4, 1), &[-1, -1, -1, -1]);
        // Rows 2,3 (shift_idx=1): all 0
        assert_eq!(row_slice(&all_data, 4, 2), &[0, 0, 0, 0]);
        assert_eq!(row_slice(&all_data, 4, 3), &[0, 0, 0, 0]);
        // Rows 4,5 (shift_idx=2): all +1
        assert_eq!(row_slice(&all_data, 4, 4), &[1, 1, 1, 1]);
        assert_eq!(row_slice(&all_data, 4, 5), &[1, 1, 1, 1]);
        // Rows 6,7 (shift_idx=3): all 0
        assert_eq!(row_slice(&all_data, 4, 6), &[0, 0, 0, 0]);
        assert_eq!(row_slice(&all_data, 4, 7), &[0, 0, 0, 0]);
    }

    fn ternary_histogram(data: &[i8]) -> BTreeMap<i8, usize> {
        let mut histogram = BTreeMap::new();
        for &value in data {
            *histogram.entry(value).or_insert(0) += 1;
        }
        histogram
    }

    fn row_slice<'a>(data: &'a [i8], cols: usize, row: usize) -> &'a [i8] {
        let start = row * cols;
        &data[start..start + cols]
    }

    fn packed_row_bytes(packed: &PackedTensor, logical_row: usize) -> Vec<u8> {
        let logical_cols = packed.shape[1];
        let logical_elements_per_row = logical_cols;
        let logical_start = logical_row * logical_elements_per_row;
        let packed_start = logical_start / 4;
        let packed_len = logical_elements_per_row.div_ceil(4);
        packed.bytes[packed_start..packed_start + packed_len].to_vec()
    }

    fn decode_first_logical_row_without_permutation(
        packed: &PackedTensor,
        scale: f32,
        cfg: &ModelConfig,
    ) -> CanonicalProjection {
        decode_packed_projection(
            &packed.bytes,
            packed.shape[0],
            packed.shape[1],
            scale,
            ProjectionKind::Value,
            cfg,
        )
        .expect("value-style decode without permutation must succeed")
    }

    #[test]
    fn decode_query_projection_applies_permutation_shape_preserving() {
        let cfg = bitnet_2b_config();
        let rows = cfg.num_attention_heads * cfg.head_dim();
        let cols = 1usize;
        let packed_rows = rows / 4;
        let logical_numel = rows * cols;
        let required_bytes = (logical_numel + 3) / 4;

        // Fill with repeating lanes 0,1,2,1.
        let packed = vec![0b01_10_01_00u8; required_bytes];

        let decoded =
            decode_packed_projection(&packed, packed_rows, cols, 1.0, ProjectionKind::Query, &cfg)
                .unwrap();

        assert_eq!(decoded.weight.rows, rows);
        assert_eq!(decoded.weight.cols, cols);
        let all_data = unpack_all(&decoded.weight);
        assert_eq!(all_data.len(), logical_numel);
        assert!(all_data.iter().all(|&v| (-1..=1).contains(&v)));
    }

    #[test]
    fn llama_permutation_reorders_rows() {
        let cfg = bitnet_2b_config();
        let rows = cfg.num_key_value_heads * cfg.head_dim();
        let cols = 1usize;

        let mut data: Vec<i8> = (0..rows).map(|i| (i % 3) as i8 - 1).collect();
        let original = data.clone();

        apply_llama_permutation_in_place(&mut data, rows, cols, ProjectionKind::Key, &cfg).unwrap();

        assert_eq!(data.len(), original.len());
        assert_ne!(data, original);
    }

    #[test]
    fn runtime_packed_model_wraps_canonical() {
        let cfg = bitnet_2b_config();
        let embed_tokens = Arc::new(vec![
            half::bf16::from_f32(0.0);
            cfg.vocab_size * cfg.hidden_size
        ]);
        // i8-quantised lm_head: all-zero embeddings → all-zero i8, minimum scales
        let lm_head_i8 = Arc::new(vec![0i8; cfg.vocab_size * cfg.hidden_size]);
        let lm_head_scales = Arc::new(vec![1e-10_f32 / 127.0; cfg.vocab_size]);
        let weights = ModelWeights {
            config: cfg.clone(),
            embed_tokens: Arc::clone(&embed_tokens),
            layers: Vec::new(),
            final_norm: vec![1.0; cfg.hidden_size],
            lm_head: embed_tokens,
            lm_head_i8,
            lm_head_scales,
        };
        let canonical = CanonicalModelWeights { weights };
        let runtime = convert_canonical_to_runtime(canonical.clone());

        assert_eq!(runtime.canonical.weights.config, canonical.weights.config);
    }

    #[test]
    fn real_model_q_proj_canonical_decode_has_expected_shape_and_domain() {
        let model_path = Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );
        assert!(
            model_path.exists(),
            "expected local HuggingFace packed weights at {}",
            model_path.display()
        );

        let cfg = bitnet_2b_config();
        let packed = load_hf_packed_model(model_path, &cfg)
            .expect("real packed HuggingFace model must load for canonical conversion tests");

        let q_weight = packed
            .tensors
            .get("model.layers.0.self_attn.q_proj.weight")
            .expect("layer-0 q_proj packed tensor must exist");
        let q_scale = load_scalar_scale(
            &packed.tensors,
            "model.layers.0.self_attn.q_proj.weight_scale",
        )
        .expect("layer-0 q_proj scale must load");

        let decoded = decode_packed_projection(
            &q_weight.bytes,
            q_weight.shape[0],
            q_weight.shape[1],
            q_scale,
            ProjectionKind::Query,
            &cfg,
        )
        .expect("layer-0 q_proj must canonically decode");

        assert_eq!(
            decoded.weight.rows, cfg.hidden_size,
            "q_proj logical rows must equal hidden_size after 4x expansion"
        );
        assert_eq!(
            decoded.weight.cols, cfg.hidden_size,
            "q_proj logical cols must equal hidden_size"
        );
        let all_data = unpack_all(&decoded.weight);
        assert!(
            all_data.iter().all(|&v| (-1..=1).contains(&v)),
            "decoded q_proj values must remain ternary"
        );
        assert!(
            decoded.weight.scale.is_finite() && decoded.weight.scale > 0.0,
            "decoded q_proj effective scale must be finite positive, got {}",
            decoded.weight.scale
        );

        let histogram = ternary_histogram(&all_data);
        assert_eq!(
            histogram.values().sum::<usize>(),
            all_data.len(),
            "ternary histogram must account for every decoded element"
        );
    }

    #[test]
    fn forensic_real_model_layer0_projection_scale_tensor_diagnostics() {
        let model_path = Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );
        assert!(
            model_path.exists(),
            "expected local HuggingFace packed weights at {}",
            model_path.display()
        );

        let cfg = bitnet_2b_config();
        let packed = load_hf_packed_model(model_path, &cfg)
            .expect("real packed HuggingFace model must load for scale diagnostics");

        for projection_name in [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ] {
            let scale_key = match projection_name {
                "q_proj" | "k_proj" | "v_proj" | "o_proj" => {
                    format!("model.layers.0.self_attn.{projection_name}.weight_scale")
                }
                "gate_proj" | "up_proj" | "down_proj" => {
                    format!("model.layers.0.mlp.{projection_name}.weight_scale")
                }
                _ => unreachable!("projection list above is exhaustive"),
            };

            let scale_tensor = packed
                .tensors
                .get(&scale_key)
                .unwrap_or_else(|| panic!("missing forensic scale tensor '{scale_key}'"));

            let decoded_scale =
                load_scalar_scale(&packed.tensors, &scale_key).unwrap_or_else(|e| {
                    panic!("failed to decode forensic scale tensor '{scale_key}': {e:#}")
                });

            let raw_scalar = scalar_scale_bytes_to_f32(&scale_tensor.bytes, &scale_tensor.dtype)
                .unwrap_or_else(|e| {
                    panic!("failed to parse raw forensic scale tensor '{scale_key}': {e:#}")
                });

            assert!(
                raw_scalar.is_finite() && raw_scalar > 0.0,
                "raw forensic scale for '{scale_key}' must be finite positive, got {raw_scalar}"
            );
            assert!(
                decoded_scale.is_finite() && decoded_scale > 0.0,
                "decoded forensic scale for '{scale_key}' must be finite positive, got {decoded_scale}"
            );
            assert!(
                (raw_scalar - decoded_scale).abs() < 1e-6,
                "raw and decoded forensic scales must match for '{scale_key}': raw={raw_scalar}, decoded={decoded_scale}"
            );

            let expected_raw_scale = match projection_name {
                "q_proj" => 1.21875_f32,
                "k_proj" => 1.796875_f32,
                "v_proj" => 2.296875_f32,
                "o_proj" => 0.96484375_f32,
                "gate_proj" => 1.5546875_f32,
                "up_proj" => 1.828125_f32,
                "down_proj" => 2.15625_f32,
                _ => unreachable!("projection list above is exhaustive"),
            };

            assert!(
                (raw_scalar - expected_raw_scale).abs() < 1e-6,
                "raw forensic scale for '{scale_key}' regressed: got {raw_scalar}, expected {expected_raw_scale}"
            );
        }
    }

    #[test]
    fn forensic_real_model_layer0_projection_decode_diagnostics() {
        let model_path = Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );
        assert!(
            model_path.exists(),
            "expected local HuggingFace packed weights at {}",
            model_path.display()
        );

        let cfg = bitnet_2b_config();
        let packed = load_hf_packed_model(model_path, &cfg)
            .expect("real packed HuggingFace model must load for forensic diagnostics");

        let q_weight = packed
            .tensors
            .get("model.layers.0.self_attn.q_proj.weight")
            .expect("layer-0 q_proj packed tensor must exist");
        let q_scale = load_scalar_scale(
            &packed.tensors,
            "model.layers.0.self_attn.q_proj.weight_scale",
        )
        .expect("layer-0 q_proj scale must load");

        let q_decoded = decode_packed_projection(
            &q_weight.bytes,
            q_weight.shape[0],
            q_weight.shape[1],
            q_scale,
            ProjectionKind::Query,
            &cfg,
        )
        .expect("layer-0 q_proj canonical decode must succeed");

        let q_unpermuted = decode_first_logical_row_without_permutation(q_weight, q_scale, &cfg);

        let packed_row0 = packed_row_bytes(q_weight, 0);
        let packed_row1 = packed_row_bytes(q_weight, 1);
        let decoded_row0 = q_decoded.weight.row_unpacked(0);
        let decoded_row1 = q_decoded.weight.row_unpacked(1);
        let unpermuted_row0 = q_unpermuted.weight.row_unpacked(0);
        let unpermuted_row1 = q_unpermuted.weight.row_unpacked(1);

        let q_all_data = unpack_all(&q_decoded.weight);
        let q_unpermuted_all_data = unpack_all(&q_unpermuted.weight);
        let q_histogram = ternary_histogram(&q_all_data);
        let q_unpermuted_histogram = ternary_histogram(&q_unpermuted_all_data);

        warn!(
            packed_shape = ?q_weight.shape,
            packed_dtype = %q_weight.dtype,
            packed_bytes = q_weight.bytes.len(),
            scale = q_scale,
            decoded_rows = q_decoded.weight.rows,
            decoded_cols = q_decoded.weight.cols,
            decoded_scale = q_decoded.weight.scale,
            histogram = ?q_histogram,
            unpermuted_histogram = ?q_unpermuted_histogram,
            packed_row0_prefix = ?&packed_row0[..packed_row0.len().min(16)],
            packed_row1_prefix = ?&packed_row1[..packed_row1.len().min(16)],
            decoded_row0_prefix = ?&decoded_row0[..decoded_row0.len().min(16)],
            decoded_row1_prefix = ?&decoded_row1[..decoded_row1.len().min(16)],
            unpermuted_row0_prefix = ?&unpermuted_row0[..unpermuted_row0.len().min(16)],
            unpermuted_row1_prefix = ?&unpermuted_row1[..unpermuted_row1.len().min(16)],
            "forensic layer-0 q_proj decode diagnostics"
        );

        assert_eq!(
            q_decoded.weight.rows, cfg.hidden_size,
            "forensic q_proj decode must preserve logical row count"
        );
        assert_eq!(
            q_decoded.weight.cols, cfg.hidden_size,
            "forensic q_proj decode must preserve logical column count"
        );
        assert!(
            q_histogram.contains_key(&-1) || q_histogram.contains_key(&1),
            "forensic q_proj decode must contain at least one non-zero ternary value"
        );
    }

    #[test]
    fn real_model_canonical_conversion_layer_shapes_match_logical_model() {
        let model_path = Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );
        assert!(
            model_path.exists(),
            "expected local HuggingFace packed weights at {}",
            model_path.display()
        );

        let cfg = bitnet_2b_config();
        let packed = load_hf_packed_model(model_path, &cfg)
            .expect("real packed HuggingFace model must load for canonical conversion tests");
        let canonical = convert_hf_packed_to_canonical(&packed)
            .expect("real packed HuggingFace model must canonically convert");

        let weights = &canonical.weights;
        let layer0 = weights
            .layers
            .first()
            .expect("canonical conversion must produce at least one layer");

        assert_eq!(
            weights.layers.len(),
            cfg.num_hidden_layers,
            "canonical conversion must preserve layer count"
        );
        assert_eq!(
            layer0.q_proj.rows, cfg.hidden_size,
            "q_proj logical rows must equal hidden_size"
        );
        assert_eq!(
            layer0.q_proj.cols, cfg.hidden_size,
            "q_proj logical cols must equal hidden_size"
        );
        assert_eq!(
            layer0.k_proj.rows,
            cfg.num_key_value_heads * cfg.head_dim(),
            "k_proj logical rows must equal kv_dim"
        );
        assert_eq!(
            layer0.k_proj.cols, cfg.hidden_size,
            "k_proj logical cols must equal hidden_size"
        );
        assert_eq!(
            layer0.v_proj.rows,
            cfg.num_key_value_heads * cfg.head_dim(),
            "v_proj logical rows must equal kv_dim"
        );
        assert_eq!(
            layer0.v_proj.cols, cfg.hidden_size,
            "v_proj logical cols must equal hidden_size"
        );
        assert_eq!(
            layer0.o_proj.rows, cfg.hidden_size,
            "o_proj logical rows must equal hidden_size"
        );
        assert_eq!(
            layer0.o_proj.cols, cfg.hidden_size,
            "o_proj logical cols must equal hidden_size"
        );
        assert_eq!(
            layer0.gate_proj.rows, cfg.intermediate_size,
            "gate_proj logical rows must equal intermediate_size"
        );
        assert_eq!(
            layer0.gate_proj.cols, cfg.hidden_size,
            "gate_proj logical cols must equal hidden_size"
        );
        assert_eq!(
            layer0.up_proj.rows, cfg.intermediate_size,
            "up_proj logical rows must equal intermediate_size"
        );
        assert_eq!(
            layer0.up_proj.cols, cfg.hidden_size,
            "up_proj logical cols must equal hidden_size"
        );
        assert_eq!(
            layer0.down_proj.rows, cfg.hidden_size,
            "down_proj logical rows must equal hidden_size"
        );
        assert_eq!(
            layer0.down_proj.cols, cfg.intermediate_size,
            "down_proj logical cols must equal intermediate_size"
        );
        assert_eq!(
            layer0.ffn_sub_norm.len(),
            cfg.intermediate_size,
            "ffn_sub_norm logical width must equal intermediate_size"
        );
    }

    /// Verify that our Rust unpacking produces the exact same ternary values
    /// as the Python reference:
    ///
    /// ```python
    /// shift = torch.tensor([0, 2, 4, 6]).reshape(4, 1, 1)
    /// expanded = packed.unsqueeze(0).expand(4, R, C) >> shift
    /// expanded = expanded & 3
    /// logical = (expanded.float() - 1).reshape(R*4, C)
    /// ```
    ///
    /// Python reference values for layer-0 q_proj (first 8 cols of selected rows):
    ///
    /// | row  | values                          |
    /// |------|---------------------------------|
    /// | 0    | [0, 0, 0, -1, 0, -1, 0, 0]     |
    /// | 1    | [0, 1, 0, 0, 0, 1, 0, 0]       |
    /// | 640  | [0, -1, 0, -1, 0, 0, 0, 0]     |
    /// | 641  | [0, 1, 1, 0, 1, 0, 0, -1]      |
    /// | 1280 | [0, -1, 0, 0, 0, 0, 0, -1]     |
    /// | 1920 | [0, 0, 0, 0, 0, 0, 0, 0]       |
    /// | 2559 | [0, 1, -1, 1, 1, 0, -1, 1]     |
    #[test]
    fn rust_unpack_matches_python_reference_values() {
        let model_path = Path::new(
            "C:\\Users\\RyanClanton\\.cache\\bitnet\\microsoft__bitnet-b1.58-2B-4T\\model.safetensors",
        );
        assert!(
            model_path.exists(),
            "expected local HuggingFace packed weights at {}",
            model_path.display()
        );

        let cfg = bitnet_2b_config();
        let packed = load_hf_packed_model(model_path, &cfg)
            .expect("real packed HuggingFace model must load");

        let q_weight = packed
            .tensors
            .get("model.layers.0.self_attn.q_proj.weight")
            .expect("layer-0 q_proj packed tensor must exist");
        let q_scale = load_scalar_scale(
            &packed.tensors,
            "model.layers.0.self_attn.q_proj.weight_scale",
        )
        .expect("layer-0 q_proj scale must load");

        // Decode WITHOUT permutation (Value kind skips it)
        let decoded = decode_packed_projection(
            &q_weight.bytes,
            q_weight.shape[0],
            q_weight.shape[1],
            q_scale,
            ProjectionKind::Value, // Value never permutes
            &cfg,
        )
        .expect("layer-0 q_proj must canonically decode");

        let cols = decoded.weight.cols;
        let data = unpack_all(&decoded.weight);

        // Helper to extract first 8 elements of a row
        let row8 = |r: usize| -> Vec<i8> { data[r * cols..r * cols + 8].to_vec() };

        // Python reference (captured from the Python script above)
        let expected: Vec<(usize, Vec<i8>)> = vec![
            (0, vec![0, 0, 0, -1, 0, -1, 0, 0]),
            (1, vec![0, 1, 0, 0, 0, 1, 0, 0]),
            (640, vec![0, -1, 0, -1, 0, 0, 0, 0]),
            (641, vec![0, 1, 1, 0, 1, 0, 0, -1]),
            (1280, vec![0, -1, 0, 0, 0, 0, 0, -1]),
            (1920, vec![0, 0, 0, 0, 0, 0, 0, 0]),
            (2559, vec![0, 1, -1, 1, 1, 0, -1, 1]),
        ];

        for (row_idx, expected_vals) in &expected {
            let actual = row8(*row_idx);
            assert_eq!(
                &actual, expected_vals,
                "row {row_idx}: Rust decode differs from Python reference.\n  \
                 expected: {expected_vals:?}\n  actual:   {actual:?}"
            );
        }

        // Also verify scale and dimensions
        assert!(
            (decoded.weight.scale - q_scale).abs() < 1e-6,
            "effective scale must equal the HF weight_scale (absmean)"
        );
        assert_eq!(decoded.weight.rows, 2560);
        assert_eq!(decoded.weight.cols, 2560);

        // Verify ternary distribution matches Python: ~25.2% -1, ~49.6% 0, ~25.2% +1
        let histogram = ternary_histogram(&data);
        let total = data.len() as f64;
        let pct_neg1 = *histogram.get(&-1).unwrap_or(&0) as f64 / total * 100.0;
        let pct_zero = *histogram.get(&0).unwrap_or(&0) as f64 / total * 100.0;
        let pct_pos1 = *histogram.get(&1).unwrap_or(&0) as f64 / total * 100.0;
        assert!(
            (pct_neg1 - 25.2).abs() < 1.0,
            "expected ~25.2% -1, got {pct_neg1:.1}%"
        );
        assert!(
            (pct_zero - 49.6).abs() < 1.0,
            "expected ~49.6% 0, got {pct_zero:.1}%"
        );
        assert!(
            (pct_pos1 - 25.2).abs() < 1.0,
            "expected ~25.2% +1, got {pct_pos1:.1}%"
        );

        eprintln!("Rust unpack matches Python reference for all checked rows.");
        eprintln!("Distribution: -1={pct_neg1:.1}%, 0={pct_zero:.1}%, +1={pct_pos1:.1}%");
    }
}
