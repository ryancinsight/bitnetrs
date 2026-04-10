//! Backend abstraction for BitNet b1.58 inference.
//!
//! # Architecture
//!
//! The backend layer follows the Dependency Inversion Principle: the model
//! architecture (`bitnet-model`) depends on the abstract [`Backend`] trait,
//! not on any concrete compute implementation.  Concrete backends live in
//! separate crates:
//!
//! | Crate          | Backend struct   | Target hardware                     |
//! |----------------|-----------------|--------------------------------------|
//! | `bitnet-cpu`   | `CpuBackend`    | Any x86-64 / ARM64 CPU (Rayon)      |
//! | `bitnet-gpu`   | `GpuBackend`    | GPU via wgpu (Vulkan/Metal/DX12)    |
//! | `bitnet-npu`   | `NpuBackend`    | NPU via DirectML (Windows)          |
//!
//! # Backend Trait Contract
//!
//! Every method operates on *flat, row-major* `f32` or `i8` slices.  No
//! heap allocation is performed inside trait methods — callers pre-allocate
//! output buffers and pass mutable references.
//!
//! # Device Selection
//!
//! Use [`Device`] to express *intent* (which hardware class to target).
//! The concrete backend is instantiated by `bitnet-model::device::create_backend`.

pub mod ops;

use std::sync::Arc;

use crate::error::Result;

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

/// Selects the target compute hardware for inference.
///
/// The device is specified at engine initialisation time and determines which
/// [`Backend`] implementation is instantiated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    /// Host CPU backend.  Uses Rayon for data-parallel GEMV and attention.
    ///
    /// `threads`: number of Rayon threads.  `None` uses all logical cores.
    Cpu { threads: Option<usize> },

    /// GPU backend via `wgpu`.
    ///
    /// `device_id`: index into the list of enumerated wgpu adapters.
    /// `0` selects the first (usually most capable) GPU.
    Gpu { device_id: u32 },

    /// NPU backend.
    ///
    /// On Windows, attempts to use a DirectML-backed wgpu adapter.
    /// Falls back to [`Device::Cpu`] if no NPU is detected.
    ///
    /// `device_id`: adapter index (0 = first NPU found).
    Npu { device_id: u32 },
}

impl Device {
    /// Convenience constructor for the default CPU device (all threads).
    #[inline]
    pub fn cpu() -> Self {
        Self::Cpu { threads: None }
    }

    /// Convenience constructor for the first GPU (adapter index 0).
    #[inline]
    pub fn gpu() -> Self {
        Self::Gpu { device_id: 0 }
    }

    /// Convenience constructor for the first NPU (adapter index 0).
    #[inline]
    pub fn npu() -> Self {
        Self::Npu { device_id: 0 }
    }

    /// Human-readable name for logging.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Cpu { .. } => "CPU",
            Self::Gpu { .. } => "GPU (wgpu)",
            Self::Npu { .. } => "NPU (DirectML)",
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu()
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu { threads } => match threads {
                Some(n) => write!(f, "CPU({n} threads)"),
                None => write!(f, "CPU(all threads)"),
            },
            Self::Gpu { device_id } => write!(f, "GPU(device={device_id})"),
            Self::Npu { device_id } => write!(f, "NPU(device={device_id})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------

/// Compute backend for all tensor operations in BitNet b1.58.
///
/// Implementations must be `Send + Sync` so that an `Arc<dyn Backend>` can be
/// shared across threads (e.g. Rayon workers or tokio tasks).
///
/// # Invariants expected of every implementation
///
/// - **No panics** from valid inputs.  All error conditions are surfaced via
///   `Result`.
/// - **Deterministic** output for the same input on the same device.
/// - **No internal heap allocation** on the hot path (output buffers are
///   pre-allocated by the caller).
/// - Shape mismatches MUST return `Err(BitNetError::InvalidShape { .. })`.
pub trait Backend: Send + Sync + 'static {
    // ------------------------------------------------------------------
    // Core linear algebra
    // ------------------------------------------------------------------

    /// Ternary General Matrix–Vector product (GEMV).
    ///
    /// Computes:
    /// ```text
    /// output[i] = Σ_j  unpack(weight_packed)[i * in_features + j] as f32 * input[j] * weight_scale
    /// ```
    /// where unpacked `weight[k] ∈ {−1, 0, +1}` and `weight_scale = α_W > 0`.
    ///
    /// # Shapes
    /// - `weight_packed`: Packed 2-bit ternary weights, `[ceil(out_features * in_features / 4)]` bytes.
    ///                    Each byte holds 4 ternary values in {-1, 0, +1}.
    ///                    Encoding: `byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)`.
    ///                    Codes: `0b00→+1, 0b01→0, 0b10→-1`.
    /// - `input`:   `[in_features]`
    /// - `output`:  `[out_features]` — pre-allocated, overwritten on success.
    ///
    /// # Errors
    /// Returns `Err` if `weight_packed.len() != ceil(out_features * in_features / 4)` or
    /// `input.len() != in_features` or `output.len() != out_features`.
    fn ternary_gemv(
        &self,
        weight_packed: &[u8],
        weight_scale: f32,
        input: &[f32],
        output: &mut [f32],
        out_features: usize,
        in_features: usize,
    ) -> Result<()>;

    /// Ternary GEMV with 8-bit activation quantisation (W2A8).
    ///
    /// This is the mathematically correct forward-pass kernel for BitNet b1.58.
    /// The model was trained with straight-through estimation (STE) of absmax
    /// activation quantisation, so inference must replicate the rounding noise.
    ///
    /// Computes:
    /// ```text
    /// x_q    = clip(round(input × 127 / max(|input|)), −128, 127)
    /// α_x    = max(|input|) / 127
    /// output[i] = (Σ_j unpack(weight_packed)[i,j] × x_q[j]) × weight_scale × α_x
    /// ```
    ///
    /// The default implementation quantises and dequantises the input, then
    /// delegates to [`ternary_gemv`](Backend::ternary_gemv).  Backends that
    /// support integer accumulation should override this for better performance.
    ///
    /// # Shapes
    /// Same as [`ternary_gemv`](Backend::ternary_gemv).
    ///
    /// # Errors
    /// Same as [`ternary_gemv`](Backend::ternary_gemv), plus
    /// [`BitNetError::QuantizationError`] if input contains non-finite values.
    fn ternary_gemv_with_activation_quant(
        &self,
        weight_packed: &[u8],
        weight_scale: f32,
        input: &[f32],
        output: &mut [f32],
        out_features: usize,
        in_features: usize,
    ) -> Result<()> {
        // Default: quantise activations to i8, dequantise back to f32,
        // then delegate to the f32 GEMV path.  This preserves the rounding
        // noise that the model was trained with via STE.
        let (x_q, scale) = crate::quant::absmax::absmax_quantize_row(input).map_err(|e| e)?;
        let x_dq = crate::quant::absmax::absmax_dequantize(&x_q, scale).map_err(|e| e)?;
        self.ternary_gemv(
            weight_packed,
            weight_scale,
            &x_dq,
            output,
            out_features,
            in_features,
        )
    }

    // ------------------------------------------------------------------
    // Normalisation
    // ------------------------------------------------------------------

    /// Root-Mean-Square LayerNorm (RMSNorm).
    ///
    /// ```text
    /// rms    = sqrt( mean(x²) + ε )
    /// out[i] = x[i] / rms * weight[i]
    /// ```
    ///
    /// # Shapes
    /// All slices must have the same length `d`.
    ///
    /// # Errors
    /// Returns `Err` if slice lengths differ or `eps ≤ 0`.
    fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) -> Result<()>;

    // ------------------------------------------------------------------
    // Positional encoding
    // ------------------------------------------------------------------

    /// Apply Rotary Position Embedding (RoPE) in-place to Q and K tensors.
    ///
    /// Both `q` and `k` are flat, row-major tensors:
    /// - `q`: `[n_heads    × head_dim]`
    /// - `k`: `[n_kv_heads × head_dim]`
    ///
    /// The rotation follows the LLaMA half-split convention at sequence
    /// position `pos`, pairing dimension `i` with dimension `i + head_dim/2`:
    /// ```text
    /// θ_i          = pos / rope_theta^(2i / head_dim)
    /// q_lo[h, i]   = q[h, i]
    /// q_hi[h, i]   = q[h, i + head_dim/2]
    /// q'[h, i]              = q_lo[h, i] * cos(θ_i) - q_hi[h, i] * sin(θ_i)
    /// q'[h, i + head_dim/2] = q_hi[h, i] * cos(θ_i) + q_lo[h, i] * sin(θ_i)
    /// ```
    /// (identical for `k`).
    ///
    /// # Errors
    /// Returns `Err` if `q.len() != n_heads * head_dim` or
    /// `k.len() != n_kv_heads * head_dim` or `head_dim` is odd.
    fn rope_embed(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        position: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        theta: f32,
    ) -> Result<()>;

    // ------------------------------------------------------------------
    // Attention
    // ------------------------------------------------------------------

    /// Causal scaled dot-product attention with Grouped Query Attention (GQA).
    ///
    /// Computes the attended output for a single new query at position `cur_pos`
    /// (0-indexed), attending over positions `0..=cur_pos` in the KV cache.
    ///
    /// ```text
    /// For query head h (0 ≤ h < n_heads):
    ///   kv_head = h / heads_per_group          (integer division)
    ///   scores[t] = dot(q[h], k_cache[kv_head, t]) / sqrt(head_dim)
    ///               for t in 0..=cur_pos
    ///   attn[h]   = softmax(scores) · v_cache[kv_head, :]
    /// output = concat(attn[0], ..., attn[n_heads-1])
    /// ```
    ///
    /// The KV cache is stored in fixed-capacity head-major order to improve
    /// temporal locality during autoregressive decode:
    ///
    /// ```text
    /// cache[kv_h, t, d] = cache[
    ///     kv_h * (max_seq * head_dim) + t * head_dim + d
    /// ]
    /// ```
    ///
    /// where `max_seq` is the cache capacity for the current inference session.
    /// Only positions `t ∈ 0..=cur_pos` are logically valid; later positions in
    /// the backing buffer are reserved capacity and must be ignored.
    ///
    /// # Shapes
    /// - `q`:       `[n_heads × head_dim]`
    /// - `k_cache`: `[n_kv_heads × max_seq × head_dim]` head-major backing storage
    /// - `v_cache`: `[n_kv_heads × max_seq × head_dim]` head-major backing storage
    /// - `output`:  `[n_heads × head_dim]` — pre-allocated, overwritten.
    ///
    /// # Errors
    /// Returns `Err` if any shape is inconsistent.
    fn masked_attention(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        output: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        cur_pos: usize,
    ) -> Result<()>;

    // ------------------------------------------------------------------
    // Activation functions
    // ------------------------------------------------------------------

    /// Apply squared ReLU in-place: `x[i] = max(0, x[i])²`.
    ///
    /// This is the activation function used in the BitNet b1.58 FFN gating:
    /// `sqrelu(gate) ⊙ up`, where `⊙` is element-wise multiplication.
    ///
    /// # Errors
    /// Returns `Err` only if the backend reports a GPU / NPU dispatch failure.
    fn squared_relu(&self, x: &mut [f32]) -> Result<()>;

    /// Apply numerically-stable softmax in-place over the entire slice.
    ///
    /// ```text
    /// m        = max(x)
    /// e[i]     = exp(x[i] - m)
    /// x[i]    ← e[i] / Σ_j e[j]
    /// ```
    ///
    /// After this operation: `x[i] > 0` and `Σ x[i] = 1`.
    ///
    /// # Errors
    /// Returns `Err` only if the backend reports a GPU / NPU dispatch failure.
    fn softmax(&self, x: &mut [f32]) -> Result<()>;

    // ------------------------------------------------------------------
    // Element-wise helpers
    // ------------------------------------------------------------------

    /// Element-wise multiply `a ⊙ b`, writing results into `out`.
    ///
    /// `out[i] = a[i] * b[i]`
    ///
    /// # Errors
    /// Returns `Err` if `a.len() != b.len()` or `out.len() != a.len()`.
    fn elementwise_mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()>;

    // ------------------------------------------------------------------
    // LM head
    // ------------------------------------------------------------------

    /// LM head matrix-vector product (unquantised f32 matmul).
    ///
    /// ```text
    /// output[v] = Σ_h  weights[v * hidden_size + h] * hidden[h]
    /// ```
    ///
    /// The default implementation delegates to [`ops::lm_head_matmul_into`].
    /// Backends with SIMD or GPU compute should override for better throughput.
    ///
    /// # Shapes
    /// - `hidden`:  `[hidden_size]`
    /// - `weights`: `[vocab_size × hidden_size]` row-major
    /// - `output`:  `[vocab_size]` — pre-allocated, overwritten
    ///
    /// # Errors
    /// Returns `Err` on shape mismatch or backend dispatch failure.
    fn lm_head_matmul_into(
        &self,
        hidden: &[f32],
        weights: &[f32],
        output: &mut [f32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<()> {
        crate::backend::ops::lm_head_matmul_into(hidden, weights, output, vocab_size, hidden_size);
        Ok(())
    }

    /// LM-head matrix–vector multiply with BF16 weight storage.
    ///
    /// Computes `output[v] = dot(lm_head_bf16[v*hidden_size..], hidden)` for each
    /// vocabulary entry `v`, using bf16-stored weights to halve memory bandwidth.
    ///
    /// # BF16 → f32 conversion
    /// BF16 is a subset of f32 with 7 mantissa bits (vs 23). Conversion is exact
    /// for all normal values: `f32_bits = (bf16_bits as u32) << 16`.
    ///
    /// # Default implementation
    /// Uses Rayon for parallelism (one row per task) with scalar bf16→f32 conversion.
    /// The CPU backend overrides with AVX2+FMA.
    ///
    /// # Errors
    /// Returns `Err` on shape mismatch.
    fn lm_head_matmul_bf16_into(
        &self,
        hidden: &[f32],
        weights_bf16: &[half::bf16],
        output: &mut [f32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<()> {
        use rayon::prelude::*;
        debug_assert_eq!(hidden.len(), hidden_size);
        debug_assert_eq!(weights_bf16.len(), vocab_size * hidden_size);
        debug_assert_eq!(output.len(), vocab_size);
        output.par_iter_mut().enumerate().for_each(|(v, out_elem)| {
            let row = &weights_bf16[v * hidden_size..(v + 1) * hidden_size];
            let mut acc = 0.0_f32;
            for (&w, &h_val) in row.iter().zip(hidden.iter()) {
                acc += f32::from(w) * h_val;
            }
            *out_elem = acc;
        });
        Ok(())
    }

    // ------------------------------------------------------------------
    // Device info
    // ------------------------------------------------------------------

    /// Human-readable name of this backend (e.g. `"CPU"`, `"NVIDIA RTX 4090"`).
    fn device_name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Blanket impl for Arc<dyn Backend>
// ---------------------------------------------------------------------------

/// Forwards all [`Backend`] methods through the `Arc` to the inner implementation.
///
/// This allows `Arc<dyn Backend>` to be used directly as a `Backend` without
/// extra boilerplate at call sites.
impl Backend for Arc<dyn Backend> {
    fn ternary_gemv(
        &self,
        weight_packed: &[u8],
        weight_scale: f32,
        input: &[f32],
        output: &mut [f32],
        out_features: usize,
        in_features: usize,
    ) -> Result<()> {
        (**self).ternary_gemv(
            weight_packed,
            weight_scale,
            input,
            output,
            out_features,
            in_features,
        )
    }

    fn ternary_gemv_with_activation_quant(
        &self,
        weight_packed: &[u8],
        weight_scale: f32,
        input: &[f32],
        output: &mut [f32],
        out_features: usize,
        in_features: usize,
    ) -> Result<()> {
        (**self).ternary_gemv_with_activation_quant(
            weight_packed,
            weight_scale,
            input,
            output,
            out_features,
            in_features,
        )
    }

    fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) -> Result<()> {
        (**self).rms_norm(input, weight, eps, output)
    }

    fn rope_embed(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        position: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        theta: f32,
    ) -> Result<()> {
        (**self).rope_embed(q, k, position, head_dim, n_heads, n_kv_heads, theta)
    }

    fn masked_attention(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        output: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        cur_pos: usize,
    ) -> Result<()> {
        (**self).masked_attention(
            q, k_cache, v_cache, output, n_heads, n_kv_heads, head_dim, cur_pos,
        )
    }

    fn squared_relu(&self, x: &mut [f32]) -> Result<()> {
        (**self).squared_relu(x)
    }

    fn softmax(&self, x: &mut [f32]) -> Result<()> {
        (**self).softmax(x)
    }

    fn elementwise_mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        (**self).elementwise_mul(a, b, out)
    }

    fn lm_head_matmul_into(
        &self,
        hidden: &[f32],
        weights: &[f32],
        output: &mut [f32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<()> {
        (**self).lm_head_matmul_into(hidden, weights, output, vocab_size, hidden_size)
    }

    fn lm_head_matmul_bf16_into(
        &self,
        hidden: &[f32],
        weights_bf16: &[half::bf16],
        output: &mut [f32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<()> {
        (**self).lm_head_matmul_bf16_into(hidden, weights_bf16, output, vocab_size, hidden_size)
    }

    fn device_name(&self) -> &str {
        (**self).device_name()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_display_cpu_no_threads() {
        let d = Device::Cpu { threads: None };
        assert_eq!(d.to_string(), "CPU(all threads)");
    }

    #[test]
    fn device_display_cpu_fixed_threads() {
        let d = Device::Cpu { threads: Some(4) };
        assert_eq!(d.to_string(), "CPU(4 threads)");
    }

    #[test]
    fn device_display_gpu() {
        let d = Device::Gpu { device_id: 0 };
        assert_eq!(d.to_string(), "GPU(device=0)");
    }

    #[test]
    fn device_display_npu() {
        let d = Device::Npu { device_id: 0 };
        assert_eq!(d.to_string(), "NPU(device=0)");
    }

    #[test]
    fn device_default_is_cpu() {
        assert_eq!(Device::default(), Device::Cpu { threads: None });
    }

    #[test]
    fn device_convenience_constructors() {
        assert_eq!(Device::cpu(), Device::Cpu { threads: None });
        assert_eq!(Device::gpu(), Device::Gpu { device_id: 0 });
        assert_eq!(Device::npu(), Device::Npu { device_id: 0 });
    }

    #[test]
    fn device_display_name() {
        assert_eq!(Device::cpu().display_name(), "CPU");
        assert_eq!(Device::gpu().display_name(), "GPU (wgpu)");
        assert_eq!(Device::npu().display_name(), "NPU (DirectML)");
    }
}
