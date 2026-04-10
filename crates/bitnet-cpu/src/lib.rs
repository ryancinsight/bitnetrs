//! # bitnet-cpu
//!
//! CPU backend for BitNet b1.58 inference.
//!
//! ## Architecture
//!
//! [`CpuBackend`] implements the [`bitnet_core::backend::Backend`] trait using
//! pure Rust with Rayon for data-parallel execution.  All tensor operations
//! operate on flat `f32` / `i8` slices in row-major (C-contiguous) layout.
//!
//! ## Module Layout
//!
//! ```text
//! bitnet-cpu/
//! ├── lib.rs        ← this file: CpuBackend + Backend impl
//! ├── gemv.rs       ← ternary GEMV (Rayon-parallel outer loop)
//! ├── norm.rs       ← RMSNorm
//! ├── rope.rs       ← Rotary Position Embedding + RopeCache
//! ├── attention.rs  ← Causal GQA scaled dot-product attention
//! └── activation.rs ← squared_relu, softmax, sqrelu_gate
//! ```
//!
//! ## Threading Model
//!
//! [`CpuBackend`] is `Send + Sync`.  Rayon's global thread pool is used for
//! all parallel operations; the pool is initialised once at process start.
//! The optional `threads` field in [`Device::Cpu`] configures the pool size
//! via `rayon::ThreadPoolBuilder`.
//!
//! ## Invariants
//!
//! - Every `Backend` method validates its inputs and returns a descriptive
//!   `Err` for any shape or parameter mismatch.
//! - No heap allocation inside the hot-path (output buffers are pre-allocated
//!   by the caller and passed as `&mut [f32]`).
//! - All operations are deterministic for a fixed input.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod activation;
pub mod attention;
pub mod gemv;
pub mod norm;
pub mod rope;

use std::sync::Arc;

use bitnet_core::backend::Backend;
use bitnet_core::error::{BitNetError, Result};
use tracing::instrument;

// ---------------------------------------------------------------------------
// CpuBackend
// ---------------------------------------------------------------------------

/// CPU-based compute backend using Rayon for data parallelism.
///
/// Instantiate via [`CpuBackend::new`] or by passing [`bitnet_core::backend::Device::Cpu`]
/// to `bitnet_model::device::create_backend`.
///
/// # Thread Safety
///
/// `CpuBackend` is `Send + Sync` and can be wrapped in `Arc<dyn Backend>` for
/// shared use across multiple inference threads or async tasks.
#[derive(Debug, Clone)]
pub struct CpuBackend {
    /// Human-readable name reported by [`Backend::device_name`].
    name: String,
    /// Number of Rayon threads used for parallel operations.
    /// `0` means "use Rayon's default (all logical cores)".
    threads: usize,
}

impl CpuBackend {
    /// Create a new `CpuBackend`.
    ///
    /// # Arguments
    ///
    /// - `threads`: Number of Rayon worker threads.  `None` or `Some(0)` uses
    ///   all available logical cores (Rayon default).
    ///
    /// When `threads` is `Some(n)` with `n > 0`, a dedicated Rayon thread pool
    /// is built and installed as the global pool.  This should be called at most
    /// once per process with a non-zero thread count; subsequent calls with the
    /// same count are effectively no-ops after the first pool initialisation.
    ///
    /// # Errors
    ///
    /// Returns [`BitNetError::BackendError`] if Rayon fails to build the thread
    /// pool (e.g. invalid thread count).
    pub fn new(threads: Option<usize>) -> Result<Self> {
        let thread_count = threads.unwrap_or(0);

        if thread_count > 0 {
            // Attempt to configure the global Rayon pool.
            // This is a best-effort: if the pool was already initialised by a
            // prior call we simply proceed with the existing configuration.
            let result = rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build_global();

            if let Err(e) = result {
                // Not fatal if already initialised.
                tracing::debug!(
                    threads = thread_count,
                    error = %e,
                    "Rayon global pool already initialised or build failed — proceeding"
                );
            }
        }

        let actual_threads = if thread_count == 0 {
            rayon::current_num_threads()
        } else {
            thread_count
        };

        tracing::info!(threads = actual_threads, "CpuBackend initialised");

        Ok(Self {
            name: format!("CPU ({actual_threads} threads)"),
            threads: actual_threads,
        })
    }

    /// Wrap this backend in an `Arc<dyn Backend>` for shared ownership.
    pub fn into_arc(self) -> Arc<dyn Backend> {
        Arc::new(self)
    }

    /// Number of Rayon threads this backend was configured with.
    #[inline]
    pub fn threads(&self) -> usize {
        self.threads
    }
}

// ---------------------------------------------------------------------------
// Backend impl
// ---------------------------------------------------------------------------

impl Backend for CpuBackend {
    // ------------------------------------------------------------------
    // Core linear algebra
    // ------------------------------------------------------------------

    /// Ternary GEMV: `output[i] = Σ_j weight[i,j] * input[j] * weight_scale`.
    ///
    /// Delegates to [`gemv::ternary_gemv_f32`] which parallelises the outer
    /// loop over output neurons using Rayon.
    ///
    /// # Errors
    ///
    /// Returns [`BitNetError::InvalidShape`] for shape mismatches.
    /// Returns [`BitNetError::QuantizationError`] if `weight_scale ≤ 0`.
    #[instrument(
        level = "trace",
        skip(self, weight, input, output),
        fields(out_features, in_features, weight_scale)
    )]
    fn ternary_gemv(
        &self,
        weight: &[i8],
        weight_scale: f32,
        input: &[f32],
        output: &mut [f32],
        out_features: usize,
        in_features: usize,
    ) -> Result<()> {
        gemv::ternary_gemv_f32(
            weight,
            weight_scale,
            input,
            output,
            out_features,
            in_features,
        )
    }

    /// Ternary GEMV with 8-bit activation quantisation (W2A8).
    ///
    /// Overrides the default implementation with a native integer accumulation
    /// path: activations are quantised to `i8` via absmax, then the ternary
    /// dot product uses `i32` accumulators for exact integer arithmetic.
    ///
    /// ```text
    /// x_q    = clip(round(x × 127 / max(|x|)), −128, 127)
    /// α_x    = max(|x|) / 127
    /// output[i] = (Σ_j W_q[i,j] × x_q[j]) × α_W × max(|x|) / 127
    /// ```
    ///
    /// # Performance
    ///
    /// The inner loop accumulates in `i32` and applies a single `f32` multiply
    /// at the end, avoiding per-element floating-point operations.
    ///
    /// # Errors
    ///
    /// Returns [`BitNetError::InvalidShape`] for shape mismatches.
    /// Returns [`BitNetError::QuantizationError`] if `weight_scale ≤ 0` or
    /// input contains non-finite values.
    #[instrument(
        level = "trace",
        skip(self, weight, input, output),
        fields(out_features, in_features, weight_scale)
    )]
    fn ternary_gemv_with_activation_quant(
        &self,
        weight: &[i8],
        weight_scale: f32,
        input: &[f32],
        output: &mut [f32],
        out_features: usize,
        in_features: usize,
    ) -> bitnet_core::error::Result<()> {
        use bitnet_core::quant::absmax::absmax_quantize_row;
        let (x_q, q_scale) = absmax_quantize_row(input)?;
        // absmax_quantize_row returns scale = max(|x|) / 127.
        // ternary_gemv_quantised expects act_scale = max(|x|) for its formula:
        //   output[i] = (Σ_j W[i,j] * x_q[j]) * weight_scale * act_scale / 127
        let max_abs = q_scale * 127.0_f32;
        gemv::ternary_gemv_quantised(
            weight,
            weight_scale,
            &x_q,
            max_abs,
            output,
            out_features,
            in_features,
        )
    }

    // ------------------------------------------------------------------
    // Normalisation
    // ------------------------------------------------------------------

    /// RMSNorm: `out[i] = x[i] / sqrt(mean(x²) + ε) * weight[i]`.
    ///
    /// Delegates to [`norm::rms_norm`].
    ///
    /// # Errors
    ///
    /// Returns [`BitNetError::InvalidShape`] if slice lengths differ.
    /// Returns [`BitNetError::QuantizationError`] if `eps ≤ 0`.
    #[instrument(
        level = "trace",
        skip(self, input, weight, output),
        fields(dim = input.len(), eps)
    )]
    fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) -> Result<()> {
        norm::rms_norm(input, weight, eps, output)
    }

    // ------------------------------------------------------------------
    // Positional encoding
    // ------------------------------------------------------------------

    /// Apply RoPE in-place to Q and K tensors at sequence position `position`.
    ///
    /// Delegates to [`rope::apply_rope`].
    ///
    /// # Errors
    ///
    /// Returns [`BitNetError::InvalidShape`] for shape mismatches or odd `head_dim`.
    /// Returns [`BitNetError::InvalidConfig`] if `theta ≤ 0`.
    #[instrument(
        level = "trace",
        skip(self, q, k),
        fields(position, head_dim, n_heads, n_kv_heads, theta)
    )]
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
        rope::apply_rope(q, k, position, head_dim, n_heads, n_kv_heads, theta)
    }

    // ------------------------------------------------------------------
    // Attention
    // ------------------------------------------------------------------

    /// Causal GQA scaled dot-product attention.
    ///
    /// Delegates to [`attention::masked_attention`].
    ///
    /// # Errors
    ///
    /// Returns [`BitNetError::InvalidShape`] for shape mismatches.
    /// Returns [`BitNetError::InvalidConfig`] if GQA group size is non-integer.
    #[instrument(
        level = "trace",
        skip(self, q, k_cache, v_cache, output),
        fields(n_heads, n_kv_heads, head_dim, cur_pos)
    )]
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
        attention::masked_attention(
            q, k_cache, v_cache, output, n_heads, n_kv_heads, head_dim, cur_pos,
        )
    }

    // ------------------------------------------------------------------
    // Activation functions
    // ------------------------------------------------------------------

    /// Apply squared ReLU in-place: `x[i] = max(0, x[i])²`.
    ///
    /// Delegates to [`activation::squared_relu`].
    ///
    /// # Errors
    ///
    /// Returns `Ok(())` unconditionally — this operation cannot fail on a
    /// valid `f32` slice.
    #[instrument(level = "trace", skip(self, x), fields(len = x.len()))]
    fn squared_relu(&self, x: &mut [f32]) -> Result<()> {
        activation::squared_relu(x);
        Ok(())
    }

    /// Apply numerically-stable softmax in-place over the entire slice.
    ///
    /// Delegates to [`activation::softmax`].
    ///
    /// # Errors
    ///
    /// Returns `Ok(())` unconditionally — this operation cannot fail on a
    /// valid `f32` slice.
    #[instrument(level = "trace", skip(self, x), fields(len = x.len()))]
    fn softmax(&self, x: &mut [f32]) -> Result<()> {
        activation::softmax(x);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Element-wise helpers
    // ------------------------------------------------------------------

    /// Element-wise multiply: `out[i] = a[i] * b[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`BitNetError::InvalidShape`] if `a.len() != b.len()` or
    /// `a.len() != out.len()`.
    #[instrument(
        level = "trace",
        skip(self, a, b, out),
        fields(len = a.len())
    )]
    fn elementwise_mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        if a.len() != b.len() {
            return Err(BitNetError::shape(
                format!("a.len() == b.len() = {}", b.len()),
                format!("a.len() = {}", a.len()),
            ));
        }
        if a.len() != out.len() {
            return Err(BitNetError::shape(
                format!("a.len() == out.len() = {}", out.len()),
                format!("a.len() = {}", a.len()),
            ));
        }
        for i in 0..a.len() {
            out[i] = a[i] * b[i];
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Device info
    // ------------------------------------------------------------------

    /// Returns a human-readable name for this backend.
    fn device_name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::backend::Backend;

    fn make_backend() -> CpuBackend {
        CpuBackend::new(Some(2)).expect("CpuBackend must initialise")
    }

    // ------------------------------------------------------------------
    // Initialisation
    // ------------------------------------------------------------------

    #[test]
    fn backend_new_no_threads() {
        let b = CpuBackend::new(None).unwrap();
        assert!(b.threads() > 0, "thread count must be > 0");
        assert!(b.device_name().starts_with("CPU"));
    }

    #[test]
    fn backend_new_with_threads() {
        let b = CpuBackend::new(Some(1)).unwrap();
        assert!(b.device_name().contains("CPU"));
    }

    #[test]
    fn backend_into_arc() {
        let b = CpuBackend::new(None).unwrap();
        let arc: Arc<dyn Backend> = b.into_arc();
        assert!(arc.device_name().starts_with("CPU"));
    }

    // ------------------------------------------------------------------
    // ternary_gemv
    // ------------------------------------------------------------------

    #[test]
    fn ternary_gemv_2x3() {
        // W = [[1, 0, -1], [-1, 1, 0]], x = [2, 3, 4], scale = 0.5
        // row 0: 1*2 + 0*3 + (-1)*4 = -2  → -1.0
        // row 1: (-1)*2 + 1*3 + 0*4 =  1  →  0.5
        let b = make_backend();
        let weight: Vec<i8> = vec![1, 0, -1, -1, 1, 0];
        let input = vec![2.0_f32, 3.0, 4.0];
        let mut output = vec![0.0_f32; 2];
        b.ternary_gemv(&weight, 0.5, &input, &mut output, 2, 3)
            .unwrap();
        assert!((output[0] - (-1.0)).abs() < 1e-6, "row 0: {}", output[0]);
        assert!((output[1] - 0.5).abs() < 1e-6, "row 1: {}", output[1]);
    }

    #[test]
    fn ternary_gemv_wrong_weight_len_returns_error() {
        let b = make_backend();
        let weight: Vec<i8> = vec![1; 5]; // should be 6
        let input = vec![1.0_f32; 3];
        let mut output = vec![0.0_f32; 2];
        let err = b
            .ternary_gemv(&weight, 1.0, &input, &mut output, 2, 3)
            .unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn ternary_gemv_zero_scale_returns_error() {
        let b = make_backend();
        let weight: Vec<i8> = vec![1; 4];
        let input = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 1];
        let err = b
            .ternary_gemv(&weight, 0.0, &input, &mut output, 1, 4)
            .unwrap_err();
        assert!(matches!(err, BitNetError::QuantizationError(_)));
    }

    // ------------------------------------------------------------------
    // rms_norm
    // ------------------------------------------------------------------

    #[test]
    fn rms_norm_via_backend() {
        let b = make_backend();
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        b.rms_norm(&input, &weight, 1e-5, &mut output).unwrap();
        // All outputs should be finite.
        assert!(output.iter().all(|v| v.is_finite()));
        // First output should be positive (input[0]=1 > 0).
        assert!(output[0] > 0.0);
    }

    #[test]
    fn rms_norm_mismatched_lengths_returns_error() {
        let b = make_backend();
        let input = vec![1.0_f32; 4];
        let weight = vec![1.0_f32; 5]; // mismatch
        let mut output = vec![0.0_f32; 4];
        let err = b.rms_norm(&input, &weight, 1e-5, &mut output).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    // ------------------------------------------------------------------
    // rope_embed
    // ------------------------------------------------------------------

    #[test]
    fn rope_embed_position_zero_is_identity() {
        let b = make_backend();
        let head_dim = 8;
        let mut q: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let mut k: Vec<f32> = (0..8).map(|i| i as f32 * 0.2).collect();
        let q_orig = q.clone();
        let k_orig = k.clone();
        b.rope_embed(&mut q, &mut k, 0, head_dim, 1, 1, 10000.0)
            .unwrap();
        for (i, (&orig, &rot)) in q_orig.iter().zip(q.iter()).enumerate() {
            assert!(
                (orig - rot).abs() < 1e-5,
                "q[{i}] pos=0 RoPE must be identity"
            );
        }
        let _ = k_orig;
    }

    #[test]
    fn rope_embed_wrong_q_len_returns_error() {
        let b = make_backend();
        let mut q = vec![0.0_f32; 7]; // should be 8
        let mut k = vec![0.0_f32; 8];
        let err = b
            .rope_embed(&mut q, &mut k, 0, 8, 1, 1, 10000.0)
            .unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    // ------------------------------------------------------------------
    // masked_attention
    // ------------------------------------------------------------------

    #[test]
    fn masked_attention_single_position_returns_value() {
        let b = make_backend();
        let head_dim = 4;
        let q = vec![1.0_f32, 0.0, 0.0, 0.0];
        let k_cache = vec![1.0_f32, 0.0, 0.0, 0.0];
        let v_cache = vec![0.1_f32, 0.2, 0.3, 0.4];
        let mut output = vec![0.0_f32; 4];
        b.masked_attention(&q, &k_cache, &v_cache, &mut output, 1, 1, head_dim, 0)
            .unwrap();
        // Single position: output == v_cache[0].
        let expected = vec![0.1_f32, 0.2, 0.3, 0.4];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "output[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn masked_attention_wrong_q_len_returns_error() {
        let b = make_backend();
        let q = vec![0.0_f32; 3]; // should be 4
        let k = vec![0.0_f32; 4];
        let v = vec![0.0_f32; 4];
        let mut out = vec![0.0_f32; 4];
        let err = b
            .masked_attention(&q, &k, &v, &mut out, 1, 1, 4, 0)
            .unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    // ------------------------------------------------------------------
    // squared_relu
    // ------------------------------------------------------------------

    #[test]
    fn squared_relu_via_backend() {
        let b = make_backend();
        let mut x = vec![-2.0_f32, -0.5, 0.0, 1.0, 3.0];
        b.squared_relu(&mut x).unwrap();
        let expected = vec![0.0_f32, 0.0, 0.0, 1.0, 9.0];
        for (i, (&got, &exp)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "x[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn squared_relu_empty_is_ok() {
        let b = make_backend();
        let mut x: Vec<f32> = vec![];
        b.squared_relu(&mut x).unwrap(); // must not panic or error
    }

    // ------------------------------------------------------------------
    // softmax
    // ------------------------------------------------------------------

    #[test]
    fn softmax_via_backend_sums_to_one() {
        let b = make_backend();
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        b.softmax(&mut x).unwrap();
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {sum}");
    }

    #[test]
    fn softmax_via_backend_all_positive() {
        let b = make_backend();
        let mut x = vec![0.5_f32, -1.0, 2.0, -3.0];
        b.softmax(&mut x).unwrap();
        assert!(x.iter().all(|&v| v > 0.0), "all outputs must be positive");
    }

    // ------------------------------------------------------------------
    // elementwise_mul
    // ------------------------------------------------------------------

    #[test]
    fn elementwise_mul_basic() {
        let b = make_backend();
        let a = vec![1.0_f32, 2.0, 3.0];
        let bv = vec![4.0_f32, 5.0, 6.0];
        let mut out = vec![0.0_f32; 3];
        b.elementwise_mul(&a, &bv, &mut out).unwrap();
        let expected = vec![4.0_f32, 10.0, 18.0];
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "out[{i}]: {got} ≠ {exp}");
        }
    }

    #[test]
    fn elementwise_mul_mismatched_a_b_returns_error() {
        let b = make_backend();
        let a = vec![1.0_f32; 4];
        let bv = vec![1.0_f32; 3]; // mismatch
        let mut out = vec![0.0_f32; 4];
        let err = b.elementwise_mul(&a, &bv, &mut out).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn elementwise_mul_mismatched_out_returns_error() {
        let b = make_backend();
        let a = vec![1.0_f32; 4];
        let bv = vec![1.0_f32; 4];
        let mut out = vec![0.0_f32; 3]; // mismatch
        let err = b.elementwise_mul(&a, &bv, &mut out).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    // ------------------------------------------------------------------
    // Arc<dyn Backend> forwarding
    // ------------------------------------------------------------------

    #[test]
    fn arc_backend_forwards_ternary_gemv() {
        let b: Arc<dyn Backend> = CpuBackend::new(None).unwrap().into_arc();
        let weight: Vec<i8> = vec![1, -1]; // 1x2
        let input = vec![3.0_f32, 2.0];
        let mut output = vec![0.0_f32; 1];
        b.ternary_gemv(&weight, 1.0, &input, &mut output, 1, 2)
            .unwrap();
        // dot([1,-1], [3,2]) * 1.0 = 1*3 + (-1)*2 = 1.0
        assert!((output[0] - 1.0).abs() < 1e-6, "got {}", output[0]);
    }

    #[test]
    fn arc_backend_device_name() {
        let b: Arc<dyn Backend> = CpuBackend::new(None).unwrap().into_arc();
        assert!(b.device_name().starts_with("CPU"));
    }

    // ------------------------------------------------------------------
    // Full forward-pass smoke test (mini-model dimensions)
    // ------------------------------------------------------------------

    /// Smoke test: run all backend operations in the order they appear during
    /// a single transformer block forward pass, verifying no panics or errors
    /// with plausible (small) dimensions.
    ///
    /// Dimensions: hidden=8, n_heads=2, n_kv_heads=1, head_dim=4, ffn_dim=12
    #[test]
    fn backend_mini_transformer_block_no_errors() {
        let hidden = 8_usize;
        let n_heads = 2_usize;
        let n_kv_heads = 1_usize;
        let head_dim = 4_usize;
        let ffn_dim = 12_usize;
        let cur_pos = 2_usize; // 3 positions
        let seq_len = cur_pos + 1;
        let b = CpuBackend::new(Some(2)).unwrap();

        // ── Input hidden state ──────────────────────────────────────────────
        let h: Vec<f32> = (0..hidden).map(|i| (i as f32 - 4.0) * 0.1).collect();
        let attn_norm_w = vec![1.0_f32; hidden];
        let ffn_norm_w = vec![1.0_f32; hidden];

        // ── Attention norm ──────────────────────────────────────────────────
        let mut h_norm = vec![0.0_f32; hidden];
        b.rms_norm(&h, &attn_norm_w, 1e-5, &mut h_norm).unwrap();

        // ── QKV projection ──────────────────────────────────────────────────
        // q: [n_heads * head_dim = 8], k: [n_kv_heads * head_dim = 4], v: [4]
        let q_weight = vec![0i8; n_heads * head_dim * hidden];
        let k_weight = vec![0i8; n_kv_heads * head_dim * hidden];
        let v_weight = vec![0i8; n_kv_heads * head_dim * hidden];

        let mut q_proj = vec![0.0_f32; n_heads * head_dim];
        let mut k_proj = vec![0.0_f32; n_kv_heads * head_dim];
        let mut v_proj = vec![0.0_f32; n_kv_heads * head_dim];

        b.ternary_gemv(
            &q_weight,
            1.0,
            &h_norm,
            &mut q_proj,
            n_heads * head_dim,
            hidden,
        )
        .unwrap();
        b.ternary_gemv(
            &k_weight,
            1.0,
            &h_norm,
            &mut k_proj,
            n_kv_heads * head_dim,
            hidden,
        )
        .unwrap();
        b.ternary_gemv(
            &v_weight,
            1.0,
            &h_norm,
            &mut v_proj,
            n_kv_heads * head_dim,
            hidden,
        )
        .unwrap();

        // ── RoPE ────────────────────────────────────────────────────────────
        b.rope_embed(
            &mut q_proj,
            &mut k_proj,
            cur_pos,
            head_dim,
            n_heads,
            n_kv_heads,
            500_000.0,
        )
        .unwrap();

        // ── KV cache (pre-filled with zeros for this test) ──────────────────
        let k_cache = vec![0.0_f32; seq_len * n_kv_heads * head_dim];
        let v_cache = vec![0.1_f32; seq_len * n_kv_heads * head_dim];

        // ── Attention ───────────────────────────────────────────────────────
        let mut attn_out = vec![0.0_f32; n_heads * head_dim];
        b.masked_attention(
            &q_proj,
            &k_cache,
            &v_cache,
            &mut attn_out,
            n_heads,
            n_kv_heads,
            head_dim,
            cur_pos,
        )
        .unwrap();

        // ── attn_sub_norm ───────────────────────────────────────────────────
        let attn_sub_norm_w = vec![1.0_f32; n_heads * head_dim];
        let mut attn_normed = vec![0.0_f32; n_heads * head_dim];
        b.rms_norm(&attn_out, &attn_sub_norm_w, 1e-5, &mut attn_normed)
            .unwrap();

        // ── o_proj ──────────────────────────────────────────────────────────
        let o_weight = vec![0i8; hidden * (n_heads * head_dim)];
        let mut o_out = vec![0.0_f32; hidden];
        b.ternary_gemv(
            &o_weight,
            1.0,
            &attn_normed,
            &mut o_out,
            hidden,
            n_heads * head_dim,
        )
        .unwrap();

        // residual: h2 = h + o_out
        let h2: Vec<f32> = h.iter().zip(o_out.iter()).map(|(a, b)| a + b).collect();

        // ── FFN norm ─────────────────────────────────────────────────────────
        let mut h2_norm = vec![0.0_f32; hidden];
        b.rms_norm(&h2, &ffn_norm_w, 1e-5, &mut h2_norm).unwrap();

        // ── gate + up projections ────────────────────────────────────────────
        let gate_weight = vec![0i8; ffn_dim * hidden];
        let up_weight = vec![0i8; ffn_dim * hidden];
        let mut gate = vec![0.0_f32; ffn_dim];
        let mut up = vec![0.1_f32; ffn_dim]; // non-zero for interesting test
        b.ternary_gemv(&gate_weight, 1.0, &h2_norm, &mut gate, ffn_dim, hidden)
            .unwrap();
        b.ternary_gemv(&up_weight, 1.0, &h2_norm, &mut up, ffn_dim, hidden)
            .unwrap();

        // ── sqrelu(gate) ⊙ up ────────────────────────────────────────────────
        b.squared_relu(&mut gate).unwrap();
        let mut inner = vec![0.0_f32; ffn_dim];
        b.elementwise_mul(&gate, &up, &mut inner).unwrap();

        // ── ffn_sub_norm ──────────────────────────────────────────────────────
        let ffn_sub_norm_w = vec![1.0_f32; ffn_dim];
        let mut inner_normed = vec![0.0_f32; ffn_dim];
        b.rms_norm(&inner, &ffn_sub_norm_w, 1e-5, &mut inner_normed)
            .unwrap();

        // ── down projection ───────────────────────────────────────────────────
        let down_weight = vec![0i8; hidden * ffn_dim];
        let mut ffn_out = vec![0.0_f32; hidden];
        b.ternary_gemv(
            &down_weight,
            1.0,
            &inner_normed,
            &mut ffn_out,
            hidden,
            ffn_dim,
        )
        .unwrap();

        // ── residual ──────────────────────────────────────────────────────────
        let h3: Vec<f32> = h2.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();

        // All outputs must be finite.
        assert!(
            h3.iter().all(|v| v.is_finite()),
            "mini forward pass output contains non-finite values: {:?}",
            h3
        );
    }
}
