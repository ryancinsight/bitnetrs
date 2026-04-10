//! # bitnet-npu
//!
//! NPU backend for BitNet b1.58 inference.
//!
//! ## Strategy
//!
//! On Windows, modern AI-capable hardware exposes a Neural Processing Unit (NPU)
//! that can be accessed via **DirectML** — Microsoft's GPU/NPU compute API.
//! wgpu on Windows supports DirectML-backed adapters, so this crate probes the
//! adapter list for NPU-like devices and, when found, uses a wgpu compute path
//! optimised for low-power AI inference.
//!
//! When no NPU is detected (or on non-Windows platforms), the backend falls back
//! transparently to [`bitnet_cpu::CpuBackend`] with no change to the API contract.
//!
//! ## Detection Heuristics
//!
//! An adapter is classified as an NPU if its reported name contains any of the
//! following vendor-specific strings (case-insensitive):
//!
//! | Vendor  | Typical NPU name substring  |
//! |---------|-----------------------------|
//! | Intel   | `npu`, `neural`             |
//! | AMD     | `npu`, `neural`             |
//! | Qualcomm| `adreno`, `npu`, `neural`   |
//! | Apple   | `neural engine`             |
//!
//! The heuristic is intentionally conservative: if no adapter clearly matches,
//! the CPU fallback is used and a `tracing::info!` message is emitted.
//!
//! ## Backend Trait Implementation
//!
//! All [`Backend`] trait methods delegate to whichever inner backend was selected
//! at construction time:
//! - [`NpuBackend::new`]: Attempt NPU detection; return an `NpuBackend` wrapping
//!   either the NPU (wgpu) backend or the CPU fallback.
//! - All compute operations: pass-through to `inner: Arc<dyn Backend>`.
//!
//! ## Invariants
//!
//! - `NpuBackend` is `Send + Sync`.
//! - The `device_name()` method always reports which backend is actually active.
//! - NPU detection never panics; errors result in silent fallback to CPU.
//! - All `Backend` methods maintain the same pre/post-conditions as the wrapped
//!   backend (shapes, scales, invariants are unchanged by the dispatch layer).
//!
//! ## Module Layout
//!
//! ```text
//! bitnet-npu/
//! ├── lib.rs     ← this file: NpuBackend + Backend impl + NPU detection
//! └── detect.rs  ← NpuInfo struct + detect_npu() function
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod detect;

use std::sync::Arc;

use bitnet_core::backend::Backend;
use bitnet_core::error::{BitNetError, Result};
use tracing::{debug, info, instrument, warn};

use detect::{detect_npu, NpuInfo};

// ---------------------------------------------------------------------------
// NpuBackend
// ---------------------------------------------------------------------------

/// NPU compute backend.
///
/// Wraps either a GPU (wgpu/DirectML NPU) backend or a [`bitnet_cpu::CpuBackend`]
/// fallback, selected transparently at construction time based on hardware
/// detection.
///
/// # Usage
///
/// ```no_run
/// use bitnet_npu::NpuBackend;
/// use bitnet_core::backend::Backend;
///
/// let backend = NpuBackend::new(0).expect("NPU backend init failed");
/// println!("Active device: {}", backend.device_name());
/// ```
///
/// # Invariants
///
/// - `NpuBackend` is `Send + Sync`.
/// - The wrapped `inner` backend satisfies all [`Backend`] invariants.
/// - `device_name()` reflects the actual hardware being used.
pub struct NpuBackend {
    /// The inner backend performing the actual compute.
    ///
    /// Either a `bitnet_gpu::GpuBackend` (when an NPU adapter was detected)
    /// or a `bitnet_cpu::CpuBackend` (fallback).
    inner: Arc<dyn Backend>,

    /// Human-readable device name for diagnostics.
    name: String,

    /// Whether the backend is actually using an NPU (vs CPU fallback).
    using_npu: bool,

    /// NPU information, if detected.
    npu_info: Option<NpuInfo>,
}

// SAFETY: The inner Arc<dyn Backend> is Send + Sync.
unsafe impl Send for NpuBackend {}
unsafe impl Sync for NpuBackend {}

impl std::fmt::Debug for NpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NpuBackend")
            .field("name", &self.name)
            .field("using_npu", &self.using_npu)
            .finish()
    }
}

impl NpuBackend {
    /// Create a new `NpuBackend`.
    ///
    /// Performs NPU detection via [`detect_npu`].  If an NPU adapter is found
    /// at index `device_id`, attempts to initialise a wgpu compute backend
    /// targeting that adapter.  If NPU detection fails or initialisation
    /// errors, silently falls back to [`bitnet_cpu::CpuBackend`].
    ///
    /// # Arguments
    ///
    /// - `device_id`: Zero-based index of the NPU adapter to prefer.
    ///   `0` selects the first detected NPU.  Ignored when falling back to CPU.
    ///
    /// # Errors
    ///
    /// Returns [`BitNetError::BackendError`] only if the CPU fallback itself
    /// fails to initialise (which should never happen on a valid system).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use bitnet_core::backend::Backend;
    /// let backend = bitnet_npu::NpuBackend::new(0).unwrap();
    /// if backend.is_using_npu() {
    ///     println!("Running on NPU: {}", backend.device_name());
    /// } else {
    ///     println!("NPU not found — using CPU fallback");
    /// }
    /// ```
    pub fn new(device_id: u32) -> Result<Self> {
        // ── Step 1: Probe for NPU adapters ────────────────────────────────
        let detected = detect_npu();

        if let Some(ref info) = detected {
            info!(
                npu_name = %info.name,
                vendor = ?info.vendor,
                adapter_type = ?info.adapter_type,
                "NPU detected"
            );
        } else {
            debug!("No NPU detected on this system; will use CPU fallback");
        }

        // ── Step 2: Attempt NPU-backed wgpu initialisation ────────────────
        //
        // On Windows, we attempt to create a wgpu backend using the detected
        // NPU adapter.  On other platforms, or if detection found nothing,
        // we skip directly to CPU fallback.
        #[cfg(target_os = "windows")]
        if let Some(ref npu_info) = detected {
            debug!(
                npu_adapter_index = npu_info.adapter_index,
                "Attempting wgpu/DirectML NPU backend initialisation"
            );

            // The adapter_index from detection is the wgpu adapter index.
            // We use it directly with the GPU backend (which uses the same
            // wgpu adapter enumeration).
            let gpu_result = bitnet_gpu::GpuBackend::new_blocking(npu_info.adapter_index);

            match gpu_result {
                Ok(gpu_backend) => {
                    let name = format!("NPU via DirectML ({})", npu_info.name);
                    info!(device = %name, "NPU backend initialised via wgpu/DirectML");
                    let inner: Arc<dyn Backend> = gpu_backend.into_arc();
                    return Ok(Self {
                        inner,
                        name,
                        using_npu: true,
                        npu_info: detected,
                    });
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        npu = %npu_info.name,
                        "wgpu/DirectML NPU initialisation failed; falling back to CPU"
                    );
                }
            }
        }

        // On non-Windows platforms, also use GPU if an NPU-like adapter is found.
        #[cfg(not(target_os = "windows"))]
        if let Some(ref npu_info) = detected {
            debug!(
                npu_name = %npu_info.name,
                "Non-Windows: attempting GPU backend for NPU-like adapter"
            );
            if let Ok(gpu_backend) = bitnet_gpu::GpuBackend::new_blocking(npu_info.adapter_index) {
                let name = format!("NPU ({})", npu_info.name);
                info!(device = %name, "NPU-like backend initialised via wgpu");
                let inner: Arc<dyn Backend> = gpu_backend.into_arc();
                return Ok(Self {
                    inner,
                    name,
                    using_npu: true,
                    npu_info: detected,
                });
            }
        }

        // ── Step 3: CPU fallback ──────────────────────────────────────────
        info!("Using CPU fallback backend (no NPU available)");
        let cpu = bitnet_cpu::CpuBackend::new(None).map_err(|e| {
            BitNetError::backend("NpuBackend", format!("CPU fallback init failed: {e}"))
        })?;

        let name = format!("NPU→CPU fallback ({})", cpu.device_name());

        Ok(Self {
            inner: cpu.into_arc(),
            name,
            using_npu: false,
            npu_info: None,
        })
    }

    /// Returns `true` if an NPU was successfully detected and is being used.
    ///
    /// When this returns `false`, all compute is performed on the CPU.
    #[inline]
    pub fn is_using_npu(&self) -> bool {
        self.using_npu
    }

    /// Returns the detected [`NpuInfo`], if an NPU was found.
    ///
    /// Returns `None` when using the CPU fallback.
    #[inline]
    pub fn npu_info(&self) -> Option<&NpuInfo> {
        self.npu_info.as_ref()
    }

    /// Wrap this backend in an `Arc<dyn Backend>` for shared ownership.
    #[inline]
    pub fn into_arc(self) -> Arc<dyn Backend> {
        Arc::new(self)
    }
}

// ---------------------------------------------------------------------------
// Backend impl
// ---------------------------------------------------------------------------

impl Backend for NpuBackend {
    // ------------------------------------------------------------------
    // Core linear algebra
    // ------------------------------------------------------------------

    /// Ternary GEMV: delegates to the detected inner backend (NPU or CPU).
    ///
    /// # Errors
    ///
    /// Propagates errors from the inner backend (shape mismatches, scale errors).
    #[instrument(
        level = "trace",
        skip(self, weight_packed, input, output),
        fields(out_features, in_features, weight_scale, backend = %self.name)
    )]
    fn ternary_gemv(
        &self,
        weight_packed: &[u8],
        weight_scale: f32,
        input: &[f32],
        output: &mut [f32],
        out_features: usize,
        in_features: usize,
    ) -> Result<()> {
        self.inner.ternary_gemv(
            weight_packed,
            weight_scale,
            input,
            output,
            out_features,
            in_features,
        )
    }

    /// Ternary GEMV with 8-bit activation quantisation: delegates to the inner backend.
    ///
    /// This ensures the inner backend's optimised path (e.g. CpuBackend's i32
    /// accumulation) is used rather than the trait's default quantise-dequantise path.
    ///
    /// # Errors
    ///
    /// Propagates errors from the inner backend.
    #[instrument(
        level = "trace",
        skip(self, weight_packed, input, output),
        fields(out_features, in_features, weight_scale, backend = %self.name)
    )]
    fn ternary_gemv_with_activation_quant(
        &self,
        weight_packed: &[u8],
        weight_scale: f32,
        input: &[f32],
        output: &mut [f32],
        out_features: usize,
        in_features: usize,
    ) -> Result<()> {
        self.inner.ternary_gemv_with_activation_quant(
            weight_packed,
            weight_scale,
            input,
            output,
            out_features,
            in_features,
        )
    }

    // ------------------------------------------------------------------
    // Normalisation
    // ------------------------------------------------------------------

    /// RMSNorm: delegates to the inner backend.
    ///
    /// # Errors
    ///
    /// Propagates shape and epsilon errors from the inner backend.
    #[instrument(
        level = "trace",
        skip(self, input, weight, output),
        fields(dim = input.len(), eps, backend = %self.name)
    )]
    fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) -> Result<()> {
        self.inner.rms_norm(input, weight, eps, output)
    }

    // ------------------------------------------------------------------
    // Positional encoding
    // ------------------------------------------------------------------

    /// Rotary Position Embedding: delegates to the inner backend.
    ///
    /// # Errors
    ///
    /// Propagates shape and configuration errors from the inner backend.
    #[instrument(
        level = "trace",
        skip(self, q, k),
        fields(position, head_dim, n_heads, n_kv_heads, theta, backend = %self.name)
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
        self.inner
            .rope_embed(q, k, position, head_dim, n_heads, n_kv_heads, theta)
    }

    // ------------------------------------------------------------------
    // Attention
    // ------------------------------------------------------------------

    /// Causal GQA attention: delegates to the inner backend.
    ///
    /// # Errors
    ///
    /// Propagates shape and configuration errors from the inner backend.
    #[instrument(
        level = "trace",
        skip(self, q, k_cache, v_cache, output),
        fields(n_heads, n_kv_heads, head_dim, cur_pos, backend = %self.name)
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
        self.inner.masked_attention(
            q, k_cache, v_cache, output, n_heads, n_kv_heads, head_dim, cur_pos,
        )
    }

    // ------------------------------------------------------------------
    // Activation functions
    // ------------------------------------------------------------------

    /// Squared ReLU in-place: `x[i] = max(0, x[i])²`.
    ///
    /// Delegates to the inner backend.
    #[instrument(level = "trace", skip(self, x), fields(len = x.len(), backend = %self.name))]
    fn squared_relu(&self, x: &mut [f32]) -> Result<()> {
        self.inner.squared_relu(x)
    }

    /// Numerically-stable softmax in-place.
    ///
    /// Delegates to the inner backend.
    #[instrument(level = "trace", skip(self, x), fields(len = x.len(), backend = %self.name))]
    fn softmax(&self, x: &mut [f32]) -> Result<()> {
        self.inner.softmax(x)
    }

    // ------------------------------------------------------------------
    // Element-wise helpers
    // ------------------------------------------------------------------

    /// Element-wise multiply: `out[i] = a[i] * b[i]`.
    ///
    /// Delegates to the inner backend.
    #[instrument(
        level = "trace",
        skip(self, a, b, out),
        fields(len = a.len(), backend = %self.name)
    )]
    fn elementwise_mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        self.inner.elementwise_mul(a, b, out)
    }

    fn lm_head_matmul_into(
        &self,
        hidden: &[f32],
        weights: &[f32],
        output: &mut [f32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<()> {
        self.inner
            .lm_head_matmul_into(hidden, weights, output, vocab_size, hidden_size)
    }

    // ------------------------------------------------------------------
    // Device info
    // ------------------------------------------------------------------

    /// Human-readable name of the active backend.
    ///
    /// Examples:
    /// - `"NPU via DirectML (Intel NPU Accelerator)"` — NPU active
    /// - `"NPU→CPU fallback (CPU (8 threads))"` — CPU fallback
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
    use bitnet_core::error::BitNetError;

    // ------------------------------------------------------------------
    // Initialisation
    // ------------------------------------------------------------------

    #[test]
    fn npu_backend_new_succeeds_always() {
        // NpuBackend::new must always succeed because it falls back to CPU
        // when no NPU is found. This test verifies the invariant.
        let backend = NpuBackend::new(0).expect("NpuBackend must always initialise");
        assert!(
            !backend.device_name().is_empty(),
            "device_name must not be empty"
        );
    }

    #[test]
    fn npu_backend_device_name_not_empty() {
        let backend = NpuBackend::new(0).unwrap();
        assert!(!backend.device_name().is_empty());
    }

    #[test]
    fn npu_backend_is_using_npu_is_bool() {
        // Just verify the method is callable and returns a bool.
        let backend = NpuBackend::new(0).unwrap();
        let _ = backend.is_using_npu(); // must not panic
    }

    #[test]
    fn npu_backend_npu_info_consistent_with_using_npu() {
        let backend = NpuBackend::new(0).unwrap();
        if backend.is_using_npu() {
            assert!(
                backend.npu_info().is_some(),
                "npu_info must be Some when is_using_npu() is true"
            );
        } else {
            assert!(
                backend.npu_info().is_none(),
                "npu_info must be None when is_using_npu() is false"
            );
        }
    }

    #[test]
    fn npu_backend_into_arc_works() {
        let backend = NpuBackend::new(0).unwrap();
        let arc: Arc<dyn Backend> = backend.into_arc();
        assert!(!arc.device_name().is_empty());
    }

    #[test]
    fn npu_backend_debug_format_non_empty() {
        let backend = NpuBackend::new(0).unwrap();
        let debug_str = format!("{backend:?}");
        assert!(!debug_str.is_empty());
        assert!(
            debug_str.contains("NpuBackend"),
            "Debug output must contain struct name"
        );
    }

    // ------------------------------------------------------------------
    // Backend trait delegation — correctness via CPU fallback
    //
    // These tests verify that delegation is correct by checking that the
    // NpuBackend produces identical results to the CPU backend for small
    // analytically-verifiable inputs.
    // ------------------------------------------------------------------

    fn make_npu_backend() -> NpuBackend {
        NpuBackend::new(0).expect("NpuBackend must init")
    }

    /// Pack a row-major `[rows, cols]` i8 ternary weight matrix into
    /// row-aligned packed 2-bit bytes.  Each row is packed independently
    /// so that `packed_cols = ceil(cols / 4)` bytes per row.
    fn pack_row_aligned(weights: &[i8], rows: usize, cols: usize) -> Vec<u8> {
        let packed_cols = (cols + 3) / 4;
        let mut packed = Vec::with_capacity(rows * packed_cols);
        for r in 0..rows {
            let row_start = r * cols;
            let row = &weights[row_start..row_start + cols];
            packed.extend_from_slice(&bitnet_core::quant::ternary::pack_ternary(row));
        }
        packed
    }

    // ---- ternary_gemv ---------------------------------------------------

    #[test]
    fn npu_ternary_gemv_2x3() {
        // W = [[1, 0, -1], [-1, 1, 0]], x = [2, 3, 4], scale = 0.5
        // row 0: 1*2 + 0*3 + (-1)*4 = -2  →  -2 * 0.5 = -1.0
        // row 1: (-1)*2 + 1*3 + 0*4 =  1  →   1 * 0.5 =  0.5
        let b = make_npu_backend();
        let weight_i8: Vec<i8> = vec![1, 0, -1, -1, 1, 0];
        let weight_packed = pack_row_aligned(&weight_i8, 2, 3);
        let input = vec![2.0_f32, 3.0, 4.0];
        let mut output = vec![0.0_f32; 2];
        b.ternary_gemv(&weight_packed, 0.5, &input, &mut output, 2, 3)
            .unwrap();
        assert!(
            (output[0] - (-1.0)).abs() < 1e-5,
            "row 0: expected -1.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 0.5).abs() < 1e-5,
            "row 1: expected 0.5, got {}",
            output[1]
        );
    }

    #[test]
    fn npu_ternary_gemv_all_zero_weight_produces_zero_output() {
        let b = make_npu_backend();
        let weight_packed = pack_row_aligned(&vec![0i8; 6], 2, 3);
        let input = vec![1.0_f32, 2.0, 3.0];
        let mut output = vec![99.0_f32; 2]; // non-zero sentinel
        b.ternary_gemv(&weight_packed, 1.0, &input, &mut output, 2, 3)
            .unwrap();
        assert!(
            output.iter().all(|&v| v == 0.0),
            "all-zero weights → all-zero output, got {:?}",
            output
        );
    }

    #[test]
    fn npu_ternary_gemv_wrong_weight_len_returns_error() {
        let b = make_npu_backend();
        // For 2×3, packed_cols = ceil(3/4) = 1, expected = 2 bytes; provide 3 (wrong).
        let weight_packed = vec![0u8; 3];
        let input = vec![1.0_f32; 3];
        let mut output = vec![0.0_f32; 2];
        let err = b
            .ternary_gemv(&weight_packed, 1.0, &input, &mut output, 2, 3)
            .unwrap_err();
        assert!(
            matches!(err, BitNetError::InvalidShape { .. }),
            "wrong weight len must return InvalidShape, got {err:?}"
        );
    }

    #[test]
    fn npu_ternary_gemv_zero_scale_returns_error() {
        let b = make_npu_backend();
        let weight_packed = pack_row_aligned(&vec![1i8; 4], 1, 4);
        let input = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 1];
        let err = b
            .ternary_gemv(&weight_packed, 0.0, &input, &mut output, 1, 4)
            .unwrap_err();
        assert!(
            matches!(err, BitNetError::QuantizationError(_)),
            "zero scale must return QuantizationError, got {err:?}"
        );
    }

    // ---- rms_norm -------------------------------------------------------

    #[test]
    fn npu_rms_norm_output_is_finite() {
        let b = make_npu_backend();
        let input = vec![1.0_f32, -2.0, 3.0, -4.0];
        let weight = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        b.rms_norm(&input, &weight, 1e-5, &mut output).unwrap();
        assert!(
            output.iter().all(|v| v.is_finite()),
            "RMSNorm output must be finite: {:?}",
            output
        );
    }

    #[test]
    fn npu_rms_norm_zero_input_produces_zero_output() {
        let b = make_npu_backend();
        let input = vec![0.0_f32; 8];
        let weight = vec![1.0_f32; 8];
        let mut output = vec![99.0_f32; 8];
        b.rms_norm(&input, &weight, 1e-5, &mut output).unwrap();
        assert!(
            output.iter().all(|&v| v == 0.0),
            "zero input → zero output: {:?}",
            output
        );
    }

    #[test]
    fn npu_rms_norm_mismatched_lengths_returns_error() {
        let b = make_npu_backend();
        let input = vec![1.0_f32; 4];
        let weight = vec![1.0_f32; 5]; // mismatch
        let mut output = vec![0.0_f32; 4];
        let err = b.rms_norm(&input, &weight, 1e-5, &mut output).unwrap_err();
        assert!(
            matches!(err, BitNetError::InvalidShape { .. }),
            "mismatched lengths must return InvalidShape, got {err:?}"
        );
    }

    #[test]
    fn npu_rms_norm_negative_eps_returns_error() {
        let b = make_npu_backend();
        let input = vec![1.0_f32; 4];
        let weight = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        let err = b.rms_norm(&input, &weight, -1e-5, &mut output).unwrap_err();
        assert!(
            matches!(err, BitNetError::QuantizationError(_)),
            "negative eps must return QuantizationError, got {err:?}"
        );
    }

    // ---- rope_embed -----------------------------------------------------

    #[test]
    fn npu_rope_position_zero_is_identity() {
        // At position 0, all rotation angles are 0 → rotation is identity.
        let b = make_npu_backend();
        let head_dim = 8_usize;
        let mut q: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.1).collect();
        let mut k: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.2).collect();
        let q_orig = q.clone();
        let k_orig = k.clone();

        b.rope_embed(&mut q, &mut k, 0, head_dim, 1, 1, 10000.0)
            .unwrap();

        for (i, (&orig, &rot)) in q_orig.iter().zip(q.iter()).enumerate() {
            assert!(
                (orig - rot).abs() < 1e-5,
                "q[{i}]: pos=0 RoPE identity violated: orig={orig}, rot={rot}"
            );
        }
        for (i, (&orig, &rot)) in k_orig.iter().zip(k.iter()).enumerate() {
            assert!(
                (orig - rot).abs() < 1e-5,
                "k[{i}]: pos=0 RoPE identity violated: orig={orig}, rot={rot}"
            );
        }
    }

    #[test]
    fn npu_rope_preserves_vector_norm() {
        // RoPE is a rotation (isometry) — it must preserve the L2 norm of each head.
        let b = make_npu_backend();
        let head_dim = 8_usize;
        let mut q: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.37) - 1.5).collect();
        let mut k: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.53) - 2.0).collect();

        let q_norm_before: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let k_norm_before: f32 = k.iter().map(|v| v * v).sum::<f32>().sqrt();

        b.rope_embed(&mut q, &mut k, 42, head_dim, 1, 1, 500_000.0)
            .unwrap();

        let q_norm_after: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let k_norm_after: f32 = k.iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!(
            (q_norm_before - q_norm_after).abs() < 1e-4,
            "Q norm changed from {q_norm_before} to {q_norm_after}"
        );
        assert!(
            (k_norm_before - k_norm_after).abs() < 1e-4,
            "K norm changed from {k_norm_before} to {k_norm_after}"
        );
    }

    #[test]
    fn npu_rope_wrong_q_length_returns_error() {
        let b = make_npu_backend();
        let mut q = vec![0.0_f32; 7]; // should be 8 for n_heads=1, head_dim=8
        let mut k = vec![0.0_f32; 8];
        let err = b
            .rope_embed(&mut q, &mut k, 0, 8, 1, 1, 10000.0)
            .unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    // ---- masked_attention -----------------------------------------------

    #[test]
    fn npu_attention_single_position_returns_value() {
        // With a single token (cur_pos=0), the only output is the value at t=0.
        // softmax([score_0]) = [1.0] → output = v_cache[0]
        let b = make_npu_backend();
        let head_dim = 4_usize;
        let q = vec![1.0_f32, 0.0, 0.0, 0.0];
        let k_cache = vec![1.0_f32, 0.0, 0.0, 0.0]; // pos=0, kv_h=0
        let v_cache = vec![0.1_f32, 0.2, 0.3, 0.4]; // pos=0, kv_h=0
        let mut output = vec![0.0_f32; 4];

        b.masked_attention(&q, &k_cache, &v_cache, &mut output, 1, 1, head_dim, 0)
            .unwrap();

        let expected = [0.1_f32, 0.2, 0.3, 0.4];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "output[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn npu_attention_output_is_finite() {
        let b = make_npu_backend();
        let head_dim = 8_usize;
        let n_heads = 4_usize;
        let n_kv_heads = 2_usize;
        let cur_pos = 3_usize;
        let seq_len = cur_pos + 1;

        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| (i as f32 - 16.0) * 0.1)
            .collect();
        let k_cache: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 - 32.0) * 0.05)
            .collect();
        let v_cache: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| i as f32 * 0.01)
            .collect();
        let mut output = vec![0.0_f32; n_heads * head_dim];

        b.masked_attention(
            &q,
            &k_cache,
            &v_cache,
            &mut output,
            n_heads,
            n_kv_heads,
            head_dim,
            cur_pos,
        )
        .unwrap();

        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn npu_attention_non_divisible_heads_returns_error() {
        let b = make_npu_backend();
        // n_heads=3, n_kv_heads=2 → 3 % 2 ≠ 0 → error
        let q = vec![0.0_f32; 12]; // 3 × 4
        let k = vec![0.0_f32; 8]; // 1 × 2 × 4
        let v = vec![0.0_f32; 8];
        let mut out = vec![0.0_f32; 12];
        let err = b
            .masked_attention(&q, &k, &v, &mut out, 3, 2, 4, 0)
            .unwrap_err();
        assert!(matches!(err, BitNetError::InvalidConfig(_)));
    }

    // ---- squared_relu ---------------------------------------------------

    #[test]
    fn npu_squared_relu_basic() {
        let b = make_npu_backend();
        let mut x = vec![-3.0_f32, -0.5, 0.0, 1.0, 2.0, 4.0];
        b.squared_relu(&mut x).unwrap();
        let expected = [0.0_f32, 0.0, 0.0, 1.0, 4.0, 16.0];
        for (i, (&got, &exp)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "x[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn npu_squared_relu_output_always_nonnegative() {
        let b = make_npu_backend();
        let mut x: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.5).collect();
        b.squared_relu(&mut x).unwrap();
        for (i, &v) in x.iter().enumerate() {
            assert!(v >= 0.0, "sqrelu output[{i}] = {v} < 0 (must be ≥ 0)");
        }
    }

    #[test]
    fn npu_squared_relu_empty_is_noop() {
        let b = make_npu_backend();
        let mut x: Vec<f32> = vec![];
        b.squared_relu(&mut x).unwrap(); // must not panic or error
    }

    // ---- softmax --------------------------------------------------------

    #[test]
    fn npu_softmax_sums_to_one() {
        let b = make_npu_backend();
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        b.softmax(&mut x).unwrap();
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1, got {sum}");
    }

    #[test]
    fn npu_softmax_all_positive() {
        let b = make_npu_backend();
        let mut x = vec![0.5_f32, -1.0, 2.0, -3.0];
        b.softmax(&mut x).unwrap();
        for (i, &v) in x.iter().enumerate() {
            assert!(v > 0.0, "softmax output[{i}] = {v} must be positive");
        }
    }

    #[test]
    fn npu_softmax_empty_is_noop() {
        let b = make_npu_backend();
        let mut x: Vec<f32> = vec![];
        b.softmax(&mut x).unwrap(); // must not panic or error
    }

    // ---- elementwise_mul ------------------------------------------------

    #[test]
    fn npu_elementwise_mul_basic() {
        let b = make_npu_backend();
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let bv = vec![0.5_f32, -1.0, 2.0, 0.0];
        let mut out = vec![0.0_f32; 4];
        b.elementwise_mul(&a, &bv, &mut out).unwrap();
        let expected = [0.5_f32, -2.0, 6.0, 0.0];
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-7,
                "out[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn npu_elementwise_mul_mismatched_lengths_returns_error() {
        let b = make_npu_backend();
        let a = vec![1.0_f32; 4];
        let bv = vec![1.0_f32; 3]; // mismatch
        let mut out = vec![0.0_f32; 4];
        let err = b.elementwise_mul(&a, &bv, &mut out).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    // ---- device_name ----------------------------------------------------

    #[test]
    fn npu_backend_device_name_via_backend_trait() {
        let b: Arc<dyn Backend> = NpuBackend::new(0).unwrap().into_arc();
        let name = b.device_name();
        assert!(!name.is_empty(), "device_name must not be empty");
        // The name must contain either "NPU" or "CPU" (indicating which is active).
        assert!(
            name.contains("NPU") || name.contains("CPU"),
            "device_name must mention NPU or CPU, got '{name}'"
        );
    }

    // ---- Cross-backend consistency: NpuBackend matches CpuBackend output ----

    /// Property: when no NPU is available (which is the case on most test machines),
    /// NpuBackend falls back to CPU and produces the same results as CpuBackend.
    #[test]
    fn npu_backend_matches_cpu_backend_when_no_npu() {
        let npu_b = NpuBackend::new(0).unwrap();

        // If an NPU is available, we cannot compare outputs (different arithmetic).
        if npu_b.is_using_npu() {
            return; // Skip — NPU produces different (valid) results
        }

        let cpu_b = bitnet_cpu::CpuBackend::new(None).unwrap();

        // Test: RMSNorm
        let input = vec![0.5_f32, -1.5, 2.0, -0.5, 1.0, 0.25, -0.75, 1.25];
        let weight = vec![1.0_f32, 0.5, 2.0, 1.5, 0.8, 1.2, 0.9, 1.1];
        let eps = 1e-5_f32;

        let mut npu_out = vec![0.0_f32; 8];
        let mut cpu_out = vec![0.0_f32; 8];

        npu_b.rms_norm(&input, &weight, eps, &mut npu_out).unwrap();
        cpu_b.rms_norm(&input, &weight, eps, &mut cpu_out).unwrap();

        for (i, (&n, &c)) in npu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!((n - c).abs() < 1e-6, "RMSNorm index {i}: npu={n}, cpu={c}");
        }
    }

    /// Full mini-forward-pass smoke test verifying no panics for all
    /// Backend operations in sequence.
    ///
    /// Uses toy dimensions (hidden=8, heads=2, kv_heads=1, head_dim=4, ffn_dim=12).
    #[test]
    fn npu_backend_mini_forward_pass_no_errors() {
        let hidden = 8_usize;
        let n_heads = 2_usize;
        let n_kv_heads = 1_usize;
        let head_dim = 4_usize;
        let ffn_dim = 12_usize;
        let cur_pos = 1_usize;
        let seq_len = cur_pos + 1;

        let b = NpuBackend::new(0).unwrap();

        // ── Hidden state ──────────────────────────────────────────────────
        let h: Vec<f32> = (0..hidden).map(|i| (i as f32 - 4.0) * 0.2).collect();
        let norm_w = vec![1.0_f32; hidden];

        // ── Pre-attention norm ────────────────────────────────────────────
        let mut h_norm = vec![0.0_f32; hidden];
        b.rms_norm(&h, &norm_w, 1e-5, &mut h_norm).unwrap();

        // ── QKV projections (all-zero weights for simplicity) ─────────────
        let q_w = pack_row_aligned(
            &vec![0i8; n_heads * head_dim * hidden],
            n_heads * head_dim,
            hidden,
        );
        let k_w = pack_row_aligned(
            &vec![0i8; n_kv_heads * head_dim * hidden],
            n_kv_heads * head_dim,
            hidden,
        );
        let v_w = pack_row_aligned(
            &vec![0i8; n_kv_heads * head_dim * hidden],
            n_kv_heads * head_dim,
            hidden,
        );

        let mut q = vec![0.0_f32; n_heads * head_dim];
        let mut k = vec![0.0_f32; n_kv_heads * head_dim];
        let mut v = vec![0.0_f32; n_kv_heads * head_dim];

        b.ternary_gemv(&q_w, 1.0, &h_norm, &mut q, n_heads * head_dim, hidden)
            .unwrap();
        b.ternary_gemv(&k_w, 1.0, &h_norm, &mut k, n_kv_heads * head_dim, hidden)
            .unwrap();
        b.ternary_gemv(&v_w, 1.0, &h_norm, &mut v, n_kv_heads * head_dim, hidden)
            .unwrap();

        // ── RoPE ──────────────────────────────────────────────────────────
        b.rope_embed(
            &mut q, &mut k, cur_pos, head_dim, n_heads, n_kv_heads, 500_000.0,
        )
        .unwrap();

        // ── Attention ─────────────────────────────────────────────────────
        let k_cache = vec![0.0_f32; seq_len * n_kv_heads * head_dim];
        let v_cache = vec![0.1_f32; seq_len * n_kv_heads * head_dim];
        let mut attn_out = vec![0.0_f32; n_heads * head_dim];

        b.masked_attention(
            &q,
            &k_cache,
            &v_cache,
            &mut attn_out,
            n_heads,
            n_kv_heads,
            head_dim,
            cur_pos,
        )
        .unwrap();

        // ── attn sub-norm + o_proj ────────────────────────────────────────
        let sub_norm_w = vec![1.0_f32; n_heads * head_dim];
        let mut attn_normed = vec![0.0_f32; n_heads * head_dim];
        b.rms_norm(&attn_out, &sub_norm_w, 1e-5, &mut attn_normed)
            .unwrap();

        let o_w = pack_row_aligned(
            &vec![0i8; hidden * n_heads * head_dim],
            hidden,
            n_heads * head_dim,
        );
        let mut o_out = vec![0.0_f32; hidden];
        b.ternary_gemv(
            &o_w,
            1.0,
            &attn_normed,
            &mut o_out,
            hidden,
            n_heads * head_dim,
        )
        .unwrap();

        // residual
        let h2: Vec<f32> = h.iter().zip(o_out.iter()).map(|(a, b)| a + b).collect();

        // ── FFN ───────────────────────────────────────────────────────────
        let ffn_norm_w = vec![1.0_f32; hidden];
        let mut h2_norm = vec![0.0_f32; hidden];
        b.rms_norm(&h2, &ffn_norm_w, 1e-5, &mut h2_norm).unwrap();

        let gate_w = pack_row_aligned(&vec![0i8; ffn_dim * hidden], ffn_dim, hidden);
        let up_w = pack_row_aligned(&vec![0i8; ffn_dim * hidden], ffn_dim, hidden);
        let down_w = pack_row_aligned(&vec![0i8; hidden * ffn_dim], hidden, ffn_dim);

        let mut gate = vec![0.0_f32; ffn_dim];
        let mut up = vec![0.1_f32; ffn_dim]; // non-zero for interesting test
        b.ternary_gemv(&gate_w, 1.0, &h2_norm, &mut gate, ffn_dim, hidden)
            .unwrap();
        b.ternary_gemv(&up_w, 1.0, &h2_norm, &mut up, ffn_dim, hidden)
            .unwrap();

        b.squared_relu(&mut gate).unwrap();
        let mut inner = vec![0.0_f32; ffn_dim];
        b.elementwise_mul(&gate, &up, &mut inner).unwrap();

        let ffn_sub_w = vec![1.0_f32; ffn_dim];
        let mut inner_normed = vec![0.0_f32; ffn_dim];
        b.rms_norm(&inner, &ffn_sub_w, 1e-5, &mut inner_normed)
            .unwrap();

        let mut ffn_out = vec![0.0_f32; hidden];
        b.ternary_gemv(&down_w, 1.0, &inner_normed, &mut ffn_out, hidden, ffn_dim)
            .unwrap();

        let h3: Vec<f32> = h2.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();

        // Verify all outputs are finite.
        assert!(
            h3.iter().all(|v| v.is_finite()),
            "Mini forward pass output contains non-finite values: {:?}",
            h3
        );
    }
}
