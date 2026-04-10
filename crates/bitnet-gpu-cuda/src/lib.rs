//! # bitnet-gpu-cuda
//!
//! CUDA backend for BitNet b1.58 inference via `cutile-rs`.
//!
//! ## Status
//!
//! This crate establishes the CUDA backend boundary in the workspace but does
//! not yet provide CUDA kernel implementations. Its purpose is to:
//!
//! - reserve the canonical crate for CUDA execution
//! - define the concrete backend type used by the GPU facade
//! - preserve architectural separation from the `wgpu` implementation
//! - provide explicit, descriptive failure modes until CUDA kernels land
//!
//! ## `cutile-rs` platform constraints
//!
//! The upstream `NVlabs/cutile-rs` project is not a single crates.io package;
//! it is a multi-crate Git workspace with host-side CUDA APIs, compiler
//! components, and MLIR/LLVM integration.
//!
//! According to the upstream repository documentation, the current supported
//! environment is:
//!
//! - Linux
//! - NVIDIA GPU with supported compute capability
//! - CUDA toolkit installed
//! - LLVM 21 with MLIR
//! - Rust 1.89+
//!
//! This means CUDA integration in this crate must be treated as a
//! platform-gated backend path. The current Windows development environment for
//! `bitnetrs` is not assumed to satisfy those requirements.
//!
//! Until a compatible Linux/CUDA/LLVM toolchain is configured and the relevant
//! `cutile-rs` workspace crates are wired in, this crate remains an explicit
//! stub backend rather than a partial or fabricated CUDA implementation.
//!
//! ## Architectural role
//!
//! The workspace GPU stack is split as follows:
//!
//! - `bitnet-gpu`       — facade / stable GPU-facing boundary
//! - `bitnet-gpu-wgpu`  — `wgpu` implementation
//! - `bitnet-gpu-cuda`  — CUDA implementation via `cutile-rs`
//!
//! Upstream crates should depend on the facade rather than this crate directly.
//!
//! ## Invariants
//!
//! - `CudaBackend` is `Send + Sync`.
//! - Construction never fabricates CUDA availability.
//! - Every [`Backend`] method returns a descriptive backend error until the real
//!   CUDA implementation is added.
//! - No method silently falls back to CPU; fallback policy belongs above this
//!   crate at the facade / factory layer.
//!
//! ## Next implementation steps
//!
//! 1. Add platform-gated Git dependencies on the required `cutile-rs` workspace
//!    crates.
//! 2. Add CUDA device/context/stream initialization through `cutile-rs` on a
//!    compatible Linux/CUDA/LLVM environment.
//! 3. Implement packed 2-bit ternary GEMV kernels.
//! 4. Add CUDA paths for RMSNorm, RoPE, attention, and lm_head.
//! 5. Integrate backend selection through the GPU facade crate.
//! 6. Add CPU-vs-CUDA numerical equivalence tests.

#![warn(missing_docs)]
#![warn(clippy::all)]

use std::sync::Arc;

use bitnet_core::backend::Backend;
use bitnet_core::error::{BitNetError, Result};
use half::bf16;

/// CUDA compute backend for BitNet inference.
///
/// This type is the canonical concrete backend for future CUDA execution.
/// At present it is a stub that reports explicit unsupported-operation errors
/// until CUDA kernels and runtime initialization are implemented.
#[derive(Debug, Clone)]
pub struct CudaBackend {
    name: String,
}

impl CudaBackend {
    /// Create a new CUDA backend.
    ///
    /// # Errors
    ///
    /// Always returns a backend error until CUDA runtime integration is
    /// implemented in this crate on a supported Linux/CUDA/LLVM toolchain.
    pub fn new(device_id: u32) -> Result<Self> {
        Err(BitNetError::backend(
            "CUDA",
            format!(
                "CUDA backend is not implemented yet (requested device_id={device_id}); \
                 cutile-rs currently requires a compatible Linux/CUDA/LLVM environment, \
                 and CUDA runtime initialization plus kernels have not been added in \
                 bitnet-gpu-cuda"
            ),
        ))
    }

    /// Blocking constructor for API symmetry with other backend crates.
    ///
    /// # Errors
    ///
    /// Always returns a backend error until CUDA runtime integration is
    /// implemented in this crate.
    #[inline]
    pub fn new_blocking(device_id: u32) -> Result<Self> {
        Self::new(device_id)
    }

    /// Wrap this backend in an `Arc<dyn Backend>`.
    #[inline]
    pub fn into_arc(self) -> Arc<dyn Backend> {
        Arc::new(self)
    }

    #[inline]
    fn unsupported<T>(&self, op: &str) -> Result<T> {
        Err(BitNetError::backend(
            "CUDA",
            format!(
                "operation '{op}' is not implemented for {}; \
                 CUDA kernels have not been added yet",
                self.name
            ),
        ))
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self {
            name: "CUDA (unimplemented)".to_string(),
        }
    }
}

impl Backend for CudaBackend {
    fn ternary_gemv(
        &self,
        _weight_packed: &[u8],
        _weight_scale: f32,
        _input: &[f32],
        _output: &mut [f32],
        _out_features: usize,
        _in_features: usize,
    ) -> Result<()> {
        self.unsupported("ternary_gemv")
    }

    fn ternary_gemv_with_activation_quant(
        &self,
        _weight_packed: &[u8],
        _weight_scale: f32,
        _input: &[f32],
        _output: &mut [f32],
        _out_features: usize,
        _in_features: usize,
    ) -> Result<()> {
        self.unsupported("ternary_gemv_with_activation_quant")
    }

    fn rms_norm(
        &self,
        _input: &[f32],
        _weight: &[f32],
        _eps: f32,
        _output: &mut [f32],
    ) -> Result<()> {
        self.unsupported("rms_norm")
    }

    fn rope_embed(
        &self,
        _q: &mut [f32],
        _k: &mut [f32],
        _position: usize,
        _head_dim: usize,
        _n_heads: usize,
        _n_kv_heads: usize,
        _theta: f32,
    ) -> Result<()> {
        self.unsupported("rope_embed")
    }

    fn masked_attention(
        &self,
        _q: &[f32],
        _k_cache: &[f32],
        _v_cache: &[f32],
        _output: &mut [f32],
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _cur_pos: usize,
    ) -> Result<()> {
        self.unsupported("masked_attention")
    }

    fn squared_relu(&self, _x: &mut [f32]) -> Result<()> {
        self.unsupported("squared_relu")
    }

    fn softmax(&self, _x: &mut [f32]) -> Result<()> {
        self.unsupported("softmax")
    }

    fn elementwise_mul(&self, _a: &[f32], _b: &[f32], _out: &mut [f32]) -> Result<()> {
        self.unsupported("elementwise_mul")
    }

    fn sqrelu_gate(&self, _gate: &[f32], _up: &[f32], _out: &mut [f32]) -> Result<()> {
        self.unsupported("sqrelu_gate")
    }

    fn lm_head_matmul_into(
        &self,
        _hidden: &[f32],
        _weights: &[f32],
        _output: &mut [f32],
        _vocab_size: usize,
        _hidden_size: usize,
    ) -> Result<()> {
        self.unsupported("lm_head_matmul_into")
    }

    fn lm_head_matmul_bf16_into(
        &self,
        _hidden: &[f32],
        _weights_bf16: &[bf16],
        _output: &mut [f32],
        _vocab_size: usize,
        _hidden_size: usize,
    ) -> Result<()> {
        self.unsupported("lm_head_matmul_bf16_into")
    }

    fn ternary_gemv_preq(
        &self,
        _weight_packed: &[u8],
        _weight_scale: f32,
        _activation_q: &[i8],
        _act_absmax: f32,
        _output: &mut [f32],
        _out_features: usize,
        _in_features: usize,
    ) -> Result<()> {
        self.unsupported("ternary_gemv_preq")
    }

    fn lm_head_matmul_i8_into(
        &self,
        _hidden: &[f32],
        _weights_i8: &[i8],
        _scales: &[f32],
        _output: &mut [f32],
        _vocab_size: usize,
        _hidden_size: usize,
    ) -> Result<()> {
        self.unsupported("lm_head_matmul_i8_into")
    }

    fn device_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_backend_new_returns_descriptive_error() {
        let err = CudaBackend::new(0).expect_err("CUDA backend stub must not initialize");
        let msg = err.to_string();
        assert!(
            msg.contains("CUDA backend is not implemented yet"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn cuda_backend_default_device_name_is_non_empty() {
        let backend = CudaBackend::default();
        assert!(
            !backend.device_name().is_empty(),
            "device name must not be empty"
        );
    }

    #[test]
    fn cuda_backend_ops_return_backend_error() {
        let backend = CudaBackend::default();

        let err = backend
            .softmax(&mut [0.0_f32, 1.0])
            .expect_err("stub softmax must fail");
        match err {
            BitNetError::BackendError { backend, message } => {
                assert_eq!(backend, "CUDA");
                assert!(
                    message.contains("softmax"),
                    "message must mention operation, got: {message}"
                );
            }
            other => panic!("expected BackendError, got {other:?}"),
        }
    }

    #[test]
    fn cuda_backend_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CudaBackend>();
    }

    #[test]
    fn cuda_backend_into_arc_preserves_device_name() {
        let backend = CudaBackend::default();
        let expected = backend.device_name().to_string();
        let arc = backend.into_arc();
        assert_eq!(arc.device_name(), expected);
    }
}
