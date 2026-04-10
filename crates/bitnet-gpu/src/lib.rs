//! # bitnet-gpu
//!
//! Facade crate for BitNet GPU backends.
//!
//! ## Purpose
//!
//! This crate is the stable GPU abstraction boundary for the workspace.
//! Model and inference crates depend on `bitnet-gpu` as the authoritative
//! GPU-facing crate, while concrete implementations live in child crates:
//!
//! - `bitnet-gpu-wgpu` — `wgpu` implementation
//! - `bitnet-gpu-cuda` — CUDA implementation via `cutile-rs`
//!
//! ## Architectural role
//!
//! This crate preserves Dependency Inversion:
//!
//! - parent crate defines the public GPU backend selection surface
//! - child crates provide concrete implementations
//! - model code does not depend on `wgpu` or CUDA directly
//!
//! ## Current exports
//!
//! At present, the `wgpu` backend is the only concrete GPU implementation
//! wired into the facade. CUDA support is introduced as a separate workspace
//! crate and can be integrated through this facade without changing upstream
//! model APIs.
//!
//! ## Invariants
//!
//! - [`GpuBackend`] remains the canonical GPU backend type exposed to the rest
//!   of the workspace.
//! - [`GpuBackend::new`] and [`GpuBackend::new_blocking`] preserve the existing
//!   initialization contract.
//! - Upstream crates continue to import `bitnet_gpu::GpuBackend` unchanged.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub use bitnet_gpu_wgpu::GpuBackend;

/// Create a `wgpu`-backed [`GpuBackend`] asynchronously.
///
/// This is the canonical facade constructor for the current GPU implementation.
///
/// # Arguments
///
/// - `device_id`: Zero-based index into the sorted GPU adapter list.
///
/// # Errors
///
/// Propagates any initialization error from the concrete `wgpu` backend.
#[inline]
pub async fn create_wgpu_backend(device_id: u32) -> anyhow::Result<GpuBackend> {
    bitnet_gpu_wgpu::GpuBackend::new(device_id).await
}

/// Create a `wgpu`-backed [`GpuBackend`] synchronously.
///
/// Convenience wrapper for non-async contexts such as CLI startup, tests,
/// and runtime backend factories.
///
/// # Arguments
///
/// - `device_id`: Zero-based index into the sorted GPU adapter list.
///
/// # Errors
///
/// Propagates any initialization error from the concrete `wgpu` backend.
#[inline]
pub fn create_wgpu_backend_blocking(device_id: u32) -> anyhow::Result<GpuBackend> {
    bitnet_gpu_wgpu::GpuBackend::new_blocking(device_id)
}

/// Backward-compatible alias for the synchronous `wgpu` facade constructor.
///
/// This preserves the naming used by the NPU integration and other call sites
/// during the workspace split.
#[inline]
pub fn new_wgpu_backend_blocking(device_id: u32) -> anyhow::Result<GpuBackend> {
    create_wgpu_backend_blocking(device_id)
}
