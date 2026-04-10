//! # bitnet-gpu
//!
//! GPU backend for BitNet b1.58 inference via [`wgpu`].
//!
//! ## Supported backends (selected automatically by wgpu)
//!
//! | Platform | Backend(s)                  |
//! |----------|-----------------------------|
//! | Windows  | Vulkan, DX12                |
//! | macOS    | Metal                       |
//! | Linux    | Vulkan                      |
//! | Any      | Software (fallback only)    |
//!
//! ## Architecture
//!
//! [`GpuBackend`] implements [`bitnet_core::backend::Backend`] by:
//!
//! 1. Uploading input slices into GPU [`GpuBuffer`]s.
//! 2. Binding the buffers to pre-compiled WGSL compute pipelines.
//! 3. Dispatching compute workgroups.
//! 4. Downloading results back to host `f32` slices.
//!
//! For hot-path operations (ternary GEMV, RMSNorm, RoPE, attention) the
//! buffers are reused across calls to avoid repeated allocation.  A small
//! pool of pre-allocated scratch buffers is managed internally.
//!
//! ## Module Layout
//!
//! ```text
//! bitnet-gpu/
//! ├── lib.rs        ← this file: GpuBackend + Backend impl
//! ├── context.rs    ← wgpu device/queue/adapter creation
//! ├── buffer.rs     ← GpuBuffer wrapper (upload / download / binding)
//! ├── pipeline.rs   ← GpuPipelines: compiled WGSL compute pipelines
//! └── shaders/
//!     ├── gemv.wgsl       ← ternary GEMV kernel
//!     ├── norm.wgsl       ← RMSNorm kernel
//!     ├── rope.wgsl       ← Rotary Position Embedding kernel
//!     └── attention.wgsl  ← Causal GQA attention kernel
//! ```
//!
//! ## Invariants
//!
//! - Every [`Backend`] method validates shapes before touching the GPU.
//! - All GPU operations are submitted synchronously from the caller's
//!   perspective (blocking poll after each submit).
//! - The CPU fallback path (via [`bitnet_cpu::CpuBackend`]) is used for
//!   operations not yet covered by a WGSL shader.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod buffer;
pub mod context;
pub mod pipeline;

use std::sync::Arc;

use bitnet_core::backend::Backend;
use bitnet_core::error::{BitNetError, Result};
use bytemuck::{Pod, Zeroable};
use tracing::{debug, instrument, warn};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, CommandEncoderDescriptor, ComputePassDescriptor, Device,
    Queue,
};

use buffer::{BufferUsage, GpuBuffer};
use context::AdapterInfo;
use pipeline::GpuPipelines;

// ---------------------------------------------------------------------------
// WGSL parameter structs (must match the uniform block layout in shaders)
// These are Pod so they can be uploaded via write_buffer.
// ---------------------------------------------------------------------------

/// Parameters for the packed 2-bit ternary GEMV shader (`gemv.wgsl`).
///
/// Layout must match `struct GemvParams` in the shader.
/// Total size: 4 × 4 = 16 bytes (satisfies 16-byte uniform alignment).
///
/// # packed_cols invariant
/// `packed_cols = ceil(in_features / 4)` — number of packed bytes per weight row.
/// This tells the shader the byte stride between rows in the packed weight buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GemvParams {
    out_features: u32,
    in_features: u32,
    weight_scale: f32,
    /// Number of packed bytes per row = ceil(in_features / 4).
    packed_cols: u32,
}

/// Parameters for the RMSNorm shader (`norm.wgsl`).
///
/// Layout must match `struct NormParams` in the shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct NormParams {
    dim: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}

/// Parameters for the RoPE shader (`rope.wgsl`).
///
/// Layout must match `struct RopeParams` in the shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RopeParams {
    position: u32,
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    rope_theta: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Parameters for the attention shader (`attention.wgsl`).
///
/// Layout must match `struct AttnParams` in the shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AttnParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    cur_pos: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ---------------------------------------------------------------------------
// GpuBackend
// ---------------------------------------------------------------------------

/// GPU compute backend using wgpu (Vulkan / Metal / DX12).
///
/// Instantiate via [`GpuBackend::new`] (async) or [`GpuBackend::new_blocking`].
///
/// # Thread Safety
///
/// `GpuBackend` is `Send + Sync`. wgpu's `Device` and `Queue` are both
/// `Send + Sync` in the wgpu crate, so wrapping in `Arc<dyn Backend>` is safe.
pub struct GpuBackend {
    device: Device,
    queue: Queue,
    pipelines: GpuPipelines,
    adapter_info: AdapterInfo,
    name: String,
    /// CPU fallback backend for operations not covered by GPU shaders.
    cpu_fallback: bitnet_cpu::CpuBackend,
}

// Safety: wgpu Device and Queue implement Send + Sync.
unsafe impl Send for GpuBackend {}
unsafe impl Sync for GpuBackend {}

impl std::fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBackend")
            .field("adapter", &self.adapter_info.name)
            .field("backend", &self.adapter_info.backend)
            .finish()
    }
}

impl GpuBackend {
    /// Create a `GpuBackend` asynchronously for the adapter at index `device_id`.
    ///
    /// # Arguments
    ///
    /// - `device_id`: Zero-based index into the sorted GPU adapter list.
    ///   `0` selects the most capable discrete GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if no GPU adapters are found, if `device_id` is out of
    /// range, if device creation fails, or if WGSL shader compilation fails.
    pub async fn new(device_id: u32) -> anyhow::Result<Self> {
        let (device, queue, adapter_info) = context::create_wgpu_device(device_id).await?;

        let pipelines = GpuPipelines::new(&device)?;

        let cpu_fallback = bitnet_cpu::CpuBackend::new(None)
            .map_err(|e| anyhow::anyhow!("CPU fallback init failed: {e}"))?;

        let name = format!("GPU ({})", adapter_info);

        tracing::info!(
            adapter = %adapter_info,
            "GpuBackend initialised"
        );

        Ok(Self {
            device,
            queue,
            pipelines,
            adapter_info,
            name,
            cpu_fallback,
        })
    }

    /// Create a `GpuBackend` synchronously using `pollster::block_on`.
    ///
    /// Convenience wrapper for non-async contexts (CLI startup, tests).
    ///
    /// # Errors
    ///
    /// Propagates any error from [`GpuBackend::new`].
    pub fn new_blocking(device_id: u32) -> anyhow::Result<Self> {
        pollster::block_on(Self::new(device_id))
    }

    /// Wrap this backend in an `Arc<dyn Backend>` for shared ownership.
    pub fn into_arc(self) -> Arc<dyn Backend> {
        Arc::new(self)
    }

    /// Return the adapter info for this backend.
    pub fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter_info
    }

    // ------------------------------------------------------------------
    // Internal dispatch helpers
    // ------------------------------------------------------------------

    /// Dispatch the ternary GEMV compute shader.
    ///
    /// Uploads weights (packed u8) and input f32, dispatches
    /// `out_features` workgroups (one per output row), downloads output.
    fn dispatch_gemv(
        &self,
        weight_packed: &[u8],
        weight_scale: f32,
        input: &[f32],
        output: &mut [f32],
        out_features: usize,
        in_features: usize,
    ) -> anyhow::Result<()> {
        // packed_cols = ceil(in_features / 4): byte stride between rows.
        let packed_cols = (in_features + 3) / 4;

        // wgpu requires buffer sizes to be multiples of 4 bytes (COPY_BUFFER_ALIGNMENT).
        // The packed weight array may need padding to the next u32 boundary.
        let raw_weight_bytes = weight_packed.len() as u64;
        let weight_bytes = (raw_weight_bytes + 3) & !3;
        let input_bytes = (input.len() * std::mem::size_of::<f32>()) as u64;
        let output_bytes = (output.len() * std::mem::size_of::<f32>()) as u64;
        let params_bytes = std::mem::size_of::<GemvParams>() as u64;

        let weight_buf = GpuBuffer::new(
            &self.device,
            weight_bytes,
            BufferUsage::Storage,
            "gemv_weight",
        )?;
        let input_buf = GpuBuffer::new(
            &self.device,
            input_bytes,
            BufferUsage::Storage,
            "gemv_input",
        )?;
        let output_buf = GpuBuffer::new(
            &self.device,
            output_bytes,
            BufferUsage::StorageReadWrite,
            "gemv_output",
        )?;
        let params_buf = GpuBuffer::new(
            &self.device,
            params_bytes,
            BufferUsage::Uniform,
            "gemv_params",
        )?;

        // Upload packed weight bytes — pad to u32 boundary if needed.
        if raw_weight_bytes == weight_bytes {
            weight_buf.upload_u8(&self.device, &self.queue, weight_packed)?;
        } else {
            let mut padded = vec![0u8; weight_bytes as usize];
            padded[..weight_packed.len()].copy_from_slice(weight_packed);
            weight_buf.upload_u8(&self.device, &self.queue, &padded)?;
        }
        input_buf.upload_f32(&self.device, &self.queue, input)?;

        let params = GemvParams {
            out_features: out_features as u32,
            in_features: in_features as u32,
            weight_scale,
            packed_cols: packed_cols as u32,
        };
        params_buf.upload_uniform(&self.device, &self.queue, &params)?;

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("gemv_bind_group"),
            layout: &self.pipelines.gemv_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: weight_buf.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("gemv_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("gemv_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.gemv);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per output row.
            pass.dispatch_workgroups(out_features as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let result = output_buf.download_f32(&self.device, &self.queue)?;
        output.copy_from_slice(&result);

        debug!(
            out_features,
            in_features, packed_cols, "dispatch_gemv complete"
        );
        Ok(())
    }

    /// Dispatch the RMSNorm compute shader.
    ///
    /// Uploads input and weight f32 slices, dispatches 1 workgroup (the entire
    /// vector is processed by a single workgroup), downloads output.
    fn dispatch_rms_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        output: &mut [f32],
    ) -> anyhow::Result<()> {
        let dim = input.len();
        let bytes = (dim * std::mem::size_of::<f32>()) as u64;
        let params_bytes = std::mem::size_of::<NormParams>() as u64;

        let input_buf = GpuBuffer::new(&self.device, bytes, BufferUsage::Storage, "norm_input")?;
        let weight_buf = GpuBuffer::new(&self.device, bytes, BufferUsage::Storage, "norm_weight")?;
        let output_buf = GpuBuffer::new(
            &self.device,
            bytes,
            BufferUsage::StorageReadWrite,
            "norm_output",
        )?;
        let params_buf = GpuBuffer::new(
            &self.device,
            params_bytes,
            BufferUsage::Uniform,
            "norm_params",
        )?;

        input_buf.upload_f32(&self.device, &self.queue, input)?;
        weight_buf.upload_f32(&self.device, &self.queue, weight)?;

        let params = NormParams {
            dim: dim as u32,
            eps,
            _pad0: 0,
            _pad1: 0,
        };
        params_buf.upload_uniform(&self.device, &self.queue, &params)?;

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("norm_bind_group"),
            layout: &self.pipelines.norm_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("norm_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("norm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.norm);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup handles the entire vector.
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let result = output_buf.download_f32(&self.device, &self.queue)?;
        output.copy_from_slice(&result);

        debug!(dim, "dispatch_rms_norm complete");
        Ok(())
    }

    /// Dispatch the RoPE compute shader for Q and K tensors.
    ///
    /// Both Q and K are read-write: the shader rotates them in place.
    /// Dispatches `n_heads + n_kv_heads` workgroups (one per head).
    fn dispatch_rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        position: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        theta: f32,
    ) -> anyhow::Result<()> {
        let q_bytes = (q.len() * std::mem::size_of::<f32>()) as u64;
        let k_bytes = (k.len() * std::mem::size_of::<f32>()) as u64;
        let params_bytes = std::mem::size_of::<RopeParams>() as u64;

        let q_buf = GpuBuffer::new(
            &self.device,
            q_bytes,
            BufferUsage::StorageReadWrite,
            "rope_q",
        )?;
        let k_buf = GpuBuffer::new(
            &self.device,
            k_bytes,
            BufferUsage::StorageReadWrite,
            "rope_k",
        )?;
        let params_buf = GpuBuffer::new(
            &self.device,
            params_bytes,
            BufferUsage::Uniform,
            "rope_params",
        )?;

        q_buf.upload_f32(&self.device, &self.queue, q)?;
        k_buf.upload_f32(&self.device, &self.queue, k)?;

        let params = RopeParams {
            position: position as u32,
            head_dim: head_dim as u32,
            n_heads: n_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            rope_theta: theta,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        params_buf.upload_uniform(&self.device, &self.queue, &params)?;

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("rope_bind_group"),
            layout: &self.pipelines.rope_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: q_buf.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_binding(),
                },
            ],
        });

        // Combined Q+K dispatch: one workgroup per head.
        let total_heads = (n_heads + n_kv_heads) as u32;
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("rope_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("rope_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.rope);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(total_heads, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Download updated Q and K.
        let q_result = q_buf.download_f32(&self.device, &self.queue)?;
        let k_result = k_buf.download_f32(&self.device, &self.queue)?;
        q.copy_from_slice(&q_result);
        k.copy_from_slice(&k_result);

        debug!(
            position,
            n_heads, n_kv_heads, head_dim, "dispatch_rope complete"
        );
        Ok(())
    }

    /// Dispatch the attention compute shader.
    ///
    /// Uploads Q, K-cache, V-cache as read-only; output as read-write.
    /// Dispatches `n_heads` workgroups (one per query head).
    fn dispatch_attention(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        output: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        cur_pos: usize,
    ) -> anyhow::Result<()> {
        let q_bytes = (q.len() * std::mem::size_of::<f32>()) as u64;
        let k_bytes = (k_cache.len() * std::mem::size_of::<f32>()) as u64;
        let v_bytes = (v_cache.len() * std::mem::size_of::<f32>()) as u64;
        let out_bytes = (output.len() * std::mem::size_of::<f32>()) as u64;
        let params_bytes = std::mem::size_of::<AttnParams>() as u64;

        let q_buf = GpuBuffer::new(&self.device, q_bytes, BufferUsage::Storage, "attn_q")?;
        let k_buf = GpuBuffer::new(&self.device, k_bytes, BufferUsage::Storage, "attn_k")?;
        let v_buf = GpuBuffer::new(&self.device, v_bytes, BufferUsage::Storage, "attn_v")?;
        let out_buf = GpuBuffer::new(
            &self.device,
            out_bytes,
            BufferUsage::StorageReadWrite,
            "attn_out",
        )?;
        let params_buf = GpuBuffer::new(
            &self.device,
            params_bytes,
            BufferUsage::Uniform,
            "attn_params",
        )?;

        q_buf.upload_f32(&self.device, &self.queue, q)?;
        k_buf.upload_f32(&self.device, &self.queue, k_cache)?;
        v_buf.upload_f32(&self.device, &self.queue, v_cache)?;

        let scale = (head_dim as f32).sqrt().recip();
        let params = AttnParams {
            n_heads: n_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            cur_pos: cur_pos as u32,
            scale,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        params_buf.upload_uniform(&self.device, &self.queue, &params)?;

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("attn_bind_group"),
            layout: &self.pipelines.attention_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: q_buf.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: v_buf.as_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("attn_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("attn_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.attention);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per query head.
            pass.dispatch_workgroups(n_heads as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let result = out_buf.download_f32(&self.device, &self.queue)?;
        output.copy_from_slice(&result);

        debug!(
            n_heads,
            n_kv_heads, head_dim, cur_pos, "dispatch_attention complete"
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Backend impl
// ---------------------------------------------------------------------------

impl Backend for GpuBackend {
    // ------------------------------------------------------------------
    // Ternary GEMV
    // ------------------------------------------------------------------

    /// Ternary GEMV: `output[i] = Σ_j weight[i,j] * input[j] * weight_scale`.
    ///
    /// Dispatches the `gemv.wgsl` compute shader.  Falls back to the CPU
    /// implementation if GPU dispatch fails (with a warning).
    ///
    /// # Errors
    ///
    /// Returns [`BitNetError::InvalidShape`] for shape mismatches.
    /// Returns [`BitNetError::QuantizationError`] if `weight_scale ≤ 0`.
    #[instrument(
        level = "trace",
        skip(self, weight_packed, input, output),
        fields(out_features, in_features, weight_scale)
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
        // Validate shapes before touching the GPU.
        // Packed format: 4 ternary values per byte → ceil(in_features / 4) bytes per row.
        let packed_cols = (in_features + 3) / 4;
        let expected_weight = out_features
            .checked_mul(packed_cols)
            .ok_or_else(|| BitNetError::shape("packed weight size fits usize", "overflow"))?;
        if weight_packed.len() != expected_weight {
            return Err(BitNetError::shape(
                format!("{out_features} * ceil({in_features}/4) = {expected_weight}"),
                format!("{}", weight_packed.len()),
            ));
        }
        if input.len() != in_features {
            return Err(BitNetError::shape(
                format!("input.len() = {in_features}"),
                format!("{}", input.len()),
            ));
        }
        if output.len() != out_features {
            return Err(BitNetError::shape(
                format!("output.len() = {out_features}"),
                format!("{}", output.len()),
            ));
        }
        if weight_scale <= 0.0 || !weight_scale.is_finite() {
            return Err(BitNetError::quant(format!(
                "weight_scale must be finite > 0, got {weight_scale}"
            )));
        }

        // Attempt GPU dispatch; fall back to CPU on error.
        match self.dispatch_gemv(
            weight_packed,
            weight_scale,
            input,
            output,
            out_features,
            in_features,
        ) {
            Ok(()) => Ok(()),
            Err(e) => {
                warn!(error = %e, "GPU GEMV dispatch failed, falling back to CPU");
                self.cpu_fallback.ternary_gemv(
                    weight_packed,
                    weight_scale,
                    input,
                    output,
                    out_features,
                    in_features,
                )
            }
        }
    }

    // ------------------------------------------------------------------
    // Ternary GEMV with activation quantisation
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // RMSNorm
    // ------------------------------------------------------------------

    /// RMSNorm via `norm.wgsl`.  Falls back to CPU on GPU error.
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
        // Shape validation.
        if input.len() != weight.len() {
            return Err(BitNetError::shape(
                format!("input.len() == weight.len() = {}", weight.len()),
                format!("input.len() = {}", input.len()),
            ));
        }
        if input.len() != output.len() {
            return Err(BitNetError::shape(
                format!("input.len() == output.len() = {}", input.len()),
                format!("output.len() = {}", output.len()),
            ));
        }
        if eps <= 0.0 || !eps.is_finite() {
            return Err(BitNetError::quant(format!(
                "eps must be finite > 0, got {eps}"
            )));
        }

        match self.dispatch_rms_norm(input, weight, eps, output) {
            Ok(()) => Ok(()),
            Err(e) => {
                warn!(error = %e, "GPU RMSNorm dispatch failed, falling back to CPU");
                self.cpu_fallback.rms_norm(input, weight, eps, output)
            }
        }
    }

    // ------------------------------------------------------------------
    // RoPE
    // ------------------------------------------------------------------

    /// Rotary Position Embedding via `rope.wgsl`.  Falls back to CPU on error.
    ///
    /// # Errors
    ///
    /// Returns [`BitNetError::InvalidShape`] for shape mismatches or odd `head_dim`.
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
        // Shape validation.
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(BitNetError::shape(
                "head_dim must be even and > 0".to_string(),
                format!("head_dim = {head_dim}"),
            ));
        }
        if q.len() != n_heads * head_dim {
            return Err(BitNetError::shape(
                format!("q.len() = {} (n_heads * head_dim)", n_heads * head_dim),
                format!("q.len() = {}", q.len()),
            ));
        }
        if k.len() != n_kv_heads * head_dim {
            return Err(BitNetError::shape(
                format!(
                    "k.len() = {} (n_kv_heads * head_dim)",
                    n_kv_heads * head_dim
                ),
                format!("k.len() = {}", k.len()),
            ));
        }
        if theta <= 0.0 || !theta.is_finite() {
            return Err(BitNetError::config(format!(
                "theta must be finite > 0, got {theta}"
            )));
        }

        match self.dispatch_rope(q, k, position, head_dim, n_heads, n_kv_heads, theta) {
            Ok(()) => Ok(()),
            Err(e) => {
                warn!(error = %e, "GPU RoPE dispatch failed, falling back to CPU");
                self.cpu_fallback
                    .rope_embed(q, k, position, head_dim, n_heads, n_kv_heads, theta)
            }
        }
    }

    // ------------------------------------------------------------------
    // Attention
    // ------------------------------------------------------------------

    /// Causal GQA attention via `attention.wgsl`.  Falls back to CPU on error.
    ///
    /// # Errors
    ///
    /// Returns [`BitNetError::InvalidShape`] for shape mismatches.
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
        // Shape validation.
        if head_dim == 0 {
            return Err(BitNetError::shape(
                "head_dim > 0".to_string(),
                "head_dim = 0".to_string(),
            ));
        }
        if n_kv_heads == 0 || n_heads == 0 {
            return Err(BitNetError::config(
                "n_heads and n_kv_heads must be > 0".to_string(),
            ));
        }
        if n_heads % n_kv_heads != 0 {
            return Err(BitNetError::config(format!(
                "n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )));
        }
        let seq_len = cur_pos + 1;
        let required_cache = seq_len * n_kv_heads * head_dim;
        if q.len() != n_heads * head_dim {
            return Err(BitNetError::shape(
                format!("{}", n_heads * head_dim),
                format!("{}", q.len()),
            ));
        }
        if k_cache.len() < required_cache {
            return Err(BitNetError::shape(
                format!(">= {required_cache}"),
                format!("{}", k_cache.len()),
            ));
        }
        if v_cache.len() < required_cache {
            return Err(BitNetError::shape(
                format!(">= {required_cache}"),
                format!("{}", v_cache.len()),
            ));
        }
        if output.len() != n_heads * head_dim {
            return Err(BitNetError::shape(
                format!("{}", n_heads * head_dim),
                format!("{}", output.len()),
            ));
        }

        match self.dispatch_attention(
            q, k_cache, v_cache, output, n_heads, n_kv_heads, head_dim, cur_pos,
        ) {
            Ok(()) => Ok(()),
            Err(e) => {
                warn!(error = %e, "GPU attention dispatch failed, falling back to CPU");
                self.cpu_fallback.masked_attention(
                    q, k_cache, v_cache, output, n_heads, n_kv_heads, head_dim, cur_pos,
                )
            }
        }
    }

    // ------------------------------------------------------------------
    // Activation functions (CPU — trivial to implement on GPU later)
    // ------------------------------------------------------------------

    /// Squared ReLU in-place: `x[i] = max(0, x[i])²`.
    ///
    /// Delegates to the CPU fallback (SIMD-optimised on modern CPUs).
    fn squared_relu(&self, x: &mut [f32]) -> Result<()> {
        self.cpu_fallback.squared_relu(x)
    }

    /// Numerically-stable softmax in-place.
    ///
    /// Delegates to the CPU fallback.
    fn softmax(&self, x: &mut [f32]) -> Result<()> {
        self.cpu_fallback.softmax(x)
    }

    // ------------------------------------------------------------------
    // Element-wise helpers
    // ------------------------------------------------------------------

    /// Element-wise multiply: `out[i] = a[i] * b[i]`.
    ///
    /// Delegates to the CPU fallback.
    fn elementwise_mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        self.cpu_fallback.elementwise_mul(a, b, out)
    }

    fn lm_head_matmul_into(
        &self,
        hidden: &[f32],
        weights: &[f32],
        output: &mut [f32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<()> {
        // Delegate to CPU backend which has SIMD-accelerated lm_head.
        // GPU-native lm_head requires persistent weight buffers (tracked: G02/FP-3).
        self.cpu_fallback
            .lm_head_matmul_into(hidden, weights, output, vocab_size, hidden_size)
    }

    /// Ternary GEMV with pre-quantised i8 activation.
    ///
    /// Dequantises the i8 activation to f32 on the CPU (cheap: max 6912 elements),
    /// then dispatches the packed 2-bit GEMV shader on the GPU.
    ///
    /// # Mathematical equivalence
    /// `output[i] = (Σ_j W[i,j] * act_q[j]) * weight_scale * act_absmax / 127`
    ///            = `(Σ_j W[i,j] * (act_q[j] * act_absmax / 127)) * weight_scale`
    ///            = GPU ternary_gemv(W, weight_scale, dequant(act_q, act_absmax), output)
    fn ternary_gemv_preq(
        &self,
        weight_packed: &[u8],
        weight_scale: f32,
        activation_q: &[i8],
        act_absmax: f32,
        output: &mut [f32],
        out_features: usize,
        in_features: usize,
    ) -> Result<()> {
        // Dequantise i8 activation to f32 on the CPU.
        // act_absmax = max(|x|); scale = act_absmax / 127.
        // This allocation is at most 6912 * 4 = 27 KiB per call — acceptable.
        let scale = act_absmax / 127.0_f32;
        let input_f32: Vec<f32> = activation_q.iter().map(|&q| q as f32 * scale).collect();
        self.ternary_gemv(
            weight_packed,
            weight_scale,
            &input_f32,
            output,
            out_features,
            in_features,
        )
    }

    // ------------------------------------------------------------------
    // Device info
    // ------------------------------------------------------------------

    /// Human-readable backend name, e.g. `"GPU (NVIDIA GeForce RTX 4090 [Vulkan / DiscreteGpu])"`.
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
    use wgpu::{Backends, Instance, InstanceDescriptor, InstanceFlags};

    /// Returns `true` if any hardware (non-CPU-software) GPU adapter is present.
    fn has_hardware_gpu() -> bool {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            flags: InstanceFlags::default(),
            ..Default::default()
        });
        instance
            .enumerate_adapters(Backends::all())
            .iter()
            .any(|a| !matches!(a.get_info().device_type, wgpu::DeviceType::Cpu))
    }

    /// Try to create a [`GpuBackend`] for device 0.
    ///
    /// Returns `None` when no hardware GPU is available (e.g. headless CI).
    /// All tests that require a GPU use this helper and skip gracefully.
    fn try_make_gpu_backend() -> Option<GpuBackend> {
        GpuBackend::new_blocking(0).ok()
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

    // ------------------------------------------------------------------
    // Initialisation
    // ------------------------------------------------------------------

    #[test]
    fn gpu_backend_new_blocking_succeeds_when_gpu_available() {
        if !has_hardware_gpu() {
            return;
        }
        let backend = GpuBackend::new_blocking(0).expect("GpuBackend should init on hardware GPU");
        assert!(
            backend.device_name().starts_with("GPU"),
            "device_name must start with 'GPU', got '{}'",
            backend.device_name()
        );
        assert!(!backend.adapter_info().name.is_empty());
    }

    #[test]
    fn gpu_backend_out_of_range_device_id_returns_error() {
        let result = GpuBackend::new_blocking(9999);
        assert!(result.is_err(), "device_id=9999 should return an error");
    }

    // ------------------------------------------------------------------
    // Ternary GEMV
    // ------------------------------------------------------------------

    #[test]
    fn gpu_ternary_gemv_2x3() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        // W = [[1, 0, -1], [-1, 1, 0]], x = [2, 3, 4], scale = 0.5
        // row 0: (1*2 + 0*3 + (-1)*4) * 0.5 = (2 + 0 - 4) * 0.5 = -1.0
        // row 1: ((-1)*2 + 1*3 + 0*4) * 0.5 = (-2 + 3 + 0) * 0.5 =  0.5
        let weight_i8: Vec<i8> = vec![1, 0, -1, -1, 1, 0];
        let weight_packed = pack_row_aligned(&weight_i8, 2, 3);
        let input = vec![2.0_f32, 3.0, 4.0];
        let mut output = vec![0.0_f32; 2];
        b.ternary_gemv(&weight_packed, 0.5, &input, &mut output, 2, 3)
            .unwrap();
        assert!(
            (output[0] - (-1.0_f32)).abs() < 1e-4,
            "row 0: expected -1.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 0.5_f32).abs() < 1e-4,
            "row 1: expected 0.5, got {}",
            output[1]
        );
    }

    #[test]
    fn gpu_ternary_gemv_wrong_shape_returns_error() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        // packed weight has 3 bytes but out_features=2, in_features=3 requires 2
        let weight_packed: Vec<u8> = vec![0u8; 3];
        let input = vec![1.0_f32; 3];
        let mut output = vec![0.0_f32; 2];
        let err = b
            .ternary_gemv(&weight_packed, 1.0, &input, &mut output, 2, 3)
            .unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn gpu_ternary_gemv_zero_scale_returns_error() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        let weight_packed = pack_row_aligned(&[1i8; 4], 1, 4);
        let input = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 1];
        let err = b
            .ternary_gemv(&weight_packed, 0.0, &input, &mut output, 1, 4)
            .unwrap_err();
        assert!(matches!(err, BitNetError::QuantizationError(_)));
    }

    /// Verify that GPU packed GEMV matches the CPU reference for a larger matrix
    /// drawn from analytical values.
    ///
    /// # Construction
    /// W = 4×8 all-ones matrix, scale = 1.0, x = [1, 1, 1, 1, 1, 1, 1, 1]
    /// Expected: output[i] = 8.0 for all i  (dot product of all-ones)
    #[test]
    fn gpu_packed_gemv_matches_cpu_all_ones() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        let weight_i8: Vec<i8> = vec![1i8; 32]; // 4×8 all-ones
        let weight_packed = pack_row_aligned(&weight_i8, 4, 8);
        let input = vec![1.0_f32; 8];
        let mut output = vec![0.0_f32; 4];
        b.ternary_gemv(&weight_packed, 1.0, &input, &mut output, 4, 8)
            .unwrap();
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 8.0).abs() < 1e-3, "output[{i}]: expected 8.0, got {v}");
        }
    }

    /// Verify GPU GEMV with alternating +1/-1 weights and unit inputs.
    ///
    /// W = 1×8 row [+1,-1,+1,-1,+1,-1,+1,-1], scale=1.0, x=[1,1,1,1,1,1,1,1]
    /// Expected: output[0] = 0.0  (cancellation)
    #[test]
    fn gpu_packed_gemv_alternating_cancels() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        let weight_i8: Vec<i8> = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let weight_packed = pack_row_aligned(&weight_i8, 1, 8);
        let input = vec![1.0_f32; 8];
        let mut output = vec![0.0_f32; 1];
        b.ternary_gemv(&weight_packed, 1.0, &input, &mut output, 1, 8)
            .unwrap();
        assert!(
            output[0].abs() < 1e-3,
            "expected 0.0 (cancellation), got {}",
            output[0]
        );
    }

    /// GPU GEMV must match the CPU result on a 64×64 matrix with mixed ternary values.
    #[test]
    fn gpu_packed_gemv_64x64_matches_cpu() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        let rows = 64usize;
        let cols = 64usize;
        // Deterministic ternary pattern: index mod 3 -> +1, 0, -1
        let weight_i8: Vec<i8> = (0..rows * cols)
            .map(|i| match i % 3 {
                0 => 1i8,
                1 => 0i8,
                _ => -1i8,
            })
            .collect();
        let weight_packed = pack_row_aligned(&weight_i8, rows, cols);
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let scale = 0.5_f32;

        // CPU reference
        let mut cpu_out = vec![0.0_f32; rows];
        for r in 0..rows {
            let mut acc = 0.0_f32;
            for c in 0..cols {
                acc += weight_i8[r * cols + c] as f32 * input[c];
            }
            cpu_out[r] = acc * scale;
        }

        // GPU output
        let mut gpu_out = vec![0.0_f32; rows];
        b.ternary_gemv(&weight_packed, scale, &input, &mut gpu_out, rows, cols)
            .unwrap();

        for (i, (&cpu, &gpu)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            assert!(
                (cpu - gpu).abs() < 1e-3,
                "row {i}: cpu={cpu:.6}, gpu={gpu:.6}, diff={:.6}",
                (cpu - gpu).abs()
            );
        }
    }

    // ------------------------------------------------------------------
    // RMSNorm
    // ------------------------------------------------------------------

    #[test]
    fn gpu_rms_norm_result_is_finite() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        b.rms_norm(&input, &weight, 1e-5, &mut output).unwrap();
        assert!(
            output.iter().all(|v| v.is_finite()),
            "all outputs must be finite, got {:?}",
            output
        );
    }

    #[test]
    fn gpu_rms_norm_wrong_lengths_returns_error() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        let input = vec![1.0_f32; 4];
        let weight = vec![1.0_f32; 5]; // length mismatch
        let mut output = vec![0.0_f32; 4];
        let err = b.rms_norm(&input, &weight, 1e-5, &mut output).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    // ------------------------------------------------------------------
    // RoPE
    // ------------------------------------------------------------------

    #[test]
    fn gpu_rope_position_zero_is_identity() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        // At position=0 all rotation angles are 0 → cos=1, sin=0 → identity.
        let head_dim = 8;
        let mut q: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let mut k: Vec<f32> = (0..8).map(|i| i as f32 * 0.2).collect();
        let q_orig = q.clone();
        b.rope_embed(&mut q, &mut k, 0, head_dim, 1, 1, 10000.0)
            .unwrap();
        for (i, (&orig, &rot)) in q_orig.iter().zip(q.iter()).enumerate() {
            assert!(
                (orig - rot).abs() < 1e-4,
                "q[{i}]: expected {orig}, got {rot} (pos=0 should be identity)"
            );
        }
    }

    #[test]
    fn gpu_rope_wrong_q_length_returns_error() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        // q should be n_heads * head_dim = 1 * 8 = 8, but we pass 7
        let mut q = vec![0.0_f32; 7];
        let mut k = vec![0.0_f32; 8];
        let err = b
            .rope_embed(&mut q, &mut k, 0, 8, 1, 1, 10000.0)
            .unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    // ------------------------------------------------------------------
    // Attention
    // ------------------------------------------------------------------

    #[test]
    fn gpu_attention_single_position_returns_value() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        let head_dim = 4;
        // Single query head, single KV head, single cached position.
        let q = vec![1.0_f32, 0.0, 0.0, 0.0];
        let k_cache = vec![1.0_f32, 0.0, 0.0, 0.0];
        let v_cache = vec![0.1_f32, 0.2, 0.3, 0.4];
        let mut output = vec![0.0_f32; 4];
        b.masked_attention(&q, &k_cache, &v_cache, &mut output, 1, 1, head_dim, 0)
            .unwrap();
        // Single position → softmax([score]) = [1.0] → output = v_cache row.
        let expected = [0.1_f32, 0.2, 0.3, 0.4];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "output[{i}]: got {got}, expected {exp}"
            );
        }
    }

    // ------------------------------------------------------------------
    // CPU-delegated ops
    // ------------------------------------------------------------------

    #[test]
    fn gpu_squared_relu_via_backend() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        let mut x = vec![-2.0_f32, 0.0, 1.0, 3.0];
        b.squared_relu(&mut x).unwrap();
        // sqrelu(x) = max(0, x)^2
        let expected = [0.0_f32, 0.0, 1.0, 9.0];
        for (i, (&got, &exp)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "x[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn gpu_softmax_sums_to_one() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        b.softmax(&mut x).unwrap();
        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax outputs must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn gpu_elementwise_mul_basic() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        let a = vec![2.0_f32, 3.0, 4.0];
        let bv = vec![0.5_f32, 2.0, 1.0];
        let mut out = vec![0.0_f32; 3];
        b.elementwise_mul(&a, &bv, &mut out).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6, "out[0]: got {}", out[0]);
        assert!((out[1] - 6.0).abs() < 1e-6, "out[1]: got {}", out[1]);
        assert!((out[2] - 4.0).abs() < 1e-6, "out[2]: got {}", out[2]);
    }

    // ------------------------------------------------------------------
    // Device info
    // ------------------------------------------------------------------

    #[test]
    fn gpu_device_name_starts_with_gpu() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        assert!(
            b.device_name().starts_with("GPU"),
            "device_name must start with 'GPU', got '{}'",
            b.device_name()
        );
    }

    #[test]
    fn gpu_backend_into_arc() {
        let Some(b) = try_make_gpu_backend() else {
            return;
        };
        let arc: Arc<dyn Backend> = b.into_arc();
        assert!(
            arc.device_name().starts_with("GPU"),
            "Arc backend device_name must start with 'GPU', got '{}'",
            arc.device_name()
        );
    }
}
