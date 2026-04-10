//! GPU buffer management for the BitNet b1.58 wgpu backend.
//!
//! # Design
//!
//! [`GpuBuffer`] wraps a [`wgpu::Buffer`] with typed upload/download helpers so
//! that the rest of the GPU backend can work with Rust slices (`&[f32]`, `&[i8]`,
//! `&[i32]`, `&[u8]`) without managing raw byte offsets.
//!
//! # Memory Model
//!
//! wgpu buffers are typed by their *usage flags* at creation time:
//!
//! | [`BufferUsage`] variant | wgpu flags                                      | Purpose                       |
//! |-------------------------|-------------------------------------------------|-------------------------------|
//! | `Storage`               | STORAGE + COPY_DST                              | GPU-side weight/activation    |
//! | `StorageReadWrite`      | STORAGE + COPY_DST + COPY_SRC                   | GPU-side mutable output       |
//! | `Uniform`               | UNIFORM + COPY_DST                              | Shader parameter structs      |
//! | `Staging`               | MAP_READ + COPY_DST                             | CPU←GPU readback              |
//! | `Upload`                | MAP_WRITE + COPY_SRC                            | CPU→GPU upload                |
//!
//! # Transfer Protocol
//!
//! CPU → GPU:
//! 1. Write data into a transient [`BufferUsage::Upload`] buffer via `map_write`.
//! 2. Submit a `copy_buffer_to_buffer` command to copy into the target storage buffer.
//! 3. Submit and poll to synchronise.
//!
//! GPU → CPU:
//! 1. Submit a `copy_buffer_to_buffer` command from storage into a staging buffer.
//! 2. Submit and poll.
//! 3. Map the staging buffer read-only and copy bytes into a `Vec<f32>`.
//! 4. Unmap the staging buffer.
//!
//! # Invariants
//!
//! - `GpuBuffer::size` always equals the byte size passed at construction.
//! - Upload/download helpers check that the slice byte size matches `self.size`.
//! - All wgpu `poll` calls use `Maintain::Wait` to guarantee synchronisation before
//!   returning to the caller.

use anyhow::{anyhow, Context};
use bytemuck::Pod;
use tracing::{debug, instrument};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindingResource, Buffer, BufferDescriptor, BufferSlice, BufferUsages, Device, Maintain, Queue,
};

// ---------------------------------------------------------------------------
// BufferUsage
// ---------------------------------------------------------------------------

/// Intended usage of a [`GpuBuffer`].
///
/// This is a higher-level abstraction over [`wgpu::BufferUsages`] that maps
/// directly to common BitNet inference patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    /// Read-only GPU storage (weights, input activations).
    ///
    /// wgpu flags: `STORAGE | COPY_DST`
    Storage,

    /// Read-write GPU storage (output activations, KV cache).
    ///
    /// wgpu flags: `STORAGE | COPY_DST | COPY_SRC`
    StorageReadWrite,

    /// Uniform buffer for shader parameters.
    ///
    /// wgpu flags: `UNIFORM | COPY_DST`
    Uniform,

    /// CPU-readable staging buffer for readback.
    ///
    /// wgpu flags: `MAP_READ | COPY_DST`
    Staging,

    /// CPU-writable upload buffer for CPU→GPU transfers.
    ///
    /// wgpu flags: `MAP_WRITE | COPY_SRC`
    Upload,
}

impl BufferUsage {
    /// Convert to the corresponding [`wgpu::BufferUsages`] bit-flags.
    pub const fn to_wgpu(self) -> BufferUsages {
        match self {
            Self::Storage => BufferUsages::STORAGE.union(BufferUsages::COPY_DST),
            Self::StorageReadWrite => BufferUsages::STORAGE
                .union(BufferUsages::COPY_DST)
                .union(BufferUsages::COPY_SRC),
            Self::Uniform => BufferUsages::UNIFORM.union(BufferUsages::COPY_DST),
            Self::Staging => BufferUsages::MAP_READ.union(BufferUsages::COPY_DST),
            Self::Upload => BufferUsages::MAP_WRITE.union(BufferUsages::COPY_SRC),
        }
    }
}

// ---------------------------------------------------------------------------
// GpuBuffer
// ---------------------------------------------------------------------------

/// A typed wrapper around a [`wgpu::Buffer`] with upload/download helpers.
///
/// # Invariants
/// - `self.size` equals the exact byte size of the underlying wgpu buffer.
/// - The buffer's usage flags match `self.usage`.
/// - After any upload, the buffer contents are synchronised with the GPU
///   before this function returns.
#[derive(Debug)]
pub struct GpuBuffer {
    /// The underlying wgpu buffer.
    buf: Buffer,
    /// Byte size of the buffer.
    pub size: u64,
    /// Human-readable label for debugging.
    pub label: String,
    /// The usage class of this buffer.
    pub usage: BufferUsage,
}

impl GpuBuffer {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Allocate an uninitialised GPU buffer of `size` bytes with the given usage.
    ///
    /// The buffer contents are undefined until data is uploaded.
    ///
    /// # Arguments
    ///
    /// - `device`:  The wgpu logical device.
    /// - `size`:    Byte size of the buffer.  Must be > 0.
    /// - `usage`:   Intended usage (determines wgpu flags).
    /// - `label`:   Human-readable label shown in GPU debug tools.
    ///
    /// # Errors
    ///
    /// Returns an error if `size == 0`.
    pub fn new(
        device: &Device,
        size: u64,
        usage: BufferUsage,
        label: impl Into<String>,
    ) -> anyhow::Result<Self> {
        if size == 0 {
            return Err(anyhow!("GpuBuffer size must be > 0"));
        }
        let label_str = label.into();
        let buf = device.create_buffer(&BufferDescriptor {
            label: Some(&label_str),
            size,
            usage: usage.to_wgpu(),
            mapped_at_creation: false,
        });
        debug!(label = %label_str, size, ?usage, "GpuBuffer allocated");
        Ok(Self {
            buf,
            size,
            label: label_str,
            usage,
        })
    }

    /// Allocate a GPU buffer and immediately initialise it with `data`.
    ///
    /// Uses `wgpu::util::DeviceExt::create_buffer_init` which maps the buffer
    /// at creation time — the most efficient path for static data (e.g. weights).
    ///
    /// # Type Parameter
    ///
    /// `T` must implement [`bytemuck::Pod`] so that a `&[T]` can be
    /// safely reinterpreted as a `&[u8]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` is empty.
    pub fn from_data<T: Pod>(
        device: &Device,
        data: &[T],
        usage: BufferUsage,
        label: impl Into<String>,
    ) -> anyhow::Result<Self> {
        if data.is_empty() {
            return Err(anyhow!("GpuBuffer::from_data: data must not be empty"));
        }
        let label_str = label.into();
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&label_str),
            contents: bytes,
            usage: usage.to_wgpu(),
        });
        let size = bytes.len() as u64;
        debug!(label = %label_str, size, ?usage, "GpuBuffer initialised from data");
        Ok(Self {
            buf,
            size,
            label: label_str,
            usage,
        })
    }

    // ------------------------------------------------------------------
    // Upload (CPU → GPU)
    // ------------------------------------------------------------------

    /// Upload `f32` data from the CPU into this GPU buffer.
    ///
    /// Creates a transient upload staging buffer, writes the data into it
    /// via `queue.write_buffer`, then copies into `self`.  The copy is
    /// submitted and polled synchronously.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len() * 4 != self.size` (byte size mismatch).
    #[instrument(level = "debug", skip(self, device, queue, data), fields(label = %self.label))]
    pub fn upload_f32(&self, device: &Device, queue: &Queue, data: &[f32]) -> anyhow::Result<()> {
        let expected_bytes = data.len() * std::mem::size_of::<f32>();
        self.check_size(expected_bytes as u64)?;
        queue.write_buffer(&self.buf, 0, bytemuck::cast_slice(data));
        device.poll(Maintain::Wait);
        Ok(())
    }

    /// Upload `i8` data from the CPU into this GPU buffer.
    ///
    /// Used for ternary weight matrices (`i8` values ∈ {-1, 0, +1}).
    /// Since wgpu shaders do not natively support `i8` storage buffers, the
    /// caller must use the `i32` variant for compute shaders.  This method
    /// uploads raw bytes; the companion [`GpuBuffer::upload_i8_as_i32`] method
    /// widens each `i8` to `i32` for shader compatibility.
    ///
    /// # Errors
    ///
    /// Returns an error if byte sizes do not match.
    #[instrument(level = "debug", skip(self, device, queue, data), fields(label = %self.label))]
    pub fn upload_i8(&self, device: &Device, queue: &Queue, data: &[i8]) -> anyhow::Result<()> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        self.check_size(bytes.len() as u64)?;
        queue.write_buffer(&self.buf, 0, bytes);
        device.poll(Maintain::Wait);
        Ok(())
    }

    /// Upload `i8` ternary weights, widening each value to `i32`.
    ///
    /// WGSL storage buffers use `array<i32>` rather than `array<i8>`, so
    /// ternary weights must be widened before upload.  This method performs
    /// the sign-extension and uploads the widened data.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len() * 4 != self.size`.
    #[instrument(level = "debug", skip(self, device, queue, data), fields(label = %self.label))]
    pub fn upload_i8_as_i32(
        &self,
        device: &Device,
        queue: &Queue,
        data: &[i8],
    ) -> anyhow::Result<()> {
        let expected_bytes = data.len() * std::mem::size_of::<i32>();
        self.check_size(expected_bytes as u64)?;
        // Widen i8 → i32 (sign-extending).
        let widened: Vec<i32> = data.iter().map(|&v| v as i32).collect();
        queue.write_buffer(&self.buf, 0, bytemuck::cast_slice(&widened));
        device.poll(Maintain::Wait);
        Ok(())
    }

    /// Upload `u8` raw bytes into this GPU buffer.
    ///
    /// Used for packed ternary weights or arbitrary byte payloads.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len() != self.size`.
    #[instrument(level = "debug", skip(self, device, queue, data), fields(label = %self.label))]
    pub fn upload_u8(&self, device: &Device, queue: &Queue, data: &[u8]) -> anyhow::Result<()> {
        self.check_size(data.len() as u64)?;
        queue.write_buffer(&self.buf, 0, data);
        device.poll(Maintain::Wait);
        Ok(())
    }

    /// Upload a `bytemuck::Pod` value as a uniform buffer.
    ///
    /// Convenience wrapper for uploading shader parameter structs.
    ///
    /// # Errors
    ///
    /// Returns an error if `std::mem::size_of::<T>() != self.size`.
    pub fn upload_uniform<T: Pod>(
        &self,
        device: &Device,
        queue: &Queue,
        value: &T,
    ) -> anyhow::Result<()> {
        let bytes = bytemuck::bytes_of(value);
        self.check_size(bytes.len() as u64)?;
        queue.write_buffer(&self.buf, 0, bytes);
        device.poll(Maintain::Wait);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Download (GPU → CPU)
    // ------------------------------------------------------------------

    /// Download the contents of this GPU buffer into a `Vec<f32>`.
    ///
    /// Creates a transient [`BufferUsage::Staging`] buffer, copies GPU data
    /// into it, then maps and reads it back into a freshly allocated `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns an error if `self.size % 4 != 0` (not a multiple of `f32` size),
    /// or if the wgpu map operation fails.
    #[instrument(level = "debug", skip(self, device, queue), fields(label = %self.label))]
    pub fn download_f32(&self, device: &Device, queue: &Queue) -> anyhow::Result<Vec<f32>> {
        if self.size % 4 != 0 {
            return Err(anyhow!(
                "GpuBuffer '{}': size {} is not a multiple of 4 (f32 size)",
                self.label,
                self.size
            ));
        }

        // Step 1: Allocate a staging buffer.
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("{}_staging_readback", self.label)),
            size: self.size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Step 2: Encode and submit a copy command.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bitnet_readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buf, 0, &staging, 0, self.size);
        queue.submit(std::iter::once(encoder.finish()));

        // Step 3: Map the staging buffer and read bytes.
        let slice: BufferSlice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(Maintain::Wait);

        receiver
            .recv()
            .context("channel closed before map callback")?
            .context("wgpu map_async failed")?;

        let data = {
            let view = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };

        staging.unmap();
        debug!(
            label = %self.label,
            n_floats = data.len(),
            "GpuBuffer downloaded"
        );

        Ok(data)
    }

    // ------------------------------------------------------------------
    // Bind group helper
    // ------------------------------------------------------------------

    /// Return a [`wgpu::BindingResource`] wrapping the full buffer slice.
    ///
    /// Used when building bind groups for compute shaders:
    ///
    /// ```ignore
    /// let entry = wgpu::BindGroupEntry {
    ///     binding: 0,
    ///     resource: my_buf.as_binding(),
    /// };
    /// ```
    #[inline]
    pub fn as_binding(&self) -> BindingResource<'_> {
        self.buf.as_entire_binding()
    }

    /// Return a reference to the raw [`wgpu::Buffer`].
    ///
    /// Needed when passing the buffer to `encoder.copy_buffer_to_buffer` or
    /// other wgpu API calls that require the raw handle.
    #[inline]
    pub fn raw(&self) -> &Buffer {
        &self.buf
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Verify that `expected_bytes` matches `self.size`.
    fn check_size(&self, expected_bytes: u64) -> anyhow::Result<()> {
        if expected_bytes != self.size {
            return Err(anyhow!(
                "GpuBuffer '{}': size mismatch — buffer is {} bytes, data is {} bytes",
                self.label,
                self.size,
                expected_bytes
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::{Backends, Instance, InstanceDescriptor, InstanceFlags};

    /// Returns true if a hardware (non-software) GPU adapter is available.
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

    /// Create a wgpu device for testing (software fallback allowed).
    async fn test_device() -> Option<(Device, Queue)> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            flags: InstanceFlags::default(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()
    }

    #[test]
    fn buffer_usage_to_wgpu_flags() {
        assert!(BufferUsage::Storage
            .to_wgpu()
            .contains(BufferUsages::STORAGE));
        assert!(BufferUsage::Storage
            .to_wgpu()
            .contains(BufferUsages::COPY_DST));
        assert!(!BufferUsage::Storage
            .to_wgpu()
            .contains(BufferUsages::COPY_SRC));

        assert!(BufferUsage::StorageReadWrite
            .to_wgpu()
            .contains(BufferUsages::COPY_SRC));

        assert!(BufferUsage::Uniform
            .to_wgpu()
            .contains(BufferUsages::UNIFORM));

        assert!(BufferUsage::Staging
            .to_wgpu()
            .contains(BufferUsages::MAP_READ));

        assert!(BufferUsage::Upload
            .to_wgpu()
            .contains(BufferUsages::MAP_WRITE));
    }

    #[test]
    fn gpu_buffer_upload_download_roundtrip() {
        // This test requires a GPU / software adapter. Skip gracefully if none.
        let Some((device, queue)) = pollster::block_on(test_device()) else {
            return;
        };

        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let byte_size = (data.len() * std::mem::size_of::<f32>()) as u64;

        // Allocate a read-write storage buffer (uploadable + downloadable).
        let buf =
            GpuBuffer::new(&device, byte_size, BufferUsage::StorageReadWrite, "test_rw").unwrap();

        // Upload.
        buf.upload_f32(&device, &queue, &data).unwrap();

        // Download and compare.
        let recovered = buf.download_f32(&device, &queue).unwrap();
        assert_eq!(recovered.len(), data.len());
        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-7,
                "index {i}: orig={orig}, recovered={rec}"
            );
        }
    }

    #[test]
    fn gpu_buffer_size_mismatch_returns_error() {
        let Some((device, queue)) = pollster::block_on(test_device()) else {
            return;
        };

        let buf = GpuBuffer::new(&device, 16, BufferUsage::StorageReadWrite, "small_buf").unwrap();
        // Try to upload 5 f32s (20 bytes) into a 16-byte buffer.
        let data = vec![1.0_f32; 5];
        let err = buf.upload_f32(&device, &queue, &data).unwrap_err();
        assert!(
            err.to_string().contains("size mismatch"),
            "error must mention size mismatch: {err}"
        );
    }

    #[test]
    fn gpu_buffer_zero_size_returns_error() {
        let Some((device, _queue)) = pollster::block_on(test_device()) else {
            return;
        };

        let err = GpuBuffer::new(&device, 0, BufferUsage::Storage, "zero").unwrap_err();
        assert!(
            err.to_string().contains("size must be > 0"),
            "error must mention zero size: {err}"
        );
    }

    #[test]
    fn gpu_buffer_i8_as_i32_upload_correct_size() {
        let Some((device, queue)) = pollster::block_on(test_device()) else {
            return;
        };

        // 4 i8 values → 4 i32 values = 16 bytes.
        let data: Vec<i8> = vec![1, -1, 0, 1];
        let byte_size = (data.len() * std::mem::size_of::<i32>()) as u64;
        let buf = GpuBuffer::new(
            &device,
            byte_size,
            BufferUsage::StorageReadWrite,
            "i8_as_i32",
        )
        .unwrap();
        // Should succeed without error.
        buf.upload_i8_as_i32(&device, &queue, &data).unwrap();
    }

    #[test]
    fn gpu_buffer_from_data_roundtrip() {
        let Some((device, queue)) = pollster::block_on(test_device()) else {
            return;
        };

        let data: Vec<f32> = vec![3.14, 2.71, 1.41, 1.73];
        let buf = GpuBuffer::from_data(
            &device,
            &data,
            BufferUsage::StorageReadWrite,
            "from_data_test",
        )
        .unwrap();

        let recovered = buf.download_f32(&device, &queue).unwrap();
        assert_eq!(recovered.len(), 4);
        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-6,
                "index {i}: orig={orig}, rec={rec}"
            );
        }
    }

    #[test]
    fn gpu_buffer_from_data_empty_returns_error() {
        let Some((device, _queue)) = pollster::block_on(test_device()) else {
            return;
        };

        let empty: Vec<f32> = vec![];
        let err = GpuBuffer::from_data(&device, &empty, BufferUsage::Storage, "empty").unwrap_err();
        assert!(
            err.to_string().contains("must not be empty"),
            "error must mention empty data: {err}"
        );
    }

    #[test]
    fn gpu_buffer_download_non_multiple_of_4_returns_error() {
        let Some((device, queue)) = pollster::block_on(test_device()) else {
            return;
        };

        // 7 bytes is not a multiple of 4.
        let buf = GpuBuffer::new(&device, 8, BufferUsage::StorageReadWrite, "odd_size").unwrap();
        // Upload 8 bytes (valid).
        let data: Vec<u8> = vec![0u8; 8];
        buf.upload_u8(&device, &queue, &data).unwrap();
        // Download as f32 from an 8-byte buffer is valid (8/4=2 floats).
        let recovered = buf.download_f32(&device, &queue).unwrap();
        assert_eq!(recovered.len(), 2);
    }

    #[test]
    fn buffer_usage_all_variants_have_nonzero_flags() {
        let variants = [
            BufferUsage::Storage,
            BufferUsage::StorageReadWrite,
            BufferUsage::Uniform,
            BufferUsage::Staging,
            BufferUsage::Upload,
        ];
        for v in variants {
            assert!(
                !v.to_wgpu().is_empty(),
                "{v:?} must have non-zero wgpu flags"
            );
        }
    }
}
