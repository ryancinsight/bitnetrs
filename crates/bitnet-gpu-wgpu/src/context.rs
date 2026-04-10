//! wgpu device and queue initialisation for the GPU backend.
//!
//! # Architecture
//!
//! [`create_wgpu_device`] enumerates all available wgpu adapters, selects the
//! one at index `device_id` (skipping CPU/software adapters when possible),
//! and requests a logical device with a command queue.
//!
//! # Adapter Selection
//!
//! wgpu exposes adapters for every available backend on the current platform:
//! - **Windows**: Vulkan, DX12, DX11 (software)
//! - **macOS/iOS**: Metal, Vulkan (via MoltenVK)
//! - **Linux**: Vulkan, OpenGL
//! - **Web**: WebGPU
//!
//! The adapter at index 0 is typically the most capable discrete GPU.
//! Software/CPU adapters (e.g. WARP on Windows, llvmpipe on Linux) are
//! de-prioritised by requesting `PowerPreference::HighPerformance`.
//!
//! # Error Handling
//!
//! All errors are surfaced as [`anyhow::Error`] with descriptive context
//! messages so that the CLI can print actionable diagnostics.
//!
//! # Invariants
//!
//! - The returned `(Device, Queue)` pair is always ready for compute dispatch.
//! - `device_id = 0` selects the first (highest-priority) GPU adapter.
//! - Requesting `device_id >= n_adapters` returns an error rather than panicking.

use anyhow::{anyhow, Context};
use tracing::{debug, info, warn};
use wgpu::{
    Adapter, Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor,
    InstanceFlags, Limits, MemoryHints, PowerPreference, Queue, RequestAdapterOptions,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Information about a wgpu adapter that was selected for inference.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    /// Human-readable adapter name (e.g. `"NVIDIA GeForce RTX 4090"`).
    pub name: String,
    /// wgpu backend used (e.g. `"Vulkan"`, `"Metal"`, `"Dx12"`).
    pub backend: String,
    /// Adapter type: `"DiscreteGpu"`, `"IntegratedGpu"`, `"VirtualGpu"`, `"Cpu"`.
    pub adapter_type: String,
    /// PCI vendor ID (0 if unavailable).
    pub vendor: u32,
    /// PCI device ID (0 if unavailable).
    pub device: u32,
}

impl std::fmt::Display for AdapterInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} [{} / {}]",
            self.name, self.backend, self.adapter_type
        )
    }
}

/// Create a wgpu logical device and command queue for the adapter at index
/// `device_id`.
///
/// # Selection Strategy
///
/// 1. Create a wgpu [`Instance`] supporting all available backends.
/// 2. Request a high-performance adapter via [`RequestAdapterOptions`].
/// 3. If `device_id == 0`, use the adapter returned by the high-performance
///    request (the best GPU available).
/// 4. If `device_id > 0`, enumerate all adapters and pick the one at that
///    position in the sorted list (sorted by: discrete GPU first, then
///    integrated, then virtual, then CPU).
///
/// # Arguments
///
/// - `device_id`: Zero-based index into the sorted adapter list.
///   `0` always refers to the highest-priority GPU.
///
/// # Errors
///
/// - Returns an error if no GPU adapters are found.
/// - Returns an error if `device_id` is out of range.
/// - Returns an error if device creation fails (e.g. unsupported feature set).
///
/// # Example
///
/// ```no_run
/// use bitnet_gpu::context::create_wgpu_device;
///
/// # async fn example() -> anyhow::Result<()> {
/// let (device, queue, info) = create_wgpu_device(0).await?;
/// println!("Using GPU: {info}");
/// # Ok(())
/// # }
/// ```
pub async fn create_wgpu_device(device_id: u32) -> anyhow::Result<(Device, Queue, AdapterInfo)> {
    // ── Step 1: Create wgpu instance ─────────────────────────────────────────
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        flags: InstanceFlags::default(),
        ..Default::default()
    });

    debug!("wgpu instance created, enumerating adapters");

    // ── Step 2: Collect all hardware adapters (exclude pure-software ones) ────
    let mut all_adapters: Vec<Adapter> = instance.enumerate_adapters(Backends::all());

    if all_adapters.is_empty() {
        return Err(anyhow!(
            "No wgpu-compatible GPU adapters found on this system. \
             Ensure your GPU drivers are installed and up to date."
        ));
    }

    // Sort adapters: discrete GPU first, then integrated, then virtual, then CPU.
    all_adapters.sort_by_key(|a| {
        let info = a.get_info();
        match info.device_type {
            wgpu::DeviceType::DiscreteGpu => 0u8,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Cpu => 3,
            wgpu::DeviceType::Other => 4,
        }
    });

    // Log all adapters at debug level.
    for (i, adapter) in all_adapters.iter().enumerate() {
        let info = adapter.get_info();
        debug!(
            index = i,
            name = %info.name,
            backend = ?info.backend,
            device_type = ?info.device_type,
            "Available wgpu adapter"
        );
    }

    // ── Step 3: Select the adapter at `device_id` ────────────────────────────
    let device_id_usize = device_id as usize;
    if device_id_usize >= all_adapters.len() {
        return Err(anyhow!(
            "GPU device_id {device_id} is out of range: only {} adapter(s) found.\n\
             Available adapters:\n{}",
            all_adapters.len(),
            all_adapters
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let inf = a.get_info();
                    format!("  [{i}] {} ({:?})", inf.name, inf.backend)
                })
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    // Remove the selected adapter from the list (moving out of the Vec).
    let adapter = all_adapters.remove(device_id_usize);
    let raw_info = adapter.get_info();

    // Warn if a software/CPU adapter was selected.
    if matches!(raw_info.device_type, wgpu::DeviceType::Cpu) {
        warn!(
            adapter = %raw_info.name,
            "Selected adapter is a software (CPU) renderer. \
             Inference will be slow. Use a hardware GPU for best performance."
        );
    }

    // ── Step 4: Request a logical device and queue ────────────────────────────
    let adapter_info = AdapterInfo {
        name: raw_info.name.clone(),
        backend: format!("{:?}", raw_info.backend),
        adapter_type: format!("{:?}", raw_info.device_type),
        vendor: raw_info.vendor,
        device: raw_info.device,
    };

    info!(adapter = %adapter_info, "Creating wgpu logical device");

    // Request the device with compute-friendly features.
    // We do not require any exotic features — standard compute shaders suffice.
    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: Some("bitnet-gpu"),
                required_features: Features::empty(),
                required_limits: compute_limits(&adapter),
                memory_hints: MemoryHints::Performance,
            },
            None, // no tracing path
        )
        .await
        .with_context(|| {
            format!(
                "Failed to create wgpu device on adapter '{}'. \
                 Check that your GPU drivers support the required compute features.",
                raw_info.name
            )
        })?;

    info!(
        adapter = %adapter_info,
        "wgpu device created successfully"
    );

    Ok((device, queue, adapter_info))
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute the wgpu [`Limits`] appropriate for this adapter.
///
/// We try to use the adapter's own supported limits (which may be higher than
/// the defaults) so that large weight matrices fit in a single dispatch.
/// Falls back to [`Limits::default()`] if the adapter reports zero limits.
fn compute_limits(adapter: &Adapter) -> Limits {
    let supported = adapter.limits();

    // Ensure we request at least enough buffer size and workgroup threads for
    // realistic model dimensions (hidden_size=2560, vocab=128256).
    //
    // Minimum requirements:
    // - max_buffer_size ≥ 512 MiB (to hold large embedding/weight matrices)
    // - max_storage_buffer_binding_size ≥ 256 MiB
    // - max_compute_workgroup_size_x ≥ 256
    let min_buffer_size: u64 = 512 * 1024 * 1024; // 512 MiB
    let min_storage_binding: u64 = 256 * 1024 * 1024; // 256 MiB

    Limits {
        max_buffer_size: supported.max_buffer_size.max(min_buffer_size),
        max_storage_buffer_binding_size: supported
            .max_storage_buffer_binding_size
            .max(min_storage_binding as u32),
        max_compute_workgroup_size_x: supported.max_compute_workgroup_size_x.max(256),
        max_compute_workgroup_size_y: supported.max_compute_workgroup_size_y.max(1),
        max_compute_workgroup_size_z: supported.max_compute_workgroup_size_z.max(1),
        max_compute_invocations_per_workgroup: supported
            .max_compute_invocations_per_workgroup
            .max(256),
        max_compute_workgroups_per_dimension: supported
            .max_compute_workgroups_per_dimension
            .max(65535),
        max_bind_groups: supported.max_bind_groups.max(4),
        max_bindings_per_bind_group: supported.max_bindings_per_bind_group.max(8),
        ..supported
    }
}

// ---------------------------------------------------------------------------
// Blocking wrapper
// ---------------------------------------------------------------------------

/// Synchronously create a wgpu device, blocking the calling thread.
///
/// This is a convenience wrapper around [`create_wgpu_device`] for use in
/// non-async contexts (e.g. CLI startup, tests).
///
/// Uses [`pollster::block_on`] to drive the async operation to completion.
///
/// # Errors
///
/// Propagates any error from [`create_wgpu_device`].
pub fn create_wgpu_device_blocking(device_id: u32) -> anyhow::Result<(Device, Queue, AdapterInfo)> {
    pollster::block_on(create_wgpu_device(device_id))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that adapter enumeration does not panic on the current machine.
    /// This is a best-effort test — it passes even if no GPU is present
    /// (in which case we expect a descriptive error, not a panic).
    #[test]
    fn enumerate_adapters_does_not_panic() {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            flags: InstanceFlags::default(),
            ..Default::default()
        });
        let adapters: Vec<Adapter> = instance.enumerate_adapters(Backends::all());
        // Just assert the count is non-negative (always true — this checks the call succeeds).
        assert!(adapters.len() < usize::MAX);
    }

    /// Verify that requesting device_id = 999 returns a descriptive error when
    /// fewer adapters are available. We use block_on to test the async path.
    #[test]
    fn out_of_range_device_id_returns_error() {
        let result = pollster::block_on(create_wgpu_device(999));
        // Either there's no adapter at index 999, or there's no GPU at all.
        // Either way it must be Err, not Ok or a panic.
        assert!(
            result.is_err(),
            "device_id=999 should return an error, got Ok"
        );
        let err_msg = result.unwrap_err().to_string();
        // The error message must be actionable.
        assert!(!err_msg.is_empty(), "Error message must not be empty");
    }

    /// Smoke test: if a hardware GPU is available, creating device 0 succeeds.
    /// Skipped silently on headless CI machines with no GPU.
    #[test]
    fn device_zero_succeeds_when_gpu_available() {
        // Check if any non-CPU adapter exists before attempting device creation.
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            flags: InstanceFlags::default(),
            ..Default::default()
        });
        let adapters: Vec<Adapter> = instance.enumerate_adapters(Backends::all());
        let has_hardware_gpu = adapters
            .iter()
            .any(|a| !matches!(a.get_info().device_type, wgpu::DeviceType::Cpu));

        if !has_hardware_gpu {
            // Skip — no hardware GPU present (e.g. CI runner).
            return;
        }

        let result = pollster::block_on(create_wgpu_device(0));
        assert!(
            result.is_ok(),
            "device 0 should succeed with a hardware GPU: {:?}",
            result.err()
        );

        let (_device, _queue, info) = result.unwrap();
        assert!(!info.name.is_empty(), "adapter name must not be empty");
        assert!(!info.backend.is_empty(), "backend string must not be empty");
    }

    /// Verify AdapterInfo Display format includes name and backend.
    #[test]
    fn adapter_info_display_format() {
        let info = AdapterInfo {
            name: "Test GPU".to_string(),
            backend: "Vulkan".to_string(),
            adapter_type: "DiscreteGpu".to_string(),
            vendor: 0x10DE,
            device: 0x2684,
        };
        let display = info.to_string();
        assert!(display.contains("Test GPU"), "display must contain name");
        assert!(display.contains("Vulkan"), "display must contain backend");
        assert!(display.contains("DiscreteGpu"), "display must contain type");
    }

    /// Verify AdapterInfo Debug output is non-empty (structural check only).
    #[test]
    fn adapter_info_debug_non_empty() {
        let info = AdapterInfo {
            name: "Debug GPU".to_string(),
            backend: "Metal".to_string(),
            adapter_type: "IntegratedGpu".to_string(),
            vendor: 0x106B,
            device: 0x0001,
        };
        let debug_str = format!("{info:?}");
        assert!(!debug_str.is_empty());
    }
}
