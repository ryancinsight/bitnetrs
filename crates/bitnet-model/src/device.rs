//! Backend factory for BitNet b1.58 model inference.
//!
//! # Architecture
//!
//! This module implements the Dependency Inversion Principle: the model
//! architecture depends on the abstract [`Backend`] trait, and this factory
//! function resolves which concrete backend to instantiate at runtime based
//! on the caller's [`Device`] preference.
//!
//! # Device Resolution Order
//!
//! | [`Device`] variant | Concrete type             | Crate          |
//! |--------------------|--------------------------|----------------|
//! | `Cpu { threads }`  | [`CpuBackend`]           | `bitnet-cpu`   |
//! | `Gpu { device_id }`| [`GpuBackend`]           | `bitnet-gpu`   |
//! | `Npu { device_id }`| [`NpuBackend`]           | `bitnet-npu`   |
//!
//! # Fallback Behaviour
//!
//! - `Npu`: If no NPU is detected, [`NpuBackend`] transparently falls back to
//!   [`CpuBackend`] without returning an error.
//! - `Gpu`: If no GPU adapter is found, returns an error (no silent fallback,
//!   to avoid unexpected performance degradation).
//! - `Cpu`: Always succeeds unless the Rayon thread pool cannot be initialised
//!   (which should never happen on a valid system).
//!
//! # Invariants
//!
//! - The returned `Arc<dyn Backend>` is `Send + Sync`.
//! - All backend instances are fully initialised and ready for dispatch.
//! - `device_name()` on the returned backend reflects the actual hardware used.

use std::sync::Arc;

use anyhow::Context;
use tracing::{info, instrument};

use bitnet_core::backend::{Backend, Device};

// ---------------------------------------------------------------------------
// create_backend
// ---------------------------------------------------------------------------

/// Instantiate the appropriate compute backend for the given [`Device`].
///
/// # Arguments
///
/// - `device`: The requested compute device.
///
/// # Returns
///
/// An `Arc<dyn Backend>` wrapping the concrete backend.
///
/// # Errors
///
/// Returns an error if the requested device/backend cannot be initialised:
/// - `Cpu`: Fails only if Rayon pool initialisation fails (extremely rare).
/// - `Gpu`: Fails if no wgpu-compatible GPU adapter is found at `device_id`.
/// - `Npu`: Falls back to CPU on detection failure; only errors if CPU also fails.
///
/// # Examples
///
/// ```no_run
/// use bitnet_core::backend::Device;
/// use bitnet_model::device::create_backend;
///
/// // CPU with 4 threads
/// let cpu_backend = create_backend(Device::Cpu { threads: Some(4) }).unwrap();
/// assert!(cpu_backend.device_name().contains("CPU"));
///
/// // First available GPU
/// let gpu_result = create_backend(Device::Gpu { device_id: 0 });
/// // May fail if no GPU is present — handle gracefully.
///
/// // NPU (falls back to CPU if not found)
/// let npu_backend = create_backend(Device::Npu { device_id: 0 }).unwrap();
/// ```
#[instrument(level = "info", fields(device = %device))]
pub fn create_backend(device: Device) -> anyhow::Result<Arc<dyn Backend>> {
    match device {
        // ── CPU backend ───────────────────────────────────────────────────
        Device::Cpu { threads } => {
            info!(threads = ?threads, "Initialising CPU backend");
            let backend =
                bitnet_cpu::CpuBackend::new(threads).context("Failed to initialise CPU backend")?;
            info!(name = %backend.device_name(), "CPU backend ready");
            Ok(backend.into_arc())
        }

        // ── GPU backend ───────────────────────────────────────────────────
        Device::Gpu { device_id } => {
            info!(device_id, "Initialising GPU backend (wgpu)");
            let backend = bitnet_gpu::GpuBackend::new_blocking(device_id).with_context(|| {
                format!(
                    "Failed to initialise GPU backend for device_id={device_id}. \
                         Check that a compatible GPU driver is installed."
                )
            })?;
            info!(name = %backend.device_name(), "GPU backend ready");
            Ok(backend.into_arc())
        }

        // ── NPU backend ───────────────────────────────────────────────────
        Device::Npu { device_id } => {
            info!(device_id, "Initialising NPU backend");
            let backend = bitnet_npu::NpuBackend::new(device_id).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to initialise NPU backend: {e}. \
                         Ensure NPU drivers are installed (Windows DirectML required)."
                )
            })?;
            let using_npu = backend.is_using_npu();
            info!(
                name = %backend.device_name(),
                using_npu,
                "NPU backend ready"
            );
            Ok(backend.into_arc())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::backend::{Backend, Device};

    // ------------------------------------------------------------------
    // CPU backend
    // ------------------------------------------------------------------

    #[test]
    fn create_cpu_backend_no_threads() {
        let backend = create_backend(Device::Cpu { threads: None })
            .expect("CPU backend must always initialise");
        assert!(
            backend.device_name().contains("CPU"),
            "CPU backend name must contain 'CPU', got '{}'",
            backend.device_name()
        );
    }

    #[test]
    fn create_cpu_backend_one_thread() {
        let backend = create_backend(Device::Cpu { threads: Some(1) })
            .expect("CPU backend with 1 thread must initialise");
        assert!(backend.device_name().contains("CPU"));
    }

    #[test]
    fn create_cpu_backend_four_threads() {
        let backend = create_backend(Device::Cpu { threads: Some(4) })
            .expect("CPU backend with 4 threads must initialise");
        assert!(backend.device_name().contains("CPU"));
    }

    #[test]
    fn cpu_backend_is_send_sync() {
        // Compile-time check: Arc<dyn Backend> must be Send + Sync.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<dyn Backend>>();
    }

    #[test]
    fn cpu_backend_ternary_gemv_works_via_factory() {
        // Verify the returned backend is fully functional.
        let backend = create_backend(Device::cpu()).unwrap();

        // 2×3 weight matrix: [[1, 0, -1], [-1, 1, 0]], scale=0.5
        // input = [2, 3, 4]
        // row 0: 1*2 + 0*3 + (-1)*4 = -2 → -2 * 0.5 = -1.0
        // row 1: (-1)*2 + 1*3 + 0*4 =  1 →  1 * 0.5 =  0.5
        let weight: Vec<i8> = vec![1, 0, -1, -1, 1, 0];
        let input = vec![2.0_f32, 3.0, 4.0];
        let mut output = vec![0.0_f32; 2];

        backend
            .ternary_gemv(&weight, 0.5, &input, &mut output, 2, 3)
            .expect("ternary_gemv must succeed");

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
    fn cpu_backend_rms_norm_works_via_factory() {
        let backend = create_backend(Device::cpu()).unwrap();
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        backend
            .rms_norm(&input, &weight, 1e-5, &mut output)
            .expect("rms_norm must succeed");
        assert!(
            output.iter().all(|v| v.is_finite()),
            "RMSNorm output must be finite"
        );
    }

    #[test]
    fn cpu_backend_softmax_sums_to_one_via_factory() {
        let backend = create_backend(Device::cpu()).unwrap();
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        backend.softmax(&mut x).expect("softmax must succeed");
        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax sum must be 1.0, got {sum}"
        );
    }

    #[test]
    fn cpu_backend_squared_relu_via_factory() {
        let backend = create_backend(Device::cpu()).unwrap();
        let mut x = vec![-2.0_f32, 0.0, 1.0, 3.0];
        backend
            .squared_relu(&mut x)
            .expect("squared_relu must succeed");
        let expected = [0.0_f32, 0.0, 1.0, 9.0];
        for (i, (&got, &exp)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "x[{i}]: got {got}, expected {exp}"
            );
        }
    }

    // ------------------------------------------------------------------
    // GPU backend
    // ------------------------------------------------------------------

    /// Attempt to create a GPU backend; skip if no hardware GPU is present.
    ///
    /// This test validates the factory function for GPU without requiring
    /// specific hardware — it passes on systems with a GPU and is skipped
    /// silently on systems without one.
    #[test]
    fn create_gpu_backend_device_zero() {
        let result = create_backend(Device::Gpu { device_id: 0 });
        match result {
            Ok(backend) => {
                assert!(
                    backend.device_name().starts_with("GPU"),
                    "GPU backend name must start with 'GPU', got '{}'",
                    backend.device_name()
                );
            }
            Err(_) => {
                // No GPU available — acceptable on CI runners / headless machines.
                // The test is a no-op in this case.
            }
        }
    }

    #[test]
    fn create_gpu_backend_invalid_device_id_returns_error() {
        let result = create_backend(Device::Gpu { device_id: 9999 });
        assert!(
            result.is_err(),
            "device_id=9999 must return an error (no 10000th GPU)"
        );
    }

    #[test]
    fn gpu_backend_error_message_is_descriptive() {
        let result = create_backend(Device::Gpu { device_id: 9999 });
        assert!(result.is_err(), "device_id=9999 must return an error");
        let msg = result.err().unwrap().to_string();
        assert!(!msg.is_empty(), "GPU error message must not be empty");
        // The error should mention the device_id or drivers.
        assert!(
            msg.contains("9999")
                || msg.contains("adapter")
                || msg.contains("GPU")
                || msg.contains("driver"),
            "GPU error must be descriptive: '{msg}'"
        );
    }

    // ------------------------------------------------------------------
    // NPU backend
    // ------------------------------------------------------------------

    #[test]
    fn create_npu_backend_always_succeeds() {
        // NpuBackend falls back to CPU if no NPU is found, so this must always succeed.
        let backend = create_backend(Device::Npu { device_id: 0 })
            .expect("NPU backend must always succeed (CPU fallback)");
        let name = backend.device_name();
        assert!(
            !name.is_empty(),
            "NPU backend device name must not be empty"
        );
        // The name must mention either NPU or CPU.
        assert!(
            name.contains("NPU") || name.contains("CPU"),
            "NPU backend name must contain 'NPU' or 'CPU', got '{name}'"
        );
    }

    #[test]
    fn npu_backend_functional_via_factory() {
        let backend = create_backend(Device::npu()).unwrap();

        // Basic smoke test: softmax must work on the fallback path.
        let mut x = vec![0.0_f32, 1.0, 2.0];
        backend
            .softmax(&mut x)
            .expect("softmax must work via NPU backend");
        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax sum = {sum}, expected 1.0"
        );
    }

    // ------------------------------------------------------------------
    // Device convenience constructors
    // ------------------------------------------------------------------

    #[test]
    fn device_cpu_convenience_constructor() {
        let d = Device::cpu();
        assert_eq!(d, Device::Cpu { threads: None });
    }

    #[test]
    fn device_gpu_convenience_constructor() {
        let d = Device::gpu();
        assert_eq!(d, Device::Gpu { device_id: 0 });
    }

    #[test]
    fn device_npu_convenience_constructor() {
        let d = Device::npu();
        assert_eq!(d, Device::Npu { device_id: 0 });
    }

    // ------------------------------------------------------------------
    // Arc<dyn Backend> send/sync and cloneability
    // ------------------------------------------------------------------

    #[test]
    fn cpu_backend_arc_can_be_cloned() {
        let backend: Arc<dyn Backend> = create_backend(Device::cpu()).unwrap();
        let cloned = Arc::clone(&backend);
        // Both point to the same underlying backend.
        assert_eq!(backend.device_name(), cloned.device_name());
        assert_eq!(Arc::strong_count(&backend), 2);
    }

    #[test]
    fn cpu_backend_arc_shared_across_threads() {
        let backend: Arc<dyn Backend> = create_backend(Device::cpu()).unwrap();
        let b2 = Arc::clone(&backend);

        // Spawn a thread and use the backend there.
        let handle = std::thread::spawn(move || {
            let mut x = vec![1.0_f32, 2.0, 3.0];
            b2.softmax(&mut x).unwrap();
            x
        });

        let result = handle.join().unwrap();
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax across threads must sum to 1.0, got {sum}"
        );
    }

    // ------------------------------------------------------------------
    // Device Display
    // ------------------------------------------------------------------

    #[test]
    fn device_display_cpu() {
        let d = Device::Cpu { threads: None };
        let s = d.to_string();
        assert!(s.contains("CPU"), "CPU device display must contain 'CPU'");
    }

    #[test]
    fn device_display_gpu() {
        let d = Device::Gpu { device_id: 2 };
        let s = d.to_string();
        assert!(s.contains("GPU"), "GPU device display must contain 'GPU'");
        assert!(s.contains('2'), "GPU device display must contain device_id");
    }

    #[test]
    fn device_display_npu() {
        let d = Device::Npu { device_id: 0 };
        let s = d.to_string();
        assert!(s.contains("NPU"), "NPU device display must contain 'NPU'");
    }
}
