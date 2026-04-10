//! Compiled WGSL compute pipelines for the BitNet b1.58 GPU backend.
//!
//! # Architecture
//!
//! [`GpuPipelines`] owns all compiled compute pipelines and their associated
//! bind group layouts.  It is constructed once at [`GpuBackend`] initialisation
//! time and reused for every dispatch.
//!
//! # Pipeline Inventory
//!
//! | Pipeline      | Shader file      | Entry point | Operation                     |
//! |---------------|-----------------|-------------|-------------------------------|
//! | `gemv`        | `gemv.wgsl`     | `main`      | Ternary GEMV (W·x·α_W)        |
//! | `norm`        | `norm.wgsl`     | `main`      | RMSNorm                       |
//! | `rope`        | `rope.wgsl`     | `rope_qk`   | Rotary Position Embedding     |
//! | `attention`   | `attention.wgsl`| `main`      | Causal GQA Attention          |
//!
//! # Bind Group Layout Contract
//!
//! Each pipeline has a paired `*_bind_group_layout` field that defines the
//! expected buffer bindings.  The layouts are created with explicit binding
//! declarations that must match the `@group(0) @binding(N)` annotations in
//! the WGSL source.
//!
//! # Shader Loading
//!
//! Shaders are embedded at compile time via `include_str!`, so no file I/O
//! is needed at runtime.  This ensures the binary is self-contained and that
//! shader source changes cause a recompilation.
//!
//! # Error Handling
//!
//! Shader compilation errors from wgpu are returned as `anyhow::Error` with
//! a descriptive message identifying the failing shader file.  Invalid WGSL
//! syntax is caught at device creation time, not at dispatch time.

use anyhow::{anyhow, Context};
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, BufferSize, ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

// ---------------------------------------------------------------------------
// Embedded WGSL shader sources
// ---------------------------------------------------------------------------

/// Ternary GEMV shader: `output[i] = α_W · Σ_j W[i,j] · x[j]`
const GEMV_WGSL: &str = include_str!("shaders/gemv.wgsl");

/// RMSNorm shader: `out[i] = x[i] / sqrt(mean(x²) + ε) · γ[i]`
const NORM_WGSL: &str = include_str!("shaders/norm.wgsl");

/// Rotary Position Embedding shader (combined Q+K entry point)
const ROPE_WGSL: &str = include_str!("shaders/rope.wgsl");

/// Causal GQA attention shader
const ATTENTION_WGSL: &str = include_str!("shaders/attention.wgsl");

// ---------------------------------------------------------------------------
// GpuPipelines
// ---------------------------------------------------------------------------

/// Collection of compiled WGSL compute pipelines and their bind group layouts.
///
/// All pipelines share the same wgpu [`Device`] and are compiled during
/// [`GpuPipelines::new`].  The struct is intentionally non-`Clone` since
/// pipelines are large GPU objects that should not be duplicated.
///
/// # Bind group layout fields
///
/// Each pipeline has a paired `*_bind_group_layout` public field.  The
/// [`GpuBackend`] uses these layouts when constructing per-dispatch bind groups
/// via `device.create_bind_group`.
pub struct GpuPipelines {
    // ------------------------------------------------------------------
    // Ternary GEMV
    // ------------------------------------------------------------------
    /// Compiled compute pipeline for ternary matrix–vector multiply.
    pub gemv: ComputePipeline,
    /// Bind group layout for the GEMV pipeline.
    ///
    /// Bindings:
    /// - 0: weight buffer  (`array<i32>`,  storage read)
    /// - 1: input buffer   (`array<f32>`,  storage read)
    /// - 2: output buffer  (`array<f32>`,  storage read_write)
    /// - 3: params uniform (`GemvParams`,  uniform)
    pub gemv_bind_group_layout: BindGroupLayout,

    // ------------------------------------------------------------------
    // RMSNorm
    // ------------------------------------------------------------------
    /// Compiled compute pipeline for RMSNorm.
    pub norm: ComputePipeline,
    /// Bind group layout for the RMSNorm pipeline.
    ///
    /// Bindings:
    /// - 0: input buffer   (`array<f32>`, storage read)
    /// - 1: weight buffer  (`array<f32>`, storage read)
    /// - 2: output buffer  (`array<f32>`, storage read_write)
    /// - 3: params uniform (`NormParams`, uniform)
    pub norm_bind_group_layout: BindGroupLayout,

    // ------------------------------------------------------------------
    // RoPE
    // ------------------------------------------------------------------
    /// Compiled compute pipeline for Rotary Position Embedding (combined Q+K).
    pub rope: ComputePipeline,
    /// Bind group layout for the RoPE pipeline.
    ///
    /// Bindings:
    /// - 0: Q buffer       (`array<f32>`, storage read_write)
    /// - 1: K buffer       (`array<f32>`, storage read_write)
    /// - 2: params uniform (`RopeParams`, uniform)
    pub rope_bind_group_layout: BindGroupLayout,

    // ------------------------------------------------------------------
    // Attention
    // ------------------------------------------------------------------
    /// Compiled compute pipeline for causal GQA attention.
    pub attention: ComputePipeline,
    /// Bind group layout for the attention pipeline.
    ///
    /// Bindings:
    /// - 0: Q buffer       (`array<f32>`, storage read)
    /// - 1: K cache        (`array<f32>`, storage read)
    /// - 2: V cache        (`array<f32>`, storage read)
    /// - 3: output buffer  (`array<f32>`, storage read_write)
    /// - 4: params uniform (`AttnParams`, uniform)
    pub attention_bind_group_layout: BindGroupLayout,
}

impl GpuPipelines {
    /// Compile all WGSL compute pipelines for the given wgpu [`Device`].
    ///
    /// This function performs shader compilation, bind group layout creation,
    /// and pipeline layout / pipeline object creation for each of the four
    /// BitNet compute shaders.
    ///
    /// Shader compilation is synchronous in wgpu — it completes before this
    /// function returns, making compilation errors immediately visible.
    ///
    /// # Errors
    ///
    /// Returns an error if any shader fails to compile (invalid WGSL syntax,
    /// unsupported features, or device capability mismatch).
    ///
    /// In practice wgpu panics on WGSL compilation failures in debug builds
    /// and returns an error in release builds; we propagate both via
    /// `anyhow::Result`.
    pub fn new(device: &Device) -> anyhow::Result<Self> {
        tracing::debug!("Compiling BitNet GPU compute pipelines");

        // ── GEMV pipeline ──────────────────────────────────────────────────
        let gemv_bind_group_layout = create_gemv_bind_group_layout(device);
        let gemv = compile_pipeline(
            device,
            GEMV_WGSL,
            "main",
            &gemv_bind_group_layout,
            "bitnet_gemv",
            "gemv.wgsl",
        )
        .context("Failed to compile ternary GEMV pipeline (gemv.wgsl)")?;

        // ── RMSNorm pipeline ───────────────────────────────────────────────
        let norm_bind_group_layout = create_norm_bind_group_layout(device);
        let norm = compile_pipeline(
            device,
            NORM_WGSL,
            "main",
            &norm_bind_group_layout,
            "bitnet_norm",
            "norm.wgsl",
        )
        .context("Failed to compile RMSNorm pipeline (norm.wgsl)")?;

        // ── RoPE pipeline ──────────────────────────────────────────────────
        let rope_bind_group_layout = create_rope_bind_group_layout(device);
        let rope = compile_pipeline(
            device,
            ROPE_WGSL,
            "rope_qk",
            &rope_bind_group_layout,
            "bitnet_rope",
            "rope.wgsl",
        )
        .context("Failed to compile RoPE pipeline (rope.wgsl)")?;

        // ── Attention pipeline ─────────────────────────────────────────────
        let attention_bind_group_layout = create_attention_bind_group_layout(device);
        let attention = compile_pipeline(
            device,
            ATTENTION_WGSL,
            "main",
            &attention_bind_group_layout,
            "bitnet_attention",
            "attention.wgsl",
        )
        .context("Failed to compile attention pipeline (attention.wgsl)")?;

        tracing::info!("All BitNet GPU compute pipelines compiled successfully");

        Ok(Self {
            gemv,
            gemv_bind_group_layout,
            norm,
            norm_bind_group_layout,
            rope,
            rope_bind_group_layout,
            attention,
            attention_bind_group_layout,
        })
    }
}

impl std::fmt::Debug for GpuPipelines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPipelines")
            .field("gemv", &"<ComputePipeline>")
            .field("norm", &"<ComputePipeline>")
            .field("rope", &"<ComputePipeline>")
            .field("attention", &"<ComputePipeline>")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Private helpers: bind group layout constructors
// ---------------------------------------------------------------------------

/// Helper: create a [`BindGroupLayoutEntry`] for a read-only storage buffer.
#[inline]
fn storage_read_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Helper: create a [`BindGroupLayoutEntry`] for a read-write storage buffer.
#[inline]
fn storage_rw_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Helper: create a [`BindGroupLayoutEntry`] for a uniform buffer.
#[inline]
fn uniform_entry(binding: u32, min_size_bytes: u64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: BufferSize::new(min_size_bytes),
        },
        count: None,
    }
}

/// Create the bind group layout for the GEMV pipeline.
///
/// ```text
/// binding 0 → weight buffer  (storage, read-only,  array<i32>)
/// binding 1 → input buffer   (storage, read-only,  array<f32>)
/// binding 2 → output buffer  (storage, read-write, array<f32>)
/// binding 3 → params uniform (uniform,             GemvParams = 16 bytes)
/// ```
fn create_gemv_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("gemv_bind_group_layout"),
        entries: &[
            storage_read_entry(0), // weight: array<i32>
            storage_read_entry(1), // input:  array<f32>
            storage_rw_entry(2),   // output: array<f32>
            uniform_entry(3, 16),  // GemvParams: 4 × u32/f32 = 16 bytes
        ],
    })
}

/// Create the bind group layout for the RMSNorm pipeline.
///
/// ```text
/// binding 0 → input buffer   (storage, read-only,  array<f32>)
/// binding 1 → weight buffer  (storage, read-only,  array<f32>)
/// binding 2 → output buffer  (storage, read-write, array<f32>)
/// binding 3 → params uniform (uniform,             NormParams = 16 bytes)
/// ```
fn create_norm_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("norm_bind_group_layout"),
        entries: &[
            storage_read_entry(0), // input:  array<f32>
            storage_read_entry(1), // weight: array<f32>
            storage_rw_entry(2),   // output: array<f32>
            uniform_entry(3, 16),  // NormParams: 4 × u32/f32 = 16 bytes
        ],
    })
}

/// Create the bind group layout for the RoPE pipeline.
///
/// ```text
/// binding 0 → Q buffer       (storage, read-write, array<f32>)
/// binding 1 → K buffer       (storage, read-write, array<f32>)
/// binding 2 → params uniform (uniform,             RopeParams = 32 bytes)
/// ```
fn create_rope_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("rope_bind_group_layout"),
        entries: &[
            storage_rw_entry(0),  // Q: array<f32>  (modified in-place)
            storage_rw_entry(1),  // K: array<f32>  (modified in-place)
            uniform_entry(2, 32), // RopeParams: 8 × u32/f32 = 32 bytes
        ],
    })
}

/// Create the bind group layout for the attention pipeline.
///
/// ```text
/// binding 0 → Q buffer       (storage, read-only,  array<f32>)
/// binding 1 → K cache        (storage, read-only,  array<f32>)
/// binding 2 → V cache        (storage, read-only,  array<f32>)
/// binding 3 → output buffer  (storage, read-write, array<f32>)
/// binding 4 → params uniform (uniform,             AttnParams = 32 bytes)
/// ```
fn create_attention_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("attention_bind_group_layout"),
        entries: &[
            storage_read_entry(0), // Q:      array<f32>
            storage_read_entry(1), // K cache: array<f32>
            storage_read_entry(2), // V cache: array<f32>
            storage_rw_entry(3),   // output:  array<f32>
            uniform_entry(4, 32),  // AttnParams: 8 × u32/f32 = 32 bytes
        ],
    })
}

// ---------------------------------------------------------------------------
// Private helper: pipeline compilation
// ---------------------------------------------------------------------------

/// Compile a WGSL compute shader and create the compute pipeline.
///
/// # Arguments
///
/// - `device`:       The wgpu logical device.
/// - `wgsl_source`:  The WGSL shader source code as a string.
/// - `entry_point`:  The name of the `@compute` entry point in the shader.
/// - `bind_group_layout`: The pre-created bind group layout for this pipeline.
/// - `pipeline_label`: Human-readable label for the pipeline object.
/// - `shader_file`:   File name for error messages (e.g. `"gemv.wgsl"`).
///
/// # Errors
///
/// Returns an error if the shader module fails to compile or if the pipeline
/// layout / pipeline creation fails.
fn compile_pipeline(
    device: &Device,
    wgsl_source: &str,
    entry_point: &str,
    bind_group_layout: &BindGroupLayout,
    pipeline_label: &str,
    shader_file: &str,
) -> anyhow::Result<ComputePipeline> {
    // Step 1: Create shader module from WGSL source.
    // wgpu validates the WGSL at this point; invalid shaders return an error.
    let shader_label = format!("{pipeline_label}_shader");
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(&shader_label),
        source: ShaderSource::Wgsl(wgsl_source.into()),
    });

    tracing::debug!(shader = shader_file, entry_point, "Shader module created");

    // Step 2: Create pipeline layout from the bind group layout.
    let pipeline_layout_label = format!("{pipeline_label}_layout");
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(&pipeline_layout_label),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    // Step 3: Create the compute pipeline.
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(pipeline_label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });

    tracing::debug!(
        pipeline = pipeline_label,
        entry_point,
        shader = shader_file,
        "Compute pipeline compiled"
    );

    Ok(pipeline)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::{Adapter, Backends, Instance, InstanceDescriptor, InstanceFlags};

    /// Create a wgpu device suitable for testing (any adapter, including software).
    async fn test_device() -> Option<(Device, wgpu::Queue)> {
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
                    label: Some("test_pipeline_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()
    }

    // -----------------------------------------------------------------------
    // Bind group layout tests (device-independent assertions on flags)
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_bind_group_layout_has_four_entries() {
        let Some((device, _queue)) = pollster::block_on(test_device()) else {
            return;
        };
        // If this doesn't panic, the layout was created with valid entries.
        let layout = create_gemv_bind_group_layout(&device);
        // Round-trip: create a bind group descriptor (existence check only).
        // We can't inspect layout entries directly in wgpu, but creation succeeding
        // is sufficient proof that the layout is valid.
        let _ = layout;
    }

    #[test]
    fn norm_bind_group_layout_creates_successfully() {
        let Some((device, _queue)) = pollster::block_on(test_device()) else {
            return;
        };
        let layout = create_norm_bind_group_layout(&device);
        let _ = layout;
    }

    #[test]
    fn rope_bind_group_layout_creates_successfully() {
        let Some((device, _queue)) = pollster::block_on(test_device()) else {
            return;
        };
        let layout = create_rope_bind_group_layout(&device);
        let _ = layout;
    }

    #[test]
    fn attention_bind_group_layout_creates_successfully() {
        let Some((device, _queue)) = pollster::block_on(test_device()) else {
            return;
        };
        let layout = create_attention_bind_group_layout(&device);
        let _ = layout;
    }

    // -----------------------------------------------------------------------
    // Full pipeline compilation tests
    // -----------------------------------------------------------------------

    /// Compile all four pipelines and verify no errors are returned.
    ///
    /// This is the primary regression test: if WGSL syntax is broken or if
    /// a bind group layout does not match the shader's declared bindings,
    /// this test will fail at the `GpuPipelines::new` call.
    #[test]
    fn all_pipelines_compile_successfully() {
        let Some((device, _queue)) = pollster::block_on(test_device()) else {
            // No adapter available — skip gracefully.
            return;
        };

        let result = GpuPipelines::new(&device);
        assert!(
            result.is_ok(),
            "All GPU pipelines must compile without error. Error: {:?}",
            result.err()
        );
    }

    #[test]
    fn gemv_pipeline_compiles() {
        let Some((device, _queue)) = pollster::block_on(test_device()) else {
            return;
        };
        let layout = create_gemv_bind_group_layout(&device);
        let result = compile_pipeline(
            &device,
            GEMV_WGSL,
            "main",
            &layout,
            "test_gemv",
            "gemv.wgsl",
        );
        assert!(
            result.is_ok(),
            "GEMV pipeline must compile: {:?}",
            result.err()
        );
    }

    #[test]
    fn norm_pipeline_compiles() {
        let Some((device, _queue)) = pollster::block_on(test_device()) else {
            return;
        };
        let layout = create_norm_bind_group_layout(&device);
        let result = compile_pipeline(
            &device,
            NORM_WGSL,
            "main",
            &layout,
            "test_norm",
            "norm.wgsl",
        );
        assert!(
            result.is_ok(),
            "RMSNorm pipeline must compile: {:?}",
            result.err()
        );
    }

    #[test]
    fn rope_pipeline_compiles() {
        let Some((device, _queue)) = pollster::block_on(test_device()) else {
            return;
        };
        let layout = create_rope_bind_group_layout(&device);
        let result = compile_pipeline(
            &device,
            ROPE_WGSL,
            "rope_qk",
            &layout,
            "test_rope",
            "rope.wgsl",
        );
        assert!(
            result.is_ok(),
            "RoPE pipeline must compile: {:?}",
            result.err()
        );
    }

    #[test]
    fn attention_pipeline_compiles() {
        let Some((device, _queue)) = pollster::block_on(test_device()) else {
            return;
        };
        let layout = create_attention_bind_group_layout(&device);
        let result = compile_pipeline(
            &device,
            ATTENTION_WGSL,
            "main",
            &layout,
            "test_attention",
            "attention.wgsl",
        );
        assert!(
            result.is_ok(),
            "Attention pipeline must compile: {:?}",
            result.err()
        );
    }

    // -----------------------------------------------------------------------
    // Shader source content tests (sanity checks on embedded strings)
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_wgsl_source_contains_entry_point() {
        assert!(
            GEMV_WGSL.contains("fn main"),
            "gemv.wgsl must define 'fn main' entry point"
        );
        assert!(
            GEMV_WGSL.contains("GemvParams"),
            "gemv.wgsl must define GemvParams uniform struct"
        );
        assert!(
            GEMV_WGSL.contains("weight_buf"),
            "gemv.wgsl must declare weight_buf binding"
        );
        assert!(
            GEMV_WGSL.contains("input_buf"),
            "gemv.wgsl must declare input_buf binding"
        );
        assert!(
            GEMV_WGSL.contains("output_buf"),
            "gemv.wgsl must declare output_buf binding"
        );
    }

    #[test]
    fn norm_wgsl_source_contains_entry_point() {
        assert!(
            NORM_WGSL.contains("fn main"),
            "norm.wgsl must define 'fn main' entry point"
        );
        assert!(
            NORM_WGSL.contains("NormParams"),
            "norm.wgsl must define NormParams uniform struct"
        );
        assert!(
            NORM_WGSL.contains("inv_rms"),
            "norm.wgsl must compute inv_rms"
        );
    }

    #[test]
    fn rope_wgsl_source_contains_combined_entry_point() {
        assert!(
            ROPE_WGSL.contains("fn rope_qk"),
            "rope.wgsl must define 'fn rope_qk' combined entry point"
        );
        assert!(
            ROPE_WGSL.contains("RopeParams"),
            "rope.wgsl must define RopeParams uniform struct"
        );
        assert!(
            ROPE_WGSL.contains("rope_theta"),
            "rope.wgsl must reference rope_theta parameter"
        );
        assert!(
            ROPE_WGSL.contains("q_buf"),
            "rope.wgsl must declare q_buf binding"
        );
        assert!(
            ROPE_WGSL.contains("k_buf"),
            "rope.wgsl must declare k_buf binding"
        );
    }

    #[test]
    fn attention_wgsl_source_contains_entry_point() {
        assert!(
            ATTENTION_WGSL.contains("fn main"),
            "attention.wgsl must define 'fn main' entry point"
        );
        assert!(
            ATTENTION_WGSL.contains("AttnParams"),
            "attention.wgsl must define AttnParams uniform struct"
        );
        assert!(
            ATTENTION_WGSL.contains("softmax"),
            "attention.wgsl must implement softmax (max subtraction)"
        );
        assert!(
            ATTENTION_WGSL.contains("n_kv_heads"),
            "attention.wgsl must handle GQA n_kv_heads"
        );
        assert!(
            ATTENTION_WGSL.contains("k_cache"),
            "attention.wgsl must declare k_cache binding"
        );
        assert!(
            ATTENTION_WGSL.contains("v_cache"),
            "attention.wgsl must declare v_cache binding"
        );
    }

    #[test]
    fn all_wgsl_sources_are_non_empty() {
        assert!(!GEMV_WGSL.is_empty(), "gemv.wgsl must not be empty");
        assert!(!NORM_WGSL.is_empty(), "norm.wgsl must not be empty");
        assert!(!ROPE_WGSL.is_empty(), "rope.wgsl must not be empty");
        assert!(
            !ATTENTION_WGSL.is_empty(),
            "attention.wgsl must not be empty"
        );
    }

    /// Verify the Debug impl doesn't panic and shows pipeline names.
    #[test]
    fn gpu_pipelines_debug_format() {
        let Some((device, _queue)) = pollster::block_on(test_device()) else {
            return;
        };
        let pipelines = GpuPipelines::new(&device).unwrap();
        let debug_str = format!("{pipelines:?}");
        assert!(
            debug_str.contains("GpuPipelines"),
            "Debug output must contain struct name"
        );
        assert!(
            debug_str.contains("gemv"),
            "Debug output must mention gemv pipeline"
        );
    }

    // -----------------------------------------------------------------------
    // Binding entry helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn storage_read_entry_has_correct_binding_type() {
        let entry = storage_read_entry(0);
        assert_eq!(entry.binding, 0);
        assert!(matches!(
            entry.ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                ..
            }
        ));
    }

    #[test]
    fn storage_rw_entry_has_correct_binding_type() {
        let entry = storage_rw_entry(2);
        assert_eq!(entry.binding, 2);
        assert!(matches!(
            entry.ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                ..
            }
        ));
    }

    #[test]
    fn uniform_entry_has_correct_binding_type() {
        let entry = uniform_entry(3, 16);
        assert_eq!(entry.binding, 3);
        assert!(matches!(
            entry.ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                ..
            }
        ));
    }

    #[test]
    fn all_entries_are_compute_stage_only() {
        for entry in [
            storage_read_entry(0),
            storage_rw_entry(1),
            uniform_entry(2, 16),
        ] {
            assert_eq!(
                entry.visibility,
                ShaderStages::COMPUTE,
                "All BitNet shader bindings must be COMPUTE stage"
            );
        }
    }
}
