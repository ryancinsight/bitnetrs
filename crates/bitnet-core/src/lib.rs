//! # bitnet-core
//!
//! Core types, quantisation mathematics, and backend abstractions for the
//! BitNet b1.58 inference engine.
//!
//! ## Crate Map
//!
//! ```text
//! bitnet-core/
//! ├── error.rs      — BitNetError + Result<T> alias
//! ├── config.rs     — ModelConfig, GenerationConfig, bitnet_2b_config()
//! ├── tensor/
//! │   ├── mod.rs    — Tensor<T>, TensorView<'_, T>
//! │   └── dtype.rs  — DType enum (F32 | F16 | BF16 | I8 | U8 | I2)
//! ├── quant/
//! │   ├── mod.rs    — re-exports
//! │   ├── absmean.rs — absmean_quantize (weights → ternary i8 + scale)
//! │   ├── absmax.rs  — absmax_quantize_row (activations → i8 + scale)
//! │   └── ternary.rs — TernaryWeight + pack/unpack helpers
//! └── backend/
//!     ├── mod.rs    — Backend trait, Device enum
//!     └── ops.rs    — Standalone f32 math (rms_norm, softmax, rope, …)
//! ```
//!
//! ## Dependency Policy
//!
//! `bitnet-core` has **no dependency** on any other `bitnet-*` crate.
//! All higher-level crates depend on `bitnet-core`, never the reverse.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod backend;
pub mod config;
pub mod error;
pub mod quant;
pub mod tensor;

// ---------------------------------------------------------------------------
// Top-level re-exports — the most commonly used items across the workspace
// ---------------------------------------------------------------------------

/// The unified error type for all BitNet operations.
pub use error::{BitNetError, Result};

/// Model and generation configuration.
pub use config::{bitnet_2b_config, GenerationConfig, ModelConfig};

/// Compute device selector.
pub use backend::Device;

/// The core backend trait — implemented by CPU, GPU and NPU crates.
pub use backend::Backend;

/// Ternary weight storage.
pub use quant::TernaryWeight;

/// Quantisation functions.
pub use quant::{absmax_quantize_row, absmean_quantize, absmean_quantize_bf16};

/// Standalone f32 math utilities.
pub use backend::ops::{
    apply_rope_to_head, elementwise_mul_f32, lm_head_matmul, rms_norm_f32, rope_cos_sin_table,
    softmax_f32, squared_relu_f32,
};

/// Tensor type and view.
pub use tensor::{DType, Tensor, TensorView};
