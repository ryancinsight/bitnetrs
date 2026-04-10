//! Quantisation primitives for BitNet b1.58.
//!
//! # Overview
//!
//! BitNet b1.58 uses a two-level quantisation scheme:
//!
//! 1. **Weight quantisation** (absmean, ternary):
//!    - Each weight matrix is quantised *once* at model-load time.
//!    - Values are stored as `i8 ∈ {−1, 0, +1}` plus a single `f32` scale.
//!    - See [`absmean`] and [`ternary::TernaryWeight`].
//!
//! 2. **Activation quantisation** (absmax, 8-bit):
//!    - Each token's hidden state is quantised *per-forward-pass* on the fly.
//!    - Values are stored transiently as `i8 ∈ [−128, 127]` plus a `f32` scale.
//!    - See [`absmax`].
//!
//! # Precision Budget
//!
//! | Component | Bits | Type | Where |
//! |-----------|------|------|-------|
//! | Weights   | 1.58 | `i8` (ternary) | Stored on disk / in memory |
//! | Activations | 8  | `i8` | Transient per token |
//! | Scales    | 32   | `f32` | One per weight matrix, one per token |
//! | Accumulator | 32 | `i32` → `f32` | Inner loop of GEMV |
//!
//! # Module Layout
//!
//! ```text
//! quant/
//! ├── mod.rs      ← this file: re-exports and top-level docs
//! ├── absmean.rs  ← absmean_quantize (f32/bf16 → ternary i8 + scale)
//! ├── absmax.rs   ← absmax_quantize_row (f32 → int8 + scale)
//! └── ternary.rs  ← TernaryWeight struct + pack/unpack helpers
//! ```

pub mod absmax;
pub mod absmean;
pub mod ternary;

// Re-export the most commonly used items at the quant:: level.
pub use absmax::{absmax_dequantize, absmax_quantize_batch, absmax_quantize_row};
pub use absmean::{absmean_dequantize, absmean_quantize, absmean_quantize_bf16};
pub use ternary::{pack_ternary, unpack_ternary, TernaryWeight};
