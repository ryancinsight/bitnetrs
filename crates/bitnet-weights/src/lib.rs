//! # bitnet-weights
//!
//! Weight loading for BitNet b1.58 inference.
//!
//! ## Overview
//!
//! This crate provides the complete pipeline for obtaining model weights:
//!
//! 1. **Download** from HuggingFace Hub via [`hf_hub::download_model_from_hf`]
//! 2. **Load** packed deployment weights via [`load_weights_from_packed`] for
//!    the default inference path
//! 3. **Optionally parse** BF16 safetensors files via
//!    [`safetensors::load_bf16_safetensors`] for master-weight loading
//! 4. **Assemble** into [`ModelWeights`] with the correct layer structure
//!
//! ## Supported Model
//!
//! Primarily targets the official HuggingFace BitNet repositories:
//! - `microsoft/bitnet-b1.58-2B-4T` for packed deployment inference weights
//! - `microsoft/bitnet-b1.58-2B-4T-bf16` for BF16 master weights
//!
//! ## Weight Name Mapping
//!
//! HuggingFace safetensors keys → [`ModelWeights`] fields:
//!
//! ```text
//! model.embed_tokens.weight                          → embed_tokens
//! model.layers.{i}.self_attn.q_proj.weight           → layers[i].q_proj     (quantised)
//! model.layers.{i}.self_attn.k_proj.weight           → layers[i].k_proj     (quantised)
//! model.layers.{i}.self_attn.v_proj.weight           → layers[i].v_proj     (quantised)
//! model.layers.{i}.self_attn.o_proj.weight           → layers[i].o_proj     (quantised)
//! model.layers.{i}.self_attn.attn_sub_norm.weight    → layers[i].attn_sub_norm
//! model.layers.{i}.mlp.gate_proj.weight              → layers[i].gate_proj  (quantised)
//! model.layers.{i}.mlp.up_proj.weight                → layers[i].up_proj    (quantised)
//! model.layers.{i}.mlp.down_proj.weight              → layers[i].down_proj  (quantised)
//! model.layers.{i}.mlp.ffn_sub_norm.weight           → layers[i].ffn_sub_norm
//! model.layers.{i}.input_layernorm.weight            → layers[i].attention_norm
//! model.layers.{i}.post_attention_layernorm.weight   → layers[i].ffn_norm
//! model.norm.weight                                  → final_norm
//! ```
//!
//! `lm_head` shares weights with `embed_tokens` (weight tying).
//!
//! ## Module Layout
//!
//! ```text
//! bitnet-weights/
//! ├── lib.rs          ← this file: public API, ModelWeights, LayerWeights
//! ├── config.rs       ← config.json parsing into ModelConfig
//! ├── safetensors.rs  ← safetensors file parsing and BF16→f32 conversion
//! ├── hf_hub.rs       ← async HuggingFace Hub HTTP download with progress bars
//! └── loader.rs       ← weight name mapping + absmean quantisation
//! ```

pub mod config;
pub mod hf_hub;
pub mod loader;
pub mod safetensors;

pub use config::{load_model_config, parse_model_config_json};
pub use loader::{load_weights_from_bf16, load_weights_from_packed, LayerWeights, ModelWeights};

// ---------------------------------------------------------------------------
// Re-export the HF Hub downloader at the crate root for convenience.
// ---------------------------------------------------------------------------

pub use hf_hub::download_model_from_hf;

// ---------------------------------------------------------------------------
// Model file constants
// ---------------------------------------------------------------------------

/// The primary packed 1.58-bit weights repository on HuggingFace.
///
/// This is the **deployment** repository: weights are stored as U8 packed
/// ternary (2 bits per value, 4 values per byte) plus BF16 per-tensor scales.
/// Use [`load_weights_from_packed`] to load from this repository.
pub const HF_REPO_PACKED: &str = "microsoft/bitnet-b1.58-2B-4T";

/// The BF16 master-weights repository on HuggingFace.
///
/// Contains full-precision BF16 weights.  Useful for fine-tuning or when you
/// want to quantise on-the-fly via [`load_weights_from_bf16`].
pub const HF_REPO_BF16: &str = "microsoft/bitnet-b1.58-2B-4T-bf16";

/// The GGUF quantised repository for use with the official C++ runtime
/// (`bitnet.cpp`).  Not used by this Rust implementation.
pub const HF_REPO_GGUF: &str = "microsoft/BitNet-b1.58-2B-4T-gguf";

/// Default safetensors filename in the BF16 repository.
pub const SAFETENSORS_FILENAME: &str = "model.safetensors";

/// Default config filename.
///
/// Parse the file with [`parse_model_config_json`] or load it from disk with
/// [`load_model_config`].
pub const CONFIG_FILENAME: &str = "config.json";

/// Default local cache directory (relative to the user's home directory).
pub const DEFAULT_CACHE_SUBDIR: &str = ".cache/bitnet";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_repo_packed_is_correct() {
        assert_eq!(HF_REPO_PACKED, "microsoft/bitnet-b1.58-2B-4T");
    }

    #[test]
    fn hf_repo_bf16_is_correct() {
        assert_eq!(HF_REPO_BF16, "microsoft/bitnet-b1.58-2B-4T-bf16");
    }

    #[test]
    fn safetensors_filename_is_correct() {
        assert_eq!(SAFETENSORS_FILENAME, "model.safetensors");
    }

    #[test]
    fn default_cache_subdir_is_non_empty() {
        assert!(!DEFAULT_CACHE_SUBDIR.is_empty());
    }

    #[test]
    fn config_filename_is_correct() {
        assert_eq!(CONFIG_FILENAME, "config.json");
    }
}
