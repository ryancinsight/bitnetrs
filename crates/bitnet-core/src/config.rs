//! Model and generation configuration for BitNet b1.58.
//!
//! # Mathematical Invariants
//!
//! For a valid [`ModelConfig`] the following must hold:
//! - `hidden_size % num_attention_heads == 0`  (head_dim is integral)
//! - `num_attention_heads % num_key_value_heads == 0`  (GQA group size is integral)
//! - `head_dim = hidden_size / num_attention_heads`
//! - `heads_per_group = num_attention_heads / num_key_value_heads`
//!
//! The canonical 2B model satisfies: 2560/20 = 128 (head_dim),
//! 20/5 = 4 (heads_per_group), and logical FFN width = 6912.

use serde::{Deserialize, Serialize};

use crate::error::{BitNetError, Result};

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------

/// Complete configuration for a BitNet b1.58 transformer model.
///
/// All dimension relationships are validated by [`ModelConfig::validate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size — number of discrete token IDs.
    /// LLaMA 3 tokenizer: 128 256.
    pub vocab_size: usize,

    /// Hidden / embedding dimension `d_model`.
    /// 2B model: 2560.
    pub hidden_size: usize,

    /// Number of transformer decoder layers.
    /// 2B model: 30.
    pub num_hidden_layers: usize,

    /// Total number of query attention heads `H`.
    /// 2B model: 20.
    pub num_attention_heads: usize,

    /// Number of key/value heads `H_kv` (Grouped Query Attention).
    /// Each KV head serves `num_attention_heads / num_key_value_heads` query heads.
    /// 2B model: 5.
    pub num_key_value_heads: usize,

    /// Feed-forward intermediate dimension.
    /// 2B model: 6912.
    pub intermediate_size: usize,

    /// Maximum sequence length (context window).
    /// 2B model: 4096.
    pub max_position_embeddings: usize,

    /// RoPE base frequency θ.
    /// 2B model: 500 000.0 (extended context variant).
    pub rope_theta: f32,

    /// ε for RMSNorm numerical stability.
    /// 2B model: 1e-5.
    pub rms_norm_eps: f32,
}

impl ModelConfig {
    /// Returns the per-head dimension: `hidden_size / num_attention_heads`.
    ///
    /// # Panics (debug only)
    /// Panics in debug builds if `hidden_size % num_attention_heads != 0`.
    #[inline]
    pub fn head_dim(&self) -> usize {
        debug_assert_eq!(
            self.hidden_size % self.num_attention_heads,
            0,
            "hidden_size must be divisible by num_attention_heads"
        );
        self.hidden_size / self.num_attention_heads
    }

    /// Number of query heads served by each KV head (GQA group size).
    ///
    /// # Panics (debug only)
    /// Panics in debug builds if `num_attention_heads % num_key_value_heads != 0`.
    #[inline]
    pub fn heads_per_group(&self) -> usize {
        debug_assert_eq!(
            self.num_attention_heads % self.num_key_value_heads,
            0,
            "num_attention_heads must be divisible by num_key_value_heads"
        );
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Total dimension of the concatenated query projection output.
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim()
    }

    /// Total dimension of the key (or value) projection output.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    /// Validate all dimension relationships.
    ///
    /// Returns `Ok(())` if the configuration is mathematically consistent, or a
    /// descriptive [`BitNetError::InvalidConfig`] otherwise.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(BitNetError::config("hidden_size must be > 0"));
        }
        if self.num_attention_heads == 0 {
            return Err(BitNetError::config("num_attention_heads must be > 0"));
        }
        if self.num_key_value_heads == 0 {
            return Err(BitNetError::config("num_key_value_heads must be > 0"));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(BitNetError::config(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(BitNetError::config(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.vocab_size == 0 {
            return Err(BitNetError::config("vocab_size must be > 0"));
        }
        if self.num_hidden_layers == 0 {
            return Err(BitNetError::config("num_hidden_layers must be > 0"));
        }
        if self.intermediate_size == 0 {
            return Err(BitNetError::config("intermediate_size must be > 0"));
        }
        if self.max_position_embeddings == 0 {
            return Err(BitNetError::config("max_position_embeddings must be > 0"));
        }
        if self.rope_theta <= 0.0 {
            return Err(BitNetError::config("rope_theta must be > 0"));
        }
        if self.rms_norm_eps <= 0.0 {
            return Err(BitNetError::config("rms_norm_eps must be > 0"));
        }
        Ok(())
    }
}

/// Canonical configuration for the BitNet b1.58 2B-4T model.
///
/// Source: the published HuggingFace `config.json`.
/// - hidden_size = 2560, n_layers = 30, n_heads = 20, n_kv_heads = 5
/// - vocab_size = 128 256 (LLaMA 3 tokenizer)
/// - logical FFN intermediate size = 6912
/// - rope_theta = 500 000.0, norm_eps = 1e-5
pub fn bitnet_2b_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 128_256,
        hidden_size: 2560,
        num_hidden_layers: 30,
        num_attention_heads: 20,
        num_key_value_heads: 5,
        intermediate_size: 6912,
        max_position_embeddings: 4096,
        rope_theta: 500_000.0,
        rms_norm_eps: 1e-5,
    }
}

// ---------------------------------------------------------------------------
// GenerationConfig
// ---------------------------------------------------------------------------

/// Hyper-parameters that govern token sampling during autoregressive generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Softmax temperature τ.
    ///
    /// `τ = 1.0` leaves the distribution unchanged.
    /// `τ < 1.0` sharpens the distribution (more deterministic).
    /// `τ → 0`  approaches greedy decoding.
    ///
    /// Applied as `logits[i] /= τ` before softmax.
    pub temperature: f32,

    /// Nucleus (top-p) sampling threshold p ∈ (0, 1].
    ///
    /// Retain the smallest set of tokens whose cumulative probability ≥ p,
    /// then re-normalise and sample.  `p = 1.0` disables nucleus filtering.
    pub top_p: f32,

    /// Top-k vocabulary truncation.
    ///
    /// Keep only the k highest-probability tokens before sampling.
    /// `0` disables top-k filtering.
    pub top_k: usize,

    /// Repetition penalty α ≥ 1.0.
    ///
    /// Penalises tokens that have already appeared in the context:
    /// `logit[t] /= α` if token `t` was generated before.
    /// `1.0` means no penalty.
    pub repetition_penalty: f32,

    /// Maximum number of new tokens to generate (excluding the prompt).
    pub max_new_tokens: usize,

    /// Seed for the pseudo-random number generator used during sampling.
    /// Deterministic for a fixed seed + prompt + config combination.
    pub seed: u64,
}

impl GenerationConfig {
    /// Greedy decoding — always pick the highest-probability token.
    pub fn greedy() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 1.0,
            max_new_tokens: 256,
            seed: 42,
        }
    }

    /// Balanced defaults suitable for chat / instruction following.
    pub fn chat_defaults() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            max_new_tokens: 512,
            seed: 0,
        }
    }

    /// Creative / high-diversity sampling.
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 0,
            repetition_penalty: 1.05,
            max_new_tokens: 1024,
            seed: 0,
        }
    }

    /// Validate the generation configuration.
    pub fn validate(&self) -> Result<()> {
        if self.temperature <= 0.0 {
            return Err(BitNetError::config("temperature must be > 0"));
        }
        if !(0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(BitNetError::config("top_p must be in (0, 1]"));
        }
        if self.repetition_penalty < 1.0 {
            return Err(BitNetError::config("repetition_penalty must be >= 1.0"));
        }
        if self.max_new_tokens == 0 {
            return Err(BitNetError::config("max_new_tokens must be > 0"));
        }
        Ok(())
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self::chat_defaults()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitnet_2b_config_is_valid() {
        let cfg = bitnet_2b_config();
        cfg.validate().expect("canonical 2B config must be valid");
    }

    #[test]
    fn bitnet_2b_head_dim_is_128() {
        let cfg = bitnet_2b_config();
        assert_eq!(cfg.head_dim(), 128, "head_dim = 2560/20 = 128");
    }

    #[test]
    fn bitnet_2b_heads_per_group_is_4() {
        let cfg = bitnet_2b_config();
        assert_eq!(
            cfg.heads_per_group(),
            4,
            "20 query heads / 5 kv heads = 4 per group"
        );
    }

    #[test]
    fn bitnet_2b_kv_dim_is_640() {
        let cfg = bitnet_2b_config();
        assert_eq!(cfg.kv_dim(), 640, "5 kv_heads * 128 head_dim = 640");
    }

    #[test]
    fn bitnet_2b_intermediate_size_is_6912() {
        let cfg = bitnet_2b_config();
        assert_eq!(
            cfg.intermediate_size, 6912,
            "logical FFN intermediate size must remain 6912"
        );
    }

    #[test]
    fn invalid_head_divisibility_rejected() {
        let mut cfg = bitnet_2b_config();
        cfg.num_attention_heads = 7; // 2560 % 7 ≠ 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_kv_head_divisibility_rejected() {
        let mut cfg = bitnet_2b_config();
        cfg.num_key_value_heads = 3; // 20 % 3 ≠ 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn generation_config_defaults_valid() {
        GenerationConfig::default()
            .validate()
            .expect("default GenerationConfig must be valid");
    }

    #[test]
    fn generation_config_greedy_valid() {
        GenerationConfig::greedy()
            .validate()
            .expect("greedy GenerationConfig must be valid");
    }

    #[test]
    fn generation_config_zero_temperature_rejected() {
        let mut cfg = GenerationConfig::default();
        cfg.temperature = 0.0;
        assert!(cfg.validate().is_err());
    }
}
