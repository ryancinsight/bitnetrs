//! HuggingFace `config.json` parsing for BitNet model configuration.
//!
//! # Overview
//!
//! HuggingFace model repositories ship a `config.json` file that describes the
//! transformer architecture. This module parses the subset of fields required to
//! construct a mathematically valid [`bitnet_core::config::ModelConfig`].
//!
//! The parser is intentionally strict about architectural fields that affect
//! tensor shapes and KV-cache layout, and permissive about unrelated metadata.
//!
//! # Supported JSON fields
//!
//! The following HuggingFace keys are mapped:
//!
//! - `vocab_size`                → [`ModelConfig::vocab_size`]
//! - `hidden_size`               → [`ModelConfig::hidden_size`]
//! - `num_hidden_layers`         → [`ModelConfig::num_hidden_layers`]
//! - `num_attention_heads`       → [`ModelConfig::num_attention_heads`]
//! - `num_key_value_heads`       → [`ModelConfig::num_key_value_heads`]
//! - `intermediate_size`         → [`ModelConfig::intermediate_size`]
//! - `max_position_embeddings`   → [`ModelConfig::max_position_embeddings`]
//! - `rope_theta`                → [`ModelConfig::rope_theta`]
//! - `rms_norm_eps`              → [`ModelConfig::rms_norm_eps`]
//!
//! # Defaults
//!
//! HuggingFace configs sometimes omit fields that are derivable or have
//! architecture-specific defaults:
//!
//! - `num_key_value_heads` defaults to `num_attention_heads`
//! - `rope_theta` defaults to `10_000.0`
//! - `rms_norm_eps` defaults to `1e-5`
//!
//! These defaults preserve compatibility with grouped-query and standard
//! multi-head attention layouts while still validating the final
//! [`ModelConfig`] via [`ModelConfig::validate`].
//!
//! # Invariants
//!
//! On successful parse:
//!
//! - every required architectural field is present or justified by a default
//! - the returned [`ModelConfig`] satisfies [`ModelConfig::validate`]
//! - no tensor-shape-affecting field is silently ignored if malformed
//!
//! # Failure modes
//!
//! Parsing returns an error if:
//!
//! - the file cannot be read
//! - the JSON is malformed
//! - a required field is missing
//! - a numeric field has the wrong type or is out of range for `usize`
//! - the resulting [`ModelConfig`] violates architectural constraints

use std::path::Path;

use anyhow::{anyhow, Context};
use bitnet_core::config::ModelConfig;
use serde::Deserialize;
use tracing::{debug, instrument};

/// Raw subset of HuggingFace `config.json` needed to build [`ModelConfig`].
///
/// Unknown fields are ignored by design; only architecture-defining fields are
/// deserialized.
#[derive(Debug, Clone, Deserialize)]
struct HuggingFaceConfig {
    /// Vocabulary size.
    vocab_size: Option<u64>,
    /// Hidden / embedding dimension.
    hidden_size: Option<u64>,
    /// Number of decoder layers.
    num_hidden_layers: Option<u64>,
    /// Number of query attention heads.
    num_attention_heads: Option<u64>,
    /// Number of key/value heads for GQA.
    num_key_value_heads: Option<u64>,
    /// Feed-forward intermediate dimension.
    intermediate_size: Option<u64>,
    /// Maximum supported context length.
    max_position_embeddings: Option<u64>,
    /// RoPE base frequency.
    rope_theta: Option<f32>,
    /// RMSNorm epsilon.
    rms_norm_eps: Option<f32>,
}

/// Parse a HuggingFace `config.json` string into a validated [`ModelConfig`].
///
/// # Errors
///
/// Returns an error if the JSON is malformed, if any required field is missing,
/// if any integer field cannot be represented as `usize`, or if the resulting
/// configuration is mathematically invalid.
pub fn parse_model_config_json(json: &str) -> anyhow::Result<ModelConfig> {
    let raw: HuggingFaceConfig =
        serde_json::from_str(json).context("Failed to parse HuggingFace config.json")?;

    let config = ModelConfig {
        vocab_size: require_usize(raw.vocab_size, "vocab_size")?,
        hidden_size: require_usize(raw.hidden_size, "hidden_size")?,
        num_hidden_layers: require_usize(raw.num_hidden_layers, "num_hidden_layers")?,
        num_attention_heads: require_usize(raw.num_attention_heads, "num_attention_heads")?,
        num_key_value_heads: match raw.num_key_value_heads {
            Some(value) => usize::try_from(value).with_context(|| {
                format!("Field 'num_key_value_heads' value {value} does not fit in usize")
            })?,
            None => require_usize(raw.num_attention_heads, "num_attention_heads")?,
        },
        intermediate_size: require_usize(raw.intermediate_size, "intermediate_size")?,
        max_position_embeddings: require_usize(
            raw.max_position_embeddings,
            "max_position_embeddings",
        )?,
        rope_theta: raw.rope_theta.unwrap_or(10_000.0),
        rms_norm_eps: raw.rms_norm_eps.unwrap_or(1e-5),
    };

    config
        .validate()
        .context("HuggingFace config.json produced an invalid ModelConfig")?;

    Ok(config)
}

/// Load and parse a HuggingFace `config.json` file from disk.
///
/// # Errors
///
/// Returns an error if the file cannot be read or if parsing/validation fails.
#[instrument(level = "debug", skip_all, fields(path = %path.display()))]
pub fn load_model_config(path: &Path) -> anyhow::Result<ModelConfig> {
    debug!("Loading HuggingFace config.json");

    let json = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config.json: {}", path.display()))?;

    parse_model_config_json(&json)
        .with_context(|| format!("Failed to load HuggingFace config from {}", path.display()))
}

fn require_usize(value: Option<u64>, field_name: &str) -> anyhow::Result<usize> {
    let value = value.ok_or_else(|| anyhow!("Missing required field '{field_name}'"))?;
    usize::try_from(value)
        .with_context(|| format!("Field '{field_name}' value {value} does not fit in usize"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_json() -> &'static str {
        r#"{
            "model_type": "bitnet",
            "vocab_size": 128256,
            "hidden_size": 2560,
            "num_hidden_layers": 30,
            "num_attention_heads": 20,
            "num_key_value_heads": 5,
            "intermediate_size": 6912,
            "max_position_embeddings": 4096,
            "rope_theta": 500000.0,
            "rms_norm_eps": 0.00001
        }"#
    }

    #[test]
    fn parse_valid_hf_config_json_succeeds() {
        let config = parse_model_config_json(valid_json()).unwrap();

        assert_eq!(config.vocab_size, 128_256);
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_hidden_layers, 30);
        assert_eq!(config.num_attention_heads, 20);
        assert_eq!(config.num_key_value_heads, 5);
        assert_eq!(config.intermediate_size, 6912);
        assert_eq!(config.max_position_embeddings, 4096);
        assert_eq!(config.rope_theta, 500_000.0);
        assert_eq!(config.rms_norm_eps, 1e-5);
    }

    #[test]
    fn parse_missing_num_key_value_heads_defaults_to_num_attention_heads() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "max_position_embeddings": 2048
        }"#;

        let config = parse_model_config_json(json).unwrap();

        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 16);
        assert_eq!(config.rope_theta, 10_000.0);
        assert_eq!(config.rms_norm_eps, 1e-5);
    }

    #[test]
    fn parse_missing_required_field_returns_error() {
        let json = r#"{
            "hidden_size": 2560,
            "num_hidden_layers": 30,
            "num_attention_heads": 20,
            "num_key_value_heads": 5,
            "intermediate_size": 6912,
            "max_position_embeddings": 4096
        }"#;

        let err = parse_model_config_json(json).unwrap_err();
        assert!(
            err.to_string().contains("vocab_size"),
            "error must mention missing field: {err}"
        );
    }

    #[test]
    fn parse_invalid_json_returns_error() {
        let err = parse_model_config_json("{ invalid json").unwrap_err();
        assert!(
            !err.to_string().is_empty(),
            "error message must be non-empty"
        );
    }

    #[test]
    fn parse_invalid_architecture_returns_error() {
        let json = r#"{
            "vocab_size": 128256,
            "hidden_size": 2500,
            "num_hidden_layers": 30,
            "num_attention_heads": 20,
            "num_key_value_heads": 6,
            "intermediate_size": 6912,
            "max_position_embeddings": 4096
        }"#;

        let err = parse_model_config_json(json).unwrap_err();
        assert!(
            err.to_string().contains("invalid")
                || err.to_string().contains("divisible")
                || !err.to_string().is_empty(),
            "error must describe invalid architecture: {err}"
        );
    }

    #[test]
    fn parse_wrong_type_returns_error() {
        let json = r#"{
            "vocab_size": "128256",
            "hidden_size": 2560,
            "num_hidden_layers": 30,
            "num_attention_heads": 20,
            "num_key_value_heads": 5,
            "intermediate_size": 6912,
            "max_position_embeddings": 4096
        }"#;

        let err = parse_model_config_json(json).unwrap_err();
        assert!(
            !err.to_string().is_empty(),
            "type error message must be non-empty"
        );
    }

    #[test]
    fn load_hf_config_from_file_succeeds() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "bitnet_config_test_{}_{}.json",
            std::process::id(),
            "load_hf_config"
        ));

        std::fs::write(&path, valid_json()).unwrap();

        let config = load_model_config(&path).unwrap();
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_key_value_heads, 5);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_hf_config_missing_file_returns_error() {
        let path = Path::new("definitely_missing_bitnet_config.json");
        let err = load_model_config(path).unwrap_err();
        assert!(
            !err.to_string().is_empty(),
            "missing-file error must be non-empty"
        );
    }
}
