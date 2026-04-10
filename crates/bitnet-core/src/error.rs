//! Error types for the BitNet inference engine.
//!
//! # Invariant
//! Every public API in this workspace returns `Result<T, BitNetError>` (aliased as
//! `crate::Result<T>`).  Panics are reserved for provably-unreachable states that
//! indicate programmer error, not runtime failures.

use thiserror::Error;

/// Unified error type for all BitNet operations.
#[derive(Debug, Error)]
pub enum BitNetError {
    /// A tensor shape was incompatible with the requested operation.
    ///
    /// Invariant: `expected` and `got` are always non-empty strings describing
    /// the shape in `[d0, d1, …]` notation.
    #[error("invalid tensor shape: expected {expected}, got {got}")]
    InvalidShape { expected: String, got: String },

    /// A quantization operation received out-of-range values or an empty slice.
    #[error("quantization error: {0}")]
    QuantizationError(String),

    /// A backend (CPU / GPU / NPU) reported a compute failure.
    #[error("backend error ({backend}): {message}")]
    BackendError { backend: String, message: String },

    /// Weight loading or format parsing failed.
    #[error("weight load error: {0}")]
    WeightLoadError(String),

    /// Tokenizer initialisation or encode/decode failed.
    #[error("tokenizer error: {0}")]
    TokenizerError(String),

    /// Wraps a standard I/O error (file not found, permission denied, …).
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// The requested compute device is not available on this machine.
    #[error("unsupported device: {0}")]
    UnsupportedDevice(String),

    /// An HTTP / network error occurred while downloading model files.
    #[error("network error: {0}")]
    NetworkError(String),

    /// A JSON or safetensors metadata parse error.
    #[error("parse error: {0}")]
    ParseError(String),

    /// The model configuration is self-contradictory (e.g. head_dim * n_heads ≠ hidden_size).
    #[error("invalid model configuration: {0}")]
    InvalidConfig(String),
}

/// Convenience alias — every fallible function in this workspace returns this.
pub type Result<T> = std::result::Result<T, BitNetError>;

// ---------------------------------------------------------------------------
// Constructor helpers
// ---------------------------------------------------------------------------

impl BitNetError {
    /// Construct a [`BitNetError::BackendError`] with the given backend name and message.
    #[inline]
    pub fn backend(backend: impl Into<String>, message: impl Into<String>) -> Self {
        Self::BackendError {
            backend: backend.into(),
            message: message.into(),
        }
    }

    /// Construct a [`BitNetError::InvalidShape`] from expected and actual shape descriptions.
    #[inline]
    pub fn shape(expected: impl Into<String>, got: impl Into<String>) -> Self {
        Self::InvalidShape {
            expected: expected.into(),
            got: got.into(),
        }
    }

    /// Construct a [`BitNetError::WeightLoadError`].
    #[inline]
    pub fn weight(msg: impl Into<String>) -> Self {
        Self::WeightLoadError(msg.into())
    }

    /// Construct a [`BitNetError::QuantizationError`].
    #[inline]
    pub fn quant(msg: impl Into<String>) -> Self {
        Self::QuantizationError(msg.into())
    }

    /// Construct a [`BitNetError::InvalidConfig`].
    #[inline]
    pub fn config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }
}
