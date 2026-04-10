//! Data-type enumeration for BitNet b1.58 tensors.
//!
//! # Design
//!
//! [`DType`] is a lightweight descriptor that carries no data — it simply
//! identifies the numeric type stored in a buffer.  It is used in the weight
//! loader to dispatch the correct conversion path when reading safetensors
//! files, and in the CLI to report the precision of loaded tensors.
//!
//! # Byte-width convention
//!
//! | Variant  | Bits | Bytes | Notes                              |
//! |----------|------|-------|------------------------------------|
//! | `F32`    |  32  |   4   | Full-precision float                |
//! | `F16`    |  16  |   2   | IEEE 754 half-precision float       |
//! | `BF16`   |  16  |   2   | Brain float (8-bit exp, 7-bit mant) |
//! | `I8`     |   8  |   1   | Signed 8-bit integer (activations)  |
//! | `U8`     |   8  |   1   | Unsigned 8-bit (packed ternary)     |
//! | `I2`     |   2  |  N/A  | 2-bit packed ternary (logical only) |

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// DType
// ---------------------------------------------------------------------------

/// Numeric element type of a tensor buffer.
///
/// This enum is a *descriptor*, not a value — it does not carry generic
/// parameters.  Use it to dispatch conversion code at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit IEEE 754 single-precision float.
    F32,

    /// 16-bit IEEE 754 half-precision float (`half::f16`).
    F16,

    /// 16-bit Brain Float (`half::bf16`) — 8-bit exponent, 7-bit mantissa.
    ///
    /// Used for the weights in the HuggingFace `bitnet-b1.58-2B-4T-bf16`
    /// safetensors checkpoint.
    BF16,

    /// Signed 8-bit integer.
    ///
    /// Used for transient quantised activations `x_q ∈ [−128, 127]` and for
    /// the unpackaged ternary weight representation `w_q ∈ {−1, 0, +1}`.
    I8,

    /// Unsigned 8-bit integer.
    ///
    /// Used for packed 2-bit-per-value ternary weight storage:
    /// four ternary values packed into one `u8` byte.
    U8,

    /// Logical 2-bit ternary.  Not a real in-memory type; each ternary value
    /// is stored as a 2-bit field inside a [`DType::U8`] byte.
    ///
    /// This variant is used purely for documentation / size-calculation
    /// purposes and is never directly allocated.
    I2,
}

impl DType {
    /// Number of bytes per element, or `None` for [`DType::I2`] (sub-byte).
    ///
    /// | Variant | Returns |
    /// |---------|---------|
    /// | F32     | Some(4) |
    /// | F16     | Some(2) |
    /// | BF16    | Some(2) |
    /// | I8      | Some(1) |
    /// | U8      | Some(1) |
    /// | I2      | None    |
    #[inline]
    pub const fn byte_size(self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 => Some(2),
            Self::BF16 => Some(2),
            Self::I8 => Some(1),
            Self::U8 => Some(1),
            Self::I2 => None,
        }
    }

    /// Number of bits per element.
    ///
    /// For [`DType::I2`] this returns `2` (the logical bit-width).
    #[inline]
    pub const fn bit_size(self) -> usize {
        match self {
            Self::F32 => 32,
            Self::F16 => 16,
            Self::BF16 => 16,
            Self::I8 => 8,
            Self::U8 => 8,
            Self::I2 => 2,
        }
    }

    /// Returns `true` if this is a floating-point type (`F32`, `F16`, `BF16`).
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }

    /// Returns `true` if this is an integer type (`I8`, `U8`, `I2`).
    #[inline]
    pub const fn is_integer(self) -> bool {
        !self.is_float()
    }

    /// Returns `true` if this type has sub-byte precision ([`DType::I2`]).
    #[inline]
    pub const fn is_sub_byte(self) -> bool {
        matches!(self, Self::I2)
    }

    /// Returns `true` if elements of this type are stored as signed values.
    ///
    /// `F32`, `F16`, `BF16`, and `I8` are signed; `U8` and `I2` are unsigned.
    #[inline]
    pub const fn is_signed(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::I8)
    }

    /// Human-readable string identifier (matches the safetensors dtype string).
    ///
    /// These strings are used when parsing / writing safetensors metadata.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::I2 => "I2",
        }
    }

    /// Parse a dtype string as returned by the safetensors file format.
    ///
    /// Accepts both upper-case (`"F32"`) and lower-case (`"f32"`) variants.
    ///
    /// # Errors
    /// Returns `None` if the string is not a known dtype.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "F32" | "FLOAT32" => Some(Self::F32),
            "F16" | "FLOAT16" | "HALF" => Some(Self::F16),
            "BF16" | "BFLOAT16" => Some(Self::BF16),
            "I8" | "INT8" => Some(Self::I8),
            "U8" | "UINT8" => Some(Self::U8),
            "I2" | "INT2" => Some(Self::I2),
            _ => None,
        }
    }

    /// Number of `u8` bytes required to store `n` elements of this type.
    ///
    /// For `I2`, this rounds up to the nearest byte (`ceil(n * 2 / 8) = ceil(n/4)`).
    ///
    /// # Panics
    /// Never panics.
    pub fn storage_bytes(self, n: usize) -> usize {
        match self {
            Self::I2 => (n + 3) / 4,
            other => n * other.byte_size().unwrap_or(1),
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_sizes_are_correct() {
        assert_eq!(DType::F32.byte_size(), Some(4));
        assert_eq!(DType::F16.byte_size(), Some(2));
        assert_eq!(DType::BF16.byte_size(), Some(2));
        assert_eq!(DType::I8.byte_size(), Some(1));
        assert_eq!(DType::U8.byte_size(), Some(1));
        assert_eq!(DType::I2.byte_size(), None);
    }

    #[test]
    fn bit_sizes_are_correct() {
        assert_eq!(DType::F32.bit_size(), 32);
        assert_eq!(DType::F16.bit_size(), 16);
        assert_eq!(DType::BF16.bit_size(), 16);
        assert_eq!(DType::I8.bit_size(), 8);
        assert_eq!(DType::U8.bit_size(), 8);
        assert_eq!(DType::I2.bit_size(), 2);
    }

    #[test]
    fn float_classification() {
        assert!(DType::F32.is_float());
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        assert!(!DType::I8.is_float());
        assert!(!DType::U8.is_float());
        assert!(!DType::I2.is_float());
    }

    #[test]
    fn integer_classification() {
        assert!(!DType::F32.is_integer());
        assert!(DType::I8.is_integer());
        assert!(DType::U8.is_integer());
        assert!(DType::I2.is_integer());
    }

    #[test]
    fn sub_byte_only_i2() {
        assert!(DType::I2.is_sub_byte());
        assert!(!DType::U8.is_sub_byte());
        assert!(!DType::I8.is_sub_byte());
        assert!(!DType::F32.is_sub_byte());
    }

    #[test]
    fn signed_classification() {
        assert!(DType::F32.is_signed());
        assert!(DType::I8.is_signed());
        assert!(!DType::U8.is_signed());
        assert!(!DType::I2.is_signed());
    }

    #[test]
    fn as_str_and_display_match() {
        for dt in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I8,
            DType::U8,
            DType::I2,
        ] {
            assert_eq!(dt.as_str(), dt.to_string());
        }
    }

    #[test]
    fn from_str_roundtrip() {
        for dt in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I8,
            DType::U8,
            DType::I2,
        ] {
            let parsed = DType::from_str(dt.as_str()).expect("must parse back");
            assert_eq!(parsed, dt, "round-trip failed for {dt}");
        }
    }

    #[test]
    fn from_str_lowercase() {
        assert_eq!(DType::from_str("f32"), Some(DType::F32));
        assert_eq!(DType::from_str("bf16"), Some(DType::BF16));
        assert_eq!(DType::from_str("i8"), Some(DType::I8));
    }

    #[test]
    fn from_str_aliases() {
        assert_eq!(DType::from_str("FLOAT32"), Some(DType::F32));
        assert_eq!(DType::from_str("BFLOAT16"), Some(DType::BF16));
        assert_eq!(DType::from_str("INT8"), Some(DType::I8));
        assert_eq!(DType::from_str("UINT8"), Some(DType::U8));
    }

    #[test]
    fn from_str_unknown_returns_none() {
        assert_eq!(DType::from_str("FLOAT8"), None);
        assert_eq!(DType::from_str(""), None);
        assert_eq!(DType::from_str("complex128"), None);
    }

    #[test]
    fn storage_bytes_f32_n_elements() {
        // 10 f32s = 40 bytes
        assert_eq!(DType::F32.storage_bytes(10), 40);
    }

    #[test]
    fn storage_bytes_bf16_n_elements() {
        // 8 bf16s = 16 bytes
        assert_eq!(DType::BF16.storage_bytes(8), 16);
    }

    #[test]
    fn storage_bytes_i8_n_elements() {
        assert_eq!(DType::I8.storage_bytes(7), 7);
    }

    #[test]
    fn storage_bytes_i2_packs_four_per_byte() {
        // ceil(4/4)=1, ceil(5/4)=2, ceil(8/4)=2, ceil(9/4)=3
        assert_eq!(DType::I2.storage_bytes(4), 1);
        assert_eq!(DType::I2.storage_bytes(5), 2);
        assert_eq!(DType::I2.storage_bytes(8), 2);
        assert_eq!(DType::I2.storage_bytes(9), 3);
    }

    #[test]
    fn storage_bytes_zero_elements() {
        for dt in [DType::F32, DType::I8, DType::I2] {
            assert_eq!(dt.storage_bytes(0), 0, "{dt} with 0 elements");
        }
    }
}
