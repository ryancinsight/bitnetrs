//! Ternary weight representation and bit-packing for BitNet b1.58.
//!
//! # Mathematical Foundation
//!
//! BitNet b1.58 quantises each weight matrix **W** to ternary values
//! {-1, 0, +1} using the *absmean* quantisation function:
//!
//! ```text
//! α_W = mean(|W|)                         (per-tensor scale)
//! W_q = clip( round(W / α_W), -1, 1 )    (ternary quantisation)
//! ```
//!
//! The effective weight used during a forward pass is:
//!
//! ```text
//! W_eff = W_q * α_W
//! ```
//!
//! so the dequantised approximation of the original weight `w` is simply
//! `w_q ∈ {-α_W, 0, +α_W}`.
//!
//! # Storage Layout
//!
//! Each ternary value `w_q ∈ {-1, 0, +1}` requires only 2 bits.  Four
//! values are packed into one `u8` byte in little-endian bit order:
//!
//! ```text
//! byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
//! ```
//!
//! where the 2-bit encoding is:
//!
//! | value | bits |
//! |-------|------|
//! |  +1   |  00  |
//! |   0   |  01  |
//! |  -1   |  10  |
//!
//! This matches the real HuggingFace packed BitNet deployment weights, which
//! use the three codes `{00, 01, 10}` and do not emit `11`.
//!
//! For a weight matrix of shape `[rows, cols]`:
//! - Unpacked (`Vec<i8>`) size: `rows * cols` bytes.
//! - Packed   (`Vec<u8>`) size: `ceil(rows * cols / 4)` bytes — a 4× reduction.
//!
//! # Invariants
//!
//! - `data.len() == rows * ((cols + 3) / 4)` (row-aligned packed byte count).
//! - `scale > 0` (guaranteed by clamping in `absmean_quantize`).
//! - Each 2-bit code in `data` is in `{0b00, 0b01, 0b10}`; `0b11` appears
//!   only as padding in the last byte when `rows * cols % 4 != 0`.

use crate::error::{BitNetError, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TernaryWeight
// ---------------------------------------------------------------------------

/// A ternary weight matrix stored in packed 2-bit format.
///
/// Each weight value in {-1, 0, +1} occupies 2 bits, with 4 values packed
/// per `u8` byte. This reduces memory usage by 4× compared to `Vec<i8>`.
///
/// ## Encoding
///
/// ```text
/// byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
/// ```
///
/// | value | bits |
/// |-------|------|
/// |  +1   |  00  |
/// |   0   |  01  |
/// |  -1   |  10  |
///
/// ## Invariants
/// - `data.len() == rows * ((cols + 3) / 4)` (row-aligned packed byte count).
/// - `scale > 0`.
/// - Each row starts at a byte boundary (requires `cols % 4 == 0` for aligned access).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TernaryWeight {
    /// Packed 2-bit-per-value storage. 4 ternary values per byte.
    pub data: Vec<u8>,
    /// Per-tensor absmean scale α_W = mean(|W|).
    pub scale: f32,
    /// Number of output features (matrix rows).
    pub rows: usize,
    /// Number of input features (matrix columns).
    pub cols: usize,
}

impl TernaryWeight {
    /// Construct a [`TernaryWeight`] from pre-packed row-aligned data.
    ///
    /// `packed_data` must have exactly `rows * ((cols + 3) / 4)` bytes,
    /// where each row is independently packed to a byte boundary.
    ///
    /// # Errors
    /// Returns [`BitNetError::InvalidShape`] if `packed_data.len()` is wrong.
    /// Returns [`BitNetError::QuantizationError`] if `scale <= 0`.
    pub fn new(packed_data: Vec<u8>, scale: f32, rows: usize, cols: usize) -> Result<Self> {
        let packed_cols = (cols + 3) / 4;
        let expected_bytes = rows * packed_cols;
        if packed_data.len() != expected_bytes {
            return Err(BitNetError::shape(
                format!("packed data: {rows} * (({cols} + 3) / 4) = {expected_bytes} bytes"),
                format!("{} bytes", packed_data.len()),
            ));
        }
        if scale <= 0.0 || !scale.is_finite() {
            return Err(BitNetError::quant(format!(
                "scale must be finite and > 0, got {scale}"
            )));
        }
        Ok(Self {
            data: packed_data,
            scale,
            rows,
            cols,
        })
    }

    /// Construct from unpacked `i8` ternary values by packing them.
    ///
    /// Each element in `data` must be in `{-1, 0, 1}`.
    ///
    /// # Errors
    /// Returns [`BitNetError::InvalidShape`] if `data.len() != rows * cols`.
    /// Returns [`BitNetError::QuantizationError`] if `scale <= 0` or any value
    /// is outside `{-1, 0, 1}`.
    pub fn from_i8(data: &[i8], scale: f32, rows: usize, cols: usize) -> Result<Self> {
        let expected = rows * cols;
        if data.len() != expected {
            return Err(BitNetError::shape(
                format!("[{rows}, {cols}] = {expected} elements"),
                format!("{} elements", data.len()),
            ));
        }
        if scale <= 0.0 || !scale.is_finite() {
            return Err(BitNetError::quant(format!(
                "scale must be finite and > 0, got {scale}"
            )));
        }
        #[cfg(debug_assertions)]
        for (i, &v) in data.iter().enumerate() {
            if v < -1 || v > 1 {
                return Err(BitNetError::quant(format!(
                    "data[{i}] = {v} is outside {{-1, 0, 1}}"
                )));
            }
        }
        let packed_cols = (cols + 3) / 4;
        let mut packed = Vec::with_capacity(rows * packed_cols);
        for r in 0..rows {
            let row_start = r * cols;
            packed.extend_from_slice(&pack_ternary(&data[row_start..row_start + cols]));
        }
        Ok(Self {
            data: packed,
            scale,
            rows,
            cols,
        })
    }

    /// Construct without validation — caller guarantees all invariants.
    #[inline]
    pub fn new_unchecked(packed_data: Vec<u8>, scale: f32, rows: usize, cols: usize) -> Self {
        Self {
            data: packed_data,
            scale,
            rows,
            cols,
        }
    }

    /// Total number of logical elements: `rows * cols`.
    #[inline]
    pub fn numel(&self) -> usize {
        self.rows * self.cols
    }

    /// Number of packed bytes per row: `(cols + 3) / 4`.
    #[inline]
    pub fn packed_cols(&self) -> usize {
        (self.cols + 3) / 4
    }

    /// Access the packed bytes for a single row.
    ///
    /// Returns a slice of `packed_cols()` bytes containing the packed
    /// ternary values for the given row.
    #[inline]
    pub fn packed_row(&self, row: usize) -> &[u8] {
        debug_assert!(row < self.rows);
        let pc = self.packed_cols();
        let start = row * pc;
        &self.data[start..start + pc]
    }

    /// Access a single element at `(row, col)` by unpacking on the fly.
    ///
    /// This is O(1) but involves bit manipulation. Not intended for hot paths.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> i8 {
        debug_assert!(row < self.rows && col < self.cols);
        let pc = self.packed_cols();
        let byte_idx = row * pc + col / 4;
        let bit_pos = (col % 4) * 2;
        let code = (self.data[byte_idx] >> bit_pos) & 0b11;
        match code {
            0b00 => 1,
            0b01 => 0,
            0b10 => -1,
            _ => 0, // 0b11: invalid, treat as zero
        }
    }

    /// Unpack and return a row as `Vec<i8>` (for debugging/testing).
    pub fn row_unpacked(&self, row: usize) -> Vec<i8> {
        let packed = self.packed_row(row);
        let n = self.cols;
        unpack_ternary(packed, n).expect("packed row data is valid")
    }

    /// Return the packed data (already packed — this is a no-op clone).
    pub fn pack(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// Compute the memory footprint of the packed representation in bytes.
    #[inline]
    pub fn packed_bytes(&self) -> usize {
        self.rows * self.packed_cols()
    }
}

// ---------------------------------------------------------------------------
// Pack / unpack helpers
// ---------------------------------------------------------------------------

/// Encode ternary `i8` values in {-1, 0, +1} into 2-bits-per-value packed `u8` bytes.
///
/// # Encoding table
///
/// This encoding matches the microsoft/bitnet-b1.58-2B-4T HuggingFace packed
/// deployment format, where the zero value (most frequent in BitNet ternary
/// weights, typically ~65%) is assigned the code `0b01`.
///
/// | i8 value | 2-bit code |
/// |----------|-----------|
/// |    +1    |    0b00   |
/// |     0    |    0b01   |
/// |    -1    |    0b10   |
///
/// **Invariant**: codes `{0b00, 0b01, 0b10}` are the only valid outputs;
/// `0b11` is never emitted (it appears only as padding in the last byte
/// when `data.len() % 4 != 0`).
///
/// Packing order within each byte (little-endian):
/// `byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)`
///
/// # Panics (debug only)
/// Panics if any element is outside {-1, 0, 1}.
pub fn pack_ternary(data: &[i8]) -> Vec<u8> {
    let packed_len = (data.len() + 3) / 4;
    let mut packed = vec![0u8; packed_len];

    for (byte_idx, chunk) in data.chunks(4).enumerate() {
        let mut byte = 0u8;
        for (bit_pos, &val) in chunk.iter().enumerate() {
            debug_assert!(val >= -1 && val <= 1, "ternary value out of range: {val}");
            let bits = match val {
                1 => 0b00,
                0 => 0b01,
                -1 => 0b10,
                _ => unreachable!("debug_assert above guarantees ternary domain"),
            };
            byte |= bits << (bit_pos * 2);
        }
        packed[byte_idx] = byte;
    }
    packed
}

/// Decode 2-bits-per-value packed `u8` bytes back into `i8` ternary values.
///
/// `n` must equal the number of original (unpacked) elements.  The function
/// discards any padding bits in the last byte if `n % 4 != 0`.
///
/// # Errors
/// Returns [`BitNetError::InvalidShape`] if `packed.len() * 4 < n`.
pub fn unpack_ternary(packed: &[u8], n: usize) -> Result<Vec<i8>> {
    let required_bytes = (n + 3) / 4;
    if packed.len() < required_bytes {
        return Err(BitNetError::shape(
            format!("≥ {required_bytes} packed bytes for {n} elements"),
            format!("{} bytes", packed.len()),
        ));
    }

    let mut out = Vec::with_capacity(n);

    'outer: for &byte in &packed[..required_bytes] {
        for bit_pos in 0..4 {
            if out.len() == n {
                break 'outer;
            }
            let bits = (byte >> (bit_pos * 2)) & 0b11;
            let val: i8 = match bits {
                0b00 => 1,
                0b01 => 0,
                0b10 => -1,
                _ => {
                    return Err(BitNetError::quant(format!(
                        "invalid 2-bit code 0b{bits:02b} in packed ternary"
                    )))
                }
            };
            out.push(val);
        }
    }

    debug_assert_eq!(out.len(), n);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // pack_ternary / unpack_ternary (unchanged helpers)
    // -----------------------------------------------------------------------

    /// Verify the round-trip property: pack(unpack(x)) == x for all ternary x.
    #[test]
    fn pack_unpack_roundtrip_full_bytes() {
        let original: Vec<i8> = vec![1, 0, -1, 1, -1, -1, 0, 0];
        let packed = pack_ternary(&original);
        assert_eq!(packed.len(), 2, "8 values → 2 bytes");
        let recovered = unpack_ternary(&packed, original.len()).unwrap();
        assert_eq!(recovered, original, "round-trip must be lossless");
    }

    #[test]
    fn pack_unpack_roundtrip_non_multiple_of_4() {
        let original: Vec<i8> = vec![1, -1, 0]; // 3 elements → 1 byte (last 2 bits padding)
        let packed = pack_ternary(&original);
        assert_eq!(packed.len(), 1);
        let recovered = unpack_ternary(&packed, 3).unwrap();
        assert_eq!(recovered, original);
    }

    #[test]
    fn pack_encoding_matches_spec() {
        // Encoding matches microsoft/bitnet-b1.58-2B-4T HuggingFace format:
        //   +1 → 0b00,  0 → 0b01,  -1 → 0b10
        //
        // For [+1, 0, -1, +1]: bits = 00 | (01<<2) | (10<<4) | (00<<6)
        // Little-endian: byte = v0 | v1<<2 | v2<<4 | v3<<6
        // bits 0-1: 00 = 0
        // bits 2-3: 01 = 4
        // bits 4-5: 10 = 32
        // bits 6-7: 00 = 0
        // total = 0 + 4 + 32 + 0 = 36 = 0x24
        let data: Vec<i8> = vec![1, 0, -1, 1];
        let packed = pack_ternary(&data);
        let expected: u8 = 0b00 | (0b01 << 2) | (0b10 << 4) | (0b00 << 6);
        assert_eq!(
            packed[0], expected,
            "pack encoding must match HuggingFace format"
        );
    }

    /// Verify that the encoding is consistent with the observed HuggingFace
    /// weight distribution: code 0b01 (→ 0) is the most frequent (~65%),
    /// codes 0b00 (→ +1) and 0b10 (→ -1) are roughly equal and less frequent.
    #[test]
    fn pack_encoding_zero_is_most_compact_code() {
        // In BitNet ternary weights, approximately 50–70% of values are zero.
        // The zero value maps to 0b01 (which in a u8 initialized to 0x00 would
        // require setting bits). The all-zero byte 0x00 packs four +1 values,
        // NOT four zeros, confirming that 0b00 ↔ +1.
        let all_plus_ones: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary(&all_plus_ones);
        assert_eq!(packed[0], 0x00, "four +1 values must pack to 0x00");

        let all_zeros: Vec<i8> = vec![0, 0, 0, 0];
        let packed = pack_ternary(&all_zeros);
        assert_eq!(packed[0], 0b01010101, "four 0 values must pack to 0x55");
    }

    #[test]
    fn all_zero_row_encodes_and_decodes() {
        let data: Vec<i8> = vec![0; 16];
        let packed = pack_ternary(&data);
        // New encoding: 0 → 0b01, so each group of 4 zeros = 01|01|01|01 = 0x55
        assert!(
            packed.iter().all(|&b| b == 0x55),
            "all 0 values must pack to 0x55 (0→0b01 encoding)"
        );
        let recovered = unpack_ternary(&packed, 16).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn all_ones_row_encodes_and_decodes() {
        let data: Vec<i8> = vec![1; 16];
        let packed = pack_ternary(&data);
        // New encoding: +1 → 0b00, so each group of 4 ones = 00|00|00|00 = 0x00
        assert!(
            packed.iter().all(|&b| b == 0x00),
            "all +1 values must pack to 0x00 (+1→0b00 encoding)"
        );
        let recovered = unpack_ternary(&packed, 16).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn all_neg_ones_row_encodes_and_decodes() {
        let data: Vec<i8> = vec![-1; 16];
        let packed = pack_ternary(&data);
        // Each group of 4 neg-ones: -1→0b10, so each byte = 10|10|10|10 = 0b10101010 = 0xAA
        // This is unchanged from any prior encoding since -1 always maps to 0b10.
        assert!(
            packed.iter().all(|&b| b == 0xAA),
            "all -1 must pack to 0xAA bytes"
        );
        let recovered = unpack_ternary(&packed, 16).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn unpack_insufficient_bytes_returns_error() {
        let packed: Vec<u8> = vec![0u8; 1]; // only 4 values
        let err = unpack_ternary(&packed, 8).unwrap_err(); // need 8 values
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    // -----------------------------------------------------------------------
    // TernaryWeight — packed `new()` constructor
    // -----------------------------------------------------------------------

    #[test]
    fn ternary_weight_new_packed_validates() {
        // 6 elements: 2 rows × 3 cols, row-aligned → 2 * ceil(3/4) = 2 packed bytes
        // Pack each row independently to byte boundary.
        let row0 = pack_ternary(&[1i8, 0, -1]);
        let row1 = pack_ternary(&[1i8, 0, -1]);
        let mut packed = Vec::new();
        packed.extend_from_slice(&row0);
        packed.extend_from_slice(&row1);
        assert_eq!(packed.len(), 2);
        let tw = TernaryWeight::new(packed, 0.5, 2, 3).unwrap();
        assert_eq!(tw.rows, 2);
        assert_eq!(tw.cols, 3);
        assert_eq!(tw.numel(), 6);
        assert_eq!(tw.scale, 0.5);
        assert_eq!(tw.get(0, 0), 1);
        assert_eq!(tw.get(0, 2), -1);
        assert_eq!(tw.get(1, 1), 0);
    }

    #[test]
    fn ternary_weight_new_packed_wrong_bytes_rejected() {
        // 6 elements need 2 packed bytes, provide 1
        let packed = vec![0u8; 1];
        let err = TernaryWeight::new(packed, 0.5, 2, 3).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    // -----------------------------------------------------------------------
    // TernaryWeight — `from_i8()` constructor
    // -----------------------------------------------------------------------

    #[test]
    fn ternary_weight_new_validates_shape() {
        let data = vec![1i8, 0, -1, 1, 0, -1];
        let tw = TernaryWeight::from_i8(&data, 0.5, 2, 3).unwrap();
        assert_eq!(tw.rows, 2);
        assert_eq!(tw.cols, 3);
        assert_eq!(tw.numel(), 6);
        assert_eq!(tw.scale, 0.5);
    }

    #[test]
    fn ternary_weight_wrong_size_rejected() {
        let data = vec![1i8, 0, -1];
        let err = TernaryWeight::from_i8(&data, 0.5, 2, 3).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn ternary_weight_nonpositive_scale_rejected() {
        let data = vec![1i8, 0, -1, 0, 1, -1];
        let err = TernaryWeight::from_i8(&data, 0.0, 2, 3).unwrap_err();
        assert!(matches!(err, BitNetError::QuantizationError(_)));
    }

    #[test]
    fn ternary_weight_row_accessor() {
        let data: Vec<i8> = vec![1, 0, -1, 0, 1, 1];
        let tw = TernaryWeight::from_i8(&data, 1.0, 2, 3).unwrap();
        assert_eq!(tw.row_unpacked(0), vec![1i8, 0, -1]);
        assert_eq!(tw.row_unpacked(1), vec![0i8, 1, 1]);
    }

    #[test]
    fn ternary_weight_from_i8_roundtrip() {
        let data: Vec<i8> = vec![1, 0, -1, 0, 1, 1, -1, -1];
        let tw = TernaryWeight::from_i8(&data, 1.0, 2, 4).unwrap();
        assert_eq!(tw.rows, 2);
        assert_eq!(tw.cols, 4);
        assert_eq!(tw.packed_cols(), 1);
        assert_eq!(tw.data.len(), 2); // 8 elements = 2 packed bytes
                                      // Verify round-trip
        for (i, &expected) in data.iter().enumerate() {
            let row = i / 4;
            let col = i % 4;
            assert_eq!(tw.get(row, col), expected, "mismatch at ({row}, {col})");
        }
    }

    #[test]
    fn ternary_weight_packed_bytes_count() {
        let cfg_cases = [(4, 4, 4), (4, 5, 8), (3, 3, 3)];
        for (rows, cols, expected_bytes) in cfg_cases {
            let data = vec![0i8; rows * cols];
            let tw = TernaryWeight::from_i8(&data, 1.0, rows, cols).unwrap();
            assert_eq!(tw.packed_bytes(), expected_bytes, "rows={rows} cols={cols}");
        }
    }

    // -----------------------------------------------------------------------
    // TernaryWeight — element access and packed_row
    // -----------------------------------------------------------------------

    #[test]
    fn ternary_weight_get_all_values() {
        // Verify get() returns correct values for each ternary code.
        let data: Vec<i8> = vec![1, 0, -1, 1];
        let tw = TernaryWeight::from_i8(&data, 1.0, 1, 4).unwrap();
        assert_eq!(tw.get(0, 0), 1);
        assert_eq!(tw.get(0, 1), 0);
        assert_eq!(tw.get(0, 2), -1);
        assert_eq!(tw.get(0, 3), 1);
    }

    #[test]
    fn ternary_weight_packed_row_slice() {
        // 2 rows × 4 cols → each row is 1 packed byte
        let data: Vec<i8> = vec![1, 1, 1, 1, -1, -1, -1, -1];
        let tw = TernaryWeight::from_i8(&data, 1.0, 2, 4).unwrap();
        assert_eq!(tw.packed_row(0), &[0x00]); // four +1 → 0x00
        assert_eq!(tw.packed_row(1), &[0xAA]); // four -1 → 0xAA
    }

    #[test]
    fn ternary_weight_pack_returns_clone() {
        let data: Vec<i8> = vec![1, 0, -1, 0];
        let tw = TernaryWeight::from_i8(&data, 1.0, 1, 4).unwrap();
        let packed_clone = tw.pack();
        assert_eq!(packed_clone, tw.data);
    }
}
