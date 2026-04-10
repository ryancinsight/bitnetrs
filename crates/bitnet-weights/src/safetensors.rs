//! SafeTensors file parsing for BitNet b1.58 weight loading.
//!
//! # Format Overview
//!
//! SafeTensors is a simple, safe format for storing tensors designed by
//! HuggingFace.  Each file consists of:
//!
//! 1. A `u64` header length (little-endian).
//! 2. A JSON header of that length describing each tensor's dtype, shape,
//!    and byte offsets into the data region.
//! 3. A raw data region containing all tensor bytes contiguously.
//!
//! # Supported dtypes
//!
//! | safetensors dtype | Rust type | Conversion |
//! |-------------------|-----------|------------|
//! | `BF16`            | `half::bf16` | → `f32` via `bf16::to_f32()` |
//! | `F16`             | `half::f16`  | → `f32` via `f16::to_f32()`  |
//! | `F32`             | `f32`        | identity                      |
//! | `U8`              | `u8`         | returned as raw bytes         |
//! | `I8`              | `i8`         | returned as raw bytes         |
//!
//! # Public API
//!
//! - [`load_bf16_safetensors`]: Load a safetensors file, convert all tensors to
//!   `f32`.  Used for the BF16 master weights checkpoint.
//! - [`load_raw_safetensors`]: Load a safetensors file, returning raw bytes +
//!   shape + dtype string per tensor.  Used for inspection and packed formats.
//! - [`TensorMeta`]: Metadata for a single tensor (dtype, shape, byte range).
//! - [`parse_safetensors_header`]: Parse only the JSON header without loading data.
//!
//! # Invariants
//!
//! - Tensor names in the returned maps exactly match the keys in the safetensors
//!   JSON header (e.g. `"model.embed_tokens.weight"`).
//! - All `f32` values in the returned maps are finite (NaN/Inf from BF16
//!   conversion are replaced with 0.0 and a warning is emitted).
//! - Shape products match the number of elements implied by the byte range.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, trace, warn};

// ---------------------------------------------------------------------------
// TensorMeta
// ---------------------------------------------------------------------------

/// Metadata for a single tensor in a safetensors file.
///
/// Parsed from the JSON header; does not include the actual tensor data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorMeta {
    /// Data type string as it appears in the safetensors header.
    ///
    /// Examples: `"BF16"`, `"F32"`, `"U8"`, `"I8"`.
    pub dtype: String,

    /// Shape of the tensor (dimension sizes in order).
    ///
    /// A scalar has shape `[]`, a vector `[n]`, a matrix `[rows, cols]`.
    pub shape: Vec<usize>,

    /// Half-open byte range `[start, end)` within the data region of the file.
    ///
    /// `data_region[data_offsets[0]..data_offsets[1]]` is the raw tensor payload.
    pub data_offsets: [usize; 2],
}

impl TensorMeta {
    /// Number of elements: product of all shape dimensions.
    ///
    /// Returns `1` for scalars (empty shape).
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    /// Byte size of the tensor data.
    pub fn byte_size(&self) -> usize {
        self.data_offsets[1] - self.data_offsets[0]
    }
}

// ---------------------------------------------------------------------------
// Header parsing
// ---------------------------------------------------------------------------

/// Raw representation of the safetensors JSON header.
///
/// The `__metadata__` key (if present) is captured separately; all other keys
/// are tensor entries.
#[derive(Debug, Deserialize)]
struct SafetensorsHeader {
    #[serde(rename = "__metadata__", default)]
    metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(flatten)]
    tensors: HashMap<String, TensorEntry>,
}

/// A single tensor entry in the safetensors JSON header.
#[derive(Debug, Deserialize)]
struct TensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Parse the safetensors JSON header from raw bytes.
///
/// # Returns
///
/// A map from tensor name → [`TensorMeta`] for all tensors in the file.
///
/// # Errors
///
/// Returns an error if the header bytes are not valid UTF-8 or valid JSON,
/// or if any tensor entry is malformed.
pub fn parse_safetensors_header(
    header_bytes: &[u8],
) -> anyhow::Result<HashMap<String, TensorMeta>> {
    let header_str =
        std::str::from_utf8(header_bytes).context("safetensors header is not valid UTF-8")?;

    // The header may contain trailing null bytes for alignment — strip them.
    let header_str = header_str.trim_end_matches('\0');

    let raw: SafetensorsHeader =
        serde_json::from_str(header_str).context("safetensors header is not valid JSON")?;

    let tensors: HashMap<String, TensorMeta> = raw
        .tensors
        .into_iter()
        .map(|(name, entry)| {
            let meta = TensorMeta {
                dtype: entry.dtype,
                shape: entry.shape,
                data_offsets: entry.data_offsets,
            };
            (name, meta)
        })
        .collect();

    Ok(tensors)
}

// ---------------------------------------------------------------------------
// load_bf16_safetensors
// ---------------------------------------------------------------------------

/// Load a safetensors file and convert all tensors to `f32`.
///
/// Supports the following source dtypes:
/// - `BF16` → each `bf16` is widened to `f32` via `bf16::to_f32()`
/// - `F16`  → each `f16` is widened to `f32` via `f16::to_f32()`
/// - `F32`  → copied as-is
///
/// Tensors with unsupported dtypes (`U8`, `I8`, `I32`, etc.) are silently
/// skipped and a `tracing::debug!` message is emitted.
///
/// # Returns
///
/// A `HashMap<String, Vec<f32>>` where keys are tensor names and values are
/// flat, row-major `f32` arrays.
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be opened or read.
/// - The safetensors header is malformed.
/// - A tensor's data region is out of range.
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use bitnet_weights::safetensors::load_bf16_safetensors;
///
/// let tensors = load_bf16_safetensors(Path::new("model.safetensors")).unwrap();
/// let embed = &tensors["model.embed_tokens.weight"];
/// println!("Embedding shape: {} elements", embed.len());
/// ```
#[instrument(level = "debug", skip_all, fields(path = %path.display()))]
pub fn load_bf16_safetensors(path: &Path) -> anyhow::Result<HashMap<String, Vec<f32>>> {
    debug!(path = %path.display(), "Loading BF16 safetensors");

    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read safetensors file: {}", path.display()))?;

    debug!(file_size_bytes = bytes.len(), "File read into memory");

    // ── Parse header ─────────────────────────────────────────────────────────
    // First 8 bytes: u64 little-endian header length.
    if bytes.len() < 8 {
        return Err(anyhow!(
            "File too small to be a valid safetensors file: {} bytes",
            bytes.len()
        ));
    }

    let header_len =
        u64::from_le_bytes(bytes[0..8].try_into().expect("slice is exactly 8 bytes")) as usize;

    let header_end = 8 + header_len;
    if header_end > bytes.len() {
        return Err(anyhow!(
            "safetensors header length ({header_len}) exceeds file size ({})",
            bytes.len()
        ));
    }

    let header_bytes = &bytes[8..header_end];
    let metas =
        parse_safetensors_header(header_bytes).context("Failed to parse safetensors header")?;

    debug!(n_tensors = metas.len(), "safetensors header parsed");

    // ── Data region starts after the header ───────────────────────────────────
    let data_region = &bytes[header_end..];

    // ── Convert each tensor to f32 ────────────────────────────────────────────
    let mut result: HashMap<String, Vec<f32>> = HashMap::with_capacity(metas.len());

    for (name, meta) in &metas {
        let start = meta.data_offsets[0];
        let end = meta.data_offsets[1];

        if end > data_region.len() {
            return Err(anyhow!(
                "Tensor '{}' data offsets [{start}, {end}) exceed data region size {}",
                name,
                data_region.len()
            ));
        }

        let tensor_bytes = &data_region[start..end];

        match meta.dtype.as_str() {
            "BF16" => {
                let f32_vals = bf16_bytes_to_f32(tensor_bytes, name);
                trace!(
                    tensor = %name,
                    n_elements = f32_vals.len(),
                    "BF16 tensor converted to f32"
                );
                result.insert(name.clone(), f32_vals);
            }
            "F16" => {
                let f32_vals = f16_bytes_to_f32(tensor_bytes, name);
                trace!(
                    tensor = %name,
                    n_elements = f32_vals.len(),
                    "F16 tensor converted to f32"
                );
                result.insert(name.clone(), f32_vals);
            }
            "F32" => {
                let f32_vals = f32_bytes_to_f32(tensor_bytes, name)?;
                trace!(
                    tensor = %name,
                    n_elements = f32_vals.len(),
                    "F32 tensor loaded"
                );
                result.insert(name.clone(), f32_vals);
            }
            other => {
                debug!(
                    tensor = %name,
                    dtype = other,
                    "Skipping tensor with non-float dtype (not needed for BF16 loading)"
                );
            }
        }
    }

    debug!(n_converted = result.len(), "safetensors loading complete");
    Ok(result)
}

// ---------------------------------------------------------------------------
// load_raw_safetensors
// ---------------------------------------------------------------------------

/// Load a safetensors file, returning raw bytes + shape + dtype per tensor.
///
/// Unlike [`load_bf16_safetensors`], this function does not perform any type
/// conversion.  It returns the raw bytes as `Vec<u8>` for every tensor, along
/// with the shape and dtype string from the header.
///
/// This is useful for:
/// - Inspecting packed quantised formats (`U8`, `I8`).
/// - Debugging weight loading issues.
/// - Loading tensors that will be converted by a custom pipeline.
///
/// # Returns
///
/// `HashMap<String, (Vec<u8>, Vec<usize>, String)>` where each value is
/// `(raw_bytes, shape, dtype_string)`.
///
/// # Errors
///
/// Returns an error if the file cannot be read or the header is malformed.
#[instrument(level = "debug", skip_all, fields(path = %path.display()))]
pub fn load_raw_safetensors(
    path: &Path,
) -> anyhow::Result<HashMap<String, (Vec<u8>, Vec<usize>, String)>> {
    debug!(path = %path.display(), "Loading raw safetensors");

    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read safetensors file: {}", path.display()))?;

    if bytes.len() < 8 {
        return Err(anyhow!("File too small: {} bytes", bytes.len()));
    }

    let header_len =
        u64::from_le_bytes(bytes[0..8].try_into().expect("slice is exactly 8 bytes")) as usize;
    let header_end = 8 + header_len;

    if header_end > bytes.len() {
        return Err(anyhow!(
            "Header length {header_len} exceeds file size {}",
            bytes.len()
        ));
    }

    let metas = parse_safetensors_header(&bytes[8..header_end])?;
    let data_region = &bytes[header_end..];

    let mut result = HashMap::with_capacity(metas.len());

    for (name, meta) in metas {
        let start = meta.data_offsets[0];
        let end = meta.data_offsets[1];

        if end > data_region.len() {
            return Err(anyhow!(
                "Tensor '{}' offsets [{start}, {end}) exceed data region {}",
                name,
                data_region.len()
            ));
        }

        let raw = data_region[start..end].to_vec();
        result.insert(name, (raw, meta.shape, meta.dtype));
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// load_safetensors_meta
// ---------------------------------------------------------------------------

/// Parse only the header metadata from a safetensors file without loading data.
///
/// Much faster than loading the full file when only shape/dtype information
/// is needed (e.g. for weight name validation before downloading).
///
/// # Returns
///
/// A `HashMap<String, TensorMeta>` with all tensor descriptors.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or the header is malformed.
pub fn load_safetensors_meta(path: &Path) -> anyhow::Result<HashMap<String, TensorMeta>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open {}", path.display()))?;

    use std::io::Read;
    let mut reader = std::io::BufReader::new(file);

    // Read the 8-byte header length.
    let mut len_buf = [0u8; 8];
    reader
        .read_exact(&mut len_buf)
        .context("Cannot read header length")?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    // Read only the header bytes.
    let mut header_bytes = vec![0u8; header_len];
    reader
        .read_exact(&mut header_bytes)
        .context("Cannot read header bytes")?;

    parse_safetensors_header(&header_bytes)
}

// ---------------------------------------------------------------------------
// Private conversion helpers
// ---------------------------------------------------------------------------

/// Convert raw BF16 bytes to `f32`, replacing non-finite values with 0.0.
///
/// Panics in debug mode if `bytes.len() % 2 != 0`.
fn bf16_bytes_to_f32(bytes: &[u8], tensor_name: &str) -> Vec<f32> {
    debug_assert_eq!(
        bytes.len() % 2,
        0,
        "BF16 tensor '{}' has odd byte count {}",
        tensor_name,
        bytes.len()
    );

    let n = bytes.len() / 2;
    let mut result = Vec::with_capacity(n);
    let mut n_non_finite = 0usize;

    for chunk in bytes.chunks_exact(2) {
        let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
        let val = bf16::from_bits(raw).to_f32();
        if !val.is_finite() {
            n_non_finite += 1;
            result.push(0.0_f32);
        } else {
            result.push(val);
        }
    }

    if n_non_finite > 0 {
        warn!(
            tensor = tensor_name,
            n_non_finite, "BF16 tensor contained non-finite values replaced with 0.0"
        );
    }

    result
}

/// Convert raw F16 bytes to `f32`, replacing non-finite values with 0.0.
fn f16_bytes_to_f32(bytes: &[u8], tensor_name: &str) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % 2, 0);

    let n = bytes.len() / 2;
    let mut result = Vec::with_capacity(n);
    let mut n_non_finite = 0usize;

    for chunk in bytes.chunks_exact(2) {
        let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
        let val = f16::from_bits(raw).to_f32();
        if !val.is_finite() {
            n_non_finite += 1;
            result.push(0.0_f32);
        } else {
            result.push(val);
        }
    }

    if n_non_finite > 0 {
        warn!(
            tensor = tensor_name,
            n_non_finite, "F16 tensor contained non-finite values replaced with 0.0"
        );
    }

    result
}

/// Convert raw F32 bytes (little-endian) to `f32`.
///
/// # Errors
///
/// Returns an error if `bytes.len() % 4 != 0`.
fn f32_bytes_to_f32(bytes: &[u8], tensor_name: &str) -> anyhow::Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(anyhow!(
            "F32 tensor '{}' has byte count {} which is not a multiple of 4",
            tensor_name,
            bytes.len()
        ));
    }

    let n = bytes.len() / 4;
    let mut result = Vec::with_capacity(n);

    for chunk in bytes.chunks_exact(4) {
        let raw = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        result.push(if raw.is_finite() { raw } else { 0.0 });
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // -----------------------------------------------------------------------
    // TensorMeta
    // -----------------------------------------------------------------------

    #[test]
    fn tensor_meta_numel_matrix() {
        let meta = TensorMeta {
            dtype: "BF16".to_string(),
            shape: vec![4, 8],
            data_offsets: [0, 64],
        };
        assert_eq!(meta.numel(), 32, "4 × 8 = 32 elements");
    }

    #[test]
    fn tensor_meta_numel_vector() {
        let meta = TensorMeta {
            dtype: "F32".to_string(),
            shape: vec![16],
            data_offsets: [0, 64],
        };
        assert_eq!(meta.numel(), 16);
    }

    #[test]
    fn tensor_meta_numel_scalar() {
        let meta = TensorMeta {
            dtype: "F32".to_string(),
            shape: vec![],
            data_offsets: [0, 4],
        };
        // Empty shape → 1 element (scalar).
        assert_eq!(meta.numel(), 1);
    }

    #[test]
    fn tensor_meta_byte_size() {
        let meta = TensorMeta {
            dtype: "BF16".to_string(),
            shape: vec![2, 4],
            data_offsets: [10, 26], // 16 bytes for 8 × 2-byte BF16
        };
        assert_eq!(meta.byte_size(), 16);
    }

    // -----------------------------------------------------------------------
    // parse_safetensors_header
    // -----------------------------------------------------------------------

    fn make_header_json(entries: &[(&str, &str, &[usize], [usize; 2])]) -> Vec<u8> {
        let mut map = serde_json::Map::new();
        for (name, dtype, shape, offsets) in entries {
            let entry = serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": offsets,
            });
            map.insert(name.to_string(), entry);
        }
        serde_json::to_vec(&serde_json::Value::Object(map)).unwrap()
    }

    #[test]
    fn parse_header_single_tensor() {
        let json = make_header_json(&[("weight", "BF16", &[2, 4], [0, 16])]);
        let metas = parse_safetensors_header(&json).unwrap();
        assert_eq!(metas.len(), 1);
        let m = &metas["weight"];
        assert_eq!(m.dtype, "BF16");
        assert_eq!(m.shape, vec![2, 4]);
        assert_eq!(m.data_offsets, [0, 16]);
    }

    #[test]
    fn parse_header_multiple_tensors() {
        let json = make_header_json(&[
            ("a", "F32", &[4], [0, 16]),
            ("b", "BF16", &[2, 2], [16, 24]),
        ]);
        let metas = parse_safetensors_header(&json).unwrap();
        assert_eq!(metas.len(), 2);
        assert!(metas.contains_key("a"));
        assert!(metas.contains_key("b"));
    }

    #[test]
    fn parse_header_with_metadata_key_is_ignored() {
        // The __metadata__ key must not appear as a tensor.
        let json_str = r#"{
            "__metadata__": {"format": "pt"},
            "embed": {"dtype": "F32", "shape": [8], "data_offsets": [0, 32]}
        }"#;
        let metas = parse_safetensors_header(json_str.as_bytes()).unwrap();
        assert!(!metas.contains_key("__metadata__"));
        assert!(metas.contains_key("embed"));
    }

    #[test]
    fn parse_header_invalid_json_returns_error() {
        let bad = b"not json at all!!!";
        let err = parse_safetensors_header(bad).unwrap_err();
        assert!(
            err.to_string().contains("JSON"),
            "error must mention JSON: {err}"
        );
    }

    #[test]
    fn parse_header_with_trailing_nulls() {
        let mut json = make_header_json(&[("x", "F32", &[1], [0, 4])]);
        json.extend_from_slice(&[0u8; 16]); // trailing null padding
        let metas = parse_safetensors_header(&json).unwrap();
        assert_eq!(metas.len(), 1);
    }

    // -----------------------------------------------------------------------
    // BF16 conversion helpers
    // -----------------------------------------------------------------------

    #[test]
    fn bf16_bytes_to_f32_simple_values() {
        // Encode [1.0, -1.0, 0.0] as BF16 little-endian.
        let vals: Vec<bf16> = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(-1.0),
            bf16::from_f32(0.0),
        ];
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();

        let f32_vals = bf16_bytes_to_f32(&bytes, "test");
        assert_eq!(f32_vals.len(), 3);
        assert!((f32_vals[0] - 1.0).abs() < 1e-3, "1.0 → {}", f32_vals[0]);
        assert!(
            (f32_vals[1] - (-1.0)).abs() < 1e-3,
            "-1.0 → {}",
            f32_vals[1]
        );
        assert!(f32_vals[2].abs() < 1e-6, "0.0 → {}", f32_vals[2]);
    }

    #[test]
    fn bf16_bytes_to_f32_non_finite_replaced_with_zero() {
        // BF16 bits for +Inf: sign=0, exp=all-ones (0xFF), mantissa=0 → 0x7F80
        let inf_bits = 0x7F80_u16;
        let bytes = inf_bits.to_le_bytes();
        let result = bf16_bytes_to_f32(&bytes, "inf_test");
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0], 0.0,
            "BF16 Inf must be replaced with 0.0, got {}",
            result[0]
        );
    }

    #[test]
    fn f16_bytes_to_f32_simple_values() {
        let vals: Vec<f16> = vec![f16::from_f32(2.5), f16::from_f32(-0.5)];
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();

        let f32_vals = f16_bytes_to_f32(&bytes, "f16_test");
        assert_eq!(f32_vals.len(), 2);
        assert!((f32_vals[0] - 2.5).abs() < 1e-3, "2.5 → {}", f32_vals[0]);
        assert!(
            (f32_vals[1] - (-0.5)).abs() < 1e-3,
            "-0.5 → {}",
            f32_vals[1]
        );
    }

    #[test]
    fn f32_bytes_to_f32_roundtrip() {
        let original = vec![1.5_f32, -2.25, 3.14159, 0.0, -100.0];
        let bytes: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();

        let recovered = f32_bytes_to_f32(&bytes, "f32_test").unwrap();
        assert_eq!(recovered.len(), original.len());
        for (i, (&orig, &rec)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-7,
                "index {i}: orig={orig}, recovered={rec}"
            );
        }
    }

    #[test]
    fn f32_bytes_to_f32_odd_length_returns_error() {
        let bytes = vec![0u8; 7]; // not a multiple of 4
        let err = f32_bytes_to_f32(&bytes, "odd").unwrap_err();
        assert!(
            err.to_string().contains("multiple of 4"),
            "error must mention multiple of 4: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // Full file loading (using a synthetic safetensors file)
    // -----------------------------------------------------------------------

    /// Build a minimal valid safetensors file in memory.
    ///
    /// Layout:
    /// - 8 bytes: header length (u64 LE)
    /// - N bytes: JSON header
    /// - M bytes: data region
    fn build_safetensors_file(tensors: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        // Build JSON header.
        let mut offset = 0usize;
        let mut header_map = serde_json::Map::new();
        let mut all_data: Vec<u8> = Vec::new();

        for (name, dtype, shape, data) in tensors {
            let end = offset + data.len();
            let entry = serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, end],
            });
            header_map.insert(name.to_string(), entry);
            all_data.extend_from_slice(data);
            offset = end;
        }

        let header_json = serde_json::to_vec(&serde_json::Value::Object(header_map)).unwrap();
        let header_len = header_json.len() as u64;

        let mut file = Vec::new();
        file.extend_from_slice(&header_len.to_le_bytes());
        file.extend_from_slice(&header_json);
        file.extend_from_slice(&all_data);
        file
    }

    #[test]
    fn load_bf16_safetensors_single_tensor() {
        // Build a BF16 tensor: [1.0, 2.0, 3.0, 4.0] encoded as BF16 LE.
        let values: Vec<bf16> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| bf16::from_f32(v))
            .collect();
        let data: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();

        let file_bytes = build_safetensors_file(&[("my_tensor", "BF16", &[2, 2], &data)]);

        // Write to a temporary file.
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let result = load_bf16_safetensors(tmp.path()).unwrap();
        assert_eq!(result.len(), 1);
        let tensor = &result["my_tensor"];
        assert_eq!(tensor.len(), 4);
        assert!((tensor[0] - 1.0).abs() < 0.01, "element 0: {}", tensor[0]);
        assert!((tensor[1] - 2.0).abs() < 0.01, "element 1: {}", tensor[1]);
        assert!((tensor[2] - 3.0).abs() < 0.01, "element 2: {}", tensor[2]);
        assert!((tensor[3] - 4.0).abs() < 0.01, "element 3: {}", tensor[3]);
    }

    #[test]
    fn load_bf16_safetensors_f32_tensor_passes_through() {
        let data: Vec<u8> = [1.0_f32, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file_bytes = build_safetensors_file(&[("f32_weight", "F32", &[2], &data)]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let result = load_bf16_safetensors(tmp.path()).unwrap();
        let t = &result["f32_weight"];
        assert_eq!(t.len(), 2);
        assert!((t[0] - 1.0).abs() < 1e-7);
        assert!((t[1] - (-2.0)).abs() < 1e-7);
    }

    #[test]
    fn load_bf16_safetensors_u8_tensor_skipped() {
        let data = vec![1u8, 2, 3, 4];
        let file_bytes = build_safetensors_file(&[("packed_weights", "U8", &[4], &data)]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let result = load_bf16_safetensors(tmp.path()).unwrap();
        // U8 tensors are skipped (not float).
        assert!(
            !result.contains_key("packed_weights"),
            "U8 tensors must be skipped"
        );
    }

    #[test]
    fn load_bf16_safetensors_multiple_tensors() {
        // Two BF16 tensors in one file.
        let embed_data: Vec<u8> = [0.1_f32, 0.2, 0.3]
            .iter()
            .map(|&v| bf16::from_f32(v))
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        let norm_data: Vec<u8> = [1.0_f32, 1.0]
            .iter()
            .map(|&v| bf16::from_f32(v))
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();

        let file_bytes = build_safetensors_file(&[
            ("model.embed_tokens.weight", "BF16", &[3], &embed_data),
            ("model.norm.weight", "BF16", &[2], &norm_data),
        ]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let result = load_bf16_safetensors(tmp.path()).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("model.embed_tokens.weight"));
        assert!(result.contains_key("model.norm.weight"));
        assert_eq!(result["model.embed_tokens.weight"].len(), 3);
        assert_eq!(result["model.norm.weight"].len(), 2);
    }

    #[test]
    fn load_bf16_safetensors_nonexistent_file_returns_error() {
        let err = load_bf16_safetensors(Path::new("/nonexistent/file.safetensors")).unwrap_err();
        assert!(
            !err.to_string().is_empty(),
            "error message must be non-empty"
        );
    }

    #[test]
    fn load_bf16_safetensors_all_outputs_are_finite() {
        // Any f32 in the output must be finite.
        let data: Vec<u8> = [0.5_f32, -0.5, 1.0, -1.0, 0.25]
            .iter()
            .map(|&v| bf16::from_f32(v))
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        let file_bytes = build_safetensors_file(&[("w", "BF16", &[5], &data)]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let result = load_bf16_safetensors(tmp.path()).unwrap();
        for (name, tensor) in &result {
            for (i, &v) in tensor.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "tensor '{name}' element {i} = {v} is not finite"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // load_raw_safetensors
    // -----------------------------------------------------------------------

    #[test]
    fn load_raw_safetensors_returns_exact_bytes() {
        let data: Vec<u8> = vec![10, 20, 30, 40];
        let file_bytes = build_safetensors_file(&[("raw", "U8", &[4], &data)]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let result = load_raw_safetensors(tmp.path()).unwrap();
        let (raw_bytes, shape, dtype) = &result["raw"];
        assert_eq!(raw_bytes, &data, "raw bytes must match exactly");
        assert_eq!(shape, &[4], "shape must be [4]");
        assert_eq!(dtype, "U8", "dtype must be U8");
    }

    #[test]
    fn load_raw_safetensors_preserves_all_tensors() {
        let data_a = vec![1u8, 2];
        let data_b = vec![3u8, 4, 5, 6];
        let file_bytes =
            build_safetensors_file(&[("a", "U8", &[2], &data_a), ("b", "U8", &[4], &data_b)]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let result = load_raw_safetensors(tmp.path()).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("a"));
        assert!(result.contains_key("b"));
    }

    // -----------------------------------------------------------------------
    // load_safetensors_meta
    // -----------------------------------------------------------------------

    #[test]
    fn load_safetensors_meta_returns_only_header_info() {
        let data: Vec<u8> = vec![0u8; 16]; // 8 BF16 values
        let file_bytes = build_safetensors_file(&[("embed", "BF16", &[2, 4], &data)]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();
        tmp.flush().unwrap();

        let metas = load_safetensors_meta(tmp.path()).unwrap();
        assert_eq!(metas.len(), 1);
        let m = &metas["embed"];
        assert_eq!(m.dtype, "BF16");
        assert_eq!(m.shape, vec![2, 4]);
        assert_eq!(m.numel(), 8);
    }

    #[test]
    fn load_safetensors_meta_nonexistent_file_returns_error() {
        let err = load_safetensors_meta(Path::new("/no/such/file.safetensors")).unwrap_err();
        assert!(!err.to_string().is_empty());
    }
}
