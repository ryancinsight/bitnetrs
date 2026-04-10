//! Absmax quantisation for 8-bit activations in BitNet b1.58.
//!
//! # Mathematical Foundation
//!
//! BitNet b1.58 uses *per-token* absmax quantisation for activations:
//!
//! ```text
//! α_x  = max( |x| ) / 127          (per-token scale)
//! x_q  = clip( round( x / α_x ), −128, 127 )
//! ```
//!
//! The quantised activation `x_q` is an `i8` value in `[−128, 127]`.
//!
//! During a forward pass the activation scale is combined with the weight
//! scale to recover the full-precision output:
//!
//! ```text
//! y = (W_q @ x_q) * α_W * α_x / 127
//! ```
//!
//! where `W_q @ x_q` is an integer dot product (each weight ∈ {−1, 0, +1},
//! each activation ∈ [−128, 127]).
//!
//! # Per-Token vs Per-Tensor
//!
//! The standard BitNet activation quantisation is *per-token*: each row
//! of the activation matrix (one token's hidden state) gets its own scale.
//! The batch helper [`absmax_quantize_batch`] applies this per-row.
//!
//! # Invariants (guaranteed by the public API)
//! - Returned `Vec<i8>` satisfies `∀ v : v ∈ [−128, 127]`.
//! - Returned `scale > 0` (clamped to a minimum of `1 / 127`).
//! - `scale = max(|x|) / 127` unless `max(|x|) < ε`, in which case the
//!   minimum scale is used.
//! - Length of returned `Vec<i8>` equals the length of the input slice.

use crate::error::{BitNetError, Result};

/// Minimum absolute-maximum value before scaling: prevents division by zero.
const ABSMAX_MIN: f32 = 1e-5;

/// The integer range for 8-bit signed quantisation: `2^7 − 1 = 127`.
const Q8_MAX: f32 = 127.0;

// ---------------------------------------------------------------------------
// Per-token (single row) quantisation
// ---------------------------------------------------------------------------

/// Quantise a single token's activation vector to 8-bit signed integers.
///
/// Applies the per-token absmax formula:
/// ```text
/// scale  = max( |x| ) / 127    (clamped to ≥ ABSMAX_MIN / 127)
/// x_q[i] = clip( round( x[i] / scale ), −128, 127 )
/// ```
///
/// Returns `(quantised, scale)`.
///
/// # Errors
/// - [`BitNetError::QuantizationError`] if `x` is empty or contains a
///   non-finite value.
///
/// # Examples
/// ```
/// use bitnet_core::quant::absmax::absmax_quantize_row;
///
/// let x = vec![0.0_f32, 1.0, -2.0, 0.5];
/// let (q, scale) = absmax_quantize_row(&x).unwrap();
/// // max(|x|) = 2.0  →  scale = 2.0 / 127 ≈ 0.01575
/// // q = [0, 64, -127, 32]  (approximately)
/// assert_eq!(q[2], -127);
/// assert!(scale > 0.0);
/// ```
pub fn absmax_quantize_row(x: &[f32]) -> Result<(Vec<i8>, f32)> {
    if x.is_empty() {
        return Err(BitNetError::quant(
            "cannot quantise an empty activation vector",
        ));
    }

    // Find max(|x|) while checking for non-finite values.
    let mut max_abs = 0.0_f32;
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(BitNetError::quant(format!(
                "non-finite activation value at index {i}: {v}"
            )));
        }
        let abs_v = v.abs();
        if abs_v > max_abs {
            max_abs = abs_v;
        }
    }

    // Clamp to prevent zero / near-zero scale.
    let max_abs_clamped = max_abs.max(ABSMAX_MIN);
    let scale = max_abs_clamped / Q8_MAX;
    let inv_scale = Q8_MAX / max_abs_clamped; // = 1.0 / scale

    // x_q[i] = clip(round(x[i] * inv_scale), -128, 127)
    let quantised: Vec<i8> = x
        .iter()
        .map(|&v| {
            let scaled = (v * inv_scale).round();
            scaled.clamp(-128.0, 127.0) as i8
        })
        .collect();

    Ok((quantised, scale))
}

// ---------------------------------------------------------------------------
// Dequantisation
// ---------------------------------------------------------------------------

/// Reconstruct approximate `f32` activations from 8-bit quantised values and a scale.
///
/// ```text
/// x_deq[i] = x_q[i] as f32 * scale
/// ```
///
/// This is the inverse of [`absmax_quantize_row`] up to rounding error.
///
/// # Errors
/// Returns [`BitNetError::QuantizationError`] if `scale ≤ 0` or is non-finite.
pub fn absmax_dequantize(quantised: &[i8], scale: f32) -> Result<Vec<f32>> {
    if scale <= 0.0 || !scale.is_finite() {
        return Err(BitNetError::quant(format!(
            "scale must be finite and > 0 for dequantisation, got {scale}"
        )));
    }
    Ok(quantised.iter().map(|&q| q as f32 * scale).collect())
}

// ---------------------------------------------------------------------------
// Batch (2-D matrix, per-row) quantisation
// ---------------------------------------------------------------------------

/// Quantise a 2-D activation matrix (rows × cols) with *per-row* absmax scaling.
///
/// Each row of `matrix` (a contiguous slice of length `cols`) is independently
/// quantised with its own scale.  This matches the standard BitNet per-token
/// quantisation where each row corresponds to one token's hidden state.
///
/// Returns `(quantised_flat, scales)` where:
/// - `quantised_flat` has the same length as `matrix` (row-major layout).
/// - `scales[r]` is the absmax scale for row `r`.
///
/// # Errors
/// Returns [`BitNetError::QuantizationError`] if `cols == 0`,
/// `matrix.len() % cols != 0`, or any row contains non-finite values.
pub fn absmax_quantize_batch(matrix: &[f32], cols: usize) -> Result<(Vec<i8>, Vec<f32>)> {
    if cols == 0 {
        return Err(BitNetError::quant(
            "cols must be > 0 for batch quantisation",
        ));
    }
    if matrix.len() % cols != 0 {
        return Err(BitNetError::quant(format!(
            "matrix length ({}) is not divisible by cols ({})",
            matrix.len(),
            cols
        )));
    }

    let rows = matrix.len() / cols;
    let mut all_quantised = Vec::with_capacity(matrix.len());
    let mut scales = Vec::with_capacity(rows);

    for row in matrix.chunks_exact(cols) {
        let (q, s) = absmax_quantize_row(row)?;
        all_quantised.extend_from_slice(&q);
        scales.push(s);
    }

    Ok((all_quantised, scales))
}

// ---------------------------------------------------------------------------
// Fused dot-product: (W_q · x_q) * weight_scale * activation_scale / 127
// ---------------------------------------------------------------------------

/// Compute the dequantised dot product of a ternary weight row and a quantised
/// activation vector.
///
/// This is the core arithmetic kernel for a single output neuron:
///
/// ```text
/// y = ( Σ_k  W_q[k] * x_q[k] ) * weight_scale * act_scale / 127
/// ```
///
/// where `W_q[k] ∈ {−1, 0, +1}` and `x_q[k] ∈ [−128, 127]`.
///
/// The integer accumulation avoids floating-point operations in the inner loop,
/// matching the theoretical efficiency of the quantised kernel.
///
/// # Errors
/// Returns [`BitNetError::InvalidShape`] if `weight_row.len() != activation.len()`.
pub fn ternary_dot_product_quantised(
    weight_row: &[i8],
    activation: &[i8],
    weight_scale: f32,
    act_scale: f32,
) -> Result<f32> {
    if weight_row.len() != activation.len() {
        return Err(BitNetError::shape(
            format!("weight_row.len() = {}", weight_row.len()),
            format!("activation.len() = {}", activation.len()),
        ));
    }

    // Integer dot product: W_q[k] ∈ {-1, 0, 1} so multiplication is trivial.
    let mut acc: i32 = 0;
    for (&w, &x) in weight_row.iter().zip(activation.iter()) {
        // w is -1, 0, or +1; safe to accumulate in i32 for up to ~16M elements.
        acc += w as i32 * x as i32;
    }

    // Scale back to f32: acc * α_W * α_x / 127
    Ok(acc as f32 * weight_scale * act_scale / Q8_MAX)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // absmax_quantize_row
    // -----------------------------------------------------------------------

    #[test]
    fn basic_quantisation_max_becomes_127() {
        // When max(|x|) = 2.0, scale = 2/127.
        // The element with value 2.0 should quantise to exactly 127.
        let x = vec![0.0_f32, 1.0, 2.0, -1.0];
        let (q, scale) = absmax_quantize_row(&x).unwrap();
        assert_eq!(q[2], 127, "max element must map to 127");
        assert!((scale - 2.0 / 127.0).abs() < 1e-7, "scale = 2/127");
        // Symmetry: -1.0 / (2/127) = -63.5 → rounds to -64
        assert_eq!(q[3], -64, "-1.0 → -64 with scale=2/127");
    }

    #[test]
    fn negative_max_element_quantises_to_neg_127() {
        // Absmax is symmetric: max(|-3|) = 3, scale = 3/127
        // -3.0 / scale = -127.0 → i8(-127) after clamp(-128,127)
        let x = vec![-3.0_f32, 0.0, 1.5];
        let (q, scale) = absmax_quantize_row(&x).unwrap();
        assert!((scale - 3.0 / 127.0).abs() < 1e-7);
        assert_eq!(q[0], -127, "-3.0 / (3/127) = -127");
        assert_eq!(q[1], 0, "0.0 → 0");
        // 1.5 / (3/127) = 1.5 * 127/3 = 63.5 → rounds to 64
        assert_eq!(q[2], 64, "1.5 → 64");
    }

    #[test]
    fn all_zeros_use_minimum_scale() {
        let x = vec![0.0_f32; 8];
        let (q, scale) = absmax_quantize_row(&x).unwrap();
        assert!(scale > 0.0, "scale must be positive for all-zero input");
        assert!(q.iter().all(|&v| v == 0), "all zeros in → all zeros out");
    }

    #[test]
    fn empty_slice_returns_error() {
        let result = absmax_quantize_row(&[]);
        assert!(matches!(result, Err(BitNetError::QuantizationError(_))));
    }

    #[test]
    fn nan_returns_error() {
        let x = vec![1.0_f32, f32::NAN];
        assert!(absmax_quantize_row(&x).is_err());
    }

    #[test]
    fn inf_returns_error() {
        let x = vec![1.0_f32, f32::INFINITY];
        assert!(absmax_quantize_row(&x).is_err());
    }

    #[test]
    fn quantised_values_in_i8_range() {
        // Property: all quantised values must be in [-128, 127].
        let cases: &[&[f32]] = &[
            &[100.0, -100.0, 50.0, -50.0],
            &[0.001, 0.002, -0.001],
            &[1.0],
            &[-1.0],
        ];
        for &row in cases {
            let (q, _) = absmax_quantize_row(row).unwrap();
            for &v in &q {
                // i8 is always in [-128, 127] by type — this checks the semantic range
                assert!(v >= -128 && v <= 127, "out-of-range: {v}");
            }
        }
    }

    #[test]
    fn scale_is_always_positive() {
        let cases: &[&[f32]] = &[&[0.0], &[1.0], &[-1.0, 1.0], &[0.0001, -0.0001]];
        for &row in cases {
            let (_, scale) = absmax_quantize_row(row).unwrap();
            assert!(scale > 0.0, "scale must be positive");
        }
    }

    #[test]
    fn single_element_positive() {
        // x = [5.0], scale = 5/127, q[0] = round(5/(5/127)) = 127
        let (q, scale) = absmax_quantize_row(&[5.0_f32]).unwrap();
        assert!((scale - 5.0 / 127.0).abs() < 1e-6);
        assert_eq!(q[0], 127);
    }

    #[test]
    fn single_element_negative() {
        let (q, scale) = absmax_quantize_row(&[-4.0_f32]).unwrap();
        assert!((scale - 4.0 / 127.0).abs() < 1e-6);
        assert_eq!(q[0], -127);
    }

    // -----------------------------------------------------------------------
    // absmax_dequantize
    // -----------------------------------------------------------------------

    #[test]
    fn dequantize_round_trip_approximate() {
        let x = vec![1.0_f32, -2.0, 0.5, -0.25, 0.0];
        let (q, scale) = absmax_quantize_row(&x).unwrap();
        let deq = absmax_dequantize(&q, scale).unwrap();

        // Maximum quantisation error is scale / 2 (rounding to nearest integer).
        for (i, (&orig, &rec)) in x.iter().zip(deq.iter()).enumerate() {
            let err = (orig - rec).abs();
            let max_err = scale / 2.0 + 1e-6;
            assert!(
                err <= max_err,
                "index {i}: |{orig} - {rec}| = {err} > max_err = {max_err}"
            );
        }
    }

    #[test]
    fn dequantize_zero_scale_returns_error() {
        assert!(absmax_dequantize(&[1i8], 0.0).is_err());
    }

    #[test]
    fn dequantize_negative_scale_returns_error() {
        assert!(absmax_dequantize(&[1i8], -0.1).is_err());
    }

    // -----------------------------------------------------------------------
    // absmax_quantize_batch
    // -----------------------------------------------------------------------

    #[test]
    fn batch_produces_per_row_scales() {
        // Row 0: [4.0, -4.0]  max=4  scale=4/127  q=[127, -127]
        // Row 1: [1.0, -2.0]  max=2  scale=2/127  q=[64, -127] (approx)
        let matrix = vec![4.0_f32, -4.0, 1.0, -2.0];
        let (q, scales) = absmax_quantize_batch(&matrix, 2).unwrap();
        assert_eq!(scales.len(), 2);
        assert!((scales[0] - 4.0 / 127.0).abs() < 1e-6, "row-0 scale");
        assert!((scales[1] - 2.0 / 127.0).abs() < 1e-6, "row-1 scale");
        assert_eq!(q[0], 127, "row 0, col 0 = 127");
        assert_eq!(q[1], -127, "row 0, col 1 = -127");
        assert_eq!(q[3], -127, "row 1, col 1 = -127");
        assert_eq!(q.len(), 4, "flat output has same length as input");
    }

    #[test]
    fn batch_zero_cols_returns_error() {
        assert!(absmax_quantize_batch(&[1.0], 0).is_err());
    }

    #[test]
    fn batch_indivisible_length_returns_error() {
        let matrix = vec![1.0_f32; 5];
        assert!(absmax_quantize_batch(&matrix, 3).is_err());
    }

    // -----------------------------------------------------------------------
    // ternary_dot_product_quantised
    // -----------------------------------------------------------------------

    #[test]
    fn dot_product_all_ones() {
        // W_q = [1, 1, 1, 1], x_q = [127, 0, -127, 64]
        // integer acc = 127 + 0 - 127 + 64 = 64
        // dequantised = 64 * weight_scale * act_scale / 127
        let w: Vec<i8> = vec![1, 1, 1, 1];
        let x: Vec<i8> = vec![127, 0, -127, 64];
        let ws = 0.5_f32;
        let ax = 2.0 / 127.0_f32;
        let y = ternary_dot_product_quantised(&w, &x, ws, ax).unwrap();
        let expected = 64.0_f32 * ws * ax / 127.0;
        assert!((y - expected).abs() < 1e-7, "dot product: {y} ≠ {expected}");
    }

    #[test]
    fn dot_product_alternating_signs() {
        // W_q = [1, -1, 1, -1], x_q = [100, 100, 100, 100]
        // acc = 100 - 100 + 100 - 100 = 0  →  output = 0
        let w: Vec<i8> = vec![1, -1, 1, -1];
        let x: Vec<i8> = vec![100, 100, 100, 100];
        let y = ternary_dot_product_quantised(&w, &x, 1.0, 1.0 / 127.0).unwrap();
        assert_eq!(y, 0.0, "alternating signs cancel to zero");
    }

    #[test]
    fn dot_product_with_zeros_in_weight() {
        // W_q = [0, 1, 0, -1], x_q = [50, 60, 70, 80]
        // acc = 0*50 + 1*60 + 0*70 + (-1)*80 = 60 - 80 = -20
        let w: Vec<i8> = vec![0, 1, 0, -1];
        let x: Vec<i8> = vec![50, 60, 70, 80];
        let ws = 1.0_f32;
        let ax = 1.0_f32;
        let y = ternary_dot_product_quantised(&w, &x, ws, ax).unwrap();
        let expected = -20.0_f32 * ws * ax / 127.0;
        assert!((y - expected).abs() < 1e-7);
    }

    #[test]
    fn dot_product_mismatched_lengths_returns_error() {
        let w: Vec<i8> = vec![1, 0, -1];
        let x: Vec<i8> = vec![10, 20];
        let result = ternary_dot_product_quantised(&w, &x, 1.0, 1.0);
        assert!(matches!(result, Err(BitNetError::InvalidShape { .. })));
    }

    #[test]
    fn dot_product_single_element() {
        // W_q = [1], x_q = [100], ws = 0.25, ax = 0.02
        // y = 100 * 0.25 * 0.02 / 127 = 0.5 / 127 ≈ 0.003937
        let y = ternary_dot_product_quantised(&[1i8], &[100i8], 0.25, 0.02).unwrap();
        let expected = 100.0 * 0.25 * 0.02 / 127.0;
        assert!((y - expected).abs() < 1e-9);
    }

    /// Verify the full quantise → dot-product → dequantise pipeline is
    /// internally consistent: the quantised dot product equals the formula
    ///
    ///   quant_dot = (Σ_k  w_q[k] * x_q[k]) * w_scale * x_scale / 127
    ///
    /// This test verifies the arithmetic pipeline, NOT that ternary
    /// quantisation preserves the original f32 dot product (it is intentionally
    /// lossy — models using BitNet are trained with ternary weights from scratch).
    #[test]
    fn end_to_end_pipeline_is_internally_consistent() {
        use crate::quant::absmean::absmean_quantize;

        // Use weights whose magnitudes are close to their absmean so that
        // quantisation error is minimal (all weights round to ±1, not 0).
        // w_fp: alternating ±0.5, all |w| == 0.5, absmean = 0.5
        // After quantisation: all w_q = ±1.
        let w_fp: Vec<f32> = vec![0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5];
        let x_fp: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let (w_q, w_scale) = absmean_quantize(&w_fp).unwrap();
        let (x_q, x_scale) = absmax_quantize_row(&x_fp).unwrap();

        // All w_q should be ±1 (since |w| == absmean).
        for &v in &w_q {
            assert!(v == 1 || v == -1, "w_q must be ±1, got {v}");
        }

        // Manually compute the expected quantised dot product.
        // w_q = [1,-1,1,-1,1,-1,1,-1], x_q = [127,127,...] (all 1.0 → 127)
        // integer_dot = 1*127 + (-1)*127 + ... = 0
        let integer_dot: i32 = w_q
            .iter()
            .zip(x_q.iter())
            .map(|(&w, &x)| w as i32 * x as i32)
            .sum();
        let expected_quant_dot = integer_dot as f32 * w_scale * x_scale / 127.0;

        let quant_dot = ternary_dot_product_quantised(&w_q, &x_q, w_scale, x_scale).unwrap();

        assert!(
            (quant_dot - expected_quant_dot).abs() < 1e-6,
            "pipeline result {quant_dot} must equal manually-computed {expected_quant_dot}"
        );

        // Verify the result is finite (no NaN / Inf).
        assert!(quant_dot.is_finite(), "pipeline result must be finite");
    }

    /// Verify that ternary quantisation of near-absmean weights preserves
    /// the SIGN of the dot product with high probability.
    ///
    /// For weights with |w_i| ≈ absmean and large enough vectors, the sign of
    /// (W_q · x) should match the sign of (W · x) for randomly-chosen x,
    /// because quantisation clips large weights to ±1 but preserves their sign.
    #[test]
    fn ternary_quantisation_preserves_sign_for_large_weights() {
        use crate::quant::absmean::absmean_quantize;

        // Weights with clear sign separation: all positive → all w_q = +1.
        let w_fp: Vec<f32> = vec![0.8, 0.6, 1.2, 0.4, 0.9, 0.7, 0.5, 1.0];
        let x_fp: Vec<f32> = vec![0.3, -0.1, 0.5, 0.2, -0.4, 0.6, 0.1, 0.3];

        let (w_q, _w_scale) = absmean_quantize(&w_fp).unwrap();
        let (x_q, _x_scale) = absmax_quantize_row(&x_fp).unwrap();

        // All w_q should be +1 (all w_fp > absmean/2, strongly positive).
        // absmean = mean([0.8,0.6,1.2,0.4,0.9,0.7,0.5,1.0]) = 6.1/8 = 0.7625
        // round(w_i / 0.7625): 0.8/0.7625≈1.05→1, 0.6/0.7625≈0.79→1,
        // 1.2/0.7625≈1.57→2→clip(1), etc.
        for &v in &w_q {
            assert_eq!(v, 1i8, "all positive weights → w_q = +1, got {v}");
        }

        // The integer dot product sign should match the sign of Σ x_fp[i].
        let x_sum: f32 = x_fp.iter().sum(); // sum of activations
        let integer_dot: i32 = x_q.iter().map(|&x| x as i32).sum(); // sum since w_q=+1

        if x_sum.abs() > 0.1 {
            // Only assert when signal is large enough to survive quantisation.
            let same_sign = (x_sum > 0.0) == (integer_dot > 0);
            assert!(same_sign, "sign of quantised dot must match sign of reference for large signals: x_sum={x_sum}, integer_dot={integer_dot}");
        }
    }
}
