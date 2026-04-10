//! Ternary General Matrix–Vector Product (GEMV) for the CPU backend.
//!
//! # Mathematical Specification
//!
//! Given:
//! - **W** ∈ {−1, 0, +1}^{M×K}  — ternary weight matrix (row-major, `i8`)
//! - **x** ∈ ℝ^K                 — input activation vector (`f32`)
//! - α_W ∈ ℝ_{>0}               — per-tensor absmean weight scale
//!
//! Compute:
//! ```text
//! y[i] = α_W · Σ_{j=0}^{K-1}  W[i,j] · x[j]   for i = 0 … M-1
//! ```
//!
//! Because W[i,j] ∈ {−1, 0, +1}, each inner-product term reduces to either
//! addition, subtraction, or a no-op — no multiplications needed per element.
//!
//! # Parallelism Strategy
//!
//! The outer loop over output neurons `i` is embarrassingly parallel.
//! Rayon's `par_iter_mut` distributes rows across all available threads.
//! Each thread computes one or more complete dot products independently,
//! so there are no race conditions and no inter-thread synchronisation on
//! the inner loop.
//!
//! # Numerical Accuracy
//!
//! The integer accumulator uses `i32` to avoid overflow.  For the worst case
//! (all weights = ±1, all activations = ±128 after int8 quantisation), an
//! `i32` accumulator can handle up to ~16 M elements without overflow
//! (2^31 / 128 ≈ 16.7 M), far exceeding any realistic hidden dimension.
//!
//! The final scaling `acc as f32 * weight_scale` converts back to `f32` after
//! the accumulation, minimising floating-point rounding error.
//!
//! # Invariants
//!
//! - `weight.len() == out_features * in_features`
//! - `input.len() == in_features`
//! - `output.len() == out_features`
//! - `weight[i] ∈ {-1, 0, 1}` for all i (assumed; validated only in debug)
//! - `weight_scale > 0`

use bitnet_core::error::{BitNetError, Result};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the ternary GEMV: `output = W · input * weight_scale`.
///
/// Uses Rayon to parallelise the outer loop over `out_features` output rows.
///
/// # Arguments
///
/// - `weight`:        Row-major `i8` slice, shape `[out_features, in_features]`.
///                    Every element must be in `{-1, 0, +1}`.
/// - `weight_scale`:  Per-tensor absmean scale α_W (must be `> 0`).
/// - `input`:         Input vector, length `in_features`.
/// - `output`:        Pre-allocated output buffer, length `out_features`.
///                    Overwritten on success.
/// - `out_features`:  Number of output neurons (matrix rows).
/// - `in_features`:   Number of input features (matrix columns).
///
/// # Errors
///
/// Returns [`BitNetError::InvalidShape`] if any slice length does not match
/// the declared dimensions.
///
/// Returns [`BitNetError::QuantizationError`] if `weight_scale <= 0` or
/// is non-finite.
///
/// # Example
///
/// ```
/// use bitnet_cpu::gemv::ternary_gemv_f32;
///
/// // 2×3 weight matrix: [[1, 0, -1], [-1, 1, 0]]
/// let weight: Vec<i8> = vec![1, 0, -1, -1, 1, 0];
/// let input = vec![2.0_f32, 3.0, 4.0];
/// let mut output = vec![0.0_f32; 2];
///
/// // Row 0: 1*2 + 0*3 + (-1)*4 = -2,  scaled by 0.5 → -1.0
/// // Row 1: (-1)*2 + 1*3 + 0*4 =  1,  scaled by 0.5 →  0.5
/// ternary_gemv_f32(&weight, 0.5, &input, &mut output, 2, 3).unwrap();
/// assert!((output[0] - (-1.0)).abs() < 1e-6);
/// assert!((output[1] - 0.5).abs() < 1e-6);
/// ```
pub fn ternary_gemv_f32(
    weight: &[i8],
    weight_scale: f32,
    input: &[f32],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) -> Result<()> {
    // ---- Validation ---------------------------------------------------------
    let expected_weight_len = out_features
        .checked_mul(in_features)
        .ok_or_else(|| BitNetError::shape("weight size fits usize", "overflow"))?;

    if weight.len() != expected_weight_len {
        return Err(BitNetError::shape(
            format!("weight.len() == {out_features} * {in_features} = {expected_weight_len}"),
            format!("weight.len() == {}", weight.len()),
        ));
    }
    if input.len() != in_features {
        return Err(BitNetError::shape(
            format!("input.len() == {in_features}"),
            format!("input.len() == {}", input.len()),
        ));
    }
    if output.len() != out_features {
        return Err(BitNetError::shape(
            format!("output.len() == {out_features}"),
            format!("output.len() == {}", output.len()),
        ));
    }
    if weight_scale <= 0.0 || !weight_scale.is_finite() {
        return Err(BitNetError::quant(format!(
            "weight_scale must be finite and > 0, got {weight_scale}"
        )));
    }

    // ---- Parallel GEMV ------------------------------------------------------
    //
    // Split `output` into per-row slices and process each in parallel.
    // The closure captures shared references to `weight`, `input`, and
    // `weight_scale` — all are `Sync`, so no data races can occur.
    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(row_idx, out_elem)| {
            let row_start = row_idx * in_features;
            let row = &weight[row_start..row_start + in_features];
            *out_elem = dot_ternary_f32(row, input) * weight_scale;
        });

    Ok(())
}

// ---------------------------------------------------------------------------
// Inner kernel: single-row ternary dot product
// ---------------------------------------------------------------------------

/// Compute the dot product of a ternary weight row and an f32 activation vector.
///
/// ```text
/// result = Σ_j  weight[j] as f32 · input[j]
/// ```
///
/// Uses an `f32` accumulator directly since the ternary multiplication is free
/// (add, subtract, or skip).  The caller applies `weight_scale` after.
///
/// # Panics (debug only)
/// Panics if `weight.len() != input.len()`.
#[inline]
pub fn dot_ternary_f32(weight: &[i8], input: &[f32]) -> f32 {
    debug_assert_eq!(
        weight.len(),
        input.len(),
        "dot_ternary_f32: weight and input lengths must match"
    );

    let mut acc = 0.0_f32;
    for (&w, &x) in weight.iter().zip(input.iter()) {
        // w ∈ {-1, 0, 1}: branch-free using cast and multiply.
        // On x86/ARM with fast FMA, the compiler may fuse this into FMADD.
        acc += w as f32 * x;
    }
    acc
}

/// Compute the dot product of a ternary weight row and an i8 quantised
/// activation vector using an integer accumulator.
///
/// ```text
/// result = Σ_j  weight[j] as i32 · activation[j] as i32
/// ```
///
/// The integer accumulator avoids any floating-point operations in the inner
/// loop, matching the W2A8 kernel design.  The caller applies both the weight
/// scale and activation scale to convert back to f32.
///
/// # Panics (debug only)
/// Panics if `weight.len() != activation.len()`.
#[inline]
pub fn dot_ternary_i8(weight: &[i8], activation: &[i8]) -> i32 {
    debug_assert_eq!(
        weight.len(),
        activation.len(),
        "dot_ternary_i8: weight and activation lengths must match"
    );

    let mut acc: i32 = 0;
    for (&w, &x) in weight.iter().zip(activation.iter()) {
        // w ∈ {-1, 0, 1}; x ∈ [-128, 127]
        // No overflow risk: |w * x| ≤ 128, sum over K ≤ 2560 → max |acc| ≤ 327 680 << i32::MAX
        acc += (w as i32) * (x as i32);
    }
    acc
}

/// Quantised GEMV variant: `output[i] = dot(W[i], x_q) * weight_scale * act_scale / 127.0`.
///
/// Combines a ternary weight matrix with an 8-bit quantised activation vector.
/// The result is dequantised to `f32` using both scales.
///
/// # Arguments
///
/// - `weight`:        Row-major ternary `i8` slice `[out_features, in_features]`.
/// - `weight_scale`:  Per-tensor absmean weight scale α_W.
/// - `activation`:    Per-token absmax quantised activation `i8` slice `[in_features]`.
/// - `act_scale`:     Per-token activation scale α_x = max(|x|) / 127.
/// - `output`:        Pre-allocated `f32` output buffer `[out_features]`.
/// - `out_features`:  Matrix rows.
/// - `in_features`:   Matrix columns.
///
/// # Errors
///
/// Returns [`BitNetError::InvalidShape`] for shape mismatches.
/// Returns [`BitNetError::QuantizationError`] for invalid scales.
pub fn ternary_gemv_quantised(
    weight: &[i8],
    weight_scale: f32,
    activation: &[i8],
    act_scale: f32,
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) -> Result<()> {
    let expected_weight_len = out_features
        .checked_mul(in_features)
        .ok_or_else(|| BitNetError::shape("weight size fits usize", "overflow"))?;

    if weight.len() != expected_weight_len {
        return Err(BitNetError::shape(
            format!("{out_features} * {in_features} = {expected_weight_len}"),
            format!("{}", weight.len()),
        ));
    }
    if activation.len() != in_features {
        return Err(BitNetError::shape(
            format!("activation.len() == {in_features}"),
            format!("activation.len() == {}", activation.len()),
        ));
    }
    if output.len() != out_features {
        return Err(BitNetError::shape(
            format!("output.len() == {out_features}"),
            format!("output.len() == {}", output.len()),
        ));
    }
    if weight_scale <= 0.0 || !weight_scale.is_finite() {
        return Err(BitNetError::quant(format!(
            "weight_scale must be finite > 0, got {weight_scale}"
        )));
    }
    if act_scale <= 0.0 || !act_scale.is_finite() {
        return Err(BitNetError::quant(format!(
            "act_scale must be finite > 0, got {act_scale}"
        )));
    }

    // Combined dequantisation factor: α_W * α_x / 127
    let combined_scale = weight_scale * act_scale / 127.0_f32;

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(row_idx, out_elem)| {
            let row_start = row_idx * in_features;
            let row = &weight[row_start..row_start + in_features];
            let acc = dot_ternary_i8(row, activation);
            *out_elem = acc as f32 * combined_scale;
        });

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // dot_ternary_f32
    // -----------------------------------------------------------------------

    #[test]
    fn dot_f32_all_ones_weight() {
        // w = [1, 1, 1], x = [2, 3, 4] → 2+3+4 = 9
        let w: Vec<i8> = vec![1, 1, 1];
        let x: Vec<f32> = vec![2.0, 3.0, 4.0];
        let result = dot_ternary_f32(&w, &x);
        assert!((result - 9.0).abs() < 1e-6, "expected 9.0, got {result}");
    }

    #[test]
    fn dot_f32_all_neg_ones_weight() {
        // w = [-1, -1], x = [5, 3] → -5 + (-3) = -8
        let w: Vec<i8> = vec![-1, -1];
        let x: Vec<f32> = vec![5.0, 3.0];
        let result = dot_ternary_f32(&w, &x);
        assert!(
            (result - (-8.0)).abs() < 1e-6,
            "expected -8.0, got {result}"
        );
    }

    #[test]
    fn dot_f32_zeros_in_weight() {
        // w = [1, 0, -1, 0], x = [1, 100, 2, 100] → 1*1 + 0*100 + (-1)*2 + 0*100 = -1
        let w: Vec<i8> = vec![1, 0, -1, 0];
        let x: Vec<f32> = vec![1.0, 100.0, 2.0, 100.0];
        let result = dot_ternary_f32(&w, &x);
        assert!(
            (result - (-1.0)).abs() < 1e-6,
            "expected -1.0, got {result}"
        );
    }

    #[test]
    fn dot_f32_all_zero_weight_produces_zero() {
        let w: Vec<i8> = vec![0; 8];
        let x: Vec<f32> = vec![99.0; 8];
        assert_eq!(dot_ternary_f32(&w, &x), 0.0);
    }

    #[test]
    fn dot_f32_single_element() {
        let w: Vec<i8> = vec![-1];
        let x: Vec<f32> = vec![7.5];
        assert!((dot_ternary_f32(&w, &x) - (-7.5)).abs() < 1e-7);
    }

    // -----------------------------------------------------------------------
    // dot_ternary_i8
    // -----------------------------------------------------------------------

    #[test]
    fn dot_i8_basic() {
        // w = [1, -1, 0], x = [100, 50, 127] → 100 - 50 + 0 = 50
        let w: Vec<i8> = vec![1, -1, 0];
        let x: Vec<i8> = vec![100, 50, 127];
        assert_eq!(dot_ternary_i8(&w, &x), 50);
    }

    #[test]
    fn dot_i8_all_ones_times_all_127() {
        // w = [1; 4], x = [127; 4] → 4 * 127 = 508
        let w: Vec<i8> = vec![1; 4];
        let x: Vec<i8> = vec![127; 4];
        assert_eq!(dot_ternary_i8(&w, &x), 508);
    }

    #[test]
    fn dot_i8_cancellation() {
        // w = [1, 1, -1, -1], x = [64; 4] → 64+64-64-64 = 0
        let w: Vec<i8> = vec![1, 1, -1, -1];
        let x: Vec<i8> = vec![64; 4];
        assert_eq!(dot_ternary_i8(&w, &x), 0);
    }

    // -----------------------------------------------------------------------
    // ternary_gemv_f32
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_2x3() {
        // W = [[1, 0, -1], [-1, 1, 0]], x = [2, 3, 4], scale = 0.5
        // row 0: 1*2 + 0*3 + (-1)*4 = -2  → -2 * 0.5 = -1.0
        // row 1: (-1)*2 + 1*3 + 0*4 =  1  →  1 * 0.5 =  0.5
        let weight: Vec<i8> = vec![1, 0, -1, -1, 1, 0];
        let input = vec![2.0_f32, 3.0, 4.0];
        let mut output = vec![0.0_f32; 2];
        ternary_gemv_f32(&weight, 0.5, &input, &mut output, 2, 3).unwrap();
        assert!((output[0] - (-1.0)).abs() < 1e-6, "row 0: {}", output[0]);
        assert!((output[1] - 0.5).abs() < 1e-6, "row 1: {}", output[1]);
    }

    #[test]
    fn gemv_identity_weight() {
        // 3×3 identity (all zeros except diag=1)
        // W = [[1,0,0],[0,1,0],[0,0,1]], x = [5, 7, 9], scale = 1.0
        let weight: Vec<i8> = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        let input = vec![5.0_f32, 7.0, 9.0];
        let mut output = vec![0.0_f32; 3];
        ternary_gemv_f32(&weight, 1.0, &input, &mut output, 3, 3).unwrap();
        assert!((output[0] - 5.0).abs() < 1e-6);
        assert!((output[1] - 7.0).abs() < 1e-6);
        assert!((output[2] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn gemv_negation_weight() {
        // All weights = -1: output = -input * scale
        let n = 4;
        let weight: Vec<i8> = vec![-1; n];
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0_f32; 1];
        ternary_gemv_f32(&weight, 2.0, &input, &mut output, 1, n).unwrap();
        // dot = -(1+2+3+4) = -10, scaled by 2 = -20
        assert!((output[0] - (-20.0)).abs() < 1e-5, "got {}", output[0]);
    }

    #[test]
    fn gemv_scale_applied_correctly() {
        // Single output neuron, all weights = 1
        let weight: Vec<i8> = vec![1, 1, 1];
        let input = vec![1.0_f32, 1.0, 1.0]; // sum = 3
        let mut output = vec![0.0_f32; 1];
        ternary_gemv_f32(&weight, 0.25, &input, &mut output, 1, 3).unwrap();
        // 3 * 0.25 = 0.75
        assert!((output[0] - 0.75).abs() < 1e-7, "got {}", output[0]);
    }

    #[test]
    fn gemv_wrong_weight_length_returns_error() {
        let weight: Vec<i8> = vec![1; 5]; // should be 6
        let input = vec![1.0_f32; 3];
        let mut output = vec![0.0_f32; 2];
        let err = ternary_gemv_f32(&weight, 1.0, &input, &mut output, 2, 3).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::InvalidShape { .. }
        ));
    }

    #[test]
    fn gemv_wrong_input_length_returns_error() {
        let weight: Vec<i8> = vec![1; 6]; // 2x3
        let input = vec![1.0_f32; 4]; // should be 3
        let mut output = vec![0.0_f32; 2];
        let err = ternary_gemv_f32(&weight, 1.0, &input, &mut output, 2, 3).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::InvalidShape { .. }
        ));
    }

    #[test]
    fn gemv_wrong_output_length_returns_error() {
        let weight: Vec<i8> = vec![1; 6]; // 2x3
        let input = vec![1.0_f32; 3];
        let mut output = vec![0.0_f32; 3]; // should be 2
        let err = ternary_gemv_f32(&weight, 1.0, &input, &mut output, 2, 3).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::InvalidShape { .. }
        ));
    }

    #[test]
    fn gemv_zero_weight_scale_returns_error() {
        let weight: Vec<i8> = vec![1; 4];
        let input = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 1];
        let err = ternary_gemv_f32(&weight, 0.0, &input, &mut output, 1, 4).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::QuantizationError(_)
        ));
    }

    #[test]
    fn gemv_negative_weight_scale_returns_error() {
        let weight: Vec<i8> = vec![1; 4];
        let input = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 1];
        let err = ternary_gemv_f32(&weight, -0.5, &input, &mut output, 1, 4).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::QuantizationError(_)
        ));
    }

    #[test]
    fn gemv_single_row_single_col() {
        let weight: Vec<i8> = vec![-1];
        let input = vec![3.0_f32];
        let mut output = vec![0.0_f32; 1];
        ternary_gemv_f32(&weight, 2.0, &input, &mut output, 1, 1).unwrap();
        // -1 * 3.0 * 2.0 = -6.0
        assert!((output[0] - (-6.0)).abs() < 1e-7, "got {}", output[0]);
    }

    #[test]
    fn gemv_all_zero_weights_produce_zero_output() {
        let weight: Vec<i8> = vec![0; 12]; // 3x4
        let input = vec![99.0_f32; 4];
        let mut output = vec![1.0_f32; 3]; // non-zero sentinel
        ternary_gemv_f32(&weight, 1.0, &input, &mut output, 3, 4).unwrap();
        assert!(
            output.iter().all(|&v| v == 0.0),
            "all-zero weights must produce zero output"
        );
    }

    // -----------------------------------------------------------------------
    // ternary_gemv_quantised
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_quantised_is_internally_consistent() {
        use bitnet_core::quant::{absmax_quantize_row, absmean_quantize};

        // Verify that ternary_gemv_quantised computes exactly:
        //   output[i] = (Σ_j  w_q[i,j] * x_q[j]) * w_scale * x_scale / 127
        //
        // We use weights whose ternary encoding is unambiguous (magnitudes
        // clearly above and below absmean so rounding is deterministic):
        //   W row 0: [+1.0, -1.0, +1.0, -1.0]  all |w| = 1.0, absmean = 1.0
        //            → w_q = [1, -1, 1, -1]
        //   W row 1: [+1.0, +1.0, -1.0, -1.0]
        //            → w_q = [1, 1, -1, -1]
        //   x: [2.0, 2.0, 2.0, 2.0]  → x_q = [127, 127, 127, 127], x_scale = 2/127
        let w_f32: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
        let x_f32: Vec<f32> = vec![2.0, 2.0, 2.0, 2.0];

        let (w_q, w_scale) = absmean_quantize(&w_f32).unwrap();
        let (x_q, x_scale) = absmax_quantize_row(&x_f32).unwrap();

        // Verify quantisation produced expected ternary values.
        assert_eq!(
            w_q,
            vec![1i8, -1, 1, -1, 1, 1, -1, -1],
            "w_q must match expected ternary"
        );
        assert!(
            (w_scale - 1.0).abs() < 1e-5,
            "w_scale must be 1.0 (absmean of all-±1)"
        );

        // Manually compute expected output using the formula.
        // Row 0: w_q = [1,-1,1,-1], x_q = [127,127,127,127]
        //   integer_dot = 127 - 127 + 127 - 127 = 0
        //   output[0] = 0 * w_scale * x_scale / 127 = 0.0
        let row0_int: i32 = w_q[0..4]
            .iter()
            .zip(x_q.iter())
            .map(|(&w, &x)| w as i32 * x as i32)
            .sum();
        let expected_row0 = row0_int as f32 * w_scale * x_scale / 127.0;

        // Row 1: w_q = [1,1,-1,-1], x_q = [127,127,127,127]
        //   integer_dot = 127 + 127 - 127 - 127 = 0
        //   output[1] = 0.0
        let row1_int: i32 = w_q[4..8]
            .iter()
            .zip(x_q.iter())
            .map(|(&w, &x)| w as i32 * x as i32)
            .sum();
        let expected_row1 = row1_int as f32 * w_scale * x_scale / 127.0;

        // Run the quantised GEMV.
        let mut quant_output = vec![0.0_f32; 2];
        ternary_gemv_quantised(&w_q, w_scale, &x_q, x_scale, &mut quant_output, 2, 4).unwrap();

        assert!(
            (quant_output[0] - expected_row0).abs() < 1e-6,
            "row 0: got {}, expected {}",
            quant_output[0],
            expected_row0
        );
        assert!(
            (quant_output[1] - expected_row1).abs() < 1e-6,
            "row 1: got {}, expected {}",
            quant_output[1],
            expected_row1
        );

        // All outputs must be finite.
        for (i, &v) in quant_output.iter().enumerate() {
            assert!(v.is_finite(), "quant_output[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn gemv_quantised_nonzero_dot_product() {
        use bitnet_core::quant::{absmax_quantize_row, absmean_quantize};

        // Test with a non-zero integer dot product so we verify the scaling formula.
        // W row 0: [+1.0, +1.0, +1.0, +1.0]  → w_q = [1,1,1,1], w_scale = 1.0
        // x: [1.0, 0.0, 0.0, 0.0]             → x_q[0] = 127, rest = 0, x_scale = 1/127
        // integer_dot = 1*127 + 1*0 + 1*0 + 1*0 = 127
        // output[0] = 127 * 1.0 * (1/127) / 127 = 1/127 ≈ 0.00787
        let w_f32: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let x_f32: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];

        let (w_q, w_scale) = absmean_quantize(&w_f32).unwrap();
        let (x_q, x_scale) = absmax_quantize_row(&x_f32).unwrap();

        let mut quant_output = vec![0.0_f32; 1];
        ternary_gemv_quantised(&w_q, w_scale, &x_q, x_scale, &mut quant_output, 1, 4).unwrap();

        // Expected: 127 * 1.0 * (1.0/127) / 127 = 1/127
        let expected = 127.0_f32 * w_scale * x_scale / 127.0;
        assert!(
            (quant_output[0] - expected).abs() < 1e-6,
            "got {}, expected {}",
            quant_output[0],
            expected
        );
        assert!(quant_output[0].is_finite());
    }

    #[test]
    fn gemv_quantised_invalid_act_scale_returns_error() {
        let weight: Vec<i8> = vec![1; 4];
        let activation: Vec<i8> = vec![100; 4];
        let mut output = vec![0.0_f32; 1];
        let err =
            ternary_gemv_quantised(&weight, 1.0, &activation, 0.0, &mut output, 1, 4).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::QuantizationError(_)
        ));
    }

    /// Property test: GEMV output scales linearly with weight_scale.
    ///
    /// Theorem: ternary_gemv(W, 2α, x) = 2 · ternary_gemv(W, α, x)
    #[test]
    fn gemv_output_linear_in_scale() {
        let weight: Vec<i8> = vec![1, -1, 0, 1, 0, -1]; // 2x3
        let input = vec![0.3_f32, -0.7, 1.1];

        let mut out1 = vec![0.0_f32; 2];
        let mut out2 = vec![0.0_f32; 2];

        ternary_gemv_f32(&weight, 0.4, &input, &mut out1, 2, 3).unwrap();
        ternary_gemv_f32(&weight, 0.8, &input, &mut out2, 2, 3).unwrap();

        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (2.0 * a - b).abs() < 1e-5,
                "linearity violated at row {i}: 2*{a}={} ≠ {b}",
                2.0 * a
            );
        }
    }

    /// Property test: GEMV with all-zero input produces zero output.
    #[test]
    fn gemv_zero_input_produces_zero_output() {
        let weight: Vec<i8> = vec![1, -1, 1, 0, 1, -1]; // 2x3
        let input = vec![0.0_f32; 3];
        let mut output = vec![99.0_f32; 2]; // non-zero sentinel

        ternary_gemv_f32(&weight, 1.0, &input, &mut output, 2, 3).unwrap();
        assert!(
            output.iter().all(|&v| v == 0.0),
            "zero input must produce zero output regardless of weights"
        );
    }
}
