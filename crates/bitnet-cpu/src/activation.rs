//! Activation functions for the CPU backend.
//!
//! # Mathematical Definitions
//!
//! ## Squared ReLU (sqrelu)
//!
//! Used as the gating activation in the BitNet b1.58 FFN:
//!
//! ```text
//! sqrelu(x_i) = max(0, x_i)²
//! ```
//!
//! The FFN computes:
//! ```text
//! inner = ffn_sub_norm( sqrelu(gate) ⊙ up )
//! output = down(inner)
//! ```
//!
//! where `⊙` is element-wise multiplication.
//!
//! ## Softmax (numerically stable)
//!
//! Used for attention score normalisation:
//!
//! ```text
//! m      = max_i(x_i)
//! e_i    = exp(x_i − m)
//! out_i  = e_i / Σ_j e_j
//! ```
//!
//! Subtracting the maximum before exponentiation prevents floating-point
//! overflow when scores are large (e.g. after scaling by 1/√head_dim).
//!
//! # Properties
//!
//! ## Squared ReLU
//! - `sqrelu(x) = 0`  for `x ≤ 0`
//! - `sqrelu(x) = x²` for `x > 0`
//! - Gradient: `2 * max(0, x)` (smooth, unlike plain ReLU)
//! - Non-negative everywhere → preserves sign of gate activation
//!
//! ## Softmax
//! - `out_i > 0` for all i
//! - `Σ_i out_i = 1.0` (probability simplex)
//! - Shift-invariant: `softmax(x + c) = softmax(x)` for any scalar c
//! - Empty slice: no-op (returns immediately)
//!
//! # Invariants
//!
//! - All functions operate in-place on `&mut [f32]`
//! - No heap allocation
//! - Finite inputs produce finite outputs (no NaN/Inf propagation for valid inputs)

use bitnet_core::error::{BitNetError, Result};

// ---------------------------------------------------------------------------
// Squared ReLU
// ---------------------------------------------------------------------------

/// Apply squared ReLU element-wise in-place: `x[i] = max(0, x[i])²`.
///
/// This is the activation function used in the BitNet b1.58 FFN gate layer.
/// It is applied to the `gate` projection output before element-wise
/// multiplication with the `up` projection:
///
/// ```text
/// inner[i] = sqrelu(gate[i]) * up[i]
/// ```
///
/// # Properties
///
/// - `sqrelu(x) = 0`   for `x ≤ 0`   (hard threshold, kills negative activations)
/// - `sqrelu(x) = x²`  for `x > 0`   (amplifies large positive values)
/// - Non-negative output: `∀i, x[i] ≥ 0` after this operation
///
/// # Performance
///
/// The implementation avoids conditional branches: `max(0, x)` is computed
/// via `f32::max`, and the squaring is a single multiply.  Compilers typically
/// auto-vectorise this loop to SIMD on x86 (SSE/AVX) and ARM (NEON).
///
/// # Example
///
/// ```
/// use bitnet_cpu::activation::squared_relu;
///
/// let mut x = vec![-2.0_f32, -0.5, 0.0, 1.0, 3.0];
/// squared_relu(&mut x);
/// assert_eq!(x, vec![0.0, 0.0, 0.0, 1.0, 9.0]);
/// ```
#[inline]
pub fn squared_relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let r = v.max(0.0); // ReLU: max(0, x)
        *v = r * r; // square
    }
}

/// Apply squared ReLU to `src` and write results to `dst`.
///
/// `dst[i] = max(0, src[i])²`
///
/// This is the non-in-place variant, useful when the original values need
/// to be preserved (e.g. for the gate/up computation where both tensors are
/// kept).
///
/// # Errors
///
/// Returns [`BitNetError::InvalidShape`] if `src.len() != dst.len()`.
///
/// # Example
///
/// ```
/// use bitnet_cpu::activation::squared_relu_into;
///
/// let src = vec![-1.0_f32, 0.5, 2.0];
/// let mut dst = vec![0.0_f32; 3];
/// squared_relu_into(&src, &mut dst).unwrap();
/// assert_eq!(dst, vec![0.0, 0.25, 4.0]);
/// ```
pub fn squared_relu_into(src: &[f32], dst: &mut [f32]) -> Result<()> {
    if src.len() != dst.len() {
        return Err(BitNetError::shape(
            format!("src.len() == dst.len() = {}", dst.len()),
            format!("src.len() = {}", src.len()),
        ));
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        let r = s.max(0.0);
        *d = r * r;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Element-wise multiplication (gate activation helper)
// ---------------------------------------------------------------------------

/// Compute the gated activation: `out[i] = sqrelu(gate[i]) * up[i]`.
///
/// This is the fused form of the BitNet b1.58 FFN inner computation:
///
/// ```text
/// out[i] = max(0, gate[i])² * up[i]
/// ```
///
/// Writing to a separate output buffer avoids aliasing issues and makes the
/// data flow explicit at call sites.
///
/// # Errors
///
/// Returns [`BitNetError::InvalidShape`] if any slice length differs.
///
/// # Example
///
/// ```
/// use bitnet_cpu::activation::sqrelu_gate;
///
/// let gate = vec![2.0_f32, -1.0, 3.0];
/// let up   = vec![0.5_f32,  2.0, 1.0];
/// let mut out = vec![0.0_f32; 3];
/// sqrelu_gate(&gate, &up, &mut out).unwrap();
/// // gate after sqrelu: [4.0, 0.0, 9.0]
/// // out = [4*0.5, 0*2, 9*1] = [2.0, 0.0, 9.0]
/// assert!((out[0] - 2.0).abs() < 1e-7);
/// assert_eq!(out[1], 0.0);
/// assert!((out[2] - 9.0).abs() < 1e-7);
/// ```
pub fn sqrelu_gate(gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<()> {
    if gate.len() != up.len() {
        return Err(BitNetError::shape(
            format!("gate.len() == up.len() = {}", up.len()),
            format!("gate.len() = {}", gate.len()),
        ));
    }
    if gate.len() != out.len() {
        return Err(BitNetError::shape(
            format!("gate.len() == out.len() = {}", out.len()),
            format!("gate.len() = {}", gate.len()),
        ));
    }
    for i in 0..gate.len() {
        let r = gate[i].max(0.0);
        out[i] = r * r * up[i];
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// Apply numerically-stable softmax in-place over the entire slice.
///
/// ```text
/// m      = max_i(x_i)          (shift for numerical stability)
/// e_i    = exp(x_i − m)        (shifted exponentials)
/// x_i   ← e_i / Σ_j e_j       (normalise to probability simplex)
/// ```
///
/// # Properties
///
/// - All outputs are positive: `x[i] > 0` after this call.
/// - Outputs sum to 1.0 (to floating-point precision).
/// - Shift-invariant: `softmax(x + c) = softmax(x)` for any scalar `c`.
/// - An empty slice is a no-op.
///
/// # Performance
///
/// Three passes over the slice:
/// 1. Find the maximum (reduction).
/// 2. Compute `exp(x_i - max)` and accumulate the sum.
/// 3. Divide by the sum.
///
/// For typical attention scores (`seq_len ≤ 4096`) this fits comfortably in L1
/// cache and the loop is auto-vectorised by the compiler.
///
/// # Example
///
/// ```
/// use bitnet_cpu::activation::softmax;
///
/// let mut x = vec![1.0_f32, 2.0, 3.0];
/// softmax(&mut x);
/// let sum: f32 = x.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1");
/// assert!(x.iter().all(|&v| v > 0.0), "all outputs must be positive");
/// ```
#[inline]
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Pass 1: find maximum for numerical stability.
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Pass 2: compute shifted exponentials and accumulate sum.
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // Pass 3: normalise.
    let inv_sum = sum.recip();
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// Apply softmax over a sub-slice `x[0..len]` in-place, leaving elements
/// beyond `len` unchanged.
///
/// This is used in the attention kernel where only `cur_pos + 1` scores are
/// valid (the rest of the score buffer is scratch space).
///
/// # Errors
///
/// Returns [`BitNetError::InvalidShape`] if `len > x.len()`.
///
/// # Example
///
/// ```
/// use bitnet_cpu::activation::softmax_partial;
///
/// let mut scores = vec![1.0_f32, 3.0, 2.0, 999.0]; // only first 3 are valid
/// softmax_partial(&mut scores, 3).unwrap();
/// let sum: f32 = scores[..3].iter().sum();
/// assert!((sum - 1.0).abs() < 1e-6);
/// assert_eq!(scores[3], 999.0, "element beyond len is untouched");
/// ```
pub fn softmax_partial(x: &mut [f32], len: usize) -> Result<()> {
    if len > x.len() {
        return Err(BitNetError::shape(
            format!("len ({len}) ≤ x.len() = {}", x.len()),
            format!("len = {len}"),
        ));
    }
    softmax(&mut x[..len]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // squared_relu
    // -----------------------------------------------------------------------

    #[test]
    fn squared_relu_negative_inputs_become_zero() {
        let mut x = vec![-10.0_f32, -1.0, -0.001, -f32::MIN_POSITIVE];
        squared_relu(&mut x);
        assert!(
            x.iter().all(|&v| v == 0.0),
            "all negative inputs must become 0, got {:?}",
            x
        );
    }

    #[test]
    fn squared_relu_positive_inputs_are_squared() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 0.5, 10.0];
        let expected = vec![1.0_f32, 4.0, 9.0, 0.25, 100.0];
        squared_relu(&mut x);
        for (i, (&got, &exp)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "index {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn squared_relu_zero_stays_zero() {
        let mut x = vec![0.0_f32; 8];
        squared_relu(&mut x);
        assert!(x.iter().all(|&v| v == 0.0), "zero input must stay zero");
    }

    #[test]
    fn squared_relu_mixed_inputs() {
        let mut x = vec![-3.0_f32, 2.0, 0.0, -0.5, 4.0, -1.0];
        squared_relu(&mut x);
        // Negatives → 0, positives → squared, zero → 0
        assert_eq!(x[0], 0.0, "-3 → 0");
        assert!((x[1] - 4.0).abs() < 1e-6, "2 → 4");
        assert_eq!(x[2], 0.0, "0 → 0");
        assert_eq!(x[3], 0.0, "-0.5 → 0");
        assert!((x[4] - 16.0).abs() < 1e-6, "4 → 16");
        assert_eq!(x[5], 0.0, "-1 → 0");
    }

    #[test]
    fn squared_relu_empty_slice_noop() {
        let mut x: Vec<f32> = vec![];
        squared_relu(&mut x); // must not panic
        assert!(x.is_empty());
    }

    #[test]
    fn squared_relu_single_element_positive() {
        let mut x = vec![5.0_f32];
        squared_relu(&mut x);
        assert!((x[0] - 25.0).abs() < 1e-6, "5² = 25, got {}", x[0]);
    }

    #[test]
    fn squared_relu_single_element_negative() {
        let mut x = vec![-5.0_f32];
        squared_relu(&mut x);
        assert_eq!(x[0], 0.0, "negative single element → 0");
    }

    #[test]
    fn squared_relu_output_always_nonnegative() {
        // Property: ∀ x, sqrelu(x) ≥ 0.
        let cases: &[f32] = &[
            f32::NEG_INFINITY,
            -1e10,
            -1.0,
            -0.001,
            0.0,
            0.001,
            1.0,
            1e10,
        ];
        for &val in cases {
            let mut x = vec![val];
            squared_relu(&mut x);
            assert!(x[0] >= 0.0, "sqrelu({val}) = {} < 0", x[0]);
        }
    }

    #[test]
    fn squared_relu_into_matches_inplace() {
        let src = vec![-2.0_f32, 0.5, 3.0, -1.5, 2.0];
        let mut dst_into = vec![0.0_f32; 5];
        squared_relu_into(&src, &mut dst_into).unwrap();

        let mut inplace = src.clone();
        squared_relu(&mut inplace);

        for (i, (&a, &b)) in dst_into.iter().zip(inplace.iter()).enumerate() {
            assert!((a - b).abs() < 1e-7, "index {i}: into={a}, inplace={b}");
        }
    }

    #[test]
    fn squared_relu_into_mismatched_lengths_returns_error() {
        let src = vec![1.0_f32; 4];
        let mut dst = vec![0.0_f32; 3];
        let err = squared_relu_into(&src, &mut dst).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    // -----------------------------------------------------------------------
    // sqrelu_gate
    // -----------------------------------------------------------------------

    #[test]
    fn sqrelu_gate_basic() {
        // gate = [2, -1, 3], up = [0.5, 2, 1]
        // sqrelu(gate) = [4, 0, 9]
        // out = [4*0.5, 0*2, 9*1] = [2.0, 0.0, 9.0]
        let gate = vec![2.0_f32, -1.0, 3.0];
        let up = vec![0.5_f32, 2.0, 1.0];
        let mut out = vec![0.0_f32; 3];
        sqrelu_gate(&gate, &up, &mut out).unwrap();
        assert!((out[0] - 2.0).abs() < 1e-7, "out[0] = {}", out[0]);
        assert_eq!(out[1], 0.0, "out[1] = {}", out[1]);
        assert!((out[2] - 9.0).abs() < 1e-7, "out[2] = {}", out[2]);
    }

    #[test]
    fn sqrelu_gate_negative_gate_zeroes_output() {
        // All negative gate → sqrelu = 0 → output = 0 regardless of up.
        let gate = vec![-1.0_f32, -2.0, -3.0];
        let up = vec![99.0_f32, 100.0, 101.0];
        let mut out = vec![1.0_f32; 3]; // non-zero sentinel
        sqrelu_gate(&gate, &up, &mut out).unwrap();
        assert!(
            out.iter().all(|&v| v == 0.0),
            "negative gate must zero output, got {:?}",
            out
        );
    }

    #[test]
    fn sqrelu_gate_zero_up_zeroes_output() {
        let gate = vec![5.0_f32, 3.0, 2.0];
        let up = vec![0.0_f32; 3];
        let mut out = vec![99.0_f32; 3];
        sqrelu_gate(&gate, &up, &mut out).unwrap();
        assert!(
            out.iter().all(|&v| v == 0.0),
            "zero up must zero output, got {:?}",
            out
        );
    }

    #[test]
    fn sqrelu_gate_mismatched_gate_up_returns_error() {
        let gate = vec![1.0_f32; 4];
        let up = vec![1.0_f32; 3];
        let mut out = vec![0.0_f32; 4];
        let err = sqrelu_gate(&gate, &up, &mut out).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn sqrelu_gate_mismatched_out_returns_error() {
        let gate = vec![1.0_f32; 4];
        let up = vec![1.0_f32; 4];
        let mut out = vec![0.0_f32; 3]; // wrong
        let err = sqrelu_gate(&gate, &up, &mut out).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn sqrelu_gate_output_nonnegative() {
        // Property: output[i] ≥ 0 always (because sqrelu ≥ 0 and we multiply, not always).
        // Actually up can be negative, so output can be negative.
        // More precisely: out[i] ≥ 0 iff up[i] ≥ 0 when gate[i] > 0.
        // This is not a general non-negativity property.
        // Test the non-negativity when up ≥ 0.
        let gate: Vec<f32> = (-5..=5).map(|i| i as f32).collect();
        let up = vec![1.0_f32; gate.len()];
        let mut out = vec![0.0_f32; gate.len()];
        sqrelu_gate(&gate, &up, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!(v >= 0.0, "index {i}: output={v} < 0 with positive up");
        }
    }

    // -----------------------------------------------------------------------
    // softmax
    // -----------------------------------------------------------------------

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1, got {sum}");
    }

    #[test]
    fn softmax_all_positive() {
        let mut x = vec![0.5_f32, -1.0, 2.0, -3.0, 1.5];
        softmax(&mut x);
        for (i, &v) in x.iter().enumerate() {
            assert!(v > 0.0, "output[{i}] = {v} must be positive");
        }
    }

    #[test]
    fn softmax_all_equal_input_is_uniform() {
        let n = 8_usize;
        let mut x = vec![2.5_f32; n];
        softmax(&mut x);
        let expected = 1.0 / n as f32;
        for (i, &v) in x.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-6,
                "output[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn softmax_large_value_dominates() {
        let mut x = vec![0.0_f32, 0.0, 100.0, 0.0];
        softmax(&mut x);
        assert!(
            x[2] > 0.999,
            "dominant logit must have prob ≈ 1, got {}",
            x[2]
        );
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn softmax_empty_slice_is_noop() {
        let mut x: Vec<f32> = vec![];
        softmax(&mut x); // must not panic
        assert!(x.is_empty());
    }

    #[test]
    fn softmax_single_element_becomes_one() {
        let mut x = vec![42.0_f32];
        softmax(&mut x);
        assert!(
            (x[0] - 1.0).abs() < 1e-6,
            "single element → 1.0, got {}",
            x[0]
        );
    }

    #[test]
    fn softmax_two_elements_sum_to_one() {
        let mut x = vec![1.0_f32, 2.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[0] < x[1], "larger logit must have larger probability");
    }

    #[test]
    fn softmax_shift_invariance() {
        // softmax(x + c) = softmax(x) for any constant c.
        let x_orig = vec![0.5_f32, -1.0, 2.0, 0.0];
        let shift = 100.0_f32;

        let mut x1 = x_orig.clone();
        let mut x2: Vec<f32> = x_orig.iter().map(|&v| v + shift).collect();

        softmax(&mut x1);
        softmax(&mut x2);

        for (i, (&a, &b)) in x1.iter().zip(x2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "index {i}: softmax(x)={a}, softmax(x+100)={b}"
            );
        }
    }

    #[test]
    fn softmax_monotone_preserving() {
        // Larger input → larger output (softmax is order-preserving).
        let mut x = vec![1.0_f32, 3.0, 2.0, 0.5];
        let order_before: Vec<usize> = {
            let mut idx: Vec<usize> = (0..4).collect();
            idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
            idx
        };
        softmax(&mut x);
        let order_after: Vec<usize> = {
            let mut idx: Vec<usize> = (0..4).collect();
            idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
            idx
        };
        assert_eq!(
            order_before, order_after,
            "softmax must preserve relative ordering"
        );
    }

    #[test]
    fn softmax_very_negative_inputs_are_handled() {
        // Inputs very far below the max should produce near-zero but valid probabilities.
        let mut x = vec![0.0_f32, -1000.0, -1000.0];
        softmax(&mut x);
        assert!(x[0] > 0.999, "dominant logit at x[0]={}", x[0]);
        for (i, &v) in x.iter().enumerate() {
            assert!(
                v.is_finite() && v >= 0.0,
                "output[{i}] = {v} must be finite and non-negative"
            );
        }
    }

    #[test]
    fn softmax_large_slice_sum_close_to_one() {
        let mut x: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum for 1024 elements = {sum}"
        );
    }

    // -----------------------------------------------------------------------
    // softmax_partial
    // -----------------------------------------------------------------------

    #[test]
    fn softmax_partial_first_n_elements_normalised() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 999.0, 999.0]; // only first 3 valid
        softmax_partial(&mut x, 3).unwrap();
        let sum: f32 = x[..3].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "first 3 elements must sum to 1, got {sum}"
        );
        // Last two elements must be unchanged.
        assert_eq!(x[3], 999.0, "x[3] must be unchanged");
        assert_eq!(x[4], 999.0, "x[4] must be unchanged");
    }

    #[test]
    fn softmax_partial_len_equals_slice_len_is_full_softmax() {
        let x_orig = vec![0.5_f32, -1.0, 2.0, 1.0];
        let mut x_partial = x_orig.clone();
        let mut x_full = x_orig.clone();

        softmax_partial(&mut x_partial, 4).unwrap();
        softmax(&mut x_full);

        for (i, (&p, &f)) in x_partial.iter().zip(x_full.iter()).enumerate() {
            assert!((p - f).abs() < 1e-7, "index {i}: partial={p}, full={f}");
        }
    }

    #[test]
    fn softmax_partial_len_zero_is_noop() {
        let mut x = vec![1.0_f32, 2.0, 3.0];
        softmax_partial(&mut x, 0).unwrap();
        // All elements unchanged.
        assert_eq!(x, vec![1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn softmax_partial_len_exceeds_slice_returns_error() {
        let mut x = vec![1.0_f32; 3];
        let err = softmax_partial(&mut x, 4).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn softmax_partial_single_element_len_one() {
        let mut x = vec![42.0_f32, 100.0];
        softmax_partial(&mut x, 1).unwrap();
        // Softmax of a single element = 1.0.
        assert!((x[0] - 1.0).abs() < 1e-6, "x[0] = {}", x[0]);
        // Second element unchanged.
        assert_eq!(x[1], 100.0);
    }

    // -----------------------------------------------------------------------
    // Analytical property tests
    // -----------------------------------------------------------------------

    /// Theorem: softmax(x)[i] = exp(x[i] - max(x)) / Σ_j exp(x[j] - max(x))
    ///
    /// Verify the explicit formula matches the implementation for a 4-element
    /// input vector with analytically computed expected values.
    #[test]
    fn softmax_matches_explicit_formula() {
        let x_orig = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut x = x_orig.clone();
        softmax(&mut x);

        let max_val = 4.0_f32;
        let exps: Vec<f32> = x_orig.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let expected: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

        for (i, (&got, &exp)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "index {i}: got {got}, expected {exp}"
            );
        }
    }

    /// Theorem: squared ReLU at a boundary:
    /// For x → 0+: sqrelu(x) → 0 continuously.
    /// For x → 0-: sqrelu(x) = 0.
    /// Continuity holds at x=0: lim_{x→0} sqrelu(x) = 0 = sqrelu(0).
    #[test]
    fn squared_relu_continuous_at_zero() {
        let epsilon = 1e-7_f32;
        let cases = [
            (0.0_f32, 0.0_f32),
            (epsilon, epsilon * epsilon),
            (-epsilon, 0.0_f32),
        ];
        for (input, expected) in cases {
            let mut x = vec![input];
            squared_relu(&mut x);
            assert!(
                (x[0] - expected).abs() < 1e-12,
                "sqrelu({input}) = {}, expected {expected}",
                x[0]
            );
        }
    }

    /// Theorem: for two elements x₀ < x₁, softmax preserves the ordering.
    /// Additionally, softmax([a, b]) = [σ(-Δ), σ(Δ)] where Δ=b-a and σ is sigmoid.
    #[test]
    fn softmax_two_elements_equals_sigmoid() {
        let a = 1.0_f32;
        let b = 3.0_f32;
        let delta = b - a;

        let mut x = vec![a, b];
        softmax(&mut x);

        // softmax([a, b])[0] = exp(a-b) / (1 + exp(a-b)) = sigmoid(-(b-a))
        // softmax([a, b])[1] = 1 / (1 + exp(a-b))        = sigmoid(b-a)
        let sigmoid_pos = 1.0_f32 / (1.0 + (-delta).exp()); // σ(Δ)
        let sigmoid_neg = 1.0_f32 - sigmoid_pos; // σ(-Δ) = 1 - σ(Δ)

        assert!(
            (x[0] - sigmoid_neg).abs() < 1e-6,
            "softmax[0]={}, expected sigmoid(-Δ)={}",
            x[0],
            sigmoid_neg
        );
        assert!(
            (x[1] - sigmoid_pos).abs() < 1e-6,
            "softmax[1]={}, expected sigmoid(Δ)={}",
            x[1],
            sigmoid_pos
        );
    }
}
