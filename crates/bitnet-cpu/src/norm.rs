//! RMSNorm implementation for the CPU backend.
//!
//! # Mathematical Specification
//!
//! Root-Mean-Square Layer Normalization (RMSNorm) is defined as:
//!
//! ```text
//! rms(x)  = sqrt( (1/d) * Σ_{i=0}^{d-1} x_i² + ε )
//! out[i]  = (x[i] / rms(x)) * γ[i]
//! ```
//!
//! where:
//! - **x** ∈ ℝ^d is the input vector
//! - **γ** ∈ ℝ^d is the learnable elementwise scale (stored as `weight`)
//! - ε > 0 is the numerical stability floor (typically 1e-5)
//!
//! Unlike LayerNorm, RMSNorm does **not** subtract the mean — it normalises
//! purely by RMS magnitude.  This matches the formulation used in LLaMA and
//! the BitNet b1.58 reference implementation.
//!
//! # BitNet b1.58 Usage
//!
//! RMSNorm appears in four positions per transformer block:
//! 1. `attention_norm`   — pre-attention normalisation of hidden state
//! 2. `attn_sub_norm`    — post-attention, pre-output-projection normalisation
//! 3. `ffn_norm`         — pre-FFN normalisation of hidden state
//! 4. `ffn_sub_norm`     — post-gate-activation, pre-down-projection normalisation
//!
//! Plus one final `norm` after all layers before the LM head.
//!
//! # Invariants
//!
//! - `input.len() == weight.len() == output.len()` (enforced at runtime)
//! - `eps > 0` (enforced at call site; default 1e-5)
//! - All output values are finite if all inputs and weights are finite

use bitnet_core::error::{BitNetError, Result};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute RMSNorm in-place, writing results to `output`.
///
/// ```text
/// rms    = sqrt( mean(x²) + ε )
/// out[i] = x[i] / rms * weight[i]
/// ```
///
/// # Arguments
///
/// - `input`:   Input vector `x`, length `d`.
/// - `weight`:  Learnable scale `γ`, length `d`.
/// - `eps`:     Numerical stability floor ε (must be > 0).
/// - `output`:  Pre-allocated output buffer, length `d`.  Overwritten.
///
/// # Errors
///
/// Returns [`BitNetError::InvalidShape`] if slice lengths differ.
/// Returns [`BitNetError::QuantizationError`] if `eps <= 0` or is non-finite.
///
/// # Example
///
/// ```
/// use bitnet_cpu::norm::rms_norm;
///
/// let input  = vec![1.0_f32, 2.0, 3.0, 4.0];
/// let weight = vec![1.0_f32; 4];
/// let mut output = vec![0.0_f32; 4];
/// rms_norm(&input, &weight, 1e-5, &mut output).unwrap();
///
/// // mean(x²) = (1+4+9+16)/4 = 7.5,  rms ≈ 2.7386
/// let expected_0 = 1.0_f32 / (7.5_f32 + 1e-5_f32).sqrt();
/// assert!((output[0] - expected_0).abs() < 1e-5);
/// ```
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) -> Result<()> {
    // ---- Validation ---------------------------------------------------------
    if input.len() != weight.len() {
        return Err(BitNetError::shape(
            format!("input.len() == weight.len() = {}", weight.len()),
            format!("input.len() = {}", input.len()),
        ));
    }
    if input.len() != output.len() {
        return Err(BitNetError::shape(
            format!("input.len() == output.len() = {}", input.len()),
            format!("output.len() = {}", output.len()),
        ));
    }
    if eps <= 0.0 || !eps.is_finite() {
        return Err(BitNetError::quant(format!(
            "eps must be finite and > 0, got {eps}"
        )));
    }
    if input.is_empty() {
        // Zero-length is valid (no-op); nothing to write.
        return Ok(());
    }

    // ---- Computation --------------------------------------------------------
    rms_norm_unchecked(input, weight, eps, output);
    Ok(())
}

/// Unchecked variant — caller guarantees all slice lengths are equal and `eps > 0`.
///
/// Used internally by the model forward pass on hot paths where the invariants
/// are guaranteed by construction.
///
/// # Panics (debug only)
/// Panics in debug builds if slice lengths differ or `eps <= 0`.
#[inline]
pub fn rms_norm_unchecked(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    debug_assert_eq!(
        input.len(),
        weight.len(),
        "rms_norm: input/weight length mismatch"
    );
    debug_assert_eq!(
        input.len(),
        output.len(),
        "rms_norm: input/output length mismatch"
    );
    debug_assert!(eps > 0.0, "rms_norm: eps must be > 0");

    let d = input.len() as f32;

    // Σ x_i²
    let sum_sq: f32 = super::simd::sum_squares_f32_fast(input);

    // rms = sqrt( mean(x²) + ε )
    let rms = (sum_sq / d + eps).sqrt();
    let inv_rms = rms.recip(); // 1 / rms

    // out[i] = x[i] * (1/rms) * γ[i]
    super::simd::mul_scale_f32_fast(input, inv_rms, weight, output);
}

/// Apply RMSNorm *in-place*, updating `buf` using `weight` as the scale.
///
/// This is equivalent to `rms_norm(buf.clone(), weight, eps, buf)` but avoids
/// an extra allocation.
///
/// # Errors
///
/// Returns the same errors as [`rms_norm`].
pub fn rms_norm_inplace(buf: &mut [f32], weight: &[f32], eps: f32) -> Result<()> {
    if buf.len() != weight.len() {
        return Err(BitNetError::shape(
            format!("buf.len() == weight.len() = {}", weight.len()),
            format!("buf.len() = {}", buf.len()),
        ));
    }
    if eps <= 0.0 || !eps.is_finite() {
        return Err(BitNetError::quant(format!(
            "eps must be finite and > 0, got {eps}"
        )));
    }
    if buf.is_empty() {
        return Ok(());
    }

    let d = buf.len() as f32;
    let sum_sq: f32 = super::simd::sum_squares_f32_fast(buf);
    let inv_rms = (sum_sq / d + eps).sqrt().recip();

    for i in 0..buf.len() {
        buf[i] *= inv_rms * weight[i];
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // rms_norm
    // -----------------------------------------------------------------------

    /// Compute the expected RMSNorm output analytically.
    fn reference_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let d = input.len() as f32;
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / d + eps).sqrt();
        input
            .iter()
            .zip(weight.iter())
            .map(|(&x, &w)| x / rms * w)
            .collect()
    }

    #[test]
    fn rms_norm_matches_reference_uniform_weight() {
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        rms_norm(&input, &weight, 1e-5, &mut output).unwrap();

        let expected = reference_rms_norm(&input, &weight, 1e-5);
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "index {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn rms_norm_scaled_weight_doubles_output() {
        // weight = [2; d] should double each output element.
        let input = vec![1.0_f32, -1.0, 1.0, -1.0];
        let weight_1 = vec![1.0_f32; 4];
        let weight_2 = vec![2.0_f32; 4];

        let mut out1 = vec![0.0_f32; 4];
        let mut out2 = vec![0.0_f32; 4];

        rms_norm(&input, &weight_1, 1e-5, &mut out1).unwrap();
        rms_norm(&input, &weight_2, 1e-5, &mut out2).unwrap();

        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (2.0 * a - b).abs() < 1e-6,
                "index {i}: 2*out1={}, out2={}",
                2.0 * a,
                b
            );
        }
    }

    #[test]
    fn rms_norm_zero_input_produces_zero_output() {
        let input = vec![0.0_f32; 8];
        let weight = vec![1.0_f32; 8];
        let mut output = vec![99.0_f32; 8]; // non-zero sentinel
        rms_norm(&input, &weight, 1e-5, &mut output).unwrap();
        // With all-zero input, sum_sq = 0, rms = sqrt(eps), out = 0 * inv_rms * w = 0
        assert!(
            output.iter().all(|&v| v == 0.0),
            "zero input must produce zero output"
        );
    }

    #[test]
    fn rms_norm_unit_input_inv_rms_is_one() {
        // x = [v; d] where v = 1/sqrt(d).
        //
        // Σ x_i² = d * v² = d * (1/d) = 1
        // mean(x²) = (1/d) * Σ x_i² = 1/d
        // rms = sqrt(mean(x²) + ε) = sqrt(1/d + ε) ≈ 1/sqrt(d) = v  (for small ε)
        // out[i] = x[i] / rms ≈ v / v = 1.0
        //
        // So RMSNorm of a uniform vector v normalises every element to ≈ 1.0
        // (when weight = 1).  This is the unit-vector-to-constant property.
        let d = 16_usize;
        let v = 1.0_f32 / (d as f32).sqrt();
        let input = vec![v; d];
        let weight = vec![1.0_f32; d];
        let mut output = vec![0.0_f32; d];
        rms_norm(&input, &weight, 1e-8, &mut output).unwrap();

        // Expected: out ≈ 1.0 for all elements (v / rms ≈ v / v = 1.0).
        for (i, &o) in output.iter().enumerate() {
            assert!(
                (o - 1.0_f32).abs() < 1e-3,
                "index {i}: out={o}, expected ≈ 1.0  (x[i]={v}, rms≈{v})"
            );
        }
    }

    #[test]
    fn rms_norm_large_input_normalises_to_order_one() {
        // Very large input: RMSNorm should bring output to order-1.
        let input: Vec<f32> = (1..=8).map(|i| i as f32 * 1000.0).collect();
        let weight = vec![1.0_f32; 8];
        let mut output = vec![0.0_f32; 8];
        rms_norm(&input, &weight, 1e-5, &mut output).unwrap();

        // RMS of [1000, 2000, ..., 8000] ≈ large, so output ≈ input/rms ≈ O(1)
        for &v in &output {
            assert!(v.abs() < 10.0, "normalised output should be O(1), got {v}");
        }
    }

    #[test]
    fn rms_norm_negative_inputs_preserve_sign() {
        let input = vec![-1.0_f32, -2.0, -3.0, -4.0];
        let weight = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        rms_norm(&input, &weight, 1e-5, &mut output).unwrap();

        // RMSNorm preserves sign.
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v < 0.0,
                "index {i}: negative input → negative output, got {v}"
            );
        }
    }

    #[test]
    fn rms_norm_mixed_sign_weight() {
        // weight = [1, -1, 1, -1] flips sign of alternate outputs.
        let input = vec![1.0_f32, 1.0, 1.0, 1.0];
        let weight = vec![1.0_f32, -1.0, 1.0, -1.0];
        let mut output = vec![0.0_f32; 4];
        rms_norm(&input, &weight, 1e-5, &mut output).unwrap();

        assert!(output[0] > 0.0, "weight=+1 → positive");
        assert!(output[1] < 0.0, "weight=-1 → negative");
        assert!(output[2] > 0.0, "weight=+1 → positive");
        assert!(output[3] < 0.0, "weight=-1 → negative");
    }

    #[test]
    fn rms_norm_single_element() {
        // x=[v], weight=[w]: rms = sqrt(v² + eps), out = v / rms * w
        let v = 3.0_f32;
        let w = 2.0_f32;
        let eps = 1e-5_f32;
        let input = vec![v];
        let weight = vec![w];
        let mut output = vec![0.0_f32; 1];
        rms_norm(&input, &weight, eps, &mut output).unwrap();

        let expected = v / (v * v + eps).sqrt() * w;
        assert!(
            (output[0] - expected).abs() < 1e-6,
            "got {}, expected {expected}",
            output[0]
        );
    }

    #[test]
    fn rms_norm_empty_slice_is_noop() {
        let input: Vec<f32> = vec![];
        let weight: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        rms_norm(&input, &weight, 1e-5, &mut output).unwrap(); // must not panic or error
    }

    // -----------------------------------------------------------------------
    // Error conditions
    // -----------------------------------------------------------------------

    #[test]
    fn rms_norm_mismatched_input_weight_returns_error() {
        let input = vec![1.0_f32; 4];
        let weight = vec![1.0_f32; 5]; // wrong length
        let mut output = vec![0.0_f32; 4];
        let err = rms_norm(&input, &weight, 1e-5, &mut output).unwrap_err();
        assert!(
            matches!(err, BitNetError::InvalidShape { .. }),
            "expected InvalidShape, got {err:?}"
        );
    }

    #[test]
    fn rms_norm_mismatched_input_output_returns_error() {
        let input = vec![1.0_f32; 4];
        let weight = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 3]; // wrong length
        let err = rms_norm(&input, &weight, 1e-5, &mut output).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn rms_norm_zero_eps_returns_error() {
        let input = vec![1.0_f32; 4];
        let weight = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        let err = rms_norm(&input, &weight, 0.0, &mut output).unwrap_err();
        assert!(matches!(err, BitNetError::QuantizationError(_)));
    }

    #[test]
    fn rms_norm_negative_eps_returns_error() {
        let input = vec![1.0_f32; 4];
        let weight = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        let err = rms_norm(&input, &weight, -1e-5, &mut output).unwrap_err();
        assert!(matches!(err, BitNetError::QuantizationError(_)));
    }

    #[test]
    fn rms_norm_nan_eps_returns_error() {
        let input = vec![1.0_f32; 4];
        let weight = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        let err = rms_norm(&input, &weight, f32::NAN, &mut output).unwrap_err();
        assert!(matches!(err, BitNetError::QuantizationError(_)));
    }

    // -----------------------------------------------------------------------
    // rms_norm_inplace
    // -----------------------------------------------------------------------

    #[test]
    fn rms_norm_inplace_matches_rms_norm() {
        let input = vec![0.5_f32, -1.5, 2.0, -0.5, 1.0, 0.25];
        let weight = vec![1.0_f32, 0.5, 2.0, 1.5, 0.8, 1.2];
        let eps = 1e-5_f32;

        // Reference: out-of-place rms_norm
        let mut ref_output = vec![0.0_f32; 6];
        rms_norm(&input, &weight, eps, &mut ref_output).unwrap();

        // In-place version
        let mut buf = input.clone();
        rms_norm_inplace(&mut buf, &weight, eps).unwrap();

        for (i, (&r, &ip)) in ref_output.iter().zip(buf.iter()).enumerate() {
            assert!((r - ip).abs() < 1e-6, "index {i}: ref={r}, inplace={ip}");
        }
    }

    #[test]
    fn rms_norm_inplace_mismatched_lengths_returns_error() {
        let mut buf = vec![1.0_f32; 4];
        let weight = vec![1.0_f32; 5];
        let err = rms_norm_inplace(&mut buf, &weight, 1e-5).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn rms_norm_inplace_zero_eps_returns_error() {
        let mut buf = vec![1.0_f32; 4];
        let weight = vec![1.0_f32; 4];
        let err = rms_norm_inplace(&mut buf, &weight, 0.0).unwrap_err();
        assert!(matches!(err, BitNetError::QuantizationError(_)));
    }

    // -----------------------------------------------------------------------
    // rms_norm_unchecked
    // -----------------------------------------------------------------------

    #[test]
    fn rms_norm_unchecked_matches_checked_version() {
        let input = vec![2.0_f32, -3.0, 1.5, -0.5];
        let weight = vec![0.8_f32, 1.2, 1.0, 0.5];
        let eps = 1e-5_f32;

        let mut out_checked = vec![0.0_f32; 4];
        let mut out_unchecked = vec![0.0_f32; 4];

        rms_norm(&input, &weight, eps, &mut out_checked).unwrap();
        rms_norm_unchecked(&input, &weight, eps, &mut out_unchecked);

        for (i, (&c, &u)) in out_checked.iter().zip(out_unchecked.iter()).enumerate() {
            assert!(
                (c - u).abs() < 1e-7,
                "index {i}: checked={c}, unchecked={u}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Mathematical properties
    // -----------------------------------------------------------------------

    /// Property: RMSNorm is idempotent when γ = rms(x) * 1_vector.
    ///
    /// If γ[i] = rms(x) for all i, then out[i] = x[i] / rms(x) * rms(x) = x[i].
    #[test]
    fn rms_norm_with_rms_scale_recovers_input() {
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let eps = 1e-8_f32;
        let d = input.len() as f32;
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / d + eps).sqrt();

        // γ[i] = rms for all i → out = input
        let weight = vec![rms; 4];
        let mut output = vec![0.0_f32; 4];
        rms_norm(&input, &weight, eps, &mut output).unwrap();

        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (inp - out).abs() < 1e-4,
                "index {i}: input={inp}, output={out}"
            );
        }
    }

    /// Property: RMSNorm with unit weight normalises L2-like: the squared mean
    /// of outputs should be approximately 1 (up to the eps perturbation).
    ///
    /// Specifically: mean(out²) = mean(x²) / rms² = mean(x²) / (mean(x²) + ε) ≈ 1 for ε → 0.
    #[test]
    fn rms_norm_output_squared_mean_near_one_for_small_eps() {
        let input: Vec<f32> = (1..=16).map(|i| i as f32 * 0.1).collect();
        let weight = vec![1.0_f32; 16];
        let mut output = vec![0.0_f32; 16];
        let eps = 1e-10_f32; // very small to minimise perturbation

        rms_norm(&input, &weight, eps, &mut output).unwrap();

        let d = 16.0_f32;
        let mean_sq_out: f32 = output.iter().map(|&v| v * v).sum::<f32>() / d;
        // mean(out²) = mean(x²) / (mean(x²) + eps) should be very close to 1
        assert!(
            (mean_sq_out - 1.0).abs() < 1e-4,
            "mean(out²) = {mean_sq_out}, expected ≈ 1"
        );
    }

    /// Property: RMSNorm is scale-invariant in the input when weight=1.
    ///
    /// For any scalar α > 0: RMSNorm(α·x, 1, ε) ≈ RMSNorm(x, 1, ε)
    /// when ε is negligible relative to mean(x²).
    #[test]
    fn rms_norm_approximately_scale_invariant() {
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32; 4];
        let eps = 1e-10_f32; // negligible vs signal

        let mut out1 = vec![0.0_f32; 4];
        let scaled: Vec<f32> = input.iter().map(|&v| v * 100.0).collect();
        let mut out2 = vec![0.0_f32; 4];

        rms_norm(&input, &weight, eps, &mut out1).unwrap();
        rms_norm(&scaled, &weight, eps, &mut out2).unwrap();

        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "index {i}: scale-invariance violated: out(x)={a}, out(100x)={b}"
            );
        }
    }
}
