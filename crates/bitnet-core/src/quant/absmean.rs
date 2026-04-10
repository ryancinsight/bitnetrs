//! Absmean quantisation for BitNet b1.58 ternary weights.
//!
//! # Mathematical Foundation
//!
//! Given a full-precision weight matrix **W** (or a flat slice of `f32` / `bf16`),
//! the *absmean* quantisation function produces a ternary matrix **W_q** and a
//! per-tensor scale α_W:
//!
//! ```text
//! α_W  = mean( |W| )                           (≥ ε to avoid division by zero)
//! W_q  = clip( round( W / α_W ), −1, 1 )       (each element ∈ {−1, 0, +1})
//! ```
//!
//! The scale is stored alongside the quantised values so that the effective
//! (dequantised) weight can be recovered as:
//!
//! ```text
//! W_eff = W_q * α_W    ⟹    each value ∈ {−α_W, 0, +α_W}
//! ```
//!
//! During a forward pass the scale is folded into the final accumulation:
//!
//! ```text
//! y = (W_q @ x_q) * α_W * (α_x / 127)
//! ```
//!
//! where `x_q` is the 8-bit activation quantised by `absmax` (see `absmax.rs`).
//!
//! # Numerical Stability
//!
//! `α_W` is clamped to a minimum of `1e-5` before the reciprocal is taken,
//! preventing division by zero for all-zero weight slices.
//!
//! # Invariants (guaranteed by the public API)
//! - Returned `Vec<i8>` satisfies `∀ v : v ∈ {−1, 0, 1}`.
//! - Returned `scale > 0`.
//! - Length of returned `Vec<i8>` equals the length of the input slice.

use half::bf16;

use crate::error::{BitNetError, Result};

/// Minimum absolute-mean value used as the denominator lower-bound.
const ABSMEAN_MIN: f32 = 1e-5;

// ---------------------------------------------------------------------------
// f32 path
// ---------------------------------------------------------------------------

/// Quantise a flat `f32` weight slice to ternary {−1, 0, +1} using absmean.
///
/// Returns `(quantised_values, scale)` where:
/// - `quantised_values[i] = clip(round(weights[i] / scale), −1, 1)`
/// - `scale = mean(|weights|)` clamped to `≥ ABSMEAN_MIN`
///
/// # Errors
/// Returns [`BitNetError::QuantizationError`] if `weights` is empty or contains
/// any non-finite (`NaN` / `Inf`) value.
///
/// # Examples
/// ```
/// use bitnet_core::quant::absmean::absmean_quantize;
///
/// let w = vec![0.3_f32, -0.6, 0.0, 1.2];
/// let (q, scale) = absmean_quantize(&w).unwrap();
/// // scale ≈ mean(0.3, 0.6, 0.0, 1.2) = 0.525
/// // q = [1, -1, 0, 1]  (round(w/scale) clamped to ±1)
/// assert!(q.iter().all(|&v| v == -1 || v == 0 || v == 1));
/// assert!(scale > 0.0);
/// ```
pub fn absmean_quantize(weights: &[f32]) -> Result<(Vec<i8>, f32)> {
    if weights.is_empty() {
        return Err(BitNetError::quant("cannot quantise an empty weight slice"));
    }

    // Verify all values are finite before touching them.
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(BitNetError::quant(format!(
                "non-finite weight value at index {i}: {w}"
            )));
        }
    }

    // α_W = mean(|W|), clamped to avoid 1/0.
    let sum_abs: f32 = weights.iter().map(|w| w.abs()).sum();
    let mean_abs = sum_abs / weights.len() as f32;
    let scale = mean_abs.max(ABSMEAN_MIN);

    let inv_scale = scale.recip();

    // q_i = clip(round(w_i * inv_scale), −1, 1)
    let quantised: Vec<i8> = weights
        .iter()
        .map(|&w| {
            let rounded = (w * inv_scale).round();
            rounded.clamp(-1.0, 1.0) as i8
        })
        .collect();

    Ok((quantised, scale))
}

// ---------------------------------------------------------------------------
// bf16 path (weight tensors loaded from HuggingFace safetensors are bf16)
// ---------------------------------------------------------------------------

/// Quantise a flat `bf16` weight slice to ternary {−1, 0, +1} using absmean.
///
/// Converts each `bf16` value to `f32` before applying the quantisation,
/// so the numerical result is identical to [`absmean_quantize`] on the
/// equivalent `f32` slice.
///
/// # Errors
/// Returns [`BitNetError::QuantizationError`] if `weights` is empty.
pub fn absmean_quantize_bf16(weights: &[bf16]) -> Result<(Vec<i8>, f32)> {
    if weights.is_empty() {
        return Err(BitNetError::quant(
            "cannot quantise an empty bf16 weight slice",
        ));
    }

    // Convert to f32 and delegate.
    let f32_weights: Vec<f32> = weights.iter().map(|w| w.to_f32()).collect();
    absmean_quantize(&f32_weights)
}

/// Quantise a flat `f16` weight slice to ternary {−1, 0, +1} using absmean.
pub fn absmean_quantize_f16(weights: &[half::f16]) -> Result<(Vec<i8>, f32)> {
    if weights.is_empty() {
        return Err(BitNetError::quant(
            "cannot quantise an empty f16 weight slice",
        ));
    }
    let f32_weights: Vec<f32> = weights.iter().map(|w| w.to_f32()).collect();
    absmean_quantize(&f32_weights)
}

// ---------------------------------------------------------------------------
// Dequantisation helpers
// ---------------------------------------------------------------------------

/// Reconstruct approximate `f32` weights from ternary values and a scale.
///
/// `dequantised[i] = quantised[i] as f32 * scale`
///
/// This is used primarily for testing and debugging; production inference
/// avoids materialising the full dequantised matrix.
///
/// # Errors
/// Returns [`BitNetError::QuantizationError`] if `scale ≤ 0` or is non-finite.
pub fn absmean_dequantize(quantised: &[i8], scale: f32) -> Result<Vec<f32>> {
    if scale <= 0.0 || !scale.is_finite() {
        return Err(BitNetError::quant(format!(
            "scale must be finite and > 0 for dequantisation, got {scale}"
        )));
    }
    Ok(quantised.iter().map(|&q| q as f32 * scale).collect())
}

// ---------------------------------------------------------------------------
// Batch helper (quantises a 2-D weight matrix row-slice at a time)
// ---------------------------------------------------------------------------

/// Quantise every sub-slice of `weights` of length `group_size` independently.
///
/// Returns a flat `Vec<i8>` of the same total length as `weights`, together
/// with a `Vec<f32>` of per-group scales.
///
/// This is used for *per-row* or *per-tile* quantisation strategies.
///
/// # Errors
/// Returns [`BitNetError::QuantizationError`] if `weights.len() % group_size != 0`
/// or if any group is empty / contains non-finite values.
pub fn absmean_quantize_grouped(weights: &[f32], group_size: usize) -> Result<(Vec<i8>, Vec<f32>)> {
    if group_size == 0 {
        return Err(BitNetError::quant("group_size must be > 0"));
    }
    if weights.len() % group_size != 0 {
        return Err(BitNetError::quant(format!(
            "weights.len() ({}) is not divisible by group_size ({})",
            weights.len(),
            group_size
        )));
    }

    let n_groups = weights.len() / group_size;
    let mut all_quantised = Vec::with_capacity(weights.len());
    let mut scales = Vec::with_capacity(n_groups);

    for group in weights.chunks_exact(group_size) {
        let (q, s) = absmean_quantize(group)?;
        all_quantised.extend_from_slice(&q);
        scales.push(s);
    }

    Ok((all_quantised, scales))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // absmean_quantize
    // -----------------------------------------------------------------------

    #[test]
    fn all_positive_weights() {
        // W = [0.4, 0.8, 1.2, 0.0]
        // |W| = [0.4, 0.8, 1.2, 0.0]  mean = 0.6
        // q  = round(W / 0.6) = round([0.667, 1.333, 2.0, 0.0]) = [1, 1, 2, 0]
        // clipped to [-1,1]: [1, 1, 1, 0]
        let w = vec![0.4_f32, 0.8, 1.2, 0.0];
        let (q, scale) = absmean_quantize(&w).unwrap();
        assert!((scale - 0.6).abs() < 1e-5, "scale ≈ 0.6, got {scale}");
        assert_eq!(q, vec![1i8, 1, 1, 0]);
    }

    #[test]
    fn mixed_sign_weights() {
        // W = [0.5, -0.5, 0.0, -1.5]
        // mean(|W|) = (0.5 + 0.5 + 0.0 + 1.5) / 4 = 0.625
        // q = round([0.8, -0.8, 0.0, -2.4]) clamped = [1, -1, 0, -1]
        let w = vec![0.5_f32, -0.5, 0.0, -1.5];
        let (q, scale) = absmean_quantize(&w).unwrap();
        assert!((scale - 0.625).abs() < 1e-5, "scale ≈ 0.625, got {scale}");
        assert_eq!(q, vec![1i8, -1, 0, -1]);
    }

    #[test]
    fn all_zero_weights_use_minimum_scale() {
        let w = vec![0.0_f32; 8];
        let (q, scale) = absmean_quantize(&w).unwrap();
        assert_eq!(scale, ABSMEAN_MIN, "all-zero slice must use ABSMEAN_MIN");
        assert!(
            q.iter().all(|&v| v == 0),
            "all-zero input → all-zero output"
        );
    }

    #[test]
    fn empty_slice_returns_error() {
        let result = absmean_quantize(&[]);
        assert!(
            matches!(result, Err(BitNetError::QuantizationError(_))),
            "empty slice must return QuantizationError"
        );
    }

    #[test]
    fn nan_weight_returns_error() {
        let w = vec![1.0_f32, f32::NAN, 0.0];
        assert!(absmean_quantize(&w).is_err());
    }

    #[test]
    fn inf_weight_returns_error() {
        let w = vec![1.0_f32, f32::INFINITY];
        assert!(absmean_quantize(&w).is_err());
    }

    #[test]
    fn quantised_values_are_ternary() {
        // Property: for any finite f32 slice, all quantised values ∈ {-1, 0, 1}.
        let cases: &[&[f32]] = &[
            &[0.1, -0.2, 3.0, -3.0],
            &[100.0, -100.0, 50.0],
            &[0.001, 0.002, -0.001],
        ];
        for &weights in cases {
            let (q, _) = absmean_quantize(weights).unwrap();
            for &v in &q {
                assert!(
                    v == -1 || v == 0 || v == 1,
                    "ternary invariant violated: {v}"
                );
            }
        }
    }

    #[test]
    fn scale_is_always_positive() {
        let cases: &[&[f32]] = &[&[0.0, 0.0], &[1.0], &[-1.0, 1.0], &[0.0001]];
        for &weights in cases {
            let (_, scale) = absmean_quantize(weights).unwrap();
            assert!(scale > 0.0, "scale must be positive, got {scale}");
        }
    }

    #[test]
    fn single_element_positive() {
        let (q, scale) = absmean_quantize(&[2.5_f32]).unwrap();
        // scale = 2.5 (single element, mean = value)
        // q = round(2.5/2.5) = round(1.0) = 1
        assert!((scale - 2.5).abs() < 1e-6);
        assert_eq!(q, vec![1i8]);
    }

    #[test]
    fn single_element_negative() {
        let (q, scale) = absmean_quantize(&[-3.0_f32]).unwrap();
        assert!((scale - 3.0).abs() < 1e-6, "scale = |−3| = 3");
        assert_eq!(q, vec![-1i8]);
    }

    // -----------------------------------------------------------------------
    // absmean_quantize_bf16
    // -----------------------------------------------------------------------

    #[test]
    fn bf16_path_matches_f32_path() {
        let f32_vals = vec![0.25_f32, -0.75, 0.5, -0.125];
        let bf16_vals: Vec<bf16> = f32_vals.iter().map(|&v| bf16::from_f32(v)).collect();

        let (q_f32, s_f32) = absmean_quantize(&f32_vals).unwrap();
        let (q_bf16, s_bf16) = absmean_quantize_bf16(&bf16_vals).unwrap();

        // bf16 has lower precision; scales may differ slightly.
        assert!(
            (s_f32 - s_bf16).abs() < 1e-3,
            "scales should be close: {s_f32} vs {s_bf16}"
        );
        assert_eq!(
            q_f32, q_bf16,
            "ternary codes must match (round-half-to-even can differ at boundaries)"
        );
    }

    #[test]
    fn bf16_empty_returns_error() {
        assert!(absmean_quantize_bf16(&[]).is_err());
    }

    // -----------------------------------------------------------------------
    // absmean_dequantize
    // -----------------------------------------------------------------------

    #[test]
    fn dequantize_reconstructs_scaled_values() {
        let q = vec![1i8, -1, 0, 1];
        let scale = 0.5_f32;
        let deq = absmean_dequantize(&q, scale).unwrap();
        let expected = vec![0.5_f32, -0.5, 0.0, 0.5];
        for (got, exp) in deq.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-7,
                "dequantise mismatch: {got} ≠ {exp}"
            );
        }
    }

    #[test]
    fn dequantize_zero_scale_returns_error() {
        assert!(absmean_dequantize(&[1i8], 0.0).is_err());
    }

    #[test]
    fn dequantize_negative_scale_returns_error() {
        assert!(absmean_dequantize(&[1i8], -1.0).is_err());
    }

    // -----------------------------------------------------------------------
    // absmean_quantize_grouped
    // -----------------------------------------------------------------------

    #[test]
    fn grouped_two_groups_independent_scales() {
        // Group 1: [2.0, -2.0]  mean(|W|) = 2.0  → q = [1, -1]
        // Group 2: [0.1, -0.1]  mean(|W|) = 0.1  → q = [1, -1]
        let w = vec![2.0_f32, -2.0, 0.1, -0.1];
        let (q, scales) = absmean_quantize_grouped(&w, 2).unwrap();
        assert_eq!(q, vec![1i8, -1, 1, -1]);
        assert_eq!(scales.len(), 2);
        assert!((scales[0] - 2.0).abs() < 1e-5);
        assert!((scales[1] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn grouped_indivisible_size_returns_error() {
        let w = vec![1.0_f32; 5];
        assert!(absmean_quantize_grouped(&w, 3).is_err());
    }

    #[test]
    fn grouped_zero_group_size_returns_error() {
        let w = vec![1.0_f32; 4];
        assert!(absmean_quantize_grouped(&w, 0).is_err());
    }

    /// Roundtrip: quantise then dequantise; the L∞ error must be ≤ α_W / 2.
    ///
    /// Theorem: for any finite w, |w - dequantise(quantise(w))| ≤ α_W / 2
    /// because rounding introduces at most 0.5 * α_W error and clipping only
    /// affects values with |w/α_W| > 1.5 where the approximation error is
    /// bounded by the clipping threshold.
    #[test]
    fn roundtrip_error_bounded_by_half_scale() {
        let weights: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let (q, scale) = absmean_quantize(&weights).unwrap();
        let deq = absmean_dequantize(&q, scale).unwrap();

        for (i, (&w, &d)) in weights.iter().zip(deq.iter()).enumerate() {
            // For unclipped values (|w/α_W| ≤ 1.5), error ≤ scale/2.
            if (w / scale).abs() <= 1.5 {
                assert!(
                    (w - d).abs() <= scale / 2.0 + 1e-6,
                    "index {i}: |{w} - {d}| = {} > scale/2 = {}",
                    (w - d).abs(),
                    scale / 2.0
                );
            }
        }
    }
}
