//! Standalone mathematical utilities shared across all backend implementations.
//!
//! These functions contain no I/O, no heap allocation beyond the returned
//! `Vec`, and no device-specific code.  They are used by both the CPU backend
//! (directly) and the GPU backend (as reference implementations for testing
//! shader outputs).
//!
//! # Mathematical Definitions
//!
//! ## RMSNorm
//!
//! Given an input vector **x** ∈ ℝ^d and learnable scale **γ** ∈ ℝ^d:
//!
//! ```text
//! rms(x)  = sqrt( (1/d) Σ_i x_i² + ε )
//! out[i]  = x[i] / rms(x) * γ[i]
//! ```
//!
//! ## Rotary Position Embedding (RoPE)
//!
//! For a head vector **x** ∈ ℝ^{head_dim} at sequence position `pos`,
//! BitNet follows the LLaMA / HuggingFace half-split RoPE convention:
//!
//! ```text
//! θ_i     = pos / rope_theta^(2i / head_dim)   for i = 0 … head_dim/2 − 1
//! x_lo[i] = x[i]
//! x_hi[i] = x[i + head_dim/2]
//! x'_lo[i] = x_lo[i] * cos(θ_i) - x_hi[i] * sin(θ_i)
//! x'_hi[i] = x_hi[i] * cos(θ_i) + x_lo[i] * sin(θ_i)
//! ```
//!
//! This is equivalent to `rotate_half(x)` in HuggingFace LLaMA-family models,
//! not the older interleaved `(2i, 2i+1)` pairing convention.
//!
//! ## Softmax (numerically stable)
//!
//! ```text
//! m     = max_i(x_i)
//! e_i   = exp(x_i − m)
//! out_i = e_i / Σ_j e_j
//! ```
//!
//! ## Squared ReLU
//!
//! ```text
//! out_i = max(0, x_i)²
//! ```

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// Compute RMSNorm in-place, writing results to `output`.
///
/// ```text
/// rms    = sqrt( mean(x²) + ε )
/// out[i] = x[i] / rms * weight[i]
/// ```
///
/// All three slices (`input`, `weight`, `output`) must have the same length
/// `d`.  The caller is responsible for ensuring this; the function panics in
/// debug builds if the lengths differ.
///
/// # Panics (debug only)
/// Panics if `input.len() != weight.len()` or `input.len() != output.len()`.
///
/// # Numerical note
/// `ε` (eps) is added *inside* the square root, not outside, to match the
/// standard Llama-family RMSNorm formulation used in the official BitNet code.
#[inline]
pub fn rms_norm_f32(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    debug_assert_eq!(
        input.len(),
        weight.len(),
        "rms_norm: input and weight must have the same length"
    );
    debug_assert_eq!(
        input.len(),
        output.len(),
        "rms_norm: input and output must have the same length"
    );

    let d = input.len() as f32;

    // mean(x²) + ε
    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / d + eps).sqrt();
    let inv_rms = rms.recip();

    for i in 0..input.len() {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

// ---------------------------------------------------------------------------
// RoPE frequency table
// ---------------------------------------------------------------------------

/// Pre-compute the cosine and sine tables for Rotary Position Embedding (RoPE).
///
/// Returns two flat `Vec<f32>` of shape `[max_seq × half_head_dim]`:
/// - `cos_table[pos * half_head_dim + i] = cos( pos / rope_theta^(2i/head_dim) )`
/// - `sin_table[pos * half_head_dim + i] = sin( pos / rope_theta^(2i/head_dim) )`
///
/// where `half_head_dim = head_dim / 2`.
///
/// # Arguments
/// - `max_seq`:    maximum sequence length to pre-compute tables for.
/// - `head_dim`:   per-head feature dimension (must be even).
/// - `theta`:      RoPE base frequency θ (e.g. 500 000.0 for the 2B model).
///
/// # Panics
/// Panics if `head_dim` is zero or odd (since RoPE requires paired dimensions).
///
/// # Derivation
/// The frequency for dimension pair `i` is:
/// ```text
/// freq_i = 1 / rope_theta^(2i / head_dim)
///        = rope_theta^(−2i / head_dim)
/// ```
/// At position `pos`: `θ_i = pos * freq_i`.
pub fn rope_cos_sin_table(max_seq: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    assert!(
        head_dim > 0 && head_dim % 2 == 0,
        "head_dim must be even and > 0"
    );

    let half = head_dim / 2;
    let total = max_seq * half;

    let mut cos_table = Vec::with_capacity(total);
    let mut sin_table = Vec::with_capacity(total);

    for pos in 0..max_seq {
        for i in 0..half {
            // freq_i = theta^(-2i / head_dim)
            let exponent = -2.0 * i as f32 / head_dim as f32;
            let freq = theta.powf(exponent);
            let angle = pos as f32 * freq;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
    }

    (cos_table, sin_table)
}

// ---------------------------------------------------------------------------
// RoPE application (single head vector)
// ---------------------------------------------------------------------------

/// Apply RoPE to a single flat head vector `x` of length `head_dim` in-place.
///
/// Uses the pre-computed `cos` and `sin` slices for the current sequence
/// position. Both `cos` and `sin` must have length `head_dim / 2`.
///
/// This helper implements the LLaMA / HuggingFace half-split convention:
///
/// ```text
/// x_lo = x[..half]
/// x_hi = x[half..]
/// x'_lo[i] = x_lo[i] * cos[i] - x_hi[i] * sin[i]
/// x'_hi[i] = x_hi[i] * cos[i] + x_lo[i] * sin[i]
/// ```
///
/// # Panics (debug only)
/// Panics if `x.len() != head_dim`, `cos.len() != head_dim / 2`, or
/// `head_dim` is odd.
#[inline]
pub fn apply_rope_to_head(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let head_dim = x.len();
    debug_assert!(head_dim % 2 == 0, "head_dim must be even");
    let half = head_dim / 2;
    debug_assert_eq!(cos.len(), half);
    debug_assert_eq!(sin.len(), half);

    for i in 0..half {
        let x_lo = x[i];
        let x_hi = x[i + half];
        let c = cos[i];
        let s = sin[i];
        x[i] = x_lo * c - x_hi * s;
        x[i + half] = x_hi * c + x_lo * s;
    }
}

// ---------------------------------------------------------------------------
// Softmax (numerically stable)
// ---------------------------------------------------------------------------

/// Apply numerically-stable softmax in-place over the entire slice.
///
/// ```text
/// m    = max_i(x_i)
/// e_i  = exp(x_i − m)
/// x_i ← e_i / Σ_j e_j
/// ```
///
/// After this operation: `x[i] > 0` and `Σ_i x[i] = 1.0` (to floating-point
/// precision).
///
/// An empty slice is a no-op.
#[inline]
pub fn softmax_f32(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Find max for numerical stability.
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x_i - max) and accumulate sum.
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // Normalise.
    let inv_sum = sum.recip();
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Squared ReLU
// ---------------------------------------------------------------------------

/// Apply squared ReLU element-wise in-place: `x[i] = max(0, x[i])²`.
///
/// This is the activation used in the BitNet b1.58 FFN gate:
/// `inner = ffn_sub_norm( sqrelu(gate) ⊙ up )`
#[inline]
pub fn squared_relu_f32(x: &mut [f32]) {
    for v in x.iter_mut() {
        let r = v.max(0.0);
        *v = r * r;
    }
}

// ---------------------------------------------------------------------------
// Element-wise multiplication
// ---------------------------------------------------------------------------

/// Element-wise multiply `a ⊙ b`, writing results to `out`.
///
/// `out[i] = a[i] * b[i]`
///
/// # Panics (debug only)
/// Panics if slice lengths differ.
#[inline]
pub fn elementwise_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len(), "elementwise_mul: a and b lengths differ");
    debug_assert_eq!(
        a.len(),
        out.len(),
        "elementwise_mul: a and out lengths differ"
    );
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

// ---------------------------------------------------------------------------
// LM-head matmul (non-quantised, full-precision)
// ---------------------------------------------------------------------------

/// Compute the language-model head projection: `logits = weights @ hidden`.
///
/// ```text
/// logits[v] = Σ_h  weights[v * hidden_size + h] * hidden[h]
/// ```
///
/// This is an unquantised `f32` matrix–vector product used only for the final
/// vocabulary projection.  Weight tying means `weights` is the transposed
/// embedding matrix.
///
/// # Arguments
/// - `hidden`:      `[hidden_size]` — the final normalised hidden state.
/// - `weights`:     `[vocab_size × hidden_size]` row-major.
/// - `vocab_size`:  number of vocabulary entries.
/// - `hidden_size`: embedding / model dimension.
///
/// # Panics (debug only)
/// Panics if `hidden.len() != hidden_size` or
/// `weights.len() != vocab_size * hidden_size`.
pub fn lm_head_matmul(
    hidden: &[f32],
    weights: &[f32],
    vocab_size: usize,
    hidden_size: usize,
) -> Vec<f32> {
    debug_assert_eq!(hidden.len(), hidden_size);
    debug_assert_eq!(weights.len(), vocab_size * hidden_size);

    let mut logits = vec![0.0_f32; vocab_size];
    for v in 0..vocab_size {
        let row_start = v * hidden_size;
        let row = &weights[row_start..row_start + hidden_size];
        let mut acc = 0.0_f32;
        for h in 0..hidden_size {
            acc += row[h] * hidden[h];
        }
        logits[v] = acc;
    }
    logits
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // rms_norm_f32
    // ------------------------------------------------------------------

    #[test]
    fn rms_norm_all_ones_weight() {
        // x = [1, 2, 3, 4], weight = [1, 1, 1, 1], eps = 0
        // mean(x²) = (1+4+9+16)/4 = 7.5,  rms = sqrt(7.5) ≈ 2.7386
        // out[i] = x[i] / rms
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        rms_norm_f32(&input, &weight, 1e-6, &mut output);

        let mean_sq: f32 = input.iter().map(|&x| x * x).sum::<f32>() / 4.0;
        let rms = (mean_sq + 1e-6_f32).sqrt();
        for (i, (&x, &o)) in input.iter().zip(output.iter()).enumerate() {
            let expected = x / rms;
            assert!((o - expected).abs() < 1e-5, "index {i}: {o} ≠ {expected}");
        }
    }

    #[test]
    fn rms_norm_scaled_weight() {
        // weight doubles the output
        let input = vec![1.0_f32, -1.0, 1.0, -1.0];
        let weight = vec![2.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        rms_norm_f32(&input, &weight, 1e-5, &mut output);

        // All |x[i]| equal, so rms = sqrt(mean(1) + eps) ≈ 1.0
        // output ≈ x[i] * 2 / 1 = ±2
        for &o in &output {
            assert!(
                (o.abs() - 2.0).abs() < 0.01,
                "output magnitude ≈ 2, got {o}"
            );
        }
    }

    #[test]
    fn rms_norm_zero_input_stays_zero() {
        let input = vec![0.0_f32; 8];
        let weight = vec![1.0_f32; 8];
        let mut output = vec![99.0_f32; 8]; // non-zero sentinel
        rms_norm_f32(&input, &weight, 1e-5, &mut output);
        assert!(
            output.iter().all(|&v| v == 0.0),
            "zero input must produce zero output"
        );
    }

    // ------------------------------------------------------------------
    // rope_cos_sin_table
    // ------------------------------------------------------------------

    #[test]
    fn rope_table_position_zero_cos_one_sin_zero() {
        // At pos=0: angle = 0 * freq = 0, so cos=1, sin=0 for all dims.
        let (cos, sin) = rope_cos_sin_table(4, 8, 10000.0);
        let half = 4_usize; // head_dim/2
        for i in 0..half {
            assert!((cos[i] - 1.0).abs() < 1e-6, "cos(0) must be 1 at dim {i}");
            assert!(sin[i].abs() < 1e-6, "sin(0) must be 0 at dim {i}");
        }
    }

    #[test]
    fn rope_table_shape() {
        let max_seq = 16;
        let head_dim = 64;
        let (cos, sin) = rope_cos_sin_table(max_seq, head_dim, 10000.0);
        let expected_len = max_seq * (head_dim / 2);
        assert_eq!(cos.len(), expected_len);
        assert_eq!(sin.len(), expected_len);
    }

    #[test]
    fn rope_table_cos_sin_identity() {
        // cos²(θ) + sin²(θ) = 1 for all entries.
        let (cos, sin) = rope_cos_sin_table(8, 16, 10000.0);
        for (c, s) in cos.iter().zip(sin.iter()) {
            let norm_sq = c * c + s * s;
            assert!((norm_sq - 1.0).abs() < 1e-5, "cos²+sin²={norm_sq} ≠ 1");
        }
    }

    // ------------------------------------------------------------------
    // apply_rope_to_head
    // ------------------------------------------------------------------

    #[test]
    fn rope_apply_zero_position_is_identity() {
        // At pos=0 all angles are 0, so cos=1, sin=0: rotation is identity.
        let (cos, sin) = rope_cos_sin_table(1, 8, 10000.0);
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = x.clone();
        apply_rope_to_head(&mut x, &cos[0..4], &sin[0..4]);
        for (i, (&orig, &rot)) in original.iter().zip(x.iter()).enumerate() {
            assert!(
                (orig - rot).abs() < 1e-5,
                "pos=0 RoPE must be identity at index {i}: {orig} vs {rot}"
            );
        }
    }

    #[test]
    fn rope_apply_preserves_norm() {
        // RoPE is a rotation — it preserves the L2 norm of each head.
        let (cos, sin) = rope_cos_sin_table(32, 8, 500_000.0);
        let half = 4_usize;
        for pos in 1..32_usize {
            let mut x = vec![0.3_f32, -0.5, 1.2, -0.8, 0.1, 0.6, -1.0, 0.4];
            let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

            let offset = pos * half;
            apply_rope_to_head(
                &mut x,
                &cos[offset..offset + half],
                &sin[offset..offset + half],
            );

            let norm_after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm_before - norm_after).abs() < 1e-4,
                "pos={pos}: norm changed from {norm_before} to {norm_after}"
            );
        }
    }

    // ------------------------------------------------------------------
    // softmax_f32
    // ------------------------------------------------------------------

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        softmax_f32(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1, got {sum}");
    }

    #[test]
    fn softmax_all_same_is_uniform() {
        let n = 8;
        let mut x = vec![0.5_f32; n];
        softmax_f32(&mut x);
        let expected = 1.0 / n as f32;
        for &v in &x {
            assert!(
                (v - expected).abs() < 1e-6,
                "uniform input → uniform output"
            );
        }
    }

    #[test]
    fn softmax_large_value_dominates() {
        // Very large value at index 2 should dominate after softmax.
        let mut x = vec![0.0_f32, 0.0, 100.0, 0.0];
        softmax_f32(&mut x);
        assert!(x[2] > 0.999, "dominant logit must have probability ≈ 1");
        assert!(x.iter().sum::<f32>() - 1.0 < 1e-5);
    }

    #[test]
    fn softmax_empty_noop() {
        let mut x: Vec<f32> = vec![];
        softmax_f32(&mut x); // must not panic
    }

    #[test]
    fn softmax_all_positive() {
        let mut x = vec![0.1_f32, 0.5, 2.0, -1.0];
        softmax_f32(&mut x);
        for &v in &x {
            assert!(v > 0.0, "all softmax outputs must be positive");
        }
    }

    // ------------------------------------------------------------------
    // squared_relu_f32
    // ------------------------------------------------------------------

    #[test]
    fn squared_relu_negative_becomes_zero() {
        let mut x = vec![-3.0_f32, -1.0, -0.001];
        squared_relu_f32(&mut x);
        assert!(x.iter().all(|&v| v == 0.0), "negative inputs → 0");
    }

    #[test]
    fn squared_relu_positive_is_squared() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 0.5];
        let expected = vec![1.0_f32, 4.0, 9.0, 0.25];
        squared_relu_f32(&mut x);
        for (i, (&v, &e)) in x.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-6, "index {i}: {v} ≠ {e}");
        }
    }

    #[test]
    fn squared_relu_zero_stays_zero() {
        let mut x = vec![0.0_f32; 4];
        squared_relu_f32(&mut x);
        assert!(x.iter().all(|&v| v == 0.0));
    }

    // ------------------------------------------------------------------
    // elementwise_mul_f32
    // ------------------------------------------------------------------

    #[test]
    fn elementwise_mul_basic() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![0.5_f32, -1.0, 2.0, 0.0];
        let mut out = vec![0.0_f32; 4];
        elementwise_mul_f32(&a, &b, &mut out);
        let expected = vec![0.5_f32, -2.0, 6.0, 0.0];
        for (i, (&v, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-7, "index {i}: {v} ≠ {e}");
        }
    }

    #[test]
    fn elementwise_mul_identity() {
        let a = vec![3.14_f32, -2.71, 0.0, 100.0];
        let b = vec![1.0_f32; 4];
        let mut out = vec![0.0_f32; 4];
        elementwise_mul_f32(&a, &b, &mut out);
        for (i, (&av, &ov)) in a.iter().zip(out.iter()).enumerate() {
            assert!(
                (av - ov).abs() < 1e-7,
                "index {i}: multiply by 1 is identity"
            );
        }
    }

    // ------------------------------------------------------------------
    // lm_head_matmul
    // ------------------------------------------------------------------

    #[test]
    fn lm_head_single_vocab_entry() {
        // vocab=1, hidden=3: logits[0] = 1*2 + (-1)*(-3) + 0.5*4 = 2+3+2 = 7
        let weights = vec![1.0_f32, -1.0, 0.5];
        let hidden = vec![2.0_f32, -3.0, 4.0];
        let logits = lm_head_matmul(&hidden, &weights, 1, 3);
        assert_eq!(logits.len(), 1);
        assert!(
            (logits[0] - 7.0).abs() < 1e-5,
            "logits[0] = 7, got {}",
            logits[0]
        );
    }

    #[test]
    fn lm_head_two_vocab_entries() {
        // vocab=2, hidden=2
        // weights = [[1, 0], [0, 1]]
        // hidden  = [3, 5]
        // logits  = [3*1 + 5*0, 3*0 + 5*1] = [3, 5]
        let weights = vec![1.0_f32, 0.0, 0.0, 1.0];
        let hidden = vec![3.0_f32, 5.0];
        let logits = lm_head_matmul(&hidden, &weights, 2, 2);
        assert_eq!(logits.len(), 2);
        assert!((logits[0] - 3.0).abs() < 1e-6);
        assert!((logits[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn lm_head_all_zero_weights() {
        let weights = vec![0.0_f32; 16];
        let hidden = vec![1.0_f32; 4];
        let logits = lm_head_matmul(&hidden, &weights, 4, 4);
        assert!(logits.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn lm_head_output_length_equals_vocab_size() {
        let vocab = 128;
        let hidden_size = 16;
        let weights = vec![0.1_f32; vocab * hidden_size];
        let hidden = vec![1.0_f32; hidden_size];
        let logits = lm_head_matmul(&hidden, &weights, vocab, hidden_size);
        assert_eq!(logits.len(), vocab);
    }
}
