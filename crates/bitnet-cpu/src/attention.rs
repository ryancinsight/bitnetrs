//! Causal Grouped Query Attention (GQA) for the CPU backend.
//!
//! # Mathematical Specification
//!
//! ## Grouped Query Attention
//!
//! BitNet b1.58 uses GQA where `n_kv_heads` key/value heads serve `n_heads`
//! query heads.  Each KV head serves `heads_per_group = n_heads / n_kv_heads`
//! query heads.
//!
//! For query head `h` (0 ≤ h < n_heads), the corresponding KV head is:
//!
//! ```text
//! kv_head(h) = h / heads_per_group    (integer division)
//! ```
//!
//! ## Scaled Dot-Product Attention
//!
//! For query head `h` at current position `cur_pos`, attending over
//! positions `0..=cur_pos` (causal mask):
//!
//! ```text
//! scale    = 1 / sqrt(head_dim)
//! score[t] = dot( q[h], k_cache[t, kv_head(h)] ) * scale
//!            for t in 0..=cur_pos
//!
//! attn     = softmax( score[0..=cur_pos] )
//!
//! out[h]   = Σ_{t=0}^{cur_pos}  attn[t] * v_cache[t, kv_head(h)]
//! ```
//!
//! The output of all heads is concatenated into a flat `[n_heads × head_dim]`
//! vector.
//!
//! ## KV Cache Layout
//!
//! Both `k_cache` and `v_cache` are flat head-major `f32` slices of shape
//! `[n_kv_heads × (cur_pos+1) × head_dim]`:
//!
//! ```text
//! cache[kv_h, t, d] = cache[ kv_h * ((cur_pos + 1) * head_dim)
//!                           + t * head_dim
//!                           + d ]
//! ```
//!
//! # Numerical Stability
//!
//! The softmax over attention scores uses the standard max-subtraction trick
//! (see [`bitnet_core::backend::ops::softmax_f32`]) to prevent overflow when
//! scores are large.
//!
//! # Invariants
//!
//! - `q.len() == n_heads * head_dim`
//! - `k_cache.len() >= n_kv_heads * (cur_pos + 1) * head_dim`
//! - `v_cache.len() >= n_kv_heads * (cur_pos + 1) * head_dim`
//! - `output.len() == n_heads * head_dim`
//! - `n_heads % n_kv_heads == 0`
//! - `head_dim > 0`

use std::cell::RefCell;

use bitnet_core::backend::ops::softmax_f32;
use bitnet_core::error::{BitNetError, Result};
use rayon::prelude::*;

thread_local! {
    static SCORE_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Causal scaled dot-product attention with Grouped Query Attention (GQA).
///
/// Computes the attended output for a single new query token at absolute
/// sequence position `cur_pos`, attending over positions `0..=cur_pos` in the
/// KV cache.
///
/// # Arguments
///
/// - `q`:        Query tensor, shape `[n_heads × head_dim]`.
/// - `k_cache`:  Key cache, shape `[n_kv_heads × (cur_pos+1) × head_dim]`.
/// - `v_cache`:  Value cache, shape `[n_kv_heads × (cur_pos+1) × head_dim]`.
/// - `output`:   Pre-allocated output, shape `[n_heads × head_dim]`.
/// - `n_heads`:  Number of query heads `H`.
/// - `n_kv_heads`: Number of key/value heads `H_kv` (must divide `H`).
/// - `head_dim`: Per-head feature dimension.
/// - `cur_pos`:  Current (0-indexed) sequence position.  The KV cache contains
///               `cur_pos + 1` positions (the current token's KV has already been stored).
///
/// # Errors
///
/// Returns [`BitNetError::InvalidShape`] if any shape is inconsistent.
/// Returns [`BitNetError::InvalidConfig`] if `n_heads % n_kv_heads != 0`.
pub fn masked_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    output: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cur_pos: usize,
) -> Result<()> {
    // ---- Validation ---------------------------------------------------------

    if head_dim == 0 {
        return Err(BitNetError::shape(
            "head_dim > 0".to_string(),
            "head_dim = 0".to_string(),
        ));
    }
    if n_kv_heads == 0 {
        return Err(BitNetError::config("n_kv_heads must be > 0".to_string()));
    }
    if n_heads == 0 {
        return Err(BitNetError::config("n_heads must be > 0".to_string()));
    }
    if n_heads % n_kv_heads != 0 {
        return Err(BitNetError::config(format!(
            "n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )));
    }

    let seq_len = cur_pos + 1; // number of positions in cache
    let head_stride = k_cache.len() / n_kv_heads; // elements per KV head in fixed-capacity cache
    let required_cache = n_kv_heads * head_stride;

    let q_expected = n_heads * head_dim;
    if q.len() != q_expected {
        return Err(BitNetError::shape(
            format!("q.len() == n_heads * head_dim = {n_heads} * {head_dim} = {q_expected}"),
            format!("q.len() = {}", q.len()),
        ));
    }
    if k_cache.len() < required_cache {
        return Err(BitNetError::shape(
            format!("k_cache.len() >= {required_cache}"),
            format!("k_cache.len() = {}", k_cache.len()),
        ));
    }
    if v_cache.len() < required_cache {
        return Err(BitNetError::shape(
            format!("v_cache.len() >= {required_cache}"),
            format!("v_cache.len() = {}", v_cache.len()),
        ));
    }
    if head_stride < seq_len * head_dim {
        return Err(BitNetError::shape(
            format!("per-head cache stride >= {}", seq_len * head_dim),
            format!("per-head cache stride = {head_stride}"),
        ));
    }
    if output.len() != q_expected {
        return Err(BitNetError::shape(
            format!("output.len() == n_heads * head_dim = {q_expected}"),
            format!("output.len() = {}", output.len()),
        ));
    }

    // ---- Attention computation ----------------------------------------------

    let scale = (head_dim as f32).sqrt().recip(); // 1 / sqrt(head_dim)
    let heads_per_group = n_heads / n_kv_heads;

    // Process heads in parallel using Rayon.
    // Each head writes to a disjoint slice of `output` (head_dim elements per head).
    output
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(h, out_head)| {
            let kv_h = h / heads_per_group;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];

            // Reuse a thread-local scores buffer to avoid per-head heap allocation.
            SCORE_BUF.with(|buf| {
                let mut scores = buf.borrow_mut();
                scores.clear();
                scores.resize(seq_len, 0.0_f32);

                // Compute attention scores: score[t] = dot(q_head, k_cache[kv_h, t]) * scale
                let k_head_base = kv_h * head_stride;
                for t in 0..seq_len {
                    let k_offset = k_head_base + t * head_dim;
                    let k_head = &k_cache[k_offset..k_offset + head_dim];
                    scores[t] = super::simd::dot_f32_f32_fast(q_head, k_head) * scale;
                }

                // Numerically-stable softmax over scores[0..seq_len].
                softmax_f32(&mut scores[..seq_len]);

                // Compute attended output: out_head = Σ_t attn[t] * v_cache[kv_h, t]
                for d in 0..head_dim {
                    out_head[d] = 0.0;
                }
                let v_head_base = kv_h * head_stride;
                for t in 0..seq_len {
                    let v_offset = v_head_base + t * head_dim;
                    let v_head = &v_cache[v_offset..v_offset + head_dim];
                    let attn_weight = scores[t];
                    super::simd::axpy_f32_fast(attn_weight, v_head, out_head);
                }
            });
        });

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a flat KV cache of shape [seq_len, n_kv_heads, head_dim] filled
    /// with constant `val` for each position t, kv head h, dim d.
    fn make_const_cache(seq_len: usize, n_kv_heads: usize, head_dim: usize, val: f32) -> Vec<f32> {
        vec![val; seq_len * n_kv_heads * head_dim]
    }

    /// Build a KV cache where cache[kv_h, t, d] = t as f32 * scale (unique per position).
    fn make_positional_cache(
        seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut cache = vec![0.0_f32; seq_len * n_kv_heads * head_dim];
        for kv_h in 0..n_kv_heads {
            for t in 0..seq_len {
                for d in 0..head_dim {
                    let idx = kv_h * seq_len * head_dim + t * head_dim + d;
                    cache[idx] = t as f32 * scale;
                }
            }
        }
        cache
    }

    // -----------------------------------------------------------------------
    // Correctness
    // -----------------------------------------------------------------------

    #[test]
    fn single_head_single_position_is_identity() {
        // With a single token (cur_pos=0), attention is trivially the value at pos 0.
        // output = softmax([score_0]) * v_0 = 1.0 * v_0 = v_0
        let head_dim = 4;
        let q = vec![1.0_f32, 0.0, 0.0, 0.0]; // 1 head × 4 dims
        let k_cache = vec![1.0_f32, 0.0, 0.0, 0.0]; // pos=0, kv_h=0
        let v_cache = vec![0.1_f32, 0.2, 0.3, 0.4]; // pos=0, kv_h=0

        let mut output = vec![0.0_f32; 4];
        masked_attention(&q, &k_cache, &v_cache, &mut output, 1, 1, head_dim, 0).unwrap();

        // With 1 position, softmax([score]) = [1.0], so out = v_cache[0]
        let expected = vec![0.1_f32, 0.2, 0.3, 0.4];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "output[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn uniform_scores_produce_averaged_values() {
        // If all attention scores are equal (e.g. query is zero → all dots = 0),
        // then softmax gives uniform distribution and output = mean(V).
        //
        // We set q=0 so all scores=0, softmax([0,0,...]) = [1/T, 1/T, ...].
        // v_cache rows are [1,0,...], [2,0,...], [3,0,...] → mean = [2, 0, ...].
        let head_dim = 2;
        let cur_pos = 2; // 3 positions: t=0,1,2
        let seq_len = 3;

        let q = vec![0.0_f32; head_dim]; // zero query → all scores equal
        let mut k_cache = vec![0.0_f32; seq_len * 1 * head_dim]; // n_kv_heads=1
                                                                 // Fill K with non-zero to avoid trivial test, but query is 0 so scores all = 0
        for i in 0..seq_len * head_dim {
            k_cache[i] = (i + 1) as f32;
        }
        let mut v_cache = vec![0.0_f32; seq_len * 1 * head_dim];
        // V[0, t, 0] = (t+1) as f32, V[0, t, 1] = 0
        for t in 0..seq_len {
            let base = t * head_dim;
            v_cache[base] = (t + 1) as f32; // [1, 2, 3]
            v_cache[base + 1] = 0.0;
        }

        let mut output = vec![0.0_f32; head_dim];
        masked_attention(&q, &k_cache, &v_cache, &mut output, 1, 1, head_dim, cur_pos).unwrap();

        // softmax([0,0,0]) = [1/3, 1/3, 1/3]
        // out[0] = (1 + 2 + 3) / 3 = 2.0
        // out[1] = 0.0
        assert!(
            (output[0] - 2.0).abs() < 1e-5,
            "uniform attn: out[0] should be 2.0, got {}",
            output[0]
        );
        assert!(
            output[1].abs() < 1e-5,
            "uniform attn: out[1] should be 0.0, got {}",
            output[1]
        );
    }

    #[test]
    fn high_score_for_last_position_attends_to_last_value() {
        // Make the query perfectly aligned with the last key → it dominates.
        // K[2] = [1, 0], Q = [1e6, 0] → score[2] >> score[0], score[1]
        // V[2] = [99, 0] → output ≈ [99, 0]
        let head_dim = 2;
        let cur_pos = 2; // t=0,1,2
        let seq_len = 3;

        let q = vec![1000.0_f32, 0.0];
        let mut k_cache = vec![0.0_f32; seq_len * head_dim]; // n_kv_heads=1
                                                             // K[0, 0] = [0, 0], K[0, 1] = [0, 0], K[0, 2] = [1, 0]
        k_cache[2 * head_dim] = 1.0;

        let mut v_cache = vec![0.0_f32; seq_len * head_dim];
        // V[0, 0] = [1, 0], V[0, 1] = [2, 0], V[0, 2] = [99, 0]
        v_cache[0 * head_dim] = 1.0;
        v_cache[1 * head_dim] = 2.0;
        v_cache[2 * head_dim] = 99.0;

        let mut output = vec![0.0_f32; head_dim];
        masked_attention(&q, &k_cache, &v_cache, &mut output, 1, 1, head_dim, cur_pos).unwrap();

        // Score[2] = 1000 * 1 / sqrt(2) >> score[0,1] = 0 → attn[2] ≈ 1.0
        // output[0] ≈ 99.0
        assert!(
            (output[0] - 99.0).abs() < 1.0,
            "dominant attention: expected ≈99, got {}",
            output[0]
        );
    }

    #[test]
    fn gqa_groups_correct_kv_head_assignment() {
        // n_heads=4, n_kv_heads=2 → heads_per_group=2
        // Q heads 0,1 use KV head 0; Q heads 2,3 use KV head 1.
        // We set KV head 0 to have V=[1,0] and KV head 1 to have V=[0,1]
        // and use zero queries (uniform attention).
        // Expected: out[0,1] = [1,0], out[2,3] = [0,1]
        let head_dim = 2;
        let n_heads = 4;
        let n_kv_heads = 2;
        let seq_len = 1; // only cur_pos=0

        let q = vec![0.0_f32; n_heads * head_dim]; // uniform attention
        let k_cache = vec![0.0_f32; seq_len * n_kv_heads * head_dim]; // all zeros → uniform

        // V cache: [kv_h=0, pos=0] = [1, 0], [kv_h=1, pos=0] = [0, 1]
        let mut v_cache = vec![0.0_f32; seq_len * n_kv_heads * head_dim];
        // layout: [kv_h=0, pos=0, d=0..1], [kv_h=1, pos=0, d=0..1]
        v_cache[0] = 1.0; // kv_h=0, pos=0, d=0
        v_cache[1] = 0.0; // kv_h=0, pos=0, d=1
        v_cache[2] = 0.0; // kv_h=1, pos=0, d=0
        v_cache[3] = 1.0; // kv_h=1, pos=0, d=1

        let mut output = vec![0.0_f32; n_heads * head_dim];
        masked_attention(
            &q,
            &k_cache,
            &v_cache,
            &mut output,
            n_heads,
            n_kv_heads,
            head_dim,
            0,
        )
        .unwrap();

        // Q heads 0,1 → KV head 0 → V=[1,0]
        assert!(
            (output[0] - 1.0).abs() < 1e-5,
            "head 0, dim 0: {}",
            output[0]
        );
        assert!(
            (output[1] - 0.0).abs() < 1e-5,
            "head 0, dim 1: {}",
            output[1]
        );
        assert!(
            (output[2] - 1.0).abs() < 1e-5,
            "head 1, dim 0: {}",
            output[2]
        );
        assert!(
            (output[3] - 0.0).abs() < 1e-5,
            "head 1, dim 1: {}",
            output[3]
        );

        // Q heads 2,3 → KV head 1 → V=[0,1]
        assert!(
            (output[4] - 0.0).abs() < 1e-5,
            "head 2, dim 0: {}",
            output[4]
        );
        assert!(
            (output[5] - 1.0).abs() < 1e-5,
            "head 2, dim 1: {}",
            output[5]
        );
        assert!(
            (output[6] - 0.0).abs() < 1e-5,
            "head 3, dim 0: {}",
            output[6]
        );
        assert!(
            (output[7] - 1.0).abs() < 1e-5,
            "head 3, dim 1: {}",
            output[7]
        );
    }

    #[test]
    fn attention_output_is_convex_combination_of_values() {
        // The output of attention must be a convex combination (weighted sum with
        // weights ≥ 0 summing to 1) of the value vectors for each head.
        //
        // Verify: for any query/key configuration, out[h] lies in the convex
        // hull of {V[t, kv_h(h)] : t = 0..=cur_pos}.
        //
        // Simple check: output[d] ∈ [min_t(V[t,d]), max_t(V[t,d])].
        let head_dim = 4;
        let cur_pos = 4; // 5 positions
        let seq_len = 5;

        // Random-ish query and keys.
        let q = vec![0.5_f32, -0.3, 0.7, 0.1];
        let mut k_cache = vec![0.0_f32; seq_len * head_dim];
        let mut v_cache = vec![0.0_f32; seq_len * head_dim];

        for t in 0..seq_len {
            for d in 0..head_dim {
                let idx = t * head_dim + d;
                k_cache[idx] = (t as f32 * 0.1 + d as f32 * 0.2) - 0.5;
                v_cache[idx] = (t + 1) as f32 * (d + 1) as f32 * 0.1;
            }
        }

        let mut output = vec![0.0_f32; head_dim];
        masked_attention(&q, &k_cache, &v_cache, &mut output, 1, 1, head_dim, cur_pos).unwrap();

        for d in 0..head_dim {
            let min_v = (0..seq_len)
                .map(|t| v_cache[t * head_dim + d])
                .fold(f32::INFINITY, f32::min);
            let max_v = (0..seq_len)
                .map(|t| v_cache[t * head_dim + d])
                .fold(f32::NEG_INFINITY, f32::max);

            assert!(
                output[d] >= min_v - 1e-5 && output[d] <= max_v + 1e-5,
                "dim {d}: output={} not in [{min_v}, {max_v}]",
                output[d]
            );
        }
    }

    #[test]
    fn output_all_finite() {
        // Ensure no NaN or Inf leaks through for typical inputs.
        let head_dim = 8;
        let cur_pos = 7;
        let seq_len = 8;
        let n_heads = 4;
        let n_kv_heads = 2;

        let q: Vec<f32> = (0..(n_heads * head_dim))
            .map(|i| (i as f32 - 16.0) * 0.1)
            .collect();
        let k_cache: Vec<f32> = (0..(seq_len * n_kv_heads * head_dim))
            .map(|i| (i as f32 - 32.0) * 0.05)
            .collect();
        let v_cache: Vec<f32> = (0..(seq_len * n_kv_heads * head_dim))
            .map(|i| (i as f32) * 0.01)
            .collect();

        let mut output = vec![0.0_f32; n_heads * head_dim];
        masked_attention(
            &q,
            &k_cache,
            &v_cache,
            &mut output,
            n_heads,
            n_kv_heads,
            head_dim,
            cur_pos,
        )
        .unwrap();

        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        }
    }

    // -----------------------------------------------------------------------
    // Error conditions
    // -----------------------------------------------------------------------

    #[test]
    fn wrong_q_length_returns_error() {
        let q = vec![0.0_f32; 7]; // should be 8
        let k = make_const_cache(1, 2, 4, 0.0);
        let v = make_const_cache(1, 2, 4, 0.0);
        let mut out = vec![0.0_f32; 8];
        let err = masked_attention(&q, &k, &v, &mut out, 2, 2, 4, 0).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn wrong_output_length_returns_error() {
        let q = vec![0.0_f32; 8];
        let k = make_const_cache(1, 2, 4, 0.0);
        let v = make_const_cache(1, 2, 4, 0.0);
        let mut out = vec![0.0_f32; 6]; // should be 8
        let err = masked_attention(&q, &k, &v, &mut out, 2, 2, 4, 0).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn k_cache_too_short_returns_error() {
        let q = vec![0.0_f32; 8];
        let k = vec![0.0_f32; 3]; // too short for 1 pos × 2 kv_heads × 4 head_dim = 8
        let v = make_const_cache(1, 2, 4, 0.0);
        let mut out = vec![0.0_f32; 8];
        let err = masked_attention(&q, &k, &v, &mut out, 2, 2, 4, 0).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn v_cache_too_short_returns_error() {
        let q = vec![0.0_f32; 8];
        let k = make_const_cache(1, 2, 4, 0.0);
        let v = vec![0.0_f32; 3]; // too short
        let mut out = vec![0.0_f32; 8];
        let err = masked_attention(&q, &k, &v, &mut out, 2, 2, 4, 0).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn n_heads_not_divisible_by_n_kv_heads_returns_error() {
        let q = vec![0.0_f32; 12]; // 3 * 4
        let k = make_const_cache(1, 2, 4, 0.0);
        let v = make_const_cache(1, 2, 4, 0.0);
        let mut out = vec![0.0_f32; 12];
        // n_heads=3, n_kv_heads=2 → 3 % 2 != 0
        let err = masked_attention(&q, &k, &v, &mut out, 3, 2, 4, 0).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidConfig(_)));
    }

    #[test]
    fn zero_head_dim_returns_error() {
        let q = vec![];
        let k = vec![];
        let v = vec![];
        let mut out = vec![];
        let err = masked_attention(&q, &k, &v, &mut out, 1, 1, 0, 0).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn zero_n_kv_heads_returns_error() {
        let q = vec![0.0_f32; 4];
        let k = vec![];
        let v = vec![];
        let mut out = vec![0.0_f32; 4];
        let err = masked_attention(&q, &k, &v, &mut out, 1, 0, 4, 0).unwrap_err();
        assert!(matches!(err, BitNetError::InvalidConfig(_)));
    }

    // -----------------------------------------------------------------------
    // Scale invariance
    // -----------------------------------------------------------------------

    /// Property: scaling the value cache by α scales the output by α.
    /// This holds because attention weights sum to 1 (softmax).
    #[test]
    fn output_scales_linearly_with_values() {
        let head_dim = 4;
        let cur_pos = 3;
        let seq_len = 4;

        let q: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.1).collect();
        let k_cache = make_positional_cache(seq_len, 1, head_dim, 0.1);
        let v1 = make_positional_cache(seq_len, 1, head_dim, 0.5);
        let v2: Vec<f32> = v1.iter().map(|&x| x * 3.0).collect(); // scaled by 3

        let mut out1 = vec![0.0_f32; head_dim];
        let mut out2 = vec![0.0_f32; head_dim];

        masked_attention(&q, &k_cache, &v1, &mut out1, 1, 1, head_dim, cur_pos).unwrap();
        masked_attention(&q, &k_cache, &v2, &mut out2, 1, 1, head_dim, cur_pos).unwrap();

        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (3.0 * a - b).abs() < 1e-5,
                "dim {i}: 3*out1={}, out2={}",
                3.0 * a,
                b
            );
        }
    }

    /// Property: attention output is invariant to adding a constant to all K
    /// vectors (the softmax subtracts max, and scores differ only in relative terms).
    ///
    /// Not quite: adding a constant c to all K*Q dot products shifts all scores
    /// by Q·c, which softmax neutralises via its shift-invariance.
    #[test]
    fn softmax_shift_invariance_in_attention() {
        let head_dim = 4;
        let cur_pos = 2;
        let seq_len = 3;

        let q = vec![0.5_f32, -0.3, 0.1, 0.8];

        // Original K cache
        let k_orig: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 - 6.0) * 0.2)
            .collect();

        // Shifted K cache: add [1, 1, 1, 1] to every key vector
        // dot(q, k_shifted[t]) = dot(q, k_orig[t]) + dot(q, [1,1,1,1])
        // So all scores shift by the same constant → softmax is unchanged.
        let q_sum: f32 = q.iter().sum();
        let k_shifted: Vec<f32> = (0..seq_len)
            .flat_map(|t| {
                let k_ref = &k_orig;
                (0..head_dim)
                    .map(move |d| k_ref[t * head_dim + d] + 1.0)
                    .collect::<Vec<f32>>()
            })
            .collect();

        let v_cache = make_positional_cache(seq_len, 1, head_dim, 0.3);

        let mut out_orig = vec![0.0_f32; head_dim];
        let mut out_shifted = vec![0.0_f32; head_dim];

        masked_attention(
            &q,
            &k_orig,
            &v_cache,
            &mut out_orig,
            1,
            1,
            head_dim,
            cur_pos,
        )
        .unwrap();
        masked_attention(
            &q,
            &k_shifted,
            &v_cache,
            &mut out_shifted,
            1,
            1,
            head_dim,
            cur_pos,
        )
        .unwrap();

        // The shift adds q_sum * scale to all scores equally → softmax unchanged → outputs equal.
        let _ = q_sum; // used in the explanation
        for (i, (&a, &b)) in out_orig.iter().zip(out_shifted.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "dim {i}: out_orig={a}, out_shifted={b}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Bitnet 2B config sanity
    // -----------------------------------------------------------------------

    /// Smoke test with the actual 2B model dimensions to ensure no shape errors.
    #[test]
    fn bitnet_2b_dimensions_smoke_test() {
        // n_heads=20, n_kv_heads=5, head_dim=128
        let n_heads = 20;
        let n_kv_heads = 5;
        let head_dim = 128;
        let cur_pos = 3; // prefill 4 tokens

        let q = vec![0.0_f32; n_heads * head_dim];
        let k_cache = vec![0.0_f32; (cur_pos + 1) * n_kv_heads * head_dim];
        let v_cache = vec![0.1_f32; (cur_pos + 1) * n_kv_heads * head_dim];
        let mut output = vec![0.0_f32; n_heads * head_dim];

        masked_attention(
            &q,
            &k_cache,
            &v_cache,
            &mut output,
            n_heads,
            n_kv_heads,
            head_dim,
            cur_pos,
        )
        .unwrap();

        // With q=0, all scores=0 → uniform attn → out = mean(V) = 0.1 per dim
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 0.1).abs() < 1e-5, "dim {i}: expected 0.1, got {v}");
        }
    }
}
