//! Rotary Position Embedding (RoPE) for the CPU backend.
//!
//! # Mathematical Foundation
//!
//! RoPE encodes absolute sequence positions as rotations in 2D subspaces of
//! the query and key head vectors.  For a head vector **x** ∈ ℝ^{head_dim}
//! at sequence position `pos`, the rotation for dimension pair `(2i, 2i+1)` is:
//!
//! ```text
//! θ_i     = pos / rope_theta^(2i / head_dim)   for i = 0 … head_dim/2 − 1
//!
//! x'_{2i}   =  x_{2i}   * cos(θ_i) − x_{2i+1} * sin(θ_i)
//! x'_{2i+1} =  x_{2i}   * sin(θ_i) + x_{2i+1} * cos(θ_i)
//! ```
//!
//! This is equivalent to complex multiplication:
//!
//! ```text
//! (x_{2i} + i·x_{2i+1}) * (cos(θ_i) + i·sin(θ_i))
//! ```
//!
//! The frequencies decay geometrically:
//!
//! ```text
//! freq_i = rope_theta^(−2i / head_dim)
//! ```
//!
//! Low-index dimensions rotate fast (high frequency), high-index dimensions
//! rotate slowly (low frequency), providing multi-scale positional information.
//!
//! # BitNet b1.58 Model Parameters
//!
//! - `rope_theta` = 500 000.0  (extended-context LLaMA 3 variant)
//! - `head_dim`   = 128        (2560 hidden / 20 heads)
//! - Applied to both Q (`n_heads = 20`) and K (`n_kv_heads = 5`) projections.
//!
//! # Efficiency
//!
//! The cos/sin values can be pre-computed once for all positions and cached.
//! The [`RopeCache`] struct provides this functionality.  For single-token
//! autoregressive decoding, the cache is indexed by the current sequence
//! position.
//!
//! # Invariants
//!
//! - `q.len() == n_heads    * head_dim`
//! - `k.len() == n_kv_heads * head_dim`
//! - `head_dim` is even (RoPE requires paired dimensions)
//! - `rope_theta > 0`
//! - All modifications are in-place; original vectors are overwritten.

use bitnet_core::error::{BitNetError, Result};

// ---------------------------------------------------------------------------
// RopeCache
// ---------------------------------------------------------------------------

/// Pre-computed cosine and sine tables for all positions up to `max_seq`.
///
/// # Layout
///
/// Both `cos` and `sin` are flat `Vec<f32>` of shape `[max_seq, half_head_dim]`
/// in row-major order:
///
/// ```text
/// cos[pos * half_head_dim + i] = cos( pos / rope_theta^(2i / head_dim) )
/// sin[pos * half_head_dim + i] = sin( pos / rope_theta^(2i / head_dim) )
/// ```
///
/// where `half_head_dim = head_dim / 2`.
#[derive(Debug, Clone)]
pub struct RopeCache {
    /// Cosine table, shape `[max_seq, half_head_dim]`.
    pub cos: Vec<f32>,
    /// Sine table, shape `[max_seq, half_head_dim]`.
    pub sin: Vec<f32>,
    /// Maximum sequence length this cache was built for.
    pub max_seq: usize,
    /// Per-head feature dimension (must be even).
    pub head_dim: usize,
    /// Half of `head_dim`: `head_dim / 2`.
    pub half_head_dim: usize,
    /// RoPE base frequency θ.
    pub theta: f32,
}

impl RopeCache {
    /// Build the full cos/sin tables for positions `0..max_seq`.
    ///
    /// # Arguments
    ///
    /// - `max_seq`:   Maximum sequence length to pre-compute.
    /// - `head_dim`:  Per-head dimension (must be even and > 0).
    /// - `theta`:     RoPE base frequency (e.g. 500 000.0).
    ///
    /// # Errors
    ///
    /// Returns [`BitNetError::InvalidConfig`] if `head_dim` is zero or odd, or
    /// if `theta` is non-positive.
    pub fn new(max_seq: usize, head_dim: usize, theta: f32) -> Result<Self> {
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(BitNetError::config(format!(
                "head_dim must be even and > 0, got {head_dim}"
            )));
        }
        if theta <= 0.0 || !theta.is_finite() {
            return Err(BitNetError::config(format!(
                "theta must be finite and > 0, got {theta}"
            )));
        }
        if max_seq == 0 {
            return Ok(Self {
                cos: Vec::new(),
                sin: Vec::new(),
                max_seq,
                head_dim,
                half_head_dim: head_dim / 2,
                theta,
            });
        }

        let half = head_dim / 2;
        let total = max_seq * half;
        let mut cos = Vec::with_capacity(total);
        let mut sin = Vec::with_capacity(total);

        for pos in 0..max_seq {
            for i in 0..half {
                // freq_i = theta^(−2i / head_dim)
                // angle_i = pos * freq_i
                let exponent = -2.0_f32 * i as f32 / head_dim as f32;
                let freq = theta.powf(exponent);
                let angle = pos as f32 * freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }

        Ok(Self {
            cos,
            sin,
            max_seq,
            head_dim,
            half_head_dim: half,
            theta,
        })
    }

    /// Return the cos/sin slices for a single sequence position.
    ///
    /// Returns `(&cos[pos*half..], &sin[pos*half..])`, each of length `half_head_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`BitNetError::InvalidShape`] if `pos >= max_seq`.
    #[inline]
    pub fn at(&self, pos: usize) -> Result<(&[f32], &[f32])> {
        if pos >= self.max_seq {
            return Err(BitNetError::shape(
                format!("pos < max_seq = {}", self.max_seq),
                format!("pos = {pos}"),
            ));
        }
        let half = self.half_head_dim;
        let offset = pos * half;
        Ok((
            &self.cos[offset..offset + half],
            &self.sin[offset..offset + half],
        ))
    }
}

// ---------------------------------------------------------------------------
// apply_rope
// ---------------------------------------------------------------------------

/// Apply Rotary Position Embeddings in-place to the Q and K tensors for a
/// single token at sequence position `position`.
///
/// Both `q` and `k` are flat, row-major tensors:
/// - `q`: `[n_heads    × head_dim]`
/// - `k`: `[n_kv_heads × head_dim]`
///
/// The rotation for each head `h` and dimension pair `(2i, 2i+1)`:
///
/// ```text
/// θ_i     = position / rope_theta^(2i / head_dim)
/// q'[h, 2i]   = q[h, 2i]   * cos(θ_i) − q[h, 2i+1] * sin(θ_i)
/// q'[h, 2i+1] = q[h, 2i]   * sin(θ_i) + q[h, 2i+1] * cos(θ_i)
/// ```
/// (identical for k).
///
/// # Arguments
///
/// - `q`:         Query tensor, shape `[n_heads × head_dim]`.  Modified in-place.
/// - `k`:         Key tensor, shape `[n_kv_heads × head_dim]`.  Modified in-place.
/// - `position`:  Absolute sequence position of this token (0-indexed).
/// - `head_dim`:  Per-head feature dimension (must be even).
/// - `n_heads`:   Number of query heads.
/// - `n_kv_heads`: Number of key/value heads.
/// - `theta`:     RoPE base frequency θ.
///
/// # Errors
///
/// Returns [`BitNetError::InvalidShape`] if:
/// - `q.len() != n_heads * head_dim`
/// - `k.len() != n_kv_heads * head_dim`
/// - `head_dim` is zero or odd
///
/// Returns [`BitNetError::InvalidConfig`] if `theta <= 0`.
///
/// # Example
///
/// ```
/// use bitnet_cpu::rope::apply_rope;
///
/// // Single query head (head_dim=4), no KV head to update.
/// let mut q = vec![1.0_f32, 0.0, 0.0, 1.0]; // 1 head × 4 dims
/// let mut k = vec![1.0_f32, 0.0];             // 1 kv_head × 2 dims — wrong!
/// // Correct: k must also have head_dim=4 elements per head.
/// let mut k = vec![1.0_f32, 0.0, 0.0, 1.0];
/// apply_rope(&mut q, &mut k, 0, 4, 1, 1, 10000.0).unwrap();
/// // At pos=0 all angles=0, so cos=1, sin=0 → q and k are unchanged.
/// assert!((q[0] - 1.0).abs() < 1e-6);
/// assert!((q[1] - 0.0).abs() < 1e-6);
/// ```
pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    position: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    theta: f32,
) -> Result<()> {
    // ---- Validation ---------------------------------------------------------
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(BitNetError::shape(
            "head_dim must be even and > 0".to_string(),
            format!("head_dim = {head_dim}"),
        ));
    }
    if theta <= 0.0 || !theta.is_finite() {
        return Err(BitNetError::config(format!(
            "theta must be finite and > 0, got {theta}"
        )));
    }

    let q_expected = n_heads * head_dim;
    if q.len() != q_expected {
        return Err(BitNetError::shape(
            format!("q.len() == n_heads * head_dim = {n_heads} * {head_dim} = {q_expected}"),
            format!("q.len() = {}", q.len()),
        ));
    }
    let k_expected = n_kv_heads * head_dim;
    if k.len() != k_expected {
        return Err(BitNetError::shape(
            format!("k.len() == n_kv_heads * head_dim = {n_kv_heads} * {head_dim} = {k_expected}"),
            format!("k.len() = {}", k.len()),
        ));
    }

    // ---- Pre-compute angles for this position --------------------------------
    // angles[i] = position / theta^(2i / head_dim)
    let half = head_dim / 2;
    let (cos_vals, sin_vals) = compute_cos_sin(position, head_dim, theta);

    // ---- Apply to Q ----------------------------------------------------------
    for h in 0..n_heads {
        let head_offset = h * head_dim;
        apply_rope_to_head_slice(
            &mut q[head_offset..head_offset + head_dim],
            &cos_vals,
            &sin_vals,
            half,
        );
    }

    // ---- Apply to K ----------------------------------------------------------
    for h in 0..n_kv_heads {
        let head_offset = h * head_dim;
        apply_rope_to_head_slice(
            &mut k[head_offset..head_offset + head_dim],
            &cos_vals,
            &sin_vals,
            half,
        );
    }

    Ok(())
}

/// Apply RoPE using a pre-built [`RopeCache`], avoiding repeated angle computation.
///
/// This is the preferred variant for autoregressive inference where many tokens
/// are processed sequentially and the cache is built once.
///
/// # Arguments
///
/// - `q`:       Query tensor `[n_heads × head_dim]`. Modified in-place.
/// - `k`:       Key tensor `[n_kv_heads × head_dim]`. Modified in-place.
/// - `position`: Sequence position of this token.
/// - `n_heads`:  Number of query heads.
/// - `n_kv_heads`: Number of key/value heads.
/// - `cache`:   Pre-built `RopeCache` for the model's `head_dim` and `theta`.
///
/// # Errors
///
/// Returns [`BitNetError::InvalidShape`] if shapes are inconsistent or
/// `position >= cache.max_seq`.
pub fn apply_rope_cached(
    q: &mut [f32],
    k: &mut [f32],
    position: usize,
    n_heads: usize,
    n_kv_heads: usize,
    cache: &RopeCache,
) -> Result<()> {
    let head_dim = cache.head_dim;
    let half = cache.half_head_dim;

    // Shape validation.
    let q_expected = n_heads * head_dim;
    if q.len() != q_expected {
        return Err(BitNetError::shape(
            format!("q.len() == {n_heads} * {head_dim} = {q_expected}"),
            format!("q.len() = {}", q.len()),
        ));
    }
    let k_expected = n_kv_heads * head_dim;
    if k.len() != k_expected {
        return Err(BitNetError::shape(
            format!("k.len() == {n_kv_heads} * {head_dim} = {k_expected}"),
            format!("k.len() = {}", k.len()),
        ));
    }

    // Retrieve pre-computed cos/sin for this position.
    let (cos_vals, sin_vals) = cache.at(position)?;

    // Apply to each query head.
    for h in 0..n_heads {
        let offset = h * head_dim;
        apply_rope_to_head_slice(&mut q[offset..offset + head_dim], cos_vals, sin_vals, half);
    }

    // Apply to each key head.
    for h in 0..n_kv_heads {
        let offset = h * head_dim;
        apply_rope_to_head_slice(&mut k[offset..offset + head_dim], cos_vals, sin_vals, half);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute cosine and sine values for all `half = head_dim/2` dimension pairs
/// at a given sequence `position`.
///
/// Returns two `Vec<f32>` each of length `half`, where:
/// - `cos[i] = cos( position / theta^(2i / head_dim) )`
/// - `sin[i] = sin( position / theta^(2i / head_dim) )`
#[inline]
fn compute_cos_sin(position: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos_vals = Vec::with_capacity(half);
    let mut sin_vals = Vec::with_capacity(half);

    for i in 0..half {
        let exponent = -2.0_f32 * i as f32 / head_dim as f32;
        let freq = theta.powf(exponent);
        let angle = position as f32 * freq;
        cos_vals.push(angle.cos());
        sin_vals.push(angle.sin());
    }

    (cos_vals, sin_vals)
}

/// Apply RoPE rotation to a single head vector in-place using the
/// **LLaMA 3 / HuggingFace half-and-half convention**.
///
/// The head vector of dimension `head_dim = 2 * half` is treated as two
/// halves: `x_lo = x[0..half]` and `x_hi = x[half..2*half]`.  For each
/// index `i` in `0..half`, the rotation pairs element `i` with element
/// `i + half`:
///
/// ```text
/// x'[i]        = x[i]        * cos[i]  -  x[i + half] * sin[i]
/// x'[i + half] = x[i + half] * cos[i]  +  x[i]        * sin[i]
/// ```
///
/// This matches `rotate_half` + `apply_rotary_pos_emb` from the HuggingFace
/// `transformers` LLaMA 3 implementation, where:
///
/// ```python
/// def rotate_half(x):
///     x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
///     return torch.cat((-x2, x1), dim=-1)
/// q_embed = q * cos + rotate_half(q) * sin
/// ```
///
/// **Important**: this is different from the "interleaved" convention used in
/// some earlier implementations (e.g. GPT-NeoX) where pairs are `(2i, 2i+1)`.
/// Using the wrong convention produces completely wrong positional encodings.
///
/// # Panics (debug only)
/// Panics if `head_slice.len() != 2 * half` or `cos.len() != half`.
#[inline]
fn apply_rope_to_head_slice(head_slice: &mut [f32], cos: &[f32], sin: &[f32], half: usize) {
    debug_assert_eq!(head_slice.len(), 2 * half, "head dimension mismatch");
    debug_assert_eq!(cos.len(), half, "cos table length mismatch");
    debug_assert_eq!(sin.len(), half, "sin table length mismatch");

    for i in 0..half {
        let x_lo = head_slice[i];
        let x_hi = head_slice[i + half];
        let c = cos[i];
        let s = sin[i];
        // LLaMA 3 half-and-half rotation:
        //   x'[i]        =  x_lo * cos[i] - x_hi * sin[i]
        //   x'[i + half] =  x_hi * cos[i] + x_lo * sin[i]
        head_slice[i] = x_lo * c - x_hi * s;
        head_slice[i + half] = x_hi * c + x_lo * s;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // RopeCache
    // -----------------------------------------------------------------------

    #[test]
    fn rope_cache_shape() {
        let max_seq = 32;
        let head_dim = 16;
        let cache = RopeCache::new(max_seq, head_dim, 10000.0).unwrap();
        let expected_len = max_seq * (head_dim / 2);
        assert_eq!(cache.cos.len(), expected_len, "cos table length");
        assert_eq!(cache.sin.len(), expected_len, "sin table length");
        assert_eq!(cache.half_head_dim, head_dim / 2);
    }

    #[test]
    fn rope_cache_position_zero_is_identity() {
        // At pos=0: angle = 0 for all dims, so cos=1 and sin=0.
        let cache = RopeCache::new(4, 8, 10000.0).unwrap();
        let (cos_pos0, sin_pos0) = cache.at(0).unwrap();
        for (i, &c) in cos_pos0.iter().enumerate() {
            assert!((c - 1.0).abs() < 1e-6, "cos[0, {i}] must be 1.0, got {c}");
        }
        for (i, &s) in sin_pos0.iter().enumerate() {
            assert!(s.abs() < 1e-6, "sin[0, {i}] must be 0.0, got {s}");
        }
    }

    #[test]
    fn rope_cache_cos_sin_unit_norm() {
        // cos²(θ) + sin²(θ) = 1 for all entries (Pythagorean identity).
        let cache = RopeCache::new(16, 32, 500_000.0).unwrap();
        for (c, s) in cache.cos.iter().zip(cache.sin.iter()) {
            let norm_sq = c * c + s * s;
            assert!((norm_sq - 1.0).abs() < 1e-5, "cos²+sin²={norm_sq} ≠ 1.0");
        }
    }

    #[test]
    fn rope_cache_out_of_bounds_returns_error() {
        let cache = RopeCache::new(4, 8, 10000.0).unwrap();
        let err = cache.at(4).unwrap_err(); // max_seq=4, so pos=4 is OOB
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::InvalidShape { .. }
        ));
    }

    #[test]
    fn rope_cache_odd_head_dim_returns_error() {
        let err = RopeCache::new(4, 7, 10000.0).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::InvalidConfig(_)
        ));
    }

    #[test]
    fn rope_cache_zero_theta_returns_error() {
        let err = RopeCache::new(4, 8, 0.0).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::InvalidConfig(_)
        ));
    }

    #[test]
    fn rope_cache_empty_max_seq() {
        // max_seq=0 should succeed but produce empty tables.
        let cache = RopeCache::new(0, 8, 10000.0).unwrap();
        assert!(cache.cos.is_empty());
        assert!(cache.sin.is_empty());
    }

    // -----------------------------------------------------------------------
    // apply_rope
    // -----------------------------------------------------------------------

    #[test]
    fn rope_position_zero_is_identity() {
        // At pos=0, all angles=0, so RoPE is the identity transformation.
        let head_dim = 8;
        let n_heads = 2;
        let n_kv = 1;
        let mut q: Vec<f32> = (0..(n_heads * head_dim)).map(|i| i as f32 * 0.1).collect();
        let mut k: Vec<f32> = (0..(n_kv * head_dim)).map(|i| i as f32 * 0.2).collect();
        let q_orig = q.clone();
        let k_orig = k.clone();

        apply_rope(&mut q, &mut k, 0, head_dim, n_heads, n_kv, 10000.0).unwrap();

        for (i, (&orig, &rot)) in q_orig.iter().zip(q.iter()).enumerate() {
            assert!(
                (orig - rot).abs() < 1e-5,
                "q[{i}]: pos=0 RoPE must be identity, got {rot}, expected {orig}"
            );
        }
        for (i, (&orig, &rot)) in k_orig.iter().zip(k.iter()).enumerate() {
            assert!(
                (orig - rot).abs() < 1e-5,
                "k[{i}]: pos=0 RoPE must be identity"
            );
        }
    }

    #[test]
    fn rope_preserves_head_norm() {
        // RoPE is an isometry (rotation) — it must preserve the L2 norm of each head.
        let head_dim = 16;
        let n_heads = 4;
        let n_kv = 2;
        let theta = 500_000.0_f32;

        for pos in [1, 5, 17, 100, 1000] {
            let mut q: Vec<f32> = (0..(n_heads * head_dim))
                .map(|i| ((i as f32 * 0.37 + 0.11) * 2.0 - 1.0))
                .collect();
            let mut k: Vec<f32> = (0..(n_kv * head_dim))
                .map(|i| ((i as f32 * 0.53 + 0.07) * 2.0 - 1.0))
                .collect();

            // Compute norms before rotation.
            let q_norms_before: Vec<f32> = q
                .chunks(head_dim)
                .map(|h| h.iter().map(|v| v * v).sum::<f32>().sqrt())
                .collect();
            let k_norms_before: Vec<f32> = k
                .chunks(head_dim)
                .map(|h| h.iter().map(|v| v * v).sum::<f32>().sqrt())
                .collect();

            apply_rope(&mut q, &mut k, pos, head_dim, n_heads, n_kv, theta).unwrap();

            // Verify norms are preserved.
            for (h, (bef, aft)) in q_norms_before
                .iter()
                .zip(
                    q.chunks(head_dim)
                        .map(|hv| hv.iter().map(|v| v * v).sum::<f32>().sqrt()),
                )
                .enumerate()
            {
                assert!(
                    (bef - aft).abs() < 1e-4,
                    "pos={pos}, q_head={h}: norm changed from {bef} to {aft}"
                );
            }
            for (h, (bef, aft)) in k_norms_before
                .iter()
                .zip(
                    k.chunks(head_dim)
                        .map(|hv| hv.iter().map(|v| v * v).sum::<f32>().sqrt()),
                )
                .enumerate()
            {
                assert!(
                    (bef - aft).abs() < 1e-4,
                    "pos={pos}, k_head={h}: norm changed from {bef} to {aft}"
                );
            }
        }
    }

    #[test]
    fn rope_wrong_q_length_returns_error() {
        let mut q = vec![0.0_f32; 15]; // should be 2*8=16
        let mut k = vec![0.0_f32; 8];
        let err = apply_rope(&mut q, &mut k, 0, 8, 2, 1, 10000.0).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::InvalidShape { .. }
        ));
    }

    #[test]
    fn rope_wrong_k_length_returns_error() {
        let mut q = vec![0.0_f32; 16];
        let mut k = vec![0.0_f32; 7]; // should be 1*8=8
        let err = apply_rope(&mut q, &mut k, 0, 8, 2, 1, 10000.0).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::InvalidShape { .. }
        ));
    }

    #[test]
    fn rope_odd_head_dim_returns_error() {
        let mut q = vec![0.0_f32; 7];
        let mut k = vec![0.0_f32; 7];
        let err = apply_rope(&mut q, &mut k, 0, 7, 1, 1, 10000.0).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::InvalidShape { .. }
        ));
    }

    #[test]
    fn rope_zero_theta_returns_error() {
        let mut q = vec![0.0_f32; 8];
        let mut k = vec![0.0_f32; 8];
        let err = apply_rope(&mut q, &mut k, 0, 8, 1, 1, 0.0).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::InvalidConfig(_)
        ));
    }

    #[test]
    fn rope_90_degree_rotation_check() {
        // For the first dimension pair (i=0), at position pos=1:
        // freq_0 = theta^0 = 1.0  (exponent = -2*0/head_dim = 0)
        // angle = pos * 1.0 = 1.0 rad
        // x = [cos(0), sin(0)] = [1.0, 0.0]
        // After RoPE: x' = [cos(1), sin(1)]
        let theta = 10000.0_f32;
        let head_dim = 2; // minimal: 1 dimension pair
        let mut q = vec![1.0_f32, 0.0]; // [x0, x1] for 1 head
        let mut k = vec![1.0_f32, 0.0];

        apply_rope(&mut q, &mut k, 1, head_dim, 1, 1, theta).unwrap();

        // freq_0 = theta^(-2*0/2) = theta^0 = 1.0
        // angle = 1 * 1.0 = 1.0 rad
        let expected_q0 = 1.0_f32 * (1.0_f32).cos() - 0.0_f32 * (1.0_f32).sin();
        let expected_q1 = 1.0_f32 * (1.0_f32).sin() + 0.0_f32 * (1.0_f32).cos();
        assert!(
            (q[0] - expected_q0).abs() < 1e-5,
            "q[0]: got {}, expected {}",
            q[0],
            expected_q0
        );
        assert!(
            (q[1] - expected_q1).abs() < 1e-5,
            "q[1]: got {}, expected {}",
            q[1],
            expected_q1
        );
    }

    #[test]
    fn rope_different_positions_give_different_rotations() {
        // Two tokens at positions 1 and 2 should have different RoPE rotations.
        let head_dim = 8;
        let initial_q = vec![1.0_f32, 0.5, -0.3, 0.8, 0.1, -0.6, 0.4, -0.2];

        let mut q1 = initial_q.clone();
        let mut k1 = vec![0.0_f32; 8];
        apply_rope(&mut q1, &mut k1, 1, head_dim, 1, 1, 10000.0).unwrap();

        let mut q2 = initial_q.clone();
        let mut k2 = vec![0.0_f32; 8];
        apply_rope(&mut q2, &mut k2, 2, head_dim, 1, 1, 10000.0).unwrap();

        // They must differ at some dimension.
        let total_diff: f32 = q1.iter().zip(q2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            total_diff > 1e-4,
            "different positions must yield different rotations; total_diff={total_diff}"
        );
    }

    // -----------------------------------------------------------------------
    // apply_rope_cached
    // -----------------------------------------------------------------------

    #[test]
    fn cached_matches_uncached() {
        let max_seq = 64;
        let head_dim = 16;
        let n_heads = 4;
        let n_kv = 2;
        let theta = 500_000.0_f32;
        let cache = RopeCache::new(max_seq, head_dim, theta).unwrap();

        for pos in [0, 1, 10, 31, 63] {
            let initial_q: Vec<f32> = (0..(n_heads * head_dim))
                .map(|i| i as f32 * 0.1 - 0.5 * n_heads as f32 * head_dim as f32 * 0.1)
                .collect();
            let initial_k: Vec<f32> = (0..(n_kv * head_dim))
                .map(|i| i as f32 * 0.2 - 0.5 * n_kv as f32 * head_dim as f32 * 0.2)
                .collect();

            // Uncached version.
            let mut q_unc = initial_q.clone();
            let mut k_unc = initial_k.clone();
            apply_rope(&mut q_unc, &mut k_unc, pos, head_dim, n_heads, n_kv, theta).unwrap();

            // Cached version.
            let mut q_cac = initial_q.clone();
            let mut k_cac = initial_k.clone();
            apply_rope_cached(&mut q_cac, &mut k_cac, pos, n_heads, n_kv, &cache).unwrap();

            for (i, (&unc, &cac)) in q_unc.iter().zip(q_cac.iter()).enumerate() {
                assert!(
                    (unc - cac).abs() < 1e-5,
                    "pos={pos}, q[{i}]: uncached={unc}, cached={cac}"
                );
            }
            for (i, (&unc, &cac)) in k_unc.iter().zip(k_cac.iter()).enumerate() {
                assert!(
                    (unc - cac).abs() < 1e-5,
                    "pos={pos}, k[{i}]: uncached={unc}, cached={cac}"
                );
            }
        }
    }

    #[test]
    fn cached_out_of_bounds_returns_error() {
        let cache = RopeCache::new(4, 8, 10000.0).unwrap();
        let mut q = vec![0.0_f32; 8];
        let mut k = vec![0.0_f32; 8];
        let err = apply_rope_cached(&mut q, &mut k, 4, 1, 1, &cache).unwrap_err();
        assert!(matches!(
            err,
            bitnet_core::error::BitNetError::InvalidShape { .. }
        ));
    }

    // -----------------------------------------------------------------------
    // Mathematical properties
    // -----------------------------------------------------------------------

    /// RoPE relative-position theorem:
    /// The inner product ⟨RoPE(q, m), RoPE(k, n)⟩ depends only on (m - n),
    /// not on the absolute positions m and n individually.
    ///
    /// Formally:
    /// ⟨R_m · q, R_n · k⟩ = Re[ Σ_i  q̃_i * k̃_i^* * e^{i(m-n)θ_i} ]
    ///
    /// We verify this numerically: rotating both vectors by the same extra
    /// offset Δ should not change their dot product.
    #[test]
    fn rope_relative_position_invariance() {
        let head_dim = 8;
        let theta = 10000.0_f32;
        // Simple head vectors.
        let q_init = vec![0.5_f32, -0.3, 0.7, 0.1, -0.2, 0.4, 0.6, -0.5];
        let k_init = vec![0.2_f32, 0.8, -0.4, 0.3, 0.5, -0.1, 0.7, -0.6];

        // Compute dot product with pos_q=3, pos_k=1 (relative pos = 2).
        let mut q_a = q_init.clone();
        let mut k_a = k_init.clone();
        let mut dummy_k_a = k_init.clone(); // placeholder for k in apply_rope call for q only
                                            // Apply RoPE to q at pos=3.
        apply_rope(&mut q_a, &mut dummy_k_a, 3, head_dim, 1, 1, theta).unwrap();
        // Apply RoPE to k at pos=1.
        let mut dummy_q_a = q_init.clone();
        apply_rope(&mut dummy_q_a, &mut k_a, 1, head_dim, 1, 1, theta).unwrap();
        let dot_a: f32 = q_a.iter().zip(k_a.iter()).map(|(a, b)| a * b).sum();

        // Compute dot product with pos_q=13, pos_k=11 (relative pos = 2, shifted by 10).
        let mut q_b = q_init.clone();
        let mut k_b = k_init.clone();
        let mut dummy_k_b = k_init.clone();
        apply_rope(&mut q_b, &mut dummy_k_b, 13, head_dim, 1, 1, theta).unwrap();
        let mut dummy_q_b = q_init.clone();
        apply_rope(&mut dummy_q_b, &mut k_b, 11, head_dim, 1, 1, theta).unwrap();
        let dot_b: f32 = q_b.iter().zip(k_b.iter()).map(|(a, b)| a * b).sum();

        // Both should have approximately the same dot product (relative pos = 2).
        assert!(
            (dot_a - dot_b).abs() < 1e-3,
            "RoPE relative-pos invariance violated: dot(pos3,pos1)={dot_a}, dot(pos13,pos11)={dot_b}"
        );
    }
}
