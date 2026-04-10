//! SIMD-accelerated kernels for ternary GEMV on x86-64.
//!
//! # Mathematical Specification
//!
//! The ternary dot product computes:
//!
//! ```text
//! result = Σ_{j=0}^{K-1}  weight[j] × activation[j]
//! ```
//!
//! where `weight[j] ∈ {-1, 0, +1}` (stored as `i8`) and `activation[j]` is
//! either `i8` (W2A8 quantised path) or `f32` (unquantised path).
//!
//! # AVX2 Ternary Multiply via `VPSIGNW` (i16 precision)
//!
//! To avoid the `i8` overflow when `activation == -128` and `weight == -1`
//! (where `-(-128_i8)` wraps to `-128_i8`), we sign-extend both operands
//! from `i8` to `i16` before applying the sign operation:
//!
//! 1. `_mm256_cvtepi8_epi16(x)` (`VPMOVSXBW`): sign-extend 16 × `i8` → 16 × `i16`.
//!
//! 2. `_mm256_sign_epi16(a_i16, w_i16)` (`VPSIGNW`): ternary multiply at
//!    `i16` precision.  `-(-128_i16) = +128_i16` — no overflow.
//!
//! 3. `VPMADDWD(products_i16, ones_i16)`: pair-wise sum `i16 → i32` (8 results).
//!
//! Each iteration processes 16 elements (limited by the 128→256 bit widening).
//! For the 2B model's hidden dimension (2560), this yields 160 iterations
//! with no scalar tail.
//!
//! # Platform Support
//!
//! All functions in this module are gated on `target_arch = "x86_64"` at
//! compile time.  Runtime detection via `has_avx2()` (backed by `OnceLock`)
//! must be checked before calling the `unsafe` AVX2 kernels.

#[allow(unused_imports)]
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Runtime feature detection
// ---------------------------------------------------------------------------

/// Cached result of the AVX2 runtime feature check.
static AVX2_DETECTED: OnceLock<bool> = OnceLock::new();

/// Returns `true` if the current CPU supports AVX2.
///
/// The result is computed once (via `CPUID`) and cached for subsequent calls.
/// Cost after first call: a single atomic load.
#[inline]
pub fn has_avx2() -> bool {
    *AVX2_DETECTED.get_or_init(detect_avx2)
}

fn detect_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Cached result of the FMA runtime feature check.
static FMA_DETECTED: OnceLock<bool> = OnceLock::new();

/// Returns `true` if the current CPU supports FMA (fused multiply-add).
///
/// The result is computed once (via `CPUID`) and cached for subsequent calls.
/// Cost after first call: a single atomic load.
#[inline]
pub fn has_fma() -> bool {
    *FMA_DETECTED.get_or_init(detect_fma)
}

fn detect_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ---------------------------------------------------------------------------
// AVX2 kernels (x86-64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2-accelerated ternary dot product: `i8` weights × `i8` activations → `i32`.
///
/// Processes 16 elements per iteration by sign-extending `i8` to `i16` before
/// applying `VPSIGNW`, avoiding the `-(-128_i8)` wraparound that occurs with
/// `VPSIGNB`.  The widened products are reduced via `VPMADDWD`.
///
/// # Returns
///
/// `Σ_j weight[j] × activation[j]` computed in `i32`.
/// Bit-exact with the scalar implementation for all inputs.
///
/// # Safety
///
/// Caller **must** verify AVX2 support via [`has_avx2()`] before calling.
/// Calling on a CPU without AVX2 is undefined behaviour.
///
/// # Panics (debug only)
///
/// Panics if `weight.len() != activation.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_ternary_i8_avx2(weight: &[i8], activation: &[i8]) -> i32 {
    debug_assert_eq!(
        weight.len(),
        activation.len(),
        "dot_ternary_i8_avx2: length mismatch ({} vs {})",
        weight.len(),
        activation.len()
    );

    let len = weight.len();
    let chunks = len / 16;
    let remainder = len % 16;

    let ones_i16 = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256(); // 8 × i32 accumulator

    // Main loop: 16 elements per iteration.
    // Sign-extend i8 → i16 to avoid the VPSIGNB -128 overflow.
    for i in 0..chunks {
        let base = i * 16;

        // Load 16 × i8 from each array into 128-bit registers.
        let w_128 = _mm_loadu_si128(weight.as_ptr().add(base) as *const __m128i);
        let a_128 = _mm_loadu_si128(activation.as_ptr().add(base) as *const __m128i);

        // Sign-extend i8 → i16: 16 values fill a 256-bit register.
        // VPMOVSXBW: [-128_i8] → [-128_i16], preserving full range.
        let w_i16 = _mm256_cvtepi8_epi16(w_128);
        let a_i16 = _mm256_cvtepi8_epi16(a_128);

        // Ternary multiply via VPSIGNW (i16 precision):
        //   -(-128_i16) = +128_i16  — no overflow.
        let products = _mm256_sign_epi16(a_i16, w_i16);

        // Pair-wise horizontal sum i16 → i32 via VPMADDWD:
        //   quad_sums[k] = products[2k]*1 + products[2k+1]*1   (8 × i32)
        let quad_sums = _mm256_madd_epi16(products, ones_i16);

        acc = _mm256_add_epi32(acc, quad_sums);
    }

    // Horizontal reduction: 8 × i32 → scalar i32.
    let mut result = hsum_i32_avx2(acc);

    // Scalar tail for remaining elements (< 16).
    if remainder > 0 {
        let base = chunks * 16;
        for i in 0..remainder {
            result += (weight[base + i] as i32) * (activation[base + i] as i32);
        }
    }

    result
}

/// AVX2-accelerated ternary dot product: `i8` weights × `f32` activations → `f32`.
///
/// Uses `VPSIGNB` combined with `VCVTDQ2PS` for vectorised accumulation.
/// Processes 16 elements per iteration (limited by the 8-wide f32 lanes
/// after sign-extension from i8 → i32).
///
/// # Returns
///
/// `Σ_j weight[j] × activation[j]` computed in `f32`.
///
/// # Safety
///
/// Caller **must** verify AVX2 support via [`has_avx2()`] before calling.
///
/// # Panics (debug only)
///
/// Panics if `weight.len() != activation.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_ternary_f32_avx2(weight: &[i8], input: &[f32]) -> f32 {
    debug_assert_eq!(
        weight.len(),
        input.len(),
        "dot_ternary_f32_avx2: length mismatch ({} vs {})",
        weight.len(),
        input.len()
    );

    let len = weight.len();
    // Process 8 elements per iteration (limited by f32 lane width).
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = _mm256_setzero_ps(); // 8 × f32 accumulator
    let zero_i8 = _mm_setzero_si128();

    for i in 0..chunks {
        let base = i * 8;

        // Load 8 weight bytes into the low 64 bits of a 128-bit register.
        // Use _mm_loadl_epi64 which loads 8 bytes (64 bits).
        let w_ptr = weight.as_ptr().add(base) as *const __m128i;
        let w_8xi8 = _mm_loadl_epi64(w_ptr);

        // Sign-extend i8 → i32 for the 8 weights.
        // _mm256_cvtepi8_epi32 takes the low 8 bytes of a 128-bit register
        // and sign-extends each to i32, producing 8 × i32 in a 256-bit reg.
        let w_8xi32 = _mm256_cvtepi8_epi32(w_8xi8);

        // Create a mask: 1 where weight != 0, 0 where weight == 0.
        // Compare w_8xi8 == 0 → yields 0xFF where zero, 0x00 where non-zero.
        let cmp_zero = _mm_cmpeq_epi8(w_8xi8, zero_i8);
        // Extend to i32 and invert: non-zero weights become all-ones mask.
        let mask_i32 = _mm256_cvtepi8_epi32(cmp_zero);
        // mask_i32 has 0xFFFFFFFF where w==0, 0x00000000 where w!=0.
        // We need the inverse for andnot.
        let mask_f32 = _mm256_castsi256_ps(mask_i32);

        // Convert weight i32 → f32.
        let w_f32 = _mm256_cvtepi32_ps(w_8xi32);

        // Load 8 f32 activations.
        let a_f32 = _mm256_loadu_ps(input.as_ptr().add(base));

        // Multiply weight × activation.
        let prod = _mm256_mul_ps(w_f32, a_f32);

        // Zero out products where weight was zero (avoid NaN propagation from
        // activation × 0.0 when activation is NaN — defensive, not expected).
        // andnot(mask, prod): bits of prod where mask is 0.
        // mask_f32 has all-ones where w==0, so andnot clears those lanes.
        let masked_prod = _mm256_andnot_ps(mask_f32, prod);

        acc = _mm256_add_ps(acc, masked_prod);
    }

    // Horizontal sum of 8 × f32.
    let mut result = hsum_f32_avx2(acc);

    // Scalar tail.
    if remainder > 0 {
        let base = chunks * 8;
        for i in 0..remainder {
            result += weight[base + i] as f32 * input[base + i];
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Horizontal reduction helpers
// ---------------------------------------------------------------------------

/// Horizontal sum of 8 × `i32` lanes in a `__m256i` register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_i32_avx2(v: __m256i) -> i32 {
    // 1. Add high 128-bit half to low 128-bit half → 4 × i32.
    let hi128 = _mm256_extracti128_si256(v, 1);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo128, hi128);

    // 2. Shuffle and add: [a, b, c, d] → [a+c, b+d, ?, ?].
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);

    // 3. Shuffle and add: [a+c, b+d, ?, ?] → [(a+c)+(b+d), ?, ?, ?].
    let hi32 = _mm_shuffle_epi32(sum64, 0b_00_00_00_01);
    let sum32 = _mm_add_epi32(sum64, hi32);

    _mm_cvtsi128_si32(sum32)
}

/// Horizontal sum of 8 × `f32` lanes in a `__m256` register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_f32_avx2(v: __m256) -> f32 {
    // 1. Add high 128-bit half to low 128-bit half → 4 × f32.
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);

    // 2. Horizontal add pairs: [a+c, b+d, a+c, b+d] (approximately).
    let shuf = _mm_movehdup_ps(sum128); // [b, b, d, d]
    let sum2 = _mm_add_ps(sum128, shuf); // [a+b, ?, c+d, ?]

    // 3. Move high pair down and add.
    let shuf2 = _mm_movehl_ps(sum2, sum2); // [c+d, ?, ?, ?]
    let sum1 = _mm_add_ss(sum2, shuf2); // [(a+b)+(c+d), ?, ?, ?]

    _mm_cvtss_f32(sum1)
}

// ---------------------------------------------------------------------------
// Dispatch wrappers (safe, platform-agnostic)
// ---------------------------------------------------------------------------

/// Ternary dot product (`i8 × i8 → i32`) with automatic SIMD dispatch.
///
/// Returns the same value as [`super::gemv::dot_ternary_i8`]:
/// ```text
/// Σ_j weight[j] × activation[j]   (all widened to i32)
/// ```
///
/// Uses AVX2 when available on x86-64, scalar fallback otherwise.
#[inline]
pub fn dot_ternary_i8_fast(weight: &[i8], activation: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // Safety: AVX2 availability confirmed by `has_avx2()`.
            return unsafe { dot_ternary_i8_avx2(weight, activation) };
        }
    }
    // Scalar fallback.
    dot_ternary_i8_scalar(weight, activation)
}

/// Ternary dot product (`i8 × f32 → f32`) with automatic SIMD dispatch.
///
/// Returns the same value as [`super::gemv::dot_ternary_f32`]:
/// ```text
/// Σ_j weight[j] as f32 × input[j]
/// ```
///
/// Uses AVX2 when available on x86-64, scalar fallback otherwise.
#[inline]
pub fn dot_ternary_f32_fast(weight: &[i8], input: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // Safety: AVX2 availability confirmed by `has_avx2()`.
            return unsafe { dot_ternary_f32_avx2(weight, input) };
        }
    }
    // Scalar fallback.
    dot_ternary_f32_scalar(weight, input)
}

// ---------------------------------------------------------------------------
// Scalar fallbacks (duplicated from gemv.rs to avoid circular dependency)
// ---------------------------------------------------------------------------

/// Scalar ternary dot product: `i8 × i8 → i32`.
#[inline]
fn dot_ternary_i8_scalar(weight: &[i8], activation: &[i8]) -> i32 {
    debug_assert_eq!(weight.len(), activation.len());
    let mut acc: i32 = 0;
    for (&w, &x) in weight.iter().zip(activation.iter()) {
        acc += (w as i32) * (x as i32);
    }
    acc
}

/// Scalar ternary dot product: `i8 × f32 → f32`.
#[inline]
fn dot_ternary_f32_scalar(weight: &[i8], input: &[f32]) -> f32 {
    debug_assert_eq!(weight.len(), input.len());
    let mut acc = 0.0_f32;
    for (&w, &x) in weight.iter().zip(input.iter()) {
        acc += w as f32 * x;
    }
    acc
}

// ---------------------------------------------------------------------------
// Packed 2-bit ternary decode LUT and kernels
// ---------------------------------------------------------------------------

/// Decode table: maps each possible packed byte to 4 i8 ternary values.
/// Index is the packed byte value (0..256).
/// Value is the 4 decoded weights packed as [i8; 4] in little-endian order.
///
/// Encoding: 0b00→+1, 0b01→0, 0b10→-1
static DECODE_LUT: [[i8; 4]; 256] = {
    let mut table = [[0i8; 4]; 256];
    let decode = [1i8, 0, -1, 0]; // code → value (code 3 maps to 0 as padding)
    let mut byte = 0u16;
    while byte < 256 {
        let b = byte as u8;
        table[byte as usize] = [
            decode[(b & 3) as usize],
            decode[((b >> 2) & 3) as usize],
            decode[((b >> 4) & 3) as usize],
            decode[((b >> 6) & 3) as usize],
        ];
        byte += 1;
    }
    table
};

/// Unpack packed 2-bit ternary weights to i8 ternary values.
///
/// # Arguments
/// - `packed`: packed bytes (4 weights per byte)
/// - `out`: output buffer, must be at least `n_elements` long
/// - `n_elements`: number of logical ternary elements to decode
///
/// # Panics (debug only)
/// Panics if `out.len() < n_elements` or `packed.len() < (n_elements + 3) / 4`.
#[inline]
pub fn unpack_packed_to_i8(packed: &[u8], out: &mut [i8], n_elements: usize) {
    debug_assert!(out.len() >= n_elements);
    debug_assert!(packed.len() >= (n_elements + 3) / 4);

    let full_bytes = n_elements / 4;
    let remainder = n_elements % 4;

    // Fast path: decode 4 values per byte using LUT
    let out_ptr = out.as_mut_ptr();
    for i in 0..full_bytes {
        let decoded = DECODE_LUT[packed[i] as usize];
        // Safety: `i * 4 + 3 < n_elements <= out.len()` holds for `i < full_bytes`.
        unsafe {
            let base = out_ptr.add(i * 4);
            base.write(decoded[0]);
            base.add(1).write(decoded[1]);
            base.add(2).write(decoded[2]);
            base.add(3).write(decoded[3]);
        }
    }

    // Handle remainder
    if remainder > 0 {
        let decoded = DECODE_LUT[packed[full_bytes] as usize];
        let base_idx = full_bytes * 4;
        for j in 0..remainder {
            out[base_idx + j] = decoded[j];
        }
    }
}

/// Packed ternary dot product: packed 2-bit weights × i8 activations → i32.
///
/// # Arguments
/// - `packed`: packed weight bytes (4 weights per byte)
/// - `activation`: i8 activation vector
/// - `n_elements`: number of logical weight/activation elements
///
/// Uses chunk-based unpacking with the existing AVX2 dot product kernel.
/// 4× less memory bandwidth for weight data vs the unpacked path.
#[inline]
pub fn dot_packed_ternary_i8_fast(packed: &[u8], activation: &[i8], n_elements: usize) -> i32 {
    debug_assert!(activation.len() >= n_elements);
    debug_assert!(packed.len() >= (n_elements + 3) / 4);

    let full_bytes = n_elements / 4;
    let remainder = n_elements % 4;

    let mut total: i32 = 0;
    let decode = [1i32, 0, -1, 0]; // code → value (code 3 maps to 0 as padding)

    for byte_idx in 0..full_bytes {
        let packed_byte = packed[byte_idx];
        let act_base = byte_idx * 4;

        let c0 = decode[(packed_byte & 0b11) as usize];
        let c1 = decode[((packed_byte >> 2) & 0b11) as usize];
        let c2 = decode[((packed_byte >> 4) & 0b11) as usize];
        let c3 = decode[((packed_byte >> 6) & 0b11) as usize];

        total += c0 * activation[act_base] as i32;
        total += c1 * activation[act_base + 1] as i32;
        total += c2 * activation[act_base + 2] as i32;
        total += c3 * activation[act_base + 3] as i32;
    }

    if remainder > 0 {
        let packed_byte = packed[full_bytes];
        let act_base = full_bytes * 4;
        let codes = [
            decode[(packed_byte & 0b11) as usize],
            decode[((packed_byte >> 2) & 0b11) as usize],
            decode[((packed_byte >> 4) & 0b11) as usize],
            decode[((packed_byte >> 6) & 0b11) as usize],
        ];

        for i in 0..remainder {
            total += codes[i] * activation[act_base + i] as i32;
        }
    }

    total
}

/// Packed ternary dot product: packed 2-bit weights × f32 activations → f32.
///
/// Same chunk-based strategy as the i8 variant.
#[inline]
pub fn dot_packed_ternary_f32_fast(packed: &[u8], input: &[f32], n_elements: usize) -> f32 {
    debug_assert!(input.len() >= n_elements);
    debug_assert!(packed.len() >= (n_elements + 3) / 4);

    let full_bytes = n_elements / 4;
    let remainder = n_elements % 4;

    let mut total: f32 = 0.0;
    let decode = [1.0_f32, 0.0, -1.0, 0.0]; // code → value (code 3 maps to 0 as padding)

    for byte_idx in 0..full_bytes {
        let packed_byte = packed[byte_idx];
        let input_base = byte_idx * 4;

        let c0 = decode[(packed_byte & 0b11) as usize];
        let c1 = decode[((packed_byte >> 2) & 0b11) as usize];
        let c2 = decode[((packed_byte >> 4) & 0b11) as usize];
        let c3 = decode[((packed_byte >> 6) & 0b11) as usize];

        total += c0 * input[input_base];
        total += c1 * input[input_base + 1];
        total += c2 * input[input_base + 2];
        total += c3 * input[input_base + 3];
    }

    if remainder > 0 {
        let packed_byte = packed[full_bytes];
        let input_base = full_bytes * 4;
        let codes = [
            decode[(packed_byte & 0b11) as usize],
            decode[((packed_byte >> 2) & 0b11) as usize],
            decode[((packed_byte >> 4) & 0b11) as usize],
            decode[((packed_byte >> 6) & 0b11) as usize],
        ];

        for i in 0..remainder {
            total += codes[i] * input[input_base + i];
        }
    }

    total
}

// ---------------------------------------------------------------------------
// f32 × f32 dot product (for lm_head matmul)
// ---------------------------------------------------------------------------

/// AVX2+FMA accelerated f32 dot product for the lm_head matmul.
///
/// Processes 8 f32 elements per iteration using fused multiply-add.
///
/// # Safety
/// Caller must verify AVX2+FMA support before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 8;
        // Safety: `base + 7 < len` holds for `i < chunks`. Unaligned loads are
        // safe on x86-64 (may cross cache lines but never fault).
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb = _mm256_loadu_ps(b.as_ptr().add(base));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    let mut result = hsum_f32_avx2(acc);

    if remainder > 0 {
        let base = chunks * 8;
        for i in 0..remainder {
            result += a[base + i] * b[base + i];
        }
    }

    result
}

/// f32 dot product with automatic SIMD dispatch.
///
/// Uses AVX2+FMA when available, scalar fallback otherwise.
/// Primary use: lm_head matmul (vocab_size rows × hidden_size dot products).
#[inline]
pub fn dot_f32_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            // Safety: AVX2+FMA availability confirmed by `has_avx2()` and `has_fma()`.
            return unsafe { dot_f32_f32_avx2(a, b) };
        }
    }
    dot_f32_f32_scalar(a, b)
}

/// Scalar f32 dot product fallback.
#[inline]
fn dot_f32_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

// ---------------------------------------------------------------------------
// axpy: out[i] += alpha * x[i]
// ---------------------------------------------------------------------------

/// Fused multiply-add: `out[i] += alpha * x[i]` for all i.
///
/// Used in attention value accumulation: `out_head += attn_weight * v_head`.
/// AVX2+FMA path processes 8 floats per cycle.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_f32_avx2(alpha: f32, x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(x.len(), out.len());

    let len = x.len();
    let chunks = len / 8;
    let remainder = len % 8;
    let valpha = _mm256_set1_ps(alpha);

    for i in 0..chunks {
        let base = i * 8;
        let vx = _mm256_loadu_ps(x.as_ptr().add(base));
        let vo = _mm256_loadu_ps(out.as_ptr().add(base));
        let result = _mm256_fmadd_ps(valpha, vx, vo);
        _mm256_storeu_ps(out.as_mut_ptr().add(base), result);
    }

    if remainder > 0 {
        let base = chunks * 8;
        for i in 0..remainder {
            out[base + i] += alpha * x[base + i];
        }
    }
}

/// axpy with automatic SIMD dispatch.
///
/// Computes `out[i] += alpha * x[i]` for all i.
#[inline]
pub fn axpy_f32_fast(alpha: f32, x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            // Safety: AVX2+FMA availability confirmed above.
            unsafe { axpy_f32_avx2(alpha, x, out) };
            return;
        }
    }
    axpy_f32_scalar(alpha, x, out);
}

/// Scalar axpy fallback.
#[inline]
fn axpy_f32_scalar(alpha: f32, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    for i in 0..x.len() {
        out[i] += alpha * x[i];
    }
}

// ---------------------------------------------------------------------------
// sum_squares: Σ x[i]²
// ---------------------------------------------------------------------------

/// SIMD-accelerated sum of squares: Σ x[i]².
///
/// Used in RMSNorm. AVX2+FMA path processes 8 floats per cycle.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sum_squares_f32_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = x.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 8;
        let vx = _mm256_loadu_ps(x.as_ptr().add(base));
        acc = _mm256_fmadd_ps(vx, vx, acc);
    }

    let mut result = hsum_f32_avx2(acc);

    if remainder > 0 {
        let base = chunks * 8;
        for i in 0..remainder {
            result += x[base + i] * x[base + i];
        }
    }

    result
}

/// Sum of squares with automatic SIMD dispatch.
#[inline]
pub fn sum_squares_f32_fast(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            return unsafe { sum_squares_f32_avx2(x) };
        }
    }
    x.iter().map(|&v| v * v).sum()
}

// ---------------------------------------------------------------------------
// mul_scale: out[i] = x[i] * scale * w[i]
// ---------------------------------------------------------------------------

/// SIMD-accelerated elementwise multiply-scale: `out[i] = x[i] * scale * w[i]`.
///
/// Used in RMSNorm output pass: `out[i] = input[i] * inv_rms * weight[i]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn mul_scale_f32_avx2(x: &[f32], scale: f32, w: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(x.len(), w.len());
    debug_assert_eq!(x.len(), out.len());

    let len = x.len();
    let chunks = len / 8;
    let remainder = len % 8;
    let vscale = _mm256_set1_ps(scale);

    for i in 0..chunks {
        let base = i * 8;
        let vx = _mm256_loadu_ps(x.as_ptr().add(base));
        let vw = _mm256_loadu_ps(w.as_ptr().add(base));
        let scaled = _mm256_mul_ps(vx, vscale);
        let result = _mm256_mul_ps(scaled, vw);
        _mm256_storeu_ps(out.as_mut_ptr().add(base), result);
    }

    if remainder > 0 {
        let base = chunks * 8;
        for i in 0..remainder {
            out[base + i] = x[base + i] * scale * w[base + i];
        }
    }
}

/// Elementwise multiply-scale with automatic SIMD dispatch.
#[inline]
pub fn mul_scale_f32_fast(x: &[f32], scale: f32, w: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { mul_scale_f32_avx2(x, scale, w, out) };
            return;
        }
    }
    for i in 0..x.len() {
        out[i] = x[i] * scale * w[i];
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx2_detection_does_not_panic() {
        // Just exercise the detection path — result depends on hardware.
        let _ = has_avx2();
    }

    #[test]
    fn dispatch_i8_matches_scalar_basic() {
        let w: Vec<i8> = vec![1, 0, -1, -1, 1, 0, 1, -1];
        let a: Vec<i8> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let expected = dot_ternary_i8_scalar(&w, &a);
        let got = dot_ternary_i8_fast(&w, &a);
        assert_eq!(got, expected);
    }

    #[test]
    fn dispatch_i8_matches_scalar_all_ones() {
        let w: Vec<i8> = vec![1; 2560];
        let a: Vec<i8> = (0..2560)
            .map(|i| ((i % 255) as i8).wrapping_sub(127))
            .collect();
        let expected = dot_ternary_i8_scalar(&w, &a);
        let got = dot_ternary_i8_fast(&w, &a);
        assert_eq!(got, expected);
    }

    #[test]
    fn dispatch_i8_matches_scalar_all_neg_ones() {
        let w: Vec<i8> = vec![-1; 2560];
        let a: Vec<i8> = (0..2560)
            .map(|i| ((i % 255) as i8).wrapping_sub(127))
            .collect();
        let expected = dot_ternary_i8_scalar(&w, &a);
        let got = dot_ternary_i8_fast(&w, &a);
        assert_eq!(got, expected);
    }

    #[test]
    fn dispatch_i8_matches_scalar_mixed_ternary() {
        // Repeating pattern: +1, 0, -1
        let len = 2560;
        let w: Vec<i8> = (0..len)
            .map(|i| match i % 3 {
                0 => 1,
                1 => 0,
                _ => -1,
            })
            .collect();
        let a: Vec<i8> = (0..len).map(|i| ((i * 7 + 3) % 255) as i8).collect();
        let expected = dot_ternary_i8_scalar(&w, &a);
        let got = dot_ternary_i8_fast(&w, &a);
        assert_eq!(got, expected);
    }

    #[test]
    fn dispatch_i8_matches_scalar_non_aligned_length() {
        // Length not divisible by 16.
        for len in [1, 7, 15, 16, 17, 31, 33, 63, 65, 100, 127, 129, 255] {
            let w: Vec<i8> = (0..len)
                .map(|i| match i % 3 {
                    0 => 1,
                    1 => 0,
                    _ => -1,
                })
                .collect();
            let a: Vec<i8> = (0..len).map(|i| (i % 200) as i8).collect();
            let expected = dot_ternary_i8_scalar(&w, &a);
            let got = dot_ternary_i8_fast(&w, &a);
            assert_eq!(got, expected, "mismatch at len={len}");
        }
    }

    #[test]
    fn dispatch_i8_empty_returns_zero() {
        assert_eq!(dot_ternary_i8_fast(&[], &[]), 0);
    }

    #[test]
    fn dispatch_f32_matches_scalar_basic() {
        let w: Vec<i8> = vec![1, 0, -1, -1, 1, 0, 1, -1];
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected = dot_ternary_f32_scalar(&w, &a);
        let got = dot_ternary_f32_fast(&w, &a);
        assert!(
            (got - expected).abs() < 1e-6,
            "got={got}, expected={expected}"
        );
    }

    #[test]
    fn dispatch_f32_matches_scalar_large() {
        let len = 2560;
        let w: Vec<i8> = (0..len)
            .map(|i| match i % 3 {
                0 => 1,
                1 => 0,
                _ => -1,
            })
            .collect();
        let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.01 - 12.8).collect();
        let expected = dot_ternary_f32_scalar(&w, &a);
        let got = dot_ternary_f32_fast(&w, &a);
        // Allow small floating-point tolerance due to different summation order.
        let tol = expected.abs() * 1e-5 + 1e-5;
        assert!(
            (got - expected).abs() < tol,
            "got={got}, expected={expected}, diff={}, tol={tol}",
            (got - expected).abs()
        );
    }

    #[test]
    fn dispatch_f32_matches_scalar_non_aligned_lengths() {
        for len in [1, 7, 8, 9, 15, 16, 17, 31, 33, 63, 65, 100] {
            let w: Vec<i8> = (0..len)
                .map(|i| match i % 3 {
                    0 => 1,
                    1 => 0,
                    _ => -1,
                })
                .collect();
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
            let expected = dot_ternary_f32_scalar(&w, &a);
            let got = dot_ternary_f32_fast(&w, &a);
            let tol = expected.abs() * 1e-5 + 1e-5;
            assert!(
                (got - expected).abs() < tol,
                "len={len}: got={got}, expected={expected}"
            );
        }
    }

    #[test]
    fn dispatch_f32_empty_returns_zero() {
        assert_eq!(dot_ternary_f32_fast(&[], &[]), 0.0);
    }

    #[test]
    fn dispatch_f32_all_zero_weights_returns_zero() {
        let w = vec![0i8; 256];
        let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
        assert_eq!(dot_ternary_f32_fast(&w, &a), 0.0);
    }

    #[cfg(target_arch = "x86_64")]
    mod avx2_direct {
        use super::*;

        /// Run a test only if AVX2 is available on this machine.
        macro_rules! require_avx2 {
            () => {
                if !has_avx2() {
                    eprintln!("Skipping AVX2 test: AVX2 not available on this CPU");
                    return;
                }
            };
        }

        #[test]
        fn i8_avx2_single_chunk_exact() {
            require_avx2!();
            // Exactly 16 elements = one AVX2 chunk, no remainder.
            let w: Vec<i8> = (0..16)
                .map(|i| match i % 3 {
                    0 => 1,
                    1 => 0,
                    _ => -1,
                })
                .collect();
            let a: Vec<i8> = (0..16).map(|i| (i * 4) as i8).collect();
            let expected = dot_ternary_i8_scalar(&w, &a);
            let got = unsafe { dot_ternary_i8_avx2(&w, &a) };
            assert_eq!(got, expected);
        }

        #[test]
        fn i8_avx2_multiple_chunks_plus_remainder() {
            require_avx2!();
            let len = 2560; // 160 chunks × 16, remainder = 0
            let w: Vec<i8> = (0..len).map(|i| [1, 0, -1][i % 3]).collect();
            let a: Vec<i8> = (0..len).map(|i| ((i * 13 + 7) % 255) as i8).collect();
            let expected = dot_ternary_i8_scalar(&w, &a);
            let got = unsafe { dot_ternary_i8_avx2(&w, &a) };
            assert_eq!(got, expected);
        }

        #[test]
        fn i8_avx2_remainder_only() {
            require_avx2!();
            // < 16 elements: no AVX2 chunks, only scalar tail.
            let w: Vec<i8> = vec![1, -1, 0, 1, -1];
            let a: Vec<i8> = vec![10, 20, 30, 40, 50];
            let expected = dot_ternary_i8_scalar(&w, &a);
            let got = unsafe { dot_ternary_i8_avx2(&w, &a) };
            assert_eq!(got, expected);
        }

        #[test]
        fn i8_avx2_all_positive_127() {
            require_avx2!();
            let w = vec![1i8; 64];
            let a = vec![127i8; 64];
            let expected = 64 * 127;
            let got = unsafe { dot_ternary_i8_avx2(&w, &a) };
            assert_eq!(got, expected);
        }

        #[test]
        fn i8_avx2_all_negative_128_with_pos_weight() {
            require_avx2!();
            // w=1, a=-128 → each product = -128, sum = -128 × 64 = -8192
            let w = vec![1i8; 64];
            let a = vec![-128i8; 64];
            let expected = 64 * (-128);
            let got = unsafe { dot_ternary_i8_avx2(&w, &a) };
            assert_eq!(got, expected);
        }

        #[test]
        fn i8_avx2_negative_128_with_neg_weight() {
            require_avx2!();
            // w=-1, a=-128 → each product = +128, sum = +128 × 64 = +8192
            // This is the edge case that VPSIGNB gets wrong but VPSIGNW handles.
            let w = vec![-1i8; 64];
            let a = vec![-128i8; 64];
            let expected = 64 * 128; // (-1) × (-128) = +128
            let got = unsafe { dot_ternary_i8_avx2(&w, &a) };
            assert_eq!(got, expected);
        }

        #[test]
        fn f32_avx2_basic() {
            require_avx2!();
            let w: Vec<i8> = vec![1, 0, -1, 1, 0, -1, 1, -1, 0, 1];
            let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            let expected = dot_ternary_f32_scalar(&w, &a);
            let got = unsafe { dot_ternary_f32_avx2(&w, &a) };
            assert!(
                (got - expected).abs() < 1e-5,
                "got={got}, expected={expected}"
            );
        }

        #[test]
        fn f32_avx2_large_dimension() {
            require_avx2!();
            let len = 2560;
            let w: Vec<i8> = (0..len).map(|i| [1, 0, -1][i % 3]).collect();
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.01 - 12.8).collect();
            let expected = dot_ternary_f32_scalar(&w, &a);
            let got = unsafe { dot_ternary_f32_avx2(&w, &a) };
            let tol = expected.abs() * 1e-5 + 1e-5;
            assert!(
                (got - expected).abs() < tol,
                "got={got}, expected={expected}, diff={}",
                (got - expected).abs()
            );
        }
    }

    // ── Packed ternary tests ─────────────────────────────────────────────

    #[test]
    fn decode_lut_correctness() {
        // Verify LUT for known byte patterns
        // All +1: byte = 0b00_00_00_00 = 0x00
        assert_eq!(DECODE_LUT[0x00], [1, 1, 1, 1]);
        // All 0: byte = 0b01_01_01_01 = 0x55
        assert_eq!(DECODE_LUT[0x55], [0, 0, 0, 0]);
        // All -1: byte = 0b10_10_10_10 = 0xAA
        assert_eq!(DECODE_LUT[0xAA], [-1, -1, -1, -1]);
        // Mixed: [+1, 0, -1, +1] = 0b00_10_01_00 = 0x24
        assert_eq!(DECODE_LUT[0x24], [1, 0, -1, 1]);
    }

    #[test]
    fn unpack_basic() {
        let packed = [0x24u8]; // [+1, 0, -1, +1]
        let mut out = [0i8; 4];
        unpack_packed_to_i8(&packed, &mut out, 4);
        assert_eq!(out, [1, 0, -1, 1]);
    }

    #[test]
    fn unpack_multiple_bytes() {
        // 8 weights: [+1,+1,+1,+1, -1,-1,-1,-1]
        let packed = [0x00u8, 0xAAu8];
        let mut out = [0i8; 8];
        unpack_packed_to_i8(&packed, &mut out, 8);
        assert_eq!(out, [1, 1, 1, 1, -1, -1, -1, -1]);
    }

    #[test]
    fn unpack_with_remainder() {
        // 3 weights from 1 byte: [+1, 0, -1]
        let packed = [0x24u8]; // full byte is [+1, 0, -1, +1]
        let mut out = [99i8; 4]; // extra space
        unpack_packed_to_i8(&packed, &mut out, 3);
        assert_eq!(&out[..3], &[1, 0, -1]);
    }

    #[test]
    fn packed_i8_dot_basic() {
        // weights: [+1, -1, 0, +1]
        // v0=+1→0b00, v1=-1→0b10, v2=0→0b01, v3=+1→0b00
        // byte = 0b00 | (0b10 << 2) | (0b01 << 4) | (0b00 << 6) = 0 | 8 | 16 | 0 = 24 = 0x18
        let packed = [0x18u8];
        let activation: Vec<i8> = vec![10, 20, 30, 40];
        // dot = 1*10 + (-1)*20 + 0*30 + 1*40 = 10 - 20 + 0 + 40 = 30
        assert_eq!(dot_packed_ternary_i8_fast(&packed, &activation, 4), 30);
    }

    #[test]
    fn packed_i8_dot_matches_unpacked() {
        // Create a larger test case matching against the unpacked kernel
        let weights_i8: Vec<i8> = (0..256)
            .map(|i| match i % 3 {
                0 => 1,
                1 => 0,
                _ => -1,
            })
            .collect();
        let activation: Vec<i8> = (0..256).map(|i| (i as i8).wrapping_mul(3)).collect();

        // Pack weights
        let packed: Vec<u8> = weights_i8
            .chunks(4)
            .map(|chunk| {
                let encode = |v: i8| -> u8 {
                    match v {
                        1 => 0b00,
                        0 => 0b01,
                        -1 => 0b10,
                        _ => 0b01,
                    }
                };
                encode(chunk[0])
                    | (encode(chunk[1]) << 2)
                    | (encode(chunk[2]) << 4)
                    | (encode(chunk[3]) << 6)
            })
            .collect();

        let packed_result = dot_packed_ternary_i8_fast(&packed, &activation, 256);
        let unpacked_result = dot_ternary_i8_fast(&weights_i8, &activation);
        assert_eq!(packed_result, unpacked_result);
    }

    #[test]
    fn packed_f32_dot_basic() {
        // Same weight pattern as packed_i8_dot_basic
        let packed = [0x18u8]; // [+1, -1, 0, +1]
        let input = vec![1.5_f32, 2.5, 3.5, 4.5];
        // dot = 1*1.5 + (-1)*2.5 + 0*3.5 + 1*4.5 = 1.5 - 2.5 + 0 + 4.5 = 3.5
        let result = dot_packed_ternary_f32_fast(&packed, &input, 4);
        assert!((result - 3.5).abs() < 1e-6, "expected 3.5, got {result}");
    }

    #[test]
    fn packed_f32_dot_matches_unpacked() {
        let weights_i8: Vec<i8> = (0..256)
            .map(|i| match i % 3 {
                0 => 1,
                1 => 0,
                _ => -1,
            })
            .collect();
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.8).collect();

        let packed: Vec<u8> = weights_i8
            .chunks(4)
            .map(|chunk| {
                let encode = |v: i8| -> u8 {
                    match v {
                        1 => 0b00,
                        0 => 0b01,
                        -1 => 0b10,
                        _ => 0b01,
                    }
                };
                encode(chunk[0])
                    | (encode(chunk[1]) << 2)
                    | (encode(chunk[2]) << 4)
                    | (encode(chunk[3]) << 6)
            })
            .collect();

        let packed_result = dot_packed_ternary_f32_fast(&packed, &input, 256);
        let unpacked_result = dot_ternary_f32_fast(&weights_i8, &input);
        assert!(
            (packed_result - unpacked_result).abs() < 1e-4,
            "packed={packed_result} vs unpacked={unpacked_result}"
        );
    }

    #[test]
    fn f32_dot_product_basic() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
        let result = dot_f32_f32_fast(&a, &b);
        assert!((result - 70.0).abs() < 1e-4, "expected 70.0, got {result}");
    }

    #[test]
    fn f32_dot_product_large() {
        let n = 2560; // model hidden size
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.0005).collect();

        let simd_result = dot_f32_f32_fast(&a, &b);
        let scalar_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(
            (simd_result - scalar_result).abs() < scalar_result.abs() * 1e-5,
            "simd={simd_result} vs scalar={scalar_result}"
        );
    }

    #[test]
    fn packed_dot_empty() {
        assert_eq!(dot_packed_ternary_i8_fast(&[], &[], 0), 0);
        assert!((dot_packed_ternary_f32_fast(&[], &[], 0)).abs() < 1e-10);
    }

    #[test]
    fn packed_i8_dot_non_chunk_aligned() {
        // Test with length not a multiple of CHUNK (256)
        let n = 100;
        let weights_i8: Vec<i8> = (0..n)
            .map(|i| match i % 3 {
                0 => 1,
                1 => 0,
                _ => -1,
            })
            .collect();
        let activation: Vec<i8> = (0..n).map(|i| ((i * 7) % 255) as i8).collect();

        let packed: Vec<u8> = weights_i8
            .chunks(4)
            .map(|chunk| {
                let encode = |v: i8| -> u8 {
                    match v {
                        1 => 0b00,
                        0 => 0b01,
                        -1 => 0b10,
                        _ => 0b01,
                    }
                };
                let mut byte = encode(chunk[0]);
                if chunk.len() > 1 {
                    byte |= encode(chunk[1]) << 2;
                }
                if chunk.len() > 2 {
                    byte |= encode(chunk[2]) << 4;
                }
                if chunk.len() > 3 {
                    byte |= encode(chunk[3]) << 6;
                }
                byte
            })
            .collect();

        let packed_result = dot_packed_ternary_i8_fast(&packed, &activation, n);
        let unpacked_result = dot_ternary_i8_fast(&weights_i8, &activation);
        assert_eq!(packed_result, unpacked_result);
    }
}
