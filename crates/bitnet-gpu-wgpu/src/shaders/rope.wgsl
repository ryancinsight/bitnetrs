// BitNet b1.58 Rotary Position Embedding (RoPE) Compute Shader
// =============================================================
//
// Applies Rotary Position Embeddings in-place to Query (Q) and Key (K) tensors.
//
// Mathematical Definition:
//
//   For head vector x ∈ R^{head_dim} at sequence position pos:
//
//   For each dimension pair (2i, 2i+1), i = 0 … head_dim/2 - 1:
//
//     θ_i         = pos / rope_theta^(2i / head_dim)
//     x'[2i]      = x[2i]   * cos(θ_i) - x[2i+1] * sin(θ_i)
//     x'[2i+1]    = x[2i]   * sin(θ_i) + x[2i+1] * cos(θ_i)
//
//   This is equivalent to complex multiplication:
//     (x[2i] + i·x[2i+1]) * exp(i·θ_i)
//
// Frequencies (decay geometrically):
//   freq_i = rope_theta^(-2i / head_dim)
//   angle_i = pos * freq_i
//
// BitNet 2B model parameters:
//   rope_theta = 500000.0
//   head_dim   = 128
//   n_heads    = 20  (for Q)
//   n_kv_heads = 5   (for K)
//
// Dispatch strategy:
//   - One workgroup per head.
//   - workgroup_id.x  ∈ [0, n_heads)      → Q heads
//   - workgroup_id.x  ∈ [n_heads, n_heads + n_kv_heads) → K heads
//   - local_id.x      ∈ [0, head_dim/2)   → dimension pair index i
//   - Workgroup size = head_dim/2 (max 128 for the 2B model)
//
// Bind group layout:
//   @group(0) @binding(0) : q_buf    (array<f32>)  [n_heads * head_dim]      read_write
//   @group(0) @binding(1) : k_buf    (array<f32>)  [n_kv_heads * head_dim]   read_write
//   @group(0) @binding(2) : params   (RopeParams)  uniform
//
// Invariants:
//   - head_dim must be even (enforced on the host before dispatch)
//   - All head vectors have unit-preserving rotation (isometry)
//   - cos²(θ_i) + sin²(θ_i) = 1 for all i (Pythagorean identity)

// ---------------------------------------------------------------------------
// Parameter uniform
// ---------------------------------------------------------------------------

struct RopeParams {
    /// Absolute sequence position of the token being encoded.
    position     : u32,
    /// Per-head feature dimension (must be even).
    head_dim     : u32,
    /// Number of query heads.
    n_heads      : u32,
    /// Number of key/value heads (≤ n_heads due to GQA).
    n_kv_heads   : u32,
    /// RoPE base frequency θ (e.g. 500000.0 for BitNet 2B).
    rope_theta   : f32,
    /// Padding to 16-byte alignment (3 × f32/u32 padding).
    _pad0        : u32,
    _pad1        : u32,
    _pad2        : u32,
}

// ---------------------------------------------------------------------------
// Bind group 0
// ---------------------------------------------------------------------------

/// Query tensor: shape [n_heads × head_dim], modified in-place.
@group(0) @binding(0) var<storage, read_write> q_buf   : array<f32>;

/// Key tensor: shape [n_kv_heads × head_dim], modified in-place.
@group(0) @binding(1) var<storage, read_write> k_buf   : array<f32>;

/// Scalar parameters.
@group(0) @binding(2) var<uniform>             params  : RopeParams;

// ---------------------------------------------------------------------------
// Entry point — Q heads
// ---------------------------------------------------------------------------

/// Apply RoPE to Q heads.
///
/// Dispatch: (n_heads, 1, 1) workgroups, workgroup_size = (head_dim/2, 1, 1).
///
/// Each invocation handles dimension pair i = local_id.x for head h = workgroup_id.x.
///
///   offset = h * head_dim + 2*i
///   x0 = q_buf[offset],  x1 = q_buf[offset+1]
///   freq_i = rope_theta^(-2i / head_dim)
///   angle  = position * freq_i
///   q_buf[offset]   = x0 * cos(angle) - x1 * sin(angle)
///   q_buf[offset+1] = x0 * sin(angle) + x1 * cos(angle)
@compute @workgroup_size(64)
fn rope_q(
    @builtin(workgroup_id)        wg_id    : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
) {
    let h        : u32 = wg_id.x;             // head index
    let pair_idx : u32 = local_id.x;          // dimension pair index i
    let head_dim : u32 = params.head_dim;
    let half_dim : u32 = head_dim >> 1u;      // head_dim / 2

    // Guard: this invocation may be a padding thread if head_dim/2 < workgroup_size.
    if h >= params.n_heads || pair_idx >= half_dim {
        return;
    }

    // Compute the rotation angle for dimension pair i at sequence position pos.
    //
    // freq_i = rope_theta^(-2i / head_dim)
    //        = exp(-2i * log(rope_theta) / head_dim)
    //
    // Using exp/log avoids integer pow which is unavailable in WGSL for f32 exponents.
    let two_i_over_d : f32 = f32(2u * pair_idx) / f32(head_dim);
    let log_theta    : f32 = log(params.rope_theta);
    let freq         : f32 = exp(-two_i_over_d * log_theta);
    let angle        : f32 = f32(params.position) * freq;

    let cos_a : f32 = cos(angle);
    let sin_a : f32 = sin(angle);

    // Linear index into the Q buffer for this head and dimension pair.
    let base : u32 = h * head_dim + pair_idx * 2u;

    let x0 : f32 = q_buf[base];
    let x1 : f32 = q_buf[base + 1u];

    // Complex rotation: (x0 + i·x1) * (cos_a + i·sin_a)
    q_buf[base]      = x0 * cos_a - x1 * sin_a;
    q_buf[base + 1u] = x0 * sin_a + x1 * cos_a;
}

// ---------------------------------------------------------------------------
// Entry point — K heads
// ---------------------------------------------------------------------------

/// Apply RoPE to K heads.
///
/// Dispatch: (n_kv_heads, 1, 1) workgroups, workgroup_size = (head_dim/2, 1, 1).
///
/// Identical computation to rope_q but operates on k_buf and uses n_kv_heads.
@compute @workgroup_size(64)
fn rope_k(
    @builtin(workgroup_id)        wg_id    : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
) {
    let h        : u32 = wg_id.x;
    let pair_idx : u32 = local_id.x;
    let head_dim : u32 = params.head_dim;
    let half_dim : u32 = head_dim >> 1u;

    // Guard: padding threads or out-of-range heads.
    if h >= params.n_kv_heads || pair_idx >= half_dim {
        return;
    }

    // Compute frequency and angle (identical formula to rope_q).
    let two_i_over_d : f32 = f32(2u * pair_idx) / f32(head_dim);
    let log_theta    : f32 = log(params.rope_theta);
    let freq         : f32 = exp(-two_i_over_d * log_theta);
    let angle        : f32 = f32(params.position) * freq;

    let cos_a : f32 = cos(angle);
    let sin_a : f32 = sin(angle);

    let base : u32 = h * head_dim + pair_idx * 2u;

    let x0 : f32 = k_buf[base];
    let x1 : f32 = k_buf[base + 1u];

    k_buf[base]      = x0 * cos_a - x1 * sin_a;
    k_buf[base + 1u] = x0 * sin_a + x1 * cos_a;
}

// ---------------------------------------------------------------------------
// Combined entry point (single dispatch for both Q and K)
// ---------------------------------------------------------------------------

/// Apply RoPE to both Q and K in a single dispatch.
///
/// Dispatch:
///   - workgroup_id.x ∈ [0, n_heads)           → processes Q head wg_id.x
///   - workgroup_id.x ∈ [n_heads, n_heads + n_kv_heads) → processes K head (wg_id.x - n_heads)
///
/// This avoids two separate dispatches at the cost of a branch in the shader.
/// For the BitNet 2B model: n_heads=20, n_kv_heads=5 → 25 total workgroups.
@compute @workgroup_size(64)
fn rope_qk(
    @builtin(workgroup_id)        wg_id    : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
) {
    let head_dim : u32 = params.head_dim;
    let half_dim : u32 = head_dim >> 1u;
    let pair_idx : u32 = local_id.x;

    // Guard: padding threads beyond half_dim.
    if pair_idx >= half_dim {
        return;
    }

    // Determine whether this workgroup handles a Q head or a K head.
    let wg_x     : u32 = wg_id.x;
    let n_q      : u32 = params.n_heads;
    let n_kv     : u32 = params.n_kv_heads;

    // Total valid workgroups = n_heads + n_kv_heads.
    if wg_x >= n_q + n_kv {
        return;
    }

    // Precompute angle for this dimension pair and position.
    let two_i_over_d : f32 = f32(2u * pair_idx) / f32(head_dim);
    let log_theta    : f32 = log(params.rope_theta);
    let freq         : f32 = exp(-two_i_over_d * log_theta);
    let angle        : f32 = f32(params.position) * freq;

    let cos_a : f32 = cos(angle);
    let sin_a : f32 = sin(angle);

    if wg_x < n_q {
        // ── Q head ──────────────────────────────────────────────────────────
        let h    : u32 = wg_x;
        let base : u32 = h * head_dim + pair_idx * 2u;

        let x0 : f32 = q_buf[base];
        let x1 : f32 = q_buf[base + 1u];

        q_buf[base]      = x0 * cos_a - x1 * sin_a;
        q_buf[base + 1u] = x0 * sin_a + x1 * cos_a;
    } else {
        // ── K head ──────────────────────────────────────────────────────────
        let h    : u32 = wg_x - n_q;
        let base : u32 = h * head_dim + pair_idx * 2u;

        let x0 : f32 = k_buf[base];
        let x1 : f32 = k_buf[base + 1u];

        k_buf[base]      = x0 * cos_a - x1 * sin_a;
        k_buf[base + 1u] = x0 * sin_a + x1 * cos_a;
    }
}
