// BitNet b1.58 Causal Scaled Dot-Product Attention with GQA — Compute Shader
// ===========================================================================
//
// Implements causal (auto-regressive) scaled dot-product attention with
// Grouped Query Attention (GQA) for decoding a single new token.
//
// Mathematical Definition (per query head h):
//
//   kv_head  = h / heads_per_group           (integer division)
//   scale    = 1 / sqrt(head_dim)
//
//   scores[t] = dot(Q[h], K_cache[t, kv_head]) * scale
//               for t in 0 .. cur_pos+1        (causal mask: attend to past)
//
//   attn[t]   = softmax(scores[0 .. cur_pos+1])[t]
//
//   out[h]    = Σ_{t=0}^{cur_pos}  attn[t] * V_cache[t, kv_head]
//
// Key design decisions:
//   - One workgroup per query head.
//   - Within the workgroup, threads cooperate to compute the dot products and
//     the weighted sum over the sequence length.
//   - Numerically-stable online softmax (max subtraction before exp).
//   - GQA: n_heads query heads share n_kv_heads key/value heads.
//
// BitNet 2B model dimensions:
//   n_heads    = 20,  n_kv_heads = 5,  heads_per_group = 4
//   head_dim   = 128
//   max_seq    = 4096
//
// KV cache layout (row-major):
//   cache[t, kv_h, d] = cache_buf[ t * n_kv_heads * head_dim
//                                 + kv_h * head_dim
//                                 + d ]
//
// Dispatch:
//   (n_heads, 1, 1) workgroups, each of size (WORKGROUP_SIZE, 1, 1).
//
// Bind group layout:
//   @group(0) @binding(0) : q_buf      (array<f32>)    [n_heads * head_dim]        read
//   @group(0) @binding(1) : k_cache    (array<f32>)    [(cur_pos+1) * n_kv_heads * head_dim]  read
//   @group(0) @binding(2) : v_cache    (array<f32>)    [(cur_pos+1) * n_kv_heads * head_dim]  read
//   @group(0) @binding(3) : output_buf (array<f32>)    [n_heads * head_dim]        read_write
//   @group(0) @binding(4) : params     (AttnParams)    uniform
//
// Invariants:
//   - cur_pos + 1 = seq_len  (number of valid KV cache positions)
//   - n_heads % n_kv_heads == 0  (GQA constraint)
//   - output is a convex combination of V vectors (weights sum to 1)

// ---------------------------------------------------------------------------
// Workgroup size constant
// ---------------------------------------------------------------------------

// 64 threads per workgroup. For head_dim=128, each thread handles 2 elements
// in the output accumulation phase (128/64=2). For seq_len up to 4096,
// threads loop over time steps.
const WORKGROUP_SIZE : u32 = 64u;

// Maximum sequence length supported (must match host-side constant).
// Used to size the shared memory score buffer.
// For BitNet 2B: 4096. Here we use 4096 as a compile-time constant.
// The host must ensure cur_pos+1 <= MAX_SEQ_LEN.
const MAX_SEQ_LEN : u32 = 4096u;

// ---------------------------------------------------------------------------
// Parameter uniform
// ---------------------------------------------------------------------------

struct AttnParams {
    /// Number of query attention heads H.
    n_heads          : u32,
    /// Number of key/value heads H_kv (divides H evenly).
    n_kv_heads       : u32,
    /// Per-head feature dimension d_h.
    head_dim         : u32,
    /// Current sequence position (0-indexed). Attends to positions 0..=cur_pos.
    cur_pos          : u32,
    /// Attention scale: 1 / sqrt(head_dim), stored as f32 to avoid re-computing.
    scale            : f32,
    /// Padding to 16-byte alignment.
    _pad0            : u32,
    _pad1            : u32,
    _pad2            : u32,
}

// ---------------------------------------------------------------------------
// Bind group 0
// ---------------------------------------------------------------------------

/// Query tensor, shape [n_heads × head_dim].
@group(0) @binding(0) var<storage, read>       q_buf      : array<f32>;

/// Key cache, shape [(cur_pos+1) × n_kv_heads × head_dim].
@group(0) @binding(1) var<storage, read>       k_cache    : array<f32>;

/// Value cache, shape [(cur_pos+1) × n_kv_heads × head_dim].
@group(0) @binding(2) var<storage, read>       v_cache    : array<f32>;

/// Output tensor, shape [n_heads × head_dim]. Written after attention.
@group(0) @binding(3) var<storage, read_write> output_buf : array<f32>;

/// Scalar parameters.
@group(0) @binding(4) var<uniform>             params     : AttnParams;

// ---------------------------------------------------------------------------
// Workgroup shared memory
// ---------------------------------------------------------------------------

// Score buffer: stores the (scaled, pre-softmax) dot products for one head.
// Size = MAX_SEQ_LEN = 4096 f32 values = 16 KiB per workgroup.
// Hardware limit is typically 32–64 KiB per workgroup; 16 KiB is safe.
var<workgroup> scores : array<f32, 4096>;

// Partial reduction buffers for the dot-product and weighted-sum phases.
// Each holds WORKGROUP_SIZE = 64 values.
var<workgroup> partial_buf : array<f32, 64>;

// Shared softmax numerics (written by thread 0, read by all).
var<workgroup> shared_max     : f32;  // max score for stable softmax
var<workgroup> shared_sum_exp : f32;  // sum of exp(score - max)

// ---------------------------------------------------------------------------
// Helper: parallel reduction (sum) over WORKGROUP_SIZE elements in partial_buf
// ---------------------------------------------------------------------------
//
// After this function, partial_buf[0] contains the sum of all 64 elements.
// All threads must call workgroupBarrier() before and after this function.
fn reduce_sum_64(tid: u32) {
    // Binary-tree reduction: stride 32, 16, 8, 4, 2, 1
    if tid < 32u { partial_buf[tid] += partial_buf[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { partial_buf[tid] += partial_buf[tid + 16u]; }
    workgroupBarrier();
    if tid <  8u { partial_buf[tid] += partial_buf[tid +  8u]; }
    workgroupBarrier();
    if tid <  4u { partial_buf[tid] += partial_buf[tid +  4u]; }
    workgroupBarrier();
    if tid <  2u { partial_buf[tid] += partial_buf[tid +  2u]; }
    workgroupBarrier();
    if tid <  1u { partial_buf[tid] += partial_buf[tid +  1u]; }
    workgroupBarrier();
}

// ---------------------------------------------------------------------------
// Helper: parallel reduction (max) over WORKGROUP_SIZE elements in partial_buf
// ---------------------------------------------------------------------------
//
// After this function, partial_buf[0] contains max of all 64 elements.
fn reduce_max_64(tid: u32) {
    if tid < 32u { partial_buf[tid] = max(partial_buf[tid], partial_buf[tid + 32u]); }
    workgroupBarrier();
    if tid < 16u { partial_buf[tid] = max(partial_buf[tid], partial_buf[tid + 16u]); }
    workgroupBarrier();
    if tid <  8u { partial_buf[tid] = max(partial_buf[tid], partial_buf[tid +  8u]); }
    workgroupBarrier();
    if tid <  4u { partial_buf[tid] = max(partial_buf[tid], partial_buf[tid +  4u]); }
    workgroupBarrier();
    if tid <  2u { partial_buf[tid] = max(partial_buf[tid], partial_buf[tid +  2u]); }
    workgroupBarrier();
    if tid <  1u { partial_buf[tid] = max(partial_buf[tid], partial_buf[tid +  1u]); }
    workgroupBarrier();
}

// ---------------------------------------------------------------------------
// Main compute entry point
// ---------------------------------------------------------------------------

/// Compute causal GQA attention for one query head per workgroup.
///
/// Workgroup mapping:
///   workgroup_id.x = query head index h  ∈ [0, n_heads)
///   local_id.x     = thread index tid    ∈ [0, WORKGROUP_SIZE)
///
/// Algorithm (per workgroup, i.e. per query head h):
///
///   1. Determine KV head index: kv_h = h / heads_per_group
///   2. For each time step t = 0 .. seq_len:
///        Compute scores[t] = dot(Q[h], K_cache[t, kv_h]) * scale
///      (threads share this work via strided accumulation + reduction)
///   3. Numerically-stable softmax over scores[0..seq_len]:
///        max_score = max(scores)
///        attn[t]   = exp(scores[t] - max_score) / Σ_t exp(scores[t] - max_score)
///   4. Compute output[h] = Σ_t attn[t] * V_cache[t, kv_h]
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)        wg_id    : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
) {
    let h   : u32 = wg_id.x;     // query head index
    let tid : u32 = local_id.x;  // thread index within workgroup

    let n_heads    : u32 = params.n_heads;
    let n_kv_heads : u32 = params.n_kv_heads;
    let head_dim   : u32 = params.head_dim;
    let cur_pos    : u32 = params.cur_pos;
    let seq_len    : u32 = cur_pos + 1u;
    let scale      : f32 = params.scale;

    // Guard: skip padding workgroups.
    if h >= n_heads {
        return;
    }

    // ── GQA: map query head to its KV head ────────────────────────────────
    let heads_per_group : u32 = n_heads / n_kv_heads;
    let kv_h            : u32 = h / heads_per_group;

    // Base offsets into the flattened Q / KV tensors.
    let q_base  : u32 = h    * head_dim;   // Q[h, *]
    let kv_stride : u32 = n_kv_heads * head_dim; // elements per time step in cache

    // =========================================================================
    // Phase 1: Compute attention scores scores[t] = dot(Q[h], K_cache[t, kv_h]) * scale
    // =========================================================================
    //
    // For each time step t, threads cooperate: each thread handles a subset of
    // the head_dim dimensions, accumulates into partial_buf, then reduces to get
    // the full dot product for that time step.
    //
    // To handle long sequences efficiently, we compute all time steps in a loop.
    // For each t, the reduction is performed across the 64 threads.

    var t : u32 = 0u;
    while t < seq_len {
        // K cache base for this time step and kv head.
        let k_base : u32 = t * kv_stride + kv_h * head_dim;

        // Strided dot product: each thread handles dimensions tid, tid+64, tid+128, ...
        var dot_acc : f32 = 0.0;
        var d : u32 = tid;
        while d < head_dim {
            dot_acc += q_buf[q_base + d] * k_cache[k_base + d];
            d += WORKGROUP_SIZE;
        }

        // Reduce across threads to sum up the partial dot products.
        partial_buf[tid] = dot_acc;
        workgroupBarrier();
        reduce_sum_64(tid);

        // Thread 0 stores the final score for this time step.
        if tid == 0u {
            scores[t] = partial_buf[0] * scale;
        }
        workgroupBarrier();

        t += 1u;
    }
    // All scores[0..seq_len] are now populated.

    // =========================================================================
    // Phase 2: Numerically-stable softmax over scores[0..seq_len]
    // =========================================================================
    //
    // Step 2a: Find max score (for numerical stability).
    //
    // Each thread finds the max over its assigned time steps, then we reduce.

    var local_max : f32 = -3.402823466e+38; // -f32::MAX
    var ts : u32 = tid;
    while ts < seq_len {
        if scores[ts] > local_max {
            local_max = scores[ts];
        }
        ts += WORKGROUP_SIZE;
    }

    partial_buf[tid] = local_max;
    workgroupBarrier();
    reduce_max_64(tid);

    if tid == 0u {
        shared_max = partial_buf[0];
    }
    workgroupBarrier();

    let max_score : f32 = shared_max;

    // Step 2b: Compute exp(score - max) and sum.
    //
    // Each thread updates its assigned time steps in the scores buffer
    // (reusing it to store exp values), and accumulates a partial sum.

    var local_sum_exp : f32 = 0.0;
    var te : u32 = tid;
    while te < seq_len {
        let e : f32 = exp(scores[te] - max_score);
        scores[te] = e;      // overwrite score with exp value
        local_sum_exp += e;
        te += WORKGROUP_SIZE;
    }
    workgroupBarrier();

    // Reduce sum_exp across threads.
    partial_buf[tid] = local_sum_exp;
    workgroupBarrier();
    reduce_sum_64(tid);

    if tid == 0u {
        shared_sum_exp = partial_buf[0];
    }
    workgroupBarrier();

    let inv_sum_exp : f32 = 1.0 / shared_sum_exp;

    // Step 2c: Normalise — convert exp values to attention weights.
    var tn : u32 = tid;
    while tn < seq_len {
        scores[tn] *= inv_sum_exp;   // scores[t] now holds attn[t] ∈ (0,1)
        tn += WORKGROUP_SIZE;
    }
    workgroupBarrier();
    // scores[0..seq_len] now contain the attention probabilities.

    // =========================================================================
    // Phase 3: Compute output[h] = Σ_t attn[t] * V_cache[t, kv_h]
    // =========================================================================
    //
    // For each output dimension d, we sum over all time steps:
    //   out[h, d] = Σ_t  attn[t] * V_cache[t, kv_h, d]
    //
    // Thread tid handles output dimensions: tid, tid+64, tid+128, ...
    // For each such dimension d, thread tid loops over all time steps.

    let out_base : u32 = h * head_dim;

    var d_out : u32 = tid;
    while d_out < head_dim {
        var weighted_sum : f32 = 0.0;
        var tv : u32 = 0u;
        while tv < seq_len {
            let v_offset : u32 = tv * kv_stride + kv_h * head_dim + d_out;
            weighted_sum += scores[tv] * v_cache[v_offset];
            tv += 1u;
        }
        output_buf[out_base + d_out] = weighted_sum;
        d_out += WORKGROUP_SIZE;
    }
    // output_buf[h * head_dim .. (h+1) * head_dim] now contains out[h].
}
