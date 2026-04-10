// BitNet b1.58 Ternary GEMV Compute Shader
// ==========================================
//
// Computes: output[i] = weight_scale * Σ_j weight[i*in_features + j] * input[j]
//
// where weight values are ternary: {-1, 0, +1} stored as i32 in a storage buffer.
//
// Dispatch: one workgroup per output element, or tiled for large matrices.
// Workgroup size: 256 threads, each handling a portion of the dot product.
//
// Bind group layout:
//   @group(0) @binding(0) : weight  (array<i32>)  [out_features * in_features]
//   @group(0) @binding(1) : input   (array<f32>)  [in_features]
//   @group(0) @binding(2) : output  (array<f32>)  [out_features]
//   @group(0) @binding(3) : params  (GemvParams)
//
// Mathematical invariant:
//   output[i] = weight_scale * dot(weight[i*K .. (i+1)*K], input[0..K])
//   where K = in_features, weight values ∈ {-1, 0, +1}

// ---------------------------------------------------------------------------
// Parameter uniform
// ---------------------------------------------------------------------------

struct GemvParams {
    /// Number of output neurons (matrix rows).
    out_features : u32,
    /// Number of input features (matrix columns).
    in_features  : u32,
    /// Per-tensor absmean scale α_W (dequantisation factor).
    weight_scale : f32,
    /// Padding to satisfy 16-byte alignment.
    _pad         : u32,
}

// ---------------------------------------------------------------------------
// Bind group 0
// ---------------------------------------------------------------------------

/// Ternary weight matrix in row-major layout.
/// Each element is an i32 with value -1, 0, or +1.
/// Shape: [out_features × in_features]
@group(0) @binding(0) var<storage, read>       weight_buf : array<i32>;

/// Input activation vector.
/// Shape: [in_features]
@group(0) @binding(1) var<storage, read>       input_buf  : array<f32>;

/// Output vector (written atomically via shared memory reduction).
/// Shape: [out_features]
@group(0) @binding(2) var<storage, read_write> output_buf : array<f32>;

/// Scalar parameters.
@group(0) @binding(3) var<uniform>             params     : GemvParams;

// ---------------------------------------------------------------------------
// Workgroup shared memory
// ---------------------------------------------------------------------------

// Each workgroup computes the dot product for a single output row.
// 256 threads each accumulate a partial sum, then we reduce.
const WORKGROUP_SIZE : u32 = 256u;

var<workgroup> partial_sums : array<f32, 256>;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Each workgroup computes output[workgroup_id.x].
///
/// Thread layout:
///   - workgroup_id.x  = output row index (0 .. out_features)
///   - local_id.x      = thread index within the workgroup (0 .. 256)
///
/// Each thread handles elements at indices:
///   local_id.x, local_id.x + WORKGROUP_SIZE, local_id.x + 2*WORKGROUP_SIZE, ...
///
/// After the strided accumulation, a parallel reduction in shared memory
/// computes the total dot product, and thread 0 writes the scaled result.
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)   wg_id    : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
) {
    let row     : u32 = wg_id.x;       // which output neuron
    let tid     : u32 = local_id.x;    // thread index within workgroup
    let K       : u32 = params.in_features;
    let M       : u32 = params.out_features;

    // Guard: skip if row is out of bounds (can happen if M is not a multiple
    // of the dispatch grid granularity).
    if row >= M {
        partial_sums[tid] = 0.0;
        workgroupBarrier();
        return;
    }

    // ── Strided accumulation ──────────────────────────────────────────────
    //
    // Thread tid accumulates:
    //   weight[row*K + tid], weight[row*K + tid + WS], weight[row*K + tid + 2*WS], …
    //
    // This pattern provides coalesced memory access for the weight matrix
    // (consecutive threads access consecutive memory locations).
    let row_offset : u32 = row * K;
    var acc : f32 = 0.0;

    var j : u32 = tid;
    while j < K {
        // weight value is i32 ∈ {-1, 0, +1}
        let w_val : i32 = weight_buf[row_offset + j];
        let x_val : f32 = input_buf[j];

        // Ternary multiply: branch-free via i32→f32 cast.
        // -1 → -1.0, 0 → 0.0, +1 → +1.0
        acc += f32(w_val) * x_val;

        j += WORKGROUP_SIZE;
    }

    // Store partial sum in shared memory.
    partial_sums[tid] = acc;

    // ── Parallel reduction in shared memory ──────────────────────────────
    //
    // Reduces WORKGROUP_SIZE = 256 partial sums to a single total.
    // Uses a standard binary-tree reduction with stride halving.
    //
    // Iteration 1: stride=128, threads 0..127 add from threads 128..255
    // Iteration 2: stride=64,  threads 0..63  add from threads 64..127
    // ...
    // Iteration 8: stride=1,   thread 0 adds from thread 1
    workgroupBarrier();

    var stride : u32 = WORKGROUP_SIZE >> 1u; // 128
    while stride > 0u {
        if tid < stride {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    // ── Write result ──────────────────────────────────────────────────────
    //
    // Only thread 0 writes the final result for this row.
    // output[row] = weight_scale * total_dot_product
    if tid == 0u {
        output_buf[row] = params.weight_scale * partial_sums[0];
    }
}
