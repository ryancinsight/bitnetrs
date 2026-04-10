// BitNet b1.58 RMSNorm Compute Shader
// =====================================
//
// Computes Root-Mean-Square Layer Normalization:
//
//   rms(x)  = sqrt( (1/d) * Σ_i x[i]² + ε )
//   out[i]  = x[i] / rms(x) * weight[i]
//
// This matches the LLaMA-family RMSNorm used in BitNet b1.58.
// Unlike LayerNorm, there is NO mean subtraction — only RMS scaling.
//
// Dispatch: one workgroup per token (one RMSNorm call per hidden vector).
// Workgroup size: 256 threads for the reduction, then broadcast.
//
// Bind group layout:
//   @group(0) @binding(0) : input   (array<f32>)  [dim]
//   @group(0) @binding(1) : weight  (array<f32>)  [dim]   (learnable scale γ)
//   @group(0) @binding(2) : output  (array<f32>)  [dim]
//   @group(0) @binding(3) : params  (NormParams)
//
// Mathematical invariant:
//   ∀i: output[i] = input[i] / sqrt(mean(input²) + ε) * weight[i]

// ---------------------------------------------------------------------------
// Parameter uniform
// ---------------------------------------------------------------------------

struct NormParams {
    /// Dimension of the input/output vectors.
    dim : u32,
    /// Numerical stability floor ε (typically 1e-5).
    eps : f32,
    /// Padding to 16-byte alignment.
    _pad0 : u32,
    _pad1 : u32,
}

// ---------------------------------------------------------------------------
// Bind group 0
// ---------------------------------------------------------------------------

/// Input hidden state vector, shape [dim].
@group(0) @binding(0) var<storage, read>       input_buf  : array<f32>;

/// Learnable elementwise scale γ, shape [dim].
@group(0) @binding(1) var<storage, read>       weight_buf : array<f32>;

/// Output vector (written by thread 0 after reduction), shape [dim].
@group(0) @binding(2) var<storage, read_write> output_buf : array<f32>;

/// Scalar parameters.
@group(0) @binding(3) var<uniform>             params     : NormParams;

// ---------------------------------------------------------------------------
// Workgroup shared memory
// ---------------------------------------------------------------------------

// Phase 1: each thread accumulates a partial sum of squared inputs.
// Phase 2: parallel reduction to compute mean(x²).
// Phase 3: all threads use the reduced rms to write their output element.

const WORKGROUP_SIZE : u32 = 256u;

/// Shared buffer for partial sums during the squared-sum reduction.
var<workgroup> partial_sq_sums : array<f32, 256>;

/// Shared storage for the computed inverse-RMS (broadcast to all threads).
var<workgroup> inv_rms_shared : f32;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Each workgroup normalises one hidden state vector of length `dim`.
///
/// Thread layout:
///   - workgroup_id.x  = token index (unused here — one dispatch per token)
///   - local_id.x      = thread index within the workgroup (0 .. 256)
///
/// Each thread handles elements:
///   tid, tid + WORKGROUP_SIZE, tid + 2*WORKGROUP_SIZE, ...
///
/// Reduction computes Σ x[i]², then thread 0 computes inv_rms and stores
/// it in shared memory.  All threads then write their output elements.
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)        wg_id    : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
) {
    let tid : u32 = local_id.x;
    let D   : u32 = params.dim;

    // ── Phase 1: Strided squared-sum accumulation ─────────────────────────
    //
    // Thread tid accumulates x[tid]², x[tid+WS]², x[tid+2*WS]², ...
    var sq_acc : f32 = 0.0;
    var i : u32 = tid;
    while i < D {
        let v : f32 = input_buf[i];
        sq_acc += v * v;
        i += WORKGROUP_SIZE;
    }
    partial_sq_sums[tid] = sq_acc;

    // ── Phase 2: Parallel reduction to sum all partial sums ───────────────
    //
    // Binary-tree reduction: stride = 128, 64, 32, 16, 8, 4, 2, 1.
    // After this, partial_sq_sums[0] = Σ_i x[i]²
    workgroupBarrier();

    var stride : u32 = WORKGROUP_SIZE >> 1u; // 128
    while stride > 0u {
        if tid < stride {
            partial_sq_sums[tid] += partial_sq_sums[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    // ── Phase 3: Thread 0 computes inv_rms and stores in shared memory ────
    //
    // mean(x²) = total_sq / D
    // rms       = sqrt(mean(x²) + ε)
    // inv_rms   = 1 / rms
    if tid == 0u {
        let total_sq : f32 = partial_sq_sums[0];
        let mean_sq  : f32 = total_sq / f32(D);
        let rms      : f32 = sqrt(mean_sq + params.eps);
        inv_rms_shared = 1.0 / rms;
    }

    // All threads wait for inv_rms_shared to be written.
    workgroupBarrier();

    // ── Phase 4: Each thread writes its output elements ───────────────────
    //
    // output[i] = input[i] * inv_rms * weight[i]
    let inv_rms : f32 = inv_rms_shared;
    var j : u32 = tid;
    while j < D {
        output_buf[j] = input_buf[j] * inv_rms * weight_buf[j];
        j += WORKGROUP_SIZE;
    }
}
