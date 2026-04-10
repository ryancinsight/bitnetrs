// BitNet b1.58 Packed 2-bit Ternary GEMV Compute Shader
// ======================================================
//
// Computes:
//   output[i] = weight_scale * sum_j(decode(weight_packed)[i*K + j] * input[j])
//
// Weights are stored as packed 2-bit ternary: 4 values per byte, 16 per u32.
// Little-endian byte layout: u32[n] = byte[4n] | (byte[4n+1]<<8) | (byte[4n+2]<<16) | (byte[4n+3]<<24)
//
// 2-bit code encoding:
//   0b00 = +1  (code 0)
//   0b01 =  0  (code 1)
//   0b10 = -1  (code 2)
//   0b11 =  0  (code 3, padding only)
//
// Dispatch: one workgroup per output row.
// Workgroup size: 256 threads, strided accumulation + shared-memory reduction.

struct GemvParams {
    out_features : u32,
    in_features  : u32,
    weight_scale : f32,
    packed_cols  : u32,
}

@group(0) @binding(0) var<storage, read>       weight_buf : array<u32>;
@group(0) @binding(1) var<storage, read>       input_buf  : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_buf : array<f32>;
@group(0) @binding(3) var<uniform>             params     : GemvParams;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> partial_sums: array<f32, 256>;

// Decode a 2-bit code to a ternary float value.
//
// Theorem: the encoding map { 0->+1, 1->0, 2->-1, 3->0 } can be implemented
// with two select operations:
//   if code == 0 -> 1.0  else:
//     if code == 2 -> -1.0  else: 0.0
//
// WGSL select(false_val, true_val, condition): returns true_val when condition is true.
fn decode_code(code: u32) -> f32 {
    return select(select(0.0, -1.0, code == 2u), 1.0, code == 0u);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)        wg_id    : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
) {
    let row         : u32 = wg_id.x;
    let tid         : u32 = local_id.x;
    let K           : u32 = params.in_features;
    let M           : u32 = params.out_features;
    let packed_cols : u32 = params.packed_cols;

    if row >= M {
        partial_sums[tid] = 0.0;
        workgroupBarrier();
        return;
    }

    // Byte-array offset for the start of this row's packed weights.
    let row_byte_base: u32 = row * packed_cols;

    var acc: f32 = 0.0;
    var j  : u32 = tid;

    while j < K {
        // Byte index within the logical (flat) packed byte array.
        let byte_idx : u32 = row_byte_base + j / 4u;

        // 2-bit position within that byte (0, 2, 4, or 6).
        let bit_shift: u32 = (j & 3u) * 2u;

        // The packed buffer is array<u32>; each u32 holds 4 bytes (little-endian).
        // Byte byte_idx resides in bits [byte_in_u32*8 .. byte_in_u32*8+7] of
        // the u32 at index byte_idx/4.
        let u32_idx     : u32 = byte_idx / 4u;
        let byte_in_u32 : u32 = byte_idx & 3u;
        let total_shift : u32 = byte_in_u32 * 8u + bit_shift;

        let code : u32  = (weight_buf[u32_idx] >> total_shift) & 3u;
        acc += decode_code(code) * input_buf[j];

        j += WORKGROUP_SIZE;
    }

    partial_sums[tid] = acc;
    workgroupBarrier();

    // Parallel reduction: WORKGROUP_SIZE -> 1.
    var stride: u32 = WORKGROUP_SIZE >> 1u;
    while stride > 0u {
        if tid < stride {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    if tid == 0u {
        output_buf[row] = params.weight_scale * partial_sums[0];
    }
}
