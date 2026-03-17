#include "common.cuh"
#include "smem_utils.cuh"

#include <torch/extension.h>

// Inline FP8 E4M3 → float conversion from raw byte
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t raw) {
    __nv_fp8_e4m3 v;
    v.__x = raw;
    return float(v);
}

// -------------------------------------------------------------------
// Decode partial kernel
// Grid: (num_tokens * REPLICATE_H, NI)
// Block: THREADS threads
// Each CTA: one head-tile (HPB heads) x one KV-block (BI entries)
// Writes partial_O [num_tokens, NI, num_heads, D_NOPE] bf16
//        partial_LSE [num_tokens, NI, num_heads]        fp32
// -------------------------------------------------------------------

template <int HPB, int BI>
__global__ void sparse_mla_decode_partial_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    bf16* __restrict__ partial_O,
    float* __restrict__ partial_LSE,
    float sm_scale,
    int num_heads,
    int num_tokens,
    int NI)
{
    static_assert(HPB == 16 && (BI == 32 || BI == 64), "supported configs");
    const int THREADS = blockDim.x;
    const int tid = threadIdx.x;

    const int REPLICATE_H = num_heads / HPB;
    const int s_i = blockIdx.x / REPLICATE_H;
    const int h_tile = blockIdx.x % REPLICATE_H;
    const int h_start = h_tile * HPB;
    const int kv_block = blockIdx.y;
    if (s_i >= num_tokens) return;

    const float sm_scale_log2e = sm_scale * LOG2E;

    // ---- shared memory layout ----
    extern __shared__ char smem_raw[];
    bf16*    q_smem     = reinterpret_cast<bf16*>(smem_raw);
    // q_smem: [HPB, DIM]  → HPB * DIM * 2 bytes
    uint8_t* kv_smem    = reinterpret_cast<uint8_t*>(q_smem + HPB * DIM);
    // kv_smem: [BI, KV_PACKED_BYTES]
    float*   score_smem = reinterpret_cast<float*>(kv_smem + BI * KV_PACKED_BYTES);
    // score_smem: [HPB, BI]
    int32_t* valid_smem = reinterpret_cast<int32_t*>(score_smem + HPB * BI);
    // valid_smem: [BI]  (1 = valid, 0 = invalid)

    // ================================================================
    // Step 1: Cooperative load of Q for this head tile
    // ================================================================
    {
        const bf16* q_base = Q + (size_t)s_i * num_heads * DIM + (size_t)h_start * DIM;
        const int q_elems = HPB * DIM; // 16 * 576 = 9216 bf16 values
        for (int i = tid; i < q_elems; i += THREADS) {
            q_smem[i] = q_base[i];
        }
    }

    // ================================================================
    // Step 2: Gather KV entries from the pool via sparse indices
    // ================================================================
    {
        const int32_t* idx_base = indices + (size_t)s_i * TOPK + kv_block * BI;

        // Load validity mask
        for (int bi = tid; bi < BI; bi += THREADS) {
            valid_smem[bi] = (idx_base[bi] >= 0) ? 1 : 0;
        }

        // Flatten the gather: total bytes = BI * KV_PACKED_BYTES
        // Load 16 bytes per iteration per thread
        constexpr int KV_TOTAL = BI * KV_PACKED_BYTES;
        constexpr int LOAD_WIDTH = 16; // 128 bits

        for (int flat = tid * LOAD_WIDTH; flat < KV_TOTAL; flat += THREADS * LOAD_WIDTH) {
            int bi = flat / KV_PACKED_BYTES;
            int byte_off = flat % KV_PACKED_BYTES;

            int idx = idx_base[bi];
            idx = (idx >= 0) ? idx : 0;

            const uint8_t* src = KV_cache + (size_t)idx * KV_PACKED_BYTES + byte_off;
            uint8_t* dst = kv_smem + flat;

            // 656 is divisible by 16, so boundary is aligned
            if (byte_off + LOAD_WIDTH <= KV_PACKED_BYTES) {
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
            }
        }
    }

    __syncthreads();

    // ================================================================
    // Step 3: Compute scores  score[h][k] = Q_nope·K_nope + Q_rope·K_rope
    // ================================================================
    {
        const int total_pairs = HPB * BI;
        for (int pair = tid; pair < total_pairs; pair += THREADS) {
            int h = pair / BI;
            int k = pair % BI;

            if (!valid_smem[k]) {
                score_smem[h * BI + k] = -1e30f;
                continue;
            }

            const bf16* q_row = q_smem + h * DIM;
            const uint8_t* kv_row = kv_smem + k * KV_PACKED_BYTES;
            const float* k_scales = reinterpret_cast<const float*>(kv_row + D_NOPE);
            const bf16* k_rope = reinterpret_cast<const bf16*>(kv_row + D_NOPE + NUM_SCALES * sizeof(float));

            float score = 0.0f;

            // Nope part: 4 blocks of 128, each with its own scale
            #pragma unroll
            for (int b = 0; b < NUM_SCALES; b++) {
                float partial = 0.0f;
                const uint8_t* k_fp8_block = kv_row + b * QUANT_TILE;
                const bf16* q_block = q_row + b * QUANT_TILE;

                // Vectorized: process 4 FP8 elements at a time
                #pragma unroll 4
                for (int d = 0; d < QUANT_TILE; d += 4) {
                    uint32_t k4 = *reinterpret_cast<const uint32_t*>(k_fp8_block + d);
                    float k0 = fp8e4m3_to_float(k4 & 0xFF);
                    float k1 = fp8e4m3_to_float((k4 >> 8) & 0xFF);
                    float k2 = fp8e4m3_to_float((k4 >> 16) & 0xFF);
                    float k3 = fp8e4m3_to_float((k4 >> 24) & 0xFF);

                    // Load Q as 2 x bf16
                    float q0 = to_float(q_block[d]);
                    float q1 = to_float(q_block[d + 1]);
                    float q2 = to_float(q_block[d + 2]);
                    float q3 = to_float(q_block[d + 3]);

                    partial += k0 * q0 + k1 * q1 + k2 * q2 + k3 * q3;
                }
                score += partial * k_scales[b];
            }

            // Rope part: 64 dims, bf16 x bf16
            {
                const bf16* q_rope = q_row + D_NOPE;
                float rope_score = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D_ROPE; d += 2) {
                    rope_score += to_float(q_rope[d])     * to_float(k_rope[d])
                                + to_float(q_rope[d + 1]) * to_float(k_rope[d + 1]);
                }
                score += rope_score;
            }

            score_smem[h * BI + k] = score;
        }
    }

    __syncthreads();

    // ================================================================
    // Step 4: Local softmax per head  &  write partial_LSE
    // ================================================================
    for (int h = tid; h < HPB; h += THREADS) {
        float* row = score_smem + h * BI;

        float max_val = -1e30f;
        for (int k = 0; k < BI; k++)
            max_val = fmaxf(max_val, row[k] * sm_scale_log2e);

        float sum_exp = 0.0f;
        for (int k = 0; k < BI; k++) {
            float s = fast_exp2f(row[k] * sm_scale_log2e - max_val);
            row[k] = s;
            sum_exp += s;
        }

        // Normalize
        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (int k = 0; k < BI; k++) row[k] *= inv_sum;

        // Write LSE
        float lse = (sum_exp > 0.0f) ? (log2f(sum_exp) + max_val) : -1e30f;
        size_t lse_idx = (size_t)s_i * NI * num_heads
                       + (size_t)kv_block * num_heads
                       + (h_start + h);
        partial_LSE[lse_idx] = lse;
    }

    __syncthreads();

    // ================================================================
    // Step 5: Compute output = weights · V_nope (dequantized)
    // ================================================================
    {
        const int total_out = HPB * D_NOPE; // 16 * 512 = 8192
        for (int elem = tid; elem < total_out; elem += THREADS) {
            int h = elem / D_NOPE;
            int d = elem % D_NOPE;
            int scale_blk = d / QUANT_TILE;

            const float* weights = score_smem + h * BI;
            float acc = 0.0f;

            for (int k = 0; k < BI; k++) {
                float w = weights[k];
                if (w == 0.0f) continue;

                const uint8_t* v_row = kv_smem + k * KV_PACKED_BYTES;
                const float* v_scales = reinterpret_cast<const float*>(v_row + D_NOPE);
                float v_val = fp8e4m3_to_float(v_row[d]) * v_scales[scale_blk];
                acc += w * v_val;
            }

            size_t out_idx = (size_t)s_i * NI * num_heads * D_NOPE
                           + (size_t)kv_block * num_heads * D_NOPE
                           + (size_t)(h_start + h) * D_NOPE
                           + d;
            partial_O[out_idx] = to_bf16(acc);
        }
    }
}

// -------------------------------------------------------------------
// Launch wrapper
// -------------------------------------------------------------------

void sparse_mla_decode_partial_launch(
    torch::Tensor Q,
    torch::Tensor KV_cache,
    torch::Tensor indices,
    torch::Tensor partial_O,
    torch::Tensor partial_LSE,
    float sm_scale,
    int num_heads,
    int num_tokens,
    int topk,
    int BI_param)
{
    const int HPB = 16;
    const int BI = BI_param;
    const int NI = topk / BI;
    const int REPLICATE_H = num_heads / HPB;
    const int THREADS = 256;

    // Shared memory size
    size_t smem_bytes = HPB * DIM * sizeof(bf16)          // q_smem
                      + BI * KV_PACKED_BYTES               // kv_smem
                      + HPB * BI * sizeof(float)           // score_smem
                      + BI * sizeof(int32_t);              // valid_smem

    dim3 grid(num_tokens * REPLICATE_H, NI);
    dim3 block(THREADS);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            sparse_mla_decode_partial_kernel<16, 64>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
    }

    if (BI == 32) {
        sparse_mla_decode_partial_kernel<16, 32><<<grid, block, smem_bytes>>>(
            reinterpret_cast<const bf16*>(Q.data_ptr()),
            reinterpret_cast<const uint8_t*>(KV_cache.data_ptr()),
            indices.data_ptr<int32_t>(),
            reinterpret_cast<bf16*>(partial_O.data_ptr()),
            partial_LSE.data_ptr<float>(),
            sm_scale, num_heads, num_tokens, NI);
    } else {
        sparse_mla_decode_partial_kernel<16, 64><<<grid, block, smem_bytes>>>(
            reinterpret_cast<const bf16*>(Q.data_ptr()),
            reinterpret_cast<const uint8_t*>(KV_cache.data_ptr()),
            indices.data_ptr<int32_t>(),
            reinterpret_cast<bf16*>(partial_O.data_ptr()),
            partial_LSE.data_ptr<float>(),
            sm_scale, num_heads, num_tokens, NI);
    }
}
