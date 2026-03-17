#include "common.cuh"
#include "smem_utils.cuh"

#include <torch/extension.h>

__device__ __forceinline__ float fp8e4m3_to_float_pf(uint8_t raw) {
    __nv_fp8_e4m3 v;
    v.__x = raw;
    return float(v);
}

// -------------------------------------------------------------------
// Prefill fused kernel: online softmax over all topk KV blocks
// Grid: (num_tokens * REPLICATE_H, 1)
// Block: THREADS
// Each CTA: one head-tile (HPB heads), loops over ALL NI KV blocks
// Output: [num_tokens, num_heads, D_NOPE]  bf16
// -------------------------------------------------------------------

template <int HPB, int BI>
__global__ void sparse_mla_prefill_fused_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    bf16* __restrict__ output,
    float sm_scale,
    int num_heads,
    int num_tokens,
    int topk)
{
    const int THREADS = blockDim.x;
    const int tid = threadIdx.x;
    const int NI = topk / BI;

    const int REPLICATE_H = num_heads / HPB;
    const int s_i = blockIdx.x / REPLICATE_H;
    const int h_tile = blockIdx.x % REPLICATE_H;
    const int h_start = h_tile * HPB;
    if (s_i >= num_tokens) return;

    const float sm_scale_log2e = sm_scale * LOG2E;

    // ---- shared memory ----
    extern __shared__ char smem_raw[];
    bf16*    q_smem     = reinterpret_cast<bf16*>(smem_raw);               // [HPB, DIM]
    uint8_t* kv_smem    = reinterpret_cast<uint8_t*>(q_smem + HPB * DIM);  // [BI, KV_PACKED_BYTES]
    float*   score_smem = reinterpret_cast<float*>(kv_smem + BI * KV_PACKED_BYTES); // [HPB, BI]
    int32_t* valid_smem = reinterpret_cast<int32_t*>(score_smem + HPB * BI);        // [BI]
    // After valid_smem: accumulator state in shared memory
    float*   acc_o_smem = reinterpret_cast<float*>(valid_smem + BI);  // [HPB, D_NOPE]
    float*   m_smem     = acc_o_smem + HPB * D_NOPE;                  // [HPB]
    float*   l_smem     = m_smem + HPB;                                // [HPB]

    // ================================================================
    // Step 1: Load Q
    // ================================================================
    {
        const bf16* q_base = Q + (size_t)s_i * num_heads * DIM + (size_t)h_start * DIM;
        for (int i = tid; i < HPB * DIM; i += THREADS)
            q_smem[i] = q_base[i];
    }

    // Initialize online softmax accumulators
    for (int i = tid; i < HPB * D_NOPE; i += THREADS)
        acc_o_smem[i] = 0.0f;
    for (int h = tid; h < HPB; h += THREADS) {
        m_smem[h] = -1e30f;
        l_smem[h] = 0.0f;
    }
    __syncthreads();

    // ================================================================
    // Step 2: Loop over NI KV blocks with online softmax
    // ================================================================
    for (int ni = 0; ni < NI; ni++) {
        // 2a: Gather this block's KV entries
        {
            const int32_t* idx_base = indices + (size_t)s_i * TOPK + ni * BI;
            for (int bi = tid; bi < BI; bi += THREADS)
                valid_smem[bi] = (idx_base[bi] >= 0) ? 1 : 0;

            constexpr int KV_TOTAL = BI * KV_PACKED_BYTES;
            for (int flat = tid * 16; flat < KV_TOTAL; flat += THREADS * 16) {
                int bi = flat / KV_PACKED_BYTES;
                int byte_off = flat % KV_PACKED_BYTES;
                int idx = idx_base[bi];
                idx = (idx >= 0) ? idx : 0;
                const uint8_t* src = KV_cache + (size_t)idx * KV_PACKED_BYTES + byte_off;
                uint8_t* dst = kv_smem + flat;
                if (byte_off + 16 <= KV_PACKED_BYTES)
                    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
            }
        }
        __syncthreads();

        // 2b: Compute scores for this block
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
                const bf16* k_rope = reinterpret_cast<const bf16*>(
                    kv_row + D_NOPE + NUM_SCALES * sizeof(float));

                float score = 0.0f;

                #pragma unroll
                for (int b = 0; b < NUM_SCALES; b++) {
                    float partial = 0.0f;
                    const uint8_t* k_fp8 = kv_row + b * QUANT_TILE;
                    const bf16* q_blk = q_row + b * QUANT_TILE;
                    #pragma unroll 4
                    for (int d = 0; d < QUANT_TILE; d += 4) {
                        uint32_t k4 = *reinterpret_cast<const uint32_t*>(k_fp8 + d);
                        partial += fp8e4m3_to_float_pf(k4 & 0xFF)         * to_float(q_blk[d])
                                 + fp8e4m3_to_float_pf((k4 >> 8) & 0xFF)  * to_float(q_blk[d+1])
                                 + fp8e4m3_to_float_pf((k4 >> 16) & 0xFF) * to_float(q_blk[d+2])
                                 + fp8e4m3_to_float_pf((k4 >> 24) & 0xFF) * to_float(q_blk[d+3]);
                    }
                    score += partial * k_scales[b];
                }

                {
                    const bf16* q_rope = q_row + D_NOPE;
                    float rope = 0.0f;
                    #pragma unroll 4
                    for (int d = 0; d < D_ROPE; d += 2)
                        rope += to_float(q_rope[d]) * to_float(k_rope[d])
                              + to_float(q_rope[d+1]) * to_float(k_rope[d+1]);
                    score += rope;
                }
                score_smem[h * BI + k] = score;
            }
        }
        __syncthreads();

        // 2c: Online softmax update + accumulate output
        // Per-head softmax update (only HPB threads needed)
        for (int h = tid; h < HPB; h += THREADS) {
            float* row = score_smem + h * BI;
            float old_m = m_smem[h];
            float old_l = l_smem[h];

            // Find new max
            float new_m = old_m;
            for (int k = 0; k < BI; k++)
                new_m = fmaxf(new_m, row[k] * sm_scale_log2e);

            // Rescale factor for old accumulators
            float alpha = fast_exp2f(old_m - new_m);

            // Compute exp2 weights and new sum
            float block_sum = 0.0f;
            for (int k = 0; k < BI; k++) {
                float w = fast_exp2f(row[k] * sm_scale_log2e - new_m);
                row[k] = w;
                block_sum += w;
            }

            m_smem[h] = new_m;
            l_smem[h] = old_l * alpha + block_sum;

            // Store alpha for output rescaling
            // Reuse m_smem temporarily? No, we need it. Use valid_smem since softmax is done.
            // Actually, let's store alpha in l_smem location... just do it inline below.
            // Store alpha * inv_new_l scaling for later: not needed, we scale acc_o directly.

            // Rescale old accumulated output
            float* o_row = acc_o_smem + h * D_NOPE;
            for (int d = 0; d < D_NOPE; d++)
                o_row[d] *= alpha;
        }
        __syncthreads();

        // Accumulate: output += weights · V_nope
        {
            const int total = HPB * D_NOPE;
            for (int elem = tid; elem < total; elem += THREADS) {
                int h = elem / D_NOPE;
                int d = elem % D_NOPE;
                int sb = d / QUANT_TILE;

                const float* weights = score_smem + h * BI;
                float acc = 0.0f;
                for (int k = 0; k < BI; k++) {
                    float w = weights[k];
                    if (w == 0.0f) continue;
                    const uint8_t* v_row = kv_smem + k * KV_PACKED_BYTES;
                    const float* v_scales = reinterpret_cast<const float*>(v_row + D_NOPE);
                    acc += w * fp8e4m3_to_float_pf(v_row[d]) * v_scales[sb];
                }
                acc_o_smem[h * D_NOPE + d] += acc;
            }
        }
        __syncthreads();
    }

    // ================================================================
    // Step 3: Finalize: normalize by l and write output
    // ================================================================
    {
        const int total = HPB * D_NOPE;
        for (int elem = tid; elem < total; elem += THREADS) {
            int h = elem / D_NOPE;
            int d = elem % D_NOPE;
            float l_val = l_smem[h];
            float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;

            size_t out_idx = (size_t)s_i * num_heads * D_NOPE
                           + (size_t)(h_start + h) * D_NOPE
                           + d;
            output[out_idx] = to_bf16(acc_o_smem[h * D_NOPE + d] * inv_l);
        }
    }
}

// -------------------------------------------------------------------
// Launch wrapper
// -------------------------------------------------------------------

void sparse_mla_prefill_launch(
    torch::Tensor Q,
    torch::Tensor KV_cache,
    torch::Tensor indices,
    torch::Tensor output,
    float sm_scale,
    int num_heads,
    int num_tokens,
    int topk,
    int BI_param)
{
    const int HPB = 16;
    const int BI = BI_param;
    const int REPLICATE_H = num_heads / HPB;
    const int THREADS = 256;

    size_t smem_bytes = HPB * DIM * sizeof(bf16)
                      + BI * KV_PACKED_BYTES
                      + HPB * BI * sizeof(float)
                      + BI * sizeof(int32_t)
                      + HPB * D_NOPE * sizeof(float)  // acc_o
                      + HPB * sizeof(float)            // m
                      + HPB * sizeof(float);           // l

    dim3 grid(num_tokens * REPLICATE_H, 1);
    dim3 block(THREADS);

    auto set_smem = [&](auto kernel_fn) {
        if (smem_bytes > 48 * 1024)
            cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    };

    if (BI == 32) {
        set_smem(sparse_mla_prefill_fused_kernel<16, 32>);
        sparse_mla_prefill_fused_kernel<16, 32><<<grid, block, smem_bytes>>>(
            reinterpret_cast<const bf16*>(Q.data_ptr()),
            reinterpret_cast<const uint8_t*>(KV_cache.data_ptr()),
            indices.data_ptr<int32_t>(),
            reinterpret_cast<bf16*>(output.data_ptr()),
            sm_scale, num_heads, num_tokens, topk);
    } else {
        set_smem(sparse_mla_prefill_fused_kernel<16, 64>);
        sparse_mla_prefill_fused_kernel<16, 64><<<grid, block, smem_bytes>>>(
            reinterpret_cast<const bf16*>(Q.data_ptr()),
            reinterpret_cast<const uint8_t*>(KV_cache.data_ptr()),
            indices.data_ptr<int32_t>(),
            reinterpret_cast<bf16*>(output.data_ptr()),
            sm_scale, num_heads, num_tokens, topk);
    }
}
