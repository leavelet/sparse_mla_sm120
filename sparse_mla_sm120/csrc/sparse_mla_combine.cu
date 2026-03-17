#include "common.cuh"

#include <torch/extension.h>

// -------------------------------------------------------------------
// Combine kernel: merge NI partial outputs via log-sum-exp rescaling
// Grid: (num_tokens * REPLICATE_H)
// Block: THREADS
//
// partial_O:   [num_tokens, NI, num_heads, D_NOPE]  bf16
// partial_LSE: [num_tokens, NI, num_heads]           fp32
// output:      [num_tokens, num_heads, D_NOPE]       bf16
// -------------------------------------------------------------------

template <int HPB>
__global__ void sparse_mla_combine_kernel(
    const bf16* __restrict__ partial_O,
    const float* __restrict__ partial_LSE,
    bf16* __restrict__ output,
    int num_heads,
    int num_tokens,
    int NI)
{
    const int THREADS = blockDim.x;
    const int tid = threadIdx.x;

    const int REPLICATE_H = num_heads / HPB;
    const int s_i = blockIdx.x / REPLICATE_H;
    const int h_tile = blockIdx.x % REPLICATE_H;
    const int h_start = h_tile * HPB;
    if (s_i >= num_tokens) return;

    // Shared memory for LSE values: [NI, HPB]
    extern __shared__ char smem_raw[];
    float* lse_smem = reinterpret_cast<float*>(smem_raw);  // [NI, HPB]

    // Load all partial LSE values for this token + head tile
    for (int i = tid; i < NI * HPB; i += THREADS) {
        int ni = i / HPB;
        int h_local = i % HPB;
        size_t idx = (size_t)s_i * NI * num_heads
                   + (size_t)ni * num_heads
                   + (h_start + h_local);
        lse_smem[ni * HPB + h_local] = partial_LSE[idx];
    }
    __syncthreads();

    // Each thread handles some (h, d) output elements
    const int total_out = HPB * D_NOPE;
    for (int elem = tid; elem < total_out; elem += THREADS) {
        int h_local = elem / D_NOPE;
        int d = elem % D_NOPE;

        // Find max LSE across splits for this head
        float lse_max = -1e30f;
        for (int ni = 0; ni < NI; ni++) {
            lse_max = fmaxf(lse_max, lse_smem[ni * HPB + h_local]);
        }

        // Compute rescaled weighted sum
        float acc = 0.0f;
        float lse_sum = 0.0f;
        for (int ni = 0; ni < NI; ni++) {
            float lse_val = lse_smem[ni * HPB + h_local];
            float scale = fast_exp2f(lse_val - lse_max);
            lse_sum += scale;

            size_t po_idx = (size_t)s_i * NI * num_heads * D_NOPE
                          + (size_t)ni * num_heads * D_NOPE
                          + (size_t)(h_start + h_local) * D_NOPE
                          + d;
            acc += scale * to_float(partial_O[po_idx]);
        }

        float inv_sum = (lse_sum > 0.0f) ? (1.0f / lse_sum) : 0.0f;
        acc *= inv_sum;

        size_t out_idx = (size_t)s_i * num_heads * D_NOPE
                       + (size_t)(h_start + h_local) * D_NOPE
                       + d;
        output[out_idx] = to_bf16(acc);
    }
}

// -------------------------------------------------------------------
// Launch wrapper
// -------------------------------------------------------------------

void sparse_mla_combine_launch(
    torch::Tensor partial_O,
    torch::Tensor partial_LSE,
    torch::Tensor output,
    int num_heads,
    int num_tokens,
    int topk,
    int BI)
{
    const int HPB = 16;
    const int NI = topk / BI;
    const int REPLICATE_H = num_heads / HPB;
    const int THREADS = 256;

    size_t smem_bytes = NI * HPB * sizeof(float);

    dim3 grid(num_tokens * REPLICATE_H);
    dim3 block(THREADS);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            sparse_mla_combine_kernel<16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
    }

    sparse_mla_combine_kernel<16><<<grid, block, smem_bytes>>>(
        reinterpret_cast<const bf16*>(partial_O.data_ptr()),
        partial_LSE.data_ptr<float>(),
        reinterpret_cast<bf16*>(output.data_ptr()),
        num_heads, num_tokens, NI);
}
