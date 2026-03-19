#include "common.cuh"

#include <torch/extension.h>

// ============================================================================
// Combine kernel: merge NSPLITS partial outputs via log-sum-exp rescaling
//
// partial_O:   [num_tokens, nsplits, num_heads, D_NOPE]  bf16
// partial_LSE: [num_tokens, nsplits, num_heads]           fp32
// output:      [num_tokens, num_heads, D_NOPE]            bf16
//
// Grid: (num_tokens * num_heads / COMBINE_HPB)
// Block: THREADS
// Each CTA handles COMBINE_HPB heads across all splits.
// ============================================================================

static constexpr int COMBINE_HPB = 4;
static constexpr int COMBINE_THREADS = 128;

__global__ void __launch_bounds__(COMBINE_THREADS)
sparse_mla_combine_kernel(
    const bf16* __restrict__ partial_O,
    const float* __restrict__ partial_LSE,
    bf16* __restrict__ output,
    int num_heads,
    int num_tokens,
    int nsplits)
{
    const int head_tiles = num_heads / COMBINE_HPB;
    const int s_i = blockIdx.x / head_tiles;
    const int h_tile = blockIdx.x % head_tiles;
    const int h_start = h_tile * COMBINE_HPB;
    if (s_i >= num_tokens) return;

    const int tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    float* lse_smem = reinterpret_cast<float*>(smem_raw);  // [nsplits * COMBINE_HPB]

    // Load LSE values for this token's head tile
    for (int i = tid; i < nsplits * COMBINE_HPB; i += COMBINE_THREADS) {
        int ni = i / COMBINE_HPB;
        int h_local = i % COMBINE_HPB;
        size_t idx = (size_t)s_i * nsplits * num_heads
                   + (size_t)ni * num_heads
                   + (h_start + h_local);
        lse_smem[ni * COMBINE_HPB + h_local] = partial_LSE[idx];
    }
    __syncthreads();

    const int total_out = COMBINE_HPB * D_NOPE;
    for (int elem = tid; elem < total_out; elem += COMBINE_THREADS) {
        int h_local = elem / D_NOPE;
        int d = elem % D_NOPE;

        float lse_max = -1e30f;
        for (int ni = 0; ni < nsplits; ni++)
            lse_max = fmaxf(lse_max, lse_smem[ni * COMBINE_HPB + h_local]);

        float acc = 0.0f;
        float lse_sum = 0.0f;
        for (int ni = 0; ni < nsplits; ni++) {
            float lse_val = lse_smem[ni * COMBINE_HPB + h_local];
            float scale = fast_exp2f(lse_val - lse_max);
            lse_sum += scale;

            size_t po_idx = (size_t)s_i * nsplits * num_heads * D_NOPE
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

// ── Launch wrapper ──────────────────────────────────────────────────────

void sparse_mla_combine_launch(
    torch::Tensor partial_O,
    torch::Tensor partial_LSE,
    torch::Tensor output,
    int num_heads,
    int num_tokens,
    int nsplits)
{
    const int head_tiles = num_heads / COMBINE_HPB;
    const int THREADS = COMBINE_THREADS;

    size_t smem_bytes = nsplits * COMBINE_HPB * sizeof(float);

    dim3 grid(num_tokens * head_tiles);
    dim3 block(THREADS);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            sparse_mla_combine_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
    }

    sparse_mla_combine_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const bf16*>(partial_O.data_ptr()),
        partial_LSE.data_ptr<float>(),
        reinterpret_cast<bf16*>(output.data_ptr()),
        num_heads, num_tokens, nsplits);
}
