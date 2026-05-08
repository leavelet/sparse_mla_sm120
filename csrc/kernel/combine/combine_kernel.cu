#include "../../arch/common.cuh"
#include "../../model/kv_cache_traits.cuh"
#include <torch/extension.h>

// ============================================================================
// Sparse MLA Combine Kernel
//
// Merges partial outputs from split-KV decode into final output.
// Separate kernel (FlashMLA compatible), launched after splitkv decode.
//
// Input:
//   partial_O:   [num_tokens, nsplits, num_heads, d_v] float32
//   partial_LSE: [num_tokens, nsplits, num_heads]      float32
//
// Output:
//   output: [num_tokens, num_heads, d_v] bfloat16
//   lse:    [num_tokens, num_heads]      float32
//
// Grid: (num_tokens, num_heads / BLOCK_H)
// Each thread block processes BLOCK_H heads for one token.
// ============================================================================

static constexpr int COMBINE_BLOCK_H = 8;
static constexpr int COMBINE_THREADS = 256;

__global__ void __launch_bounds__(COMBINE_THREADS)
sparse_mla_combine_kernel(
    const float* __restrict__ partial_O,
    const float* __restrict__ partial_LSE,
    bf16* __restrict__ output,
    float* __restrict__ out_lse,
    int num_heads,
    int nsplits,
    int d_v)
{
    const int token_idx = blockIdx.x;
    const int h_block = blockIdx.y;
    const int h_start = h_block * COMBINE_BLOCK_H;
    const int tid = threadIdx.x;

    // Each thread handles a subset of (head, dim) within the BLOCK_H heads
    // Total work = BLOCK_H * d_v = 8 * 512 = 4096 elements
    // With 256 threads: 16 elements per thread

    constexpr int ELEMS_PER_THREAD = COMBINE_BLOCK_H * D_V / COMBINE_THREADS;  // 16

    // Step 1: Find max LSE across all splits for each head in this block
    // Use shared memory for LSE values
    extern __shared__ char smem[];
    float* lse_smem = reinterpret_cast<float*>(smem);  // [nsplits, BLOCK_H]

    // Load all LSE values cooperatively
    for (int i = tid; i < nsplits * COMBINE_BLOCK_H; i += COMBINE_THREADS) {
        int sp = i / COMBINE_BLOCK_H;
        int h_local = i % COMBINE_BLOCK_H;
        int h = h_start + h_local;
        if (h < num_heads) {
            size_t lse_idx = (size_t)token_idx * nsplits * num_heads
                           + (size_t)sp * num_heads + h;
            lse_smem[sp * COMBINE_BLOCK_H + h_local] = partial_LSE[lse_idx];
        } else {
            lse_smem[sp * COMBINE_BLOCK_H + h_local] = -1e30f;
        }
    }
    __syncthreads();

    // Each thread processes its assigned (head, dim) pairs
    for (int elem = tid; elem < COMBINE_BLOCK_H * D_V; elem += COMBINE_THREADS) {
        int h_local = elem / D_V;
        int d = elem % D_V;
        int h = h_start + h_local;

        if (h >= num_heads) continue;

        // Find max LSE for this head
        float max_lse = -1e30f;
        for (int sp = 0; sp < nsplits; sp++)
            max_lse = fmaxf(max_lse, lse_smem[sp * COMBINE_BLOCK_H + h_local]);

        // Weighted sum
        float acc = 0.f;
        float scale_sum = 0.f;
        for (int sp = 0; sp < nsplits; sp++) {
            float lse_val = lse_smem[sp * COMBINE_BLOCK_H + h_local];
            float scale = exp2f(lse_val - max_lse);
            scale_sum += scale;

            size_t po_idx = (size_t)token_idx * nsplits * num_heads * D_V
                          + (size_t)sp * num_heads * D_V
                          + (size_t)h * D_V + d;
            acc += scale * partial_O[po_idx];
        }

        // Normalize and write
        float inv_scale = (scale_sum > 0.f) ? (1.f / scale_sum) : 0.f;
        size_t out_idx = (size_t)token_idx * num_heads * D_V + (size_t)h * D_V + d;
        output[out_idx] = __float2bfloat16(acc * inv_scale);

        // Write LSE (only once per head, use d==0)
        if (d == 0) {
            float final_lse = (scale_sum > 0.f) ? (log2f(scale_sum) + max_lse) : -1e30f;
            size_t lse_out_idx = (size_t)token_idx * num_heads + h;
            out_lse[lse_out_idx] = final_lse;
        }
    }
}

// Launch wrapper
void sparse_mla_combine_launch(
    torch::Tensor partial_O,
    torch::Tensor partial_LSE,
    torch::Tensor output,
    torch::Tensor out_lse,
    int nsplits,
    cudaStream_t stream)
{
    int num_tokens = partial_O.size(0);
    int num_heads = partial_O.size(2);
    int d_v = partial_O.size(3);

    dim3 grid(num_tokens, (num_heads + COMBINE_BLOCK_H - 1) / COMBINE_BLOCK_H);
    dim3 block(COMBINE_THREADS);
    int smem_bytes = nsplits * COMBINE_BLOCK_H * sizeof(float);

    sparse_mla_combine_kernel<<<grid, block, smem_bytes, stream>>>(
        partial_O.data_ptr<float>(),
        partial_LSE.data_ptr<float>(),
        reinterpret_cast<bf16*>(output.data_ptr()),
        out_lse.data_ptr<float>(),
        num_heads, nsplits, d_v);
}
