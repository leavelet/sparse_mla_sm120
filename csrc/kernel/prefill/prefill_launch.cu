#include "prefill_kernel.cuh"
#include <torch/extension.h>

// ============================================================================
// Sparse MLA prefill: launch helpers and dispatch.
// SG (single-group, 16 heads/CTA) for h<=16.
// MG (multi-group, 32 heads/CTA) for h>16 — 2x KV reuse + deferred row_sum.
// ============================================================================

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE,
          bool BF16_QK = KVCacheTraits<MT>::USE_BF16_QK>
void launch_prefill_sg(
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
    bf16* output, float* out_lse,
    float sm_scale, int num_tokens,
    size_t stride_kv_block,
    const float* attn_sink,
    const int* topk_length_ptr,
    cudaStream_t stream)
{
    constexpr size_t smem_bytes = SmemLayout<MT, CM, BF16_QK>::TOTAL;
    constexpr int REPLICATE_H = NUM_HEADS / HPB;
    dim3 grid(num_tokens * REPLICATE_H);
    dim3 block(BLOCK_THREADS);

    auto kernel = sparse_mla_prefill_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE, BF16_QK>;
    static bool configured = false;
    if (!configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        configured = true;
    }

    PrefillColdParams cold{sm_scale, num_tokens, stride_kv_block, attn_sink, topk_length_ptr};
    cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
    void* args[] = {
        (void*)&Q, (void*)&KV_cache, (void*)&indices,
        (void*)&output, (void*)&out_lse, (void*)&cold
    };
    CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE,
          bool BF16_QK = KVCacheTraits<MT>::USE_BF16_QK>
void launch_prefill_mg(
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
    bf16* output, float* out_lse,
    float sm_scale, int num_tokens,
    size_t stride_kv_block,
    const float* attn_sink,
    const int* topk_length_ptr,
    cudaStream_t stream)
{
    constexpr size_t smem_bytes = SmemLayoutMG<MT, CM, BF16_QK>::TOTAL;
    constexpr int REPLICATE_H = NUM_HEADS / MG_HEADS_PER_CTA;
    dim3 grid(num_tokens * REPLICATE_H);
    dim3 block(BLOCK_THREADS);

    auto kernel = sparse_mla_prefill_mg_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE, BF16_QK>;
    static bool configured = false;
    if (!configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        configured = true;
    }

    PrefillColdParams cold{sm_scale, num_tokens, stride_kv_block, attn_sink, topk_length_ptr};
    cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
    void* args[] = {
        (void*)&Q, (void*)&KV_cache, (void*)&indices,
        (void*)&output, (void*)&out_lse, (void*)&cold
    };
    CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

// ============================================================================
// External entry points — dispatch SG (h<=16) vs MG (h>16)
// ============================================================================

void sparse_mla_prefill_launch_v32(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output, torch::Tensor out_lse,
    float sm_scale, int num_heads, int num_tokens, int topk,
    int page_block_size, int stride_kv_row,
    const float* attn_sink_ptr,
    const int* topk_length_ptr,
    cudaStream_t stream)
{
    auto Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr());
    auto KV_ptr = reinterpret_cast<const uint8_t*>(KV_cache.data_ptr());
    auto idx_ptr = indices.data_ptr<int32_t>();
    auto O_ptr = reinterpret_cast<bf16*>(output.data_ptr());
    auto LSE_ptr = out_lse.data_ptr<float>();
    size_t stride_kv_block = (size_t)page_block_size * stride_kv_row;

    TORCH_CHECK(topk == 2048, "V32 prefill requires topk=2048, got ", topk);

    if (num_heads <= HPB) {
        #define DISPATCH_SG(NH) \
            launch_prefill_sg<ModelType::V32, ComputeMode::FP8, NH, 2048, 1>( \
                Q_ptr, KV_ptr, idx_ptr, O_ptr, LSE_ptr, \
                sm_scale, num_tokens, stride_kv_block, attn_sink_ptr, topk_length_ptr, stream)
        switch (num_heads) {
        case 16:  DISPATCH_SG(16); break;
        default:  TORCH_CHECK(false, "V32 prefill SG: unsupported num_heads=", num_heads);
        }
        #undef DISPATCH_SG
    } else {
        #define DISPATCH_MG(NH) \
            launch_prefill_mg<ModelType::V32, ComputeMode::FP8, NH, 2048, 1>( \
                Q_ptr, KV_ptr, idx_ptr, O_ptr, LSE_ptr, \
                sm_scale, num_tokens, stride_kv_block, attn_sink_ptr, topk_length_ptr, stream)
        switch (num_heads) {
        case 64:  DISPATCH_MG(64); break;
        case 128: DISPATCH_MG(128); break;
        default:  TORCH_CHECK(false, "V32 prefill MG: unsupported num_heads=", num_heads);
        }
        #undef DISPATCH_MG
    }
}

void sparse_mla_prefill_launch_model1(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output, torch::Tensor out_lse,
    float sm_scale, int num_heads, int num_tokens, int topk,
    int page_block_size, int stride_kv_row,
    const float* attn_sink_ptr,
    const int* topk_length_ptr,
    bool bf16_qk,
    cudaStream_t stream)
{
    auto Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr());
    auto KV_ptr = reinterpret_cast<const uint8_t*>(KV_cache.data_ptr());
    auto idx_ptr = indices.data_ptr<int32_t>();
    auto O_ptr = reinterpret_cast<bf16*>(output.data_ptr());
    auto LSE_ptr = out_lse.data_ptr<float>();
    size_t stride_kv_block = (size_t)page_block_size * stride_kv_row;

    TORCH_CHECK(page_block_size == 64, "MODEL1 prefill: page_block_size must be 64, got ", page_block_size);

    #define DISPATCH_MG(NH, TK) \
        if (bf16_qk) { \
            launch_prefill_mg<ModelType::MODEL1, ComputeMode::FP8, NH, TK, 64, true>( \
                Q_ptr, KV_ptr, idx_ptr, O_ptr, LSE_ptr, \
                sm_scale, num_tokens, stride_kv_block, attn_sink_ptr, topk_length_ptr, stream); \
        } else { \
            launch_prefill_mg<ModelType::MODEL1, ComputeMode::FP8, NH, TK, 64, false>( \
                Q_ptr, KV_ptr, idx_ptr, O_ptr, LSE_ptr, \
                sm_scale, num_tokens, stride_kv_block, attn_sink_ptr, topk_length_ptr, stream); \
        }

    if (topk == 512) {
        switch (num_heads) {
        case 64:  DISPATCH_MG(64, 512); break;
        case 128: DISPATCH_MG(128, 512); break;
        default:  TORCH_CHECK(false, "MODEL1 prefill: unsupported num_heads=", num_heads);
        }
    } else if (topk == 1024) {
        switch (num_heads) {
        case 64:  DISPATCH_MG(64, 1024); break;
        case 128: DISPATCH_MG(128, 1024); break;
        default:  TORCH_CHECK(false, "MODEL1 prefill: unsupported num_heads=", num_heads);
        }
    } else {
        TORCH_CHECK(false, "MODEL1 prefill: unsupported topk=", topk);
    }
    #undef DISPATCH_MG
}
