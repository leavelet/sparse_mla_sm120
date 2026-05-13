#include "prefill_kernel.cuh"
#include <torch/extension.h>

// ============================================================================
// Sparse MLA prefill: launch helpers and dispatch.
// SG (single-group, 16 heads/CTA) for h<=16.
// MG (multi-group, 32 heads/CTA) for h>16.
// Supports dual-cache (extra_k_cache) for V4 SWA + compressed.
// ============================================================================

template <ModelType MT, ComputeMode CM, int NUM_HEADS,
          bool BF16_QK = KVCacheTraits<MT>::USE_BF16_QK>
void launch_prefill_sg(
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
    const uint8_t* extra_KV_cache, const int32_t* extra_indices,
    bf16* output, float* out_lse, float* out_max_logits,
    float sm_scale, int num_tokens, int topk, int page_block_size,
    size_t stride_kv_block,
    const float* attn_sink,
    const int* topk_length_ptr,
    int extra_topk, size_t stride_extra_kv_block, int extra_page_block_size,
    cudaStream_t stream)
{
    constexpr size_t smem_bytes = SmemLayout<MT, CM, BF16_QK>::TOTAL;
    constexpr int REPLICATE_H = NUM_HEADS / HPB;
    dim3 grid(num_tokens * REPLICATE_H);
    dim3 block(BLOCK_THREADS);

    auto kernel = sparse_mla_prefill_kernel<MT, CM, NUM_HEADS, BF16_QK>;
    static bool configured = false;
    if (!configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        configured = true;
    }

    PrefillColdParams cold{sm_scale, num_tokens, topk, stride_kv_block, page_block_size,
                           attn_sink, topk_length_ptr,
                           extra_topk, stride_extra_kv_block, extra_page_block_size};
    cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
    void* args[] = {
        (void*)&Q, (void*)&KV_cache, (void*)&indices,
        (void*)&extra_KV_cache, (void*)&extra_indices,
        (void*)&output, (void*)&out_lse, (void*)&out_max_logits, (void*)&cold
    };
    CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

template <ModelType MT, ComputeMode CM, int NUM_HEADS,
          bool BF16_QK = KVCacheTraits<MT>::USE_BF16_QK>
void launch_prefill_mg(
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
    const uint8_t* extra_KV_cache, const int32_t* extra_indices,
    bf16* output, float* out_lse, float* out_max_logits,
    float sm_scale, int num_tokens, int topk, int page_block_size,
    size_t stride_kv_block,
    const float* attn_sink,
    const int* topk_length_ptr,
    int extra_topk, size_t stride_extra_kv_block, int extra_page_block_size,
    cudaStream_t stream)
{
    constexpr size_t smem_bytes = SmemLayoutMG<MT, CM, BF16_QK>::TOTAL;
    constexpr int REPLICATE_H = NUM_HEADS / MG_HEADS_PER_CTA;
    dim3 grid(num_tokens * REPLICATE_H);
    dim3 block(BLOCK_THREADS);

    auto kernel = sparse_mla_prefill_mg_kernel<MT, CM, NUM_HEADS, BF16_QK>;
    static bool configured = false;
    if (!configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        configured = true;
    }

    PrefillColdParams cold{sm_scale, num_tokens, topk, stride_kv_block, page_block_size,
                           attn_sink, topk_length_ptr,
                           extra_topk, stride_extra_kv_block, extra_page_block_size};
    cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
    void* args[] = {
        (void*)&Q, (void*)&KV_cache, (void*)&indices,
        (void*)&extra_KV_cache, (void*)&extra_indices,
        (void*)&output, (void*)&out_lse, (void*)&out_max_logits, (void*)&cold
    };
    CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

// ============================================================================
// External entry points
// ============================================================================

void sparse_mla_prefill_launch_v32(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output, torch::Tensor out_lse,
    float sm_scale, int num_heads, int num_tokens, int topk,
    int page_block_size, int stride_kv_row,
    const float* attn_sink_ptr,
    const int* topk_length_ptr,
    float* out_max_logits_ptr,
    const uint8_t* extra_kv, const int32_t* extra_idx,
    int extra_topk, int extra_page_block_size, int extra_stride_kv_row,
    cudaStream_t stream)
{
    auto Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr());
    auto KV_ptr = reinterpret_cast<const uint8_t*>(KV_cache.data_ptr());
    auto idx_ptr = indices.data_ptr<int32_t>();
    auto O_ptr = reinterpret_cast<bf16*>(output.data_ptr());
    auto LSE_ptr = out_lse.data_ptr<float>();
    size_t stride_kv_block = (size_t)page_block_size * stride_kv_row;
    size_t stride_extra = (extra_kv != nullptr)
        ? (size_t)extra_page_block_size * extra_stride_kv_row : 0;

    if (num_heads <= HPB) {
        #define DISPATCH_SG(NH) \
            launch_prefill_sg<ModelType::V32, ComputeMode::FP8, NH>( \
                Q_ptr, KV_ptr, idx_ptr, extra_kv, extra_idx, \
                O_ptr, LSE_ptr, out_max_logits_ptr, \
                sm_scale, num_tokens, topk, page_block_size, stride_kv_block, \
                attn_sink_ptr, topk_length_ptr, \
                extra_topk, stride_extra, extra_page_block_size, stream)
        switch (num_heads) {
        case 16:  DISPATCH_SG(16); break;
        default:  TORCH_CHECK(false, "V32 prefill SG: unsupported num_heads=", num_heads);
        }
        #undef DISPATCH_SG
    } else {
        #define DISPATCH_MG(NH) \
            launch_prefill_mg<ModelType::V32, ComputeMode::FP8, NH>( \
                Q_ptr, KV_ptr, idx_ptr, extra_kv, extra_idx, \
                O_ptr, LSE_ptr, out_max_logits_ptr, \
                sm_scale, num_tokens, topk, page_block_size, stride_kv_block, \
                attn_sink_ptr, topk_length_ptr, \
                extra_topk, stride_extra, extra_page_block_size, stream)
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
    float* out_max_logits_ptr,
    const uint8_t* extra_kv, const int32_t* extra_idx,
    int extra_topk, int extra_page_block_size, int extra_stride_kv_row,
    cudaStream_t stream)
{
    auto Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr());
    auto KV_ptr = reinterpret_cast<const uint8_t*>(KV_cache.data_ptr());
    auto idx_ptr = indices.data_ptr<int32_t>();
    auto O_ptr = reinterpret_cast<bf16*>(output.data_ptr());
    auto LSE_ptr = out_lse.data_ptr<float>();
    size_t stride_kv_block = (size_t)page_block_size * stride_kv_row;
    size_t stride_extra = (extra_kv != nullptr)
        ? (size_t)extra_page_block_size * extra_stride_kv_row : 0;

    #define DISPATCH_MG(NH) \
        if (bf16_qk) { \
            launch_prefill_mg<ModelType::MODEL1, ComputeMode::FP8, NH, true>( \
                Q_ptr, KV_ptr, idx_ptr, extra_kv, extra_idx, \
                O_ptr, LSE_ptr, out_max_logits_ptr, \
                sm_scale, num_tokens, topk, page_block_size, stride_kv_block, \
                attn_sink_ptr, topk_length_ptr, \
                extra_topk, stride_extra, extra_page_block_size, stream); \
        } else { \
            launch_prefill_mg<ModelType::MODEL1, ComputeMode::FP8, NH, false>( \
                Q_ptr, KV_ptr, idx_ptr, extra_kv, extra_idx, \
                O_ptr, LSE_ptr, out_max_logits_ptr, \
                sm_scale, num_tokens, topk, page_block_size, stride_kv_block, \
                attn_sink_ptr, topk_length_ptr, \
                extra_topk, stride_extra, extra_page_block_size, stream); \
        }

    switch (num_heads) {
    case 64:  DISPATCH_MG(64); break;
    case 128: DISPATCH_MG(128); break;
    default:  TORCH_CHECK(false, "MODEL1 prefill: unsupported num_heads=", num_heads);
    }
    #undef DISPATCH_MG
}
