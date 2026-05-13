#include "prefill_kernel.cuh"
#include <torch/extension.h>

// ============================================================================
// Sparse MLA prefill: launch helpers and dispatch.
// SG (single-group, 16 heads/CTA) for h<=16.
// MG (multi-group, 32 heads/CTA) for h>16 — 2x KV reuse + deferred row_sum.
// ============================================================================

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
void launch_prefill_sg(
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
    bf16* output, float* out_lse,
    float sm_scale, int num_tokens,
    size_t stride_kv_block,
    cudaStream_t stream)
{
    constexpr size_t smem_bytes = SmemLayout<MT, CM>::TOTAL;
    constexpr int REPLICATE_H = NUM_HEADS / HPB;
    dim3 grid(num_tokens * REPLICATE_H);
    dim3 block(BLOCK_THREADS);

    auto kernel = sparse_mla_prefill_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE>;
    static bool configured = false;
    if (!configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        configured = true;
    }

    // SG is single-cache only; stride_kv_block_extra is unused.
    PrefillColdParams cold{sm_scale, num_tokens, stride_kv_block, /*stride_kv_block_extra=*/(size_t)0};
    cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
    void* args[] = {
        (void*)&Q, (void*)&KV_cache, (void*)&indices,
        (void*)&output, (void*)&out_lse, (void*)&cold
    };
    CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

// Dual-cache aware MG prefill launcher. When TOPK_EXTRA == 0 the kv_cache_extra
// / indices_extra pointers may be nullptr and stride_kv_block_extra is unused;
// the kernel template instantiation produces single-cache code via
// if-constexpr dead-code-elim, matching the prior behavior.
template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE, int TOPK_EXTRA = 0, int PAGE_BLOCK_SIZE_EXTRA = PAGE_BLOCK_SIZE>
void launch_prefill_mg(
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
    const uint8_t* KV_cache_extra, const int32_t* indices_extra,
    bf16* output, float* out_lse,
    float sm_scale, int num_tokens,
    size_t stride_kv_block, size_t stride_kv_block_extra,
    cudaStream_t stream)
{
    constexpr size_t smem_bytes = SmemLayoutMG<MT, CM>::TOTAL;
    constexpr int REPLICATE_H = NUM_HEADS / MG_HEADS_PER_CTA;
    dim3 grid(num_tokens * REPLICATE_H);
    dim3 block(BLOCK_THREADS);

    auto kernel = sparse_mla_prefill_mg_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE, TOPK_EXTRA, PAGE_BLOCK_SIZE_EXTRA>;
    static bool configured = false;
    if (!configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        configured = true;
    }

    PrefillColdParams cold{sm_scale, num_tokens, stride_kv_block, stride_kv_block_extra};
    cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
    void* args[] = {
        (void*)&Q, (void*)&KV_cache, (void*)&indices,
        (void*)&KV_cache_extra, (void*)&indices_extra,
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
                sm_scale, num_tokens, stride_kv_block, stream)
        switch (num_heads) {
        case 16:  DISPATCH_SG(16); break;
        default:  TORCH_CHECK(false, "V32 prefill SG: unsupported num_heads=", num_heads);
        }
        #undef DISPATCH_SG
    } else {
        #define DISPATCH_MG(NH) \
            launch_prefill_mg<ModelType::V32, ComputeMode::FP8, NH, 2048, 1>( \
                Q_ptr, KV_ptr, idx_ptr, \
                /*KV_cache_extra=*/nullptr, /*indices_extra=*/nullptr, \
                O_ptr, LSE_ptr, \
                sm_scale, num_tokens, \
                stride_kv_block, /*stride_kv_block_extra=*/(size_t)0, \
                stream)
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
    cudaStream_t stream)
{
    auto Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr());
    auto KV_ptr = reinterpret_cast<const uint8_t*>(KV_cache.data_ptr());
    auto idx_ptr = indices.data_ptr<int32_t>();
    auto O_ptr = reinterpret_cast<bf16*>(output.data_ptr());
    auto LSE_ptr = out_lse.data_ptr<float>();
    size_t stride_kv_block = (size_t)page_block_size * stride_kv_row;

    TORCH_CHECK(page_block_size == 64, "MODEL1 prefill: page_block_size must be 64, got ", page_block_size);

    // MODEL1 always uses MG (h=64 or h=128, both > HPB)
    #define DISPATCH_MG(NH, TK) \
        launch_prefill_mg<ModelType::MODEL1, ComputeMode::FP8, NH, TK, 64>( \
            Q_ptr, KV_ptr, idx_ptr, \
            /*KV_cache_extra=*/nullptr, /*indices_extra=*/nullptr, \
            O_ptr, LSE_ptr, \
            sm_scale, num_tokens, \
            stride_kv_block, /*stride_kv_block_extra=*/(size_t)0, \
            stream)

    if (topk == 128) {
        switch (num_heads) {
        case 64:  DISPATCH_MG(64, 128); break;
        case 128: DISPATCH_MG(128, 128); break;
        default:  TORCH_CHECK(false, "MODEL1 prefill: unsupported num_heads=", num_heads);
        }
    } else if (topk == 512) {
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

// Dual-cache MODEL1 prefill entry point. Hardcoded to
// (topk_main=128, topk_extra=128) for the DSv4-sm120 case; add more
// (topk_main, topk_extra) combinations here as needed.
void sparse_mla_prefill_launch_model1_dual(
    torch::Tensor Q,
    torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor KV_cache_extra, torch::Tensor indices_extra,
    torch::Tensor output, torch::Tensor out_lse,
    float sm_scale, int num_heads, int num_tokens,
    int topk, int topk_extra,
    int page_block_size, int stride_kv_row,
    int page_block_size_extra, int stride_kv_row_extra,
    cudaStream_t stream)
{
    auto Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr());
    auto KV_ptr = reinterpret_cast<const uint8_t*>(KV_cache.data_ptr());
    auto idx_ptr = indices.data_ptr<int32_t>();
    auto KV_extra_ptr = reinterpret_cast<const uint8_t*>(KV_cache_extra.data_ptr());
    auto idx_extra_ptr = indices_extra.data_ptr<int32_t>();
    auto O_ptr = reinterpret_cast<bf16*>(output.data_ptr());
    auto LSE_ptr = out_lse.data_ptr<float>();
    size_t stride_kv_block = (size_t)page_block_size * stride_kv_row;
    size_t stride_kv_block_extra = (size_t)page_block_size_extra * stride_kv_row_extra;

    TORCH_CHECK(page_block_size == 64,
        "MODEL1 dual prefill: main page_block_size must be 64, got ", page_block_size);
    // DSv4 compressed cache: page_block_size_extra = main_block_size / compress_ratio.
    TORCH_CHECK(page_block_size_extra == 64 || page_block_size_extra == 2,
        "MODEL1 dual prefill: extra page_block_size must be 64 or 2, got ",
        page_block_size_extra);

    #define DISPATCH_DUAL_MG(NH, TK, TK_EX, PBSX) \
        launch_prefill_mg<ModelType::MODEL1, ComputeMode::FP8, NH, TK, 64, TK_EX, PBSX>( \
            Q_ptr, KV_ptr, idx_ptr, KV_extra_ptr, idx_extra_ptr, \
            O_ptr, LSE_ptr, sm_scale, num_tokens, \
            stride_kv_block, stride_kv_block_extra, stream)

    if (topk == 128 && topk_extra == 128 && page_block_size_extra == 64) {
        switch (num_heads) {
        case 64:  DISPATCH_DUAL_MG(64, 128, 128, 64); break;
        case 128: DISPATCH_DUAL_MG(128, 128, 128, 64); break;
        default:
            TORCH_CHECK(false, "MODEL1 dual prefill: unsupported num_heads=", num_heads);
        }
    } else if (topk == 128 && topk_extra == 512 && page_block_size_extra == 64) {
        // DSv4-Flash C4A: SWA window=128, indexer top_k=512, compress_ratio=4.
        switch (num_heads) {
        case 64:  DISPATCH_DUAL_MG(64, 128, 512, 64); break;
        case 128: DISPATCH_DUAL_MG(128, 128, 512, 64); break;
        default:
            TORCH_CHECK(false, "MODEL1 dual prefill: unsupported num_heads=", num_heads);
        }
    } else if (topk == 128 && topk_extra == 512 && page_block_size_extra == 2) {
        // DSv4-Flash C128A: SWA window=128, indexer top_k=512, compress_ratio=128.
        switch (num_heads) {
        case 64:  DISPATCH_DUAL_MG(64, 128, 512, 2); break;
        case 128: DISPATCH_DUAL_MG(128, 128, 512, 2); break;
        default:
            TORCH_CHECK(false, "MODEL1 dual prefill: unsupported num_heads=", num_heads);
        }
    } else {
        TORCH_CHECK(false, "MODEL1 dual prefill: unsupported "
                    "(topk, topk_extra, page_block_size_extra)=(",
                    topk, ", ", topk_extra, ", ", page_block_size_extra,
                    "); supported: (128,128,64), (128,512,64), (128,512,2)");
    }
    #undef DISPATCH_DUAL_MG
}
