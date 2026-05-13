#include "decode_kernel.cuh"
#include "decode_v2_kernel.cuh"
#include <torch/extension.h>

// ============================================================================
// Split-KV decode: launch helpers and dispatch.
// ============================================================================

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int TILES_PER_SPLIT, int PAGE_BLOCK_SIZE,
          bool BF16_QK = KVCacheTraits<MT>::USE_BF16_QK>
void launch_decode(
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
    float* partial_O, float* partial_LSE,
    float sm_scale, int num_tokens,
    size_t stride_kv_block,
    cudaStream_t stream)
{
    constexpr size_t smem_bytes = SmemLayout<MT, CM, BF16_QK>::TOTAL;
    constexpr int REPLICATE_H = (NUM_HEADS + HPB - 1) / HPB;
    constexpr int NI = TOPK / BI;
    constexpr int NSPLITS = NI / TILES_PER_SPLIT;
    dim3 grid(num_tokens * REPLICATE_H, NSPLITS);
    dim3 block(BLOCK_THREADS);

    auto kernel = sparse_mla_decode_kernel<MT, CM, NUM_HEADS, TOPK, TILES_PER_SPLIT, PAGE_BLOCK_SIZE, BF16_QK>;
    static bool configured = false;
    if (!configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        configured = true;
    }

    DecodeColdParams cold{sm_scale, num_tokens, stride_kv_block};

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{grid, block, smem_bytes, stream, attrs, 1};
    void* args[] = {
        (void*)&Q, (void*)&KV_cache, (void*)&indices,
        (void*)&partial_O, (void*)&partial_LSE, (void*)&cold
    };
    CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE,
          bool BF16_QK = KVCacheTraits<MT>::USE_BF16_QK>
void dispatch_tiles(
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
    float* partial_O, float* partial_LSE,
    float sm_scale, int num_tokens, int tiles_per_split,
    size_t stride_kv_block,
    cudaStream_t stream)
{
    constexpr int NI = TOPK / BI;

    #define CASE(TPS) \
        case TPS: \
            if constexpr (NI >= TPS && NI % TPS == 0) { \
                launch_decode<MT, CM, NUM_HEADS, TOPK, TPS, PAGE_BLOCK_SIZE, BF16_QK>( \
                    Q, KV_cache, indices, partial_O, partial_LSE, \
                    sm_scale, num_tokens, stride_kv_block, stream); \
            } else { \
                TORCH_CHECK(false, "Invalid tiles_per_split=", TPS, " for TOPK=", TOPK); \
            } \
            break;

    switch (tiles_per_split) {
        CASE(2)
        CASE(4)
        CASE(8)
        CASE(16)
        CASE(32)
        default: TORCH_CHECK(false, "Unsupported tiles_per_split=", tiles_per_split);
    }
    #undef CASE
}

// ============================================================================
// External entry points
// ============================================================================

void sparse_mla_splitkv_launch_v32(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor partial_O, torch::Tensor partial_LSE,
    float sm_scale, int num_heads, int num_tokens, int topk,
    int tiles_per_split, int page_block_size, int stride_kv_row,
    cudaStream_t stream)
{
    auto Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr());
    auto KV_ptr = reinterpret_cast<const uint8_t*>(KV_cache.data_ptr());
    auto idx_ptr = indices.data_ptr<int32_t>();
    auto PO_ptr = partial_O.data_ptr<float>();
    auto LSE_ptr = partial_LSE.data_ptr<float>();
    size_t stride_kv_block = (size_t)page_block_size * stride_kv_row;

    TORCH_CHECK(topk == 2048, "V32 decode requires topk=2048, got ", topk);

    // V32: page_block_size=1 (flat addressing), but template it anyway for consistency
    #define DISPATCH(NH) \
        dispatch_tiles<ModelType::V32, ComputeMode::FP8, NH, 2048, 1>( \
            Q_ptr, KV_ptr, idx_ptr, PO_ptr, LSE_ptr, \
            sm_scale, num_tokens, tiles_per_split, \
            stride_kv_block, stream)

    switch (num_heads) {
    case 8:   DISPATCH(8); break;
    case 16:  DISPATCH(16); break;
    case 64:  DISPATCH(64); break;
    case 128: DISPATCH(128); break;
    default:  TORCH_CHECK(false, "V32 decode: unsupported num_heads=", num_heads);
    }
    #undef DISPATCH
}

void sparse_mla_splitkv_launch_model1(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor partial_O, torch::Tensor partial_LSE,
    float sm_scale, int num_heads, int num_tokens, int topk,
    int tiles_per_split, int page_block_size, int stride_kv_row,
    bool bf16_qk,
    cudaStream_t stream)
{
    auto Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr());
    auto KV_ptr = reinterpret_cast<const uint8_t*>(KV_cache.data_ptr());
    auto idx_ptr = indices.data_ptr<int32_t>();
    auto PO_ptr = partial_O.data_ptr<float>();
    auto LSE_ptr = partial_LSE.data_ptr<float>();
    size_t stride_kv_block = (size_t)page_block_size * stride_kv_row;

    TORCH_CHECK(page_block_size == 64, "MODEL1 decode: page_block_size must be 64, got ", page_block_size);

    #define DISPATCH(NH, TK) \
        if (bf16_qk) { \
            dispatch_tiles<ModelType::MODEL1, ComputeMode::FP8, NH, TK, 64, true>( \
                Q_ptr, KV_ptr, idx_ptr, PO_ptr, LSE_ptr, \
                sm_scale, num_tokens, tiles_per_split, \
                stride_kv_block, stream); \
        } else { \
            dispatch_tiles<ModelType::MODEL1, ComputeMode::FP8, NH, TK, 64, false>( \
                Q_ptr, KV_ptr, idx_ptr, PO_ptr, LSE_ptr, \
                sm_scale, num_tokens, tiles_per_split, \
                stride_kv_block, stream); \
        }

    if (topk == 512) {
        switch (num_heads) {
        case 8:   DISPATCH(8, 512); break;
        case 64:  DISPATCH(64, 512); break;
        case 128: DISPATCH(128, 512); break;
        default:  TORCH_CHECK(false, "MODEL1 decode: unsupported num_heads=", num_heads);
        }
    } else if (topk == 1024) {
        switch (num_heads) {
        case 16:  DISPATCH(16, 1024); break;
        case 64:  DISPATCH(64, 1024); break;
        case 128: DISPATCH(128, 1024); break;
        default:  TORCH_CHECK(false, "MODEL1 decode: unsupported num_heads=", num_heads);
        }
    } else {
        TORCH_CHECK(false, "MODEL1 decode: unsupported topk=", topk);
    }
    #undef DISPATCH
}

// ============================================================================
// V2 decode: scheduler-driven launch (FlashMLA-compatible)
// ============================================================================

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE,
          bool BF16_QK = KVCacheTraits<MT>::USE_BF16_QK>
void launch_decode_v2(
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
    const uint8_t* extra_KV_cache, const int32_t* extra_indices,
    float* o_accum, float* lse_accum,
    bf16* output, float* out_lse,
    const DecodingSchedMeta* sched_meta, const int* num_splits_ptr,
    float sm_scale, int num_batches, int s_q, int topk,
    size_t stride_kv_block, int num_sm_parts,
    size_t stride_oa_split, size_t stride_oa_sq,
    size_t stride_la_split, size_t stride_la_sq,
    const float* attn_sink,
    const int* topk_length, int extra_topk, const int* extra_topk_length,
    size_t stride_extra_kv_block,
    cudaStream_t stream)
{
    constexpr size_t smem_bytes = SmemLayout<MT, CM, BF16_QK>::TOTAL;
    constexpr int REPLICATE_H = (NUM_HEADS + HPB - 1) / HPB;

    auto kernel = sparse_mla_decode_v2_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE, BF16_QK>;
    static bool configured = false;
    if (!configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        configured = true;
    }

    dim3 grid(REPLICATE_H, s_q, num_sm_parts);
    dim3 block(BLOCK_THREADS);

    DecodeV2ColdParams cold{sm_scale, num_batches, s_q, stride_kv_block, topk,
                            stride_oa_split, stride_oa_sq,
                            stride_la_split, stride_la_sq, attn_sink,
                            topk_length, extra_topk, extra_topk_length,
                            stride_extra_kv_block};

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{grid, block, smem_bytes, stream, attrs, 1};
    void* args[] = {
        (void*)&Q, (void*)&KV_cache, (void*)&indices,
        (void*)&extra_KV_cache, (void*)&extra_indices,
        (void*)&o_accum, (void*)&lse_accum,
        (void*)&output, (void*)&out_lse,
        (void*)&sched_meta, (void*)&num_splits_ptr,
        (void*)&cold
    };
    CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

void sparse_mla_splitkv_v2_launch_v32(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor o_accum, torch::Tensor lse_accum,
    torch::Tensor output, torch::Tensor out_lse,
    torch::Tensor sched_meta, torch::Tensor num_splits,
    float sm_scale, int num_heads, int num_batches, int s_q, int topk,
    int page_block_size, int stride_kv_row, int num_sm_parts,
    const float* attn_sink,
    const uint8_t* extra_kv, const int32_t* extra_idx,
    const int* topk_length_ptr, int extra_topk, const int* extra_topk_length_ptr,
    int extra_page_block_size, int extra_stride_kv_row,
    cudaStream_t stream)
{
    auto Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr());
    auto KV_ptr = reinterpret_cast<const uint8_t*>(KV_cache.data_ptr());
    auto idx_ptr = indices.data_ptr<int32_t>();
    auto OA_ptr = o_accum.data_ptr<float>();
    auto LA_ptr = lse_accum.data_ptr<float>();
    auto O_ptr = reinterpret_cast<bf16*>(output.data_ptr());
    auto LSE_ptr = out_lse.data_ptr<float>();
    auto* meta_ptr = reinterpret_cast<const DecodingSchedMeta*>(sched_meta.data_ptr());
    auto* ns_ptr = num_splits.data_ptr<int>();
    size_t stride_kv_block = (size_t)page_block_size * stride_kv_row;
    size_t stride_extra = (extra_kv != nullptr)
        ? (size_t)extra_page_block_size * extra_stride_kv_row : 0;

    size_t stride_oa_split = (size_t)s_q * num_heads * D_V;
    size_t stride_oa_sq = (size_t)num_heads * D_V;
    size_t stride_la_split = (size_t)s_q * num_heads;
    size_t stride_la_sq = (size_t)num_heads;

    TORCH_CHECK(topk == 2048, "V32 decode v2 requires topk=2048");

    #define DISPATCH_V2(NH) \
        launch_decode_v2<ModelType::V32, ComputeMode::FP8, NH, 2048, 1>( \
            Q_ptr, KV_ptr, idx_ptr, extra_kv, extra_idx, \
            OA_ptr, LA_ptr, O_ptr, LSE_ptr, \
            meta_ptr, ns_ptr, sm_scale, num_batches, s_q, topk, \
            stride_kv_block, num_sm_parts, \
            stride_oa_split, stride_oa_sq, stride_la_split, stride_la_sq, \
            attn_sink, topk_length_ptr, extra_topk, extra_topk_length_ptr, \
            stride_extra, stream)

    switch (num_heads) {
    case 8:   DISPATCH_V2(8); break;
    case 16:  DISPATCH_V2(16); break;
    case 64:  DISPATCH_V2(64); break;
    case 128: DISPATCH_V2(128); break;
    default:  TORCH_CHECK(false, "V32 decode v2: unsupported num_heads=", num_heads);
    }
    #undef DISPATCH_V2
}

void sparse_mla_splitkv_v2_launch_model1(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor o_accum, torch::Tensor lse_accum,
    torch::Tensor output, torch::Tensor out_lse,
    torch::Tensor sched_meta, torch::Tensor num_splits,
    float sm_scale, int num_heads, int num_batches, int s_q, int topk,
    int page_block_size, int stride_kv_row, int num_sm_parts,
    const float* attn_sink,
    const uint8_t* extra_kv, const int32_t* extra_idx,
    const int* topk_length_ptr, int extra_topk, const int* extra_topk_length_ptr,
    int extra_page_block_size, int extra_stride_kv_row,
    bool bf16_qk,
    cudaStream_t stream)
{
    auto Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr());
    auto KV_ptr = reinterpret_cast<const uint8_t*>(KV_cache.data_ptr());
    auto idx_ptr = indices.data_ptr<int32_t>();
    auto OA_ptr = o_accum.data_ptr<float>();
    auto LA_ptr = lse_accum.data_ptr<float>();
    auto O_ptr = reinterpret_cast<bf16*>(output.data_ptr());
    auto LSE_ptr = out_lse.data_ptr<float>();
    auto* meta_ptr = reinterpret_cast<const DecodingSchedMeta*>(sched_meta.data_ptr());
    auto* ns_ptr = num_splits.data_ptr<int>();
    size_t stride_kv_block = (size_t)page_block_size * stride_kv_row;
    size_t stride_extra = (extra_kv != nullptr)
        ? (size_t)extra_page_block_size * extra_stride_kv_row : 0;

    size_t stride_oa_split = (size_t)s_q * num_heads * D_V;
    size_t stride_oa_sq = (size_t)num_heads * D_V;
    size_t stride_la_split = (size_t)s_q * num_heads;
    size_t stride_la_sq = (size_t)num_heads;

    TORCH_CHECK(page_block_size == 64, "MODEL1 decode v2: page_block_size must be 64");

    #define DISPATCH_V2(NH, TK) \
        if (bf16_qk) { \
            launch_decode_v2<ModelType::MODEL1, ComputeMode::FP8, NH, TK, 64, true>( \
                Q_ptr, KV_ptr, idx_ptr, extra_kv, extra_idx, \
                OA_ptr, LA_ptr, O_ptr, LSE_ptr, \
                meta_ptr, ns_ptr, sm_scale, num_batches, s_q, topk, \
                stride_kv_block, num_sm_parts, \
                stride_oa_split, stride_oa_sq, stride_la_split, stride_la_sq, \
                attn_sink, topk_length_ptr, extra_topk, extra_topk_length_ptr, \
                stride_extra, stream); \
        } else { \
            launch_decode_v2<ModelType::MODEL1, ComputeMode::FP8, NH, TK, 64, false>( \
                Q_ptr, KV_ptr, idx_ptr, extra_kv, extra_idx, \
                OA_ptr, LA_ptr, O_ptr, LSE_ptr, \
                meta_ptr, ns_ptr, sm_scale, num_batches, s_q, topk, \
                stride_kv_block, num_sm_parts, \
                stride_oa_split, stride_oa_sq, stride_la_split, stride_la_sq, \
                attn_sink, topk_length_ptr, extra_topk, extra_topk_length_ptr, \
                stride_extra, stream); \
        }

    if (topk == 512) {
        switch (num_heads) {
        case 8:   DISPATCH_V2(8, 512); break;
        case 64:  DISPATCH_V2(64, 512); break;
        case 128: DISPATCH_V2(128, 512); break;
        default:  TORCH_CHECK(false, "MODEL1 decode v2: unsupported num_heads=", num_heads);
        }
    } else if (topk == 1024) {
        switch (num_heads) {
        case 16:  DISPATCH_V2(16, 1024); break;
        case 64:  DISPATCH_V2(64, 1024); break;
        case 128: DISPATCH_V2(128, 1024); break;
        default:  TORCH_CHECK(false, "MODEL1 decode v2: unsupported num_heads=", num_heads);
        }
    } else {
        TORCH_CHECK(false, "MODEL1 decode v2: unsupported topk=", topk);
    }
    #undef DISPATCH_V2
}
