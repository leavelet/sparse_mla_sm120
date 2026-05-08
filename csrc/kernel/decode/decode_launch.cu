#include "decode_kernel.cuh"
#include <torch/extension.h>

// ============================================================================
// Split-KV decode: launch helpers and dispatch.
// ============================================================================

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int TILES_PER_SPLIT>
void launch_decode(
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
    float* partial_O, float* partial_LSE,
    float sm_scale, int num_tokens,
    int page_block_size, size_t stride_kv_block,
    cudaStream_t stream)
{
    constexpr size_t smem_bytes = SmemLayout<MT, CM>::TOTAL;
    constexpr int REPLICATE_H = NUM_HEADS / HPB;
    constexpr int NI = TOPK / BI;
    constexpr int NSPLITS = NI / TILES_PER_SPLIT;
    dim3 grid(num_tokens * REPLICATE_H, NSPLITS);
    dim3 block(BLOCK_THREADS);

    auto kernel = sparse_mla_decode_kernel<MT, CM, NUM_HEADS, TOPK, TILES_PER_SPLIT>;
    static bool configured = false;
    if (!configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        configured = true;
    }
    kernel<<<grid, block, smem_bytes, stream>>>(
        Q, KV_cache, indices, partial_O, partial_LSE,
        sm_scale, num_tokens, page_block_size, stride_kv_block);
}

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK>
void dispatch_tiles(
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
    float* partial_O, float* partial_LSE,
    float sm_scale, int num_tokens, int tiles_per_split,
    int page_block_size, size_t stride_kv_block,
    cudaStream_t stream)
{
    constexpr int NI = TOPK / BI;

    #define CASE(TPS) \
        case TPS: \
            if constexpr (NI >= TPS && NI % TPS == 0) { \
                launch_decode<MT, CM, NUM_HEADS, TOPK, TPS>( \
                    Q, KV_cache, indices, partial_O, partial_LSE, \
                    sm_scale, num_tokens, page_block_size, stride_kv_block, stream); \
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

    #define DISPATCH(NH) \
        dispatch_tiles<ModelType::V32, ComputeMode::FP8, NH, 2048>( \
            Q_ptr, KV_ptr, idx_ptr, PO_ptr, LSE_ptr, \
            sm_scale, num_tokens, tiles_per_split, \
            page_block_size, stride_kv_block, stream)

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
    cudaStream_t stream)
{
    auto Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr());
    auto KV_ptr = reinterpret_cast<const uint8_t*>(KV_cache.data_ptr());
    auto idx_ptr = indices.data_ptr<int32_t>();
    auto PO_ptr = partial_O.data_ptr<float>();
    auto LSE_ptr = partial_LSE.data_ptr<float>();
    size_t stride_kv_block = (size_t)page_block_size * stride_kv_row;

    #define DISPATCH(NH, TK) \
        dispatch_tiles<ModelType::MODEL1, ComputeMode::FP8, NH, TK>( \
            Q_ptr, KV_ptr, idx_ptr, PO_ptr, LSE_ptr, \
            sm_scale, num_tokens, tiles_per_split, \
            page_block_size, stride_kv_block, stream)

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
