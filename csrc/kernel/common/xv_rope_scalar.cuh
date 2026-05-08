#pragma once

#include "../../arch/common.cuh"
#include "../../model/kv_cache_traits.cuh"
#include "kv_cache_io.cuh"

// XV rope scalar (MODEL1 only): CUDA core FMA for d_rope=64 output dims.
//
// Each lane handles 2 rope dims (lane * 2 and lane * 2 + 1).
// Each lane independently reads ALL ENTRIES_PER_WARP entries' v_rope from
// global memory and does weighted sum.
//
// Weights are distributed across tid values in the MMA output mapping:
//   tid=0 has weights for entries [qk_nb+0, qk_nb+1]
//   tid=1 has weights for entries [qk_nb+2, qk_nb+3]
//   etc.
// We use __shfl_sync to fetch weights from the correct tid's lane.
//
// ~57 ns/tile (benchmark), completely hidden by IO prefetch (~1375 ns/tile).
// Zero smem, zero barriers.

template <ModelType MT>
__device__ __forceinline__ void xv_rope_scalar(
    float acc_rope[4],  // [dim0_head_gid, dim1_head_gid, dim0_head_gid8, dim1_head_gid8]
    float w0, float w1,  // this tid's softmax weights for head gid (entries tid*2, tid*2+1)
    float w2, float w3,  // this tid's softmax weights for head gid+8
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ KV_cache,
    int qk_nb,
    int lane,
    int page_block_size,
    size_t stride_kv_block)
{
    if constexpr (!KVCacheTraits<MT>::V_HAS_ROPE) return;

    using KV = KVCacheTraits<MT>;
    using IO = KVIOTraits<MT>;
    const int gid = lane >> 2;
    const int d_base = lane * 2;

    if (d_base >= D_ROPE) return;

    // Fetch all 8 entries' weights for this head via shfl
    float w_all_h0[8], w_all_h8[8];
    #pragma unroll
    for (int t = 0; t < 4; t++) {
        int src_lane = (gid << 2) | t;
        w_all_h0[t * 2]     = __shfl_sync(0xffffffff, w0, src_lane);
        w_all_h0[t * 2 + 1] = __shfl_sync(0xffffffff, w1, src_lane);
        w_all_h8[t * 2]     = __shfl_sync(0xffffffff, w2, src_lane);
        w_all_h8[t * 2 + 1] = __shfl_sync(0xffffffff, w3, src_lane);
    }

    // Read all 8 entries' v_rope and accumulate
    float sum_d0_h0 = 0, sum_d1_h0 = 0;
    float sum_d0_h8 = 0, sum_d1_h8 = 0;

    #pragma unroll
    for (int e = 0; e < ENTRIES_PER_WARP; e++) {
        int idx = indices[qk_nb + e];
        idx = (idx >= 0) ? idx : 0;

        // MODEL1 footer: block-structured rope address
        int block_idx = idx / page_block_size;
        int local_idx = idx % page_block_size;
        const bf16* rope = reinterpret_cast<const bf16*>(
            KV_cache + (size_t)block_idx * stride_kv_block
                     + (size_t)local_idx * IO::IO_STRIDE
                     + KV::KV_ROPE_GMEM_OFFSET);

        float vr_d0 = __bfloat162float(rope[d_base]);
        float vr_d1 = __bfloat162float(rope[d_base + 1]);

        sum_d0_h0 += w_all_h0[e] * vr_d0;
        sum_d1_h0 += w_all_h0[e] * vr_d1;
        sum_d0_h8 += w_all_h8[e] * vr_d0;
        sum_d1_h8 += w_all_h8[e] * vr_d1;
    }

    acc_rope[0] += sum_d0_h0;
    acc_rope[1] += sum_d1_h0;
    acc_rope[2] += sum_d0_h8;
    acc_rope[3] += sum_d1_h8;
}
