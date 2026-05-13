#pragma once

#include "../../arch/common.cuh"
#include "../../arch/mma_sm120.cuh"
#include "../../arch/ldmatrix_sm120.cuh"
#include "../../model/kv_cache_traits.cuh"
#include "kv_cache_io.cuh"

// XV rope via BF16 MMA m16n8k16 (MODEL1 only).
//
// Computes: acc_rope[h][d] += Σ_e softmax_weight[h][e] * v_rope[e][d]
// Per tile: 64 entries, 16 heads, 64 rope dims.
//
// Each warp handles 1 n-tile of 8 rope dims (n_start = mwarp * 8).
// 4 k-steps of 16 entries each cover all 64 entries. No cross-warp reduction.
//
// A operand: softmax weights as bf16, stored to smem (overlay on dead v_trans),
//            loaded via ldmatrix.x4.
// B operand: V rope bf16, loaded from global memory (L2 cached), scalar packing.
// C output:  c0=C[gid][tid*2], c1=C[gid][tid*2+1],
//            c2=C[gid+8][tid*2], c3=C[gid+8][tid*2+1]

template <ModelType MT>
__device__ __forceinline__ void xv_rope_mma(
    float acc_rope[4],
    float w0, float w1, float w2, float w3,
    const int32_t* __restrict__ tile_indices,
    const uint8_t* __restrict__ KV_cache,
    int mwarp,
    int lane,
    size_t stride_kv_block,
    int page_block_size,
    bf16* weight_smem)
{
    if constexpr (!KVCacheTraits<MT>::V_HAS_ROPE) return;

    using KV = KVCacheTraits<MT>;
    using IO = KVIOTraits<MT>;
    const int gid = lane >> 2;
    const int tid = lane & 3;
    const int qk_nb = mwarp * ENTRIES_PER_WARP;

    weight_smem[gid * BI + qk_nb + tid * 2]           = __float2bfloat16(w0);
    weight_smem[gid * BI + qk_nb + tid * 2 + 1]       = __float2bfloat16(w1);
    weight_smem[(gid + 8) * BI + qk_nb + tid * 2]     = __float2bfloat16(w2);
    weight_smem[(gid + 8) * BI + qk_nb + tid * 2 + 1] = __float2bfloat16(w3);
    bar_sync_t<2, MATH_THREADS>();

    int n_start = mwarp * 8;
    int dim_n = n_start + gid;

    for (int ks = 0; ks < BI / 16; ks++) {
        int k_base = ks * 16;

        uint32_t a0, a1, a2, a3;
        ldmatrix_load_A_bf16(a0, a1, a2, a3,
            weight_smem + k_base, BI, lane);

        auto load_rope_v = [&](int entry_offset) -> uint16_t {
            int idx = tile_indices[entry_offset];
            if (idx < 0) return 0;
            const uint8_t* base;
            if constexpr (KV::SCALE_IN_KV_SMEM) {
                base = KV_cache + (size_t)idx * IO::IO_STRIDE;
            } else {
                const unsigned pbs = (unsigned)page_block_size;
                unsigned bi = (unsigned)idx / pbs;
                unsigned li = (unsigned)idx % pbs;
                base = KV_cache + (size_t)bi * stride_kv_block
                                + (size_t)li * IO::IO_STRIDE;
            }
            const bf16* rp = reinterpret_cast<const bf16*>(base + KV::KV_ROPE_GMEM_OFFSET);
            return *reinterpret_cast<const uint16_t*>(&rp[dim_n]);
        };

        uint16_t v0 = load_rope_v(k_base + tid * 2);
        uint16_t v1 = load_rope_v(k_base + tid * 2 + 1);
        uint16_t v8 = load_rope_v(k_base + tid * 2 + 8);
        uint16_t v9 = load_rope_v(k_base + tid * 2 + 9);

        uint32_t b0 = (uint32_t)v0 | ((uint32_t)v1 << 16);
        uint32_t b1 = (uint32_t)v8 | ((uint32_t)v9 << 16);

        MmaBf16Result r = mma_bf16_m16n8k16(
            a0, a1, a2, a3, b0, b1,
            acc_rope[0], acc_rope[1], acc_rope[2], acc_rope[3]);
        acc_rope[0] = r.d0;
        acc_rope[1] = r.d1;
        acc_rope[2] = r.d2;
        acc_rope[3] = r.d3;
    }
    bar_sync_t<2, MATH_THREADS>();
}
