#pragma once

#include "../../arch/common.cuh"
#include "../../model/kv_cache_traits.cuh"

// V transpose: byte-level [BI × V_CHUNK] → [V_CHUNK × BI] in smem.
//
// Required for FP8 XV MMA path (ldmatrix needs column-major B operand).
// FP8 data cannot use ldmatrix.trans (it operates at b16 granularity,
// swapping pairs of FP8 bytes rather than individual bytes).
//
// Uses uint32 loads (4 FP8 bytes at once) for throughput.
// The V_TRANS_STRIDE includes padding for bank conflict avoidance.

template <ModelType MT, ComputeMode CM>
__device__ __forceinline__ void transpose_v_chunk(
    uint8_t* __restrict__ v_trans,
    const uint8_t* __restrict__ kv_smem,
    int v_off)
{
    using KV = KVCacheTraits<MT>;
    using CT = ComputeTraits<MT, CM>;
    constexpr int V_CHUNK_LOCAL = CT::V_CHUNK;
    constexpr int KV_STRIDE = KV::KV_SMEM_STRIDE;
    constexpr int VT_STRIDE = CT::V_TRANS_STRIDE;
    constexpr int WORK = (V_CHUNK_LOCAL / 4) * BI;

    for (int idx = threadIdx.x; idx < WORK; idx += MATH_THREADS) {
        int d4 = (idx / BI) * 4;
        int e  = idx % BI;

        // Handle MODEL1 padding: when v_off + d4 >= D_NOPE, read zeros
        uint32_t val;
        if constexpr (KV::V_HAS_ROPE) {
            // MODEL1: pad 448→512, last chunk reads beyond D_NOPE
            if (v_off + d4 + 3 < KV::D_NOPE)
                val = *reinterpret_cast<const uint32_t*>(kv_smem + e * KV_STRIDE + v_off + d4);
            else
                val = 0;  // padding region
        } else {
            // V32: all 512 bytes are valid nope
            val = *reinterpret_cast<const uint32_t*>(kv_smem + e * KV_STRIDE + v_off + d4);
        }

        v_trans[(d4 + 0) * VT_STRIDE + e] = (uint8_t)(val);
        v_trans[(d4 + 1) * VT_STRIDE + e] = (uint8_t)(val >> 8);
        v_trans[(d4 + 2) * VT_STRIDE + e] = (uint8_t)(val >> 16);
        v_trans[(d4 + 3) * VT_STRIDE + e] = (uint8_t)(val >> 24);
    }
}
