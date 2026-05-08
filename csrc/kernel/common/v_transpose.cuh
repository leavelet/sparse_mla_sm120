#pragma once

#include "../../arch/common.cuh"
#include "../../model/kv_cache_traits.cuh"

// V transpose: byte-level [BI × V_CHUNK] → [V_CHUNK × BI] in smem.
//
// Required for FP8 XV MMA path (ldmatrix needs column-major B operand).
// Uses PRMT (byte permute) for 4×4 block transpose: load 4 uint32 from
// 4 entries, permute bytes to transpose, store 4 uint32 to 4 dim rows.
// 4× STS.32 instead of 4× STS.U8 → better smem store throughput.

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

    // Process 4×4 byte blocks: (V_CHUNK/4) × (BI/4) blocks total
    constexpr int N_BLOCKS = (V_CHUNK_LOCAL / 4) * (BI / 4);

    for (int bid = threadIdx.x; bid < N_BLOCKS; bid += MATH_THREADS) {
        int bd = (bid / (BI / 4)) * 4;   // dim block start (0, 4, 8, ...)
        int be = (bid % (BI / 4)) * 4;   // entry block start (0, 4, 8, ...)

        // Load 4 rows of 4 bytes each
        uint32_t r0, r1, r2, r3;
        if constexpr (KV::V_HAS_ROPE) {
            if (v_off + bd + 3 < KV::D_NOPE) {
                r0 = *reinterpret_cast<const uint32_t*>(kv_smem + (be+0)*KV_STRIDE + v_off + bd);
                r1 = *reinterpret_cast<const uint32_t*>(kv_smem + (be+1)*KV_STRIDE + v_off + bd);
                r2 = *reinterpret_cast<const uint32_t*>(kv_smem + (be+2)*KV_STRIDE + v_off + bd);
                r3 = *reinterpret_cast<const uint32_t*>(kv_smem + (be+3)*KV_STRIDE + v_off + bd);
            } else {
                r0 = r1 = r2 = r3 = 0;
            }
        } else {
            r0 = *reinterpret_cast<const uint32_t*>(kv_smem + (be+0)*KV_STRIDE + v_off + bd);
            r1 = *reinterpret_cast<const uint32_t*>(kv_smem + (be+1)*KV_STRIDE + v_off + bd);
            r2 = *reinterpret_cast<const uint32_t*>(kv_smem + (be+2)*KV_STRIDE + v_off + bd);
            r3 = *reinterpret_cast<const uint32_t*>(kv_smem + (be+3)*KV_STRIDE + v_off + bd);
        }

        // 4×4 byte transpose using PRMT
        // r0=[a00 a01 a02 a03], r1=[a10 a11 a12 a13], etc.
        // Want: c0=[a00 a10 a20 a30], c1=[a01 a11 a21 a31], etc.
        uint32_t t01_lo, t01_hi, t23_lo, t23_hi;
        asm volatile("prmt.b32 %0, %1, %2, 0x5140;\n" : "=r"(t01_lo) : "r"(r0), "r"(r1));
        asm volatile("prmt.b32 %0, %1, %2, 0x7362;\n" : "=r"(t01_hi) : "r"(r0), "r"(r1));
        asm volatile("prmt.b32 %0, %1, %2, 0x5140;\n" : "=r"(t23_lo) : "r"(r2), "r"(r3));
        asm volatile("prmt.b32 %0, %1, %2, 0x7362;\n" : "=r"(t23_hi) : "r"(r2), "r"(r3));

        uint32_t c0, c1, c2, c3;
        asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(c0) : "r"(t01_lo), "r"(t23_lo));
        asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(c1) : "r"(t01_lo), "r"(t23_lo));
        asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(c2) : "r"(t01_hi), "r"(t23_hi));
        asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(c3) : "r"(t01_hi), "r"(t23_hi));

        // Store 4 transposed columns as uint32
        *reinterpret_cast<uint32_t*>(v_trans + (bd+0)*VT_STRIDE + be) = c0;
        *reinterpret_cast<uint32_t*>(v_trans + (bd+1)*VT_STRIDE + be) = c1;
        *reinterpret_cast<uint32_t*>(v_trans + (bd+2)*VT_STRIDE + be) = c2;
        *reinterpret_cast<uint32_t*>(v_trans + (bd+3)*VT_STRIDE + be) = c3;
    }
}
