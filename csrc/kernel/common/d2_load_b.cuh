#pragma once

#include "../../arch/common.cuh"

// D2: Direct B operand production for FP8 XV MMA (m16n8k32).
//
// Produces QMMA.16832 B operand registers directly from kv_smem[entry][dim],
// bypassing the V transpose buffer entirely.
//
// B register layout: b0 byte j = V[entry_base + tid*4+j][dim + gid]   (K=0..15)
//                    b1 byte j = V[entry_base+16 + tid*4+j][dim + gid] (K=16..31)
//
// Method: 4× LDS.32 (read 4 entries' 4-byte dim chunks) + 3× PRMT (extract
// the target dim byte from each and pack). Total: 8 LDS.32 + 6 PRMT per (b0,b1).
//
// Bank conflict: stride/4 mod 32 = {528/4%32=4(V32), 464/4%32=20(MODEL1)}.
// Within a quad, 4 threads read at 4× stride apart → banks separated by
// 4×(stride/4%32) mod 32 = {16(V32), 16(MODEL1)}. Max 2-way conflict.

template <int KV_STRIDE>
__device__ __forceinline__ void d2_load_b_fp8(
    uint32_t& b0, uint32_t& b1,
    const uint8_t* __restrict__ kv_smem,
    int entry_base,
    int dim,
    int lane)
{
    const int gid = lane >> 2;
    const int tid = lane & 3;
    const int d = dim + gid;
    const int d_base = d & ~3;
    const int d_sel = d & 3;
    const uint32_t sel = ((4 + d_sel) << 4) | d_sel;

    // b0: entries entry_base + tid*4 + {0,1,2,3}
    {
        uint32_t r0 = *reinterpret_cast<const uint32_t*>(
            kv_smem + (entry_base + tid*4+0)*KV_STRIDE + d_base);
        uint32_t r1 = *reinterpret_cast<const uint32_t*>(
            kv_smem + (entry_base + tid*4+1)*KV_STRIDE + d_base);
        uint32_t r2 = *reinterpret_cast<const uint32_t*>(
            kv_smem + (entry_base + tid*4+2)*KV_STRIDE + d_base);
        uint32_t r3 = *reinterpret_cast<const uint32_t*>(
            kv_smem + (entry_base + tid*4+3)*KV_STRIDE + d_base);
        uint32_t t01, t23;
        asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t01) : "r"(r0), "r"(r1), "r"(sel));
        asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t23) : "r"(r2), "r"(r3), "r"(sel));
        asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(b0) : "r"(t01), "r"(t23));
    }

    // b1: entries entry_base+16 + tid*4 + {0,1,2,3}
    {
        uint32_t r0 = *reinterpret_cast<const uint32_t*>(
            kv_smem + (entry_base+16 + tid*4+0)*KV_STRIDE + d_base);
        uint32_t r1 = *reinterpret_cast<const uint32_t*>(
            kv_smem + (entry_base+16 + tid*4+1)*KV_STRIDE + d_base);
        uint32_t r2 = *reinterpret_cast<const uint32_t*>(
            kv_smem + (entry_base+16 + tid*4+2)*KV_STRIDE + d_base);
        uint32_t r3 = *reinterpret_cast<const uint32_t*>(
            kv_smem + (entry_base+16 + tid*4+3)*KV_STRIDE + d_base);
        uint32_t t01, t23;
        asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t01) : "r"(r0), "r"(r1), "r"(sel));
        asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t23) : "r"(r2), "r"(r3), "r"(sel));
        asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(b1) : "r"(t01), "r"(t23));
    }
}
