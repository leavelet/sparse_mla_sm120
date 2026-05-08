#pragma once

#include "../../arch/common.cuh"
#include "../../arch/mma_sm120.cuh"
#include "../../arch/ldmatrix_sm120.cuh"
#include "../../model/kv_cache_traits.cuh"

// Q rope: preload to registers and compute QK rope via BF16 MMA.
//
// Q rope is [HPB × D_ROPE] BF16 in smem (from quantize_q_to_smem).
// Preloaded to registers once before the main loop (survives across all tiles).
// KV rope is read from global memory per-entry during QK computation.

struct QRopeRegs {
    uint32_t a[N_ROPE_CHUNKS][4];
};

__device__ __forceinline__ QRopeRegs preload_q_rope_regs(
    const bf16* q_rope_smem, int lane)
{
    QRopeRegs regs;
    #pragma unroll
    for (int ks = 0; ks < N_ROPE_CHUNKS; ks++)
        ldmatrix_load_A_bf16(regs.a[ks][0], regs.a[ks][1],
                              regs.a[ks][2], regs.a[ks][3],
                              q_rope_smem + ks * 16, D_ROPE, lane);
    return regs;
}

// Compute QK rope contribution.
// kv_rope_ptr: pointer to the rope BF16 data for this entry (already computed
// by caller using kv_token_data_addr + KV_ROPE_GMEM_OFFSET).
__device__ __forceinline__ void compute_qk_rope(
    float qk[4], const QRopeRegs& qr,
    const bf16* kv_rope_ptr, int lane)
{
    const int tid = lane & 3;
    float ra[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int ks = 0; ks < N_ROPE_CHUNKS; ks++) {
        int ko = ks * 16;
        uint32_t b0 = *reinterpret_cast<const uint32_t*>(kv_rope_ptr + ko + tid * 2);
        uint32_t b1 = *reinterpret_cast<const uint32_t*>(kv_rope_ptr + ko + 8 + tid * 2);
        MmaBf16Result r = mma_bf16_m16n8k16(
            qr.a[ks][0], qr.a[ks][1], qr.a[ks][2], qr.a[ks][3],
            b0, b1, ra[0], ra[1], ra[2], ra[3]);
        ra[0] = r.d0; ra[1] = r.d1; ra[2] = r.d2; ra[3] = r.d3;
    }
    qk[0] += ra[0]; qk[1] += ra[1]; qk[2] += ra[2]; qk[3] += ra[3];
}
