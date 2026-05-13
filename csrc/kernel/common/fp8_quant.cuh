#pragma once

#include "../../arch/common.cuh"
#include "../../arch/barrier.cuh"
#include "../../model/kv_cache_traits.cuh"

// On-the-fly Q quantization: BF16 → FP8 E4M3 with per-tile scaling.
//
// Steps:
//   1. Copy Q rope to smem (BF16, unquantized)
//   2. Compute per-tile absmax via atomicMax
//   3. Compute scale = absmax / FP8_MAX (power-of-2 friendly)
//   4. Quantize Q nope to FP8 and write to smem
//
// Template on ModelType to get correct Q_NOPE_STRIDE and NUM_SCALES.

template <ModelType MT, int _MATH_THREADS>
__device__ __forceinline__ void quantize_q_to_smem(
    uint8_t* q_nope_fp8,
    float* q_nope_sc,
    bf16* q_rope,
    const bf16* q_base,
    float* reduce_buf,
    int valid_hpb = HPB)
{
    using KV = KVCacheTraits<MT>;
    constexpr int D_NOPE = KV::D_NOPE;
    constexpr int Q_NOPE_STRIDE = KV::Q_NOPE_STRIDE;
    constexpr int QUANT_TILE = KV::QUANT_TILE;
    constexpr int NUM_SCALES = KV::NUM_SCALES;
    constexpr int DIM = KV::D_QK;

    float* amax = reduce_buf;

    // Step 1: copy Q rope to smem (only valid heads from gmem; zero-fill rest)
    for (int i = threadIdx.x; i < HPB * D_ROPE; i += _MATH_THREADS) {
        int h = i / D_ROPE, d = i % D_ROPE;
        q_rope[h * D_ROPE + d] = (h < valid_hpb)
            ? q_base[h * DIM + D_NOPE + d] : __float2bfloat16(0.f);
    }
    // Step 2: init amax
    for (int i = threadIdx.x; i < HPB * NUM_SCALES; i += _MATH_THREADS)
        amax[i] = 0.f;
    bar_sync_t<2, _MATH_THREADS>();

    // Compute absmax per tile (only valid heads)
    for (int idx = threadIdx.x; idx < valid_hpb * D_NOPE; idx += _MATH_THREADS) {
        int h = idx / D_NOPE, blk = (idx % D_NOPE) / QUANT_TILE;
        atomicMax(reinterpret_cast<int*>(&amax[h * NUM_SCALES + blk]),
                  __float_as_int(fabsf(__bfloat162float(q_base[h * DIM + idx % D_NOPE]))));
    }
    bar_sync_t<2, _MATH_THREADS>();

    // Step 3: compute scale, rounded up to power-of-2 for exact UE8M0 block-scaled MMA
    for (int i = threadIdx.x; i < HPB * NUM_SCALES; i += _MATH_THREADS) {
        float raw = fmaxf(amax[i], 1e-4f) / FP8_MAX;
        uint32_t bits = __float_as_uint(raw);
        if (bits & 0x007FFFFF)
            bits = (bits + 0x00800000) & 0x7F800000;
        q_nope_sc[i] = __uint_as_float(bits);
    }
    bar_sync_t<2, _MATH_THREADS>();

    // Step 4: quantize (valid heads from gmem; zero-fill rest)
    for (int idx = threadIdx.x; idx < HPB * D_NOPE; idx += _MATH_THREADS) {
        int h = idx / D_NOPE, d = idx % D_NOPE, blk = d / QUANT_TILE;
        if (h < valid_hpb) {
            float si = 1.f / q_nope_sc[h * NUM_SCALES + blk];
            float v = fmaxf(FP8_MIN, fminf(FP8_MAX, __bfloat162float(q_base[h * DIM + d]) * si));
            __nv_fp8_e4m3 fp8v(v);
            q_nope_fp8[h * Q_NOPE_STRIDE + d] = fp8v.__x;
        } else {
            q_nope_fp8[h * Q_NOPE_STRIDE + d] = 0;
        }
    }
}
