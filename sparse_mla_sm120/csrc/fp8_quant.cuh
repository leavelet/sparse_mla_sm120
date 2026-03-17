#pragma once

#include "common.cuh"

// On-the-fly BF16 -> FP8 E4M3 quantization with per-tile scaling.
// Each tile of QUANT_TILE (128) elements gets one FP32 scale.

// Quantize a bf16 value to fp8 e4m3 given a pre-computed inverse scale.
__device__ __forceinline__ fp8 bf16_to_fp8(bf16 val, float scale_inv) {
    float f = to_float(val) * scale_inv;
    f = fmaxf(FP8_MIN, fminf(FP8_MAX, f));
    return static_cast<fp8>(__nv_fp8_e4m3(f));
}

// Compute the absmax of a bf16 array in shared memory using warp reduction.
// Each thread processes `elems_per_thread` elements.
// Returns the scale = absmax / FP8_MAX for the tile.
__device__ __forceinline__ float compute_tile_scale_warp(
    const bf16* __restrict__ tile_smem,
    int tile_size,
    int tid_in_warp)
{
    float local_max = 0.0f;
    for (int i = tid_in_warp; i < tile_size; i += 32) {
        float v = fabsf(to_float(tile_smem[i]));
        local_max = fmaxf(local_max, v);
    }
    local_max = warp_reduce_max(local_max);
    local_max = fmaxf(local_max, 1e-4f);
    return local_max * FP8_MAX_INV;
}

// Quantize a tile of bf16 values to fp8 e4m3 in-place in shared memory.
// Writes fp8 output and returns the scale.
// tile_bf16: input [tile_size] bf16
// tile_fp8: output [tile_size] fp8
// All threads in a warp participate.
__device__ __forceinline__ float quantize_tile_warp(
    const bf16* __restrict__ tile_bf16,
    fp8* __restrict__ tile_fp8,
    int tile_size,
    int tid_in_warp)
{
    float scale = compute_tile_scale_warp(tile_bf16, tile_size, tid_in_warp);
    float scale_inv = 1.0f / scale;

    for (int i = tid_in_warp; i < tile_size; i += 32) {
        tile_fp8[i] = bf16_to_fp8(tile_bf16[i], scale_inv);
    }
    return scale;
}
