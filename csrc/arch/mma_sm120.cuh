#pragma once

#include "common.cuh"

// SM120 MMA instruction wrappers.
//
// Standard (no scale):
//   FP8:  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//   BF16: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// Block-scaled (UE8M0 scale applied in hardware, zero overhead):
//   FP8:  mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32
//         .row.col.f32.e4m3.e4m3.f32.ue8m0

struct MmaFp8Result { float d0, d1, d2, d3; };
struct MmaBf16Result { float d0, d1, d2, d3; };

__device__ __forceinline__ MmaFp8Result mma_fp8_m16n8k32(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    MmaFp8Result r;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    return r;
}

__device__ __forceinline__ MmaFp8Result mma_fp8_block_scaled_m16n8k32(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint8_t scale_a, uint8_t scale_b)
{
    MmaFp8Result r;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32"
        ".row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(static_cast<uint32_t>(scale_a)),
          "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)),
          "r"(static_cast<uint32_t>(scale_b)),
          "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)));
    return r;
}

__device__ __forceinline__ MmaBf16Result mma_bf16_m16n8k16(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    MmaBf16Result r;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    return r;
}
