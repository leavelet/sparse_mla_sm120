#pragma once

#include "common.cuh"

// SM120 MMA instruction wrappers.
// SM120 uses mma.sync (register-based), NOT wgmma/umma.
//
// FP8:  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//       A: 4 x uint32 (16x32 e4m3), B: 2 x uint32 (8x32 e4m3), C/D: 4 x float (16x8)
//
// BF16: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//       A: 4 x uint32 (16x16 bf16), B: 2 x uint32 (8x16 bf16), C/D: 4 x float (16x8)

struct MmaFp8Result {
    float d0, d1, d2, d3;
};

// m16n8k32 FP8 E4M3 MMA: C += A * B^T
__device__ __forceinline__ MmaFp8Result mma_fp8_m16n8k32(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    MmaFp8Result r;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    return r;
}

struct MmaBf16Result {
    float d0, d1, d2, d3;
};

// m16n8k16 BF16 MMA: C += A * B^T
__device__ __forceinline__ MmaBf16Result mma_bf16_m16n8k16(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    MmaBf16Result r;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    return r;
}

// ldmatrix: load 4 x 8x8 matrices from shared memory (for MMA operand A)
__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

// ldmatrix: load 2 x 8x8 matrices from shared memory (for MMA operand B)
__device__ __forceinline__ void ldmatrix_x2(
    uint32_t& r0, uint32_t& r1,
    const void* smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
}

// ldmatrix transposed: load 2 x 8x8 from shared, transposed (for B operand of m16n8k*)
__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1,
    const void* smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}
