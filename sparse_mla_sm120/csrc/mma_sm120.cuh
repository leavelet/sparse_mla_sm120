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

// ── ldmatrix-based MMA operand loaders ──────────────────────────────────
//
// These exploit the fact that FP8 m16n8k32 uses the SAME register mapping
// as BF16 m16n8k16 when 2 FP8 bytes are treated as 1 b16 element:
//   16×32 FP8  = 16×16 "b16"  = 4 × (8×8 b16)  → ldmatrix.x4
//    8×32 FP8  =  8×16 "b16"  = 2 × (8×8 b16)  → ldmatrix.x2

// Load FP8 A operand [16 rows × 32 cols] via ldmatrix.m8n8.x4.b16.
// smem_base: pointer to row 0, column 'ko' of the FP8 matrix.
// stride: row stride in BYTES (must be 16-byte aligned).
__device__ __forceinline__ void ldmatrix_load_A_fp8(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    const uint8_t* smem_base, int stride, int lane)
{
    // Map 32 threads → 4 submatrices of 8×8 b16 (= 8×16 FP8):
    //   threads  0– 7: matrix0 → rows 0–7,  FP8 cols  0–15
    //   threads  8–15: matrix1 → rows 8–15, FP8 cols  0–15
    //   threads 16–23: matrix2 → rows 0–7,  FP8 cols 16–31
    //   threads 24–31: matrix3 → rows 8–15, FP8 cols 16–31
    int row = (lane & 7) + ((lane >> 3) & 1) * 8;
    int col = (lane >> 4) * 16;
    const void* ptr = smem_base + row * stride + col;
    ldmatrix_x4(a0, a1, a2, a3, ptr);
}

// Load FP8 B operand [8 rows × 32 cols] via ldmatrix.m8n8.x2.b16.
// smem_base: pointer to row 0, column 'ko' of the FP8 matrix.
// stride: row stride in BYTES (must be 16-byte aligned).
__device__ __forceinline__ void ldmatrix_load_B_fp8(
    uint32_t& b0, uint32_t& b1,
    const uint8_t* smem_base, int stride, int lane)
{
    // threads  0– 7: matrix0 → rows 0–7, FP8 cols  0–15
    // threads  8–15: matrix1 → rows 0–7, FP8 cols 16–31
    // threads 16–31: addresses ignored by hardware for .x2
    int row = lane & 7;
    int col = ((lane >> 3) & 1) * 16;
    const void* ptr = smem_base + row * stride + col;
    ldmatrix_x2(b0, b1, ptr);
}

// Load BF16 A operand [16 rows × 16 cols] via ldmatrix.m8n8.x4.b16.
// smem_base: pointer to row 0, column 'ko' of the BF16 matrix.
// stride: row stride in BF16 ELEMENTS (byte stride = stride*2, must be 16B-aligned).
__device__ __forceinline__ void ldmatrix_load_A_bf16(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    const bf16* smem_base, int stride_elems, int lane)
{
    int row = (lane & 7) + ((lane >> 3) & 1) * 8;
    int col = (lane >> 4) * 8;
    const void* ptr = smem_base + row * stride_elems + col;
    ldmatrix_x4(a0, a1, a2, a3, ptr);
}
