#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// DeepSeek V3.2 MLA dimensions (hardcoded for performance)
constexpr int D_NOPE = 512;    // k_nope / v dimension
constexpr int D_ROPE = 64;     // k_rope dimension
constexpr int DIM = 576;       // D_NOPE + D_ROPE
constexpr int TOPK = 2048;     // sparse top-k
constexpr int QUANT_TILE = 128; // FP8 quantization tile size
constexpr int NUM_SCALES = D_NOPE / QUANT_TILE; // 4 scales per token

// KV cache packed layout: 656 bytes per token
// [0:512) = fp8 nope, [512:528) = 4xfp32 scales, [528:656) = 64xbf16 rope
constexpr int KV_PACKED_BYTES = D_NOPE + NUM_SCALES * sizeof(float) + D_ROPE * sizeof(__nv_bfloat16);
static_assert(KV_PACKED_BYTES == 656);

constexpr float LOG2E = 1.44269504088896340736f;
constexpr float FP8_MAX = 448.0f;
constexpr float FP8_MIN = -448.0f;
constexpr float FP8_MAX_INV = 1.0f / 448.0f;

using bf16 = __nv_bfloat16;
using fp8 = __nv_fp8_e4m3;
using bf16_2 = __nv_bfloat162;

__device__ __forceinline__ float to_float(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16(x); }

__device__ __forceinline__ float fast_exp2f(float x) { return exp2f(x); }

// Warp-level reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            abort();                                                           \
        }                                                                      \
    } while (0)
