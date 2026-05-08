#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

using bf16 = __nv_bfloat16;
using fp8 = __nv_fp8_e4m3;
using bf16_2 = __nv_bfloat162;

constexpr float LOG2E = 1.44269504088896340736f;
constexpr float FP8_MAX = 448.0f;
constexpr float FP8_MIN = -448.0f;
constexpr float FP8_MAX_INV = 1.0f / 448.0f;

__device__ __forceinline__ float to_float(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float fast_exp2f(float x) { return exp2f(x); }

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

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            abort();                                                           \
        }                                                                      \
    } while (0)
#endif
