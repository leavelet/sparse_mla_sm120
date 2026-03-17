#pragma once

#include "common.cuh"

// Async global -> shared copy (cp.async)
__device__ __forceinline__ void cp_async_4B(void* smem_ptr, const void* gmem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_8B(void* smem_ptr, const void* gmem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;\n"
        :: "r"(addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Vectorized shared memory store (128-bit)
__device__ __forceinline__ void store_128b(void* smem_ptr, uint4 data) {
    *reinterpret_cast<uint4*>(smem_ptr) = data;
}

// Vectorized shared memory load (128-bit)
__device__ __forceinline__ uint4 load_128b(const void* smem_ptr) {
    return *reinterpret_cast<const uint4*>(smem_ptr);
}
