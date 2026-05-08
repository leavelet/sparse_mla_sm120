#pragma once

#include "common.cuh"

// cp.async: global → shared async copy

__device__ __forceinline__ void cp_async_4B(void* smem_ptr, const void* gmem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_8B(void* smem_ptr, const void* gmem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_16B_l2(void* smem_ptr, const void* gmem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n"); }

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// cp.async.bulk: SM90+ bulk global → shared (mbarrier-based completion)
__device__ __forceinline__ void cp_async_bulk_g2s(
    void* smem_dst, const void* gmem_src, uint32_t bytes, uint64_t* mbar)
{
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];\n"
        :: "r"(dst_addr), "l"(gmem_src), "r"(bytes), "r"(mbar_addr));
}
