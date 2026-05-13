#pragma once

#include "common.cuh"

// CLC (Cooperative Launch Control) primitives for SM100+.
// Cluster barrier (SM90+), DSMEM (SM90+), work stealing via try_cancel (SM100+).

__device__ __forceinline__ void cluster_arrive() {
    asm volatile("barrier.cluster.arrive.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_wait() {
    asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_sync() {
    cluster_arrive();
    cluster_wait();
}

__device__ __forceinline__ uint32_t block_rank_in_cluster() {
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
    return rank;
}

// Map local smem address to remote CTA's DSMEM address within cluster.
// After this, regular ld.shared on the returned address reads remote CTA's smem.
__device__ __forceinline__ uint32_t dsmem_map(const void* smem_ptr, uint32_t remote_rank) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t result;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                 : "=r"(result) : "r"(smem_addr), "r"(remote_rank));
    return result;
}

__device__ __forceinline__ float dsmem_read_f32(uint32_t dsmem_addr) {
    float val;
    asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(val) : "r"(dsmem_addr));
    return val;
}

__device__ __forceinline__ float4 dsmem_read_f128(uint32_t dsmem_addr) {
    float4 val;
    asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                 : "r"(dsmem_addr));
    return val;
}

// Fence for mbarrier init visibility across cluster.
__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// ── CLC work stealing (SM100+) ──────────────────────────────────────

struct CLCResult {
    bool is_valid;
    int x, y, z;
};

// Issue try_cancel: async request to cancel an unscheduled cluster.
// Response (16B) written to `response`, tracked by `mbar`.
// Caller must first mbarrier_arrive_expect_tx(mbar, 16).
__device__ __forceinline__ void clc_try_cancel(uint64_t* mbar, void* response) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint32_t resp_addr = static_cast<uint32_t>(__cvta_generic_to_shared(response));
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta"
        ".mbarrier::complete_tx::bytes.b128 [%0], [%1];\n"
        :: "r"(resp_addr), "r"(mbar_addr) : "memory");
}

// Decode the 16B opaque response from try_cancel.
__device__ __forceinline__ CLCResult clc_decode_response(const void* response) {
    uint32_t resp_addr = static_cast<uint32_t>(__cvta_generic_to_shared(response));
    uint32_t is_canceled, x, y, z;
    asm volatile(
        "{\n"
        " .reg .b128 resp;\n"
        " .reg .pred p;\n"
        " ld.shared.b128 resp, [%4];\n"
        " clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, resp;\n"
        " selp.u32 %0, 1, 0, p;\n"
        " @p clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %1, resp;\n"
        " @p clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %2, resp;\n"
        " @p clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %3, resp;\n"
        "}\n"
        : "=r"(is_canceled), "=r"(x), "=r"(y), "=r"(z)
        : "r"(resp_addr));
    return {is_canceled != 0, (int)x, (int)y, (int)z};
}
