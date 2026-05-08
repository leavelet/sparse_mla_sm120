#pragma once

#include "common.cuh"

// barrier.cta with immediate operands (17 ns vs 22 ns for register operands)
template <int ID, int CNT>
__device__ __forceinline__ void bar_arrive_t() {
    asm volatile("barrier.cta.arrive %0, %1;\n" :: "n"(ID), "n"(CNT) : "memory");
}

template <int ID, int CNT>
__device__ __forceinline__ void bar_sync_t() {
    asm volatile("barrier.cta.sync %0, %1;\n" :: "n"(ID), "n"(CNT) : "memory");
}

// mbarrier (SM90+) for async copy tracking
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* mbar) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(addr));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n .reg .b64 state;\n"
        " mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\n"
        "}\n" :: "r"(addr), "r"(tx_bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar, uint32_t phase) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n .reg .pred p;\n"
            " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            " selp.u32 %0, 1, 0, p;\n"
            "}\n" : "=r"(done) : "r"(addr), "r"(phase));
    }
}
