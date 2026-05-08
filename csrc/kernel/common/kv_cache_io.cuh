#pragma once

#include "../../arch/cp_async.cuh"
#include "../../arch/barrier.cuh"
#include "../../model/kv_cache_traits.cuh"

// KV cache IO: gather BI entries from global KV pool to smem.
//
// FlashMLA ABI: stride_kv_row = bytes_per_token (V32: 656, MODEL1: 584).
// The IO stride used for address calculation is the DATA stride:
//   V32:    656 (nope+scale+rope all contiguous, 656 % 16 = 0 ✓)
//   MODEL1: 576 (nope+rope only, footer scales excluded)
//           576 % 16 = 0 ✓ for cp.async.bulk
//
// V32 uses flat addressing: kv_ptr + global_idx * 656.
// MODEL1 uses block-structured addressing (footer layout):
//   data:  kv_ptr + block_idx * stride_kv_block + local_idx * 576
//   scale: kv_ptr + block_idx * stride_kv_block + page_block_size * 576 + local_idx * 8
//
// Reference: FlashMLA SM90 splitkv_mla.cuh L538-555, SM100 kernel.cuh L657-714.

template <ModelType MT>
struct KVIOTraits {
    using KV = KVCacheTraits<MT>;
    // V32: IO_STRIDE = KV_GMEM_STRIDE = 656 (inline, bulk copy includes scale)
    // MODEL1: IO_STRIDE = D_NOPE + D_ROPE*2 = 576 (footer, data portion only)
    static constexpr int IO_STRIDE = KV::SCALE_IN_KV_SMEM
        ? KV::KV_GMEM_STRIDE
        : (KV::D_NOPE + D_ROPE * sizeof(bf16));
    static_assert(IO_STRIDE % 16 == 0, "IO stride must be 16B aligned for cp.async.bulk");
};

// Bulk gather token nope data (and inline scales for V32) from global to smem.
// V32: flat addressing (idx * 656). MODEL1: block-structured (footer layout).
template <ModelType MT>
__device__ __forceinline__ void io_bulk_gather_tile(
    uint8_t* dst,
    const int32_t* indices,
    const uint8_t* __restrict__ kv_ptr,
    uint64_t* mbar,
    int io_tid,
    int page_block_size,
    size_t stride_kv_block)
{
    using KV = KVCacheTraits<MT>;
    using IO = KVIOTraits<MT>;
    constexpr int COPY_BYTES = KV::KV_SMEM_COPY_BYTES;
    constexpr int SMEM_STRIDE = KV::KV_SMEM_STRIDE;

    if (io_tid == 0)
        mbarrier_arrive_expect_tx(mbar, BI * COPY_BYTES);

    #pragma unroll 1
    for (int bi = io_tid; bi < BI; bi += IO_THREADS) {
        int idx = indices[bi];
        idx = (idx >= 0) ? idx : 0;

        const uint8_t* src;
        if constexpr (KV::SCALE_IN_KV_SMEM) {
            src = kv_ptr + (size_t)idx * IO::IO_STRIDE;
        } else {
            int block_idx = idx / page_block_size;
            int local_idx = idx % page_block_size;
            src = kv_ptr + (size_t)block_idx * stride_kv_block
                         + (size_t)local_idx * IO::IO_STRIDE;
        }
        cp_async_bulk_g2s(dst + bi * SMEM_STRIDE, src, COPY_BYTES, mbar);
    }
}

// Gather MODEL1 footer scales to separate smem buffer.
// V32: no-op (scales already in bulk-copied kv_smem).
// MODEL1 footer: scale at block_base + page_block_size * DATA_STRIDE + local_idx * 8.
template <ModelType MT>
__device__ __forceinline__ void io_gather_scales(
    uint8_t* scale_dst,
    const int32_t* indices,
    const uint8_t* __restrict__ kv_ptr,
    int io_tid,
    int page_block_size,
    size_t stride_kv_block)
{
    using KV = KVCacheTraits<MT>;
    using IO = KVIOTraits<MT>;
    if constexpr (KV::SCALE_IN_KV_SMEM) return;

    constexpr int SCALE_BYTES = KV::SCALE_BYTES_PER_TOKEN;

    for (int bi = io_tid; bi < BI; bi += IO_THREADS) {
        int idx = indices[bi];
        idx = (idx >= 0) ? idx : 0;

        int block_idx = idx / page_block_size;
        int local_idx = idx % page_block_size;
        const uint8_t* src = kv_ptr + (size_t)block_idx * stride_kv_block
                                     + (size_t)page_block_size * IO::IO_STRIDE
                                     + (size_t)local_idx * SCALE_BYTES;
        *reinterpret_cast<uint64_t*>(scale_dst + bi * SCALE_BYTES) =
            __ldg(reinterpret_cast<const uint64_t*>(src));
    }
}
