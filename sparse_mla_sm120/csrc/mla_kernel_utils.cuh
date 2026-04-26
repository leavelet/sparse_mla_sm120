#pragma once

#include "common.cuh"
#include "mma_sm120.cuh"
#include "smem_utils.cuh"

// ============================================================================
// Shared utility functions for MLA kernels (prefill + decode).
// Functions are parameterized via template params for caller-specific
// tile sizes and strides. Constants from common.cuh (D_NOPE, D_ROPE,
// QUANT_TILE, NUM_SCALES, KV_PACKED_BYTES, etc.) are used directly.
// ============================================================================

// ── Barrier helpers — ID as immediate, count as register ────────────────

template <int ID, int CNT>
__device__ __forceinline__ void bar_arrive_t() {
    asm volatile("barrier.cta.arrive %0, %1;\n" :: "n"(ID), "n"(CNT) : "memory");
}
template <int ID, int CNT>
__device__ __forceinline__ void bar_sync_t() {
    asm volatile("barrier.cta.sync %0, %1;\n" :: "n"(ID), "n"(CNT) : "memory");
}

// ── cp.async with L2 prefetch hint ──────────────────────────────────────

__device__ __forceinline__ void cp_async_16B_l2(void* smem_ptr, const void* gmem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(addr), "l"(gmem_ptr));
}

// ── mbarrier helpers (SM90+) ────────────────────────────────────────────

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* mbar) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(addr));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint64_t* mbar, uint32_t tx_bytes)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n .reg .b64 state;\n"
        " mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\n"
        "}\n" :: "r"(addr), "r"(tx_bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(
    uint64_t* mbar, uint32_t phase)
{
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

// ── cp.async.bulk: global → shared::cta (SM90+) ────────────────────────

__device__ __forceinline__ void cp_async_bulk_g2s(
    void* smem_dst, const void* gmem_src,
    uint32_t bytes, uint64_t* mbar)
{
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];\n"
        :: "r"(dst_addr), "l"(gmem_src), "r"(bytes), "r"(mbar_addr));
}

// ── IO path: gather KV tile (legacy cp.async.cg, used by decode) ────────

template <int _KV_SMEM_STRIDE, int _IO_THREADS, int _BI>
__device__ __forceinline__ void io_gather_tile(
    uint8_t* dst, const int32_t* ib,
    const uint8_t* __restrict__ KV_cache, int io_tid)
{
    constexpr int CP_BYTES = 16;
    constexpr int CHUNKS_PER_ENTRY = _KV_SMEM_STRIDE / CP_BYTES;
    static_assert(_KV_SMEM_STRIDE % CP_BYTES == 0, "stride must be 16B-aligned");
    constexpr int TOTAL_CHUNKS = _BI * CHUNKS_PER_ENTRY;

    #pragma unroll 1
    for (int c = io_tid; c < TOTAL_CHUNKS; c += _IO_THREADS) {
        int bi = c / CHUNKS_PER_ENTRY;
        int bo = (c - bi * CHUNKS_PER_ENTRY) * CP_BYTES;
        int idx = ib[bi]; idx = (idx >= 0) ? idx : 0;
        cp_async_16B_l2(dst + bi * _KV_SMEM_STRIDE + bo,
                        KV_cache + (size_t)idx * KV_PACKED_BYTES + bo);
    }
    cp_async_commit();
    cp_async_wait_all();
}

// ── IO path: bulk gather KV tile (cp.async.bulk + mbarrier) ─────────────
// 1 instruction per entry (528B each) vs 33 cp.async.cg per entry.
// Only io_tid==0 calls arrive_expect_tx; all IO threads issue bulk copies.

template <int _KV_SMEM_STRIDE, int _IO_THREADS, int _BI>
__device__ __forceinline__ void io_bulk_gather_tile(
    uint8_t* dst, const int32_t* ib,
    const uint8_t* __restrict__ KV_cache,
    uint64_t* mbar, int io_tid)
{
    static_assert(_KV_SMEM_STRIDE % 16 == 0, "copy size must be multiple of 16");

    if (io_tid == 0)
        mbarrier_arrive_expect_tx(mbar, _BI * _KV_SMEM_STRIDE);

    #pragma unroll 1
    for (int bi = io_tid; bi < _BI; bi += _IO_THREADS) {
        int idx = ib[bi];
        idx = (idx >= 0) ? idx : 0;
        cp_async_bulk_g2s(
            dst + bi * _KV_SMEM_STRIDE,
            KV_cache + (size_t)idx * KV_PACKED_BYTES,
            _KV_SMEM_STRIDE, mbar);
    }
}

// ── Q quantization to smem ──────────────────────────────────────────────
// Quantizes HPB heads of Q nope to FP8 in smem, copies rope to smem.

template <int _HPB, int _Q_NOPE_STRIDE, int _MATH_THREADS>
__device__ __forceinline__ void quantize_q_to_smem(
    uint8_t* q_nope_fp8, float* q_nope_sc, bf16* q_rope,
    const bf16* q_base, float* reduce_buf)
{
    float* amax = reduce_buf;
    for (int i = threadIdx.x; i < _HPB * D_ROPE; i += _MATH_THREADS) {
        int h = i / D_ROPE, d = i % D_ROPE;
        q_rope[h * D_ROPE + d] = q_base[h * DIM + D_NOPE + d];
    }
    for (int i = threadIdx.x; i < _HPB * NUM_SCALES; i += _MATH_THREADS)
        amax[i] = 0.f;
    bar_sync_t<2, _MATH_THREADS>();

    for (int idx = threadIdx.x; idx < _HPB * D_NOPE; idx += _MATH_THREADS) {
        int h = idx / D_NOPE, blk = (idx % D_NOPE) / QUANT_TILE;
        atomicMax(reinterpret_cast<int*>(&amax[h * NUM_SCALES + blk]),
                  __float_as_int(fabsf(__bfloat162float(q_base[h * DIM + idx % D_NOPE]))));
    }
    bar_sync_t<2, _MATH_THREADS>();

    for (int i = threadIdx.x; i < _HPB * NUM_SCALES; i += _MATH_THREADS)
        q_nope_sc[i] = fmaxf(amax[i], 1e-4f) / FP8_MAX;
    bar_sync_t<2, _MATH_THREADS>();

    for (int idx = threadIdx.x; idx < _HPB * D_NOPE; idx += _MATH_THREADS) {
        int h = idx / D_NOPE, d = idx % D_NOPE, blk = d / QUANT_TILE;
        float si = 1.f / q_nope_sc[h * NUM_SCALES + blk];
        float v = fmaxf(FP8_MIN, fminf(FP8_MAX, __bfloat162float(q_base[h * DIM + d]) * si));
        __nv_fp8_e4m3 fp8v(v);
        q_nope_fp8[h * _Q_NOPE_STRIDE + d] = fp8v.__x;
    }
}

// ── QK nope MMA ─────────────────────────────────────────────────────────

template <int _Q_NOPE_STRIDE, int _KV_SMEM_STRIDE, int _QK_NOPE_KSTEPS>
__device__ __forceinline__ void compute_qk_nope(
    float qk[4], const uint8_t* q_nope_fp8, const float* q_nope_sc,
    const uint8_t* kv_smem, int qk_nb, int lane)
{
    const int gid = lane >> 2, tid = lane & 3;
    #pragma unroll
    for (int blk = 0; blk < NUM_SCALES; blk++) {
        float ab[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int ks = 0; ks < _QK_NOPE_KSTEPS; ks++) {
            int ko = blk * QUANT_TILE + ks * 32;
            uint32_t a0, a1, a2, a3, b0, b1;
            ldmatrix_load_A_fp8(a0, a1, a2, a3,
                q_nope_fp8 + ko, _Q_NOPE_STRIDE, lane);
            ldmatrix_load_B_fp8(b0, b1,
                kv_smem + qk_nb * _KV_SMEM_STRIDE + ko, _KV_SMEM_STRIDE, lane);
            MmaFp8Result r = mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1,
                                               ab[0], ab[1], ab[2], ab[3]);
            ab[0] = r.d0; ab[1] = r.d1; ab[2] = r.d2; ab[3] = r.d3;
        }
        float qs0 = q_nope_sc[gid * NUM_SCALES + blk];
        float qs1 = q_nope_sc[(gid + 8) * NUM_SCALES + blk];
        int e0 = qk_nb + tid * 2, e1 = e0 + 1;
        float k0 = reinterpret_cast<const float*>(kv_smem + e0 * _KV_SMEM_STRIDE + D_NOPE)[blk];
        float k1 = reinterpret_cast<const float*>(kv_smem + e1 * _KV_SMEM_STRIDE + D_NOPE)[blk];
        qk[0] += ab[0] * qs0 * k0; qk[1] += ab[1] * qs0 * k1;
        qk[2] += ab[2] * qs1 * k0; qk[3] += ab[3] * qs1 * k1;
    }
}

// ── Q rope registers ────────────────────────────────────────────────────

static constexpr int N_ROPE_CHUNKS = D_ROPE / 16;  // 4

struct QRopeRegs {
    uint32_t a[N_ROPE_CHUNKS][4];
};

__device__ __forceinline__ QRopeRegs preload_q_rope_regs(
    const bf16* q_rope_smem, int lane)
{
    QRopeRegs regs;
    #pragma unroll
    for (int ks = 0; ks < N_ROPE_CHUNKS; ks++)
        ldmatrix_load_A_bf16(regs.a[ks][0], regs.a[ks][1],
                              regs.a[ks][2], regs.a[ks][3],
                              q_rope_smem + ks * 16, D_ROPE, lane);
    return regs;
}

__device__ __forceinline__ void compute_qk_rope(
    float qk[4], const QRopeRegs& qr, const bf16* g_rope, int lane)
{
    const int tid = lane & 3;
    float ra[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int ks = 0; ks < N_ROPE_CHUNKS; ks++) {
        int ko = ks * 16;
        uint32_t b0 = *reinterpret_cast<const uint32_t*>(g_rope + ko + tid * 2);
        uint32_t b1 = *reinterpret_cast<const uint32_t*>(g_rope + ko + 8 + tid * 2);
        MmaBf16Result r = mma_bf16_m16n8k16(
            qr.a[ks][0], qr.a[ks][1], qr.a[ks][2], qr.a[ks][3],
            b0, b1, ra[0], ra[1], ra[2], ra[3]);
        ra[0] = r.d0; ra[1] = r.d1; ra[2] = r.d2; ra[3] = r.d3;
    }
    qk[0] += ra[0]; qk[1] += ra[1]; qk[2] += ra[2]; qk[3] += ra[3];
}

// ── V transpose helper ──────────────────────────────────────────────────

template <int _V_CHUNK, int _V_TRANS_STRIDE, int _KV_SMEM_STRIDE,
          int _MATH_THREADS, int _BI>
__device__ __forceinline__ void transpose_v_chunk(
    uint8_t* __restrict__ v_trans,
    const uint8_t* __restrict__ kv_smem,
    int v_off, int /*lane*/)
{
    static_assert(_V_CHUNK % 4 == 0, "V_CHUNK must be a multiple of 4");
    static_assert(_KV_SMEM_STRIDE % 4 == 0, "KV stride must be 4B-aligned");
    constexpr int BI_SHIFT = (_BI == 64) ? 6 : ((_BI == 32) ? 5 : 0);
    static_assert(BI_SHIFT != 0, "BI must be 32 or 64");
    constexpr int WORK = (_V_CHUNK / 4) * _BI;
    for (int idx = threadIdx.x; idx < WORK; idx += _MATH_THREADS) {
        int d4 = (idx >> BI_SHIFT) * 4;
        int e  = idx & (_BI - 1);
        uint32_t val = *reinterpret_cast<const uint32_t*>(
            kv_smem + e * _KV_SMEM_STRIDE + v_off + d4);
        v_trans[(d4 + 0) * _V_TRANS_STRIDE + e] = (uint8_t)(val);
        v_trans[(d4 + 1) * _V_TRANS_STRIDE + e] = (uint8_t)(val >> 8);
        v_trans[(d4 + 2) * _V_TRANS_STRIDE + e] = (uint8_t)(val >> 16);
        v_trans[(d4 + 3) * _V_TRANS_STRIDE + e] = (uint8_t)(val >> 24);
    }
}
