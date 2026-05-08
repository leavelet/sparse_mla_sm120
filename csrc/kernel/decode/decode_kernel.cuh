#pragma once

#include "../../arch/common.cuh"
#include "../../arch/mma_sm120.cuh"
#include "../../arch/ldmatrix_sm120.cuh"
#include "../../arch/barrier.cuh"
#include "../../arch/cp_async.cuh"
#include "../../model/kv_cache_traits.cuh"
#include "../../model/scale_convert.cuh"
#include "../common/smem_layout.cuh"
#include "../common/kv_cache_io.cuh"
#include "../common/fp8_quant.cuh"
#include "../common/online_softmax.cuh"
#include "../common/q_rope.cuh"
#include "../common/v_transpose.cuh"
#include "../common/xv_rope_mma.cuh"

// ============================================================================
// Sparse MLA Decode Kernel — split-KV decode (separate combine kernel)
//
// Template params (all constexpr):
//   MT:              ModelType (V32 / MODEL1)
//   CM:              ComputeMode (FP8 / BF16) — currently FP8 only
//   NUM_HEADS:       16, 32, 64, 128
//   TOPK:            512, 1024, 2048
//   TILES_PER_SPLIT: 2, 4, 8, 16, 32 (must divide NI = TOPK/BI)
// ============================================================================

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int TILES_PER_SPLIT>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
sparse_mla_decode_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    float* __restrict__ partial_O,
    float* __restrict__ partial_LSE,
    float sm_scale,
    int num_tokens,
    int page_block_size,
    size_t stride_kv_block)
{
    using KV = KVCacheTraits<MT>;
    using CT = ComputeTraits<MT, CM>;
    using L = SmemLayout<MT, CM>;
    using IO = KVIOTraits<MT>;

    static constexpr int NI = TOPK / BI;
    static constexpr int NSPLITS = NI / TILES_PER_SPLIT;
    static constexpr int REPLICATE_H = NUM_HEADS / HPB;
    static constexpr int QK_NOPE_KSTEPS = KV::QUANT_TILE / 32;

    const int s_i = blockIdx.x / REPLICATE_H;
    const int h_tile = blockIdx.x % REPLICATE_H;
    const int h_start = h_tile * HPB;
    const int split_idx = blockIdx.y;
    if (s_i >= num_tokens) return;

    constexpr int tile_start_stride = TILES_PER_SPLIT * BI;

    const int warp_rank = threadIdx.x / 32;
    const int wy = warp_rank / 4;

    extern __shared__ char smem_raw[];
    auto sm = SmemPtrs<MT, CM>::init(smem_raw);

    if (threadIdx.x == 0) {
        mbarrier_init(sm.mbar_kv + 0, 1);
        mbarrier_init(sm.mbar_kv + 1, 1);
    }
    bar_sync_t<3, BLOCK_THREADS>();

    // ── IO warps ────────────────────────────────────────────────────
    if (wy == 2) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));

        const int io_tid = threadIdx.x - N_MATH_WARPS * 32;
        const int32_t* idx_base = indices + (size_t)s_i * TOPK + split_idx * tile_start_stride;

        // Prologue: gather tile 0
        io_bulk_gather_tile<MT>(
            sm.kv_bufs[0], idx_base, KV_cache, sm.mbar_kv + 0, io_tid,
            page_block_size, stride_kv_block);
        io_gather_scales<MT>(
            sm.kv_scale_bufs[0], idx_base, KV_cache, io_tid,
            page_block_size, stride_kv_block);
        // [F3] Ensure scale stores visible to math warps (mbarrier only tracks cp.async.bulk)
        __threadfence_block();

        #pragma unroll
        for (int ti = 0; ti < TILES_PER_SPLIT; ti++) {
            if (ti + 1 < TILES_PER_SPLIT) {
                io_bulk_gather_tile<MT>(
                    sm.kv_bufs[(ti + 1) & 1],
                    idx_base + (ti + 1) * BI, KV_cache,
                    sm.mbar_kv + ((ti + 1) & 1), io_tid,
                    page_block_size, stride_kv_block);
                io_gather_scales<MT>(
                    sm.kv_scale_bufs[(ti + 1) & 1],
                    idx_base + (ti + 1) * BI, KV_cache, io_tid,
                    page_block_size, stride_kv_block);
                // [F3] threadfence for scale visibility in main loop
                __threadfence_block();
            }
            bar_sync_t<1, BLOCK_THREADS>();
        }

    // ── Math warps ──────────────────────────────────────────────────
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(232));

        const int lane = threadIdx.x & 31;
        const int mwarp = warp_rank;
        const int gid = lane >> 2, tid = lane & 3;
        const float sm_scale_log2e = sm_scale * LOG2E;
        const bf16* q_base = Q + (size_t)s_i * NUM_HEADS * KV::D_QK + (size_t)h_start * KV::D_QK;
        const int32_t* idx_base = indices + (size_t)s_i * TOPK + split_idx * tile_start_stride;

        quantize_q_to_smem<MT, MATH_THREADS>(
            sm.q_nope_fp8, sm.q_nope_sc, sm.q_rope, q_base, sm.reduce_buf);
        QRopeRegs q_rope_regs = preload_q_rope_regs(sm.q_rope, lane);

        for (int h = threadIdx.x; h < HPB; h += MATH_THREADS) {
            sm.m_smem[h] = -1e30f;
            sm.l_smem[h] = 0.f;
        }

        float acc_o[CT::ACC_TILES][4];
        #pragma unroll
        for (int t = 0; t < CT::ACC_TILES; t++)
            acc_o[t][0] = acc_o[t][1] = acc_o[t][2] = acc_o[t][3] = 0.f;

        // MODEL1 rope output accumulator (dims D_NOPE..D_NOPE+D_ROPE-1)
        float acc_rope[4] = {0.f, 0.f, 0.f, 0.f};

        bar_sync_t<2, MATH_THREADS>();
        mbarrier_wait_parity(sm.mbar_kv + 0, 0);

        // ── Main loop — QK + softmax + XV ───────────────────────────
        #pragma unroll
        for (int ti = 0; ti < TILES_PER_SPLIT; ti++) {
            uint8_t* kv_smem = sm.kv_bufs[ti & 1];
            const int32_t* ib = idx_base + ti * BI;
            const int qk_nb = mwarp * ENTRIES_PER_WARP;

            for (int i = threadIdx.x; i < CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = 0.f;

            // ── QK nope (FP8 MMA) ──────────────────────────────────
            float qk[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
                float ab[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int ks = 0; ks < QK_NOPE_KSTEPS; ks++) {
                    int ko = blk * KV::QUANT_TILE + ks * 32;
                    uint32_t a0, a1, a2, a3, b0, b1;
                    ldmatrix_load_A_fp8(a0, a1, a2, a3,
                        sm.q_nope_fp8 + ko, KV::Q_NOPE_STRIDE, lane);
                    ldmatrix_load_B_fp8(b0, b1,
                        kv_smem + qk_nb * KV::KV_SMEM_STRIDE + ko, KV::KV_SMEM_STRIDE, lane);
                    MmaFp8Result r = mma_fp8_m16n8k32(
                        a0, a1, a2, a3, b0, b1,
                        ab[0], ab[1], ab[2], ab[3]);
                    ab[0] = r.d0; ab[1] = r.d1; ab[2] = r.d2; ab[3] = r.d3;
                }
                // Scale: Q_scale × KV_scale
                float qs0 = sm.q_nope_sc[gid * KV::NUM_SCALES + blk];
                float qs1 = sm.q_nope_sc[(gid + 8) * KV::NUM_SCALES + blk];
                float k0, k1;
                if constexpr (KV::SCALE_IN_KV_SMEM) {
                    k0 = reinterpret_cast<const float*>(kv_smem + (qk_nb + tid * 2) * KV::KV_SMEM_STRIDE + KV::D_NOPE)[blk];
                    k1 = reinterpret_cast<const float*>(kv_smem + (qk_nb + tid * 2 + 1) * KV::KV_SMEM_STRIDE + KV::D_NOPE)[blk];
                } else {
                    // MODEL1: scales from separate scale buffer
                    uint8_t s0 = sm.kv_scale_bufs[ti & 1][(qk_nb + tid * 2) * KV::SCALE_BYTES_PER_TOKEN + blk];
                    uint8_t s1 = sm.kv_scale_bufs[ti & 1][(qk_nb + tid * 2 + 1) * KV::SCALE_BYTES_PER_TOKEN + blk];
                    k0 = ue8m0_to_fp32(s0);
                    k1 = ue8m0_to_fp32(s1);
                }
                qk[0] += ab[0] * qs0 * k0; qk[1] += ab[1] * qs0 * k1;
                qk[2] += ab[2] * qs1 * k0; qk[3] += ab[3] * qs1 * k1;
            }

            // ── QK rope (BF16 MMA) ─────────────────────────────────
            {
                int entry_idx = ib[qk_nb + gid];
                entry_idx = (entry_idx >= 0) ? entry_idx : 0;
                const uint8_t* rope_base;
                if constexpr (KV::SCALE_IN_KV_SMEM) {
                    rope_base = KV_cache + (size_t)entry_idx * IO::IO_STRIDE;
                } else {
                    int bi = entry_idx / page_block_size;
                    int li = entry_idx % page_block_size;
                    rope_base = KV_cache + (size_t)bi * stride_kv_block
                                         + (size_t)li * IO::IO_STRIDE;
                }
                const bf16* rope_ptr = reinterpret_cast<const bf16*>(
                    rope_base + KV::KV_ROPE_GMEM_OFFSET);
                compute_qk_rope(qk, q_rope_regs, rope_ptr, lane);
            }

            // ── Invalid index masking ──────────────────────────────
            {
                int e0 = qk_nb + tid * 2, e1 = e0 + 1;
                if (ib[e0] < 0) { qk[0] = -1e30f; qk[2] = -1e30f; }
                if (ib[e1] < 0) { qk[1] = -1e30f; qk[3] = -1e30f; }
            }

            // ── Online softmax ──────────────────────────────────────
            float s[4] = { qk[0] * sm_scale_log2e, qk[1] * sm_scale_log2e,
                           qk[2] * sm_scale_log2e, qk[3] * sm_scale_log2e };

            float lm0, lm1;
            softmax_warp_max(s, lm0, lm1);
            if (tid == 0) {
                sm.reduce_buf[mwarp * HPB + gid] = lm0;
                sm.reduce_buf[mwarp * HPB + gid + 8] = lm1;
            }
            bar_sync_t<2, MATH_THREADS>();

            if (threadIdx.x < HPB) {
                int h = threadIdx.x;
                float old_m = sm.m_smem[h], tm = -1e30f;
                #pragma unroll
                for (int w = 0; w < N_MATH_WARPS; w++)
                    tm = fmaxf(tm, sm.reduce_buf[w * HPB + h]);
                float nm = fmaxf(old_m, tm);
                float alpha = exp2f(old_m - nm);
                sm.m_smem[h] = nm;
                sm.l_smem[h] *= alpha;
                sm.reduce_buf[h] = alpha;
                sm.reduce_buf[HPB + h] = nm;
            }
            bar_sync_t<2, MATH_THREADS>();

            float alpha0 = sm.reduce_buf[gid], alpha1 = sm.reduce_buf[gid + 8];
            float nm0 = sm.reduce_buf[HPB + gid], nm1 = sm.reduce_buf[HPB + gid + 8];

            #pragma unroll
            for (int t = 0; t < CT::ACC_TILES; t++) {
                acc_o[t][0] *= alpha0; acc_o[t][1] *= alpha0;
                acc_o[t][2] *= alpha1; acc_o[t][3] *= alpha1;
            }
            if constexpr (KV::V_HAS_ROPE) {
                acc_rope[0] *= alpha0; acc_rope[1] *= alpha0;
                acc_rope[2] *= alpha1; acc_rope[3] *= alpha1;
            }

            float w0 = exp2f(s[0] - nm0), w1 = exp2f(s[1] - nm0);
            float w2 = exp2f(s[2] - nm1), w3 = exp2f(s[3] - nm1);

            float ls0, ls1;
            softmax_warp_sum(w0, w1, w2, w3, ls0, ls1);
            if (tid == 0) {
                sm.sum_reduce_buf[mwarp * HPB + gid] = ls0;
                sm.sum_reduce_buf[mwarp * HPB + gid + 8] = ls1;
            }

            // ── V scale atomicMax ───────────────────────────────────
            // V_CHUNK = QUANT_TILE → 1:1 scale mapping, no max-of-tiles.
            {
                int e0i = qk_nb + tid * 2, e1i = e0i + 1;
                #pragma unroll
                for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
                    float vsc0, vsc1;
                    if constexpr (KV::SCALE_IN_KV_SMEM) {
                        vsc0 = reinterpret_cast<const float*>(kv_smem + e0i * KV::KV_SMEM_STRIDE + KV::D_NOPE)[vc];
                        vsc1 = reinterpret_cast<const float*>(kv_smem + e1i * KV::KV_SMEM_STRIDE + KV::D_NOPE)[vc];
                    } else {
                        vsc0 = ue8m0_to_fp32(sm.kv_scale_bufs[ti & 1][e0i * KV::SCALE_BYTES_PER_TOKEN + vc]);
                        vsc1 = ue8m0_to_fp32(sm.kv_scale_bufs[ti & 1][e1i * KV::SCALE_BYTES_PER_TOKEN + vc]);
                    }
                    float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
                    float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
                    atomicMax(reinterpret_cast<int*>(&sm.w_head_sc_all[vc * HPB + gid]),
                        __float_as_int(fmaxf(fabsf(ws00), fabsf(ws01))));
                    atomicMax(reinterpret_cast<int*>(&sm.w_head_sc_all[vc * HPB + gid + 8]),
                        __float_as_int(fmaxf(fabsf(ws10), fabsf(ws11))));
                }
            }
            bar_sync_t<2, MATH_THREADS>();

            if (threadIdx.x < HPB) {
                int h = threadIdx.x;
                float ts = 0.f;
                #pragma unroll
                for (int w = 0; w < N_MATH_WARPS; w++)
                    ts += sm.sum_reduce_buf[w * HPB + h];
                sm.l_smem[h] += ts;
            }
            for (int i = threadIdx.x; i < CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = fmaxf(sm.w_head_sc_all[i], 1e-10f) / FP8_MAX;
            bar_sync_t<2, MATH_THREADS>();

            // ── XV nope MMA ─────────────────────────────────────────
            {
                int e0i = qk_nb + tid * 2, e1i = e0i + 1;

                #pragma unroll
                for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
                    const int v_off = vc * CT::V_CHUNK;
                    float* vc_sc = sm.w_head_sc_all + vc * HPB;

                    {
                        float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
                        float vsc0, vsc1;
                        if constexpr (KV::SCALE_IN_KV_SMEM) {
                            vsc0 = reinterpret_cast<const float*>(kv_smem + e0i * KV::KV_SMEM_STRIDE + KV::D_NOPE)[vc];
                            vsc1 = reinterpret_cast<const float*>(kv_smem + e1i * KV::KV_SMEM_STRIDE + KV::D_NOPE)[vc];
                        } else {
                            vsc0 = ue8m0_to_fp32(sm.kv_scale_bufs[ti & 1][e0i * KV::SCALE_BYTES_PER_TOKEN + vc]);
                            vsc1 = ue8m0_to_fp32(sm.kv_scale_bufs[ti & 1][e1i * KV::SCALE_BYTES_PER_TOKEN + vc]);
                        }
                        float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
                        float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
                        __nv_fp8_e4m3 f00(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws00 * si0)));
                        __nv_fp8_e4m3 f01(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws01 * si0)));
                        __nv_fp8_e4m3 f10(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws10 * si1)));
                        __nv_fp8_e4m3 f11(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws11 * si1)));
                        sm.w_fp8[gid * (BI + 16) + e0i] = f00.__x;
                        sm.w_fp8[gid * (BI + 16) + e1i] = f01.__x;
                        sm.w_fp8[(gid + 8) * (BI + 16) + e0i] = f10.__x;
                        sm.w_fp8[(gid + 8) * (BI + 16) + e1i] = f11.__x;
                    }

                    transpose_v_chunk<MT, CM>(sm.v_trans, kv_smem, v_off);
                    bar_sync_t<2, MATH_THREADS>();

                    #pragma unroll
                    for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
                        int ti_acc = vc * CT::NT_PER_WARP_XV + nt;
                        int nl = mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
                        float xv[4] = {0.f, 0.f, 0.f, 0.f};
                        #pragma unroll
                        for (int kstep = 0; kstep < CT::XV_KSTEPS; kstep++) {
                            int ko = kstep * 32;
                            uint32_t a0, a1, a2, a3, b0, b1;
                            ldmatrix_load_A_fp8(a0, a1, a2, a3,
                                sm.w_fp8 + ko, BI + 16, lane);
                            ldmatrix_load_B_fp8(b0, b1,
                                sm.v_trans + nl * CT::V_TRANS_STRIDE + ko,
                                CT::V_TRANS_STRIDE, lane);
                            MmaFp8Result r = mma_fp8_m16n8k32(
                                a0, a1, a2, a3, b0, b1,
                                xv[0], xv[1], xv[2], xv[3]);
                            xv[0] = r.d0; xv[1] = r.d1; xv[2] = r.d2; xv[3] = r.d3;
                        }
                        float sc0 = vc_sc[gid], sc1 = vc_sc[gid + 8];
                        acc_o[ti_acc][0] += xv[0] * sc0; acc_o[ti_acc][1] += xv[1] * sc0;
                        acc_o[ti_acc][2] += xv[2] * sc1; acc_o[ti_acc][3] += xv[3] * sc1;
                    }

                    if (vc < CT::N_V_CHUNKS - 1)
                        bar_sync_t<2, MATH_THREADS>();
                }
            }

            // ── XV rope BF16 MMA (MODEL1 only) ─────────────────────
            // BF16 m16n8k16 MMA: each warp handles 8 rope dims (n_start=mwarp*8).
            // A = softmax weights (bf16 in smem, overlay on v_trans).
            // B = V rope from global (L2 cached, scalar loads).
            // ~1011 ns/tile (M7 benchmark), hidden by IO prefetch (~1375 ns/tile).
            if constexpr (KV::V_HAS_ROPE) {
                xv_rope_mma<MT>(acc_rope, w0, w1, w2, w3,
                    ib, KV_cache, mwarp, lane,
                    page_block_size, stride_kv_block,
                    reinterpret_cast<bf16*>(sm.v_trans));
            }

            bar_arrive_t<1, BLOCK_THREADS>();
            if (ti + 1 < TILES_PER_SPLIT) {
                const int next_phase = ((ti + 1) >> 1) & 1;
                mbarrier_wait_parity(sm.mbar_kv + ((ti + 1) & 1), next_phase);
            }
        }

        // ── Write partial_O (float32) and partial_LSE ────────────────
        // partial_O layout: [num_tokens, NSPLITS, NUM_HEADS, D_V] float32
        // partial_LSE layout: [num_tokens, NSPLITS, NUM_HEADS] float32
        //
        // Use smem staging (reuse kv_bufs[0]) to coalesce float32 writes.
        // Staging layout: [HPB heads × D_V floats] = HPB * D_V * 4 bytes
        // kv_bufs[0] = BI * KV_SMEM_STRIDE bytes, D_V*HPB*4 = 32768 ≤ 33792 ✓

        float il0 = (sm.l_smem[gid] > 0.f) ? (1.f / sm.l_smem[gid]) : 0.f;
        float il1 = (sm.l_smem[gid + 8] > 0.f) ? (1.f / sm.l_smem[gid + 8]) : 0.f;

        // Scatter normalized acc_o (float32) to staging
        float* staging_f32 = reinterpret_cast<float*>(sm.kv_bufs[0]);
        constexpr int F32_STAGING_STRIDE = D_V;  // floats per head row

        #pragma unroll
        for (int t = 0; t < CT::ACC_TILES; t++) {
            constexpr int _NT8 = CT::NT_PER_WARP_XV * 8;
            int c = t / CT::NT_PER_WARP_XV, lnt = t % CT::NT_PER_WARP_XV;
            int d0 = c * CT::V_CHUNK + mwarp * _NT8 + lnt * 8 + tid * 2;
            staging_f32[gid * F32_STAGING_STRIDE + d0]       = acc_o[t][0] * il0;
            staging_f32[gid * F32_STAGING_STRIDE + d0 + 1]   = acc_o[t][1] * il0;
            staging_f32[(gid+8) * F32_STAGING_STRIDE + d0]   = acc_o[t][2] * il1;
            staging_f32[(gid+8) * F32_STAGING_STRIDE + d0+1] = acc_o[t][3] * il1;
        }

        // MODEL1: rope staging — each warp writes its 8 independent dims.
        // MMA output: c0=C[gid][tid*2], c1=C[gid][tid*2+1],
        //             c2=C[gid+8][tid*2], c3=C[gid+8][tid*2+1].
        // Each warp handles n_start=mwarp*8, so no cross-warp conflicts.
        if constexpr (KV::V_HAS_ROPE) {
            int n_start = mwarp * 8;
            int d0 = KV::D_NOPE + n_start + tid * 2;
            staging_f32[gid * F32_STAGING_STRIDE + d0]       = acc_rope[0] * il0;
            staging_f32[gid * F32_STAGING_STRIDE + d0 + 1]   = acc_rope[1] * il0;
            staging_f32[(gid+8) * F32_STAGING_STRIDE + d0]   = acc_rope[2] * il1;
            staging_f32[(gid+8) * F32_STAGING_STRIDE + d0+1] = acc_rope[3] * il1;
        }
        bar_sync_t<2, MATH_THREADS>();

        // Coalesced write staging → partial_O (float32, 128-bit = float4 per store)
        {
            constexpr size_t h_stride = D_V;
            constexpr size_t split_stride = (size_t)NUM_HEADS * D_V;
            constexpr size_t token_stride = (size_t)NSPLITS * split_stride;
            const size_t po_base = (size_t)s_i * token_stride
                                 + (size_t)split_idx * split_stride
                                 + (size_t)h_start * h_stride;
            constexpr int FLOATS_PER_STORE = 4;  // float4 = 16 bytes
            constexpr int STORES_PER_HEAD = D_V / FLOATS_PER_STORE;  // 128
            for (int idx = threadIdx.x; idx < HPB * STORES_PER_HEAD; idx += MATH_THREADS) {
                int h = idx / STORES_PER_HEAD;
                int d4 = (idx - h * STORES_PER_HEAD) * FLOATS_PER_STORE;
                float4 val = *reinterpret_cast<const float4*>(
                    &staging_f32[h * F32_STAGING_STRIDE + d4]);
                *reinterpret_cast<float4*>(
                    &partial_O[po_base + h * h_stride + d4]) = val;
            }
        }

        // Write partial_LSE
        if (threadIdx.x < HPB) {
            int h = threadIdx.x;
            float lse = softmax_lse(sm.m_smem[h], sm.l_smem[h]);
            constexpr size_t lse_split_stride = (size_t)NUM_HEADS;
            constexpr size_t lse_token_stride = (size_t)NSPLITS * lse_split_stride;
            size_t lse_idx = (size_t)s_i * lse_token_stride
                           + (size_t)split_idx * lse_split_stride
                           + (h_start + h);
            partial_LSE[lse_idx] = lse;
        }
    }
}
