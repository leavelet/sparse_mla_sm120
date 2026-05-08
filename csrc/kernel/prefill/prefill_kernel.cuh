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
// Sparse MLA Prefill Kernel — single-pass (no split-KV, no combine)
//
// Structurally identical to decode main loop (QK→softmax→XV), but:
//   - Iterates over ALL NI = TOPK/BI tiles (no split)
//   - Writes direct BF16 output (no partial_O + combine)
//   - No PDL (no dependent kernel)
//
// Template params (all constexpr):
//   MT:              ModelType (V32 / MODEL1)
//   CM:              ComputeMode (FP8 / BF16) — currently FP8 only
//   NUM_HEADS:       16, 64, 128
//   TOPK:            512, 1024, 2048
//   PAGE_BLOCK_SIZE: 1 (V32) or 64 (MODEL1)
// ============================================================================

struct PrefillColdParams {
    float sm_scale;
    int num_tokens;
    size_t stride_kv_block;
};

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
sparse_mla_prefill_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    bf16* __restrict__ output,
    float* __restrict__ out_lse,
    __grid_constant__ const PrefillColdParams cold)
{
    const float sm_scale = cold.sm_scale;
    const int num_tokens = cold.num_tokens;
    constexpr int page_block_size = PAGE_BLOCK_SIZE;
    const size_t stride_kv_block = cold.stride_kv_block;
    using KV = KVCacheTraits<MT>;
    using CT = ComputeTraits<MT, CM>;
    using L = SmemLayout<MT, CM>;
    using IO = KVIOTraits<MT>;

    static constexpr int NI = TOPK / BI;
    static constexpr int REPLICATE_H = NUM_HEADS / HPB;
    static constexpr int QK_NOPE_KSTEPS = KV::QUANT_TILE / 32;

    const int s_i = blockIdx.x / REPLICATE_H;
    const int h_tile = blockIdx.x % REPLICATE_H;
    const int h_start = h_tile * HPB;
    if (s_i >= num_tokens) return;

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
        const int32_t* idx_base = indices + (size_t)s_i * TOPK;
        const uint64_t kv_l2_policy = create_l2_evict_first_policy();

        // Prologue: gather tile 0
        io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(
            sm.kv_bufs[0], idx_base, KV_cache, sm.mbar_kv + 0, io_tid,
            stride_kv_block, kv_l2_policy);
        io_gather_scales<MT, PAGE_BLOCK_SIZE>(
            sm.kv_scale_bufs[0], idx_base, KV_cache, io_tid,
            stride_kv_block);
        __threadfence_block();

        #pragma unroll 1
        for (int ti = 0; ti < NI; ti++) {
            if (ti + 1 < NI) {
                io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(
                    sm.kv_bufs[(ti + 1) & 1],
                    idx_base + (ti + 1) * BI, KV_cache,
                    sm.mbar_kv + ((ti + 1) & 1), io_tid,
                    stride_kv_block, kv_l2_policy);
                io_gather_scales<MT, PAGE_BLOCK_SIZE>(
                    sm.kv_scale_bufs[(ti + 1) & 1],
                    idx_base + (ti + 1) * BI, KV_cache, io_tid,
                    stride_kv_block);
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
        const int32_t* idx_base = indices + (size_t)s_i * TOPK;

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

        float acc_rope[4] = {0.f, 0.f, 0.f, 0.f};

        bar_sync_t<2, MATH_THREADS>();
        mbarrier_wait_parity(sm.mbar_kv + 0, 0);

        // ── Main loop — QK + softmax + XV ───────────────────────────
        #pragma unroll 1
        for (int ti = 0; ti < NI; ti++) {
            uint8_t* kv_smem = sm.kv_bufs[ti & 1];
            const int32_t* ib = idx_base + ti * BI;
            const int qk_nb = mwarp * ENTRIES_PER_WARP;
            uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;

            const uint8_t* entry_base[ENTRIES_PER_WARP];
            if constexpr (KV::V_HAS_ROPE) {
                #pragma unroll
                for (int e = 0; e < ENTRIES_PER_WARP; e++) {
                    int idx = ib[qk_nb + e];
                    idx = (idx >= 0) ? idx : 0;
                    int bi_e = idx / page_block_size;
                    int li_e = idx % page_block_size;
                    entry_base[e] = KV_cache + (size_t)bi_e * stride_kv_block
                                             + (size_t)li_e * IO::IO_STRIDE;
                }
            } else {
                int idx = ib[qk_nb + gid];
                idx = (idx >= 0) ? idx : 0;
                entry_base[gid] = KV_cache + (size_t)idx * IO::IO_STRIDE;
            }

            for (int i = threadIdx.x; i < CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = 0.f;

            KVRopePrefetch rope_pf = prefetch_kv_rope(
                reinterpret_cast<const bf16*>(entry_base[gid] + KV::KV_ROPE_GMEM_OFFSET), lane);

            // ── QK nope (block-scaled FP8 MMA) ─────────────────────
            float qk[4] = {0.f, 0.f, 0.f, 0.f};
            const uint8_t* kv_gid_base = kv_warp_base + gid * KV::KV_SMEM_STRIDE;
            #pragma unroll
            for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
                uint8_t sfa = fp32_to_ue8m0(
                    sm.q_nope_sc[(gid + (lane & 1) * 8) * KV::NUM_SCALES + blk]);
                uint8_t sfb;
                if constexpr (KV::SCALE_IN_KV_SMEM) {
                    sfb = fp32_to_ue8m0(
                        reinterpret_cast<const float*>(kv_gid_base + KV::D_NOPE)[blk]);
                } else {
                    sfb = sm.kv_scale_bufs[ti & 1][(qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN + blk];
                }

                #pragma unroll
                for (int ks = 0; ks < QK_NOPE_KSTEPS; ks++) {
                    int ko = blk * KV::QUANT_TILE + ks * 32;
                    uint32_t a0, a1, a2, a3, b0, b1;
                    ldmatrix_load_A_fp8(a0, a1, a2, a3,
                        sm.q_nope_fp8 + ko, KV::Q_NOPE_STRIDE, lane);
                    ldmatrix_load_B_fp8(b0, b1,
                        kv_warp_base + ko, KV::KV_SMEM_STRIDE, lane);
                    MmaFp8Result r = mma_fp8_block_scaled_m16n8k32(
                        a0, a1, a2, a3, b0, b1,
                        qk[0], qk[1], qk[2], qk[3], sfa, sfb);
                    qk[0] = r.d0; qk[1] = r.d1; qk[2] = r.d2; qk[3] = r.d3;
                }
            }

            // ── QK rope (BF16 MMA, uses prefetched B operands) ──────
            compute_qk_rope(qk, q_rope_regs, rope_pf);

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
            bar_sync_t<2, MATH_THREADS>();
            if (tid == 0) {
                sm.sum_reduce_buf[mwarp * HPB + gid] = ls0;
                sm.sum_reduce_buf[mwarp * HPB + gid + 8] = ls1;
            }

            // ── V scale cache + atomicMax ───────────────────────────
            float vsc_cache[CT::N_V_CHUNKS][2];
            {
                const int e0i = qk_nb + tid * 2, e1i = e0i + 1;
                const uint8_t* e0_base = kv_warp_base + tid * 2 * KV::KV_SMEM_STRIDE;
                const uint8_t* e1_base = e0_base + KV::KV_SMEM_STRIDE;
                #pragma unroll
                for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
                    if constexpr (KV::SCALE_IN_KV_SMEM) {
                        vsc_cache[vc][0] = reinterpret_cast<const float*>(e0_base + KV::D_NOPE)[vc];
                        vsc_cache[vc][1] = reinterpret_cast<const float*>(e1_base + KV::D_NOPE)[vc];
                    } else {
                        vsc_cache[vc][0] = ue8m0_to_fp32(sm.kv_scale_bufs[ti & 1][e0i * KV::SCALE_BYTES_PER_TOKEN + vc]);
                        vsc_cache[vc][1] = ue8m0_to_fp32(sm.kv_scale_bufs[ti & 1][e1i * KV::SCALE_BYTES_PER_TOKEN + vc]);
                    }
                    float ws00 = w0 * vsc_cache[vc][0], ws01 = w1 * vsc_cache[vc][1];
                    float ws10 = w2 * vsc_cache[vc][0], ws11 = w3 * vsc_cache[vc][1];
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
                const int e0i = qk_nb + tid * 2, e1i = e0i + 1;

                transpose_v_chunk<MT, CM>(sm.v_trans_bufs[0], kv_smem, 0);

                #pragma unroll
                for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
                    const int v_off = vc * CT::V_CHUNK;
                    float* vc_sc = sm.w_head_sc_all + vc * HPB;
                    const int buf_idx = L::DOUBLE_BUF_XV ? (vc & 1) : 0;
                    uint8_t* cur_vt = sm.v_trans_bufs[buf_idx];
                    uint8_t* cur_wfp8 = sm.w_fp8_bufs[buf_idx];

                    {
                        float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
                        float vsc0 = vsc_cache[vc][0], vsc1 = vsc_cache[vc][1];
                        float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
                        float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
                        __nv_fp8_e4m3 f00(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws00 * si0)));
                        __nv_fp8_e4m3 f01(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws01 * si0)));
                        __nv_fp8_e4m3 f10(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws10 * si1)));
                        __nv_fp8_e4m3 f11(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws11 * si1)));
                        cur_wfp8[gid * (BI + 16) + e0i] = f00.__x;
                        cur_wfp8[gid * (BI + 16) + e1i] = f01.__x;
                        cur_wfp8[(gid + 8) * (BI + 16) + e0i] = f10.__x;
                        cur_wfp8[(gid + 8) * (BI + 16) + e1i] = f11.__x;
                    }

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
                                cur_wfp8 + ko, BI + 16, lane);
                            ldmatrix_load_B_fp8(b0, b1,
                                cur_vt + nl * CT::V_TRANS_STRIDE + ko,
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

                    if (vc < CT::N_V_CHUNKS - 1) {
                        const int next_buf = L::DOUBLE_BUF_XV ? ((vc + 1) & 1) : 0;
                        transpose_v_chunk<MT, CM>(sm.v_trans_bufs[next_buf], kv_smem, (vc + 1) * CT::V_CHUNK);
                        if constexpr (!L::DOUBLE_BUF_XV)
                            bar_sync_t<2, MATH_THREADS>();
                    }
                }
            }

            // ── XV rope BF16 MMA (MODEL1 only) ─────────────────────
            if constexpr (KV::V_HAS_ROPE) {
                bar_sync_t<2, MATH_THREADS>();
                xv_rope_mma<MT, PAGE_BLOCK_SIZE>(acc_rope, w0, w1, w2, w3,
                    ib, KV_cache, mwarp, lane,
                    stride_kv_block,
                    reinterpret_cast<bf16*>(sm.v_trans_bufs[0]));
            }

            bar_arrive_t<1, BLOCK_THREADS>();
            if (ti + 1 < NI) {
                const int next_phase = ((ti + 1) >> 1) & 1;
                mbarrier_wait_parity(sm.mbar_kv + ((ti + 1) & 1), next_phase);
            }
        }

        // ── Write BF16 output and LSE ────────────────────────────────
        // output layout: [num_tokens, NUM_HEADS, D_V] bfloat16
        // out_lse layout: [num_tokens, NUM_HEADS] float32
        //
        // Stage normalized BF16 to kv_bufs[0] for coalesced writes.
        // kv_bufs[0] capacity: V32=33792B, MODEL1=29696B; need HPB*D_V*2=16384B ✓

        float il0 = (sm.l_smem[gid] > 0.f) ? (1.f / sm.l_smem[gid]) : 0.f;
        float il1 = (sm.l_smem[gid + 8] > 0.f) ? (1.f / sm.l_smem[gid + 8]) : 0.f;

        bf16* staging_bf16 = reinterpret_cast<bf16*>(sm.kv_bufs[0]);
        constexpr int BF16_STAGING_STRIDE = D_V;

        #pragma unroll
        for (int t = 0; t < CT::ACC_TILES; t++) {
            constexpr int _NT8 = CT::NT_PER_WARP_XV * 8;
            int c = t / CT::NT_PER_WARP_XV, lnt = t % CT::NT_PER_WARP_XV;
            int d0 = c * CT::V_CHUNK + mwarp * _NT8 + lnt * 8 + tid * 2;
            staging_bf16[gid * BF16_STAGING_STRIDE + d0]       = __float2bfloat16(acc_o[t][0] * il0);
            staging_bf16[gid * BF16_STAGING_STRIDE + d0 + 1]   = __float2bfloat16(acc_o[t][1] * il0);
            staging_bf16[(gid+8) * BF16_STAGING_STRIDE + d0]   = __float2bfloat16(acc_o[t][2] * il1);
            staging_bf16[(gid+8) * BF16_STAGING_STRIDE + d0+1] = __float2bfloat16(acc_o[t][3] * il1);
        }

        if constexpr (KV::V_HAS_ROPE) {
            int n_start = mwarp * 8;
            int d0 = KV::D_NOPE + n_start + tid * 2;
            staging_bf16[gid * BF16_STAGING_STRIDE + d0]       = __float2bfloat16(acc_rope[0] * il0);
            staging_bf16[gid * BF16_STAGING_STRIDE + d0 + 1]   = __float2bfloat16(acc_rope[1] * il0);
            staging_bf16[(gid+8) * BF16_STAGING_STRIDE + d0]   = __float2bfloat16(acc_rope[2] * il1);
            staging_bf16[(gid+8) * BF16_STAGING_STRIDE + d0+1] = __float2bfloat16(acc_rope[3] * il1);
        }
        bar_sync_t<2, MATH_THREADS>();

        // Coalesced BF16 write: uint4 = 128-bit = 8 bf16 per store
        {
            constexpr size_t h_stride = D_V;
            constexpr size_t token_stride = (size_t)NUM_HEADS * D_V;
            const size_t out_base = (size_t)s_i * token_stride
                                  + (size_t)h_start * h_stride;
            constexpr int BF16_PER_STORE = 8;
            constexpr int STORES_PER_HEAD = D_V / BF16_PER_STORE;  // 64
            for (int idx = threadIdx.x; idx < HPB * STORES_PER_HEAD; idx += MATH_THREADS) {
                int h = idx / STORES_PER_HEAD;
                int d8 = (idx - h * STORES_PER_HEAD) * BF16_PER_STORE;
                uint4 v = *reinterpret_cast<const uint4*>(
                    &staging_bf16[h * BF16_STAGING_STRIDE + d8]);
                *reinterpret_cast<uint4*>(&output[out_base + h * h_stride + d8]) = v;
            }
        }

        // Write LSE
        if (threadIdx.x < HPB) {
            int h = threadIdx.x;
            float lse = softmax_lse(sm.m_smem[h], sm.l_smem[h]);
            size_t lse_idx = (size_t)s_i * NUM_HEADS + h_start + h;
            out_lse[lse_idx] = lse;
        }
    }
}

// ============================================================================
// Multi-Group (MG) Prefill Kernel — 2 head groups per CTA
//
// Processes 2×HPB = 32 heads per CTA. KV loaded once, reused for both groups.
// Key optimizations over SG:
//   - 2x KV reuse (V transpose shared, smem KV shared)
//   - Deferred row_sum (warp_l_partial in registers, reduce once at end)
//   - Better compute/load ratio → higher MMA utilization
//
// Used for NUM_HEADS > HPB (h=64, 128). h=16 stays on SG.
// ============================================================================

static constexpr int MG_N_HG = 2;
static constexpr int MG_HEADS_PER_CTA = MG_N_HG * HPB;  // 32

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
sparse_mla_prefill_mg_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    bf16* __restrict__ output,
    float* __restrict__ out_lse,
    __grid_constant__ const PrefillColdParams cold)
{
    const float sm_scale = cold.sm_scale;
    const int num_tokens = cold.num_tokens;
    constexpr int page_block_size = PAGE_BLOCK_SIZE;
    const size_t stride_kv_block = cold.stride_kv_block;
    using KV = KVCacheTraits<MT>;
    using CT = ComputeTraits<MT, CM>;
    using LMG = SmemLayoutMG<MT, CM>;
    using IO = KVIOTraits<MT>;
    using SMG = SmemPtrsMG<MT, CM>;

    static constexpr int NI = TOPK / BI;
    static constexpr int REPLICATE_H = NUM_HEADS / MG_HEADS_PER_CTA;
    static constexpr int QK_NOPE_KSTEPS = KV::QUANT_TILE / 32;

    const int s_i = blockIdx.x / REPLICATE_H;
    const int h_tile = blockIdx.x % REPLICATE_H;
    const int h_start = h_tile * MG_HEADS_PER_CTA;
    if (s_i >= num_tokens) return;

    const int warp_rank = threadIdx.x / 32;
    const int wy = warp_rank / 4;

    extern __shared__ char smem_raw[];
    auto sm = SMG::init(smem_raw);

    if (threadIdx.x == 0) {
        mbarrier_init(sm.mbar_kv + 0, 1);
        mbarrier_init(sm.mbar_kv + 1, 1);
    }
    bar_sync_t<3, BLOCK_THREADS>();

    // ── IO warps (identical to SG) ──────────────────────────────────
    if (wy == 2) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));

        const int io_tid = threadIdx.x - N_MATH_WARPS * 32;
        const int32_t* idx_base = indices + (size_t)s_i * TOPK;
        const uint64_t kv_l2_policy = create_l2_evict_first_policy();

        io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(
            sm.kv_bufs[0], idx_base, KV_cache, sm.mbar_kv + 0, io_tid,
            stride_kv_block, kv_l2_policy);
        io_gather_scales<MT, PAGE_BLOCK_SIZE>(
            sm.kv_scale_bufs[0], idx_base, KV_cache, io_tid, stride_kv_block);
        __threadfence_block();

        #pragma unroll 1
        for (int ti = 0; ti < NI; ti++) {
            if (ti + 1 < NI) {
                io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(
                    sm.kv_bufs[(ti + 1) & 1],
                    idx_base + (ti + 1) * BI, KV_cache,
                    sm.mbar_kv + ((ti + 1) & 1), io_tid,
                    stride_kv_block, kv_l2_policy);
                io_gather_scales<MT, PAGE_BLOCK_SIZE>(
                    sm.kv_scale_bufs[(ti + 1) & 1],
                    idx_base + (ti + 1) * BI, KV_cache, io_tid,
                    stride_kv_block);
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
        const int32_t* idx_base = indices + (size_t)s_i * TOPK;

        // ── Quantize Q for both groups (serial, reuse reduce_buf) ───
        #pragma unroll
        for (int g = 0; g < MG_N_HG; g++) {
            const bf16* q_base_g = Q + (size_t)s_i * NUM_HEADS * KV::D_QK
                                     + (size_t)(h_start + g * HPB) * KV::D_QK;
            quantize_q_to_smem<MT, MATH_THREADS>(
                sm.q_nope_fp8[g], sm.q_nope_sc[g],
                sm.q_rope + g * HPB * D_ROPE,
                q_base_g, sm.reduce_buf);
        }

        // Preload Q rope to registers for both groups
        QRopeRegs q_rope_regs[MG_N_HG];
        #pragma unroll
        for (int g = 0; g < MG_N_HG; g++)
            q_rope_regs[g] = preload_q_rope_regs(sm.q_rope + g * HPB * D_ROPE, lane);

        // Init per-group softmax state
        for (int i = threadIdx.x; i < MG_N_HG * HPB; i += MATH_THREADS) {
            sm.m_smem[i] = -1e30f;
            sm.l_smem[i] = 0.f;
        }

        // Per-group accumulators
        float acc_o[MG_N_HG][CT::ACC_TILES][4];
        #pragma unroll
        for (int g = 0; g < MG_N_HG; g++)
            #pragma unroll
            for (int t = 0; t < CT::ACC_TILES; t++)
                acc_o[g][t][0] = acc_o[g][t][1] = acc_o[g][t][2] = acc_o[g][t][3] = 0.f;

        float acc_rope[MG_N_HG][4];
        #pragma unroll
        for (int g = 0; g < MG_N_HG; g++)
            acc_rope[g][0] = acc_rope[g][1] = acc_rope[g][2] = acc_rope[g][3] = 0.f;

        // Deferred row_sum accumulators (register-only, no smem per tile)
        float warp_l_partial[MG_N_HG][2] = {};

        bar_sync_t<2, MATH_THREADS>();
        mbarrier_wait_parity(sm.mbar_kv + 0, 0);

        // ── Main loop ───────────────────────────────────────────────
        #pragma unroll 1
        for (int ti = 0; ti < NI; ti++) {
            uint8_t* kv_smem = sm.kv_bufs[ti & 1];
            const int32_t* ib = idx_base + ti * BI;
            const int qk_nb = mwarp * ENTRIES_PER_WARP;
            uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;

            // Entry base precomputation (shared across groups)
            const uint8_t* entry_base[ENTRIES_PER_WARP];
            if constexpr (KV::V_HAS_ROPE) {
                #pragma unroll
                for (int e = 0; e < ENTRIES_PER_WARP; e++) {
                    int idx = ib[qk_nb + e];
                    idx = (idx >= 0) ? idx : 0;
                    int bi_e = idx / page_block_size;
                    int li_e = idx % page_block_size;
                    entry_base[e] = KV_cache + (size_t)bi_e * stride_kv_block
                                             + (size_t)li_e * IO::IO_STRIDE;
                }
            } else {
                int idx = ib[qk_nb + gid];
                idx = (idx >= 0) ? idx : 0;
                entry_base[gid] = KV_cache + (size_t)idx * IO::IO_STRIDE;
            }

            // Prefetch KV rope (shared across groups — same entry)
            KVRopePrefetch rope_pf = prefetch_kv_rope(
                reinterpret_cast<const bf16*>(entry_base[gid] + KV::KV_ROPE_GMEM_OFFSET), lane);

            // Init per-group w_head_sc_all
            for (int i = threadIdx.x; i < MG_N_HG * CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = 0.f;

            // ── QK + softmax for both groups ────────────────────────
            float w_grp[MG_N_HG][4];
            float vsc_cache[MG_N_HG][CT::N_V_CHUNKS][2];

            #pragma unroll
            for (int g = 0; g < MG_N_HG; g++) {
                const uint8_t* kv_gid_base = kv_warp_base + gid * KV::KV_SMEM_STRIDE;

                // QK nope (block-scaled FP8 MMA)
                float qk[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
                    uint8_t sfa = fp32_to_ue8m0(
                        sm.q_nope_sc[g][(gid + (lane & 1) * 8) * KV::NUM_SCALES + blk]);
                    uint8_t sfb;
                    if constexpr (KV::SCALE_IN_KV_SMEM) {
                        sfb = fp32_to_ue8m0(
                            reinterpret_cast<const float*>(kv_gid_base + KV::D_NOPE)[blk]);
                    } else {
                        sfb = sm.kv_scale_bufs[ti & 1][(qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN + blk];
                    }
                    #pragma unroll
                    for (int ks = 0; ks < QK_NOPE_KSTEPS; ks++) {
                        int ko = blk * KV::QUANT_TILE + ks * 32;
                        uint32_t a0, a1, a2, a3, b0, b1;
                        ldmatrix_load_A_fp8(a0, a1, a2, a3,
                            sm.q_nope_fp8[g] + ko, KV::Q_NOPE_STRIDE, lane);
                        ldmatrix_load_B_fp8(b0, b1,
                            kv_warp_base + ko, KV::KV_SMEM_STRIDE, lane);
                        MmaFp8Result r = mma_fp8_block_scaled_m16n8k32(
                            a0, a1, a2, a3, b0, b1,
                            qk[0], qk[1], qk[2], qk[3], sfa, sfb);
                        qk[0] = r.d0; qk[1] = r.d1; qk[2] = r.d2; qk[3] = r.d3;
                    }
                }

                // QK rope (reuses prefetched B operands)
                compute_qk_rope(qk, q_rope_regs[g], rope_pf);

                // Invalid index masking
                {
                    int e0 = qk_nb + tid * 2, e1 = e0 + 1;
                    if (ib[e0] < 0) { qk[0] = -1e30f; qk[2] = -1e30f; }
                    if (ib[e1] < 0) { qk[1] = -1e30f; qk[3] = -1e30f; }
                }

                float s[4] = { qk[0] * sm_scale_log2e, qk[1] * sm_scale_log2e,
                               qk[2] * sm_scale_log2e, qk[3] * sm_scale_log2e };

                // Warp max → reduce_buf (per-group offset)
                float lm0, lm1;
                softmax_warp_max(s, lm0, lm1);
                if (tid == 0) {
                    sm.reduce_buf[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid] = lm0;
                    sm.reduce_buf[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid + 8] = lm1;
                }
                // Store s for later use
                w_grp[g][0] = s[0]; w_grp[g][1] = s[1];
                w_grp[g][2] = s[2]; w_grp[g][3] = s[3];
            }
            bar_sync_t<2, MATH_THREADS>();

            // Cross-warp max for both groups
            if (threadIdx.x < MG_N_HG * HPB) {
                int g = threadIdx.x / HPB, h = threadIdx.x % HPB;
                float old_m = sm.m_smem[g * SMG::ML_GRP_STRIDE + h], tm = -1e30f;
                #pragma unroll
                for (int w = 0; w < N_MATH_WARPS; w++)
                    tm = fmaxf(tm, sm.reduce_buf[g * SMG::REDUCE_GRP_STRIDE + w * HPB + h]);
                float nm = fmaxf(old_m, tm);
                float alpha = exp2f(old_m - nm);
                sm.m_smem[g * SMG::ML_GRP_STRIDE + h] = nm;
                sm.l_smem[g * SMG::ML_GRP_STRIDE + h] *= alpha;
                sm.reduce_buf[g * SMG::REDUCE_GRP_STRIDE + h] = alpha;
                sm.reduce_buf[g * SMG::REDUCE_GRP_STRIDE + HPB + h] = nm;
            }
            bar_sync_t<2, MATH_THREADS>();

            // Rescale + exp weights for both groups
            #pragma unroll
            for (int g = 0; g < MG_N_HG; g++) {
                float alpha0 = sm.reduce_buf[g * SMG::REDUCE_GRP_STRIDE + gid];
                float alpha1 = sm.reduce_buf[g * SMG::REDUCE_GRP_STRIDE + gid + 8];
                float nm0 = sm.reduce_buf[g * SMG::REDUCE_GRP_STRIDE + HPB + gid];
                float nm1 = sm.reduce_buf[g * SMG::REDUCE_GRP_STRIDE + HPB + gid + 8];

                #pragma unroll
                for (int t = 0; t < CT::ACC_TILES; t++) {
                    acc_o[g][t][0] *= alpha0; acc_o[g][t][1] *= alpha0;
                    acc_o[g][t][2] *= alpha1; acc_o[g][t][3] *= alpha1;
                }
                if constexpr (KV::V_HAS_ROPE) {
                    acc_rope[g][0] *= alpha0; acc_rope[g][1] *= alpha0;
                    acc_rope[g][2] *= alpha1; acc_rope[g][3] *= alpha1;
                }

                float w0 = exp2f(w_grp[g][0] - nm0), w1 = exp2f(w_grp[g][1] - nm0);
                float w2 = exp2f(w_grp[g][2] - nm1), w3 = exp2f(w_grp[g][3] - nm1);
                w_grp[g][0] = w0; w_grp[g][1] = w1;
                w_grp[g][2] = w2; w_grp[g][3] = w3;

                // Deferred row_sum: accumulate in registers
                float ls0, ls1;
                softmax_warp_sum(w0, w1, w2, w3, ls0, ls1);
                warp_l_partial[g][0] += ls0;
                warp_l_partial[g][1] += ls1;

                // V scale cache + atomicMax
                const int e0i = qk_nb + tid * 2, e1i = e0i + 1;
                const uint8_t* e0_base = kv_warp_base + tid * 2 * KV::KV_SMEM_STRIDE;
                const uint8_t* e1_base = e0_base + KV::KV_SMEM_STRIDE;
                #pragma unroll
                for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
                    if constexpr (KV::SCALE_IN_KV_SMEM) {
                        vsc_cache[g][vc][0] = reinterpret_cast<const float*>(e0_base + KV::D_NOPE)[vc];
                        vsc_cache[g][vc][1] = reinterpret_cast<const float*>(e1_base + KV::D_NOPE)[vc];
                    } else {
                        vsc_cache[g][vc][0] = ue8m0_to_fp32(sm.kv_scale_bufs[ti & 1][e0i * KV::SCALE_BYTES_PER_TOKEN + vc]);
                        vsc_cache[g][vc][1] = ue8m0_to_fp32(sm.kv_scale_bufs[ti & 1][e1i * KV::SCALE_BYTES_PER_TOKEN + vc]);
                    }
                    float ws00 = w0 * vsc_cache[g][vc][0], ws01 = w1 * vsc_cache[g][vc][1];
                    float ws10 = w2 * vsc_cache[g][vc][0], ws11 = w3 * vsc_cache[g][vc][1];
                    atomicMax(reinterpret_cast<int*>(&sm.w_head_sc_all[g * SMG::WSC_GRP_STRIDE + vc * HPB + gid]),
                        __float_as_int(fmaxf(fabsf(ws00), fabsf(ws01))));
                    atomicMax(reinterpret_cast<int*>(&sm.w_head_sc_all[g * SMG::WSC_GRP_STRIDE + vc * HPB + gid + 8]),
                        __float_as_int(fmaxf(fabsf(ws10), fabsf(ws11))));
                }
            }
            bar_sync_t<2, MATH_THREADS>();

            // Normalize w_head_sc_all (both groups)
            for (int i = threadIdx.x; i < MG_N_HG * CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = fmaxf(sm.w_head_sc_all[i], 1e-10f) / FP8_MAX;
            bar_sync_t<2, MATH_THREADS>();

            // ── XV nope MMA (V transpose shared) ────────────────────
            {
                // Prologue: transpose chunk 0 (once, shared)
                transpose_v_chunk<MT, CM>(sm.v_trans, kv_smem, 0);

                #pragma unroll
                for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
                    // W quantize for both groups → separate w_fp8 buffers
                    #pragma unroll
                    for (int g = 0; g < MG_N_HG; g++) {
                        float* vc_sc = sm.w_head_sc_all + g * SMG::WSC_GRP_STRIDE + vc * HPB;
                        uint8_t* cur_wfp8 = sm.w_fp8 + g * SMG::WFP8_GRP_SIZE;
                        float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
                        float w0 = w_grp[g][0], w1 = w_grp[g][1];
                        float w2 = w_grp[g][2], w3 = w_grp[g][3];
                        float vsc0 = vsc_cache[g][vc][0], vsc1 = vsc_cache[g][vc][1];
                        float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
                        float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
                        __nv_fp8_e4m3 f00(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws00 * si0)));
                        __nv_fp8_e4m3 f01(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws01 * si0)));
                        __nv_fp8_e4m3 f10(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws10 * si1)));
                        __nv_fp8_e4m3 f11(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws11 * si1)));
                        const int e0i = qk_nb + tid * 2, e1i = e0i + 1;
                        cur_wfp8[gid * (BI + 16) + e0i] = f00.__x;
                        cur_wfp8[gid * (BI + 16) + e1i] = f01.__x;
                        cur_wfp8[(gid + 8) * (BI + 16) + e0i] = f10.__x;
                        cur_wfp8[(gid + 8) * (BI + 16) + e1i] = f11.__x;
                    }

                    bar_sync_t<2, MATH_THREADS>();

                    // MMA for both groups (shared v_trans, per-group w_fp8)
                    #pragma unroll
                    for (int g = 0; g < MG_N_HG; g++) {
                        float* vc_sc = sm.w_head_sc_all + g * SMG::WSC_GRP_STRIDE + vc * HPB;
                        uint8_t* cur_wfp8 = sm.w_fp8 + g * SMG::WFP8_GRP_SIZE;
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
                                    cur_wfp8 + ko, BI + 16, lane);
                                ldmatrix_load_B_fp8(b0, b1,
                                    sm.v_trans + nl * CT::V_TRANS_STRIDE + ko,
                                    CT::V_TRANS_STRIDE, lane);
                                MmaFp8Result r = mma_fp8_m16n8k32(
                                    a0, a1, a2, a3, b0, b1,
                                    xv[0], xv[1], xv[2], xv[3]);
                                xv[0] = r.d0; xv[1] = r.d1; xv[2] = r.d2; xv[3] = r.d3;
                            }
                            float sc0 = vc_sc[gid], sc1 = vc_sc[gid + 8];
                            acc_o[g][ti_acc][0] += xv[0] * sc0; acc_o[g][ti_acc][1] += xv[1] * sc0;
                            acc_o[g][ti_acc][2] += xv[2] * sc1; acc_o[g][ti_acc][3] += xv[3] * sc1;
                        }
                    }

                    if (vc < CT::N_V_CHUNKS - 1) {
                        transpose_v_chunk<MT, CM>(sm.v_trans, kv_smem, (vc + 1) * CT::V_CHUNK);
                        bar_sync_t<2, MATH_THREADS>();
                    }
                }
            }

            // ── XV rope BF16 MMA (MODEL1, both groups) ──────────────
            if constexpr (KV::V_HAS_ROPE) {
                bar_sync_t<2, MATH_THREADS>();
                #pragma unroll
                for (int g = 0; g < MG_N_HG; g++) {
                    xv_rope_mma<MT, PAGE_BLOCK_SIZE>(
                        acc_rope[g], w_grp[g][0], w_grp[g][1], w_grp[g][2], w_grp[g][3],
                        ib, KV_cache, mwarp, lane, stride_kv_block,
                        reinterpret_cast<bf16*>(sm.v_trans));
                }
            }

            bar_arrive_t<1, BLOCK_THREADS>();
            if (ti + 1 < NI) {
                const int next_phase = ((ti + 1) >> 1) & 1;
                mbarrier_wait_parity(sm.mbar_kv + ((ti + 1) & 1), next_phase);
            }
        }

        // ── Finalize deferred row_sum ───────────────────────────────
        // Write warp_l_partial to smem for cross-warp reduction
        #pragma unroll
        for (int g = 0; g < MG_N_HG; g++) {
            if (tid == 0) {
                sm.reduce_buf[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid] = warp_l_partial[g][0];
                sm.reduce_buf[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid + 8] = warp_l_partial[g][1];
            }
        }
        bar_sync_t<2, MATH_THREADS>();

        if (threadIdx.x < MG_N_HG * HPB) {
            int g = threadIdx.x / HPB, h = threadIdx.x % HPB;
            float ts = 0.f;
            #pragma unroll
            for (int w = 0; w < N_MATH_WARPS; w++)
                ts += sm.reduce_buf[g * SMG::REDUCE_GRP_STRIDE + w * HPB + h];
            sm.l_smem[g * SMG::ML_GRP_STRIDE + h] += ts;
        }
        bar_sync_t<2, MATH_THREADS>();

        // ── Epilogue: BF16 output for both groups (serial) ─────────
        // Reuse kv_bufs[0] for BF16 staging (16KB needed, 29-33KB available)
        bf16* staging_bf16 = reinterpret_cast<bf16*>(sm.kv_bufs[0]);
        constexpr int BF16_STAGING_STRIDE = D_V;
        constexpr size_t h_stride = D_V;
        constexpr size_t token_stride = (size_t)NUM_HEADS * D_V;

        #pragma unroll
        for (int g = 0; g < MG_N_HG; g++) {
            float il0 = (sm.l_smem[g * SMG::ML_GRP_STRIDE + gid] > 0.f)
                ? (1.f / sm.l_smem[g * SMG::ML_GRP_STRIDE + gid]) : 0.f;
            float il1 = (sm.l_smem[g * SMG::ML_GRP_STRIDE + gid + 8] > 0.f)
                ? (1.f / sm.l_smem[g * SMG::ML_GRP_STRIDE + gid + 8]) : 0.f;

            #pragma unroll
            for (int t = 0; t < CT::ACC_TILES; t++) {
                constexpr int _NT8 = CT::NT_PER_WARP_XV * 8;
                int c = t / CT::NT_PER_WARP_XV, lnt = t % CT::NT_PER_WARP_XV;
                int d0 = c * CT::V_CHUNK + mwarp * _NT8 + lnt * 8 + tid * 2;
                staging_bf16[gid * BF16_STAGING_STRIDE + d0]       = __float2bfloat16(acc_o[g][t][0] * il0);
                staging_bf16[gid * BF16_STAGING_STRIDE + d0 + 1]   = __float2bfloat16(acc_o[g][t][1] * il0);
                staging_bf16[(gid+8) * BF16_STAGING_STRIDE + d0]   = __float2bfloat16(acc_o[g][t][2] * il1);
                staging_bf16[(gid+8) * BF16_STAGING_STRIDE + d0+1] = __float2bfloat16(acc_o[g][t][3] * il1);
            }

            if constexpr (KV::V_HAS_ROPE) {
                int n_start = mwarp * 8;
                int d0 = KV::D_NOPE + n_start + tid * 2;
                staging_bf16[gid * BF16_STAGING_STRIDE + d0]       = __float2bfloat16(acc_rope[g][0] * il0);
                staging_bf16[gid * BF16_STAGING_STRIDE + d0 + 1]   = __float2bfloat16(acc_rope[g][1] * il0);
                staging_bf16[(gid+8) * BF16_STAGING_STRIDE + d0]   = __float2bfloat16(acc_rope[g][2] * il1);
                staging_bf16[(gid+8) * BF16_STAGING_STRIDE + d0+1] = __float2bfloat16(acc_rope[g][3] * il1);
            }
            bar_sync_t<2, MATH_THREADS>();

            // Coalesced write
            {
                const int g_h_start = h_start + g * HPB;
                const size_t out_base = (size_t)s_i * token_stride + (size_t)g_h_start * h_stride;
                constexpr int BF16_PER_STORE = 8;
                constexpr int STORES_PER_HEAD = D_V / BF16_PER_STORE;
                for (int idx = threadIdx.x; idx < HPB * STORES_PER_HEAD; idx += MATH_THREADS) {
                    int h = idx / STORES_PER_HEAD;
                    int d8 = (idx - h * STORES_PER_HEAD) * BF16_PER_STORE;
                    uint4 v = *reinterpret_cast<const uint4*>(
                        &staging_bf16[h * BF16_STAGING_STRIDE + d8]);
                    *reinterpret_cast<uint4*>(&output[out_base + h * h_stride + d8]) = v;
                }
            }

            // Write LSE for this group
            if (threadIdx.x < HPB) {
                int h = threadIdx.x;
                float lse = softmax_lse(sm.m_smem[g * SMG::ML_GRP_STRIDE + h],
                                         sm.l_smem[g * SMG::ML_GRP_STRIDE + h]);
                size_t lse_idx = (size_t)s_i * NUM_HEADS + (h_start + g * HPB + h);
                out_lse[lse_idx] = lse;
            }

            if (g < MG_N_HG - 1)
                bar_sync_t<2, MATH_THREADS>();
        }
    }
}
