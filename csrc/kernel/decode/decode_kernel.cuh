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
#include "../common/d2_load_b.cuh"
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

struct DecodeColdParams {
    float sm_scale;
    int num_tokens;
    size_t stride_kv_block;
};

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int TILES_PER_SPLIT, int PAGE_BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
sparse_mla_decode_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    float* __restrict__ partial_O,
    float* __restrict__ partial_LSE,
    __grid_constant__ const DecodeColdParams cold)
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
    static constexpr int NSPLITS = NI / TILES_PER_SPLIT;
    static constexpr int REPLICATE_H = (NUM_HEADS + HPB - 1) / HPB;
    static constexpr int QK_NOPE_KSTEPS = KV::QUANT_TILE / 32;

    static constexpr int VALID_HPB = (NUM_HEADS < HPB) ? NUM_HEADS : HPB;
    static_assert(NUM_HEADS % VALID_HPB == 0, "NUM_HEADS must be a multiple of VALID_HPB");

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
        const uint64_t kv_l2_policy = create_l2_evict_first_policy();

        // Prologue: gather tile 0
        io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(
            sm.kv_bufs[0], idx_base, KV_cache, sm.mbar_kv + 0, io_tid,
            stride_kv_block, kv_l2_policy);
        io_gather_scales<MT, PAGE_BLOCK_SIZE>(
            sm.kv_scale_bufs[0], idx_base, KV_cache, io_tid,
            stride_kv_block);
        __threadfence_block();

        #pragma unroll
        for (int ti = 0; ti < TILES_PER_SPLIT; ti++) {
            if (ti + 1 < TILES_PER_SPLIT) {
                io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(
                    sm.kv_bufs[(ti + 1) & 1],
                    idx_base + (ti + 1) * BI, KV_cache,
                    sm.mbar_kv + ((ti + 1) & 1), io_tid,
                    stride_kv_block, kv_l2_policy);
                io_gather_scales<MT, PAGE_BLOCK_SIZE>(
                    sm.kv_scale_bufs[(ti + 1) & 1],
                    idx_base + (ti + 1) * BI, KV_cache, io_tid,
                    stride_kv_block);
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
            sm.q_nope_fp8, sm.q_nope_sc, sm.q_rope, q_base, sm.reduce_buf, VALID_HPB);
        QRopeRegs q_rope_regs = preload_q_rope_regs(sm.q_rope, lane);

        for (int h = threadIdx.x; h < HPB; h += MATH_THREADS)
            sm.m_smem[h] = -1e30f;

        float acc_o[CT::ACC_TILES][4];
        #pragma unroll
        for (int t = 0; t < CT::ACC_TILES; t++)
            acc_o[t][0] = acc_o[t][1] = acc_o[t][2] = acc_o[t][3] = 0.f;

        float acc_rope[4] = {0.f, 0.f, 0.f, 0.f};
        float warp_l[2] = {0.f, 0.f};

        bar_sync_t<2, MATH_THREADS>();
        mbarrier_wait_parity(sm.mbar_kv + 0, 0);

        // ── Main loop — QK + softmax + XV ───────────────────────────
        #pragma unroll
        for (int ti = 0; ti < TILES_PER_SPLIT; ti++) {
            uint8_t* kv_smem = sm.kv_bufs[ti & 1];
            const int32_t* ib = idx_base + ti * BI;
            const int qk_nb = mwarp * ENTRIES_PER_WARP;
            uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;

            // Precompute entry base pointers for global KV access.
            // V32 (flat addressing): only gid's entry needed (for QK rope).
            // MODEL1 (block-structured): all 8 entries precomputed (div/mod expensive).
            const uint8_t* entry_base[ENTRIES_PER_WARP];
            if constexpr (KV::V_HAS_ROPE) {
                // MODEL1: precompute all 8 — used by QK rope + XV rope
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
                // V32: only precompute gid's entry (QK rope needs only this one)
                int idx = ib[qk_nb + gid];
                idx = (idx >= 0) ? idx : 0;
                entry_base[gid] = KV_cache + (size_t)idx * IO::IO_STRIDE;
            }

            for (int i = threadIdx.x; i < CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = 0.f;

            // Prefetch KV rope B operands into registers — loads issue now,
            // data arrives during the ~16 QK nope MMAs below (~300 cycle overlap).
            KVRopePrefetch rope_pf = prefetch_kv_rope(
                reinterpret_cast<const bf16*>(entry_base[gid] + KV::KV_ROPE_GMEM_OFFSET), lane);

            // ── QK nope (block-scaled FP8 MMA) ─────────────────────
            // sfa: UE8M0 Q scale for M-row = gid + (lane&1)*8
            // sfb: UE8M0 K scale for N-row = gid (entry gid in this warp)
            // Hardware applies: D[m][n] = scale_A[m] * scale_B[n] * (A×B) + C
            float qk[4] = {0.f, 0.f, 0.f, 0.f};
            // Precompute gid's scale buffer base for this warp
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

            // ── Online softmax (deferred sum, conditional rescale) ──
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
                sm.reduce_buf[h] = alpha;
                sm.reduce_buf[HPB + h] = nm;
            }
            bar_sync_t<2, MATH_THREADS>();

            float alpha0 = sm.reduce_buf[gid], alpha1 = sm.reduce_buf[gid + 8];
            float nm0 = sm.reduce_buf[HPB + gid], nm1 = sm.reduce_buf[HPB + gid + 8];

            if (alpha0 < 1.0f || alpha1 < 1.0f) {
                #pragma unroll
                for (int t = 0; t < CT::ACC_TILES; t++) {
                    acc_o[t][0] *= alpha0; acc_o[t][1] *= alpha0;
                    acc_o[t][2] *= alpha1; acc_o[t][3] *= alpha1;
                }
                if constexpr (KV::V_HAS_ROPE) {
                    acc_rope[0] *= alpha0; acc_rope[1] *= alpha0;
                    acc_rope[2] *= alpha1; acc_rope[3] *= alpha1;
                }
                warp_l[0] *= alpha0;
                warp_l[1] *= alpha1;
            }

            float w0 = exp2f(s[0] - nm0), w1 = exp2f(s[1] - nm0);
            float w2 = exp2f(s[2] - nm1), w3 = exp2f(s[3] - nm1);

            float ls0, ls1;
            softmax_warp_sum(w0, w1, w2, w3, ls0, ls1);
            warp_l[0] += ls0;
            warp_l[1] += ls1;

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

            for (int i = threadIdx.x; i < CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = fmaxf(sm.w_head_sc_all[i], 1e-10f) / FP8_MAX;
            bar_sync_t<2, MATH_THREADS>();

            // ── XV nope (batch W quant + D2 direct B) ────────────────
            {
                const int e0i = qk_nb + tid * 2, e1i = e0i + 1;

                // Batch W quant: all V chunks at once, single barrier
                #pragma unroll
                for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
                    float* vc_sc = sm.w_head_sc_all + vc * HPB;
                    uint8_t* wfp8 = sm.w_fp8 + vc * L::SMEM_W_FP8_ONE;
                    float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
                    float vsc0 = vsc_cache[vc][0], vsc1 = vsc_cache[vc][1];
                    float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
                    float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
                    __nv_fp8_e4m3 f00(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws00 * si0)));
                    __nv_fp8_e4m3 f01(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws01 * si0)));
                    __nv_fp8_e4m3 f10(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws10 * si1)));
                    __nv_fp8_e4m3 f11(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws11 * si1)));
                    wfp8[gid * (BI + 16) + e0i] = f00.__x;
                    wfp8[gid * (BI + 16) + e1i] = f01.__x;
                    wfp8[(gid + 8) * (BI + 16) + e0i] = f10.__x;
                    wfp8[(gid + 8) * (BI + 16) + e1i] = f11.__x;
                }
                bar_sync_t<2, MATH_THREADS>();

                // Batch MMA: all V chunks, no barriers between
                #pragma unroll
                for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
                    float* vc_sc = sm.w_head_sc_all + vc * HPB;
                    uint8_t* wfp8 = sm.w_fp8 + vc * L::SMEM_W_FP8_ONE;
                    #pragma unroll
                    for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
                        int ti_acc = vc * CT::NT_PER_WARP_XV + nt;
                        int dim = vc * CT::V_CHUNK + mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
                        float xv[4] = {0.f, 0.f, 0.f, 0.f};
                        #pragma unroll
                        for (int kstep = 0; kstep < CT::XV_KSTEPS; kstep++) {
                            int ko = kstep * 32;
                            uint32_t a0, a1, a2, a3, b0, b1;
                            ldmatrix_load_A_fp8(a0, a1, a2, a3,
                                wfp8 + ko, BI + 16, lane);
                            d2_load_b_fp8<KV::KV_SMEM_STRIDE>(b0, b1,
                                kv_smem, kstep * 32, dim, lane);
                            MmaFp8Result r = mma_fp8_m16n8k32(
                                a0, a1, a2, a3, b0, b1,
                                xv[0], xv[1], xv[2], xv[3]);
                            xv[0] = r.d0; xv[1] = r.d1; xv[2] = r.d2; xv[3] = r.d3;
                        }
                        float sc0 = vc_sc[gid], sc1 = vc_sc[gid + 8];
                        acc_o[ti_acc][0] += xv[0] * sc0; acc_o[ti_acc][1] += xv[1] * sc0;
                        acc_o[ti_acc][2] += xv[2] * sc1; acc_o[ti_acc][3] += xv[3] * sc1;
                    }
                }
                // Sync before next tile's w_head_sc_all zeroing (V32 has no rope barrier)
                if constexpr (!KV::V_HAS_ROPE)
                    bar_sync_t<2, MATH_THREADS>();
            }

            // ── XV rope BF16 MMA (MODEL1 only) ─────────────────────
            // Scratch overlays on w_fp8_bufs (dead after XV nope phase)
            if constexpr (KV::V_HAS_ROPE) {
                bar_sync_t<2, MATH_THREADS>();
                xv_rope_mma<MT, PAGE_BLOCK_SIZE>(acc_rope, w0, w1, w2, w3,
                    ib, KV_cache, mwarp, lane,
                    stride_kv_block,
                    reinterpret_cast<bf16*>(sm.w_fp8));
            }

            bar_arrive_t<1, BLOCK_THREADS>();
            if (ti + 1 < TILES_PER_SPLIT) {
                const int next_phase = ((ti + 1) >> 1) & 1;
                mbarrier_wait_parity(sm.mbar_kv + ((ti + 1) & 1), next_phase);
            }
        }

        // ── Finalize deferred row_sum ────────────────────────────────
        if (tid == 0) {
            sm.reduce_buf[mwarp * HPB + gid] = warp_l[0];
            sm.reduce_buf[mwarp * HPB + gid + 8] = warp_l[1];
        }
        bar_sync_t<2, MATH_THREADS>();
        if (threadIdx.x < HPB) {
            int h = threadIdx.x;
            float ts = 0.f;
            #pragma unroll
            for (int w = 0; w < N_MATH_WARPS; w++)
                ts += sm.reduce_buf[w * HPB + h];
            sm.l_smem[h] = ts;
        }
        bar_sync_t<2, MATH_THREADS>();

        // ── Write partial_O (float32) and partial_LSE ────────────────
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
            constexpr int FLOATS_PER_WIDE_STORE = 8;  // v8.b32 = 256-bit for L2::evict_last
            constexpr int WIDE_STORES_PER_HEAD = D_V / FLOATS_PER_WIDE_STORE;  // 64
            for (int idx = threadIdx.x; idx < VALID_HPB * WIDE_STORES_PER_HEAD; idx += MATH_THREADS) {
                int h = idx / WIDE_STORES_PER_HEAD;
                int d8 = (idx - h * WIDE_STORES_PER_HEAD) * FLOATS_PER_WIDE_STORE;
                float4 v0 = *reinterpret_cast<const float4*>(
                    &staging_f32[h * F32_STAGING_STRIDE + d8]);
                float4 v1 = *reinterpret_cast<const float4*>(
                    &staging_f32[h * F32_STAGING_STRIDE + d8 + 4]);
                store_8f_evict_last(&partial_O[po_base + h * h_stride + d8], v0, v1);
            }
        }

        // Write partial_LSE
        if (threadIdx.x < VALID_HPB) {
            int h = threadIdx.x;
            float lse = softmax_lse(sm.m_smem[h], sm.l_smem[h]);
            constexpr size_t lse_split_stride = (size_t)NUM_HEADS;
            constexpr size_t lse_token_stride = (size_t)NSPLITS * lse_split_stride;
            size_t lse_idx = (size_t)s_i * lse_token_stride
                           + (size_t)split_idx * lse_split_stride
                           + (h_start + h);
            partial_LSE[lse_idx] = lse;
        }

        cudaTriggerProgrammaticLaunchCompletion();
    }
}
