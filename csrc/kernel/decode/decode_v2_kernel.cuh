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
#include "../sched/sched_params.h"

// ============================================================================
// Sparse MLA Decode V2 — scheduler-driven split-KV decode
//
// Replaces v1's fixed (token × REPLICATE_H, NSPLITS) grid with
// scheduler-driven (REPLICATE_H, s_q, num_sm_parts) grid.
//
// Key differences from v1:
//   - TILES_PER_SPLIT removed (runtime from scheduler metadata)
//   - One CTA may process blocks from multiple batch elements
//   - Supports is_no_split direct bf16 output (skip combine)
//   - Per-batch split indexing via num_splits_ptr prefix sum
// ============================================================================

struct DecodeV2ColdParams {
    float sm_scale;
    int num_batches;
    int s_q;
    size_t stride_kv_block;
    int topk;
    // o_accum strides: [total_splits, s_q, NUM_HEADS, D_V]
    size_t stride_oa_split;   // s_q * NUM_HEADS * D_V
    size_t stride_oa_sq;      // NUM_HEADS * D_V
    // lse_accum strides: [total_splits, s_q, NUM_HEADS]
    size_t stride_la_split;   // s_q * NUM_HEADS
    size_t stride_la_sq;      // NUM_HEADS
    const float* attn_sink;   // [NUM_HEADS] float32, nullptr = disabled
    // V4 features (MODEL1 only)
    const int* topk_length;        // [num_batches] int32, nullptr = uniform topk
    int extra_topk;                // 0 = no extra cache
    const int* extra_topk_length;  // [num_batches] int32, nullptr = uniform extra_topk
    size_t stride_extra_kv_block;  // extra cache block stride
};

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE,
          bool BF16_QK = KVCacheTraits<MT>::USE_BF16_QK>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
sparse_mla_decode_v2_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ extra_KV_cache,
    const int32_t* __restrict__ extra_indices,
    float* __restrict__ o_accum,
    float* __restrict__ lse_accum,
    bf16* __restrict__ output,
    float* __restrict__ out_lse,
    const DecodingSchedMeta* __restrict__ sched_meta,
    const int* __restrict__ num_splits_ptr,
    __grid_constant__ const DecodeV2ColdParams cold)
{
    const float sm_scale = cold.sm_scale;
    const int num_batches = cold.num_batches;
    const int s_q = cold.s_q;
    constexpr int page_block_size = PAGE_BLOCK_SIZE;
    const size_t stride_kv_block = cold.stride_kv_block;
    const int topk = cold.topk;
    using KV = KVCacheTraits<MT>;
    using CT = ComputeTraits<MT, CM>;
    using L = SmemLayout<MT, CM, BF16_QK>;
    using IO = KVIOTraits<MT>;

    static constexpr int REPLICATE_H = (NUM_HEADS + HPB - 1) / HPB;
    static constexpr int QK_NOPE_KSTEPS = KV::QUANT_TILE / 32;
    static constexpr int VALID_HPB = (NUM_HEADS < HPB) ? NUM_HEADS : HPB;

    const int h_tile = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int h_start = h_tile * HPB;

    const DecodingSchedMeta meta = sched_meta[partition_idx];

    const int warp_rank = threadIdx.x / 32;
    const int wy = warp_rank / 4;

    extern __shared__ char smem_raw[];
    auto sm = SmemPtrs<MT, CM, BF16_QK>::init(smem_raw);

    if (threadIdx.x == 0) {
        mbarrier_init(sm.mbar_kv + 0, 1);
        mbarrier_init(sm.mbar_kv + 1, 1);
    }
    bar_sync_t<3, BLOCK_THREADS>();

    // ── Batch loop driven by scheduler metadata ─────────────────────
    for (int req = meta.begin_req_idx; req <= meta.end_req_idx && req < num_batches; req++) {
        const int s_i = req * s_q + s_q_idx;

        // Per-batch topk_length and extra_topk (match FlashMLA: orig_topk_padded = max(ceil(tl, BI), BI))
        const int topk_len = cold.topk_length ? __ldg(cold.topk_length + req) : topk;
        const int orig_topk_padded = max(((topk_len + BI - 1) / BI) * BI, BI);
        const int num_orig_blocks = orig_topk_padded / BI;
        const int extra_topk_len = (cold.extra_topk > 0)
            ? (cold.extra_topk_length ? __ldg(cold.extra_topk_length + req) : cold.extra_topk) : 0;
        const int total_blocks = (orig_topk_padded + ((extra_topk_len + BI - 1) / BI) * BI) / BI;

        const int start_block = (req == meta.begin_req_idx) ? meta.begin_block_idx : 0;
        const int end_block = (req == meta.end_req_idx) ? meta.end_block_idx : total_blocks;
        const int num_tiles = end_block - start_block;
        if (num_tiles <= 0) continue;

        const bool is_no_split = (req == meta.begin_req_idx)
            ? !meta.is_first_req_splitted
            : ((req == meta.end_req_idx) ? !meta.is_last_req_splitted : true);
        const int split_offset = (req == meta.begin_req_idx) ? meta.begin_split_idx : 0;
        const int split_idx = num_splits_ptr[req] + split_offset;

        // Reinit mbarriers between batch elements
        if (req != meta.begin_req_idx) {
            if (threadIdx.x == 0) {
                mbarrier_inval(sm.mbar_kv + 0);
                mbarrier_inval(sm.mbar_kv + 1);
                mbarrier_init(sm.mbar_kv + 0, 1);
                mbarrier_init(sm.mbar_kv + 1, 1);
            }
            bar_sync_t<3, BLOCK_THREADS>();
        }

    // ── IO warps ────────────────────────────────────────────────────
    if (wy == 2) {
        if (req == meta.begin_req_idx)
            asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));

        const int io_tid = threadIdx.x - N_MATH_WARPS * 32;
        const uint64_t kv_l2_policy = create_l2_evict_first_policy();

        // Per-tile IO gather — select main vs extra cache based on block index
        auto io_gather_one = [&](int global_blk, int buf_idx) {
            const int32_t* idx_ptr;
            const uint8_t* kv_ptr;
            size_t kv_str;
            if (global_blk < num_orig_blocks) {
                idx_ptr = indices + (size_t)s_i * topk + (size_t)global_blk * BI;
                kv_ptr = KV_cache; kv_str = stride_kv_block;
            } else {
                int eb = global_blk - num_orig_blocks;
                idx_ptr = extra_indices + (size_t)s_i * cold.extra_topk + (size_t)eb * BI;
                kv_ptr = extra_KV_cache; kv_str = cold.stride_extra_kv_block;
            }
            io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(
                sm.kv_bufs[buf_idx], idx_ptr, kv_ptr,
                sm.mbar_kv + buf_idx, io_tid, kv_str, kv_l2_policy);
            io_gather_scales<MT, PAGE_BLOCK_SIZE>(
                sm.kv_scale_bufs[buf_idx], idx_ptr, kv_ptr, io_tid, kv_str);
        };

        // Prologue: gather tile 0
        io_gather_one(start_block, 0);
        __threadfence_block();

        #pragma unroll 1
        for (int ti = 0; ti < num_tiles; ti++) {
            if (ti + 1 < num_tiles) {
                io_gather_one(start_block + ti + 1, (ti + 1) & 1);
                __threadfence_block();
            }
            bar_sync_t<1, BLOCK_THREADS>();
        }

    // ── Math warps ──────────────────────────────────────────────────
    } else {
        if (req == meta.begin_req_idx)
            asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(232));

        const int lane = threadIdx.x & 31;
        const int mwarp = warp_rank;
        const int gid = lane >> 2, tid = lane & 3;
        const float sm_scale_log2e = sm_scale * LOG2E;
        const bf16* q_base = Q + (size_t)s_i * NUM_HEADS * KV::D_QK + (size_t)h_start * KV::D_QK;

        if constexpr (BF16_QK) {
            load_q_bf16_to_smem<MT, MATH_THREADS>(
                sm.q_nope_bf16, sm.q_rope, q_base, VALID_HPB);
        } else {
            quantize_q_to_smem<MT, MATH_THREADS>(
                sm.q_nope_fp8, sm.q_nope_sc, sm.q_rope, q_base, sm.reduce_buf, VALID_HPB);
        }
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

        // ── Main loop — QK + softmax + XV ────────────────────────────
        #pragma unroll 1
        for (int ti = 0; ti < num_tiles; ti++) {
            uint8_t* kv_smem = sm.kv_bufs[ti & 1];
            const int global_blk = start_block + ti;
            const bool is_extra_tile = (global_blk >= num_orig_blocks);

            // Per-tile index base and KV cache pointer (main vs extra)
            const int32_t* ib;
            const uint8_t* tile_kv_cache;
            size_t tile_stride;
            if (!is_extra_tile) {
                ib = indices + (size_t)s_i * topk + (size_t)global_blk * BI;
                tile_kv_cache = KV_cache;
                tile_stride = stride_kv_block;
            } else {
                int extra_blk = global_blk - num_orig_blocks;
                ib = extra_indices + (size_t)s_i * cold.extra_topk + (size_t)extra_blk * BI;
                tile_kv_cache = extra_KV_cache;
                tile_stride = cold.stride_extra_kv_block;
            }

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
                    entry_base[e] = tile_kv_cache + (size_t)bi_e * tile_stride
                                                  + (size_t)li_e * IO::IO_STRIDE;
                }
            } else {
                int idx = ib[qk_nb + gid];
                idx = (idx >= 0) ? idx : 0;
                entry_base[gid] = tile_kv_cache + (size_t)idx * IO::IO_STRIDE;
            }

            for (int i = threadIdx.x; i < CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = 0.f;

            // Zero kv_smem rows for invalid (-1) entries so XV NoPE MMA
            // picks up exact-zero V instead of slot-0 garbage (0 × fp8_NaN = NaN).
            {
                constexpr int BYTES_PER_LANE = (KV::KV_SMEM_COPY_BYTES + 31) / 32;
                #pragma unroll
                for (int e = 0; e < ENTRIES_PER_WARP; e++) {
                    if (ib[qk_nb + e] < 0) {
                        uint8_t* row = kv_smem + (qk_nb + e) * KV::KV_SMEM_STRIDE;
                        #pragma unroll
                        for (int b = 0; b < BYTES_PER_LANE; b++) {
                            int off = lane * BYTES_PER_LANE + b;
                            if (off < KV::KV_SMEM_COPY_BYTES) row[off] = 0;
                        }
                    }
                }
                bar_sync_t<2, MATH_THREADS>();
            }

            KVRopePrefetch rope_pf = prefetch_kv_rope(
                reinterpret_cast<const bf16*>(entry_base[gid] + KV::KV_ROPE_GMEM_OFFSET), lane);

            // QK nope
            float qk[4] = {0.f, 0.f, 0.f, 0.f};
            const uint8_t* kv_gid_base = kv_warp_base + gid * KV::KV_SMEM_STRIDE;

            if constexpr (BF16_QK) {
            #pragma unroll
            for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
                float scale_f = ue8m0_to_fp32(
                    sm.kv_scale_bufs[ti & 1][(qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN + blk]);
                #pragma unroll
                for (int ks = 0; ks < KV::QUANT_TILE / 16; ks++) {
                    int ko = blk * KV::QUANT_TILE + ks * 16;
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_load_A_bf16(a0, a1, a2, a3,
                        sm.q_nope_bf16 + ko, KV::Q_NOPE_BF16_STRIDE, lane);
                    uint16_t p0 = *reinterpret_cast<const uint16_t*>(kv_gid_base + ko + 2 * tid);
                    uint16_t p1 = *reinterpret_cast<const uint16_t*>(kv_gid_base + ko + 2 * tid + 8);
                    uint32_t f16x2_0, f16x2_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2_0) : "h"(p0));
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2_1) : "h"(p1));
                    __half2 h2_0 = *reinterpret_cast<__half2*>(&f16x2_0);
                    __half2 h2_1 = *reinterpret_cast<__half2*>(&f16x2_1);
                    float fk0 = __low2float(h2_0) * scale_f, fk1 = __high2float(h2_0) * scale_f;
                    float fk2 = __low2float(h2_1) * scale_f, fk3 = __high2float(h2_1) * scale_f;
                    uint32_t b0, b1;
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b0) : "f"(fk1), "f"(fk0));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b1) : "f"(fk3), "f"(fk2));
                    MmaBf16Result r = mma_bf16_m16n8k16(
                        a0, a1, a2, a3, b0, b1,
                        qk[0], qk[1], qk[2], qk[3]);
                    qk[0] = r.d0; qk[1] = r.d1; qk[2] = r.d2; qk[3] = r.d3;
                }
            }
            } else {
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
            }

            compute_qk_rope(qk, q_rope_regs, rope_pf);

            // Invalid index masking: negative indices + topk_length overflow
            {
                int e0 = qk_nb + tid * 2, e1 = e0 + 1;
                if (ib[e0] < 0) { qk[0] = -1e30f; qk[2] = -1e30f; }
                if (ib[e1] < 0) { qk[1] = -1e30f; qk[3] = -1e30f; }
                // Mask entries beyond topk_length (main tiles) or extra_topk_length (extra tiles)
                if (!is_extra_tile) {
                    int abs0 = global_blk * BI + e0, abs1 = global_blk * BI + e1;
                    if (abs0 >= topk_len) { qk[0] = -1e30f; qk[2] = -1e30f; }
                    if (abs1 >= topk_len) { qk[1] = -1e30f; qk[3] = -1e30f; }
                } else if (extra_topk_len > 0) {
                    int extra_blk = global_blk - num_orig_blocks;
                    int abs0 = extra_blk * BI + e0, abs1 = extra_blk * BI + e1;
                    if (abs0 >= extra_topk_len) { qk[0] = -1e30f; qk[2] = -1e30f; }
                    if (abs1 >= extra_topk_len) { qk[1] = -1e30f; qk[3] = -1e30f; }
                }
            }

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

            // XV nope
            {
                const int e0i = qk_nb + tid * 2, e1i = e0i + 1;
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
                if constexpr (!KV::V_HAS_ROPE)
                    bar_sync_t<2, MATH_THREADS>();
            }

            // XV rope
            if constexpr (KV::V_HAS_ROPE) {
                bar_sync_t<2, MATH_THREADS>();
                xv_rope_mma<MT, PAGE_BLOCK_SIZE>(acc_rope, w0, w1, w2, w3,
                    ib, tile_kv_cache, mwarp, lane,
                    tile_stride,
                    reinterpret_cast<bf16*>(sm.w_fp8));
            }

            bar_arrive_t<1, BLOCK_THREADS>();
            if (ti + 1 < num_tiles) {
                const int next_phase = ((ti + 1) >> 1) & 1;
                mbarrier_wait_parity(sm.mbar_kv + ((ti + 1) & 1), next_phase);
            }
        }
        // ── End main loop ───────────────────────────────────────────

        // Finalize deferred row_sum
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

        // ── Epilogue ────────────────────────────────────────────────
        float il0, il1;
        if (is_no_split && cold.attn_sink != nullptr) {
            float s0 = (gid < VALID_HPB) ? (__ldg(cold.attn_sink + h_start + gid) * LOG2E) : -1e30f;
            float s1 = (gid + 8 < VALID_HPB) ? (__ldg(cold.attn_sink + h_start + gid + 8) * LOG2E) : -1e30f;
            float d0 = sm.l_smem[gid] + exp2f(s0 - sm.m_smem[gid]);
            float d1 = sm.l_smem[gid + 8] + exp2f(s1 - sm.m_smem[gid + 8]);
            il0 = (d0 > 0.f) ? (1.f / d0) : 0.f;
            il1 = (d1 > 0.f) ? (1.f / d1) : 0.f;
        } else {
            il0 = (sm.l_smem[gid] > 0.f) ? (1.f / sm.l_smem[gid]) : 0.f;
            il1 = (sm.l_smem[gid + 8] > 0.f) ? (1.f / sm.l_smem[gid + 8]) : 0.f;
        }

        if (is_no_split) {
            // Direct bf16 output — no combine needed
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

            // Write bf16 output
            constexpr size_t h_stride = D_V;
            constexpr size_t token_stride = (size_t)NUM_HEADS * D_V;
            const size_t out_base = (size_t)s_i * token_stride + (size_t)h_start * h_stride;
            constexpr int BF16_PER_STORE = 8;
            constexpr int STORES_PER_HEAD = D_V / BF16_PER_STORE;
            for (int idx = threadIdx.x; idx < VALID_HPB * STORES_PER_HEAD; idx += MATH_THREADS) {
                int h = idx / STORES_PER_HEAD;
                int d8 = (idx - h * STORES_PER_HEAD) * BF16_PER_STORE;
                uint4 v = *reinterpret_cast<const uint4*>(
                    &staging_bf16[h * BF16_STAGING_STRIDE + d8]);
                *reinterpret_cast<uint4*>(&output[out_base + h * h_stride + d8]) = v;
            }

            // Write LSE (with attn_sink merged)
            if (threadIdx.x < VALID_HPB) {
                int h = threadIdx.x;
                float lse = softmax_lse(sm.m_smem[h], sm.l_smem[h]);
                if (cold.attn_sink != nullptr) {
                    float sink_log2 = __ldg(cold.attn_sink + h_start + h) * LOG2E;
                    if (lse != -1e30f)
                        lse += log2f(1.f + exp2f(sink_log2 - lse));
                    else
                        lse = sink_log2;
                }
                size_t lse_idx = (size_t)s_i * NUM_HEADS + h_start + h;
                out_lse[lse_idx] = lse;
            }
        } else {
            // Partial output → o_accum / lse_accum
            float* staging_f32 = reinterpret_cast<float*>(sm.kv_bufs[0]);
            constexpr int F32_STAGING_STRIDE = D_V;

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
            if constexpr (KV::V_HAS_ROPE) {
                int n_start = mwarp * 8;
                int d0 = KV::D_NOPE + n_start + tid * 2;
                staging_f32[gid * F32_STAGING_STRIDE + d0]       = acc_rope[0] * il0;
                staging_f32[gid * F32_STAGING_STRIDE + d0 + 1]   = acc_rope[1] * il0;
                staging_f32[(gid+8) * F32_STAGING_STRIDE + d0]   = acc_rope[2] * il1;
                staging_f32[(gid+8) * F32_STAGING_STRIDE + d0+1] = acc_rope[3] * il1;
            }
            bar_sync_t<2, MATH_THREADS>();

            // Write to o_accum[split_idx]
            {
                const size_t oa_base = (size_t)split_idx * cold.stride_oa_split
                                     + (size_t)s_q_idx * cold.stride_oa_sq
                                     + (size_t)h_start * D_V;
                constexpr int FLOATS_PER_WIDE_STORE = 8;
                constexpr int WIDE_STORES_PER_HEAD = D_V / FLOATS_PER_WIDE_STORE;
                for (int idx = threadIdx.x; idx < VALID_HPB * WIDE_STORES_PER_HEAD; idx += MATH_THREADS) {
                    int h = idx / WIDE_STORES_PER_HEAD;
                    int d8 = (idx - h * WIDE_STORES_PER_HEAD) * FLOATS_PER_WIDE_STORE;
                    float4 v0 = *reinterpret_cast<const float4*>(
                        &staging_f32[h * F32_STAGING_STRIDE + d8]);
                    float4 v1 = *reinterpret_cast<const float4*>(
                        &staging_f32[h * F32_STAGING_STRIDE + d8 + 4]);
                    store_8f_evict_last(&o_accum[oa_base + h * D_V + d8], v0, v1);
                }
            }

            // Write lse_accum[split_idx]
            if (threadIdx.x < VALID_HPB) {
                int h = threadIdx.x;
                float lse = softmax_lse(sm.m_smem[h], sm.l_smem[h]);
                size_t lse_idx = (size_t)split_idx * cold.stride_la_split
                               + (size_t)s_q_idx * cold.stride_la_sq
                               + h_start + h;
                lse_accum[lse_idx] = lse;
            }
        }
    } // end math warp

    } // end batch loop

    // Ensure all warps (IO + math) finish before PDL signals next CTA.
    // IO warps may exit the batch loop before math warps finish epilogue writes.
    bar_sync_t<1, BLOCK_THREADS>();
    cudaTriggerProgrammaticLaunchCompletion();
}
