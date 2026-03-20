#include "mla_kernel_utils.cuh"

#include <torch/extension.h>

// ============================================================================
// Sparse MLA Decode Kernel — MMA-based split-KV with in-kernel combine
//
// Template params: NUM_HEADS, TILES_PER_SPLIT (both constexpr).
// NSPLITS = NI / TILES_PER_SPLIT derived at compile time.
//
// Each CTA computes QK + softmax + XV for its tile range, writes partial
// results to workspace, then the last CTA per (token, head_tile) merges
// all splits and writes the final output — no separate combine kernel.
// ============================================================================

// ── Tile geometry ───────────────────────────────────────────────────────
static constexpr int HPB = 16;
static constexpr int BI  = 64;
static constexpr int NI  = TOPK / BI;                           // 32
static constexpr int N_MATH_WARPS  = 8;
static constexpr int N_IO_WARPS    = 4;
static constexpr int N_TOTAL_WARPS = N_MATH_WARPS + N_IO_WARPS; // 12
static constexpr int BLOCK_THREADS = N_TOTAL_WARPS * 32;        // 384
static constexpr int MATH_THREADS  = N_MATH_WARPS * 32;         // 256
static constexpr int IO_THREADS    = N_IO_WARPS * 32;           // 128

static constexpr int V_CHUNK        = 128;
static constexpr int N_V_CHUNKS     = D_NOPE / V_CHUNK;                    // 4
static constexpr int ENTRIES_PER_WARP = BI / N_MATH_WARPS;                 // 8
static constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / N_MATH_WARPS;         // 2
static constexpr int ACC_TILES      = N_V_CHUNKS * NT_PER_WARP_XV;        // 8
static constexpr int XV_KSTEPS      = BI / 32;                            // 2
static constexpr int QK_NOPE_KSTEPS = QUANT_TILE / 32;                    // 4

// ── Smem strides ────────────────────────────────────────────────────────
static constexpr int KV_SMEM_STRIDE = D_NOPE + NUM_SCALES * sizeof(float); // 528
static constexpr int Q_NOPE_STRIDE  = D_NOPE + 16;    // 528
static constexpr int V_TRANS_STRIDE = BI + 16;         // 80
static constexpr int W_FP8_STRIDE   = BI + 16;         // 80

static constexpr int KV_ROPE_BYTE_OFFSET = D_NOPE + NUM_SCALES * sizeof(float);

// ── Output staging ──────────────────────────────────────────────────────
static constexpr int OUT_STAGING_STRIDE = D_NOPE + 8;    // 520 bf16 elements
static constexpr int OUT_VEC = 8;
static constexpr int OUT_TILES_PER_HEAD = D_NOPE / OUT_VEC; // 64

// ── Smem layout ─────────────────────────────────────────────────────────
static constexpr size_t SMEM_Q_NOPE     = HPB * Q_NOPE_STRIDE;
static constexpr size_t SMEM_Q_SC       = HPB * NUM_SCALES * sizeof(float);
static constexpr size_t SMEM_Q_ROPE     = HPB * D_ROPE * sizeof(bf16);
static constexpr size_t SMEM_KV_BUF     = BI * KV_SMEM_STRIDE;
static constexpr size_t SMEM_REDUCE     = N_MATH_WARPS * HPB * sizeof(float);
static constexpr size_t SMEM_SUM_REDUCE = N_MATH_WARPS * HPB * sizeof(float);
static constexpr size_t SMEM_M          = HPB * sizeof(float);
static constexpr size_t SMEM_L          = HPB * sizeof(float);
static constexpr size_t SMEM_W_SC_ALL   = N_V_CHUNKS * HPB * sizeof(float);
static constexpr size_t SMEM_W_FP8      = HPB * W_FP8_STRIDE;
static constexpr size_t SMEM_V_TRANS    = V_CHUNK * V_TRANS_STRIDE;
static constexpr size_t SMEM_MBAR_KV    = 2 * sizeof(uint64_t);  // double-buffered

namespace dec {
    static constexpr size_t OFF_Q_NOPE    = 0;
    static constexpr size_t OFF_Q_SC      = OFF_Q_NOPE    + SMEM_Q_NOPE;
    static constexpr size_t OFF_Q_ROPE    = OFF_Q_SC      + SMEM_Q_SC;
    static constexpr size_t OFF_KV0       = OFF_Q_ROPE    + SMEM_Q_ROPE;
    static constexpr size_t OFF_KV1       = OFF_KV0       + SMEM_KV_BUF;
    static constexpr size_t OFF_REDUCE    = OFF_KV1       + SMEM_KV_BUF;
    static constexpr size_t OFF_SUM_RED   = OFF_REDUCE    + SMEM_REDUCE;
    static constexpr size_t OFF_M         = OFF_SUM_RED   + SMEM_SUM_REDUCE;
    static constexpr size_t OFF_L         = OFF_M         + SMEM_M;
    static constexpr size_t OFF_W_SC_ALL  = OFF_L         + SMEM_L;
    static constexpr size_t OFF_W_FP8     = OFF_W_SC_ALL  + SMEM_W_SC_ALL;
    static constexpr size_t OFF_V_TRANS   = OFF_W_FP8     + SMEM_W_FP8;
    static constexpr size_t OFF_MBAR_KV   = (OFF_V_TRANS + SMEM_V_TRANS + 7) / 8 * 8;
    static constexpr size_t TOTAL         = OFF_MBAR_KV   + SMEM_MBAR_KV;
}
static_assert(dec::TOTAL < 100 * 1024, "decode smem exceeds 100 KB");

struct DecodeSmem {
    uint8_t* q_nope_fp8;
    float*   q_nope_sc;
    bf16*    q_rope;
    uint8_t* kv_bufs[2];
    float*   reduce_buf;
    float*   sum_reduce_buf;
    float*   m_smem;
    float*   l_smem;
    float*   w_head_sc_all;
    uint8_t* w_fp8;
    uint8_t* v_trans;
    uint64_t* mbar_kv;

    __device__ static DecodeSmem init(char* b) {
        return {
            (uint8_t*)(b + dec::OFF_Q_NOPE),
            (float*)  (b + dec::OFF_Q_SC),
            (bf16*)   (b + dec::OFF_Q_ROPE),
            {(uint8_t*)(b + dec::OFF_KV0), (uint8_t*)(b + dec::OFF_KV1)},
            (float*)  (b + dec::OFF_REDUCE),
            (float*)  (b + dec::OFF_SUM_RED),
            (float*)  (b + dec::OFF_M),
            (float*)  (b + dec::OFF_L),
            (float*)  (b + dec::OFF_W_SC_ALL),
            (uint8_t*)(b + dec::OFF_W_FP8),
            (uint8_t*)(b + dec::OFF_V_TRANS),
            (uint64_t*)(b + dec::OFF_MBAR_KV),
        };
    }
};


// ============================================================================
// Decode kernel — split-KV with in-kernel combine
// ============================================================================

template <int NUM_HEADS, int TILES_PER_SPLIT>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
sparse_mla_decode_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    bf16* __restrict__ partial_O,
    float* __restrict__ partial_LSE,
    bf16* __restrict__ output,
    uint32_t* __restrict__ semaphores,
    float sm_scale,
    int num_tokens)
{
    static constexpr int REPLICATE_H = NUM_HEADS / HPB;
    static constexpr int NSPLITS = NI / TILES_PER_SPLIT;

    const int s_i = blockIdx.x / REPLICATE_H;
    const int h_tile = blockIdx.x % REPLICATE_H;
    const int h_start = h_tile * HPB;
    const int split_idx = blockIdx.y;
    if (s_i >= num_tokens) return;

    constexpr int tile_start_stride = TILES_PER_SPLIT * BI;

    const int warp_rank = threadIdx.x / 32;
    const int wy = warp_rank / 4;

    extern __shared__ char smem_raw[];
    DecodeSmem sm = DecodeSmem::init(smem_raw);

    if (threadIdx.x == 0) {
        mbarrier_init(sm.mbar_kv + 0, 1);
        mbarrier_init(sm.mbar_kv + 1, 1);
    }
    bar_sync_t<3, BLOCK_THREADS>();

    // ── IO warps ────────────────────────────────────────────────────────
    if (wy == 2) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));

        const int io_tid = threadIdx.x - N_MATH_WARPS * 32;
        const int32_t* idx_base = indices + (size_t)s_i * TOPK + split_idx * tile_start_stride;

        io_bulk_gather_tile<KV_SMEM_STRIDE, IO_THREADS, BI>(
            sm.kv_bufs[0], idx_base, KV_cache, sm.mbar_kv + 0, io_tid);

        #pragma unroll
        for (int ti = 0; ti < TILES_PER_SPLIT; ti++) {
            if (ti + 1 < TILES_PER_SPLIT)
                io_bulk_gather_tile<KV_SMEM_STRIDE, IO_THREADS, BI>(
                    sm.kv_bufs[(ti + 1) & 1],
                    idx_base + (ti + 1) * BI, KV_cache,
                    sm.mbar_kv + ((ti + 1) & 1), io_tid);
            bar_sync_t<1, BLOCK_THREADS>();
        }

    // ── Math warps ──────────────────────────────────────────────────────
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(200));

        const int lane = threadIdx.x & 31;
        const int mwarp = warp_rank;
        const int gid = lane >> 2, tid = lane & 3;
        const float sm_scale_log2e = sm_scale * LOG2E;
        const bf16* q_base = Q + (size_t)s_i * NUM_HEADS * DIM + (size_t)h_start * DIM;
        const int32_t* idx_base = indices + (size_t)s_i * TOPK + split_idx * tile_start_stride;

        quantize_q_to_smem<HPB, Q_NOPE_STRIDE, MATH_THREADS>(
            sm.q_nope_fp8, sm.q_nope_sc, sm.q_rope, q_base, sm.reduce_buf);
        QRopeRegs q_rope_regs = preload_q_rope_regs(sm.q_rope, lane);

        for (int h = threadIdx.x; h < HPB; h += MATH_THREADS) {
            sm.m_smem[h] = -1e30f;
            sm.l_smem[h] = 0.f;
        }

        float acc_o[ACC_TILES][4];
        #pragma unroll
        for (int t = 0; t < ACC_TILES; t++)
            acc_o[t][0] = acc_o[t][1] = acc_o[t][2] = acc_o[t][3] = 0.f;

        bar_sync_t<2, MATH_THREADS>();
        mbarrier_wait_parity(sm.mbar_kv + 0, 0);

        // ── Main loop — QK + softmax + XV ───────────────────────────
        #pragma unroll
        for (int ti = 0; ti < TILES_PER_SPLIT; ti++) {
            uint8_t* kv_smem = sm.kv_bufs[ti & 1];
            const int32_t* ib = idx_base + ti * BI;
            const int qk_nb = mwarp * ENTRIES_PER_WARP;

            for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = 0.f;

            float qk[4] = {0.f, 0.f, 0.f, 0.f};
            compute_qk_nope<Q_NOPE_STRIDE, KV_SMEM_STRIDE, QK_NOPE_KSTEPS>(
                qk, sm.q_nope_fp8, sm.q_nope_sc, kv_smem, qk_nb, lane);

            {
                int entry_idx = ib[qk_nb + gid];
                entry_idx = (entry_idx >= 0) ? entry_idx : 0;
                const bf16* g_rope = reinterpret_cast<const bf16*>(
                    KV_cache + (size_t)entry_idx * KV_PACKED_BYTES + KV_ROPE_BYTE_OFFSET);
                compute_qk_rope(qk, q_rope_regs, g_rope, lane);
            }

            {
                int e0 = qk_nb + tid * 2, e1 = e0 + 1;
                if (ib[e0] < 0) { qk[0] = -1e30f; qk[2] = -1e30f; }
                if (ib[e1] < 0) { qk[1] = -1e30f; qk[3] = -1e30f; }
            }

            float s[4] = { qk[0] * sm_scale_log2e, qk[1] * sm_scale_log2e,
                           qk[2] * sm_scale_log2e, qk[3] * sm_scale_log2e };

            float lm0 = fmaxf(s[0], s[1]), lm1 = fmaxf(s[2], s[3]);
            lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 1));
            lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 2));
            lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 1));
            lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 2));
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
            for (int t = 0; t < ACC_TILES; t++) {
                acc_o[t][0] *= alpha0; acc_o[t][1] *= alpha0;
                acc_o[t][2] *= alpha1; acc_o[t][3] *= alpha1;
            }

            float w0 = exp2f(s[0] - nm0), w1 = exp2f(s[1] - nm0);
            float w2 = exp2f(s[2] - nm1), w3 = exp2f(s[3] - nm1);

            float ls0 = w0 + w1, ls1 = w2 + w3;
            ls0 += __shfl_xor_sync(0xffffffff, ls0, 1);
            ls0 += __shfl_xor_sync(0xffffffff, ls0, 2);
            ls1 += __shfl_xor_sync(0xffffffff, ls1, 1);
            ls1 += __shfl_xor_sync(0xffffffff, ls1, 2);
            if (tid == 0) {
                sm.sum_reduce_buf[mwarp * HPB + gid] = ls0;
                sm.sum_reduce_buf[mwarp * HPB + gid + 8] = ls1;
            }

            {
                int e0i = qk_nb + tid * 2, e1i = e0i + 1;
                #pragma unroll
                for (int vc = 0; vc < N_V_CHUNKS; vc++) {
                    float vsc0 = reinterpret_cast<const float*>(
                        kv_smem + e0i * KV_SMEM_STRIDE + D_NOPE)[vc];
                    float vsc1 = reinterpret_cast<const float*>(
                        kv_smem + e1i * KV_SMEM_STRIDE + D_NOPE)[vc];
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
            for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = fmaxf(sm.w_head_sc_all[i], 1e-10f) / FP8_MAX;
            bar_sync_t<2, MATH_THREADS>();

            // ── XV MMA ──────────────────────────────────────────────
            {
                int e0i = qk_nb + tid * 2, e1i = e0i + 1;

                #pragma unroll
                for (int vc = 0; vc < N_V_CHUNKS; vc++) {
                    const int v_off = vc * V_CHUNK;
                    float* vc_sc = sm.w_head_sc_all + vc * HPB;

                    {
                        float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
                        float vsc0 = reinterpret_cast<const float*>(
                            kv_smem + e0i * KV_SMEM_STRIDE + D_NOPE)[vc];
                        float vsc1 = reinterpret_cast<const float*>(
                            kv_smem + e1i * KV_SMEM_STRIDE + D_NOPE)[vc];
                        float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
                        float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
                        __nv_fp8_e4m3 f00(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws00 * si0)));
                        __nv_fp8_e4m3 f01(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws01 * si0)));
                        __nv_fp8_e4m3 f10(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws10 * si1)));
                        __nv_fp8_e4m3 f11(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws11 * si1)));
                        sm.w_fp8[gid * W_FP8_STRIDE + e0i] = f00.__x;
                        sm.w_fp8[gid * W_FP8_STRIDE + e1i] = f01.__x;
                        sm.w_fp8[(gid + 8) * W_FP8_STRIDE + e0i] = f10.__x;
                        sm.w_fp8[(gid + 8) * W_FP8_STRIDE + e1i] = f11.__x;
                    }

                    transpose_v_chunk<V_CHUNK, V_TRANS_STRIDE, KV_SMEM_STRIDE, MATH_THREADS, BI>(
                        sm.v_trans, kv_smem, v_off, lane);
                    bar_sync_t<2, MATH_THREADS>();

                    #pragma unroll
                    for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
                        int ti_acc = vc * NT_PER_WARP_XV + nt;
                        int nl = mwarp * (NT_PER_WARP_XV * 8) + nt * 8;
                        float xv[4] = {0.f, 0.f, 0.f, 0.f};
                        #pragma unroll
                        for (int kstep = 0; kstep < XV_KSTEPS; kstep++) {
                            int ko = kstep * 32;
                            uint32_t a0, a1, a2, a3, b0, b1;
                            ldmatrix_load_A_fp8(a0, a1, a2, a3,
                                sm.w_fp8 + ko, W_FP8_STRIDE, lane);
                            ldmatrix_load_B_fp8(b0, b1,
                                sm.v_trans + nl * V_TRANS_STRIDE + ko,
                                V_TRANS_STRIDE, lane);
                            MmaFp8Result r = mma_fp8_m16n8k32(
                                a0, a1, a2, a3, b0, b1,
                                xv[0], xv[1], xv[2], xv[3]);
                            xv[0] = r.d0; xv[1] = r.d1; xv[2] = r.d2; xv[3] = r.d3;
                        }
                        float sc0 = vc_sc[gid], sc1 = vc_sc[gid + 8];
                        acc_o[ti_acc][0] += xv[0] * sc0; acc_o[ti_acc][1] += xv[1] * sc0;
                        acc_o[ti_acc][2] += xv[2] * sc1; acc_o[ti_acc][3] += xv[3] * sc1;
                    }

                    if (vc < N_V_CHUNKS - 1)
                        bar_sync_t<2, MATH_THREADS>();
                }
            }

            bar_arrive_t<1, BLOCK_THREADS>();
            if (ti + 1 < TILES_PER_SPLIT) {
                const int next_phase = ((ti + 1) >> 1) & 1;
                mbarrier_wait_parity(sm.mbar_kv + ((ti + 1) & 1), next_phase);
            }
        }

        // ── Write partial output via smem staging ───────────────────
        bf16* staging = reinterpret_cast<bf16*>(sm.kv_bufs[0]);

        float il0 = (sm.l_smem[gid] > 0.f) ? (1.f / sm.l_smem[gid]) : 0.f;
        float il1 = (sm.l_smem[gid + 8] > 0.f) ? (1.f / sm.l_smem[gid + 8]) : 0.f;

        #pragma unroll
        for (int t = 0; t < ACC_TILES; t++) {
            constexpr int _NT8 = NT_PER_WARP_XV * 8;
            int c = t / NT_PER_WARP_XV, lnt = t % NT_PER_WARP_XV;
            int d0 = c * V_CHUNK + mwarp * _NT8 + lnt * 8 + tid * 2;
            *reinterpret_cast<bf16_2*>(&staging[gid * OUT_STAGING_STRIDE + d0]) =
                __floats2bfloat162_rn(acc_o[t][0] * il0, acc_o[t][1] * il0);
            *reinterpret_cast<bf16_2*>(&staging[(gid + 8) * OUT_STAGING_STRIDE + d0]) =
                __floats2bfloat162_rn(acc_o[t][2] * il1, acc_o[t][3] * il1);
        }
        bar_sync_t<2, MATH_THREADS>();

        // Write partial_O to workspace
        {
            constexpr size_t h_stride = (size_t)D_NOPE;
            constexpr size_t split_stride = (size_t)NUM_HEADS * D_NOPE;
            constexpr size_t token_stride = (size_t)NSPLITS * split_stride;
            const size_t po_base = (size_t)s_i * token_stride
                                 + (size_t)split_idx * split_stride
                                 + (size_t)h_start * h_stride;
            for (int idx = threadIdx.x; idx < HPB * OUT_TILES_PER_HEAD; idx += MATH_THREADS) {
                int h = idx / OUT_TILES_PER_HEAD;
                int d8 = idx - h * OUT_TILES_PER_HEAD;
                uint4 val = *reinterpret_cast<const uint4*>(
                    &staging[h * OUT_STAGING_STRIDE + d8 * OUT_VEC]);
                *reinterpret_cast<uint4*>(
                    &partial_O[po_base + h * h_stride + d8 * OUT_VEC]) = val;
            }
        }

        // Write partial_LSE to workspace
        if (threadIdx.x < HPB) {
            int h = threadIdx.x;
            float m = sm.m_smem[h];
            float l = sm.l_smem[h];
            float lse = (l > 0.f) ? (log2f(l) + m) : -1e30f;
            constexpr size_t lse_split_stride = (size_t)NUM_HEADS;
            constexpr size_t lse_token_stride = (size_t)NSPLITS * lse_split_stride;
            size_t lse_idx = (size_t)s_i * lse_token_stride
                           + (size_t)split_idx * lse_split_stride
                           + (h_start + h);
            partial_LSE[lse_idx] = lse;
        }

        // ── In-kernel combine via atomicInc ─────────────────────────
        __threadfence();

        bool is_last = false;
        if (threadIdx.x == 0) {
            uint32_t old;
            constexpr uint32_t limit = NSPLITS - 1;
            asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;\n"
                : "=r"(old)
                : "l"(&semaphores[s_i * REPLICATE_H + h_tile]), "r"(limit));
            reinterpret_cast<uint32_t*>(sm.reduce_buf)[0] = (old == limit) ? 1u : 0u;
        }
        bar_sync_t<2, MATH_THREADS>();
        is_last = (reinterpret_cast<uint32_t*>(sm.reduce_buf)[0] == 1u);

        if (!is_last) return;

        // ── Last CTA: merge all NSPLITS and write final output ──────
        // Optimized: use own acc_o as seed, direct global reads (no smem
        // staging / barriers in merge loop), only smem for final output.
        __threadfence();

        float* lse_buf = sm.reduce_buf;

        // Load all LSE values into smem (single cooperative load + 1 barrier)
        for (int i = threadIdx.x; i < NSPLITS * HPB; i += MATH_THREADS) {
            int sp = i / HPB, h = i % HPB;
            constexpr size_t lse_split_stride2 = (size_t)NUM_HEADS;
            constexpr size_t lse_token_stride2 = (size_t)NSPLITS * lse_split_stride2;
            size_t lse_idx = (size_t)s_i * lse_token_stride2
                           + (size_t)sp * lse_split_stride2
                           + (h_start + h);
            lse_buf[sp * HPB + h] = partial_LSE[lse_idx];
        }
        bar_sync_t<2, MATH_THREADS>();

        // Own LSE (already in smem from above, or compute from m/l)
        float my_lse0 = lse_buf[split_idx * HPB + gid];
        float my_lse1 = lse_buf[split_idx * HPB + gid + 8];

        // Find max LSE across all splits
        float max_lse0 = my_lse0, max_lse1 = my_lse1;
        #pragma unroll
        for (int sp = 0; sp < NSPLITS; sp++) {
            max_lse0 = fmaxf(max_lse0, lse_buf[sp * HPB + gid]);
            max_lse1 = fmaxf(max_lse1, lse_buf[sp * HPB + gid + 8]);
        }

        // Seed merge_acc from own acc_o (float32, no bf16 round-trip)
        float my_sc0 = exp2f(my_lse0 - max_lse0);
        float my_sc1 = exp2f(my_lse1 - max_lse1);
        float scale_sum0 = my_sc0, scale_sum1 = my_sc1;

        // acc_o is un-normalized. merge_acc = normalized * scale = (acc_o * il) * sc
        float merge_acc[ACC_TILES][4];
        #pragma unroll
        for (int t = 0; t < ACC_TILES; t++) {
            merge_acc[t][0] = acc_o[t][0] * il0 * my_sc0;
            merge_acc[t][1] = acc_o[t][1] * il0 * my_sc0;
            merge_acc[t][2] = acc_o[t][2] * il1 * my_sc1;
            merge_acc[t][3] = acc_o[t][3] * il1 * my_sc1;
        }

        // Merge other NSPLITS-1 splits — direct global reads, NO barriers
        constexpr size_t po_split_stride = (size_t)NUM_HEADS * D_NOPE;
        constexpr size_t po_token_stride = (size_t)NSPLITS * po_split_stride;
        const size_t po_token_base = (size_t)s_i * po_token_stride
                                   + (size_t)h_start * D_NOPE;

        #pragma unroll 1
        for (int sp = 0; sp < NSPLITS; sp++) {
            if (sp == split_idx) continue;

            float sc0 = exp2f(lse_buf[sp * HPB + gid] - max_lse0);
            float sc1 = exp2f(lse_buf[sp * HPB + gid + 8] - max_lse1);
            scale_sum0 += sc0;
            scale_sum1 += sc1;

            const bf16* po_ptr = partial_O + po_token_base
                               + (size_t)sp * po_split_stride;

            #pragma unroll
            for (int t = 0; t < ACC_TILES; t++) {
                constexpr int _NT8 = NT_PER_WARP_XV * 8;
                int c = t / NT_PER_WARP_XV, lnt = t % NT_PER_WARP_XV;
                int d0 = c * V_CHUNK + mwarp * _NT8 + lnt * 8 + tid * 2;
                bf16_2 v01 = *reinterpret_cast<const bf16_2*>(&po_ptr[gid * D_NOPE + d0]);
                bf16_2 v23 = *reinterpret_cast<const bf16_2*>(&po_ptr[(gid + 8) * D_NOPE + d0]);
                float2 f01 = __bfloat1622float2(v01);
                float2 f23 = __bfloat1622float2(v23);

                merge_acc[t][0] += sc0 * f01.x;
                merge_acc[t][1] += sc0 * f01.y;
                merge_acc[t][2] += sc1 * f23.x;
                merge_acc[t][3] += sc1 * f23.y;
            }
        }

        // Normalize and write to smem staging for coalesced output
        float inv0 = (scale_sum0 > 0.f) ? (1.f / scale_sum0) : 0.f;
        float inv1 = (scale_sum1 > 0.f) ? (1.f / scale_sum1) : 0.f;

        #pragma unroll
        for (int t = 0; t < ACC_TILES; t++) {
            constexpr int _NT8 = NT_PER_WARP_XV * 8;
            int c = t / NT_PER_WARP_XV, lnt = t % NT_PER_WARP_XV;
            int d0 = c * V_CHUNK + mwarp * _NT8 + lnt * 8 + tid * 2;
            *reinterpret_cast<bf16_2*>(&staging[gid * OUT_STAGING_STRIDE + d0]) =
                __floats2bfloat162_rn(merge_acc[t][0] * inv0, merge_acc[t][1] * inv0);
            *reinterpret_cast<bf16_2*>(&staging[(gid + 8) * OUT_STAGING_STRIDE + d0]) =
                __floats2bfloat162_rn(merge_acc[t][2] * inv1, merge_acc[t][3] * inv1);
        }
        bar_sync_t<2, MATH_THREADS>();

        // Coalesced write to final output
        {
            const size_t out_base = (size_t)s_i * NUM_HEADS * D_NOPE
                                  + (size_t)h_start * D_NOPE;
            for (int idx = threadIdx.x; idx < HPB * OUT_TILES_PER_HEAD; idx += MATH_THREADS) {
                int h = idx / OUT_TILES_PER_HEAD;
                int d8 = idx - h * OUT_TILES_PER_HEAD;
                uint4 val = *reinterpret_cast<const uint4*>(
                    &staging[h * OUT_STAGING_STRIDE + d8 * OUT_VEC]);
                *reinterpret_cast<uint4*>(
                    &output[out_base + h * D_NOPE + d8 * OUT_VEC]) = val;
            }
        }
    }
}

// ============================================================================
// Launch helpers
// ============================================================================

template <int NUM_HEADS, int TILES_PER_SPLIT>
void launch_decode(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                   bf16* partial_O, float* partial_LSE,
                   bf16* output, uint32_t* semaphores,
                   float sm_scale, int num_tokens)
{
    constexpr size_t smem_bytes = dec::TOTAL;
    constexpr int REPLICATE_H = NUM_HEADS / HPB;
    constexpr int NSPLITS = NI / TILES_PER_SPLIT;
    dim3 grid(num_tokens * REPLICATE_H, NSPLITS);
    dim3 block(BLOCK_THREADS);

    static bool configured = false;
    if (!configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(sparse_mla_decode_kernel<NUM_HEADS, TILES_PER_SPLIT>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            smem_bytes);
        configured = true;
    }
    sparse_mla_decode_kernel<NUM_HEADS, TILES_PER_SPLIT><<<grid, block, smem_bytes>>>(
        Q, KV_cache, indices, partial_O, partial_LSE,
        output, semaphores, sm_scale, num_tokens);
}

template <int NUM_HEADS>
void dispatch_tiles(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                    bf16* partial_O, float* partial_LSE,
                    bf16* output, uint32_t* semaphores,
                    float sm_scale, int num_tokens, int tiles_per_split)
{
    switch (tiles_per_split) {
    case 2:  launch_decode<NUM_HEADS, 2> (Q, KV_cache, indices, partial_O, partial_LSE, output, semaphores, sm_scale, num_tokens); break;
    case 4:  launch_decode<NUM_HEADS, 4> (Q, KV_cache, indices, partial_O, partial_LSE, output, semaphores, sm_scale, num_tokens); break;
    case 8:  launch_decode<NUM_HEADS, 8> (Q, KV_cache, indices, partial_O, partial_LSE, output, semaphores, sm_scale, num_tokens); break;
    case 16: launch_decode<NUM_HEADS, 16>(Q, KV_cache, indices, partial_O, partial_LSE, output, semaphores, sm_scale, num_tokens); break;
    case 32: launch_decode<NUM_HEADS, 32>(Q, KV_cache, indices, partial_O, partial_LSE, output, semaphores, sm_scale, num_tokens); break;
    default: TORCH_CHECK(false, "tiles_per_split must be 2,4,8,16,32; got ", tiles_per_split);
    }
}

void sparse_mla_decode_launch(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor partial_O, torch::Tensor partial_LSE,
    torch::Tensor output, torch::Tensor semaphores,
    float sm_scale, int num_heads, int num_tokens, int topk,
    int tiles_per_split, int nsplits)
{
    TORCH_CHECK(topk == TOPK, "topk must be ", TOPK);
    TORCH_CHECK(num_tokens <= 64, "decode path requires num_tokens <= 64; got ", num_tokens);

    auto Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr());
    auto KV_ptr = reinterpret_cast<const uint8_t*>(KV_cache.data_ptr());
    auto idx_ptr = indices.data_ptr<int32_t>();
    auto PO_ptr = reinterpret_cast<bf16*>(partial_O.data_ptr());
    auto LSE_ptr = partial_LSE.data_ptr<float>();
    auto O_ptr = reinterpret_cast<bf16*>(output.data_ptr());
    auto sem_ptr = reinterpret_cast<uint32_t*>(semaphores.data_ptr<int32_t>());

    switch (num_heads) {
    case 16:  dispatch_tiles<16> (Q_ptr, KV_ptr, idx_ptr, PO_ptr, LSE_ptr, O_ptr, sem_ptr, sm_scale, num_tokens, tiles_per_split); break;
    case 32:  dispatch_tiles<32> (Q_ptr, KV_ptr, idx_ptr, PO_ptr, LSE_ptr, O_ptr, sem_ptr, sm_scale, num_tokens, tiles_per_split); break;
    case 64:  dispatch_tiles<64> (Q_ptr, KV_ptr, idx_ptr, PO_ptr, LSE_ptr, O_ptr, sem_ptr, sm_scale, num_tokens, tiles_per_split); break;
    case 128: dispatch_tiles<128>(Q_ptr, KV_ptr, idx_ptr, PO_ptr, LSE_ptr, O_ptr, sem_ptr, sm_scale, num_tokens, tiles_per_split); break;
    default:  TORCH_CHECK(false, "num_heads must be 16,32,64,128; got ", num_heads);
    }
}
