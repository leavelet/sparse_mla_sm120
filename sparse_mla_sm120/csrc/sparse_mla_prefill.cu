#include "mla_kernel_utils.cuh"

#include <torch/extension.h>

// ============================================================================
// Sparse MLA Prefill Kernels — Optimized for SM120a
//
// All tile sizes, smem layout offsets, loop bounds, and barrier parameters
// are constexpr. The compiler can resolve every address offset and loop
// trip count at compile time.
// ============================================================================

// ── Tile geometry (all constexpr) ───────────────────────────────────────
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

// ── Smem strides (padded for bank-conflict avoidance + ldmatrix alignment) ──
static constexpr int KV_SMEM_STRIDE = D_NOPE + NUM_SCALES * sizeof(float); // 528
static constexpr int Q_NOPE_STRIDE  = D_NOPE + 16;    // 528: stride/4=132, 132%32=4
static constexpr int V_TRANS_STRIDE = BI + 16;         // 80: 16B-aligned, 80/4=20, 20%32=20
static constexpr int W_FP8_STRIDE   = BI + 16;         // 80: 16B-aligned

// ── KV cache layout ─────────────────────────────────────────────────────
static constexpr int KV_ROPE_BYTE_OFFSET = D_NOPE + NUM_SCALES * sizeof(float); // 528

// ── Output staging (reuses kv_bufs after main loop for coalesced global writes) ──
// Padded stride: D_NOPE=512 bf16 → stride/4=256, 256%32=0 → 32-way conflict!
// Pad to 520: stride_bytes=1040, 1040/4=260, 260%32=4 → 4-way. 16B-aligned (1040%16=0).
static constexpr int OUT_STAGING_STRIDE = D_NOPE + 8;    // 520 bf16 elements
static constexpr int OUT_VEC = 8;                          // bf16 per uint4 store
static constexpr int OUT_TILES_PER_HEAD = D_NOPE / OUT_VEC; // 64
static constexpr size_t OUT_STAGING_BYTES = HPB * OUT_STAGING_STRIDE * sizeof(bf16); // 16640

// ── Smem layout: constexpr offsets from base ────────────────────────────
static constexpr size_t SMEM_Q_NOPE     = HPB * Q_NOPE_STRIDE;
static constexpr size_t SMEM_Q_SC       = HPB * NUM_SCALES * sizeof(float);
static constexpr size_t SMEM_Q_ROPE     = HPB * D_ROPE * sizeof(bf16);
static constexpr size_t SMEM_KV_BUF     = BI * KV_SMEM_STRIDE;
static_assert(OUT_STAGING_BYTES <= SMEM_KV_BUF, "staging must fit in one kv_buf");
static constexpr size_t SMEM_REDUCE     = N_MATH_WARPS * HPB * sizeof(float);
static constexpr size_t SMEM_SUM_REDUCE = N_MATH_WARPS * HPB * sizeof(float);
static constexpr size_t SMEM_M          = HPB * sizeof(float);
static constexpr size_t SMEM_L          = HPB * sizeof(float);
static constexpr size_t SMEM_W_SC_ALL   = N_V_CHUNKS * HPB * sizeof(float);
static constexpr size_t SMEM_W_FP8      = HPB * W_FP8_STRIDE;
static constexpr size_t SMEM_V_TRANS    = V_CHUNK * V_TRANS_STRIDE;

static constexpr size_t SMEM_MBAR_KV = 2 * sizeof(uint64_t);  // 16 bytes, double-buffered

// Single-group kernel smem offsets
namespace sg {
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

// Multi-group kernel smem offsets (N_HG=2)
static constexpr int MG_N_HG = 2;
static constexpr int MG_HEADS_PER_CTA = MG_N_HG * HPB;  // 32

// MG-specific smem sizes (doubled for parallel group processing)
static constexpr size_t SMEM_REDUCE_MG    = MG_N_HG * N_MATH_WARPS * HPB * sizeof(float);
static constexpr size_t SMEM_SUM_RED_MG   = MG_N_HG * N_MATH_WARPS * HPB * sizeof(float);
static constexpr size_t SMEM_W_FP8_MG     = MG_N_HG * HPB * W_FP8_STRIDE;
static constexpr int    MG_REDUCE_GRP_STRIDE = N_MATH_WARPS * HPB; // 128

namespace mg {
    static constexpr size_t OFF_Q_NOPE0   = 0;
    static constexpr size_t OFF_Q_NOPE1   = OFF_Q_NOPE0   + SMEM_Q_NOPE;
    static constexpr size_t OFF_Q_SC0     = OFF_Q_NOPE1   + SMEM_Q_NOPE;
    static constexpr size_t OFF_Q_SC1     = OFF_Q_SC0     + SMEM_Q_SC;
    static constexpr size_t OFF_KV0       = OFF_Q_SC1     + SMEM_Q_SC;
    static constexpr size_t OFF_KV1       = OFF_KV0       + SMEM_KV_BUF;
    static constexpr size_t OFF_REDUCE    = OFF_KV1       + SMEM_KV_BUF;
    static constexpr size_t OFF_SUM_RED   = OFF_REDUCE    + SMEM_REDUCE_MG;
    static constexpr size_t OFF_M         = OFF_SUM_RED   + SMEM_SUM_RED_MG;
    static constexpr size_t OFF_L         = OFF_M         + MG_N_HG * SMEM_M;
    static constexpr size_t OFF_W_SC_ALL  = OFF_L         + MG_N_HG * SMEM_L;
    static constexpr size_t OFF_W_FP8     = OFF_W_SC_ALL  + MG_N_HG * SMEM_W_SC_ALL;
    static constexpr size_t OFF_V_TRANS   = OFF_W_FP8     + SMEM_W_FP8_MG;
    static constexpr size_t OFF_MBAR_KV   = (OFF_V_TRANS + SMEM_V_TRANS + 7) / 8 * 8;
    static constexpr size_t TOTAL         = OFF_MBAR_KV   + SMEM_MBAR_KV;
}

static_assert(sg::TOTAL < 100 * 1024, "SG smem exceeds 100 KB");
static_assert(mg::TOTAL < 100 * 1024, "MG smem exceeds 100 KB");

// ── Wrappers for shared utility functions with prefill-specific constants ──

__device__ __forceinline__ void prefill_io_gather_tile(
    uint8_t* dst, const int32_t* ib,
    const uint8_t* __restrict__ KV_cache, int io_tid)
{
    io_gather_tile<KV_SMEM_STRIDE, IO_THREADS, BI>(dst, ib, KV_cache, io_tid);
}

__device__ __forceinline__ void prefill_io_bulk_gather_tile(
    uint8_t* dst, const int32_t* ib,
    const uint8_t* __restrict__ KV_cache,
    uint64_t* mbar, int io_tid)
{
    io_bulk_gather_tile<KV_SMEM_STRIDE, IO_THREADS, BI>(
        dst, ib, KV_cache, mbar, io_tid);
}

__device__ __forceinline__ void prefill_quantize_q(
    uint8_t* q_nope_fp8, float* q_nope_sc, bf16* q_rope,
    const bf16* q_base, float* reduce_buf)
{
    quantize_q_to_smem<HPB, Q_NOPE_STRIDE, MATH_THREADS>(
        q_nope_fp8, q_nope_sc, q_rope, q_base, reduce_buf);
}

__device__ __forceinline__ void prefill_compute_qk_nope(
    float qk[4], const uint8_t* q_nope_fp8, const float* q_nope_sc,
    const uint8_t* kv_smem, int qk_nb, int lane)
{
    compute_qk_nope<Q_NOPE_STRIDE, KV_SMEM_STRIDE, QK_NOPE_KSTEPS>(
        qk, q_nope_fp8, q_nope_sc, kv_smem, qk_nb, lane);
}

__device__ __forceinline__ void prefill_transpose_v_chunk(
    uint8_t* __restrict__ v_trans,
    const uint8_t* __restrict__ kv_smem,
    int v_off, int lane)
{
    transpose_v_chunk<V_CHUNK, V_TRANS_STRIDE, KV_SMEM_STRIDE, MATH_THREADS, BI>(
        v_trans, kv_smem, v_off, lane);
}

// ── Smem layout structs — constexpr offsets ─────────────────────────────
struct PrefillSmem {
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

    __device__ static PrefillSmem init(char* b) {
        return {
            (uint8_t*)(b + sg::OFF_Q_NOPE),
            (float*)  (b + sg::OFF_Q_SC),
            (bf16*)   (b + sg::OFF_Q_ROPE),
            {(uint8_t*)(b + sg::OFF_KV0), (uint8_t*)(b + sg::OFF_KV1)},
            (float*)  (b + sg::OFF_REDUCE),
            (float*)  (b + sg::OFF_SUM_RED),
            (float*)  (b + sg::OFF_M),
            (float*)  (b + sg::OFF_L),
            (float*)  (b + sg::OFF_W_SC_ALL),
            (uint8_t*)(b + sg::OFF_W_FP8),
            (uint8_t*)(b + sg::OFF_V_TRANS),
            (uint64_t*)(b + sg::OFF_MBAR_KV),
        };
    }
};

struct PrefillSmemMG {
    uint8_t* q_nope_fp8[MG_N_HG];
    float*   q_nope_sc[MG_N_HG];
    uint8_t* kv_bufs[2];
    float*   reduce_buf;
    float*   sum_reduce_buf;
    float*   m_smem;
    float*   l_smem;
    float*   w_head_sc_all;
    uint8_t* w_fp8_base;
    uint8_t* v_trans;
    uint64_t* mbar_kv;

    __device__ static PrefillSmemMG init(char* b) {
        PrefillSmemMG s;
        s.q_nope_fp8[0] = (uint8_t*)(b + mg::OFF_Q_NOPE0);
        s.q_nope_fp8[1] = (uint8_t*)(b + mg::OFF_Q_NOPE1);
        s.q_nope_sc[0]  = (float*)  (b + mg::OFF_Q_SC0);
        s.q_nope_sc[1]  = (float*)  (b + mg::OFF_Q_SC1);
        s.kv_bufs[0]    = (uint8_t*)(b + mg::OFF_KV0);
        s.kv_bufs[1]    = (uint8_t*)(b + mg::OFF_KV1);
        s.reduce_buf    = (float*)  (b + mg::OFF_REDUCE);
        s.sum_reduce_buf= (float*)  (b + mg::OFF_SUM_RED);
        s.m_smem        = (float*)  (b + mg::OFF_M);
        s.l_smem        = (float*)  (b + mg::OFF_L);
        s.w_head_sc_all = (float*)  (b + mg::OFF_W_SC_ALL);
        s.w_fp8_base    = (uint8_t*)(b + mg::OFF_W_FP8);
        s.v_trans       = (uint8_t*)(b + mg::OFF_V_TRANS);
        s.mbar_kv       = (uint64_t*)(b + mg::OFF_MBAR_KV);
        return s;
    }
    __device__ __forceinline__ uint8_t* w_fp8_grp(int g) {
        return w_fp8_base + g * HPB * W_FP8_STRIDE;
    }
    __device__ bf16* q_rope_staging() { return (bf16*)v_trans; }
};


// ============================================================================
// Single-group prefill kernel (num_heads <= 16, memory-bound, 88% BW)
// ============================================================================
template <int NUM_HEADS>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
sparse_mla_prefill_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    bf16* __restrict__ output,
    float sm_scale,
    int num_tokens)
{
    static constexpr int REPLICATE_H = NUM_HEADS / HPB;
    const int s_i = blockIdx.x / REPLICATE_H;
    const int h_start = (blockIdx.x % REPLICATE_H) * HPB;
    if (s_i >= num_tokens) return;

    const int warp_rank = threadIdx.x / 32;
    const int wy = warp_rank / 4;

    extern __shared__ char smem_raw[];
    PrefillSmem sm = PrefillSmem::init(smem_raw);

    // Initialize mbarriers before warp specialization (use barrier 3 for sync)
    if (threadIdx.x == 0) {
        mbarrier_init(sm.mbar_kv + 0, 1);
        mbarrier_init(sm.mbar_kv + 1, 1);
    }
    bar_sync_t<3, BLOCK_THREADS>();

    if (wy == 2) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));

        const int io_tid = threadIdx.x - N_MATH_WARPS * 32;
        const int32_t* idx_base = indices + (size_t)s_i * TOPK;

        prefill_io_bulk_gather_tile(sm.kv_bufs[0], idx_base,
                                    KV_cache, sm.mbar_kv + 0, io_tid);

        for (int ni = 0; ni < NI; ni++) {
            if (ni + 1 < NI)
                prefill_io_bulk_gather_tile(sm.kv_bufs[(ni + 1) & 1],
                               idx_base + (ni + 1) * BI, KV_cache,
                               sm.mbar_kv + ((ni + 1) & 1), io_tid);
            bar_sync_t<1, BLOCK_THREADS>();
        }

    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(200));

        const int lane = threadIdx.x & 31;
        const int mwarp = warp_rank;
        const int gid = lane >> 2, tid = lane & 3;
        const float sm_scale_log2e = sm_scale * LOG2E;
        const bf16* q_base = Q + (size_t)s_i * NUM_HEADS * DIM + (size_t)h_start * DIM;
        const int32_t* idx_base = indices + (size_t)s_i * TOPK;

        prefill_quantize_q(sm.q_nope_fp8, sm.q_nope_sc, sm.q_rope,
                           q_base, sm.reduce_buf);
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

        for (int ni = 0; ni < NI; ni++) {
            uint8_t* kv_smem = sm.kv_bufs[ni & 1];
            const int32_t* ib = idx_base + ni * BI;
            const int qk_nb = mwarp * ENTRIES_PER_WARP;

            for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = 0.f;

            float qk[4] = {0.f, 0.f, 0.f, 0.f};
            prefill_compute_qk_nope(qk, sm.q_nope_fp8, sm.q_nope_sc, kv_smem, qk_nb, lane);

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

            // ── Online softmax (4 barriers) ─────────────────────────
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

                    prefill_transpose_v_chunk(sm.v_trans, kv_smem, v_off, lane);
                    bar_sync_t<2, MATH_THREADS>();

                    #pragma unroll
                    for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
                        int ti = vc * NT_PER_WARP_XV + nt;
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
                        acc_o[ti][0] += xv[0] * sc0; acc_o[ti][1] += xv[1] * sc0;
                        acc_o[ti][2] += xv[2] * sc1; acc_o[ti][3] += xv[3] * sc1;
                    }

                    if (vc < N_V_CHUNKS - 1)
                        bar_sync_t<2, MATH_THREADS>();
                }
            }

            bar_arrive_t<1, BLOCK_THREADS>();
            if (ni + 1 < NI) {
                const int next_phase = ((ni + 1) >> 1) & 1;
                mbarrier_wait_parity(sm.mbar_kv + ((ni + 1) & 1), next_phase);
            }
        }

        // ── Coalesced output via smem staging ───────────────────
        // Reuse kv_bufs[0] as staging buffer (IO warps are done)
        bf16* staging = reinterpret_cast<bf16*>(sm.kv_bufs[0]);

        float il0 = (sm.l_smem[gid] > 0.f) ? (1.f / sm.l_smem[gid]) : 0.f;
        float il1 = (sm.l_smem[gid + 8] > 0.f) ? (1.f / sm.l_smem[gid + 8]) : 0.f;

        // Step 1: scatter acc_o → staging[head][dim] (bf16_2 writes, padded stride)
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

        // Step 2: coalesced 128-bit writes from staging → global
        {
            const size_t out_base = (size_t)s_i * NUM_HEADS * D_NOPE + (size_t)h_start * D_NOPE;
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
// Multi-group prefill kernel (num_heads > 16, e.g. 128)
//
// Optimized: both head groups processed in parallel within each NI
// iteration, cutting barrier count from ~23 to ~11 per tile.
//   Phase 1: QK + softmax for both groups with 4 combined barriers (was 8)
//   Phase 2: w_fp8 for both groups → 1 barrier → MMA both groups (7 total, was 15)
// ============================================================================
template <int NUM_HEADS>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
sparse_mla_prefill_mg_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    bf16* __restrict__ output,
    float sm_scale,
    int num_tokens)
{
    static constexpr int REPLICATE_H = NUM_HEADS / MG_HEADS_PER_CTA;
    const int s_i = blockIdx.x / REPLICATE_H;
    const int h_start = (blockIdx.x % REPLICATE_H) * MG_HEADS_PER_CTA;
    if (s_i >= num_tokens) return;

    const int warp_rank = threadIdx.x / 32;
    const int wy = warp_rank / 4;

    extern __shared__ char smem_raw[];
    PrefillSmemMG sm = PrefillSmemMG::init(smem_raw);

    if (threadIdx.x == 0) {
        mbarrier_init(sm.mbar_kv + 0, 1);
        mbarrier_init(sm.mbar_kv + 1, 1);
    }
    bar_sync_t<3, BLOCK_THREADS>();

    if (wy == 2) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));
        const int io_tid = threadIdx.x - N_MATH_WARPS * 32;
        const int32_t* idx_base = indices + (size_t)s_i * TOPK;
        prefill_io_bulk_gather_tile(sm.kv_bufs[0], idx_base,
                                    KV_cache, sm.mbar_kv + 0, io_tid);
        for (int ni = 0; ni < NI; ni++) {
            if (ni + 1 < NI)
                prefill_io_bulk_gather_tile(sm.kv_bufs[(ni + 1) & 1],
                               idx_base + (ni + 1) * BI, KV_cache,
                               sm.mbar_kv + ((ni + 1) & 1), io_tid);
            bar_sync_t<1, BLOCK_THREADS>();
        }
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(200));

        const int lane = threadIdx.x & 31;
        const int mwarp = warp_rank;
        const int gid = lane >> 2, tid = lane & 3;
        const float sm_scale_log2e = sm_scale * LOG2E;
        const int32_t* idx_base = indices + (size_t)s_i * TOPK;

        QRopeRegs q_rope_regs[MG_N_HG];
        float acc_o[MG_N_HG][ACC_TILES][4];

        #pragma unroll
        for (int g = 0; g < MG_N_HG; g++) {
            const bf16* q_base_g = Q + (size_t)s_i * NUM_HEADS * DIM
                                     + (size_t)(h_start + g * HPB) * DIM;
            bf16* rope_staging = sm.q_rope_staging();
            prefill_quantize_q(sm.q_nope_fp8[g], sm.q_nope_sc[g],
                               rope_staging, q_base_g, sm.reduce_buf);
            q_rope_regs[g] = preload_q_rope_regs(rope_staging, lane);
            bar_sync_t<2, MATH_THREADS>();

            for (int h = threadIdx.x; h < HPB; h += MATH_THREADS) {
                sm.m_smem[g * HPB + h] = -1e30f;
                sm.l_smem[g * HPB + h] = 0.f;
            }
            #pragma unroll
            for (int t = 0; t < ACC_TILES; t++)
                acc_o[g][t][0] = acc_o[g][t][1] = acc_o[g][t][2] = acc_o[g][t][3] = 0.f;
        }

        bar_sync_t<2, MATH_THREADS>();
        mbarrier_wait_parity(sm.mbar_kv + 0, 0);

        float weights[MG_N_HG][4];

        for (int ni = 0; ni < NI; ni++) {
            uint8_t* kv_smem = sm.kv_bufs[ni & 1];
            const int32_t* ib = idx_base + ni * BI;
            const int qk_nb = mwarp * ENTRIES_PER_WARP;
            const int e0i = qk_nb + tid * 2, e1i = e0i + 1;

            // ── PHASE 1: QK + softmax — both groups in parallel ─────

            // Clear w_head_sc_all for both groups
            for (int i = threadIdx.x; i < MG_N_HG * N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = 0.f;

            // Compute QK for both groups (no barriers between groups)
            float sc_g[MG_N_HG][4];

            // Load rope B operands once (shared across groups)
            int entry_idx = ib[qk_nb + gid];
            entry_idx = (entry_idx >= 0) ? entry_idx : 0;
            const bf16* g_rope = reinterpret_cast<const bf16*>(
                KV_cache + (size_t)entry_idx * KV_PACKED_BYTES + KV_ROPE_BYTE_OFFSET);

            #pragma unroll
            for (int g = 0; g < MG_N_HG; g++) {
                float qk[4] = {0.f, 0.f, 0.f, 0.f};
                prefill_compute_qk_nope(qk, sm.q_nope_fp8[g], sm.q_nope_sc[g],
                                kv_smem, qk_nb, lane);
                compute_qk_rope(qk, q_rope_regs[g], g_rope, lane);
                {
                    if (ib[e0i] < 0) { qk[0] = -1e30f; qk[2] = -1e30f; }
                    if (ib[e1i] < 0) { qk[1] = -1e30f; qk[3] = -1e30f; }
                }
                sc_g[g][0] = qk[0]*sm_scale_log2e; sc_g[g][1] = qk[1]*sm_scale_log2e;
                sc_g[g][2] = qk[2]*sm_scale_log2e; sc_g[g][3] = qk[3]*sm_scale_log2e;
            }

            // ── Combined max reduction for both groups (1 barrier) ──
            #pragma unroll
            for (int g = 0; g < MG_N_HG; g++) {
                float lm0 = fmaxf(sc_g[g][0], sc_g[g][1]);
                float lm1 = fmaxf(sc_g[g][2], sc_g[g][3]);
                lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 1));
                lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 2));
                lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 1));
                lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 2));
                if (tid == 0) {
                    sm.reduce_buf[g * MG_REDUCE_GRP_STRIDE + mwarp * HPB + gid] = lm0;
                    sm.reduce_buf[g * MG_REDUCE_GRP_STRIDE + mwarp * HPB + gid + 8] = lm1;
                }
            }
            bar_sync_t<2, MATH_THREADS>();

            // ── Cross-warp max + alpha/nm for both groups (1 barrier) ──
            if (threadIdx.x < MG_N_HG * HPB) {
                const int g = threadIdx.x / HPB;
                const int h = threadIdx.x % HPB;
                float* g_m = sm.m_smem + g * HPB;
                float* g_l = sm.l_smem + g * HPB;
                float old_m = g_m[h], tm = -1e30f;
                #pragma unroll
                for (int w = 0; w < N_MATH_WARPS; w++)
                    tm = fmaxf(tm, sm.reduce_buf[g * MG_REDUCE_GRP_STRIDE + w * HPB + h]);
                float nm = fmaxf(old_m, tm);
                float alpha = exp2f(old_m - nm);
                g_m[h] = nm;
                g_l[h] *= alpha;
                sm.reduce_buf[g * MG_N_HG * HPB + h] = alpha;
                sm.reduce_buf[g * MG_N_HG * HPB + HPB + h] = nm;
            }
            bar_sync_t<2, MATH_THREADS>();

            // ── Rescale + exp weights for both groups (no barrier) ──
            #pragma unroll
            for (int g = 0; g < MG_N_HG; g++) {
                float alpha0 = sm.reduce_buf[g * MG_N_HG * HPB + gid];
                float alpha1 = sm.reduce_buf[g * MG_N_HG * HPB + gid + 8];
                float nm0 = sm.reduce_buf[g * MG_N_HG * HPB + HPB + gid];
                float nm1 = sm.reduce_buf[g * MG_N_HG * HPB + HPB + gid + 8];
                #pragma unroll
                for (int t = 0; t < ACC_TILES; t++) {
                    acc_o[g][t][0] *= alpha0; acc_o[g][t][1] *= alpha0;
                    acc_o[g][t][2] *= alpha1; acc_o[g][t][3] *= alpha1;
                }
                weights[g][0] = exp2f(sc_g[g][0] - nm0);
                weights[g][1] = exp2f(sc_g[g][1] - nm0);
                weights[g][2] = exp2f(sc_g[g][2] - nm1);
                weights[g][3] = exp2f(sc_g[g][3] - nm1);
            }

            // ── Combined sum + atomicMax for both groups (1 barrier) ──
            #pragma unroll
            for (int g = 0; g < MG_N_HG; g++) {
                float w0 = weights[g][0], w1 = weights[g][1];
                float w2 = weights[g][2], w3 = weights[g][3];
                float ls0 = w0 + w1, ls1 = w2 + w3;
                ls0 += __shfl_xor_sync(0xffffffff, ls0, 1);
                ls0 += __shfl_xor_sync(0xffffffff, ls0, 2);
                ls1 += __shfl_xor_sync(0xffffffff, ls1, 1);
                ls1 += __shfl_xor_sync(0xffffffff, ls1, 2);
                if (tid == 0) {
                    sm.sum_reduce_buf[g * MG_REDUCE_GRP_STRIDE + mwarp * HPB + gid] = ls0;
                    sm.sum_reduce_buf[g * MG_REDUCE_GRP_STRIDE + mwarp * HPB + gid + 8] = ls1;
                }

                float* g_wsc = sm.w_head_sc_all + g * N_V_CHUNKS * HPB;
                #pragma unroll
                for (int vc = 0; vc < N_V_CHUNKS; vc++) {
                    float vsc0 = reinterpret_cast<const float*>(kv_smem + e0i * KV_SMEM_STRIDE + D_NOPE)[vc];
                    float vsc1 = reinterpret_cast<const float*>(kv_smem + e1i * KV_SMEM_STRIDE + D_NOPE)[vc];
                    atomicMax(reinterpret_cast<int*>(&g_wsc[vc * HPB + gid]),
                        __float_as_int(fmaxf(fabsf(w0 * vsc0), fabsf(w1 * vsc1))));
                    atomicMax(reinterpret_cast<int*>(&g_wsc[vc * HPB + gid + 8]),
                        __float_as_int(fmaxf(fabsf(w2 * vsc0), fabsf(w3 * vsc1))));
                }
            }
            bar_sync_t<2, MATH_THREADS>();

            // ── Normalize scales + accumulate l for both groups (1 barrier) ──
            if (threadIdx.x < MG_N_HG * HPB) {
                const int g = threadIdx.x / HPB;
                const int h = threadIdx.x % HPB;
                float* g_l = sm.l_smem + g * HPB;
                float ts = 0.f;
                #pragma unroll
                for (int w = 0; w < N_MATH_WARPS; w++)
                    ts += sm.sum_reduce_buf[g * MG_REDUCE_GRP_STRIDE + w * HPB + h];
                g_l[h] += ts;
            }
            for (int i = threadIdx.x; i < MG_N_HG * N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = fmaxf(sm.w_head_sc_all[i], 1e-10f) / FP8_MAX;
            bar_sync_t<2, MATH_THREADS>();

            // ── PHASE 2: XV — both groups' w_fp8 + shared V transpose ──
            {
                #pragma unroll
                for (int vc = 0; vc < N_V_CHUNKS; vc++) {
                    const int v_off = vc * V_CHUNK;

                    // V transpose (shared across groups)
                    prefill_transpose_v_chunk(sm.v_trans, kv_smem, v_off, lane);

                    // Quantize w_fp8 for BOTH groups simultaneously (different smem rows)
                    #pragma unroll
                    for (int g = 0; g < MG_N_HG; g++) {
                        float* g_wsc = sm.w_head_sc_all + g * N_V_CHUNKS * HPB + vc * HPB;
                        float gw0 = weights[g][0], gw1 = weights[g][1];
                        float gw2 = weights[g][2], gw3 = weights[g][3];

                        float si0 = 1.f / g_wsc[gid], si1 = 1.f / g_wsc[gid + 8];
                        float vsc0 = reinterpret_cast<const float*>(kv_smem + e0i * KV_SMEM_STRIDE + D_NOPE)[vc];
                        float vsc1 = reinterpret_cast<const float*>(kv_smem + e1i * KV_SMEM_STRIDE + D_NOPE)[vc];
                        float ws00 = gw0 * vsc0, ws01 = gw1 * vsc1;
                        float ws10 = gw2 * vsc0, ws11 = gw3 * vsc1;
                        __nv_fp8_e4m3 f00(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws00 * si0)));
                        __nv_fp8_e4m3 f01(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws01 * si0)));
                        __nv_fp8_e4m3 f10(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws10 * si1)));
                        __nv_fp8_e4m3 f11(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws11 * si1)));
                        uint8_t* gw = sm.w_fp8_grp(g);
                        gw[gid * W_FP8_STRIDE + e0i] = f00.__x;
                        gw[gid * W_FP8_STRIDE + e1i] = f01.__x;
                        gw[(gid + 8) * W_FP8_STRIDE + e0i] = f10.__x;
                        gw[(gid + 8) * W_FP8_STRIDE + e1i] = f11.__x;
                    }

                    bar_sync_t<2, MATH_THREADS>();

                    // XV MMA for BOTH groups (reading from group-specific w_fp8 rows)
                    #pragma unroll
                    for (int g = 0; g < MG_N_HG; g++) {
                        float* g_wsc = sm.w_head_sc_all + g * N_V_CHUNKS * HPB + vc * HPB;
                        uint8_t* gw = sm.w_fp8_grp(g);

                        #pragma unroll
                        for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
                            int ti = vc * NT_PER_WARP_XV + nt;
                            int nl = mwarp * (NT_PER_WARP_XV * 8) + nt * 8;
                            float xv[4] = {0.f, 0.f, 0.f, 0.f};
                            #pragma unroll
                            for (int kstep = 0; kstep < XV_KSTEPS; kstep++) {
                                int ko = kstep * 32;
                                uint32_t a0, a1, a2, a3, b0, b1;
                                ldmatrix_load_A_fp8(a0, a1, a2, a3, gw + ko, W_FP8_STRIDE, lane);
                                ldmatrix_load_B_fp8(b0, b1, sm.v_trans + nl * V_TRANS_STRIDE + ko, V_TRANS_STRIDE, lane);
                                MmaFp8Result r = mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, xv[0], xv[1], xv[2], xv[3]);
                                xv[0] = r.d0; xv[1] = r.d1; xv[2] = r.d2; xv[3] = r.d3;
                            }
                            float sc0 = g_wsc[gid], sc1 = g_wsc[gid + 8];
                            acc_o[g][ti][0] += xv[0] * sc0; acc_o[g][ti][1] += xv[1] * sc0;
                            acc_o[g][ti][2] += xv[2] * sc1; acc_o[g][ti][3] += xv[3] * sc1;
                        }
                    }

                    if (vc < N_V_CHUNKS - 1)
                        bar_sync_t<2, MATH_THREADS>();
                }
            }

            bar_arrive_t<1, BLOCK_THREADS>();
            if (ni + 1 < NI) {
                const int next_phase = ((ni + 1) >> 1) & 1;
                mbarrier_wait_parity(sm.mbar_kv + ((ni + 1) & 1), next_phase);
            }
        }

        // ── Coalesced output via smem staging ───────────────────
        bf16* staging = reinterpret_cast<bf16*>(sm.kv_bufs[0]);

        #pragma unroll
        for (int g = 0; g < MG_N_HG; g++) {
            float* g_l = sm.l_smem + g * HPB;
            int g_h_base = h_start + g * HPB;
            float il0 = (g_l[gid] > 0.f) ? (1.f / g_l[gid]) : 0.f;
            float il1 = (g_l[gid + 8] > 0.f) ? (1.f / g_l[gid + 8]) : 0.f;

            #pragma unroll
            for (int t = 0; t < ACC_TILES; t++) {
                constexpr int _NT8 = NT_PER_WARP_XV * 8;
                int c = t / NT_PER_WARP_XV, lnt = t % NT_PER_WARP_XV;
                int d0 = c * V_CHUNK + mwarp * _NT8 + lnt * 8 + tid * 2;
                *reinterpret_cast<bf16_2*>(&staging[gid * OUT_STAGING_STRIDE + d0]) =
                    __floats2bfloat162_rn(acc_o[g][t][0] * il0, acc_o[g][t][1] * il0);
                *reinterpret_cast<bf16_2*>(&staging[(gid + 8) * OUT_STAGING_STRIDE + d0]) =
                    __floats2bfloat162_rn(acc_o[g][t][2] * il1, acc_o[g][t][3] * il1);
            }
            bar_sync_t<2, MATH_THREADS>();

            {
                const size_t out_base = (size_t)s_i * NUM_HEADS * D_NOPE + (size_t)g_h_base * D_NOPE;
                for (int idx = threadIdx.x; idx < HPB * OUT_TILES_PER_HEAD; idx += MATH_THREADS) {
                    int h = idx / OUT_TILES_PER_HEAD;
                    int d8 = idx - h * OUT_TILES_PER_HEAD;
                    uint4 val = *reinterpret_cast<const uint4*>(
                        &staging[h * OUT_STAGING_STRIDE + d8 * OUT_VEC]);
                    *reinterpret_cast<uint4*>(
                        &output[out_base + h * D_NOPE + d8 * OUT_VEC]) = val;
                }
            }
            if (g < MG_N_HG - 1)
                bar_sync_t<2, MATH_THREADS>();
        }
    }
}

// ============================================================================
// Launch dispatcher — template instantiation + runtime dispatch
// ============================================================================

template <int NUM_HEADS>
void launch_sg(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
               bf16* output, float sm_scale, int num_tokens,
               cudaStream_t stream)
{
    constexpr size_t smem_bytes = sg::TOTAL;
    constexpr int REPLICATE_H = NUM_HEADS / HPB;
    dim3 grid(num_tokens * REPLICATE_H);
    dim3 block(BLOCK_THREADS);

    static bool configured = false;
    if (!configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(sparse_mla_prefill_kernel<NUM_HEADS>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            smem_bytes);
        configured = true;
    }
    sparse_mla_prefill_kernel<NUM_HEADS><<<grid, block, smem_bytes, stream>>>(
        Q, KV_cache, indices, output, sm_scale, num_tokens);
}

template <int NUM_HEADS>
void launch_mg(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
               bf16* output, float sm_scale, int num_tokens,
               cudaStream_t stream)
{
    constexpr size_t smem_bytes = mg::TOTAL;
    constexpr int REPLICATE_H = NUM_HEADS / MG_HEADS_PER_CTA;
    dim3 grid(num_tokens * REPLICATE_H);
    dim3 block(BLOCK_THREADS);

    static bool configured = false;
    if (!configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(sparse_mla_prefill_mg_kernel<NUM_HEADS>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            smem_bytes);
        configured = true;
    }
    sparse_mla_prefill_mg_kernel<NUM_HEADS><<<grid, block, smem_bytes, stream>>>(
        Q, KV_cache, indices, output, sm_scale, num_tokens);
}

void sparse_mla_prefill_launch(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output, float sm_scale,
    int num_heads, int num_tokens, int topk, int BI_param,
    cudaStream_t stream)
{
    TORCH_CHECK(topk == TOPK, "topk must be ", TOPK, " (compile-time constant); got ", topk);

    auto Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr());
    auto KV_ptr = reinterpret_cast<const uint8_t*>(KV_cache.data_ptr());
    auto idx_ptr = indices.data_ptr<int32_t>();
    auto out_ptr = reinterpret_cast<bf16*>(output.data_ptr());

    // Dispatch: NUM_HEADS is a template parameter → REPLICATE_H, address
    // arithmetic with NUM_HEADS*DIM, and all derived constants are constexpr.
    // TOPK is already constexpr (2048) via common.cuh → NI, TOTAL_KV_CHUNKS, etc.
    switch (num_heads) {
    case 16:  launch_sg<16> (Q_ptr, KV_ptr, idx_ptr, out_ptr, sm_scale, num_tokens, stream); break;
    case 32:  launch_mg<32> (Q_ptr, KV_ptr, idx_ptr, out_ptr, sm_scale, num_tokens, stream); break;
    case 64:  launch_mg<64> (Q_ptr, KV_ptr, idx_ptr, out_ptr, sm_scale, num_tokens, stream); break;
    case 128: launch_mg<128>(Q_ptr, KV_ptr, idx_ptr, out_ptr, sm_scale, num_tokens, stream); break;
    default:
        TORCH_CHECK(false, "num_heads must be 16, 32, 64, or 128; got ", num_heads);
    }
}
