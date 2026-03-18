#include "common.cuh"
#include "mma_sm120.cuh"
#include "smem_utils.cuh"

#include <torch/extension.h>

// ============================================================================
// Sparse MLA Prefill Kernel — Optimized for SM120a
//
// Key optimizations vs v1:
//   - Batched XV scale computation (all 4 V chunks in one 2-barrier pass)
//   - Separate sum_reduce_buf eliminates 1 softmax barrier/tile
//   - All tile sizes are constexpr for full compile-time unrolling
//   - cp.async with L2 prefetch hints
//   - 11 barriers/tile (vs 25 in v1): 4 softmax + 4 XV ready + 3 XV done
//
// Architecture: 12 warps (384 threads) = 8 math + 4 IO
//   Math: setmaxnreg 200, warps 0-7
//   IO:   setmaxnreg 32,  warps 8-11
//   Barriers: 0 = data ready, 1 = buf consumed, 2 = math internal
//   Double-buffered KV: 2 × 64 × 528 = 66 KB
//   Total smem: ~87 KB < 100 KB
// ============================================================================

static constexpr int HPB = 16;
static constexpr int BI = 64;
static constexpr int NI = TOPK / BI;  // 32
static constexpr int N_MATH_WARPS = 8;
static constexpr int N_IO_WARPS = 4;
static constexpr int N_TOTAL_WARPS = N_MATH_WARPS + N_IO_WARPS;
static constexpr int BLOCK_THREADS = N_TOTAL_WARPS * 32;
static constexpr int MATH_THREADS = N_MATH_WARPS * 32;
static constexpr int IO_THREADS = N_IO_WARPS * 32;
static constexpr int V_CHUNK = 128;
static constexpr int N_V_CHUNKS = D_NOPE / V_CHUNK;
static constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / N_MATH_WARPS;
static constexpr int ACC_TILES = N_V_CHUNKS * NT_PER_WARP_XV;

static constexpr int KV_SMEM_STRIDE = D_NOPE + NUM_SCALES * sizeof(float);  // 528
static constexpr int Q_NOPE_STRIDE = D_NOPE + 16;   // 528: stride/4=132, 132%32=4
static constexpr int V_TRANS_STRIDE = BI + 16;       // 80:  16B-aligned for ldmatrix, 80/4=20, 20%32=20
static constexpr int W_FP8_STRIDE = BI + 16;         // 80:  16B-aligned for ldmatrix

// Smem sizes (all constexpr)
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
static constexpr size_t SMEM_TOTAL      = SMEM_Q_NOPE + SMEM_Q_SC + SMEM_Q_ROPE
                                        + 2 * SMEM_KV_BUF + SMEM_REDUCE + SMEM_SUM_REDUCE
                                        + SMEM_M + SMEM_L + SMEM_W_SC_ALL
                                        + SMEM_W_FP8 + SMEM_V_TRANS;

__device__ __forceinline__ void bar_arrive(int id, int cnt) {
    asm volatile("barrier.cta.arrive %0, %1;\n" :: "r"(id), "r"(cnt) : "memory");
}
__device__ __forceinline__ void bar_sync(int id, int cnt) {
    asm volatile("barrier.cta.sync %0, %1;\n" :: "r"(id), "r"(cnt) : "memory");
}

__device__ __forceinline__ void cp_async_16B_l2(void* smem_ptr, const void* gmem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(addr), "l"(gmem_ptr));
}

// ============================================================================
// Smem layout struct — all pointers derived from a single base
// ============================================================================
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

    __device__ static PrefillSmem init(char* base) {
        PrefillSmem s;
        char* p = base;
        s.q_nope_fp8    = (uint8_t*)p; p += SMEM_Q_NOPE;
        s.q_nope_sc     = (float*)p;   p += SMEM_Q_SC;
        s.q_rope        = (bf16*)p;    p += SMEM_Q_ROPE;
        s.kv_bufs[0]    = (uint8_t*)p; p += SMEM_KV_BUF;
        s.kv_bufs[1]    = (uint8_t*)p; p += SMEM_KV_BUF;
        s.reduce_buf    = (float*)p;   p += SMEM_REDUCE;
        s.sum_reduce_buf= (float*)p;   p += SMEM_SUM_REDUCE;
        s.m_smem        = (float*)p;   p += SMEM_M;
        s.l_smem        = (float*)p;   p += SMEM_L;
        s.w_head_sc_all = (float*)p;   p += SMEM_W_SC_ALL;
        s.w_fp8         = (uint8_t*)p; p += SMEM_W_FP8;
        s.v_trans       = (uint8_t*)p;
        return s;
    }
};

// ============================================================================
// IO path: gather KV nope+scales into double buffer with L2 prefetch
// ============================================================================
__device__ __forceinline__ void io_gather_tile(
    uint8_t* dst, const int32_t* ib,
    const uint8_t* __restrict__ KV_cache, int io_tid)
{
    constexpr int TOT = BI * KV_SMEM_STRIDE;
    for (int flat = io_tid * 16; flat < TOT; flat += IO_THREADS * 16) {
        int bi = flat / KV_SMEM_STRIDE, bo = flat % KV_SMEM_STRIDE;
        int idx = ib[bi]; idx = (idx >= 0) ? idx : 0;
        if (bo + 16 <= KV_SMEM_STRIDE)
            cp_async_16B_l2(dst + flat, KV_cache + (size_t)idx * KV_PACKED_BYTES + bo);
    }
    cp_async_commit();
    cp_async_wait_all();
}

// ============================================================================
// Math helpers
// ============================================================================
__device__ __forceinline__ void quantize_q_to_smem(
    uint8_t* q_nope_fp8, float* q_nope_sc, bf16* q_rope,
    const bf16* q_base, float* reduce_buf)
{
    float* amax = reduce_buf;
    for (int i = threadIdx.x; i < HPB * D_ROPE; i += MATH_THREADS) {
        int h = i / D_ROPE, d = i % D_ROPE;
        q_rope[h * D_ROPE + d] = q_base[h * DIM + D_NOPE + d];
    }
    for (int i = threadIdx.x; i < HPB * NUM_SCALES; i += MATH_THREADS)
        amax[i] = 0.f;
    bar_sync(2, MATH_THREADS);

    for (int idx = threadIdx.x; idx < HPB * D_NOPE; idx += MATH_THREADS) {
        int h = idx / D_NOPE, blk = (idx % D_NOPE) / QUANT_TILE;
        atomicMax(reinterpret_cast<int*>(&amax[h * NUM_SCALES + blk]),
                  __float_as_int(fabsf(__bfloat162float(q_base[h * DIM + idx % D_NOPE]))));
    }
    bar_sync(2, MATH_THREADS);

    for (int i = threadIdx.x; i < HPB * NUM_SCALES; i += MATH_THREADS)
        q_nope_sc[i] = fmaxf(amax[i], 1e-4f) / FP8_MAX;
    bar_sync(2, MATH_THREADS);

    for (int idx = threadIdx.x; idx < HPB * D_NOPE; idx += MATH_THREADS) {
        int h = idx / D_NOPE, d = idx % D_NOPE, blk = d / QUANT_TILE;
        float si = 1.f / q_nope_sc[h * NUM_SCALES + blk];
        float v = fmaxf(FP8_MIN, fminf(FP8_MAX, __bfloat162float(q_base[h * DIM + d]) * si));
        __nv_fp8_e4m3 fp8v(v);
        q_nope_fp8[h * Q_NOPE_STRIDE + d] = fp8v.__x;
    }
}

__device__ __forceinline__ void compute_qk_nope(
    float qk[4], const uint8_t* q_nope_fp8, const float* q_nope_sc,
    const uint8_t* kv_smem, int qk_nb, int lane)
{
    int gid = lane >> 2, tid = lane & 3;
    #pragma unroll
    for (int blk = 0; blk < NUM_SCALES; blk++) {
        float ab[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int ks = 0; ks < QUANT_TILE / 32; ks++) {
            int ko = blk * QUANT_TILE + ks * 32;
            uint32_t a0, a1, a2, a3, b0, b1;
            ldmatrix_load_A_fp8(a0, a1, a2, a3,
                q_nope_fp8 + ko, Q_NOPE_STRIDE, lane);
            ldmatrix_load_B_fp8(b0, b1,
                kv_smem + qk_nb * KV_SMEM_STRIDE + ko, KV_SMEM_STRIDE, lane);
            MmaFp8Result r = mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1,
                                               ab[0], ab[1], ab[2], ab[3]);
            ab[0] = r.d0; ab[1] = r.d1; ab[2] = r.d2; ab[3] = r.d3;
        }
        float qs0 = q_nope_sc[gid * NUM_SCALES + blk];
        float qs1 = q_nope_sc[(gid + 8) * NUM_SCALES + blk];
        int e0 = qk_nb + tid * 2, e1 = qk_nb + tid * 2 + 1;
        float k0 = reinterpret_cast<const float*>(kv_smem + e0 * KV_SMEM_STRIDE + D_NOPE)[blk];
        float k1 = reinterpret_cast<const float*>(kv_smem + e1 * KV_SMEM_STRIDE + D_NOPE)[blk];
        qk[0] += ab[0] * qs0 * k0; qk[1] += ab[1] * qs0 * k1;
        qk[2] += ab[2] * qs1 * k0; qk[3] += ab[3] * qs1 * k1;
    }
}

// Q rope MMA fragments preloaded to registers (16 uint32 = 16 regs per thread)
static constexpr int N_ROPE_CHUNKS = D_ROPE / 16;  // 4
struct QRopeRegs {
    uint32_t a[N_ROPE_CHUNKS][4];  // [k-chunk][a0,a1,a2,a3]
};

__device__ __forceinline__ QRopeRegs preload_q_rope_regs(
    const bf16* q_rope_smem, int lane)
{
    QRopeRegs regs;
    #pragma unroll
    for (int ks = 0; ks < N_ROPE_CHUNKS; ks++) {
        ldmatrix_load_A_bf16(regs.a[ks][0], regs.a[ks][1],
                              regs.a[ks][2], regs.a[ks][3],
                              q_rope_smem + ks * 16, D_ROPE, lane);
    }
    return regs;
}

__device__ __forceinline__ void compute_qk_rope(
    float qk[4], const QRopeRegs& q_rope_regs, const bf16* g_rope, int lane)
{
    int tid = lane & 3;
    float ra[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int ks = 0; ks < N_ROPE_CHUNKS; ks++) {
        int ko = ks * 16;
        uint32_t b0, b1;
        b0 = *reinterpret_cast<const uint32_t*>(g_rope + ko + tid * 2);
        b1 = *reinterpret_cast<const uint32_t*>(g_rope + ko + 8 + tid * 2);
        MmaBf16Result r = mma_bf16_m16n8k16(
            q_rope_regs.a[ks][0], q_rope_regs.a[ks][1],
            q_rope_regs.a[ks][2], q_rope_regs.a[ks][3],
            b0, b1, ra[0], ra[1], ra[2], ra[3]);
        ra[0] = r.d0; ra[1] = r.d1; ra[2] = r.d2; ra[3] = r.d3;
    }
    qk[0] += ra[0]; qk[1] += ra[1]; qk[2] += ra[2]; qk[3] += ra[3];
}

// ============================================================================
// Main prefill kernel
// ============================================================================
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
sparse_mla_prefill_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    bf16* __restrict__ output,
    float sm_scale,
    int num_heads, int num_tokens, int topk)
{
    const int REPLICATE_H = num_heads / HPB;
    const int s_i = blockIdx.x / REPLICATE_H;
    const int h_start = (blockIdx.x % REPLICATE_H) * HPB;
    if (s_i >= num_tokens) return;

    const int warp_rank = threadIdx.x / 32;
    const int wy = warp_rank / 4;

    extern __shared__ char smem_raw[];
    PrefillSmem sm = PrefillSmem::init(smem_raw);

    // ================================================================
    // IO PATH (warps 8-11)
    // ================================================================
    if (wy == 2) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));

        const int io_tid = threadIdx.x - N_MATH_WARPS * 32;
        const int32_t* idx_base = indices + (size_t)s_i * topk;

        io_gather_tile(sm.kv_bufs[0], idx_base, KV_cache, io_tid);
        bar_arrive(0, BLOCK_THREADS);

        for (int ni = 0; ni < NI; ni++) {
            if (ni + 1 < NI)
                io_gather_tile(sm.kv_bufs[(ni + 1) & 1],
                               idx_base + (ni + 1) * BI, KV_cache, io_tid);
            bar_sync(1, BLOCK_THREADS);
            if (ni + 1 < NI)
                bar_arrive(0, BLOCK_THREADS);
        }

    // ================================================================
    // MATH PATH (warps 0-7)
    // ================================================================
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(200));

        const int lane = threadIdx.x & 31;
        const int mwarp = warp_rank;
        const int gid = lane >> 2, tid = lane & 3;
        const float sm_scale_log2e = sm_scale * LOG2E;
        const bf16* q_base = Q + (size_t)s_i * num_heads * DIM + (size_t)h_start * DIM;
        const int32_t* idx_base = indices + (size_t)s_i * topk;

        // ---- Q quantization (one-time, 4 barriers) ----
        quantize_q_to_smem(sm.q_nope_fp8, sm.q_nope_sc, sm.q_rope,
                           q_base, sm.reduce_buf);

        // Preload Q rope MMA fragments to registers (avoids smem reads in hot loop)
        QRopeRegs q_rope_regs = preload_q_rope_regs(sm.q_rope, lane);

        // Init softmax state
        for (int h = threadIdx.x; h < HPB; h += MATH_THREADS) {
            sm.m_smem[h] = -1e30f;
            sm.l_smem[h] = 0.f;
        }

        float acc_o[ACC_TILES][4];
        #pragma unroll
        for (int t = 0; t < ACC_TILES; t++)
            acc_o[t][0] = acc_o[t][1] = acc_o[t][2] = acc_o[t][3] = 0.f;

        bar_sync(2, MATH_THREADS);
        bar_sync(0, BLOCK_THREADS);  // wait for IO tile 0

        // ---- Main loop over KV tiles ----
        for (int ni = 0; ni < NI; ni++) {
            uint8_t* kv_smem = sm.kv_bufs[ni & 1];
            const int32_t* ib = idx_base + ni * BI;
            const int qk_nb = mwarp * 8;

            // Init batched XV scales for this tile (overlaps with QK MMA below)
            for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = 0.f;

            // ---- QK MMA nope (FP8) ----
            float qk[4] = {0.f, 0.f, 0.f, 0.f};
            compute_qk_nope(qk, sm.q_nope_fp8, sm.q_nope_sc, kv_smem, qk_nb, lane);

            // ---- QK MMA rope (BF16, from global L1-cached) ----
            {
                int ne = qk_nb + gid;
                int entry_idx = ib[ne];
                entry_idx = (entry_idx >= 0) ? entry_idx : 0;
                const bf16* g_rope = reinterpret_cast<const bf16*>(
                    KV_cache + (size_t)entry_idx * KV_PACKED_BYTES
                    + D_NOPE + NUM_SCALES * sizeof(float));
                compute_qk_rope(qk, q_rope_regs, g_rope, lane);
            }

            // ---- Mask invalid entries ----
            {
                int e0 = qk_nb + tid * 2, e1 = qk_nb + tid * 2 + 1;
                if (ib[e0] < 0) { qk[0] = -1e30f; qk[2] = -1e30f; }
                if (ib[e1] < 0) { qk[1] = -1e30f; qk[3] = -1e30f; }
            }

            // ============================================================
            // Online softmax (4 barriers, was 5)
            // ============================================================
            float s[4] = { qk[0] * sm_scale_log2e, qk[1] * sm_scale_log2e,
                           qk[2] * sm_scale_log2e, qk[3] * sm_scale_log2e };

            // Warp-level max reduce
            float lm0 = fmaxf(s[0], s[1]), lm1 = fmaxf(s[2], s[3]);
            lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 1));
            lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 2));
            lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 1));
            lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 2));
            if (tid == 0) {
                sm.reduce_buf[mwarp * HPB + gid] = lm0;
                sm.reduce_buf[mwarp * HPB + gid + 8] = lm1;
            }

            bar_sync(2, MATH_THREADS);  // BARRIER 1: local max ready

            if (threadIdx.x < HPB) {
                int h = threadIdx.x;
                float old_m = sm.m_smem[h], tm = -1e30f;
                for (int w = 0; w < N_MATH_WARPS; w++)
                    tm = fmaxf(tm, sm.reduce_buf[w * HPB + h]);
                float nm = fmaxf(old_m, tm);
                float alpha = exp2f(old_m - nm);
                sm.m_smem[h] = nm;
                sm.l_smem[h] *= alpha;
                sm.reduce_buf[h] = alpha;
                sm.reduce_buf[HPB + h] = nm;
            }

            bar_sync(2, MATH_THREADS);  // BARRIER 2: alpha, nm ready

            float alpha0 = sm.reduce_buf[gid], alpha1 = sm.reduce_buf[gid + 8];
            float nm0 = sm.reduce_buf[HPB + gid], nm1 = sm.reduce_buf[HPB + gid + 8];

            #pragma unroll
            for (int t = 0; t < ACC_TILES; t++) {
                acc_o[t][0] *= alpha0; acc_o[t][1] *= alpha0;
                acc_o[t][2] *= alpha1; acc_o[t][3] *= alpha1;
            }

            float w0 = exp2f(s[0] - nm0), w1 = exp2f(s[1] - nm0);
            float w2 = exp2f(s[2] - nm1), w3 = exp2f(s[3] - nm1);

            // Write local sums to SEPARATE buffer (no conflict with alpha/nm)
            float ls0 = w0 + w1, ls1 = w2 + w3;
            ls0 += __shfl_xor_sync(0xffffffff, ls0, 1);
            ls0 += __shfl_xor_sync(0xffffffff, ls0, 2);
            ls1 += __shfl_xor_sync(0xffffffff, ls1, 1);
            ls1 += __shfl_xor_sync(0xffffffff, ls1, 2);
            if (tid == 0) {
                sm.sum_reduce_buf[mwarp * HPB + gid] = ls0;
                sm.sum_reduce_buf[mwarp * HPB + gid + 8] = ls1;
            }

            // Batched XV scale computation: atomicMax for all 4 V chunks
            {
                int e0i = qk_nb + tid * 2, e1i = qk_nb + tid * 2 + 1;
                #pragma unroll
                for (int vc = 0; vc < N_V_CHUNKS; vc++) {
                    float vsc0 = reinterpret_cast<const float*>(
                        kv_smem + e0i * KV_SMEM_STRIDE + D_NOPE)[vc];
                    float vsc1 = reinterpret_cast<const float*>(
                        kv_smem + e1i * KV_SMEM_STRIDE + D_NOPE)[vc];
                    float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
                    float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
                    atomicMax(reinterpret_cast<int*>(
                        &sm.w_head_sc_all[vc * HPB + gid]),
                        __float_as_int(fmaxf(fabsf(ws00), fabsf(ws01))));
                    atomicMax(reinterpret_cast<int*>(
                        &sm.w_head_sc_all[vc * HPB + gid + 8]),
                        __float_as_int(fmaxf(fabsf(ws10), fabsf(ws11))));
                }
            }

            bar_sync(2, MATH_THREADS);  // BARRIER 3: sums + atomicMax done

            // Reduce sums and normalize XV scales
            if (threadIdx.x < HPB) {
                int h = threadIdx.x;
                float ts = 0.f;
                for (int w = 0; w < N_MATH_WARPS; w++)
                    ts += sm.sum_reduce_buf[w * HPB + h];
                sm.l_smem[h] += ts;
            }
            for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += MATH_THREADS)
                sm.w_head_sc_all[i] = fmaxf(sm.w_head_sc_all[i], 1e-10f) / FP8_MAX;

            bar_sync(2, MATH_THREADS);  // BARRIER 4: l_smem + all XV scales ready

            // ============================================================
            // XV MMA (4 V chunks × {quantize+transpose → barrier → MMA})
            // Saves 3 barriers per V chunk vs v1 (scale already computed)
            // ============================================================
            {
                int e0i = qk_nb + tid * 2, e1i = qk_nb + tid * 2 + 1;

                #pragma unroll
                for (int vc = 0; vc < N_V_CHUNKS; vc++) {
                    int v_off = vc * V_CHUNK;
                    float* vc_sc = sm.w_head_sc_all + vc * HPB;

                    // Quantize softmax weights × V_scale to FP8
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

                    // V transpose (KV nope region)
                    for (int idx = threadIdx.x; idx < V_CHUNK * BI; idx += MATH_THREADS) {
                        int d = idx / BI, e = idx % BI;
                        sm.v_trans[d * V_TRANS_STRIDE + e] =
                            kv_smem[e * KV_SMEM_STRIDE + v_off + d];
                    }

                    bar_sync(2, MATH_THREADS);  // XV BARRIER A: w_fp8 + v_trans ready

                    // XV MMA (FP8, ldmatrix loads)
                    #pragma unroll
                    for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
                        int ti = vc * NT_PER_WARP_XV + nt;
                        int nl = mwarp * (NT_PER_WARP_XV * 8) + nt * 8;
                        float xv[4] = {0.f, 0.f, 0.f, 0.f};
                        #pragma unroll
                        for (int kstep = 0; kstep < BI / 32; kstep++) {
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

                    // Skip final barrier for last V chunk (no reuse conflict)
                    if (vc < N_V_CHUNKS - 1)
                        bar_sync(2, MATH_THREADS);  // XV BARRIER B: MMA done
                }
            }

            // Signal IO: current buffer consumed
            bar_arrive(1, BLOCK_THREADS);
            if (ni + 1 < NI)
                bar_sync(0, BLOCK_THREADS);  // wait for next tile
        }

        // ---- Finalize: divide by sum ----
        #pragma unroll
        for (int t = 0; t < ACC_TILES; t++) {
            int c = t / NT_PER_WARP_XV, lnt = t % NT_PER_WARP_XV;
            int vb = c * V_CHUNK + mwarp * (NT_PER_WARP_XV * 8) + lnt * 8;
            int h0 = h_start + gid, h1 = h_start + gid + 8;
            int d0 = vb + tid * 2, d1 = vb + tid * 2 + 1;
            float il0 = (sm.l_smem[gid] > 0.f) ? (1.f / sm.l_smem[gid]) : 0.f;
            float il1 = (sm.l_smem[gid + 8] > 0.f) ? (1.f / sm.l_smem[gid + 8]) : 0.f;
            size_t b0 = (size_t)s_i * num_heads * D_NOPE + (size_t)h0 * D_NOPE;
            size_t b1 = (size_t)s_i * num_heads * D_NOPE + (size_t)h1 * D_NOPE;
            output[b0 + d0] = __float2bfloat16(acc_o[t][0] * il0);
            output[b0 + d1] = __float2bfloat16(acc_o[t][1] * il0);
            output[b1 + d0] = __float2bfloat16(acc_o[t][2] * il1);
            output[b1 + d1] = __float2bfloat16(acc_o[t][3] * il1);
        }
    }
}

// ============================================================================
// Multi-group prefill kernel — processes N_HG head groups per CTA
// Shares KV tile across groups, cutting DRAM reads by N_HG×.
// Phase 1: QK + softmax per group (sequential, sharing reduce_buf)
// Phase 2: XV with shared V transpose, per-group w_fp8 quantize
// ============================================================================
static constexpr int MG_N_HG = 2;
static constexpr int MG_HEADS_PER_CTA = MG_N_HG * HPB;  // 32

static constexpr size_t MG_SMEM_TOTAL =
    MG_N_HG * SMEM_Q_NOPE                      // per-group Q nope fp8
    + MG_N_HG * SMEM_Q_SC                       // per-group Q scales
    + 2 * SMEM_KV_BUF                           // shared double-buf KV
    + SMEM_REDUCE + SMEM_SUM_REDUCE             // shared reduce bufs
    + MG_N_HG * SMEM_M + MG_N_HG * SMEM_L      // per-group softmax state
    + MG_N_HG * SMEM_W_SC_ALL                   // per-group XV scales
    + SMEM_W_FP8 + SMEM_V_TRANS;                // shared XV buffers (v_trans also used as q_rope staging)

static_assert(MG_SMEM_TOTAL < 100 * 1024, "Multi-group smem exceeds 100 KB");

struct PrefillSmemMG {
    uint8_t* q_nope_fp8[MG_N_HG];
    float*   q_nope_sc[MG_N_HG];
    uint8_t* kv_bufs[2];
    float*   reduce_buf;
    float*   sum_reduce_buf;
    float*   m_smem;       // [MG_N_HG * HPB]
    float*   l_smem;       // [MG_N_HG * HPB]
    float*   w_head_sc_all; // [MG_N_HG * N_V_CHUNKS * HPB]
    uint8_t* w_fp8;
    uint8_t* v_trans;       // first 2KB overlapped as q_rope staging during init

    __device__ static PrefillSmemMG init(char* base) {
        PrefillSmemMG s;
        char* p = base;
        for (int g = 0; g < MG_N_HG; g++) { s.q_nope_fp8[g] = (uint8_t*)p; p += SMEM_Q_NOPE; }
        for (int g = 0; g < MG_N_HG; g++) { s.q_nope_sc[g] = (float*)p; p += SMEM_Q_SC; }
        s.kv_bufs[0]    = (uint8_t*)p; p += SMEM_KV_BUF;
        s.kv_bufs[1]    = (uint8_t*)p; p += SMEM_KV_BUF;
        s.reduce_buf    = (float*)p;   p += SMEM_REDUCE;
        s.sum_reduce_buf= (float*)p;   p += SMEM_SUM_REDUCE;
        s.m_smem        = (float*)p;   p += MG_N_HG * SMEM_M;
        s.l_smem        = (float*)p;   p += MG_N_HG * SMEM_L;
        s.w_head_sc_all = (float*)p;   p += MG_N_HG * SMEM_W_SC_ALL;
        s.w_fp8         = (uint8_t*)p; p += SMEM_W_FP8;
        s.v_trans       = (uint8_t*)p;
        return s;
    }
    __device__ bf16* q_rope_staging() { return (bf16*)v_trans; }
};

__global__ void __launch_bounds__(BLOCK_THREADS, 1)
sparse_mla_prefill_mg_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    bf16* __restrict__ output,
    float sm_scale,
    int num_heads, int num_tokens, int topk)
{
    const int REPLICATE_H = num_heads / MG_HEADS_PER_CTA;
    const int s_i = blockIdx.x / REPLICATE_H;
    const int h_start = (blockIdx.x % REPLICATE_H) * MG_HEADS_PER_CTA;
    if (s_i >= num_tokens) return;

    const int warp_rank = threadIdx.x / 32;
    const int wy = warp_rank / 4;

    extern __shared__ char smem_raw[];
    PrefillSmemMG sm = PrefillSmemMG::init(smem_raw);

    if (wy == 2) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));
        const int io_tid = threadIdx.x - N_MATH_WARPS * 32;
        const int32_t* idx_base = indices + (size_t)s_i * topk;
        io_gather_tile(sm.kv_bufs[0], idx_base, KV_cache, io_tid);
        bar_arrive(0, BLOCK_THREADS);
        for (int ni = 0; ni < NI; ni++) {
            if (ni + 1 < NI)
                io_gather_tile(sm.kv_bufs[(ni + 1) & 1],
                               idx_base + (ni + 1) * BI, KV_cache, io_tid);
            bar_sync(1, BLOCK_THREADS);
            if (ni + 1 < NI)
                bar_arrive(0, BLOCK_THREADS);
        }
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(200));

        const int lane = threadIdx.x & 31;
        const int mwarp = warp_rank;
        const int gid = lane >> 2, tid = lane & 3;
        const float sm_scale_log2e = sm_scale * LOG2E;
        const int32_t* idx_base = indices + (size_t)s_i * topk;

        // ---- Init: quantize Q and preload rope for all groups ----
        QRopeRegs q_rope_regs[MG_N_HG];
        float acc_o[MG_N_HG][ACC_TILES][4];

        #pragma unroll
        for (int g = 0; g < MG_N_HG; g++) {
            const bf16* q_base_g = Q + (size_t)s_i * num_heads * DIM
                                     + (size_t)(h_start + g * HPB) * DIM;
            bf16* rope_staging = sm.q_rope_staging();
            quantize_q_to_smem(sm.q_nope_fp8[g], sm.q_nope_sc[g],
                               rope_staging, q_base_g, sm.reduce_buf);
            q_rope_regs[g] = preload_q_rope_regs(rope_staging, lane);
            bar_sync(2, MATH_THREADS);

            for (int h = threadIdx.x; h < HPB; h += MATH_THREADS) {
                sm.m_smem[g * HPB + h] = -1e30f;
                sm.l_smem[g * HPB + h] = 0.f;
            }
            #pragma unroll
            for (int t = 0; t < ACC_TILES; t++)
                acc_o[g][t][0] = acc_o[g][t][1] = acc_o[g][t][2] = acc_o[g][t][3] = 0.f;
        }

        bar_sync(2, MATH_THREADS);
        bar_sync(0, BLOCK_THREADS);

        // ---- Main loop ----
        float weights[MG_N_HG][4];

        for (int ni = 0; ni < NI; ni++) {
            uint8_t* kv_smem = sm.kv_bufs[ni & 1];
            const int32_t* ib = idx_base + ni * BI;
            const int qk_nb = mwarp * 8;

            // ============================================================
            // PHASE 1: QK + softmax for each group (sequential)
            // ============================================================
            #pragma unroll
            for (int g = 0; g < MG_N_HG; g++) {
                float* g_m = sm.m_smem + g * HPB;
                float* g_l = sm.l_smem + g * HPB;
                float* g_wsc = sm.w_head_sc_all + g * N_V_CHUNKS * HPB;

                for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += MATH_THREADS)
                    g_wsc[i] = 0.f;

                float qk[4] = {0.f, 0.f, 0.f, 0.f};
                compute_qk_nope(qk, sm.q_nope_fp8[g], sm.q_nope_sc[g],
                                kv_smem, qk_nb, lane);
                {
                    int ne = qk_nb + gid;
                    int entry_idx = ib[ne];
                    entry_idx = (entry_idx >= 0) ? entry_idx : 0;
                    const bf16* g_rope = reinterpret_cast<const bf16*>(
                        KV_cache + (size_t)entry_idx * KV_PACKED_BYTES
                        + D_NOPE + NUM_SCALES * sizeof(float));
                    compute_qk_rope(qk, q_rope_regs[g], g_rope, lane);
                }
                {
                    int e0 = qk_nb + tid * 2, e1 = qk_nb + tid * 2 + 1;
                    if (ib[e0] < 0) { qk[0] = -1e30f; qk[2] = -1e30f; }
                    if (ib[e1] < 0) { qk[1] = -1e30f; qk[3] = -1e30f; }
                }

                float s[4] = { qk[0]*sm_scale_log2e, qk[1]*sm_scale_log2e,
                                qk[2]*sm_scale_log2e, qk[3]*sm_scale_log2e };
                float lm0 = fmaxf(s[0],s[1]), lm1 = fmaxf(s[2],s[3]);
                lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff,lm0,1));
                lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff,lm0,2));
                lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff,lm1,1));
                lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff,lm1,2));
                if (tid==0) { sm.reduce_buf[mwarp*HPB+gid]=lm0; sm.reduce_buf[mwarp*HPB+gid+8]=lm1; }
                bar_sync(2, MATH_THREADS);

                if (threadIdx.x < HPB) {
                    int h = threadIdx.x;
                    float old_m = g_m[h], tm = -1e30f;
                    for (int w = 0; w < N_MATH_WARPS; w++) tm = fmaxf(tm, sm.reduce_buf[w*HPB+h]);
                    float nm = fmaxf(old_m, tm), alpha = exp2f(old_m - nm);
                    g_m[h] = nm; g_l[h] *= alpha;
                    sm.reduce_buf[h] = alpha; sm.reduce_buf[HPB+h] = nm;
                }
                bar_sync(2, MATH_THREADS);

                float alpha0 = sm.reduce_buf[gid], alpha1 = sm.reduce_buf[gid+8];
                float nm0 = sm.reduce_buf[HPB+gid], nm1 = sm.reduce_buf[HPB+gid+8];
                #pragma unroll
                for (int t = 0; t < ACC_TILES; t++) {
                    acc_o[g][t][0]*=alpha0; acc_o[g][t][1]*=alpha0;
                    acc_o[g][t][2]*=alpha1; acc_o[g][t][3]*=alpha1;
                }

                float w0=exp2f(s[0]-nm0), w1=exp2f(s[1]-nm0);
                float w2=exp2f(s[2]-nm1), w3=exp2f(s[3]-nm1);
                weights[g][0]=w0; weights[g][1]=w1; weights[g][2]=w2; weights[g][3]=w3;

                float ls0=w0+w1, ls1=w2+w3;
                ls0+=__shfl_xor_sync(0xffffffff,ls0,1); ls0+=__shfl_xor_sync(0xffffffff,ls0,2);
                ls1+=__shfl_xor_sync(0xffffffff,ls1,1); ls1+=__shfl_xor_sync(0xffffffff,ls1,2);
                if (tid==0) { sm.sum_reduce_buf[mwarp*HPB+gid]=ls0; sm.sum_reduce_buf[mwarp*HPB+gid+8]=ls1; }

                {
                    int e0i=qk_nb+tid*2, e1i=qk_nb+tid*2+1;
                    #pragma unroll
                    for (int vc = 0; vc < N_V_CHUNKS; vc++) {
                        float vsc0 = reinterpret_cast<const float*>(kv_smem+e0i*KV_SMEM_STRIDE+D_NOPE)[vc];
                        float vsc1 = reinterpret_cast<const float*>(kv_smem+e1i*KV_SMEM_STRIDE+D_NOPE)[vc];
                        atomicMax(reinterpret_cast<int*>(&g_wsc[vc*HPB+gid]),
                            __float_as_int(fmaxf(fabsf(w0*vsc0),fabsf(w1*vsc1))));
                        atomicMax(reinterpret_cast<int*>(&g_wsc[vc*HPB+gid+8]),
                            __float_as_int(fmaxf(fabsf(w2*vsc0),fabsf(w3*vsc1))));
                    }
                }
                bar_sync(2, MATH_THREADS);

                if (threadIdx.x < HPB) {
                    int h = threadIdx.x; float ts = 0.f;
                    for (int w = 0; w < N_MATH_WARPS; w++) ts += sm.sum_reduce_buf[w*HPB+h];
                    g_l[h] += ts;
                }
                for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += MATH_THREADS)
                    g_wsc[i] = fmaxf(g_wsc[i], 1e-10f) / FP8_MAX;
                bar_sync(2, MATH_THREADS);
            } // end Phase 1 group loop

            // ============================================================
            // PHASE 2: XV — shared V transpose, per-group w_fp8 + MMA
            // ============================================================
            {
                int e0i = qk_nb + tid * 2, e1i = qk_nb + tid * 2 + 1;

                #pragma unroll
                for (int vc = 0; vc < N_V_CHUNKS; vc++) {
                    int v_off = vc * V_CHUNK;

                    // V transpose (shared across all groups)
                    for (int idx = threadIdx.x; idx < V_CHUNK * BI; idx += MATH_THREADS) {
                        int d = idx / BI, e = idx % BI;
                        sm.v_trans[d * V_TRANS_STRIDE + e] =
                            kv_smem[e * KV_SMEM_STRIDE + v_off + d];
                    }

                    #pragma unroll
                    for (int g = 0; g < MG_N_HG; g++) {
                        float* g_wsc = sm.w_head_sc_all + g * N_V_CHUNKS * HPB + vc * HPB;
                        float gw0=weights[g][0], gw1=weights[g][1];
                        float gw2=weights[g][2], gw3=weights[g][3];

                        float si0=1.f/g_wsc[gid], si1=1.f/g_wsc[gid+8];
                        float vsc0 = reinterpret_cast<const float*>(
                            kv_smem+e0i*KV_SMEM_STRIDE+D_NOPE)[vc];
                        float vsc1 = reinterpret_cast<const float*>(
                            kv_smem+e1i*KV_SMEM_STRIDE+D_NOPE)[vc];
                        float ws00=gw0*vsc0, ws01=gw1*vsc1, ws10=gw2*vsc0, ws11=gw3*vsc1;
                        __nv_fp8_e4m3 f00(fmaxf(-FP8_MAX,fminf(FP8_MAX,ws00*si0)));
                        __nv_fp8_e4m3 f01(fmaxf(-FP8_MAX,fminf(FP8_MAX,ws01*si0)));
                        __nv_fp8_e4m3 f10(fmaxf(-FP8_MAX,fminf(FP8_MAX,ws10*si1)));
                        __nv_fp8_e4m3 f11(fmaxf(-FP8_MAX,fminf(FP8_MAX,ws11*si1)));
                        sm.w_fp8[gid*W_FP8_STRIDE+e0i]=f00.__x;
                        sm.w_fp8[gid*W_FP8_STRIDE+e1i]=f01.__x;
                        sm.w_fp8[(gid+8)*W_FP8_STRIDE+e0i]=f10.__x;
                        sm.w_fp8[(gid+8)*W_FP8_STRIDE+e1i]=f11.__x;

                        bar_sync(2, MATH_THREADS); // w_fp8 + v_trans ready

                        #pragma unroll
                        for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
                            int ti=vc*NT_PER_WARP_XV+nt;
                            int nl=mwarp*(NT_PER_WARP_XV*8)+nt*8;
                            float xv[4]={0.f,0.f,0.f,0.f};
                            #pragma unroll
                            for (int kstep=0; kstep<BI/32; kstep++) {
                                int ko=kstep*32; uint32_t a0,a1,a2,a3,b0,b1;
                                ldmatrix_load_A_fp8(a0,a1,a2,a3, sm.w_fp8+ko, W_FP8_STRIDE, lane);
                                ldmatrix_load_B_fp8(b0,b1, sm.v_trans+nl*V_TRANS_STRIDE+ko, V_TRANS_STRIDE, lane);
                                MmaFp8Result r = mma_fp8_m16n8k32(a0,a1,a2,a3,b0,b1,xv[0],xv[1],xv[2],xv[3]);
                                xv[0]=r.d0;xv[1]=r.d1;xv[2]=r.d2;xv[3]=r.d3;
                            }
                            float sc0=g_wsc[gid], sc1=g_wsc[gid+8];
                            acc_o[g][ti][0]+=xv[0]*sc0; acc_o[g][ti][1]+=xv[1]*sc0;
                            acc_o[g][ti][2]+=xv[2]*sc1; acc_o[g][ti][3]+=xv[3]*sc1;
                        }

                        bool is_very_last = (vc == N_V_CHUNKS-1) && (g == MG_N_HG-1);
                        if (!is_very_last)
                            bar_sync(2, MATH_THREADS); // MMA done, protect w_fp8/v_trans
                    } // end group loop
                } // end V chunk loop
            }

            bar_arrive(1, BLOCK_THREADS);
            if (ni + 1 < NI)
                bar_sync(0, BLOCK_THREADS);
        } // end tile loop

        // ---- Finalize ----
        #pragma unroll
        for (int g = 0; g < MG_N_HG; g++) {
            float* g_l = sm.l_smem + g * HPB;
            int g_h_base = h_start + g * HPB;
            #pragma unroll
            for (int t = 0; t < ACC_TILES; t++) {
                int c=t/NT_PER_WARP_XV, lnt=t%NT_PER_WARP_XV;
                int vb=c*V_CHUNK+mwarp*(NT_PER_WARP_XV*8)+lnt*8;
                int h0=g_h_base+gid, h1=g_h_base+gid+8, d0=vb+tid*2, d1=vb+tid*2+1;
                float il0=(g_l[gid]>0.f)?(1.f/g_l[gid]):0.f;
                float il1=(g_l[gid+8]>0.f)?(1.f/g_l[gid+8]):0.f;
                size_t b0=(size_t)s_i*num_heads*D_NOPE+(size_t)h0*D_NOPE;
                size_t b1=(size_t)s_i*num_heads*D_NOPE+(size_t)h1*D_NOPE;
                output[b0+d0]=__float2bfloat16(acc_o[g][t][0]*il0);
                output[b0+d1]=__float2bfloat16(acc_o[g][t][1]*il0);
                output[b1+d0]=__float2bfloat16(acc_o[g][t][2]*il1);
                output[b1+d1]=__float2bfloat16(acc_o[g][t][3]*il1);
            }
        }
    }
}

// ============================================================================
// Launch function — dispatches single-group or multi-group based on num_heads
// ============================================================================
void sparse_mla_prefill_launch(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output, float sm_scale,
    int num_heads, int num_tokens, int topk, int BI_param)
{
    dim3 block(BLOCK_THREADS);
    const int head_groups = num_heads / HPB;

    if (head_groups > 1 && (head_groups % MG_N_HG == 0)) {
        // Multi-group kernel: N_HG=2, processes 32 heads per CTA
        constexpr size_t smem_bytes = MG_SMEM_TOTAL;
        const int REPLICATE_H = num_heads / MG_HEADS_PER_CTA;
        dim3 grid(num_tokens * REPLICATE_H);

        static bool mg_configured = false;
        if (!mg_configured && smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(sparse_mla_prefill_mg_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_bytes);
            mg_configured = true;
        }
        sparse_mla_prefill_mg_kernel<<<grid, block, smem_bytes>>>(
            reinterpret_cast<const bf16*>(Q.data_ptr()),
            reinterpret_cast<const uint8_t*>(KV_cache.data_ptr()),
            indices.data_ptr<int32_t>(),
            reinterpret_cast<bf16*>(output.data_ptr()),
            sm_scale, num_heads, num_tokens, topk);
    } else {
        // Single-group kernel: original path
        constexpr size_t smem_bytes = SMEM_TOTAL;
        const int REPLICATE_H = num_heads / HPB;
        dim3 grid(num_tokens * REPLICATE_H);

        static bool sg_configured = false;
        if (!sg_configured && smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(sparse_mla_prefill_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_bytes);
            sg_configured = true;
        }
        sparse_mla_prefill_kernel<<<grid, block, smem_bytes>>>(
            reinterpret_cast<const bf16*>(Q.data_ptr()),
            reinterpret_cast<const uint8_t*>(KV_cache.data_ptr()),
            indices.data_ptr<int32_t>(),
            reinterpret_cast<bf16*>(output.data_ptr()),
            sm_scale, num_heads, num_tokens, topk);
    }
}
