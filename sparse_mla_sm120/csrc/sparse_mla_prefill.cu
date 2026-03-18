#include "common.cuh"
#include "mma_sm120.cuh"

#include <torch/extension.h>

// ============================================================================
// Sparse MLA Prefill — FP8 MMA QK+XV, in-register online softmax
//
// Grid: [num_tokens * ceil(num_heads / HPB)]
// Block: 256 threads (8 warps), HPB=16, BI=64
//
// QK → scores in registers → softmax in registers (cross-warp max via smem)
// → FP8 weights written directly to w_fp8 smem → XV MMA
// No scores_smem needed.
// ============================================================================

static constexpr int HPB = 16;
static constexpr int BI = 64;
static constexpr int NWARPS = 8;
static constexpr int THREADS = NWARPS * 32;
static constexpr int V_CHUNK = 128;
static constexpr int N_V_CHUNKS = D_NOPE / V_CHUNK;
static constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / NWARPS;  // 2
static constexpr int ACC_TILES = N_V_CHUNKS * NT_PER_WARP_XV; // 8

__device__ __forceinline__ void load_A_fp8(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    const uint8_t* smem, int stride, int k_off, int lane)
{
    int g = lane >> 2, t = lane & 3;
    const uint8_t* r0 = smem + g * stride + k_off;
    const uint8_t* r1 = smem + (g + 8) * stride + k_off;
    a0 = *reinterpret_cast<const uint32_t*>(r0 + t * 4);
    a1 = *reinterpret_cast<const uint32_t*>(r1 + t * 4);
    a2 = *reinterpret_cast<const uint32_t*>(r0 + 16 + t * 4);
    a3 = *reinterpret_cast<const uint32_t*>(r1 + 16 + t * 4);
}

__device__ __forceinline__ void load_B_fp8(
    uint32_t& b0, uint32_t& b1,
    const uint8_t* smem, int stride, int n_base, int k_off, int lane)
{
    int g = lane >> 2, t = lane & 3;
    const uint8_t* r = smem + (n_base + g) * stride + k_off;
    b0 = *reinterpret_cast<const uint32_t*>(r + t * 4);
    b1 = *reinterpret_cast<const uint32_t*>(r + 16 + t * 4);
}

__device__ __forceinline__ void load_A_bf16(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    const bf16* smem, int stride, int k_off, int lane)
{
    int g = lane >> 2, t = lane & 3;
    const bf16* r0 = smem + g * stride + k_off;
    const bf16* r1 = smem + (g + 8) * stride + k_off;
    a0 = *reinterpret_cast<const uint32_t*>(r0 + t * 2);
    a1 = *reinterpret_cast<const uint32_t*>(r1 + t * 2);
    a2 = *reinterpret_cast<const uint32_t*>(r0 + 8 + t * 2);
    a3 = *reinterpret_cast<const uint32_t*>(r1 + 8 + t * 2);
}

__global__ void __launch_bounds__(THREADS, 1)
sparse_mla_prefill_mma_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    bf16* __restrict__ output,
    float sm_scale,
    int num_heads,
    int num_tokens,
    int topk)
{
    const int NI = topk / BI;
    const int REPLICATE_H = num_heads / HPB;
    const int s_i = blockIdx.x / REPLICATE_H;
    const int h_tile = blockIdx.x % REPLICATE_H;
    const int h_start = h_tile * HPB;
    if (s_i >= num_tokens) return;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int gid = lane >> 2;
    const int tid = lane & 3;
    const float sm_scale_log2e = sm_scale * LOG2E;

    extern __shared__ char smem_raw[];
    char* sp = smem_raw;

    uint8_t* q_nope_fp8  = (uint8_t*)sp;  sp += HPB * D_NOPE;           // 8192
    float*   q_nope_sc   = (float*)sp;    sp += HPB * NUM_SCALES * 4;   // 256
    bf16*    q_rope_smem = (bf16*)sp;     sp += HPB * D_ROPE * 2;      // 2048
    uint8_t* kv_smem     = (uint8_t*)sp;  sp += BI * KV_PACKED_BYTES;   // 41984
    // Cross-warp reduction: [NWARPS][HPB] for local max/sum
    float*   reduce_buf  = (float*)sp;    sp += NWARPS * HPB * 4;       // 512
    float*   m_smem      = (float*)sp;    sp += HPB * 4;                // 64
    float*   l_smem      = (float*)sp;    sp += HPB * 4;                // 64
    // XV buffers
    uint8_t* w_fp8       = (uint8_t*)sp;  sp += HPB * BI;               // 1024
    float*   w_head_sc   = (float*)sp;    sp += HPB * 4;                // 64
    uint8_t* v_trans     = (uint8_t*)sp;  sp += V_CHUNK * BI;           // 8192
    // Total: ~62 KB (saved 4 KB vs scores_smem version)

    float acc_o[ACC_TILES][4];
    #pragma unroll
    for (int t = 0; t < ACC_TILES; t++)
        acc_o[t][0] = acc_o[t][1] = acc_o[t][2] = acc_o[t][3] = 0.f;

    // ---- Q quantization (same as before) ----
    const bf16* q_base = Q + (size_t)s_i * num_heads * DIM + (size_t)h_start * DIM;

    for (int i = threadIdx.x; i < HPB * D_ROPE; i += THREADS)
        q_rope_smem[(i / D_ROPE) * D_ROPE + i % D_ROPE] =
            q_base[(i / D_ROPE) * DIM + D_NOPE + i % D_ROPE];

    // Use reduce_buf as temp for absmax (512B >> 64 floats needed)
    float* amax = reduce_buf;
    for (int i = threadIdx.x; i < HPB * NUM_SCALES; i += THREADS) amax[i] = 0.f;
    __syncthreads();

    for (int idx = threadIdx.x; idx < HPB * D_NOPE; idx += THREADS) {
        int h = idx / D_NOPE, blk = (idx % D_NOPE) / QUANT_TILE;
        float v = fabsf(__bfloat162float(q_base[h * DIM + idx % D_NOPE]));
        atomicMax(reinterpret_cast<int*>(&amax[h * NUM_SCALES + blk]), __float_as_int(v));
    }
    __syncthreads();

    for (int i = threadIdx.x; i < HPB * NUM_SCALES; i += THREADS)
        q_nope_sc[i] = fmaxf(amax[i], 1e-4f) / FP8_MAX;
    __syncthreads();

    for (int idx = threadIdx.x; idx < HPB * D_NOPE; idx += THREADS) {
        int h = idx / D_NOPE, d = idx % D_NOPE, blk = d / QUANT_TILE;
        float si = 1.f / q_nope_sc[h * NUM_SCALES + blk];
        float v = fmaxf(FP8_MIN, fminf(FP8_MAX, __bfloat162float(q_base[h * DIM + d]) * si));
        __nv_fp8_e4m3 fp8v(v);
        q_nope_fp8[h * D_NOPE + d] = fp8v.__x;
    }

    for (int h = threadIdx.x; h < HPB; h += THREADS) {
        m_smem[h] = -1e30f;
        l_smem[h] = 0.f;
    }
    __syncthreads();

    // ================================================================
    // Main loop
    // ================================================================
    for (int ni = 0; ni < NI; ni++) {
        const int32_t* ib = indices + (size_t)s_i * topk + ni * BI;

        // ---- Gather KV ----
        {
            constexpr int TOT = BI * KV_PACKED_BYTES;
            for (int flat = threadIdx.x * 16; flat < TOT; flat += THREADS * 16) {
                int bi = flat / KV_PACKED_BYTES, bo = flat % KV_PACKED_BYTES;
                int idx = ib[bi]; idx = (idx >= 0) ? idx : 0;
                if (bo + 16 <= KV_PACKED_BYTES)
                    *reinterpret_cast<uint4*>(kv_smem + flat) =
                        *reinterpret_cast<const uint4*>(KV_cache + (size_t)idx * KV_PACKED_BYTES + bo);
            }
        }
        __syncthreads();

        // ---- QK MMA (each warp: 8 entries) ----
        int qk_nb = warp_id * 8;
        float qk[4] = {0.f, 0.f, 0.f, 0.f};

        #pragma unroll
        for (int blk = 0; blk < NUM_SCALES; blk++) {
            float ab[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int ks = 0; ks < QUANT_TILE / 32; ks++) {
                int ko = blk * QUANT_TILE + ks * 32;
                uint32_t a0, a1, a2, a3, b0, b1;
                load_A_fp8(a0, a1, a2, a3, q_nope_fp8, D_NOPE, ko, lane);
                load_B_fp8(b0, b1, kv_smem, KV_PACKED_BYTES, qk_nb, ko, lane);
                MmaFp8Result r = mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1,
                                                   ab[0], ab[1], ab[2], ab[3]);
                ab[0] = r.d0; ab[1] = r.d1; ab[2] = r.d2; ab[3] = r.d3;
            }
            float qs0 = q_nope_sc[gid * NUM_SCALES + blk];
            float qs1 = q_nope_sc[(gid + 8) * NUM_SCALES + blk];
            int e0 = qk_nb + tid * 2, e1 = qk_nb + tid * 2 + 1;
            float k0 = reinterpret_cast<const float*>(kv_smem + e0 * KV_PACKED_BYTES + D_NOPE)[blk];
            float k1 = reinterpret_cast<const float*>(kv_smem + e1 * KV_PACKED_BYTES + D_NOPE)[blk];
            qk[0] += ab[0] * qs0 * k0;  qk[1] += ab[1] * qs0 * k1;
            qk[2] += ab[2] * qs1 * k0;  qk[3] += ab[3] * qs1 * k1;
        }

        // ---- QK MMA rope ----
        {
            float ra[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int ks = 0; ks < D_ROPE / 16; ks++) {
                int ko = ks * 16;
                uint32_t a0, a1, a2, a3, b0, b1;
                load_A_bf16(a0, a1, a2, a3, q_rope_smem, D_ROPE, ko, lane);
                int ne = qk_nb + gid;
                const bf16* erp = reinterpret_cast<const bf16*>(
                    kv_smem + ne * KV_PACKED_BYTES + D_NOPE + NUM_SCALES * sizeof(float));
                b0 = *reinterpret_cast<const uint32_t*>(erp + ko + tid * 2);
                b1 = *reinterpret_cast<const uint32_t*>(erp + ko + 8 + tid * 2);
                MmaBf16Result r = mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1,
                                                     ra[0], ra[1], ra[2], ra[3]);
                ra[0] = r.d0; ra[1] = r.d1; ra[2] = r.d2; ra[3] = r.d3;
            }
            qk[0] += ra[0]; qk[1] += ra[1]; qk[2] += ra[2]; qk[3] += ra[3];
        }

        // Mask invalid
        {
            int e0 = qk_nb + tid * 2, e1 = qk_nb + tid * 2 + 1;
            if (ib[e0] < 0) { qk[0] = -1e30f; qk[2] = -1e30f; }
            if (ib[e1] < 0) { qk[1] = -1e30f; qk[3] = -1e30f; }
        }

        // ---- In-register online softmax ----
        // Step 1: scale scores
        float s[4] = {qk[0] * sm_scale_log2e, qk[1] * sm_scale_log2e,
                      qk[2] * sm_scale_log2e, qk[3] * sm_scale_log2e};

        // Step 2: per-warp local max (within quad → 8 entries per head)
        float lm0 = fmaxf(s[0], s[1]);  // head gid
        float lm1 = fmaxf(s[2], s[3]);  // head gid+8
        lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 1));
        lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 2));
        lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 1));
        lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 2));

        // Step 3: cross-warp max reduction
        if (tid == 0) {
            reduce_buf[warp_id * HPB + gid]     = lm0;
            reduce_buf[warp_id * HPB + gid + 8] = lm1;
        }
        __syncthreads();

        // Step 4: thread h computes global max, alpha. Updates m_smem, l_smem *= alpha.
        // Broadcasts alpha at reduce_buf[h], new_max at reduce_buf[HPB+h].
        if (threadIdx.x < HPB) {
            int h = threadIdx.x;
            float old_m = m_smem[h];
            float tile_max = -1e30f;
            for (int w = 0; w < NWARPS; w++)
                tile_max = fmaxf(tile_max, reduce_buf[w * HPB + h]);
            float new_m = fmaxf(old_m, tile_max);
            float alpha = exp2f(old_m - new_m);
            m_smem[h] = new_m;
            l_smem[h] *= alpha;               // rescale running sum BEFORE overwrite
            reduce_buf[h] = alpha;
            reduce_buf[HPB + h] = new_m;
        }
        __syncthreads();

        // Step 5: all threads read alpha and new_max, rescale acc_o, compute weights
        float alpha0 = reduce_buf[gid], alpha1 = reduce_buf[gid + 8];
        #pragma unroll
        for (int t = 0; t < ACC_TILES; t++) {
            acc_o[t][0] *= alpha0; acc_o[t][1] *= alpha0;
            acc_o[t][2] *= alpha1; acc_o[t][3] *= alpha1;
        }

        float nm0 = reduce_buf[HPB + gid], nm1 = reduce_buf[HPB + gid + 8];
        float w0 = exp2f(s[0] - nm0), w1 = exp2f(s[1] - nm0);
        float w2 = exp2f(s[2] - nm1), w3 = exp2f(s[3] - nm1);
        __syncthreads();  // ensure all threads have read alpha/new_max before overwriting reduce_buf

        // Step 6: per-warp sum → cross-warp sum → update l_smem
        float lsum0 = w0 + w1, lsum1 = w2 + w3;
        lsum0 += __shfl_xor_sync(0xffffffff, lsum0, 1);
        lsum0 += __shfl_xor_sync(0xffffffff, lsum0, 2);
        lsum1 += __shfl_xor_sync(0xffffffff, lsum1, 1);
        lsum1 += __shfl_xor_sync(0xffffffff, lsum1, 2);

        if (tid == 0) {
            reduce_buf[warp_id * HPB + gid]     = lsum0;
            reduce_buf[warp_id * HPB + gid + 8] = lsum1;
        }
        __syncthreads();

        if (threadIdx.x < HPB) {
            int h = threadIdx.x;
            float total_sum = 0.f;
            for (int w = 0; w < NWARPS; w++)
                total_sum += reduce_buf[w * HPB + h];
            l_smem[h] += total_sum;
        }
        __syncthreads();

        // ---- XV MMA per V chunk ----
        // Each thread holds w0..w3 (float weights for its 2 entries × 2 heads)
        #pragma unroll
        for (int vc = 0; vc < N_V_CHUNKS; vc++) {
            int v_off = vc * V_CHUNK;

            // Quantize w_scaled = weight * V_scale to FP8
            // Per-head max via atomicMax
            for (int h = threadIdx.x; h < HPB; h += THREADS)
                w_head_sc[h] = 0.f;
            __syncthreads();

            // Each thread has weights for 2 entries × 2 heads
            // Read V_scale for those entries
            int e0_idx = qk_nb + tid * 2, e1_idx = qk_nb + tid * 2 + 1;
            float vsc0 = reinterpret_cast<const float*>(kv_smem + e0_idx * KV_PACKED_BYTES + D_NOPE)[vc];
            float vsc1 = reinterpret_cast<const float*>(kv_smem + e1_idx * KV_PACKED_BYTES + D_NOPE)[vc];

            float ws00 = w0 * vsc0, ws01 = w1 * vsc1;  // head gid
            float ws10 = w2 * vsc0, ws11 = w3 * vsc1;  // head gid+8

            // atomicMax for per-head scale
            atomicMax(reinterpret_cast<int*>(&w_head_sc[gid]),     __float_as_int(fmaxf(fabsf(ws00), fabsf(ws01))));
            atomicMax(reinterpret_cast<int*>(&w_head_sc[gid + 8]), __float_as_int(fmaxf(fabsf(ws10), fabsf(ws11))));
            __syncthreads();

            for (int h = threadIdx.x; h < HPB; h += THREADS)
                w_head_sc[h] = fmaxf(w_head_sc[h], 1e-10f) / FP8_MAX;
            __syncthreads();

            // Quantize and write 4 FP8 values per thread
            {
                float si0 = 1.f / w_head_sc[gid], si1 = 1.f / w_head_sc[gid + 8];
                __nv_fp8_e4m3 f00(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws00 * si0)));
                __nv_fp8_e4m3 f01(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws01 * si0)));
                __nv_fp8_e4m3 f10(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws10 * si1)));
                __nv_fp8_e4m3 f11(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws11 * si1)));
                w_fp8[gid       * BI + e0_idx] = f00.__x;
                w_fp8[gid       * BI + e1_idx] = f01.__x;
                w_fp8[(gid + 8) * BI + e0_idx] = f10.__x;
                w_fp8[(gid + 8) * BI + e1_idx] = f11.__x;
            }

            // Transpose V chunk
            for (int idx = threadIdx.x; idx < V_CHUNK * BI; idx += THREADS) {
                int d = idx / BI, e = idx % BI;
                v_trans[d * BI + e] = kv_smem[e * KV_PACKED_BYTES + v_off + d];
            }
            __syncthreads();

            // XV FP8 MMA
            #pragma unroll
            for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
                int tile_idx = vc * NT_PER_WARP_XV + nt;
                int n_local = warp_id * (NT_PER_WARP_XV * 8) + nt * 8;

                float xv[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int kstep = 0; kstep < BI / 32; kstep++) {
                    int ko = kstep * 32;
                    uint32_t a0, a1, a2, a3, b0, b1;
                    load_A_fp8(a0, a1, a2, a3, w_fp8, BI, ko, lane);
                    load_B_fp8(b0, b1, v_trans, BI, n_local, ko, lane);
                    MmaFp8Result r = mma_fp8_m16n8k32(
                        a0, a1, a2, a3, b0, b1, xv[0], xv[1], xv[2], xv[3]);
                    xv[0] = r.d0; xv[1] = r.d1; xv[2] = r.d2; xv[3] = r.d3;
                }
                float sc0 = w_head_sc[gid], sc1 = w_head_sc[gid + 8];
                acc_o[tile_idx][0] += xv[0] * sc0;
                acc_o[tile_idx][1] += xv[1] * sc0;
                acc_o[tile_idx][2] += xv[2] * sc1;
                acc_o[tile_idx][3] += xv[3] * sc1;
            }
            __syncthreads();
        }
    }

    // ---- Finalize ----
    #pragma unroll
    for (int t = 0; t < ACC_TILES; t++) {
        int c = t / NT_PER_WARP_XV, lnt = t % NT_PER_WARP_XV;
        int v_base = c * V_CHUNK + warp_id * (NT_PER_WARP_XV * 8) + lnt * 8;
        int h0 = h_start + gid, h1 = h_start + gid + 8;
        int d0 = v_base + tid * 2, d1 = v_base + tid * 2 + 1;
        float inv_l0 = (l_smem[gid] > 0.f) ? (1.f / l_smem[gid]) : 0.f;
        float inv_l1 = (l_smem[gid + 8] > 0.f) ? (1.f / l_smem[gid + 8]) : 0.f;
        size_t base0 = (size_t)s_i * num_heads * D_NOPE + (size_t)h0 * D_NOPE;
        size_t base1 = (size_t)s_i * num_heads * D_NOPE + (size_t)h1 * D_NOPE;
        output[base0 + d0] = __float2bfloat16(acc_o[t][0] * inv_l0);
        output[base0 + d1] = __float2bfloat16(acc_o[t][1] * inv_l0);
        output[base1 + d0] = __float2bfloat16(acc_o[t][2] * inv_l1);
        output[base1 + d1] = __float2bfloat16(acc_o[t][3] * inv_l1);
    }
}

void sparse_mla_prefill_launch(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output, float sm_scale,
    int num_heads, int num_tokens, int topk, int BI_param)
{
    const int REPLICATE_H = num_heads / HPB;
    size_t smem_bytes = HPB * D_NOPE
                      + HPB * NUM_SCALES * 4
                      + HPB * D_ROPE * 2
                      + BI * KV_PACKED_BYTES
                      + NWARPS * HPB * 4    // reduce_buf
                      + HPB * 4 + HPB * 4   // m, l
                      + HPB * BI            // w_fp8
                      + HPB * 4             // w_head_sc
                      + V_CHUNK * BI;       // v_trans

    dim3 grid(num_tokens * REPLICATE_H);
    dim3 block(THREADS);

    if (smem_bytes > 48 * 1024)
        cudaFuncSetAttribute(sparse_mla_prefill_mma_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    sparse_mla_prefill_mma_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const bf16*>(Q.data_ptr()),
        reinterpret_cast<const uint8_t*>(KV_cache.data_ptr()),
        indices.data_ptr<int32_t>(),
        reinterpret_cast<bf16*>(output.data_ptr()),
        sm_scale, num_heads, num_tokens, topk);
}
