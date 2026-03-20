/*
 * FP8 MQA Logits Kernel for SM120 (Blackwell)
 *
 * logits[i, j] = sum_h( max( sum_d(q[i,h,d] * kv[j,d]) * kv_scale[j], 0 ) * w[i,h] )
 *
 * B128 XOR swizzle, compact stride=128, outer-product k-loop.
 * Matches Triton's approach: BLOCK_H=64, fully unrolled, aggressive register use.
 */

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "mma_sm120.cuh"

constexpr int HEAD_DIM   = 128;
constexpr int NUM_WARPS  = 4;
constexpr int BLOCK_SIZE = NUM_WARPS * 32;
constexpr int K_ITERS    = HEAD_DIM / 32;
constexpr int CHUNKS_ROW = HEAD_DIM / 16;

template <int NUM_HEADS, int BLOCK_KV, int BLOCK_H>
__device__ __forceinline__ void mqa_mma_core(
    const uint8_t* __restrict__ q_base,
    const float*   __restrict__ w_base,
    uint8_t* kv_s, float* sc_s, uint8_t* q_s,
    float* a0, float* a1, int wkb, int lane)
{
    static_assert(NUM_HEADS % BLOCK_H == 0 && BLOCK_H % 16 == 0);
    constexpr int NUM_H_TILES  = NUM_HEADS / BLOCK_H;
    constexpr int M_TILES      = BLOCK_H / 16;
    constexpr int NT           = BLOCK_KV / 8 / NUM_WARPS;
    constexpr int Q_LOAD_ELEMS = BLOCK_H * CHUNKS_ROW;

    const int tid = threadIdx.x;

    #pragma unroll 1
    for (int ht = 0; ht < NUM_H_TILES; ht++) {
        const int h = ht * BLOCK_H;

        #pragma unroll
        for (int i = tid; i < Q_LOAD_ELEMS; i += BLOCK_SIZE) {
            const int r = i / CHUNKS_ROW, c = i % CHUNKS_ROW;
            const int cb = c * 16;
            int4 v = __ldg(reinterpret_cast<const int4*>(
                q_base + (h + r) * HEAD_DIM + cb));
            *reinterpret_cast<int4*>(
                &q_s[r * HEAD_DIM + swizzle_b128_col(r, cb)]) = v;
        }
        __syncthreads();

        float d[M_TILES][NT][4] = {};

        #pragma unroll
        for (int k = 0; k < K_ITERS; k++) {
            uint32_t ra[M_TILES][4];
            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++)
                ldmatrix_load_A_fp8_swz(ra[mt][0], ra[mt][1], ra[mt][2], ra[mt][3],
                    q_s, mt, k, lane);

            uint32_t rb[NT][2];
            #pragma unroll
            for (int n = 0; n < NT; n++)
                ldmatrix_load_B_fp8_swz(rb[n][0], rb[n][1],
                    kv_s, wkb + n * 8, k, lane);

            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++) {
                #pragma unroll
                for (int n = 0; n < NT; n++) {
                    auto r = mma_fp8_m16n8k32(
                        ra[mt][0], ra[mt][1], ra[mt][2], ra[mt][3],
                        rb[n][0], rb[n][1],
                        d[mt][n][0], d[mt][n][1], d[mt][n][2], d[mt][n][3]);
                    d[mt][n][0] = r.d0; d[mt][n][1] = r.d1;
                    d[mt][n][2] = r.d2; d[mt][n][3] = r.d3;
                }
            }
        }

        #pragma unroll
        for (int mt = 0; mt < M_TILES; mt++) {
            const float w0 = __ldg(&w_base[h + mt * 16 + (lane >> 2)]);
            const float w1 = __ldg(&w_base[h + mt * 16 + (lane >> 2) + 8]);
            #pragma unroll
            for (int n = 0; n < NT; n++) {
                const int lc = (lane & 3) * 2;
                const float s0 = sc_s[wkb + n * 8 + lc];
                const float s1 = sc_s[wkb + n * 8 + lc + 1];
                a0[n] += fmaxf(d[mt][n][0] * s0, 0.f) * w0 + fmaxf(d[mt][n][2] * s0, 0.f) * w1;
                a1[n] += fmaxf(d[mt][n][1] * s1, 0.f) * w0 + fmaxf(d[mt][n][3] * s1, 0.f) * w1;
            }
        }
        __syncthreads();
    }
}

template <int NT>
__device__ __forceinline__ void mqa_reduce_store(
    float* a0, float* a1, float* dst, int kvb, int wkb, int lane,
    int lo, int hi)
{
    #pragma unroll
    for (int n = 0; n < NT; n++) {
        a0[n] += __shfl_xor_sync(0xffffffff, a0[n], 4);
        a0[n] += __shfl_xor_sync(0xffffffff, a0[n], 8);
        a0[n] += __shfl_xor_sync(0xffffffff, a0[n], 16);
        a1[n] += __shfl_xor_sync(0xffffffff, a1[n], 4);
        a1[n] += __shfl_xor_sync(0xffffffff, a1[n], 8);
        a1[n] += __shfl_xor_sync(0xffffffff, a1[n], 16);
    }
    const int g = lane >> 2, c = lane & 3;
    if (g < NT) {
        const int p = kvb + wkb + g * 8 + c * 2;
        if (p >= lo && p + 1 < hi)
            *reinterpret_cast<float2*>(&dst[p]) = make_float2(a0[g], a1[g]);
        else {
            if (p >= lo && p < hi)       dst[p]     = a0[g];
            if (p + 1 >= lo && p + 1 < hi) dst[p + 1] = a1[g];
        }
    }
}

// ── Ragged kernel ──
template <int NUM_HEADS, int BLOCK_KV, int BLOCK_H>
__global__ void fp8_mqa_logits_ragged_kernel(
    const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KV,
    const float* __restrict__ KV_scale, const float* __restrict__ W,
    const int* __restrict__ K_start, const int* __restrict__ K_end,
    float* __restrict__ out, int seq_kv, int stride_out)
{
    constexpr int NT = BLOCK_KV / 8 / NUM_WARPS;
    constexpr int KV_TOTAL = BLOCK_KV * CHUNKS_ROW;
    const int qi = blockIdx.x, kvti = blockIdx.y, kvb = kvti * BLOCK_KV;
    const int ks = K_start[qi], ke = K_end[qi];
    if (kvb >= ke || kvb + BLOCK_KV <= ks) return;

    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    extern __shared__ uint8_t smem[];
    uint8_t* kv_s = smem;
    float*   sc_s = reinterpret_cast<float*>(kv_s + BLOCK_KV * HEAD_DIM);
    uint8_t* q_s  = reinterpret_cast<uint8_t*>(sc_s + BLOCK_KV);

    #pragma unroll 2
    for (int i = tid; i < KV_TOTAL; i += BLOCK_SIZE) {
        const int r = i / CHUNKS_ROW, c = i % CHUNKS_ROW, cb = c * 16, g = kvb + r;
        int4 v = {0,0,0,0};
        if (g < seq_kv) v = __ldg(reinterpret_cast<const int4*>(KV + g * HEAD_DIM + cb));
        *reinterpret_cast<int4*>(&kv_s[r * HEAD_DIM + swizzle_b128_col(r, cb)]) = v;
    }
    for (int i = tid; i < BLOCK_KV; i += BLOCK_SIZE)
        sc_s[i] = (kvb + i < seq_kv) ? __ldg(&KV_scale[kvb + i]) : 0.f;
    __syncthreads();

    const int wkb = wid * NT * 8;
    float a0[NT] = {}, a1[NT] = {};
    mqa_mma_core<NUM_HEADS, BLOCK_KV, BLOCK_H>(
        Q + qi * NUM_HEADS * HEAD_DIM, W + qi * NUM_HEADS,
        kv_s, sc_s, q_s, a0, a1, wkb, lane);
    mqa_reduce_store<NT>(a0, a1, out + qi * stride_out, kvb, wkb, lane, ks, ke);
}

// ── Paged kernel ──
template <int NUM_HEADS, int NEXT_N, int BLOCK_H>
__global__ void fp8_mqa_logits_paged_kernel(
    const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KV,
    const float* __restrict__ KV_scale, const float* __restrict__ W,
    const int* __restrict__ ctx_lens, const int* __restrict__ block_tbl,
    float* __restrict__ out, int stride_bt, int stride_out,
    int stride_kv_block, int stride_kvs_block)
{
    constexpr int BKV = 64, NT = BKV / 8 / NUM_WARPS, KV_TOTAL = BKV * CHUNKS_ROW;
    const int qi = blockIdx.x, pi = blockIdx.y, bi = qi / NEXT_N;
    const int ctx_len = ctx_lens[bi], kvb = pi * BKV;
    if (kvb >= ctx_len) return;
    const int phys = block_tbl[bi * stride_bt + pi];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;

    extern __shared__ uint8_t smem[];
    uint8_t* kv_s = smem;
    float*   sc_s = reinterpret_cast<float*>(kv_s + BKV * HEAD_DIM);
    uint8_t* q_s  = reinterpret_cast<uint8_t*>(sc_s + BKV);

    #pragma unroll 2
    for (int i = tid; i < KV_TOTAL; i += BLOCK_SIZE) {
        const int r = i / CHUNKS_ROW, c = i % CHUNKS_ROW, cb = c * 16;
        int4 v = __ldg(reinterpret_cast<const int4*>(KV + phys * stride_kv_block + r * HEAD_DIM + cb));
        *reinterpret_cast<int4*>(&kv_s[r * HEAD_DIM + swizzle_b128_col(r, cb)]) = v;
    }
    for (int i = tid; i < BKV; i += BLOCK_SIZE)
        sc_s[i] = __ldg(&KV_scale[phys * stride_kvs_block + i]);
    __syncthreads();

    const int wkb = wid * NT * 8;
    float a0[NT] = {}, a1[NT] = {};
    mqa_mma_core<NUM_HEADS, BKV, BLOCK_H>(
        Q + qi * NUM_HEADS * HEAD_DIM, W + qi * NUM_HEADS,
        kv_s, sc_s, q_s, a0, a1, wkb, lane);
    mqa_reduce_store<NT>(a0, a1, out + qi * stride_out, kvb, wkb, lane, 0, ctx_len);
}

// ── Config ──
template <int NH> struct BestBlockH;
template <> struct BestBlockH<128> { static constexpr int value = 64; };
template <> struct BestBlockH<16>  { static constexpr int value = 16; };

template <int BKV, int BH>
static constexpr int smem_for() { return BKV * HEAD_DIM + BKV * 4 + BH * HEAD_DIM; }

// ── Dispatch ──
template <int NH>
static void dispatch_ragged(
    const uint8_t* Q, const uint8_t* KV, const float* KVs, const float* W,
    const int* Ks, const int* Ke, float* out,
    int sq, int skv, int so, int msk, cudaStream_t stream)
{
    constexpr int BKV = 128, BH = BestBlockH<NH>::value;
    const int ow = (msk > 0) ? msk : skv;
    fp8_mqa_logits_ragged_kernel<NH, BKV, BH>
        <<<dim3(sq, (ow + BKV - 1) / BKV), BLOCK_SIZE, smem_for<BKV, BH>(), stream>>>(
            Q, KV, KVs, W, Ks, Ke, out, skv, so);
}

template <int NH, int NN>
static void dispatch_paged(
    const uint8_t* Q, const uint8_t* KV, const float* KVs, const float* W,
    const int* cl, const int* bt, float* out,
    int tq, int mbl, int sbt, int so, int skb, int sksb,
    cudaStream_t stream)
{
    constexpr int BKV = 64, BH = BestBlockH<NH>::value;
    fp8_mqa_logits_paged_kernel<NH, NN, BH>
        <<<dim3(tq, mbl), BLOCK_SIZE, smem_for<BKV, BH>(), stream>>>(
            Q, KV, KVs, W, cl, bt, out, sbt, so, skb, sksb);
}

// ── Launch wrappers ──
void fp8_mqa_logits_ragged_launch(
    torch::Tensor Q, torch::Tensor KV, torch::Tensor KV_scale,
    torch::Tensor W, torch::Tensor K_start, torch::Tensor K_end,
    torch::Tensor logits, int max_seqlen_k, cudaStream_t stream)
{
    int sq = Q.size(0), nh = Q.size(1), skv = KV.size(0), so = logits.stride(0);
    TORCH_CHECK(nh == 16 || nh == 128); TORCH_CHECK(Q.size(2) == HEAD_DIM);
    auto q = static_cast<const uint8_t*>(Q.data_ptr());
    auto kv = static_cast<const uint8_t*>(KV.data_ptr());
    if (nh == 128)
        dispatch_ragged<128>(q, kv, KV_scale.data_ptr<float>(), W.data_ptr<float>(),
            K_start.data_ptr<int>(), K_end.data_ptr<int>(), logits.data_ptr<float>(), sq, skv, so, max_seqlen_k, stream);
    else
        dispatch_ragged<16>(q, kv, KV_scale.data_ptr<float>(), W.data_ptr<float>(),
            K_start.data_ptr<int>(), K_end.data_ptr<int>(), logits.data_ptr<float>(), sq, skv, so, max_seqlen_k, stream);
}

void fp8_mqa_logits_paged_launch(
    torch::Tensor Q, torch::Tensor KV, torch::Tensor KV_scale,
    torch::Tensor W, torch::Tensor ctx_lens, torch::Tensor block_table,
    torch::Tensor logits, int next_n, cudaStream_t stream)
{
    int tq = Q.size(0), nh = Q.size(1), so = logits.stride(0);
    int sbt = block_table.stride(0), mbl = block_table.size(1);
    int skb = KV.stride(0), sksb = KV_scale.stride(0);
    TORCH_CHECK(nh == 16 || nh == 128); TORCH_CHECK(next_n == 1);
    auto q = static_cast<const uint8_t*>(Q.data_ptr());
    auto kv = static_cast<const uint8_t*>(KV.data_ptr());
    #define LP(NH) dispatch_paged<NH,1>(q,kv,KV_scale.data_ptr<float>(),W.data_ptr<float>(),\
        ctx_lens.data_ptr<int>(),block_table.data_ptr<int>(),logits.data_ptr<float>(),tq,mbl,sbt,so,skb,sksb,stream)
    if (nh == 128) { LP(128); } else { LP(16); }
    #undef LP
}
