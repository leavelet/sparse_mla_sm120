#include "../../arch/common.cuh"
#include "../../model/kv_cache_traits.cuh"
#include <torch/extension.h>

// ============================================================================
// Sparse MLA Combine Kernel — FlashMLA-style vectorized implementation
//
// Merges partial outputs from split-KV decode into final output.
// 8 warps × 32 lanes = 256 threads. 1 warp per head.
// float4 vectorized loads with split-level prefetch.
//
// Input:
//   partial_O:   [num_tokens, nsplits, num_heads, D_V] float32
//   partial_LSE: [num_tokens, nsplits, num_heads]      float32
//
// Output:
//   output: [num_tokens, num_heads, D_V] bfloat16
//   lse:    [num_tokens, num_heads]      float32
//
// Grid: (num_tokens, 1, ceil(num_heads / BLOCK_H))
// ============================================================================

static constexpr int COMBINE_BLOCK_H = 8;
static constexpr int COMBINE_THREADS = COMBINE_BLOCK_H * 32;  // 256
static constexpr int COMBINE_ELEMS_PER_THREAD = D_V / (32 * 4);  // 512/(32*4) = 4

struct CombineParams {
    const float* partial_O;
    const float* partial_LSE;
    bf16* output;
    float* out_lse;
    int num_heads;
    int nsplits;
    // Per-head attention-sink logit, shape [num_heads], float32. nullptr =
    // no sink (output unchanged). FlashMLA convention: out *= sigmoid(lse - sink)
    // AND merge sink into returned LSE: lse' = log(exp(lse) + exp(sink)).
    // Padded heads carry -inf which yields factor == 1.
    const float* attn_sink;
};

template <int MAX_SPLITS>
__global__ void __launch_bounds__(COMBINE_THREADS)
sparse_mla_combine_kernel(__grid_constant__ const CombineParams params)
{
    cudaGridDependencySynchronize();

    const float* __restrict__ partial_O = params.partial_O;
    const float* __restrict__ partial_LSE = params.partial_LSE;
    bf16* __restrict__ output = params.output;
    float* __restrict__ out_lse = params.out_lse;
    const float* __restrict__ attn_sink = params.attn_sink;  // [num_heads] or nullptr
    const int num_heads = params.num_heads;
    const int nsplits = params.nsplits;

    const int token_idx = blockIdx.x;
    const int h_block = blockIdx.z;
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;

    const int h = h_block * COMBINE_BLOCK_H + warp_idx;
    if (h >= num_heads) return;

    if (nsplits == 1) {
        // Single split: just convert float32 → bf16, no combine needed.
        // attn_sink scaling: output[h] *= sigmoid(lse_h - sink_h). lse is
        // already in log2 space (the decode kernel writes `m + log2(l)`),
        // and sink is in raw-log space, so compare via LOG2E. -inf sink
        // (padded heads) yields factor == 1 — no-op.
        const float* src = partial_O
            + (size_t)token_idx * nsplits * num_heads * D_V
            + (size_t)h * D_V;
        bf16* dst = output
            + (size_t)token_idx * num_heads * D_V
            + (size_t)h * D_V;

        float sink_factor = 1.0f;
        if (attn_sink != nullptr) {
            float lse_h = partial_LSE[(size_t)token_idx * nsplits * num_heads + h];
            float sink_log2 = attn_sink[h] * LOG2E;
            sink_factor = 1.0f / (1.0f + exp2f(sink_log2 - lse_h));
        }

        #pragma unroll
        for (int i = 0; i < COMBINE_ELEMS_PER_THREAD; ++i) {
            float4 v = *(const float4*)(src + lane_idx * 4 + i * 128);
            bf16 b[4];
            b[0] = __float2bfloat16(v.x * sink_factor);
            b[1] = __float2bfloat16(v.y * sink_factor);
            b[2] = __float2bfloat16(v.z * sink_factor);
            b[3] = __float2bfloat16(v.w * sink_factor);
            *(uint64_t*)(dst + lane_idx * 4 + i * 128) = *(const uint64_t*)b;
        }

        if (lane_idx == 0) {
            size_t lse_idx = (size_t)token_idx * nsplits * num_heads + h;
            size_t lse_out_idx = (size_t)token_idx * num_heads + h;
            float lse_h = partial_LSE[lse_idx];
            // FlashMLA V4 convention: merge sink into LSE.
            if (attn_sink != nullptr) {
                float sink_log2 = attn_sink[h] * LOG2E;
                if (lse_h != -1e30f)
                    lse_h += log2f(1.f + exp2f(sink_log2 - lse_h));
                else
                    lse_h = sink_log2;
            }
            out_lse[lse_out_idx] = lse_h;
        }
        return;
    }

    // Stride for partial_O: [num_tokens, nsplits, num_heads, D_V]
    const size_t split_stride = (size_t)num_heads * D_V;  // stride across splits (in floats)
    const float* oaccum_ptr = partial_O
        + (size_t)token_idx * nsplits * num_heads * D_V
        + (size_t)h * D_V;

    // LSE stride: [num_tokens, nsplits, num_heads]
    const size_t lse_split_stride = (size_t)num_heads;
    const float* lse_ptr = partial_LSE
        + (size_t)token_idx * nsplits * num_heads
        + h;

    // ── LSE reduction via warp shuffle ──────────────────────────────
    __shared__ float smem_buf[COMBINE_BLOCK_H][MAX_SPLITS];

    constexpr int NUM_LSE_PER_THREAD = (MAX_SPLITS + 31) / 32;
    float local_lse[NUM_LSE_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
        int sp = i * 32 + lane_idx;
        local_lse[i] = (sp < nsplits) ? lse_ptr[sp * lse_split_stride] : -1e30f;
    }

    // Warp-wide max
    float max_lse = -1e30f;
    #pragma unroll
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
        max_lse = fmaxf(max_lse, local_lse[i]);
    max_lse = warp_reduce_max(max_lse);
    if (max_lse == -1e30f) max_lse = 0.f;

    // Warp-wide sum of exp2(lse - max)
    float sum_lse = 0.f;
    #pragma unroll
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
        sum_lse += exp2f(local_lse[i] - max_lse);
    sum_lse = warp_reduce_sum(sum_lse);

    // Global LSE and per-split scale factors
    float global_lse = (sum_lse > 0.f) ? (log2f(sum_lse) + max_lse) : -1e30f;

    // Merge attn_sink LSE (MODEL1 V4): logsumexp(global_lse, attn_sink)
    if (params.attn_sink != nullptr) {
        float sink_log2 = __ldg(params.attn_sink + h) * LOG2E;
        if (global_lse != -1e30f)
            global_lse += log2f(1.f + exp2f(sink_log2 - global_lse));
        else
            global_lse = sink_log2;
    }

    if (lane_idx == 0) {
        size_t lse_out_idx = (size_t)token_idx * num_heads + h;
        out_lse[lse_out_idx] = global_lse;
    }

    // Write per-split scale factors to smem
    #pragma unroll
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
        int sp = i * 32 + lane_idx;
        if (sp < MAX_SPLITS)
            smem_buf[warp_idx][sp] = exp2f(local_lse[i] - global_lse);
    }
    __syncwarp();

    // ── Accumulation with prefetch ──────────────────────────────────
    // Prefetch split 0 data
    float4 datas[COMBINE_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < COMBINE_ELEMS_PER_THREAD; ++i)
        datas[i] = *(const float4*)(oaccum_ptr + lane_idx * 4 + i * 128);

    float4 result[COMBINE_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < COMBINE_ELEMS_PER_THREAD; ++i)
        result[i] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll 1
    for (int sp = 0; sp < nsplits; ++sp) {
        float lse_scale = smem_buf[warp_idx][sp];
        #pragma unroll
        for (int i = 0; i < COMBINE_ELEMS_PER_THREAD; ++i) {
            result[i].x += lse_scale * datas[i].x;
            result[i].y += lse_scale * datas[i].y;
            result[i].z += lse_scale * datas[i].z;
            result[i].w += lse_scale * datas[i].w;
            // Prefetch next split
            if (sp != nsplits - 1) {
                datas[i] = *(const float4*)(oaccum_ptr + (size_t)(sp + 1) * split_stride + lane_idx * 4 + i * 128);
            }
        }
    }

    // ── Write output (bf16, packed uint64_t) ────────────────────────
    // attn_sink output-scaling is implicit here: the per-split scale factors
    // (smem_buf[sp] = exp2(raw_lse_i - global_lse_merged)) already divide
    // the accumulated sum by (sum_exp(raw_lse) + exp(sink)), which is the
    // sigmoid(global_lse_raw - sink) factor we want. Do NOT apply a second
    // explicit sink scaling — that would double-count.
    bf16* o_ptr = output
        + (size_t)token_idx * num_heads * D_V
        + (size_t)h * D_V;

    #pragma unroll
    for (int i = 0; i < COMBINE_ELEMS_PER_THREAD; ++i) {
        bf16 b[4];
        b[0] = __float2bfloat16(result[i].x);
        b[1] = __float2bfloat16(result[i].y);
        b[2] = __float2bfloat16(result[i].z);
        b[3] = __float2bfloat16(result[i].w);
        *(uint64_t*)(o_ptr + lane_idx * 4 + i * 128) = *(const uint64_t*)b;
    }
}

// ============================================================================
// V2 Combine — per-batch split indexing via num_splits_ptr
//
// Input:
//   o_accum:   [total_splits, s_q, num_heads, D_V] float32
//   lse_accum: [total_splits, s_q, num_heads]      float32
//
// Output:
//   output: [batch * s_q, num_heads, D_V] bfloat16
//   lse:    [batch * s_q, num_heads]      float32
//
// Grid: (batch * s_q, 1, ceil(num_heads / BLOCK_H))
// ============================================================================

struct CombineV2Params {
    const float* o_accum;
    const float* lse_accum;
    bf16* output;
    float* out_lse;
    const int* num_splits_ptr;   // [batch + 1] prefix sum
    int num_heads;
    int s_q;
    size_t stride_oa_split;      // s_q * num_heads * D_V
    size_t stride_la_split;      // s_q * num_heads
    const float* attn_sink;
};

template <int MAX_SPLITS>
__global__ void __launch_bounds__(COMBINE_THREADS)
sparse_mla_combine_v2_kernel(__grid_constant__ const CombineV2Params params)
{
    cudaGridDependencySynchronize();

    const int batch_sq_idx = blockIdx.x;
    const int batch_idx = batch_sq_idx / params.s_q;
    const int s_q_idx = batch_sq_idx % params.s_q;
    const int h_block = blockIdx.z;
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;
    const int h = h_block * COMBINE_BLOCK_H + warp_idx;
    if (h >= params.num_heads) return;

    const int start_split = __ldg(params.num_splits_ptr + batch_idx);
    const int end_split = __ldg(params.num_splits_ptr + batch_idx + 1);
    const int my_nsplits = end_split - start_split;

    if (my_nsplits <= 1) return;

    const float* __restrict__ oaccum_ptr = params.o_accum
        + (size_t)start_split * params.stride_oa_split
        + (size_t)s_q_idx * params.num_heads * D_V
        + (size_t)h * D_V;
    const size_t oa_split_stride = params.stride_oa_split;

    const float* __restrict__ lse_ptr = params.lse_accum
        + (size_t)start_split * params.stride_la_split
        + (size_t)s_q_idx * params.num_heads
        + h;
    const size_t la_split_stride = params.stride_la_split;

    // LSE reduction (identical algorithm to v1)
    __shared__ float smem_buf[COMBINE_BLOCK_H][MAX_SPLITS];

    constexpr int NUM_LSE_PER_THREAD = (MAX_SPLITS + 31) / 32;
    float local_lse[NUM_LSE_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
        int sp = i * 32 + lane_idx;
        local_lse[i] = (sp < my_nsplits) ? lse_ptr[sp * la_split_stride] : -1e30f;
    }

    float max_lse = -1e30f;
    #pragma unroll
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
        max_lse = fmaxf(max_lse, local_lse[i]);
    max_lse = warp_reduce_max(max_lse);
    if (max_lse == -1e30f) max_lse = 0.f;

    float sum_lse = 0.f;
    #pragma unroll
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
        sum_lse += exp2f(local_lse[i] - max_lse);
    sum_lse = warp_reduce_sum(sum_lse);

    float global_lse = (sum_lse > 0.f) ? (log2f(sum_lse) + max_lse) : -1e30f;

    if (params.attn_sink != nullptr) {
        float sink_log2 = __ldg(params.attn_sink + h) * LOG2E;
        if (global_lse != -1e30f)
            global_lse += log2f(1.f + exp2f(sink_log2 - global_lse));
        else
            global_lse = sink_log2;
    }

    if (lane_idx == 0) {
        size_t lse_out_idx = (size_t)batch_sq_idx * params.num_heads + h;
        params.out_lse[lse_out_idx] = global_lse;
    }

    #pragma unroll
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
        int sp = i * 32 + lane_idx;
        if (sp < MAX_SPLITS)
            smem_buf[warp_idx][sp] = exp2f(local_lse[i] - global_lse);
    }
    __syncwarp();

    // Accumulation with prefetch (identical algorithm to v1)
    float4 datas[COMBINE_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < COMBINE_ELEMS_PER_THREAD; ++i)
        datas[i] = *(const float4*)(oaccum_ptr + lane_idx * 4 + i * 128);

    float4 result[COMBINE_ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < COMBINE_ELEMS_PER_THREAD; ++i)
        result[i] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll 1
    for (int sp = 0; sp < my_nsplits; ++sp) {
        float lse_scale = smem_buf[warp_idx][sp];
        #pragma unroll
        for (int i = 0; i < COMBINE_ELEMS_PER_THREAD; ++i) {
            result[i].x += lse_scale * datas[i].x;
            result[i].y += lse_scale * datas[i].y;
            result[i].z += lse_scale * datas[i].z;
            result[i].w += lse_scale * datas[i].w;
            if (sp != my_nsplits - 1) {
                datas[i] = *(const float4*)(oaccum_ptr + (size_t)(sp + 1) * oa_split_stride + lane_idx * 4 + i * 128);
            }
        }
    }

    bf16* o_ptr = params.output
        + (size_t)batch_sq_idx * params.num_heads * D_V
        + (size_t)h * D_V;

    #pragma unroll
    for (int i = 0; i < COMBINE_ELEMS_PER_THREAD; ++i) {
        bf16 b[4];
        b[0] = __float2bfloat16(result[i].x);
        b[1] = __float2bfloat16(result[i].y);
        b[2] = __float2bfloat16(result[i].z);
        b[3] = __float2bfloat16(result[i].w);
        *(uint64_t*)(o_ptr + lane_idx * 4 + i * 128) = *(const uint64_t*)b;
    }
}

// ── MAX_SPLITS dispatch macro ───────────────────────────────────────
#define COMBINE_SPLITS_SWITCH(NSPLITS, NAME, ...)       \
    [&] {                                               \
        if ((NSPLITS) <= 32) {                          \
            constexpr int NAME = 32;                    \
            return __VA_ARGS__();                        \
        } else if ((NSPLITS) <= 64) {                   \
            constexpr int NAME = 64;                    \
            return __VA_ARGS__();                        \
        } else if ((NSPLITS) <= 128) {                  \
            constexpr int NAME = 128;                   \
            return __VA_ARGS__();                        \
        } else {                                        \
            TORCH_CHECK(false, "nsplits=", (NSPLITS),   \
                        " exceeds MAX_SPLITS=128");     \
        }                                               \
    }()

// ── Launch wrapper ──────────────────────────────────────────────────
void sparse_mla_combine_launch(
    torch::Tensor partial_O,
    torch::Tensor partial_LSE,
    torch::Tensor output,
    torch::Tensor out_lse,
    int nsplits,
    c10::optional<torch::Tensor> attn_sink,
    cudaStream_t stream)
{
    int num_tokens = partial_O.size(0);
    int num_heads = partial_O.size(2);

    COMBINE_SPLITS_SWITCH(nsplits, MAX_SPLITS, [&] {
        dim3 grid(num_tokens, 1, (num_heads + COMBINE_BLOCK_H - 1) / COMBINE_BLOCK_H);
        dim3 block(COMBINE_THREADS);
        size_t smem_bytes = COMBINE_BLOCK_H * MAX_SPLITS * sizeof(float);

        auto kernel = sparse_mla_combine_kernel<MAX_SPLITS>;
        if (smem_bytes > 48 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        }

        CombineParams params{
            partial_O.data_ptr<float>(),
            partial_LSE.data_ptr<float>(),
            reinterpret_cast<bf16*>(output.data_ptr()),
            out_lse.data_ptr<float>(),
            num_heads, nsplits,
            attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr
        };

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t config{grid, block, smem_bytes, stream, attrs, 1};
        void* args[] = { (void*)&params };
        CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
    });
}

// ── V2 launch wrapper ───────────────────────────────────────────────
void sparse_mla_combine_v2_launch(
    torch::Tensor o_accum,
    torch::Tensor lse_accum,
    torch::Tensor output,
    torch::Tensor out_lse,
    torch::Tensor num_splits_ptr,
    int batch, int s_q, int num_heads,
    int max_nsplits,
    c10::optional<torch::Tensor> attn_sink,
    cudaStream_t stream)
{
    size_t stride_oa_split = (size_t)s_q * num_heads * D_V;
    size_t stride_la_split = (size_t)s_q * num_heads;

    COMBINE_SPLITS_SWITCH(max_nsplits, MAX_SPLITS, [&] {
        dim3 grid(batch * s_q, 1, (num_heads + COMBINE_BLOCK_H - 1) / COMBINE_BLOCK_H);
        dim3 block(COMBINE_THREADS);
        size_t smem_bytes = COMBINE_BLOCK_H * MAX_SPLITS * sizeof(float);

        auto kernel = sparse_mla_combine_v2_kernel<MAX_SPLITS>;
        if (smem_bytes > 48 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        }

        CombineV2Params params{
            o_accum.data_ptr<float>(),
            lse_accum.data_ptr<float>(),
            reinterpret_cast<bf16*>(output.data_ptr()),
            out_lse.data_ptr<float>(),
            num_splits_ptr.data_ptr<int>(),
            num_heads, s_q,
            stride_oa_split, stride_la_split,
            attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr
        };

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t config{grid, block, smem_bytes, stream, attrs, 1};
        void* args[] = { (void*)&params };
        CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
    });
}
