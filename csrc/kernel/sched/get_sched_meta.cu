#include "sched_params.h"
#include "../../arch/common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// GPU scheduler kernel — single warp, computes per-partition block assignments.
// Replicates FlashMLA's get_mla_metadata_kernel exactly.

struct SchedParams {
    int b;
    int topk;
    int extra_topk;
    int block_size_n;
    int fixed_overhead;
    int num_sm_parts;
    const int* topk_length;        // [b], may be nullptr
    const int* extra_topk_length;  // [b], may be nullptr
    DecodingSchedMeta* sched_meta; // [num_sm_parts]
    int* num_splits;               // [b+1]
};

static __device__ int ceil_to(int x, int m) { return ((x + m - 1) / m) * m; }

__global__ void get_sched_meta_kernel(__grid_constant__ const SchedParams p) {
    extern __shared__ int smem[];
    const int b = p.b;
    int* num_blocks_sh   = smem;
    int* num_splits_sh   = smem + b;
    int* first_block_sh  = smem + 2 * b + 1;
    int* last_block_sh   = smem + 3 * b + 1;

    int total_num_blocks = 0;

    // Phase 1: compute per-batch block counts (parallel)
    for (int i = threadIdx.x; i < b; i += 32) {
        int cur_s_k = p.topk_length ? __ldg(p.topk_length + i) : p.topk;
        if (cur_s_k == 0) cur_s_k = 1;
        if (p.extra_topk > 0) {
            cur_s_k = ceil_to(cur_s_k, p.block_size_n);
            cur_s_k += p.extra_topk_length ? __ldg(p.extra_topk_length + i) : p.extra_topk;
        }

        int first_block = 0;
        int last_block = (cur_s_k > 0) ? ((cur_s_k - 1) / p.block_size_n) : 0;
        int nb = last_block - first_block + 1;

        num_blocks_sh[i] = nb;
        first_block_sh[i] = first_block;
        last_block_sh[i] = last_block;
        total_num_blocks += nb + p.fixed_overhead;
    }

    // Warp-wide sum
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1)
        total_num_blocks += __shfl_xor_sync(0xffffffff, total_num_blocks, offset);
    __syncwarp();

    // Phase 2: greedy scheduling (thread 0)
    if (threadIdx.x == 0) {
        int payload = (total_num_blocks + p.num_sm_parts - 1) / p.num_sm_parts
                    + p.fixed_overhead;

        int now_req = 0, now_block = 0, now_n_split = 0, cum_splits = 0;
        num_splits_sh[0] = 0;

        for (int part = 0; part < p.num_sm_parts; part++) {
            DecodingSchedMeta meta;
            meta.begin_req_idx = now_req;
            meta.begin_block_idx = (now_req < b)
                ? (now_block + first_block_sh[now_req]) : 0;
            meta.begin_split_idx = now_n_split;
            meta.is_first_req_splitted = (now_block != 0) ? 1 : 0;

            int remain = payload;
            while (now_req < b) {
                int nb = num_blocks_sh[now_req];
                int remain_blocks = nb - now_block;
                if (remain >= remain_blocks + p.fixed_overhead) {
                    cum_splits += now_n_split + 1;
                    num_splits_sh[now_req + 1] = cum_splits;
                    remain -= remain_blocks + p.fixed_overhead;
                    now_req++;
                    now_block = 0;
                    now_n_split = 0;
                } else {
                    if (remain > p.fixed_overhead) {
                        now_block += remain - p.fixed_overhead;
                        now_n_split++;
                    }
                    remain = 0;
                    break;
                }
            }

            meta.end_req_idx = (now_block > 0) ? now_req : (now_req - 1);
            if (now_block > 0) {
                meta.end_block_idx = now_block + first_block_sh[now_req];
            } else if (now_req > 0) {
                meta.end_block_idx = last_block_sh[now_req - 1] + 1;
            } else {
                meta.end_block_idx = 0;
            }

            meta.is_last_req_splitted =
                (meta.end_req_idx < b &&
                 meta.end_block_idx != last_block_sh[meta.end_req_idx] + 1) ? 1 : 0;

            if (meta.begin_req_idx == meta.end_req_idx) {
                int flag = meta.is_first_req_splitted | meta.is_last_req_splitted;
                meta.is_first_req_splitted = flag;
                meta.is_last_req_splitted = flag;
            }

            meta._pad = 0;
            p.sched_meta[part] = meta;
        }
    }
    __syncwarp();

    // Phase 3: write num_splits (parallel)
    for (int i = threadIdx.x; i <= b; i += 32)
        p.num_splits[i] = num_splits_sh[i];
}

// ── Launch wrapper ──────────────────────────────────────────────────

void launch_get_sched_meta(
    int b, int topk, int extra_topk,
    int block_size_n, int fixed_overhead, int num_sm_parts,
    const int* topk_length, const int* extra_topk_length,
    DecodingSchedMeta* sched_meta, int* num_splits,
    cudaStream_t stream)
{
    SchedParams params{b, topk, extra_topk, block_size_n, fixed_overhead,
                       num_sm_parts, topk_length, extra_topk_length,
                       sched_meta, num_splits};

    size_t smem = sizeof(int) * (4 * b + 1);
    cudaLaunchConfig_t config{dim3(1), dim3(32), smem, stream, nullptr, 0};
    void* args[] = {(void*)&params};
    CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)get_sched_meta_kernel, args));
}

// ── Python entry point ──────────────────────────────────────────────

void get_decode_metadata(
    int b, int topk, int extra_topk,
    int num_sm_parts, int fixed_overhead,
    c10::optional<torch::Tensor> topk_length,
    c10::optional<torch::Tensor> extra_topk_length,
    torch::Tensor sched_meta,
    torch::Tensor num_splits)
{
    constexpr int BI = 64;
    const int* tl = topk_length.has_value() ? topk_length->data_ptr<int>() : nullptr;
    const int* etl = extra_topk_length.has_value() ? extra_topk_length->data_ptr<int>() : nullptr;
    auto* meta = reinterpret_cast<DecodingSchedMeta*>(sched_meta.data_ptr());
    int* ns = num_splits.data_ptr<int>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(sched_meta.get_device()).stream();

    launch_get_sched_meta(b, topk, extra_topk, BI, fixed_overhead, num_sm_parts,
                          tl, etl, meta, ns, stream);
}
