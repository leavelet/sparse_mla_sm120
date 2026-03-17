#include <torch/extension.h>

// Forward declarations of launch wrappers from .cu files
void sparse_mla_decode_partial_launch(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor partial_O, torch::Tensor partial_LSE,
    float sm_scale, int num_heads, int num_tokens, int topk, int BI);

void sparse_mla_combine_launch(
    torch::Tensor partial_O, torch::Tensor partial_LSE,
    torch::Tensor output,
    int num_heads, int num_tokens, int topk, int BI);

void sparse_mla_prefill_launch(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output,
    float sm_scale, int num_heads, int num_tokens, int topk, int BI);

// Unified forward entry point
void sparse_mla_fwd(
    torch::Tensor Q,          // [num_tokens, num_heads, dim]  bf16
    torch::Tensor KV_cache,   // [pool_size, 1, kv_packed]     uint8
    torch::Tensor indices,    // [num_tokens, topk]            int32
    torch::Tensor output,     // [num_tokens, num_heads, d_v]  bf16
    float sm_scale,
    int d_v,
    int d_rope,
    int topk)
{
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bf16");
    TORCH_CHECK(Q.is_cuda() && KV_cache.is_cuda() && indices.is_cuda(),
                "All tensors must be on CUDA");
    TORCH_CHECK(Q.is_contiguous() && KV_cache.is_contiguous() && indices.is_contiguous(),
                "All tensors must be contiguous");
    TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");

    int num_tokens = Q.size(0);
    int num_heads = Q.size(1);
    int dim = Q.size(2);
    TORCH_CHECK(dim == d_v + d_rope, "dim mismatch");
    TORCH_CHECK(num_heads % 16 == 0, "num_heads must be divisible by 16 (HPB)");
    TORCH_CHECK(indices.size(0) == num_tokens, "indices batch mismatch");
    TORCH_CHECK(indices.size(-1) == topk, "topk mismatch");

    // Choose BI based on available shared memory and num_heads.
    // BI=32 uses ~40 KB smem (fits default 48 KB).
    // BI=64 uses ~61 KB smem (needs 100 KB config on SM120).
    // For decode (small batch), BI=32 with more splits gives better SM utilization.
    int BI = 32;

    int NI = topk / BI;

    // Heuristic: use split-KV (decode path) for small batches, fused for large
    const int DECODE_THRESHOLD = 64;

    if (num_tokens <= DECODE_THRESHOLD) {
        // Decode path: partial + combine
        auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(Q.device());
        auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());

        auto partial_O = torch::empty({num_tokens, NI, num_heads, d_v}, opts_bf16);
        auto partial_LSE = torch::empty({num_tokens, NI, num_heads}, opts_fp32);

        sparse_mla_decode_partial_launch(
            Q, KV_cache, indices, partial_O, partial_LSE,
            sm_scale, num_heads, num_tokens, topk, BI);

        sparse_mla_combine_launch(
            partial_O, partial_LSE, output,
            num_heads, num_tokens, topk, BI);
    } else {
        // Prefill path: fused online softmax
        sparse_mla_prefill_launch(
            Q, KV_cache, indices, output,
            sm_scale, num_heads, num_tokens, topk, BI);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mla_fwd", &sparse_mla_fwd, "Sparse MLA forward (SM120)");
}
