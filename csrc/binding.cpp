#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "model/model_type.h"

namespace {

cudaStream_t get_current_stream(const torch::Tensor& tensor) {
    return at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

ModelType infer_model_type(int d_qk) {
    if (d_qk == 576) return ModelType::V32;
    if (d_qk == 512) return ModelType::MODEL1;
    TORCH_CHECK(false, "Unsupported d_qk=", d_qk, "; expected 576 (V32) or 512 (MODEL1)");
}

}  // namespace

// Forward declarations — split-KV decode
void sparse_mla_splitkv_launch_v32(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor partial_O, torch::Tensor partial_LSE,
    float sm_scale, int num_heads, int num_tokens, int topk,
    int tiles_per_split, int page_block_size, int stride_kv_row,
    cudaStream_t stream);

void sparse_mla_splitkv_launch_model1(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor partial_O, torch::Tensor partial_LSE,
    float sm_scale, int num_heads, int num_tokens, int topk,
    int tiles_per_split, int page_block_size, int stride_kv_row,
    cudaStream_t stream);

// Forward declarations — combine
void sparse_mla_combine_launch(
    torch::Tensor partial_O, torch::Tensor partial_LSE,
    torch::Tensor output, torch::Tensor out_lse,
    int nsplits, cudaStream_t stream);

// Forward declarations — prefill
void sparse_mla_prefill_launch_v32(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output, torch::Tensor out_lse,
    float sm_scale, int num_heads, int num_tokens, int topk,
    int page_block_size, int stride_kv_row,
    cudaStream_t stream);

void sparse_mla_prefill_launch_model1(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output, torch::Tensor out_lse,
    float sm_scale, int num_heads, int num_tokens, int topk,
    int page_block_size, int stride_kv_row,
    cudaStream_t stream);

// ── Python-facing functions ─────────────────────────────────────────

static constexpr int HPB = 16;

void sparse_mla_splitkv_fwd(
    torch::Tensor Q,
    torch::Tensor KV_cache,
    torch::Tensor indices,
    torch::Tensor partial_O,
    torch::Tensor partial_LSE,
    float sm_scale,
    int topk,
    int tiles_per_split,
    int stride_kv_row,
    int page_block_size)
{
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bf16");
    TORCH_CHECK(Q.is_cuda() && KV_cache.is_cuda() && indices.is_cuda());
    TORCH_CHECK(partial_O.dtype() == torch::kFloat32, "partial_O must be float32");
    TORCH_CHECK(partial_LSE.dtype() == torch::kFloat32, "partial_LSE must be float32");

    int num_tokens = Q.size(0);
    int num_heads = Q.size(1);
    int d_qk = Q.size(2);
    TORCH_CHECK(num_tokens <= 64, "decode path requires num_tokens <= 64");
    TORCH_CHECK(num_heads % HPB == 0);
    TORCH_CHECK(page_block_size > 0, "page_block_size must be > 0");

    ModelType mt = infer_model_type(d_qk);
    const cudaStream_t stream = get_current_stream(Q);

    switch (mt) {
    case ModelType::V32:
        sparse_mla_splitkv_launch_v32(
            Q, KV_cache, indices, partial_O, partial_LSE,
            sm_scale, num_heads, num_tokens, topk,
            tiles_per_split, page_block_size, stride_kv_row, stream);
        break;
    case ModelType::MODEL1:
        sparse_mla_splitkv_launch_model1(
            Q, KV_cache, indices, partial_O, partial_LSE,
            sm_scale, num_heads, num_tokens, topk,
            tiles_per_split, page_block_size, stride_kv_row, stream);
        break;
    }
}

void sparse_mla_combine_fwd(
    torch::Tensor partial_O,
    torch::Tensor partial_LSE,
    torch::Tensor output,
    torch::Tensor out_lse,
    int nsplits)
{
    TORCH_CHECK(partial_O.is_cuda() && output.is_cuda());
    const cudaStream_t stream = get_current_stream(partial_O);
    sparse_mla_combine_launch(partial_O, partial_LSE, output, out_lse, nsplits, stream);
}

void sparse_mla_prefill_fwd(
    torch::Tensor Q,
    torch::Tensor KV_cache,
    torch::Tensor indices,
    torch::Tensor output,
    torch::Tensor out_lse,
    float sm_scale,
    int topk,
    int stride_kv_row,
    int page_block_size)
{
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bf16");
    TORCH_CHECK(Q.is_cuda() && KV_cache.is_cuda() && indices.is_cuda());
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bf16");
    TORCH_CHECK(out_lse.dtype() == torch::kFloat32, "out_lse must be float32");

    int num_tokens = Q.size(0);
    int num_heads = Q.size(1);
    int d_qk = Q.size(2);
    TORCH_CHECK(num_heads % HPB == 0);
    TORCH_CHECK(page_block_size > 0, "page_block_size must be > 0");

    ModelType mt = infer_model_type(d_qk);
    const cudaStream_t stream = get_current_stream(Q);

    switch (mt) {
    case ModelType::V32:
        sparse_mla_prefill_launch_v32(
            Q, KV_cache, indices, output, out_lse,
            sm_scale, num_heads, num_tokens, topk,
            page_block_size, stride_kv_row, stream);
        break;
    case ModelType::MODEL1:
        sparse_mla_prefill_launch_model1(
            Q, KV_cache, indices, output, out_lse,
            sm_scale, num_heads, num_tokens, topk,
            page_block_size, stride_kv_row, stream);
        break;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mla_splitkv_fwd", &sparse_mla_splitkv_fwd,
          "Split-KV decode forward (SM120, V32+MODEL1)");
    m.def("sparse_mla_combine_fwd", &sparse_mla_combine_fwd,
          "Combine partial outputs from split-KV decode");
    m.def("sparse_mla_prefill_fwd", &sparse_mla_prefill_fwd,
          "Sparse MLA prefill forward (SM120, V32+MODEL1)");
}
