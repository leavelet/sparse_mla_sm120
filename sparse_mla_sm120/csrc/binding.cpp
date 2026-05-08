#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

cudaStream_t get_current_stream(const torch::Tensor& tensor) {
    return at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

}  // namespace

void sparse_mla_decode_launch(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor partial_O, torch::Tensor partial_LSE,
    torch::Tensor output, torch::Tensor semaphores,
    float sm_scale, int num_heads, int num_tokens, int topk,
    int tiles_per_split, int nsplits, cudaStream_t stream);

void sparse_mla_prefill_launch(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output,
    float sm_scale, int num_heads, int num_tokens, int topk, int BI,
    cudaStream_t stream);

static constexpr int DECODE_HPB = 16;

void sparse_mla_decode_fwd(
    torch::Tensor Q,
    torch::Tensor KV_cache,
    torch::Tensor indices,
    torch::Tensor output,
    torch::Tensor partial_O,
    torch::Tensor partial_LSE,
    torch::Tensor semaphores,
    float sm_scale,
    int d_v,
    int d_rope,
    int topk,
    int tiles_per_split,
    int nsplits)
{
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bf16");
    TORCH_CHECK(Q.is_cuda() && KV_cache.is_cuda() && indices.is_cuda(),
                "All tensors must be on CUDA");

    int num_tokens = Q.size(0);
    int num_heads = Q.size(1);
    TORCH_CHECK(num_tokens <= 64, "decode path requires num_tokens <= 64");
    TORCH_CHECK(num_heads % DECODE_HPB == 0);

    const cudaStream_t stream = get_current_stream(Q);
    sparse_mla_decode_launch(
        Q, KV_cache, indices,
        partial_O, partial_LSE,
        output, semaphores,
        sm_scale, num_heads, num_tokens, topk,
        tiles_per_split, nsplits, stream);
}

void sparse_mla_prefill_fwd(
    torch::Tensor Q,
    torch::Tensor KV_cache,
    torch::Tensor indices,
    torch::Tensor output,
    float sm_scale,
    int d_v,
    int d_rope,
    int topk)
{
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bf16");
    TORCH_CHECK(Q.is_cuda() && KV_cache.is_cuda() && indices.is_cuda(),
                "All tensors must be on CUDA");

    int num_tokens = Q.size(0);
    int num_heads = Q.size(1);
    int dim = Q.size(2);
    TORCH_CHECK(dim == d_v + d_rope, "dim mismatch");
    TORCH_CHECK(num_heads % DECODE_HPB == 0);

    const cudaStream_t stream = get_current_stream(Q);
    sparse_mla_prefill_launch(
        Q, KV_cache, indices, output,
        sm_scale, num_heads, num_tokens, topk, 64, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mla_decode_fwd", &sparse_mla_decode_fwd, "Sparse MLA decode forward (SM120)");
    m.def("sparse_mla_prefill_fwd", &sparse_mla_prefill_fwd, "Sparse MLA prefill forward (SM120)");
}
