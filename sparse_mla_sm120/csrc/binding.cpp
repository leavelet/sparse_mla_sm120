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

void fp8_mqa_logits_ragged_launch(
    torch::Tensor Q, torch::Tensor KV, torch::Tensor KV_scale,
    torch::Tensor W, torch::Tensor K_start, torch::Tensor K_end,
    torch::Tensor logits, int max_seqlen_k, cudaStream_t stream);

void fp8_mqa_logits_paged_launch(
    torch::Tensor Q, torch::Tensor KV, torch::Tensor KV_scale,
    torch::Tensor W, torch::Tensor ctx_lens, torch::Tensor block_table,
    torch::Tensor logits, int next_n, cudaStream_t stream);

static constexpr int DECODE_HPB = 16;
static constexpr int MQA_HEAD_DIM = 128;

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

void fp8_mqa_logits_ragged_fwd(
    torch::Tensor Q,
    torch::Tensor KV,
    torch::Tensor KV_scale,
    torch::Tensor W,
    torch::Tensor K_start,
    torch::Tensor K_end,
    torch::Tensor logits,
    int max_seqlen_k)
{
    TORCH_CHECK(Q.is_cuda() && KV.is_cuda() && KV_scale.is_cuda() &&
                W.is_cuda() && K_start.is_cuda() && K_end.is_cuda() &&
                logits.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(Q.dim() == 3 && Q.size(2) == MQA_HEAD_DIM,
                "Q must be (seq_q, num_heads, 128)");
    TORCH_CHECK(KV.dim() == 2 && KV.size(1) == MQA_HEAD_DIM,
                "KV must be (seq_kv, 128)");
    TORCH_CHECK(Q.size(1) % 16 == 0, "num_heads must be a multiple of 16");
    TORCH_CHECK(Q.is_contiguous() && KV.is_contiguous(),
                "Q and KV must be contiguous");

    const cudaStream_t stream = get_current_stream(Q);
    fp8_mqa_logits_ragged_launch(Q, KV, KV_scale, W, K_start, K_end,
                                 logits, max_seqlen_k, stream);
}

void fp8_mqa_logits_paged_fwd(
    torch::Tensor Q,
    torch::Tensor KV,
    torch::Tensor KV_scale,
    torch::Tensor W,
    torch::Tensor ctx_lens,
    torch::Tensor block_table,
    torch::Tensor logits,
    int next_n)
{
    TORCH_CHECK(Q.is_cuda() && KV.is_cuda() && KV_scale.is_cuda() &&
                W.is_cuda() && ctx_lens.is_cuda() && block_table.is_cuda() &&
                logits.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(Q.dim() == 3 && Q.size(2) == MQA_HEAD_DIM,
                "Q must be (batch*next_n, num_heads, 128)");
    TORCH_CHECK(KV.dim() == 3 && KV.size(1) == 64 && KV.size(2) == MQA_HEAD_DIM,
                "KV must be (num_blocks, 64, 128)");
    TORCH_CHECK(Q.size(1) % 16 == 0, "num_heads must be a multiple of 16");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");

    const cudaStream_t stream = get_current_stream(Q);
    fp8_mqa_logits_paged_launch(Q, KV, KV_scale, W, ctx_lens, block_table,
                                logits, next_n, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mla_decode_fwd", &sparse_mla_decode_fwd, "Sparse MLA decode forward (SM120)");
    m.def("sparse_mla_prefill_fwd", &sparse_mla_prefill_fwd, "Sparse MLA prefill forward (SM120)");
    m.def("fp8_mqa_logits_ragged_fwd", &fp8_mqa_logits_ragged_fwd,
          "FP8 MQA logits ragged forward (SM120)");
    m.def("fp8_mqa_logits_paged_fwd", &fp8_mqa_logits_paged_fwd,
          "FP8 MQA logits paged forward (SM120)");
}
