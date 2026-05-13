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

void sparse_mla_splitkv_launch_model1_dual(
    torch::Tensor Q,
    torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor KV_cache_extra, torch::Tensor indices_extra,
    torch::Tensor partial_O, torch::Tensor partial_LSE,
    float sm_scale, int num_heads, int num_tokens,
    int topk, int topk_extra,
    int page_block_size, int stride_kv_row,
    int page_block_size_extra, int stride_kv_row_extra,
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

void sparse_mla_prefill_launch_model1_dual(
    torch::Tensor Q,
    torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor KV_cache_extra, torch::Tensor indices_extra,
    torch::Tensor output, torch::Tensor out_lse,
    float sm_scale, int num_heads, int num_tokens,
    int topk, int topk_extra,
    int page_block_size, int stride_kv_row,
    int page_block_size_extra, int stride_kv_row_extra,
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
    int page_block_size,
    // Dual-cache extras: accepted but currently no-op (API-skin phase).
    // Kernel-side dual-cache scoring lands separately; when nullopt the
    // existing single-cache path is unchanged.
    c10::optional<torch::Tensor> KV_cache_extra = c10::nullopt,
    c10::optional<torch::Tensor> indices_extra = c10::nullopt,
    c10::optional<torch::Tensor> topk_length = c10::nullopt,
    c10::optional<torch::Tensor> topk_length_extra = c10::nullopt,
    c10::optional<torch::Tensor> attn_sink = c10::nullopt)
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

    // attn_sink + topk_length are not yet kernel-wired (entries past
    // topk_length must be -1-padded by the caller). Warn if provided.
    if (topk_length.has_value() || topk_length_extra.has_value()
        || attn_sink.has_value()) {
        TORCH_WARN_ONCE(
            "sparse_mla_splitkv_fwd: topk_length / topk_length_extra / "
            "attn_sink are accepted but not yet plumbed to the kernel. "
            "Pad indices with -1 beyond the valid range; attn_sink is "
            "currently dropped.");
    }

    ModelType mt = infer_model_type(d_qk);
    const cudaStream_t stream = get_current_stream(Q);

    // Dual-cache path: routed when both KV_cache_extra and indices_extra
    // are provided. Only MODEL1 is supported in this slice.
    if (KV_cache_extra.has_value() && indices_extra.has_value()) {
        TORCH_CHECK(mt == ModelType::MODEL1,
            "Dual-cache decode is only implemented for MODEL1 currently; "
            "got d_qk=", d_qk);
        torch::Tensor KV_extra = KV_cache_extra.value();
        torch::Tensor idx_extra = indices_extra.value();
        TORCH_CHECK(KV_extra.is_cuda() && idx_extra.is_cuda(),
            "KV_cache_extra and indices_extra must be CUDA tensors");
        // Derive extra-cache page_block_size / stride. Match the main-cache
        // convention: stride_kv_row from KV_extra.stride(-2) bytes,
        // page_block_size from KV_extra.shape[-3].
        int page_block_size_extra = (int)KV_extra.size(-3);
        int stride_kv_row_extra = (int)(KV_extra.stride(-2) * KV_extra.element_size());
        int topk_extra = (int)idx_extra.size(-1);

        sparse_mla_splitkv_launch_model1_dual(
            Q, KV_cache, indices, KV_extra, idx_extra,
            partial_O, partial_LSE,
            sm_scale, num_heads, num_tokens,
            topk, topk_extra,
            page_block_size, stride_kv_row,
            page_block_size_extra, stride_kv_row_extra,
            stream);
        return;
    }

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
    int page_block_size,
    // Dual-cache extras: accepted but currently no-op (API-skin phase).
    c10::optional<torch::Tensor> KV_cache_extra = c10::nullopt,
    c10::optional<torch::Tensor> indices_extra = c10::nullopt,
    c10::optional<torch::Tensor> topk_length = c10::nullopt,
    c10::optional<torch::Tensor> topk_length_extra = c10::nullopt,
    c10::optional<torch::Tensor> attn_sink = c10::nullopt)
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

    if (topk_length.has_value() || topk_length_extra.has_value()
        || attn_sink.has_value()) {
        TORCH_WARN_ONCE(
            "sparse_mla_prefill_fwd: topk_length / topk_length_extra / "
            "attn_sink are accepted but not yet plumbed to the kernel. "
            "Pad indices with -1 beyond the valid range; attn_sink is "
            "currently dropped.");
    }

    ModelType mt = infer_model_type(d_qk);
    const cudaStream_t stream = get_current_stream(Q);

    // Dual-cache path: routed when both KV_cache_extra and indices_extra
    // are provided. Only MODEL1 is supported in this slice.
    if (KV_cache_extra.has_value() && indices_extra.has_value()) {
        TORCH_CHECK(mt == ModelType::MODEL1,
            "Dual-cache prefill is only implemented for MODEL1 currently; "
            "got d_qk=", d_qk);
        torch::Tensor KV_extra = KV_cache_extra.value();
        torch::Tensor idx_extra = indices_extra.value();
        TORCH_CHECK(KV_extra.is_cuda() && idx_extra.is_cuda(),
            "KV_cache_extra and indices_extra must be CUDA tensors");
        int page_block_size_extra = (int)KV_extra.size(-3);
        int stride_kv_row_extra = (int)(KV_extra.stride(-2) * KV_extra.element_size());
        int topk_extra = (int)idx_extra.size(-1);

        sparse_mla_prefill_launch_model1_dual(
            Q, KV_cache, indices, KV_extra, idx_extra,
            output, out_lse,
            sm_scale, num_heads, num_tokens,
            topk, topk_extra,
            page_block_size, stride_kv_row,
            page_block_size_extra, stride_kv_row_extra,
            stream);
        return;
    }

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
          "Split-KV decode forward (SM120, V32+MODEL1)",
          py::arg("Q"),
          py::arg("KV_cache"),
          py::arg("indices"),
          py::arg("partial_O"),
          py::arg("partial_LSE"),
          py::arg("sm_scale"),
          py::arg("topk"),
          py::arg("tiles_per_split"),
          py::arg("stride_kv_row"),
          py::arg("page_block_size"),
          // Dual-cache extras (API-skin; currently no-op in kernel)
          py::arg("KV_cache_extra") = py::none(),
          py::arg("indices_extra") = py::none(),
          py::arg("topk_length") = py::none(),
          py::arg("topk_length_extra") = py::none(),
          py::arg("attn_sink") = py::none());
    m.def("sparse_mla_combine_fwd", &sparse_mla_combine_fwd,
          "Combine partial outputs from split-KV decode");
    m.def("sparse_mla_prefill_fwd", &sparse_mla_prefill_fwd,
          "Sparse MLA prefill forward (SM120, V32+MODEL1)",
          py::arg("Q"),
          py::arg("KV_cache"),
          py::arg("indices"),
          py::arg("output"),
          py::arg("out_lse"),
          py::arg("sm_scale"),
          py::arg("topk"),
          py::arg("stride_kv_row"),
          py::arg("page_block_size"),
          // Dual-cache extras (API-skin; currently no-op in kernel)
          py::arg("KV_cache_extra") = py::none(),
          py::arg("indices_extra") = py::none(),
          py::arg("topk_length") = py::none(),
          py::arg("topk_length_extra") = py::none(),
          py::arg("attn_sink") = py::none());
}
