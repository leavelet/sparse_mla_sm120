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
    int nsplits, c10::optional<torch::Tensor> attn_sink,
    cudaStream_t stream);

void sparse_mla_combine_v2_launch(
    torch::Tensor o_accum, torch::Tensor lse_accum,
    torch::Tensor output, torch::Tensor out_lse,
    torch::Tensor num_splits_ptr,
    int batch, int s_q, int num_heads, int max_nsplits,
    c10::optional<torch::Tensor> attn_sink,
    cudaStream_t stream);

// Forward declarations — split-KV decode v2 (scheduler-driven)
void sparse_mla_splitkv_v2_launch_v32(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor o_accum, torch::Tensor lse_accum,
    torch::Tensor output, torch::Tensor out_lse,
    torch::Tensor sched_meta, torch::Tensor num_splits,
    float sm_scale, int num_heads, int num_batches, int s_q, int topk,
    int page_block_size, int stride_kv_row, int num_sm_parts,
    const float* attn_sink,
    const uint8_t* extra_kv, const int32_t* extra_idx,
    const int* topk_length_ptr, int extra_topk, const int* extra_topk_length_ptr,
    int extra_page_block_size, int extra_stride_kv_row,
    cudaStream_t stream);

void sparse_mla_splitkv_v2_launch_model1(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor o_accum, torch::Tensor lse_accum,
    torch::Tensor output, torch::Tensor out_lse,
    torch::Tensor sched_meta, torch::Tensor num_splits,
    float sm_scale, int num_heads, int num_batches, int s_q, int topk,
    int page_block_size, int stride_kv_row, int num_sm_parts,
    const float* attn_sink,
    const uint8_t* extra_kv, const int32_t* extra_idx,
    const int* topk_length_ptr, int extra_topk, const int* extra_topk_length_ptr,
    int extra_page_block_size, int extra_stride_kv_row,
    cudaStream_t stream);

// Forward declarations — scheduler
void get_decode_metadata(
    int b, int topk, int extra_topk,
    int num_sm_parts, int fixed_overhead,
    c10::optional<torch::Tensor> topk_length,
    c10::optional<torch::Tensor> extra_topk_length,
    torch::Tensor sched_meta,
    torch::Tensor num_splits);

// Forward declarations — prefill
void sparse_mla_prefill_launch_v32(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output, torch::Tensor out_lse,
    float sm_scale, int num_heads, int num_tokens, int topk,
    int page_block_size, int stride_kv_row,
    const float* attn_sink_ptr,
    const int* topk_length_ptr,
    cudaStream_t stream);

void sparse_mla_prefill_launch_model1(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output, torch::Tensor out_lse,
    float sm_scale, int num_heads, int num_tokens, int topk,
    int page_block_size, int stride_kv_row,
    const float* attn_sink_ptr,
    const int* topk_length_ptr,
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
    TORCH_CHECK(num_heads > 0 && num_heads <= 128);
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

void sparse_mla_splitkv_v2_fwd(
    torch::Tensor Q,
    torch::Tensor KV_cache,
    torch::Tensor indices,
    torch::Tensor o_accum,
    torch::Tensor lse_accum,
    torch::Tensor output,
    torch::Tensor out_lse,
    torch::Tensor sched_meta,
    torch::Tensor num_splits,
    float sm_scale,
    int topk,
    int stride_kv_row,
    int page_block_size,
    int num_sm_parts,
    c10::optional<torch::Tensor> attn_sink,
    c10::optional<torch::Tensor> extra_k_cache,
    c10::optional<torch::Tensor> extra_indices_t,
    c10::optional<torch::Tensor> topk_length_t,
    int extra_topk,
    c10::optional<torch::Tensor> extra_topk_length_t)
{
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bf16");
    TORCH_CHECK(Q.is_cuda() && KV_cache.is_cuda() && indices.is_cuda());

    int num_batches = Q.size(0);
    int num_heads = Q.size(1);
    int d_qk = Q.size(2);
    int s_q = 1;
    TORCH_CHECK(num_heads > 0 && num_heads <= 128);

    const float* sink_ptr = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
    const uint8_t* extra_kv = extra_k_cache.has_value()
        ? reinterpret_cast<const uint8_t*>(extra_k_cache->data_ptr()) : nullptr;
    const int32_t* extra_idx = extra_indices_t.has_value()
        ? extra_indices_t->data_ptr<int32_t>() : nullptr;
    const int* tl_ptr = topk_length_t.has_value()
        ? topk_length_t->data_ptr<int>() : nullptr;
    const int* etl_ptr = extra_topk_length_t.has_value()
        ? extra_topk_length_t->data_ptr<int>() : nullptr;
    int extra_pbs = extra_k_cache.has_value() ? extra_k_cache->size(-3) : 1;
    int extra_stride = extra_k_cache.has_value()
        ? (int)(extra_k_cache->stride(-2) * extra_k_cache->element_size()) : 0;

    ModelType mt = infer_model_type(d_qk);
    const cudaStream_t stream = get_current_stream(Q);

    switch (mt) {
    case ModelType::V32:
        sparse_mla_splitkv_v2_launch_v32(
            Q, KV_cache, indices, o_accum, lse_accum, output, out_lse,
            sched_meta, num_splits,
            sm_scale, num_heads, num_batches, s_q, topk,
            page_block_size, stride_kv_row, num_sm_parts, sink_ptr,
            extra_kv, extra_idx, tl_ptr, extra_topk, etl_ptr,
            extra_pbs, extra_stride, stream);
        break;
    case ModelType::MODEL1:
        sparse_mla_splitkv_v2_launch_model1(
            Q, KV_cache, indices, o_accum, lse_accum, output, out_lse,
            sched_meta, num_splits,
            sm_scale, num_heads, num_batches, s_q, topk,
            page_block_size, stride_kv_row, num_sm_parts, sink_ptr,
            extra_kv, extra_idx, tl_ptr, extra_topk, etl_ptr,
            extra_pbs, extra_stride, stream);
        break;
    }
}

void sparse_mla_combine_fwd(
    torch::Tensor partial_O,
    torch::Tensor partial_LSE,
    torch::Tensor output,
    torch::Tensor out_lse,
    int nsplits,
    c10::optional<torch::Tensor> attn_sink)
{
    TORCH_CHECK(partial_O.is_cuda() && output.is_cuda());
    if (attn_sink.has_value()) {
        TORCH_CHECK(attn_sink->dtype() == torch::kFloat32, "attn_sink must be float32");
        TORCH_CHECK(attn_sink->is_cuda(), "attn_sink must be on CUDA");
    }
    const cudaStream_t stream = get_current_stream(partial_O);
    sparse_mla_combine_launch(partial_O, partial_LSE, output, out_lse, nsplits, attn_sink, stream);
}

void sparse_mla_combine_v2_fwd(
    torch::Tensor o_accum,
    torch::Tensor lse_accum,
    torch::Tensor output,
    torch::Tensor out_lse,
    torch::Tensor num_splits_ptr,
    int batch,
    int max_nsplits,
    c10::optional<torch::Tensor> attn_sink)
{
    TORCH_CHECK(o_accum.is_cuda() && output.is_cuda());
    int s_q = 1;
    int num_heads = o_accum.size(2);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(o_accum.get_device()).stream();
    sparse_mla_combine_v2_launch(
        o_accum, lse_accum, output, out_lse, num_splits_ptr,
        batch, s_q, num_heads, max_nsplits, attn_sink, stream);
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
    c10::optional<torch::Tensor> attn_sink,
    c10::optional<torch::Tensor> topk_length)
{
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bf16");
    TORCH_CHECK(Q.is_cuda() && KV_cache.is_cuda() && indices.is_cuda());
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bf16");
    TORCH_CHECK(out_lse.dtype() == torch::kFloat32, "out_lse must be float32");
    if (attn_sink.has_value()) {
        TORCH_CHECK(attn_sink->dtype() == torch::kFloat32, "attn_sink must be float32");
        TORCH_CHECK(attn_sink->is_cuda(), "attn_sink must be on CUDA");
    }

    int num_tokens = Q.size(0);
    int num_heads = Q.size(1);
    int d_qk = Q.size(2);
    TORCH_CHECK(num_heads > 0 && num_heads <= 128);
    TORCH_CHECK(page_block_size > 0, "page_block_size must be > 0");

    ModelType mt = infer_model_type(d_qk);
    const cudaStream_t stream = get_current_stream(Q);
    const float* sink_ptr = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
    const int* tl_ptr = topk_length.has_value() ? topk_length->data_ptr<int>() : nullptr;

    switch (mt) {
    case ModelType::V32:
        sparse_mla_prefill_launch_v32(
            Q, KV_cache, indices, output, out_lse,
            sm_scale, num_heads, num_tokens, topk,
            page_block_size, stride_kv_row, sink_ptr, tl_ptr, stream);
        break;
    case ModelType::MODEL1:
        sparse_mla_prefill_launch_model1(
            Q, KV_cache, indices, output, out_lse,
            sm_scale, num_heads, num_tokens, topk,
            page_block_size, stride_kv_row, sink_ptr, tl_ptr, stream);
        break;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mla_splitkv_fwd", &sparse_mla_splitkv_fwd,
          "Split-KV decode forward (SM120, V32+MODEL1)");
    m.def("sparse_mla_splitkv_v2_fwd", &sparse_mla_splitkv_v2_fwd,
          "Split-KV decode v2 forward (scheduler-driven, V4-compatible)",
          py::arg("Q"), py::arg("KV_cache"), py::arg("indices"),
          py::arg("o_accum"), py::arg("lse_accum"),
          py::arg("output"), py::arg("out_lse"),
          py::arg("sched_meta"), py::arg("num_splits"),
          py::arg("sm_scale"), py::arg("topk"),
          py::arg("stride_kv_row"), py::arg("page_block_size"),
          py::arg("num_sm_parts"),
          py::arg("attn_sink") = py::none(),
          py::arg("extra_k_cache") = py::none(),
          py::arg("extra_indices") = py::none(),
          py::arg("topk_length") = py::none(),
          py::arg("extra_topk") = 0,
          py::arg("extra_topk_length") = py::none());
    m.def("sparse_mla_combine_fwd", &sparse_mla_combine_fwd,
          "Combine partial outputs from split-KV decode",
          py::arg("partial_O"), py::arg("partial_LSE"),
          py::arg("output"), py::arg("out_lse"),
          py::arg("nsplits"), py::arg("attn_sink") = py::none());
    m.def("sparse_mla_combine_v2_fwd", &sparse_mla_combine_v2_fwd,
          "Combine v2: per-batch split indexing via num_splits_ptr",
          py::arg("o_accum"), py::arg("lse_accum"),
          py::arg("output"), py::arg("out_lse"),
          py::arg("num_splits_ptr"), py::arg("batch"),
          py::arg("max_nsplits"),
          py::arg("attn_sink") = py::none());
    m.def("get_decode_metadata", &get_decode_metadata,
          "Compute decode scheduler metadata (GPU, 1 warp)",
          py::arg("b"), py::arg("topk"), py::arg("extra_topk"),
          py::arg("num_sm_parts"), py::arg("fixed_overhead"),
          py::arg("topk_length") = py::none(),
          py::arg("extra_topk_length") = py::none(),
          py::arg("sched_meta"), py::arg("num_splits"));
    m.def("sparse_mla_prefill_fwd", &sparse_mla_prefill_fwd,
          "Sparse MLA prefill forward (SM120, V32+MODEL1)",
          py::arg("Q"), py::arg("KV_cache"), py::arg("indices"),
          py::arg("output"), py::arg("out_lse"),
          py::arg("sm_scale"), py::arg("topk"),
          py::arg("stride_kv_row"), py::arg("page_block_size"),
          py::arg("attn_sink") = py::none(),
          py::arg("topk_length") = py::none());
}
