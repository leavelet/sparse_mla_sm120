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

// Returns the per-token byte stride to pass to the kernel so that
// `page_block_size * stride == block byte stride`. Mirrors the Python
// `_effective_stride_kv_row` helper: when callers (e.g., vLLM) pad the
// *block* stride for alignment, the natural per-token stride times
// page_block_size doesn't equal the actual block-to-block stride, so we
// encode the padding into the per-row override.
int effective_stride_kv_row(const torch::Tensor& kv) {
    const int natural_row_bytes = (int)(kv.stride(-2) * kv.element_size());
    const int block_stride_bytes = (int)(kv.stride(0) * kv.element_size());
    const int page_block_size = (int)kv.size(-3);
    if (block_stride_bytes == page_block_size * natural_row_bytes) {
        return natural_row_bytes;
    }
    TORCH_CHECK(block_stride_bytes % page_block_size == 0,
        "kv_cache block stride ", block_stride_bytes,
        " not divisible by page_block_size ", page_block_size,
        "; cannot encode padding via stride_kv_row override");
    return block_stride_bytes / page_block_size;
}

}  // namespace

// Forward declarations — split-KV decode (v1)
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

// Forward declarations — combine (v1 + v2)
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

void sparse_mla_prefill_launch_model1_dual(
    torch::Tensor Q,
    torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor KV_cache_extra, torch::Tensor indices_extra,
    torch::Tensor output, torch::Tensor out_lse,
    float sm_scale, int num_heads, int num_tokens,
    int topk, int topk_extra,
    int page_block_size, int stride_kv_row,
    int page_block_size_extra, int stride_kv_row_extra,
    const float* attn_sink_ptr,
    const int* topk_length_ptr,
    const int* topk_length_extra_ptr,
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
    // Dual-cache extras: when both KV_cache_extra and indices_extra are
    // provided, routes to the dual-cache MODEL1 launcher. v1 decode does
    // not consume topk_length (use sparse_mla_splitkv_v2_fwd for that).
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
    TORCH_CHECK(num_heads > 0 && num_heads <= 128);
    TORCH_CHECK(page_block_size > 0, "page_block_size must be > 0");

    // v1 decode does not wire topk_length into the kernel. Callers that
    // need length-aware masking should use sparse_mla_splitkv_v2_fwd.
    if (topk_length.has_value() || topk_length_extra.has_value()) {
        TORCH_WARN_ONCE(
            "sparse_mla_splitkv_fwd (v1): topk_length / topk_length_extra "
            "are accepted but not wired into v1 decode. Pad indices with "
            "-1 beyond the valid range, or use sparse_mla_splitkv_v2_fwd.");
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
        int page_block_size_extra = (int)KV_extra.size(-3);
        int stride_kv_row_extra = effective_stride_kv_row(KV_extra);
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
        ? effective_stride_kv_row(extra_k_cache.value()) : 0;

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
    c10::optional<torch::Tensor> attn_sink = c10::nullopt)
{
    TORCH_CHECK(partial_O.is_cuda() && output.is_cuda());
    if (attn_sink.has_value()) {
        const torch::Tensor& s = attn_sink.value();
        TORCH_CHECK(s.is_cuda(), "attn_sink must be a CUDA tensor");
        TORCH_CHECK(s.dtype() == torch::kFloat32, "attn_sink must be float32");
        TORCH_CHECK(s.dim() == 1, "attn_sink must be 1-D (per-head)");
        TORCH_CHECK(s.is_contiguous(), "attn_sink must be contiguous");
    }
    const cudaStream_t stream = get_current_stream(partial_O);
    sparse_mla_combine_launch(partial_O, partial_LSE, output, out_lse, nsplits,
                              attn_sink, stream);
}

void sparse_mla_combine_v2_fwd(
    torch::Tensor o_accum,
    torch::Tensor lse_accum,
    torch::Tensor output,
    torch::Tensor out_lse,
    torch::Tensor num_splits_ptr,
    int batch,
    int max_nsplits,
    c10::optional<torch::Tensor> attn_sink = c10::nullopt)
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
    // Dual-cache + V4 extras. When both KV_cache_extra and indices_extra
    // are provided, routes to the dual-cache MODEL1 prefill launcher.
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
    TORCH_CHECK(num_heads > 0 && num_heads <= 128);
    TORCH_CHECK(page_block_size > 0, "page_block_size must be > 0");

    // Validate + extract attn_sink (per-head, [num_heads], float32).
    const float* attn_sink_ptr = nullptr;
    if (attn_sink.has_value()) {
        const torch::Tensor& s = attn_sink.value();
        TORCH_CHECK(s.is_cuda(), "attn_sink must be a CUDA tensor");
        TORCH_CHECK(s.dtype() == torch::kFloat32, "attn_sink must be float32");
        TORCH_CHECK(s.dim() == 1 && (int)s.size(0) == num_heads,
            "attn_sink must be shape [num_heads]");
        TORCH_CHECK(s.is_contiguous(), "attn_sink must be contiguous");
        attn_sink_ptr = s.data_ptr<float>();
    }
    const int* tl_ptr = topk_length.has_value() ? topk_length->data_ptr<int>() : nullptr;
    const int* etl_ptr = topk_length_extra.has_value()
        ? topk_length_extra->data_ptr<int>() : nullptr;

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
        int stride_kv_row_extra = effective_stride_kv_row(KV_extra);
        int topk_extra = (int)idx_extra.size(-1);

        sparse_mla_prefill_launch_model1_dual(
            Q, KV_cache, indices, KV_extra, idx_extra,
            output, out_lse,
            sm_scale, num_heads, num_tokens,
            topk, topk_extra,
            page_block_size, stride_kv_row,
            page_block_size_extra, stride_kv_row_extra,
            attn_sink_ptr, tl_ptr, etl_ptr,
            stream);
        return;
    }

    switch (mt) {
    case ModelType::V32:
        sparse_mla_prefill_launch_v32(
            Q, KV_cache, indices, output, out_lse,
            sm_scale, num_heads, num_tokens, topk,
            page_block_size, stride_kv_row, attn_sink_ptr, tl_ptr, stream);
        break;
    case ModelType::MODEL1:
        sparse_mla_prefill_launch_model1(
            Q, KV_cache, indices, output, out_lse,
            sm_scale, num_heads, num_tokens, topk,
            page_block_size, stride_kv_row, attn_sink_ptr, tl_ptr, stream);
        break;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mla_splitkv_fwd", &sparse_mla_splitkv_fwd,
          "Split-KV decode forward (SM120, V32+MODEL1) — v1, supports dual-cache",
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
          py::arg("KV_cache_extra") = py::none(),
          py::arg("indices_extra") = py::none(),
          py::arg("topk_length") = py::none(),
          py::arg("topk_length_extra") = py::none(),
          py::arg("attn_sink") = py::none());
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
          "Combine partial outputs from split-KV decode (v1)",
          py::arg("partial_O"),
          py::arg("partial_LSE"),
          py::arg("output"),
          py::arg("out_lse"),
          py::arg("nsplits"),
          py::arg("attn_sink") = py::none());
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
          "Sparse MLA prefill forward (SM120, V32+MODEL1) — supports dual-cache",
          py::arg("Q"),
          py::arg("KV_cache"),
          py::arg("indices"),
          py::arg("output"),
          py::arg("out_lse"),
          py::arg("sm_scale"),
          py::arg("topk"),
          py::arg("stride_kv_row"),
          py::arg("page_block_size"),
          py::arg("KV_cache_extra") = py::none(),
          py::arg("indices_extra") = py::none(),
          py::arg("topk_length") = py::none(),
          py::arg("topk_length_extra") = py::none(),
          py::arg("attn_sink") = py::none());
}
