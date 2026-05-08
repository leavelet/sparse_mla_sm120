// Stub implementations for prefill launch functions.
// Will be replaced when prefill kernels are implemented.

#include <torch/extension.h>
#include "../../arch/common.cuh"

void sparse_mla_prefill_launch_v32(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output, float sm_scale,
    int num_heads, int num_tokens, int topk, cudaStream_t stream)
{
    TORCH_CHECK(false, "V32 prefill not yet implemented in flash_mla_sm120");
}

void sparse_mla_prefill_launch_model1(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output, float sm_scale,
    int num_heads, int num_tokens, int topk, cudaStream_t stream)
{
    TORCH_CHECK(false, "MODEL1 prefill not yet implemented in flash_mla_sm120");
}
