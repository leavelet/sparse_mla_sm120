#pragma once

#include "common.cuh"

// Online softmax state for one head, maintained in registers.
struct SoftmaxState {
    float m;       // running max (in log2 space, i.e. raw_max * sm_scale * log2e)
    float l;       // running sum of exp2(score * sm_scale * log2e - m)
};

__device__ __forceinline__ SoftmaxState softmax_init() {
    return {-1e30f, 0.0f};
}

// Update softmax state with a new block of scores.
// scores[i] are raw dot-product scores (NOT scaled).
// Returns the rescale factor alpha = exp2(old_m - new_m) to apply to old accumulators.
__device__ __forceinline__ float softmax_update(
    SoftmaxState& state,
    float* scores,
    int n,
    float sm_scale_log2e)
{
    float new_max = state.m;
    for (int i = 0; i < n; i++) {
        new_max = fmaxf(new_max, scores[i] * sm_scale_log2e);
    }

    float alpha = fast_exp2f(state.m - new_max);
    state.l *= alpha;

    for (int i = 0; i < n; i++) {
        scores[i] = fast_exp2f(scores[i] * sm_scale_log2e - new_max);
        state.l += scores[i];
    }
    state.m = new_max;
    return alpha;
}

// Finalize: compute 1/l for normalization.
__device__ __forceinline__ float softmax_finalize(const SoftmaxState& state) {
    return (state.l > 0.0f) ? (1.0f / state.l) : 0.0f;
}

// Compute LSE (log-sum-exp) in log2 space for the combine kernel.
// lse = log2(l) + m
__device__ __forceinline__ float softmax_lse(const SoftmaxState& state) {
    if (state.l <= 0.0f) return -1e30f;
    return log2f(state.l) + state.m;
}
