"""
Prefill correctness tests for flash_mla_sm120.
Same sparse attention as decode, but num_tokens > 64 (prefill threshold).

Reuses quantization helpers from test_decode.py.
"""

import torch
import pytest
import math
import flash_mla_sm120
from test_decode import (
    quantize_kv_v32, dequantize_kv_v32,
    quantize_kv_model1, dequantize_kv_model1,
    ref_sparse_attn_decode,
)


def run_prefill_test(model_type, d_qk, d_v, topk, num_heads, num_tokens,
                     block_size=64, num_blocks=64, atol=0.01):
    torch.manual_seed(42)
    sm_scale = d_qk ** -0.5
    s_kv = num_blocks * block_size

    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                            device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)

    if model_type == "V32":
        kv_packed = quantize_kv_v32(kv_bf16)
        kv_dequant = dequantize_kv_v32(kv_packed)
    else:
        kv_packed = quantize_kv_model1(kv_bf16)
        kv_dequant = dequantize_kv_model1(kv_packed)

    q = (torch.randn(num_tokens, 1, num_heads, d_qk,
                      device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (num_tokens, 1, topk),
                             device="cuda", dtype=torch.int32)
    indices[:, :, -10:] = -1

    ref_out, ref_lse = ref_sparse_attn_decode(q, kv_dequant, indices, sm_scale, d_v)

    q_flat = q.view(-1, num_heads, d_qk)
    idx_flat = indices.view(-1, topk)

    out, lse = flash_mla_sm120.sparse_mla_prefill_fwd(
        q_flat, kv_packed, idx_flat, sm_scale, d_v)
    out = out.view_as(ref_out)

    err = (out.float() - ref_out.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    return max_err, mean_err


class TestV32Prefill:
    """V32 prefill: d_qk=576, topk=2048"""

    @pytest.mark.parametrize("num_heads,num_tokens", [
        (128, 65),   # V3.2 TP1, just above threshold
        (128, 128),  # V3.2 TP1, larger batch
        (64, 65),    # GLM 5.1 TP1
        (16, 100),   # V3.2 TP8
    ])
    def test_correctness(self, num_heads, num_tokens):
        max_err, mean_err = run_prefill_test(
            "V32", d_qk=576, d_v=512, topk=2048,
            num_heads=num_heads, num_tokens=num_tokens)
        print(f"\n  V32 prefill h={num_heads} tokens={num_tokens}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f}")
        assert max_err < 0.001, f"V32 prefill failed: max_err={max_err}"


class TestMODEL1Prefill:
    """MODEL1 prefill: d_qk=512, topk=512/1024"""

    @pytest.mark.parametrize("num_heads,topk,num_tokens", [
        (64, 512, 65),     # V4 Flash TP1
        (128, 1024, 65),   # V4 Pro TP1
        (128, 1024, 100),  # V4 Pro TP1, larger batch
    ])
    def test_correctness(self, num_heads, topk, num_tokens):
        max_err, mean_err = run_prefill_test(
            "MODEL1", d_qk=512, d_v=512, topk=topk,
            num_heads=num_heads, num_tokens=num_tokens)
        print(f"\n  MODEL1 prefill h={num_heads} topk={topk} tokens={num_tokens}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f}")
        assert max_err < 0.001, f"MODEL1 prefill failed: max_err={max_err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
