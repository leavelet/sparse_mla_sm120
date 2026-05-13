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
                     block_size=64, num_blocks=64, bf16_qk=True):
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

    out, max_logits, lse = flash_mla_sm120.sparse_mla_prefill_fwd(
        q_flat, kv_packed, idx_flat, sm_scale, d_v, bf16_qk=bf16_qk)
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
    @pytest.mark.parametrize("bf16_qk", [True, False], ids=["bf16qk", "fp8qk"])
    def test_correctness(self, num_heads, topk, num_tokens, bf16_qk):
        max_err, mean_err = run_prefill_test(
            "MODEL1", d_qk=512, d_v=512, topk=topk,
            num_heads=num_heads, num_tokens=num_tokens, bf16_qk=bf16_qk)
        tag = "BF16" if bf16_qk else "FP8"
        threshold = 0.001 if bf16_qk else 0.002
        print(f"\n  MODEL1 prefill [{tag}] h={num_heads} topk={topk} tokens={num_tokens}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f}")
        assert max_err < threshold, (
            f"MODEL1 prefill [{tag}] failed: max_err={max_err} > {threshold}")


# ── attn_sink Prefill Tests (MODEL1 only) ───────────────────────────

def run_prefill_attn_sink_test(num_heads, topk, num_tokens, attn_sink_mode="random",
                                block_size=64, num_blocks=64, seed=42):
    torch.manual_seed(seed)
    d_qk, d_v = 512, 512
    sm_scale = d_qk ** -0.5
    s_kv = num_blocks * block_size

    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                            device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_packed = quantize_kv_model1(kv_bf16)
    kv_dequant = dequantize_kv_model1(kv_packed)

    q = (torch.randn(num_tokens, 1, num_heads, d_qk,
                      device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (num_tokens, 1, topk),
                             device="cuda", dtype=torch.int32)
    indices[:, :, -10:] = -1

    if attn_sink_mode == "random":
        attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 2.0
    elif attn_sink_mode == "neg_inf":
        attn_sink = torch.full((num_heads,), float("-inf"), device="cuda", dtype=torch.float32)
    elif attn_sink_mode == "large_pos":
        attn_sink = torch.full((num_heads,), 20.0, device="cuda", dtype=torch.float32)
    elif attn_sink_mode == "zero":
        attn_sink = torch.zeros(num_heads, device="cuda", dtype=torch.float32)
    elif attn_sink_mode == "mixed":
        attn_sink = torch.zeros(num_heads, device="cuda", dtype=torch.float32)
        attn_sink[:num_heads // 2] = 15.0
        attn_sink[num_heads // 2:] = -15.0
    else:
        raise ValueError(f"Unknown mode: {attn_sink_mode}")

    ref_out, ref_lse = ref_sparse_attn_decode(q, kv_dequant, indices, sm_scale, d_v)
    ref_lse_f = ref_lse.float()
    sink_f = attn_sink.view(1, 1, num_heads)
    scale = 1.0 / (1.0 + torch.exp(sink_f - ref_lse_f))
    ref_out_sink = (ref_out.float() * scale.unsqueeze(-1)).to(torch.bfloat16)

    q_flat = q.view(-1, num_heads, d_qk)
    idx_flat = indices.view(-1, topk)
    out, max_logits, lse = flash_mla_sm120.sparse_mla_prefill_fwd(
        q_flat, kv_packed, idx_flat, sm_scale, d_v, attn_sink)
    out = out.view_as(ref_out_sink)

    err = (out.float() - ref_out_sink.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    return max_err, mean_err


class TestAttnSinkPrefill:
    """Test attn_sink in prefill kernel epilogue — comprehensive coverage."""

    @pytest.mark.parametrize("num_heads,topk,num_tokens", [
        (64, 512, 65),     (64, 512, 128),   (64, 512, 256),
        (128, 1024, 65),   (128, 1024, 128),  (128, 1024, 256),
    ])
    def test_random_sink(self, num_heads, topk, num_tokens):
        max_err, mean_err = run_prefill_attn_sink_test(
            num_heads, topk, num_tokens, "random")
        print(f"\n  prefill sink random h={num_heads} topk={topk} tok={num_tokens}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f}")
        assert max_err < 0.002, f"max_err={max_err}"

    @pytest.mark.parametrize("num_heads,topk", [
        (64, 512), (128, 1024),
    ])
    def test_neg_inf_sink(self, num_heads, topk):
        """attn_sink = -inf should have no effect."""
        max_err_nosink, _ = run_prefill_test(
            "MODEL1", d_qk=512, d_v=512, topk=topk,
            num_heads=num_heads, num_tokens=65)
        max_err_sink, _ = run_prefill_attn_sink_test(
            num_heads, topk, 65, "neg_inf")
        print(f"\n  prefill sink -inf h={num_heads}: nosink={max_err_nosink:.6f} "
              f"sink={max_err_sink:.6f}")
        assert max_err_sink < 0.001, f"max_err={max_err_sink}"

    @pytest.mark.parametrize("num_heads,topk", [
        (64, 512), (128, 1024),
    ])
    def test_large_pos_sink(self, num_heads, topk):
        max_err, _ = run_prefill_attn_sink_test(num_heads, topk, 65, "large_pos")
        print(f"\n  prefill sink large_pos h={num_heads}: max_err={max_err:.6f}")
        assert max_err < 0.002, f"max_err={max_err}"

    @pytest.mark.parametrize("num_heads,topk", [
        (64, 512), (128, 1024),
    ])
    def test_zero_sink(self, num_heads, topk):
        max_err, _ = run_prefill_attn_sink_test(num_heads, topk, 65, "zero")
        print(f"\n  prefill sink zero h={num_heads}: max_err={max_err:.6f}")
        assert max_err < 0.002, f"max_err={max_err}"

    @pytest.mark.parametrize("num_heads,topk", [
        (64, 512), (128, 1024),
    ])
    def test_mixed_sink(self, num_heads, topk):
        max_err, _ = run_prefill_attn_sink_test(num_heads, topk, 100, "mixed")
        print(f"\n  prefill sink mixed h={num_heads}: max_err={max_err:.6f}")
        assert max_err < 0.002, f"max_err={max_err}"

    def test_race_check(self):
        """Run 3x with different seeds."""
        for seed in [42, 123, 999]:
            max_err, _ = run_prefill_attn_sink_test(
                128, 1024, 100, "random", seed=seed)
            assert max_err < 0.002, f"race check seed={seed}: max_err={max_err}"
        print("\n  prefill attn_sink race check (3 seeds): PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
