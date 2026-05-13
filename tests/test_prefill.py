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

    @pytest.mark.parametrize("num_heads,topk,num_tokens", [
        (64, 128, 65),     # V4 SWA window=128
        (128, 128, 65),
        (64, 256, 80),     # arbitrary topk
    ])
    def test_v4_swa_topk(self, num_heads, topk, num_tokens):
        """V4 SWA prefill: topk=window_size=128 (runtime, no template)."""
        max_err, mean_err = run_prefill_test(
            "MODEL1", d_qk=512, d_v=512, topk=topk,
            num_heads=num_heads, num_tokens=num_tokens, bf16_qk=True)
        print(f"\n  MODEL1 SWA prefill h={num_heads} topk={topk} tokens={num_tokens}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f}")
        assert max_err < 0.0015, (
            f"MODEL1 SWA prefill failed: max_err={max_err}")


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


# ── Dual-Cache (extra_k_cache) Prefill Tests ──────────────────────────

def ref_dual_cache_attn(q, kv_main_dq, kv_extra_dq, main_idx, extra_idx,
                         sm_scale, d_v, topk_length=None, attn_sink=None):
    """FP32 reference for dual-cache sparse attention.

    q:            (num_tokens, num_heads, d_qk) bf16
    kv_main_dq:   (num_blocks, block_size, 1, d_qk) bf16 — main (SWA) cache
    kv_extra_dq:  (num_blocks, block_size, 1, d_qk) bf16 — extra (compressed) cache
    main_idx:     (num_tokens, topk) int32
    extra_idx:    (num_tokens, extra_topk) int32
    """
    T, H, D = q.shape
    topk = main_idx.shape[-1]
    extra_topk = extra_idx.shape[-1]

    kv_main_flat = kv_main_dq.view(-1, D).float()
    kv_extra_flat = kv_extra_dq.view(-1, D).float()
    q_f = q.float()

    main_fix = main_idx.clamp(min=0)
    main_inv = main_idx < 0
    extra_fix = extra_idx.clamp(min=0)
    extra_inv = extra_idx < 0

    g_main = kv_main_flat.index_select(0, main_fix.view(-1)).view(T, topk, D)
    g_extra = kv_extra_flat.index_select(0, extra_fix.view(-1)).view(T, extra_topk, D)
    gathered = torch.cat([g_main, g_extra], dim=1)

    P = torch.einsum("thd,tsd->ths", q_f, gathered) * sm_scale

    invalid = torch.cat([main_inv, extra_inv], dim=-1)
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")

    if topk_length is not None:
        for t in range(T):
            tl = topk_length[t].item()
            if tl < topk:
                P[t, :, tl:topk] = float("-inf")

    lse = torch.logsumexp(P, dim=-1)
    lse_safe = lse.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))

    out = torch.einsum("ths,tsd->thd", weights, gathered[..., :d_v])

    if attn_sink is not None:
        sink_f = attn_sink.float().view(1, H)
        scale = 1.0 / (1.0 + torch.exp(sink_f - lse))
        out = out * scale.unsqueeze(-1)

    return out.to(torch.bfloat16), lse


def run_dual_cache_test(num_heads, topk, extra_topk, num_tokens,
                         block_size=64, num_blocks=64, num_extra_blocks=64,
                         attn_sink_val=None, topk_length_mode=None, seed=42):
    """Run dual-cache prefill and compare to reference."""
    torch.manual_seed(seed)
    d_qk, d_v = 512, 512
    sm_scale = d_qk ** -0.5
    s_kv_main = num_blocks * block_size
    s_kv_extra = num_extra_blocks * block_size

    kv_main = (torch.randn(num_blocks, block_size, 1, d_qk,
                             device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_extra = (torch.randn(num_extra_blocks, block_size, 1, d_qk,
                              device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_main_p = quantize_kv_model1(kv_main)
    kv_extra_p = quantize_kv_model1(kv_extra)
    kv_main_dq = dequantize_kv_model1(kv_main_p)
    kv_extra_dq = dequantize_kv_model1(kv_extra_p)

    q = (torch.randn(num_tokens, num_heads, d_qk,
                       device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    main_idx = torch.randint(0, s_kv_main, (num_tokens, topk),
                              device="cuda", dtype=torch.int32)
    extra_idx = torch.randint(0, s_kv_extra, (num_tokens, extra_topk),
                               device="cuda", dtype=torch.int32)
    main_idx[:, -5:] = -1

    topk_length = None
    if topk_length_mode == "short":
        topk_length = torch.randint(topk // 2, topk, (num_tokens,),
                                     device="cuda", dtype=torch.int32)
    elif topk_length_mode == "variable":
        topk_length = torch.randint(1, topk + 1, (num_tokens,),
                                     device="cuda", dtype=torch.int32)

    attn_sink = None
    if attn_sink_val is not None:
        attn_sink = torch.full((num_heads,), attn_sink_val,
                                device="cuda", dtype=torch.float32)

    ref_out, ref_lse = ref_dual_cache_attn(
        q, kv_main_dq, kv_extra_dq, main_idx, extra_idx,
        sm_scale, d_v, topk_length, attn_sink)

    out, ml, lse = flash_mla_sm120.sparse_mla_prefill_fwd(
        q, kv_main_p, main_idx, sm_scale, d_v,
        attn_sink=attn_sink, topk_length=topk_length,
        extra_k_cache=kv_extra_p, extra_indices=extra_idx,
        extra_topk=extra_topk)

    err = (out.float() - ref_out.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    return max_err, mean_err


class TestDualCachePrefill:
    """Dual-cache prefill: SWA (main) + compressed (extra) V4 config."""

    @pytest.mark.parametrize("num_heads,topk,extra_topk,num_tokens", [
        (64, 128, 512, 65),     # V4 Flash C4A: SWA=128, compressed=512
        (64, 128, 512, 128),    # larger batch
        (128, 128, 1024, 65),   # V4 Pro C4A: SWA=128, compressed=1024
        (128, 128, 1024, 100),
        (64, 128, 256, 65),     # smaller extra_topk
        (64, 512, 512, 65),     # equal main and extra
    ])
    def test_correctness(self, num_heads, topk, extra_topk, num_tokens):
        max_err, mean_err = run_dual_cache_test(
            num_heads, topk, extra_topk, num_tokens)
        print(f"\n  dual-cache h={num_heads} topk={topk}+{extra_topk} "
              f"tok={num_tokens}: max_err={max_err:.6f} mean_err={mean_err:.6f}")
        assert max_err < 0.0015, f"dual-cache failed: max_err={max_err}"

    def test_swa_only(self):
        """extra_k_cache=None — should match single-cache."""
        torch.manual_seed(42)
        d_qk, d_v, h, topk = 512, 512, 64, 128
        nb, bs, num_tokens = 64, 64, 65
        sm_scale = d_qk ** -0.5

        kv = (torch.randn(nb, bs, 1, d_qk, device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
        kv_p = quantize_kv_model1(kv)
        q = torch.randn(num_tokens, h, d_qk, device="cuda", dtype=torch.bfloat16) / 10
        idx = torch.randint(0, nb * bs, (num_tokens, topk), device="cuda", dtype=torch.int32)

        out_single, _, _ = flash_mla_sm120.sparse_mla_prefill_fwd(
            q, kv_p, idx, sm_scale, d_v)
        out_dual, _, _ = flash_mla_sm120.sparse_mla_prefill_fwd(
            q, kv_p, idx, sm_scale, d_v,
            extra_k_cache=None, extra_indices=None, extra_topk=0)

        err = (out_single.float() - out_dual.float()).abs().max().item()
        print(f"\n  swa-only: single vs dual(extra=None) max_diff={err:.6f}")
        assert err == 0.0, f"swa-only mismatch: {err}"

    @pytest.mark.parametrize("num_heads,topk,extra_topk", [
        (64, 128, 512), (128, 128, 1024),
    ])
    def test_with_attn_sink(self, num_heads, topk, extra_topk):
        max_err, _ = run_dual_cache_test(
            num_heads, topk, extra_topk, 65, attn_sink_val=2.0)
        print(f"\n  dual-cache+sink h={num_heads}: max_err={max_err:.6f}")
        assert max_err < 0.002, f"dual-cache+sink failed: max_err={max_err}"

    @pytest.mark.parametrize("num_heads,topk,extra_topk", [
        (64, 128, 512), (128, 128, 1024),
    ])
    def test_with_topk_length(self, num_heads, topk, extra_topk):
        """topk_length masks main cache entries; extra cache fully valid."""
        max_err, _ = run_dual_cache_test(
            num_heads, topk, extra_topk, 65, topk_length_mode="variable")
        print(f"\n  dual-cache+topk_length h={num_heads}: max_err={max_err:.6f}")
        assert max_err < 0.0015, f"dual-cache+topk_length failed: max_err={max_err}"

    def test_all_extra_invalid(self):
        """All extra indices = -1, only main cache contributes."""
        torch.manual_seed(42)
        d_qk, d_v, h, topk, extra_topk = 512, 512, 64, 128, 512
        nb, bs, num_tokens = 64, 64, 65
        sm_scale = d_qk ** -0.5

        kv_main = (torch.randn(nb, bs, 1, d_qk, device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
        kv_extra = (torch.randn(nb, bs, 1, d_qk, device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
        kv_main_p = quantize_kv_model1(kv_main)
        kv_extra_p = quantize_kv_model1(kv_extra)

        q = torch.randn(num_tokens, h, d_qk, device="cuda", dtype=torch.bfloat16) / 10
        main_idx = torch.randint(0, nb * bs, (num_tokens, topk), device="cuda", dtype=torch.int32)
        extra_idx = torch.full((num_tokens, extra_topk), -1, device="cuda", dtype=torch.int32)

        out_dual, _, _ = flash_mla_sm120.sparse_mla_prefill_fwd(
            q, kv_main_p, main_idx, sm_scale, d_v,
            extra_k_cache=kv_extra_p, extra_indices=extra_idx, extra_topk=extra_topk)

        out_single, _, _ = flash_mla_sm120.sparse_mla_prefill_fwd(
            q, kv_main_p, main_idx, sm_scale, d_v)

        err = (out_dual.float() - out_single.float()).abs().max().item()
        print(f"\n  all-extra-invalid: max_diff={err:.6f}")
        assert err < 1e-4, f"all-extra-invalid mismatch: {err}"

    def test_race_check(self):
        """Run 3x with different seeds to catch non-determinism."""
        for seed in [42, 123, 999]:
            max_err, _ = run_dual_cache_test(128, 128, 1024, 65, seed=seed)
            assert max_err < 0.0015, f"race check seed={seed}: max_err={max_err}"
        print("\n  dual-cache race check (3 seeds): PASS")

    def test_out_param(self):
        """Verify out= pre-allocated buffer works with dual-cache."""
        torch.manual_seed(42)
        d_qk, d_v, h, topk, extra_topk = 512, 512, 64, 128, 512
        nb, bs, num_tokens = 64, 64, 65
        sm_scale = d_qk ** -0.5

        kv_main_p = quantize_kv_model1(
            (torch.randn(nb, bs, 1, d_qk, device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1))
        kv_extra_p = quantize_kv_model1(
            (torch.randn(nb, bs, 1, d_qk, device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1))
        q = torch.randn(num_tokens, h, d_qk, device="cuda", dtype=torch.bfloat16) / 10
        main_idx = torch.randint(0, nb * bs, (num_tokens, topk), device="cuda", dtype=torch.int32)
        extra_idx = torch.randint(0, nb * bs, (num_tokens, extra_topk), device="cuda", dtype=torch.int32)

        out_alloc, _, _ = flash_mla_sm120.sparse_mla_prefill_fwd(
            q, kv_main_p, main_idx, sm_scale, d_v,
            extra_k_cache=kv_extra_p, extra_indices=extra_idx, extra_topk=extra_topk)

        out_buf = torch.empty_like(out_alloc)
        out_pre, _, _ = flash_mla_sm120.sparse_mla_prefill_fwd(
            q, kv_main_p, main_idx, sm_scale, d_v, out=out_buf,
            extra_k_cache=kv_extra_p, extra_indices=extra_idx, extra_topk=extra_topk)

        assert out_pre.data_ptr() == out_buf.data_ptr(), "out buffer not used"
        assert (out_alloc == out_pre).all(), "out= results differ"
        print("\n  dual-cache out= param: PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
