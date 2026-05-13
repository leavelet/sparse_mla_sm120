"""Reproduce e2e gibberish: run flash_mla_with_kvcache on realistic-magnitude
inputs (Q ~ unit-stdev randn, K with the same) and see if output collapses
to zero / flat / NaN. If yes, the FP8 precision wall is the e2e blocker."""
import torch
import pytest
import math

from flash_mla_sm120.interface import flash_mla_with_kvcache, FlashMLASchedMeta
from tests.test_decode import quantize_kv_model1, dequantize_kv_model1, ref_sparse_attn_decode


def _run(q_scale: float, k_scale: float, with_sink: bool, with_extra: bool, with_topklen: bool = False):
    """q_scale/k_scale: pre-clamp multiplier. 0.1 = test-friendly; 1.0 = real."""
    torch.manual_seed(0)
    batch, s_q, h_q, d_qk, d_v = 4, 1, 128, 512, 512
    topk = 128
    main_block_size = 64
    num_blocks = 32
    s_kv = num_blocks * main_block_size
    sm_scale = d_qk ** -0.5

    # FP8-friendly synthesis: randn × scale, then clamp to keep amax bounded.
    kv_bf16 = (torch.randn(num_blocks, main_block_size, 1, d_qk,
                            device="cuda", dtype=torch.bfloat16) * k_scale).clamp(-k_scale * 10, k_scale * 10)
    kv_packed = quantize_kv_model1(kv_bf16)
    kv_dequant = dequantize_kv_model1(kv_packed)

    q = (torch.randn(batch, s_q, h_q, d_qk,
                      device="cuda", dtype=torch.bfloat16) * q_scale).clamp(-q_scale * 10, q_scale * 10)
    indices = torch.randint(0, s_kv, (batch, s_q, topk), device="cuda", dtype=torch.int32)

    extra_kv_packed = None
    extra_indices = None
    extra_topk = 512
    if with_extra:
        extra_block_size = 2  # C128A
        extra_nb = (extra_topk * 2 + extra_block_size - 1) // extra_block_size
        extra_bf16 = (torch.randn(extra_nb, extra_block_size, 1, d_qk,
                                   device="cuda", dtype=torch.bfloat16) * k_scale
                      ).clamp(-k_scale * 10, k_scale * 10)
        extra_kv_packed = quantize_kv_model1(extra_bf16)
        extra_indices = torch.randint(0, extra_nb * extra_block_size,
                                       (batch, s_q, extra_topk),
                                       dtype=torch.int32, device="cuda")

    attn_sink = None
    if with_sink:
        attn_sink = (torch.randn(h_q, device="cuda", dtype=torch.float32) * 2.0)

    topk_length = None
    extra_topk_length = None
    if with_topklen:
        topk_length = torch.full((batch * s_q,), topk // 2, dtype=torch.int32, device="cuda")
        if with_extra:
            extra_topk_length = torch.full((batch * s_q,), extra_topk // 2, dtype=torch.int32, device="cuda")

    out, lse = flash_mla_with_kvcache(
        q, kv_packed, None, None, d_v,
        FlashMLASchedMeta(), None,
        softmax_scale=sm_scale,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_kv_packed,
        extra_indices_in_kvcache=extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )

    out_finite = torch.isfinite(out).all().item()
    out_abs_mean = out.abs().float().mean().item()
    out_abs_max = out.abs().float().max().item()
    nz_frac = (out != 0).float().mean().item()
    lse_finite = torch.isfinite(lse).all().item()
    lse_mean = lse.float().mean().item()
    lse_max = lse.float().max().item()
    return {
        "finite": out_finite,
        "abs_mean": out_abs_mean,
        "abs_max": out_abs_max,
        "nonzero_frac": nz_frac,
        "lse_finite": lse_finite,
        "lse_mean": lse_mean,
        "lse_max": lse_max,
    }


def test_magnitudes_no_extras():
    for q_scale, k_scale in [(0.1, 0.1), (0.3, 0.3), (1.0, 1.0), (3.0, 3.0)]:
        r = _run(q_scale, k_scale, with_sink=False, with_extra=False)
        print(f"  q={q_scale}/k={k_scale}: finite={r['finite']} abs_mean={r['abs_mean']:.4e} abs_max={r['abs_max']:.4e} nz={r['nonzero_frac']:.4f}")
        assert r["finite"], f"NaN/Inf at q={q_scale} k={k_scale}"


def test_magnitudes_with_dual_and_sink():
    for q_scale, k_scale in [(0.1, 0.1), (1.0, 1.0), (3.0, 3.0)]:
        r = _run(q_scale, k_scale, with_sink=True, with_extra=True)
        print(f"  q={q_scale}/k={k_scale} +sink+ext: finite={r['finite']} abs_mean={r['abs_mean']:.4e} abs_max={r['abs_max']:.4e} nz={r['nonzero_frac']:.4f}")
        assert r["finite"], f"NaN/Inf at q={q_scale} k={k_scale}"


def test_realistic_full_combo():
    """Mimic vLLM's call: real magnitudes + dual cache + sink + topk_length."""
    for q_scale, k_scale in [(0.5, 0.5), (1.0, 1.0), (3.0, 3.0)]:
        r = _run(q_scale, k_scale, with_sink=True, with_extra=True, with_topklen=True)
        print(f"  q={q_scale}/k={k_scale} full: finite={r['finite']} abs_mean={r['abs_mean']:.4e} abs_max={r['abs_max']:.4e} nz={r['nonzero_frac']:.4f}")
        assert r["finite"], f"NaN/Inf at q={q_scale} k={k_scale}"
        assert r["nonzero_frac"] > 0.5, f"output mostly zero at q={q_scale} k={k_scale}"
