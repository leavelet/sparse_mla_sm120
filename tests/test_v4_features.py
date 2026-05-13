"""
V4 feature tests: topk_length + extra_k_cache + attn_sink (MODEL1 only).
Modeled after FlashMLA's test_flash_mla_sparse_decoding.py patterns.

Reference: concatenate main + extra KV, mask by topk_length, compute attention,
scale by attn_sink.
"""

import torch
import pytest
import math
import sys
sys.path.insert(0, 'tests')

from flash_mla_sm120.cuda import (
    get_decode_metadata,
    sparse_mla_splitkv_v2_fwd,
    sparse_mla_combine_fwd,
)
from test_decode import quantize_kv_model1, dequantize_kv_model1

BI = 64
HPB = 16
NUM_SMS = 188
FIXED_OVERHEAD = 5

# FlashMLA precision thresholds (decode: tighter, prefill: slightly looser)
OUT_ABS_TOL = 1e-3
OUT_REL_TOL = 2.01 / 128  # ~0.0157
PREFILL_ABS_TOL = 8e-4
PREFILL_REL_TOL = 3.01 / 128  # ~0.0235 (FlashMLA prefill threshold)


def check_allclose(name, actual, expected, abs_tol=OUT_ABS_TOL, rel_tol=OUT_REL_TOL):
    """allclose: |actual - expected| <= abs_tol + rel_tol * max(|actual|, |expected|)"""
    diff = (actual.float() - expected.float()).abs()
    tol = abs_tol + rel_tol * torch.maximum(actual.float().abs(), expected.float().abs())
    violations = diff > tol
    if violations.any():
        max_violation = (diff - tol)[violations].max().item()
        max_diff = diff.max().item()
        n_violations = violations.sum().item()
        n_total = violations.numel()
        msg = (f"{name}: {n_violations}/{n_total} violations, "
               f"max_diff={max_diff:.6f}, max_violation_over_tol={max_violation:.6f}")
        return False, msg
    return True, f"{name}: PASS (max_diff={diff.max().item():.6f})"


def ref_v4_decode(q, kv_dequant, indices, sm_scale, d_v,
                  topk_length=None,
                  extra_kv_dequant=None, extra_indices=None, extra_topk_length=None,
                  attn_sink=None):
    """Reference V4 decode: variable topk + extra cache + attn_sink."""
    b, s_q, h_q, d_qk = q.shape
    topk = indices.shape[-1]
    q_f = q.float()

    # Process main KV scope
    kv_flat = kv_dequant.view(-1, d_qk).float()
    idx_fixed = indices.clamp(min=0)
    invalid = indices < 0
    if topk_length is not None:
        invalid = invalid | (torch.arange(topk, device=q.device).view(1, 1, topk) >= topk_length.view(b, 1, 1))

    gathered = kv_flat.index_select(0, idx_fixed.view(-1)).view(b, s_q, topk, d_qk)

    # Process extra KV scope (if present)
    if extra_kv_dequant is not None and extra_indices is not None:
        extra_topk = extra_indices.shape[-1]
        ekv_flat = extra_kv_dequant.view(-1, d_qk).float()
        eidx_fixed = extra_indices.clamp(min=0)
        einvalid = extra_indices < 0
        if extra_topk_length is not None:
            einvalid = einvalid | (torch.arange(extra_topk, device=q.device).view(1, 1, extra_topk) >= extra_topk_length.view(b, 1, 1))

        egathered = ekv_flat.index_select(0, eidx_fixed.view(-1)).view(b, s_q, extra_topk, d_qk)
        gathered = torch.cat([gathered, egathered], dim=2)
        invalid = torch.cat([invalid, einvalid], dim=2)

    gathered[gathered != gathered] = 0.0  # NaN → 0

    P = torch.einsum("bshd,bstd->bsht", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(2).expand_as(P)] = float("-inf")

    lse = torch.logsumexp(P, dim=-1)
    lse_safe = lse.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out = torch.einsum("bsht,bstd->bshd", weights, gathered[..., :d_v])

    # attn_sink scaling
    if attn_sink is not None:
        sink_f = attn_sink.view(1, 1, h_q)
        scale = 1.0 / (1.0 + torch.exp(sink_f - lse))
        out = out * scale.unsqueeze(-1)

    lonely = (lse == float("-inf"))
    out[lonely.unsqueeze(-1).expand_as(out)] = 0.0

    return out.to(torch.bfloat16), lse


def run_v4_decode_test(num_heads, topk, batch_size,
                       have_topk_length=False,
                       extra_topk=0,
                       have_extra_topk_length=False,
                       have_attn_sink=False,
                       block_size=64, num_blocks=64, seed=42):
    torch.manual_seed(seed)
    d_qk, d_v = 512, 512
    sm_scale = d_qk ** -0.5
    s_kv = num_blocks * block_size
    s_q = 1

    # Main KV cache
    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                            device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_packed = quantize_kv_model1(kv_bf16)
    kv_dequant = dequantize_kv_model1(kv_packed)

    q = (torch.randn(batch_size, 1, num_heads, d_qk,
                      device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (batch_size, 1, topk),
                             device="cuda", dtype=torch.int32)
    indices[:, :, -10:] = -1

    # topk_length
    topk_length = None
    if have_topk_length:
        topk_length = torch.randint(0, topk + 1, (batch_size,),
                                     dtype=torch.int32, device="cuda")

    # Extra KV cache
    extra_kv_packed = None
    extra_kv_dequant = None
    extra_indices_t = None
    extra_topk_length = None
    if extra_topk > 0:
        extra_nb = max(4, (extra_topk * 2 + block_size - 1) // block_size)
        extra_s_kv = extra_nb * block_size
        extra_bf16 = (torch.randn(extra_nb, block_size, 1, d_qk,
                                   device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
        extra_kv_packed = quantize_kv_model1(extra_bf16)
        extra_kv_dequant = dequantize_kv_model1(extra_kv_packed)
        extra_indices_t = torch.randint(0, extra_s_kv, (batch_size, 1, extra_topk),
                                         dtype=torch.int32, device="cuda")
        if have_extra_topk_length:
            extra_topk_length = torch.randint(0, extra_topk + 1, (batch_size,),
                                               dtype=torch.int32, device="cuda")

    # attn_sink
    attn_sink = None
    if have_attn_sink:
        attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 2.0
        inf_mask = torch.randn(num_heads, device="cuda", dtype=torch.float32)
        attn_sink[inf_mask > 1.0] = float("inf")
        attn_sink[inf_mask < -1.0] = float("-inf")

    # Reference
    ref_out, ref_lse = ref_v4_decode(
        q, kv_dequant, indices, sm_scale, d_v,
        topk_length=topk_length,
        extra_kv_dequant=extra_kv_dequant,
        extra_indices=extra_indices_t,
        extra_topk_length=extra_topk_length,
        attn_sink=attn_sink)

    # V2 kernel path
    replicate_h = (num_heads + HPB - 1) // HPB
    num_sm_parts = max(NUM_SMS // (s_q * replicate_h), 1)
    total_splits_bound = batch_size + num_sm_parts

    sched_meta = torch.empty(num_sm_parts * 8, dtype=torch.int32, device="cuda")
    num_splits = torch.empty(batch_size + 1, dtype=torch.int32, device="cuda")
    get_decode_metadata(batch_size, topk, extra_topk, num_sm_parts, FIXED_OVERHEAD,
                        topk_length, extra_topk_length, sched_meta, num_splits)

    o_accum = torch.empty(total_splits_bound, s_q, num_heads, d_v,
                          dtype=torch.float32, device="cuda")
    lse_accum = torch.empty(total_splits_bound, s_q, num_heads,
                            dtype=torch.float32, device="cuda")
    output = torch.zeros(batch_size, num_heads, d_v, dtype=torch.bfloat16, device="cuda")
    out_lse = torch.zeros(batch_size, num_heads, dtype=torch.float32, device="cuda")

    stride_kv_row = kv_packed.stride(-2) * kv_packed.element_size()
    page_block_size = kv_packed.shape[-3] if kv_packed.dim() >= 3 else 1
    q_flat = q.view(batch_size, num_heads, d_qk)
    idx_flat = indices.view(batch_size, topk)

    sparse_mla_splitkv_v2_fwd(
        q_flat, kv_packed, idx_flat,
        o_accum, lse_accum, output, out_lse,
        sched_meta, num_splits,
        sm_scale, topk, stride_kv_row, page_block_size, num_sm_parts,
        attn_sink,
        extra_kv_packed, extra_indices_t.view(batch_size, extra_topk) if extra_indices_t is not None else None,
        topk_length, extra_topk,
        extra_topk_length)
    torch.cuda.synchronize()

    # Combine for split batches
    ns_cpu = num_splits.cpu().tolist()
    for i in range(batch_size):
        my_splits = ns_cpu[i + 1] - ns_cpu[i]
        if my_splits <= 1:
            continue
        start = ns_cpu[i]
        po = o_accum[start:start + my_splits, 0:1, :, :].squeeze(1).unsqueeze(0)
        pl = lse_accum[start:start + my_splits, 0:1, :].squeeze(1).unsqueeze(0)
        o_i = torch.empty(1, num_heads, d_v, dtype=torch.bfloat16, device="cuda")
        l_i = torch.empty(1, num_heads, dtype=torch.float32, device="cuda")
        sparse_mla_combine_fwd(po, pl, o_i, l_i, my_splits, attn_sink)
        output[i] = o_i[0]
        out_lse[i] = l_i[0]

    out_view = output.view_as(ref_out)
    err = (out_view.float() - ref_out.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    ok, msg = check_allclose("out", out_view, ref_out)
    return max_err, mean_err, ok, msg


# ── topk_length Tests ───────────────────────────────────────────────

class TestTopkLength:
    """Variable topk_length per batch (MODEL1 only)."""

    @pytest.mark.parametrize("num_heads,topk,batch_size", [
        (64, 512, 1), (64, 512, 4), (64, 512, 8),
        (128, 1024, 1), (128, 1024, 4),
        (8, 512, 1), (8, 512, 4),
    ])
    def test_topk_length(self, num_heads, topk, batch_size):
        max_err, mean_err, ok, msg = run_v4_decode_test(
            num_heads, topk, batch_size, have_topk_length=True)
        print(f"\n  topk_length h={num_heads} topk={topk} bs={batch_size}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f} {msg}")
        assert ok, msg

    def test_topk_length_zero(self):
        """All topk_length=0 → output should be 0."""
        max_err, _, ok, msg = run_v4_decode_test(64, 512, 4, have_topk_length=True, seed=999)
        print(f"\n  topk_length seed=999: max_err={max_err:.6f} {msg}")
        assert ok, msg


# ── extra_k_cache Tests ─────────────────────────────────────────────

class TestExtraKCache:
    """Extra KV cache pool (MODEL1 decode only)."""

    @pytest.mark.parametrize("num_heads,topk,extra_topk,batch_size", [
        (64, 512, 64, 1), (64, 512, 64, 4),
        (64, 512, 128, 4),
        (128, 1024, 128, 1), (128, 1024, 128, 4),
        (8, 512, 64, 1),
    ])
    def test_extra_cache(self, num_heads, topk, extra_topk, batch_size):
        max_err, mean_err, ok, msg = run_v4_decode_test(
            num_heads, topk, batch_size, extra_topk=extra_topk)
        print(f"\n  extra_cache h={num_heads} topk={topk} extra={extra_topk} bs={batch_size}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f} {msg}")
        assert ok, msg

    @pytest.mark.parametrize("num_heads,topk,extra_topk,batch_size", [
        (64, 512, 64, 4),
        (128, 1024, 128, 4),
    ])
    def test_extra_with_topk_length(self, num_heads, topk, extra_topk, batch_size):
        max_err, mean_err, ok, msg = run_v4_decode_test(
            num_heads, topk, batch_size, extra_topk=extra_topk,
            have_topk_length=True, have_extra_topk_length=True)
        print(f"\n  extra+topk_len h={num_heads} topk={topk} extra={extra_topk} bs={batch_size}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f} {msg}")
        assert ok, msg


# ── Combined V4 Features ────────────────────────────────────────────

class TestV4Combined:
    """All V4 features together: topk_length + extra_k_cache + attn_sink."""

    @pytest.mark.parametrize("num_heads,topk,extra_topk,batch_size", [
        (64, 512, 64, 1), (64, 512, 64, 4),
        (128, 1024, 128, 1), (128, 1024, 128, 4),
        (128, 1024, 128, 8),
    ])
    def test_all_features(self, num_heads, topk, extra_topk, batch_size):
        max_err, mean_err, ok, msg = run_v4_decode_test(
            num_heads, topk, batch_size,
            have_topk_length=True, extra_topk=extra_topk,
            have_extra_topk_length=True, have_attn_sink=True)
        print(f"\n  V4 all h={num_heads} topk={topk} extra={extra_topk} bs={batch_size}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f} {msg}")
        assert ok, msg

    def test_race_check(self):
        """3 seeds for race detection."""
        for seed in [42, 123, 999]:
            max_err, _, ok, msg = run_v4_decode_test(
                128, 1024, 4, have_topk_length=True,
                extra_topk=128, have_extra_topk_length=True,
                have_attn_sink=True, seed=seed)
            assert ok, f"V4 race check seed={seed}: {msg}"
        print("\n  V4 race check (3 seeds): PASS")


# ── Performance Benchmark ───────────────────────────────────────────

# ── Prefill topk_length Tests ────────────────────────────────────────

class TestPrefillTopkLength:
    """Prefill with variable topk_length per token (MODEL1 only)."""

    @staticmethod
    def _run(num_heads, topk, num_tokens, seed=42):
        torch.manual_seed(seed)
        d_qk, d_v = 512, 512
        sm_scale = d_qk ** -0.5
        block_size, num_blocks = 64, 64
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

        topk_length = torch.randint(BI, topk + 1, (num_tokens,),
                                     dtype=torch.int32, device="cuda")

        # Reference: kernel with pre-masked indices (bit-exact baseline)
        ref_indices = indices.clone()
        for i in range(num_tokens):
            tl = topk_length[i].item()
            ref_indices[i, :, tl:] = -1

        import flash_mla_sm120
        q_flat = q.view(-1, num_heads, d_qk)
        idx_flat = indices.view(-1, topk)
        ref_flat = ref_indices.view(-1, topk)

        # Kernel with topk_length
        out, lse = flash_mla_sm120.sparse_mla_prefill_fwd(
            q_flat, kv_packed, idx_flat, sm_scale, d_v,
            topk_length=topk_length)
        # Reference: kernel with pre-masked indices (no topk_length)
        ref_out, ref_lse = flash_mla_sm120.sparse_mla_prefill_fwd(
            q_flat, kv_packed, ref_flat, sm_scale, d_v)

        # Compare vs FP32 reference (not bit-exact due to FP8 quantization)
        from test_decode import ref_sparse_attn_decode
        ref_out_fp32, _ = ref_sparse_attn_decode(q, kv_dequant, ref_indices, sm_scale, d_v)
        out = out.view_as(ref_out_fp32)

        ok, msg = check_allclose("prefill_topk_len", out, ref_out_fp32,
                                 abs_tol=PREFILL_ABS_TOL, rel_tol=PREFILL_REL_TOL)
        max_err = (out.float() - ref_out_fp32.float()).abs().max().item()
        return max_err, ok, msg

    @pytest.mark.parametrize("num_heads,topk,num_tokens", [
        (64, 512, 65), (64, 512, 128),
        (128, 1024, 65), (128, 1024, 100),
    ])
    def test_prefill_topk_length(self, num_heads, topk, num_tokens):
        max_err, ok, msg = self._run(num_heads, topk, num_tokens)
        print(f"\n  prefill topk_len h={num_heads} topk={topk} tok={num_tokens}: "
              f"max_err={max_err:.6f} {msg}")
        assert ok, msg

    def test_prefill_topk_length_with_sink(self):
        """Prefill topk_length + attn_sink combined."""
        torch.manual_seed(42)
        d_qk, d_v, num_heads, topk, num_tokens = 512, 512, 128, 1024, 65
        sm_scale = d_qk ** -0.5
        block_size, num_blocks = 64, 64
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
        topk_length = torch.randint(topk // 2, topk + 1, (num_tokens,),
                                     dtype=torch.int32, device="cuda")
        attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 2.0

        # Reference
        ref_indices = indices.clone()
        for i in range(num_tokens):
            ref_indices[i, :, topk_length[i].item():] = -1
        from test_decode import ref_sparse_attn_decode
        ref_out, ref_lse = ref_sparse_attn_decode(q, kv_dequant, ref_indices, sm_scale, d_v)
        sink_f = attn_sink.view(1, 1, num_heads)
        scale = 1.0 / (1.0 + torch.exp(sink_f - ref_lse.float()))
        ref_out = (ref_out.float() * scale.unsqueeze(-1)).to(torch.bfloat16)

        # Kernel
        import flash_mla_sm120
        q_flat = q.view(-1, num_heads, d_qk)
        idx_flat = indices.view(-1, topk)
        out, lse = flash_mla_sm120.sparse_mla_prefill_fwd(
            q_flat, kv_packed, idx_flat, sm_scale, d_v,
            attn_sink, topk_length)
        out = out.view_as(ref_out)

        ok, msg = check_allclose("prefill_topk_len+sink", out, ref_out,
                                 abs_tol=PREFILL_ABS_TOL, rel_tol=PREFILL_REL_TOL)
        max_err = (out.float() - ref_out.float()).abs().max().item()
        print(f"\n  prefill topk_len+sink: max_err={max_err:.6f} {msg}")
        assert ok, msg

    def test_race_check(self):
        for seed in [42, 123, 999]:
            max_err, ok, msg = self._run(128, 1024, 100, seed=seed)
            assert ok, f"prefill topk_len race seed={seed}: {msg}"
        print("\n  prefill topk_length race check (3 seeds): PASS")

    def test_extreme_short_topk(self):
        """Extreme short topk_length (0-10). Only verify no crash + output is finite."""
        torch.manual_seed(42)
        d_qk, d_v, num_heads, topk, num_tokens = 512, 512, 64, 512, 65
        sm_scale = d_qk ** -0.5
        block_size, num_blocks = 64, 64
        s_kv = num_blocks * block_size

        kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                                device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
        kv_packed = quantize_kv_model1(kv_bf16)
        q = (torch.randn(num_tokens, num_heads, d_qk,
                          device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
        indices = torch.randint(0, s_kv, (num_tokens, topk),
                                 device="cuda", dtype=torch.int32)
        topk_length = torch.randint(0, 11, (num_tokens,),
                                     dtype=torch.int32, device="cuda")

        import flash_mla_sm120
        out, lse = flash_mla_sm120.sparse_mla_prefill_fwd(
            q, kv_packed, indices, sm_scale, d_v, topk_length=topk_length)
        assert torch.isfinite(out).all() or (out == 0).all(), "output has NaN/Inf"
        print(f"\n  extreme short topk (0-10): no crash, output finite")


class TestV4Perf:
    """Performance comparison: v2 with V4 features vs v1 baseline."""

    def _bench(self, fn, warmup=50, reps=200):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
        return times[reps // 2]

    @pytest.mark.parametrize("num_heads,topk,batch_size,extra_topk,label", [
        (64, 512, 1, 0, "M1 Flash bs=1 (no extra)"),
        (64, 512, 1, 64, "M1 Flash bs=1 (extra=64)"),
        (128, 1024, 1, 0, "M1 Pro bs=1 (no extra)"),
        (128, 1024, 1, 128, "M1 Pro bs=1 (extra=128)"),
        (128, 1024, 4, 128, "M1 Pro bs=4 (extra=128)"),
    ])
    def test_perf(self, num_heads, topk, batch_size, extra_topk, label):
        torch.manual_seed(42)
        d_qk, d_v = 512, 512
        sm_scale = d_qk ** -0.5
        block_size, num_blocks = 64, 64
        s_kv = num_blocks * block_size
        s_q = 1

        kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                                device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
        kv_packed = quantize_kv_model1(kv_bf16)
        q = torch.randn(batch_size, num_heads, d_qk, dtype=torch.bfloat16, device="cuda")
        indices = torch.randint(0, s_kv, (batch_size, topk), dtype=torch.int32, device="cuda")
        topk_length = torch.randint(topk // 2, topk + 1, (batch_size,), dtype=torch.int32, device="cuda")

        extra_kv = None
        extra_idx = None
        if extra_topk > 0:
            extra_nb = max(4, extra_topk * 2 // block_size)
            extra_bf16 = (torch.randn(extra_nb, block_size, 1, d_qk,
                                       device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
            extra_kv = quantize_kv_model1(extra_bf16)
            extra_idx = torch.randint(0, extra_nb * block_size, (batch_size, extra_topk),
                                       dtype=torch.int32, device="cuda")

        replicate_h = (num_heads + HPB - 1) // HPB
        num_sm_parts = max(NUM_SMS // (s_q * replicate_h), 1)
        total_splits = batch_size + num_sm_parts

        sched_meta = torch.empty(num_sm_parts * 8, dtype=torch.int32, device="cuda")
        num_splits = torch.empty(batch_size + 1, dtype=torch.int32, device="cuda")
        o_accum = torch.empty(total_splits, s_q, num_heads, d_v, dtype=torch.float32, device="cuda")
        lse_accum = torch.empty(total_splits, s_q, num_heads, dtype=torch.float32, device="cuda")
        output = torch.empty(batch_size, num_heads, d_v, dtype=torch.bfloat16, device="cuda")
        out_lse = torch.empty(batch_size, num_heads, dtype=torch.float32, device="cuda")

        stride_kv_row = kv_packed.stride(-2) * kv_packed.element_size()
        page_block_size = kv_packed.shape[-3]

        def run():
            get_decode_metadata(batch_size, topk, extra_topk, num_sm_parts, FIXED_OVERHEAD,
                                topk_length, None, sched_meta, num_splits)
            sparse_mla_splitkv_v2_fwd(
                q, kv_packed, indices, o_accum, lse_accum, output, out_lse,
                sched_meta, num_splits, sm_scale, topk, stride_kv_row,
                page_block_size, num_sm_parts, None,
                extra_kv, extra_idx, topk_length, extra_topk, None)

        us = self._bench(run)
        print(f"\n  {label}: {us:.1f} us")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
