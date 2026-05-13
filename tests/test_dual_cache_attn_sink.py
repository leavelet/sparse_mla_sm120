"""Dual-cache MODEL1 decode/prefill with attn_sink + asymmetric topk/page_block_size.

Covers the kernel instantiations that DSv4-Flash actually uses end-to-end on
sm120 but which the existing test_dual_cache_{decode,prefill}.py do not
exercise:

  * (topk=128, topk_extra=128, page_block_size=64, page_block_size_extra=64)
  * (topk=128, topk_extra=512, page_block_size=64, page_block_size_extra=64)
      — DSv4 C4A layers (compress_ratio=4 → 256/4 = 64)
  * (topk=128, topk_extra=512, page_block_size=64, page_block_size_extra=2)
      — DSv4 C128A layers (compress_ratio=128 → 256/128 = 2)

Plus a non-zero attn_sink per head (with padded heads carrying -inf so
factor == 1 there) to validate the post-output scaling
`output *= sigmoid(lse - attn_sink)` from upstream FlashMLA convention.

Reference: dequantize both caches, gather union of indices, single softmax,
optional attn_sink post-scale.
"""
import math
import pytest
import torch

import flash_mla_sm120
from tests.test_decode import (
    quantize_kv_model1,
    dequantize_kv_model1,
)


def ref_dual_cache_attn_with_sink(
    q, kv_main_dequant, idx_main,
    kv_extra_dequant, idx_extra, sm_scale, d_v,
    attn_sink=None,
):
    """Reference: dual-cache sparse attention with optional attn_sink post-scaling.

    q:                (num_tokens, num_heads, d_qk) bf16
    kv_main_dequant:  (num_blocks_m, block_size_m, 1, d_qk) bf16
    idx_main:         (num_tokens, topk_main) int32 (-1 = invalid)
    kv_extra_dequant: (num_blocks_e, block_size_e, 1, d_qk) bf16
    idx_extra:        (num_tokens, topk_extra) int32 (-1 = invalid)
    attn_sink:        (num_heads,) float32 or None
    """
    num_tokens, h_q, d_qk = q.shape
    q_f = q.float()

    main_flat = kv_main_dequant.view(-1, d_qk).float()
    extra_flat = kv_extra_dequant.view(-1, d_qk).float()

    gathered_main = main_flat.index_select(0, idx_main.clamp(min=0).view(-1)) \
        .view(num_tokens, idx_main.size(-1), d_qk)
    gathered_extra = extra_flat.index_select(0, idx_extra.clamp(min=0).view(-1)) \
        .view(num_tokens, idx_extra.size(-1), d_qk)

    gathered = torch.cat([gathered_main, gathered_extra], dim=-2)
    invalid = torch.cat([idx_main < 0, idx_extra < 0], dim=-1)

    P = torch.einsum("nhd,ntd->nht", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")
    lse = torch.logsumexp(P, dim=-1)              # (n, h), natural-log
    lse_safe = lse.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out = torch.einsum("nht,ntd->nhd", weights, gathered[..., :d_v]).to(torch.bfloat16)

    if attn_sink is not None:
        sink = attn_sink.float().to(q.device)
        # sigmoid(lse - sink) per head; -inf sink → factor == 1, +inf → 0.
        factor = torch.sigmoid(lse - sink.unsqueeze(0))  # (n, h)
        out = (out.float() * factor.unsqueeze(-1)).to(torch.bfloat16)
    return out, lse


# ── Common cache + index builders ────────────────────────────────────

def _make_cache(num_blocks, block_size, d_qk, device="cuda"):
    kv_bf16 = (
        torch.randn(num_blocks, block_size, 1, d_qk,
                    device=device, dtype=torch.bfloat16) / 10
    ).clamp(-1, 1)
    packed = quantize_kv_model1(kv_bf16)
    dequant = dequantize_kv_model1(packed)
    return packed, dequant


def _make_indices(num_tokens, topk, s_kv, device="cuda", invalid_tail=5):
    idx = torch.randint(0, s_kv, (num_tokens, topk),
                        device=device, dtype=torch.int32)
    if invalid_tail > 0:
        idx[:, -invalid_tail:] = -1
    return idx


def _make_attn_sink(num_heads, real_heads=None, device="cuda"):
    """Per-head sink. Real heads (first `real_heads`) get small learned values;
    padded heads (rest) get -inf, which must result in factor == 1 (no-op)."""
    if real_heads is None:
        real_heads = num_heads
    sink = torch.full((num_heads,), -float("inf"),
                      device=device, dtype=torch.float32)
    # Mix of small positive and negative logits to exercise both directions
    # of the sigmoid factor.
    real = torch.tensor(
        [(-1.0) ** i * (0.5 + 0.1 * (i % 7)) for i in range(real_heads)],
        device=device, dtype=torch.float32,
    )
    sink[:real_heads] = real
    return sink


# ── Tests ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "topk_extra,block_size_extra",
    [
        (128, 64),  # symmetric: matches existing test_dual_cache_*
        (512, 64),  # DSv4 C4A
        (512, 2),   # DSv4 C128A
    ],
    ids=["sym-128-bs64", "c4a-512-bs64", "c128a-512-bs2"],
)
@pytest.mark.parametrize("num_heads", [64, 128])
@pytest.mark.parametrize("with_sink", [False, True])
def test_dual_cache_decode_combos(topk_extra, block_size_extra, num_heads, with_sink):
    torch.manual_seed(0)
    d_qk, d_v, main_block_size = 512, 512, 64
    sm_scale = d_qk ** -0.5
    num_tokens = 32

    num_blocks_main = 32
    # Pick num_blocks_extra so the extra-cache pool can hold topk_extra entries.
    num_blocks_extra = max(32, (topk_extra * 4 + block_size_extra - 1) // block_size_extra)
    kv_main_packed, kv_main_dequant = _make_cache(num_blocks_main, main_block_size, d_qk)
    kv_extra_packed, kv_extra_dequant = _make_cache(num_blocks_extra, block_size_extra, d_qk)

    q = (
        torch.randn(num_tokens, num_heads, d_qk,
                    device="cuda", dtype=torch.bfloat16) / 10
    ).clamp(-1, 1)
    idx_main = _make_indices(num_tokens, 128, num_blocks_main * main_block_size)
    idx_extra = _make_indices(num_tokens, topk_extra, num_blocks_extra * block_size_extra)

    # First half of heads get real sink, rest -inf (padded).
    attn_sink = _make_attn_sink(num_heads, real_heads=num_heads // 2) if with_sink else None

    ref_out, _ = ref_dual_cache_attn_with_sink(
        q, kv_main_dequant, idx_main, kv_extra_dequant, idx_extra, sm_scale, d_v,
        attn_sink=attn_sink,
    )

    out, _ = flash_mla_sm120.sparse_mla_decode_fwd(
        q, kv_main_packed, idx_main, sm_scale, d_v,
        extra_kv_cache=kv_extra_packed,
        extra_indices=idx_extra,
        attn_sink=attn_sink,
    )

    err = (out.float() - ref_out.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    print(f"\n  dual decode tk={128} tk_ex={topk_extra} bs_ex={block_size_extra} "
          f"h={num_heads} sink={with_sink}: max_err={max_err:.6f} mean_err={mean_err:.6f}")
    # Wider tolerance for (512, 2) combo (FP8 noise compounds over more entries).
    tol = 0.02 if topk_extra == 512 else 0.01
    assert max_err < tol, f"dual decode failed: max_err={max_err}"


@pytest.mark.parametrize(
    "topk_extra,block_size_extra",
    [
        (128, 64),
        (512, 64),
        (512, 2),
    ],
    ids=["sym-128-bs64", "c4a-512-bs64", "c128a-512-bs2"],
)
@pytest.mark.parametrize("num_heads", [64, 128])
@pytest.mark.parametrize("with_sink", [False, True])
def test_dual_cache_prefill_combos(topk_extra, block_size_extra, num_heads, with_sink):
    """Same matrix as decode but routes to the MG prefill kernel (num_tokens > 64)."""
    torch.manual_seed(0)
    d_qk, d_v, main_block_size = 512, 512, 64
    sm_scale = d_qk ** -0.5
    num_tokens = 96  # > _DECODE_THRESHOLD (64)

    num_blocks_main = 32
    num_blocks_extra = max(32, (topk_extra * 4 + block_size_extra - 1) // block_size_extra)
    kv_main_packed, kv_main_dequant = _make_cache(num_blocks_main, main_block_size, d_qk)
    kv_extra_packed, kv_extra_dequant = _make_cache(num_blocks_extra, block_size_extra, d_qk)

    q = (
        torch.randn(num_tokens, num_heads, d_qk,
                    device="cuda", dtype=torch.bfloat16) / 10
    ).clamp(-1, 1)
    idx_main = _make_indices(num_tokens, 128, num_blocks_main * main_block_size)
    idx_extra = _make_indices(num_tokens, topk_extra, num_blocks_extra * block_size_extra)

    attn_sink = _make_attn_sink(num_heads, real_heads=num_heads // 2) if with_sink else None

    ref_out, _ = ref_dual_cache_attn_with_sink(
        q, kv_main_dequant, idx_main, kv_extra_dequant, idx_extra, sm_scale, d_v,
        attn_sink=attn_sink,
    )

    out, _ = flash_mla_sm120.sparse_mla_prefill_fwd(
        q, kv_main_packed, idx_main, sm_scale, d_v,
        extra_kv_cache=kv_extra_packed,
        extra_indices=idx_extra,
        attn_sink=attn_sink,
    )

    err = (out.float() - ref_out.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    print(f"\n  dual prefill tk={128} tk_ex={topk_extra} bs_ex={block_size_extra} "
          f"h={num_heads} sink={with_sink}: max_err={max_err:.6f} mean_err={mean_err:.6f}")
    tol = 0.02 if topk_extra == 512 else 0.01
    assert max_err < tol, f"dual prefill failed: max_err={max_err}"
