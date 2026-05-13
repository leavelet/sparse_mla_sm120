"""End-to-end tests for the V2 path through flash_mla_with_kvcache.

Covers the scheduler-driven decode v2 + combine v2 wired through
`flash_mla_sm120.interface.flash_mla_with_kvcache`, including the DSv4
PAGE_BLOCK_SIZE_EXTRA=2 (C128A) dual-cache instantiation.
"""

import torch
import pytest

from flash_mla_sm120.interface import (
    FlashMLASchedMeta,
    flash_mla_with_kvcache,
)
from tests.test_decode import quantize_kv_model1, dequantize_kv_model1
from tests.test_v4_features import ref_v4_decode, check_allclose, OUT_ABS_TOL, OUT_REL_TOL


def _run(num_heads, topk, batch_size,
         extra_topk=0, extra_block_size=64,
         have_attn_sink=False, seed=0):
    torch.manual_seed(seed)
    d_qk, d_v = 512, 512
    sm_scale = d_qk ** -0.5
    main_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * main_block_size
    s_q = 1

    kv_bf16 = (torch.randn(num_blocks, main_block_size, 1, d_qk,
                            device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_packed = quantize_kv_model1(kv_bf16)
    kv_dequant = dequantize_kv_model1(kv_packed)

    q = (torch.randn(batch_size, s_q, num_heads, d_qk,
                      device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (batch_size, s_q, topk),
                             device="cuda", dtype=torch.int32)
    indices[:, :, -10:] = -1

    extra_kv_packed = None
    extra_kv_dequant = None
    extra_indices_t = None
    if extra_topk > 0:
        extra_nb = max(4, (extra_topk * 2 + extra_block_size - 1) // extra_block_size)
        extra_s_kv = extra_nb * extra_block_size
        extra_bf16 = (torch.randn(extra_nb, extra_block_size, 1, d_qk,
                                   device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
        extra_kv_packed = quantize_kv_model1(extra_bf16)
        extra_kv_dequant = dequantize_kv_model1(extra_kv_packed)
        extra_indices_t = torch.randint(0, extra_s_kv,
                                         (batch_size, s_q, topk if extra_topk == 0 else extra_topk),
                                         dtype=torch.int32, device="cuda")
        extra_indices_t = extra_indices_t[:, :, :extra_topk]

    attn_sink = None
    if have_attn_sink:
        attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 2.0

    ref_out, ref_lse = ref_v4_decode(
        q, kv_dequant, indices, sm_scale, d_v,
        extra_kv_dequant=extra_kv_dequant,
        extra_indices=extra_indices_t,
        attn_sink=attn_sink,
    )

    # The interface wraps decode_v2 inside, with indices reshaped to
    # (b*s_q, [1,] topk). flash_mla_with_kvcache expects indices shape
    # (b, s_q, [1,] topk) — match the original sparse-attention layout.
    out, lse = flash_mla_with_kvcache(
        q, kv_packed, None, None, d_v,
        FlashMLASchedMeta(), None,
        softmax_scale=sm_scale,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_kv_packed,
        extra_indices_in_kvcache=extra_indices_t,
    )
    # interface returns (batch, s_q, h_q, d_v); ref returns (b, s_q, h_q, d_v)
    out_check = out.view_as(ref_out)
    ok, msg = check_allclose(
        f"interface_v2[h={num_heads},topk={topk},ext={extra_topk}/{extra_block_size},sink={have_attn_sink}]",
        out_check, ref_out, abs_tol=OUT_ABS_TOL, rel_tol=OUT_REL_TOL,
    )
    return ok, msg


@pytest.mark.parametrize("num_heads,topk,batch", [
    (64, 128, 4),
    (64, 512, 4),
    (128, 128, 4),
    (128, 512, 4),
])
def test_single_cache(num_heads, topk, batch):
    ok, msg = _run(num_heads, topk, batch)
    assert ok, msg


@pytest.mark.parametrize("num_heads,topk,extra_topk,ext_pbs", [
    # C4A: same block size on extra cache
    (64, 128, 512, 64),
    (128, 128, 512, 64),
    # C128A: smaller block size on extra cache (DSv4 compressed)
    (64, 128, 512, 2),
    (128, 128, 512, 2),
])
def test_dual_cache(num_heads, topk, extra_topk, ext_pbs):
    ok, msg = _run(num_heads, topk, 4, extra_topk=extra_topk, extra_block_size=ext_pbs)
    assert ok, msg


@pytest.mark.parametrize("num_heads,topk,extra_topk,ext_pbs", [
    (64, 128, 512, 64),
    (128, 128, 512, 2),
])
def test_dual_cache_with_sink(num_heads, topk, extra_topk, ext_pbs):
    ok, msg = _run(num_heads, topk, 4, extra_topk=extra_topk,
                   extra_block_size=ext_pbs, have_attn_sink=True)
    assert ok, msg
