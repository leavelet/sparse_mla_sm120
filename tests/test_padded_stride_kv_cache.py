"""Regression: kernel must handle kv_cache tensors with a *padded* block stride.

vLLM pads each block of its DSv4 paged kv_cache to a multiple of 576 bytes
for the FlashMLA ABI: actual_block_bytes = block_size * 584 (37376 for the
64-token MODEL1 block); padded_block_bytes = round_up(37376, 576) = 37440.
The extra 64 bytes live at the end of each block and are never written/read.

The Python binding for the C extension must propagate this padded block
stride to the kernel as a per-row override (so page_block_size * stride ==
actual block byte stride). Prior to the fix, the binding's
`extra_stride_kv_row` was read as `tensor.stride(-2) * element_size()`,
which always returned the natural 584, dropping the per-block padding for
the extra (compressed) cache and causing the kernel to read at the wrong
block-byte offsets — producing all-NaN output starting at block 1.

The fix mirrors `_effective_stride_kv_row` in C++ via the helper
`effective_stride_kv_row` in csrc/binding.cpp.
"""
import pytest
import torch

import flash_mla_sm120
from flash_mla_sm120.ops import (
    sparse_mla_decode_v2_fwd,
    sparse_mla_prefill_fwd,
)
from tests.test_decode import quantize_kv_model1, dequantize_kv_model1


def _to_padded_block_stride(kv_packed, padded_block_bytes):
    """Wrap kv_packed (natural-stride) into a padded-block-stride view.

    kv_packed: shape (num_blocks, block_size, 1, head_bytes) uint8,
               naturally packed (stride(0) = block_size * head_bytes).
    Returns a view with the same logical contents and per-byte data,
    but stride(0) = padded_block_bytes (e.g., 37440 vs natural 37376).
    The extra padding bytes per block stay zero.
    """
    num_blocks, block_size, kvh, head_bytes = kv_packed.shape
    natural_block_bytes = block_size * kvh * head_bytes
    assert padded_block_bytes >= natural_block_bytes
    storage = torch.zeros(
        num_blocks * padded_block_bytes, dtype=torch.uint8, device=kv_packed.device
    )
    src = kv_packed.contiguous().view(num_blocks, natural_block_bytes)
    storage_view = storage.view(num_blocks, padded_block_bytes)
    storage_view[:, :natural_block_bytes] = src
    return torch.as_strided(
        storage,
        size=(num_blocks, block_size, kvh, head_bytes),
        stride=(padded_block_bytes, kvh * head_bytes, head_bytes, 1),
    ), storage  # keep storage alive


def _ref_dual_cache(q, kv_main_dq, idx_main, kv_extra_dq, idx_extra, sm_scale, d_v,
                    topk_length=None, extra_topk_length=None, attn_sink=None):
    num_tokens, h_q, d_qk = q.shape
    main_flat = kv_main_dq.view(-1, d_qk).float()
    extra_flat = kv_extra_dq.view(-1, d_qk).float()
    gm = main_flat.index_select(0, idx_main.clamp(min=0).view(-1)).view(
        num_tokens, idx_main.size(-1), d_qk)
    ge = extra_flat.index_select(0, idx_extra.clamp(min=0).view(-1)).view(
        num_tokens, idx_extra.size(-1), d_qk)
    gathered = torch.cat([gm, ge], dim=-2)
    invalid = torch.cat([idx_main < 0, idx_extra < 0], dim=-1)
    P = torch.einsum("nhd,ntd->nht", q.float(), gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")
    if topk_length is not None:
        mask = torch.arange(idx_main.size(-1), device=q.device).unsqueeze(0) \
            >= topk_length.long().unsqueeze(-1)
        P[..., : idx_main.size(-1)][mask.unsqueeze(1).expand(-1, h_q, -1)] = float("-inf")
    if extra_topk_length is not None:
        mask = torch.arange(idx_extra.size(-1), device=q.device).unsqueeze(0) \
            >= extra_topk_length.long().unsqueeze(-1)
        P[..., idx_main.size(-1):][mask.unsqueeze(1).expand(-1, h_q, -1)] = float("-inf")
    if attn_sink is not None:
        P = torch.cat(
            [P, attn_sink.float().view(1, h_q, 1).expand(num_tokens, -1, 1)], dim=-1
        )
    lse = torch.logsumexp(P, dim=-1)
    safe = lse.clone(); safe[safe == float("-inf")] = float("+inf")
    w = torch.exp(P - safe.unsqueeze(-1))
    if attn_sink is not None:
        w = w[..., :-1]
    out = torch.einsum("nht,ntd->nhd", w, gathered[..., :d_v])
    return out.to(torch.bfloat16), lse


def _make_inputs(num_blocks, block_size=64, num_heads=32, num_tokens=1,
                 d_qk=512, topk_main=128, topk_extra=512, swa_len=24, ex_len=6,
                 seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    kv_main_bf16 = (torch.randn(
        num_blocks, block_size, 1, d_qk, device=device, dtype=torch.bfloat16
    ) / 10).clamp(-1, 1)
    kv_extra_bf16 = (torch.randn(
        num_blocks, block_size, 1, d_qk, device=device, dtype=torch.bfloat16
    ) / 10).clamp(-1, 1)
    kv_main_packed = quantize_kv_model1(kv_main_bf16)
    kv_extra_packed = quantize_kv_model1(kv_extra_bf16)
    kv_main_dq = dequantize_kv_model1(kv_main_packed)
    kv_extra_dq = dequantize_kv_model1(kv_extra_packed)

    q = (torch.randn(num_tokens, num_heads, d_qk,
                     device=device, dtype=torch.bfloat16) / 10).clamp(-1, 1)
    s_kv = num_blocks * block_size
    idx_main = torch.randint(0, s_kv, (num_tokens, topk_main),
                             device=device, dtype=torch.int32)
    idx_extra = torch.randint(0, s_kv, (num_tokens, topk_extra),
                              device=device, dtype=torch.int32)
    idx_main[:, swa_len:] = -1
    idx_extra[:, ex_len:] = -1
    topk_length = torch.tensor([swa_len], device=device, dtype=torch.int32)
    extra_topk_length = torch.tensor([ex_len], device=device, dtype=torch.int32)
    attn_sink = torch.randn(num_heads, device=device, dtype=torch.float32) * 0.5
    return (q, kv_main_packed, idx_main, kv_main_dq,
            kv_extra_packed, idx_extra, kv_extra_dq,
            topk_length, extra_topk_length, attn_sink)


# vllm's DSv4 SlidingWindowMLASpec sets alignment=576, so for
# block_size=64 and per-token bytes=584, padded_block_bytes = round_up(64*584, 576)
# = round_up(37376, 576) = 37440. That's the live padded stride.
_PADDED_BLOCK_BYTES = 37440
_NATURAL_BLOCK_BYTES = 64 * 584  # 37376


@pytest.mark.parametrize("pad_main,pad_extra", [
    (False, True),   # only extra padded — this is the bug we fixed
    (True, False),   # only main padded
    (True, True),    # both padded — what vllm actually does
])
def test_decode_v2_padded_kv_cache(pad_main, pad_extra):
    """sparse_mla_decode_v2_fwd must produce finite output matching reference
    when k_cache and/or extra_k_cache use vllm's padded block stride."""
    inputs = _make_inputs(num_blocks=32)
    (q, kv_main, idx_main, kv_main_dq, kv_extra, idx_extra, kv_extra_dq,
     topk_length, extra_topk_length, attn_sink) = inputs
    sm_scale = q.shape[-1] ** -0.5
    d_v = 512

    storage_refs = []
    if pad_main:
        kv_main, store = _to_padded_block_stride(kv_main, _PADDED_BLOCK_BYTES)
        storage_refs.append(store)
    if pad_extra:
        kv_extra, store = _to_padded_block_stride(kv_extra, _PADDED_BLOCK_BYTES)
        storage_refs.append(store)

    out, _ = sparse_mla_decode_v2_fwd(
        q.contiguous(), kv_main, idx_main, sm_scale, d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        extra_kv_cache=kv_extra,
        extra_indices=idx_extra,
        extra_topk_length=extra_topk_length,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all(), \
        f"non-finite output with pad_main={pad_main}, pad_extra={pad_extra}"

    ref_out, _ = _ref_dual_cache(
        q, kv_main_dq, idx_main, kv_extra_dq, idx_extra,
        sm_scale, d_v, topk_length=topk_length,
        extra_topk_length=extra_topk_length, attn_sink=attn_sink,
    )
    err = (out.float() - ref_out.float()).abs().max().item()
    assert err < 0.01, f"max abs err {err} too large (pad_main={pad_main}, " \
                       f"pad_extra={pad_extra})"
    # Keep storage refs alive until kernel finishes
    del storage_refs


@pytest.mark.parametrize("pad_extra", [True, False])
def test_prefill_padded_extra_kv_cache(pad_extra):
    """sparse_mla_prefill_fwd must also propagate the padded extra stride."""
    inputs = _make_inputs(num_blocks=32, num_tokens=128)
    (q, kv_main, idx_main, kv_main_dq, kv_extra, idx_extra, kv_extra_dq,
     topk_length, extra_topk_length, attn_sink) = inputs
    sm_scale = q.shape[-1] ** -0.5
    d_v = 512

    storage_ref = None
    if pad_extra:
        kv_extra, storage_ref = _to_padded_block_stride(kv_extra, _PADDED_BLOCK_BYTES)

    # prefill expects per-token topk_length / extra_topk_length matching num_tokens
    num_tokens = q.shape[0]
    tl = topk_length.repeat(num_tokens)
    etl = extra_topk_length.repeat(num_tokens)

    out, _ = sparse_mla_prefill_fwd(
        q.contiguous(), kv_main, idx_main, sm_scale, d_v,
        attn_sink=attn_sink,
        topk_length=tl,
        extra_kv_cache=kv_extra,
        extra_indices=idx_extra,
        extra_topk_length=etl,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all(), f"non-finite (pad_extra={pad_extra})"
    del storage_ref
