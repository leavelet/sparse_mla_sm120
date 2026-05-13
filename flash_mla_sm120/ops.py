import torch
from typing import Optional, Tuple


def _load_lib():
    import flash_mla_sm120.cuda as _C
    return _C


_HPB = 16
_FIXED_OVERHEAD = 5
_BI = 64

_num_sms_cache: dict = {}

def _get_num_sms() -> int:
    dev = torch.cuda.current_device()
    if dev not in _num_sms_cache:
        _num_sms_cache[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _num_sms_cache[dev]


def _effective_stride_kv_row(kv_cache: torch.Tensor) -> int:
    """Return the per-token byte stride the kernel should use.

    The kernel computes stride_kv_block = page_block_size * stride_kv_row.
    Some callers (e.g. vLLM) pad the block stride for alignment, so
    page_block_size * per_token_stride != actual block-to-block stride.
    We detect this and override stride_kv_row so the formula comes out
    to the correct block byte stride. The kernel uses IO::IO_STRIDE
    (a model constant) for per-token offsets within a block, so this
    override is safe.
    """
    if kv_cache.dim() < 3:
        return kv_cache.stride(-2) * kv_cache.element_size()
    page_block_size = kv_cache.shape[-3]
    natural_row_bytes = kv_cache.stride(-2) * kv_cache.element_size()
    block_stride_bytes = kv_cache.stride(0) * kv_cache.element_size()
    if block_stride_bytes == page_block_size * natural_row_bytes:
        return natural_row_bytes
    assert block_stride_bytes % page_block_size == 0, (
        f"kv_cache block stride {block_stride_bytes} not divisible by "
        f"page_block_size {page_block_size}"
    )
    return block_stride_bytes // page_block_size


def sparse_mla_decode_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    bf16_qk: bool = True,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    extra_topk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _C = _load_lib()
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]

    assert q.is_contiguous()
    assert kv_cache.is_contiguous()
    assert indices.dtype == torch.int32 and indices.is_contiguous()

    stride_kv_row = _effective_stride_kv_row(kv_cache)
    page_block_size = kv_cache.shape[-3] if kv_cache.dim() >= 3 else 1

    replicate_h = (num_heads + _HPB - 1) // _HPB
    num_sm_parts = max(_get_num_sms() // (num_tokens * replicate_h), 1)
    total_splits_bound = num_tokens + num_sm_parts

    sched_meta = torch.empty(num_sm_parts * 8, dtype=torch.int32, device=q.device)
    num_splits = torch.empty(num_tokens + 1, dtype=torch.int32, device=q.device)
    _C.get_decode_metadata(
        num_tokens, topk, extra_topk, num_sm_parts, _FIXED_OVERHEAD,
        topk_length, extra_topk_length, sched_meta, num_splits)

    s_q = 1
    o_accum = torch.empty(
        (total_splits_bound, s_q, num_heads, d_v),
        dtype=torch.float32, device=q.device)
    lse_accum = torch.empty(
        (total_splits_bound, s_q, num_heads),
        dtype=torch.float32, device=q.device)
    output = torch.empty(
        (num_tokens, num_heads, d_v),
        dtype=torch.bfloat16, device=q.device)
    out_lse = torch.empty(
        (num_tokens, num_heads),
        dtype=torch.float32, device=q.device)

    _C.sparse_mla_splitkv_v2_fwd(
        q, kv_cache, indices,
        o_accum, lse_accum, output, out_lse,
        sched_meta, num_splits,
        sm_scale, topk, stride_kv_row, page_block_size,
        num_sm_parts, attn_sink,
        extra_k_cache, extra_indices, topk_length,
        extra_topk, extra_topk_length, bf16_qk)

    ni = topk // _BI
    if extra_topk > 0:
        ni += (extra_topk + _BI - 1) // _BI
    max_nsplits = min(ni, num_sm_parts)

    _C.sparse_mla_combine_v2_fwd(
        o_accum, lse_accum, output, out_lse,
        num_splits, num_tokens, max_nsplits, attn_sink)

    return output, out_lse


def sparse_mla_prefill_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    bf16_qk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _C = _load_lib()
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]

    assert q.is_contiguous()
    assert kv_cache.is_contiguous()
    assert indices.dtype == torch.int32 and indices.is_contiguous()

    stride_kv_row = _effective_stride_kv_row(kv_cache)
    page_block_size = kv_cache.shape[-3] if kv_cache.dim() >= 3 else 1

    output = torch.empty(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=q.device
    )
    lse = torch.empty(
        (num_tokens, num_heads), dtype=torch.float32, device=q.device
    )
    max_logits = torch.empty(
        (num_tokens, num_heads), dtype=torch.float32, device=q.device
    )
    _C.sparse_mla_prefill_fwd(
        q, kv_cache, indices, output, lse,
        sm_scale, topk, stride_kv_row, page_block_size,
        attn_sink, topk_length, bf16_qk, max_logits,
    )
    return output, max_logits, lse


