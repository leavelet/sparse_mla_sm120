"""FlashMLA-compatible Python interface for flash_mla_sm120.

API signatures match flash_mla_interface.py so that sglang/vllm
can use this as a drop-in SM120 backend.
"""

from typing import Optional, Tuple
import dataclasses
import torch


@dataclasses.dataclass
class FlashMLASchedMeta:
    @dataclasses.dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int
        causal: bool
        is_fp8_kvcache: bool
        topk: Optional[int]
        extra_page_block_size: Optional[int]
        extra_topk: Optional[int]

    have_initialized: bool = False
    config: Optional[Config] = None
    tile_scheduler_metadata: Optional[torch.Tensor] = None
    num_splits: Optional[torch.Tensor] = None
    num_sm_parts: int = 0
    max_nsplits: int = 0


def get_mla_metadata(*args, **kwargs) -> Tuple[FlashMLASchedMeta, None]:
    return FlashMLASchedMeta(), None


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    bf16_qk: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from .ops import _load_lib, _effective_stride_kv_row, _get_num_sms, _HPB, _FIXED_OVERHEAD, _BI

    _C = _load_lib()
    sched_meta = tile_scheduler_metadata
    assert isinstance(sched_meta, FlashMLASchedMeta)
    assert num_splits is None

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    assert indices is not None, "flash_mla_sm120 only supports sparse attention"
    assert is_fp8_kvcache, "flash_mla_sm120 requires FP8 KV cache"

    topk = indices.shape[-1]
    extra_topk = 0
    extra_k_page_block_size = None
    if extra_indices_in_kvcache is not None:
        extra_topk = extra_indices_in_kvcache.shape[-1]
    if extra_k_cache is not None:
        extra_k_page_block_size = extra_k_cache.shape[1]

    q_input = q.reshape(-1, q.shape[-2], q.shape[-1])
    num_tokens, num_heads, d_qk = q_input.shape
    idx_input = indices.reshape(num_tokens, -1)

    extra_idx_input = None
    if extra_indices_in_kvcache is not None:
        extra_idx_input = extra_indices_in_kvcache.reshape(num_tokens, -1)

    stride_kv_row = _effective_stride_kv_row(k_cache)
    page_block_size = k_cache.shape[-3] if k_cache.dim() >= 3 else 1

    if not sched_meta.have_initialized:
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            b=q.shape[0],
            s_q=q.shape[1] if q.dim() == 4 else 1,
            h_q=num_heads,
            page_block_size=k_cache.shape[1],
            h_k=k_cache.shape[2],
            causal=causal,
            is_fp8_kvcache=is_fp8_kvcache,
            topk=topk,
            extra_page_block_size=extra_k_page_block_size,
            extra_topk=extra_topk if extra_topk > 0 else None,
        )
        replicate_h = (num_heads + _HPB - 1) // _HPB
        sched_meta.num_sm_parts = max(_get_num_sms() // (num_tokens * replicate_h), 1)
        ni = topk // _BI
        if extra_topk > 0:
            ni += (extra_topk + _BI - 1) // _BI
        sched_meta.max_nsplits = min(ni, sched_meta.num_sm_parts)

        sched_meta.tile_scheduler_metadata = torch.empty(
            sched_meta.num_sm_parts * 8, dtype=torch.int32, device=q.device)
        sched_meta.num_splits = torch.empty(
            num_tokens + 1, dtype=torch.int32, device=q.device)
        _C.get_decode_metadata(
            num_tokens, topk, extra_topk, sched_meta.num_sm_parts, _FIXED_OVERHEAD,
            topk_length, extra_topk_length,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits)
    else:
        c = sched_meta.config
        assert c.b == q.shape[0]
        assert c.s_q == (q.shape[1] if q.dim() == 4 else 1)
        assert c.h_q == num_heads
        assert c.page_block_size == k_cache.shape[1]
        assert c.h_k == k_cache.shape[2]
        assert c.causal == causal
        assert c.is_fp8_kvcache == is_fp8_kvcache
        assert c.topk == topk
        assert c.extra_page_block_size == extra_k_page_block_size
        assert c.extra_topk == (extra_topk if extra_topk > 0 else None)

    num_sm_parts = sched_meta.num_sm_parts
    total_splits_bound = num_tokens + num_sm_parts
    s_q = 1

    o_accum = torch.empty(
        (total_splits_bound, s_q, num_heads, head_dim_v),
        dtype=torch.float32, device=q.device)
    lse_accum = torch.empty(
        (total_splits_bound, s_q, num_heads),
        dtype=torch.float32, device=q.device)
    output = torch.empty(
        (num_tokens, num_heads, head_dim_v),
        dtype=torch.bfloat16, device=q.device)
    out_lse = torch.empty(
        (num_tokens, num_heads),
        dtype=torch.float32, device=q.device)

    _C.sparse_mla_splitkv_v2_fwd(
        q_input, k_cache, idx_input,
        o_accum, lse_accum, output, out_lse,
        sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
        softmax_scale, topk, stride_kv_row, page_block_size,
        num_sm_parts, attn_sink,
        extra_k_cache, extra_idx_input, topk_length,
        extra_topk, extra_topk_length, bf16_qk)

    _C.sparse_mla_combine_v2_fwd(
        o_accum, lse_accum, output, out_lse,
        sched_meta.num_splits, num_tokens, sched_meta.max_nsplits, attn_sink)

    batch = q.shape[0]
    s_q_dim = q.shape[1] if q.dim() == 4 else 1
    output = output.reshape(batch, s_q_dim, num_heads, head_dim_v)
    lse = out_lse.reshape(batch, s_q_dim, num_heads).permute(0, 2, 1).contiguous()

    return output, lse


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .ops import sparse_mla_prefill_fwd

    idx = indices.squeeze(1) if indices.dim() == 3 else indices
    out, max_logits, lse = sparse_mla_prefill_fwd(
        q, kv, idx, sm_scale, d_v,
        attn_sink=attn_sink, topk_length=topk_length)

    return out, max_logits, lse
