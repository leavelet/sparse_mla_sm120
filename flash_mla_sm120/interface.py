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
    from .ops import sparse_mla_decode_fwd

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

    if not sched_meta.have_initialized:
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            b=q.shape[0],
            s_q=q.shape[1] if q.dim() == 4 else 1,
            h_q=q.shape[-2],
            page_block_size=k_cache.shape[1],
            h_k=k_cache.shape[2],
            causal=causal,
            is_fp8_kvcache=is_fp8_kvcache,
            topk=topk,
            extra_page_block_size=extra_k_page_block_size,
            extra_topk=extra_topk if extra_topk > 0 else None,
        )
    else:
        c = sched_meta.config
        assert c.b == q.shape[0]
        assert c.s_q == (q.shape[1] if q.dim() == 4 else 1)
        assert c.h_q == q.shape[-2]
        assert c.page_block_size == k_cache.shape[1]
        assert c.h_k == k_cache.shape[2]
        assert c.causal == causal
        assert c.is_fp8_kvcache == is_fp8_kvcache
        assert c.topk == topk
        assert c.extra_page_block_size == extra_k_page_block_size
        assert c.extra_topk == (extra_topk if extra_topk > 0 else None)

    q_input = q.reshape(-1, q.shape[-2], q.shape[-1])
    idx_input = indices.reshape(q_input.shape[0], -1)

    extra_idx_input = None
    if extra_indices_in_kvcache is not None:
        extra_idx_input = extra_indices_in_kvcache.reshape(q_input.shape[0], -1)

    output, real_lse = sparse_mla_decode_fwd(
        q_input, k_cache, idx_input, softmax_scale, head_dim_v,
        attn_sink=attn_sink, bf16_qk=bf16_qk,
        extra_k_cache=extra_k_cache, extra_indices=extra_idx_input,
        topk_length=topk_length, extra_topk_length=extra_topk_length,
        extra_topk=extra_topk,
    )

    batch = q.shape[0]
    s_q = q.shape[1] if q.dim() == 4 else 1
    h_q = q.shape[-2]
    output = output.reshape(batch, s_q, h_q, head_dim_v)
    lse = real_lse.reshape(batch, s_q, h_q).permute(0, 2, 1).contiguous()

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
