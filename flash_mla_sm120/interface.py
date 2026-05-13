"""FlashMLA-compatible Python interface for flash_mla_sm120.

API signatures match flash_mla_interface.py exactly so that sglang/vllm
can use this as a drop-in SM120 backend.

Precision behavior (matching FlashMLA):
  - prefill (flash_mla_sparse_fwd): BF16 compute by default
  - decode  (flash_mla_with_kvcache): configurable (FP8 for perf, BF16 for precision)
FlashMLA on SM90 uses BF16 MMA for all paths (dequant FP8→BF16 first).
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
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from .ops import (_DECODE_THRESHOLD, sparse_mla_decode_fwd,
                        sparse_mla_decode_v2_fwd, sparse_mla_prefill_fwd)

    sched_meta = tile_scheduler_metadata
    assert isinstance(sched_meta, FlashMLASchedMeta)
    assert num_splits is None

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    assert indices is not None, "flash_mla_sm120 only supports sparse attention"
    assert is_fp8_kvcache, "flash_mla_sm120 requires FP8 KV cache"

    q_input = q.reshape(-1, q.shape[-2], q.shape[-1])  # [b*s_q, h_q, d_qk]
    idx_input = indices.reshape(q_input.shape[0], -1)   # [b*s_q, topk]

    # If caller provided an output buffer, pass it through as a view that
    # matches the underlying kernel's (num_tokens, num_heads, d_v) layout
    # so the combine/prefill kernel writes directly into it (no copy).
    out_view = None
    if out is not None:
        out_view = out.view(q_input.shape[0], q_input.shape[1], head_dim_v)

    extra_idx_input = None
    if extra_indices_in_kvcache is not None:
        extra_idx_input = extra_indices_in_kvcache.reshape(q_input.shape[0], -1)

    # Dispatch:
    # - num_tokens <= _DECODE_THRESHOLD: scheduler-driven v2 decode + v2 combine
    #   (is_no_split fast path writes bf16 directly; combine_v2 is CUDA-graph
    #   friendly and per-batch early-exits for unsplit rows).
    # - num_tokens > _DECODE_THRESHOLD: prefill kernel (single-pass, direct bf16).
    num_tokens = q_input.shape[0]
    if num_tokens <= _DECODE_THRESHOLD:
        output, real_lse = sparse_mla_decode_v2_fwd(
            q_input, k_cache, idx_input, softmax_scale, head_dim_v,
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=out_view,
            extra_kv_cache=extra_k_cache,
            extra_indices=extra_idx_input,
            extra_topk_length=extra_topk_length,
        )
    else:
        output, real_lse = sparse_mla_prefill_fwd(
            q_input, k_cache, idx_input, softmax_scale, head_dim_v,
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=out_view,
            extra_kv_cache=extra_k_cache,
            extra_indices=extra_idx_input,
            extra_topk_length=extra_topk_length,
        )

    batch = q.shape[0]
    s_q = q.shape[1] if q.dim() == 4 else 1
    h_q = q.shape[-2]
    output = output.reshape(batch, s_q, h_q, head_dim_v)
    # real_lse is (b*s_q, h_q) → reshape to (b, s_q, h_q) then permute to (b, h_q, s_q)
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
    out: Optional[torch.Tensor] = None,
    # Dual-cache extras (API-skin; currently no-op in kernel).
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .ops import sparse_mla_fwd

    extra_idx = None
    if extra_indices_in_kvcache is not None:
        # Match the squeeze-on-dim-3 convention applied to `indices`.
        extra_idx = (extra_indices_in_kvcache.squeeze(1)
                     if extra_indices_in_kvcache.dim() == 3
                     else extra_indices_in_kvcache)

    # Caller may pass `out` with shape (num_tokens, h_q, d_v) — the kernel's
    # native output layout — so forward it directly with no view gymnastics.
    result = sparse_mla_fwd(
        q, kv, indices.squeeze(1) if indices.dim() == 3 else indices,
        sm_scale, d_v, out=out,
        extra_kv_cache=extra_k_cache,
        extra_indices=extra_idx,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        attn_sink=attn_sink,
    )

    if isinstance(result, tuple):
        output, lse = result
    else:
        output, lse = result, None

    if lse is None:
        lse = torch.zeros(q.shape[0], q.shape[1], dtype=torch.float32, device=q.device)
    max_logits = torch.zeros(q.shape[0], q.shape[1], dtype=torch.float32, device=q.device)

    return output, max_logits, lse
