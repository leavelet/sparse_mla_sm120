import torch
from typing import Tuple


def _load_lib():
    import flash_mla_sm120.cuda as _C
    return _C


_DECODE_THRESHOLD = 64


def _compute_decode_splits(num_tokens, num_heads, topk):
    hpb = 16
    bi = 64
    ni = topk // bi

    replicate_h = num_heads // hpb
    ctas_per_split = num_tokens * replicate_h

    target_total_ctas = 128
    nsplits = min(ni, max(1, (target_total_ctas + ctas_per_split - 1) // ctas_per_split))
    min_tiles = 2
    nsplits = min(nsplits, ni // min_tiles)
    nsplits = max(nsplits, 1)
    while nsplits > 1 and ni % nsplits != 0:
        nsplits -= 1
    tiles_per_split = ni // nsplits
    return nsplits, tiles_per_split


def sparse_mla_decode_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode forward: splitkv + combine.

    Returns:
        output: [num_tokens, num_heads, d_v] bfloat16
        lse: [num_tokens, num_heads] float32
    """
    _C = _load_lib()
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]

    assert q.is_contiguous()
    assert kv_cache.is_contiguous()
    assert indices.dtype == torch.int32 and indices.is_contiguous()
    assert num_tokens <= _DECODE_THRESHOLD

    stride_kv_row = kv_cache.stride(-2) * kv_cache.element_size()
    page_block_size = kv_cache.shape[-3] if kv_cache.dim() >= 3 else 1
    nsplits, tiles_per_split = _compute_decode_splits(num_tokens, num_heads, topk)

    # Allocate workspace (float32 for combine precision)
    partial_O = torch.empty(
        (num_tokens, nsplits, num_heads, d_v),
        dtype=torch.float32, device=q.device,
    )
    partial_LSE = torch.empty(
        (num_tokens, nsplits, num_heads),
        dtype=torch.float32, device=q.device,
    )

    # Split-KV decode
    _C.sparse_mla_splitkv_fwd(
        q, kv_cache, indices, partial_O, partial_LSE,
        sm_scale, topk, tiles_per_split, stride_kv_row, page_block_size,
    )

    # Combine
    output = torch.empty(
        (num_tokens, num_heads, d_v),
        dtype=torch.bfloat16, device=q.device,
    )
    lse = torch.empty(
        (num_tokens, num_heads),
        dtype=torch.float32, device=q.device,
    )
    _C.sparse_mla_combine_fwd(partial_O, partial_LSE, output, lse, nsplits)

    return output, lse


def sparse_mla_prefill_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    _C = _load_lib()
    num_tokens, num_heads, d_qk = q.shape
    d_rope = 64
    topk = indices.shape[-1]

    assert q.is_contiguous()
    assert kv_cache.is_contiguous()
    assert indices.dtype == torch.int32 and indices.is_contiguous()

    output = torch.empty(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=q.device
    )
    _C.sparse_mla_prefill_fwd(
        q, kv_cache, indices, output,
        sm_scale, d_v, d_rope, topk,
    )
    return output


def sparse_mla_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
):
    num_tokens = q.shape[0]
    if num_tokens <= _DECODE_THRESHOLD:
        return sparse_mla_decode_fwd(q, kv_cache, indices, sm_scale, d_v)
    output = sparse_mla_prefill_fwd(q, kv_cache, indices, sm_scale, d_v)
    return output, None  # prefill doesn't return LSE yet
