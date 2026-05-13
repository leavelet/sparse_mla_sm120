import torch
from typing import Optional, Tuple


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
    out: Optional[torch.Tensor] = None,
    # Dual-cache extras (API-skin; currently no-op in kernel).
    # Plumbed to the C++ binding which emits TORCH_WARN_ONCE if any are set,
    # making it visible that the second-cache contribution is being dropped.
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode forward: splitkv + combine.

    If `out` is provided, the combine kernel writes directly into it
    (no extra allocation/copy). `out` must be a contiguous bfloat16 tensor
    with shape (num_tokens, num_heads, d_v).

    Returns:
        output: [num_tokens, num_heads, d_v] bfloat16 (same storage as `out` if provided)
        lse: [num_tokens, num_heads] float32
    """
    _C = _load_lib()
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]

    assert q.is_contiguous()
    # kv_cache is paged and may be unsqueezed/padded by the caller; the kernel
    # consumes stride_kv_row + page_block_size, so we only require the innermost
    # head-bytes dim to be contiguous (stride == 1).
    assert kv_cache.stride(-1) == 1, (
        f"kv_cache innermost dim must be contiguous; got strides {kv_cache.stride()}"
    )
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
        KV_cache_extra=extra_kv_cache,
        indices_extra=extra_indices,
        topk_length=topk_length,
        topk_length_extra=extra_topk_length,
        attn_sink=attn_sink,
    )

    # Combine — write directly into caller-provided `out` when available.
    if out is None:
        output = torch.empty(
            (num_tokens, num_heads, d_v),
            dtype=torch.bfloat16, device=q.device,
        )
    else:
        assert out.shape == (num_tokens, num_heads, d_v), (
            f"out shape {tuple(out.shape)} != expected "
            f"{(num_tokens, num_heads, d_v)}"
        )
        assert out.dtype == torch.bfloat16
        assert out.is_contiguous()
        output = out
    lse = torch.empty(
        (num_tokens, num_heads),
        dtype=torch.float32, device=q.device,
    )
    # attn_sink scaling lives in the combine kernel: output *= sigmoid(lse - sink)
    # per head. Caller convention from upstream FlashMLA — attn_sink is a 1-D
    # float32 tensor of length num_heads (padded heads carry -inf so factor==1).
    _C.sparse_mla_combine_fwd(partial_O, partial_LSE, output, lse, nsplits,
                              attn_sink=attn_sink)

    return output, lse


def sparse_mla_prefill_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    out: Optional[torch.Tensor] = None,
    # Dual-cache extras (API-skin; currently no-op in kernel).
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _C = _load_lib()
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]

    assert q.is_contiguous()
    # Same relaxation as decode: kernel takes explicit stride_kv_row, only
    # requires innermost head-bytes dim contiguous.
    assert kv_cache.stride(-1) == 1, (
        f"kv_cache innermost dim must be contiguous; got strides {kv_cache.stride()}"
    )
    assert indices.dtype == torch.int32 and indices.is_contiguous()

    stride_kv_row = kv_cache.stride(-2) * kv_cache.element_size()
    page_block_size = kv_cache.shape[-3] if kv_cache.dim() >= 3 else 1

    if out is None:
        output = torch.empty(
            (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=q.device
        )
    else:
        assert out.shape == (num_tokens, num_heads, d_v), (
            f"out shape {tuple(out.shape)} != expected "
            f"{(num_tokens, num_heads, d_v)}"
        )
        assert out.dtype == torch.bfloat16
        assert out.is_contiguous()
        output = out
    lse = torch.empty(
        (num_tokens, num_heads), dtype=torch.float32, device=q.device
    )
    _C.sparse_mla_prefill_fwd(
        q, kv_cache, indices, output, lse,
        sm_scale, topk, stride_kv_row, page_block_size,
        KV_cache_extra=extra_kv_cache,
        indices_extra=extra_indices,
        topk_length=topk_length,
        topk_length_extra=extra_topk_length,
        attn_sink=attn_sink,
    )
    return output, lse


def sparse_mla_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    out: Optional[torch.Tensor] = None,
    # Dual-cache extras (API-skin; currently no-op in kernel).
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
):
    kwargs = dict(
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        attn_sink=attn_sink,
    )
    num_tokens = q.shape[0]
    if num_tokens <= _DECODE_THRESHOLD:
        return sparse_mla_decode_fwd(
            q, kv_cache, indices, sm_scale, d_v, out=out, **kwargs
        )
    return sparse_mla_prefill_fwd(
        q, kv_cache, indices, sm_scale, d_v, out=out, **kwargs
    )
