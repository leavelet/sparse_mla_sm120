import torch
from typing import Optional, Tuple


def _load_lib():
    import flash_mla_sm120.cuda as _C
    return _C


_DECODE_THRESHOLD = 64


def _effective_stride_kv_row(kv_cache: torch.Tensor) -> int:
    """Return the per-token byte stride the kernel should use.

    The kernel computes ``stride_kv_block = page_block_size * stride_kv_row``
    to advance from one paged block to the next. Some callers (e.g. vLLM)
    pad the *block* stride for alignment so the natural per-token stride
    times block_size doesn't equal the actual block-to-block stride. The
    only knob the binding exposes is ``stride_kv_row``, so override it so
    ``page_block_size * stride_kv_row == actual_block_byte_stride`` — the
    kernel uses ``IO::IO_STRIDE`` (not stride_kv_row) for per-token offsets
    *within* a block, so this remains correct.

    For a paged tensor shaped ``(num_blocks, block_size, head_kv, bytes)``
    or ``(num_blocks, block_size, bytes)`` the block stride is ``stride[0]``
    and the per-token stride is ``stride[1]``.
    """
    if kv_cache.dim() < 3:
        return kv_cache.stride(-2) * kv_cache.element_size()
    page_block_size = kv_cache.shape[-3]
    natural_row_bytes = kv_cache.stride(-2) * kv_cache.element_size()
    # Outermost dim is num_blocks; its stride is the block byte stride.
    block_stride_bytes = kv_cache.stride(0) * kv_cache.element_size()
    if block_stride_bytes == page_block_size * natural_row_bytes:
        return natural_row_bytes
    assert block_stride_bytes % page_block_size == 0, (
        f"kv_cache block stride {block_stride_bytes} not divisible by "
        f"page_block_size {page_block_size}; cannot encode padding via "
        f"stride_kv_row override"
    )
    return block_stride_bytes // page_block_size


def _compute_decode_splits(num_tokens, num_heads, topk):
    """v1 decode split-KV heuristic (used by the legacy v1 path).
    For v2 (scheduler-driven), splits are computed on-device by
    get_decode_metadata and this function is not called.
    """
    hpb = 16
    bi = 64
    ni = topk // bi

    replicate_h = (num_heads + hpb - 1) // hpb
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
    attn_sink: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    # Dual-cache extras + V4 features. When extras are present, routes through
    # the dual-cache v1 launcher (kernel-side dual-cache + attn_sink). topk_length
    # is not yet wired into v1 decode — callers needing length-aware decoding
    # should use the v2 scheduler-driven path (sparse_mla_decode_v2_fwd) instead.
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode forward (v1): splitkv + combine.

    If `out` is provided, the combine kernel writes directly into it
    (no extra allocation/copy). `out` must be a contiguous bfloat16 tensor
    with shape (num_tokens, num_heads, d_v).

    Returns:
        output: [num_tokens, num_heads, d_v] bfloat16 (same storage as `out` if provided)
        lse: [num_tokens, num_heads] float32  (FlashMLA convention: includes attn_sink merge)
    """
    _C = _load_lib()
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]

    assert q.is_contiguous()
    assert kv_cache.stride(-1) == 1, (
        f"kv_cache innermost dim must be contiguous; got strides {kv_cache.stride()}"
    )
    assert indices.dtype == torch.int32 and indices.is_contiguous()
    assert num_tokens <= _DECODE_THRESHOLD

    stride_kv_row = _effective_stride_kv_row(kv_cache)
    page_block_size = kv_cache.shape[-3] if kv_cache.dim() >= 3 else 1
    nsplits, tiles_per_split = _compute_decode_splits(num_tokens, num_heads, topk)

    partial_O = torch.empty(
        (num_tokens, nsplits, num_heads, d_v),
        dtype=torch.float32, device=q.device,
    )
    partial_LSE = torch.empty(
        (num_tokens, nsplits, num_heads),
        dtype=torch.float32, device=q.device,
    )

    _C.sparse_mla_splitkv_fwd(
        q, kv_cache, indices, partial_O, partial_LSE,
        sm_scale, topk, tiles_per_split, stride_kv_row, page_block_size,
        KV_cache_extra=extra_kv_cache,
        indices_extra=extra_indices,
        topk_length=topk_length,
        topk_length_extra=extra_topk_length,
        attn_sink=attn_sink,
    )

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
    # per head, AND lse' = log(exp(lse) + exp(sink)) (FlashMLA V4 convention).
    _C.sparse_mla_combine_fwd(partial_O, partial_LSE, output, lse, nsplits,
                              attn_sink=attn_sink)

    return output, lse


_HPB = 16
_FIXED_OVERHEAD = 64


def _compute_num_sm_parts(num_tokens: int, num_heads: int, s_q: int,
                           device: torch.device) -> int:
    """FlashMLA partitioning heuristic: num_SMs // (s_q * replicate_h)."""
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    replicate_h = (num_heads + _HPB - 1) // _HPB
    return max(num_sms // (s_q * replicate_h), 1)


def sparse_mla_decode_v2_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    num_sm_parts: Optional[int] = None,
    fixed_overhead: int = _FIXED_OVERHEAD,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode v2 (scheduler-driven).

    Uses the GPU scheduler to assign per-batch splits dynamically, then runs
    decode_v2 (with direct-bf16 epilogue for is_no_split batches) and combine_v2
    for batches that did split. Both paths apply attn_sink uniformly.

    Returns:
        output: [num_tokens, num_heads, d_v] bfloat16
        lse:    [num_tokens, num_heads]      float32 (includes attn_sink merge)
    """
    _C = _load_lib()
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]
    extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0

    assert q.is_contiguous()
    assert kv_cache.stride(-1) == 1
    assert indices.dtype == torch.int32 and indices.is_contiguous()

    stride_kv_row = _effective_stride_kv_row(kv_cache)
    page_block_size = kv_cache.shape[-3] if kv_cache.dim() >= 3 else 1

    s_q = 1
    if num_sm_parts is None:
        num_sm_parts = _compute_num_sm_parts(num_tokens, num_heads, s_q, q.device)

    # Scheduler metadata. Flat int32 buffer reinterpreted as DecodingSchedMeta
    # (8 int32 fields per entry) by the kernel.
    sched_meta = torch.empty((num_sm_parts * 8,), dtype=torch.int32, device=q.device)
    num_splits = torch.empty((num_tokens + 1,), dtype=torch.int32, device=q.device)
    _C.get_decode_metadata(
        num_tokens, topk, extra_topk,
        num_sm_parts, fixed_overhead,
        topk_length, extra_topk_length,
        sched_meta, num_splits,
    )

    # FlashMLA upper bound on total per-batch splits across all batches.
    total_splits_bound = num_tokens + num_sm_parts
    o_accum = torch.empty(
        (total_splits_bound, s_q, num_heads, d_v),
        dtype=torch.float32, device=q.device,
    )
    lse_accum = torch.empty(
        (total_splits_bound, s_q, num_heads),
        dtype=torch.float32, device=q.device,
    )

    if out is None:
        output = torch.zeros(
            (num_tokens, num_heads, d_v),
            dtype=torch.bfloat16, device=q.device,
        )
    else:
        assert out.shape == (num_tokens, num_heads, d_v)
        assert out.dtype == torch.bfloat16 and out.is_contiguous()
        output = out
    out_lse = torch.zeros(
        (num_tokens, num_heads),
        dtype=torch.float32, device=q.device,
    )

    _C.sparse_mla_splitkv_v2_fwd(
        q, kv_cache, indices,
        o_accum, lse_accum,
        output, out_lse,
        sched_meta, num_splits,
        sm_scale, topk,
        stride_kv_row, page_block_size,
        num_sm_parts,
        attn_sink=attn_sink,
        extra_k_cache=extra_kv_cache,
        extra_indices=extra_indices,
        topk_length=topk_length,
        extra_topk=extra_topk,
        extra_topk_length=extra_topk_length,
    )

    # combine_v2 — always launched (CUDA-graph friendly). Per-batch early-exits
    # for is_no_split batches (those wrote bf16 directly in decode_v2).
    # max_nsplits is a static upper bound (num_sm_parts) so we can select the
    # right MAX_SPLITS dispatch template without GPU→CPU sync — required for
    # CUDA graph capture. Per-batch nsplits cannot exceed num_sm_parts.
    _C.sparse_mla_combine_v2_fwd(
        o_accum, lse_accum, output, out_lse, num_splits,
        num_tokens, num_sm_parts,
        attn_sink=attn_sink,
    )

    return output, out_lse


def sparse_mla_prefill_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _C = _load_lib()
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]

    assert q.is_contiguous()
    assert kv_cache.stride(-1) == 1, (
        f"kv_cache innermost dim must be contiguous; got strides {kv_cache.stride()}"
    )
    assert indices.dtype == torch.int32 and indices.is_contiguous()

    stride_kv_row = _effective_stride_kv_row(kv_cache)
    page_block_size = kv_cache.shape[-3] if kv_cache.dim() >= 3 else 1

    if out is None:
        output = torch.empty(
            (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=q.device
        )
    else:
        assert out.shape == (num_tokens, num_heads, d_v)
        assert out.dtype == torch.bfloat16 and out.is_contiguous()
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
        # Route through v2 when topk_length / extra_topk_length is set so the
        # kernel-side QK mask is honored. v1 decode ignores topk_length (callers
        # would have to pre-pad indices with -1, which vLLM does not do).
        if topk_length is not None or extra_topk_length is not None:
            return sparse_mla_decode_v2_fwd(
                q, kv_cache, indices, sm_scale, d_v, out=out, **kwargs
            )
        return sparse_mla_decode_fwd(
            q, kv_cache, indices, sm_scale, d_v, out=out, **kwargs
        )
    return sparse_mla_prefill_fwd(
        q, kv_cache, indices, sm_scale, d_v, out=out, **kwargs
    )
