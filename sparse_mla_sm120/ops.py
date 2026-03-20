import torch

def _load_lib():
    import sparse_mla_sm120._C as _C
    return _C

_decode_workspace = {}

def _get_decode_workspace(num_heads, nsplits, device):
    """Pre-allocate and cache workspace tensors for decode split-KV."""
    key = (num_heads, nsplits, device)
    ws = _decode_workspace.get(key)
    max_tokens = 64
    d_v = 512
    hpb = 16
    if ws is None:
        ws = {
            'partial_O': torch.empty(
                (max_tokens, nsplits, num_heads, d_v),
                dtype=torch.bfloat16, device=device),
            'partial_LSE': torch.empty(
                (max_tokens, nsplits, num_heads),
                dtype=torch.float32, device=device),
            'semaphores': torch.zeros(
                (max_tokens * (num_heads // hpb),),
                dtype=torch.int32, device=device),
        }
        _decode_workspace[key] = ws
    return ws


def _compute_decode_splits(num_tokens, num_heads):
    """Compute NSPLITS and TILES_PER_SPLIT for decode.

    Decode is latency/workspace bound more than occupancy bound. Empirically,
    the best split choices on SM120 are very stable for the main DeepSeek
    decode shapes, so prefer a small measured policy table over a single
    occupancy-style heuristic.
    """
    hpb = 16
    bi = 64
    topk = 2048
    ni = topk // bi  # 32

    # Measured on RTX PRO 6000 Blackwell Server Edition.
    if num_heads == 16:
        return 16, 2
    if num_heads == 128:
        if num_tokens <= 1:
            return 16, 2
        if num_tokens <= 2:
            return 8, 4
        if num_tokens <= 5:
            return 4, 8
        if num_tokens <= 8:
            return 2, 16

    replicate_h = num_heads // hpb
    ctas_per_split = num_tokens * replicate_h

    # Fallback for less common head counts / batch sizes.
    target_total_ctas = 128
    nsplits = min(ni, max(1, (target_total_ctas + ctas_per_split - 1) // ctas_per_split))
    min_tiles = 2
    nsplits = min(nsplits, ni // min_tiles)
    nsplits = max(nsplits, 1)
    while nsplits > 1 and ni % nsplits != 0:
        nsplits -= 1
    tiles_per_split = ni // nsplits
    return nsplits, tiles_per_split


def sparse_mla_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Sparse MLA forward pass on SM120 with native FP8 KV cache.

    Args:
        q: [num_tokens, num_heads, dim] bf16, dim = d_v + d_rope (576)
        kv_cache: [pool_size, 1, kv_dim] uint8, packed FP8 nope + FP32 scales + BF16 rope (656)
        indices: [num_tokens, topk] int32, -1 = invalid
        sm_scale: softmax scale factor
        d_v: value head dimension (512)

    Returns:
        output: [num_tokens, num_heads, d_v] bf16
    """
    _C = _load_lib()

    assert q.dtype == torch.bfloat16, f"q must be bf16, got {q.dtype}"
    assert q.is_contiguous()
    assert kv_cache.is_contiguous()
    assert indices.dtype == torch.int32

    num_tokens, num_heads, dim = q.shape
    d_rope = dim - d_v
    topk = indices.shape[-1]

    output = torch.empty(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=q.device
    )

    decode_threshold = 64
    if num_tokens <= decode_threshold:
        nsplits, tiles_per_split = _compute_decode_splits(num_tokens, num_heads)
        ws = _get_decode_workspace(num_heads, nsplits, q.device)
        _C.sparse_mla_decode_fwd(
            q, kv_cache, indices, output,
            ws['partial_O'], ws['partial_LSE'], ws['semaphores'],
            sm_scale, d_v, d_rope, topk,
            tiles_per_split, nsplits)
    else:
        _C.sparse_mla_prefill_fwd(
            q, kv_cache, indices, output,
            sm_scale, d_v, d_rope, topk)

    return output
