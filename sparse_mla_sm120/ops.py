import torch

def _load_lib():
    import sparse_mla_sm120._C as _C
    return _C

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

    _C.sparse_mla_fwd(q, kv_cache, indices, output, sm_scale, d_v, d_rope, topk)
    return output
