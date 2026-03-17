"""Correctness tests for sparse_mla_sm120 CUDA kernels.

Compares the kernel output against a pure-PyTorch reference.
The kernel operates on FP8 KV cache (packed format), so tolerance
accounts for FP8 quantization error.

Run:
    pytest tests/test_sparse_mla.py -v -s
"""

import itertools

import pytest
import torch


# ── KV cache packing / unpacking ──────────────────────────────────────────

D_NOPE = 512
D_ROPE = 64
DIM = D_NOPE + D_ROPE  # 576
QUANT_TILE = 128
NUM_SCALES = D_NOPE // QUANT_TILE  # 4
KV_PACKED_BYTES = D_NOPE + NUM_SCALES * 4 + D_ROPE * 2  # 656
TOPK = 2048
DEVICE = "cuda"
FP8_MAX = 448.0


def pack_kv_cache_fp8(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Convert bf16 KV cache to packed FP8 format matching sglang layout.

    Input:  kv_bf16 [pool_size, 1, 576] bf16
    Output: packed  [pool_size, 1, 656] uint8
    """
    pool_size = kv_bf16.shape[0]
    kv_f = kv_bf16.squeeze(1).float()  # [pool_size, 576]

    packed = torch.zeros(pool_size, 1, KV_PACKED_BYTES, dtype=torch.uint8, device=kv_bf16.device)

    nope = kv_f[:, :D_NOPE]  # [pool_size, 512]
    rope = kv_bf16.squeeze(1)[:, D_NOPE:]  # [pool_size, 64] bf16

    for b in range(NUM_SCALES):
        tile = nope[:, b * QUANT_TILE : (b + 1) * QUANT_TILE]  # [pool_size, 128]
        amax = tile.abs().amax(dim=1).clamp(min=1e-4)  # [pool_size]
        scale = amax / FP8_MAX  # [pool_size]
        scale_inv = 1.0 / scale

        # Quantize to fp8 e4m3
        fp8_vals = (tile * scale_inv.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX)
        fp8_tensor = fp8_vals.to(torch.float8_e4m3fn)

        # Store fp8 bytes
        byte_offset = b * QUANT_TILE
        packed[:, 0, byte_offset : byte_offset + QUANT_TILE] = fp8_tensor.view(torch.uint8)

        # Store fp32 scale
        scale_bytes = scale.to(torch.float32).view(torch.uint8).reshape(pool_size, 4)
        scale_offset = D_NOPE + b * 4
        packed[:, 0, scale_offset : scale_offset + 4] = scale_bytes

    # Store rope as bf16
    rope_bytes = rope.contiguous().view(torch.uint8).reshape(pool_size, D_ROPE * 2)
    packed[:, 0, D_NOPE + NUM_SCALES * 4 :] = rope_bytes

    return packed


def unpack_kv_cache_fp8(packed: torch.Tensor) -> torch.Tensor:
    """Dequantize packed FP8 KV cache back to bf16 for reference comparison.

    Input:  packed [pool_size, 1, 656] uint8
    Output: kv_bf16 [pool_size, 1, 576] bf16
    """
    pool_size = packed.shape[0]
    result = torch.zeros(pool_size, 1, DIM, dtype=torch.bfloat16, device=packed.device)

    for b in range(NUM_SCALES):
        byte_offset = b * QUANT_TILE
        fp8_raw = packed[:, 0, byte_offset : byte_offset + QUANT_TILE]
        fp8_tensor = fp8_raw.view(torch.float8_e4m3fn).float()  # [pool_size, 128]

        scale_offset = D_NOPE + b * 4
        scale_bytes = packed[:, 0, scale_offset : scale_offset + 4]
        scale = scale_bytes.contiguous().view(torch.float32).squeeze(-1)  # [pool_size]

        dequant = fp8_tensor * scale.unsqueeze(1)
        result[:, 0, b * QUANT_TILE : (b + 1) * QUANT_TILE] = dequant.to(torch.bfloat16)

    # Rope
    rope_bytes = packed[:, 0, D_NOPE + NUM_SCALES * 4 :]
    rope = rope_bytes.contiguous().view(torch.bfloat16).reshape(pool_size, D_ROPE)
    result[:, 0, D_NOPE:] = rope

    return result


# ── Reference implementation ──────────────────────────────────────────────


def sparse_attention_ref(
    q: torch.Tensor,       # [batch, num_heads, dim] bf16
    kv: torch.Tensor,      # [pool_size, 1, dim] bf16 (dequantized)
    indices: torch.Tensor,  # [batch, topk] int32
    sm_scale: float,
    d_v: int,
) -> torch.Tensor:
    """Pure PyTorch reference for sparse MLA attention."""
    batch, num_heads, dim = q.shape

    q_f = q.float()
    kv_f = kv.squeeze(1).float()

    output = torch.zeros(batch, num_heads, d_v, device=q.device, dtype=torch.float32)

    for b in range(batch):
        idx = indices[b]  # [topk]
        valid = idx >= 0
        if not valid.any():
            continue

        valid_idx = idx[valid].long()
        k_sel = kv_f[valid_idx]  # [n_valid, dim]

        scores = torch.einsum("hd,nd->hn", q_f[b], k_sel) * sm_scale
        weights = torch.softmax(scores, dim=-1)

        v_sel = k_sel[:, :d_v]
        output[b] = torch.einsum("hn,nd->hd", weights, v_sel)

    return output.to(torch.bfloat16)


# ── Kernel wrapper ────────────────────────────────────────────────────────


def sm120_sparse_mla(
    q: torch.Tensor,
    kv_packed: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
) -> torch.Tensor:
    """Wrapper for our SM120 CUDA kernel."""
    import sparse_mla_sm120
    return sparse_mla_sm120.sparse_mla_fwd(
        q=q.contiguous(),
        kv_cache=kv_packed.contiguous(),
        indices=indices.contiguous(),
        sm_scale=sm_scale,
        d_v=d_v,
    )


# ── Test configs ──────────────────────────────────────────────────────────

BATCH_SIZES = [1, 4, 8]
HEAD_COUNTS = [16, 128]

# Higher tolerance for FP8 quantization error
FP8_ATOL = 0.1
FP8_RTOL = 0.1


@pytest.mark.parametrize(
    "batch_size,num_heads",
    list(itertools.product(BATCH_SIZES, HEAD_COUNTS)),
)
def test_correctness_all_valid(batch_size: int, num_heads: int) -> None:
    """All indices valid — compare kernel vs reference (via FP8 dequant)."""
    torch.manual_seed(42)
    pool_size = 4096
    sm_scale = DIM ** -0.5

    q = torch.randn(batch_size, num_heads, DIM, device=DEVICE, dtype=torch.bfloat16)
    kv_bf16 = torch.randn(pool_size, 1, DIM, device=DEVICE, dtype=torch.bfloat16)
    indices = torch.randint(0, pool_size, (batch_size, TOPK), device=DEVICE, dtype=torch.int32)

    kv_packed = pack_kv_cache_fp8(kv_bf16)
    kv_dequant = unpack_kv_cache_fp8(kv_packed)

    ref = sparse_attention_ref(q, kv_dequant, indices, sm_scale, D_NOPE)
    out = sm120_sparse_mla(q, kv_packed, indices, sm_scale, D_NOPE)

    print(f"\n  batch={batch_size}, heads={num_heads}")
    print(f"  max_abs_err = {(out.float() - ref.float()).abs().max().item():.4f}")
    print(f"  mean_abs_err = {(out.float() - ref.float()).abs().mean().item():.6f}")

    torch.testing.assert_close(out.float(), ref.float(), atol=FP8_ATOL, rtol=FP8_RTOL)


@pytest.mark.parametrize(
    "batch_size,num_heads",
    list(itertools.product(BATCH_SIZES, HEAD_COUNTS)),
)
def test_correctness_mixed_invalid(batch_size: int, num_heads: int) -> None:
    """Mix of valid and -1 indices."""
    torch.manual_seed(123)
    pool_size = 4096
    sm_scale = DIM ** -0.5
    n_valid = 128

    q = torch.randn(batch_size, num_heads, DIM, device=DEVICE, dtype=torch.bfloat16)
    kv_bf16 = torch.randn(pool_size, 1, DIM, device=DEVICE, dtype=torch.bfloat16)

    indices = torch.full((batch_size, TOPK), -1, device=DEVICE, dtype=torch.int32)
    indices[:, :n_valid] = torch.randint(0, pool_size, (batch_size, n_valid), device=DEVICE, dtype=torch.int32)

    kv_packed = pack_kv_cache_fp8(kv_bf16)
    kv_dequant = unpack_kv_cache_fp8(kv_packed)

    ref = sparse_attention_ref(q, kv_dequant, indices, sm_scale, D_NOPE)
    out = sm120_sparse_mla(q, kv_packed, indices, sm_scale, D_NOPE)

    print(f"\n  batch={batch_size}, heads={num_heads}, valid={n_valid}/{TOPK}")
    print(f"  max_abs_err = {(out.float() - ref.float()).abs().max().item():.4f}")

    torch.testing.assert_close(out.float(), ref.float(), atol=FP8_ATOL, rtol=FP8_RTOL)


@pytest.mark.parametrize("num_heads", HEAD_COUNTS)
def test_no_nan_inf(num_heads: int) -> None:
    """Output must be finite even with mostly-invalid indices."""
    torch.manual_seed(7)
    pool_size = 4096
    sm_scale = DIM ** -0.5
    batch_size = 4

    q = torch.randn(batch_size, num_heads, DIM, device=DEVICE, dtype=torch.bfloat16)
    kv_bf16 = torch.randn(pool_size, 1, DIM, device=DEVICE, dtype=torch.bfloat16)

    indices = torch.full((batch_size, TOPK), -1, device=DEVICE, dtype=torch.int32)
    indices[:, :32] = torch.randint(0, pool_size, (batch_size, 32), device=DEVICE, dtype=torch.int32)

    kv_packed = pack_kv_cache_fp8(kv_bf16)

    out = sm120_sparse_mla(q, kv_packed, indices, sm_scale, D_NOPE)
    assert not torch.isnan(out).any(), "output contains NaN"
    assert not torch.isinf(out).any(), "output contains Inf"


@pytest.mark.parametrize("num_heads", HEAD_COUNTS)
def test_prefill_path(num_heads: int) -> None:
    """Test prefill path (num_tokens > 64)."""
    torch.manual_seed(99)
    pool_size = 4096
    sm_scale = DIM ** -0.5
    batch_size = 128  # triggers prefill path

    q = torch.randn(batch_size, num_heads, DIM, device=DEVICE, dtype=torch.bfloat16)
    kv_bf16 = torch.randn(pool_size, 1, DIM, device=DEVICE, dtype=torch.bfloat16)
    indices = torch.randint(0, pool_size, (batch_size, TOPK), device=DEVICE, dtype=torch.int32)

    kv_packed = pack_kv_cache_fp8(kv_bf16)
    kv_dequant = unpack_kv_cache_fp8(kv_packed)

    ref = sparse_attention_ref(q, kv_dequant, indices, sm_scale, D_NOPE)
    out = sm120_sparse_mla(q, kv_packed, indices, sm_scale, D_NOPE)

    print(f"\n  batch={batch_size}, heads={num_heads} (prefill)")
    print(f"  max_abs_err = {(out.float() - ref.float()).abs().max().item():.4f}")

    torch.testing.assert_close(out.float(), ref.float(), atol=FP8_ATOL, rtol=FP8_RTOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
