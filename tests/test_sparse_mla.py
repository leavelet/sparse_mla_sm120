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
    """Convert bf16 KV cache (V32 layout) to packed FP8 format using UE8M0 scales.

    UE8M0 (power-of-2-only) is required because the kernel reinterprets the
    4-byte scale field as a float32 with the implicit exponent-only invariant
    inherited from FlashMLA upstream — arbitrary float32 scales produce wrong
    results.

    Input:  kv_bf16 [pool_size, 1, 576] bf16
    Output: packed  [pool_size, 1, 1, 656] uint8 — adds the explicit
                    page_block_size=1 axis the kernel infers from kv.shape[-3].
    """
    from tests.test_decode import quantize_kv_v32
    # quantize_kv_v32 wants (num_blocks, block_size, 1, 576). Treat each
    # pool slot as its own one-token block.
    kv_4d = kv_bf16.unsqueeze(2) if kv_bf16.dim() == 3 else kv_bf16  # (pool, 1, 1, 576)
    return quantize_kv_v32(kv_4d)


def unpack_kv_cache_fp8(packed: torch.Tensor) -> torch.Tensor:
    """Dequantize packed FP8 V32 KV cache back to bf16 for reference comparison.

    Input:  packed [pool_size, 1, 1, 656] uint8
    Output: kv_bf16 [pool_size, 1, 576] bf16
    """
    from tests.test_decode import dequantize_kv_v32
    out_4d = dequantize_kv_v32(packed)  # (pool, 1, 1, 576)
    return out_4d.squeeze(2)


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
    from flash_mla_sm120.ops import sparse_mla_fwd
    out, _ = sparse_mla_fwd(
        q=q.contiguous(),
        kv_cache=kv_packed.contiguous(),
        indices=indices.contiguous(),
        sm_scale=sm_scale,
        d_v=d_v,
    )
    return out


def sm120_sparse_mla_decode(
    q: torch.Tensor,
    kv_packed: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
) -> torch.Tensor:
    from flash_mla_sm120.ops import sparse_mla_decode_fwd
    out, _ = sparse_mla_decode_fwd(
        q=q.contiguous(),
        kv_cache=kv_packed.contiguous(),
        indices=indices.contiguous(),
        sm_scale=sm_scale,
        d_v=d_v,
    )
    return out


def sm120_sparse_mla_prefill(
    q: torch.Tensor,
    kv_packed: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
) -> torch.Tensor:
    from flash_mla_sm120.ops import sparse_mla_prefill_fwd
    out, _ = sparse_mla_prefill_fwd(
        q=q.contiguous(),
        kv_cache=kv_packed.contiguous(),
        indices=indices.contiguous(),
        sm_scale=sm_scale,
        d_v=d_v,
    )
    return out


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

    # Scale to FP8-friendly range; raw randn (stdev=1) magnitudes exceed
    # UE8M0 power-of-2 scale granularity and produce out-of-tolerance noise.
    q = (torch.randn(batch_size, num_heads, DIM, device=DEVICE,
                      dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_bf16 = (torch.randn(pool_size, 1, DIM, device=DEVICE,
                            dtype=torch.bfloat16) / 10).clamp(-1, 1)
    indices = torch.randint(0, pool_size, (batch_size, TOPK), device=DEVICE, dtype=torch.int32)

    kv_packed = pack_kv_cache_fp8(kv_bf16)
    kv_dequant = unpack_kv_cache_fp8(kv_packed)

    ref = sparse_attention_ref(q, kv_dequant, indices, sm_scale, D_NOPE)
    out = sm120_sparse_mla_decode(q, kv_packed, indices, sm_scale, D_NOPE)

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

    # Scale to FP8-friendly range; raw randn (stdev=1) magnitudes exceed
    # UE8M0 power-of-2 scale granularity and produce out-of-tolerance noise.
    q = (torch.randn(batch_size, num_heads, DIM, device=DEVICE,
                      dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_bf16 = (torch.randn(pool_size, 1, DIM, device=DEVICE,
                            dtype=torch.bfloat16) / 10).clamp(-1, 1)

    indices = torch.full((batch_size, TOPK), -1, device=DEVICE, dtype=torch.int32)
    indices[:, :n_valid] = torch.randint(0, pool_size, (batch_size, n_valid), device=DEVICE, dtype=torch.int32)

    kv_packed = pack_kv_cache_fp8(kv_bf16)
    kv_dequant = unpack_kv_cache_fp8(kv_packed)

    ref = sparse_attention_ref(q, kv_dequant, indices, sm_scale, D_NOPE)
    out = sm120_sparse_mla_decode(q, kv_packed, indices, sm_scale, D_NOPE)

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

    # Scale to FP8-friendly range; raw randn (stdev=1) magnitudes exceed
    # UE8M0 power-of-2 scale granularity and produce out-of-tolerance noise.
    q = (torch.randn(batch_size, num_heads, DIM, device=DEVICE,
                      dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_bf16 = (torch.randn(pool_size, 1, DIM, device=DEVICE,
                            dtype=torch.bfloat16) / 10).clamp(-1, 1)

    indices = torch.full((batch_size, TOPK), -1, device=DEVICE, dtype=torch.int32)
    indices[:, :32] = torch.randint(0, pool_size, (batch_size, 32), device=DEVICE, dtype=torch.int32)

    kv_packed = pack_kv_cache_fp8(kv_bf16)

    out = sm120_sparse_mla_decode(q, kv_packed, indices, sm_scale, D_NOPE)
    assert not torch.isnan(out).any(), "output contains NaN"
    assert not torch.isinf(out).any(), "output contains Inf"


@pytest.mark.parametrize("num_heads", HEAD_COUNTS)
def test_prefill_path(num_heads: int) -> None:
    """Test prefill path (num_tokens > 64).

    Prefill uses FP8 MMA for both QK (Q quantized to FP8) and XV
    (weights pre-scaled by V_scale then quantized to FP8), so tolerance
    is higher than decode which uses direct bf16×fp8 scalar dot products.
    """
    torch.manual_seed(99)
    pool_size = 4096
    sm_scale = DIM ** -0.5
    batch_size = 128  # triggers prefill path

    # Scale to FP8-friendly range; raw randn (stdev=1) magnitudes exceed
    # UE8M0 power-of-2 scale granularity and produce out-of-tolerance noise.
    q = (torch.randn(batch_size, num_heads, DIM, device=DEVICE,
                      dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_bf16 = (torch.randn(pool_size, 1, DIM, device=DEVICE,
                            dtype=torch.bfloat16) / 10).clamp(-1, 1)
    indices = torch.randint(0, pool_size, (batch_size, TOPK), device=DEVICE, dtype=torch.int32)

    kv_packed = pack_kv_cache_fp8(kv_bf16)
    kv_dequant = unpack_kv_cache_fp8(kv_packed)

    ref = sparse_attention_ref(q, kv_dequant, indices, sm_scale, D_NOPE)
    out = sm120_sparse_mla_prefill(q, kv_packed, indices, sm_scale, D_NOPE)

    err = (out.float() - ref.float()).abs()
    print(f"\n  batch={batch_size}, heads={num_heads} (prefill)")
    print(f"  max_abs_err = {err.max().item():.4f}, mismatch@0.1 = {(err>0.1).float().mean()*100:.3f}%")

    torch.testing.assert_close(out.float(), ref.float(), atol=0.3, rtol=0.6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
