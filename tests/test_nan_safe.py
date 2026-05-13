"""
NaN-safe invalid-index handling tests.
Verifies: 0 × NaN = NaN propagation is blocked for invalid entries (idx < 0).
Covers XV NoPE (FP8 MMA) and XV rope (BF16 MMA from global).
"""
import torch
import pytest
import math
from flash_mla_sm120.ops import sparse_mla_decode_fwd, sparse_mla_prefill_fwd, _effective_stride_kv_row
from tests.test_decode import quantize_kv_model1


def _make_nan_seeded_kv(num_blocks, block_size, d_qk):
    """Create KV cache where slot 0 has FP8 NaN bytes (0x7F) in nope region
    and BF16 NaN in rope region. Other slots have normal data."""
    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                            device="cuda", dtype=torch.bfloat16) * 0.5).clamp(-2, 2)
    kv_packed = quantize_kv_model1(kv_bf16)

    # Poison slot 0: write FP8 NaN (0x7F) into nope region
    flat = kv_packed.view(-1)
    data_stride = 448 + 64 * 2  # 576 bytes data per token in MODEL1
    # Slot 0 is at byte offset 0 in block 0
    for d in range(448):  # nope region
        flat[d] = 0x7F  # FP8 E4M3 NaN
    # Poison rope region with BF16 NaN (0x7FC0)
    rope_start = 448
    for d in range(0, 128, 2):
        flat[rope_start + d] = 0xC0
        flat[rope_start + d + 1] = 0x7F

    return kv_packed


class TestNaNSafeInvalidIndex:
    """Verify that invalid entries (idx < 0) don't cause NaN propagation
    through the XV MMA path when slot 0 contains FP8/BF16 NaN."""

    @pytest.mark.parametrize("num_heads,topk", [
        (64, 512), (128, 1024),
    ])
    def test_decode_v1_nan_safe(self, num_heads, topk):
        torch.manual_seed(42)
        d_qk, d_v = 512, 512
        batch = 2
        sm_scale = 1.0 / math.sqrt(d_qk)

        kv_packed = _make_nan_seeded_kv(64, 64, d_qk).view(-1, 64, 1, 584)
        q = torch.randn(batch, num_heads, d_qk, device="cuda", dtype=torch.bfloat16) * 0.5
        indices = torch.randint(1, 4096, (batch, topk), device="cuda", dtype=torch.int32)
        # Make ~25% of entries invalid (pointing to NaN-poisoned slot 0 via clamp)
        indices[:, -topk//4:] = -1

        output, lse = sparse_mla_decode_fwd(q, kv_packed, indices, sm_scale, d_v)

        assert torch.isfinite(output).all(), (
            f"NaN/Inf in decode v1 output! "
            f"nan_count={torch.isnan(output).sum().item()}, "
            f"inf_count={torch.isinf(output).sum().item()}"
        )
        assert (output != 0).float().mean().item() > 0.3, "output mostly zero"
        print(f"\n  decode v1 h={num_heads} topk={topk}: finite, "
              f"abs_mean={output.abs().float().mean():.4e}")

    @pytest.mark.parametrize("num_heads,topk", [
        (64, 512), (128, 1024),
    ])
    def test_prefill_nan_safe(self, num_heads, topk):
        torch.manual_seed(42)
        d_qk, d_v = 512, 512
        num_tokens = 4
        sm_scale = 1.0 / math.sqrt(d_qk)

        kv_packed = _make_nan_seeded_kv(64, 64, d_qk).view(-1, 64, 1, 584)
        q = torch.randn(num_tokens, num_heads, d_qk, device="cuda", dtype=torch.bfloat16) * 0.5
        indices = torch.randint(1, 4096, (num_tokens, topk), device="cuda", dtype=torch.int32)
        indices[:, -topk//4:] = -1

        output, lse = sparse_mla_prefill_fwd(q, kv_packed, indices, sm_scale, d_v)

        assert torch.isfinite(output).all(), (
            f"NaN/Inf in prefill output! "
            f"nan_count={torch.isnan(output).sum().item()}"
        )
        assert (output != 0).float().mean().item() > 0.3, "output mostly zero"
        print(f"\n  prefill h={num_heads} topk={topk}: finite, "
              f"abs_mean={output.abs().float().mean():.4e}")

    def test_all_invalid_entries(self):
        """ALL entries invalid — output should be zero (or near-zero), not NaN."""
        torch.manual_seed(42)
        d_qk, d_v, topk, num_heads = 512, 512, 512, 64
        sm_scale = 1.0 / math.sqrt(d_qk)

        kv_packed = _make_nan_seeded_kv(64, 64, d_qk).view(-1, 64, 1, 584)
        q = torch.randn(1, num_heads, d_qk, device="cuda", dtype=torch.bfloat16)
        indices = torch.full((1, topk), -1, device="cuda", dtype=torch.int32)

        output, lse = sparse_mla_decode_fwd(q, kv_packed, indices, sm_scale, d_v)

        assert torch.isfinite(output).all(), "NaN with all-invalid entries"
        print(f"\n  all-invalid: abs_max={output.abs().max():.4e}")


class TestPaddedStride:
    """Verify _effective_stride_kv_row handles vLLM-style padded block strides."""

    def test_natural_stride(self):
        """Non-padded tensor: stride should be natural."""
        kv = torch.zeros(32, 64, 1, 584, dtype=torch.uint8, device="cuda")
        assert _effective_stride_kv_row(kv) == 584

    def test_padded_stride(self):
        """Padded tensor: stride should be overridden to account for padding."""
        # Simulate vLLM padding: block stride = 37440 vs natural 37376
        bpt = 584
        block_size = 64
        num_blocks = 32
        padded_block_bytes = 37440  # 37376 + 64 padding
        buf = torch.zeros(num_blocks * padded_block_bytes, dtype=torch.uint8, device="cuda")
        kv = torch.as_strided(buf, (num_blocks, block_size, 1, bpt),
                              (padded_block_bytes, bpt, bpt, 1))
        result = _effective_stride_kv_row(kv)
        expected = padded_block_bytes // block_size  # 585
        assert result == expected, f"Expected {expected}, got {result}"
        print(f"\n  padded stride: natural=584, effective={result}")

    def test_2d_tensor(self):
        """2D tensor (flat addressing): stride should be per-element."""
        kv = torch.zeros(4096, 584, dtype=torch.uint8, device="cuda")
        assert _effective_stride_kv_row(kv) == 584
