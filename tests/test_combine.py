"""
Independent correctness and performance tests for the combine kernel.

The combine kernel merges partial outputs from split-KV decode:
  Input:  partial_O [num_tokens, nsplits, num_heads, d_v] float32
          partial_LSE [num_tokens, nsplits, num_heads] float32
  Output: out [num_tokens, num_heads, d_v] bfloat16
          lse [num_tokens, num_heads] float32

Reference: LSE-weighted merge in float64.
"""

import torch
import pytest
import time
import flash_mla_sm120
import flash_mla_sm120.cuda


# ── Reference implementation (float64 for precision) ─────────────────

def combine_ref(partial_O, partial_LSE):
    """Python reference for combine kernel.

    partial_O:   [num_tokens, nsplits, num_heads, d_v] float32
    partial_LSE: [num_tokens, nsplits, num_heads] float32

    Returns:
        out: [num_tokens, num_heads, d_v] bfloat16
        lse: [num_tokens, num_heads] float32
    """
    # Work in float64 for reference precision
    po = partial_O.double()       # [T, S, H, D]
    pl = partial_LSE.double()     # [T, S, H]

    # Max LSE across splits
    max_lse = pl.max(dim=1).values  # [T, H]

    # Scale factors: exp2(lse - max_lse)
    scales = torch.pow(2.0, pl - max_lse.unsqueeze(1))  # [T, S, H]
    scale_sum = scales.sum(dim=1)  # [T, H]

    # Weighted sum of partial_O
    weighted_o = (po * scales.unsqueeze(-1)).sum(dim=1)  # [T, H, D]

    # Normalize
    inv_scale_sum = torch.where(scale_sum > 0, 1.0 / scale_sum, torch.zeros_like(scale_sum))
    out = (weighted_o * inv_scale_sum.unsqueeze(-1)).float().to(torch.bfloat16)

    # LSE in log2 space
    lse = torch.where(
        scale_sum > 0,
        torch.log2(scale_sum) + max_lse,
        torch.full_like(max_lse, -1e30)
    ).float()

    return out, lse


def make_partial_data(num_tokens, nsplits, num_heads, d_v, device="cuda"):
    """Generate realistic partial_O and partial_LSE for testing."""
    # Simulate real kernel output: normalized partial attention + LSE
    partial_O = torch.randn(num_tokens, nsplits, num_heads, d_v,
                            dtype=torch.float32, device=device) * 0.1
    # LSE values: realistic range (typically 0-10 in log2 space)
    partial_LSE = torch.randn(num_tokens, nsplits, num_heads,
                               dtype=torch.float32, device=device) * 3.0
    return partial_O, partial_LSE


# ── Correctness tests ────────────────────────────────────────────────

class TestCombineCorrectness:

    @pytest.mark.parametrize("nsplits", [1, 2, 4, 8, 16, 32])
    @pytest.mark.parametrize("num_heads", [8, 16, 64, 128])
    def test_basic(self, nsplits, num_heads):
        """Test combine against float64 reference."""
        torch.manual_seed(42)
        num_tokens = 4
        d_v = 512

        partial_O, partial_LSE = make_partial_data(
            num_tokens, nsplits, num_heads, d_v)

        # Reference
        ref_out, ref_lse = combine_ref(partial_O, partial_LSE)

        # Kernel
        output = torch.empty(num_tokens, num_heads, d_v,
                             dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(num_tokens, num_heads,
                          dtype=torch.float32, device="cuda")
        flash_mla_sm120.cuda.sparse_mla_combine_fwd(
            partial_O, partial_LSE, output, lse, nsplits)
        torch.cuda.synchronize()

        # Compare output (bf16 tolerance)
        out_err = (output.float() - ref_out.float()).abs()
        max_out_err = out_err.max().item()
        mean_out_err = out_err.mean().item()

        # Compare LSE (float32)
        lse_err = (lse - ref_lse).abs()
        max_lse_err = lse_err.max().item()

        print(f"\n  nsplits={nsplits}, h={num_heads}: "
              f"out max_err={max_out_err:.6f} mean_err={mean_out_err:.6f}, "
              f"lse max_err={max_lse_err:.6f}")

        assert max_out_err < 0.01, f"Output error too large: {max_out_err}"
        assert max_lse_err < 0.01, f"LSE error too large: {max_lse_err}"

    def test_single_split(self):
        """nsplits=1: output should be partial_O converted to bf16."""
        torch.manual_seed(7)
        num_tokens, num_heads, d_v = 2, 64, 512

        partial_O, partial_LSE = make_partial_data(
            num_tokens, 1, num_heads, d_v)

        output = torch.empty(num_tokens, num_heads, d_v,
                             dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(num_tokens, num_heads,
                          dtype=torch.float32, device="cuda")
        flash_mla_sm120.cuda.sparse_mla_combine_fwd(
            partial_O, partial_LSE, output, lse, 1)
        torch.cuda.synchronize()

        expected = partial_O.squeeze(1).to(torch.bfloat16)
        err = (output.float() - expected.float()).abs().max().item()
        print(f"\n  single split: max_err={err:.6f}")
        assert err < 1e-3, f"Single split should be identity: err={err}"

    def test_all_neg_inf_lse(self):
        """All LSE = -inf: output should be zero."""
        num_tokens, nsplits, num_heads, d_v = 1, 4, 16, 512

        partial_O = torch.randn(num_tokens, nsplits, num_heads, d_v,
                                dtype=torch.float32, device="cuda")
        partial_LSE = torch.full((num_tokens, nsplits, num_heads),
                                  -1e30, dtype=torch.float32, device="cuda")

        output = torch.empty(num_tokens, num_heads, d_v,
                             dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(num_tokens, num_heads,
                          dtype=torch.float32, device="cuda")
        flash_mla_sm120.cuda.sparse_mla_combine_fwd(
            partial_O, partial_LSE, output, lse, nsplits)
        torch.cuda.synchronize()

        assert not torch.isnan(output).any(), "NaN in output"
        assert not torch.isinf(output).any(), "Inf in output"

    def test_large_lse_range(self):
        """Splits with very different LSE values (numerical stability)."""
        torch.manual_seed(99)
        num_tokens, nsplits, num_heads, d_v = 2, 8, 64, 512

        partial_O, _ = make_partial_data(num_tokens, nsplits, num_heads, d_v)
        partial_LSE = torch.zeros(num_tokens, nsplits, num_heads,
                                   dtype=torch.float32, device="cuda")
        # One split dominates with much larger LSE
        partial_LSE[:, 0, :] = 100.0
        partial_LSE[:, 1:, :] = -100.0

        ref_out, ref_lse = combine_ref(partial_O, partial_LSE)

        output = torch.empty(num_tokens, num_heads, d_v,
                             dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(num_tokens, num_heads,
                          dtype=torch.float32, device="cuda")
        flash_mla_sm120.cuda.sparse_mla_combine_fwd(
            partial_O, partial_LSE, output, lse, nsplits)
        torch.cuda.synchronize()

        err = (output.float() - ref_out.float()).abs().max().item()
        print(f"\n  large LSE range: max_err={err:.6f}")
        assert err < 0.01, f"Numerical instability: {err}"
        assert not torch.isnan(output).any()


# ── Performance tests ────────────────────────────────────────────────

class TestCombinePerformance:

    @staticmethod
    def bench_combine(num_tokens, nsplits, num_heads, d_v=512,
                       warmup=20, iters=100):
        partial_O = torch.randn(num_tokens, nsplits, num_heads, d_v,
                                dtype=torch.float32, device="cuda")
        partial_LSE = torch.randn(num_tokens, nsplits, num_heads,
                                   dtype=torch.float32, device="cuda")
        output = torch.empty(num_tokens, num_heads, d_v,
                             dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(num_tokens, num_heads,
                          dtype=torch.float32, device="cuda")

        for _ in range(warmup):
            flash_mla_sm120.cuda.sparse_mla_combine_fwd(
                partial_O, partial_LSE, output, lse, nsplits)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            flash_mla_sm120.cuda.sparse_mla_combine_fwd(
                partial_O, partial_LSE, output, lse, nsplits)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters * 1000  # us

    @pytest.mark.parametrize("config", [
        # (num_tokens, nsplits, num_heads, description)
        (1, 16, 128, "V3.2 TP1 bs=1"),
        (1, 16, 16,  "V3.2 TP8 bs=1"),
        (1, 16, 64,  "GLM5.1 TP1 bs=1"),
        (1, 4,  64,  "V4 Flash TP1 bs=1"),
        (1, 8,  128, "V4 Pro TP1 bs=1"),
        (8, 16, 128, "V3.2 TP1 bs=8"),
        (8, 4,  64,  "V4 Flash TP1 bs=8"),
    ])
    def test_perf(self, config):
        num_tokens, nsplits, num_heads, desc = config
        us = self.bench_combine(num_tokens, nsplits, num_heads)

        # Effective data moved
        d_v = 512
        read_bytes = num_tokens * nsplits * num_heads * d_v * 4  # float32 partial_O
        read_bytes += num_tokens * nsplits * num_heads * 4        # float32 LSE
        write_bytes = num_tokens * num_heads * d_v * 2           # bf16 output
        write_bytes += num_tokens * num_heads * 4                 # float32 LSE
        total_bytes = read_bytes + write_bytes

        gbps = total_bytes / (us * 1e-6) / 1e9

        print(f"\n  {desc}: {us:.1f} us, {gbps:.1f} GB/s "
              f"(tokens={num_tokens}, splits={nsplits}, heads={num_heads})")

        # Combine should be fast (< 50 us for typical configs)
        assert us < 100, f"Combine too slow: {us:.1f} us"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
