"""
Independent correctness tests for the V2 combine kernel.

The V2 combine kernel merges partial outputs from scheduler-driven split-KV decode:
  Input:  o_accum   [total_splits, s_q, num_heads, d_v] float32
          lse_accum [total_splits, s_q, num_heads]      float32
          num_splits_ptr [batch + 1] int32 prefix sum
  Output: out [batch * s_q, num_heads, d_v] bfloat16
          lse [batch * s_q, num_heads]      float32

Reference: LSE-weighted merge in float64.
"""

import torch
import pytest
import flash_mla_sm120
import flash_mla_sm120.cuda


def combine_v2_ref(o_accum, lse_accum, num_splits_cpu, batch, num_heads, d_v):
    """Python reference for V2 combine kernel."""
    results_o = []
    results_lse = []
    for b in range(batch):
        start = num_splits_cpu[b].item()
        end = num_splits_cpu[b + 1].item()
        ns = end - start
        if ns <= 1:
            o_split = o_accum[start, 0].to(torch.bfloat16)
            l_split = lse_accum[start, 0]
            results_o.append(o_split)
            results_lse.append(l_split)
            continue

        po = o_accum[start:end, 0].double()
        pl = lse_accum[start:end, 0].double()
        max_lse = pl.max(dim=0).values
        scales = torch.pow(2.0, pl - max_lse.unsqueeze(0))
        scale_sum = scales.sum(dim=0)
        weighted_o = (po * scales.unsqueeze(-1)).sum(dim=0)
        inv = torch.where(scale_sum > 0, 1.0 / scale_sum, torch.zeros_like(scale_sum))
        out = (weighted_o * inv.unsqueeze(-1)).float().to(torch.bfloat16)
        lse = torch.where(
            scale_sum > 0,
            torch.log2(scale_sum) + max_lse,
            torch.full_like(max_lse, -1e30)
        ).float()
        results_o.append(out)
        results_lse.append(lse)
    return torch.stack(results_o), torch.stack(results_lse)


def make_v2_data(batch, nsplits_per_batch, num_heads, d_v, device="cuda"):
    """Generate V2 combine test data with uniform splits per batch."""
    total_splits = batch * nsplits_per_batch
    o_accum = torch.randn(total_splits, 1, num_heads, d_v,
                          dtype=torch.float32, device=device) * 0.1
    lse_accum = torch.randn(total_splits, 1, num_heads,
                            dtype=torch.float32, device=device) * 3.0
    num_splits = torch.arange(0, batch + 1, dtype=torch.int32, device=device) * nsplits_per_batch
    return o_accum, lse_accum, num_splits


class TestCombineV2Correctness:

    @pytest.mark.parametrize("nsplits", [2, 4, 8, 16, 32])
    @pytest.mark.parametrize("num_heads", [8, 64, 128])
    def test_basic(self, nsplits, num_heads):
        torch.manual_seed(42)
        batch, d_v = 4, 512
        o_accum, lse_accum, num_splits = make_v2_data(
            batch, nsplits, num_heads, d_v)

        ref_out, ref_lse = combine_v2_ref(
            o_accum, lse_accum, num_splits.cpu(), batch, num_heads, d_v)

        output = torch.empty(batch, num_heads, d_v,
                             dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(batch, num_heads, dtype=torch.float32, device="cuda")
        flash_mla_sm120.cuda.sparse_mla_combine_v2_fwd(
            o_accum, lse_accum, output, lse,
            num_splits, batch, nsplits)
        torch.cuda.synchronize()

        out_err = (output.float() - ref_out.float()).abs()
        max_out_err = out_err.max().item()
        lse_err = (lse - ref_lse).abs()
        max_lse_err = lse_err.max().item()

        print(f"\n  nsplits={nsplits}, h={num_heads}: "
              f"out max_err={max_out_err:.6f}, lse max_err={max_lse_err:.6f}")
        assert max_out_err < 0.01, f"Output error too large: {max_out_err}"
        assert max_lse_err < 0.01, f"LSE error too large: {max_lse_err}"

    def test_variable_splits(self):
        """Test with variable per-batch split counts."""
        torch.manual_seed(42)
        batch, num_heads, d_v = 4, 64, 512
        splits = [2, 5, 3, 8]
        total = sum(splits)
        prefix = [0]
        for s in splits:
            prefix.append(prefix[-1] + s)

        o_accum = torch.randn(total, 1, num_heads, d_v,
                              dtype=torch.float32, device="cuda") * 0.1
        lse_accum = torch.randn(total, 1, num_heads,
                                dtype=torch.float32, device="cuda") * 3.0
        num_splits = torch.tensor(prefix, dtype=torch.int32, device="cuda")

        ref_out, ref_lse = combine_v2_ref(
            o_accum, lse_accum, num_splits.cpu(), batch, num_heads, d_v)

        output = torch.empty(batch, num_heads, d_v,
                             dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(batch, num_heads, dtype=torch.float32, device="cuda")
        flash_mla_sm120.cuda.sparse_mla_combine_v2_fwd(
            o_accum, lse_accum, output, lse,
            num_splits, batch, max(splits))
        torch.cuda.synchronize()

        err = (output.float() - ref_out.float()).abs().max().item()
        print(f"\n  variable splits {splits}: max_err={err:.6f}")
        assert err < 0.01, f"Variable splits error: {err}"

    def test_large_lse_range(self):
        """Splits with very different LSE values (numerical stability)."""
        torch.manual_seed(99)
        batch, nsplits, num_heads, d_v = 2, 8, 64, 512
        o_accum, lse_accum, num_splits = make_v2_data(
            batch, nsplits, num_heads, d_v)
        for b in range(batch):
            start = b * nsplits
            lse_accum[start, 0, :] = 100.0
            lse_accum[start+1:start+nsplits, 0, :] = -100.0

        ref_out, ref_lse = combine_v2_ref(
            o_accum, lse_accum, num_splits.cpu(), batch, num_heads, d_v)

        output = torch.empty(batch, num_heads, d_v,
                             dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(batch, num_heads, dtype=torch.float32, device="cuda")
        flash_mla_sm120.cuda.sparse_mla_combine_v2_fwd(
            o_accum, lse_accum, output, lse,
            num_splits, batch, nsplits)
        torch.cuda.synchronize()

        err = (output.float() - ref_out.float()).abs().max().item()
        print(f"\n  large LSE range: max_err={err:.6f}")
        assert err < 0.01, f"Numerical instability: {err}"
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
