"""
Correctness tests for FP8 MQA logits CUDA kernels.

Reference computation (PyTorch, float32):
    logits[i, j] = sum_h( max( sum_d(q[i,h,d] * kv[j,d]) * kv_scale[j], 0 ) * w[i,h] )
"""
import torch
import pytest
import math

import sparse_mla_sm120._C as _C


# ── PyTorch reference ──────────────────────────────────────────
def mqa_logits_ref(q_fp8, kv_fp8, kv_scale, weights, k_start=None, k_end=None):
    """Compute MQA logits in FP32 from FP8 inputs."""
    q = q_fp8.float()       # (seq_q, H, D)
    kv = kv_fp8.float()     # (seq_kv, D)

    # dots[i, h, j] = q[i,h,:] . kv[j,:]
    dots = torch.einsum("ihd,jd->ihj", q, kv)

    # scale, relu, weight, sum over heads
    dots = dots * kv_scale[None, None, :]
    dots = torch.relu(dots)
    logits = (dots * weights[:, :, None]).sum(dim=1)  # (seq_q, seq_kv)

    if k_start is not None and k_end is not None:
        for i in range(q.size(0)):
            ks = k_start[i].item()
            ke = k_end[i].item()
            logits[i, :ks] = 0.0
            logits[i, ke:] = 0.0

    return logits


# ── Helpers ────────────────────────────────────────────────────
def make_fp8(shape, scale=0.5, device="cuda"):
    """Random tensor quantized to FP8 E4M3."""
    x = torch.randn(shape, device=device) * scale
    return x.to(torch.float8_e4m3fn)


def allclose(a, b, atol=0.5, rtol=0.02):
    """Check closeness, returning max absolute error for diagnostics."""
    diff = (a.float() - b.float()).abs()
    ref_scale = b.float().abs().clamp(min=1.0)
    rel_err = (diff / ref_scale).max().item()
    abs_err = diff.max().item()
    ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    return ok, abs_err, rel_err


# ── Ragged tests ──────────────────────────────────────────────
class TestRaggedMQA:

    @pytest.mark.parametrize("num_heads", [16, 128])
    @pytest.mark.parametrize("seq_q", [1, 4, 16])
    @pytest.mark.parametrize("seq_kv", [128, 256, 512, 1024])
    def test_basic(self, num_heads, seq_q, seq_kv):
        """Full-range ragged test (k_start=0, k_end=seq_kv for all queries)."""
        D = 128
        q = make_fp8((seq_q, num_heads, D))
        kv = make_fp8((seq_kv, D))
        kv_scale = torch.rand(seq_kv, device="cuda") * 0.1 + 0.01
        w = torch.randn(seq_q, num_heads, device="cuda") * 0.1

        k_start = torch.zeros(seq_q, dtype=torch.int32, device="cuda")
        k_end = torch.full((seq_q,), seq_kv, dtype=torch.int32, device="cuda")

        align = 256
        stride = ((seq_kv + align - 1) // align) * align
        logits = torch.zeros(seq_q, stride, dtype=torch.float32, device="cuda")
        out = logits[:, :seq_kv]

        _C.fp8_mqa_logits_ragged_fwd(q, kv, kv_scale, w, k_start, k_end, out, seq_kv)

        ref = mqa_logits_ref(q, kv, kv_scale, w, k_start, k_end)

        ok, abs_err, rel_err = allclose(out, ref)
        assert ok, (
            f"Ragged MQA mismatch: num_heads={num_heads}, seq_q={seq_q}, "
            f"seq_kv={seq_kv}, abs_err={abs_err:.4f}, rel_err={rel_err:.4f}"
        )

    @pytest.mark.parametrize("num_heads", [16, 128])
    def test_ragged_ranges(self, num_heads):
        """Test with non-trivial k_start/k_end per query."""
        D = 128
        seq_q = 8
        seq_kv = 512

        q = make_fp8((seq_q, num_heads, D))
        kv = make_fp8((seq_kv, D))
        kv_scale = torch.rand(seq_kv, device="cuda") * 0.05 + 0.01
        w = torch.randn(seq_q, num_heads, device="cuda") * 0.1

        k_start = torch.randint(0, seq_kv // 2, (seq_q,), dtype=torch.int32, device="cuda")
        k_end = k_start + torch.randint(64, seq_kv // 2, (seq_q,), dtype=torch.int32, device="cuda")
        k_end = k_end.clamp(max=seq_kv)

        max_k = k_end.max().item()
        align = 256
        stride = ((max_k + align - 1) // align) * align
        logits = torch.zeros(seq_q, stride, dtype=torch.float32, device="cuda")
        out = logits[:, :max_k]

        _C.fp8_mqa_logits_ragged_fwd(q, kv, kv_scale, w, k_start, k_end, out, max_k)

        ref = mqa_logits_ref(q, kv, kv_scale, w, k_start, k_end)[:, :max_k]

        ok, abs_err, rel_err = allclose(out, ref)
        assert ok, (
            f"Ragged ranges mismatch: num_heads={num_heads}, "
            f"abs_err={abs_err:.4f}, rel_err={rel_err:.4f}"
        )

    def test_partial_tile(self):
        """seq_kv not aligned to BLOCK_KV=128."""
        D = 128
        num_heads = 16
        seq_q = 2
        seq_kv = 200  # not a multiple of 128

        q = make_fp8((seq_q, num_heads, D))
        kv = make_fp8((seq_kv, D))
        kv_scale = torch.rand(seq_kv, device="cuda") * 0.05 + 0.01
        w = torch.randn(seq_q, num_heads, device="cuda") * 0.1

        k_start = torch.zeros(seq_q, dtype=torch.int32, device="cuda")
        k_end = torch.full((seq_q,), seq_kv, dtype=torch.int32, device="cuda")

        align = 256
        stride = ((seq_kv + align - 1) // align) * align
        logits = torch.zeros(seq_q, stride, dtype=torch.float32, device="cuda")
        out = logits[:, :seq_kv]

        _C.fp8_mqa_logits_ragged_fwd(q, kv, kv_scale, w, k_start, k_end, out, seq_kv)

        ref = mqa_logits_ref(q, kv, kv_scale, w, k_start, k_end)

        ok, abs_err, rel_err = allclose(out, ref)
        assert ok, f"Partial tile mismatch: abs_err={abs_err:.4f}, rel_err={rel_err:.4f}"


# ── Paged tests ───────────────────────────────────────────────
class TestPagedMQA:

    @pytest.mark.parametrize("num_heads", [16, 128])
    @pytest.mark.parametrize("batch", [1, 4])
    @pytest.mark.parametrize("ctx_len", [64, 128, 256, 512])
    def test_basic(self, num_heads, batch, ctx_len):
        """Basic paged test with identity block table."""
        D = 128
        page_size = 64
        next_n = 1

        num_pages_per_seq = (ctx_len + page_size - 1) // page_size
        total_pages = batch * num_pages_per_seq

        q = make_fp8((batch * next_n, num_heads, D))
        kv = make_fp8((total_pages, page_size, D))
        kv_scale = torch.rand(total_pages, page_size, device="cuda") * 0.05 + 0.01
        w = torch.randn(batch * next_n, num_heads, device="cuda") * 0.1
        ctx_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device="cuda")

        block_table = torch.zeros(batch, num_pages_per_seq, dtype=torch.int32, device="cuda")
        for b in range(batch):
            for p in range(num_pages_per_seq):
                block_table[b, p] = b * num_pages_per_seq + p

        align = 256
        stride = ((ctx_len + align - 1) // align) * align
        logits = torch.zeros(batch * next_n, stride, dtype=torch.float32, device="cuda")
        out = logits[:, :ctx_len]

        _C.fp8_mqa_logits_paged_fwd(q, kv, kv_scale, w, ctx_lens, block_table, out, next_n)

        # Reference: reconstruct flat KV from pages
        flat_kv_list = []
        flat_scale_list = []
        for b in range(batch):
            pages = block_table[b]
            kv_tokens = []
            sc_tokens = []
            for p in range(num_pages_per_seq):
                phys = pages[p].item()
                kv_tokens.append(kv[phys])
                sc_tokens.append(kv_scale[phys])
            flat_kv_list.append(torch.cat(kv_tokens, dim=0)[:ctx_len])
            flat_scale_list.append(torch.cat(sc_tokens, dim=0)[:ctx_len])

        for b in range(batch):
            for t in range(next_n):
                qi = b * next_n + t
                q_single = q[qi:qi+1]
                w_single = w[qi:qi+1]
                ref = mqa_logits_ref(q_single, flat_kv_list[b], flat_scale_list[b], w_single)
                actual = out[qi, :ctx_len]

                ok, abs_err, rel_err = allclose(actual.unsqueeze(0), ref)
                assert ok, (
                    f"Paged MQA mismatch: num_heads={num_heads}, batch={batch}, "
                    f"ctx_len={ctx_len}, b={b}, t={t}, "
                    f"abs_err={abs_err:.4f}, rel_err={rel_err:.4f}"
                )

    def test_variable_context(self):
        """Batch with variable context lengths."""
        D = 128
        num_heads = 16
        page_size = 64
        batch = 4
        next_n = 1
        ctx_lens_list = [128, 64, 256, 192]

        max_ctx = max(ctx_lens_list)
        max_pages = (max_ctx + page_size - 1) // page_size
        total_pages = sum((c + page_size - 1) // page_size for c in ctx_lens_list)

        q = make_fp8((batch * next_n, num_heads, D))
        kv = make_fp8((total_pages, page_size, D))
        kv_scale = torch.rand(total_pages, page_size, device="cuda") * 0.05 + 0.01
        w = torch.randn(batch * next_n, num_heads, device="cuda") * 0.1
        ctx_lens = torch.tensor(ctx_lens_list, dtype=torch.int32, device="cuda")

        block_table = torch.zeros(batch, max_pages, dtype=torch.int32, device="cuda")
        page_offset = 0
        for b in range(batch):
            np = (ctx_lens_list[b] + page_size - 1) // page_size
            for p in range(np):
                block_table[b, p] = page_offset + p
            page_offset += np

        align = 256
        stride = ((max_ctx + align - 1) // align) * align
        logits = torch.zeros(batch * next_n, stride, dtype=torch.float32, device="cuda")
        out = logits[:, :max_ctx]

        _C.fp8_mqa_logits_paged_fwd(q, kv, kv_scale, w, ctx_lens, block_table, out, next_n)

        page_offset = 0
        for b in range(batch):
            cl = ctx_lens_list[b]
            np = (cl + page_size - 1) // page_size
            flat_kv = torch.cat([kv[page_offset + p] for p in range(np)], dim=0)[:cl]
            flat_sc = torch.cat([kv_scale[page_offset + p] for p in range(np)], dim=0)[:cl]
            page_offset += np

            qi = b * next_n
            q_s = q[qi:qi+1]
            w_s = w[qi:qi+1]
            ref = mqa_logits_ref(q_s, flat_kv, flat_sc, w_s)
            actual = out[qi, :cl]

            ok, abs_err, rel_err = allclose(actual.unsqueeze(0), ref)
            assert ok, (
                f"Variable ctx mismatch: b={b}, ctx_len={cl}, "
                f"abs_err={abs_err:.4f}, rel_err={rel_err:.4f}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
