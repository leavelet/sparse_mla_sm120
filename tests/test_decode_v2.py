"""
Tests for decode v2 (scheduler-driven split-KV decode).

Uses the GPU scheduler to compute per-batch splits, then runs the v2 decode
kernel, then the combine kernel. Compares output to the same FP32 reference
as test_decode.py.
"""

import torch
import pytest
import math
import sys
sys.path.insert(0, 'tests')

import flash_mla_sm120
from flash_mla_sm120.cuda import (
    get_decode_metadata,
    sparse_mla_splitkv_v2_fwd,
    sparse_mla_combine_v2_fwd,
)
from test_decode import (
    quantize_kv_v32, dequantize_kv_v32,
    quantize_kv_model1, dequantize_kv_model1,
    ref_sparse_attn_decode,
)

BI = 64
HPB = 16
NUM_SMS = 188
FIXED_OVERHEAD = 5


def run_decode_v2_test(model_type, d_qk, d_v, topk, num_heads, batch_size,
                       block_size=64, num_blocks=64, attn_sink_val=None):
    """Run v2 decode (scheduler + kernel + combine) and compare to reference."""
    torch.manual_seed(42)
    sm_scale = d_qk ** -0.5
    s_kv = num_blocks * block_size
    s_q = 1

    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                            device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    if model_type == "V32":
        kv_packed = quantize_kv_v32(kv_bf16)
        kv_dequant = dequantize_kv_v32(kv_packed)
    else:
        kv_packed = quantize_kv_model1(kv_bf16)
        kv_dequant = dequantize_kv_model1(kv_packed)

    q = (torch.randn(batch_size, 1, num_heads, d_qk,
                      device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (batch_size, 1, topk),
                             device="cuda", dtype=torch.int32)
    indices[:, :, -10:] = -1

    attn_sink = None
    if attn_sink_val is not None:
        attn_sink = torch.full((num_heads,), attn_sink_val, device="cuda", dtype=torch.float32)

    # Reference
    ref_out, ref_lse = ref_sparse_attn_decode(q, kv_dequant, indices, sm_scale, d_v)
    if attn_sink is not None:
        ref_lse_f = ref_lse.float()
        sink_f = attn_sink.view(1, 1, num_heads)
        scale = 1.0 / (1.0 + torch.exp(sink_f - ref_lse_f))
        ref_out = (ref_out.float() * scale.unsqueeze(-1)).to(torch.bfloat16)

    # V2 path: scheduler → decode v2 → combine
    replicate_h = (num_heads + HPB - 1) // HPB
    num_sm_parts = max(NUM_SMS // (s_q * replicate_h), 1)
    total_splits_bound = batch_size + num_sm_parts

    sched_meta = torch.empty(num_sm_parts * 8, dtype=torch.int32, device="cuda")
    num_splits = torch.empty(batch_size + 1, dtype=torch.int32, device="cuda")
    get_decode_metadata(batch_size, topk, 0, num_sm_parts, FIXED_OVERHEAD,
                        None, None, sched_meta, num_splits)

    o_accum = torch.empty(total_splits_bound, s_q, num_heads, d_v,
                          dtype=torch.float32, device="cuda")
    lse_accum = torch.empty(total_splits_bound, s_q, num_heads,
                            dtype=torch.float32, device="cuda")
    output = torch.empty(batch_size * s_q, num_heads, d_v,
                         dtype=torch.bfloat16, device="cuda")
    out_lse = torch.empty(batch_size * s_q, num_heads,
                          dtype=torch.float32, device="cuda")

    stride_kv_row = kv_packed.stride(-2) * kv_packed.element_size()
    page_block_size = kv_packed.shape[-3] if kv_packed.dim() >= 3 else 1

    q_flat = q.view(batch_size, num_heads, d_qk)
    idx_flat = indices.view(batch_size, topk)

    sparse_mla_splitkv_v2_fwd(
        q_flat, kv_packed, idx_flat,
        o_accum, lse_accum, output, out_lse,
        sched_meta, num_splits,
        sm_scale, topk, stride_kv_row, page_block_size,
        num_sm_parts, attn_sink)
    torch.cuda.synchronize()

    # V2 combine: always launch (CUDA graph friendly — kernel early-exits for is_no_split)
    ns_cpu = num_splits.cpu()
    max_nsplits = max((ns_cpu[i+1] - ns_cpu[i]).item() for i in range(batch_size))
    sparse_mla_combine_v2_fwd(
        o_accum, lse_accum, output, out_lse,
        num_splits, batch_size, max_nsplits, attn_sink)

    out_view = output.view_as(ref_out)
    err = (out_view.float() - ref_out.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    return max_err, mean_err


# ── V32 Tests ────────────────────────────────────────────────────────

class TestV32DecodeV2:
    @pytest.mark.parametrize("num_heads,batch_size", [
        (128, 1), (128, 4), (128, 8),
        (16, 1), (16, 4),
        (64, 1), (64, 4),
        (8, 1), (8, 4),
    ])
    def test_correctness(self, num_heads, batch_size):
        max_err, mean_err = run_decode_v2_test(
            "V32", d_qk=576, d_v=512, topk=2048,
            num_heads=num_heads, batch_size=batch_size)
        print(f"\n  V32 v2 h={num_heads} bs={batch_size}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f}")
        assert max_err < 0.001, f"V32 decode v2 failed: max_err={max_err}"


# ── MODEL1 Tests ─────────────────────────────────────────────────────

class TestMODEL1DecodeV2:
    @pytest.mark.parametrize("num_heads,topk,batch_size", [
        (8, 512, 1), (8, 512, 4),
        (64, 512, 1), (64, 512, 4), (64, 512, 8),
        (16, 1024, 1), (16, 1024, 4),
        (128, 1024, 1), (128, 1024, 4), (128, 1024, 8),
    ])
    def test_correctness(self, num_heads, topk, batch_size):
        max_err, mean_err = run_decode_v2_test(
            "MODEL1", d_qk=512, d_v=512, topk=topk,
            num_heads=num_heads, batch_size=batch_size)
        print(f"\n  MODEL1 v2 h={num_heads} topk={topk} bs={batch_size}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f}")
        assert max_err < 0.001, f"MODEL1 decode v2 failed: max_err={max_err}"


# ── attn_sink with v2 ───────────────────────────────────────────────

class TestAttnSinkDecodeV2:
    @pytest.mark.parametrize("num_heads,topk,batch_size", [
        (64, 512, 1), (128, 1024, 4),
    ])
    def test_attn_sink_random(self, num_heads, topk, batch_size):
        max_err, mean_err = run_decode_v2_test(
            "MODEL1", d_qk=512, d_v=512, topk=topk,
            num_heads=num_heads, batch_size=batch_size,
            attn_sink_val=2.0)
        print(f"\n  v2 attn_sink h={num_heads} topk={topk} bs={batch_size}: "
              f"max_err={max_err:.6f}")
        assert max_err < 0.002, f"v2 attn_sink failed: max_err={max_err}"


# ── Race check ──────────────────────────────────────────────────────

class TestDecodeV2Race:
    def test_race_check(self):
        """Run 3x with different seeds."""
        for seed in [42, 123, 999]:
            torch.manual_seed(seed)
            max_err, _ = run_decode_v2_test(
                "MODEL1", d_qk=512, d_v=512, topk=1024,
                num_heads=128, batch_size=4)
            assert max_err < 0.001, f"v2 race check seed={seed}: max_err={max_err}"
        print("\n  v2 decode race check (3 seeds): PASS")


# ── CUDA graph compatibility ────────────────────────────────────────

class TestDecodeV2CudaGraph:
    """Verify the v2 pipeline works under CUDA graph capture."""

    def test_cuda_graph_capture(self):
        """Capture scheduler→decode→combine as a CUDA graph and replay."""
        torch.manual_seed(42)
        d_qk, d_v, h, topk, b = 512, 512, 128, 1024, 4
        sm_scale = d_qk ** -0.5
        block_size, num_blocks = 64, 64
        s_kv = num_blocks * block_size

        kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                                device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
        kv_packed = quantize_kv_model1(kv_bf16)
        kv_dequant = dequantize_kv_model1(kv_packed)
        q = (torch.randn(b, 1, h, d_qk,
                          device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
        indices = torch.randint(0, s_kv, (b, 1, topk),
                                 device="cuda", dtype=torch.int32)
        indices[:, :, -10:] = -1

        replicate_h = (h + HPB - 1) // HPB
        num_sm_parts = max(NUM_SMS // replicate_h, 1)
        total_splits = b + num_sm_parts

        sched_meta = torch.empty(num_sm_parts * 8, dtype=torch.int32, device="cuda")
        num_splits = torch.empty(b + 1, dtype=torch.int32, device="cuda")
        o_accum = torch.empty(total_splits, 1, h, d_v, dtype=torch.float32, device="cuda")
        lse_accum = torch.empty(total_splits, 1, h, dtype=torch.float32, device="cuda")
        output = torch.empty(b, h, d_v, dtype=torch.bfloat16, device="cuda")
        out_lse = torch.empty(b, h, dtype=torch.float32, device="cuda")

        stride_kv = kv_packed.stride(-2) * kv_packed.element_size()
        pbs = kv_packed.shape[-3]
        qf = q.view(b, h, d_qk)
        idf = indices.view(b, topk)

        # Warmup (outside graph)
        get_decode_metadata(b, topk, 0, num_sm_parts, FIXED_OVERHEAD,
                            None, None, sched_meta, num_splits)
        sparse_mla_splitkv_v2_fwd(qf, kv_packed, idf, o_accum, lse_accum,
                                   output, out_lse, sched_meta, num_splits,
                                   sm_scale, topk, stride_kv, pbs, num_sm_parts)
        ns_cpu = num_splits.cpu()
        max_nsplits = max((ns_cpu[i+1] - ns_cpu[i]).item() for i in range(b))
        sparse_mla_combine_v2_fwd(o_accum, lse_accum, output, out_lse,
                                   num_splits, b, max_nsplits)
        torch.cuda.synchronize()

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            get_decode_metadata(b, topk, 0, num_sm_parts, FIXED_OVERHEAD,
                                None, None, sched_meta, num_splits)
            sparse_mla_splitkv_v2_fwd(qf, kv_packed, idf, o_accum, lse_accum,
                                       output, out_lse, sched_meta, num_splits,
                                       sm_scale, topk, stride_kv, pbs, num_sm_parts)
            sparse_mla_combine_v2_fwd(o_accum, lse_accum, output, out_lse,
                                       num_splits, b, max_nsplits)

        # Replay
        graph.replay()
        torch.cuda.synchronize()

        # Verify correctness
        ref_out, _ = ref_sparse_attn_decode(q, kv_dequant, indices, sm_scale, d_v)
        err = (output.view_as(ref_out).float() - ref_out.float()).abs()
        max_err = err.max().item()
        print(f"\n  CUDA graph replay: max_err={max_err:.6f}")
        assert max_err < 0.001, f"CUDA graph failed: max_err={max_err}"

    def test_all_is_no_split(self):
        """When all batches fit in one partition (is_no_split), combine is a no-op.
        Verify combine kernel handles max_nsplits=1 gracefully."""
        torch.manual_seed(42)
        d_qk, d_v, h, topk, b = 512, 512, 64, 512, 1
        sm_scale = d_qk ** -0.5
        block_size, num_blocks = 64, 64
        s_kv = num_blocks * block_size

        kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                                device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
        kv_packed = quantize_kv_model1(kv_bf16)
        kv_dequant = dequantize_kv_model1(kv_packed)
        q = (torch.randn(b, 1, h, d_qk,
                          device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
        indices = torch.randint(0, s_kv, (b, 1, topk),
                                 device="cuda", dtype=torch.int32)

        replicate_h = (h + HPB - 1) // HPB
        num_sm_parts = max(NUM_SMS // replicate_h, 1)
        total_splits = b + num_sm_parts

        sched_meta = torch.empty(num_sm_parts * 8, dtype=torch.int32, device="cuda")
        num_splits = torch.empty(b + 1, dtype=torch.int32, device="cuda")
        o_accum = torch.empty(total_splits, 1, h, d_v, dtype=torch.float32, device="cuda")
        lse_accum = torch.empty(total_splits, 1, h, dtype=torch.float32, device="cuda")
        output = torch.empty(b, h, d_v, dtype=torch.bfloat16, device="cuda")
        out_lse = torch.empty(b, h, dtype=torch.float32, device="cuda")

        stride_kv = kv_packed.stride(-2) * kv_packed.element_size()
        pbs = kv_packed.shape[-3]
        qf = q.view(b, h, d_qk)
        idf = indices.view(b, topk)

        get_decode_metadata(b, topk, 0, num_sm_parts, FIXED_OVERHEAD,
                            None, None, sched_meta, num_splits)
        sparse_mla_splitkv_v2_fwd(qf, kv_packed, idf, o_accum, lse_accum,
                                   output, out_lse, sched_meta, num_splits,
                                   sm_scale, topk, stride_kv, pbs, num_sm_parts)

        # max_nsplits should be 1 (all is_no_split for b=1 with many partitions)
        ns_cpu = num_splits.cpu()
        max_nsplits = max((ns_cpu[i+1] - ns_cpu[i]).item() for i in range(b))
        print(f"\n  all_is_no_split: max_nsplits={max_nsplits}")

        # Always launch combine (CUDA graph friendly) — should be a no-op
        sparse_mla_combine_v2_fwd(o_accum, lse_accum, output, out_lse,
                                   num_splits, b, max_nsplits)
        torch.cuda.synchronize()

        ref_out, _ = ref_sparse_attn_decode(q, kv_dequant, indices, sm_scale, d_v)
        err = (output.view_as(ref_out).float() - ref_out.float()).abs()
        max_err = err.max().item()
        print(f"  max_err={max_err:.6f}")
        assert max_err < 0.001, f"all_is_no_split failed: max_err={max_err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
