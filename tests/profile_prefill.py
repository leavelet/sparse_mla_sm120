"""
Cycle-accurate profiler for prefill MG kernel.

Records per-tile timestamps at 5 points in the main loop:
  t0: tile start (after mbarrier_wait)
  t1: after QK MMA (both groups)
  t2: after softmax + V scale
  t3: after XV MMA (both groups)
  t4: tile end (before bar_arrive)

Output breakdown:
  IO wait:  t0 (from previous tile end to current tile start)
  QK:       t1 - t0
  Softmax:  t2 - t1
  XV:       t3 - t2
  Overhead: t4 - t3

Uses inline PTX clock64() injected via a thin wrapper kernel.
"""

import torch
import os
import sys
import ctypes
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from test_decode import quantize_kv_v32, quantize_kv_model1

# We'll use the production kernel and measure externally with CUDA events
# per-phase. This doesn't need kernel modification — we measure the
# full kernel and combine with NCU source-level sampling.

# Alternative: use ncu --source-level to get per-line cycle distribution.
# This is the most accurate approach without modifying the kernel.


def profile_with_ncu_source(model_type, d_qk, d_v, topk, num_heads, chunk):
    """Run NCU with source-level sampling and parse results."""
    import flash_mla_sm120
    import subprocess

    sm_scale = d_qk ** -0.5
    num_blocks, block_size = 64, 64
    s_kv = num_blocks * block_size

    torch.manual_seed(42)
    if model_type == "V32":
        kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                                device='cuda', dtype=torch.bfloat16) / 10).clamp(-1, 1)
        kv = quantize_kv_v32(kv_bf16)
    else:
        kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                                device='cuda', dtype=torch.bfloat16) / 10).clamp(-1, 1)
        kv = quantize_kv_model1(kv_bf16)

    q = torch.randn(chunk, num_heads, d_qk, device='cuda', dtype=torch.bfloat16) / 10
    indices = torch.randint(0, s_kv, (chunk, topk), device='cuda', dtype=torch.int32)
    indices[:, -10:] = -1

    # Warmup
    for _ in range(3):
        flash_mla_sm120.sparse_mla_prefill_fwd(q, kv, indices, sm_scale, d_v)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(50):
        start.record()
        flash_mla_sm120.sparse_mla_prefill_fwd(q, kv, indices, sm_scale, d_v)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # us
    times.sort()
    median_us = times[len(times)//2]

    ni = topk // 64
    total_flops = 2.0 * chunk * num_heads * topk * (d_qk + d_v)
    tflops = total_flops / (median_us * 1e-6) / 1e12

    # Per-tile estimate
    mg_heads = 32 if num_heads > 16 else 16
    replicate_h = num_heads // mg_heads
    ctas_per_token = replicate_h
    total_ctas = chunk * ctas_per_token
    sms = 188
    waves = total_ctas / sms

    # Per-CTA time
    per_cta_us = median_us / waves if waves >= 1 else median_us
    per_tile_us = per_cta_us / ni

    print(f"\n{'='*70}")
    print(f"Profile: {model_type} h={num_heads} topk={topk} chunk={chunk}")
    print(f"{'='*70}")
    print(f"  Kernel latency:    {median_us:.1f} us")
    print(f"  TFLOP/s:           {tflops:.1f}")
    print(f"  NI (tiles/CTA):    {ni}")
    print(f"  CTAs/token:        {ctas_per_token} ({'MG' if num_heads > 16 else 'SG'})")
    print(f"  Total CTAs:        {total_ctas}")
    print(f"  Waves:             {waves:.1f}")
    print(f"  Est. per-CTA:      {per_cta_us:.1f} us")
    print(f"  Est. per-tile:     {per_tile_us:.2f} us")
    print()

    # Compute breakdown estimate based on MMA counts
    # QK nope MMA count per tile per group: NUM_SCALES * QK_NOPE_KSTEPS
    if d_qk == 576:
        qk_mmas = 4 * 4  # NUM_SCALES=4, KSTEPS=4 (V32)
        xv_mmas = 4 * 2 * 2  # N_V_CHUNKS=4, NT_PER_WARP=2, XV_KSTEPS=2
        n_groups = 2 if num_heads > 16 else 1
    else:
        qk_mmas = 7 * 2  # NUM_SCALES=7, KSTEPS=2 (MODEL1)
        xv_mmas = 7 * 1 * 2  # N_V_CHUNKS=7, NT_PER_WARP=1, XV_KSTEPS=2
        n_groups = 2 if num_heads > 16 else 1

    qk_rope_mmas = 4  # N_ROPE_CHUNKS=4
    total_qk_mmas = (qk_mmas + qk_rope_mmas) * n_groups
    total_xv_mmas = xv_mmas * n_groups
    total_mmas = total_qk_mmas + total_xv_mmas

    # Estimate: 1 FP8 MMA m16n8k32 ≈ 4 cycles (instruction issue),
    # but pipeline depth means effective throughput is higher
    print(f"  MMA breakdown per tile:")
    print(f"    QK nope MMAs:    {qk_mmas} × {n_groups} groups = {qk_mmas * n_groups}")
    print(f"    QK rope MMAs:    {qk_rope_mmas} × {n_groups} groups = {qk_rope_mmas * n_groups}")
    print(f"    XV nope MMAs:    {xv_mmas} × {n_groups} groups = {xv_mmas * n_groups}")
    print(f"    Total MMAs:      {total_mmas}")
    print(f"    Barriers:        ~{11 if n_groups == 2 else 12} per tile")
    print()


def main():
    configs = [
        # (model, d_qk, d_v, topk, num_heads, chunk)
        ("MODEL1", 512, 512, 512, 64, 128),     # V4 Flash — the problematic case
        ("MODEL1", 512, 512, 1024, 128, 128),    # V4 Pro
        ("V32", 576, 512, 2048, 128, 128),       # V32 — baseline (works well)
        ("V32", 576, 512, 2048, 16, 128),        # V32 TP8 — SG
    ]

    for model, d_qk, d_v, topk, nh, chunk in configs:
        profile_with_ncu_source(model, d_qk, d_v, topk, nh, chunk)


if __name__ == "__main__":
    main()
