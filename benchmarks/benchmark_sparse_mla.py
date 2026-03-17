"""
Benchmark for sparse_mla_sm120 CUDA kernel.

Reports: latency, TFLOP/s, effective bandwidth, roofline analysis.

Usage:
    python benchmarks/benchmark_sparse_mla.py
"""

import torch
import time
import sys

D_V = 512
D_ROPE = 64
DIM = D_V + D_ROPE  # 576
TOPK = 2048
POOL_SIZE = 336_000
QUANT_TILE = 128
NUM_SCALES = D_V // QUANT_TILE
KV_PACKED_BYTES = D_V + NUM_SCALES * 4 + D_ROPE * 2  # 656
FP8_MAX = 448.0

# RTX PRO 6000 specs
BW_TB = 1.6       # TB/s GDDR7
BF16_TFLOPS = 380  # TFLOP/s
FP8_TFLOPS = 700   # TFLOP/s


def pack_kv_cache_fp8(kv_bf16):
    pool_size = kv_bf16.shape[0]
    kv_f = kv_bf16.squeeze(1).float()
    packed = torch.zeros(pool_size, 1, KV_PACKED_BYTES, dtype=torch.uint8, device=kv_bf16.device)
    nope = kv_f[:, :D_V]
    rope = kv_bf16.squeeze(1)[:, D_V:]
    for b in range(NUM_SCALES):
        tile = nope[:, b * QUANT_TILE : (b + 1) * QUANT_TILE]
        amax = tile.abs().amax(dim=1).clamp(min=1e-4)
        scale = amax / FP8_MAX
        scale_inv = 1.0 / scale
        fp8_vals = (tile * scale_inv.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        packed[:, 0, b * QUANT_TILE : (b + 1) * QUANT_TILE] = fp8_vals.view(torch.uint8)
        scale_bytes = scale.to(torch.float32).view(torch.uint8).reshape(pool_size, 4)
        packed[:, 0, D_V + b * 4 : D_V + (b + 1) * 4] = scale_bytes
    rope_bytes = rope.contiguous().view(torch.uint8).reshape(pool_size, D_ROPE * 2)
    packed[:, 0, D_V + NUM_SCALES * 4:] = rope_bytes
    return packed


def flops(num_tokens, num_heads):
    return 2.0 * num_tokens * num_heads * TOPK * (DIM + D_V)


def kv_bytes_read(num_tokens):
    return num_tokens * TOPK * KV_PACKED_BYTES


def q_bytes_read(num_tokens, num_heads):
    return num_tokens * num_heads * DIM * 2


def output_bytes_written(num_tokens, num_heads):
    return num_tokens * num_heads * D_V * 2


def torch_sparse_ref(q, kv_packed, indices, sm_scale):
    """PyTorch reference using dequantized bf16."""
    pool_size = kv_packed.shape[0]
    # Dequant the needed entries only
    batch, num_heads, dim = q.shape
    idx = indices.long().clamp(min=0)  # [batch, topk]
    flat_idx = idx.reshape(-1)

    kv_raw = kv_packed[flat_idx, 0]  # [batch*topk, 656] uint8
    # Dequant nope
    nope_parts = []
    for b in range(NUM_SCALES):
        fp8_raw = kv_raw[:, b*QUANT_TILE:(b+1)*QUANT_TILE]
        fp8_f = fp8_raw.view(torch.float8_e4m3fn).float()
        scale_bytes = kv_raw[:, D_V + b*4 : D_V + (b+1)*4]
        scale = scale_bytes.contiguous().view(torch.float32).squeeze(-1)
        nope_parts.append(fp8_f * scale.unsqueeze(1))
    nope = torch.cat(nope_parts, dim=1)  # [batch*topk, 512]

    rope_bytes = kv_raw[:, D_V + NUM_SCALES * 4:]
    rope = rope_bytes.contiguous().view(torch.bfloat16).reshape(-1, D_ROPE).float()

    kv_dequant = torch.cat([nope, rope], dim=1).reshape(batch, TOPK, DIM)

    scores = torch.einsum("bhd,bkd->bhk", q.float(), kv_dequant) * sm_scale
    # Mask invalid
    valid = indices >= 0
    scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhk,bkd->bhd", weights, kv_dequant[:, :, :D_V]).to(torch.bfloat16)


def bench_fn(fn, warmup=10, rep=50):
    """Benchmark a function, return median time in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times)//2]  # median


def main():
    dev = torch.cuda.get_device_properties(0)
    print(f"Device: {dev.name}, SM {dev.major}{dev.minor}, {dev.multi_processor_count} SMs")
    print(f"GDDR7 BW: {BW_TB} TB/s, BF16: {BF16_TFLOPS} TFLOP/s, FP8: {FP8_TFLOPS} TFLOP/s")
    print(f"Pool: {POOL_SIZE}, topk: {TOPK}, dim: {DIM}, d_v: {D_V}, kv_packed: {KV_PACKED_BYTES}")
    print()

    # Pre-allocate
    torch.manual_seed(42)
    kv_bf16 = torch.randn(POOL_SIZE, 1, DIM, device="cuda", dtype=torch.bfloat16)
    kv_packed = pack_kv_cache_fp8(kv_bf16)
    del kv_bf16
    torch.cuda.empty_cache()

    import sparse_mla_sm120

    # ── Roofline analysis ──
    print("=" * 110)
    print("Roofline analysis")
    print("=" * 110)
    print(f"{'Scenario':<28} {'FLOPs':>10} {'KV Read':>10} {'Q Read':>8} {'AI':>7} "
          f"{'BW min':>9} {'BF16 min':>9} {'FP8 min':>9} {'Bound':>10}")
    print("-" * 110)
    for nt, label in [(1, "decode bs=1"), (4, "decode bs=4"), (8, "decode bs=8"),
                       (128, "prefill 128"), (1024, "prefill 1K")]:
        for nh, hl in [(16, "16h"), (128, "128h")]:
            f = flops(nt, nh)
            kv_b = kv_bytes_read(nt)
            q_b = q_bytes_read(nt, nh)
            total_b = kv_b + q_b
            ai = f / total_b
            mem_ms = total_b / (BW_TB * 1e12) * 1e3
            bf16_ms = f / (BF16_TFLOPS * 1e12) * 1e3
            fp8_ms = f / (FP8_TFLOPS * 1e12) * 1e3
            bound = "compute" if ai > FP8_TFLOPS / BW_TB else "memory"
            print(f"  {label} {hl:<18} {f/1e9:>8.1f}G {kv_b/1e6:>8.1f}M {q_b/1e6:>6.1f}M "
                  f"{ai:>6.0f} {mem_ms:>7.3f}ms {bf16_ms:>7.3f}ms {fp8_ms:>7.3f}ms {bound:>10}")
    print()

    # ── Decode benchmark ──
    print("=" * 110)
    print("Decode latency (ms) — split-KV path")
    print("=" * 110)
    print(f"{'bs':>4} {'heads':>6} {'sm120_cuda ms':>14} {'torch_ref ms':>13} "
          f"{'speedup':>8} {'TFLOP/s':>8} {'eff BW TB/s':>12} {'% BW peak':>10}")
    print("-" * 110)

    for nh in [16, 128]:
        for bs in [1, 2, 4, 8]:
            q = torch.randn(bs, nh, DIM, device="cuda", dtype=torch.bfloat16)
            indices = torch.randint(0, POOL_SIZE, (bs, TOPK), device="cuda", dtype=torch.int32)
            output = torch.empty(bs, nh, D_V, device="cuda", dtype=torch.bfloat16)

            def run_kernel():
                sparse_mla_sm120.sparse_mla_fwd(q, kv_packed, indices, DIM**-0.5, D_V)

            def run_torch():
                torch_sparse_ref(q, kv_packed, indices, DIM**-0.5)

            ms_kernel = bench_fn(run_kernel, warmup=10, rep=50)
            ms_torch = bench_fn(run_torch, warmup=5, rep=20)

            f = flops(bs, nh)
            tflops = f / (ms_kernel * 1e-3) / 1e12
            total_bytes = kv_bytes_read(bs) + q_bytes_read(bs, nh) + output_bytes_written(bs, nh)
            eff_bw = total_bytes / (ms_kernel * 1e-3) / 1e12
            bw_pct = eff_bw / BW_TB * 100

            print(f"  {bs:>2}   {nh:>4}   {ms_kernel:>12.3f}   {ms_torch:>11.3f}   "
                  f"{ms_torch/ms_kernel:>6.1f}x   {tflops:>6.1f}   {eff_bw:>10.3f}   {bw_pct:>8.1f}%")

    print()

    # ── Prefill benchmark ──
    print("=" * 110)
    print("Prefill throughput (TFLOP/s) — fused online-softmax path")
    print("=" * 110)
    print(f"{'tokens':>7} {'heads':>6} {'sm120_cuda ms':>14} {'TFLOP/s':>8} {'eff BW TB/s':>12}")
    print("-" * 110)

    for nh in [16, 128]:
        for nt in [128, 256, 512, 1024]:
            q = torch.randn(nt, nh, DIM, device="cuda", dtype=torch.bfloat16)
            indices = torch.randint(0, POOL_SIZE, (nt, TOPK), device="cuda", dtype=torch.int32)

            def run_kernel():
                sparse_mla_sm120.sparse_mla_fwd(q, kv_packed, indices, DIM**-0.5, D_V)

            ms_kernel = bench_fn(run_kernel, warmup=5, rep=20)
            f = flops(nt, nh)
            tflops = f / (ms_kernel * 1e-3) / 1e12
            total_bytes = kv_bytes_read(nt) + q_bytes_read(nt, nh) + output_bytes_written(nt, nh)
            eff_bw = total_bytes / (ms_kernel * 1e-3) / 1e12
            print(f"  {nt:>5}   {nh:>4}   {ms_kernel:>12.3f}   {tflops:>6.1f}   {eff_bw:>10.3f}")

    # ── Workspace memory ──
    print()
    print("=" * 80)
    print("Workspace memory comparison (MB)")
    print("=" * 80)
    BI = 32
    NI = TOPK // BI
    for nt, label in [(8, "decode bs=8"), (1024, "prefill 1K")]:
        for nh in [16, 128]:
            ws_split = (nt * NI * nh * D_V * 2 + nt * NI * nh * 4) / 1e6
            ws_torch = (nt * TOPK * DIM * 2 + nt * nh * TOPK * 4 * 2) / 1e6
            ws_fused = 0.0
            path = "split-KV" if nt <= 64 else "fused"
            ws_ours = ws_split if nt <= 64 else ws_fused
            print(f"  {label} {nh}h: ours={ws_ours:.0f} MB ({path}), torch={ws_torch:.0f} MB")


if __name__ == "__main__":
    main()
