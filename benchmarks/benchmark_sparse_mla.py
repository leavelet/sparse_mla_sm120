"""
Benchmark for sparse_mla_sm120 CUDA kernel.

Realistic configs for DeepSeek V3.2 sparse MLA:
  Prefill: chunked prefill, chunk_size=2048 or 4096, total_len=4K..64K
  Decode:  bs=1,2,4,8, total_len=4K..64K

Reports: latency, TFLOP/s, effective bandwidth, roofline analysis.

Usage:
    python benchmarks/benchmark_sparse_mla.py
"""

import torch
import sys

D_V = 512
D_ROPE = 64
DIM = D_V + D_ROPE  # 576
TOPK = 2048
QUANT_TILE = 128
NUM_SCALES = D_V // QUANT_TILE
KV_PACKED_BYTES = D_V + NUM_SCALES * 4 + D_ROPE * 2  # 656
FP8_MAX = 448.0

# RTX PRO 6000 Blackwell specs
BW_TB = 1.6       # TB/s GDDR7
BF16_TFLOPS = 380  # TFLOP/s
FP8_TFLOPS = 700   # TFLOP/s
NUM_SMS = 188


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
        fp8_vals = (tile / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        packed[:, 0, b * QUANT_TILE : (b + 1) * QUANT_TILE] = fp8_vals.view(torch.uint8)
        scale_bytes = scale.to(torch.float32).view(torch.uint8).reshape(pool_size, 4)
        packed[:, 0, D_V + b * 4 : D_V + (b + 1) * 4] = scale_bytes
    rope_bytes = rope.contiguous().view(torch.uint8).reshape(pool_size, D_ROPE * 2)
    packed[:, 0, D_V + NUM_SCALES * 4:] = rope_bytes
    return packed


def flops(num_tokens, num_heads):
    # QK: 2 * tokens * heads * topk * dim, XV: 2 * tokens * heads * topk * d_v
    return 2.0 * num_tokens * num_heads * TOPK * (DIM + D_V)


def kv_bytes_read(num_tokens):
    return num_tokens * TOPK * KV_PACKED_BYTES


def q_bytes_read(num_tokens, num_heads):
    return num_tokens * num_heads * DIM * 2


def output_bytes_written(num_tokens, num_heads):
    return num_tokens * num_heads * D_V * 2


def total_bytes(num_tokens, num_heads):
    return kv_bytes_read(num_tokens) + q_bytes_read(num_tokens, num_heads) + output_bytes_written(num_tokens, num_heads)


def bench_fn(fn, warmup=10, rep=50):
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
    return times[len(times) // 2]


def main():
    dev = torch.cuda.get_device_properties(0)
    print(f"Device: {dev.name}, SM {dev.major}{dev.minor}, {dev.multi_processor_count} SMs")
    print(f"GDDR7 BW: {BW_TB} TB/s, BF16: {BF16_TFLOPS} TFLOP/s, FP8: {FP8_TFLOPS} TFLOP/s")
    print(f"topk: {TOPK}, dim: {DIM}, d_v: {D_V}, kv_packed: {KV_PACKED_BYTES} B/entry")
    print()

    torch.manual_seed(42)
    pool_size = 65536  # max total_len
    kv_bf16 = torch.randn(pool_size, 1, DIM, device="cuda", dtype=torch.bfloat16)
    kv_packed = pack_kv_cache_fp8(kv_bf16)
    del kv_bf16
    torch.cuda.empty_cache()

    import sparse_mla_sm120

    sm_scale = DIM ** -0.5

    # ── Roofline ──
    print("=" * 120)
    print("Roofline analysis (per configuration)")
    print("=" * 120)
    print(f"{'Config':<35} {'FLOPs':>10} {'Bytes':>10} {'AI':>7} "
          f"{'BW min':>9} {'FP8 min':>9} {'Bound':>10}")
    print("-" * 120)
    for label, nt, nh in [
        ("decode bs=1 16h", 1, 16), ("decode bs=1 128h", 1, 128),
        ("decode bs=8 16h", 8, 16), ("decode bs=8 128h", 8, 128),
        ("prefill chunk=2048 16h", 2048, 16), ("prefill chunk=2048 128h", 2048, 128),
        ("prefill chunk=4096 16h", 4096, 16), ("prefill chunk=4096 128h", 4096, 128),
    ]:
        f = flops(nt, nh)
        tb = total_bytes(nt, nh)
        ai = f / tb
        mem_ms = tb / (BW_TB * 1e12) * 1e3
        fp8_ms = f / (FP8_TFLOPS * 1e12) * 1e3
        bound = "compute" if ai > FP8_TFLOPS / BW_TB * 1e12 / 1e12 else "memory"
        print(f"  {label:<33} {f/1e9:>8.1f}G {tb/1e6:>8.1f}M {ai:>6.0f} "
              f"{mem_ms:>7.3f}ms {fp8_ms:>7.3f}ms {bound:>10}")
    print()

    # ── Prefill benchmark (chunked prefill) ──
    print("=" * 120)
    print("Prefill benchmark — chunked prefill (chunk triggers prefill path: chunk > 64)")
    print("=" * 120)
    print(f"{'total_len':>10} {'chunk':>6} {'heads':>6} {'latency ms':>12} "
          f"{'TFLOP/s':>8} {'eff BW TB/s':>12} {'% BW':>6} {'% FP8':>6}")
    print("-" * 120)

    for nh in [16, 128]:
        for total_len_k in [4, 8, 16, 32, 64]:
            total_len = total_len_k * 1024
            if total_len > pool_size:
                continue
            for chunk in [2048, 4096]:
                if chunk > total_len:
                    continue
                q = torch.randn(chunk, nh, DIM, device="cuda", dtype=torch.bfloat16)
                indices = torch.randint(0, total_len, (chunk, TOPK), device="cuda", dtype=torch.int32)

                def run():
                    sparse_mla_sm120.sparse_mla_prefill_fwd(
                        q, kv_packed[:total_len], indices, sm_scale, D_V
                    )

                ms = bench_fn(run, warmup=5, rep=20)
                f = flops(chunk, nh)
                tflops = f / (ms * 1e-3) / 1e12
                tb = total_bytes(chunk, nh)
                eff_bw = tb / (ms * 1e-3) / 1e12
                bw_pct = eff_bw / BW_TB * 100
                fp8_pct = tflops / FP8_TFLOPS * 100

                print(f"  {total_len:>8} {chunk:>6} {nh:>6}   {ms:>10.3f}   "
                      f"{tflops:>6.1f}   {eff_bw:>10.3f}   {bw_pct:>5.1f}% {fp8_pct:>5.1f}%")
        print()

    # ── Decode benchmark ──
    print("=" * 120)
    print("Decode benchmark — split-KV path (bs <= 64)")
    print("=" * 120)
    print(f"{'total_len':>10} {'bs':>4} {'heads':>6} {'latency ms':>12} "
          f"{'TFLOP/s':>8} {'eff BW TB/s':>12} {'% BW':>6}")
    print("-" * 120)

    for nh in [16, 128]:
        for total_len_k in [4, 8, 16, 32, 64]:
            total_len = total_len_k * 1024
            if total_len > pool_size:
                continue
            for bs in [1, 2, 4, 8]:
                q = torch.randn(bs, nh, DIM, device="cuda", dtype=torch.bfloat16)
                indices = torch.randint(0, total_len, (bs, TOPK), device="cuda", dtype=torch.int32)

                def run():
                    sparse_mla_sm120.sparse_mla_decode_fwd(
                        q, kv_packed[:total_len], indices, sm_scale, D_V
                    )

                ms = bench_fn(run, warmup=5, rep=30)
                f = flops(bs, nh)
                tflops = f / (ms * 1e-3) / 1e12
                tb = total_bytes(bs, nh)
                eff_bw = tb / (ms * 1e-3) / 1e12
                bw_pct = eff_bw / BW_TB * 100

                print(f"  {total_len:>8} {bs:>4} {nh:>6}   {ms:>10.3f}   "
                      f"{tflops:>6.1f}   {eff_bw:>10.3f}   {bw_pct:>5.1f}%")
        print()


if __name__ == "__main__":
    main()
