"""
Benchmark for flash_mla_sm120 sparse MLA CUDA kernels.

Realistic configs for DeepSeek V3.2 / V4 sparse MLA:
  Prefill: chunked prefill, chunk_size=2048 or 4096, total_len=4K..64K
  Decode:  bs=1,2,4,8, total_len=4K..64K

Reports: latency, TFLOP/s, effective bandwidth, roofline analysis.

Usage:
    CUDA_VISIBLE_DEVICES=3 python benchmarks/benchmark_sparse_mla.py
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from test_decode import quantize_kv_v32, quantize_kv_model1

# RTX PRO 6000 Blackwell specs
BW_TB = 1.6       # TB/s GDDR7
BF16_TFLOPS = 380  # TFLOP/s
FP8_TFLOPS = 700   # TFLOP/s
NUM_SMS = 188

# Model configs
MODELS = {
    'V32': dict(d_qk=576, d_v=512, d_nope=512, d_rope=64, topk=2048, bpt=656),
    'MODEL1_Flash': dict(d_qk=512, d_v=512, d_nope=448, d_rope=64, topk=512, bpt=584),
    'MODEL1_Pro': dict(d_qk=512, d_v=512, d_nope=448, d_rope=64, topk=1024, bpt=584),
}


def flops(num_tokens, num_heads, d_qk, d_v, topk):
    return 2.0 * num_tokens * num_heads * topk * (d_qk + d_v)


def total_bytes(num_tokens, num_heads, d_qk, d_v, topk, bpt):
    kv = num_tokens * topk * bpt
    q = num_tokens * num_heads * d_qk * 2
    out = num_tokens * num_heads * d_v * 2
    return kv + q + out


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


def make_kv_cache(model, pool_blocks, block_size, device):
    """Create quantized KV cache for a given model config."""
    d_qk = model['d_qk']
    kv_bf16 = (torch.randn(pool_blocks, block_size, 1, d_qk,
                            device=device, dtype=torch.bfloat16) / 10).clamp(-1, 1)
    if d_qk == 576:
        return quantize_kv_v32(kv_bf16)
    else:
        return quantize_kv_model1(kv_bf16)


def main():
    dev = torch.cuda.get_device_properties(0)
    print(f"Device: {dev.name}, SM {dev.major}{dev.minor}, {dev.multi_processor_count} SMs")
    print(f"GDDR7 BW: {BW_TB} TB/s, BF16: {BF16_TFLOPS} TFLOP/s, FP8: {FP8_TFLOPS} TFLOP/s")
    print()

    torch.manual_seed(42)
    import flash_mla_sm120

    block_size = 64
    pool_blocks = 1024  # 64K tokens

    # ── Roofline ──
    print("=" * 120)
    print("Roofline analysis")
    print("=" * 120)
    print(f"  {'Config':<45} {'FLOPs':>10} {'Bytes':>10} {'AI':>7} "
          f"{'BW min':>9} {'FP8 min':>9} {'Bound':>10}")
    print("-" * 120)
    for label, nt, nh, mname in [
        ("V32 decode bs=1 h=128", 1, 128, 'V32'),
        ("V32 decode bs=8 h=128", 8, 128, 'V32'),
        ("V32 prefill chunk=2048 h=128", 2048, 128, 'V32'),
        ("V32 prefill chunk=4096 h=128", 4096, 128, 'V32'),
        ("MODEL1 decode bs=1 h=128 topk=1024", 1, 128, 'MODEL1_Pro'),
        ("MODEL1 prefill chunk=2048 h=128 topk=1024", 2048, 128, 'MODEL1_Pro'),
        ("MODEL1 prefill chunk=2048 h=64 topk=512", 2048, 64, 'MODEL1_Flash'),
    ]:
        m = MODELS[mname]
        f = flops(nt, nh, m['d_qk'], m['d_v'], m['topk'])
        tb = total_bytes(nt, nh, m['d_qk'], m['d_v'], m['topk'], m['bpt'])
        ai = f / tb
        mem_ms = tb / (BW_TB * 1e12) * 1e3
        fp8_ms = f / (FP8_TFLOPS * 1e12) * 1e3
        bound = "compute" if fp8_ms > mem_ms else "memory"
        print(f"  {label:<45} {f/1e9:>8.1f}G {tb/1e6:>8.1f}M {ai:>6.0f} "
              f"{mem_ms:>7.3f}ms {fp8_ms:>7.3f}ms {bound:>10}")
    print()

    # ── V32 Prefill ──
    print("=" * 120)
    print("V32 Prefill (d_qk=576, topk=2048)")
    print("=" * 120)
    m = MODELS['V32']
    kv_packed = make_kv_cache(m, pool_blocks, block_size, 'cuda')
    s_kv = pool_blocks * block_size
    sm_scale = m['d_qk'] ** -0.5

    print(f"  {'chunk':>6} {'heads':>6} {'latency ms':>12} "
          f"{'TFLOP/s':>8} {'eff BW TB/s':>12} {'% BW':>6} {'% FP8':>6}")
    print("-" * 80)

    for nh in [16, 64, 128]:
        for chunk in [128, 512, 2048, 4096]:
            q = torch.randn(chunk, nh, m['d_qk'], device='cuda', dtype=torch.bfloat16)
            indices = torch.randint(0, s_kv, (chunk, m['topk']), device='cuda', dtype=torch.int32)

            def run():
                flash_mla_sm120.sparse_mla_prefill_fwd(q, kv_packed, indices, sm_scale, m['d_v'])

            ms = bench_fn(run, warmup=5, rep=20)
            f = flops(chunk, nh, m['d_qk'], m['d_v'], m['topk'])
            tflops_val = f / (ms * 1e-3) / 1e12
            tb = total_bytes(chunk, nh, m['d_qk'], m['d_v'], m['topk'], m['bpt'])
            eff_bw = tb / (ms * 1e-3) / 1e12
            bw_pct = eff_bw / BW_TB * 100
            fp8_pct = tflops_val / FP8_TFLOPS * 100

            print(f"  {chunk:>6} {nh:>6}   {ms:>10.3f}   "
                  f"{tflops_val:>6.1f}   {eff_bw:>10.3f}   {bw_pct:>5.1f}% {fp8_pct:>5.1f}%")
        print()

    del kv_packed
    torch.cuda.empty_cache()

    # ── MODEL1 Prefill ──
    for mname, nh_list in [('MODEL1_Flash', [64]), ('MODEL1_Pro', [128])]:
        m = MODELS[mname]
        print("=" * 120)
        print(f"{mname} Prefill (d_qk={m['d_qk']}, topk={m['topk']})")
        print("=" * 120)
        kv_packed = make_kv_cache(m, pool_blocks, block_size, 'cuda')
        s_kv = pool_blocks * block_size
        sm_scale = m['d_qk'] ** -0.5

        print(f"  {'chunk':>6} {'heads':>6} {'latency ms':>12} "
              f"{'TFLOP/s':>8} {'eff BW TB/s':>12} {'% BW':>6} {'% FP8':>6}")
        print("-" * 80)

        for nh in nh_list:
            for chunk in [128, 512, 2048, 4096]:
                q = torch.randn(chunk, nh, m['d_qk'], device='cuda', dtype=torch.bfloat16)
                indices = torch.randint(0, s_kv, (chunk, m['topk']), device='cuda', dtype=torch.int32)

                def run():
                    flash_mla_sm120.sparse_mla_prefill_fwd(q, kv_packed, indices, sm_scale, m['d_v'])

                ms = bench_fn(run, warmup=5, rep=20)
                f = flops(chunk, nh, m['d_qk'], m['d_v'], m['topk'])
                tflops_val = f / (ms * 1e-3) / 1e12
                tb = total_bytes(chunk, nh, m['d_qk'], m['d_v'], m['topk'], m['bpt'])
                eff_bw = tb / (ms * 1e-3) / 1e12
                bw_pct = eff_bw / BW_TB * 100
                fp8_pct = tflops_val / FP8_TFLOPS * 100

                print(f"  {chunk:>6} {nh:>6}   {ms:>10.3f}   "
                      f"{tflops_val:>6.1f}   {eff_bw:>10.3f}   {bw_pct:>5.1f}% {fp8_pct:>5.1f}%")
            print()

        del kv_packed
        torch.cuda.empty_cache()

    # ── V32 Decode (for comparison) ──
    print("=" * 120)
    print("V32 Decode (d_qk=576, topk=2048) — split-KV + combine")
    print("=" * 120)
    m = MODELS['V32']
    kv_packed = make_kv_cache(m, pool_blocks, block_size, 'cuda')
    sm_scale = m['d_qk'] ** -0.5

    print(f"  {'bs':>4} {'heads':>6} {'latency ms':>12} "
          f"{'TFLOP/s':>8} {'eff BW TB/s':>12} {'% BW':>6}")
    print("-" * 80)

    for nh in [16, 128]:
        for bs in [1, 4, 8]:
            q = torch.randn(bs, nh, m['d_qk'], device='cuda', dtype=torch.bfloat16)
            indices = torch.randint(0, s_kv, (bs, m['topk']), device='cuda', dtype=torch.int32)

            def run():
                flash_mla_sm120.sparse_mla_decode_fwd(q, kv_packed, indices, sm_scale, m['d_v'])

            ms = bench_fn(run, warmup=10, rep=50)
            f = flops(bs, nh, m['d_qk'], m['d_v'], m['topk'])
            tflops_val = f / (ms * 1e-3) / 1e12
            tb = total_bytes(bs, nh, m['d_qk'], m['d_v'], m['topk'], m['bpt'])
            eff_bw = tb / (ms * 1e-3) / 1e12
            bw_pct = eff_bw / BW_TB * 100

            print(f"  {bs:>4} {nh:>6}   {ms:>10.3f}   "
                  f"{tflops_val:>6.1f}   {eff_bw:>10.3f}   {bw_pct:>5.1f}%")
        print()


if __name__ == "__main__":
    main()
