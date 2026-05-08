"""
Micro benchmark: FP8 XV vs BF16 XV per V-chunk timing.

Uses the actual kernel to measure XV phase cost by comparing MODEL1 configs
with different N_V_CHUNKS. The per-chunk overhead is:
  FP8:  V_transpose + W_quantize(atomicMax+normalize+FP8) + FP8_MMA(2 QMMA)
  BF16: dequant(FP8→BF16) + W_bf16_store + BF16_MMA(4 HMMA)

We compare the current code (BF16 XV for MODEL1) vs the FP8 XV baseline.
Since we can't easily switch at runtime, we measure the full kernel and
extract per-tile and per-chunk estimates.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from test_decode import quantize_kv_model1, quantize_kv_v32
import flash_mla_sm120

def bench(fn, warmup=50, reps=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # us
    times.sort()
    return times[len(times)//2]

def profile_kernel(model, d_qk, topk, num_heads, chunk, block_size=64, num_blocks=64):
    torch.manual_seed(42)
    sm_scale = d_qk ** -0.5
    s_kv = num_blocks * block_size

    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                            device='cuda', dtype=torch.bfloat16) / 10).clamp(-1, 1)
    if model == "V32":
        kv = quantize_kv_v32(kv_bf16)
    else:
        kv = quantize_kv_model1(kv_bf16)

    q = torch.randn(chunk, num_heads, d_qk, device='cuda', dtype=torch.bfloat16) / 10
    indices = torch.randint(0, s_kv, (chunk, topk), device='cuda', dtype=torch.int32)
    indices[:, -10:] = -1

    if model == "V32":
        fn = lambda: flash_mla_sm120.sparse_mla_decode_fwd(q, kv, indices, sm_scale, 512)
    else:
        fn = lambda: flash_mla_sm120.sparse_mla_decode_fwd(q, kv, indices, sm_scale, 512)

    us = bench(fn)
    return us

def main():
    print("="*70)
    print("XV Phase Micro Analysis: FP8 vs BF16 overhead per V-chunk")
    print("="*70)

    # Decode: all use SG kernel. Compare MODEL1 (BF16 XV) vs V32 (FP8 XV)
    # Both have similar structure, but MODEL1 has N_V_CHUNKS=7, V32 has N_V_CHUNKS=4
    # And MODEL1 uses BF16 XV, V32 uses FP8 XV

    # MODEL1 decode configs (BF16 XV path)
    print("\n--- MODEL1 Decode (BF16 XV, N_V_CHUNKS=7) ---")
    for nh, topk in [(64, 512), (128, 1024)]:
        for bs in [1, 4]:
            us = profile_kernel("MODEL1", 512, topk, nh, bs)
            ni = topk // 64
            print(f"  h={nh} topk={topk} bs={bs}: {us:.1f} us  "
                  f"(NI={ni}, ~{us/ni:.1f} us/tile)")

    # V32 decode configs (FP8 XV path) for comparison
    print("\n--- V32 Decode (FP8 XV, N_V_CHUNKS=4) ---")
    for nh in [128, 16]:
        for bs in [1, 4]:
            us = profile_kernel("V32", 576, 2048, nh, bs)
            ni = 2048 // 64
            print(f"  h={nh} topk=2048 bs={bs}: {us:.1f} us  "
                  f"(NI={ni}, ~{us/ni:.1f} us/tile)")

    # Prefill: MODEL1 uses MG (BF16 XV), V32 uses MG (FP8 XV)
    print("\n--- MODEL1 Prefill MG (BF16 XV) ---")
    for nh, topk, chunk in [(64, 512, 128), (128, 1024, 128)]:
        torch.manual_seed(42)
        kv_bf16 = (torch.randn(64, 64, 1, 512, device='cuda', dtype=torch.bfloat16) / 10).clamp(-1, 1)
        kv = quantize_kv_model1(kv_bf16)
        q = torch.randn(chunk, nh, 512, device='cuda', dtype=torch.bfloat16) / 10
        idx = torch.randint(0, 4096, (chunk, topk), device='cuda', dtype=torch.int32)
        idx[:, -10:] = -1

        fn = lambda: flash_mla_sm120.sparse_mla_prefill_fwd(q, kv, idx, 512**-0.5, 512)
        us = bench(fn)
        ni = topk // 64
        mg_heads = 32
        ctas = chunk * (nh // mg_heads)
        waves = ctas / 188
        per_cta = us / max(waves, 1)
        per_tile = per_cta / ni

        flops = 2.0 * chunk * nh * topk * (512 + 512)
        tflops = flops / (us * 1e-6) / 1e12
        print(f"  h={nh} topk={topk} chunk={chunk}: {us:.1f} us, {tflops:.1f} TFLOP/s  "
              f"(NI={ni}, ~{per_tile:.1f} us/tile/CTA)")

    print("\n--- V32 Prefill MG (FP8 XV) ---")
    for nh, chunk in [(128, 128), (64, 128)]:
        torch.manual_seed(42)
        kv_bf16 = (torch.randn(64, 64, 1, 576, device='cuda', dtype=torch.bfloat16) / 10).clamp(-1, 1)
        kv = quantize_kv_v32(kv_bf16)
        q = torch.randn(chunk, nh, 576, device='cuda', dtype=torch.bfloat16) / 10
        idx = torch.randint(0, 4096, (chunk, 2048), device='cuda', dtype=torch.int32)
        idx[:, -10:] = -1

        fn = lambda: flash_mla_sm120.sparse_mla_prefill_fwd(q, kv, idx, 576**-0.5, 512)
        us = bench(fn)
        ni = 2048 // 64
        mg_heads = 32
        ctas = chunk * (nh // mg_heads)
        waves = ctas / 188
        per_cta = us / max(waves, 1)
        per_tile = per_cta / ni

        flops = 2.0 * chunk * nh * 2048 * (576 + 512)
        tflops = flops / (us * 1e-6) / 1e12
        print(f"  h={nh} topk=2048 chunk={chunk}: {us:.1f} us, {tflops:.1f} TFLOP/s  "
              f"(NI={ni}, ~{per_tile:.1f} us/tile/CTA)")

    # Summary: compute per-V-chunk cost
    print("\n" + "="*70)
    print("Per V-chunk cost estimate:")
    print("  FP8 XV (V32):  V_transpose + W_quantize + 2×QMMA per warp per group")
    print("  BF16 XV (M1):  dequant(4096 FP8→BF16) + W_bf16_store + 4×HMMA per warp per group")
    print("  QMMA m16n8k32: ~2 cycles issue latency")
    print("  HMMA m16n8k16: ~2 cycles issue latency")
    print("  V_transpose(64×64 bytes): ~800 SHF+LOP3+STS instructions")
    print("  W_quantize: atomicMax + normalize + FP8 clamp+convert")
    print("  Dequant(4096): ~5 instructions × 16 elements/thread = 80 inst/thread")

if __name__ == "__main__":
    main()
