"""Benchmark sparse MLA prefill kernel."""

import sys
import torch

sys.path.insert(0, "tests")
from test_sparse_mla import pack_kv_cache_fp8
from sparse_mla_sm120.ops import sparse_mla_prefill_fwd

torch.manual_seed(42)

pool_size = 65536
topk = 2048
d_v = 512
dim = 576
sm_scale = 1.0 / (dim**0.5)

kv_bf16 = torch.randn(pool_size, 1, dim, dtype=torch.bfloat16, device="cuda")
kv = pack_kv_cache_fp8(kv_bf16)
del kv_bf16


def bench(fn, warmup=5, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    return times[len(times) // 2]


header = f"{'Heads':>5} {'Tokens':>6} {'ms':>8} {'TFLOP/s':>8}"
print(header)
print("-" * len(header))

for num_heads in [16, 128]:
    for num_tokens in [512, 2048, 4096]:
        q = torch.randn(num_tokens, num_heads, dim, dtype=torch.bfloat16, device="cuda")
        idx = torch.randint(
            0, pool_size, (num_tokens, topk), dtype=torch.int32, device="cuda"
        )
        flops = 2.0 * num_tokens * num_heads * topk * (dim + d_v)
        ms = bench(lambda: sparse_mla_prefill_fwd(q, kv, idx, sm_scale))
        tflops = flops / ms / 1e9
        print(f"{num_heads:>5} {num_tokens:>6} {ms:>8.3f} {tflops:>8.1f}")
    print()
