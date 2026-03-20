"""
Benchmark FP8 MQA logits: CUDA kernel vs Triton kernel.

Side-by-side latency (us), TFLOP/s, effective bandwidth (GB/s), and speedup.
DeepSeek V3.2 configs: num_heads in {16, 128}, head_dim = 128.
"""
import torch
import argparse

import sparse_mla_sm120._C as _C
from sparse_mla_sm120.triton_mqa_logits import (
    fp8_mqa_logits_ragged_triton,
    fp8_paged_mqa_logits_triton,
)


def benchmark_fn(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


def make_fp8(shape, scale=0.5):
    return (torch.randn(shape, device="cuda") * scale).to(torch.float8_e4m3fn)


def compute_flops_ragged(seq_q, seq_kv, num_heads, D=128):
    block_kv = 128
    num_kv_tiles = (seq_kv + block_kv - 1) // block_kv
    n_tiles = block_kv // 8
    flops_per_tile = (num_heads // 16) * (D // 32) * n_tiles * 16 * 8 * 32 * 2
    return seq_q * num_kv_tiles * flops_per_tile


def compute_bytes_ragged(seq_q, seq_kv, num_heads, D=128):
    """Effective bytes touched (DRAM-level, assuming KV fits in L2 for reuse)."""
    q_bytes = seq_q * num_heads * D          # fp8, read once
    kv_bytes = seq_kv * D                    # fp8, read once into L2, reused across q tokens
    kv_scale_bytes = seq_kv * 4              # fp32, read once
    w_bytes = seq_q * num_heads * 4          # fp32, read once
    out_bytes = seq_q * seq_kv * 4           # fp32, write once
    return q_bytes + kv_bytes + kv_scale_bytes + w_bytes + out_bytes


def compute_flops_paged(total_q, num_pages, num_heads, D=128):
    page_size = 64
    n_tiles = page_size // 8
    flops_per_page = (num_heads // 16) * (D // 32) * n_tiles * 16 * 8 * 32 * 2
    return total_q * num_pages * flops_per_page


def compute_bytes_paged(batch, ctx_len, num_heads, D=128):
    page_size = 64
    num_pages = (ctx_len + page_size - 1) // page_size
    total_q = batch
    q_bytes = total_q * num_heads * D
    kv_bytes = num_pages * page_size * D * batch   # each batch reads its own pages
    kv_scale_bytes = num_pages * page_size * 4 * batch
    w_bytes = total_q * num_heads * 4
    out_bytes = total_q * ctx_len * 4
    return q_bytes + kv_bytes + kv_scale_bytes + w_bytes + out_bytes


def bench_ragged(seq_q, seq_kv, num_heads, D=128):
    q = make_fp8((seq_q, num_heads, D))
    kv = make_fp8((seq_kv, D))
    kv_scale = torch.rand(seq_kv, device="cuda") * 0.05 + 0.01
    w = torch.randn(seq_q, num_heads, device="cuda") * 0.1
    k_start = torch.zeros(seq_q, dtype=torch.int32, device="cuda")
    k_end = torch.full((seq_q,), seq_kv, dtype=torch.int32, device="cuda")

    align = 256
    stride = ((seq_kv + align - 1) // align) * align

    logits_cuda = torch.zeros(seq_q, stride, dtype=torch.float32, device="cuda")
    out_cuda = logits_cuda[:, :seq_kv]

    def run_cuda():
        out_cuda.zero_()
        _C.fp8_mqa_logits_ragged_fwd(q, kv, kv_scale, w, k_start, k_end, out_cuda, seq_kv)

    us_cuda = benchmark_fn(run_cuda)

    def run_triton():
        fp8_mqa_logits_ragged_triton(q, kv, kv_scale, w, k_start, k_end, max_seqlen_k=seq_kv)

    us_triton = benchmark_fn(run_triton)

    total_flops = compute_flops_ragged(seq_q, seq_kv, num_heads, D)
    total_bytes = compute_bytes_ragged(seq_q, seq_kv, num_heads, D)
    tflops_cuda   = total_flops / 1e12 / (us_cuda / 1e6)
    tflops_triton = total_flops / 1e12 / (us_triton / 1e6)
    bw_cuda       = total_bytes / 1e9 / (us_cuda / 1e6)
    bw_triton     = total_bytes / 1e9 / (us_triton / 1e6)

    return us_cuda, tflops_cuda, bw_cuda, us_triton, tflops_triton, bw_triton


def bench_paged(batch, next_n, ctx_len, num_heads, D=128):
    page_size = 64
    num_pages = (ctx_len + page_size - 1) // page_size
    total_pages = batch * num_pages
    total_q = batch * next_n

    q = make_fp8((total_q, num_heads, D))
    kv = make_fp8((total_pages, page_size, D))
    kv_scale = torch.rand(total_pages, page_size, device="cuda") * 0.05 + 0.01
    w = torch.randn(total_q, num_heads, device="cuda") * 0.1
    ctx_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device="cuda")

    block_table = torch.zeros(batch, num_pages, dtype=torch.int32, device="cuda")
    for b in range(batch):
        for p in range(num_pages):
            block_table[b, p] = b * num_pages + p

    align = 256
    stride = ((ctx_len + align - 1) // align) * align

    logits_cuda = torch.zeros(total_q, stride, dtype=torch.float32, device="cuda")
    out_cuda = logits_cuda[:, :ctx_len]

    def run_cuda():
        out_cuda.zero_()
        _C.fp8_mqa_logits_paged_fwd(q, kv, kv_scale, w, ctx_lens, block_table, out_cuda, next_n)

    us_cuda = benchmark_fn(run_cuda)

    def run_triton():
        fp8_paged_mqa_logits_triton(
            q, kv, kv_scale, w, ctx_lens, block_table, ctx_len, next_n=next_n)

    us_triton = benchmark_fn(run_triton)

    total_flops = compute_flops_paged(total_q, num_pages, num_heads, D)
    total_bytes = compute_bytes_paged(batch, ctx_len, num_heads, D)
    tflops_cuda   = total_flops / 1e12 / (us_cuda / 1e6)
    tflops_triton = total_flops / 1e12 / (us_triton / 1e6)
    bw_cuda       = total_bytes / 1e9 / (us_cuda / 1e6)
    bw_triton     = total_bytes / 1e9 / (us_triton / 1e6)

    return us_cuda, tflops_cuda, bw_cuda, us_triton, tflops_triton, bw_triton


def main():
    parser = argparse.ArgumentParser(description="Benchmark FP8 MQA logits: CUDA vs Triton")
    parser.add_argument("--mode", choices=["ragged", "paged", "all"], default="all")
    args = parser.parse_args()

    torch.manual_seed(42)
    dev = torch.cuda.get_device_name()
    print(f"Device: {dev}\n")

    hdr_ragged = (
        f"{'seq_q':>6} {'seq_kv':>7} {'H':>4} │"
        f"{'CUDA μs':>9} {'TF/s':>7} {'GB/s':>7} │"
        f"{'Triton μs':>10} {'TF/s':>7} {'GB/s':>7} │"
        f"{'speedup':>8}"
    )
    hdr_paged = (
        f"{'batch':>6} {'ctx':>7} {'H':>4} │"
        f"{'CUDA μs':>9} {'TF/s':>7} {'GB/s':>7} │"
        f"{'Triton μs':>10} {'TF/s':>7} {'GB/s':>7} │"
        f"{'speedup':>8}"
    )

    if args.mode in ("ragged", "all"):
        print("=" * 96)
        print("  RAGGED FP8 MQA LOGITS  (CUDA vs Triton)")
        print("=" * 96)
        print(hdr_ragged)
        print("─" * 96)

        for num_heads in [16, 128]:
            for seq_q in [1, 4, 16]:
                for seq_kv in [1024, 2048, 4096, 8192, 16384]:
                    uc, tc, bc, ut, tt, bt = bench_ragged(seq_q, seq_kv, num_heads)
                    sp = ut / uc
                    print(
                        f"{seq_q:>6} {seq_kv:>7} {num_heads:>4} │"
                        f"{uc:>9.1f} {tc:>7.2f} {bc:>7.1f} │"
                        f"{ut:>10.1f} {tt:>7.2f} {bt:>7.1f} │"
                        f"{sp:>7.2f}x"
                    )
            print()

    if args.mode in ("paged", "all"):
        print("=" * 96)
        print("  PAGED FP8 MQA LOGITS  (CUDA vs Triton)")
        print("=" * 96)
        print(hdr_paged)
        print("─" * 96)

        for num_heads in [16, 128]:
            for batch in [1, 8, 32]:
                for ctx_len in [1024, 4096, 8192, 16384]:
                    uc, tc, bc, ut, tt, bt = bench_paged(batch, 1, ctx_len, num_heads)
                    sp = ut / uc
                    print(
                        f"{batch:>6} {ctx_len:>7} {num_heads:>4} │"
                        f"{uc:>9.1f} {tc:>7.2f} {bc:>7.1f} │"
                        f"{ut:>10.1f} {tt:>7.2f} {bt:>7.1f} │"
                        f"{sp:>7.2f}x"
                    )
            print()


if __name__ == "__main__":
    main()
