"""Sweep decode split configurations for sparse MLA."""

from __future__ import annotations

import argparse

import torch

import sparse_mla_sm120._C as _C

from profile_decode import (
    D_ROPE,
    D_V,
    DIM,
    HPB,
    MAX_TOKENS,
    TOPK,
    bench,
    make_indices,
    pack_kv_cache_fp8,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--total-len", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--index-pattern", choices=["random", "sequential", "clustered"], default="random")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    q = torch.randn(args.num_tokens, args.num_heads, DIM, device="cuda", dtype=torch.bfloat16)
    kv_cache = pack_kv_cache_fp8(torch.randn(args.total_len, 1, DIM, device="cuda", dtype=torch.bfloat16))
    indices = make_indices(args.num_tokens, args.total_len, args.index_pattern)
    output = torch.empty((args.num_tokens, args.num_heads, D_V), dtype=torch.bfloat16, device="cuda")
    sm_scale = DIM ** -0.5

    best = None
    for tiles_per_split in (2, 4, 8, 16, 32):
        if (TOPK // 64) % tiles_per_split != 0:
            continue
        nsplits = (TOPK // 64) // tiles_per_split
        partial_O = torch.empty((MAX_TOKENS, nsplits, args.num_heads, D_V), dtype=torch.bfloat16, device="cuda")
        partial_LSE = torch.empty((MAX_TOKENS, nsplits, args.num_heads), dtype=torch.float32, device="cuda")
        semaphores = torch.zeros((MAX_TOKENS * (args.num_heads // HPB),), dtype=torch.int32, device="cuda")

        def run() -> None:
            _C.sparse_mla_decode_fwd(
                q,
                kv_cache,
                indices,
                output,
                partial_O,
                partial_LSE,
                semaphores,
                sm_scale,
                D_V,
                D_ROPE,
                TOPK,
                tiles_per_split,
                nsplits,
            )

        median_ms = bench(run, warmup=args.warmup, rep=args.repeat)
        print(f"tiles_per_split={tiles_per_split:2d} nsplits={nsplits:2d} median_ms={median_ms:.6f}")
        if best is None or median_ms < best[1]:
            best = ((tiles_per_split, nsplits), median_ms)

    assert best is not None
    print(
        f"best tiles_per_split={best[0][0]} nsplits={best[0][1]} median_ms={best[1]:.6f}"
    )


if __name__ == "__main__":
    main()
