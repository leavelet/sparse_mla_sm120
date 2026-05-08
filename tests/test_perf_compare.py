"""Performance comparison: old sparse_mla_sm120 vs new flash_mla_sm120.

Only V32 configs are compared (old package doesn't support MODEL1).
New package also benchmarked for MODEL1 configs.
"""

import torch
import time
import sys
sys.path.insert(0, '.')


def bench_kernel(fn, warmup=50, reps=200, use_cuda_graph=True):
    """Median latency in microseconds. Uses CUDA graph to eliminate launch overhead."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if use_cuda_graph:
        # Capture CUDA graph
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            fn()  # dry run in capture stream
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn()
        torch.cuda.synchronize()

        # Benchmark the graph replay
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            graph.replay()
            ends[i].record()
        torch.cuda.synchronize()
    else:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()

    times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
    return times[reps // 2]  # median in us


def quantize_kv_v32(kv_bf16):
    """V32 FP8 quantization (same as test_decode.py)."""
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    nb, bs, hk, d = kv_bf16.shape
    kv = kv_bf16.squeeze(2)
    bpt = d_nope + num_tiles * 4 + d_rope * 2  # 656
    result = torch.zeros(nb, bs, bpt, dtype=torch.uint8, device=kv.device)
    for ti in range(num_tiles):
        tile = kv[..., ti*tile_size:(ti+1)*tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = torch.pow(2, torch.clamp_min(amax / 448.0, 1e-4).log2().ceil())
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[..., ti*tile_size:(ti+1)*tile_size] = fp8.view(torch.uint8)
        sb = scale.to(torch.float32).contiguous().view(torch.uint8).reshape(nb, bs, 4)
        result[..., d_nope + ti*4 : d_nope + (ti+1)*4] = sb
    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8).reshape(nb, bs, d_rope*2)
    result[..., d_nope + num_tiles*4:] = rope
    return result.view(nb, bs, 1, bpt)


def quantize_kv_model1(kv_bf16):
    """MODEL1 FP8 footer quantization (same as test_decode.py)."""
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2
    scale_bytes = num_tiles + 1
    bpt = data_stride + scale_bytes
    nb, bs, hk, d = kv_bf16.shape
    kv = kv_bf16.squeeze(2)
    block_bytes = bs * bpt
    result_flat = torch.zeros(nb, block_bytes, dtype=torch.uint8, device=kv.device)
    for ti in range(num_tiles):
        tile = kv[..., ti*tile_size:(ti+1)*tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = torch.pow(2, torch.clamp_min(amax / 448.0, 1e-4).log2().ceil())
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        bits = scale.to(torch.float32).view(torch.int32)
        ue8m0 = ((bits >> 23) & 0xFF).to(torch.uint8)
        for tok in range(bs):
            data_off = tok * data_stride + ti * tile_size
            result_flat[:, data_off:data_off+tile_size] = fp8[:, tok].view(torch.uint8)
            scale_off = bs * data_stride + tok * scale_bytes + ti
            result_flat[:, scale_off] = ue8m0[:, tok]
    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8).reshape(nb, bs, d_rope * 2)
    for tok in range(bs):
        rope_off = tok * data_stride + d_nope
        result_flat[:, rope_off:rope_off+d_rope*2] = rope[:, tok]
    return result_flat.view(nb, bs, 1, bpt)


def run_v32_comparison():
    """Compare old sparse_mla_sm120 vs new flash_mla_sm120 on V32."""
    import sparse_mla_sm120
    import flash_mla_sm120

    d_qk, d_v, topk = 576, 512, 2048
    num_blocks, block_size = 64, 64
    sm_scale = d_qk ** -0.5

    torch.manual_seed(42)
    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                            device='cuda', dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_packed = quantize_kv_v32(kv_bf16)

    # Old package uses flat kv: [pool_size, 1, 656]
    kv_flat = kv_packed.view(-1, 1, 656)

    configs = [
        (128, 1, "V3.2 TP1 bs=1"),
        (128, 4, "V3.2 TP1 bs=4"),
        (128, 8, "V3.2 TP1 bs=8"),
        (16, 1, "V3.2 TP8 bs=1"),
        (16, 4, "V3.2 TP8 bs=4"),
        (64, 1, "GLM5.1 TP1 bs=1"),
        (64, 4, "GLM5.1 TP1 bs=4"),
    ]

    print(f"{'Config':<25s} {'Old':>8s} {'New':>8s} {'Speedup':>8s}  {'Old(noCG)':>10s} {'New(noCG)':>10s}")
    print("-" * 75)

    for num_heads, batch_size, label in configs:
        q = torch.randn(batch_size, num_heads, d_qk, device='cuda', dtype=torch.bfloat16) / 10
        indices = torch.randint(0, num_blocks * block_size, (batch_size, topk),
                                device='cuda', dtype=torch.int32)
        indices[:, -10:] = -1

        old_cg = bench_kernel(lambda: sparse_mla_sm120.sparse_mla_decode_fwd(
            q, kv_flat, indices, sm_scale, d_v), use_cuda_graph=True)
        new_cg = bench_kernel(lambda: flash_mla_sm120.sparse_mla_decode_fwd(
            q, kv_packed, indices, sm_scale, d_v), use_cuda_graph=True)
        old_nocg = bench_kernel(lambda: sparse_mla_sm120.sparse_mla_decode_fwd(
            q, kv_flat, indices, sm_scale, d_v), use_cuda_graph=False)
        new_nocg = bench_kernel(lambda: flash_mla_sm120.sparse_mla_decode_fwd(
            q, kv_packed, indices, sm_scale, d_v), use_cuda_graph=False)

        speedup = old_cg / new_cg
        print(f"{label:<25s} {old_cg:>7.1f} {new_cg:>7.1f} {speedup:>7.2f}x  {old_nocg:>10.1f} {new_nocg:>10.1f}")


def run_model1_perf():
    """Benchmark new flash_mla_sm120 on MODEL1 configs (no old baseline)."""
    import flash_mla_sm120

    d_qk, d_v = 512, 512
    num_blocks, block_size = 64, 64
    sm_scale = d_qk ** -0.5

    torch.manual_seed(42)
    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                            device='cuda', dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_packed = quantize_kv_model1(kv_bf16)

    configs = [
        (64, 512, 1, "V4 Flash TP1 bs=1"),
        (64, 512, 4, "V4 Flash TP1 bs=4"),
        (64, 512, 8, "V4 Flash TP1 bs=8"),
        (128, 1024, 1, "V4 Pro TP1 bs=1"),
        (128, 1024, 4, "V4 Pro TP1 bs=4"),
    ]

    print(f"\n{'Config':<25s} {'CG (us)':>8s} {'noCG (us)':>10s}")
    print("-" * 48)

    for num_heads, topk, batch_size, label in configs:
        q = torch.randn(batch_size, num_heads, d_qk, device='cuda', dtype=torch.bfloat16) / 10
        indices = torch.randint(0, num_blocks * block_size, (batch_size, topk),
                                device='cuda', dtype=torch.int32)
        indices[:, -10:] = -1

        cg = bench_kernel(lambda: flash_mla_sm120.sparse_mla_decode_fwd(
            q, kv_packed, indices, sm_scale, d_v), use_cuda_graph=True)
        nocg = bench_kernel(lambda: flash_mla_sm120.sparse_mla_decode_fwd(
            q, kv_packed, indices, sm_scale, d_v), use_cuda_graph=False)

        print(f"{label:<25s} {cg:>7.1f} {nocg:>10.1f}")


def flops_prefill(num_tokens, num_heads, d_qk, d_v, topk):
    return 2.0 * num_tokens * num_heads * topk * (d_qk + d_v)


def run_v32_prefill_comparison():
    """Compare old sparse_mla_sm120 vs new flash_mla_sm120 on V32 prefill."""
    import sparse_mla_sm120
    import flash_mla_sm120

    d_qk, d_v, topk = 576, 512, 2048
    num_blocks, block_size = 64, 64
    sm_scale = d_qk ** -0.5
    s_kv = num_blocks * block_size

    torch.manual_seed(42)
    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                            device='cuda', dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_packed = quantize_kv_v32(kv_bf16)
    kv_flat = kv_packed.view(-1, 1, 656)

    configs = [
        (128, 128, "V3.2 TP1 chunk=128"),
        (128, 512, "V3.2 TP1 chunk=512"),
        (128, 2048, "V3.2 TP1 chunk=2048"),
        (128, 4096, "V3.2 TP1 chunk=4096"),
        (64, 128, "GLM5.1 TP1 chunk=128"),
        (64, 2048, "GLM5.1 TP1 chunk=2048"),
        (16, 128, "V3.2 TP8 chunk=128"),
        (16, 2048, "V3.2 TP8 chunk=2048"),
    ]

    print(f"{'Config':<30s} {'Old (us)':>10s} {'New (us)':>10s} {'Speedup':>8s} "
          f"{'Old TFLOP/s':>12s} {'New TFLOP/s':>12s}")
    print("-" * 96)

    for num_heads, chunk, label in configs:
        q = torch.randn(chunk, num_heads, d_qk, device='cuda', dtype=torch.bfloat16) / 10
        indices = torch.randint(0, s_kv, (chunk, topk), device='cuda', dtype=torch.int32)
        indices[:, -10:] = -1

        old_us = bench_kernel(lambda: sparse_mla_sm120.sparse_mla_prefill_fwd(
            q, kv_flat, indices, sm_scale, d_v), use_cuda_graph=False)
        new_us = bench_kernel(lambda: flash_mla_sm120.sparse_mla_prefill_fwd(
            q, kv_packed, indices, sm_scale, d_v), use_cuda_graph=False)

        f = flops_prefill(chunk, num_heads, d_qk, d_v, topk)
        old_tflops = f / (old_us * 1e-6) / 1e12
        new_tflops = f / (new_us * 1e-6) / 1e12
        speedup = old_us / new_us

        print(f"{label:<30s} {old_us:>9.1f} {new_us:>9.1f} {speedup:>7.2f}x "
              f"{old_tflops:>11.1f} {new_tflops:>11.1f}")


def run_model1_prefill_perf():
    """Benchmark new flash_mla_sm120 on MODEL1 prefill (no old baseline)."""
    import flash_mla_sm120

    d_qk, d_v = 512, 512
    num_blocks, block_size = 64, 64
    sm_scale = d_qk ** -0.5
    s_kv = num_blocks * block_size

    torch.manual_seed(42)
    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                            device='cuda', dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_packed = quantize_kv_model1(kv_bf16)

    configs = [
        (64, 512, 128, "V4 Flash chunk=128"),
        (64, 512, 2048, "V4 Flash chunk=2048"),
        (64, 512, 4096, "V4 Flash chunk=4096"),
        (128, 1024, 128, "V4 Pro chunk=128"),
        (128, 1024, 2048, "V4 Pro chunk=2048"),
        (128, 1024, 4096, "V4 Pro chunk=4096"),
    ]

    print(f"\n{'Config':<30s} {'us':>10s} {'TFLOP/s':>10s}")
    print("-" * 55)

    for num_heads, topk, chunk, label in configs:
        q = torch.randn(chunk, num_heads, d_qk, device='cuda', dtype=torch.bfloat16) / 10
        indices = torch.randint(0, s_kv, (chunk, topk), device='cuda', dtype=torch.int32)
        indices[:, -10:] = -1

        us = bench_kernel(lambda: flash_mla_sm120.sparse_mla_prefill_fwd(
            q, kv_packed, indices, sm_scale, d_v), use_cuda_graph=False)

        f = flops_prefill(chunk, num_heads, d_qk, d_v, topk)
        tflops = f / (us * 1e-6) / 1e12

        print(f"{label:<30s} {us:>9.1f} {tflops:>9.1f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Performance Comparison: sparse_mla_sm120 (old) vs flash_mla_sm120 (new)")
    print("=" * 60)
    print("\n--- V32 Decode (old vs new) ---")
    run_v32_comparison()

    print("\n--- MODEL1 Decode (new only) ---")
    run_model1_perf()

    print("\n--- V32 Prefill (old vs new) ---")
    run_v32_prefill_comparison()

    print("\n--- MODEL1 Prefill (new only) ---")
    run_model1_prefill_perf()
