"""Mimic the exact tensor shapes vLLM passes to flash_mla_with_kvcache and
flash_mla_sparse_fwd, and verify output is non-zero."""
import torch

from flash_mla_sm120.interface import flash_mla_with_kvcache, flash_mla_sparse_fwd, FlashMLASchedMeta
from tests.test_decode import quantize_kv_model1, dequantize_kv_model1


def _make_paged_kv(num_blocks: int, block_size: int, d_qk: int, k_scale: float):
    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                            device="cuda", dtype=torch.bfloat16) * k_scale
              ).clamp(-k_scale * 5, k_scale * 5)
    return quantize_kv_model1(kv_bf16)


def test_decode_shapes_swa_only():
    """compress_ratio=1 layer: swa-only, no extra cache."""
    torch.manual_seed(0)
    num_tokens, num_heads, d_qk, d_v = 8, 128, 512, 512
    main_block, main_nb = 64, 64
    window = 128
    sm_scale = d_qk ** -0.5

    # vLLM shape conventions (per deepseek_v4_attention.py:856 q=q.unsqueeze(1)):
    q = (torch.randn(num_tokens, 1, num_heads, d_qk, device="cuda", dtype=torch.bfloat16)).clamp(-3, 3)
    kv_packed = _make_paged_kv(main_nb, main_block, d_qk, 1.0)
    # swa_cache_layer.kv_cache.unsqueeze(-2) → (nb, bs, 1, head_bytes)
    swa_cache = kv_packed  # already (nb, bs, 1, hb)
    # decode_swa_indices: (max_tokens, 1, window_size)
    swa_indices = torch.randint(0, main_nb * main_block, (num_tokens, 1, window), dtype=torch.int32, device="cuda")
    swa_lens = torch.full((num_tokens,), window, dtype=torch.int32, device="cuda")
    attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 2.0
    # output: (num_tokens, num_heads, d_v), then .unsqueeze(1) gives 4D view
    output = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device="cuda")

    out, _ = flash_mla_with_kvcache(
        q=q,
        k_cache=swa_cache,
        block_table=None,
        head_dim_v=d_v,
        tile_scheduler_metadata=FlashMLASchedMeta(),
        cache_seqlens=None,
        is_fp8_kvcache=True,
        indices=swa_indices,
        topk_length=swa_lens,
        softmax_scale=sm_scale,
        attn_sink=attn_sink,
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        extra_topk_length=None,
        out=output.unsqueeze(1),
    )

    print(f"  swa-only: out shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()} "
          f"abs_mean={out.abs().float().mean().item():.4e} nz={(out != 0).float().mean().item():.4f}")
    print(f"  output (via out= side-effect): abs_mean={output.abs().float().mean().item():.4e}")
    assert torch.isfinite(output).all()
    assert (output != 0).float().mean().item() > 0.5, "output mostly zero — bug!"


def test_decode_shapes_c4a():
    """compress_ratio=4 layer: dual-cache C4A."""
    torch.manual_seed(0)
    num_tokens, num_heads, d_qk, d_v = 8, 128, 512, 512
    main_block, main_nb = 64, 64
    window, topk = 128, 512
    sm_scale = d_qk ** -0.5

    q = (torch.randn(num_tokens, 1, num_heads, d_qk, device="cuda", dtype=torch.bfloat16)).clamp(-3, 3)
    swa_cache = _make_paged_kv(main_nb, main_block, d_qk, 1.0)
    compressed_cache = _make_paged_kv(main_nb, main_block, d_qk, 1.0)  # C4A: same block size
    swa_indices = torch.randint(0, main_nb * main_block, (num_tokens, 1, window), dtype=torch.int32, device="cuda")
    swa_lens = torch.full((num_tokens,), window, dtype=torch.int32, device="cuda")
    # C4A: topk_indices = global_indices.view(num_decode_tokens, 1, -1)
    topk_indices = torch.randint(0, main_nb * main_block, (num_tokens, 1, topk), dtype=torch.int32, device="cuda")
    topk_lens = torch.full((num_tokens,), topk, dtype=torch.int32, device="cuda")
    attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 2.0
    output = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device="cuda")

    out, _ = flash_mla_with_kvcache(
        q=q,
        k_cache=swa_cache,
        block_table=None,
        head_dim_v=d_v,
        tile_scheduler_metadata=FlashMLASchedMeta(),
        cache_seqlens=None,
        is_fp8_kvcache=True,
        indices=swa_indices,
        topk_length=swa_lens,
        softmax_scale=sm_scale,
        attn_sink=attn_sink,
        extra_k_cache=compressed_cache,
        extra_indices_in_kvcache=topk_indices,
        extra_topk_length=topk_lens,
        out=output.unsqueeze(1),
    )

    print(f"  c4a: out shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()} "
          f"abs_mean={out.abs().float().mean().item():.4e} nz={(out != 0).float().mean().item():.4f}")
    print(f"  output (via out= side-effect): abs_mean={output.abs().float().mean().item():.4e}")
    assert torch.isfinite(output).all()
    assert (output != 0).float().mean().item() > 0.5, "output mostly zero — bug!"


def test_decode_shapes_c128a():
    """compress_ratio=128: extra cache block_size=2."""
    torch.manual_seed(0)
    num_tokens, num_heads, d_qk, d_v = 8, 128, 512, 512
    main_block, main_nb = 64, 64
    extra_block, extra_nb = 2, 4096  # block_size=2 for C128A
    window, topk = 128, 512
    sm_scale = d_qk ** -0.5

    q = (torch.randn(num_tokens, 1, num_heads, d_qk, device="cuda", dtype=torch.bfloat16)).clamp(-3, 3)
    swa_cache = _make_paged_kv(main_nb, main_block, d_qk, 1.0)
    compressed_cache = _make_paged_kv(extra_nb, extra_block, d_qk, 1.0)
    swa_indices = torch.randint(0, main_nb * main_block, (num_tokens, 1, window), dtype=torch.int32, device="cuda")
    swa_lens = torch.full((num_tokens,), window, dtype=torch.int32, device="cuda")
    topk_indices = torch.randint(0, extra_nb * extra_block, (num_tokens, 1, topk), dtype=torch.int32, device="cuda")
    topk_lens = torch.full((num_tokens,), topk, dtype=torch.int32, device="cuda")
    attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 2.0
    output = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device="cuda")

    out, _ = flash_mla_with_kvcache(
        q=q,
        k_cache=swa_cache,
        block_table=None,
        head_dim_v=d_v,
        tile_scheduler_metadata=FlashMLASchedMeta(),
        cache_seqlens=None,
        is_fp8_kvcache=True,
        indices=swa_indices,
        topk_length=swa_lens,
        softmax_scale=sm_scale,
        attn_sink=attn_sink,
        extra_k_cache=compressed_cache,
        extra_indices_in_kvcache=topk_indices,
        extra_topk_length=topk_lens,
        out=output.unsqueeze(1),
    )

    print(f"  c128a: out shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()} "
          f"abs_mean={out.abs().float().mean().item():.4e} nz={(out != 0).float().mean().item():.4f}")
    print(f"  output (via out= side-effect): abs_mean={output.abs().float().mean().item():.4e}")
    assert torch.isfinite(output).all()
    assert (output != 0).float().mean().item() > 0.5, "output mostly zero — bug!"


def test_decode_small_extra_topk_length():
    """Reproduce the e2e flat-logits bug: small/zero extra_topk_length."""
    torch.manual_seed(0)
    num_tokens, num_heads, d_qk, d_v = 4, 128, 512, 512
    main_block, main_nb = 64, 64
    extra_block, extra_nb = 2, 4096
    window, topk = 128, 512
    sm_scale = d_qk ** -0.5

    q = (torch.randn(num_tokens, 1, num_heads, d_qk, device="cuda", dtype=torch.bfloat16)).clamp(-3, 3)
    swa_cache = _make_paged_kv(main_nb, main_block, d_qk, 1.0)
    compressed_cache = _make_paged_kv(extra_nb, extra_block, d_qk, 1.0)
    swa_indices = torch.randint(0, main_nb * main_block, (num_tokens, 1, window), dtype=torch.int32, device="cuda")
    topk_indices = torch.randint(0, extra_nb * extra_block, (num_tokens, 1, topk), dtype=torch.int32, device="cuda")
    attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 2.0

    for swa_len_val, extra_len_val in [(1, 0), (2, 0), (3, 0), (4, 1), (4, 2), (32, 8), (128, 64)]:
        swa_lens = torch.full((num_tokens,), swa_len_val, dtype=torch.int32, device="cuda")
        topk_lens = torch.full((num_tokens,), extra_len_val, dtype=torch.int32, device="cuda")
        output = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device="cuda")

        out, _ = flash_mla_with_kvcache(
            q=q, k_cache=swa_cache, block_table=None, head_dim_v=d_v,
            tile_scheduler_metadata=FlashMLASchedMeta(), cache_seqlens=None,
            is_fp8_kvcache=True, indices=swa_indices, topk_length=swa_lens,
            softmax_scale=sm_scale, attn_sink=attn_sink,
            extra_k_cache=compressed_cache, extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens, out=output.unsqueeze(1),
        )
        am = output.abs().float().mean().item()
        nz = (output != 0).float().mean().item()
        fin = torch.isfinite(output).all().item()
        status = "OK" if nz > 0.5 and fin else "BAD"
        print(f"  swa_len={swa_len_val:3d} extra_len={extra_len_val:3d}: finite={fin} abs_mean={am:.4e} nz={nz:.4f} [{status}]")


def test_prefill_varying_topk_length():
    """Reproduce e2e bug: 4-token prefill chunk where token i has topk_length=i+1
    (causal: each token sees one more token of context than the prior)."""
    torch.manual_seed(0)
    num_tokens, num_heads, d_qk, d_v = 4, 128, 512, 512
    main_block, main_nb = 64, 64
    window = 128
    sm_scale = d_qk ** -0.5

    # vLLM prefill: q is 3D
    q = (torch.randn(num_tokens, num_heads, d_qk, device="cuda", dtype=torch.bfloat16)).clamp(-3, 3)
    kv_packed = _make_paged_kv(main_nb, main_block, d_qk, 1.0)
    swa_kv_paged = kv_packed
    indices = torch.randint(0, main_nb * main_block, (num_tokens, 1, window), dtype=torch.int32, device="cuda")
    # Causal: position i attends to i+1 tokens
    swa_lens = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 2.0
    output = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device="cuda")

    out, _, _ = flash_mla_sparse_fwd(
        q=q, kv=swa_kv_paged, indices=indices, sm_scale=sm_scale,
        attn_sink=attn_sink, topk_length=swa_lens, out=output,
    )

    # Per-token output non-zero check
    for i in range(num_tokens):
        am = output[i].abs().float().mean().item()
        nz = (output[i] != 0).float().mean().item()
        fin = torch.isfinite(output[i]).all().item()
        status = "OK" if nz > 0.5 and fin else "FLAT/ZERO"
        print(f"  token {i} (topk_len={swa_lens[i].item()}): finite={fin} abs_mean={am:.4e} nz={nz:.4f} [{status}]")
    assert torch.isfinite(output).all()
    assert (output != 0).float().mean().item() > 0.5, "output mostly zero — bug!"


def test_prefill_shapes():
    """vLLM prefill path via flash_mla_sparse_fwd."""
    torch.manual_seed(0)
    num_tokens, num_heads, d_qk, d_v = 32, 128, 512, 512
    main_block, main_nb = 64, 64
    window = 128
    sm_scale = d_qk ** -0.5

    # vLLM prefill: q is 3D (num_tokens, num_heads, d_qk), no s_q axis
    q = (torch.randn(num_tokens, num_heads, d_qk, device="cuda", dtype=torch.bfloat16)).clamp(-3, 3)
    kv_packed = _make_paged_kv(main_nb, main_block, d_qk, 1.0)
    swa_kv_paged = kv_packed
    # prefill_swa_indices: (max_tokens, 1, window_size)
    indices = torch.randint(0, main_nb * main_block, (num_tokens, 1, window), dtype=torch.int32, device="cuda")
    swa_lens = torch.full((num_tokens,), window, dtype=torch.int32, device="cuda")
    attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 2.0
    output = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device="cuda")

    out, _, _ = flash_mla_sparse_fwd(
        q=q,
        kv=swa_kv_paged,
        indices=indices,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        topk_length=swa_lens,
        out=output,
    )

    print(f"  prefill swa-only: out shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()} "
          f"abs_mean={out.abs().float().mean().item():.4e} nz={(out != 0).float().mean().item():.4f}")
    print(f"  output (via out= side-effect): abs_mean={output.abs().float().mean().item():.4e}")
    assert torch.isfinite(output).all()
    assert (output != 0).float().mean().item() > 0.5, "output mostly zero — bug!"
