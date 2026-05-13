"""Dual-cache MODEL1 decode correctness test.

Runs sparse_mla_decode_fwd with extra_kv_cache + extra_indices set and compares
the kernel output against a torch reference computed by dequantizing both
caches, gathering all entries (main ∪ extra), and applying a single softmax
over the union.

Currently only the (TOPK=128, TOPK_EXTRA=128) (NUM_HEADS in {64, 128})
combination is instantiated in dispatch_tiles_dual; new combinations need
matching kernel template instantiations in decode_launch.cu.
"""
import pytest
import torch

import flash_mla_sm120
from tests.test_decode import (
    quantize_kv_model1,
    dequantize_kv_model1,
)


def ref_dual_cache_attn(q, kv_main_dequant, idx_main,
                        kv_extra_dequant, idx_extra, sm_scale, d_v):
    """Reference: attend over union of (idx_main into kv_main, idx_extra into kv_extra).

    q:                (num_tokens, num_heads, d_qk) bf16
    kv_main_dequant:  (num_blocks_m, block_size, 1, d_qk) bf16
    idx_main:         (num_tokens, topk_main) int32 (-1 = invalid)
    kv_extra_dequant: (num_blocks_e, block_size, 1, d_qk) bf16
    idx_extra:        (num_tokens, topk_extra) int32 (-1 = invalid)
    """
    num_tokens, h_q, d_qk = q.shape
    q_f = q.float()

    main_flat = kv_main_dequant.view(-1, d_qk).float()
    extra_flat = kv_extra_dequant.view(-1, d_qk).float()

    gathered_main = main_flat.index_select(0, idx_main.clamp(min=0).view(-1)) \
        .view(num_tokens, idx_main.size(-1), d_qk)
    gathered_extra = extra_flat.index_select(0, idx_extra.clamp(min=0).view(-1)) \
        .view(num_tokens, idx_extra.size(-1), d_qk)

    gathered = torch.cat([gathered_main, gathered_extra], dim=-2)  # (n, tk+tk_ex, d_qk)
    invalid = torch.cat([idx_main < 0, idx_extra < 0], dim=-1)      # (n, tk+tk_ex)

    P = torch.einsum("nhd,ntd->nht", q_f, gathered) * sm_scale       # (n, h, tk+tk_ex)
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")
    lse = torch.logsumexp(P, dim=-1)
    lse_safe = lse.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out = torch.einsum("nht,ntd->nhd", weights, gathered[..., :d_v])
    return out.to(torch.bfloat16), lse


@pytest.mark.parametrize("num_heads", [64, 128])
@pytest.mark.parametrize("num_tokens", [1, 8, 32, 64])
def test_dual_cache_model1_decode_128_128(num_heads, num_tokens):
    """MODEL1: topk_main=128 + topk_extra=128, dispatched to dispatch_tiles_dual."""
    torch.manual_seed(0)
    d_qk = 512
    d_v = 512
    block_size = 64
    sm_scale = d_qk ** -0.5

    # Two independent paged caches with the same packed layout.
    num_blocks_main, num_blocks_extra = 32, 32
    kv_main_bf16 = (torch.randn(num_blocks_main, block_size, 1, d_qk,
                                device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_extra_bf16 = (torch.randn(num_blocks_extra, block_size, 1, d_qk,
                                 device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)

    kv_main_packed = quantize_kv_model1(kv_main_bf16)
    kv_main_dequant = dequantize_kv_model1(kv_main_packed)
    kv_extra_packed = quantize_kv_model1(kv_extra_bf16)
    kv_extra_dequant = dequantize_kv_model1(kv_extra_packed)

    s_kv_main = num_blocks_main * block_size
    s_kv_extra = num_blocks_extra * block_size
    topk_main, topk_extra = 128, 128

    q = (torch.randn(num_tokens, num_heads, d_qk,
                     device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    idx_main = torch.randint(0, s_kv_main, (num_tokens, topk_main),
                             device="cuda", dtype=torch.int32)
    idx_extra = torch.randint(0, s_kv_extra, (num_tokens, topk_extra),
                              device="cuda", dtype=torch.int32)
    # Mark a few entries invalid to exercise -1 masking.
    idx_main[:, -5:] = -1
    idx_extra[:, -3:] = -1

    ref_out, _ = ref_dual_cache_attn(
        q, kv_main_dequant, idx_main, kv_extra_dequant, idx_extra, sm_scale, d_v
    )

    out, _ = flash_mla_sm120.sparse_mla_decode_fwd(
        q, kv_main_packed, idx_main, sm_scale, d_v,
        extra_kv_cache=kv_extra_packed,
        extra_indices=idx_extra,
    )

    err = (out.float() - ref_out.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    print(f"\n  dual-cache decode h={num_heads} tokens={num_tokens}: "
          f"max_err={max_err:.6f} mean_err={mean_err:.6f}")
    # Slightly looser bound than single-cache test_decode (0.001) because
    # FP8 quantization noise compounds across more indices.
    assert max_err < 0.01, f"dual-cache decode failed: max_err={max_err}"
