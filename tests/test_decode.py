"""
Decode correctness tests for flash_mla_sm120.
Covers V32 (DeepSeek V3.2, GLM 5.1) and MODEL1 (DeepSeek V4 Flash/Pro).

Uses FlashMLA-compatible quantization:
  V32:    656B/token [nope 512B | scale 16B FP32 | rope 128B]
  MODEL1: 584B/token [nope 448B | rope 128B | scale 8B UE8M0]
"""

import torch
import pytest
import math
import struct
import flash_mla_sm120
import flash_mla_sm120.cuda


# ── Quantization helpers (matching FlashMLA quant.py) ────────────────

def _cast_scale_inv_to_ue8m0(scales_inv):
    """Round scale to nearest power-of-2 (FlashMLA convention)."""
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil())


def _fp32_to_ue8m0_bytes(scale_fp32):
    """Convert FP32 power-of-2 scale to UE8M0 byte (exponent extraction)."""
    bits = scale_fp32.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


# ── V32 pack/unpack ─────────────────────────────────────────────────

def quantize_kv_v32(kv_bf16):
    """Pack bf16 KV → V32 FP8 format (656B/token).
    Input:  (num_blocks, block_size, 1, 576) bf16
    Output: (num_blocks, block_size, 1, 656) uint8 view
    """
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    nb, bs, hk, d = kv_bf16.shape
    assert d == 576 and hk == 1
    kv = kv_bf16.squeeze(2)

    bpt = d_nope + num_tiles * 4 + d_rope * 2  # 656
    result = torch.zeros(nb, bs, bpt, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[..., ti*tile_size:(ti+1)*tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[..., ti*tile_size:(ti+1)*tile_size] = fp8.view(torch.uint8)
        sb = scale.to(torch.float32).contiguous().view(torch.uint8).reshape(nb, bs, 4)
        result[..., d_nope + ti*4 : d_nope + (ti+1)*4] = sb

    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8).reshape(nb, bs, d_rope*2)
    result[..., d_nope + num_tiles*4:] = rope
    return result.view(nb, bs, 1, bpt)


def dequantize_kv_v32(packed):
    """Unpack V32 FP8 → bf16."""
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    nb, bs, hk, bpt = packed.shape
    p = packed.view(nb, bs, -1)
    result = torch.zeros(nb, bs, 576, dtype=torch.bfloat16, device=p.device)
    for ti in range(num_tiles):
        fp8 = p[..., ti*tile_size:(ti+1)*tile_size].view(torch.float8_e4m3fn).float()
        sc = p[..., d_nope+ti*4:d_nope+(ti+1)*4].contiguous().view(torch.float32).squeeze(-1)
        result[..., ti*tile_size:(ti+1)*tile_size] = (fp8 * sc.unsqueeze(-1)).to(torch.bfloat16)
    result[..., d_nope:] = p[..., d_nope+num_tiles*4:].contiguous().view(torch.bfloat16).reshape(nb, bs, d_rope)
    return result.view(nb, bs, 1, 576)


# ── MODEL1 pack/unpack ──────────────────────────────────────────────

def quantize_kv_model1(kv_bf16):
    """Pack bf16 KV → MODEL1 FP8 FOOTER format.
    Input:  (num_blocks, block_size, 1, 512) bf16
    Output: (num_blocks, block_size, 1, 584) uint8 view

    Physical layout per block (FOOTER):
      [0 : block_size*576)              Token data (each: nope 448B FP8 + rope 128B BF16)
      [block_size*576 : block_size*584) Scale footer (each: 7×UE8M0 + 1 pad)
    The output tensor has shape (nb, bs, 1, 584) but physical bytes are footer-ordered.
    """
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2  # 576
    scale_bytes = num_tiles + 1  # 8 (7 UE8M0 + 1 pad)
    bpt = data_stride + scale_bytes  # 584
    nb, bs, hk, d = kv_bf16.shape
    assert d == 512 and hk == 1
    kv = kv_bf16.squeeze(2)

    # Allocate flat byte buffer per block with footer layout
    block_bytes = bs * bpt
    result_flat = torch.zeros(nb, block_bytes, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[..., ti*tile_size:(ti+1)*tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        ue8m0 = _fp32_to_ue8m0_bytes(scale)

        for tok in range(bs):
            data_off = tok * data_stride + ti * tile_size
            result_flat[:, data_off:data_off+tile_size] = fp8[:, tok].view(torch.uint8)
            scale_off = bs * data_stride + tok * scale_bytes + ti
            result_flat[:, scale_off] = ue8m0[:, tok]

    # Rope at offset d_nope within each token's data region
    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    rope = rope.reshape(nb, bs, d_rope * 2)
    for tok in range(bs):
        rope_off = tok * data_stride + d_nope
        result_flat[:, rope_off:rope_off+d_rope*2] = rope[:, tok]

    return result_flat.view(nb, bs, 1, bpt)


def dequantize_kv_model1(packed):
    """Unpack MODEL1 FP8 FOOTER format → bf16."""
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2  # 576
    scale_bytes = num_tiles + 1  # 8
    bpt = data_stride + scale_bytes  # 584
    nb, bs, hk, _ = packed.shape
    result = torch.zeros(nb, bs, 512, dtype=torch.bfloat16, device=packed.device)

    p = packed.view(nb, bs * bpt)  # flat byte view per block

    for tok in range(bs):
        data_off = tok * data_stride
        scale_off = bs * data_stride + tok * scale_bytes
        for ti in range(num_tiles):
            fp8_off = data_off + ti * tile_size
            fp8 = p[:, fp8_off:fp8_off+tile_size].view(torch.float8_e4m3fn).float()
            ue8m0 = p[:, scale_off + ti]
            scale = torch.pow(2.0, ue8m0.float() - 127.0)
            result[:, tok, ti*tile_size:(ti+1)*tile_size] = (
                fp8 * scale.unsqueeze(-1)).to(torch.bfloat16)
        rope_off = data_off + d_nope
        rope_bytes = p[:, rope_off:rope_off+d_rope*2].contiguous()
        result[:, tok, d_nope:] = rope_bytes.view(torch.bfloat16).reshape(nb, d_rope)

    return result.view(nb, bs, 1, 512)


# ── Reference implementation ────────────────────────────────────────

def ref_sparse_attn_decode(q, kv_dequant, indices, sm_scale, d_v):
    """
    q:           (b, s_q, h_q, d_qk) bf16
    kv_dequant:  (num_blocks, block_size, 1, d_qk) bf16
    indices:     (b, s_q, topk) int32
    """
    b, s_q, h_q, d_qk = q.shape
    topk = indices.shape[-1]

    kv_flat = kv_dequant.view(-1, d_qk).float()
    q_f = q.float()

    idx_fixed = indices.clamp(min=0)
    invalid = indices < 0

    gathered = kv_flat.index_select(0, idx_fixed.view(-1)).view(b, s_q, topk, d_qk)
    P = torch.einsum("bshd,bstd->bsht", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(2).expand_as(P)] = float("-inf")

    lse = torch.logsumexp(P, dim=-1)
    lse_safe = lse.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))

    out = torch.einsum("bsht,bstd->bshd", weights, gathered[..., :d_v])
    return out.to(torch.bfloat16), lse


# ── Helper to run decode and compare ────────────────────────────────

def run_decode_test(model_type, d_qk, d_v, topk, num_heads, batch_size,
                    block_size=64, num_blocks=64, atol=0.01):
    torch.manual_seed(42)
    sm_scale = d_qk ** -0.5
    s_kv = num_blocks * block_size

    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                            device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)

    if model_type == "V32":
        kv_packed = quantize_kv_v32(kv_bf16)
        kv_dequant = dequantize_kv_v32(kv_packed)
    else:
        kv_packed = quantize_kv_model1(kv_bf16)
        kv_dequant = dequantize_kv_model1(kv_packed)

    q = (torch.randn(batch_size, 1, num_heads, d_qk,
                      device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (batch_size, 1, topk),
                             device="cuda", dtype=torch.int32)
    indices[:, :, -10:] = -1

    ref_out, ref_lse = ref_sparse_attn_decode(q, kv_dequant, indices, sm_scale, d_v)

    q_flat = q.view(-1, num_heads, d_qk)
    idx_flat = indices.view(-1, topk)

    out, lse = flash_mla_sm120.sparse_mla_decode_fwd(
        q_flat, kv_packed, idx_flat, sm_scale, d_v)
    out = out.view_as(ref_out)

    err = (out.float() - ref_out.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    return max_err, mean_err


# ── V32 Tests ────────────────────────────────────────────────────────

class TestV32Decode:
    """DeepSeek V3.2 / GLM 5.1 (d_qk=576, 656B/token)"""

    @pytest.mark.parametrize("num_heads,batch_size", [
        (128, 1), (128, 4),   # V3.2 TP1
        (16, 1), (16, 4),     # V3.2 TP8
        (64, 1), (64, 4),     # GLM 5.1 TP1
    ])
    def test_correctness(self, num_heads, batch_size):
        max_err, mean_err = run_decode_test(
            "V32", d_qk=576, d_v=512, topk=2048,
            num_heads=num_heads, batch_size=batch_size)
        print(f"\n  V32 h={num_heads} bs={batch_size}: max_err={max_err:.6f} mean_err={mean_err:.6f}")
        assert max_err < 0.001, f"V32 decode failed: max_err={max_err}"


# ── MODEL1 Tests ─────────────────────────────────────────────────────

class TestMODEL1Decode:
    """DeepSeek V4 Flash/Pro (d_qk=512, 584B/token)"""

    @pytest.mark.parametrize("num_heads,topk,batch_size", [
        (64, 512, 1),  (64, 512, 4),    # V4 Flash TP1
        (128, 1024, 1), (128, 1024, 4),  # V4 Pro TP1
    ])
    def test_correctness(self, num_heads, topk, batch_size):
        max_err, mean_err = run_decode_test(
            "MODEL1", d_qk=512, d_v=512, topk=topk,
            num_heads=num_heads, batch_size=batch_size)
        print(f"\n  MODEL1 h={num_heads} topk={topk} bs={batch_size}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f}")
        assert max_err < 0.001, f"MODEL1 decode failed: max_err={max_err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
