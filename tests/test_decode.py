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
                    block_size=64, num_blocks=64, bf16_qk=True):
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
        q_flat, kv_packed, idx_flat, sm_scale, d_v, bf16_qk=bf16_qk)
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
        (8, 1),  (8, 4),      # GLM 5.1 TP8
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
        (8, 512, 1),   (8, 512, 4),     # V4 Flash TP8
        (64, 512, 1),  (64, 512, 4),    # V4 Flash TP1
        (16, 1024, 1), (16, 1024, 4),   # V4 Pro TP8
        (128, 1024, 1), (128, 1024, 4),  # V4 Pro TP1
    ])
    @pytest.mark.parametrize("bf16_qk", [True, False], ids=["bf16qk", "fp8qk"])
    def test_correctness(self, num_heads, topk, batch_size, bf16_qk):
        max_err, mean_err = run_decode_test(
            "MODEL1", d_qk=512, d_v=512, topk=topk,
            num_heads=num_heads, batch_size=batch_size, bf16_qk=bf16_qk)
        tag = "BF16" if bf16_qk else "FP8"
        threshold = 0.001 if bf16_qk else 0.002
        print(f"\n  MODEL1[{tag}] h={num_heads} topk={topk} bs={batch_size}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f}")
        assert max_err < threshold, (
            f"MODEL1 decode [{tag}] failed: max_err={max_err} > {threshold}")


# ── attn_sink Tests (MODEL1 only) ───────────────────────────────────

def run_decode_attn_sink_test(num_heads, topk, batch_size, attn_sink_mode="random",
                               block_size=64, num_blocks=64, seed=42):
    """Run decode with attn_sink and compare to reference.

    attn_sink_mode: "random", "neg_inf" (no effect), "large_pos" (dominates),
                    "zero", "mixed" (half heads large, half small)
    """
    torch.manual_seed(seed)
    d_qk, d_v = 512, 512
    sm_scale = d_qk ** -0.5
    s_kv = num_blocks * block_size

    kv_bf16 = (torch.randn(num_blocks, block_size, 1, d_qk,
                            device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    kv_packed = quantize_kv_model1(kv_bf16)
    kv_dequant = dequantize_kv_model1(kv_packed)

    q = (torch.randn(batch_size, 1, num_heads, d_qk,
                      device="cuda", dtype=torch.bfloat16) / 10).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (batch_size, 1, topk),
                             device="cuda", dtype=torch.int32)
    indices[:, :, -10:] = -1

    if attn_sink_mode == "random":
        attn_sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 2.0
    elif attn_sink_mode == "neg_inf":
        attn_sink = torch.full((num_heads,), float("-inf"), device="cuda", dtype=torch.float32)
    elif attn_sink_mode == "large_pos":
        attn_sink = torch.full((num_heads,), 20.0, device="cuda", dtype=torch.float32)
    elif attn_sink_mode == "zero":
        attn_sink = torch.zeros(num_heads, device="cuda", dtype=torch.float32)
    elif attn_sink_mode == "mixed":
        attn_sink = torch.zeros(num_heads, device="cuda", dtype=torch.float32)
        attn_sink[:num_heads // 2] = 15.0
        attn_sink[num_heads // 2:] = -15.0
    else:
        raise ValueError(f"Unknown mode: {attn_sink_mode}")

    ref_out, ref_lse = ref_sparse_attn_decode(q, kv_dequant, indices, sm_scale, d_v)
    ref_lse_f = ref_lse.float()
    sink_f = attn_sink.view(1, 1, num_heads)
    scale = 1.0 / (1.0 + torch.exp(sink_f - ref_lse_f))
    ref_out_sink = (ref_out.float() * scale.unsqueeze(-1)).to(torch.bfloat16)

    q_flat = q.view(-1, num_heads, d_qk)
    idx_flat = indices.view(-1, topk)
    out, lse = flash_mla_sm120.sparse_mla_decode_fwd(
        q_flat, kv_packed, idx_flat, sm_scale, d_v, attn_sink)
    out = out.view_as(ref_out_sink)

    err = (out.float() - ref_out_sink.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    return max_err, mean_err


class TestAttnSinkDecode:
    """Test attn_sink LSE merge in combine kernel — comprehensive coverage."""

    @pytest.mark.parametrize("num_heads,topk,batch_size", [
        (64, 512, 1),    (64, 512, 4),    (64, 512, 8),
        (128, 1024, 1),  (128, 1024, 4),  (128, 1024, 8),
        (16, 1024, 1),   (16, 1024, 4),    # TP8 Pro config
    ])
    def test_random_sink(self, num_heads, topk, batch_size):
        max_err, mean_err = run_decode_attn_sink_test(
            num_heads, topk, batch_size, "random")
        print(f"\n  attn_sink random h={num_heads} topk={topk} bs={batch_size}: "
              f"max_err={max_err:.6f} mean_err={mean_err:.6f}")
        assert max_err < 0.002, f"max_err={max_err}"

    @pytest.mark.parametrize("num_heads,topk", [
        (64, 512), (128, 1024),
    ])
    def test_neg_inf_sink(self, num_heads, topk):
        """attn_sink = -inf should have no effect (same as no sink)."""
        max_err_nosink, _ = run_decode_test(
            "MODEL1", d_qk=512, d_v=512, topk=topk,
            num_heads=num_heads, batch_size=1)
        max_err_sink, _ = run_decode_attn_sink_test(
            num_heads, topk, 1, "neg_inf")
        print(f"\n  attn_sink -inf h={num_heads}: nosink_err={max_err_nosink:.6f} "
              f"sink_err={max_err_sink:.6f}")
        assert max_err_sink < 0.001, f"max_err={max_err_sink}"

    @pytest.mark.parametrize("num_heads,topk", [
        (64, 512), (128, 1024),
    ])
    def test_large_pos_sink(self, num_heads, topk):
        """Large positive attn_sink should suppress sparse attention output toward 0."""
        max_err, _ = run_decode_attn_sink_test(
            num_heads, topk, 1, "large_pos")
        print(f"\n  attn_sink large_pos h={num_heads}: max_err={max_err:.6f}")
        assert max_err < 0.002, f"max_err={max_err}"

    @pytest.mark.parametrize("num_heads,topk", [
        (64, 512), (128, 1024),
    ])
    def test_zero_sink(self, num_heads, topk):
        max_err, _ = run_decode_attn_sink_test(num_heads, topk, 1, "zero")
        print(f"\n  attn_sink zero h={num_heads}: max_err={max_err:.6f}")
        assert max_err < 0.002, f"max_err={max_err}"

    @pytest.mark.parametrize("num_heads,topk", [
        (64, 512), (128, 1024),
    ])
    def test_mixed_sink(self, num_heads, topk):
        """Half heads large sink, half small — tests per-head independence."""
        max_err, _ = run_decode_attn_sink_test(num_heads, topk, 4, "mixed")
        print(f"\n  attn_sink mixed h={num_heads}: max_err={max_err:.6f}")
        assert max_err < 0.002, f"max_err={max_err}"

    def test_race_check(self):
        """Run 3x with different seeds to catch non-deterministic races."""
        for seed in [42, 123, 999]:
            max_err, _ = run_decode_attn_sink_test(
                128, 1024, 4, "random", seed=seed)
            assert max_err < 0.002, f"race check seed={seed}: max_err={max_err}"
        print("\n  attn_sink race check (3 seeds): PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
