"""
Precision regression tests across data magnitudes.
Verifies kernel output matches reference at realistic (not just unit-test) scales.
Covers both decode and prefill paths.
"""
import torch
import pytest
import math
import flash_mla_sm120.cuda
from flash_mla_sm120.ops import sparse_mla_decode_fwd, sparse_mla_prefill_fwd

# ── Quantization helpers (matching FlashMLA quant.py) ────────────────

def _cast_scale_inv_to_ue8m0(scales_inv):
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil())

def _fp32_to_ue8m0_bytes(scale_fp32):
    bits = scale_fp32.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


# ── V32 quantize/dequantize ─────────────────────────────────────────

def quantize_kv_v32(kv_bf16):
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    nb, bs, hk, d = kv_bf16.shape
    kv = kv_bf16.squeeze(2)
    bpt = d_nope + num_tiles * 4 + d_rope * 2
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


# ── MODEL1 quantize/dequantize ──────────────────────────────────────

def quantize_kv_model1(kv_bf16):
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
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        ue8m0 = _fp32_to_ue8m0_bytes(scale)
        for tok in range(bs):
            data_off = tok * data_stride + ti * tile_size
            result_flat[:, data_off:data_off+tile_size] = fp8[:, tok].view(torch.uint8)
            scale_off = bs * data_stride + tok * scale_bytes + ti
            result_flat[:, scale_off] = ue8m0[:, tok]
    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    rope = rope.reshape(nb, bs, d_rope * 2)
    for tok in range(bs):
        rope_off = tok * data_stride + d_nope
        result_flat[:, rope_off:rope_off+d_rope*2] = rope[:, tok]
    return result_flat.view(nb, bs, 1, bpt)


def dequantize_kv_model1(packed):
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2
    scale_bytes = num_tiles + 1
    bpt = data_stride + scale_bytes
    nb, bs, hk, _ = packed.shape
    result = torch.zeros(nb, bs, 512, dtype=torch.bfloat16, device=packed.device)
    p = packed.view(nb, bs * bpt)
    for tok in range(bs):
        data_off = tok * data_stride
        scale_off = bs * data_stride + tok * scale_bytes
        for ti in range(num_tiles):
            fp8_off = data_off + ti * tile_size
            fp8 = p[:, fp8_off:fp8_off+tile_size].view(torch.float8_e4m3fn).float()
            ue8m0 = p[:, scale_off + ti]
            scale = torch.pow(2.0, ue8m0.float() - 127.0)
            result[:, tok, ti*tile_size:(ti+1)*tile_size] = (fp8 * scale.unsqueeze(-1)).to(torch.bfloat16)
        rope_off = data_off + d_nope
        rope = p[:, rope_off:rope_off+d_rope*2].contiguous().view(torch.bfloat16).reshape(nb, d_rope)
        result[:, tok, d_nope:] = rope
    return result.view(nb, bs, 1, 512)


# ── Reference computation ───────────────────────────────────────────

def vectorized_reference(q, kv_deq, indices, sm_scale, d_qk, d_v=512):
    """Vectorized BF16 reference using dequantized KV cache."""
    kv_flat = kv_deq.view(-1, d_qk).float()
    gathered_kv = kv_flat[indices.long()]
    k_gathered = gathered_kv[..., :d_qk]
    v_gathered = gathered_kv[..., :d_v]
    scores = torch.einsum("...hd,...kd->...hk", q.float(), k_gathered) * sm_scale
    m = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - m)
    l = exp_scores.sum(dim=-1, keepdim=True)
    attn = exp_scores / l
    return torch.einsum("...hk,...kd->...hd", attn, v_gathered).to(torch.bfloat16)


def measure_error(output, ref_output):
    """Compute error metrics including FlashMLA allclose pass rate."""
    diff = (output.float() - ref_output.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ref_mean = ref_output.float().abs().mean().item()

    abs_tol, rel_tol = 1e-3, 2.01 / 128
    threshold = abs_tol + rel_tol * torch.maximum(output.float().abs(), ref_output.float().abs())
    pass_rate = (diff <= threshold).float().mean().item()

    return {
        "max_err": max_err,
        "mean_err": mean_err,
        "ref_mean": ref_mean,
        "rel_err": mean_err / max(ref_mean, 1e-10),
        "flashmla_pass_rate": pass_rate,
    }


# ── Decode test runner ───────────────────────────────────────────────

def run_decode_test(model_type, num_heads, topk, magnitude, batch=2,
                    num_blocks=64, block_size=64):
    torch.manual_seed(42)
    if model_type == "v32":
        d_qk, d_v = 576, 512
        quantize_fn, dequant_fn = quantize_kv_v32, dequantize_kv_v32
    else:
        d_qk, d_v = 512, 512
        quantize_fn, dequant_fn = quantize_kv_model1, dequantize_kv_model1

    sm_scale = 1.0 / math.sqrt(d_qk)
    q = torch.randn(batch, num_heads, d_qk, device="cuda", dtype=torch.bfloat16) * magnitude
    kv_bf16 = torch.randn(num_blocks, block_size, 1, d_qk, device="cuda", dtype=torch.bfloat16) * magnitude
    kv_packed = quantize_fn(kv_bf16)
    kv_deq = dequant_fn(kv_packed)
    indices = torch.randint(0, num_blocks * block_size, (batch, topk),
                            device="cuda", dtype=torch.int32)

    output, lse = sparse_mla_decode_fwd(
        q, kv_packed.view(-1, block_size, 1, kv_packed.shape[-1]),
        indices, sm_scale, d_v)
    ref_output = vectorized_reference(q, kv_deq, indices, sm_scale, d_qk, d_v)
    return measure_error(output, ref_output)


# ── Prefill test runner ──────────────────────────────────────────────

def run_prefill_test(model_type, num_heads, topk, magnitude, num_tokens=4,
                     num_blocks=64, block_size=64):
    torch.manual_seed(42)
    if model_type == "v32":
        d_qk, d_v = 576, 512
        quantize_fn, dequant_fn = quantize_kv_v32, dequantize_kv_v32
    else:
        d_qk, d_v = 512, 512
        quantize_fn, dequant_fn = quantize_kv_model1, dequantize_kv_model1

    sm_scale = 1.0 / math.sqrt(d_qk)
    q = torch.randn(num_tokens, num_heads, d_qk, device="cuda", dtype=torch.bfloat16) * magnitude
    kv_bf16 = torch.randn(num_blocks, block_size, 1, d_qk, device="cuda", dtype=torch.bfloat16) * magnitude
    kv_packed = quantize_fn(kv_bf16)
    kv_deq = dequant_fn(kv_packed)
    indices = torch.randint(0, num_blocks * block_size, (num_tokens, topk),
                            device="cuda", dtype=torch.int32)

    output, lse = sparse_mla_prefill_fwd(
        q, kv_packed.view(-1, block_size, 1, kv_packed.shape[-1]),
        indices, sm_scale, d_v)
    ref_output = vectorized_reference(q, kv_deq, indices, sm_scale, d_qk, d_v)
    return measure_error(output, ref_output)


# ── Decode precision regression tests ────────────────────────────────

@pytest.mark.parametrize("magnitude", [0.1, 0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("model_type,num_heads,topk", [
    ("v32", 128, 2048),
    ("model1", 64, 512),
    ("model1", 128, 1024),
])
def test_decode_precision_vs_magnitude(model_type, num_heads, topk, magnitude):
    result = run_decode_test(model_type, num_heads, topk, magnitude, batch=2)
    print(f"\n[decode] {model_type} h={num_heads} topk={topk} mag={magnitude}: "
          f"rel_err={result['rel_err']:.4%} flashmla_pass={result['flashmla_pass_rate']:.2%} "
          f"max_err={result['max_err']:.6f}")
    assert result["rel_err"] < 0.10, (
        f"Relative error {result['rel_err']:.2%} exceeds 10% at magnitude={magnitude}")


# ── Prefill precision regression tests ───────────────────────────────

@pytest.mark.parametrize("magnitude", [0.1, 0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("model_type,num_heads,topk", [
    ("v32", 128, 2048),
    ("model1", 64, 512),
    ("model1", 128, 1024),
])
def test_prefill_precision_vs_magnitude(model_type, num_heads, topk, magnitude):
    result = run_prefill_test(model_type, num_heads, topk, magnitude, num_tokens=4)
    print(f"\n[prefill] {model_type} h={num_heads} topk={topk} mag={magnitude}: "
          f"rel_err={result['rel_err']:.4%} flashmla_pass={result['flashmla_pass_rate']:.2%} "
          f"max_err={result['max_err']:.6f}")
    assert result["rel_err"] < 0.10, (
        f"Relative error {result['rel_err']:.2%} exceeds 10% at magnitude={magnitude}")


# ── Production config tests ──────────────────────────────────────────

@pytest.mark.parametrize("magnitude", [0.1, 1.0, 5.0])
def test_model1_flash_decode(magnitude):
    """MODEL1 Flash (h=64, topk=512) — TP1 decode."""
    r = run_decode_test("model1", 64, 512, magnitude, batch=4)
    print(f"\n[decode] MODEL1 Flash mag={magnitude}: rel={r['rel_err']:.4%} pass={r['flashmla_pass_rate']:.2%}")

@pytest.mark.parametrize("magnitude", [0.1, 1.0, 5.0])
def test_model1_pro_decode(magnitude):
    """MODEL1 Pro (h=128, topk=1024) — TP1 decode."""
    r = run_decode_test("model1", 128, 1024, magnitude, batch=2)
    print(f"\n[decode] MODEL1 Pro mag={magnitude}: rel={r['rel_err']:.4%} pass={r['flashmla_pass_rate']:.2%}")

@pytest.mark.parametrize("magnitude", [0.1, 1.0, 5.0])
def test_model1_flash_prefill(magnitude):
    """MODEL1 Flash (h=64, topk=512) — prefill."""
    r = run_prefill_test("model1", 64, 512, magnitude, num_tokens=4)
    print(f"\n[prefill] MODEL1 Flash mag={magnitude}: rel={r['rel_err']:.4%} pass={r['flashmla_pass_rate']:.2%}")

@pytest.mark.parametrize("magnitude", [0.1, 1.0, 5.0])
def test_model1_pro_prefill(magnitude):
    """MODEL1 Pro (h=128, topk=1024) — prefill."""
    r = run_prefill_test("model1", 128, 1024, magnitude, num_tokens=4)
    print(f"\n[prefill] MODEL1 Pro mag={magnitude}: rel={r['rel_err']:.4%} pass={r['flashmla_pass_rate']:.2%}")


if __name__ == "__main__":
    for path_name, runner in [("decode", run_decode_test), ("prefill", run_prefill_test)]:
        print(f"\n=== {path_name.upper()} Precision vs Data Magnitude ===\n")
        configs = [("v32", 128, 2048), ("model1", 64, 512), ("model1", 128, 1024)]
        for model_type, num_heads, topk in configs:
            print(f"--- {model_type} h={num_heads} topk={topk} ---")
            kwargs = {"batch": 2} if path_name == "decode" else {"num_tokens": 4}
            for mag in [0.1, 0.5, 1.0, 2.0, 5.0]:
                r = runner(model_type, num_heads, topk, mag, **kwargs)
                print(f"  mag={mag:4.1f}  rel_err={r['rel_err']:.4%}  "
                      f"flashmla_pass={r['flashmla_pass_rate']:.2%}  "
                      f"max_err={r['max_err']:.6f}  mean_err={r['mean_err']:.6f}")
            print()
