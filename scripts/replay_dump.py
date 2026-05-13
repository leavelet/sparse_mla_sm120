"""Replay a /tmp/dsv4_kernel_dump_prefill.pt dump.

Loads the captured kernel inputs from a real vLLM inference call and runs:
  * the flash_mla_sm120 kernel (same call vLLM would have made)
  * a torch reference (dequantize → gather → softmax → optional sink)

Reports max/mean abs error per token. If errors are large, the kernel is
producing wrong output for these specific inputs. If errors are small,
the kernel is correct on this slice and the gibberish must be from
downstream code (output projection, MoE, layernorm, etc.) or from other
layers we have not yet dumped.
"""
import argparse
import sys
import torch

import flash_mla_sm120
from tests.test_decode import dequantize_kv_model1


def ref_attn_with_sink(
    q, swa_kv_paged, swa_indices,
    extra_kv_paged, extra_indices,
    sm_scale, d_v, attn_sink=None,
):
    num_tokens, h_q, d_qk = q.shape
    q_f = q.float()

    # SWA cache: dequantize the paged FP8 to bf16, then gather by paged-coord
    # slot ids (idx = block * block_size + slot, the kernel splits via
    # idx / page_block_size and idx % page_block_size).
    # swa_kv_paged is the kernel's input shape (nb, bs, 1, head_bytes); the
    # dequantize helper expects exactly this 4D shape.
    swa_dequant = dequantize_kv_model1(swa_kv_paged)  # (nb, bs, 1, d_qk)
    swa_flat = swa_dequant.view(-1, d_qk).float()
    # swa_indices may be (num_tokens, 1, topk) or (num_tokens, topk).
    topk_swa = swa_indices.size(-1)
    swa_idx_flat = swa_indices.reshape(num_tokens, topk_swa)
    swa_gathered = swa_flat.index_select(
        0, swa_idx_flat.clamp(min=0).view(-1)
    ).view(num_tokens, topk_swa, d_qk)
    invalid = (swa_idx_flat < 0)
    gathered = swa_gathered

    if extra_kv_paged is not None and extra_indices is not None:
        extra_dequant = dequantize_kv_model1(extra_kv_paged)
        extra_flat = extra_dequant.view(-1, d_qk).float()
        topk_ex = extra_indices.size(-1)
        extra_idx_flat = extra_indices.reshape(num_tokens, topk_ex)
        extra_gathered = extra_flat.index_select(
            0, extra_idx_flat.clamp(min=0).view(-1)
        ).view(num_tokens, topk_ex, d_qk)
        gathered = torch.cat([gathered, extra_gathered], dim=-2)
        invalid = torch.cat([invalid, (extra_idx_flat < 0)], dim=-1)

    P = torch.einsum("nhd,ntd->nht", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")
    lse = torch.logsumexp(P, dim=-1)  # (n, h)
    lse_safe = lse.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out = torch.einsum("nht,ntd->nhd", weights, gathered[..., :d_v]).to(torch.bfloat16)

    if attn_sink is not None:
        sink = attn_sink.float().to(q.device)
        factor = torch.sigmoid(lse - sink.unsqueeze(0))
        out = (out.float() * factor.unsqueeze(-1)).to(torch.bfloat16)
    return out, lse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/dsv4_kernel_dump_prefill.pt")
    args = ap.parse_args()

    d = torch.load(args.dump, map_location="cuda", weights_only=False)
    print(f"layer={d['layer_prefix']} swa_only={d['swa_only']} "
          f"compress_ratio={d['compress_ratio']} sm_scale={d['sm_scale']}")

    q = d["q"].cuda()
    swa_kv_paged = d["swa_kv_paged"].cuda()
    swa_indices = d["swa_indices"].cuda()
    extra_kv_paged = (d["extra_kv_paged"].cuda()
                      if d["extra_kv_paged"] is not None else None)
    extra_indices = (d["extra_indices"].cuda()
                     if d["extra_indices"] is not None else None)
    attn_sink = (d["attn_sink"].cuda()
                 if d["attn_sink"] is not None else None)
    sm_scale = d["sm_scale"]
    d_v = 512

    print(f"shapes: q={tuple(q.shape)} swa_kv={tuple(swa_kv_paged.shape)} "
          f"swa_idx={tuple(swa_indices.shape)} "
          f"extra_kv={tuple(extra_kv_paged.shape) if extra_kv_paged is not None else None} "
          f"extra_idx={tuple(extra_indices.shape) if extra_indices is not None else None} "
          f"sink={tuple(attn_sink.shape) if attn_sink is not None else None}")
    if attn_sink is not None:
        finite = attn_sink[torch.isfinite(attn_sink)]
        print(f"  attn_sink: {finite.numel()} finite "
              f"(min={finite.min().item():.4f}, max={finite.max().item():.4f}), "
              f"{(~torch.isfinite(attn_sink)).sum().item()} -inf")
    print(f"  swa_indices: invalid count = {(swa_indices < 0).sum().item()} "
          f"of {swa_indices.numel()}")
    if extra_indices is not None:
        print(f"  extra_indices: invalid count = {(extra_indices < 0).sum().item()} "
              f"of {extra_indices.numel()}")

    # Reference
    ref_out, ref_lse = ref_attn_with_sink(
        q, swa_kv_paged, swa_indices,
        extra_kv_paged, extra_indices,
        sm_scale, d_v, attn_sink=attn_sink,
    )
    print(f"ref out shape={tuple(ref_out.shape)} "
          f"abs mean={ref_out.abs().float().mean().item():.6f}")

    # Kernel
    kernel_out, kernel_lse = flash_mla_sm120.sparse_mla_prefill_fwd(
        q, swa_kv_paged, swa_indices, sm_scale, d_v,
        extra_kv_cache=extra_kv_paged,
        extra_indices=extra_indices,
        attn_sink=attn_sink,
    )
    print(f"kernel out shape={tuple(kernel_out.shape)} "
          f"abs mean={kernel_out.abs().float().mean().item():.6f}")

    # Compare
    err = (kernel_out.float() - ref_out.float()).abs()
    print(f"\n=== KERNEL vs REFERENCE ===")
    print(f"  max_err   = {err.max().item():.6f}")
    print(f"  mean_err  = {err.mean().item():.6f}")
    print(f"  ref mean  = {ref_out.abs().float().mean().item():.6f}")
    print(f"  relative  = {(err.mean() / ref_out.abs().float().mean().clamp(min=1e-9)).item():.4f}")
    # Per-token max err
    per_tok = err.flatten(1).max(dim=1).values
    print(f"  per-token max err: min={per_tok.min().item():.6f} "
          f"max={per_tok.max().item():.6f}")


if __name__ == "__main__":
    main()
