"""
Triton kernels for NSA MQA logits computation.

Math:
  For each query token i and KV position j:
    logits[i, j] = sum_h( max( sum_d(q[i,h,d] * kv[j,d]) * kv_scale[j], 0 ) * weights[i,h] )
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Paged MQA logits  (decode / target_verify / draft_extend)
# ---------------------------------------------------------------------------


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    Q_ptr,
    KV_ptr,
    KV_SCALE_ptr,
    W_ptr,
    CTX_LENS_ptr,
    BT_ptr,
    LOGITS_ptr,
    stride_q_token: tl.int64,
    stride_q_head: tl.int64,
    stride_kv_block: tl.int64,
    stride_kvs_block: tl.int64,
    stride_bt_batch: tl.int64,
    stride_logits_token: tl.int64,
    stride_w_token: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    MAX_CONTEXT_LEN,
    NEXT_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Grid: (batch*next_n, num_logical_pages)"""
    q_token_idx = tl.program_id(0)
    page_logical = tl.program_id(1)
    batch_idx = q_token_idx // NEXT_N

    context_len = tl.load(CTX_LENS_ptr + batch_idx)
    kv_start = page_logical * BLOCK_KV
    if kv_start >= context_len:
        return

    phys_page = tl.load(BT_ptr + batch_idx * stride_bt_batch + page_logical)

    kv_range = tl.arange(0, BLOCK_KV)
    d_range = tl.arange(0, HEAD_DIM)

    kv_ptrs = (
        KV_ptr
        + phys_page * stride_kv_block
        + kv_range[:, None] * HEAD_DIM
        + d_range[None, :]
    )
    kv_vals = tl.load(kv_ptrs)

    kv_scale = tl.load(
        KV_SCALE_ptr + phys_page * stride_kvs_block + kv_range
    )

    accum = tl.zeros((BLOCK_KV,), dtype=tl.float32)

    for h_start in tl.static_range(0, NUM_HEADS, BLOCK_H):
        h_range = h_start + tl.arange(0, BLOCK_H)

        q_ptrs = (
            Q_ptr
            + q_token_idx * stride_q_token
            + h_range[:, None] * stride_q_head
            + d_range[None, :]
        )
        q_vals = tl.load(q_ptrs)

        dots = tl.dot(q_vals, tl.trans(kv_vals))
        dots = tl.maximum(dots * kv_scale[None, :], 0.0)

        w = tl.load(W_ptr + q_token_idx * stride_w_token + h_range)
        accum += tl.sum(dots * w[:, None], axis=0)

    valid = (kv_start + kv_range) < context_len
    tl.store(
        LOGITS_ptr + q_token_idx * stride_logits_token + kv_start + kv_range,
        accum,
        mask=valid,
    )


def fp8_paged_mqa_logits_triton(
    q_flat,           # (batch*next_n, num_heads, head_dim) fp8
    kv_fp8,           # (num_kv_blocks, block_kv, head_dim) fp8
    kv_scale,         # (num_kv_blocks, block_kv) fp32
    weights,          # (batch*next_n, num_heads) fp32
    context_lens,     # (batch,) int32
    block_table,      # (batch, max_block_len) int32
    max_context_len,  # int
    next_n=1,
):
    """Paged MQA logits via Triton. Takes pre-unpacked kv_fp8 / kv_scale."""
    total_q, num_heads, head_dim = q_flat.shape
    block_kv = kv_fp8.shape[1]
    assert block_kv == 64
    max_block_len = block_table.shape[1]

    align = 256
    stride = ((max_context_len + align - 1) // align) * align
    logits = torch.empty(
        (total_q, stride), dtype=torch.float32, device=q_flat.device
    )
    logits = logits[:, :max_context_len]

    block_h = min(64, num_heads)
    while num_heads % block_h != 0 and block_h > 16:
        block_h //= 2

    grid = (total_q, max_block_len)

    _fp8_paged_mqa_logits_kernel[grid](
        q_flat, kv_fp8, kv_scale, weights,
        context_lens, block_table, logits,
        q_flat.stride(0), q_flat.stride(1),
        kv_fp8.stride(0), kv_scale.stride(0),
        block_table.stride(0), logits.stride(0), weights.stride(0),
        NUM_HEADS=num_heads, HEAD_DIM=head_dim,
        BLOCK_KV=block_kv, MAX_CONTEXT_LEN=max_context_len,
        NEXT_N=next_n, BLOCK_H=block_h,
        num_warps=4, num_stages=2,
    )
    return logits


# ---------------------------------------------------------------------------
# Ragged MQA logits  (extend / prefill)
# ---------------------------------------------------------------------------


@triton.jit
def _fp8_mqa_logits_kernel(
    Q_ptr,
    KV_ptr,
    KV_SCALE_ptr,
    W_ptr,
    KS_ptr,
    KE_ptr,
    LOGITS_ptr,
    stride_q_token: tl.int64,
    stride_q_head: tl.int64,
    stride_kv_token: tl.int64,
    stride_logits_token: tl.int64,
    stride_w_token: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQ_LEN_KV,
    BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Grid: (seq_len, num_kv_tiles)"""
    q_idx = tl.program_id(0)
    kv_tile_idx = tl.program_id(1)

    k_start = tl.load(KS_ptr + q_idx)
    k_end = tl.load(KE_ptr + q_idx)

    kv_base = kv_tile_idx * BLOCK_KV
    if kv_base >= k_end or (kv_base + BLOCK_KV) <= k_start:
        return

    kv_range = tl.arange(0, BLOCK_KV)
    kv_pos = kv_base + kv_range
    d_range = tl.arange(0, HEAD_DIM)

    kv_ptrs = KV_ptr + kv_pos[:, None] * stride_kv_token + d_range[None, :]
    kv_mask = kv_pos[:, None] < SEQ_LEN_KV
    kv_vals = tl.load(kv_ptrs, mask=kv_mask, other=0.0)

    kv_scale = tl.load(
        KV_SCALE_ptr + kv_pos, mask=kv_pos < SEQ_LEN_KV, other=0.0
    )

    accum = tl.zeros((BLOCK_KV,), dtype=tl.float32)

    for h_start in tl.static_range(0, NUM_HEADS, BLOCK_H):
        h_range = h_start + tl.arange(0, BLOCK_H)

        q_ptrs = (
            Q_ptr
            + q_idx * stride_q_token
            + h_range[:, None] * stride_q_head
            + d_range[None, :]
        )
        q_vals = tl.load(q_ptrs)

        dots = tl.dot(q_vals, tl.trans(kv_vals))
        dots = tl.maximum(dots * kv_scale[None, :], 0.0)

        w = tl.load(W_ptr + q_idx * stride_w_token + h_range)
        accum += tl.sum(dots * w[:, None], axis=0)

    valid = (kv_pos >= k_start) & (kv_pos < k_end)
    tl.store(LOGITS_ptr + q_idx * stride_logits_token + kv_pos, accum, mask=valid)


def fp8_mqa_logits_ragged_triton(
    q,                   # (seq_len, num_heads, head_dim) fp8
    kv_fp8,              # (seq_len_kv, head_dim) fp8
    kv_scale,            # (seq_len_kv,) fp32
    weights,             # (seq_len, num_heads) fp32
    cu_seq_len_k_start,  # (seq_len,) int32
    cu_seq_len_k_end,    # (seq_len,) int32
    max_seqlen_k=0,
):
    """Ragged MQA logits via Triton."""
    seq_len, num_heads, head_dim = q.shape
    seq_len_kv = kv_fp8.shape[0]

    align_unit = 256
    if max_seqlen_k == 0:
        stride_logits = ((seq_len_kv + align_unit) + 3) // 4 * 4
        logits = torch.empty(
            (seq_len, stride_logits), dtype=torch.float32, device=q.device
        )
        logits = logits[:, :seq_len_kv]
    else:
        stride_logits = ((max_seqlen_k + align_unit - 1) // align_unit) * align_unit
        logits = torch.empty(
            (seq_len, stride_logits), dtype=torch.float32, device=q.device
        )
        logits = logits[:, :max_seqlen_k]

    output_width = logits.shape[1]
    tile_kv = 128
    num_kv_tiles = (output_width + tile_kv - 1) // tile_kv

    block_h = min(64, num_heads)
    while num_heads % block_h != 0 and block_h > 16:
        block_h //= 2

    grid = (seq_len, num_kv_tiles)

    _fp8_mqa_logits_kernel[grid](
        q, kv_fp8, kv_scale, weights,
        cu_seq_len_k_start, cu_seq_len_k_end, logits,
        q.stride(0), q.stride(1), kv_fp8.stride(0),
        logits.stride(0), weights.stride(0),
        NUM_HEADS=num_heads, HEAD_DIM=head_dim,
        SEQ_LEN_KV=seq_len_kv, BLOCK_KV=tile_kv, BLOCK_H=block_h,
        num_warps=4, num_stages=4,
    )
    return logits
