"""
Comprehensive tests for the GPU scheduler kernel (get_decode_metadata).

The scheduler computes per-partition block assignments for split-KV decode.
It must produce correct DecodingSchedMeta and num_splits_ptr arrays that:
1. Cover ALL blocks for ALL batch elements (no work lost)
2. Never overlap (each block assigned to exactly one partition)
3. Correctly track split boundaries via num_splits_ptr prefix sum
4. Handle variable topk_length per batch
5. Handle extra_topk (extra KV cache blocks appended after main)
6. Handle edge cases: b=1, empty topk, all-same topk, etc.
"""

import torch
import pytest
import math

# Import the C++ scheduler
from flash_mla_sm120.cuda import get_decode_metadata

BI = 64  # block_size_n (tokens per block)


def run_scheduler(b, topk, extra_topk=0, num_sm_parts=8, fixed_overhead=5,
                  topk_length=None, extra_topk_length=None, device='cuda'):
    """Run scheduler and return parsed metadata."""
    sched_meta = torch.empty(num_sm_parts * 8, dtype=torch.int32, device=device)
    num_splits = torch.empty(b + 1, dtype=torch.int32, device=device)

    tl = topk_length if topk_length is not None else None
    etl = extra_topk_length if extra_topk_length is not None else None

    get_decode_metadata(b, topk, extra_topk, num_sm_parts, fixed_overhead,
                        tl, etl, sched_meta, num_splits)
    torch.cuda.synchronize()

    meta = sched_meta.view(num_sm_parts, 8).cpu().tolist()
    ns = num_splits.cpu().tolist()

    partitions = []
    for i in range(num_sm_parts):
        m = meta[i]
        partitions.append({
            'begin_req': m[0], 'end_req': m[1],
            'begin_block': m[2], 'end_block': m[3],
            'begin_split_idx': m[4],
            'is_first_split': m[5], 'is_last_split': m[6],
        })
    return partitions, ns


def compute_expected_blocks(b, topk, extra_topk, topk_length, extra_topk_length):
    """Compute expected num_blocks per batch element."""
    blocks = []
    for i in range(b):
        cur = topk_length[i] if topk_length is not None else topk
        if cur == 0:
            cur = 1
        if extra_topk > 0:
            cur = math.ceil(cur / BI) * BI  # pad to block boundary
            cur += extra_topk_length[i] if extra_topk_length is not None else extra_topk
        nb = math.ceil(cur / BI)
        blocks.append(nb)
    return blocks


def verify_coverage(partitions, expected_blocks, b, num_sm_parts):
    """Verify every block of every batch is covered exactly once."""
    # Track which blocks are assigned per batch
    assigned = {i: set() for i in range(b)}

    for part_idx, p in enumerate(partitions):
        if p['begin_req'] >= b:
            continue  # empty partition

        for req in range(p['begin_req'], p['end_req'] + 1):
            if req >= b:
                break
            start = p['begin_block'] if req == p['begin_req'] else 0
            end = p['end_block'] if req == p['end_req'] else expected_blocks[req]

            for blk in range(start, end):
                assert blk not in assigned[req], \
                    f"Block {blk} of batch {req} assigned twice (partition {part_idx})"
                assigned[req].add(blk)

    # Verify all blocks covered
    for i in range(b):
        expected = set(range(expected_blocks[i]))
        assert assigned[i] == expected, \
            f"Batch {i}: expected blocks {expected}, got {assigned[i]}"


def verify_num_splits(partitions, num_splits, b, num_sm_parts):
    """Verify num_splits prefix sum is consistent with partitions."""
    assert num_splits[0] == 0, f"num_splits[0] should be 0, got {num_splits[0]}"

    # Count splits per batch from partition assignments
    splits_per_batch = [0] * b
    for p in partitions:
        if p['begin_req'] >= b:
            continue
        for req in range(p['begin_req'], p['end_req'] + 1):
            if req >= b:
                break
            splits_per_batch[req] += 1

    # Verify prefix sum
    cum = 0
    for i in range(b):
        cum += splits_per_batch[i]
        assert num_splits[i + 1] == cum, \
            f"num_splits[{i+1}] = {num_splits[i+1]}, expected {cum}"


def verify_monotonic_partitions(partitions, b):
    """Verify partitions are in order (no backward jumps)."""
    last_req = -1
    last_block = -1
    for i, p in enumerate(partitions):
        if p['begin_req'] >= b:
            continue
        assert p['begin_req'] >= last_req, \
            f"Partition {i} begins at req {p['begin_req']} but last was {last_req}"
        if p['begin_req'] == last_req:
            assert p['begin_block'] >= last_block, \
                f"Partition {i} begins at block {p['begin_block']} but last ended at {last_block}"
        last_req = p['end_req']
        last_block = p['end_block']


# ── Basic Tests ─────────────────────────────────────────────────────

class TestSchedulerBasic:
    """Basic scheduler correctness for uniform topk."""

    @pytest.mark.parametrize("b,topk,num_sm_parts", [
        (1, 512, 4),
        (1, 512, 8),
        (1, 1024, 8),
        (1, 2048, 16),
        (4, 512, 8),
        (4, 1024, 16),
        (8, 512, 8),
        (8, 1024, 32),
        (8, 2048, 32),
        (16, 512, 32),
        (32, 512, 64),
    ])
    def test_uniform_topk(self, b, topk, num_sm_parts):
        partitions, ns = run_scheduler(b, topk, num_sm_parts=num_sm_parts)
        expected_blocks = compute_expected_blocks(b, topk, 0, None, None)
        verify_coverage(partitions, expected_blocks, b, num_sm_parts)
        verify_num_splits(partitions, ns, b, num_sm_parts)
        verify_monotonic_partitions(partitions, b)

        total_splits = ns[b]
        assert total_splits >= b, f"Need at least {b} splits, got {total_splits}"
        print(f"\n  b={b} topk={topk} parts={num_sm_parts}: "
              f"total_splits={total_splits} splits/batch={[ns[i+1]-ns[i] for i in range(b)]}")

    def test_single_batch_many_partitions(self):
        """b=1 with many partitions → many splits for one batch."""
        partitions, ns = run_scheduler(1, 2048, num_sm_parts=32)
        expected_blocks = [2048 // BI]  # 32 blocks
        verify_coverage(partitions, expected_blocks, 1, 32)
        verify_num_splits(partitions, ns, 1, 32)
        total_splits = ns[1]
        print(f"\n  b=1 topk=2048 parts=32: splits={total_splits}")
        assert total_splits >= 1

    def test_many_batches_few_partitions(self):
        """b=16 with only 4 partitions → multiple batches per partition."""
        partitions, ns = run_scheduler(16, 512, num_sm_parts=4)
        expected_blocks = compute_expected_blocks(16, 512, 0, None, None)
        verify_coverage(partitions, expected_blocks, 16, 4)
        verify_num_splits(partitions, ns, 16, 4)
        print(f"\n  b=16 topk=512 parts=4: splits={[ns[i+1]-ns[i] for i in range(16)]}")


# ── topk_length Tests ───────────────────────────────────────────────

class TestSchedulerTopkLength:
    """Variable topk_length per batch."""

    def test_variable_topk(self):
        """Different topk per batch element."""
        b = 4
        tl = torch.tensor([256, 512, 128, 512], dtype=torch.int32, device='cuda')
        partitions, ns = run_scheduler(b, 512, topk_length=tl, num_sm_parts=8)
        expected_blocks = compute_expected_blocks(b, 512, 0, tl.tolist(), None)
        verify_coverage(partitions, expected_blocks, b, 8)
        verify_num_splits(partitions, ns, b, 8)
        print(f"\n  variable topk [256,512,128,512]: blocks={expected_blocks} splits={[ns[i+1]-ns[i] for i in range(b)]}")

    def test_all_minimum_topk(self):
        """All batches have minimal topk (1 block each)."""
        b = 8
        tl = torch.full((b,), BI, dtype=torch.int32, device='cuda')
        partitions, ns = run_scheduler(b, 512, topk_length=tl, num_sm_parts=8)
        expected_blocks = compute_expected_blocks(b, 512, 0, tl.tolist(), None)
        verify_coverage(partitions, expected_blocks, b, 8)
        verify_num_splits(partitions, ns, b, 8)
        print(f"\n  all min topk={BI}: blocks={expected_blocks} splits={[ns[i+1]-ns[i] for i in range(b)]}")

    def test_one_large_rest_small(self):
        """One batch has max topk, rest have 1 block."""
        b = 4
        tl = torch.tensor([1024, 64, 64, 64], dtype=torch.int32, device='cuda')
        partitions, ns = run_scheduler(b, 1024, topk_length=tl, num_sm_parts=8)
        expected_blocks = compute_expected_blocks(b, 1024, 0, tl.tolist(), None)
        verify_coverage(partitions, expected_blocks, b, 8)
        verify_num_splits(partitions, ns, b, 8)
        print(f"\n  one large [1024,64,64,64]: blocks={expected_blocks} splits={[ns[i+1]-ns[i] for i in range(b)]}")

    def test_zero_topk(self):
        """topk_length=0 should be treated as 1."""
        b = 2
        tl = torch.tensor([0, 512], dtype=torch.int32, device='cuda')
        partitions, ns = run_scheduler(b, 512, topk_length=tl, num_sm_parts=4)
        # topk=0 → cur_s_k=1 → 1 block
        expected_blocks = [1, 512 // BI]
        verify_coverage(partitions, expected_blocks, b, 4)
        verify_num_splits(partitions, ns, b, 4)
        print(f"\n  zero topk [0,512]: blocks={expected_blocks}")

    def test_topk_not_block_aligned(self):
        """topk_length not a multiple of BI."""
        b = 3
        tl = torch.tensor([100, 200, 300], dtype=torch.int32, device='cuda')
        partitions, ns = run_scheduler(b, 512, topk_length=tl, num_sm_parts=4)
        expected_blocks = [math.ceil(100/BI), math.ceil(200/BI), math.ceil(300/BI)]
        verify_coverage(partitions, expected_blocks, b, 4)
        verify_num_splits(partitions, ns, b, 4)
        print(f"\n  unaligned topk [100,200,300]: blocks={expected_blocks}")


# ── extra_topk Tests ────────────────────────────────────────────────

class TestSchedulerExtraTopk:
    """Extra KV cache blocks appended after main blocks."""

    def test_uniform_extra(self):
        """Fixed extra_topk for all batches."""
        b = 4
        partitions, ns = run_scheduler(b, 512, extra_topk=128, num_sm_parts=8)
        expected_blocks = compute_expected_blocks(b, 512, 128, None, None)
        verify_coverage(partitions, expected_blocks, b, 8)
        verify_num_splits(partitions, ns, b, 8)
        print(f"\n  uniform extra=128: blocks={expected_blocks}")

    def test_variable_extra(self):
        """Variable extra_topk_length per batch."""
        b = 4
        etl = torch.tensor([64, 128, 0, 256], dtype=torch.int32, device='cuda')
        partitions, ns = run_scheduler(b, 512, extra_topk=256,
                                        extra_topk_length=etl, num_sm_parts=8)
        expected_blocks = compute_expected_blocks(b, 512, 256, None, etl.tolist())
        verify_coverage(partitions, expected_blocks, b, 8)
        verify_num_splits(partitions, ns, b, 8)
        print(f"\n  variable extra [64,128,0,256]: blocks={expected_blocks}")

    def test_topk_length_plus_extra(self):
        """Both topk_length and extra_topk_length variable."""
        b = 4
        tl = torch.tensor([256, 512, 128, 512], dtype=torch.int32, device='cuda')
        etl = torch.tensor([128, 64, 256, 0], dtype=torch.int32, device='cuda')
        partitions, ns = run_scheduler(b, 512, extra_topk=256,
                                        topk_length=tl, extra_topk_length=etl,
                                        num_sm_parts=8)
        expected_blocks = compute_expected_blocks(b, 512, 256, tl.tolist(), etl.tolist())
        verify_coverage(partitions, expected_blocks, b, 8)
        verify_num_splits(partitions, ns, b, 8)
        print(f"\n  topk+extra: blocks={expected_blocks}")


# ── Edge Cases ──────────────────────────────────────────────────────

class TestSchedulerEdgeCases:
    """Edge cases and stress tests."""

    def test_b_equals_1(self):
        partitions, ns = run_scheduler(1, 512, num_sm_parts=4)
        expected_blocks = [512 // BI]
        verify_coverage(partitions, expected_blocks, 1, 4)
        verify_num_splits(partitions, ns, 1, 4)

    def test_parts_equals_1(self):
        """Single partition must handle all work."""
        partitions, ns = run_scheduler(4, 512, num_sm_parts=1)
        expected_blocks = compute_expected_blocks(4, 512, 0, None, None)
        verify_coverage(partitions, expected_blocks, 4, 1)
        verify_num_splits(partitions, ns, 4, 1)
        # Single partition → all batches should have 1 split
        for i in range(4):
            assert ns[i+1] - ns[i] == 1, f"Batch {i} should have 1 split"

    def test_more_parts_than_blocks(self):
        """More partitions than total blocks → many empty partitions."""
        partitions, ns = run_scheduler(1, 64, num_sm_parts=16)  # 1 block total
        expected_blocks = [1]
        verify_coverage(partitions, expected_blocks, 1, 16)
        verify_num_splits(partitions, ns, 1, 16)

    def test_large_batch(self):
        """Stress test with large batch."""
        b = 32
        partitions, ns = run_scheduler(b, 512, num_sm_parts=64)
        expected_blocks = compute_expected_blocks(b, 512, 0, None, None)
        verify_coverage(partitions, expected_blocks, b, 64)
        verify_num_splits(partitions, ns, b, 64)

    @pytest.mark.parametrize("fixed_overhead", [0, 1, 3, 5, 10])
    def test_different_overhead(self, fixed_overhead):
        partitions, ns = run_scheduler(4, 512, num_sm_parts=8,
                                        fixed_overhead=fixed_overhead)
        expected_blocks = compute_expected_blocks(4, 512, 0, None, None)
        verify_coverage(partitions, expected_blocks, 4, 8)
        verify_num_splits(partitions, ns, 4, 8)

    def test_realistic_model1_flash(self):
        """MODEL1 Flash TP8: b=1, h=8, topk=512, 188 SMs → num_sm_parts=188."""
        num_sm_parts = max(188 // (1 * 1), 1)  # s_q=1, REPLICATE_H=1 for h=8
        partitions, ns = run_scheduler(1, 512, num_sm_parts=num_sm_parts)
        expected_blocks = [512 // BI]  # 8 blocks
        verify_coverage(partitions, expected_blocks, 1, num_sm_parts)
        verify_num_splits(partitions, ns, 1, num_sm_parts)
        total_splits = ns[1]
        print(f"\n  M1 Flash TP8: parts={num_sm_parts} splits={total_splits}")

    def test_realistic_model1_pro(self):
        """MODEL1 Pro TP1: b=1, h=128, topk=1024, num_sm_parts=188/8=23."""
        num_sm_parts = max(188 // (1 * 8), 1)
        partitions, ns = run_scheduler(1, 1024, num_sm_parts=num_sm_parts)
        expected_blocks = [1024 // BI]  # 16 blocks
        verify_coverage(partitions, expected_blocks, 1, num_sm_parts)
        verify_num_splits(partitions, ns, 1, num_sm_parts)
        total_splits = ns[1]
        print(f"\n  M1 Pro TP1: parts={num_sm_parts} splits={total_splits}")

    def test_realistic_v32(self):
        """V32 TP1: b=1, h=128, topk=2048, num_sm_parts=188/8=23."""
        num_sm_parts = max(188 // (1 * 8), 1)
        partitions, ns = run_scheduler(1, 2048, num_sm_parts=num_sm_parts)
        expected_blocks = [2048 // BI]  # 32 blocks
        verify_coverage(partitions, expected_blocks, 1, num_sm_parts)
        verify_num_splits(partitions, ns, 1, num_sm_parts)
        total_splits = ns[1]
        print(f"\n  V32 TP1: parts={num_sm_parts} splits={total_splits}")

    def test_deterministic(self):
        """Run 3x and verify results are identical (no races)."""
        results = []
        for _ in range(3):
            tl = torch.tensor([256, 512, 128, 512], dtype=torch.int32, device='cuda')
            partitions, ns = run_scheduler(4, 512, topk_length=tl, num_sm_parts=8)
            results.append((partitions, ns))
        for i in range(1, 3):
            assert results[i][1] == results[0][1], f"num_splits differs on run {i}"
            for j in range(len(results[0][0])):
                assert results[i][0][j] == results[0][0][j], \
                    f"partition {j} differs on run {i}"
        print("\n  deterministic (3 runs): PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
