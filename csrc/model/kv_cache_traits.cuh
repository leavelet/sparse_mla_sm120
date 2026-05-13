#pragma once

#include "model_type.h"
#include "../arch/common.cuh"

// KVCacheTraits<ModelType>: compile-time constants for KV cache layout.
//
// These determine smem strides, MMA loop counts, IO gather sizes,
// and all dimension-dependent kernel parameters.
//
// Both model types share: D_ROPE=64, D_V=512, HPB=16, BI=64.

template <ModelType MT> struct KVCacheTraits;

template <>
struct KVCacheTraits<ModelType::V32> {
    // Dimensions
    static constexpr int D_NOPE = 512;
    static constexpr int D_ROPE = 64;
    static constexpr int D_QK = D_NOPE + D_ROPE;  // 576
    static constexpr int D_V = 512;

    // FP8 quantization
    static constexpr int QUANT_TILE = 128;
    static constexpr int NUM_SCALES = D_NOPE / QUANT_TILE;  // 4

    // KV cache layout (FlashMLA ABI): INLINE, 656 bytes per token
    //   [0:512)   FP8 E4M3 nope (4 tiles × 128)
    //   [512:528) 4 × FP32 scale
    //   [528:656) BF16 rope (64 elements × 2B)
    static constexpr bool SCALE_INLINE = true;
    static constexpr int SCALE_BYTES_PER_TOKEN = NUM_SCALES * sizeof(float);  // 16
    static constexpr int KV_GMEM_STRIDE = D_NOPE + SCALE_BYTES_PER_TOKEN + D_ROPE * sizeof(bf16);  // 656
    static constexpr int KV_SCALE_GMEM_OFFSET = D_NOPE;  // 512
    static constexpr int KV_ROPE_GMEM_OFFSET = D_NOPE + SCALE_BYTES_PER_TOKEN;  // 528

    // Smem layout: bulk copy includes nope + scales (528B)
    // stride=528: 528/4=132, 132%32=4 → 4-way bank conflict (acceptable)
    static constexpr int KV_SMEM_STRIDE = D_NOPE + SCALE_BYTES_PER_TOKEN;  // 528
    static constexpr int KV_SMEM_COPY_BYTES = KV_SMEM_STRIDE;  // copy 528B per entry
    // V32: scales are within the bulk-copied region → accessible from kv_smem
    static constexpr bool SCALE_IN_KV_SMEM = true;

    // Q nope stride (padded for ldmatrix alignment + bank conflict avoidance)
    static constexpr int Q_NOPE_STRIDE = D_NOPE + 16;  // 528

    // V = pure nope (no rope component)
    static constexpr bool V_HAS_ROPE = false;

    // FP32→UE8M0 scale conversion for block-scaled MMA
    // FlashMLA stores power-of-2 FP32 scales → bit-shift gives exact UE8M0
    __device__ static __forceinline__ uint8_t scale_to_ue8m0(float scale) {
        return static_cast<uint8_t>((__float_as_uint(scale) >> 23) & 0xFF);
    }

    // V32: stays FP8 QK (Q quantized to FP8, block-scaled MMA)
    static constexpr bool USE_BF16_QK = false;
    static constexpr int Q_NOPE_BF16_STRIDE = D_NOPE + 8;  // 520 bf16 elements (unused for V32)
};

template <>
struct KVCacheTraits<ModelType::MODEL1> {
    // Dimensions
    static constexpr int D_NOPE = 448;
    static constexpr int D_ROPE = 64;
    static constexpr int D_QK = D_NOPE + D_ROPE;  // 512
    static constexpr int D_V = 512;  // = D_NOPE + D_ROPE

    // FP8 quantization
    static constexpr int QUANT_TILE = 64;
    static constexpr int NUM_SCALES = 7;  // D_NOPE / QUANT_TILE = 448/64

    // KV cache layout (FlashMLA ABI): FOOTER, 584 logical bytes per token
    // Physical layout per block (page_block_size tokens):
    //   [0 : block_size*576)                nope+rope data (576B each)
    //     per token: [0:448) FP8 nope, [448:576) BF16 rope
    //   [block_size*576 : block_size*584)   scale footer (8B each: 7×UE8M0 + 1 pad)
    //
    // stride_kv_row = 584 = logical bytes_per_token (PyTorch API stride, NOT IO stride)
    // IO stride = 576 (data only, 16B aligned for cp.async.bulk)
    static constexpr bool SCALE_INLINE = false;  // scales in footer, not inline
    static constexpr int SCALE_BYTES_PER_TOKEN = 8;
    static constexpr int KV_GMEM_STRIDE = D_NOPE + D_ROPE * sizeof(bf16) + SCALE_BYTES_PER_TOKEN;  // 584
    static constexpr int KV_ROPE_GMEM_OFFSET = D_NOPE;  // 448
    static constexpr int KV_SCALE_GMEM_OFFSET = D_NOPE + D_ROPE * sizeof(bf16);  // 576

    // Smem layout (nope only + padding, no rope, no inline scales)
    // stride=464: 464/4=116, 116%32=20 → clean (M4b benchmark verified: 12.9 ns/MMA)
    // Must be 16B aligned for cp.async.bulk: 464%16=0 ✓
    static constexpr int KV_SMEM_STRIDE = D_NOPE + 16;  // 464
    static constexpr int KV_SMEM_COPY_BYTES = D_NOPE;  // copy 448B nope per entry
    // MODEL1: scales NOT in the bulk-copied region → loaded separately to kv_scale_bufs
    static constexpr bool SCALE_IN_KV_SMEM = false;

    // Q nope stride
    static constexpr int Q_NOPE_STRIDE = D_NOPE + 16;  // 464

    // V = nope[0:448] + rope[0:64]
    // XV nope: V_CHUNK=128, pad 448→512, MMA same as V32
    // XV rope: CUDA core scalar FMA from global (zero smem, M5b verified)
    static constexpr bool V_HAS_ROPE = true;

    // UE8M0 scales are native — no conversion needed
    __device__ static __forceinline__ uint8_t scale_to_ue8m0(uint8_t scale) {
        return scale;
    }

    // MODEL1: BF16 QK (Q stays BF16, K dequanted online from FP8)
    static constexpr bool USE_BF16_QK = true;
    static constexpr int Q_NOPE_BF16_STRIDE = D_NOPE + 8;  // 456 bf16 elements (912 bytes, 4-way bank conflict)
};

// ============================================================================
// Shared constants across all model types
// ============================================================================

static constexpr int HPB = 16;
static constexpr int BI = 64;
static constexpr int D_ROPE = 64;  // universal
static constexpr int D_V = 512;    // universal

// Warp configuration
static constexpr int N_MATH_WARPS = 8;
static constexpr int N_IO_WARPS = 4;
static constexpr int N_TOTAL_WARPS = N_MATH_WARPS + N_IO_WARPS;  // 12
static constexpr int BLOCK_THREADS = N_TOTAL_WARPS * 32;          // 384
static constexpr int MATH_THREADS = N_MATH_WARPS * 32;            // 256
static constexpr int IO_THREADS = N_IO_WARPS * 32;                // 128

static constexpr int ENTRIES_PER_WARP = BI / N_MATH_WARPS;        // 8
static constexpr int N_ROPE_CHUNKS = D_ROPE / 16;                 // 4

// Output staging (reuses KV buffer after main loop)
static constexpr int OUT_STAGING_STRIDE = D_V + 8;  // 520 bf16 elements
static constexpr int OUT_VEC = 8;
static constexpr int OUT_TILES_PER_HEAD = D_V / OUT_VEC;  // 64

// ============================================================================
// ComputeMode + ModelType dependent parameters
// ============================================================================
//
// V_CHUNK = QUANT_TILE for each model (1:1 scale mapping, no max-of-tiles):
//   V32:    V_CHUNK=128 (QUANT_TILE=128, 4 chunks for D_NOPE=512)
//   MODEL1: V_CHUNK=64  (QUANT_TILE=64,  7 chunks for D_NOPE=448)
//
// FP8 mode:
//   QK nope: FP8 MMA m16n8k32
//   QK rope: BF16 MMA m16n8k16
//   XV nope: FP8 MMA m16n8k32 (W quantized to FP8, V stays FP8)
//   XV rope (MODEL1): BF16 MMA m16n8k16 (B from global, L2 cached)
//   Byte transpose required for V (FP8)
//
// BF16 mode:
//   IO dequants FP8 KV → BF16 in smem
//   QK/XV: BF16 MMA m16n8k16
//   V in smem is BF16 → ldmatrix.x2.trans (no byte transpose)

template <ModelType MT, ComputeMode CM>
struct ComputeTraits;

template <ModelType MT>
struct ComputeTraits<MT, ComputeMode::FP8> {
    using KV = KVCacheTraits<MT>;
    static constexpr int V_CHUNK = KV::QUANT_TILE;                        // V32=128, MODEL1=64
    static constexpr int N_V_CHUNKS = KV::D_NOPE / V_CHUNK;              // V32=4, MODEL1=7
    static constexpr int V_TRANS_STRIDE = BI + 16;                        // 80
    static constexpr int W_FP8_STRIDE = BI + 16;                          // 80
    static constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / N_MATH_WARPS;    // V32=2, MODEL1=1
    static constexpr int ACC_TILES = N_V_CHUNKS * NT_PER_WARP_XV;        // V32=8, MODEL1=7
    static constexpr int XV_KSTEPS = BI / 32;                             // 2 (FP8 k=32)
};

template <ModelType MT>
struct ComputeTraits<MT, ComputeMode::BF16> {
    using KV = KVCacheTraits<MT>;
    static constexpr int V_CHUNK = KV::QUANT_TILE;
    static constexpr int N_V_CHUNKS = KV::D_NOPE / V_CHUNK;
    static constexpr int V_TRANS_STRIDE = BI + 8;                          // 72 bf16 elements
    static constexpr int W_FP8_STRIDE = 0;
    static constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / N_MATH_WARPS;
    static constexpr int ACC_TILES = N_V_CHUNKS * NT_PER_WARP_XV;
    static constexpr int XV_KSTEPS = BI / 16;                             // 4 (BF16 k=16)
};
