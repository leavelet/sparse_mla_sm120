#pragma once

// ModelType determines KV cache layout, dimensions, and scale format.
// V32:    DeepSeek V3.2, GLM 5.1 — d_nope=512, FP32 scale inline, 656B/token
// MODEL1: DeepSeek V4 Flash/Pro  — d_nope=448, UE8M0 scale footer, 584B/token
enum class ModelType { V32, MODEL1 };

// ComputeMode determines the MMA precision path.
//
// FP8:  QK and XV use FP8 MMA (block-scaled with UE8M0).
//       Highest throughput. Q is quantized to FP8 on the fly.
//       KV remains FP8 in smem — no dequant needed.
//
// BF16: QK and XV use BF16 MMA. FP8 KV is dequantized to BF16 in smem.
//       Matches FlashMLA's precision behavior (which always uses BF16 MMA).
//       ~40-50% throughput of FP8 path, but higher accuracy.
//       FlashMLA's prefill always uses this mode.
//
// For FlashMLA compatibility, the default should match FlashMLA:
//   - prefill: BF16 (matching FlashMLA sparse_fwd)
//   - decode: configurable (FP8 for performance, BF16 for precision)
enum class ComputeMode { FP8, BF16 };
