#pragma once

#include "../../model/kv_cache_traits.cuh"

// Smem layout: constexpr offset computation for each buffer.
// Parameterized by ModelType, ComputeMode, and BF16_QK.
//
// BF16_QK controls whether Q nope is stored as BF16 (with online K dequant)
// or FP8 (with block-scaled MMA). Default from KVCacheTraits::USE_BF16_QK.
// Both instantiations exist for MODEL1 to allow Python-level A/B comparison.

template <ModelType MT, ComputeMode CM, bool BF16_QK = KVCacheTraits<MT>::USE_BF16_QK>
struct SmemLayout {
    using KV = KVCacheTraits<MT>;
    using CT = ComputeTraits<MT, CM>;

    static constexpr size_t SMEM_Q_NOPE = BF16_QK
        ? HPB * KV::Q_NOPE_BF16_STRIDE * sizeof(bf16)
        : HPB * KV::Q_NOPE_STRIDE;
    static constexpr size_t SMEM_Q_SC   = BF16_QK ? 0 : HPB * KV::NUM_SCALES * sizeof(float);
    static constexpr size_t SMEM_Q_ROPE = HPB * D_ROPE * sizeof(bf16);

    static constexpr size_t SMEM_KV_BUF = BI * KV::KV_SMEM_STRIDE;
    static constexpr bool NEED_SCALE_BUF = (KV::KV_SMEM_COPY_BYTES < KV::KV_SCALE_GMEM_OFFSET + KV::SCALE_BYTES_PER_TOKEN);
    static constexpr size_t SMEM_KV_SCALE_BUF = NEED_SCALE_BUF ? BI * KV::SCALE_BYTES_PER_TOKEN : 0;

    static constexpr size_t SMEM_REDUCE   = N_MATH_WARPS * HPB * sizeof(float);
    static constexpr size_t SMEM_M        = HPB * sizeof(float);
    static constexpr size_t SMEM_L        = HPB * sizeof(float);
    static constexpr size_t SMEM_W_SC_ALL = CT::N_V_CHUNKS * HPB * sizeof(float);
    static constexpr size_t SMEM_W_FP8_ONE = (CM == ComputeMode::FP8) ? HPB * (BI + 16) : 0;
    static constexpr size_t SMEM_W_FP8   = SMEM_W_FP8_ONE * CT::N_V_CHUNKS;
    static constexpr size_t SMEM_MBAR_KV = 2 * sizeof(uint64_t);

    static constexpr size_t OFF_Q_NOPE  = 0;
    static constexpr size_t OFF_Q_SC    = OFF_Q_NOPE  + SMEM_Q_NOPE;
    static constexpr size_t OFF_Q_ROPE  = OFF_Q_SC    + SMEM_Q_SC;
    static constexpr size_t OFF_KV0     = OFF_Q_ROPE  + SMEM_Q_ROPE;
    static constexpr size_t OFF_KV1     = OFF_KV0     + SMEM_KV_BUF;
    static constexpr size_t OFF_KV_SC0  = OFF_KV1     + SMEM_KV_BUF;
    static constexpr size_t OFF_KV_SC1  = OFF_KV_SC0  + SMEM_KV_SCALE_BUF;
    static constexpr size_t OFF_REDUCE  = OFF_KV_SC1  + SMEM_KV_SCALE_BUF;
    static constexpr size_t OFF_SUM_RED = OFF_REDUCE;
    static constexpr size_t OFF_M       = OFF_REDUCE  + SMEM_REDUCE;
    static constexpr size_t OFF_L       = OFF_M       + SMEM_M;
    static constexpr size_t OFF_W_SC_ALL= OFF_L       + SMEM_L;
    static constexpr size_t OFF_W_FP8   = OFF_W_SC_ALL+ SMEM_W_SC_ALL;
    static constexpr size_t OFF_MBAR_KV = (OFF_W_FP8 + SMEM_W_FP8 + 7) / 8 * 8;
    static constexpr size_t TOTAL       = OFF_MBAR_KV + SMEM_MBAR_KV;

    static_assert(TOTAL <= 101376, "SG smem exceeds 99KB per-block limit");
};

template <ModelType MT, ComputeMode CM, bool BF16_QK = KVCacheTraits<MT>::USE_BF16_QK>
struct SmemLayoutMG {
    using KV = KVCacheTraits<MT>;
    using CT = ComputeTraits<MT, CM>;
    static constexpr int N_HG = 2;

    static constexpr size_t SMEM_Q_NOPE = BF16_QK
        ? HPB * KV::Q_NOPE_BF16_STRIDE * sizeof(bf16)
        : HPB * KV::Q_NOPE_STRIDE;
    static constexpr size_t SMEM_Q_SC   = BF16_QK ? 0 : HPB * KV::NUM_SCALES * sizeof(float);
    static constexpr size_t SMEM_KV_BUF = BI * KV::KV_SMEM_STRIDE;
    static constexpr size_t SMEM_KV_SCALE_BUF = SmemLayout<MT, CM, BF16_QK>::NEED_SCALE_BUF ? BI * KV::SCALE_BYTES_PER_TOKEN : 0;

    static constexpr size_t SMEM_REDUCE_MG = N_HG * N_MATH_WARPS * HPB * sizeof(float);
    static constexpr size_t SMEM_M       = N_HG * HPB * sizeof(float);
    static constexpr size_t SMEM_L       = N_HG * HPB * sizeof(float);
    static constexpr size_t SMEM_W_SC_ALL= N_HG * CT::N_V_CHUNKS * HPB * sizeof(float);
    static constexpr size_t SMEM_W_FP8_MG= (CM == ComputeMode::FP8) ? N_HG * HPB * (BI + 16) : 0;
    static constexpr size_t SMEM_SCRATCH = N_HG * HPB * D_ROPE * sizeof(bf16);
    static constexpr size_t SMEM_MBAR_KV = 2 * sizeof(uint64_t);

    static constexpr size_t OFF_Q_NOPE0  = 0;
    static constexpr size_t OFF_Q_NOPE1  = OFF_Q_NOPE0 + SMEM_Q_NOPE;
    static constexpr size_t OFF_Q_SC0    = OFF_Q_NOPE1 + SMEM_Q_NOPE;
    static constexpr size_t OFF_Q_SC1    = OFF_Q_SC0   + SMEM_Q_SC;
    static constexpr size_t OFF_KV0      = OFF_Q_SC1   + SMEM_Q_SC;
    static constexpr size_t OFF_KV1      = OFF_KV0     + SMEM_KV_BUF;
    static constexpr size_t OFF_KV_SC0   = OFF_KV1     + SMEM_KV_BUF;
    static constexpr size_t OFF_KV_SC1   = OFF_KV_SC0  + SMEM_KV_SCALE_BUF;
    static constexpr size_t OFF_REDUCE   = OFF_KV_SC1  + SMEM_KV_SCALE_BUF;
    static constexpr size_t OFF_M        = OFF_REDUCE  + SMEM_REDUCE_MG;
    static constexpr size_t OFF_L        = OFF_M       + SMEM_M;
    static constexpr size_t OFF_W_SC_ALL = OFF_L       + SMEM_L;
    static constexpr size_t OFF_W_FP8    = OFF_W_SC_ALL+ SMEM_W_SC_ALL;
    static constexpr size_t OFF_SCRATCH  = OFF_W_FP8   + SMEM_W_FP8_MG;
    static constexpr size_t OFF_MBAR_KV  = (OFF_SCRATCH + SMEM_SCRATCH + 7) / 8 * 8;
    static constexpr size_t TOTAL        = OFF_MBAR_KV + SMEM_MBAR_KV;

    static_assert(TOTAL <= 101376, "MG smem exceeds 99KB per-block limit");
};

// SG convenience accessor
template <ModelType MT, ComputeMode CM, bool BF16_QK = KVCacheTraits<MT>::USE_BF16_QK>
struct SmemPtrs {
    using L = SmemLayout<MT, CM, BF16_QK>;

    uint8_t* q_nope_fp8;
    bf16*    q_nope_bf16;
    float*   q_nope_sc;
    bf16*    q_rope;
    uint8_t* kv_bufs[2];
    uint8_t* kv_scale_bufs[2];
    float*   reduce_buf;
    float*   sum_reduce_buf;
    float*   m_smem;
    float*   l_smem;
    float*   w_head_sc_all;
    uint8_t* w_fp8;
    uint64_t* mbar_kv;

    __device__ static SmemPtrs init(char* base) {
        SmemPtrs s;
        if constexpr (BF16_QK) {
            s.q_nope_bf16 = (bf16*)(base + L::OFF_Q_NOPE);
            s.q_nope_fp8  = nullptr;
            s.q_nope_sc   = nullptr;
        } else {
            s.q_nope_fp8  = (uint8_t*)(base + L::OFF_Q_NOPE);
            s.q_nope_bf16 = nullptr;
            s.q_nope_sc   = (float*)(base + L::OFF_Q_SC);
        }
        s.q_rope         = (bf16*)   (base + L::OFF_Q_ROPE);
        s.kv_bufs[0]     = (uint8_t*)(base + L::OFF_KV0);
        s.kv_bufs[1]     = (uint8_t*)(base + L::OFF_KV1);
        if constexpr (L::NEED_SCALE_BUF) {
            s.kv_scale_bufs[0] = (uint8_t*)(base + L::OFF_KV_SC0);
            s.kv_scale_bufs[1] = (uint8_t*)(base + L::OFF_KV_SC1);
        } else {
            s.kv_scale_bufs[0] = nullptr;
            s.kv_scale_bufs[1] = nullptr;
        }
        s.reduce_buf     = (float*)  (base + L::OFF_REDUCE);
        s.sum_reduce_buf = (float*)  (base + L::OFF_SUM_RED);
        s.m_smem         = (float*)  (base + L::OFF_M);
        s.l_smem         = (float*)  (base + L::OFF_L);
        s.w_head_sc_all  = (float*)  (base + L::OFF_W_SC_ALL);
        s.w_fp8          = (uint8_t*)(base + L::OFF_W_FP8);
        s.mbar_kv        = (uint64_t*)(base + L::OFF_MBAR_KV);
        return s;
    }
};

// MG convenience accessor
template <ModelType MT, ComputeMode CM, bool BF16_QK = KVCacheTraits<MT>::USE_BF16_QK>
struct SmemPtrsMG {
    using LMG = SmemLayoutMG<MT, CM, BF16_QK>;
    using CT = ComputeTraits<MT, CM>;

    static constexpr int N_HG = LMG::N_HG;
    static constexpr int REDUCE_GRP_STRIDE = N_MATH_WARPS * HPB;
    static constexpr int ML_GRP_STRIDE = HPB;
    static constexpr int WSC_GRP_STRIDE = CT::N_V_CHUNKS * HPB;
    static constexpr int WFP8_GRP_SIZE = HPB * (BI + 16);

    uint8_t* q_nope_fp8[N_HG];
    bf16*    q_nope_bf16[N_HG];
    float*   q_nope_sc[N_HG];
    bf16*    q_rope;
    uint8_t* kv_bufs[2];
    uint8_t* kv_scale_bufs[2];
    float*   reduce_buf;
    float*   m_smem;
    float*   l_smem;
    float*   w_head_sc_all;
    uint8_t* w_fp8;
    uint8_t* scratch;
    uint64_t* mbar_kv;

    __device__ static SmemPtrsMG init(char* base) {
        SmemPtrsMG s;
        if constexpr (BF16_QK) {
            s.q_nope_bf16[0] = (bf16*)(base + LMG::OFF_Q_NOPE0);
            s.q_nope_bf16[1] = (bf16*)(base + LMG::OFF_Q_NOPE1);
            s.q_nope_fp8[0] = nullptr; s.q_nope_fp8[1] = nullptr;
            s.q_nope_sc[0] = nullptr;  s.q_nope_sc[1] = nullptr;
        } else {
            s.q_nope_fp8[0]  = (uint8_t*)(base + LMG::OFF_Q_NOPE0);
            s.q_nope_fp8[1]  = (uint8_t*)(base + LMG::OFF_Q_NOPE1);
            s.q_nope_bf16[0] = nullptr; s.q_nope_bf16[1] = nullptr;
            s.q_nope_sc[0]   = (float*)(base + LMG::OFF_Q_SC0);
            s.q_nope_sc[1]   = (float*)(base + LMG::OFF_Q_SC1);
        }
        s.q_rope         = (bf16*)   (base + LMG::OFF_SCRATCH);
        s.kv_bufs[0]     = (uint8_t*)(base + LMG::OFF_KV0);
        s.kv_bufs[1]     = (uint8_t*)(base + LMG::OFF_KV1);
        if constexpr (SmemLayout<MT, CM, BF16_QK>::NEED_SCALE_BUF) {
            s.kv_scale_bufs[0] = (uint8_t*)(base + LMG::OFF_KV_SC0);
            s.kv_scale_bufs[1] = (uint8_t*)(base + LMG::OFF_KV_SC1);
        } else {
            s.kv_scale_bufs[0] = nullptr;
            s.kv_scale_bufs[1] = nullptr;
        }
        s.reduce_buf     = (float*)  (base + LMG::OFF_REDUCE);
        s.m_smem         = (float*)  (base + LMG::OFF_M);
        s.l_smem         = (float*)  (base + LMG::OFF_L);
        s.w_head_sc_all  = (float*)  (base + LMG::OFF_W_SC_ALL);
        s.w_fp8          = (uint8_t*)(base + LMG::OFF_W_FP8);
        s.scratch        = (uint8_t*)(base + LMG::OFF_SCRATCH);
        s.mbar_kv        = (uint64_t*)(base + LMG::OFF_MBAR_KV);
        return s;
    }
};
