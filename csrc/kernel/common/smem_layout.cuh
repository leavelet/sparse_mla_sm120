#pragma once

#include "../../model/kv_cache_traits.cuh"

// Smem layout: constexpr offset computation for each buffer.
// Parameterized by ModelType and ComputeMode.
//
// Buffers (decode / prefill SG):
//   q_nope_fp8, q_nope_sc, q_rope, kv_buf×2, [kv_scale_buf×2 for MODEL1],
//   reduce_buf, sum_reduce_buf (or union), m_smem, l_smem,
//   w_head_sc_all, w_fp8 (FP8 mode), v_trans, mbar_kv
//
// All offsets are in bytes.

template <ModelType MT, ComputeMode CM>
struct SmemLayout {
    using KV = KVCacheTraits<MT>;
    using CT = ComputeTraits<MT, CM>;

    // Q buffers
    static constexpr size_t SMEM_Q_NOPE     = HPB * KV::Q_NOPE_STRIDE;
    static constexpr size_t SMEM_Q_SC       = HPB * KV::NUM_SCALES * sizeof(float);
    static constexpr size_t SMEM_Q_ROPE     = HPB * D_ROPE * sizeof(bf16);

    // KV double buffer
    static constexpr size_t SMEM_KV_BUF     = BI * KV::KV_SMEM_STRIDE;

    // KV scale buffer: needed when bulk copy doesn't include scales.
    // V32: copies 528B (nope+scale), scales in kv_smem → no extra buffer.
    // MODEL1: copies 448B (nope only), scales at offset 576 → need separate buffer.
    static constexpr bool NEED_SCALE_BUF = (KV::KV_SMEM_COPY_BYTES < KV::KV_SCALE_GMEM_OFFSET + KV::SCALE_BYTES_PER_TOKEN);
    static constexpr size_t SMEM_KV_SCALE_BUF = NEED_SCALE_BUF ? BI * KV::SCALE_BYTES_PER_TOKEN : 0;

    // Cross-warp reduction (O1 overlap: reduce_buf and sum_reduce_buf share memory)
    static constexpr size_t SMEM_REDUCE     = N_MATH_WARPS * HPB * sizeof(float);

    // Per-head online softmax state
    static constexpr size_t SMEM_M          = HPB * sizeof(float);
    static constexpr size_t SMEM_L          = HPB * sizeof(float);

    // XV phase — w_fp8 always double-buffered (D2 eliminates v_trans)
    static constexpr size_t SMEM_W_SC_ALL   = CT::N_V_CHUNKS * HPB * sizeof(float);
    static constexpr size_t SMEM_W_FP8_ONE  = (CM == ComputeMode::FP8) ? HPB * (BI + 16) : 0;
    static constexpr size_t SMEM_W_FP8      = SMEM_W_FP8_ONE * 2;

    // Mbarrier (double-buffered)
    static constexpr size_t SMEM_MBAR_KV    = 2 * sizeof(uint64_t);

    // Offsets — O1 overlap: reduce_buf and sum_reduce_buf share the same memory
    static constexpr size_t OFF_Q_NOPE    = 0;
    static constexpr size_t OFF_Q_SC      = OFF_Q_NOPE    + SMEM_Q_NOPE;
    static constexpr size_t OFF_Q_ROPE    = OFF_Q_SC      + SMEM_Q_SC;
    static constexpr size_t OFF_KV0       = OFF_Q_ROPE    + SMEM_Q_ROPE;
    static constexpr size_t OFF_KV1       = OFF_KV0       + SMEM_KV_BUF;
    static constexpr size_t OFF_KV_SC0    = OFF_KV1       + SMEM_KV_BUF;
    static constexpr size_t OFF_KV_SC1    = OFF_KV_SC0    + SMEM_KV_SCALE_BUF;
    static constexpr size_t OFF_REDUCE    = OFF_KV_SC1    + SMEM_KV_SCALE_BUF;
    static constexpr size_t OFF_SUM_RED   = OFF_REDUCE;  // O1: shares memory with reduce_buf
    static constexpr size_t OFF_M         = OFF_REDUCE    + SMEM_REDUCE;
    static constexpr size_t OFF_L         = OFF_M         + SMEM_M;
    static constexpr size_t OFF_W_SC_ALL  = OFF_L         + SMEM_L;
    static constexpr size_t OFF_W_FP8_0   = OFF_W_SC_ALL  + SMEM_W_SC_ALL;
    static constexpr size_t OFF_W_FP8_1   = OFF_W_FP8_0   + SMEM_W_FP8_ONE;
    static constexpr size_t OFF_MBAR_KV   = (OFF_W_FP8_0 + SMEM_W_FP8 + 7) / 8 * 8;
    static constexpr size_t TOTAL         = OFF_MBAR_KV   + SMEM_MBAR_KV;

    static_assert(TOTAL <= 101376, "SG smem exceeds 99KB per-block limit");
};

// MG (multi-group) layout: 2 head groups, reduce/sum_reduce union (O1)
template <ModelType MT, ComputeMode CM>
struct SmemLayoutMG {
    using KV = KVCacheTraits<MT>;
    using CT = ComputeTraits<MT, CM>;
    static constexpr int N_HG = 2;

    static constexpr size_t SMEM_Q_NOPE     = HPB * KV::Q_NOPE_STRIDE;
    static constexpr size_t SMEM_Q_SC       = HPB * KV::NUM_SCALES * sizeof(float);
    static constexpr size_t SMEM_KV_BUF     = BI * KV::KV_SMEM_STRIDE;
    static constexpr size_t SMEM_KV_SCALE_BUF = SmemLayout<MT, CM>::NEED_SCALE_BUF ? BI * KV::SCALE_BYTES_PER_TOKEN : 0;

    // O1 overlap: reduce_buf and sum_reduce_buf share the same memory
    static constexpr size_t SMEM_REDUCE_MG  = N_HG * N_MATH_WARPS * HPB * sizeof(float);

    static constexpr size_t SMEM_M          = N_HG * HPB * sizeof(float);
    static constexpr size_t SMEM_L          = N_HG * HPB * sizeof(float);
    static constexpr size_t SMEM_W_SC_ALL   = N_HG * CT::N_V_CHUNKS * HPB * sizeof(float);
    static constexpr size_t SMEM_W_FP8_MG   = (CM == ComputeMode::FP8) ? N_HG * HPB * (BI + 16) : 0;
    // Scratch area: q_rope staging (O2 overlap) and xv_rope weight staging
    // q_rope needs: N_HG * HPB * D_ROPE * sizeof(bf16) = 4096 bytes
    // xv_rope needs: HPB * BI * sizeof(bf16) = 2048 bytes
    static constexpr size_t SMEM_SCRATCH    = N_HG * HPB * D_ROPE * sizeof(bf16);
    static constexpr size_t SMEM_MBAR_KV    = 2 * sizeof(uint64_t);

    static constexpr size_t OFF_Q_NOPE0   = 0;
    static constexpr size_t OFF_Q_NOPE1   = OFF_Q_NOPE0   + SMEM_Q_NOPE;
    static constexpr size_t OFF_Q_SC0     = OFF_Q_NOPE1   + SMEM_Q_NOPE;
    static constexpr size_t OFF_Q_SC1     = OFF_Q_SC0     + SMEM_Q_SC;
    static constexpr size_t OFF_KV0       = OFF_Q_SC1     + SMEM_Q_SC;
    static constexpr size_t OFF_KV1       = OFF_KV0       + SMEM_KV_BUF;
    static constexpr size_t OFF_KV_SC0    = OFF_KV1       + SMEM_KV_BUF;
    static constexpr size_t OFF_KV_SC1    = OFF_KV_SC0    + SMEM_KV_SCALE_BUF;
    // O1: single buffer used as both reduce and sum_reduce
    static constexpr size_t OFF_REDUCE    = OFF_KV_SC1    + SMEM_KV_SCALE_BUF;
    static constexpr size_t OFF_M         = OFF_REDUCE    + SMEM_REDUCE_MG;
    static constexpr size_t OFF_L         = OFF_M         + SMEM_M;
    static constexpr size_t OFF_W_SC_ALL  = OFF_L         + SMEM_L;
    static constexpr size_t OFF_W_FP8     = OFF_W_SC_ALL  + SMEM_W_SC_ALL;
    // Scratch: q_rope staging (O2 overlap) + xv_rope weight staging
    static constexpr size_t OFF_SCRATCH   = OFF_W_FP8     + SMEM_W_FP8_MG;
    static constexpr size_t OFF_MBAR_KV   = (OFF_SCRATCH + SMEM_SCRATCH + 7) / 8 * 8;
    static constexpr size_t TOTAL         = OFF_MBAR_KV   + SMEM_MBAR_KV;

    static_assert(TOTAL <= 101376, "MG smem exceeds 99KB per-block limit");
};

// MG convenience accessor
template <ModelType MT, ComputeMode CM>
struct SmemPtrsMG {
    using LMG = SmemLayoutMG<MT, CM>;
    using CT = ComputeTraits<MT, CM>;

    static constexpr int N_HG = LMG::N_HG;
    static constexpr int REDUCE_GRP_STRIDE = N_MATH_WARPS * HPB;
    static constexpr int ML_GRP_STRIDE = HPB;
    static constexpr int WSC_GRP_STRIDE = CT::N_V_CHUNKS * HPB;
    static constexpr int WFP8_GRP_SIZE = HPB * (BI + 16);

    uint8_t* q_nope_fp8[N_HG];
    float*   q_nope_sc[N_HG];
    bf16*    q_rope;           // O2: both groups stored sequentially in v_trans area
    uint8_t* kv_bufs[2];
    uint8_t* kv_scale_bufs[2];
    float*   reduce_buf;       // [N_HG * N_MATH_WARPS * HPB]
    float*   m_smem;           // [N_HG * HPB]
    float*   l_smem;           // [N_HG * HPB]
    float*   w_head_sc_all;    // [N_HG * N_V_CHUNKS * HPB]
    uint8_t* w_fp8;            // [N_HG * HPB * (BI+16)]
    uint8_t* scratch;          // q_rope staging + xv_rope weight staging
    uint64_t* mbar_kv;

    __device__ static SmemPtrsMG init(char* base) {
        SmemPtrsMG s;
        s.q_nope_fp8[0]  = (uint8_t*)(base + LMG::OFF_Q_NOPE0);
        s.q_nope_fp8[1]  = (uint8_t*)(base + LMG::OFF_Q_NOPE1);
        s.q_nope_sc[0]   = (float*)  (base + LMG::OFF_Q_SC0);
        s.q_nope_sc[1]   = (float*)  (base + LMG::OFF_Q_SC1);
        s.q_rope         = (bf16*)   (base + LMG::OFF_SCRATCH);
        s.kv_bufs[0]     = (uint8_t*)(base + LMG::OFF_KV0);
        s.kv_bufs[1]     = (uint8_t*)(base + LMG::OFF_KV1);
        if constexpr (SmemLayout<MT,CM>::NEED_SCALE_BUF) {
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

// SG convenience accessor (initialized from smem base pointer)
template <ModelType MT, ComputeMode CM>
struct SmemPtrs {
    using L = SmemLayout<MT, CM>;

    uint8_t* q_nope_fp8;
    float*   q_nope_sc;
    bf16*    q_rope;
    uint8_t* kv_bufs[2];
    uint8_t* kv_scale_bufs[2];  // nullptr for V32 (inline scales)
    float*   reduce_buf;
    float*   sum_reduce_buf;
    float*   m_smem;
    float*   l_smem;
    float*   w_head_sc_all;
    uint8_t* w_fp8_bufs[2];
    uint64_t* mbar_kv;

    __device__ static SmemPtrs init(char* base) {
        SmemPtrs s;
        s.q_nope_fp8     = (uint8_t*)(base + L::OFF_Q_NOPE);
        s.q_nope_sc      = (float*)  (base + L::OFF_Q_SC);
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
        s.w_fp8_bufs[0]  = (uint8_t*)(base + L::OFF_W_FP8_0);
        s.w_fp8_bufs[1]  = (uint8_t*)(base + L::OFF_W_FP8_1);
        s.mbar_kv        = (uint64_t*)(base + L::OFF_MBAR_KV);
        return s;
    }
};
