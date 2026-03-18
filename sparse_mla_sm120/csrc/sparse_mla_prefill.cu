#include "common.cuh"
#include "mma_sm120.cuh"
#include "smem_utils.cuh"

#include <torch/extension.h>

// ============================================================================
// Sparse MLA Prefill — Warp-specialized, BI=64, rope-from-global
//
// 12 warps (384 threads): 8 math (warps 0-7) + 4 IO (warps 8-11)
// KV smem stores only nope+scales (528B/entry), rope read from global (L1 cached)
// Double-buffered KV: 2 × 64 × 528 = 66 KB → total smem ~86 KB < 100 KB
//
// Barriers: bar 0 = data ready, bar 1 = buf consumed, bar 2 = math internal
// ============================================================================

static constexpr int HPB = 16;
static constexpr int BI = 64;
static constexpr int N_MATH_WARPS = 8;
static constexpr int N_TOTAL_WARPS = 12;
static constexpr int BLOCK_THREADS = N_TOTAL_WARPS * 32;
static constexpr int MATH_THREADS = N_MATH_WARPS * 32;
static constexpr int IO_THREADS = (N_TOTAL_WARPS - N_MATH_WARPS) * 32;
static constexpr int V_CHUNK = 128;
static constexpr int N_V_CHUNKS = D_NOPE / V_CHUNK;
static constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / N_MATH_WARPS;
static constexpr int ACC_TILES = N_V_CHUNKS * NT_PER_WARP_XV;

// KV entry in smem: nope FP8 (512B) + scales F32 (16B) = 528B (no rope)
static constexpr int KV_SMEM_STRIDE = D_NOPE + NUM_SCALES * sizeof(float);  // 528

// Padded strides to avoid smem bank conflicts (need stride/4 coprime with 32)
static constexpr int Q_NOPE_STRIDE = D_NOPE + 16;  // 528: 132%32=4 ✓
static constexpr int V_TRANS_STRIDE = BI + 4;       // 68:  17%32=17 ✓
static constexpr int W_FP8_STRIDE = BI + 4;         // 68:  17%32=17 ✓ (was 64: 16%32=16 → 2-way conflict)

__device__ __forceinline__ void bar_arrive(int id, int cnt) {
    asm volatile("barrier.cta.arrive %0, %1;\n" :: "r"(id), "r"(cnt) : "memory");
}
__device__ __forceinline__ void bar_sync(int id, int cnt) {
    asm volatile("barrier.cta.sync %0, %1;\n" :: "r"(id), "r"(cnt) : "memory");
}

__device__ __forceinline__ void load_A_fp8(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    const uint8_t* s, int stride, int ko, int lane) {
    int g=lane>>2,t=lane&3;
    const uint8_t *r0=s+g*stride+ko, *r1=s+(g+8)*stride+ko;
    a0=*(const uint32_t*)(r0+t*4); a1=*(const uint32_t*)(r1+t*4);
    a2=*(const uint32_t*)(r0+16+t*4); a3=*(const uint32_t*)(r1+16+t*4);
}
__device__ __forceinline__ void load_B_fp8(
    uint32_t& b0, uint32_t& b1,
    const uint8_t* s, int stride, int nb, int ko, int lane) {
    int g=lane>>2,t=lane&3;
    const uint8_t* r=s+(nb+g)*stride+ko;
    b0=*(const uint32_t*)(r+t*4); b1=*(const uint32_t*)(r+16+t*4);
}
__device__ __forceinline__ void load_A_bf16(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    const bf16* s, int stride, int ko, int lane) {
    int g=lane>>2,t=lane&3;
    const bf16 *r0=s+g*stride+ko, *r1=s+(g+8)*stride+ko;
    a0=*(const uint32_t*)(r0+t*2); a1=*(const uint32_t*)(r1+t*2);
    a2=*(const uint32_t*)(r0+8+t*2); a3=*(const uint32_t*)(r1+8+t*2);
}

__global__ void __launch_bounds__(BLOCK_THREADS, 1)
sparse_mla_prefill_mma_kernel(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    bf16* __restrict__ output,
    float sm_scale, int num_heads, int num_tokens, int topk)
{
    const int NI = topk / BI;
    const int REPLICATE_H = num_heads / HPB;
    const int s_i = blockIdx.x / REPLICATE_H;
    const int h_tile = blockIdx.x % REPLICATE_H;
    const int h_start = h_tile * HPB;
    if (s_i >= num_tokens) return;

    const int warp_rank = threadIdx.x / 32;
    const int wy = warp_rank / 4;

    extern __shared__ char smem_raw[];
    char* sp = smem_raw;
    uint8_t* q_nope_fp8 = (uint8_t*)sp;  sp += HPB * Q_NOPE_STRIDE;
    float*   q_nope_sc  = (float*)sp;    sp += HPB * NUM_SCALES * 4;
    bf16*    q_rope_smem= (bf16*)sp;     sp += HPB * D_ROPE * 2;
    uint8_t* kv_buf0    = (uint8_t*)sp;  sp += BI * KV_SMEM_STRIDE;
    uint8_t* kv_buf1    = (uint8_t*)sp;  sp += BI * KV_SMEM_STRIDE;
    float*   reduce_buf = (float*)sp;    sp += N_MATH_WARPS * HPB * 4;
    float*   m_smem     = (float*)sp;    sp += HPB * 4;
    float*   l_smem     = (float*)sp;    sp += HPB * 4;
    uint8_t* w_fp8      = (uint8_t*)sp;  sp += HPB * W_FP8_STRIDE;
    float*   w_head_sc  = (float*)sp;    sp += HPB * 4;
    uint8_t* v_trans    = (uint8_t*)sp;  sp += V_CHUNK * V_TRANS_STRIDE;
    uint8_t* kv_bufs[2] = {kv_buf0, kv_buf1};

    // ================================================================
    // setmaxnreg FIRST — before any heavy code
    // ================================================================
    if (wy == 2) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));

        // ============================================================
        // IO PATH: gather KV nope+scales (skip rope) into double buffer
        // ============================================================
        int io_tid = threadIdx.x - N_MATH_WARPS * 32;
        const int32_t* idx_base = indices + (size_t)s_i * topk;

        auto gather_tile = [&](uint8_t* dst, int tile_idx) {
            const int32_t* ib = idx_base + tile_idx * BI;
            // Load nope (512B) + scales (16B) = 528B per entry, skip rope
            constexpr int TOT = BI * KV_SMEM_STRIDE;
            for (int flat = io_tid * 16; flat < TOT; flat += IO_THREADS * 16) {
                int bi = flat / KV_SMEM_STRIDE;
                int bo = flat % KV_SMEM_STRIDE;
                int idx = ib[bi]; idx = (idx >= 0) ? idx : 0;
                if (bo + 16 <= KV_SMEM_STRIDE)
                    cp_async_16B(dst + flat,
                                 KV_cache + (size_t)idx * KV_PACKED_BYTES + bo);
            }
            cp_async_commit();
            cp_async_wait_all();
        };

        gather_tile(kv_bufs[0], 0);
        bar_arrive(0, BLOCK_THREADS);

        for (int ni = 0; ni < NI; ni++) {
            if (ni + 1 < NI)
                gather_tile(kv_bufs[(ni + 1) & 1], ni + 1);
            bar_sync(1, BLOCK_THREADS);
            if (ni + 1 < NI)
                bar_arrive(0, BLOCK_THREADS);
        }

    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(200));

        // ============================================================
        // MATH PATH: Q quant, QK+softmax+XV main loop
        // ============================================================
        const int lane = threadIdx.x & 31;
        const int mwarp = warp_rank;
        const int gid = lane >> 2, tid = lane & 3;
        const float sm_scale_log2e = sm_scale * LOG2E;
        const bf16* q_base = Q + (size_t)s_i * num_heads * DIM + (size_t)h_start * DIM;
        const int32_t* idx_base = indices + (size_t)s_i * topk;

        // Q quantization
        for (int i = threadIdx.x; i < HPB*D_ROPE; i += MATH_THREADS)
            q_rope_smem[(i/D_ROPE)*D_ROPE+i%D_ROPE] = q_base[(i/D_ROPE)*DIM+D_NOPE+i%D_ROPE];
        float* amax = reduce_buf;
        for (int i = threadIdx.x; i < HPB*NUM_SCALES; i += MATH_THREADS) amax[i] = 0.f;
        bar_sync(2, MATH_THREADS);
        for (int idx = threadIdx.x; idx < HPB*D_NOPE; idx += MATH_THREADS) {
            int h=idx/D_NOPE, blk=(idx%D_NOPE)/QUANT_TILE;
            atomicMax(reinterpret_cast<int*>(&amax[h*NUM_SCALES+blk]),
                      __float_as_int(fabsf(__bfloat162float(q_base[h*DIM+idx%D_NOPE]))));
        }
        bar_sync(2, MATH_THREADS);
        for (int i = threadIdx.x; i < HPB*NUM_SCALES; i += MATH_THREADS)
            q_nope_sc[i] = fmaxf(amax[i], 1e-4f) / FP8_MAX;
        bar_sync(2, MATH_THREADS);
        for (int idx = threadIdx.x; idx < HPB*D_NOPE; idx += MATH_THREADS) {
            int h=idx/D_NOPE, d=idx%D_NOPE, blk=d/QUANT_TILE;
            float si=1.f/q_nope_sc[h*NUM_SCALES+blk];
            float v=fmaxf(FP8_MIN,fminf(FP8_MAX,__bfloat162float(q_base[h*DIM+d])*si));
            __nv_fp8_e4m3 fp8v(v); q_nope_fp8[h*Q_NOPE_STRIDE+d]=fp8v.__x;
        }
        for (int h = threadIdx.x; h < HPB; h += MATH_THREADS) { m_smem[h]=-1e30f; l_smem[h]=0.f; }
        bar_sync(2, MATH_THREADS);

        float acc_o[ACC_TILES][4];
        #pragma unroll
        for (int t = 0; t < ACC_TILES; t++)
            acc_o[t][0]=acc_o[t][1]=acc_o[t][2]=acc_o[t][3]=0.f;

        bar_sync(0, BLOCK_THREADS);  // wait for IO tile 0

        for (int ni = 0; ni < NI; ni++) {
            uint8_t* kv_smem = kv_bufs[ni & 1];
            const int32_t* ib = idx_base + ni * BI;

            // ---- QK MMA nope (from smem, stride=528) ----
            int qk_nb = mwarp * 8;
            float qk[4] = {0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int blk = 0; blk < NUM_SCALES; blk++) {
                float ab[4] = {0.f,0.f,0.f,0.f};
                #pragma unroll
                for (int ks = 0; ks < QUANT_TILE/32; ks++) {
                    int ko = blk*QUANT_TILE+ks*32;
                    uint32_t a0,a1,a2,a3,b0,b1;
                    load_A_fp8(a0,a1,a2,a3, q_nope_fp8, Q_NOPE_STRIDE, ko, lane);
                    load_B_fp8(b0,b1, kv_smem, KV_SMEM_STRIDE, qk_nb, ko, lane);
                    MmaFp8Result r=mma_fp8_m16n8k32(a0,a1,a2,a3,b0,b1,ab[0],ab[1],ab[2],ab[3]);
                    ab[0]=r.d0;ab[1]=r.d1;ab[2]=r.d2;ab[3]=r.d3;
                }
                float qs0=q_nope_sc[gid*NUM_SCALES+blk], qs1=q_nope_sc[(gid+8)*NUM_SCALES+blk];
                int e0=qk_nb+tid*2, e1=qk_nb+tid*2+1;
                float k0=reinterpret_cast<const float*>(kv_smem+e0*KV_SMEM_STRIDE+D_NOPE)[blk];
                float k1=reinterpret_cast<const float*>(kv_smem+e1*KV_SMEM_STRIDE+D_NOPE)[blk];
                qk[0]+=ab[0]*qs0*k0; qk[1]+=ab[1]*qs0*k1;
                qk[2]+=ab[2]*qs1*k0; qk[3]+=ab[3]*qs1*k1;
            }

            // ---- QK MMA rope (from GLOBAL, L1 cached) ----
            {
                float ra[4]={0.f,0.f,0.f,0.f};
                int ne = qk_nb + gid;
                int entry_idx = ib[ne]; entry_idx = (entry_idx >= 0) ? entry_idx : 0;
                const bf16* g_rope = reinterpret_cast<const bf16*>(
                    KV_cache + (size_t)entry_idx * KV_PACKED_BYTES + D_NOPE + NUM_SCALES * sizeof(float));
                #pragma unroll
                for (int ks = 0; ks < D_ROPE/16; ks++) {
                    int ko = ks*16;
                    uint32_t a0,a1,a2,a3,b0,b1;
                    load_A_bf16(a0,a1,a2,a3, q_rope_smem, D_ROPE, ko, lane);
                    // B from global memory (L1 cached from KV gather)
                    b0 = *reinterpret_cast<const uint32_t*>(g_rope + ko + tid*2);
                    b1 = *reinterpret_cast<const uint32_t*>(g_rope + ko + 8 + tid*2);
                    MmaBf16Result r=mma_bf16_m16n8k16(a0,a1,a2,a3,b0,b1,ra[0],ra[1],ra[2],ra[3]);
                    ra[0]=r.d0;ra[1]=r.d1;ra[2]=r.d2;ra[3]=r.d3;
                }
                qk[0]+=ra[0];qk[1]+=ra[1];qk[2]+=ra[2];qk[3]+=ra[3];
            }

            // Mask
            { int e0=qk_nb+tid*2,e1=qk_nb+tid*2+1;
              if(ib[e0]<0){qk[0]=-1e30f;qk[2]=-1e30f;}
              if(ib[e1]<0){qk[1]=-1e30f;qk[3]=-1e30f;} }

            // ---- Online softmax ----
            float s[4]={qk[0]*sm_scale_log2e,qk[1]*sm_scale_log2e,qk[2]*sm_scale_log2e,qk[3]*sm_scale_log2e};
            float lm0=fmaxf(s[0],s[1]),lm1=fmaxf(s[2],s[3]);
            lm0=fmaxf(lm0,__shfl_xor_sync(0xffffffff,lm0,1));
            lm0=fmaxf(lm0,__shfl_xor_sync(0xffffffff,lm0,2));
            lm1=fmaxf(lm1,__shfl_xor_sync(0xffffffff,lm1,1));
            lm1=fmaxf(lm1,__shfl_xor_sync(0xffffffff,lm1,2));
            if(tid==0){reduce_buf[mwarp*HPB+gid]=lm0;reduce_buf[mwarp*HPB+gid+8]=lm1;}
            bar_sync(2,MATH_THREADS);
            if(threadIdx.x<HPB){int h=threadIdx.x;float old_m=m_smem[h],tm=-1e30f;
              for(int w=0;w<N_MATH_WARPS;w++)tm=fmaxf(tm,reduce_buf[w*HPB+h]);
              float nm=fmaxf(old_m,tm),alpha=exp2f(old_m-nm);
              m_smem[h]=nm;l_smem[h]*=alpha;reduce_buf[h]=alpha;reduce_buf[HPB+h]=nm;}
            bar_sync(2,MATH_THREADS);
            float alpha0=reduce_buf[gid],alpha1=reduce_buf[gid+8];
            #pragma unroll
            for(int t=0;t<ACC_TILES;t++){acc_o[t][0]*=alpha0;acc_o[t][1]*=alpha0;acc_o[t][2]*=alpha1;acc_o[t][3]*=alpha1;}
            float nm0=reduce_buf[HPB+gid],nm1=reduce_buf[HPB+gid+8];
            float w0=exp2f(s[0]-nm0),w1=exp2f(s[1]-nm0),w2=exp2f(s[2]-nm1),w3=exp2f(s[3]-nm1);
            bar_sync(2,MATH_THREADS);
            float ls0=w0+w1,ls1=w2+w3;
            ls0+=__shfl_xor_sync(0xffffffff,ls0,1);ls0+=__shfl_xor_sync(0xffffffff,ls0,2);
            ls1+=__shfl_xor_sync(0xffffffff,ls1,1);ls1+=__shfl_xor_sync(0xffffffff,ls1,2);
            if(tid==0){reduce_buf[mwarp*HPB+gid]=ls0;reduce_buf[mwarp*HPB+gid+8]=ls1;}
            bar_sync(2,MATH_THREADS);
            if(threadIdx.x<HPB){int h=threadIdx.x;float ts=0.f;
              for(int w=0;w<N_MATH_WARPS;w++)ts+=reduce_buf[w*HPB+h];l_smem[h]+=ts;}
            bar_sync(2,MATH_THREADS);

            // ---- XV MMA (V data from smem at stride=528) ----
            #pragma unroll
            for(int vc=0;vc<N_V_CHUNKS;vc++){
                int v_off=vc*V_CHUNK;
                for(int h=threadIdx.x;h<HPB;h+=MATH_THREADS)w_head_sc[h]=0.f;
                bar_sync(2,MATH_THREADS);
                int e0i=qk_nb+tid*2,e1i=qk_nb+tid*2+1;
                float vsc0=reinterpret_cast<const float*>(kv_smem+e0i*KV_SMEM_STRIDE+D_NOPE)[vc];
                float vsc1=reinterpret_cast<const float*>(kv_smem+e1i*KV_SMEM_STRIDE+D_NOPE)[vc];
                float ws00=w0*vsc0,ws01=w1*vsc1,ws10=w2*vsc0,ws11=w3*vsc1;
                atomicMax(reinterpret_cast<int*>(&w_head_sc[gid]),__float_as_int(fmaxf(fabsf(ws00),fabsf(ws01))));
                atomicMax(reinterpret_cast<int*>(&w_head_sc[gid+8]),__float_as_int(fmaxf(fabsf(ws10),fabsf(ws11))));
                bar_sync(2,MATH_THREADS);
                for(int h=threadIdx.x;h<HPB;h+=MATH_THREADS)w_head_sc[h]=fmaxf(w_head_sc[h],1e-10f)/FP8_MAX;
                bar_sync(2,MATH_THREADS);
                {float si0=1.f/w_head_sc[gid],si1=1.f/w_head_sc[gid+8];
                 __nv_fp8_e4m3 f00(fmaxf(-FP8_MAX,fminf(FP8_MAX,ws00*si0)));
                 __nv_fp8_e4m3 f01(fmaxf(-FP8_MAX,fminf(FP8_MAX,ws01*si0)));
                 __nv_fp8_e4m3 f10(fmaxf(-FP8_MAX,fminf(FP8_MAX,ws10*si1)));
                 __nv_fp8_e4m3 f11(fmaxf(-FP8_MAX,fminf(FP8_MAX,ws11*si1)));
                 w_fp8[gid*W_FP8_STRIDE+e0i]=f00.__x;w_fp8[gid*W_FP8_STRIDE+e1i]=f01.__x;
                 w_fp8[(gid+8)*W_FP8_STRIDE+e0i]=f10.__x;w_fp8[(gid+8)*W_FP8_STRIDE+e1i]=f11.__x;}
                // V transpose from smem (stride=528, nope part only)
                for(int idx=threadIdx.x;idx<V_CHUNK*BI;idx+=MATH_THREADS){
                    int d=idx/BI,e=idx%BI;
                    v_trans[d*V_TRANS_STRIDE+e]=kv_smem[e*KV_SMEM_STRIDE+v_off+d];}
                bar_sync(2,MATH_THREADS);
                #pragma unroll
                for(int nt=0;nt<NT_PER_WARP_XV;nt++){
                    int ti=vc*NT_PER_WARP_XV+nt,nl=mwarp*(NT_PER_WARP_XV*8)+nt*8;
                    float xv[4]={0.f,0.f,0.f,0.f};
                    #pragma unroll
                    for(int kstep=0;kstep<BI/32;kstep++){
                        int ko=kstep*32; uint32_t a0,a1,a2,a3,b0,b1;
                        load_A_fp8(a0,a1,a2,a3,w_fp8,W_FP8_STRIDE,ko,lane);
                        load_B_fp8(b0,b1,v_trans,V_TRANS_STRIDE,nl,ko,lane);
                        MmaFp8Result r=mma_fp8_m16n8k32(a0,a1,a2,a3,b0,b1,xv[0],xv[1],xv[2],xv[3]);
                        xv[0]=r.d0;xv[1]=r.d1;xv[2]=r.d2;xv[3]=r.d3;}
                    float sc0=w_head_sc[gid],sc1=w_head_sc[gid+8];
                    acc_o[ti][0]+=xv[0]*sc0;acc_o[ti][1]+=xv[1]*sc0;
                    acc_o[ti][2]+=xv[2]*sc1;acc_o[ti][3]+=xv[3]*sc1;}
                bar_sync(2,MATH_THREADS);
            }

            // Signal IO: consumed
            bar_arrive(1, BLOCK_THREADS);
            if (ni + 1 < NI)
                bar_sync(0, BLOCK_THREADS);
        }

        // Finalize
        #pragma unroll
        for(int t=0;t<ACC_TILES;t++){
            int c=t/NT_PER_WARP_XV,lnt=t%NT_PER_WARP_XV;
            int vb=c*V_CHUNK+mwarp*(NT_PER_WARP_XV*8)+lnt*8;
            int h0=h_start+gid,h1=h_start+gid+8,d0=vb+tid*2,d1=vb+tid*2+1;
            float il0=(l_smem[gid]>0.f)?(1.f/l_smem[gid]):0.f;
            float il1=(l_smem[gid+8]>0.f)?(1.f/l_smem[gid+8]):0.f;
            size_t b0=(size_t)s_i*num_heads*D_NOPE+(size_t)h0*D_NOPE;
            size_t b1=(size_t)s_i*num_heads*D_NOPE+(size_t)h1*D_NOPE;
            output[b0+d0]=__float2bfloat16(acc_o[t][0]*il0);
            output[b0+d1]=__float2bfloat16(acc_o[t][1]*il0);
            output[b1+d0]=__float2bfloat16(acc_o[t][2]*il1);
            output[b1+d1]=__float2bfloat16(acc_o[t][3]*il1);
        }
    }
}

void sparse_mla_prefill_launch(
    torch::Tensor Q, torch::Tensor KV_cache, torch::Tensor indices,
    torch::Tensor output, float sm_scale,
    int num_heads, int num_tokens, int topk, int BI_param)
{
    const int REPLICATE_H = num_heads / HPB;
    size_t smem_bytes = HPB*Q_NOPE_STRIDE + HPB*NUM_SCALES*4 + HPB*D_ROPE*2
                      + 2*BI*KV_SMEM_STRIDE
                      + N_MATH_WARPS*HPB*4 + HPB*4 + HPB*4
                      + HPB*W_FP8_STRIDE + HPB*4 + V_CHUNK*V_TRANS_STRIDE;

    dim3 grid(num_tokens * REPLICATE_H);
    dim3 block(BLOCK_THREADS);

    if (smem_bytes > 48*1024)
        cudaFuncSetAttribute(sparse_mla_prefill_mma_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    sparse_mla_prefill_mma_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const bf16*>(Q.data_ptr()),
        reinterpret_cast<const uint8_t*>(KV_cache.data_ptr()),
        indices.data_ptr<int32_t>(),
        reinterpret_cast<bf16*>(output.data_ptr()),
        sm_scale, num_heads, num_tokens, topk);
}
