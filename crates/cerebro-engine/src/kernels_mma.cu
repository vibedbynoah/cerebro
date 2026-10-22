// Tensor-core kernels (Ampere sm_80 and newer). Compiled to a separate PTX module by build.rs and
// loaded only when the GPU has compute capability 8.0+; older GPUs keep the dp4a kernels.

// Y[t][row] (+)= W[row] . X[t] for T tokens on int8 tensor cores. W and X are int8 with one f32 scale
// per 32 values, so one m16n8k32 MMA covers exactly one quant block: its int32 result is scaled by
// the row's and the token's block scales and added to f32 accumulators.
// Block tile: 64 rows x 64 tokens, 128 k per shared-memory stage (4 quant blocks), double-buffered
// with cp.async. 4 warps in a 2x2 grid, each computing 32 rows x 32 tokens (2 x 4 MMA tiles).
// Requires cols % 32 == 0; rows and T are arbitrary.

#include <cuda_fp16.h>

#define BM 64
#define BN 64
#define BK 128
#define LDS (BK + 16) // padded row stride: the 8 rows a fragment load touches fall in distinct banks

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int n = pred ? 16 : 0; // src-size 0 zero-fills the 16 bytes
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem, bool pred) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int n = pred ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// Exact int32 -> float for |v| < 2^22 without the slow I2F unit (16 per clock per SM on Ampere):
// put v in the mantissa of 1.5 * 2^23 and subtract that. A 32-deep int8 product is at most
// 32 * 127 * 127 = 516,128, well inside the range.
__device__ __forceinline__ float i2f_small(int v) {
    return __int_as_float(v + 0x4B400000) - 12582912.0f;
}

struct Stage {
    signed char a[BM * LDS];
    signed char b[BN * LDS];
    float wd[BK / 32][BM];
    float xd[BK / 32][BN];
};

extern "C" __global__ void __launch_bounds__(128) matmul_q8_mma(
    const signed char* __restrict__ qs, const float* __restrict__ ds, const signed char* __restrict__ xq,
    const float* __restrict__ xd, float* __restrict__ y, int rows, int cols, int T, int add) {
    __shared__ __align__(16) Stage st[2];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, tg = lane & 3;
    // Token tiles vary fastest (gridDim.x): the blocks sharing one weight tile run together, so it
    // is read from memory once and from L2 for the other token tiles.
    const int r0 = blockIdx.y * BM, t0 = blockIdx.x * BN;
    const int wm = (warp & 1) * 32, wn = (warp >> 1) * 32;
    const int nb = cols >> 5;
    const int stages = (cols + BK - 1) / BK;

    auto load = [&](int s, int k0) {
        Stage& S = st[s];
        // 64 rows x 128 bytes of W and of X: 512 16-byte chunks each, 4 per thread.
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int c = tid + i * 128, r = c >> 3, k = k0 + (c & 7) * 16;
            bool okw = (r0 + r < rows) && (k < cols);
            cp_async16(&S.a[r * LDS + (c & 7) * 16], qs + (size_t)(okw ? r0 + r : 0) * cols + (okw ? k : 0), okw);
            bool okx = (t0 + r < T) && (k < cols);
            cp_async16(&S.b[r * LDS + (c & 7) * 16], xq + (size_t)(okx ? t0 + r : 0) * cols + (okx ? k : 0), okx);
        }
        // Scales: 4 blocks x 64 rows (and tokens), 2 of each per thread.
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = tid + i * 128, kb = c >> 6, r = c & 63, b = (k0 >> 5) + kb;
            bool okw = (r0 + r < rows) && (b < nb);
            cp_async4(&S.wd[kb][r], ds + (okw ? (size_t)(r0 + r) * nb + b : 0), okw);
            bool okx = (t0 + r < T) && (b < nb);
            cp_async4(&S.xd[kb][r], xd + (okx ? (size_t)(t0 + r) * nb + b : 0), okx);
        }
        cp_commit();
    };

    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    load(0, 0);
    for (int s = 0; s < stages; ++s) {
        if (s + 1 < stages) { load((s + 1) & 1, (s + 1) * BK); cp_wait<1>(); }
        else cp_wait<0>();
        __syncthreads();
        const Stage& S = st[s & 1];
        #pragma unroll
        for (int kb = 0; kb < BK / 32; ++kb) {
            const int ko = kb * 32 + tg * 4;
            unsigned a[2][4], b[4][2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const signed char* p = &S.a[(wm + i * 16 + g) * LDS + ko];
                a[i][0] = *(const unsigned*)p;
                a[i][1] = *(const unsigned*)(p + 8 * LDS);
                a[i][2] = *(const unsigned*)(p + 16);
                a[i][3] = *(const unsigned*)(p + 8 * LDS + 16);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const signed char* p = &S.b[(wn + j * 8 + g) * LDS + ko];
                b[j][0] = *(const unsigned*)p;
                b[j][1] = *(const unsigned*)(p + 16);
            }
            float sw[2][2], sx[4][2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) { sw[i][0] = S.wd[kb][wm + i * 16 + g]; sw[i][1] = S.wd[kb][wm + i * 16 + g + 8]; }
            #pragma unroll
            for (int j = 0; j < 4; ++j) { sx[j][0] = S.xd[kb][wn + j * 8 + tg * 2]; sx[j][1] = S.xd[kb][wn + j * 8 + tg * 2 + 1]; }
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int c0, c1, c2, c3;
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                        "{%10,%10,%10,%10};\n"
                        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                        : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]), "r"(b[j][0]), "r"(b[j][1]), "r"(0));
                    acc[i][j][0] = fmaf(i2f_small(c0), sw[i][0] * sx[j][0], acc[i][j][0]);
                    acc[i][j][1] = fmaf(i2f_small(c1), sw[i][0] * sx[j][1], acc[i][j][1]);
                    acc[i][j][2] = fmaf(i2f_small(c2), sw[i][1] * sx[j][0], acc[i][j][2]);
                    acc[i][j][3] = fmaf(i2f_small(c3), sw[i][1] * sx[j][1], acc[i][j][3]);
                }
        }
        __syncthreads(); // the next iteration's load overwrites this stage
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int r = r0 + wm + i * 16 + g + (e >> 1) * 8, t = t0 + wn + j * 8 + tg * 2 + (e & 1);
                if (r < rows && t < T) {
                    float* o = y + (size_t)t * rows + r;
                    *o = add ? *o + acc[i][j][e] : acc[i][j][e];
                }
            }
}

// Causal flash attention for prompt processing on f16 tensor cores (head_dim 128). A block takes one
// query head and 64 tokens (16 per warp). Per tile of 64 cached keys: S = Q K^T with m16n8k16 MMAs,
// online softmax in registers (exp2 domain, scale folded into Q), then O += P V with P reused straight
// from the S accumulators as the A operand and V fragments read with ldmatrix.trans.
// Q rows come from the fused QKV buffer (f32, RoPE already applied); K/V from the f16 caches.
#define FHD 128
#define FKT 64
#define FLD (FHD + 8) // padded smem row (272 bytes): conflict-free fragment loads and ldmatrix

__device__ __forceinline__ unsigned pack_h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<unsigned*>(&h);
}

__device__ __forceinline__ void mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

extern "C" __global__ void __launch_bounds__(128) attn_prefill_mma(
    const float* __restrict__ qkv, const __half* __restrict__ kc, const __half* __restrict__ vc,
    float* __restrict__ out, int n_head, int n_kv, const int* __restrict__ posp, int group, float scale, int T) {
    __shared__ __align__(16) __half ks[FKT * FLD];
    __shared__ __align__(16) __half vs[FKT * FLD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tg = lane & 3;
    const int h = blockIdx.x, kvh = h / group, t0 = blockIdx.y * 64;
    const int pos0 = *posp;
    const int row = n_head * FHD + 2 * n_kv * FHD, kvd = n_kv * FHD;
    const int qa = t0 + warp * 16 + g, qb = qa + 8; // this thread's two query rows
    const float qsc = scale * 1.4426950408889634f;

    // Q fragments for 8 k-steps of 16 dims.
    unsigned qf[8][4];
    {
        const float* ra = qkv + (size_t)min(qa, T - 1) * row + h * FHD;
        const float* rb = qkv + (size_t)min(qb, T - 1) * row + h * FHD;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int d = k * 16 + tg * 2;
            float2 a0 = *reinterpret_cast<const float2*>(ra + d), a2 = *reinterpret_cast<const float2*>(ra + d + 8);
            float2 b0 = *reinterpret_cast<const float2*>(rb + d), b2 = *reinterpret_cast<const float2*>(rb + d + 8);
            qf[k][0] = pack_h2(a0.x * qsc, a0.y * qsc);
            qf[k][1] = pack_h2(b0.x * qsc, b0.y * qsc);
            qf[k][2] = pack_h2(a2.x * qsc, a2.y * qsc);
            qf[k][3] = pack_h2(b2.x * qsc, b2.y * qsc);
        }
    }
    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float ma = -1e30f, mb = -1e30f, la = 0.f, lb = 0.f;
    const int lim_a = pos0 + qa, lim_b = pos0 + qb; // last key each row may see
    const int nkeys = pos0 + min(T, t0 + 64);
    const int warp_last = pos0 + t0 + warp * 16 + 15;

    for (int k0 = 0; k0 < nkeys; k0 += FKT) {
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int c = tid + i * 128, r = c >> 4, d = (c & 15) * 8, p = k0 + r;
            bool ok = p < nkeys;
            size_t src = ok ? (size_t)p * kvd + kvh * FHD + d : 0;
            cp_async16(&ks[r * FLD + d], kc + src, ok);
            cp_async16(&vs[r * FLD + d], vc + src, ok);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        if (k0 > warp_last) continue; // every key in this tile is in this warp's future

        float s[8][4];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
            const __half* kr = &ks[(j * 8 + g) * FLD + tg * 2];
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                mma16816(s[j], qf[k], *reinterpret_cast<const unsigned*>(kr + k * 16),
                         *reinterpret_cast<const unsigned*>(kr + k * 16 + 8));
        }
        // Causal mask and row maxima (each row's 64 scores live in the 4 threads of a quad).
        float xa = -1e30f, xb = -1e30f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int p = k0 + j * 8 + tg * 2;
            if (p > lim_a) s[j][0] = -INFINITY;
            if (p + 1 > lim_a) s[j][1] = -INFINITY;
            if (p > lim_b) s[j][2] = -INFINITY;
            if (p + 1 > lim_b) s[j][3] = -INFINITY;
            xa = fmaxf(xa, fmaxf(s[j][0], s[j][1]));
            xb = fmaxf(xb, fmaxf(s[j][2], s[j][3]));
        }
        xa = fmaxf(xa, __shfl_xor_sync(0xffffffff, xa, 1)); xa = fmaxf(xa, __shfl_xor_sync(0xffffffff, xa, 2));
        xb = fmaxf(xb, __shfl_xor_sync(0xffffffff, xb, 1)); xb = fmaxf(xb, __shfl_xor_sync(0xffffffff, xb, 2));
        float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
        float ca = exp2f(ma - na), cb = exp2f(mb - nb);
        ma = na; mb = nb;
        float sa = 0.f, sb = 0.f;
        unsigned pf[4][4]; // P as A fragments for 4 k-steps of 16 keys
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float p0 = exp2f(s[j][0] - na), p1 = exp2f(s[j][1] - na), p2 = exp2f(s[j][2] - nb), p3 = exp2f(s[j][3] - nb);
            sa += p0 + p1; sb += p2 + p3;
            pf[j >> 1][(j & 1) * 2] = pack_h2(p0, p1);
            pf[j >> 1][(j & 1) * 2 + 1] = pack_h2(p2, p3);
        }
        la = la * ca + sa; lb = lb * cb + sb; // per-thread partial sums; reduced across the quad at the end
        #pragma unroll
        for (int n = 0; n < 16; ++n) { o[n][0] *= ca; o[n][1] *= ca; o[n][2] *= cb; o[n][3] *= cb; }
        // O += P V. ldmatrix.x4.trans: lanes 0-15 address keys kk*16 + (lane & 15) at dims n*8, lanes
        // 16-31 the same keys at dims n*8 + 8: B fragments for two n-tiles per instruction.
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            #pragma unroll
            for (int n = 0; n < 16; n += 2) {
                const __half* vp = &vs[(kk * 16 + (lane & 15)) * FLD + n * 8 + (lane >> 4) * 8];
                unsigned sp = (unsigned)__cvta_generic_to_shared(vp), b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(sp));
                mma16816(o[n], pf[kk], b0, b1);
                mma16816(o[n + 1], pf[kk], b2, b3);
            }
        }
    }
    la += __shfl_xor_sync(0xffffffff, la, 1); la += __shfl_xor_sync(0xffffffff, la, 2);
    lb += __shfl_xor_sync(0xffffffff, lb, 1); lb += __shfl_xor_sync(0xffffffff, lb, 2);
    float ia = la > 0.f ? 1.f / la : 0.f, ib = lb > 0.f ? 1.f / lb : 0.f;
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        int d = n * 8 + tg * 2;
        if (qa < T) *reinterpret_cast<float2*>(out + (size_t)qa * n_head * FHD + h * FHD + d) = make_float2(o[n][0] * ia, o[n][1] * ia);
        if (qb < T) *reinterpret_cast<float2*>(out + (size_t)qb * n_head * FHD + h * FHD + d) = make_float2(o[n][2] * ib, o[n][3] * ib);
    }
}

// The same flash attention for head size 256 (Qwen3.8). Holding a 256-wide Q and O per thread would
// need ~250 registers, so each (head, 64 tokens) runs as two blocks (blockIdx.z), each producing 128
// of the 256 output dimensions: S = Q K^T over all 256 dims is computed by both (1.5x the FLOPs of one
// block, no spills). Key tiles are 32 wide so K (256 dims) and V (128 dims) fit 48 KB of shared memory.
#define GHD 256
#define GDV 128
#define GKT 32
#define GLK (GHD + 8)
#define GLV (GDV + 8)

extern "C" __global__ void __launch_bounds__(128) attn_prefill_mma256(
    const float* __restrict__ qkv, const __half* __restrict__ kc, const __half* __restrict__ vc,
    float* __restrict__ out, int n_head, int n_kv, const int* __restrict__ posp, int group, float scale, int T) {
    __shared__ __align__(16) __half ks[GKT * GLK];
    __shared__ __align__(16) __half vs[GKT * GLV];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tg = lane & 3;
    const int h = blockIdx.x, kvh = h / group, t0 = blockIdx.y * 64, dz = blockIdx.z * GDV;
    const int pos0 = *posp;
    const int row = n_head * GHD + 2 * n_kv * GHD, kvd = n_kv * GHD;
    const int qa = t0 + warp * 16 + g, qb = qa + 8;
    const float qsc = scale * 1.4426950408889634f;

    unsigned qf[16][4];
    {
        const float* ra = qkv + (size_t)min(qa, T - 1) * row + h * GHD;
        const float* rb = qkv + (size_t)min(qb, T - 1) * row + h * GHD;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            int d = k * 16 + tg * 2;
            float2 a0 = *reinterpret_cast<const float2*>(ra + d), a2 = *reinterpret_cast<const float2*>(ra + d + 8);
            float2 b0 = *reinterpret_cast<const float2*>(rb + d), b2 = *reinterpret_cast<const float2*>(rb + d + 8);
            qf[k][0] = pack_h2(a0.x * qsc, a0.y * qsc);
            qf[k][1] = pack_h2(b0.x * qsc, b0.y * qsc);
            qf[k][2] = pack_h2(a2.x * qsc, a2.y * qsc);
            qf[k][3] = pack_h2(b2.x * qsc, b2.y * qsc);
        }
    }
    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float ma = -1e30f, mb = -1e30f, la = 0.f, lb = 0.f;
    const int lim_a = pos0 + qa, lim_b = pos0 + qb;
    const int nkeys = pos0 + min(T, t0 + 64);
    const int warp_last = pos0 + t0 + warp * 16 + 15;

    for (int k0 = 0; k0 < nkeys; k0 += GKT) {
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int c = tid + i * 128, r = c >> 5, d = (c & 31) * 8, p = k0 + r;
            bool ok = p < nkeys;
            cp_async16(&ks[r * GLK + d], kc + (ok ? (size_t)p * kvd + kvh * GHD + d : 0), ok);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int c = tid + i * 128, r = c >> 4, d = (c & 15) * 8, p = k0 + r;
            bool ok = p < nkeys;
            cp_async16(&vs[r * GLV + d], vc + (ok ? (size_t)p * kvd + kvh * GHD + dz + d : 0), ok);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        if (k0 > warp_last) continue;

        float s[4][4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
            const __half* kr = &ks[(j * 8 + g) * GLK + tg * 2];
            #pragma unroll
            for (int k = 0; k < 16; ++k)
                mma16816(s[j], qf[k], *reinterpret_cast<const unsigned*>(kr + k * 16),
                         *reinterpret_cast<const unsigned*>(kr + k * 16 + 8));
        }
        float xa = -1e30f, xb = -1e30f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int p = k0 + j * 8 + tg * 2;
            if (p > lim_a) s[j][0] = -INFINITY;
            if (p + 1 > lim_a) s[j][1] = -INFINITY;
            if (p > lim_b) s[j][2] = -INFINITY;
            if (p + 1 > lim_b) s[j][3] = -INFINITY;
            xa = fmaxf(xa, fmaxf(s[j][0], s[j][1]));
            xb = fmaxf(xb, fmaxf(s[j][2], s[j][3]));
        }
        xa = fmaxf(xa, __shfl_xor_sync(0xffffffff, xa, 1)); xa = fmaxf(xa, __shfl_xor_sync(0xffffffff, xa, 2));
        xb = fmaxf(xb, __shfl_xor_sync(0xffffffff, xb, 1)); xb = fmaxf(xb, __shfl_xor_sync(0xffffffff, xb, 2));
        float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
        float ca = exp2f(ma - na), cb = exp2f(mb - nb);
        ma = na; mb = nb;
        float sa = 0.f, sb = 0.f;
        unsigned pf[2][4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float p0 = exp2f(s[j][0] - na), p1 = exp2f(s[j][1] - na), p2 = exp2f(s[j][2] - nb), p3 = exp2f(s[j][3] - nb);
            sa += p0 + p1; sb += p2 + p3;
            pf[j >> 1][(j & 1) * 2] = pack_h2(p0, p1);
            pf[j >> 1][(j & 1) * 2 + 1] = pack_h2(p2, p3);
        }
        la = la * ca + sa; lb = lb * cb + sb;
        #pragma unroll
        for (int n = 0; n < 16; ++n) { o[n][0] *= ca; o[n][1] *= ca; o[n][2] *= cb; o[n][3] *= cb; }
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            #pragma unroll
            for (int n = 0; n < 16; n += 2) {
                const __half* vp = &vs[(kk * 16 + (lane & 15)) * GLV + n * 8 + (lane >> 4) * 8];
                unsigned sp = (unsigned)__cvta_generic_to_shared(vp), b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(sp));
                mma16816(o[n], pf[kk], b0, b1);
                mma16816(o[n + 1], pf[kk], b2, b3);
            }
        }
    }
    la += __shfl_xor_sync(0xffffffff, la, 1); la += __shfl_xor_sync(0xffffffff, la, 2);
    lb += __shfl_xor_sync(0xffffffff, lb, 1); lb += __shfl_xor_sync(0xffffffff, lb, 2);
    float ia = la > 0.f ? 1.f / la : 0.f, ib = lb > 0.f ? 1.f / lb : 0.f;
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        int d = dz + n * 8 + tg * 2;
        if (qa < T) *reinterpret_cast<float2*>(out + (size_t)qa * n_head * GHD + h * GHD + d) = make_float2(o[n][0] * ia, o[n][1] * ia);
        if (qb < T) *reinterpret_cast<float2*>(out + (size_t)qb * n_head * GHD + h * GHD + d) = make_float2(o[n][2] * ib, o[n][3] * ib);
    }
}
