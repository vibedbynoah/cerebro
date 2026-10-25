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

// `ys` is the distance between two tokens' rows in y (rows, or more when this product fills only
// some of the rows of a wider output).
extern "C" __global__ void __launch_bounds__(128) matmul_q8_mma(
    const signed char* __restrict__ qs, const float* __restrict__ ds, const signed char* __restrict__ xq,
    const float* __restrict__ xd, float* __restrict__ y, int rows, int cols, int T, int add, int ys) {
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
                    float* o = y + (size_t)t * ys + r;
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


// attn_prefill_mma256 with a sliding window and the query row stride as parameters (Gemma 4's
// sliding layers: 256-wide heads, a 512-position window, q in its own buffer, scale 1).
extern "C" __global__ void __launch_bounds__(128) attn_prefill_mma256w(
    const float* __restrict__ qkv, const __half* __restrict__ kc, const __half* __restrict__ vc,
    float* __restrict__ out, int n_head, int n_kv, const int* __restrict__ posp, int group, float scale, int T,
    int row, int window) {
    __shared__ __align__(16) __half ks[GKT * GLK];
    __shared__ __align__(16) __half vs[GKT * GLV];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tg = lane & 3;
    const int h = blockIdx.x, kvh = h / group, t0 = blockIdx.y * 64, dz = blockIdx.z * GDV;
    const int pos0 = *posp;
    const int kvd = n_kv * GHD;
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
    // Sliding window: a query at position p sees keys p - window + 1 .. p.
    const int low_a = window > 0 ? lim_a - window + 1 : 0, low_b = window > 0 ? lim_b - window + 1 : 0;
    const int nkeys = pos0 + min(T, t0 + 64);
    const int warp_last = pos0 + t0 + warp * 16 + 15;
    const int kbeg = window > 0 ? max(0, pos0 + t0 + 1 - window) / GKT * GKT : 0;
    const int warp_first = window > 0 ? pos0 + t0 + warp * 16 + 1 - window : 0;

    for (int k0 = kbeg; k0 < nkeys; k0 += GKT) {
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
        if (k0 > warp_last || k0 + GKT <= warp_first) continue;

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
            if (p > lim_a || p < low_a) s[j][0] = -INFINITY;
            if (p + 1 > lim_a || p + 1 < low_a) s[j][1] = -INFINITY;
            if (p > lim_b || p < low_b) s[j][2] = -INFINITY;
            if (p + 1 > lim_b || p + 1 < low_b) s[j][3] = -INFINITY;
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


// The same product with a 128 x 64 block and a 64 x 32 tile per warp: each fragment loaded from
// shared memory feeds twice as many tensor-core operations (24 loads per 16 mma instead of 16 per
// 8), for big matrices and long prompts. 64-deep stages keep two stages in 48 KB of shared memory.
#define BM2 128
#define BN2 64
#define BK2 64
#define LDS2 (BK2 + 16)

struct Stage2 {
    signed char a[BM2 * LDS2];
    signed char b[BN2 * LDS2];
    float wd[BK2 / 32][BM2];
    float xd[BK2 / 32][BN2];
};

extern "C" __global__ void __launch_bounds__(128) matmul_q8_mma_big(
    const signed char* __restrict__ qs, const float* __restrict__ ds, const signed char* __restrict__ xq,
    const float* __restrict__ xd, float* __restrict__ y, int rows, int cols, int T, int add, int ys) {
    __shared__ __align__(16) Stage2 st[2];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, tg = lane & 3;
    const int r0 = blockIdx.y * BM2, t0 = blockIdx.x * BN2;
    const int wm = (warp & 1) * 64, wn = (warp >> 1) * 32;
    const int nb = cols >> 5;
    const int stages = (cols + BK2 - 1) / BK2;

    auto load = [&](int s, int k0) {
        Stage2& S = st[s];
        // W: 128 rows x 64 bytes = 512 16-byte chunks, 4 per thread; X: 64 x 64 = 256, 2 per thread.
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int c = tid + i * 128, r = c >> 2, k = k0 + (c & 3) * 16;
            bool ok = (r0 + r < rows) && (k < cols);
            cp_async16(&S.a[r * LDS2 + (c & 3) * 16], qs + (size_t)(ok ? r0 + r : 0) * cols + (ok ? k : 0), ok);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = tid + i * 128, r = c >> 2, k = k0 + (c & 3) * 16;
            bool ok = (t0 + r < T) && (k < cols);
            cp_async16(&S.b[r * LDS2 + (c & 3) * 16], xq + (size_t)(ok ? t0 + r : 0) * cols + (ok ? k : 0), ok);
        }
        // Scales: 2 blocks x 128 rows (2 per thread) and 2 x 64 tokens (1 per thread).
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = tid + i * 128, kb = c >> 7, r = c & 127, b = (k0 >> 5) + kb;
            bool ok = (r0 + r < rows) && (b < nb);
            cp_async4(&S.wd[kb][r], ds + (ok ? (size_t)(r0 + r) * nb + b : 0), ok);
        }
        {
            int kb = tid >> 6, r = tid & 63, b = (k0 >> 5) + kb;
            bool ok = (t0 + r < T) && (b < nb);
            cp_async4(&S.xd[kb][r], xd + (ok ? (size_t)(t0 + r) * nb + b : 0), ok);
        }
        cp_commit();
    };

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    load(0, 0);
    for (int s = 0; s < stages; ++s) {
        if (s + 1 < stages) { load((s + 1) & 1, (s + 1) * BK2); cp_wait<1>(); }
        else cp_wait<0>();
        __syncthreads();
        const Stage2& S = st[s & 1];
        #pragma unroll
        for (int kb = 0; kb < BK2 / 32; ++kb) {
            const int ko = kb * 32 + tg * 4;
            unsigned a[4][4], b[4][2];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const signed char* p = &S.a[(wm + i * 16 + g) * LDS2 + ko];
                a[i][0] = *(const unsigned*)p;
                a[i][1] = *(const unsigned*)(p + 8 * LDS2);
                a[i][2] = *(const unsigned*)(p + 16);
                a[i][3] = *(const unsigned*)(p + 8 * LDS2 + 16);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const signed char* p = &S.b[(wn + j * 8 + g) * LDS2 + ko];
                b[j][0] = *(const unsigned*)p;
                b[j][1] = *(const unsigned*)(p + 16);
            }
            float sw[4][2], sx[4][2];
            #pragma unroll
            for (int i = 0; i < 4; ++i) { sw[i][0] = S.wd[kb][wm + i * 16 + g]; sw[i][1] = S.wd[kb][wm + i * 16 + g + 8]; }
            #pragma unroll
            for (int j = 0; j < 4; ++j) { sx[j][0] = S.xd[kb][wn + j * 8 + tg * 2]; sx[j][1] = S.xd[kb][wn + j * 8 + tg * 2 + 1]; }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
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
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int r = r0 + wm + i * 16 + g + (e >> 1) * 8, t = t0 + wn + j * 8 + tg * 2 + (e & 1);
                if (r < rows && t < T) {
                    float* o = y + (size_t)t * ys + r;
                    *o = add ? *o + acc[i][j][e] : acc[i][j][e];
                }
            }
}

// The same product with a 128 x 128 block of 8 warps (2 x 4, each 64 rows x 32 tokens): every
// weight tile staged in shared memory serves 128 tokens instead of 64, halving the weight traffic
// per output, and twice the warps per block hide the loads. For long prompts on big matrices.
#define BN3 128
struct Stage3 {
    signed char a[BM2 * LDS2];
    signed char b[BN3 * LDS2];
    float wd[BK2 / 32][BM2];
    float xd[BK2 / 32][BN3];
};

extern "C" __global__ void __launch_bounds__(256) matmul_q8_mma_x(
    const signed char* __restrict__ qs, const float* __restrict__ ds, const signed char* __restrict__ xq,
    const float* __restrict__ xd, float* __restrict__ y, int rows, int cols, int T, int add, int ys) {
    __shared__ __align__(16) Stage3 st[2];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, tg = lane & 3;
    const int r0 = blockIdx.y * BM2, t0 = blockIdx.x * BN3;
    const int wm = (warp & 1) * 64, wn = (warp >> 1) * 32;
    const int nb = cols >> 5;
    const int stages = (cols + BK2 - 1) / BK2;

    auto load = [&](int s, int k0) {
        Stage3& S = st[s];
        // W: 128 rows x 64 bytes = 512 16-byte chunks, 2 per thread; X: 128 x 64, 2 per thread.
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = tid + i * 256, r = c >> 2, k = k0 + (c & 3) * 16;
            bool ok = (r0 + r < rows) && (k < cols);
            cp_async16(&S.a[r * LDS2 + (c & 3) * 16], qs + (size_t)(ok ? r0 + r : 0) * cols + (ok ? k : 0), ok);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = tid + i * 256, r = c >> 2, k = k0 + (c & 3) * 16;
            bool ok = (t0 + r < T) && (k < cols);
            cp_async16(&S.b[r * LDS2 + (c & 3) * 16], xq + (size_t)(ok ? t0 + r : 0) * cols + (ok ? k : 0), ok);
        }
        // Scales: 2 blocks x 128 rows and 2 x 128 tokens, one of each per thread.
        {
            int kb = tid >> 7, r = tid & 127, b = (k0 >> 5) + kb;
            bool ok = (r0 + r < rows) && (b < nb);
            cp_async4(&S.wd[kb][r], ds + (ok ? (size_t)(r0 + r) * nb + b : 0), ok);
            bool okx = (t0 + r < T) && (b < nb);
            cp_async4(&S.xd[kb][r], xd + (okx ? (size_t)(t0 + r) * nb + b : 0), okx);
        }
        cp_commit();
    };

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    load(0, 0);
    for (int s = 0; s < stages; ++s) {
        if (s + 1 < stages) { load((s + 1) & 1, (s + 1) * BK2); cp_wait<1>(); }
        else cp_wait<0>();
        __syncthreads();
        const Stage3& S = st[s & 1];
        #pragma unroll
        for (int kb = 0; kb < BK2 / 32; ++kb) {
            const int ko = kb * 32 + tg * 4;
            // ldmatrix: one instruction per A fragment (four 8x16-byte matrices: rows 0-7 / 8-15 x
            // bytes 0-15 / 16-31, which is the m16n8k32 A layout) and per two B fragments.
            unsigned a[4][4], b[4][2];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const signed char* p = &S.a[(wm + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS2 + kb * 32 + (lane >> 4) * 16];
                unsigned sp = (unsigned)__cvta_generic_to_shared(p);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3]) : "r"(sp));
            }
            #pragma unroll
            for (int j = 0; j < 4; j += 2) {
                const signed char* p = &S.b[(wn + j * 8 + (lane & 7) + (lane >> 4) * 8) * LDS2 + kb * 32 + ((lane >> 3) & 1) * 16];
                unsigned sp = (unsigned)__cvta_generic_to_shared(p);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[j][0]), "=r"(b[j][1]), "=r"(b[j + 1][0]), "=r"(b[j + 1][1]) : "r"(sp));
            }
            float sw[4][2], sx[4][2];
            #pragma unroll
            for (int i = 0; i < 4; ++i) { sw[i][0] = S.wd[kb][wm + i * 16 + g]; sw[i][1] = S.wd[kb][wm + i * 16 + g + 8]; }
            #pragma unroll
            for (int j = 0; j < 4; ++j) { sx[j][0] = S.xd[kb][wn + j * 8 + tg * 2]; sx[j][1] = S.xd[kb][wn + j * 8 + tg * 2 + 1]; }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
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
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int r = r0 + wm + i * 16 + g + (e >> 1) * 8, t = t0 + wn + j * 8 + tg * 2 + (e & 1);
                if (r < rows && t < T) {
                    float* o = y + (size_t)t * ys + r;
                    *o = add ? *o + acc[i][j][e] : acc[i][j][e];
                }
            }
}

// ---- Q4_K straight from the file's blocks (no int8 copy) ------------------------------------------
// Y[t][row] (+)= W[row] . X[t] with W in Q4_K (144-byte blocks of 256: d, dmin, 12 scale bytes,
// 128 bytes of nibbles). Each 64-value stage of a row is one 32-byte chunk of its block: byte k
// holds value k of sub-block 2j (low nibble) and of sub-block 2j+1 (high nibble), so a fragment's
// four consecutive weights are one 32-bit load and a mask: the weights go from the file's bytes to
// the tensor cores with no conversion pass and no int8 copy in memory. Per 32 values:
//   sum_k w_k x_k = d*sc * sum_k q_k x_k - dmin*m * sum_k x_k,
// the first from the MMA (q in 0..15), the second from the activations' block sums.
// Block tile 128 rows x 64 tokens, 4 warps (each 64 x 32), stages double-buffered with cp.async.
#define QK_BM 128
#define QK_BN 64
#define QK_LDA 48 // 32 bytes + 16 padding: a fragment's 8 rows hit distinct banks
#define QK_LDB 80

struct StageQ4 {
    unsigned char a[QK_BM * QK_LDA];
    signed char b[QK_BN * QK_LDB];
    float wsc[2][QK_BM];
    float wmn[2][QK_BM];
    float xd[2][QK_BN];
    float xs[2][QK_BN];
};

__device__ __forceinline__ void q4k_scale_min(int j, const unsigned char* q, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

extern "C" __global__ void __launch_bounds__(128) mmq_q4k(
    const unsigned char* __restrict__ w, const signed char* __restrict__ xq, const float* __restrict__ xd,
    float* __restrict__ y, int rows, int cols, int T, int add, int ys) {
    __shared__ __align__(16) StageQ4 st[2];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, tg = lane & 3;
    const int r0 = blockIdx.y * QK_BM, t0 = blockIdx.x * QK_BN;
    const int wm = (warp & 1) * 64, wn = (warp >> 1) * 32;
    const int nb = cols >> 5;
    const size_t rb = (size_t)(cols >> 8) * 144;
    const int stages = cols >> 6;

    auto load = [&](int s, int k0) {
        StageQ4& S = st[s];
        const int sb = k0 >> 8, j = (k0 >> 6) & 3;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = tid + i * 128, r = c >> 1, h = c & 1;
            bool ok = r0 + r < rows;
            cp_async16(&S.a[r * QK_LDA + 16 * h], w + (ok ? (size_t)(r0 + r) * rb + (size_t)sb * 144 + 16 + 32 * j + 16 * h : 0), ok);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = tid + i * 128, r = c >> 2, k = k0 + (c & 3) * 16;
            bool ok = t0 + r < T;
            cp_async16(&S.b[r * QK_LDB + (c & 3) * 16], xq + (ok ? (size_t)(t0 + r) * cols + k : 0), ok);
        }
        {
            int kb = tid >> 6, r = tid & 63, b = (k0 >> 5) + kb;
            bool ok = t0 + r < T;
            cp_async4(&S.xd[kb][r], xd + (ok ? (size_t)(t0 + r) * nb + b : 0), ok);
        }
        cp_commit();
        // The two sub-blocks' scales and mins of this thread's row, from the block header.
        {
            int r = tid;
            float s0 = 0.f, s1 = 0.f, m0f = 0.f, m1f = 0.f;
            if (r0 + r < rows) {
                const unsigned char* hb = w + (size_t)(r0 + r) * rb + (size_t)sb * 144;
                float d = __half2float(*reinterpret_cast<const __half*>(hb));
                float dm = __half2float(*reinterpret_cast<const __half*>(hb + 2));
                int sc, m;
                q4k_scale_min(2 * j, hb + 4, sc, m);
                s0 = d * sc;
                m0f = dm * m;
                q4k_scale_min(2 * j + 1, hb + 4, sc, m);
                s1 = d * sc;
                m1f = dm * m;
            }
            S.wsc[0][r] = s0;
            S.wsc[1][r] = s1;
            S.wmn[0][r] = m0f;
            S.wmn[1][r] = m1f;
        }
    };

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][jj][e] = 0.f;

    load(0, 0);
    for (int s = 0; s < stages; ++s) {
        if (s + 1 < stages) {
            load((s + 1) & 1, (s + 1) * 64);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        StageQ4& S = st[s & 1];
        // Each token's sum of its two 32-value blocks, times its scale (the min term).
        {
            int kb = tid >> 6, r = tid & 63;
            const int* xb = reinterpret_cast<const int*>(&S.b[r * QK_LDB + kb * 32]);
            int sum = 0;
            #pragma unroll
            for (int q = 0; q < 8; ++q) sum = __dp4a(xb[q], 0x01010101, sum);
            S.xs[kb][r] = S.xd[kb][r] * (float)sum;
        }
        __syncthreads();
        unsigned wa[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned char* p = &S.a[(wm + i * 16 + g) * QK_LDA + tg * 4];
            wa[i][0] = *(const unsigned*)p;
            wa[i][1] = *(const unsigned*)(p + 8 * QK_LDA);
            wa[i][2] = *(const unsigned*)(p + 16);
            wa[i][3] = *(const unsigned*)(p + 8 * QK_LDA + 16);
        }
        #pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            const int ko = kb * 32 + tg * 4;
            unsigned b[4][2];
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const signed char* p = &S.b[(wn + jj * 8 + g) * QK_LDB + ko];
                b[jj][0] = *(const unsigned*)p;
                b[jj][1] = *(const unsigned*)(p + 16);
            }
            float sw[4][2], mw[4][2], sx[4][2], xsv[4][2];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int rr = wm + i * 16 + g;
                sw[i][0] = S.wsc[kb][rr];
                sw[i][1] = S.wsc[kb][rr + 8];
                mw[i][0] = S.wmn[kb][rr];
                mw[i][1] = S.wmn[kb][rr + 8];
            }
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                int tt = wn + jj * 8 + tg * 2;
                sx[jj][0] = S.xd[kb][tt];
                sx[jj][1] = S.xd[kb][tt + 1];
                xsv[jj][0] = S.xs[kb][tt];
                xsv[jj][1] = S.xs[kb][tt + 1];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                unsigned a0 = kb ? (wa[i][0] >> 4) & 0x0F0F0F0F : wa[i][0] & 0x0F0F0F0F;
                unsigned a1 = kb ? (wa[i][1] >> 4) & 0x0F0F0F0F : wa[i][1] & 0x0F0F0F0F;
                unsigned a2 = kb ? (wa[i][2] >> 4) & 0x0F0F0F0F : wa[i][2] & 0x0F0F0F0F;
                unsigned a3 = kb ? (wa[i][3] >> 4) & 0x0F0F0F0F : wa[i][3] & 0x0F0F0F0F;
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    int c0, c1, c2, c3;
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                        "{%10,%10,%10,%10};\n"
                        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b[jj][0]), "r"(b[jj][1]), "r"(0));
                    acc[i][jj][0] += i2f_small(c0) * sw[i][0] * sx[jj][0] - mw[i][0] * xsv[jj][0];
                    acc[i][jj][1] += i2f_small(c1) * sw[i][0] * sx[jj][1] - mw[i][0] * xsv[jj][1];
                    acc[i][jj][2] += i2f_small(c2) * sw[i][1] * sx[jj][0] - mw[i][1] * xsv[jj][0];
                    acc[i][jj][3] += i2f_small(c3) * sw[i][1] * sx[jj][1] - mw[i][1] * xsv[jj][1];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int r = r0 + wm + i * 16 + g + (e >> 1) * 8, t = t0 + wn + jj * 8 + tg * 2 + (e & 1);
                if (r < rows && t < T) {
                    float* o = y + (size_t)t * ys + r;
                    *o = add ? *o + acc[i][jj][e] : acc[i][jj][e];
                }
            }
}

// mmq_q4k with a 128 x 128 block of 8 warps (2 x 4, each 64 rows x 32 tokens): every Q4_K chunk
// staged in shared memory serves 128 tokens. The rows' block headers arrive with cp.async like the
// weights; after each stage lands, threads 0..127 decode their row's two scales and mins while all
// 256 threads sum the tokens' 32-value blocks, behind one barrier.
#define QX_BN 128

struct StageQ4X {
    unsigned char a[QK_BM * QK_LDA];
    signed char b[QX_BN * QK_LDB];
    unsigned char hdr[QK_BM * 16];
    float wsc[2][QK_BM];
    float wmn[2][QK_BM];
    float xd[2][QX_BN];
    float xs[2][QX_BN];
};

// Two blocks per SM (<= 128 registers): with one, half the warps were missing to hide latency.
extern "C" __global__ void __launch_bounds__(256, 2) mmq_q4k_x(
    const unsigned char* __restrict__ w, const signed char* __restrict__ xq, const float* __restrict__ xd,
    float* __restrict__ y, int rows, int cols, int T, int add, int ys) {
    __shared__ __align__(16) StageQ4X st[2];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, tg = lane & 3;
    const int r0 = blockIdx.y * QK_BM, t0 = blockIdx.x * QX_BN;
    const int wm = (warp & 1) * 64, wn = (warp >> 1) * 32;
    const int nb = cols >> 5;
    const size_t rb = (size_t)(cols >> 8) * 144;
    const int stages = cols >> 6;

    auto load = [&](int s, int k0) {
        StageQ4X& S = st[s];
        const int sb = k0 >> 8, j = (k0 >> 6) & 3;
        // Weights: 128 rows x 32 bytes = 256 chunks; headers: 128 x 16 bytes; one or two per thread.
        {
            int r = tid >> 1, h = tid & 1;
            bool ok = r0 + r < rows;
            const unsigned char* base = w + (ok ? (size_t)(r0 + r) * rb + (size_t)sb * 144 : 0);
            cp_async16(&S.a[r * QK_LDA + 16 * h], base + (ok ? 16 + 32 * j + 16 * h : 0), ok);
            if (h == 0) cp_async16(&S.hdr[r * 16], base, ok);
        }
        // Activations: 128 tokens x 64 bytes = 512 chunks, two per thread; scales 2 x 128.
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = tid + i * 256, r = c >> 2, k = k0 + (c & 3) * 16;
            bool ok = t0 + r < T;
            cp_async16(&S.b[r * QK_LDB + (c & 3) * 16], xq + (ok ? (size_t)(t0 + r) * cols + k : 0), ok);
        }
        {
            int kb = tid >> 7, r = tid & 127, b = (k0 >> 5) + kb;
            bool ok = t0 + r < T;
            cp_async4(&S.xd[kb][r], xd + (ok ? (size_t)(t0 + r) * nb + b : 0), ok);
        }
        cp_commit();
    };

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][jj][e] = 0.f;

    load(0, 0);
    for (int s = 0; s < stages; ++s) {
        if (s + 1 < stages) {
            load((s + 1) & 1, (s + 1) * 64);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        StageQ4X& S = st[s & 1];
        {
            int kb = tid >> 7, r = tid & 127;
            const int* xb = reinterpret_cast<const int*>(&S.b[r * QK_LDB + kb * 32]);
            int sum = 0;
            #pragma unroll
            for (int q = 0; q < 8; ++q) sum = __dp4a(xb[q], 0x01010101, sum);
            S.xs[kb][r] = S.xd[kb][r] * (float)sum;
            if (tid < QK_BM) {
                const unsigned char* hb = &S.hdr[tid * 16];
                const int j = (s & 3);
                float d = __half2float(*reinterpret_cast<const __half*>(hb));
                float dm = __half2float(*reinterpret_cast<const __half*>(hb + 2));
                int sc, m;
                q4k_scale_min(2 * j, hb + 4, sc, m);
                S.wsc[0][tid] = d * sc;
                S.wmn[0][tid] = dm * m;
                q4k_scale_min(2 * j + 1, hb + 4, sc, m);
                S.wsc[1][tid] = d * sc;
                S.wmn[1][tid] = dm * m;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            const int ko = kb * 32 + tg * 4;
            unsigned b[4][2];
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const signed char* p = &S.b[(wn + jj * 8 + g) * QK_LDB + ko];
                b[jj][0] = *(const unsigned*)p;
                b[jj][1] = *(const unsigned*)(p + 16);
            }
            float sx[4][2], xsv[4][2];
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                int tt = wn + jj * 8 + tg * 2;
                sx[jj][0] = S.xd[kb][tt];
                sx[jj][1] = S.xd[kb][tt + 1];
                xsv[jj][0] = S.xs[kb][tt];
                xsv[jj][1] = S.xs[kb][tt + 1];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned char* p = &S.a[(wm + i * 16 + g) * QK_LDA + tg * 4];
                const int sh = kb * 4;
                unsigned a0 = (*(const unsigned*)p >> sh) & 0x0F0F0F0F;
                unsigned a1 = (*(const unsigned*)(p + 8 * QK_LDA) >> sh) & 0x0F0F0F0F;
                unsigned a2 = (*(const unsigned*)(p + 16) >> sh) & 0x0F0F0F0F;
                unsigned a3 = (*(const unsigned*)(p + 8 * QK_LDA + 16) >> sh) & 0x0F0F0F0F;
                const int rr = wm + i * 16 + g;
                const float sw0 = S.wsc[kb][rr], sw1 = S.wsc[kb][rr + 8], mw0 = S.wmn[kb][rr], mw1 = S.wmn[kb][rr + 8];
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    int c0, c1, c2, c3;
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                        "{%10,%10,%10,%10};\n"
                        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b[jj][0]), "r"(b[jj][1]), "r"(0));
                    acc[i][jj][0] += i2f_small(c0) * sw0 * sx[jj][0] - mw0 * xsv[jj][0];
                    acc[i][jj][1] += i2f_small(c1) * sw0 * sx[jj][1] - mw0 * xsv[jj][1];
                    acc[i][jj][2] += i2f_small(c2) * sw1 * sx[jj][0] - mw1 * xsv[jj][0];
                    acc[i][jj][3] += i2f_small(c3) * sw1 * sx[jj][1] - mw1 * xsv[jj][1];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int r = r0 + wm + i * 16 + g + (e >> 1) * 8, t = t0 + wn + jj * 8 + tg * 2 + (e & 1);
                if (r < rows && t < T) {
                    float* o = y + (size_t)t * ys + r;
                    *o = add ? *o + acc[i][jj][e] : acc[i][jj][e];
                }
            }
}

// Flash attention for 512-wide heads (Gemma 4's global layers), causal, optional sliding window,
// any scale. A 512-wide Q and O per thread would need far more registers than there are, so each
// (head, 64 tokens) runs as four blocks (blockIdx.z), each producing 128 of the 512 output
// dimensions, and Q is re-read from global memory (L2) in two 256-wide halves for every 16-key tile
// instead of being held. K tiles of 16 keys x 512 dims and V tiles of 16 x 128 sit in shared memory.
#define H5D 512
#define H5V 128
#define H5KT 16
#define H5LK (H5D + 8)
#define H5LV (H5V + 8)

extern "C" __global__ void __launch_bounds__(128) attn_prefill_mma512w(
    const float* __restrict__ q, const __half* __restrict__ kc, const __half* __restrict__ vc,
    float* __restrict__ out, int n_head, int n_kv, const int* __restrict__ posp, int group, float scale, int T,
    int row, int window) {
    __shared__ __align__(16) __half ks[H5KT * H5LK];
    __shared__ __align__(16) __half vs[H5KT * H5LV];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tg = lane & 3;
    const int h = blockIdx.x, kvh = h / group, t0 = blockIdx.y * 64, dz = blockIdx.z * H5V;
    const int pos0 = *posp;
    const int kvd = n_kv * H5D;
    const int qa = t0 + warp * 16 + g, qb = qa + 8;
    const float qsc = scale * 1.4426950408889634f;
    const float* ra = q + (size_t)min(qa, T - 1) * row + h * H5D;
    const float* rb = q + (size_t)min(qb, T - 1) * row + h * H5D;

    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float ma = -1e30f, mb = -1e30f, la = 0.f, lb = 0.f;
    const int lim_a = pos0 + qa, lim_b = pos0 + qb;
    const int low_a = window > 0 ? lim_a - window + 1 : 0, low_b = window > 0 ? lim_b - window + 1 : 0;
    const int nkeys = pos0 + min(T, t0 + 64);
    const int warp_last = pos0 + t0 + warp * 16 + 15;
    const int kbeg = window > 0 ? max(0, pos0 + t0 + 1 - window) / H5KT * H5KT : 0;
    const int warp_first = window > 0 ? pos0 + t0 + warp * 16 + 1 - window : 0;

    for (int k0 = kbeg; k0 < nkeys; k0 += H5KT) {
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int c = tid + i * 128, r = c >> 6, d = (c & 63) * 8, p = k0 + r;
            bool ok = p < nkeys;
            cp_async16(&ks[r * H5LK + d], kc + (ok ? (size_t)p * kvd + kvh * H5D + d : 0), ok);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = tid + i * 128, r = c >> 4, d = (c & 15) * 8, p = k0 + r;
            bool ok = p < nkeys;
            cp_async16(&vs[r * H5LV + d], vc + (ok ? (size_t)p * kvd + kvh * H5D + dz + d : 0), ok);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        if (k0 > warp_last || k0 + H5KT <= warp_first) continue;

        float s[2][4];
        #pragma unroll
        for (int j = 0; j < 2; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
        #pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            unsigned qf[16][4];
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                int d = hf * 256 + k * 16 + tg * 2;
                float2 a0 = *reinterpret_cast<const float2*>(ra + d), a2 = *reinterpret_cast<const float2*>(ra + d + 8);
                float2 b0 = *reinterpret_cast<const float2*>(rb + d), b2 = *reinterpret_cast<const float2*>(rb + d + 8);
                qf[k][0] = pack_h2(a0.x * qsc, a0.y * qsc);
                qf[k][1] = pack_h2(b0.x * qsc, b0.y * qsc);
                qf[k][2] = pack_h2(a2.x * qsc, a2.y * qsc);
                qf[k][3] = pack_h2(b2.x * qsc, b2.y * qsc);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const __half* kr = &ks[(j * 8 + g) * H5LK + hf * 256 + tg * 2];
                #pragma unroll
                for (int k = 0; k < 16; ++k)
                    mma16816(s[j], qf[k], *reinterpret_cast<const unsigned*>(kr + k * 16),
                             *reinterpret_cast<const unsigned*>(kr + k * 16 + 8));
            }
        }
        float xa = -1e30f, xb = -1e30f;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int p = k0 + j * 8 + tg * 2;
            if (p > lim_a || p < low_a) s[j][0] = -INFINITY;
            if (p + 1 > lim_a || p + 1 < low_a) s[j][1] = -INFINITY;
            if (p > lim_b || p < low_b) s[j][2] = -INFINITY;
            if (p + 1 > lim_b || p + 1 < low_b) s[j][3] = -INFINITY;
            xa = fmaxf(xa, fmaxf(s[j][0], s[j][1]));
            xb = fmaxf(xb, fmaxf(s[j][2], s[j][3]));
        }
        xa = fmaxf(xa, __shfl_xor_sync(0xffffffff, xa, 1)); xa = fmaxf(xa, __shfl_xor_sync(0xffffffff, xa, 2));
        xb = fmaxf(xb, __shfl_xor_sync(0xffffffff, xb, 1)); xb = fmaxf(xb, __shfl_xor_sync(0xffffffff, xb, 2));
        float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
        float ca = exp2f(ma - na), cb = exp2f(mb - nb);
        ma = na; mb = nb;
        float sa = 0.f, sb = 0.f;
        unsigned pf[4];
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float p0 = exp2f(s[j][0] - na), p1 = exp2f(s[j][1] - na), p2 = exp2f(s[j][2] - nb), p3 = exp2f(s[j][3] - nb);
            sa += p0 + p1; sb += p2 + p3;
            pf[j * 2] = pack_h2(p0, p1);
            pf[j * 2 + 1] = pack_h2(p2, p3);
        }
        la = la * ca + sa; lb = lb * cb + sb;
        #pragma unroll
        for (int n = 0; n < 16; ++n) { o[n][0] *= ca; o[n][1] *= ca; o[n][2] *= cb; o[n][3] *= cb; }
        #pragma unroll
        for (int n = 0; n < 16; n += 2) {
            const __half* vp = &vs[(lane & 15) * H5LV + n * 8 + (lane >> 4) * 8];
            unsigned sp = (unsigned)__cvta_generic_to_shared(vp), b0, b1, b2, b3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(sp));
            mma16816(o[n], pf, b0, b1);
            mma16816(o[n + 1], pf, b2, b3);
        }
    }
    la += __shfl_xor_sync(0xffffffff, la, 1); la += __shfl_xor_sync(0xffffffff, la, 2);
    lb += __shfl_xor_sync(0xffffffff, lb, 1); lb += __shfl_xor_sync(0xffffffff, lb, 2);
    float ia = la > 0.f ? 1.f / la : 0.f, ib = lb > 0.f ? 1.f / lb : 0.f;
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        int d = dz + n * 8 + tg * 2;
        if (qa < T) *reinterpret_cast<float2*>(out + (size_t)qa * n_head * H5D + h * H5D + d) = make_float2(o[n][0] * ia, o[n][1] * ia);
        if (qb < T) *reinterpret_cast<float2*>(out + (size_t)qb * n_head * H5D + h * H5D + d) = make_float2(o[n][2] * ib, o[n][3] * ib);
    }
}

// ---- Tensor-core rate probe ------------------------------------------------------------------------
// Each warp runs `iters` rounds of 8 independent MMAs of one kind (0: f16 in, f32 accumulate; 1: f16
// in, f16 accumulate; 2: int8 in, int32 accumulate; 3: int4 in, int32 accumulate) and writes a value so none is optimised away.
// GeForce Ampere runs kind 0 at half the rate of kind 1 unless the FP16/BF16 issue rate is unlocked.
extern "C" __global__ void __launch_bounds__(256) tensor_probe(int kind, int iters, float* out) {
    const unsigned a0 = 0x3c003c00u ^ threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, b0 = a0 + 5, b1 = a0 + 7;
    float acc = 0.f;
    if (kind == 0) {
        float c[8][4] = {};
        for (int i = 0; i < iters; ++i)
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(c[k][0]), "+f"(c[k][1]), "+f"(c[k][2]), "+f"(c[k][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        #pragma unroll
        for (int k = 0; k < 8; ++k) acc += c[k][0] + c[k][3];
    } else if (kind == 1) {
        unsigned c[8][2] = {};
        for (int i = 0; i < iters; ++i)
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                             : "+r"(c[k][0]), "+r"(c[k][1])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        #pragma unroll
        for (int k = 0; k < 8; ++k) acc += __int_as_float(c[k][0] & 0x3fffffff);
    } else if (kind == 3) {
        int c[8][4] = {};
        for (int i = 0; i < iters; ++i)
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(c[k][0]), "+r"(c[k][1]), "+r"(c[k][2]), "+r"(c[k][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        #pragma unroll
        for (int k = 0; k < 8; ++k) acc += (float)(c[k][0] + c[k][3]);
    } else {
        int c[8][4] = {};
        for (int i = 0; i < iters; ++i)
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(c[k][0]), "+r"(c[k][1]), "+r"(c[k][2]), "+r"(c[k][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        #pragma unroll
        for (int k = 0; k < 8; ++k) acc += (float)(c[k][0] + c[k][3]);
    }
    if (acc == 12345.f) out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
