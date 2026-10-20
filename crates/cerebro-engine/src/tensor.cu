// General tensor kernels for the encoders and generators that are not language models (vision,
// speech, speech synthesis, images): an f16 tensor-core GEMM with f32 accumulation, and the
// normalisations, activations and layout changes around it. Activations are f32 rows of channels
// ([tokens][channels], pixel-major for images); GEMM operands are f16 copies. Ampere (sm_80) and
// newer; compiled to a separate PTX module by build.rs.

#include <cuda_fp16.h>

__device__ __forceinline__ float block_sum(float v, float* red); // below, with the normalisations

// ---- GEMM: Y[t*ldy + r] = alpha * sum_k A[r*lda + k] * B[t*ldb + k] (+ bias[r]) (+ Y) ---------------
// A: rows x K (weights, or K for attention), B: T x K (activations). K % 8 == 0, lda/ldb % 8 == 0.
// Block tile 128 rows x 64 tokens x 32 k, 8 warps in a 4x2 grid (32 rows x 32 tokens each), double
// buffered with cp.async.

#define GBM 128
#define GBN 64
#define GBK 32
#define GLD (GBK + 8) // padded row (80 bytes): conflict-free fragment loads

__device__ __forceinline__ void cpa16(void* smem, const void* gmem, bool pred) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// f16 accumulation (twice the f32-accumulating rate on GeForce cards): c holds two half2.
__device__ __forceinline__ void mma_h16(unsigned* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

extern "C" __global__ void __launch_bounds__(256) gemm_f16(
    const __half* __restrict__ A, int lda, const __half* __restrict__ B, int ldb, float* __restrict__ Y, int ldy,
    int rows, int T, int K, const float* __restrict__ bias, int add, float alpha) {
    __shared__ __align__(16) __half sa[2][GBM * GLD];
    __shared__ __align__(16) __half sb[2][GBN * GLD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, tg = lane & 3;
    const int r0 = blockIdx.x * GBM, t0 = blockIdx.y * GBN;
    const int wm = (warp & 3) * 32, wn = (warp >> 2) * 32;
    const int stages = (K + GBK - 1) / GBK;

    auto load = [&](int s, int k0) {
        // A: 128 rows x 4 chunks of 8 halfs = 512 chunks, 2 per thread. B: 64 x 4 = 256, 1 per thread.
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = tid + i * 256, r = c >> 2, k = k0 + (c & 3) * 8;
            bool ok = (r0 + r < rows) && (k < K);
            cpa16(&sa[s][r * GLD + (c & 3) * 8], A + (size_t)(ok ? r0 + r : 0) * lda + (ok ? k : 0), ok);
        }
        {
            int c = tid, r = c >> 2, k = k0 + (c & 3) * 8;
            bool ok = (t0 + r < T) && (k < K);
            cpa16(&sb[s][r * GLD + (c & 3) * 8], B + (size_t)(ok ? t0 + r : 0) * ldb + (ok ? k : 0), ok);
        }
        cpa_commit();
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
        if (s + 1 < stages) { load((s + 1) & 1, (s + 1) * GBK); cpa_wait<1>(); }
        else cpa_wait<0>();
        __syncthreads();
        const __half* SA = sa[s & 1];
        const __half* SB = sb[s & 1];
        #pragma unroll
        for (int kk = 0; kk < GBK; kk += 16) {
            unsigned a[2][4], b[4][2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const __half* p = &SA[(wm + i * 16 + g) * GLD + kk + tg * 2];
                a[i][0] = *(const unsigned*)p;
                a[i][1] = *(const unsigned*)(p + 8 * GLD);
                a[i][2] = *(const unsigned*)(p + 8);
                a[i][3] = *(const unsigned*)(p + 8 * GLD + 8);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const __half* p = &SB[(wn + j * 8 + g) * GLD + kk + tg * 2];
                b[j][0] = *(const unsigned*)p;
                b[j][1] = *(const unsigned*)(p + 8);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) mma_f16(acc[i][j], a[i], b[j][0], b[j][1]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int r = r0 + wm + i * 16 + g + (e >> 1) * 8, t = t0 + wn + j * 8 + tg * 2 + (e & 1);
                if (r < rows && t < T) {
                    float v = alpha * acc[i][j][e] + (bias ? bias[r] : 0.f);
                    float* o = Y + (size_t)t * ldy + r;
                    *o = add ? *o + v : v;
                }
            }
}

// ---- implicit-GEMM convolution ------------------------------------------------------------------------
// out[p][o] = bias[o] + sum_{ky,kx,ch} W[o][(ky*k + kx)*c + ch] * x[iy][ix][ch] (+ out when add) for
// output pixel p, iy = oy*s + ky - k/2, ix = ox*s + kx - k/2 (zero outside). x is f32 [h][w][c] with
// c % 8 == 0; W is f16 with rows of k*k*c values ordered (ky, kx, channel), so each 8-value k chunk is
// 8 consecutive channels of one input pixel. The patch matrix is never built: B tiles are gathered
// from x straight into shared memory (as f16) while the previous tile multiplies. A 1x1 convolution
// over n pixels is a linear layer with f32 input.
// Block tile 128 output channels x BN pixels x 32 k, 8 warps. BN 64: 4x2 warps of 32x32;
// BN 128: 2x4 warps of 64x32 (twice the MMAs per fragment load; for large images).
// HACC: products are summed in f16 over each 32-value k slice, then added to the f32 accumulators
// (the slice sum is short, so its rounding stays near f16's own; measured against the f32 reference).
template <int BN, bool HACC>
__device__ __forceinline__ void conv_body(
    const __half* __restrict__ W, int ldw, const float* __restrict__ x, float* __restrict__ Y, int oc,
    int h, int w, int c, int k, int s, int oh, int ow, const float* __restrict__ bias, int add) {
    constexpr int WNW = BN / 32, WMW = 8 / WNW; // warps along pixels and along channels
    constexpr int MI = GBM / WMW / 16, NJ = 4;  // 16-row and 8-pixel MMA tiles per warp
    constexpr int BCH = BN / 64;                // B chunks per thread
    __shared__ __align__(16) __half sa[2][GBM * GLD];
    __shared__ __align__(16) __half sb[2][BN * GLD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, tg = lane & 3;
    const int r0 = blockIdx.x * GBM, t0 = blockIdx.y * BN;
    const int wm = (warp % WMW) * (GBM / WMW), wn = (warp / WMW) * 32;
    const int K = k * k * c, np = oh * ow, pad = k / 2;
    const int stages = (K + GBK - 1) / GBK;
    int br[BCH], oy[BCH], ox[BCH];
    #pragma unroll
    for (int u = 0; u < BCH; ++u) {
        br[u] = (tid + u * 256) >> 2;
        int pix = t0 + br[u];
        oy[u] = pix / ow;
        ox[u] = pix % ow;
    }
    const int bk = (tid & 3) * 8;

    auto load_a = [&](int st, int k0) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int cc = tid + i * 256, r = cc >> 2, kk = k0 + (cc & 3) * 8;
            bool ok = (r0 + r < oc) && (kk < K);
            cpa16(&sa[st][r * GLD + (cc & 3) * 8], W + (size_t)(ok ? r0 + r : 0) * ldw + (ok ? kk : 0), ok);
        }
        cpa_commit();
    };
    float4 lo[BCH], hi[BCH];
    auto fetch_b = [&](int k0) {
        int kk = k0 + bk;
        int tap = kk / c, ch = kk - tap * c, ky = tap / k, kx = tap - ky * k;
        #pragma unroll
        for (int u = 0; u < BCH; ++u) {
            lo[u] = hi[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            int iy = oy[u] * s + ky - pad, ix = ox[u] * s + kx - pad;
            if (t0 + br[u] < np && kk < K && iy >= 0 && iy < h && ix >= 0 && ix < w) {
                const float4* src = reinterpret_cast<const float4*>(x + ((size_t)iy * w + ix) * c + ch);
                lo[u] = src[0];
                hi[u] = src[1];
            }
        }
    };
    auto store_b = [&](int st) {
        #pragma unroll
        for (int u = 0; u < BCH; ++u) {
            __half2 v[4] = {__floats2half2_rn(lo[u].x, lo[u].y), __floats2half2_rn(lo[u].z, lo[u].w), __floats2half2_rn(hi[u].x, hi[u].y), __floats2half2_rn(hi[u].z, hi[u].w)};
            *reinterpret_cast<uint4*>(&sb[st][br[u] * GLD + bk]) = *reinterpret_cast<uint4*>(v);
        }
    };

    float acc[MI][NJ][4];
    #pragma unroll
    for (int i = 0; i < MI; ++i)
        #pragma unroll
        for (int j = 0; j < NJ; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    load_a(0, 0);
    fetch_b(0);
    store_b(0);
    for (int st = 0; st < stages; ++st) {
        const bool more = st + 1 < stages;
        if (more) { load_a((st + 1) & 1, (st + 1) * GBK); fetch_b((st + 1) * GBK); cpa_wait<1>(); }
        else cpa_wait<0>();
        __syncthreads();
        const __half* SA = sa[st & 1];
        const __half* SB = sb[st & 1];
        unsigned hacc[MI][NJ][2];
        if (HACC) {
            #pragma unroll
            for (int i = 0; i < MI; ++i)
                #pragma unroll
                for (int j = 0; j < NJ; ++j) hacc[i][j][0] = hacc[i][j][1] = 0u;
        }
        #pragma unroll
        for (int kk = 0; kk < GBK; kk += 16) {
            unsigned a[MI][4], b[NJ][2];
            #pragma unroll
            for (int i = 0; i < MI; ++i) {
                const __half* p = &SA[(wm + i * 16 + g) * GLD + kk + tg * 2];
                a[i][0] = *(const unsigned*)p;
                a[i][1] = *(const unsigned*)(p + 8 * GLD);
                a[i][2] = *(const unsigned*)(p + 8);
                a[i][3] = *(const unsigned*)(p + 8 * GLD + 8);
            }
            #pragma unroll
            for (int j = 0; j < NJ; ++j) {
                const __half* p = &SB[(wn + j * 8 + g) * GLD + kk + tg * 2];
                b[j][0] = *(const unsigned*)p;
                b[j][1] = *(const unsigned*)(p + 8);
            }
            #pragma unroll
            for (int i = 0; i < MI; ++i)
                #pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    if (HACC) mma_h16(hacc[i][j], a[i], b[j][0], b[j][1]);
                    else mma_f16(acc[i][j], a[i], b[j][0], b[j][1]);
                }
        }
        if (HACC) {
            #pragma unroll
            for (int i = 0; i < MI; ++i)
                #pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    float2 lo = __half22float2(*reinterpret_cast<__half2*>(&hacc[i][j][0]));
                    float2 hi = __half22float2(*reinterpret_cast<__half2*>(&hacc[i][j][1]));
                    acc[i][j][0] += lo.x;
                    acc[i][j][1] += lo.y;
                    acc[i][j][2] += hi.x;
                    acc[i][j][3] += hi.y;
                }
        }
        // The other buffer was last read before the previous barrier.
        if (more) store_b((st + 1) & 1);
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < MI; ++i)
        #pragma unroll
        for (int j = 0; j < NJ; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int r = r0 + wm + i * 16 + g + (e >> 1) * 8, t = t0 + wn + j * 8 + tg * 2 + (e & 1);
                if (r < oc && t < np) {
                    float v = acc[i][j][e] + (bias ? bias[r] : 0.f);
                    float* o = Y + (size_t)t * oc + r;
                    *o = add ? *o + v : v;
                }
            }
}

extern "C" __global__ void __launch_bounds__(256) conv_f16(
    const __half* __restrict__ W, int ldw, const float* __restrict__ x, float* __restrict__ Y, int oc,
    int h, int w, int c, int k, int s, int oh, int ow, const float* __restrict__ bias, int add) {
    conv_body<64, false>(W, ldw, x, Y, oc, h, w, c, k, s, oh, ow, bias, add);
}

extern "C" __global__ void __launch_bounds__(256) conv_f16_h(
    const __half* __restrict__ W, int ldw, const float* __restrict__ x, float* __restrict__ Y, int oc,
    int h, int w, int c, int k, int s, int oh, int ow, const float* __restrict__ bias, int add) {
    conv_body<64, true>(W, ldw, x, Y, oc, h, w, c, k, s, oh, ow, bias, add);
}

extern "C" __global__ void __launch_bounds__(256) conv_f16_wide_h(
    const __half* __restrict__ W, int ldw, const float* __restrict__ x, float* __restrict__ Y, int oc,
    int h, int w, int c, int k, int s, int oh, int ow, const float* __restrict__ bias, int add) {
    conv_body<128, true>(W, ldw, x, Y, oc, h, w, c, k, s, oh, ow, bias, add);
}

extern "C" __global__ void __launch_bounds__(256) conv_f16_wide(
    const __half* __restrict__ W, int ldw, const float* __restrict__ x, float* __restrict__ Y, int oc,
    int h, int w, int c, int k, int s, int oh, int ow, const float* __restrict__ bias, int add) {
    conv_body<128, false>(W, ldw, x, Y, oc, h, w, c, k, s, oh, ow, bias, add);
}

// ---- flash attention, head size 64 ---------------------------------------------------------------------
// out[i][h*64..] = softmax(q_i . k_j * scale) v_j over keys j < nk (j <= i when causal), never forming
// the score matrix. Block: 64 queries of one head (grid.y), 4 warps x 16 rows. Per tile of 64 keys:
// S = Q K^T on f16 tensor cores, online softmax in registers (exp2 domain, scale folded into Q), and
// O += P V with P taken straight from the S accumulators as the A operand. K and V are read as f32
// and stored to shared memory as f16 (V transposed, so its B fragments are contiguous pairs).
#define FA_D 64
#define FA_T 64
#define FA_LD (FA_D + 8)

__device__ __forceinline__ unsigned fa_pack(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<unsigned*>(&h);
}

extern "C" __global__ void __launch_bounds__(128) flash_attn64(
    const float* __restrict__ q, int ldq, const float* __restrict__ k, int ldk, const float* __restrict__ v, int ldv,
    float* __restrict__ out, int ldo, int nq, int nk, float scale, int causal) {
    __shared__ __align__(16) __half qs[FA_T * FA_LD];
    __shared__ __align__(16) __half ks[FA_T * FA_LD];
    __shared__ __align__(16) __half vt[FA_D * FA_LD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, tg = lane & 3;
    const int q0 = blockIdx.x * FA_T, hoff = blockIdx.y * FA_D;
    const float qscale = scale * 1.44269504f;

    // Q tile: 64 rows x 64 dims, 32 values per thread (8 float4).
    for (int i = tid; i < FA_T * FA_D / 4; i += 128) {
        int r = i / (FA_D / 4), c4 = (i % (FA_D / 4)) * 4;
        float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < nq) x = *reinterpret_cast<const float4*>(q + (size_t)(q0 + r) * ldq + hoff + c4);
        *reinterpret_cast<uint2*>(&qs[r * FA_LD + c4]) = make_uint2(fa_pack(x.x * qscale, x.y * qscale), fa_pack(x.z * qscale, x.w * qscale));
    }
    __syncthreads();
    unsigned qa[4][4];
    const int wr = warp * 16;
    #pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        const __half* p = &qs[(wr + g) * FA_LD + kk * 16 + tg * 2];
        qa[kk][0] = *(const unsigned*)p;
        qa[kk][1] = *(const unsigned*)(p + 8 * FA_LD);
        qa[kk][2] = *(const unsigned*)(p + 8);
        qa[kk][3] = *(const unsigned*)(p + 8 * FA_LD + 8);
    }

    float o[8][4];
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        #pragma unroll
        for (int e = 0; e < 4; ++e) o[j][e] = 0.f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    const int row0 = q0 + wr + g, row1 = row0 + 8;
    const int kend = causal ? min(nk, q0 + FA_T) : nk;

    for (int k0 = 0; k0 < kend; k0 += FA_T) {
        __syncthreads();
        for (int i = tid; i < FA_T * FA_D / 4; i += 128) {
            int r = i / (FA_D / 4), c4 = (i % (FA_D / 4)) * 4;
            float4 x = make_float4(0.f, 0.f, 0.f, 0.f), y = x;
            if (k0 + r < nk) {
                x = *reinterpret_cast<const float4*>(k + (size_t)(k0 + r) * ldk + hoff + c4);
                y = *reinterpret_cast<const float4*>(v + (size_t)(k0 + r) * ldv + hoff + c4);
            }
            *reinterpret_cast<uint2*>(&ks[r * FA_LD + c4]) = make_uint2(fa_pack(x.x, x.y), fa_pack(x.z, x.w));
            vt[(c4 + 0) * FA_LD + r] = __float2half(y.x);
            vt[(c4 + 1) * FA_LD + r] = __float2half(y.y);
            vt[(c4 + 2) * FA_LD + r] = __float2half(y.z);
            vt[(c4 + 3) * FA_LD + r] = __float2half(y.w);
        }
        __syncthreads();

        float s[8][4];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const __half* p = &ks[(j * 8 + g) * FA_LD + kk * 16 + tg * 2];
                mma_f16(s[j], qa[kk], *(const unsigned*)p, *(const unsigned*)(p + 8));
            }
        }
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int c = k0 + j * 8 + tg * 2;
            if (c >= nk || (causal && c > row0)) s[j][0] = -1e30f;
            if (c + 1 >= nk || (causal && c + 1 > row0)) s[j][1] = -1e30f;
            if (c >= nk || (causal && c > row1)) s[j][2] = -1e30f;
            if (c + 1 >= nk || (causal && c + 1 > row1)) s[j][3] = -1e30f;
            mx0 = fmaxf(mx0, fmaxf(s[j][0], s[j][1]));
            mx1 = fmaxf(mx1, fmaxf(s[j][2], s[j][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
        const float c0 = exp2f(m0 - mx0), c1 = exp2f(m1 - mx1);
        m0 = mx0;
        m1 = mx1;
        float r0 = 0.f, r1 = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            s[j][0] = exp2f(s[j][0] - m0);
            s[j][1] = exp2f(s[j][1] - m0);
            s[j][2] = exp2f(s[j][2] - m1);
            s[j][3] = exp2f(s[j][3] - m1);
            r0 += s[j][0] + s[j][1];
            r1 += s[j][2] + s[j][3];
            o[j][0] *= c0;
            o[j][1] *= c0;
            o[j][2] *= c1;
            o[j][3] *= c1;
        }
        l0 = l0 * c0 + r0;
        l1 = l1 * c1 + r1;
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            unsigned pa[4] = {fa_pack(s[2 * kk][0], s[2 * kk][1]), fa_pack(s[2 * kk][2], s[2 * kk][3]),
                              fa_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]), fa_pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __half* p = &vt[(j * 8 + g) * FA_LD + kk * 16 + tg * 2];
                mma_f16(o[j], pa, *(const unsigned*)p, *(const unsigned*)(p + 8));
            }
        }
    }
    l0 += __shfl_xor_sync(0xffffffff, l0, 1);
    l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1);
    l1 += __shfl_xor_sync(0xffffffff, l1, 2);
    const float i0 = 1.f / l0, i1 = 1.f / l1;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int c = hoff + j * 8 + tg * 2;
        if (row0 < nq) *reinterpret_cast<float2*>(out + (size_t)row0 * ldo + c) = make_float2(o[j][0] * i0, o[j][1] * i0);
        if (row1 < nq) *reinterpret_cast<float2*>(out + (size_t)row1 * ldo + c) = make_float2(o[j][2] * i1, o[j][3] * i1);
    }
}

// ---- GroupNorm statistics across the whole card --------------------------------------------------------
// Pass 1: block b sums a range of 64 pixels per channel, shifted by one sample of the channel's group
// (pixel 0) so the variance keeps its precision: (sum(x - K), sum((x - K)^2)), coalesced across
// channels. Pass 2: one block per group adds the partials and writes (mean, inv std).
extern "C" __global__ void group_partial(const float* __restrict__ x, float* __restrict__ part, int n, int c, int groups) {
    int p0 = blockIdx.x * 64, p1 = min(n, p0 + 64), gs = c / groups;
    for (int ch = threadIdx.x; ch < c; ch += blockDim.x) {
        float k = x[(ch / gs) * gs];
        float s1 = 0.f, s2 = 0.f;
        for (int p = p0; p < p1; ++p) {
            float d = x[(size_t)p * c + ch] - k;
            s1 += d;
            s2 += d * d;
        }
        float* o = part + ((size_t)blockIdx.x * c + ch) * 2;
        o[0] = s1;
        o[1] = s2;
    }
}

extern "C" __global__ void group_merge(const float* __restrict__ x, const float* __restrict__ part, float2* __restrict__ stats, int parts, int n, int c, int groups, float eps) {
    __shared__ float red[32];
    int g = blockIdx.x, gs = c / groups;
    float s1 = 0.f, s2 = 0.f;
    for (int i = threadIdx.x; i < parts * gs; i += blockDim.x) {
        const float* q = part + ((size_t)(i / gs) * c + g * gs + i % gs) * 2;
        s1 += q[0];
        s2 += q[1];
    }
    s1 = block_sum(s1, red);
    s2 = block_sum(s2, red);
    if (threadIdx.x == 0) {
        float cnt = (float)n * gs, m = s1 / cnt;
        stats[g] = make_float2(x[g * gs] + m, rsqrtf(fmaxf(s2 / cnt - m * m, 0.f) + eps));
    }
}

// ---- conversions and layout ---------------------------------------------------------------------------

// y (rows x ldy, f16) = x (rows x cols, f32 with row stride ldx), zero padding cols..ldy.
extern "C" __global__ void to_f16(const float* __restrict__ x, int ldx, __half* __restrict__ y, int ldy, int rows, int cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * ldy) return;
    int r = (int)(i / ldy), c = (int)(i % ldy);
    y[i] = __float2half(c < cols ? x[(size_t)r * ldx + c] : 0.f);
}

// yT (cols x ldt, f16) = x^T for x (rows x cols, f32 with stride ldx): a head's values as GEMM weights.
extern "C" __global__ void transpose_f16(const float* __restrict__ x, int ldx, __half* __restrict__ y, int ldt, int rows, int cols) {
    __shared__ float tile[32][33];
    int bx = blockIdx.x * 32, by = blockIdx.y * 32;
    int tx = threadIdx.x, ty = threadIdx.y;
    for (int j = ty; j < 32; j += 8) {
        int r = by + j, c = bx + tx;
        tile[j][tx] = (r < rows && c < cols) ? x[(size_t)r * ldx + c] : 0.f;
    }
    __syncthreads();
    for (int j = ty; j < 32; j += 8) {
        int c = bx + j, r = by + tx; // output row c (a column of x), column r
        if (c < cols && r < ldt) y[(size_t)c * ldt + r] = __float2half(r < rows ? tile[tx][j] : 0.f);
    }
}

// Patches for a k x k convolution with stride s and padding k/2: x [h][w][c] f32 -> P [oh*ow rows from
// row0][c*k*k (channel, ky, kx), padded to ldp] f16.
extern "C" __global__ void im2col_f16(const float* __restrict__ x, int h, int w, int c, int k, int s, int ow,
                                      int row0, int nrows, __half* __restrict__ P, int ldp) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nrows * ldp) return;
    int p = (int)(i / ldp), col = (int)(i % ldp);
    float v = 0.f;
    if (col < c * k * k) {
        int ch = col / (k * k), rem = col % (k * k), ky = rem / k, kx = rem % k;
        int pix = row0 + p, oy = pix / ow, ox = pix % ow;
        int iy = oy * s + ky - k / 2, ix = ox * s + kx - k / 2;
        if (iy >= 0 && iy < h && ix >= 0 && ix < w) v = x[((size_t)iy * w + ix) * c + ch];
    }
    P[i] = __float2half(v);
}

extern "C" __global__ void upsample2(const float* __restrict__ x, float* __restrict__ y, int h, int w, int c) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)4 * h * w * c;
    if (i >= n) return;
    int ch = (int)(i % c);
    size_t pix = i / c;
    int oy = (int)(pix / (2 * w)), ox = (int)(pix % (2 * w));
    y[i] = x[((size_t)(oy / 2) * w + ox / 2) * c + ch];
}

// out [n][ca + cb] = [a | b] per row.
extern "C" __global__ void concat_rows(const float* __restrict__ a, int ca, const float* __restrict__ b, int cb, float* __restrict__ y, int n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int cc = ca + cb;
    if (i >= (size_t)n * cc) return;
    int r = (int)(i / cc), c = (int)(i % cc);
    y[i] = c < ca ? a[(size_t)r * ca + c] : b[(size_t)r * cb + (c - ca)];
}

// ---- elementwise --------------------------------------------------------------------------------------

// op: 0 SiLU, 1 GELU (tanh), 2 GELU (erf).
extern "C" __global__ void act(float* __restrict__ x, size_t n, int op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (op == 0) v = v / (1.f + __expf(-v));
    else if (op == 1) v = 0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    else v = 0.5f * v * (1.f + erff(v * 0.70710678f));
    x[i] = v;
}

// y[r][j] = a * gelu_erf(b) for a = x[r][j], b = x[r][inner + j] (GEGLU), x rows of 2*inner.
extern "C" __global__ void geglu(const float* __restrict__ x, float* __restrict__ y, int n, int inner) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * inner) return;
    int r = (int)(i / inner), j = (int)(i % inner);
    float a = x[(size_t)r * 2 * inner + j], b = x[(size_t)r * 2 * inner + inner + j];
    y[i] = a * 0.5f * b * (1.f + erff(b * 0.70710678f));
}

// x[r][c] += v[c] (a per-channel vector, e.g. a time embedding) or y += x elementwise when v is null.
extern "C" __global__ void add_rows(float* __restrict__ x, const float* __restrict__ v, const float* __restrict__ other, size_t n, int c) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += v ? v[i % c] : other[i];
}

// ---- normalisations -------------------------------------------------------------------------------------

__device__ __forceinline__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) red[w] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? red[threadIdx.x] : 0.f;
    if (w == 0)
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    __syncthreads();
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    v = red[0];
    __syncthreads();
    return v;
}

// One block per row: LayerNorm (rms == 0) or RMSNorm (rms == 1), with optional bias.
extern "C" __global__ void row_norm(const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ w,
                                    const float* __restrict__ b, int dim, float eps, int rms) {
    __shared__ float red[32];
    const float* xr = x + (size_t)blockIdx.x * dim;
    float* yr = y + (size_t)blockIdx.x * dim;
    float mean = 0.f;
    if (!rms) {
        float s = 0.f;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i];
        mean = block_sum(s, red) / dim;
    }
    float s2 = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float d = xr[i] - mean; s2 += d * d; }
    float inv = rsqrtf(block_sum(s2, red) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = (xr[i] - mean) * inv * w[i] + (b ? b[i] : 0.f);
}

// y = GroupNorm(x) * w + b, then SiLU when silu != 0.
extern "C" __global__ void group_apply(const float* __restrict__ x, float* __restrict__ y, const float2* __restrict__ stats,
                                       const float* __restrict__ w, const float* __restrict__ b, size_t n, int c, int groups, int silu) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ch = (int)(i % c);
    float2 st = stats[ch / (c / groups)];
    float v = (x[i] - st.x) * st.y * w[ch] + b[ch];
    if (silu) v = v / (1.f + __expf(-v));
    y[i] = v;
}

// Row softmax of S (rows x ld, valid columns n; row i sees columns 0..=i when causal) scaled by
// `scale`, written as f16 P (rows x ld, zero padded): the probabilities for the second attention
// GEMM. One block per row.
extern "C" __global__ void softmax_f16(const float* __restrict__ S, __half* __restrict__ P, int n, int ld, float scale, int causal) {
    __shared__ float red[32];
    if (causal && (int)blockIdx.x + 1 < n) n = blockIdx.x + 1;
    const float* s = S + (size_t)blockIdx.x * ld;
    __half* p = P + (size_t)blockIdx.x * ld;
    float m = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, s[i]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) red[w] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mm = red[0];
        for (int i = 1; i < (int)(blockDim.x >> 5); i++) mm = fmaxf(mm, red[i]);
        red[0] = mm;
    }
    __syncthreads();
    m = red[0];
    __syncthreads();
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += __expf((s[i] - m) * scale);
    float inv = 1.f / block_sum(sum, red);
    for (int i = threadIdx.x; i < ld; i += blockDim.x) p[i] = __float2half(i < n ? __expf((s[i] - m) * scale) * inv : 0.f);
}
