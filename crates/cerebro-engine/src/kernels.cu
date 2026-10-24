#include <cuda_fp16.h>

// Cerebro Engine CUDA kernels. Compiled to PTX by build.rs with nvcc and embedded in the binary;
// the driver JIT-compiles the PTX for the user's GPU (Turing sm_75 and newer).
//
// Decode is memory-bound, so each kernel does as much as possible per launch: norms quantise their
// output for the next matmul, matmuls can add straight into the residual, Q/K/V and gate/up are one
// matrix each. Positions and token ids are read from device memory, so one decode step is captured
// once as a CUDA graph and replayed for every token.

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// Block-wide sum for a block of up to 1024 threads; every thread gets the result.
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = lane < ((blockDim.x + 31) >> 5) ? red[lane] : 0.f;
    t = warp_sum(t);
    __syncthreads();
    return t;
}

// y = W x for int8 weights (one f32 scale per 32) and int8 activations. One warp per row, each lane
// reading 16 bytes per iteration. If `add` is set the result is added to y (residual connection).
extern "C" __global__ void matvec_q8(const signed char* __restrict__ qs, const float* __restrict__ ds,
                                     const signed char* __restrict__ xq, const float* __restrict__ xd,
                                     float* __restrict__ y, int rows, int cols, int add) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    int nb = cols >> 5;
    const signed char* w = qs + (size_t)row * cols;
    const float* d = ds + (size_t)row * nb;
    float sum = 0.f;
    #pragma unroll 4
    for (int i = lane * 16; i < cols; i += 512) {
        int4 wv = __ldg(reinterpret_cast<const int4*>(w + i));
        int4 xv = *reinterpret_cast<const int4*>(xq + i);
        int acc = __dp4a(wv.x, xv.x, 0);
        acc = __dp4a(wv.y, xv.y, acc);
        acc = __dp4a(wv.z, xv.z, acc);
        acc = __dp4a(wv.w, xv.w, acc);
        int b = i >> 5;
        sum += (float)acc * __ldg(d + b) * xd[b];
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = add ? y[row] + sum : sum;
}

// Quantise one block of 32 values per warp.
__device__ __forceinline__ void quant32(float v, signed char* xq, float* xd, int i, int lane) {
    float amax = warp_max(fabsf(v));
    float d = amax / 127.f;
    float id = d > 0.f ? 1.f / d : 0.f;
    xq[i] = (signed char)__float2int_rn(v * id);
    if (lane == 0) xd[i >> 5] = d;
}

extern "C" __global__ void quantize_q8(const float* __restrict__ x, signed char* __restrict__ xq,
                                       float* __restrict__ xd, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i & ~31) >= n) return;
    quant32(x[i], xq, xd, i, threadIdx.x & 31);
}

// RMSNorm then quantise for the next matmul; also writes the normed f32 to y if y is not null.
// One block of 1024 threads, n <= 8192.
extern "C" __global__ void rmsnorm_quant(const float* __restrict__ x, const float* __restrict__ w,
                                         signed char* __restrict__ xq, float* __restrict__ xd,
                                         float* __restrict__ y, int n, float eps) {
    __shared__ float red[32];
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    float s = rsqrtf(block_sum(ss, red) / n + eps);
    for (int base = 0; base < n; base += blockDim.x) {
        int i = base + threadIdx.x;
        float v = i < n ? x[i] * s * w[i] : 0.f;
        if (i < n) {
            if (y) y[i] = v;
            quant32(v, xq, xd, i, threadIdx.x & 31);
        }
    }
}

// After the fused QKV matmul for T tokens at positions pos0..pos0+T-1 (T = gridDim.y; decode uses
// T = 1 with pos0 read from device memory): per-head RMSNorm (optional) + NeoX RoPE for Q (in place)
// and K (written into the f16 K cache), V copied into the f16 V cache. Grid: (n_head + n_kv, T).
// Per head of q and k: optional RMS norm, then RoPE on the first `rd` dimensions (pairs i, i + rd/2;
// rd == hd rotates the whole head), and k and v into the f16 cache. Grid (n_head + n_kv, T), block hd/2.
// Interleaved multi-axis rotary (Qwen3-VL / Qwen3.5): which of a token's time, height and width
// positions frequency j uses. `sec` packs the four section sizes one byte each; 0 = plain RoPE.
__device__ __forceinline__ float rope_axis(int j, int sec, int pt, int ph, int pw) {
    if (sec == 0) return (float)pt;
    int s0 = sec & 255, s1 = (sec >> 8) & 255, s2 = (sec >> 16) & 255, s3 = (sec >> 24) & 255;
    int total = s0 + s1 + s2 + s3;
    if (total == 0) return (float)pt;
    int k = j % total;
    if (k % 3 == 1 && k < 3 * s1) return (float)ph;
    if (k % 3 == 2 && k < 3 * s2) return (float)pw;
    if (k % 3 == 0 && k < 3 * s0) return (float)pt;
    return 0.f;
}

extern "C" __global__ void qkv_post(float* __restrict__ qkv, const float* __restrict__ qn, const float* __restrict__ kn,
                                    int has_norm, __half* __restrict__ kc, __half* __restrict__ vc,
                                    int n_head, int n_kv, int hd, const int* __restrict__ posp, float base, float eps, int rd,
                                    const int* __restrict__ rtab, int sec, const float* __restrict__ ftab, float msc) {
    // ftab: the model's rotary frequencies (YaRN-adjusted for an extended context); msc: YaRN's
    // scale on the rotated values (1 at the trained length).
    __shared__ float red[32];
    __shared__ float hv[256];
    int t = blockIdx.y;
    // posp[0] is the cache slot of token 0, posp[1] how far rotary positions lag the slots (the
    // positions images saved); rtab, when given, holds (time, height, width) per token instead.
    int pos = posp[0] + t;
    int pt = pos - posp[1], ph = pt, pw = pt;
    if (rtab) {
        pt = rtab[3 * t];
        ph = rtab[3 * t + 1];
        pw = rtab[3 * t + 2];
    }
    int h = blockIdx.x, i = threadIdx.x, half = hd >> 1;
    int kvd = n_kv * hd;
    int row = n_head * hd + 2 * kvd;
    float* base_t = qkv + (size_t)t * row;
    bool is_k = h >= n_head;
    float* src = is_k ? base_t + n_head * hd + (h - n_head) * hd : base_t + h * hd;
    float a = src[i], b = src[i + half];
    if (has_norm) {
        const float* nw = is_k ? kn : qn;
        float s = rsqrtf(block_sum(a * a + b * b, red) / hd + eps);
        a = a * s * nw[i];
        b = b * s * nw[i + half];
    }
    float ra = a, rb = b;
    if (rd == hd) {
        float inv = ftab ? ftab[i] : powf(base, -2.f * i / hd);
        float sn, cs;
        sincosf(rope_axis(i, sec, pt, ph, pw) * inv, &sn, &cs);
        ra = (a * cs - b * sn) * msc;
        rb = (a * sn + b * cs) * msc;
    } else {
        // Partial rotation: pairs (i, i + rd/2) for i < rd/2 cross this thread's (i, i + half).
        // Every thread reaches the same barriers; only the first rd/2 rotate.
        hv[i] = a;
        hv[i + half] = b;
        __syncthreads();
        int rh = rd >> 1;
        float x0 = 0.f, x1 = 0.f;
        if (i < rh) {
            x0 = hv[i];
            x1 = hv[i + rh];
        }
        __syncthreads();
        if (i < rh) {
            float inv = ftab ? ftab[i] : powf(base, -2.f * i / rd);
            float sn, cs;
            sincosf(rope_axis(i, sec, pt, ph, pw) * inv, &sn, &cs);
            hv[i] = (x0 * cs - x1 * sn) * msc;
            hv[i + rh] = (x0 * sn + x1 * cs) * msc;
        }
        __syncthreads();
        ra = hv[i];
        rb = hv[i + half];
    }
    if (is_k) {
        size_t o = (size_t)pos * kvd + (h - n_head) * hd;
        kc[o + i] = __float2half(ra);
        kc[o + i + half] = __float2half(rb);
        const float* v = base_t + n_head * hd + kvd + (h - n_head) * hd;
        vc[o + i] = __float2half(v[i]);
        vc[o + i + half] = __float2half(v[i + half]);
    } else {
        src[i] = ra;
        src[i + half] = rb;
    }
}

// Split-K decode attention ("flash decoding"): grid (n_head, splits); block s covers positions
// [s*chunk, (s+1)*chunk) of the len = *pos + 1 cached ones with 4 warps and an online softmax, and
// writes its partial max, sum and weighted values. attn_combine merges the splits. Long contexts use
// every SM instead of one block per head. head_dim up to 256.
extern "C" __global__ void attn_partial(const float* __restrict__ q, const __half* __restrict__ kc,
                                        const __half* __restrict__ vc, float* __restrict__ pm, float* __restrict__ pl,
                                        float* __restrict__ pacc, int n_kv, int hd, const int* __restrict__ posp,
                                        int group, float scale, int chunk) {
    __shared__ float sm[4], sl[4];
    __shared__ float sacc[4][256];
    int len = *posp + 1;
    int head = blockIdx.x, split = blockIdx.y, splits = gridDim.y;
    int start = split * chunk, end = min(len, start + chunk);
    int kvh = head / group;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int per = (hd + 31) >> 5;
    size_t pidx = (size_t)head * splits + split;
    if (start >= len) {
        if (threadIdx.x == 0) { pm[pidx] = -1e30f; pl[pidx] = 0.f; }
        return;
    }
    const float* qh = q + head * hd;
    float qv[8], acc[8];
    for (int j = 0; j < per; ++j) { int d = lane + j * 32; qv[j] = d < hd ? qh[d] : 0.f; acc[j] = 0.f; }
    float m = -1e30f, l = 0.f;
    int stride = n_kv * hd;
    for (int t = start + warp; t < end; t += 4) {
        const __half* kt = kc + (size_t)t * stride + kvh * hd;
        float s = 0.f;
        for (int j = 0; j < per; ++j) { int d = lane + j * 32; if (d < hd) s += qv[j] * __half2float(kt[d]); }
        s = warp_sum(s) * scale;
        float mn = fmaxf(m, s);
        float c = __expf(m - mn), p = __expf(s - mn);
        l = l * c + p;
        const __half* vt = vc + (size_t)t * stride + kvh * hd;
        for (int j = 0; j < per; ++j) { int d = lane + j * 32; if (d < hd) acc[j] = acc[j] * c + p * __half2float(vt[d]); }
        m = mn;
    }
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    for (int j = 0; j < per; ++j) { int d = lane + j * 32; if (d < hd) sacc[warp][d] = acc[j]; }
    __syncthreads();
    if (warp == 0) {
        float M = -1e30f;
        for (int w = 0; w < 4; ++w) M = fmaxf(M, sm[w]);
        float L = 0.f;
        for (int w = 0; w < 4; ++w) L += sl[w] * __expf(sm[w] - M);
        for (int j = 0; j < per; ++j) {
            int d = lane + j * 32;
            if (d < hd) {
                float o = 0.f;
                for (int w = 0; w < 4; ++w) o += sacc[w][d] * __expf(sm[w] - M);
                pacc[pidx * hd + d] = o;
            }
        }
        if (lane == 0) { pm[pidx] = M; pl[pidx] = L; }
    }
}

// Decode attention for grouped-query models: one block per (kv head, split) serves every query head
// that shares the kv head, so K and V are read once per group instead of once per query head.
// Scores: 8 lanes share a K row, each reading one full 32-byte sector, so a warp reads 4 whole rows
// per load and reduces each with 3 shuffles. Values: each lane owns HD/32 contiguous dims and reads
// them in one load per row. Compiled per head size and group size so registers fit the model. The
// split's range is derived from *posp at run time, so the launch shape is fixed (graph capture)
// while the work follows the context length. Writes the same partials as attn_partial; attn_combine
// merges them.
template <int HD> struct DecVec;
template <> struct DecVec<64> { typedef uint32_t T; };   // 2 halves per lane in the V pass
template <> struct DecVec<128> { typedef uint2 T; };     // 4 halves
template <> struct DecVec<256> { typedef uint4 T; };     // 8 halves

template <int HD, int G>
__device__ __forceinline__ void attn_decode_gqa_t(const float* __restrict__ q, const __half* __restrict__ kc,
                                                  const __half* __restrict__ vc, float* __restrict__ pm,
                                                  float* __restrict__ pl, float* __restrict__ pacc, int n_kv,
                                                  const int* __restrict__ posp, int group, float scale) {
    constexpr int PER = HD / 32;     // V dims per lane
    constexpr int KL = HD / 8;       // K halves per lane (8 lanes per row)
    constexpr int KV4 = KL / 8;      // ... as 16-byte loads
    __shared__ float qs[G * HD];
    __shared__ float wm[4][G], wl[4][G];
    __shared__ float wacc[4][G * HD];
    const int len = *posp + 1;
    const int kvh = blockIdx.x, split = blockIdx.y, splits = gridDim.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int chunk = (((len + splits - 1) / splits) + 31) & ~31;
    const int start = split * chunk, end = min(len, start + chunk);
    if (start >= len) {
        if (threadIdx.x < group) {
            size_t i = (size_t)(kvh * group + threadIdx.x) * splits + split;
            pm[i] = -1e30f;
            pl[i] = 0.f;
        }
        return;
    }
    for (int i = threadIdx.x; i < G * HD; i += blockDim.x) qs[i] = i < group * HD ? q[(size_t)kvh * group * HD + i] * scale : 0.f;
    __syncthreads();

    const size_t stride = (size_t)n_kv * HD;
    const int sub = lane & 7, rq = lane >> 3;  // K pass: this lane's slice of the row, and which of 4 rows
    float m[G], l[G], acc[G][PER];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -1e30f;
        l[g] = 0.f;
#pragma unroll
        for (int j = 0; j < PER; ++j) acc[g][j] = 0.f;
    }
    for (int base = start + warp * 32; base < end; base += 128) {
        // Scores for the tile's 32 positions, 4 rows per step.
        float s[G];
#pragma unroll
        for (int g = 0; g < G; ++g) s[g] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = base + i * 4 + rq;
            float part[G];
#pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.f;
            if (row < end) {
                const uint4* kt = reinterpret_cast<const uint4*>(kc + row * stride + (size_t)kvh * HD + sub * KL);
#pragma unroll
                for (int c = 0; c < KV4; ++c) {
                    const uint4 raw = kt[c];
                    const __half2* h = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 k = __half22float2(h[e]);
                        const int d = sub * KL + c * 8 + e * 2;
#pragma unroll
                        for (int g = 0; g < G; ++g) part[g] += k.x * qs[g * HD + d] + k.y * qs[g * HD + d + 1];
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float v = part[g];
                v += __shfl_xor_sync(0xffffffff, v, 1);
                v += __shfl_xor_sync(0xffffffff, v, 2);
                v += __shfl_xor_sync(0xffffffff, v, 4);
                // Position base + lane is row (lane >> 2) * 4 + (lane & 3): taken from step lane >> 2.
                const float mine = __shfl_sync(0xffffffff, v, (lane & 3) << 3);
                if ((lane >> 2) == i) s[g] = mine;
            }
        }
        const bool valid = base + lane < end;
        float p[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float mx = warp_max(valid ? s[g] : -1e30f);
            const float mn = fmaxf(m[g], mx);
            const float c = __expf(m[g] - mn);
            p[g] = valid ? __expf(s[g] - mn) : 0.f;
            l[g] = l[g] * c + warp_sum(p[g]);
#pragma unroll
            for (int j = 0; j < PER; ++j) acc[g][j] *= c;
            m[g] = mn;
        }
        // Values: lane owns dims [lane * PER, lane * PER + PER), one load per row.
        const int n = min(32, end - base);
        for (int j0 = 0; j0 < n; j0 += 4) {
            typename DecVec<HD>::T raw[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int j = min(j0 + u, n - 1);
                raw[u] = *reinterpret_cast<const typename DecVec<HD>::T*>(vc + (base + j) * stride + (size_t)kvh * HD + lane * PER);
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const __half2* h = reinterpret_cast<const __half2*>(&raw[u]);
                float v[PER];
#pragma unroll
                for (int e = 0; e < PER / 2; ++e) {
                    const float2 f = __half22float2(h[e]);
                    v[2 * e] = f.x;
                    v[2 * e + 1] = f.y;
                }
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float pj = j0 + u < n ? __shfl_sync(0xffffffff, p[g], (j0 + u) & 31) : 0.f;
#pragma unroll
                    for (int e = 0; e < PER; ++e) acc[g][e] += pj * v[e];
                }
            }
        }
    }
    // Merge the four warps.
    if (lane == 0)
#pragma unroll
        for (int g = 0; g < G; ++g) { wm[warp][g] = m[g]; wl[warp][g] = l[g]; }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int e = 0; e < PER; ++e) wacc[warp][g * HD + lane * PER + e] = acc[g][e];
    __syncthreads();
    for (int i = threadIdx.x; i < group * HD; i += blockDim.x) {
        const int g = i / HD, d = i % HD;
        float M = -1e30f;
        for (int w = 0; w < 4; ++w) M = fmaxf(M, wm[w][g]);
        float o = 0.f;
        for (int w = 0; w < 4; ++w) o += wacc[w][g * HD + d] * __expf(wm[w][g] - M);
        pacc[((size_t)(kvh * group + g) * splits + split) * HD + d] = o;
    }
    if (threadIdx.x < group) {
        const int g = threadIdx.x;
        float M = -1e30f;
        for (int w = 0; w < 4; ++w) M = fmaxf(M, wm[w][g]);
        float L = 0.f;
        for (int w = 0; w < 4; ++w) L += wl[w][g] * __expf(wm[w][g] - M);
        const size_t i = (size_t)(kvh * group + g) * splits + split;
        pm[i] = M;
        pl[i] = L;
    }
}

#define ATTN_DECODE_GQA(HD, G)                                                                                         \
    extern "C" __global__ void attn_decode_gqa_##HD##_##G(const float* __restrict__ q, const __half* __restrict__ kc, \
                                                          const __half* __restrict__ vc, float* __restrict__ pm,      \
                                                          float* __restrict__ pl, float* __restrict__ pacc, int n_kv, \
                                                          const int* __restrict__ posp, int group, float scale) {     \
        attn_decode_gqa_t<HD, G>(q, kc, vc, pm, pl, pacc, n_kv, posp, group, scale);                                  \
    }
ATTN_DECODE_GQA(64, 1) ATTN_DECODE_GQA(64, 2) ATTN_DECODE_GQA(64, 4) ATTN_DECODE_GQA(64, 8)
ATTN_DECODE_GQA(128, 1) ATTN_DECODE_GQA(128, 2) ATTN_DECODE_GQA(128, 4) ATTN_DECODE_GQA(128, 8)
ATTN_DECODE_GQA(256, 1) ATTN_DECODE_GQA(256, 2) ATTN_DECODE_GQA(256, 4) ATTN_DECODE_GQA(256, 8)

extern "C" __global__ void attn_combine(const float* __restrict__ pm, const float* __restrict__ pl,
                                        const float* __restrict__ pacc, float* __restrict__ out, int hd, int splits) {
    int head = blockIdx.x, d = threadIdx.x;
    if (d >= hd) return;
    float M = -1e30f;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, pm[head * splits + s]);
    float L = 0.f, o = 0.f;
    for (int s = 0; s < splits; ++s) {
        float w = __expf(pm[head * splits + s] - M);
        L += pl[head * splits + s] * w;
        o += pacc[((size_t)head * splits + s) * hd + d] * w;
    }
    out[head * hd + d] = o / L;
}

// Causal attention for a prefill chunk: query t (blockIdx.y) at position *pos0 + t attends to cached
// positions 0..pos. One block per (head, token), 8 warps, online softmax.
extern "C" __global__ void attn_prefill(const float* __restrict__ qkv, const __half* __restrict__ kc,
                                        const __half* __restrict__ vc, float* __restrict__ out,
                                        int n_head, int n_kv, int hd, const int* __restrict__ posp, int group, float scale) {
    __shared__ float sm[8], sl[8];
    __shared__ float sacc[8][256];
    int t = blockIdx.y;
    int len = *posp + t + 1;
    int head = blockIdx.x, kvh = head / group;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int per = (hd + 31) >> 5;
    int row = n_head * hd + 2 * n_kv * hd;
    const float* qh = qkv + (size_t)t * row + head * hd;
    float qv[8], acc[8];
    for (int j = 0; j < per; ++j) { int d = lane + j * 32; qv[j] = d < hd ? qh[d] : 0.f; acc[j] = 0.f; }
    float m = -1e30f, l = 0.f;
    int stride = n_kv * hd;
    for (int p = warp; p < len; p += 8) {
        const __half* kt = kc + (size_t)p * stride + kvh * hd;
        float s = 0.f;
        for (int j = 0; j < per; ++j) { int d = lane + j * 32; if (d < hd) s += qv[j] * __half2float(kt[d]); }
        s = warp_sum(s) * scale;
        float mn = fmaxf(m, s);
        float c = __expf(m - mn), pr = __expf(s - mn);
        l = l * c + pr;
        const __half* vt = vc + (size_t)p * stride + kvh * hd;
        for (int j = 0; j < per; ++j) { int d = lane + j * 32; if (d < hd) acc[j] = acc[j] * c + pr * __half2float(vt[d]); }
        m = mn;
    }
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    for (int j = 0; j < per; ++j) { int d = lane + j * 32; if (d < hd) sacc[warp][d] = acc[j]; }
    __syncthreads();
    if (warp == 0) {
        float M = -1e30f;
        for (int w = 0; w < 8; ++w) M = fmaxf(M, sm[w]);
        float L = 0.f;
        for (int w = 0; w < 8; ++w) L += sl[w] * __expf(sm[w] - M);
        for (int j = 0; j < per; ++j) {
            int d = lane + j * 32;
            if (d < hd) {
                float o = 0.f;
                for (int w = 0; w < 8; ++w) o += sacc[w][d] * __expf(sm[w] - M);
                out[(size_t)t * n_head * hd + head * hd + d] = o / L;
            }
        }
    }
}

// act = silu(gate) * up from the fused gate|up output, quantised for the down projection.
extern "C" __global__ void silu_mul_quant(const float* __restrict__ gu, signed char* __restrict__ xq,
                                          float* __restrict__ xd, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i & ~31) >= n) return;
    float g = gu[i], u = gu[n + i];
    quant32(g / (1.f + __expf(-g)) * u, xq, xd, i, threadIdx.x & 31);
}

// Embedding lookup for the token id in device memory.
extern "C" __global__ void embed(const signed char* __restrict__ qs, const float* __restrict__ ds,
                                 const int* __restrict__ tok, float* __restrict__ x, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    size_t r = (size_t)(*tok) * dim;
    x[i] = (float)qs[r + i] * ds[(r + i) >> 5];
}

// Greedy pick on the GPU: index of the largest logit, written to out[0]; also advances *pos and
// stores the token as the next input so the graph can run again without the host.
extern "C" __global__ void argmax_next(const float* __restrict__ logits, int n, int* __restrict__ out,
                                       int* __restrict__ tok, int* __restrict__ pos) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float best = -1e30f;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = logits[i];
        if (v > best) { best = v; idx = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best, o);
        int oi = __shfl_xor_sync(0xffffffff, idx, o);
        if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) { bv[warp] = best; bi[warp] = idx; }
    __syncthreads();
    if (warp == 0) {
        int nw = blockDim.x >> 5;
        best = lane < nw ? bv[lane] : -1e30f;
        idx = lane < nw ? bi[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, best, o);
            int oi = __shfl_xor_sync(0xffffffff, idx, o);
            if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
        }
        if (lane == 0) { out[0] = idx; *tok = idx; *pos = *pos + 1; }
    }
}

// ---- Batched (prefill / verification) kernels: T tokens per launch ----------------------------

extern "C" __global__ void embed_batch(const signed char* __restrict__ qs, const float* __restrict__ ds,
                                       const int* __restrict__ toks, float* __restrict__ x, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= dim) return;
    size_t r = (size_t)toks[t] * dim;
    x[(size_t)t * dim + i] = (float)qs[r + i] * ds[(r + i) >> 5];
}

// Row t of x (grid T): RMSNorm then quantise into row t of xq/xd.
extern "C" __global__ void rmsnorm_quant_batch(const float* __restrict__ x, const float* __restrict__ w,
                                               signed char* __restrict__ xq, float* __restrict__ xd, int n, float eps,
                                               float* __restrict__ yout) {
    __shared__ float red[32];
    const float* xr = x + (size_t)blockIdx.x * n;
    signed char* q = xq + (size_t)blockIdx.x * n;
    float* d = xd + (size_t)blockIdx.x * (n >> 5);
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    float s = rsqrtf(block_sum(ss, red) / n + eps);
    for (int base = 0; base < n; base += blockDim.x) {
        int i = base + threadIdx.x;
        if (i < n) {
            float v = xr[i] * s * w[i];
            if (yout) yout[(size_t)blockIdx.x * n + i] = v;
            quant32(v, q, d, i, threadIdx.x & 31);
        }
    }
}

// Y[t][row] (+)= W[row] . X[t] for T tokens. One warp per weight row; each 16-byte weight chunk is
// loaded once and used for TT tokens at a time, so the weights are read T/TT times instead of T.
#define TT 16
extern "C" __global__ void matmul_q8_batch(const signed char* __restrict__ qs, const float* __restrict__ ds,
                                           const signed char* __restrict__ xq, const float* __restrict__ xd,
                                           float* __restrict__ y, int rows, int cols, int T, int add) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    int nb = cols >> 5;
    const signed char* w = qs + (size_t)row * cols;
    const float* d = ds + (size_t)row * nb;
    for (int t0 = 0; t0 < T; t0 += TT) {
        int tn = min(TT, T - t0);
        float sum[TT];
        #pragma unroll
        for (int k = 0; k < TT; ++k) sum[k] = 0.f;
        for (int i = lane * 16; i < cols; i += 512) {
            int4 wv = __ldg(reinterpret_cast<const int4*>(w + i));
            int b = i >> 5;
            float wd = __ldg(d + b);
            #pragma unroll
            for (int k = 0; k < TT; ++k) {
                if (k < tn) {
                    const signed char* xr = xq + (size_t)(t0 + k) * cols;
                    int4 xv = *reinterpret_cast<const int4*>(xr + i);
                    int acc = __dp4a(wv.x, xv.x, 0);
                    acc = __dp4a(wv.y, xv.y, acc);
                    acc = __dp4a(wv.z, xv.z, acc);
                    acc = __dp4a(wv.w, xv.w, acc);
                    sum[k] += (float)acc * wd * xd[(size_t)(t0 + k) * nb + b];
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < TT; ++k) {
            if (k < tn) {
                float v = warp_sum(sum[k]);
                if (lane == 0) {
                    float* o = y + (size_t)(t0 + k) * rows + row;
                    *o = add ? *o + v : v;
                }
            }
        }
    }
}

extern "C" __global__ void silu_mul_quant_batch(const float* __restrict__ gu, signed char* __restrict__ xq,
                                                float* __restrict__ xd, int ffn, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i & ~31) >= ffn * T) return;
    int t = i / ffn, j = i - t * ffn;
    const float* r = gu + (size_t)t * 2 * ffn;
    float g = r[j], u = r[ffn + j];
    quant32(g / (1.f + __expf(-g)) * u, xq, xd, i, threadIdx.x & 31);
}

// ---- Mixture of experts: router and K-quant expert kernels ------------------------------------
//
// Experts resident in VRAM keep their GGUF K-quant format (4-6 bits per weight, so about twice as
// many fit as with an 8-bit repack). Q4_K (144 B) and Q5_K (176 B) blocks are used as stored; Q6_K
// blocks are padded from 210 to 224 bytes at upload so every 16-byte load is aligned. A block of
// 256 values is processed in 16-value units with dp4a against int8 activations.

__device__ __forceinline__ void scale_min_k4(int j, const unsigned char* q, int& sc, int& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
__device__ __forceinline__ int dot16(int4 a, int4 b) {
    return __dp4a(a.w, b.w, __dp4a(a.z, b.z, __dp4a(a.y, b.y, __dp4a(a.x, b.x, 0))));
}
__device__ __forceinline__ int sum16(int4 x) {
    const int one = 0x01010101;
    return __dp4a(x.w, one, __dp4a(x.z, one, __dp4a(x.y, one, __dp4a(x.x, one, 0))));
}
__device__ __forceinline__ int4 and4(int4 a, int m) { return make_int4(a.x & m, a.y & m, a.z & m, a.w & m); }
__device__ __forceinline__ int4 shr4(int4 a, int s) {
    return make_int4((int)((unsigned)a.x >> s), (int)((unsigned)a.y >> s), (int)((unsigned)a.z >> s), (int)((unsigned)a.w >> s));
}
__device__ __forceinline__ int4 or4(int4 a, int4 b) { return make_int4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w); }
__device__ __forceinline__ int4 shl4(int4 a, int s) { return make_int4(a.x << s, a.y << s, a.z << s, a.w << s); }

// ---- i-quants and Q2_K kept in their own format (models whose int8 copy would not fit) ----------
//
// GGUF type codes: Q2_K 10, IQ2_XXS 16, IQ3_XXS 18, IQ1_S 19, IQ4_NL 20 (32-value blocks, taken eight
// at a time as one 144-byte group), IQ3_S 21, IQ2_S 22, IQ4_XS 23, IQ1_M 29. Each 256-value group has
// 8 units of 32 values; iq_vals writes unit u's values (block scale included) as floats, following
// src/quant_more.rs's decoder line for line. Decoding is the cheap part: these rows are read at
// memory speed whichever way the products are formed.
#include "iq_tables.cuh"

__device__ __forceinline__ bool is_iq(int dtype) {
    return dtype == 10 || dtype == 16 || dtype == 18 || dtype == 19 || dtype == 20 || dtype == 21 || dtype == 22 || dtype == 23 || dtype == 29 || dtype == 41;
}
__device__ __forceinline__ float h16(const unsigned char* p) {
    return __half2float(__ushort_as_half((unsigned short)(p[0] | (p[1] << 8))));
}
__device__ __forceinline__ unsigned int u32at(const unsigned char* p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}
__device__ __forceinline__ unsigned int u16at(const unsigned char* p) { return (unsigned int)p[0] | ((unsigned int)p[1] << 8); }
__device__ __forceinline__ float gbyte(unsigned long long g, int j) { return (float)((g >> (8 * j)) & 0xff); }
__device__ __forceinline__ float gsbyte(unsigned long long g, int j) { return (float)(signed char)((g >> (8 * j)) & 0xff); }
__device__ __forceinline__ float sgn(unsigned int s, int j) { return (s >> j) & 1 ? -1.f : 1.f; }

__device__ void iq_vals(int dtype, const unsigned char* b, int u, float* v) {
    if (dtype == 41) {
        // Q1_0 as laid out on the GPU: [d0 d1 | 32 sign bytes], 32 weights per unit u.
        float d = h16(b + 2 * (u >> 2));
        unsigned int s = u32at(b + 4 + 4 * u);
        for (int j = 0; j < 32; ++j) v[j] = (s >> j) & 1 ? d : -d;
        return;
    }
    if (dtype == 20) {
        const unsigned char* q = b + 18 * u;
        float d = h16(q);
        for (int j = 0; j < 16; ++j) {
            v[j] = d * kvalues_iq4nl_d[q[2 + j] & 0xf];
            v[j + 16] = d * kvalues_iq4nl_d[q[2 + j] >> 4];
        }
    } else if (dtype == 23) {
        float d = h16(b);
        unsigned int sh = u16at(b + 2);
        int ls = ((b[4 + u / 2] >> (4 * (u % 2))) & 0xf) | (((sh >> (2 * u)) & 3) << 4);
        float dl = d * (ls - 32);
        const unsigned char* q = b + 8 + 16 * u;
        for (int j = 0; j < 16; ++j) {
            v[j] = dl * kvalues_iq4nl_d[q[j] & 0xf];
            v[j + 16] = dl * kvalues_iq4nl_d[q[j] >> 4];
        }
    } else if (dtype == 16) {
        float d = h16(b);
        unsigned int a0 = u32at(b + 2 + 8 * u), a1 = u32at(b + 6 + 8 * u);
        float db = d * (0.5f + (a1 >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            unsigned long long g = iq2xxs_grid_d[(a0 >> (8 * l)) & 0xff];
            unsigned int sg = ksigns_iq2xs_d[(a1 >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j) v[l * 8 + j] = db * gbyte(g, j) * sgn(sg, j);
        }
    } else if (dtype == 22) {
        float d = h16(b);
        int sc = b[74 + u];
        unsigned int qh = b[66 + u];
        for (int l = 0; l < 4; ++l) {
            float db = d * (0.5f + (l < 2 ? (sc & 0xf) : (sc >> 4))) * 0.25f;
            unsigned long long g = iq2s_grid_d[b[2 + 4 * u + l] | ((qh << (8 - 2 * l)) & 0x300)];
            unsigned int sg = b[34 + 4 * u + l];
            for (int j = 0; j < 8; ++j) v[l * 8 + j] = db * gbyte(g, j) * sgn(sg, j);
        }
    } else if (dtype == 18) {
        float d = h16(b);
        unsigned int aux = u32at(b + 66 + 4 * u);
        float db = d * (0.5f + (aux >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            unsigned int sg = ksigns_iq2xs_d[(aux >> (7 * l)) & 127];
            unsigned int g1 = iq3xxs_grid_d[b[2 + 8 * u + 2 * l]], g2 = iq3xxs_grid_d[b[2 + 8 * u + 2 * l + 1]];
            for (int j = 0; j < 4; ++j) {
                v[l * 8 + j] = db * (float)((g1 >> (8 * j)) & 0xff) * sgn(sg, j);
                v[l * 8 + j + 4] = db * (float)((g2 >> (8 * j)) & 0xff) * sgn(sg, j + 4);
            }
        }
    } else if (dtype == 21) {
        float d = h16(b);
        int sc = b[106 + u / 2];
        float db = d * (1 + 2 * (u % 2 == 0 ? (sc & 0xf) : (sc >> 4)));
        unsigned int hb = b[66 + u];
        for (int l = 0; l < 4; ++l) {
            unsigned int g1 = iq3s_grid_d[b[2 + 8 * u + 2 * l] | ((hb << (8 - 2 * l)) & 256)];
            unsigned int g2 = iq3s_grid_d[b[2 + 8 * u + 2 * l + 1] | ((hb << (7 - 2 * l)) & 256)];
            unsigned int sg = b[74 + 4 * u + l];
            for (int j = 0; j < 4; ++j) {
                v[l * 8 + j] = db * (float)((g1 >> (8 * j)) & 0xff) * sgn(sg, j);
                v[l * 8 + j + 4] = db * (float)((g2 >> (8 * j)) & 0xff) * sgn(sg, j + 4);
            }
        }
    } else if (dtype == 19) {
        float d = h16(b);
        unsigned int qh = u16at(b + 34 + 2 * u);
        float dl = d * (2 * ((qh >> 12) & 7) + 1);
        float delta = qh & 0x8000 ? -0.125f : 0.125f;
        for (int l = 0; l < 4; ++l) {
            unsigned long long g = iq1s_grid_d[b[2 + 4 * u + l] | (((qh >> (3 * l)) & 7) << 8)];
            for (int j = 0; j < 8; ++j) v[l * 8 + j] = dl * (gsbyte(g, j) + delta);
        }
    } else if (dtype == 29) {
        unsigned int sc0 = u16at(b + 48), sc1 = u16at(b + 50), sc2 = u16at(b + 52), sc3 = u16at(b + 54);
        unsigned int dbits = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
        float d = __half2float(__ushort_as_half((unsigned short)dbits));
        unsigned int scs[4] = {sc0, sc1, sc2, sc3};
        unsigned int sv = scs[u / 2] >> (6 * (u % 2));
        float dl0 = d * (2 * (sv & 7) + 1), dl1 = d * (2 * ((sv >> 3) & 7) + 1);
        const unsigned char* qs = b + 4 * u;
        const unsigned char* qh = b + 32 + 2 * u;
        int idx[4] = {qs[0] | ((qh[0] << 8) & 0x700), qs[1] | ((qh[0] << 4) & 0x700), qs[2] | ((qh[1] << 8) & 0x700), qs[3] | ((qh[1] << 4) & 0x700)};
        float dl[4] = {qh[0] & 0x08 ? -0.125f : 0.125f, qh[0] & 0x80 ? -0.125f : 0.125f, qh[1] & 0x08 ? -0.125f : 0.125f, qh[1] & 0x80 ? -0.125f : 0.125f};
        for (int l = 0; l < 4; ++l) {
            unsigned long long g = iq1s_grid_d[idx[l]];
            float sc = l < 2 ? dl0 : dl1;
            for (int j = 0; j < 8; ++j) v[l * 8 + j] = sc * (gsbyte(g, j) + dl[l]);
        }
    } else {
        // Q2_K: unit u = 32 values of half n = u / 4 at shift 2 * (u % 4); a scale and a minimum per 16.
        float d = h16(b + 80), dmin = h16(b + 82);
        int n = u >> 2, k = u & 3;
        const unsigned char* q = b + 16 + 32 * n;
        int c0 = b[8 * n + 2 * k], c1 = b[8 * n + 2 * k + 1];
        for (int l = 0; l < 16; ++l) {
            v[l] = d * (c0 & 0xf) * ((q[l] >> (2 * k)) & 3) - dmin * (c0 >> 4);
            v[16 + l] = d * (c1 & 0xf) * ((q[16 + l] >> (2 * k)) & 3) - dmin * (c1 >> 4);
        }
    }
}

// The same units as integer dot products for decoding: signed int8 words built in registers, four
// values per dp4a against the int8 activations, one float scale per unit (or per 16 values).
// Four sign bits -> 0xff in the bytes whose bit is set.
__device__ __forceinline__ int smask4(unsigned int s) { return (int)(((s & 15u) * 0x00204081u & 0x01010101u) * 0xffu); }
// Four unsigned grid bytes with four sign bits applied.
__device__ __forceinline__ int signed4(unsigned int g, unsigned int s) { int m = smask4(s); return __vsub4((int)g ^ m, m); }
__device__ __forceinline__ int xw(const signed char* x, int k) { return *reinterpret_cast<const int*>(x + 4 * k); }
// IQ4_NL / IQ4_XS: four nibbles (bytes 0-3 of q, low or high half) through the value table.
__device__ __forceinline__ int nl4(unsigned int q, int hi) {
    int sh = hi ? 4 : 0;
    return (int)(unsigned char)kvalues_iq4nl_d[(q >> sh) & 15] | ((int)(unsigned char)kvalues_iq4nl_d[(q >> (sh + 8)) & 15] << 8)
         | ((int)(unsigned char)kvalues_iq4nl_d[(q >> (sh + 16)) & 15] << 16) | ((int)(unsigned char)kvalues_iq4nl_d[(q >> (sh + 24)) & 15] << 24);
}

__device__ float iq_dot(int dtype, const unsigned char* b, int u, const signed char* xq, const float* xd) {
    const signed char* x = xq + u * 32;
    const int one = 0x01010101;
    if (dtype == 41) {
        // Q1_0: sum(+-x) = 2 * sum(x where the bit is set) - sum(x).
        unsigned int s = u32at(b + 4 + 4 * u);
        int on = 0, all = 0;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            unsigned int nib = (s >> (4 * k)) & 0xf;
            int m = (nib & 1 ? 0xff : 0) | (nib & 2 ? 0xff00 : 0) | (nib & 4 ? 0xff0000 : 0) | (nib & 8 ? 0xff000000 : 0);
            int xv = xw(x, k);
            on = __dp4a(xv & m, one, on);
            all = __dp4a(xv, one, all);
        }
        return h16(b + 2 * (u >> 2)) * (float)(2 * on - all) * xd[u];
    }
    if (dtype == 20 || dtype == 23) {
        const unsigned char* q;
        float dl;
        if (dtype == 20) {
            q = b + 18 * u + 2;
            dl = h16(b + 18 * u);
        } else {
            unsigned int sh = u16at(b + 2);
            int ls = ((b[4 + u / 2] >> (4 * (u % 2))) & 0xf) | (((sh >> (2 * u)) & 3) << 4);
            dl = h16(b) * (ls - 32);
            q = b + 8 + 16 * u;
        }
        int acc = 0;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            unsigned int w = u32at(q + 4 * k);
            acc = __dp4a(nl4(w, 0), xw(x, k), acc);
            acc = __dp4a(nl4(w, 1), xw(x, 4 + k), acc);
        }
        return dl * acc * xd[u];
    }
    if (dtype == 16) {
        unsigned int a0 = u32at(b + 2 + 8 * u), a1 = u32at(b + 6 + 8 * u);
        int acc = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            unsigned long long g = iq2xxs_grid_d[(a0 >> (8 * l)) & 0xff];
            unsigned int sg = ksigns_iq2xs_d[(a1 >> (7 * l)) & 127];
            acc = __dp4a(signed4((unsigned int)g, sg), xw(x, 2 * l), acc);
            acc = __dp4a(signed4((unsigned int)(g >> 32), sg >> 4), xw(x, 2 * l + 1), acc);
        }
        return h16(b) * (2 * (a1 >> 28) + 1) * 0.125f * acc * xd[u];
    }
    if (dtype == 22) {
        int sc = b[74 + u];
        unsigned int qh = b[66 + u];
        int a01 = 0, a23 = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            unsigned long long g = iq2s_grid_d[b[2 + 4 * u + l] | ((qh << (8 - 2 * l)) & 0x300)];
            unsigned int sg = b[34 + 4 * u + l];
            int p = __dp4a(signed4((unsigned int)g, sg), xw(x, 2 * l), __dp4a(signed4((unsigned int)(g >> 32), sg >> 4), xw(x, 2 * l + 1), 0));
            if (l < 2) a01 += p; else a23 += p;
        }
        return h16(b) * 0.125f * ((2 * (sc & 0xf) + 1) * a01 + (2 * (sc >> 4) + 1) * a23) * xd[u];
    }
    if (dtype == 18) {
        unsigned int aux = u32at(b + 66 + 4 * u);
        int acc = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            unsigned int sg = ksigns_iq2xs_d[(aux >> (7 * l)) & 127];
            acc = __dp4a(signed4(iq3xxs_grid_d[b[2 + 8 * u + 2 * l]], sg), xw(x, 2 * l), acc);
            acc = __dp4a(signed4(iq3xxs_grid_d[b[2 + 8 * u + 2 * l + 1]], sg >> 4), xw(x, 2 * l + 1), acc);
        }
        return h16(b) * (2 * (aux >> 28) + 1) * 0.25f * acc * xd[u];
    }
    if (dtype == 21) {
        int sc = b[106 + u / 2];
        unsigned int hb = b[66 + u];
        int acc = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            unsigned int sg = b[74 + 4 * u + l];
            acc = __dp4a(signed4(iq3s_grid_d[b[2 + 8 * u + 2 * l] | ((hb << (8 - 2 * l)) & 256)], sg), xw(x, 2 * l), acc);
            acc = __dp4a(signed4(iq3s_grid_d[b[2 + 8 * u + 2 * l + 1] | ((hb << (7 - 2 * l)) & 256)], sg >> 4), xw(x, 2 * l + 1), acc);
        }
        return h16(b) * (1 + 2 * (u % 2 == 0 ? (sc & 0xf) : (sc >> 4))) * acc * xd[u];
    }
    if (dtype == 19) {
        unsigned int qh = u16at(b + 34 + 2 * u);
        int acc = 0, sx = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            unsigned long long g = iq1s_grid_d[b[2 + 4 * u + l] | (((qh >> (3 * l)) & 7) << 8)];
            acc = __dp4a((int)(unsigned int)g, xw(x, 2 * l), acc);
            acc = __dp4a((int)(unsigned int)(g >> 32), xw(x, 2 * l + 1), acc);
            sx = __dp4a(xw(x, 2 * l), one, __dp4a(xw(x, 2 * l + 1), one, sx));
        }
        float delta = qh & 0x8000 ? -0.125f : 0.125f;
        return h16(b) * (2 * ((qh >> 12) & 7) + 1) * (acc + delta * sx) * xd[u];
    }
    if (dtype == 29) {
        unsigned int sc0 = u16at(b + 48), sc1 = u16at(b + 50), sc2 = u16at(b + 52), sc3 = u16at(b + 54);
        unsigned int dbits = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
        float d = __half2float(__ushort_as_half((unsigned short)dbits));
        unsigned int scs[4] = {sc0, sc1, sc2, sc3};
        unsigned int sv = scs[u / 2] >> (6 * (u % 2));
        const unsigned char* qs = b + 4 * u;
        const unsigned char* qh = b + 32 + 2 * u;
        int idx[4] = {qs[0] | ((qh[0] << 8) & 0x700), qs[1] | ((qh[0] << 4) & 0x700), qs[2] | ((qh[1] << 8) & 0x700), qs[3] | ((qh[1] << 4) & 0x700)};
        bool neg[4] = {(qh[0] & 0x08) != 0, (qh[0] & 0x80) != 0, (qh[1] & 0x08) != 0, (qh[1] & 0x80) != 0};
        float part[2] = {0.f, 0.f};
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            unsigned long long g = iq1s_grid_d[idx[l]];
            int a = __dp4a((int)(unsigned int)g, xw(x, 2 * l), __dp4a((int)(unsigned int)(g >> 32), xw(x, 2 * l + 1), 0));
            int sx = __dp4a(xw(x, 2 * l), one, __dp4a(xw(x, 2 * l + 1), one, 0));
            part[l >> 1] += a + (neg[l] ? -0.125f : 0.125f) * sx;
        }
        return d * ((2 * (sv & 7) + 1) * part[0] + (2 * ((sv >> 3) & 7) + 1) * part[1]) * xd[u];
    }
    // Q2_K
    int n = u >> 2, k = u & 3;
    const unsigned char* q = b + 16 + 32 * n;
    int c0 = b[8 * n + 2 * k], c1 = b[8 * n + 2 * k + 1];
    int a0 = 0, a1 = 0, s0 = 0, s1 = 0;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        int v0 = (int)((u32at(q + 4 * w) >> (2 * k)) & 0x03030303u);
        int v1 = (int)((u32at(q + 16 + 4 * w) >> (2 * k)) & 0x03030303u);
        a0 = __dp4a(v0, xw(x, w), a0);
        a1 = __dp4a(v1, xw(x, 4 + w), a1);
        s0 = __dp4a(xw(x, w), one, s0);
        s1 = __dp4a(xw(x, 4 + w), one, s1);
    }
    return (h16(b + 80) * ((c0 & 0xf) * a0 + (c1 & 0xf) * a1) - h16(b + 82) * ((c0 >> 4) * s0 + (c1 >> 4) * s1)) * xd[u];
}

// Unit u of a block: Q4_K/Q5_K have 8 (j = u/2 picks the 32-byte qs chunk, half = u%2 its 16 bytes;
// low nibbles belong to sub-block 2j, high to 2j+1); Q6_K has 16 (half n, group g, 16-value part hh).
__device__ __forceinline__ float unit_dot(int dtype, const unsigned char* blk, int u,
                                          const signed char* xq, const float* xd) {
    if (is_iq(dtype)) return iq_dot(dtype, blk, u, xq, xd);
    if (dtype == 12 || dtype == 13) {
        int j = u >> 1, half = u & 1, sb0 = 2 * j, sb1 = sb0 + 1;
        float d = __half2float(*reinterpret_cast<const __half*>(blk));
        float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
        int4 x0 = *reinterpret_cast<const int4*>(xq + sb0 * 32 + half * 16);
        int4 x1 = *reinterpret_cast<const int4*>(xq + sb1 * 32 + half * 16);
        int4 lo, hi;
        if (dtype == 12) {
            int4 q = *reinterpret_cast<const int4*>(blk + 16 + j * 32 + half * 16);
            lo = and4(q, 0x0F0F0F0F);
            hi = and4(shr4(q, 4), 0x0F0F0F0F);
        } else {
            int4 q = *reinterpret_cast<const int4*>(blk + 48 + j * 32 + half * 16);
            int4 h = *reinterpret_cast<const int4*>(blk + 16 + half * 16);
            lo = or4(and4(q, 0x0F0F0F0F), shl4(and4(shr4(h, sb0), 0x01010101), 4));
            hi = or4(and4(shr4(q, 4), 0x0F0F0F0F), shl4(and4(shr4(h, sb1), 0x01010101), 4));
        }
        int sc0, m0, sc1, m1;
        scale_min_k4(sb0, blk + 4, sc0, m0);
        scale_min_k4(sb1, blk + 4, sc1, m1);
        return xd[sb0] * (d * sc0 * dot16(lo, x0) - dmin * m0 * sum16(x0))
             + xd[sb1] * (d * sc1 * dot16(hi, x1) - dmin * m1 * sum16(x1));
    } else {
        int n = u >> 3, g = (u >> 1) & 3, hh = u & 1;
        float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
        int4 ql = *reinterpret_cast<const int4*>(blk + n * 64 + (g & 1) * 32 + hh * 16);
        int4 qh = *reinterpret_cast<const int4*>(blk + 128 + n * 32 + hh * 16);
        int4 lo = g < 2 ? and4(ql, 0x0F0F0F0F) : and4(shr4(ql, 4), 0x0F0F0F0F);
        int4 q = or4(lo, shl4(and4(shr4(qh, 2 * g), 0x03030303), 4));
        int xo = n * 128 + g * 32 + hh * 16;
        int4 x = *reinterpret_cast<const int4*>(xq + xo);
        int sc = (signed char)blk[192 + n * 8 + 2 * g + hh];
        return d * sc * xd[n * 4 + g] * (float)(dot16(q, x) - 32 * sum16(x));
    }
}

// K-quant rows (Q4_K / Q5_K / Q6_K, GPU layout) -> int8 with one scale per 32 values, the layout the
// tensor-core prompt GEMM reads. One warp per 32-value sub-block, one value per lane; the bit unpacking
// is unit_dot's. Used for prompt batches of weights kept in their own format (models too big for the
// int8 layout), so they get the tensor-core GEMM too; decoding keeps the native-format matvec.
extern "C" __global__ void k_to_q8(const unsigned char* __restrict__ w, int dtype, int bb, int rows, int cols,
                                   signed char* __restrict__ qs, float* __restrict__ ds) {
    // One thread per 32-value sub-block: two 16-byte loads of packed weights, the values in
    // registers, the int8 scale from their maximum, two 16-byte stores.
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int nsb = cols >> 5;
    if (id >= (long long)rows * nsb) return;
    int row = (int)(id / nsb), sb = (int)(id - (long long)row * nsb);
    int b = sb >> 3, j = sb & 7;
    const unsigned char* blk = w + ((size_t)row * (cols >> 8) + b) * bb;
    float v[32];
    if (is_iq(dtype)) {
        iq_vals(dtype, blk, j, v);
    } else if (dtype == 12 || dtype == 13) {
        float d = __half2float(*reinterpret_cast<const __half*>(blk));
        float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
        int sc, m;
        scale_min_k4(j, blk + 4, sc, m);
        float a = d * sc, c = dmin * m;
        const unsigned char* q = blk + (dtype == 12 ? 16 : 48) + (j >> 1) * 32;
        int4 q0 = *reinterpret_cast<const int4*>(q), q1 = *reinterpret_cast<const int4*>(q + 16);
        const unsigned char* qb0 = reinterpret_cast<const unsigned char*>(&q0);
        const unsigned char* qb1 = reinterpret_cast<const unsigned char*>(&q1);
        int sh = (j & 1) * 4;
        if (dtype == 12) {
            #pragma unroll
            for (int l = 0; l < 16; ++l) {
                v[l] = a * ((qb0[l] >> sh) & 15) - c;
                v[16 + l] = a * ((qb1[l] >> sh) & 15) - c;
            }
        } else {
            int4 h0 = *reinterpret_cast<const int4*>(blk + 16), h1 = *reinterpret_cast<const int4*>(blk + 32);
            const unsigned char* hb0 = reinterpret_cast<const unsigned char*>(&h0);
            const unsigned char* hb1 = reinterpret_cast<const unsigned char*>(&h1);
            #pragma unroll
            for (int l = 0; l < 16; ++l) {
                v[l] = a * (((qb0[l] >> sh) & 15) | (((hb0[l] >> j) & 1) << 4)) - c;
                v[16 + l] = a * (((qb1[l] >> sh) & 15) | (((hb1[l] >> j) & 1) << 4)) - c;
            }
        }
    } else {
        int n = j >> 2, g = j & 3;
        float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
        const unsigned char* ql = blk + n * 64 + (g & 1) * 32;
        const unsigned char* qh = blk + 128 + n * 32;
        int4 l0 = *reinterpret_cast<const int4*>(ql), l1 = *reinterpret_cast<const int4*>(ql + 16);
        int4 h0 = *reinterpret_cast<const int4*>(qh), h1 = *reinterpret_cast<const int4*>(qh + 16);
        const unsigned char* lb0 = reinterpret_cast<const unsigned char*>(&l0);
        const unsigned char* lb1 = reinterpret_cast<const unsigned char*>(&l1);
        const unsigned char* hb0 = reinterpret_cast<const unsigned char*>(&h0);
        const unsigned char* hb1 = reinterpret_cast<const unsigned char*>(&h1);
        float s0 = d * (signed char)blk[192 + n * 8 + 2 * g], s1 = d * (signed char)blk[192 + n * 8 + 2 * g + 1];
        int lsh = g < 2 ? 0 : 4, hsh = 2 * g;
        #pragma unroll
        for (int l = 0; l < 16; ++l) {
            v[l] = s0 * ((((lb0[l] >> lsh) & 15) | (((hb0[l] >> hsh) & 3) << 4)) - 32);
            v[16 + l] = s1 * ((((lb1[l] >> lsh) & 15) | (((hb1[l] >> hsh) & 3) << 4)) - 32);
        }
    }
    float amax = 0.f;
    #pragma unroll
    for (int l = 0; l < 32; ++l) amax = fmaxf(amax, fabsf(v[l]));
    float dq = amax / 127.f, id_ = dq > 0.f ? 1.f / dq : 0.f;
    int4 o[2];
    signed char* ob = reinterpret_cast<signed char*>(o);
    #pragma unroll
    for (int l = 0; l < 32; ++l) ob[l] = (signed char)__float2int_rn(v[l] * id_);
    int4* dst = reinterpret_cast<int4*>(qs + (size_t)row * cols + sb * 32);
    dst[0] = o[0];
    dst[1] = o[1];
    ds[(size_t)row * nsb + sb] = dq;
}

// GGUF Q8_0 rows (34-byte blocks: f16 scale, 32 int8) -> the int8 + f32-scale layout the prompt
// GEMM reads. One thread per block.
// Q6_K blocks (210 bytes) widened to the GPU's aligned 224-byte stride, one byte per thread.
extern "C" __global__ void q6k_widen(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[(i / 210) * 224 + i % 210] = src[i];
}

extern "C" __global__ void q80_split(const unsigned char* __restrict__ w, int nblocks,
                                     signed char* __restrict__ qs, float* __restrict__ ds) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const unsigned char* b = w + i * 34;
    ds[i] = __half2float(*reinterpret_cast<const __half*>(b));
    #pragma unroll
    for (int j = 0; j < 32; ++j) qs[i * 32 + j] = (signed char)b[2 + j];
}

// Batched expert matvec over the k chosen experts (gridDim.y = k). sel holds [ids k | weights k |
// cache slots k]; experts not cached (slot < 0) are skipped here and computed on the CPU. Expert j's
// matrix starts at pool + slot*slot_bytes + off; its input at xq + j*xstride; output at y + j*rows.
extern "C" __global__ void moe_matvec(const unsigned char* __restrict__ pool, long long slot_bytes, long long off,
                                      const int* __restrict__ sel, int dtype, int bb,
                                      const signed char* __restrict__ xq, const float* __restrict__ xd, int xstride,
                                      float* __restrict__ y, int rows, int cols) {
    int k = blockIdx.y;
    int slot = sel[2 * gridDim.y + k];
    if (slot < 0) return;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    int nb = cols >> 8;
    int upb = dtype == 14 ? 16 : 8;
    const unsigned char* w = pool + slot * slot_bytes + off + (size_t)row * nb * bb;
    const signed char* x = xq + (size_t)k * xstride;
    const float* d = xd + (size_t)k * (xstride >> 5);
    float sum = 0.f;
    for (int u = lane; u < nb * upb; u += 32) {
        int b = u / upb;
        sum += unit_dot(dtype, w + (size_t)b * bb, u - b * upb, x + b * 256, d + b * 8);
    }
    sum = warp_sum(sum);
    if (lane == 0) y[(size_t)k * rows + row] = sum;
}

// Router: scores for n_expert experts (f32 weights), softmax, top-k, renormalised weights. One block of
// 1024 threads (32 warps); writes k expert ids and weights.
extern "C" __global__ void router_topk(const float* __restrict__ wr, const float* __restrict__ h, int dim,
                                       int n_expert, int k, const int* __restrict__ slot_tab, int* __restrict__ sel) {
    int* ids = sel;
    float* wts = reinterpret_cast<float*>(sel + k);
    __shared__ float sc[512];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int e = warp; e < n_expert; e += 32) {
        const float* r = wr + (size_t)e * dim;
        float s = 0.f;
        for (int i = lane; i < dim; i += 32) s += r[i] * h[i];
        s = warp_sum(s);
        if (lane == 0) sc[e] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = -1e30f;
        for (int e = 0; e < n_expert; ++e) mx = fmaxf(mx, sc[e]);
        for (int e = 0; e < n_expert; ++e) sc[e] = __expf(sc[e] - mx);
        float tot = 0.f;
        for (int t = 0; t < k; ++t) {
            int best = 0;
            float bv = -1.f;
            for (int e = 0; e < n_expert; ++e) if (sc[e] > bv) { bv = sc[e]; best = e; }
            ids[t] = best;
            wts[t] = bv;
            tot += bv;
            sc[best] = -2.f;
        }
        for (int t = 0; t < k; ++t) {
            wts[t] /= tot;
            sel[2 * k + t] = slot_tab[ids[t]];
        }
    }
}

// act_k = silu(gate_k) * up_k for n experts (gate outputs first, then up outputs), quantised per expert.
extern "C" __global__ void moe_silu_quant(const float* __restrict__ gu, signed char* __restrict__ xq,
                                          float* __restrict__ xd, int eff, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i & ~31) >= eff * n) return;
    float g = gu[i], u = gu[eff * n + i];
    quant32(g / (1.f + __expf(-g)) * u, xq, xd, i, threadIdx.x & 31);
}

// x += sum over cached experts of weight * eo[j] + cpu (the CPU experts' weighted sum).
extern "C" __global__ void moe_combine(float* __restrict__ x, const float* __restrict__ eo, const int* __restrict__ sel,
                                       int k, const float* __restrict__ cpu, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const float* w = reinterpret_cast<const float*>(sel + k);
    float s = cpu[i];
    for (int j = 0; j < k; ++j) if (sel[2 * k + j] >= 0) s += w[j] * eo[(size_t)j * dim + i];
    x[i] += s;
}

// Dense matvec on K-quant weights in their stored format (MoE models keep attention and output this
// way to leave more VRAM for experts). One warp per row; `add` accumulates into y.
extern "C" __global__ void matvec_k(const unsigned char* __restrict__ w, int dtype, int bb,
                                    const signed char* __restrict__ xq, const float* __restrict__ xd,
                                    float* __restrict__ y, int rows, int cols, int add) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    int nb = cols >> 8;
    int upb = dtype == 14 ? 16 : 8;
    const unsigned char* wr = w + (size_t)row * nb * bb;
    float sum = 0.f;
    for (int u = lane; u < nb * upb; u += 32) {
        int b = u / upb;
        sum += unit_dot(dtype, wr + (size_t)b * bb, u - b * upb, xq + b * 256, xd + b * 8);
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = add ? y[row] + sum : sum;
}

// ---- Batched MoE (prompt processing) ------------------------------------------------------------

// Y[i][row] (+)= W[row] . X[r_i] for n tokens, K-quant weights. r_i = idx[i] when idx is given (the
// tokens routed to one expert), else i. Output row stride ystride. One warp per weight row; each
// weight unit is loaded once per tile of TT tokens.
extern "C" __global__ void kmatmul(const unsigned char* __restrict__ w, int dtype, int bb,
                                   const signed char* __restrict__ xq, const float* __restrict__ xd,
                                   const int* __restrict__ idx, int n, float* __restrict__ y, int ystride,
                                   int rows, int cols, int add) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    int nb = cols >> 8;
    int upb = dtype == 14 ? 16 : 8;
    const unsigned char* wr = w + (size_t)row * nb * bb;
    for (int t0 = 0; t0 < n; t0 += TT) {
        int tn = min(TT, n - t0);
        float sum[TT];
        #pragma unroll
        for (int k = 0; k < TT; ++k) sum[k] = 0.f;
        for (int u = lane; u < nb * upb; u += 32) {
            int b = u / upb, uu = u - b * upb;
            const unsigned char* blk = wr + (size_t)b * bb;
            #pragma unroll
            for (int k = 0; k < TT; ++k) {
                if (k < tn) {
                    int r = idx ? idx[t0 + k] : t0 + k;
                    sum[k] += unit_dot(dtype, blk, uu, xq + (size_t)r * cols + b * 256, xd + (size_t)r * (cols >> 5) + b * 8);
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < TT; ++k) {
            if (k < tn) {
                float v = warp_sum(sum[k]);
                if (lane == 0) {
                    float* o = y + (size_t)(t0 + k) * ystride + row;
                    *o = add ? *o + v : v;
                }
            }
        }
    }
}

// Router for T tokens (one block each): sel + t*3k = [ids k | weights k | cache slots k].
extern "C" __global__ void router_topk_batch(const float* __restrict__ wr, const float* __restrict__ h, int dim,
                                             int n_expert, int k, const int* __restrict__ slot_tab, int* __restrict__ sel) {
    __shared__ float sc[512];
    const float* ht = h + (size_t)blockIdx.x * dim;
    int* st = sel + (size_t)blockIdx.x * 3 * k;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int e = warp; e < n_expert; e += 32) {
        const float* r = wr + (size_t)e * dim;
        float s = 0.f;
        for (int i = lane; i < dim; i += 32) s += r[i] * ht[i];
        s = warp_sum(s);
        if (lane == 0) sc[e] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = -1e30f;
        for (int e = 0; e < n_expert; ++e) mx = fmaxf(mx, sc[e]);
        for (int e = 0; e < n_expert; ++e) sc[e] = __expf(sc[e] - mx);
        float tot = 0.f;
        float* wts = reinterpret_cast<float*>(st + k);
        for (int t = 0; t < k; ++t) {
            int best = 0;
            float bv = -1.f;
            for (int e = 0; e < n_expert; ++e) if (sc[e] > bv) { bv = sc[e]; best = e; }
            st[t] = best;
            wts[t] = bv;
            tot += bv;
            sc[best] = -2.f;
        }
        for (int t = 0; t < k; ++t) {
            wts[t] /= tot;
            st[2 * k + t] = slot_tab[st[t]];
        }
    }
}

// x[idx[i]] += w[i] * eo[i] for the n tokens one expert processed.
extern "C" __global__ void scatter_add(float* __restrict__ x, const float* __restrict__ eo, const int* __restrict__ idx,
                                       const float* __restrict__ w, int n, int dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (d >= dim || i >= n) return;
    x[(size_t)idx[i] * dim + d] += w[i] * eo[(size_t)i * dim + d];
}

extern "C" __global__ void add_inplace(float* __restrict__ x, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// Tiled causal attention for prompt processing (head_dim 128). A block takes one KV head and a tile
// of tokens, i.e. every query head sharing that KV head (GQA) for those tokens: 64 query vectors, 8
// per warp. K and V are read from the cache 32 positions at a time into shared memory and reused by
// all 64 queries, instead of each query streaming the whole cache from global memory.
#define AQ 8      // query vectors per warp
#define AK 32     // keys per tile
#define AHD 128
extern "C" __global__ void attn_prefill_tiled(const float* __restrict__ qkv, const __half* __restrict__ kc,
                                              const __half* __restrict__ vc, float* __restrict__ out,
                                              int n_head, int n_kv, const int* __restrict__ posp, int group,
                                              float scale, int T, int qt) {
    __shared__ __half ks[AK][AHD + 2];
    __shared__ __half vs[AK][AHD];
    __shared__ __half qs[8][AQ][AHD];
    int kvh = blockIdx.x, t0 = blockIdx.y * qt;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int pos0 = *posp;
    int row = n_head * AHD + 2 * n_kv * AHD;
    int kvd = n_kv * AHD;
    // This warp's query vectors: index qi -> token t0 + qi / group, head kvh*group + qi % group.
    int tok[AQ];
    bool live[AQ];
    for (int j = 0; j < AQ; ++j) {
        int qi = warp * AQ + j;
        int t = t0 + qi / group;
        live[j] = qi < qt * group && t < T;
        tok[j] = t;
        int h = kvh * group + qi % group;
        for (int d = lane; d < AHD; d += 32)
            qs[warp][j][d] = __float2half(live[j] ? qkv[(size_t)t * row + h * AHD + d] * scale : 0.f);
    }
    float m[AQ], l[AQ], acc[AQ][4];
    for (int j = 0; j < AQ; ++j) { m[j] = -1e30f; l[j] = 0.f; for (int c = 0; c < 4; ++c) acc[j][c] = 0.f; }
    int t_last = min(T, t0 + qt) - 1;
    int nkeys = pos0 + t_last + 1;
    for (int k0 = 0; k0 < nkeys; k0 += AK) {
        __syncthreads();
        for (int i = threadIdx.x; i < AK * AHD; i += blockDim.x) {
            int kk = i / AHD, d = i - kk * AHD;
            int p = k0 + kk;
            __half kv = __float2half(0.f), vv = __float2half(0.f);
            if (p < nkeys) {
                kv = kc[(size_t)p * kvd + kvh * AHD + d];
                vv = vc[(size_t)p * kvd + kvh * AHD + d];
            }
            ks[kk][d] = kv;
            vs[kk][d] = vv;
        }
        __syncthreads();
        // Lane = key k0 + lane: scores for this warp's queries.
        int p = k0 + lane;
        float sc[AQ];
        for (int j = 0; j < AQ; ++j) sc[j] = 0.f;
        for (int d = 0; d < AHD; d += 2) {
            float2 kf = __half22float2(*reinterpret_cast<const __half2*>(&ks[lane][d]));
            #pragma unroll
            for (int j = 0; j < AQ; ++j) {
                float2 qf = __half22float2(*reinterpret_cast<const __half2*>(&qs[warp][j][d]));
                sc[j] += qf.x * kf.x + qf.y * kf.y;
            }
        }
        #pragma unroll
        for (int j = 0; j < AQ; ++j) {
            bool ok = live[j] && p <= pos0 + tok[j];
            float s = ok ? sc[j] : -1e30f;
            float mx = warp_max(s);
            float mn = fmaxf(m[j], mx);
            float pr = ok ? __expf(s - mn) : 0.f;
            float c = __expf(m[j] - mn);
            l[j] = l[j] * c + warp_sum(pr);
            for (int cc = 0; cc < 4; ++cc) acc[j][cc] *= c;
            m[j] = mn;
            sc[j] = pr;
        }
        // Accumulate V: lane owns dims lane*4 .. lane*4+3.
        for (int kk = 0; kk < AK; ++kk) {
            float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&vs[kk][lane * 4]));
            float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&vs[kk][lane * 4 + 2]));
            #pragma unroll
            for (int j = 0; j < AQ; ++j) {
                float pj = __shfl_sync(0xffffffff, sc[j], kk);
                acc[j][0] += pj * v01.x;
                acc[j][1] += pj * v01.y;
                acc[j][2] += pj * v23.x;
                acc[j][3] += pj * v23.y;
            }
        }
    }
    for (int j = 0; j < AQ; ++j) {
        if (!live[j]) continue;
        int qi = warp * AQ + j;
        int h = kvh * group + qi % group;
        float inv = l[j] > 0.f ? 1.f / l[j] : 0.f;
        float* o = out + (size_t)tok[j] * n_head * AHD + h * AHD + lane * 4;
        o[0] = acc[j][0] * inv;
        o[1] = acc[j][1] * inv;
        o[2] = acc[j][2] * inv;
        o[3] = acc[j][3] * inv;
    }
}

// ---- Qwen3.5 / Qwen3.8 hybrid layers (Gated DeltaNet + gated attention) ------------------------
// Rows of the fused input projection, per token: [ q|k|v (ch) | z (nv*dv) | alpha (nv) | beta (nv) ].

// Causal depthwise convolution over the last kc inputs of each channel, then SiLU, for T tokens in
// order. `hist` keeps the kc-1 previous inputs per channel (oldest first) across calls. One thread
// per channel; kc <= 9.
extern "C" __global__ void delta_conv(float* __restrict__ row, int ld, const float* __restrict__ w,
                                      float* __restrict__ hist, int ch, int kc, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float h[8];
    for (int j = 0; j < kc - 1; j++) h[j] = hist[(size_t)j * ch + c];
    const float* wc = w + (size_t)c * kc;
    for (int t = 0; t < T; t++) {
        float x = row[(size_t)t * ld + c];
        float y = wc[kc - 1] * x;
        for (int j = 0; j < kc - 1; j++) y += wc[j] * h[j];
        for (int j = 0; j + 1 < kc - 1; j++) h[j] = h[j + 1];
        if (kc > 1) h[kc - 2] = x;
        row[(size_t)t * ld + c] = y / (1.f + __expf(-y));
    }
    for (int j = 0; j < kc - 1; j++) hist[(size_t)j * ch + c] = h[j];
}

// L2-normalises each q and k head (2*nk heads of dk); q heads also take `qscale`. Grid (2*nk, T),
// block dk threads.
extern "C" __global__ void delta_l2(float* __restrict__ row, int ld, int nk, int dk, float qscale) {
    __shared__ float red[32];
    int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    float* v = row + (size_t)t * ld + (size_t)h * dk;
    float x = v[i];
    float s = rsqrtf(block_sum(x * x, red) + 1e-6f);
    if (h < nk) s *= qscale;
    v[i] = x * s;
}

// The delta rule for T tokens in order. Grid (nv, dv/32), block 128: 32 state columns of one value
// head, each column's dk rows split over 4 thread groups; the slice stays in shared memory for all
// T tokens and is written back at the end. Per token: S *= decay; kv = S^T k; S += k (beta (v - kv))^T;
// o = S^T q. Value head h reads key head h % nk. Shared memory: (dk + 4) * 32 floats.
extern "C" __global__ void delta_rule(const float* __restrict__ row, int ld, int ch, int nk, int dk, int nv, int dv,
                                      const float* __restrict__ a, const float* __restrict__ dtb,
                                      float* __restrict__ state, float* __restrict__ out, int ldo, int T) {
    extern __shared__ float sm[];
    float* S = sm;
    float* part = sm + dk * 32;
    int h = blockIdx.x, tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    int j = blockIdx.y * 32 + tx;
    int kh = h % nk;
    float* st = state + (size_t)h * dk * dv;
    for (int i = ty; i < dk; i += 4) S[i * 32 + tx] = st[(size_t)i * dv + j];
    __syncthreads();
    for (int t = 0; t < T; t++) {
        const float* r = row + (size_t)t * ld;
        const float* q = r + kh * dk;
        const float* k = r + (nk + kh) * dk;
        float al = r[ch + nv * dv + h], be = r[ch + nv * dv + nv + h];
        float x = al + dtb[h];
        float sp = x > 20.f ? x : log1pf(__expf(x));
        float decay = __expf(a[h] * sp);
        float b = 1.f / (1.f + __expf(-be));
        float kv = 0.f;
        for (int i = ty; i < dk; i += 4) {
            float s = S[i * 32 + tx] * decay;
            S[i * 32 + tx] = s;
            kv += s * k[i];
        }
        part[ty * 32 + tx] = kv;
        __syncthreads();
        kv = part[tx] + part[32 + tx] + part[64 + tx] + part[96 + tx];
        float delta = (r[2 * nk * dk + h * dv + j] - kv) * b;
        __syncthreads();
        float o = 0.f;
        for (int i = ty; i < dk; i += 4) {
            float s = S[i * 32 + tx] + k[i] * delta;
            S[i * 32 + tx] = s;
            o += s * q[i];
        }
        part[ty * 32 + tx] = o;
        __syncthreads();
        if (ty == 0) out[(size_t)t * ldo + h * dv + j] = part[tx] + part[32 + tx] + part[64 + tx] + part[96 + tx];
        __syncthreads();
    }
    for (int i = ty; i < dk; i += 4) st[(size_t)i * dv + j] = S[i * 32 + tx];
}

// Gated RMSNorm of each value head's output: o = rmsnorm(o) * w * silu(z). Grid (nv, T), block dv.
extern "C" __global__ void delta_gnorm(float* __restrict__ out, int ldo, const float* __restrict__ row, int ld, int zoff,
                                       const float* __restrict__ w, int dv, float eps) {
    __shared__ float red[32];
    int h = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
    float* o = out + (size_t)t * ldo + (size_t)h * dv;
    float x = o[j];
    float r = rsqrtf(block_sum(x * x, red) / dv + eps);
    float z = row[(size_t)t * ld + zoff + h * dv + j];
    o[j] = x * r * w[j] * (z / (1.f + __expf(-z)));
}

// Gated attention input: rows [q_h | gate_h per head (2*qd) | k (kvd) | v (kvd)] become rows
// [q (qd) | k | v] in `qkv` and the gates in `gate` (T x qd). Grid (n_head + 2*n_kv, T), block hd.
extern "C" __global__ void unpack_qgate(const float* __restrict__ src, float* __restrict__ qkv, float* __restrict__ gate,
                                        int n_head, int n_kv, int hd) {
    int b = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    int qd = n_head * hd, kvd = n_kv * hd;
    const float* s = src + (size_t)t * (2 * qd + 2 * kvd);
    float* d = qkv + (size_t)t * (qd + 2 * kvd);
    if (b < n_head) {
        d[b * hd + i] = s[2 * b * hd + i];
        gate[(size_t)t * qd + b * hd + i] = s[(2 * b + 1) * hd + i];
    } else {
        int k = b - n_head;
        d[qd + k * hd + i] = s[2 * qd + k * hd + i];
    }
}

// att *= sigmoid(gate), n values.
extern "C" __global__ void gate_mul(float* __restrict__ att, const float* __restrict__ gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) att[i] *= 1.f / (1.f + __expf(-gate[i]));
}

// x[t] += g_t * so[t], g_t = sigmoid(gi . h[t]) (1 when gi is null): a MoE layer's shared expert
// output, scaled by its gate on the layer's normed input. One block per token.
extern "C" __global__ void shared_gate_add(float* __restrict__ x, const float* __restrict__ so, const float* __restrict__ gi,
                                           const float* __restrict__ h, int dim) {
    __shared__ float red[32];
    __shared__ float gate;
    const size_t o = (size_t)blockIdx.x * dim;
    float sum = 0.f;
    if (gi) for (int i = threadIdx.x; i < dim; i += blockDim.x) sum += gi[i] * h[o + i];
    sum = warp_sum(sum);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tsum = 0.f;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) tsum += red[w];
        gate = gi ? 1.f / (1.f + __expf(-tsum)) : 1.f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < dim; i += blockDim.x) x[o + i] += gate * so[o + i];
}

// Decode matvec for Q1_0 weights (GPU layout: 256-weight blocks of [d0 d1 | 32 sign bytes]).
// One-bit weights are 1/8 the bytes of the int8 activation, so re-reading the activation for every
// row (one warp per row) made the kernel bound by activation traffic: here each warp computes four
// rows from one load of each activation unit. A unit's signs become +-1 bytes arithmetically
// (bits spread with a multiply, then 2b - 1) for one dp4a per four weights.
#define Q1_ROWS 4
extern "C" __global__ void matvec_q1(const unsigned char* __restrict__ w, const signed char* __restrict__ xq,
                                     const float* __restrict__ xd, float* __restrict__ y, int rows, int cols, int add) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * Q1_ROWS;
    if (r0 >= rows) return;
    int units = cols >> 5;
    size_t rb = (size_t)(cols >> 8) * 36;
    float sum[Q1_ROWS];
    #pragma unroll
    for (int r = 0; r < Q1_ROWS; ++r) sum[r] = 0.f;
    for (int u = lane; u < units; u += 32) {
        const int4* xp = reinterpret_cast<const int4*>(xq + (size_t)u * 32);
        int4 xa = xp[0], xb = xp[1];
        int xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        float sx = xd[u];
        int blk = u >> 3, uu = u & 7;
        #pragma unroll
        for (int r = 0; r < Q1_ROWS; ++r) {
            if (r0 + r >= rows) break;
            const unsigned char* b = w + (size_t)(r0 + r) * rb + (size_t)blk * 36;
            unsigned int s = *reinterpret_cast<const unsigned int*>(b + 4 + 4 * uu);
            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                unsigned int bits = (((s >> (4 * k)) & 0xfu) * 0x00204081u) & 0x01010101u;
                acc = __dp4a((int)__vsub4(bits << 1, 0x01010101u), xv[k], acc);
            }
            sum[r] += h16(b + 2 * (uu >> 2)) * sx * (float)acc;
        }
    }
    #pragma unroll
    for (int r = 0; r < Q1_ROWS; ++r) {
        float v = warp_sum(sum[r]);
        if (lane == 0 && r0 + r < rows) y[r0 + r] = add ? y[r0 + r] + v : v;
    }
}
