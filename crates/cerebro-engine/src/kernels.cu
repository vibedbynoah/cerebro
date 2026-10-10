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
extern "C" __global__ void qkv_post(float* __restrict__ qkv, const float* __restrict__ qn, const float* __restrict__ kn,
                                    int has_norm, __half* __restrict__ kc, __half* __restrict__ vc,
                                    int n_head, int n_kv, int hd, const int* __restrict__ posp, float base, float eps) {
    __shared__ float red[32];
    int t = blockIdx.y;
    int pos = *posp + t;
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
    float inv = powf(base, -2.f * i / hd);
    float sn, cs;
    sincosf(pos * inv, &sn, &cs);
    float ra = a * cs - b * sn, rb = a * sn + b * cs;
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
                                               signed char* __restrict__ xq, float* __restrict__ xd, int n, float eps) {
    __shared__ float red[32];
    const float* xr = x + (size_t)blockIdx.x * n;
    signed char* q = xq + (size_t)blockIdx.x * n;
    float* d = xd + (size_t)blockIdx.x * (n >> 5);
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    float s = rsqrtf(block_sum(ss, red) / n + eps);
    for (int base = 0; base < n; base += blockDim.x) {
        int i = base + threadIdx.x;
        if (i < n) quant32(xr[i] * s * w[i], q, d, i, threadIdx.x & 31);
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

// Unit u of a block: Q4_K/Q5_K have 8 (j = u/2 picks the 32-byte qs chunk, half = u%2 its 16 bytes;
// low nibbles belong to sub-block 2j, high to 2j+1); Q6_K has 16 (half n, group g, 16-value part hh).
__device__ __forceinline__ float unit_dot(int dtype, const unsigned char* blk, int u,
                                          const signed char* xq, const float* xd) {
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
