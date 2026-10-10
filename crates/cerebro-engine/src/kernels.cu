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

// After the fused QKV matmul: per-head RMSNorm (optional) + NeoX RoPE for Q (in place) and K (written
// into the K cache at *pos), and V copied into the V cache. Grid: n_head + n_kv blocks of hd/2.
extern "C" __global__ void qkv_post(float* __restrict__ qkv, const float* __restrict__ qn, const float* __restrict__ kn,
                                    int has_norm, float* __restrict__ kc, float* __restrict__ vc,
                                    int n_head, int n_kv, int hd, const int* __restrict__ posp, float base, float eps) {
    __shared__ float red[32];
    int pos = *posp;
    int h = blockIdx.x, i = threadIdx.x, half = hd >> 1;
    int kvd = n_kv * hd;
    bool is_k = h >= n_head;
    float* src = is_k ? qkv + n_head * hd + (h - n_head) * hd : qkv + h * hd;
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
    float* o = is_k ? kc + (size_t)pos * kvd + (h - n_head) * hd : src;
    o[i] = a * cs - b * sn;
    o[i + half] = a * sn + b * cs;
    if (is_k) {
        // V for this KV head goes to the cache too.
        const float* v = qkv + n_head * hd + kvd + (h - n_head) * hd;
        float* vo = vc + (size_t)pos * kvd + (h - n_head) * hd;
        vo[i] = v[i];
        vo[i + half] = v[i + half];
    }
}

// Decode attention over positions 0..*pos. One block per head, 8 warps, online softmax per warp,
// combined at the end; head_dim up to 256.
extern "C" __global__ void attention(const float* __restrict__ q, const float* __restrict__ kc,
                                     const float* __restrict__ vc, float* __restrict__ out,
                                     int n_kv, int hd, const int* __restrict__ posp, int group, float scale) {
    __shared__ float sm[8], sl[8];
    __shared__ float sacc[8][256];
    int len = *posp + 1;
    int head = blockIdx.x, kvh = head / group;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int per = (hd + 31) >> 5;
    const float* qh = q + head * hd;
    float qv[8], acc[8];
    for (int j = 0; j < per; ++j) { int d = lane + j * 32; qv[j] = d < hd ? qh[d] : 0.f; acc[j] = 0.f; }
    float m = -1e30f, l = 0.f;
    int stride = n_kv * hd;
    for (int t = warp; t < len; t += 8) {
        const float* kt = kc + (size_t)t * stride + kvh * hd;
        float s = 0.f;
        for (int j = 0; j < per; ++j) { int d = lane + j * 32; if (d < hd) s += qv[j] * kt[d]; }
        s = warp_sum(s) * scale;
        float mn = fmaxf(m, s);
        float c = __expf(m - mn), p = __expf(s - mn);
        l = l * c + p;
        const float* vt = vc + (size_t)t * stride + kvh * hd;
        for (int j = 0; j < per; ++j) { int d = lane + j * 32; if (d < hd) acc[j] = acc[j] * c + p * vt[d]; }
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
                out[head * hd + d] = o / L;
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
