// Cerebro Engine CUDA kernels. Compiled to PTX by build.rs with nvcc and embedded in the binary;
// the driver JIT-compiles the PTX for the user's GPU (Turing sm_75 and newer).

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// One warp per output row; each lane reads 16 int8 weights per iteration (one int4 load).
extern "C" __global__ void matvec_q8(const signed char* __restrict__ qs, const float* __restrict__ ds,
                                     const signed char* __restrict__ xq, const float* __restrict__ xd,
                                     float* __restrict__ y, int rows, int cols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    int nb = cols >> 5;
    const signed char* w = qs + (size_t)row * cols;
    const float* d = ds + (size_t)row * nb;
    float sum = 0.f;
    for (int i = lane * 16; i < cols; i += 512) {
        int4 wv = *reinterpret_cast<const int4*>(w + i);
        int4 xv = *reinterpret_cast<const int4*>(xq + i);
        int acc = 0;
        acc = __dp4a(wv.x, xv.x, acc);
        acc = __dp4a(wv.y, xv.y, acc);
        acc = __dp4a(wv.z, xv.z, acc);
        acc = __dp4a(wv.w, xv.w, acc);
        int b = i >> 5;
        sum += (float)acc * d[b] * xd[b];
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

// One warp per block of 32 activations: scale = max|x| / 127.
extern "C" __global__ void quantize_q8(const float* __restrict__ x, signed char* __restrict__ xq,
                                       float* __restrict__ xd, int n) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int i = warp * 32 + lane;
    if (warp * 32 >= n) return;
    float v = x[i];
    float amax = warp_max(fabsf(v));
    float d = amax / 127.f;
    float id = d > 0.f ? 1.f / d : 0.f;
    xq[i] = (signed char)__float2int_rn(v * id);
    if (lane == 0) xd[warp] = d;
}

// y = x * w / rms(x); one block.
extern "C" __global__ void rmsnorm(const float* __restrict__ x, const float* __restrict__ w,
                                   float* __restrict__ y, int n, float eps) {
    __shared__ float red[32];
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        v = warp_sum(v);
        if (threadIdx.x == 0) red[0] = rsqrtf(v / n + eps);
    }
    __syncthreads();
    float s = red[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * s * w[i];
}

// Per-head RMSNorm (optional) and NeoX RoPE. One block per head, head_dim/2 threads.
// If `dst` is not null the result is written there (the KV cache) instead of in place.
extern "C" __global__ void norm_rope(float* __restrict__ v, const float* __restrict__ nw, int has_norm,
                                     int hd, int pos, float base, float eps, float* __restrict__ dst) {
    __shared__ float red[32];
    float* h = v + blockIdx.x * hd;
    int i = threadIdx.x, half = hd >> 1;
    float a = h[i], b = h[i + half];
    if (has_norm) {
        float ss = warp_sum(a * a + b * b);
        if ((i & 31) == 0) red[i >> 5] = ss;
        __syncthreads();
        if (i < 32) {
            float t = i < ((half + 31) >> 5) ? red[i] : 0.f;
            t = warp_sum(t);
            if (i == 0) red[0] = rsqrtf(t / hd + eps);
        }
        __syncthreads();
        float s = red[0];
        a = a * s * nw[i];
        b = b * s * nw[i + half];
    }
    float inv = powf(base, -2.f * i / hd);
    float sn, cs;
    sincosf(pos * inv, &sn, &cs);
    float* o = dst ? dst + blockIdx.x * hd : h;
    o[i] = a * cs - b * sn;
    o[i + half] = a * sn + b * cs;
}

extern "C" __global__ void copy_f32(const float* __restrict__ src, float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// Decode attention for one query token over `len` cached positions. One block per head, 8 warps;
// each warp takes every 8th position with an online softmax; head_dim up to 256 (8 values per lane).
extern "C" __global__ void attention(const float* __restrict__ q, const float* __restrict__ kc,
                                     const float* __restrict__ vc, float* __restrict__ out,
                                     int n_kv, int hd, int len, int group, float scale) {
    __shared__ float sm[8], sl[8];
    __shared__ float sacc[8][256];
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

extern "C" __global__ void silu_mul(float* __restrict__ g, const float* __restrict__ u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = g[i]; g[i] = v / (1.f + __expf(-v)) * u[i]; }
}

extern "C" __global__ void add_inplace(float* __restrict__ x, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
