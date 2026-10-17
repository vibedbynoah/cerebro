// Tensor-core kernels (Ampere sm_80 and newer). Compiled to a separate PTX module by build.rs and
// loaded only when the GPU has compute capability 8.0+; older GPUs keep the dp4a kernels.

// Y[t][row] (+)= W[row] . X[t] for T tokens on int8 tensor cores. W and X are int8 with one f32 scale
// per 32 values, so one m16n8k32 MMA covers exactly one quant block: its int32 result is scaled by
// the row's and the token's block scales and added to f32 accumulators.
// Block tile: 64 rows x 64 tokens, 128 k per shared-memory stage (4 quant blocks), double-buffered
// with cp.async. 4 warps in a 2x2 grid, each computing 32 rows x 32 tokens (2 x 4 MMA tiles).
// Requires cols % 32 == 0; rows and T are arbitrary.

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
    const int r0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
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
                    acc[i][j][0] += (float)c0 * sw[i][0] * sx[j][0];
                    acc[i][j][1] += (float)c1 * sw[i][0] * sx[j][1];
                    acc[i][j][2] += (float)c2 * sw[i][1] * sx[j][0];
                    acc[i][j][3] += (float)c3 * sw[i][1] * sx[j][1];
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
