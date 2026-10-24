//! Quantised weight formats and the row kernels that use them. Weights stay quantised; a row is
//! dequantised block by block into registers/stack and dotted with an f32 activation. The block
//! layouts are the GGUF ones (Q8_0, Q4_K, Q5_K, Q6_K) so standard files load without conversion.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    // Formats read through quant_more (dequantise-and-dot on the CPU, repacked for the GPU).
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q2K,
    Q3K,
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ3XXS,
    IQ3S,
    IQ1S,
    IQ1M,
    IQ4NL,
    IQ4XS,
    /// PrismML's 1-bit "Q1_0_g128" (Bonsai): a sign bit per weight (1 = +d, 0 = -d), one f16 scale
    /// per 128 weights; 18 bytes per block.
    Q1_0,
}

impl DType {
    pub fn from_ggml(t: u32) -> Option<Self> {
        Some(match t {
            0 => DType::F32,
            1 => DType::F16,
            8 => DType::Q8_0,
            12 => DType::Q4K,
            13 => DType::Q5K,
            14 => DType::Q6K,
            30 => DType::BF16,
            2 => DType::Q4_0,
            3 => DType::Q4_1,
            6 => DType::Q5_0,
            7 => DType::Q5_1,
            10 => DType::Q2K,
            11 => DType::Q3K,
            16 => DType::IQ2XXS,
            17 => DType::IQ2XS,
            18 => DType::IQ3XXS,
            19 => DType::IQ1S,
            20 => DType::IQ4NL,
            21 => DType::IQ3S,
            22 => DType::IQ2S,
            23 => DType::IQ4XS,
            29 => DType::IQ1M,
            41 => DType::Q1_0,
            _ => return None,
        })
    }
    pub fn block_elems(self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::BF16 => 1,
            DType::Q8_0 | DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::IQ4NL => 32,
            DType::Q1_0 => 128,
            _ => 256,
        }
    }
    pub fn block_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q8_0 => 34,
            DType::Q4K => 144,
            DType::Q5K => 176,
            DType::Q6K => 210,
            DType::Q4_0 => 18,
            DType::Q4_1 => 20,
            DType::Q5_0 => 22,
            DType::Q5_1 => 24,
            DType::Q2K => 84,
            DType::Q3K => 110,
            DType::IQ2XXS => 66,
            DType::IQ2XS => 74,
            DType::IQ2S => 82,
            DType::IQ3XXS => 98,
            DType::IQ3S => 110,
            DType::IQ1S => 50,
            DType::IQ1M => 56,
            DType::IQ4NL => 18,
            DType::IQ4XS => 136,
            DType::Q1_0 => 18,
        }
    }
    pub fn row_bytes(self, cols: usize) -> usize {
        cols / self.block_elems() * self.block_bytes()
    }
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[inline]
fn rd16(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Dequantises one block into `out` (length = block_elems).
pub fn dequant_block(t: DType, b: &[u8], out: &mut [f32]) {
    match t {
        DType::F32 => out[0] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        DType::F16 => out[0] = rd16(b, 0),
        DType::BF16 => out[0] = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16),
        DType::Q8_0 => {
            let d = rd16(b, 0);
            for (i, o) in out.iter_mut().enumerate().take(32) {
                *o = d * (b[2 + i] as i8) as f32;
            }
        }
        DType::Q4K => {
            let d = rd16(b, 0);
            let dmin = rd16(b, 2);
            let scales = &b[4..16];
            let qs = &b[16..144];
            let mut is = 0;
            for j in 0..4 {
                let (sc1, m1) = scale_min_k4(is, scales);
                let (sc2, m2) = scale_min_k4(is + 1, scales);
                let (d1, m1) = (d * sc1 as f32, dmin * m1 as f32);
                let (d2, m2) = (d * sc2 as f32, dmin * m2 as f32);
                let q = &qs[j * 32..j * 32 + 32];
                let o = &mut out[j * 64..j * 64 + 64];
                for l in 0..32 {
                    o[l] = d1 * (q[l] & 0xF) as f32 - m1;
                    o[l + 32] = d2 * (q[l] >> 4) as f32 - m2;
                }
                is += 2;
            }
        }
        DType::Q5K => {
            let d = rd16(b, 0);
            let dmin = rd16(b, 2);
            let scales = &b[4..16];
            let qh = &b[16..48];
            let qs = &b[48..176];
            let (mut u1, mut u2) = (1u8, 2u8);
            let mut is = 0;
            for j in 0..4 {
                let (sc1, m1) = scale_min_k4(is, scales);
                let (sc2, m2) = scale_min_k4(is + 1, scales);
                let (d1, m1) = (d * sc1 as f32, dmin * m1 as f32);
                let (d2, m2) = (d * sc2 as f32, dmin * m2 as f32);
                let q = &qs[j * 32..j * 32 + 32];
                let o = &mut out[j * 64..j * 64 + 64];
                for l in 0..32 {
                    o[l] = d1 * ((q[l] & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 }) as f32 - m1;
                    o[l + 32] = d2 * ((q[l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 }) as f32 - m2;
                }
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        DType::Q6K => {
            let ql = &b[0..128];
            let qh = &b[128..192];
            let sc = &b[192..208];
            let d = rd16(b, 208);
            for n in 0..2 {
                let ql = &ql[n * 64..];
                let qh = &qh[n * 32..];
                let sc = &sc[n * 8..];
                let o = &mut out[n * 128..n * 128 + 128];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    o[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                    o[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                    o[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                    o[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                }
            }
        }
        DType::Q1_0 => {
            let d = rd16(b, 0);
            for (j, o) in out.iter_mut().enumerate().take(128) {
                *o = if (b[2 + j / 8] >> (j % 8)) & 1 != 0 { d } else { -d };
            }
        }
        _ => crate::quant_more::dequant_block(t, b, out),
    }
}

/// Dequantises a whole row of `cols` elements.
pub fn dequant_row(t: DType, row: &[u8], out: &mut [f32]) {
    let be = t.block_elems();
    let bb = t.block_bytes();
    for (i, chunk) in out.chunks_mut(be).enumerate() {
        dequant_block(t, &row[i * bb..(i + 1) * bb], chunk);
    }
}

#[inline]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() % 16 == 0 && std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: FMA (and AVX) support checked at run time; lengths are equal multiples of 16.
            return unsafe { dot_f32_fma(a, b) };
        }
    }
    // Eight independent accumulators so the compiler vectorises and pipelines the loop.
    let mut acc = [0f32; 8];
    let chunks = a.len() / 8;
    for i in 0..chunks {
        for k in 0..8 {
            acc[k] += a[i * 8 + k] * b[i * 8 + k];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let n = a.len().min(b.len());
    let mut i = 0;
    while i < n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)), _mm256_loadu_ps(b.as_ptr().add(i + 8)), s1);
        i += 16;
    }
    hsum_ps(_mm256_add_ps(s0, s1))
}

fn has_f16c() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// f32 -> IEEE half bits, round to nearest even (the conversion F16C does).
pub(crate) fn to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    sign | v as u16
}

/// dst = src as halves (the CPU conversation cache's storage).
pub(crate) fn store_f16(dst: &mut [u16], src: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_f16c() && src.len() % 8 == 0 {
            // SAFETY: F16C checked at run time; lengths are equal multiples of 8.
            unsafe { store_f16_c(dst, src) };
            return;
        }
    }
    for (o, &v) in dst.iter_mut().zip(src) {
        *o = to_f16(v);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn store_f16_c(dst: &mut [u16], src: &[f32]) {
    use std::arch::x86_64::*;
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i < n {
        let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(src.as_ptr().add(i)));
        _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), h);
        i += 8;
    }
}

/// q . k with k stored as halves.
#[inline]
pub(crate) fn dot_f16(q: &[f32], k: &[u16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_f16c() && q.len() % 16 == 0 {
            // SAFETY: F16C and FMA checked at run time; lengths are equal multiples of 16.
            return unsafe { dot_f16_c(q, k) };
        }
    }
    q.iter().zip(k).map(|(a, &b)| a * f16_to_f32(b)).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f16_c(q: &[f32], k: &[u16]) -> f32 {
    use std::arch::x86_64::*;
    let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let n = q.len().min(k.len());
    let mut i = 0;
    while i < n {
        let k0 = _mm256_cvtph_ps(_mm_loadu_si128(k.as_ptr().add(i).cast()));
        let k1 = _mm256_cvtph_ps(_mm_loadu_si128(k.as_ptr().add(i + 8).cast()));
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q.as_ptr().add(i)), k0, s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(q.as_ptr().add(i + 8)), k1, s1);
        i += 16;
    }
    hsum_ps(_mm256_add_ps(s0, s1))
}

/// Attention scores over cached keys: out[i] = q . k[i * stride .. + q.len()] for i < out.len().
/// Four keys share each load of q and one final reduction (a dot per key, with its own horizontal
/// sum, measured at ~11 GB/s on the i9-12900H: overhead, not memory).
pub(crate) fn scores_f16(q: &[f32], k: &[u16], stride: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_f16c() && q.len() % 8 == 0 {
            // SAFETY: F16C and FMA checked at run time; every key row lies inside `k` (asserted).
            assert!(out.is_empty() || (out.len() - 1) * stride + q.len() <= k.len());
            unsafe { scores_f16_c(q, k, stride, out) };
            return;
        }
    }
    for (i, o) in out.iter_mut().enumerate() {
        *o = dot_f16(q, &k[i * stride..i * stride + q.len()]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn scores_f16_c(q: &[f32], k: &[u16], stride: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let hd = q.len();
    let n = out.len();
    let kp = k.as_ptr();
    let ld = |row: usize, c: usize| _mm256_cvtph_ps(_mm_loadu_si128(kp.add(row * stride + c).cast()));
    let mut i = 0;
    while i + 4 <= n {
        let (mut a0, mut a1, mut a2, mut a3) = (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut c = 0;
        while c < hd {
            let qv = _mm256_loadu_ps(q.as_ptr().add(c));
            a0 = _mm256_fmadd_ps(qv, ld(i, c), a0);
            a1 = _mm256_fmadd_ps(qv, ld(i + 1, c), a1);
            a2 = _mm256_fmadd_ps(qv, ld(i + 2, c), a2);
            a3 = _mm256_fmadd_ps(qv, ld(i + 3, c), a3);
            c += 8;
        }
        // Reduce the four accumulators into four lanes at once.
        let s01 = _mm256_hadd_ps(a0, a1);
        let s23 = _mm256_hadd_ps(a2, a3);
        let s = _mm256_hadd_ps(s01, s23);
        let r = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        _mm_storeu_ps(out.as_mut_ptr().add(i), r);
        i += 4;
    }
    while i < n {
        let mut a = _mm256_setzero_ps();
        let mut c = 0;
        while c < hd {
            a = _mm256_fmadd_ps(_mm256_loadu_ps(q.as_ptr().add(c)), ld(i, c), a);
            c += 8;
        }
        out[i] = hsum_ps(a);
        i += 1;
    }
}

/// out += sum_i p[i] * v[i * stride .. + out.len()]: the attention-weighted values. The output
/// stays in registers, 64 values at a time, across every position.
pub(crate) fn weighted_f16(p: &[f32], v: &[u16], stride: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_f16c() && out.len() % 64 == 0 {
            // SAFETY: F16C and FMA checked at run time; every value row lies inside `v` (asserted).
            assert!(p.is_empty() || (p.len() - 1) * stride + out.len() <= v.len());
            unsafe { weighted_f16_c(p, v, stride, out) };
            return;
        }
    }
    let hd = out.len();
    for (i, &w) in p.iter().enumerate() {
        axpy_f16(out, w, &v[i * stride..i * stride + hd]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn weighted_f16_c(p: &[f32], v: &[u16], stride: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let vp = v.as_ptr();
    let mut b = 0;
    while b < out.len() {
        let o = out.as_mut_ptr().add(b);
        let mut acc = [_mm256_setzero_ps(); 8];
        for (r, a) in acc.iter_mut().enumerate() {
            *a = _mm256_loadu_ps(o.add(r * 8));
        }
        for (i, &w) in p.iter().enumerate() {
            let wv = _mm256_set1_ps(w);
            let row = vp.add(i * stride + b);
            for (r, a) in acc.iter_mut().enumerate() {
                *a = _mm256_fmadd_ps(wv, _mm256_cvtph_ps(_mm_loadu_si128(row.add(r * 8).cast())), *a);
            }
        }
        for (r, a) in acc.iter().enumerate() {
            _mm256_storeu_ps(o.add(r * 8), *a);
        }
        b += 64;
    }
}

/// y += a * x with x stored as halves.
#[inline]
pub(crate) fn axpy_f16(y: &mut [f32], a: f32, x: &[u16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_f16c() && y.len() % 8 == 0 {
            // SAFETY: F16C and FMA checked at run time; lengths are equal multiples of 8.
            unsafe { axpy_f16_c(y, a, x) };
            return;
        }
    }
    for (o, &v) in y.iter_mut().zip(x) {
        *o += a * f16_to_f32(v);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn axpy_f16_c(y: &mut [f32], a: f32, x: &[u16]) {
    use std::arch::x86_64::*;
    let av = _mm256_set1_ps(a);
    let n = y.len().min(x.len());
    let mut i = 0;
    while i < n {
        let p = y.as_mut_ptr().add(i);
        let xv = _mm256_cvtph_ps(_mm_loadu_si128(x.as_ptr().add(i).cast()));
        _mm256_storeu_ps(p, _mm256_fmadd_ps(av, xv, _mm256_loadu_ps(p)));
        i += 8;
    }
}

/// Dot product of one quantised row with an f32 vector.
pub fn dot_row(t: DType, row: &[u8], x: &[f32]) -> f32 {
    let be = t.block_elems();
    let bb = t.block_bytes();
    match t {
        DType::F32 => {
            let mut s = 0.0;
            for (i, xv) in x.iter().enumerate() {
                s += f32::from_le_bytes([row[i * 4], row[i * 4 + 1], row[i * 4 + 2], row[i * 4 + 3]]) * xv;
            }
            s
        }
        DType::Q8_0 => {
            // Fused: integer weights times f32 activations, scaled once per block.
            let mut s = 0.0;
            for (bi, xb) in x.chunks_exact(32).enumerate() {
                let b = &row[bi * 34..bi * 34 + 34];
                let d = rd16(b, 0);
                let mut acc = [0f32; 8];
                for i in 0..4 {
                    for k in 0..8 {
                        acc[k] += (b[2 + i * 8 + k] as i8) as f32 * xb[i * 8 + k];
                    }
                }
                s += d * acc.iter().sum::<f32>();
            }
            s
        }
        _ => {
            let mut buf = [0f32; 256];
            let mut s = 0.0;
            for (bi, xb) in x.chunks_exact(be).enumerate() {
                dequant_block(t, &row[bi * bb..(bi + 1) * bb], &mut buf[..be]);
                s += dot_f32(&buf[..be], xb);
            }
            s
        }
    }
}

/// Round half away from zero to i8 (|x| <= 127.5). f32::round is a library call on the baseline
/// x86-64 target (no SSE4.1), which made activation quantisation a measurable share of each token.
#[inline(always)]
fn round_i8(x: f32) -> i8 {
    (x + 0.5f32.copysign(x)) as i8
}

/// e^x to about 2e-7 relative error, branch-free so loops over it vectorise (f32::exp is a library
/// call per element; softmax, SiLU and routing call it hundreds of thousands of times per token).
#[inline(always)]
pub(crate) fn exp_fast(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E + 0.5f32.copysign(x)) as i32 as f32;
    let r = x - n * 0.693_145_75 - n * 1.428_606_8e-6;
    // Taylor/minimax polynomial for e^r, |r| <= ln2/2.
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (0.166_666_67 + r * (0.041_666_668 + r * (0.008_333_334 + r * 0.001_388_889)))));
    f32::from_bits(((n as i32 + 127) as u32) << 23) * p
}

/// Activations quantised to 8 bits in blocks of 32 (one f32 scale per block), so Q8_0 weights can be
/// multiplied with integer instructions. Built once per step and shared by every matrix that reads it.
pub struct QAct {
    pub d: Vec<f32>,
    pub q: Vec<i8>,
    /// Sum of the int8 values of each block (for the K-quant minimum terms).
    pub s: Vec<i32>,
    /// The same activations quantised with one scale per 256 (the K-quant block size), so a K-quant
    /// block's integer products can be summed across its sub-blocks and converted to float once.
    pub kq: Vec<i8>,
    pub kd: Vec<f32>,
    /// Sums of kq per 16 values (minimum terms and the Q6_K offset).
    pub ks: Vec<i32>,
    /// `q + 128` as unsigned bytes: the unsigned operand VNNI multiplies natively, so the prefill
    /// tiles need no per-pair sign fix-up (the 128 offset is taken back with each weight block's sum).
    pub qu: Vec<u8>,
}

impl QAct {
    pub fn new(len: usize) -> Self {
        let k = if len % 256 == 0 { len } else { 0 };
        Self { d: vec![0.0; len / 32], q: vec![0; len], s: vec![0; len / 32], kq: vec![0; k], kd: vec![0.0; k / 256], ks: vec![0; k / 16], qu: vec![128; len] }
    }

    pub fn quantize(&mut self, x: &[f32]) {
        for (bi, xb) in x.chunks_exact(32).enumerate() {
            let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            self.d[bi] = d;
            let mut sum = 0i32;
            for (i, v) in xb.iter().enumerate() {
                let q = round_i8(v * id);
                self.q[bi * 32 + i] = q;
                self.qu[bi * 32 + i] = (q as i16 + 128) as u8;
                sum += q as i32;
            }
            self.s[bi] = sum;
        }
        if !self.kq.is_empty() {
            for (bi, xb) in x.chunks_exact(256).enumerate() {
                let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                self.kd[bi] = d;
                for (gi, g) in xb.chunks_exact(16).enumerate() {
                    let mut sum = 0i32;
                    for (i, v) in g.iter().enumerate() {
                        let q = round_i8(v * id);
                        self.kq[bi * 256 + gi * 16 + i] = q;
                        sum += q as i32;
                    }
                    self.ks[bi * 16 + gi] = sum;
                }
            }
        }
    }
}

/// Q8_0 weight row times 8-bit activations.
pub fn dot_q8_0_q8(row: &[u8], a: &QAct) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avxvnni") {
            // SAFETY: AVX-VNNI (which implies AVX2) was checked at run time just above.
            return unsafe { dot_q8_0_q8_vnni(row, a) };
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 support was checked at run time just above.
            return unsafe { dot_q8_0_q8_avx2(row, a) };
        }
    }
    let mut s = 0.0;
    for (bi, d) in a.d.iter().enumerate() {
        let b = &row[bi * 34..bi * 34 + 34];
        let mut acc = 0i32;
        for i in 0..32 {
            acc += (b[2 + i] as i8) as i32 * a.q[bi * 32 + i] as i32;
        }
        s += rd16(b, 0) * d * acc as f32;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_0_q8_avx2(row: &[u8], a: &QAct) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let ones = _mm256_set1_epi16(1);
    for (bi, d) in a.d.iter().enumerate() {
        let b = row.as_ptr().add(bi * 34);
        let wd = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
        let w = _mm256_loadu_si256(b.add(2) as *const __m256i);
        let x = _mm256_loadu_si256(a.q.as_ptr().add(bi * 32) as *const __m256i);
        // maddubs needs one unsigned operand: take the absolute weights and move their signs onto x.
        let aw = _mm256_sign_epi8(w, w);
        let sx = _mm256_sign_epi8(x, w);
        let p16 = _mm256_maddubs_epi16(aw, sx);
        let p32 = _mm256_madd_epi16(p16, ones);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(wd * d), acc);
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let s = _mm_add_ps(hi, _mm256_castps256_ps128(acc));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// IQ4_NL row (32-value blocks: f16 scale, 16 bytes of 4-bit indices into a fixed table of 16 int8
/// values) times the per-32 int8 activations. Unsloth Dynamic files use IQ4_NL for expert down
/// projections and large lookup tables.
pub fn dot_iq4nl_q8(row: &[u8], a: &QAct) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 support was checked at run time just above.
            return unsafe { dot_iq4nl_q8_avx2(row, a) };
        }
    }
    let mut s = 0.0;
    for (bi, d) in a.d.iter().enumerate() {
        let b = &row[bi * 18..bi * 18 + 18];
        let mut acc = 0i32;
        for j in 0..16 {
            acc += crate::iq_tables::KVALUES_IQ4NL[(b[2 + j] & 0xf) as usize] as i32 * a.q[bi * 32 + j] as i32;
            acc += crate::iq_tables::KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as i32 * a.q[bi * 32 + j + 16] as i32;
        }
        s += rd16(b, 0) * d * acc as f32;
    }
    s
}

/// The 16 table values come out of one byte shuffle per nibble half; the rest is the Q8_0 kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq4nl_q8_avx2(row: &[u8], a: &QAct) -> f32 {
    use std::arch::x86_64::*;
    let values = _mm_loadu_si128(crate::iq_tables::KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4 = _mm_set1_epi8(0x0f);
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();
    for (bi, d) in a.d.iter().enumerate() {
        let b = row.as_ptr().add(bi * 18);
        let wd = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
        let q4 = _mm_loadu_si128(b.add(2) as *const __m128i);
        let lo = _mm_shuffle_epi8(values, _mm_and_si128(q4, m4));
        let hi = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q4, 4), m4));
        let w = _mm256_set_m128i(hi, lo);
        let x = _mm256_loadu_si256(a.q.as_ptr().add(bi * 32) as *const __m256i);
        let aw = _mm256_sign_epi8(w, w);
        let sx = _mm256_sign_epi8(x, w);
        let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(wd * d), acc);
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let s = _mm_add_ps(hi, _mm256_castps256_ps128(acc));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// IQ2_S row (256-value blocks: f16 scale, 32 grid-index low bytes, 32 sign bytes, 8 bytes of index
/// high bits, 8 bytes of two 4-bit sub-scales) times the per-256 int8 activations (`a.kq`). Each
/// 8-value group is a row of a 1024-entry grid of unsigned magnitudes with its own sign byte;
/// a 16-value half-block's weight is d * (2s + 1) / 8. None when the activations have no per-256
/// form (length not a multiple of 256).
pub fn dot_iq2s_k8(row: &[u8], a: &QAct) -> Option<f32> {
    if a.kq.is_empty() {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 support was checked at run time just above.
            return Some(unsafe { dot_iq2s_k8_avx2(row, a) });
        }
    }
    let mut s = 0.0;
    for (bi, kd) in a.kd.iter().enumerate() {
        let b = &row[bi * 82..bi * 82 + 82];
        let mut acc = 0i32;
        for ib in 0..8 {
            let sc = b[74 + ib];
            for l in 0..4 {
                let idx = b[2 + 4 * ib + l] as usize | (((b[66 + ib] as usize) << (8 - 2 * l)) & 0x300);
                let g = crate::iq_tables::IQ2S_GRID[idx].to_le_bytes();
                let sg = b[34 + 4 * ib + l];
                let mut part = 0i32;
                for j in 0..8 {
                    let v = g[j] as i32 * if sg >> j & 1 == 1 { -1 } else { 1 };
                    part += v * a.kq[bi * 256 + ib * 32 + l * 8 + j] as i32;
                }
                let scale = if l < 2 { sc & 0xf } else { sc >> 4 } as i32;
                acc += part * (2 * scale + 1);
            }
        }
        s += rd16(b, 0) * kd * acc as f32 * 0.125;
    }
    Some(s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq2s_k8_avx2(row: &[u8], a: &QAct) -> f32 {
    use std::arch::x86_64::*;
    // Byte i of a lane takes sign byte (lane * 2 + i / 8); then bit (i % 8) of it.
    let spread = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    let bits = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
    let one8 = _mm256_set1_epi8(1);
    let mut accf = 0f32;
    for (bi, kd) in a.kd.iter().enumerate() {
        let b = row.as_ptr().add(bi * 82);
        let d = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
        let qs = b.add(2);
        let sgn = b.add(34);
        let qh = b.add(66);
        let sc = b.add(74);
        let mut acc = _mm256_setzero_si256();
        for ib in 0..8 {
            let h = *qh.add(ib) as usize;
            let g = |l: usize| crate::iq_tables::IQ2S_GRID[*qs.add(4 * ib + l) as usize | ((h << (8 - 2 * l)) & 0x300)] as i64;
            let w = _mm256_set_epi64x(g(3), g(2), g(1), g(0));
            // The four sign bytes, each spread over its eight values: 0xff where the bit is set.
            let s4 = std::ptr::read_unaligned(sgn.add(4 * ib) as *const u32);
            let sv = _mm256_shuffle_epi8(_mm256_set1_epi32(s4 as i32), spread);
            let neg = _mm256_cmpeq_epi8(_mm256_and_si256(sv, bits), bits);
            let x = _mm256_loadu_si256(a.kq.as_ptr().add(bi * 256 + ib * 32) as *const __m256i);
            // Weights are unsigned magnitudes: put the signs on the activations.
            let sx = _mm256_sign_epi8(x, _mm256_or_si256(neg, one8));
            let p16 = _mm256_maddubs_epi16(w, sx);
            let s = *sc.add(ib);
            let scales = _mm256_set_m128i(_mm_set1_epi16(2 * (s >> 4) as i16 + 1), _mm_set1_epi16(2 * (s & 0xf) as i16 + 1));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, scales));
        }
        let hi = _mm256_extracti128_si256(acc, 1);
        let s = _mm_add_epi32(hi, _mm256_castsi256_si128(acc));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        accf += d * kd * _mm_cvtsi128_si32(s) as f32 * 0.125;
    }
    accf
}

/// Q8_0 row times four tokens' activations at once: each weight block is loaded and sign-split once
/// and reused for the four tokens (batched prefill). Same per-token result as `dot_q8_0_q8`.
pub fn dot_q8_0_q8_x4(row: &[u8], a: [&QAct; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avxvnni") {
            // SAFETY: AVX-VNNI (which implies AVX2) was checked at run time just above.
            return unsafe { dot_q8_0_q8_x4_vnni(row, a) };
        }
    }
    [dot_q8_0_q8(row, a[0]), dot_q8_0_q8(row, a[1]), dot_q8_0_q8(row, a[2]), dot_q8_0_q8(row, a[3])]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn dot_q8_0_q8_x4_vnni(row: &[u8], a: [&QAct; 4]) -> [f32; 4] {
    use std::arch::x86_64::*;
    let mut acc = [_mm256_setzero_ps(); 4];
    for bi in 0..a[0].d.len() {
        let b = row.as_ptr().add(bi * 34);
        let wd = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
        let w = _mm256_loadu_si256(b.add(2) as *const __m256i);
        let aw = _mm256_sign_epi8(w, w);
        for k in 0..4 {
            let x = _mm256_loadu_si256(a[k].q.as_ptr().add(bi * 32) as *const __m256i);
            let p32 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, _mm256_sign_epi8(x, w));
            acc[k] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(wd * a[k].d[bi]), acc[k]);
        }
    }
    let mut out = [0f32; 4];
    for k in 0..4 {
        let hi = _mm256_extractf128_ps(acc[k], 1);
        let s = _mm_add_ps(hi, _mm256_castps256_ps128(acc[k]));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        out[k] = _mm_cvtss_f32(s);
    }
    out
}

/// Bytes per (8-row group, 32-column block) in the interleaved Q8_0 layout: the eight rows' f16
/// scales, then for each of the block's eight 4-byte chunks the eight rows' 4 bytes, each weight +
/// 128. Exactly the file's size: decode reads weights at memory speed, so every byte counts (f32
/// scales, 6% more bytes, measured 6% slower writing).
pub const Q8X8_BLOCK: usize = 16 + 256;
const Q8X8_Q: usize = 16;

/// Q8_0 rows (`rows` a multiple of 8) re-laid for the processor: eight rows interleaved per block so
/// one VNNI instruction yields eight rows' partial sums in its eight lanes (no horizontal sums), the
/// weights offset to unsigned (the activations stay signed; their block sums take the offset back)
/// and the eight scales together for one F16C conversion. 272 bytes per 8 x 32 weights, as in the file.
pub fn repack_q8_0_x8(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let nb = cols / 32;
    let rb = nb * 34;
    let mut out = vec![0u8; rows / 8 * nb * Q8X8_BLOCK];
    for g in 0..rows / 8 {
        for b in 0..nb {
            let o = &mut out[(g * nb + b) * Q8X8_BLOCK..(g * nb + b + 1) * Q8X8_BLOCK];
            for r in 0..8 {
                let blk = &src[(g * 8 + r) * rb + b * 34..];
                o[r * 2..r * 2 + 2].copy_from_slice(&blk[0..2]);
                for k in 0..8 {
                    for j in 0..4 {
                        o[Q8X8_Q + k * 32 + r * 4 + j] = (blk[2 + k * 4 + j] as i8 as i16 + 128) as u8;
                    }
                }
            }
        }
    }
    out
}

/// One 8-row group of an interleaved Q8_0 matrix times one activation vector: the eight rows' dots.
pub fn gemv_q8_0_x8(packed: &[u8], nb: usize, g: usize, a: &QAct) -> [f32; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avxvnni") && has_f16c() {
            // SAFETY: AVX-VNNI (implies AVX2) and F16C checked at run time; the group lies inside `packed`.
            assert!((g + 1) * nb * Q8X8_BLOCK <= packed.len());
            return unsafe { gemv_q8_0_x8_vnni(packed, nb, g, a) };
        }
    }
    let mut out = [0f32; 8];
    for b in 0..nb {
        let o = &packed[(g * nb + b) * Q8X8_BLOCK..];
        for (r, v) in out.iter_mut().enumerate() {
            let d = f16_to_f32(u16::from_le_bytes([o[r * 2], o[r * 2 + 1]]));
            let mut s = 0i32;
            for k in 0..8 {
                for j in 0..4 {
                    s += (o[Q8X8_Q + k * 32 + r * 4 + j] as i32 - 128) * a.q[b * 32 + k * 4 + j] as i32;
                }
            }
            *v += d * a.d[b] * s as f32;
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni,f16c")]
unsafe fn gemv_q8_0_x8_vnni(packed: &[u8], nb: usize, g: usize, a: &QAct) -> [f32; 8] {
    use std::arch::x86_64::*;
    let base = packed.as_ptr().add(g * nb * Q8X8_BLOCK);
    let xq = a.q.as_ptr();
    let mut acc = _mm256_setzero_ps();
    for b in 0..nb {
        let o = base.add(b * Q8X8_BLOCK);
        // Two independent chains: one VNNI accumulator would make each instruction wait for the last.
        let (mut d0, mut d1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for k in (0..8).step_by(2) {
            let x0 = _mm256_set1_epi32((xq.add(b * 32 + k * 4) as *const i32).read_unaligned());
            let x1 = _mm256_set1_epi32((xq.add(b * 32 + k * 4 + 4) as *const i32).read_unaligned());
            d0 = _mm256_dpbusd_avx_epi32(d0, _mm256_loadu_si256(o.add(Q8X8_Q + k * 32) as *const __m256i), x0);
            d1 = _mm256_dpbusd_avx_epi32(d1, _mm256_loadu_si256(o.add(Q8X8_Q + 32 + k * 32) as *const __m256i), x1);
        }
        // (w + 128) . x = w . x + 128 * sum(x).
        let dot = _mm256_sub_epi32(_mm256_add_epi32(d0, d1), _mm256_set1_epi32(128 * a.s[b]));
        let sc = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(o as *const __m128i)), _mm256_set1_ps(a.d[b]));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), sc, acc);
    }
    let mut out = [0f32; 8];
    _mm256_storeu_ps(out.as_mut_ptr(), acc);
    out
}

/// One 8-row group times four tokens: each block's weights are loaded once for the four.
pub fn gemm_q8_0_x8_4(packed: &[u8], nb: usize, g: usize, a: [&QAct; 4]) -> [[f32; 8]; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avxvnni") && has_f16c() {
            // SAFETY: as gemv_q8_0_x8.
            assert!((g + 1) * nb * Q8X8_BLOCK <= packed.len());
            return unsafe { gemm_q8_0_x8_4_vnni(packed, nb, g, a) };
        }
    }
    a.map(|qa| gemv_q8_0_x8(packed, nb, g, qa))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni,f16c")]
unsafe fn gemm_q8_0_x8_4_vnni(packed: &[u8], nb: usize, g: usize, a: [&QAct; 4]) -> [[f32; 8]; 4] {
    use std::arch::x86_64::*;
    let base = packed.as_ptr().add(g * nb * Q8X8_BLOCK);
    let mut acc = [_mm256_setzero_ps(); 4];
    for b in 0..nb {
        let o = base.add(b * Q8X8_BLOCK);
        let w: [__m256i; 8] = std::array::from_fn(|k| _mm256_loadu_si256(o.add(Q8X8_Q + k * 32) as *const __m256i));
        let ws = _mm256_cvtph_ps(_mm_loadu_si128(o as *const __m128i));
        for t in 0..4 {
            let xq = a[t].q.as_ptr().add(b * 32);
            let mut dot = _mm256_setzero_si256();
            for (k, wk) in w.iter().enumerate() {
                dot = _mm256_dpbusd_avx_epi32(dot, *wk, _mm256_set1_epi32((xq.add(k * 4) as *const i32).read_unaligned()));
            }
            let dot = _mm256_sub_epi32(dot, _mm256_set1_epi32(128 * a[t].s[b]));
            acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), _mm256_mul_ps(ws, _mm256_set1_ps(a[t].d[b])), acc[t]);
        }
    }
    let mut out = [[0f32; 8]; 4];
    for t in 0..4 {
        _mm256_storeu_ps(out[t].as_mut_ptr(), acc[t]);
    }
    out
}

/// Bytes per (8-row group, 256-column block) in the interleaved Q4_K layout: d and dmin (f16) for
/// the eight rows, the eight sub-blocks' 6-bit scales and minimums decoded (sub-block-major, eight
/// rows each), then for each nibble pair j (sub-blocks 2j, 2j+1) and 4-byte chunk k the eight rows'
/// 4 bytes of nibbles exactly as stored (low nibble: sub-block 2j, high: 2j + 1).
pub const Q4KX8_BLOCK: usize = 16 + 16 + 64 + 64 + 1024;
const Q4KX8_SC: usize = 32;
const Q4KX8_M: usize = 96;
const Q4KX8_Q: usize = 160;

/// Q4_K rows (`rows` a multiple of 8) re-laid for the processor: one VNNI instruction takes the
/// unsigned nibbles of eight rows against four signed activations, giving eight rows' partial
/// sums in its lanes. 1184 bytes per 8 x 256 weights against 1152 in the file.
pub fn repack_q4k_x8(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let nb = cols / 256;
    let rb = nb * 144;
    let mut out = vec![0u8; rows / 8 * nb * Q4KX8_BLOCK];
    for g in 0..rows / 8 {
        for b in 0..nb {
            let o = &mut out[(g * nb + b) * Q4KX8_BLOCK..(g * nb + b + 1) * Q4KX8_BLOCK];
            for r in 0..8 {
                let blk = &src[(g * 8 + r) * rb + b * 144..(g * 8 + r) * rb + (b + 1) * 144];
                o[r * 2..r * 2 + 2].copy_from_slice(&blk[0..2]);
                o[16 + r * 2..16 + r * 2 + 2].copy_from_slice(&blk[2..4]);
                for s in 0..8 {
                    let (sc, m) = scale_min_k4(s, &blk[4..16]);
                    o[Q4KX8_SC + s * 8 + r] = sc;
                    o[Q4KX8_M + s * 8 + r] = m;
                }
                for j in 0..4 {
                    for k in 0..8 {
                        let dst = Q4KX8_Q + (j * 8 + k) * 32 + r * 4;
                        o[dst..dst + 4].copy_from_slice(&blk[16 + j * 32 + k * 4..16 + j * 32 + k * 4 + 4]);
                    }
                }
            }
        }
    }
    out
}

/// One 8-row group of an interleaved Q4_K matrix times one activation vector (`QAct::kq`).
pub fn gemv_q4k_x8(packed: &[u8], nb: usize, g: usize, a: &QAct) -> [f32; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avxvnni") && has_f16c() && !a.kq.is_empty() {
            // SAFETY: AVX-VNNI (implies AVX2) checked at run time; the group lies inside `packed`.
            assert!((g + 1) * nb * Q4KX8_BLOCK <= packed.len() && a.kq.len() >= nb * 256);
            return unsafe { gemm_q4k_x8_vnni::<1>(packed, nb, g, [a])[0] };
        }
    }
    let mut out = [0f32; 8];
    for b in 0..nb {
        let o = &packed[(g * nb + b) * Q4KX8_BLOCK..];
        let f = |i: usize| f16_to_f32(u16::from_le_bytes([o[i], o[i + 1]]));
        for (r, v) in out.iter_mut().enumerate() {
            let (mut acc, mut msum) = (0i32, 0i32);
            for s in 0..8 {
                let (j, hi) = (s / 2, s % 2 == 1);
                let mut dot = 0i32;
                for k in 0..8 {
                    for i in 0..4 {
                        let byte = o[Q4KX8_Q + (j * 8 + k) * 32 + r * 4 + i];
                        let q = if hi { byte >> 4 } else { byte & 15 } as i32;
                        dot += q * a.kq[b * 256 + j * 64 + if hi { 32 } else { 0 } + k * 4 + i] as i32;
                    }
                }
                acc += o[Q4KX8_SC + s * 8 + r] as i32 * dot;
                msum += o[Q4KX8_M + s * 8 + r] as i32 * (a.ks[b * 16 + 2 * s] + a.ks[b * 16 + 2 * s + 1]);
            }
            *v += a.kd[b] * (f(r * 2) * acc as f32 - f(16 + r * 2) * msum as f32);
        }
    }
    out
}

/// One 8-row group times four tokens: each block's nibbles are loaded and split once for the four.
pub fn gemm_q4k_x8_4(packed: &[u8], nb: usize, g: usize, a: [&QAct; 4]) -> [[f32; 8]; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avxvnni") && has_f16c() && a.iter().all(|q| !q.kq.is_empty()) {
            // SAFETY: as gemv_q4k_x8.
            assert!((g + 1) * nb * Q4KX8_BLOCK <= packed.len() && a.iter().all(|q| q.kq.len() >= nb * 256));
            return unsafe { gemm_q4k_x8_vnni::<4>(packed, nb, g, a) };
        }
    }
    a.map(|qa| gemv_q4k_x8(packed, nb, g, qa))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni,f16c")]
unsafe fn gemm_q4k_x8_vnni<const T: usize>(packed: &[u8], nb: usize, g: usize, a: [&QAct; T]) -> [[f32; 8]; T] {
    use std::arch::x86_64::*;
    let base = packed.as_ptr().add(g * nb * Q4KX8_BLOCK);
    let m4 = _mm256_set1_epi8(0xF);
    let mut facc = [_mm256_setzero_ps(); T];
    for b in 0..nb {
        let o = base.add(b * Q4KX8_BLOCK);
        let mut acc = [_mm256_setzero_si256(); T];
        let mut msum = [_mm256_setzero_si256(); T];
        for j in 0..4 {
            let mut dl = [_mm256_setzero_si256(); T];
            let mut dh = [_mm256_setzero_si256(); T];
            for k in 0..8 {
                let v = _mm256_loadu_si256(o.add(Q4KX8_Q + (j * 8 + k) * 32) as *const __m256i);
                let lo = _mm256_and_si256(v, m4);
                let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
                for t in 0..T {
                    let xq = a[t].kq.as_ptr().add(b * 256 + j * 64 + k * 4);
                    dl[t] = _mm256_dpbusd_avx_epi32(dl[t], lo, _mm256_set1_epi32((xq as *const i32).read_unaligned()));
                    dh[t] = _mm256_dpbusd_avx_epi32(dh[t], hi, _mm256_set1_epi32((xq.add(32) as *const i32).read_unaligned()));
                }
            }
            let s0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(o.add(Q4KX8_SC + 2 * j * 8) as *const __m128i));
            let s1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(o.add(Q4KX8_SC + (2 * j + 1) * 8) as *const __m128i));
            let n0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(o.add(Q4KX8_M + 2 * j * 8) as *const __m128i));
            let n1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(o.add(Q4KX8_M + (2 * j + 1) * 8) as *const __m128i));
            for t in 0..T {
                acc[t] = _mm256_add_epi32(acc[t], _mm256_add_epi32(_mm256_mullo_epi32(dl[t], s0), _mm256_mullo_epi32(dh[t], s1)));
                let ks = a[t].ks.as_ptr().add(b * 16 + 4 * j);
                let x0 = _mm256_set1_epi32(*ks + *ks.add(1));
                let x1 = _mm256_set1_epi32(*ks.add(2) + *ks.add(3));
                msum[t] = _mm256_add_epi32(msum[t], _mm256_add_epi32(_mm256_mullo_epi32(n0, x0), _mm256_mullo_epi32(n1, x1)));
            }
        }
        let d = _mm256_cvtph_ps(_mm_loadu_si128(o as *const __m128i));
        let dmin = _mm256_cvtph_ps(_mm_loadu_si128(o.add(16) as *const __m128i));
        for t in 0..T {
            let kd = _mm256_set1_ps(*a[t].kd.get_unchecked(b));
            let v = _mm256_fmsub_ps(d, _mm256_cvtepi32_ps(acc[t]), _mm256_mul_ps(dmin, _mm256_cvtepi32_ps(msum[t])));
            facc[t] = _mm256_fmadd_ps(v, kd, facc[t]);
        }
    }
    let mut out = [[0f32; 8]; T];
    for t in 0..T {
        _mm256_storeu_ps(out[t].as_mut_ptr(), facc[t]);
    }
    out
}

/// A prefill tile: two Q8_0 rows times four tokens' activations, out[r][t]. Every loaded block
/// serves several products, the activations enter VNNI unsigned (`QAct::qu`; each weight block's
/// sum, shared by the four tokens, takes the offset back), and per block the float steps are those
/// of `dot_q8_0_q8_x4` in the same order, so results are bit-identical to it. None without
/// AVX-VNNI.
pub fn gemm_q8_0_2x4(rows: [&[u8]; 2], a: [&QAct; 4]) -> Option<[[f32; 4]; 2]> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avxvnni") {
            // SAFETY: AVX-VNNI (which implies AVX2) was checked at run time just above.
            return Some(unsafe { gemm_q8_0_2x4_vnni(rows, a) });
        }
    }
    let _ = (rows, a);
    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn gemm_q8_0_2x4_vnni(rows: [&[u8]; 2], a: [&QAct; 4]) -> [[f32; 4]; 2] {
    use std::arch::x86_64::*;
    let mut acc = [[_mm256_setzero_ps(); 4]; 2];
    let ones = _mm256_set1_epi8(1);
    let zero = _mm256_setzero_si256();
    for bi in 0..a[0].d.len() {
        let b0 = rows[0].as_ptr().add(bi * 34);
        let b1 = rows[1].as_ptr().add(bi * 34);
        let wd = [f16_to_f32(u16::from_le_bytes([*b0, *b0.add(1)])), f16_to_f32(u16::from_le_bytes([*b1, *b1.add(1)]))];
        let w = [_mm256_loadu_si256(b0.add(2) as *const __m256i), _mm256_loadu_si256(b1.add(2) as *const __m256i)];
        // 128 * (sum of each lane's four weights): what the unsigned offset added.
        let corr = [_mm256_slli_epi32(_mm256_dpbusd_avx_epi32(zero, ones, w[0]), 7), _mm256_slli_epi32(_mm256_dpbusd_avx_epi32(zero, ones, w[1]), 7)];
        for t in 0..4 {
            let x = _mm256_loadu_si256(a[t].qu.as_ptr().add(bi * 32) as *const __m256i);
            let dx = a[t].d[bi];
            for r in 0..2 {
                let p = _mm256_sub_epi32(_mm256_dpbusd_avx_epi32(zero, x, w[r]), corr[r]);
                acc[r][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(wd[r] * dx), acc[r][t]);
            }
        }
    }
    let mut out = [[0f32; 4]; 2];
    for r in 0..2 {
        for t in 0..4 {
            out[r][t] = hsum_ps(acc[r][t]);
        }
    }
    out
}

/// Four Q8_0 rows times one activation vector (decode matvec): each activation block is loaded once
/// for four rows, the four accumulators are independent (no single FMA chain to wait on), and the
/// four block scales convert in one F16C instruction. Each row's arithmetic is the same, in the same
/// order, as `dot_q8_0_q8`, so the results are bit-identical. None when AVX-VNNI or F16C is missing.
pub fn dot_q8_0_q8_rows4(rows: [&[u8]; 4], a: &QAct) -> Option<[f32; 4]> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avxvnni") && std::arch::is_x86_feature_detected!("f16c") {
            // SAFETY: AVX-VNNI (implies AVX2) and F16C were checked at run time just above.
            return Some(unsafe { dot_q8_0_q8_rows4_vnni(rows, a) });
        }
    }
    let _ = (rows, a);
    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni,f16c")]
unsafe fn dot_q8_0_q8_rows4_vnni(rows: [&[u8]; 4], a: &QAct) -> [f32; 4] {
    use std::arch::x86_64::*;
    let mut acc = [_mm256_setzero_ps(); 4];
    let p = [rows[0].as_ptr(), rows[1].as_ptr(), rows[2].as_ptr(), rows[3].as_ptr()];
    for (bi, d) in a.d.iter().enumerate() {
        let off = bi * 34;
        let x = _mm256_loadu_si256(a.q.as_ptr().add(bi * 32) as *const __m256i);
        // The four rows' f16 scales, converted together (exact, like f16_to_f32).
        let h = _mm_set_epi16(0, 0, 0, 0, (p[3].add(off) as *const i16).read_unaligned(), (p[2].add(off) as *const i16).read_unaligned(), (p[1].add(off) as *const i16).read_unaligned(), (p[0].add(off) as *const i16).read_unaligned());
        let mut wd = [0f32; 4];
        _mm_storeu_ps(wd.as_mut_ptr(), _mm_cvtph_ps(h));
        for k in 0..4 {
            let w = _mm256_loadu_si256(p[k].add(off + 2) as *const __m256i);
            let p32 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
            acc[k] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(wd[k] * d), acc[k]);
        }
    }
    let mut out = [0f32; 4];
    for k in 0..4 {
        let hi = _mm256_extractf128_ps(acc[k], 1);
        let s = _mm_add_ps(hi, _mm256_castps256_ps128(acc[k]));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        out[k] = _mm_cvtss_f32(s);
    }
    out
}

/// AVX-VNNI (Alder Lake and newer): one vpdpbusd replaces maddubs + madd per 32 values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn dot_q8_0_q8_vnni(row: &[u8], a: &QAct) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    for (bi, d) in a.d.iter().enumerate() {
        let b = row.as_ptr().add(bi * 34);
        let wd = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
        let w = _mm256_loadu_si256(b.add(2) as *const __m256i);
        let x = _mm256_loadu_si256(a.q.as_ptr().add(bi * 32) as *const __m256i);
        let p32 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(wd * d), acc);
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let s = _mm_add_ps(hi, _mm256_castps256_ps128(acc));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// K-quant (Q4_K, Q5_K, Q6_K) weight row times 8-bit activations, at the weights' stored size: the
/// 4/5/6-bit values are unpacked in registers and multiplied with integer instructions. Each 256-value
/// block holds 8 sub-blocks of 32 (Q4/Q5: scale and minimum) or 16 of 16 (Q6: signed scale).
pub fn dot_k_q8(t: DType, row: &[u8], a: &QAct) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 support was checked at run time just above.
            return unsafe {
                if a.kq.is_empty() {
                    match t {
                        DType::Q4K => dot_q4k_avx2(row, a),
                        DType::Q5K => dot_q5k_avx2(row, a),
                        _ => dot_q6k_avx2(row, a),
                    }
                } else {
                    match t {
                        DType::Q4K => dot_q4k_k8_avx2(row, a),
                        DType::Q5K => dot_q5k_k8_avx2(row, a),
                        _ => dot_q6k_k8_avx2(row, a),
                    }
                }
            };
        }
    }
    dot_k_q8_scalar(t, row, a)
}

/// Q4_K row times per-256 int8 activations: per block, the eight sub-blocks' products are scaled in
/// integer (madd with the 6-bit scales) and summed, then converted to float once.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4k_k8_avx2(row: &[u8], a: &QAct) -> f32 {
    use std::arch::x86_64::*;
    let m4 = _mm256_set1_epi8(0xF);
    let mut facc = _mm256_setzero_ps();
    let mut mins = 0f32;
    for blk in 0..row.len() / 144 {
        let b = row.as_ptr().add(blk * 144);
        let d = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*b.add(2), *b.add(3)]));
        let sc = std::slice::from_raw_parts(b.add(4), 12);
        let xq = a.kq.as_ptr().add(blk * 256);
        let ks = a.ks.as_ptr().add(blk * 16);
        let mut acc = _mm256_setzero_si256();
        let mut msum = 0i32;
        for j in 0..4 {
            let (s0, m0) = scale_min_k4(2 * j, sc);
            let (s1, m1) = scale_min_k4(2 * j + 1, sc);
            let qv = _mm256_loadu_si256(b.add(16 + j * 32) as *const __m256i);
            let lo = _mm256_and_si256(qv, m4);
            let hi = _mm256_and_si256(_mm256_srli_epi16(qv, 4), m4);
            let xl = _mm256_loadu_si256(xq.add(j * 64) as *const __m256i);
            let xh = _mm256_loadu_si256(xq.add(j * 64 + 32) as *const __m256i);
            let pl = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, xl), _mm256_set1_epi16(s0 as i16));
            let ph = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, xh), _mm256_set1_epi16(s1 as i16));
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(pl, ph));
            msum += m0 as i32 * (*ks.add(4 * j) + *ks.add(4 * j + 1)) + m1 as i32 * (*ks.add(4 * j + 2) + *ks.add(4 * j + 3));
        }
        let kd = *a.kd.get_unchecked(blk);
        facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(d * kd), facc);
        mins += dmin * kd * msum as f32;
    }
    hsum_ps(facc) - mins
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q5k_k8_avx2(row: &[u8], a: &QAct) -> f32 {
    use std::arch::x86_64::*;
    let m4 = _mm256_set1_epi8(0xF);
    let sixteen = _mm256_set1_epi8(16);
    let mut facc = _mm256_setzero_ps();
    let mut mins = 0f32;
    for blk in 0..row.len() / 176 {
        let b = row.as_ptr().add(blk * 176);
        let d = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*b.add(2), *b.add(3)]));
        let sc = std::slice::from_raw_parts(b.add(4), 12);
        let qh = _mm256_loadu_si256(b.add(16) as *const __m256i);
        let xq = a.kq.as_ptr().add(blk * 256);
        let ks = a.ks.as_ptr().add(blk * 16);
        let mut acc = _mm256_setzero_si256();
        let mut msum = 0i32;
        for j in 0..4 {
            let (s0, m0) = scale_min_k4(2 * j, sc);
            let (s1, m1) = scale_min_k4(2 * j + 1, sc);
            let qv = _mm256_loadu_si256(b.add(48 + j * 32) as *const __m256i);
            let b0 = _mm256_set1_epi8(1i8.wrapping_shl(2 * j as u32));
            let b1 = _mm256_set1_epi8(1i8.wrapping_shl(2 * j as u32 + 1));
            let h0 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, b0), b0), sixteen);
            let h1 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, b1), b1), sixteen);
            let lo = _mm256_or_si256(_mm256_and_si256(qv, m4), h0);
            let hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qv, 4), m4), h1);
            let xl = _mm256_loadu_si256(xq.add(j * 64) as *const __m256i);
            let xh = _mm256_loadu_si256(xq.add(j * 64 + 32) as *const __m256i);
            let pl = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, xl), _mm256_set1_epi16(s0 as i16));
            let ph = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, xh), _mm256_set1_epi16(s1 as i16));
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(pl, ph));
            msum += m0 as i32 * (*ks.add(4 * j) + *ks.add(4 * j + 1)) + m1 as i32 * (*ks.add(4 * j + 2) + *ks.add(4 * j + 3));
        }
        let kd = *a.kd.get_unchecked(blk);
        facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(d * kd), facc);
        mins += dmin * kd * msum as f32;
    }
    hsum_ps(facc) - mins
}

/// Q6_K row times per-256 int8 activations: unsigned 6-bit values times activations, scaled per 16 in
/// integer; the -32 offset is applied once per block from the activation sums.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q6k_k8_avx2(row: &[u8], a: &QAct) -> f32 {
    use std::arch::x86_64::*;
    let m4 = _mm256_set1_epi8(0xF);
    let m2 = _mm256_set1_epi8(3);
    let mut facc = _mm256_setzero_ps();
    let mut off = 0f32;
    for blk in 0..row.len() / 210 {
        let b = row.as_ptr().add(blk * 210);
        let d = f16_to_f32(u16::from_le_bytes([*b.add(208), *b.add(209)]));
        let xq = a.kq.as_ptr().add(blk * 256);
        let ks = a.ks.as_ptr().add(blk * 16);
        let mut acc = _mm256_setzero_si256();
        let mut osum = 0i32;
        for n in 0..2 {
            let ql0 = _mm256_loadu_si256(b.add(n * 64) as *const __m256i);
            let ql1 = _mm256_loadu_si256(b.add(n * 64 + 32) as *const __m256i);
            let qh = _mm256_loadu_si256(b.add(128 + n * 32) as *const __m256i);
            let sc = b.add(192 + n * 8) as *const i8;
            for g in 0..4usize {
                let ql = if g % 2 == 0 { ql0 } else { ql1 };
                let lo = if g < 2 { _mm256_and_si256(ql, m4) } else { _mm256_and_si256(_mm256_srli_epi16(ql, 4), m4) };
                let hi = _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128((2 * g) as i32)), m2);
                let q = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 4));
                let x = _mm256_loadu_si256(xq.add(n * 128 + g * 32) as *const __m256i);
                let (s1, s2) = (*sc.add(2 * g) as i16, *sc.add(2 * g + 1) as i16);
                let scales = _mm256_setr_epi16(s1, s1, s1, s1, s1, s1, s1, s1, s2, s2, s2, s2, s2, s2, s2, s2);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), scales));
                let k = n * 8 + 2 * g;
                osum += s1 as i32 * *ks.add(k) + s2 as i32 * *ks.add(k + 1);
            }
        }
        let kd = *a.kd.get_unchecked(blk);
        facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(d * kd), facc);
        off += d * kd * 32.0 * osum as f32;
    }
    hsum_ps(facc) - off
}

/// Unpacks sub-block `sb` (32 values) of a K-quant block into unsigned values and returns
/// (values, scale for the first 16, scale for the last 16, minimum) as integers; value =
/// d*scale*q - dmin*min (Q4/Q5) or d*scale*(q-32) (Q6, min unused).
fn k_sub(t: DType, b: &[u8], sb: usize, q: &mut [i32; 32]) -> (i32, i32, i32) {
    match t {
        DType::Q4K => {
            let (sc, m) = scale_min_k4(sb, &b[4..16]);
            let qs = &b[16 + (sb / 2) * 32..16 + (sb / 2) * 32 + 32];
            for l in 0..32 {
                q[l] = if sb % 2 == 0 { (qs[l] & 0xF) as i32 } else { (qs[l] >> 4) as i32 };
            }
            (sc as i32, sc as i32, m as i32)
        }
        DType::Q5K => {
            let (sc, m) = scale_min_k4(sb, &b[4..16]);
            let qh = &b[16..48];
            let qs = &b[48 + (sb / 2) * 32..48 + (sb / 2) * 32 + 32];
            for l in 0..32 {
                let lo = if sb % 2 == 0 { qs[l] & 0xF } else { qs[l] >> 4 };
                q[l] = (lo | if qh[l] & (1 << sb) != 0 { 16 } else { 0 }) as i32;
            }
            (sc as i32, sc as i32, m as i32)
        }
        _ => {
            // Q6_K: half n (128 values) holds groups g = 0..3 of 32, interleaved in ql/qh.
            let (n, g) = (sb / 4, sb % 4);
            let ql = &b[n * 64..];
            let qh = &b[128 + n * 32..];
            let sc = &b[192 + n * 8..];
            for l in 0..32 {
                // Groups 0/1 use the low nibbles of ql[0..32]/ql[32..64], groups 2/3 the high ones.
                let v = ql[l + (g % 2) * 32];
                let lo = if g < 2 { v & 0xF } else { v >> 4 };
                q[l] = (lo | (((qh[l] >> (2 * g)) & 3) << 4)) as i32;
            }
            (sc[2 * g] as i8 as i32, sc[2 * g + 1] as i8 as i32, 0)
        }
    }
}

fn dot_k_q8_scalar(t: DType, row: &[u8], a: &QAct) -> f32 {
    let bb = t.block_bytes();
    let mut q = [0i32; 32];
    let mut s = 0f32;
    for blk in 0..row.len() / bb {
        let b = &row[blk * bb..(blk + 1) * bb];
        for sb in 0..8 {
            let ai = blk * 8 + sb;
            let x = &a.q[ai * 32..ai * 32 + 32];
            let (s1, s2, m) = k_sub(t, b, sb, &mut q);
            if t == DType::Q6K {
                let d = rd16(b, 208);
                let p1: i32 = (0..16).map(|l| (q[l] - 32) * x[l] as i32).sum();
                let p2: i32 = (16..32).map(|l| (q[l] - 32) * x[l] as i32).sum();
                s += d * a.d[ai] * (s1 * p1 + s2 * p2) as f32;
            } else {
                let (d, dmin) = (rd16(b, 0), rd16(b, 2));
                let p: i32 = (0..32).map(|l| q[l] * x[l] as i32).sum();
                s += a.d[ai] * (d * s1 as f32 * p as f32 - dmin * m as f32 * a.s[ai] as f32);
            }
        }
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_ps(acc: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(acc, 1);
    let s = _mm_add_ps(hi, _mm256_castps256_ps128(acc));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4k_avx2(row: &[u8], a: &QAct) -> f32 {
    use std::arch::x86_64::*;
    let m4 = _mm256_set1_epi8(0xF);
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();
    let mut mins = 0f32;
    for blk in 0..row.len() / 144 {
        let b = &row[blk * 144..blk * 144 + 144];
        let (d, dmin) = (rd16(b, 0), rd16(b, 2));
        for j in 0..4 {
            let qv = _mm256_loadu_si256(b.as_ptr().add(16 + j * 32) as *const __m256i);
            let lo = _mm256_and_si256(qv, m4);
            let hi = _mm256_and_si256(_mm256_srli_epi16(qv, 4), m4);
            for (h, w) in [(0usize, lo), (1, hi)] {
                let sb = 2 * j + h;
                let ai = blk * 8 + sb;
                let (sc, m) = scale_min_k4(sb, &b[4..16]);
                let x = _mm256_loadu_si256(a.q.as_ptr().add(ai * 32) as *const __m256i);
                let p = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), ones);
                let xd = *a.d.get_unchecked(ai);
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * sc as f32 * xd), acc);
                mins += dmin * m as f32 * xd * *a.s.get_unchecked(ai) as f32;
            }
        }
    }
    hsum_ps(acc) - mins
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q5k_avx2(row: &[u8], a: &QAct) -> f32 {
    use std::arch::x86_64::*;
    let m4 = _mm256_set1_epi8(0xF);
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();
    let mut mins = 0f32;
    for blk in 0..row.len() / 176 {
        let b = &row[blk * 176..blk * 176 + 176];
        let (d, dmin) = (rd16(b, 0), rd16(b, 2));
        let qh = _mm256_loadu_si256(b.as_ptr().add(16) as *const __m256i);
        for j in 0..4 {
            let qv = _mm256_loadu_si256(b.as_ptr().add(48 + j * 32) as *const __m256i);
            for h in 0..2usize {
                let sb = 2 * j + h;
                let nib = if h == 0 { _mm256_and_si256(qv, m4) } else { _mm256_and_si256(_mm256_srli_epi16(qv, 4), m4) };
                // Bit sb of each qh byte becomes 16.
                let bit = _mm256_set1_epi8(1i8.wrapping_shl(sb as u32));
                let hb = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, bit), bit), _mm256_set1_epi8(16));
                let w = _mm256_or_si256(nib, hb);
                let ai = blk * 8 + sb;
                let (sc, m) = scale_min_k4(sb, &b[4..16]);
                let x = _mm256_loadu_si256(a.q.as_ptr().add(ai * 32) as *const __m256i);
                let p = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), ones);
                let xd = *a.d.get_unchecked(ai);
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * sc as f32 * xd), acc);
                mins += dmin * m as f32 * xd * *a.s.get_unchecked(ai) as f32;
            }
        }
    }
    hsum_ps(acc) - mins
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q6k_avx2(row: &[u8], a: &QAct) -> f32 {
    use std::arch::x86_64::*;
    let m4 = _mm256_set1_epi8(0xF);
    let m2 = _mm256_set1_epi8(3);
    let off = _mm256_set1_epi8(32);
    let mut acc = _mm256_setzero_ps();
    for blk in 0..row.len() / 210 {
        let b = &row[blk * 210..blk * 210 + 210];
        let d = rd16(b, 208);
        for n in 0..2 {
            let ql0 = _mm256_loadu_si256(b.as_ptr().add(n * 64) as *const __m256i);
            let ql1 = _mm256_loadu_si256(b.as_ptr().add(n * 64 + 32) as *const __m256i);
            let qh = _mm256_loadu_si256(b.as_ptr().add(128 + n * 32) as *const __m256i);
            let sc = &b[192 + n * 8..];
            for g in 0..4usize {
                let ql = if g % 2 == 0 { ql0 } else { ql1 };
                let lo = if g < 2 { _mm256_and_si256(ql, m4) } else { _mm256_and_si256(_mm256_srli_epi16(ql, 4), m4) };
                let hi = _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128((2 * g) as i32)), m2);
                // q - 32 in [-32, 31]; maddubs needs an unsigned operand, so move the sign onto x.
                let qs = _mm256_sub_epi8(_mm256_or_si256(lo, _mm256_slli_epi16(hi, 4)), off);
                let ai = blk * 8 + n * 4 + g;
                let x = _mm256_loadu_si256(a.q.as_ptr().add(ai * 32) as *const __m256i);
                let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(qs, qs), _mm256_sign_epi8(x, qs));
                // Lanes 0-7 hold values 0-15 (scale 2g), lanes 8-15 values 16-31 (scale 2g+1).
                let (s1, s2) = (sc[2 * g] as i8 as i16, sc[2 * g + 1] as i8 as i16);
                let scales = _mm256_setr_epi16(s1, s1, s1, s1, s1, s1, s1, s1, s2, s2, s2, s2, s2, s2, s2, s2);
                let p = _mm256_madd_epi16(p16, scales);
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * *a.d.get_unchecked(ai)), acc);
            }
        }
    }
    hsum_ps(acc)
}

#[cfg(test)]
mod kq_tests {
    use super::*;

    fn rand_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    #[test]
    fn k_quant_integer_dots_track_the_float_dot() {
        for t in [DType::Q4K, DType::Q5K, DType::Q6K] {
            let cols = 512;
            let mut row = rand_bytes(t.row_bytes(cols), 7);
            // Sane f16 scales: d = 0.01, dmin = 0.005 (Q6_K keeps d at the end).
            for blk in 0..cols / 256 {
                let o = blk * t.block_bytes();
                if t == DType::Q6K {
                    row[o + 208..o + 210].copy_from_slice(&0x211Eu16.to_le_bytes());
                } else {
                    row[o..o + 2].copy_from_slice(&0x211Eu16.to_le_bytes());
                    row[o + 2..o + 4].copy_from_slice(&0x1D1Eu16.to_le_bytes());
                }
            }
            let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.37).sin()).collect();
            let mut qa = QAct::new(cols);
            qa.quantize(&x);
            // Per-32 path first (kq cleared selects it).
            qa.kq.clear();
            // Reference: the float dot with the same quantised activations, so only kernel error remains.
            let xq: Vec<f32> = (0..cols).map(|i| qa.q[i] as f32 * qa.d[i / 32]).collect();
            let want = dot_row(t, &row, &xq);
            let scalar = dot_k_q8_scalar(t, &row, &qa);
            let fast = dot_k_q8(t, &row, &qa);
            let tol = want.abs() * 1e-3 + 1e-3;
            assert!((scalar - want).abs() <= tol, "{t:?} scalar {scalar} vs {want}");
            assert!((fast - scalar).abs() <= scalar.abs() * 1e-4 + 1e-3, "{t:?} simd {fast} vs scalar {scalar}");
            // Per-256 activations (the default path): exact against the float dot on the same values.
            let mut qk = QAct::new(cols);
            qk.quantize(&x);
            let xk: Vec<f32> = (0..cols).map(|i| qk.kq[i] as f32 * qk.kd[i / 256]).collect();
            let want_k = dot_row(t, &row, &xk);
            let got_k = dot_k_q8(t, &row, &qk);
            assert!((got_k - want_k).abs() <= want_k.abs() * 1e-3 + 1e-3, "{t:?} k8 {got_k} vs {want_k}");
        }
    }
}

#[cfg(test)]
mod exp_tests {
    use super::*;

    #[test]
    fn fast_exp_matches_libm() {
        let mut x = -80.0f32;
        while x < 80.0 {
            let (a, b) = (exp_fast(x), x.exp());
            assert!(((a - b) / b).abs() < 1e-6, "{x}: {a} vs {b}");
            x += 0.137;
        }
    }
}

#[cfg(test)]
mod q8_tests {
    use super::*;

    #[test]
    fn integer_q8_dot_tracks_the_float_dot() {
        let mut row = vec![0u8; 34 * 4];
        for b in 0..4 {
            row[b * 34..b * 34 + 2].copy_from_slice(&0x2e66u16.to_le_bytes());
            for i in 0..32 {
                row[b * 34 + 2 + i] = (((i * 7 + b * 13) % 255) as i32 - 127) as i8 as u8;
            }
        }
        let x: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.21).cos()).collect();
        let mut qa = QAct::new(128);
        qa.quantize(&x);
        let want = dot_row(DType::Q8_0, &row, &x);
        let got = dot_q8_0_q8(&row, &qa);
        assert!((got - want).abs() <= want.abs() * 0.02 + 0.05, "{got} vs {want}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_q4k_matches_the_file_layout() {
        let (rows, cols) = (16, 512);
        let nb = cols / 256;
        let mut src = Vec::new();
        let mut x = 777u32;
        let mut rnd = || {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (x >> 16) as u8
        };
        for r in 0..rows {
            for b in 0..nb {
                src.extend_from_slice(&to_f16(0.01 + 0.001 * (r + b) as f32).to_le_bytes());
                src.extend_from_slice(&to_f16(0.003 + 0.0005 * r as f32).to_le_bytes());
                for _ in 0..140 {
                    src.push(rnd());
                }
            }
        }
        let packed = repack_q4k_x8(&src, rows, cols);
        let acts: Vec<QAct> = (0..4)
            .map(|t| {
                let mut a = QAct::new(cols);
                a.quantize(&(0..cols).map(|i| ((i * 7 + t * 31) % 23) as f32 / 11.0 - 1.0).collect::<Vec<_>>());
                a
            })
            .collect();
        for g in 0..rows / 8 {
            let tile = gemm_q4k_x8_4(&packed, nb, g, [&acts[0], &acts[1], &acts[2], &acts[3]]);
            for (t, a) in acts.iter().enumerate() {
                let one = gemv_q4k_x8(&packed, nb, g, a);
                for r in 0..8 {
                    let want = dot_k_q8(DType::Q4K, &src[(g * 8 + r) * nb * 144..(g * 8 + r + 1) * nb * 144], a);
                    for got in [one[r], tile[t][r]] {
                        assert!((got - want).abs() <= 1e-3 * want.abs().max(1e-2), "g{g} r{r} t{t}: {got} vs {want}");
                    }
                }
            }
        }
    }

    #[test]
    fn interleaved_q8_0_matches_the_file_layout() {
        let (rows, cols) = (16, 128);
        let nb = cols / 32;
        let mut src = Vec::new();
        let mut x = 12345u32;
        for r in 0..rows {
            for b in 0..nb {
                src.extend_from_slice(&to_f16(0.002 * (1 + r + b) as f32).to_le_bytes());
                for i in 0..32 {
                    x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    src.push(if i == 3 { 0x80 } else { (x >> 16) as u8 });
                }
            }
        }
        let packed = repack_q8_0_x8(&src, rows, cols);
        let acts: Vec<QAct> = (0..4)
            .map(|t| {
                let mut a = QAct::new(cols);
                a.quantize(&(0..cols).map(|i| ((i * 7 + t * 31) % 23) as f32 / 11.0 - 1.0).collect::<Vec<_>>());
                a
            })
            .collect();
        for g in 0..rows / 8 {
            let tile = gemm_q8_0_x8_4(&packed, nb, g, [&acts[0], &acts[1], &acts[2], &acts[3]]);
            for (t, a) in acts.iter().enumerate() {
                let one = gemv_q8_0_x8(&packed, nb, g, a);
                for r in 0..8 {
                    let want = dot_q8_0_q8(&src[(g * 8 + r) * nb * 34..(g * 8 + r + 1) * nb * 34], a);
                    for got in [one[r], tile[t][r]] {
                        assert!((got - want).abs() <= 1e-4 * want.abs().max(1e-3), "g{g} r{r} t{t}: {got} vs {want}");
                    }
                }
            }
        }
    }

    #[test]
    fn prefill_tile_is_bit_identical_to_the_four_token_kernel() {
        let cols = 256;
        let nb = cols / 32;
        // Two Q8_0 rows: per block an f16 scale and 32 int8 weights, including -128 and 127.
        let mk = |seed: u32| -> Vec<u8> {
            let mut row = Vec::new();
            let mut x = seed;
            for b in 0..nb {
                row.extend_from_slice(&to_f16(0.01 + b as f32 * 0.003).to_le_bytes());
                for i in 0..32 {
                    x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    let v = if i == 0 { -128i8 } else if i == 1 { 127 } else { (x >> 16) as i8 };
                    row.push(v as u8);
                }
            }
            row
        };
        let (r0, r1) = (mk(1), mk(7));
        let acts: Vec<QAct> = (0..4)
            .map(|t| {
                let mut a = QAct::new(cols);
                let x: Vec<f32> = (0..cols).map(|i| ((i * 13 + t * 101) % 61) as f32 / 30.0 - 1.0).collect();
                a.quantize(&x);
                a
            })
            .collect();
        let Some(tile) = gemm_q8_0_2x4([&r0, &r1], [&acts[0], &acts[1], &acts[2], &acts[3]]) else { return };
        for (r, row) in [&r0, &r1].into_iter().enumerate() {
            let want = dot_q8_0_q8_x4(row, [&acts[0], &acts[1], &acts[2], &acts[3]]);
            assert_eq!(tile[r].map(f32::to_bits), want.map(f32::to_bits), "row {r}");
        }
    }

    #[test]
    fn blocked_attention_kernels_match_one_key_at_a_time() {
        let (hd, stride, n) = (128, 1024, 37);
        let k: Vec<u16> = (0..n * stride).map(|i| to_f16(((i * 7919 % 2003) as f32 - 1000.0) * 0.001)).collect();
        let q: Vec<f32> = (0..hd).map(|i| (i as f32 * 0.37).cos()).collect();
        let mut s = vec![0f32; n];
        scores_f16(&q, &k[64..], stride, &mut s);
        for (i, v) in s.iter().enumerate() {
            let want = dot_f16(&q, &k[64 + i * stride..64 + i * stride + hd]);
            assert!((v - want).abs() <= 1e-4 * want.abs().max(1.0), "score {i}: {v} vs {want}");
        }
        let p: Vec<f32> = (0..n).map(|i| 1.0 / (1.0 + i as f32)).collect();
        let mut o = vec![0.25f32; hd];
        weighted_f16(&p, &k[64..], stride, &mut o);
        let mut want = vec![0.25f32; hd];
        for (i, &w) in p.iter().enumerate() {
            axpy_f16(&mut want, w, &k[64 + i * stride..64 + i * stride + hd]);
        }
        for (a, b) in o.iter().zip(&want) {
            assert!((a - b).abs() <= 1e-4 * b.abs().max(1.0));
        }
    }

    #[test]
    fn f16_cache_kernels_match_the_scalar_definitions() {
        let src: Vec<f32> = (0..64).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.0371 + if i == 5 { 65504.0 } else { 0.0 }).collect();
        let mut h = vec![0u16; 64];
        store_f16(&mut h, &src);
        let scalar: Vec<u16> = src.iter().map(|&v| to_f16(v)).collect();
        assert_eq!(h, scalar, "F16C and the scalar rounding differ");
        let q: Vec<f32> = (0..64).map(|i| (i as f32 * 0.13).sin()).collect();
        let want: f32 = q.iter().zip(&h).map(|(a, &b)| a * f16_to_f32(b)).sum();
        assert!((dot_f16(&q, &h) - want).abs() < 1e-3 * want.abs().max(1.0));
        let mut y = vec![1.0f32; 64];
        axpy_f16(&mut y, 0.5, &h);
        for (i, v) in y.iter().enumerate() {
            assert!((v - (1.0 + 0.5 * f16_to_f32(h[i]))).abs() < 1e-3);
        }
    }

    #[test]
    fn f16_round_trips_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x3555) - 0.333_251_95).abs() < 1e-6);
        assert!(f16_to_f32(0x0001) > 0.0);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn vnni_q8_dot_equals_avx2() {
        if !std::arch::is_x86_feature_detected!("avxvnni") {
            return;
        }
        let nb = 64;
        let mut row: Vec<u8> = (0..nb * 34).map(|i| (i as u32).wrapping_mul(2654435761).rotate_right(13) as u8).collect();
        for b in 0..nb {
            row[b * 34..b * 34 + 2].copy_from_slice(&0x2c00u16.to_le_bytes());
        }
        let x: Vec<f32> = (0..nb * 32).map(|i| (i as f32 * 0.21).cos()).collect();
        let mut a = QAct::new(x.len());
        a.quantize(&x);
        // SAFETY: feature checked above.
        let (v, w) = unsafe { (dot_q8_0_q8_vnni(&row, &a), dot_q8_0_q8_avx2(&row, &a)) };
        assert!((v - w).abs() <= 1e-3 * w.abs().max(1.0), "{v} vs {w}");
    }

    /// The four-row kernel gives exactly the one-row results, scales of every kind included
    /// (normal, subnormal, zero, negative).
    #[test]
    fn rows4_is_bit_identical_to_one_row() {
        let nb = 32;
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let rows: Vec<Vec<u8>> = (0..4)
            .map(|k| {
                let mut r: Vec<u8> = (0..nb * 34).map(|_| next() as u8).collect();
                for b in 0..nb {
                    let scale: u16 = match (b + k) % 5 { 0 => 0x0001, 1 => 0x0000, 2 => 0xb800, 3 => 0x2c00, _ => (next() as u16) & 0x7bff };
                    r[b * 34..b * 34 + 2].copy_from_slice(&scale.to_le_bytes());
                }
                r
            })
            .collect();
        let x: Vec<f32> = (0..nb * 32).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let mut a = QAct::new(x.len());
        a.quantize(&x);
        let Some(four) = dot_q8_0_q8_rows4([&rows[0], &rows[1], &rows[2], &rows[3]], &a) else { return };
        for k in 0..4 {
            assert_eq!(four[k].to_bits(), dot_q8_0_q8(&rows[k], &a).to_bits(), "row {k}");
        }
    }

    #[test]
    fn q8_0_dot_matches_dequantised_dot() {
        let mut row = vec![0u8; 34 * 2];
        for b in 0..2 {
            row[b * 34..b * 34 + 2].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5
            for i in 0..32 {
                row[b * 34 + 2 + i] = ((i as i32 - 16 + b as i32) as i8) as u8;
            }
        }
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut deq = vec![0f32; 64];
        dequant_row(DType::Q8_0, &row, &mut deq);
        let want: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((dot_row(DType::Q8_0, &row, &x) - want).abs() < 1e-4);
    }
}

// ---- Integer dots for the remaining i-quants and Q2_K (AVX2) ------------------------------------
// The same scheme as dot_iq2s_k8: per-256 int8 activations (QAct::kq), the weights unpacked in
// registers from their grids, signs moved onto the activations, maddubs + madd with the per-32 (or
// per-16) scales, one float conversion per block. IQ1's +-1/8 offset is summed in integer too (as
// 8 * value + sign, divided by 8 at the end). None without AVX2 or per-256 activations: the caller
// then takes the float path.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let s = _mm_add_epi32(_mm256_extracti128_si256(v, 1), _mm256_castsi256_si128(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    _mm_cvtsi128_si32(s)
}

/// 0xff in each byte whose bit is set in the four sign bytes of `s4` (one byte per 8 values).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sign_mask(s4: u32) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let spread = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    let bits = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
    let sv = _mm256_shuffle_epi8(_mm256_set1_epi32(s4 as i32), spread);
    _mm256_cmpeq_epi8(_mm256_and_si256(sv, bits), bits)
}

macro_rules! avx2_or_none {
    ($a:expr, $f:expr) => {{
        if $a.kq.is_empty() {
            return None;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 support was checked at run time just above.
                return Some(unsafe { $f });
            }
        }
        None
    }};
}

/// IQ2_XXS (66 bytes per 256): grid magnitudes, 7-bit sign codes, a 4-bit scale per 32.
pub fn dot_iq2xxs_k8(row: &[u8], a: &QAct) -> Option<f32> {
    avx2_or_none!(a, dot_iq2xxs_k8_avx2(row, a))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq2xxs_k8_avx2(row: &[u8], a: &QAct) -> f32 {
    use crate::iq_tables::{IQ2XXS_GRID, KSIGNS_IQ2XS};
    use std::arch::x86_64::*;
    let one8 = _mm256_set1_epi8(1);
    let mut s = 0f32;
    for (bi, kd) in a.kd.iter().enumerate() {
        let b = row.as_ptr().add(bi * 66);
        let d = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
        let mut acc = _mm256_setzero_si256();
        for ib in 0..8 {
            let a0 = std::ptr::read_unaligned(b.add(2 + 8 * ib) as *const u32);
            let a1 = std::ptr::read_unaligned(b.add(6 + 8 * ib) as *const u32);
            let g = |l: u32| IQ2XXS_GRID[((a0 >> (8 * l)) & 0xff) as usize] as i64;
            let w = _mm256_set_epi64x(g(3), g(2), g(1), g(0));
            let sg = |l: u32| KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize] as u32;
            let neg = sign_mask(sg(0) | sg(1) << 8 | sg(2) << 16 | sg(3) << 24);
            let x = _mm256_loadu_si256(a.kq.as_ptr().add(bi * 256 + ib * 32) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(w, _mm256_sign_epi8(x, _mm256_or_si256(neg, one8)));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, _mm256_set1_epi16(2 * (a1 >> 28) as i16 + 1)));
        }
        s += d * kd * hsum_epi32(acc) as f32 * 0.125;
    }
    s
}

/// IQ3_XXS (98 bytes per 256): eight 4-value grid entries per 32, 7-bit sign codes, a 4-bit scale.
pub fn dot_iq3xxs_k8(row: &[u8], a: &QAct) -> Option<f32> {
    avx2_or_none!(a, dot_iq3xxs_k8_avx2(row, a))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq3xxs_k8_avx2(row: &[u8], a: &QAct) -> f32 {
    use crate::iq_tables::{IQ3XXS_GRID, KSIGNS_IQ2XS};
    use std::arch::x86_64::*;
    let one8 = _mm256_set1_epi8(1);
    let mut s = 0f32;
    for (bi, kd) in a.kd.iter().enumerate() {
        let b = row.as_ptr().add(bi * 98);
        let d = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
        let mut acc = _mm256_setzero_si256();
        for ib in 0..8 {
            let q = b.add(2 + 8 * ib);
            let g = |k: usize| IQ3XXS_GRID[*q.add(k) as usize] as i32;
            let w = _mm256_set_epi32(g(7), g(6), g(5), g(4), g(3), g(2), g(1), g(0));
            let aux = std::ptr::read_unaligned(b.add(66 + 4 * ib) as *const u32);
            let sg = |l: u32| KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize] as u32;
            let neg = sign_mask(sg(0) | sg(1) << 8 | sg(2) << 16 | sg(3) << 24);
            let x = _mm256_loadu_si256(a.kq.as_ptr().add(bi * 256 + ib * 32) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(w, _mm256_sign_epi8(x, _mm256_or_si256(neg, one8)));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, _mm256_set1_epi16(2 * (aux >> 28) as i16 + 1)));
        }
        s += d * kd * hsum_epi32(acc) as f32 * 0.25;
    }
    s
}

/// IQ3_S (110 bytes per 256): 9-bit grid indices (high bits in qh), explicit sign bytes, a 4-bit
/// scale per 32.
pub fn dot_iq3s_k8(row: &[u8], a: &QAct) -> Option<f32> {
    avx2_or_none!(a, dot_iq3s_k8_avx2(row, a))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq3s_k8_avx2(row: &[u8], a: &QAct) -> f32 {
    use crate::iq_tables::IQ3S_GRID;
    use std::arch::x86_64::*;
    let one8 = _mm256_set1_epi8(1);
    let mut s = 0f32;
    for (bi, kd) in a.kd.iter().enumerate() {
        let b = row.as_ptr().add(bi * 110);
        let d = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
        let (qs, qh, signs, scales) = (b.add(2), b.add(66), b.add(74), b.add(106));
        let mut acc = _mm256_setzero_si256();
        for ib in 0..8 {
            let hb = *qh.add(ib) as usize;
            let g = |k: usize| IQ3S_GRID[*qs.add(8 * ib + k) as usize | ((hb << (8 - k)) & 256)] as i32;
            let w = _mm256_set_epi32(g(7), g(6), g(5), g(4), g(3), g(2), g(1), g(0));
            let neg = sign_mask(std::ptr::read_unaligned(signs.add(4 * ib) as *const u32));
            let x = _mm256_loadu_si256(a.kq.as_ptr().add(bi * 256 + ib * 32) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(w, _mm256_sign_epi8(x, _mm256_or_si256(neg, one8)));
            let sc = *scales.add(ib / 2);
            let nib = if ib % 2 == 0 { sc & 0xf } else { sc >> 4 };
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, _mm256_set1_epi16(1 + 2 * nib as i16)));
        }
        s += d * kd * hsum_epi32(acc) as f32;
    }
    s
}

/// Q1_0 (18 bytes per 128): each int8 activation with the weight's sign, summed per 128 and scaled
/// by the block's scale and the activation block's (two weight blocks per 256 activations).
pub fn dot_q1_0_k8(row: &[u8], a: &QAct) -> Option<f32> {
    avx2_or_none!(a, dot_q1_0_k8_avx2(row, a))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q1_0_k8_avx2(row: &[u8], a: &QAct) -> f32 {
    use std::arch::x86_64::*;
    let ones8 = _mm256_set1_epi8(1);
    let ones16 = _mm256_set1_epi16(1);
    let minus = _mm256_set1_epi8(-1);
    let mut s = 0f32;
    for (bi, kd) in a.kd.iter().enumerate() {
        for half in 0..2 {
            let b = row.as_ptr().add((2 * bi + half) * 18);
            let d = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
            let mut acc = _mm256_setzero_si256();
            for q in 0..4 {
                let bits = u32::from_le_bytes([*b.add(2 + 4 * q), *b.add(3 + 4 * q), *b.add(4 + 4 * q), *b.add(5 + 4 * q)]);
                // +1 where the bit is set, -1 where it is not.
                let w = _mm256_or_si256(_mm256_andnot_si256(sign_mask(bits), minus), ones8);
                let x = _mm256_loadu_si256(a.kq.as_ptr().add(bi * 256 + half * 128 + q * 32) as *const __m256i);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x, w)), ones16));
            }
            s += d * kd * hsum_epi32(acc) as f32;
        }
    }
    s
}

/// IQ1_S (50 bytes per 256): grid values in {-1, 0, 1} plus an offset of +-1/8 per 32, a 3-bit
/// scale per 32.
pub fn dot_iq1s_k8(row: &[u8], a: &QAct) -> Option<f32> {
    avx2_or_none!(a, dot_iq1s_k8_avx2(row, a))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq1s_k8_avx2(row: &[u8], a: &QAct) -> f32 {
    use crate::iq_tables::IQ1S_GRID;
    use std::arch::x86_64::*;
    let mut s = 0f32;
    for (bi, kd) in a.kd.iter().enumerate() {
        let b = row.as_ptr().add(bi * 50);
        let d = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
        let mut acc = _mm256_setzero_si256();
        let mut off = 0i32;
        for ib in 0..8 {
            let qh = u16::from_le_bytes([*b.add(34 + 2 * ib), *b.add(35 + 2 * ib)]) as usize;
            let g = |l: usize| IQ1S_GRID[*b.add(2 + 4 * ib + l) as usize | (((qh >> (3 * l)) & 7) << 8)] as i64;
            let w = _mm256_set_epi64x(g(3), g(2), g(1), g(0));
            let x = _mm256_loadu_si256(a.kq.as_ptr().add(bi * 256 + ib * 32) as *const __m256i);
            // |w| (0 or 1) times x with w's sign.
            let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
            let sc = 2 * ((qh >> 12) & 7) as i32 + 1;
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, _mm256_set1_epi16(sc as i16)));
            let sumx = *a.ks.get_unchecked(bi * 16 + 2 * ib) + *a.ks.get_unchecked(bi * 16 + 2 * ib + 1);
            off += sc * if qh & 0x8000 != 0 { -sumx } else { sumx };
        }
        s += d * kd * (8 * hsum_epi32(acc) + off) as f32 * 0.125;
    }
    s
}

/// IQ1_M (56 bytes per 256): like IQ1_S with the offset sign per 8 values, two 3-bit scales per 32
/// and the block scale spread over the scale words.
pub fn dot_iq1m_k8(row: &[u8], a: &QAct) -> Option<f32> {
    avx2_or_none!(a, dot_iq1m_k8_avx2(row, a))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq1m_k8_avx2(row: &[u8], a: &QAct) -> f32 {
    use crate::iq_tables::IQ1S_GRID;
    use std::arch::x86_64::*;
    let ones = _mm256_set1_epi8(1);
    let mut s = 0f32;
    for (bi, kd) in a.kd.iter().enumerate() {
        let b = row.as_ptr().add(bi * 56);
        let w16 = |o: usize| u16::from_le_bytes([*b.add(o), *b.add(o + 1)]);
        let sc = [w16(48), w16(50), w16(52), w16(54)];
        let d = f16_to_f32((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        let mut acc = _mm256_setzero_si256();
        let mut accd = _mm256_setzero_si256();
        for ib in 0..8 {
            let sv = sc[ib / 2] >> (6 * (ib % 2));
            let (s0, s1) = (2 * (sv & 7) as i16 + 1, 2 * ((sv >> 3) & 7) as i16 + 1);
            let (q0, q1, q2, q3) = (*b.add(4 * ib) as usize, *b.add(4 * ib + 1) as usize, *b.add(4 * ib + 2) as usize, *b.add(4 * ib + 3) as usize);
            let (h0, h1) = (*b.add(32 + 2 * ib) as usize, *b.add(33 + 2 * ib) as usize);
            let w = _mm256_set_epi64x(
                IQ1S_GRID[q3 | ((h1 << 4) & 0x700)] as i64,
                IQ1S_GRID[q2 | ((h1 << 8) & 0x700)] as i64,
                IQ1S_GRID[q1 | ((h0 << 4) & 0x700)] as i64,
                IQ1S_GRID[q0 | ((h0 << 8) & 0x700)] as i64,
            );
            let x = _mm256_loadu_si256(a.kq.as_ptr().add(bi * 256 + ib * 32) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
            let scales = _mm256_set_m128i(_mm_set1_epi16(s1), _mm_set1_epi16(s0));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, scales));
            // The offsets: pairs of x summed, then weighted by +-scale per 8 values (4 lanes each).
            let dl = |bit: bool, sc: i16| if bit { -sc } else { sc };
            let dw = _mm256_set_epi16(
                dl(h1 & 0x80 != 0, s1), dl(h1 & 0x80 != 0, s1), dl(h1 & 0x80 != 0, s1), dl(h1 & 0x80 != 0, s1),
                dl(h1 & 0x08 != 0, s1), dl(h1 & 0x08 != 0, s1), dl(h1 & 0x08 != 0, s1), dl(h1 & 0x08 != 0, s1),
                dl(h0 & 0x80 != 0, s0), dl(h0 & 0x80 != 0, s0), dl(h0 & 0x80 != 0, s0), dl(h0 & 0x80 != 0, s0),
                dl(h0 & 0x08 != 0, s0), dl(h0 & 0x08 != 0, s0), dl(h0 & 0x08 != 0, s0), dl(h0 & 0x08 != 0, s0),
            );
            accd = _mm256_add_epi32(accd, _mm256_madd_epi16(_mm256_maddubs_epi16(ones, x), dw));
        }
        s += d * kd * (8 * hsum_epi32(acc) + hsum_epi32(accd)) as f32 * 0.125;
    }
    s
}

/// Q2_K (84 bytes per 256): 2-bit values, a 4-bit scale and a 4-bit minimum per 16.
pub fn dot_q2k_k8(row: &[u8], a: &QAct) -> Option<f32> {
    avx2_or_none!(a, dot_q2k_k8_avx2(row, a))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q2k_k8_avx2(row: &[u8], a: &QAct) -> f32 {
    use std::arch::x86_64::*;
    let m3 = _mm256_set1_epi8(3);
    let mut s = 0f32;
    for (bi, kd) in a.kd.iter().enumerate() {
        let b = row.as_ptr().add(bi * 84);
        let d = f16_to_f32(u16::from_le_bytes([*b.add(80), *b.add(81)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*b.add(82), *b.add(83)]));
        let ks = a.ks.as_ptr().add(bi * 16);
        let mut acc = _mm256_setzero_si256();
        let mut msum = 0i32;
        let mut is = 0usize;
        for n in 0..2 {
            let qv = _mm256_loadu_si256(b.add(16 + 32 * n) as *const __m256i);
            for k in 0..4 {
                let w = _mm256_and_si256(_mm256_srl_epi16(qv, _mm_cvtsi32_si128(2 * k as i32)), m3);
                let x = _mm256_loadu_si256(a.kq.as_ptr().add(bi * 256 + n * 128 + k * 32) as *const __m256i);
                let (c0, c1) = (*b.add(is), *b.add(is + 1));
                let scales = _mm256_set_m128i(_mm_set1_epi16((c1 & 0xf) as i16), _mm_set1_epi16((c0 & 0xf) as i16));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), scales));
                let g = (n * 128 + k * 32) / 16;
                msum += (c0 >> 4) as i32 * *ks.add(g) + (c1 >> 4) as i32 * *ks.add(g + 1);
                is += 2;
            }
        }
        s += d * kd * hsum_epi32(acc) as f32 - dmin * kd * msum as f32;
    }
    s
}

#[cfg(test)]
mod iq_tests {
    use super::*;

    fn rand_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect()
    }

    fn acts(n: usize) -> (Vec<f32>, QAct) {
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin() * 2.0 + 0.1).collect();
        let mut qa = QAct::new(n);
        qa.quantize(&x);
        (x, qa)
    }

    fn close(got: f32, want: f32) {
        assert!((got - want).abs() <= want.abs() * 0.03 + 0.05, "{got} vs {want}");
    }

    #[test]
    fn iq4nl_integer_dot_tracks_the_float_dot() {
        let mut row = rand_bytes(18 * 16, 7);
        for b in 0..16 {
            row[b * 18..b * 18 + 2].copy_from_slice(&0x2c00u16.to_le_bytes());
        }
        let (x, qa) = acts(512);
        close(dot_iq4nl_q8(&row, &qa), dot_row(DType::IQ4NL, &row, &x));
    }

    #[test]
    fn iq2s_integer_dot_tracks_the_float_dot() {
        let mut row = rand_bytes(82 * 4, 11);
        for b in 0..4 {
            row[b * 82..b * 82 + 2].copy_from_slice(&0x2400u16.to_le_bytes());
        }
        let (x, qa) = acts(1024);
        close(dot_iq2s_k8(&row, &qa).unwrap(), dot_row(DType::IQ2S, &row, &x));
    }

    /// A random row of `n` blocks of `bb` bytes with a sane f16 scale at byte `at` of each block.
    fn iq_row(n: usize, bb: usize, at: usize, seed: u64) -> Vec<u8> {
        let mut row = rand_bytes(bb * n, seed);
        for b in 0..n {
            row[b * bb + at..b * bb + at + 2].copy_from_slice(&0x2400u16.to_le_bytes());
        }
        row
    }

    #[test]
    fn low_bit_integer_dots_track_the_float_dot() {
        let (x, qa) = acts(1024);
        close(dot_iq2xxs_k8(&iq_row(4, 66, 0, 21), &qa).unwrap(), dot_row(DType::IQ2XXS, &iq_row(4, 66, 0, 21), &x));
        close(dot_iq3xxs_k8(&iq_row(4, 98, 0, 22), &qa).unwrap(), dot_row(DType::IQ3XXS, &iq_row(4, 98, 0, 22), &x));
        close(dot_iq3s_k8(&iq_row(4, 110, 0, 23), &qa).unwrap(), dot_row(DType::IQ3S, &iq_row(4, 110, 0, 23), &x));
        close(dot_iq1s_k8(&iq_row(4, 50, 0, 24), &qa).unwrap(), dot_row(DType::IQ1S, &iq_row(4, 50, 0, 24), &x));
        close(dot_q2k_k8(&iq_row(4, 84, 80, 26), &qa).unwrap(), dot_row(DType::Q2K, &iq_row(4, 84, 80, 26), &x));
        // IQ1_M keeps its scale in the top nibbles of the scale words: a mid-range f16 there.
        let mut m = rand_bytes(56 * 4, 25);
        for b in 0..4 {
            let bits = 0x2400u16;
            for (w, nib) in [(48usize, bits & 0xf), (50, (bits >> 4) & 0xf), (52, (bits >> 8) & 0xf), (54, bits >> 12)] {
                m[b * 56 + w + 1] = (m[b * 56 + w + 1] & 0x0f) | ((nib as u8) << 4);
            }
        }
        close(dot_iq1m_k8(&m, &qa).unwrap(), dot_row(DType::IQ1M, &m, &x));
        // Q1_0: random signs and scales, 18 bytes per 128.
        let mut q1 = vec![0u8; x.len() / 128 * 18];
        for (i, blk) in q1.chunks_exact_mut(18).enumerate() {
            blk[..2].copy_from_slice(&crate::quant::to_f16(0.01 + 0.001 * i as f32).to_le_bytes());
            for (j, v) in blk[2..].iter_mut().enumerate() {
                *v = ((i * 131 + j * 29) % 251) as u8;
            }
        }
        close(dot_q1_0_k8(&q1, &qa).unwrap(), dot_row(DType::Q1_0, &q1, &x));
    }

}
