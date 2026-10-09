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
            _ => return None,
        })
    }
    pub fn block_elems(self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::BF16 => 1,
            DType::Q8_0 => 32,
            DType::Q4K | DType::Q5K | DType::Q6K => 256,
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
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
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

/// Activations quantised to 8 bits in blocks of 32 (one f32 scale per block), so Q8_0 weights can be
/// multiplied with integer instructions. Built once per step and shared by every matrix that reads it.
pub struct QAct {
    pub d: Vec<f32>,
    pub q: Vec<i8>,
}

impl QAct {
    pub fn new(len: usize) -> Self {
        Self { d: vec![0.0; len / 32], q: vec![0; len] }
    }

    pub fn quantize(&mut self, x: &[f32]) {
        for (bi, xb) in x.chunks_exact(32).enumerate() {
            let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            self.d[bi] = d;
            for (i, v) in xb.iter().enumerate() {
                self.q[bi * 32 + i] = (v * id).round() as i8;
            }
        }
    }
}

/// Q8_0 weight row times 8-bit activations.
pub fn dot_q8_0_q8(row: &[u8], a: &QAct) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
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
    fn f16_round_trips_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x3555) - 0.333_251_95).abs() < 1e-6);
        assert!(f16_to_f32(0x0001) > 0.0);
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
