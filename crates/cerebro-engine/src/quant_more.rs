//! Dequantisation for the remaining GGUF formats: the legacy 32-value blocks (Q4_0, Q4_1, Q5_0,
//! Q5_1), the low K-quants (Q2_K, Q3_K) and the i-quants (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
//! IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS) that Unsloth Dynamic (UD) files are built from. Block layouts and
//! value mappings follow the GGUF format definition; rows of these types go through the generic
//! dequantise-and-dot path on the CPU and are repacked for the GPU.

use crate::iq_tables::*;
use crate::quant::{f16_to_f32, DType};

#[inline]
fn h(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

#[inline]
fn u16le(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

#[inline]
fn u32le(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

#[inline]
fn grid8(g: u64, j: usize) -> f32 {
    ((g >> (8 * j)) & 0xff) as u8 as f32
}

#[inline]
fn grid4(g: u32, j: usize) -> f32 {
    ((g >> (8 * j)) & 0xff) as u8 as f32
}

#[inline]
fn sign(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
}

/// Dequantises one block of `t` (one of the formats this module covers) into `out`.
pub fn dequant_block(t: DType, b: &[u8], out: &mut [f32]) {
    match t {
        DType::Q4_0 => {
            let d = h(b, 0);
            for j in 0..16 {
                out[j] = d * ((b[2 + j] & 0xf) as f32 - 8.0);
                out[j + 16] = d * ((b[2 + j] >> 4) as f32 - 8.0);
            }
        }
        DType::Q4_1 => {
            let (d, m) = (h(b, 0), h(b, 2));
            for j in 0..16 {
                out[j] = d * (b[4 + j] & 0xf) as f32 + m;
                out[j + 16] = d * (b[4 + j] >> 4) as f32 + m;
            }
        }
        DType::Q5_0 | DType::Q5_1 => {
            let (d, m, o) = if t == DType::Q5_0 { (h(b, 0), 0.0, 2) } else { (h(b, 0), h(b, 2), 4) };
            let qh = u32le(b, o);
            let qs = &b[o + 4..];
            let off = if t == DType::Q5_0 { 16.0 } else { 0.0 };
            for j in 0..16 {
                let x0 = ((qs[j] & 0xf) as u32 | (((qh >> j) << 4) & 0x10)) as f32;
                let x1 = ((qs[j] >> 4) as u32 | ((qh >> (j + 12)) & 0x10)) as f32;
                out[j] = d * (x0 - off) + m;
                out[j + 16] = d * (x1 - off) + m;
            }
        }
        DType::Q2K => {
            // scales[16] | qs[64] | d | dmin
            let (d, min) = (h(b, 80), h(b, 82));
            let mut y = 0;
            let mut is = 0;
            for n in 0..2 {
                let q = &b[16 + n * 32..];
                for shift in [0, 2, 4, 6] {
                    for half in 0..2 {
                        let sc = b[is];
                        is += 1;
                        let (dl, ml) = (d * (sc & 0xf) as f32, min * (sc >> 4) as f32);
                        for l in 0..16 {
                            out[y] = dl * ((q[l + 16 * half] >> shift) & 3) as f32 - ml;
                            y += 1;
                        }
                    }
                }
            }
        }
        DType::Q3K => {
            // hmask[32] | qs[64] | scales[12] | d
            let d = h(b, 108);
            let mut aux = [u32le(b, 96), u32le(b, 100), u32le(b, 104), 0];
            let tmp = aux[2];
            let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
            aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
            aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
            aux[0] = (aux[0] & k2) | ((tmp & k1) << 4);
            aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
            let scales: Vec<i8> = aux.iter().flat_map(|w| w.to_le_bytes()).map(|x| x as i8).collect();
            let hm = &b[0..32];
            let mut m = 1u8;
            let mut y = 0;
            let mut is = 0;
            for n in 0..2 {
                let q = &b[32 + n * 32..];
                for shift in [0, 2, 4, 6] {
                    for half in 0..2 {
                        let dl = d * (scales[is] as f32 - 32.0);
                        is += 1;
                        for l in 0..16 {
                            let li = l + 16 * half;
                            let v = ((q[li] >> shift) & 3) as i32 - if hm[li] & m != 0 { 0 } else { 4 };
                            out[y] = dl * v as f32;
                            y += 1;
                        }
                    }
                    m <<= 1;
                }
            }
        }
        DType::IQ2XXS => {
            let d = h(b, 0);
            for ib in 0..8 {
                let a0 = u32le(b, 2 + 8 * ib);
                let a1 = u32le(b, 6 + 8 * ib);
                let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
                for l in 0..4 {
                    let g = IQ2XXS_GRID[((a0 >> (8 * l)) & 0xff) as usize];
                    let s = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                    for j in 0..8 {
                        out[ib * 32 + l * 8 + j] = db * grid8(g, j) * sign(s, j);
                    }
                }
            }
        }
        DType::IQ2XS => {
            // d | qs[32] u16 | scales[8]
            let d = h(b, 0);
            for ib in 0..8 {
                let sc = b[66 + ib];
                let db = [d * (0.5 + (sc & 0xf) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25];
                for l in 0..4 {
                    let q = u16le(b, 2 + 2 * (4 * ib + l));
                    let g = IQ2XS_GRID[(q & 511) as usize];
                    let s = KSIGNS_IQ2XS[(q >> 9) as usize];
                    for j in 0..8 {
                        out[ib * 32 + l * 8 + j] = db[l / 2] * grid8(g, j) * sign(s, j);
                    }
                }
            }
        }
        DType::IQ2S => {
            // d | qs[64] (grid low bits, then signs) | qh[8] | scales[8]
            let d = h(b, 0);
            let qs = &b[2..34];
            let signs = &b[34..66];
            let qh = &b[66..74];
            for ib in 0..8 {
                let sc = b[74 + ib];
                let db = [d * (0.5 + (sc & 0xf) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25];
                for l in 0..4 {
                    let idx = qs[4 * ib + l] as usize | (((qh[ib] as usize) << (8 - 2 * l)) & 0x300);
                    let g = IQ2S_GRID[idx];
                    let s = signs[4 * ib + l];
                    for j in 0..8 {
                        out[ib * 32 + l * 8 + j] = db[l / 2] * grid8(g, j) * sign(s, j);
                    }
                }
            }
        }
        DType::IQ3XXS => {
            // d | qs[64] grid indices | 32 bytes scales and signs
            let d = h(b, 0);
            for ib in 0..8 {
                let aux = u32le(b, 66 + 4 * ib);
                let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let s = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                    let g1 = IQ3XXS_GRID[b[2 + 8 * ib + 2 * l] as usize];
                    let g2 = IQ3XXS_GRID[b[2 + 8 * ib + 2 * l + 1] as usize];
                    for j in 0..4 {
                        out[ib * 32 + l * 8 + j] = db * grid4(g1, j) * sign(s, j);
                        out[ib * 32 + l * 8 + j + 4] = db * grid4(g2, j) * sign(s, j + 4);
                    }
                }
            }
        }
        DType::IQ3S => {
            // d | qs[64] | qh[8] | signs[32] | scales[4]
            let d = h(b, 0);
            let (qs, qh, signs, scales) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
            for ib in 0..8 {
                let sc = scales[ib / 2];
                let db = d * (1 + 2 * if ib % 2 == 0 { sc & 0xf } else { sc >> 4 }) as f32;
                let hb = qh[ib] as usize;
                for l in 0..4 {
                    let g1 = IQ3S_GRID[qs[8 * ib + 2 * l] as usize | ((hb << (8 - 2 * l)) & 256)];
                    let g2 = IQ3S_GRID[qs[8 * ib + 2 * l + 1] as usize | ((hb << (7 - 2 * l)) & 256)];
                    let s = signs[4 * ib + l];
                    for j in 0..4 {
                        out[ib * 32 + l * 8 + j] = db * grid4(g1, j) * sign(s, j);
                        out[ib * 32 + l * 8 + j + 4] = db * grid4(g2, j) * sign(s, j + 4);
                    }
                }
            }
        }
        DType::IQ1S => {
            // d | qs[32] | qh[8] u16
            let d = h(b, 0);
            for ib in 0..8 {
                let qh = u16le(b, 34 + 2 * ib);
                let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
                let delta = if qh & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
                for l in 0..4 {
                    let g = IQ1S_GRID[b[2 + 4 * ib + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8)];
                    for j in 0..8 {
                        out[ib * 32 + l * 8 + j] = dl * (((g >> (8 * j)) & 0xff) as u8 as i8 as f32 + delta);
                    }
                }
            }
        }
        DType::IQ1M => {
            // qs[32] | qh[16] | scales[8]
            let sc = [u16le(b, 48), u16le(b, 50), u16le(b, 52), u16le(b, 54)];
            let dbits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            let d = f16_to_f32(dbits);
            for ib in 0..8 {
                let s = sc[ib / 2] >> (6 * (ib % 2));
                let dl = [d * (2 * (s & 7) + 1) as f32, d * (2 * ((s >> 3) & 7) + 1) as f32];
                let qs = &b[4 * ib..];
                let qh = &b[32 + 2 * ib..];
                let idx = [
                    qs[0] as usize | (((qh[0] as usize) << 8) & 0x700),
                    qs[1] as usize | (((qh[0] as usize) << 4) & 0x700),
                    qs[2] as usize | (((qh[1] as usize) << 8) & 0x700),
                    qs[3] as usize | (((qh[1] as usize) << 4) & 0x700),
                ];
                let delta = [qh[0] & 0x08, qh[0] & 0x80, qh[1] & 0x08, qh[1] & 0x80].map(|x| if x != 0 { -IQ1S_DELTA } else { IQ1S_DELTA });
                for l in 0..4 {
                    let g = IQ1S_GRID[idx[l]];
                    for j in 0..8 {
                        out[ib * 32 + l * 8 + j] = dl[l / 2] * (((g >> (8 * j)) & 0xff) as u8 as i8 as f32 + delta[l]);
                    }
                }
            }
        }
        DType::IQ4NL => {
            let d = h(b, 0);
            for j in 0..16 {
                out[j] = d * KVALUES_IQ4NL[(b[2 + j] & 0xf) as usize] as f32;
                out[j + 16] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
            }
        }
        DType::IQ4XS => {
            // d | scales_h u16 | scales_l[4] | qs[128]
            let d = h(b, 0);
            let sh = u16le(b, 2);
            for ib in 0..8 {
                let ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) as i32 | ((((sh >> (2 * ib)) & 3) as i32) << 4);
                let dl = d * (ls - 32) as f32;
                let qs = &b[8 + 16 * ib..];
                for j in 0..16 {
                    out[ib * 32 + j] = dl * KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as f32;
                    out[ib * 32 + j + 16] = dl * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
                }
            }
        }
        _ => unreachable!("format handled in quant.rs"),
    }
}
