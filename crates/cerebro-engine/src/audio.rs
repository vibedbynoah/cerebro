//! Audio encoder for Qwen3-ASR style projectors (`clip.projector_type = qwen3a`): 16 kHz speech
//! in, rows of the language model's width out, ready for `Session::feed_parts` as `Part::Audio`.
//!
//! Follows the reference implementation (llama.cpp `tools/mtmd/models/qwen3a.cpp` and the
//! `qwen3a` preprocessor in `mtmd-audio.cpp`, which matches Whisper's feature extractor): a 128-bin
//! log-mel spectrogram (25 ms periodic Hann windows every 10 ms, reflect-padded, log10, clamped to
//! 8 below the maximum and scaled), split into windows of 800 frames (8 s) encoded one at a time;
//! each window in chunks of 100 frames, each chunk through three 3x3 stride-2 convolutions with
//! GELU (13 tokens per chunk), a linear projection and per-chunk learned positions; then a
//! pre-norm transformer over the window's tokens and a two-layer projector. f32 throughout, on the
//! vision encoder's kernels.

use std::path::Path;

use crate::gguf::Gguf;
use crate::vision::{attention, f32s, layer_norm_all, lin, linear, norm, Lin};
use crate::Error;

/// Samples per second the encoder expects.
pub const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const CHUNK: usize = 100;
const WINDOW: usize = 800;
const TOKENS_PER_CHUNK: usize = 13;
const MEL_FLOOR: f64 = 5.960_464_477_539_063e-8;

/// The encoder's output: `tokens` rows of `dim` values, in time order.
#[derive(Debug, Clone)]
pub struct AudioEmbedding {
    pub tokens: usize,
    pub dim: usize,
    pub data: Vec<f32>,
}

struct Block {
    ln1: (Vec<f32>, Vec<f32>),
    qkv: Lin,
    out: Lin,
    ln2: (Vec<f32>, Vec<f32>),
    up: Lin,
    down: Lin,
}

#[cfg(feature = "cuda")]
#[path = "audio_gpu.rs"]
mod gpu;

pub struct AudioModel {
    /// The encoder on the graphics card, built on first use (None: no card this backend runs on).
    #[cfg(feature = "cuda")]
    gpu: std::sync::OnceLock<Option<gpu::GAudio>>,
    n_mel: usize,
    dim: usize,
    heads: usize,
    eps: f32,
    /// Output width (the language model's width).
    pub proj: usize,
    conv: [Lin; 3],
    conv_out: Lin,
    pos: Vec<f32>,
    blocks: Vec<Block>,
    post: (Vec<f32>, Vec<f32>),
    mm1: Lin,
    mm2: Lin,
    filters: Vec<f32>,
    hann: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

/// GELU with the exact error function (PyTorch's default), the reference's `gelu_erf`.
pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// erf to about 1.2e-7 (Numerical Recipes' Chebyshev fit of erfc).
pub fn erf(x: f32) -> f32 {
    let z = (x as f64).abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.265_512_23 + t * (1.000_023_68 + t * (0.374_091_96 + t * (0.096_784_18 + t * (-0.186_288_06 + t * (0.278_868_07 + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))))).exp();
    let v = 1.0 - r;
    (if x >= 0.0 { v } else { -v }) as f32
}

fn gelu_erf_all(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = gelu_erf(*v));
}

/// The Slaney mel filterbank with area normalisation (librosa's default), n_mel x (n_fft/2 + 1).
pub fn mel_filters(n_mel: usize, n_fft: usize, sample_rate: f64) -> Vec<f32> {
    let (min_log_hz, lin_slope) = (1000.0f64, 3.0 / 200.0);
    let min_log_mel = min_log_hz * lin_slope;
    let log_step = 6.4f64.ln() / 27.0;
    let hz_to_mel = |f: f64| if f < min_log_hz { f * lin_slope } else { min_log_mel + (f / min_log_hz).ln() / log_step };
    let mel_to_hz = |m: f64| if m < min_log_mel { m / lin_slope } else { min_log_hz * ((m - min_log_mel) * log_step).exp() };
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(sample_rate / 2.0));
    let hz: Vec<f64> = (0..n_mel + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mel + 1) as f64)).collect();
    let bins = n_fft / 2 + 1;
    let step = sample_rate / n_fft as f64;
    let mut out = vec![0f32; n_mel * bins];
    for m in 0..n_mel {
        let (l, c, r) = (hz[m], hz[m + 1], hz[m + 2]);
        let enorm = 2.0 / (r - l).max(1e-30);
        for k in 0..bins {
            let f = k as f64 * step;
            let w = if f >= l && f <= c {
                (f - l) / (c - l).max(1e-30)
            } else if f > c && f <= r {
                (r - f) / (r - c).max(1e-30)
            } else {
                0.0
            };
            out[m * bins + k] = (w * enorm) as f32;
        }
    }
    out
}

/// Mel spectrogram windows for the encoder: each `n_mel` rows of `frames` values (padded to a
/// multiple of the chunk length with zeros), at most `WINDOW` frames of audio each.
pub struct MelWindow {
    pub frames: usize,
    pub data: Vec<f32>,
}

impl AudioModel {
    /// Reads a projector file. Only Qwen3-ASR style projectors are supported for now.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let g = Gguf::open(path)?;
        let ptype = g.str("clip.audio.projector_type").or(g.str("clip.projector_type")).unwrap_or("");
        if ptype != "qwen3a" {
            return Err(Error::Unsupported(format!("This audio projector type is not supported yet: {ptype}")));
        }
        let u = |k: &str| g.u64(&format!("clip.audio.{k}")).map(|v| v as usize);
        let (n_mel, dim, heads) = (u("num_mel_bins")?, u("embedding_length")?, u("attention.head_count")?);
        if dim % heads != 0 || (dim / heads) % 16 != 0 {
            return Err(Error::Unsupported(format!("audio shape (width {dim}, heads {heads})")));
        }
        let eps = g.f32("clip.audio.attention.layer_norm_epsilon").unwrap_or(1e-5);
        // Convolutions as matrices over (input channel, kernel row, kernel column).
        let conv_lin = |i: usize| -> Result<Lin, Error> {
            let t = g.tensor(&format!("a.conv2d.{i}.weight"))?;
            let out = t.dims[3] as usize;
            let w = f32s(&g, &format!("a.conv2d.{i}.weight"))?;
            let bias = f32s(&g, &format!("a.conv2d.{i}.bias"))?;
            Ok(Lin { rows: out, cols: w.len() / out, w, bias })
        };
        let conv = [conv_lin(1)?, conv_lin(2)?, conv_lin(3)?];
        let conv_out = lin(&g, "a.conv_out")?;
        let pos = f32s(&g, "a.position_embd.weight")?;
        let n_layer = u("block_count")?;
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("a.blk.{i}");
            // q, k and v stacked into one matrix, as the attention kernel takes them.
            let (q, k, v) = (lin(&g, &format!("{p}.attn_q"))?, lin(&g, &format!("{p}.attn_k"))?, lin(&g, &format!("{p}.attn_v"))?);
            let qkv = Lin { rows: 3 * dim, cols: dim, w: [q.w, k.w, v.w].concat(), bias: [q.bias, k.bias, v.bias].concat() };
            blocks.push(Block {
                ln1: norm(&g, &format!("{p}.ln1"))?,
                qkv,
                out: lin(&g, &format!("{p}.attn_out"))?,
                ln2: norm(&g, &format!("{p}.ln2"))?,
                up: lin(&g, &format!("{p}.ffn_up"))?,
                down: lin(&g, &format!("{p}.ffn_down"))?,
            });
        }
        let post = norm(&g, "a.post_ln")?;
        let mm1 = lin(&g, "mm.a.mlp.1")?;
        let mm2 = lin(&g, "mm.a.mlp.2")?;
        if conv_out.rows != dim || conv[0].cols != 9 || mm1.cols != dim {
            return Err(Error::Format("audio projector has an unexpected shape".into()));
        }
        let hann: Vec<f32> = (0..N_FFT).map(|i| (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos())) as f32).collect();
        let cos = (0..N_FFT).map(|i| (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos() as f32).collect();
        let sin = (0..N_FFT).map(|i| (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).sin() as f32).collect();
        Ok(AudioModel {
            #[cfg(feature = "cuda")]
            gpu: std::sync::OnceLock::new(),
            n_mel,
            dim,
            heads, eps, proj: mm2.rows, conv, conv_out, pos, blocks, post, mm1, mm2, filters: mel_filters(n_mel, N_FFT, SAMPLE_RATE as f64), hann, cos, sin })
    }

    /// The log-mel windows of 16 kHz mono samples.
    pub fn mel(&self, samples: &[f32]) -> Vec<MelWindow> {
        let n = samples.len();
        if n == 0 {
            return Vec::new();
        }
        let pad = N_FFT / 2;
        let mut x = vec![0f32; n + 2 * pad];
        for i in 0..pad {
            x[i] = samples.get(pad - i).copied().unwrap_or(0.0);
            x[n + pad + i] = if n >= 2 + i { samples[n - 2 - i] } else { 0.0 };
        }
        x[pad..pad + n].copy_from_slice(samples);
        let total = x.len();
        let n_len = (total - N_FFT) / HOP + 1;
        let computed = n_len.min(total / HOP + 1);
        let bins = N_FFT / 2 + 1;
        let n_mel = self.n_mel;
        let mut mel = vec![0f32; n_mel * n_len];
        {
            let out = crate::vision::Ptr::new(mel.as_mut_ptr());
            crate::pool::global().run(computed.div_ceil(32), &|ti| {
                let mut frame = vec![0f32; N_FFT];
                let mut power = vec![0f64; bins];
                for i in ti * 32..((ti + 1) * 32).min(computed) {
                    let off = i * HOP;
                    for j in 0..N_FFT {
                        frame[j] = if off + j < total { self.hann[j] * x[off + j] } else { 0.0 };
                    }
                    for (k, pk) in power.iter_mut().enumerate() {
                        let (mut re, mut im) = (0f64, 0f64);
                        for (j, &v) in frame.iter().enumerate() {
                            let a = (k * j) % N_FFT;
                            re += (v * self.cos[a]) as f64;
                            im -= (v * self.sin[a]) as f64;
                        }
                        *pk = re * re + im * im;
                    }
                    for m in 0..n_mel {
                        let row = &self.filters[m * bins..(m + 1) * bins];
                        let s: f64 = row.iter().zip(&power).map(|(f, p)| *f as f64 * p).sum();
                        // SAFETY: each frame (column) is written by one task.
                        unsafe { *out.at(m * n_len + i) = s.max(MEL_FLOOR).log10() as f32 };
                    }
                }
            });
        }
        for i in computed..n_len {
            for m in 0..n_mel {
                mel[m * n_len + i] = (1e-10f64).log10() as f32;
            }
        }
        let max = mel.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) - 8.0;
        mel.iter_mut().for_each(|v| *v = (v.max(max) + 4.0) / 4.0);
        let eff = n_len.min(n / HOP + 1);
        let mut out = Vec::new();
        let mut off = 0;
        while off < eff {
            let len = WINDOW.min(eff - off);
            let frames = len.div_ceil(CHUNK) * CHUNK;
            let mut data = vec![0f32; n_mel * frames];
            for m in 0..n_mel {
                let take = len.min(n_len - off);
                data[m * frames..m * frames + take].copy_from_slice(&mel[m * n_len + off..m * n_len + off + take]);
            }
            out.push(MelWindow { frames, data });
            off += WINDOW;
        }
        out
    }

    /// Encodes 16 kHz mono samples: on the graphics card when there is one the tensor backend
    /// runs on (CEREBRO_TENSOR_CPU forces the processor), otherwise on the processor.
    pub fn encode(&self, samples: &[f32]) -> Result<AudioEmbedding, Error> {
        #[cfg(feature = "cuda")]
        if let Some(g) = crate::gtensor::Gx::get() {
            let enc = self.gpu.get_or_init(|| gpu::GAudio::new(g, self).map_err(|e| eprintln!("speech encoder on the graphics card: {e}")).ok());
            if let Some(enc) = enc {
                return self.encode_with(samples, &|w| enc.encode_window(g, self, w));
            }
        }
        self.encode_cpu(samples)
    }

    /// `encode` on the processor.
    pub fn encode_cpu(&self, samples: &[f32]) -> Result<AudioEmbedding, Error> {
        self.encode_with(samples, &|w| self.encode_window(w))
    }

    fn encode_with(&self, samples: &[f32], window: &dyn Fn(&MelWindow) -> Result<Vec<f32>, Error>) -> Result<AudioEmbedding, Error> {
        let windows = self.mel(samples);
        if windows.is_empty() {
            return Err(Error::Request("the recording is empty".into()));
        }
        let mut data = Vec::new();
        let mut tokens = 0;
        for w in &windows {
            let e = window(w)?;
            tokens += e.len() / self.proj;
            data.extend(e);
        }
        Ok(AudioEmbedding { tokens, dim: self.proj, data })
    }

    /// One window: chunks through the convolutions, then the transformer over all its tokens.
    fn encode_window(&self, w: &MelWindow) -> Result<Vec<f32>, Error> {
        let (n_mel, dim) = (self.n_mel, self.dim);
        let n_chunks = w.frames / CHUNK;
        let n = n_chunks * TOKENS_PER_CHUNK;
        let mut x = vec![0f32; n * dim];
        for c in 0..n_chunks {
            // [1][mel][frame] for this chunk, as positions x channels.
            let mut act: Vec<f32> = Vec::with_capacity(n_mel * CHUNK);
            for m in 0..n_mel {
                act.extend_from_slice(&w.data[m * w.frames + c * CHUNK..m * w.frames + (c + 1) * CHUNK]);
            }
            let (mut h, mut wd, mut ch) = (n_mel, CHUNK, 1usize);
            // `act` is [h][w][ch] after the first convolution; the input is [ch=1][h][w] = [h][w][1].
            for conv in &self.conv {
                let (oh, ow) = ((h + 2 - 3) / 2 + 1, (wd + 2 - 3) / 2 + 1);
                let cols = ch * 9;
                let mut patches = vec![0f32; oh * ow * cols];
                for y in 0..oh {
                    for xo in 0..ow {
                        let row = &mut patches[(y * ow + xo) * cols..(y * ow + xo + 1) * cols];
                        for ic in 0..ch {
                            for kh in 0..3 {
                                let iy = (2 * y + kh) as isize - 1;
                                for kw in 0..3 {
                                    let ix = (2 * xo + kw) as isize - 1;
                                    if iy >= 0 && (iy as usize) < h && ix >= 0 && (ix as usize) < wd {
                                        row[(ic * 3 + kh) * 3 + kw] = act[((iy as usize) * wd + ix as usize) * ch + ic];
                                    }
                                }
                            }
                        }
                    }
                }
                let mut out = vec![0f32; oh * ow * conv.rows];
                linear(conv, &patches, oh * ow, &mut out, false);
                gelu_erf_all(&mut out);
                act = out;
                (h, wd, ch) = (oh, ow, conv.rows);
            }
            // Tokens are the frame axis; features run channel-major over the mel axis (c * h + y).
            let feat = ch * h;
            let mut flat = vec![0f32; wd * feat];
            for t in 0..wd {
                for oc in 0..ch {
                    for y in 0..h {
                        flat[t * feat + oc * h + y] = act[(y * wd + t) * ch + oc];
                    }
                }
            }
            if wd != TOKENS_PER_CHUNK || feat != self.conv_out.cols {
                return Err(Error::Format("audio convolutions gave an unexpected shape".into()));
            }
            let dst = &mut x[c * TOKENS_PER_CHUNK * dim..(c + 1) * TOKENS_PER_CHUNK * dim];
            linear(&self.conv_out, &flat, wd, dst, false);
            for t in 0..wd {
                for d in 0..dim {
                    dst[t * dim + d] += self.pos[t * dim + d];
                }
            }
        }
        let hd = dim / self.heads;
        let mut h = vec![0f32; n * dim];
        let mut qkv = vec![0f32; n * 3 * dim];
        let mut att = vec![0f32; n * dim];
        let mut ff = vec![0f32; n * self.blocks.first().map_or(0, |b| b.up.rows)];
        for b in &self.blocks {
            layer_norm_all(&x, &b.ln1.0, &b.ln1.1, self.eps, dim, &mut h);
            linear(&b.qkv, &h, n, &mut qkv, false);
            attention(&qkv, n, self.heads, hd, &mut att);
            linear(&b.out, &att, n, &mut x, true);
            layer_norm_all(&x, &b.ln2.0, &b.ln2.1, self.eps, dim, &mut h);
            linear(&b.up, &h, n, &mut ff, false);
            gelu_erf_all(&mut ff);
            linear(&b.down, &ff, n, &mut x, true);
        }
        layer_norm_all(&x, &self.post.0, &self.post.1, self.eps, dim, &mut h);
        let mut mid = vec![0f32; n * self.mm1.rows];
        linear(&self.mm1, &h, n, &mut mid, false);
        gelu_erf_all(&mut mid);
        let mut out = vec![0f32; n * self.proj];
        linear(&self.mm2, &mid, n, &mut out, false);
        Ok(out)
    }
}

/// 16-bit PCM or 32-bit float WAV to mono f32 at the file's rate: (samples, rate).
pub fn read_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32), Error> {
    let bad = || Error::Request("not a WAV file this reader understands (PCM 16-bit or float 32-bit)".into());
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad());
    }
    let (mut fmt, mut data) = (None, None);
    let mut i = 12;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let len = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        let body = &bytes[i + 8..(i + 8 + len).min(bytes.len())];
        match id {
            b"fmt " => fmt = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        i += 8 + len + (len & 1);
    }
    let (fmt, data) = (fmt.ok_or_else(bad)?, data.ok_or_else(bad)?);
    if fmt.len() < 16 {
        return Err(bad());
    }
    let format = u16::from_le_bytes([fmt[0], fmt[1]]);
    let channels = u16::from_le_bytes([fmt[2], fmt[3]]).max(1) as usize;
    let rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
    let bits = u16::from_le_bytes([fmt[14], fmt[15]]);
    let frames: Vec<f32> = match (format, bits) {
        (1, 16) => data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect(),
        (3, 32) => data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        _ => return Err(bad()),
    };
    let mono = frames.chunks(channels).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect();
    Ok((mono, rate))
}

/// Linear resampling to `to` Hz (speech is band-limited well below the 8 kHz the encoder keeps).
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64 / from as u64) as usize;
    (0..n)
        .map(|i| {
            let p = i as f64 * from as f64 / to as f64;
            let j = p.floor() as usize;
            let f = (p - j as f64) as f32;
            let a = x[j.min(x.len() - 1)];
            let b = x[(j + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_and_gelu_match_known_values() {
        assert!((erf(0.5) - 0.520_499_9).abs() < 1e-6);
        assert!((erf(-1.5) + 0.966_105_1).abs() < 1e-6);
        assert!((gelu_erf(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu_erf(-2.0) + 0.045_500_26).abs() < 1e-6);
    }

    #[test]
    fn mel_filters_are_triangles_with_slaney_area() {
        let f = mel_filters(128, 400, 16000.0);
        let bins = 201;
        assert_eq!(f.len(), 128 * bins);
        // Each filter is non-negative and peaks once; the lowest one starts at bin 0.
        for m in 0..128 {
            let row = &f[m * bins..(m + 1) * bins];
            assert!(row.iter().all(|v| *v >= 0.0));
            assert!(row.iter().any(|v| *v > 0.0), "filter {m} is empty");
        }
        assert_eq!(f[0], 0.0);
    }

    #[test]
    fn wav_reading_and_resampling() {
        let mut w = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes()); // PCM
        w.extend_from_slice(&1u16.to_le_bytes()); // mono
        w.extend_from_slice(&8000u32.to_le_bytes());
        w.extend_from_slice(&16000u32.to_le_bytes());
        w.extend_from_slice(&2u16.to_le_bytes());
        w.extend_from_slice(&16u16.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&4u32.to_le_bytes());
        w.extend_from_slice(&16384i16.to_le_bytes());
        w.extend_from_slice(&(-16384i16).to_le_bytes());
        let (s, rate) = read_wav(&w).expect("wav");
        assert_eq!((rate, s.len()), (8000, 2));
        assert!((s[0] - 0.5).abs() < 1e-6 && (s[1] + 0.5).abs() < 1e-6);
        assert_eq!(resample(&s, 8000, 16000).len(), 4);
        assert!(read_wav(b"nope").is_err());
    }
}
