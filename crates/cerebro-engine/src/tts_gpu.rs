//! The speech decoder (code2wav) on the graphics card (`gtensor`): the same stages as
//! `TtsDecoder::decode`, with every matrix read from the file again in f32 and uploaded as f16
//! (the processor copy keeps the file's 8-bit packing for its own kernels).

use super::{conv_w, f32s, transpose_weight, Rng, TtsDecoder, SWA};
use crate::gguf::Gguf;
use crate::gtensor::{Act, DBuf, DConv, Gx, DW};
use crate::vision::Lin;
use crate::Error;

/// A matrix in f32 from a tensor (ggml layout: input dimension first) and its bias, if any.
fn lin(g: &Gguf, name: &str) -> Result<Lin, Error> {
    lin_part(g, name, 0)
}

/// Matrix `i` of a stacked [cols, rows, n] tensor.
fn lin_part(g: &Gguf, name: &str, i: usize) -> Result<Lin, Error> {
    let t = g.tensor(name)?;
    let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let bias = match name.strip_suffix(".weight") {
        Some(base) if g.tensors.contains_key(&format!("{base}.bias")) => f32s(g, &format!("{base}.bias"))?,
        _ => vec![0.0; rows],
    };
    let w = f32s(g, name)?;
    Ok(Lin { rows, cols, w: w[i * rows * cols..(i + 1) * rows * cols].to_vec(), bias })
}

struct GLayer {
    ln1: DBuf,
    q: DW,
    k: DW,
    v: DW,
    o: DW,
    q_norm: Option<DBuf>,
    k_norm: Option<DBuf>,
    ln2: DBuf,
    gate: DW,
    up: DW,
    down: DW,
    ls1: Option<DBuf>,
    ls2: Option<DBuf>,
}

fn glayer(x: &Gx, g: &Gguf, p: &str) -> Result<GLayer, Error> {
    let w = |n: &str| x.weights(&lin(g, &format!("{p}.{n}.weight"))?);
    let v = |n: &str| x.up(&f32s(g, &format!("{p}.{n}.weight"))?);
    let opt = |n: &str| -> Result<Option<DBuf>, Error> { if g.tensors.contains_key(&format!("{p}.{n}.weight")) { v(n).map(Some) } else { Ok(None) } };
    Ok(GLayer {
        ln1: v("ln1")?,
        q: w("attn_q")?,
        k: w("attn_k")?,
        v: w("attn_v")?,
        o: w("attn_out")?,
        q_norm: opt("attn_q_norm")?,
        k_norm: opt("attn_k_norm")?,
        ln2: v("ln2")?,
        gate: w("ffn_gate")?,
        up: w("ffn_up")?,
        down: w("ffn_down")?,
        ls1: opt("ls1")?,
        ls2: opt("ls2")?,
    })
}

/// `layers_forward` for a whole sequence from position 0 (no cache): heads == kv heads.
#[allow(clippy::too_many_arguments)]
fn layers(x: &Gx, layers: &[GLayer], h: &mut DBuf, n: usize, dim: usize, heads: usize, eps: f32, theta: f32, window: usize) -> Result<(), Error> {
    let Some(l0) = layers.first() else { return Ok(()) };
    let qd = l0.q.rows;
    let hd = qd / heads;
    let half = hd / 2;
    let mut cs = Vec::with_capacity(n * hd);
    for p in 0..n {
        for i in 0..half {
            let (s, c) = (p as f32 * theta.powf(-2.0 * i as f32 / hd as f32)).sin_cos();
            cs.extend([c, s]);
        }
    }
    let cs = x.up(&cs)?;
    for l in layers {
        if l.k.rows != qd {
            return Err(Error::Unsupported("grouped attention in the speech decoder on the graphics card".into()));
        }
        let t = x.row_norm(h, n, dim, &l.ln1, None, eps, true)?;
        let (mut q, mut k, mut v) = (x.buf(n * qd)?, x.buf(n * qd)?, x.buf(n * qd)?);
        x.linear(&t, n, &l.q, &mut q, false)?;
        x.linear(&t, n, &l.k, &mut k, false)?;
        x.linear(&t, n, &l.v, &mut v, false)?;
        if let Some(w) = &l.q_norm {
            q = x.row_norm(&q, n * heads, hd, w, None, eps, true)?;
        }
        if let Some(w) = &l.k_norm {
            k = x.row_norm(&k, n * heads, hd, w, None, eps, true)?;
        }
        x.rope_halves(&mut q, &cs, n, heads, hd, qd, 1)?;
        x.rope_halves(&mut k, &cs, n, heads, hd, qd, 1)?;
        let mut att = x.buf(n * qd)?;
        x.attention_window(&q.slice(..), qd, &k.slice(..), qd, &v.slice(..), qd, n, n, heads, hd, true, window, &mut att)?;
        let mut o = x.buf(n * dim)?;
        x.linear(&att, n, &l.o, &mut o, false)?;
        x.add_scaled(h, &o, l.ls1.as_ref(), n * dim, dim)?;
        let t = x.row_norm(h, n, dim, &l.ln2, None, eps, true)?;
        let ff = l.gate.rows;
        let (mut gt, mut ut) = (x.buf(n * ff)?, x.buf(n * ff)?);
        x.linear(&t, n, &l.gate, &mut gt, false)?;
        x.linear(&t, n, &l.up, &mut ut, false)?;
        x.swiglu(&mut gt, &ut, n * ff)?;
        let mut dn = x.buf(n * dim)?;
        x.linear(&gt, n, &l.down, &mut dn, false)?;
        x.add_scaled(h, &dn, l.ls2.as_ref(), n * dim, dim)?;
    }
    Ok(())
}

struct GUp {
    conv: (DW, usize, usize, DBuf),
    dw: DBuf,
    dw_b: DBuf,
    norm: (DBuf, DBuf),
    pw1: DW,
    pw2: DW,
    gamma: DBuf,
}

struct GRes {
    a1: (DBuf, DBuf),
    conv1: DConv,
    a2: (DBuf, DBuf),
    conv2: DConv,
}

pub(super) struct GDecoder {
    q_first_out: DW,
    q_rest_out: DW,
    pre_conv: DConv,
    tfm_in: DW,
    tfm: Vec<GLayer>,
    tfm_norm: DBuf,
    tfm_out: DW,
    up: Vec<GUp>,
    dac_entry: DConv,
    dac: Vec<((DBuf, DBuf), (DW, usize, usize, DBuf), usize, Vec<GRes>)>,
    post_snake: (DBuf, DBuf),
    post_conv: DConv,
}

impl GDecoder {
    pub(super) fn load(x: &Gx, path: &std::path::Path) -> Result<Self, Error> {
        let g = Gguf::open(path)?;
        let v = |n: &str| x.up(&f32s(&g, n)?);
        let conv = |n: &str, dil: usize| -> Result<DConv, Error> {
            let c = conv_w(&g, n)?;
            x.conv1d(&c.w, c.ic, c.k, dil)
        };
        let tconv = |n: &str| -> Result<(DW, usize, usize, DBuf), Error> {
            let (w2, oc, k, b) = transpose_weight(&g, n)?;
            Ok((x.weights(&w2)?, oc, k, x.up(&b)?))
        };
        let mut up = Vec::new();
        for i in 0.. {
            let p = format!("a.gen.wav.up.blk.{i}");
            if !g.tensors.contains_key(&format!("{p}.conv.weight")) {
                break;
            }
            up.push(GUp {
                conv: tconv(&format!("{p}.conv"))?,
                dw: v(&format!("{p}.dwconv.weight"))?,
                dw_b: v(&format!("{p}.dwconv.bias"))?,
                norm: (v(&format!("{p}.norm.weight"))?, v(&format!("{p}.norm.bias"))?),
                pw1: x.weights(&lin(&g, &format!("{p}.pw1.weight"))?)?,
                pw2: x.weights(&lin(&g, &format!("{p}.pw2.weight"))?)?,
                gamma: v(&format!("{p}.gamma"))?,
            });
        }
        let mut dac = Vec::new();
        for i in 0.. {
            let p = format!("a.gen.wav.dac.blk.{i}");
            if !g.tensors.contains_key(&format!("{p}.conv.weight")) {
                break;
            }
            let upc = tconv(&format!("{p}.conv"))?;
            let stride = upc.2 / 2;
            let mut res = Vec::new();
            for (j, dil) in [1usize, 3, 9].into_iter().enumerate() {
                let r = format!("{p}.res.{j}");
                res.push(GRes {
                    a1: (v(&format!("{r}.act1.alpha"))?, v(&format!("{r}.act1.beta"))?),
                    conv1: conv(&format!("{r}.conv1"), dil)?,
                    a2: (v(&format!("{r}.act2.alpha"))?, v(&format!("{r}.act2.beta"))?),
                    conv2: conv(&format!("{r}.conv2"), 1)?,
                });
            }
            dac.push(((v(&format!("{p}.snake.alpha"))?, v(&format!("{p}.snake.beta"))?), upc, stride, res));
        }
        Ok(GDecoder {
            q_first_out: x.weights(&lin(&g, "a.gen.wav.quant.first.out_proj.weight")?)?,
            q_rest_out: x.weights(&lin(&g, "a.gen.wav.quant.rest.out_proj.weight")?)?,
            pre_conv: conv("a.gen.wav.pre_conv", 1)?,
            tfm_in: x.weights(&lin(&g, "a.gen.wav.tfm.in_proj.weight")?)?,
            tfm: (0..8).map(|i| glayer(x, &g, &format!("a.gen.wav.tfm.blk.{i}"))).collect::<Result<Vec<_>, _>>()?,
            tfm_norm: v("a.gen.wav.tfm.output_norm.weight")?,
            tfm_out: x.weights(&lin(&g, "a.gen.wav.tfm.out_proj.weight")?)?,
            up,
            dac_entry: conv("a.gen.wav.dac.entry", 1)?,
            dac,
            post_snake: (v("a.gen.wav.dac.post_snake.alpha")?, v("a.gen.wav.dac.post_snake.beta")?),
            post_conv: conv("a.gen.wav.dac.post_conv", 1)?,
        })
    }

    pub(super) fn decode(&self, x: &Gx, d: &TtsDecoder, frames: &[Vec<usize>]) -> Result<Vec<f32>, Error> {
        let n = frames.len();
        let qd = d.q_dim;
        // 1. Codebooks: the first through its projection, the other 15 summed then projected.
        let (mut sem, mut ac) = (vec![0f32; n * qd], vec![0f32; n * qd]);
        for (t, f) in frames.iter().enumerate() {
            sem[t * qd..(t + 1) * qd].copy_from_slice(TtsDecoder::row(&d.q_first, f[0], qd));
            for (gi, table) in d.q_rest.iter().enumerate() {
                let r = TtsDecoder::row(table, f[gi + 1], qd);
                for k in 0..qd {
                    ac[t * qd + k] += r[k];
                }
            }
        }
        let mut h = x.buf(n * self.q_first_out.rows)?;
        x.linear(&x.up(&sem)?, n, &self.q_first_out, &mut h, false)?;
        x.linear(&x.up(&ac)?, n, &self.q_rest_out, &mut h, true)?;
        // 2. Causal convolution, 3. the sliding-window transformer.
        let c0 = x.conv1d_run(&h, n, &self.pre_conv)?;
        let td = self.tfm_in.rows;
        let mut t = x.buf(n * td)?;
        x.linear(&c0, n, &self.tfm_in, &mut t, false)?;
        layers(x, &self.tfm, &mut t, n, td, 16, 1e-5, 10_000.0, SWA)?;
        let t = x.row_norm(&t, n, td, &self.tfm_norm, None, 1e-5, true)?;
        let mut c = self.tfm_out.rows;
        let mut y = x.buf(n * c)?;
        x.linear(&t, n, &self.tfm_out, &mut y, false)?;
        let mut len = n;
        // 4. Two 2x upsampling stages, each a transposed convolution and a ConvNeXt block.
        for u in &self.up {
            let (w2, oc, k, b) = &u.conv;
            y = x.conv_transpose(&y, len, w2, *oc, *k, 2, b)?;
            len *= 2;
            c = *oc;
            let kdw = u.dw.len() / c;
            let z = x.dwconv_causal(&y, len, c, &u.dw, &u.dw_b, kdw)?;
            let z = x.row_norm(&z, len, c, &u.norm.0, Some(&u.norm.1), 1e-6, false)?;
            let mut mid = x.buf(len * u.pw1.rows)?;
            x.linear(&z, len, &u.pw1, &mut mid, false)?;
            x.act(&mut mid, len * u.pw1.rows, Act::GeluTanh)?;
            let mut o = x.buf(len * c)?;
            x.linear(&mid, len, &u.pw2, &mut o, false)?;
            x.add_scaled(&mut y, &o, Some(&u.gamma), len * c, c)?;
        }
        // 5. The DAC decoder.
        y = x.conv1d_run(&y, len, &self.dac_entry)?;
        c = self.dac_entry.w.rows;
        for (sn, (w2, oc, k, b), stride, res) in &self.dac {
            let s = x.snake(&y, len * c, c, &sn.0, &sn.1)?;
            y = x.conv_transpose(&s, len, w2, *oc, *k, *stride, b)?;
            len *= stride;
            c = *oc;
            for r in res {
                let hh = x.snake(&y, len * c, c, &r.a1.0, &r.a1.1)?;
                let hh = x.conv1d_run(&hh, len, &r.conv1)?;
                let hh = x.snake(&hh, len * c, c, &r.a2.0, &r.a2.1)?;
                let hh = x.conv1d_run(&hh, len, &r.conv2)?;
                x.add_scaled(&mut y, &hh, None, len * c, c)?;
            }
        }
        let s = x.snake(&y, len * c, c, &self.post_snake.0, &self.post_snake.1)?;
        let out = x.conv1d_run(&s, len, &self.post_conv)?;
        Ok(x.down(&out)?.into_iter().map(|v| v.clamp(-1.0, 1.0)).collect())
    }
}

/// One predictor layer with its matrices fused for one-token steps: [q | k | v] and [gate | up].
struct PLayer {
    ln1: DBuf,
    qkv: DW,
    o: DW,
    q_norm: Option<DBuf>,
    k_norm: Option<DBuf>,
    ln2: DBuf,
    gate_up: DW,
    down: DW,
}

/// Stacks matrices with the same column count by rows.
fn stack(parts: &[Lin]) -> Lin {
    let cols = parts[0].cols;
    let mut w = Vec::new();
    let mut bias = Vec::new();
    for p in parts {
        w.extend_from_slice(&p.w);
        bias.extend_from_slice(&p.bias);
    }
    Lin { rows: parts.iter().map(|p| p.rows).sum(), cols, w, bias }
}

/// The code predictor on the card: one token per step through its layers, with the keys and
/// values of the frame's (at most 17) positions kept on the card, sampling on the card too.
pub(super) struct GPredictor {
    /// proj_in applied to every row of each acoustic codebook's table, and of the first
    /// codebook's (vocab x dim each): a step's input is one row of these.
    tables: Vec<DBuf>,
    out_table: DBuf,
    layers: Vec<PLayer>,
    proj_in: DW,
    norm: DBuf,
    heads_w: Vec<DW>,
    /// Rotary (cos, sin) pairs for each position of a frame.
    cs: Vec<DBuf>,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    /// Step buffers, reused (the predictor runs on one thread).
    scratch: std::sync::Mutex<Option<Scratch>>,
}

struct Scratch {
    h: DBuf,
    qkv: DBuf,
    att: DBuf,
    gu: DBuf,
    logits: DBuf,
    cache: Vec<(DBuf, DBuf)>,
    codes: cudarc::driver::CudaSlice<i32>,
}

impl GPredictor {
    pub(super) fn load(x: &Gx, path: &std::path::Path, d: &TtsDecoder) -> Result<Self, Error> {
        let g = Gguf::open(path)?;
        let mut layers = Vec::new();
        for i in 0..d.cp_layers.len() {
            let p = format!("a.gen.code.blk.{i}");
            let l = |n: &str| lin(&g, &format!("{p}.{n}.weight"));
            let v = |n: &str| x.up(&f32s(&g, &format!("{p}.{n}.weight"))?);
            let opt = |n: &str| -> Result<Option<DBuf>, Error> { if g.tensors.contains_key(&format!("{p}.{n}.weight")) { v(n).map(Some) } else { Ok(None) } };
            if g.tensors.contains_key(&format!("{p}.ls1.weight")) || g.tensors.contains_key(&format!("{p}.ls2.weight")) {
                return Err(Error::Unsupported("layer scales in the code predictor on the graphics card".into()));
            }
            layers.push(PLayer {
                ln1: v("ln1")?,
                qkv: x.weights(&stack(&[l("attn_q")?, l("attn_k")?, l("attn_v")?]))?,
                o: x.weights(&l("attn_out")?)?,
                q_norm: opt("attn_q_norm")?,
                k_norm: opt("attn_k_norm")?,
                ln2: v("ln2")?,
                gate_up: x.weights(&stack(&[l("ffn_gate")?, l("ffn_up")?]))?,
                down: x.weights(&l("ffn_down")?)?,
            });
        }
        let heads_w = (0..d.cp_heads_w.len()).map(|i| x.weights(&lin_part(&g, "a.gen.code.head.weight", i)?)).collect::<Result<Vec<_>, _>>()?;
        let hd = layers.first().map_or(0, |l| l.o.cols / d.cp_heads);
        let half = hd / 2;
        let cs = (0..d.cp_heads_w.len() + 2)
            .map(|p| {
                let row: Vec<f32> = (0..half)
                    .flat_map(|i| {
                        let (s, c) = (p as f32 * 1_000_000f32.powf(-2.0 * i as f32 / hd as f32)).sin_cos();
                        [c, s]
                    })
                    .collect();
                x.up(&row)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let proj_in = x.weights(&lin(&g, "a.gen.code.proj_in.weight")?)?;
        let td = d.talker_dim;
        let project = |t: &[f32]| -> Result<DBuf, Error> {
            let n = t.len() / td;
            let mut out = x.buf(n * proj_in.rows)?;
            x.linear(&x.up(t)?, n, &proj_in, &mut out, false)?;
            Ok(out)
        };
        let tables = d.cp_embd.iter().map(|t| project(t)).collect::<Result<Vec<_>, _>>()?;
        let out_table = project(&d.out_embd)?;
        Ok(GPredictor {
            tables,
            out_table,
            layers,
            proj_in,
            norm: x.up(&f32s(&g, "a.gen.code.output_norm.weight")?)?,
            heads_w,
            cs,
            heads: d.cp_heads,
            kv_heads: d.cp_kv_heads,
            hd,
            scratch: std::sync::Mutex::new(None),
        })
    }

    /// One position (`sc.h`, already projected: the model's width) at `pos` through the layers.
    fn step(&self, x: &Gx, d: &TtsDecoder, sc: &mut Scratch, pos: usize) -> Result<(), Error> {
        let eps = d.cp_eps;
        for (l, (kc, vc)) in self.layers.iter().zip(sc.cache.iter_mut()) {
            x.gemv(&l.qkv, &sc.h, &mut sc.qkv, false, Some((&l.ln1, eps)), false)?;
            x.qk_norm_rope_cache(&mut sc.qkv, l.q_norm.as_ref(), l.k_norm.as_ref(), &self.cs[pos], kc, vc, pos, self.heads, self.kv_heads, self.hd, eps)?;
            x.attend_one(&sc.qkv, kc, vc, pos + 1, self.heads, self.kv_heads, self.hd, &mut sc.att)?;
            x.gemv(&l.o, &sc.att, &mut sc.h, true, None, false)?;
            x.gemv(&l.gate_up, &sc.h, &mut sc.gu, false, Some((&l.ln2, eps)), false)?;
            x.gemv(&l.down, &sc.gu, &mut sc.h, true, None, true)?;
        }
        Ok(())
    }

    /// `TtsDecoder::predict_frame` on the card: the 15 draws on the card too (their uniforms drawn
    /// here in the processor sampler's order), so the frame waits for the card once.
    pub(super) fn predict_frame(&self, x: &Gx, d: &TtsDecoder, h_state: &[f32], code0: usize, top_k: usize, top_p: f32, rng: &mut Rng) -> Result<(Vec<usize>, Vec<f32>), Error> {
        if top_p < 1.0 {
            return Err(Error::Unsupported("top-p sampling in the code predictor on the graphics card".into()));
        }
        let td = d.talker_dim;
        let n_ac = self.heads_w.len();
        let (dim, kvd) = (self.proj_in.rows, self.kv_heads * self.hd);
        let mut guard = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(Scratch {
                h: x.zeros(dim)?,
                qkv: x.zeros(self.layers[0].qkv.rows)?,
                att: x.zeros(self.heads * self.hd)?,
                gu: x.zeros(self.layers[0].gate_up.rows)?,
                logits: x.zeros(d.code_vocab)?,
                cache: (0..self.layers.len()).map(|_| Ok((x.zeros((n_ac + 2) * kvd)?, x.zeros((n_ac + 2) * kvd)?))).collect::<Result<Vec<_>, Error>>()?,
                codes: x.codes(n_ac)?,
            });
        }
        let sc = guard.as_mut().expect("scratch");
        let us: Vec<f32> = (0..n_ac).map(|_| rng.next_f32()).collect();
        let u = x.up(&us)?;
        x.linear(&x.up(h_state)?, 1, &self.proj_in, &mut sc.h, false)?;
        self.step(x, d, sc, 0)?;
        x.copy_into_from(&self.out_table, code0 * dim, dim, &mut sc.h)?;
        for gi in 0..n_ac {
            self.step(x, d, sc, gi + 1)?;
            x.gemv(&self.heads_w[gi], &sc.h, &mut sc.logits, false, Some((&self.norm, d.cp_eps)), false)?;
            x.sample_topk(&sc.logits, d.code_vocab, top_k, &u, gi, &mut sc.codes)?;
            if gi + 1 < n_ac {
                x.gather_row(&self.tables[gi], &sc.codes, gi, &mut sc.h, dim)?;
            }
        }
        let got = x.down_codes(&sc.codes)?;
        let mut out = vec![code0];
        let mut sum = TtsDecoder::row(&d.out_embd, code0, td).to_vec();
        for (gi, &c) in got.iter().enumerate() {
            let c = c as usize;
            out.push(c);
            for (s, v) in sum.iter_mut().zip(TtsDecoder::row(&d.cp_embd[gi], c, td)) {
                *s += v;
            }
        }
        Ok((out, sum))
    }
}
