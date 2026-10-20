//! The image pipeline on the graphics card (`gtensor`): the same text encoder, UNet and VAE
//! decoder as the processor path, built from its loaded weights (uploaded as f16), with every
//! activation kept on the card between layers.

use super::{Attn, Conv, Resnet, TextEncoder, Transformer, UNet, VaeDecoder};
use crate::gtensor::{Act, DBuf, DConv, Gx, DW};
use crate::Error;

type Norm = (DBuf, DBuf);

fn norm(g: &Gx, n: &(Vec<f32>, Vec<f32>)) -> Result<Norm, Error> {
    Ok((g.up(&n.0)?, g.up(&n.1)?))
}

fn conv(g: &Gx, c: &Conv) -> Result<DConv, Error> {
    g.conv(&c.lin, c.ic, c.k, c.stride)
}

struct GResnet {
    n1: Norm,
    c1: DConv,
    temb: Option<DW>,
    n2: Norm,
    c2: DConv,
    skip: Option<DConv>,
    eps: f32,
}

impl GResnet {
    fn new(g: &Gx, r: &Resnet) -> Result<Self, Error> {
        Ok(GResnet {
            n1: norm(g, &r.n1)?,
            c1: conv(g, &r.c1)?,
            temb: r.temb.as_ref().map(|l| g.weights(l)).transpose()?,
            n2: norm(g, &r.n2)?,
            c2: conv(g, &r.c2)?,
            skip: r.skip.as_ref().map(|c| conv(g, c)).transpose()?,
            eps: r.eps,
        })
    }

    /// `semb`: SiLU of the time embedding (one row), when the block takes it.
    fn run(&self, g: &Gx, x: &DBuf, h: usize, w: usize, c: usize, semb: Option<&DBuf>) -> Result<DBuf, Error> {
        let n = h * w;
        let t = g.group_norm(x, n, c, 32, &self.n1.0, &self.n1.1, self.eps, true)?;
        let (mut t, _, _) = g.conv2d(&t, h, w, &self.c1)?;
        let oc = self.c1.w.rows;
        if let (Some(l), Some(e)) = (&self.temb, semb) {
            let mut add = g.buf(oc)?;
            g.linear(e, 1, l, &mut add, false)?;
            g.add_vec(&mut t, &add, n * oc, oc)?;
        }
        let t = g.group_norm(&t, n, oc, 32, &self.n2.0, &self.n2.1, self.eps, true)?;
        let (mut t, _, _) = g.conv2d(&t, h, w, &self.c2)?;
        match &self.skip {
            Some(s) => {
                let (sk, _, _) = g.conv2d(x, h, w, s)?;
                g.add(&mut t, &sk, n * oc)?;
            }
            None => g.add(&mut t, x, n * oc)?,
        }
        Ok(t)
    }
}

struct GAttn {
    q: DW,
    k: DW,
    v: DW,
    out: DW,
}

impl GAttn {
    fn new(g: &Gx, a: &Attn) -> Result<Self, Error> {
        Ok(GAttn { q: g.weights(&a.q)?, k: g.weights(&a.k)?, v: g.weights(&a.v)?, out: g.weights(&a.out)? })
    }

    /// Adds attention of the `n` rows of x (to themselves, or to the `nc` rows of ctx) to `acc`.
    fn run_add(&self, g: &Gx, x: &DBuf, n: usize, ctx: Option<(&DBuf, usize)>, heads: usize, acc: &mut DBuf) -> Result<(), Error> {
        let dim = self.q.rows;
        let mut q = g.buf(n * dim)?;
        g.linear(x, n, &self.q, &mut q, false)?;
        let (src, nk) = ctx.unwrap_or((x, n));
        let (mut k, mut v) = (g.buf(nk * dim)?, g.buf(nk * dim)?);
        g.linear(src, nk, &self.k, &mut k, false)?;
        g.linear(src, nk, &self.v, &mut v, false)?;
        let mut att = g.buf(n * dim)?;
        g.attention(&q.slice(..), dim, &k.slice(..), dim, &v.slice(..), dim, n, nk, heads, dim / heads, false, &mut att)?;
        g.linear(&att, n, &self.out, acc, true)
    }
}

struct GTransformer {
    norm: Norm,
    proj_in: DW,
    n1: Norm,
    a1: GAttn,
    n2: Norm,
    a2: GAttn,
    n3: Norm,
    ff1: DW,
    ff2: DW,
    proj_out: DW,
}

impl GTransformer {
    fn new(g: &Gx, t: &Transformer) -> Result<Self, Error> {
        Ok(GTransformer {
            norm: norm(g, &t.norm)?,
            proj_in: g.weights(&t.proj_in)?,
            n1: norm(g, &t.n1)?,
            a1: GAttn::new(g, &t.a1)?,
            n2: norm(g, &t.n2)?,
            a2: GAttn::new(g, &t.a2)?,
            n3: norm(g, &t.n3)?,
            ff1: g.weights(&t.ff1)?,
            ff2: g.weights(&t.ff2)?,
            proj_out: g.weights(&t.proj_out)?,
        })
    }

    fn run(&self, g: &Gx, x: &DBuf, n: usize, c: usize, ctx: &DBuf, nc: usize) -> Result<DBuf, Error> {
        let heads = c / 64;
        let t = g.group_norm(x, n, c, 32, &self.norm.0, &self.norm.1, 1e-6, false)?;
        let mut h = g.buf(n * c)?;
        g.linear(&t, n, &self.proj_in, &mut h, false)?;
        let t = g.row_norm(&h, n, c, &self.n1.0, Some(&self.n1.1), 1e-5, false)?;
        self.a1.run_add(g, &t, n, None, heads, &mut h)?;
        let t = g.row_norm(&h, n, c, &self.n2.0, Some(&self.n2.1), 1e-5, false)?;
        self.a2.run_add(g, &t, n, Some((ctx, nc)), heads, &mut h)?;
        let t = g.row_norm(&h, n, c, &self.n3.0, Some(&self.n3.1), 1e-5, false)?;
        let inner = self.ff1.rows / 2;
        let mut f = g.buf(n * 2 * inner)?;
        g.linear(&t, n, &self.ff1, &mut f, false)?;
        let f = g.geglu(&f, n, inner)?;
        g.linear(&f, n, &self.ff2, &mut h, true)?;
        let mut out = g.copy(x)?;
        g.linear(&h, n, &self.proj_out, &mut out, true)?;
        Ok(out)
    }
}

pub(super) struct GText {
    tok_embd: Vec<f32>,
    pos_embd: Vec<f32>,
    dim: usize,
    heads: usize,
    layers: Vec<(Norm, DW, DW, DW, DW, Norm, DW, DW)>,
    final_ln: Norm,
}

impl GText {
    pub(super) fn new(g: &Gx, t: TextEncoder) -> Result<Self, Error> {
        let layers = t
            .layers
            .iter()
            .map(|l| Ok((norm(g, &l.ln1)?, g.weights(&l.q)?, g.weights(&l.k)?, g.weights(&l.v)?, g.weights(&l.o)?, norm(g, &l.ln2)?, g.weights(&l.fc1)?, g.weights(&l.fc2)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(GText { final_ln: norm(g, &t.final_ln)?, tok_embd: t.tok_embd, pos_embd: t.pos_embd, dim: t.dim, heads: t.heads, layers })
    }

    /// The prompt's embeddings, left on the card.
    pub(super) fn encode(&self, g: &Gx, ids: &[u32]) -> Result<DBuf, Error> {
        let (n, d) = (ids.len(), self.dim);
        let mut x0 = vec![0f32; n * d];
        for (t, &id) in ids.iter().enumerate() {
            for j in 0..d {
                x0[t * d + j] = self.tok_embd[id as usize * d + j] + self.pos_embd[t * d + j];
            }
        }
        let mut x = g.up(&x0)?;
        for (ln1, q, k, v, o, ln2, fc1, fc2) in &self.layers {
            let h = g.row_norm(&x, n, d, &ln1.0, Some(&ln1.1), 1e-5, false)?;
            let (mut qq, mut kk, mut vv) = (g.buf(n * d)?, g.buf(n * d)?, g.buf(n * d)?);
            g.linear(&h, n, q, &mut qq, false)?;
            g.linear(&h, n, k, &mut kk, false)?;
            g.linear(&h, n, v, &mut vv, false)?;
            let mut att = g.buf(n * d)?;
            g.attention(&qq.slice(..), d, &kk.slice(..), d, &vv.slice(..), d, n, n, self.heads, d / self.heads, true, &mut att)?;
            g.linear(&att, n, o, &mut x, true)?;
            let h = g.row_norm(&x, n, d, &ln2.0, Some(&ln2.1), 1e-5, false)?;
            let mut f = g.buf(n * fc1.rows)?;
            g.linear(&h, n, fc1, &mut f, false)?;
            g.act(&mut f, n * fc1.rows, Act::GeluErf)?;
            g.linear(&f, n, fc2, &mut x, true)?;
        }
        g.row_norm(&x, n, d, &self.final_ln.0, Some(&self.final_ln.1), 1e-5, false)
    }
}

pub(super) struct GUNet {
    t1: DW,
    t2: DW,
    conv_in: DConv,
    down: Vec<(Vec<GResnet>, Vec<GTransformer>, Option<DConv>)>,
    mid: (GResnet, GTransformer, GResnet),
    up: Vec<(Vec<GResnet>, Vec<GTransformer>, Option<DConv>)>,
    norm_out: Norm,
    conv_out: DConv,
    base: usize,
}

impl GUNet {
    pub(super) fn new(g: &Gx, u: UNet) -> Result<Self, Error> {
        let rs = |v: &[Resnet]| v.iter().map(|r| GResnet::new(g, r)).collect::<Result<Vec<_>, Error>>();
        let ts = |v: &[Transformer]| v.iter().map(|t| GTransformer::new(g, t)).collect::<Result<Vec<_>, Error>>();
        Ok(GUNet {
            t1: g.weights(&u.t1)?,
            t2: g.weights(&u.t2)?,
            conv_in: conv(g, &u.conv_in)?,
            down: u.down.iter().map(|b| Ok((rs(&b.resnets)?, ts(&b.attns)?, b.down.as_ref().map(|c| conv(g, c)).transpose()?))).collect::<Result<Vec<_>, Error>>()?,
            mid: (GResnet::new(g, &u.mid.0)?, GTransformer::new(g, &u.mid.1)?, GResnet::new(g, &u.mid.2)?),
            up: u.up.iter().map(|b| Ok((rs(&b.resnets)?, ts(&b.attns)?, b.up.as_ref().map(|c| conv(g, c)).transpose()?))).collect::<Result<Vec<_>, Error>>()?,
            norm_out: norm(g, &u.norm_out)?,
            conv_out: conv(g, &u.conv_out)?,
            base: u.base,
        })
    }

    pub(super) fn run(&self, g: &Gx, x: &[f32], h: usize, w: usize, t: f32, ctx: &DBuf, nc: usize) -> Result<Vec<f32>, Error> {
        let half = self.base / 2;
        let mut te = vec![0f32; self.base];
        for i in 0..half {
            let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
            te[i] = (t * f).cos();
            te[half + i] = (t * f).sin();
        }
        let te = g.up(&te)?;
        let mut e1 = g.buf(self.t1.rows)?;
        g.linear(&te, 1, &self.t1, &mut e1, false)?;
        g.act(&mut e1, self.t1.rows, Act::Silu)?;
        let mut semb = g.buf(self.t2.rows)?;
        g.linear(&e1, 1, &self.t2, &mut semb, false)?;
        // Every ResNet block takes SiLU of the embedding.
        g.act(&mut semb, self.t2.rows, Act::Silu)?;

        let (mut x, mut hh, mut ww) = g.conv2d(&g.up(x)?, h, w, &self.conv_in)?;
        let mut c = self.base;
        let mut skips: Vec<(DBuf, usize)> = vec![(g.copy(&x)?, c)];
        for (resnets, attns, down) in &self.down {
            for (j, r) in resnets.iter().enumerate() {
                x = r.run(g, &x, hh, ww, c, Some(&semb))?;
                c = r.c1.w.rows;
                if let Some(a) = attns.get(j) {
                    x = a.run(g, &x, hh * ww, c, ctx, nc)?;
                }
                skips.push((g.copy(&x)?, c));
            }
            if let Some(d) = down {
                let (y, oh, ow) = g.conv2d(&x, hh, ww, d)?;
                x = y;
                (hh, ww) = (oh, ow);
                skips.push((g.copy(&x)?, c));
            }
        }
        x = self.mid.0.run(g, &x, hh, ww, c, Some(&semb))?;
        x = self.mid.1.run(g, &x, hh * ww, c, ctx, nc)?;
        x = self.mid.2.run(g, &x, hh, ww, c, Some(&semb))?;
        for (resnets, attns, up) in &self.up {
            for (j, r) in resnets.iter().enumerate() {
                let (s, sc) = skips.pop().expect("skip connection");
                let cat = g.concat(&x, c, &s, sc, hh * ww)?;
                x = r.run(g, &cat, hh, ww, c + sc, Some(&semb))?;
                c = r.c1.w.rows;
                if let Some(a) = attns.get(j) {
                    x = a.run(g, &x, hh * ww, c, ctx, nc)?;
                }
            }
            if let Some(u) = up {
                let upx = g.upsample2(&x, hh, ww, c)?;
                (hh, ww) = (hh * 2, ww * 2);
                x = g.conv2d(&upx, hh, ww, u)?.0;
            }
        }
        let t = g.group_norm(&x, hh * ww, c, 32, &self.norm_out.0, &self.norm_out.1, 1e-5, true)?;
        g.down(&g.conv2d(&t, hh, ww, &self.conv_out)?.0)
    }
}

pub(super) struct GVae {
    post_quant: Option<DConv>,
    conv_in: DConv,
    mid: (GResnet, Norm, GAttn, GResnet),
    up: Vec<(Vec<GResnet>, Option<DConv>)>,
    norm_out: Norm,
    conv_out: DConv,
}

impl GVae {
    pub(super) fn new(g: &Gx, v: &VaeDecoder) -> Result<Self, Error> {
        Ok(GVae {
            post_quant: v.post_quant.as_ref().map(|c| conv(g, c)).transpose()?,
            conv_in: conv(g, &v.conv_in)?,
            mid: (GResnet::new(g, &v.mid.0)?, norm(g, &v.mid.1)?, GAttn::new(g, &v.mid.2)?, GResnet::new(g, &v.mid.3)?),
            up: v.up.iter().map(|(rs, u)| Ok((rs.iter().map(|r| GResnet::new(g, r)).collect::<Result<Vec<_>, Error>>()?, u.as_ref().map(|c| conv(g, c)).transpose()?))).collect::<Result<Vec<_>, Error>>()?,
            norm_out: norm(g, &v.norm_out)?,
            conv_out: conv(g, &v.conv_out)?,
        })
    }

    pub(super) fn run(&self, g: &Gx, z: &[f32], h: usize, w: usize) -> Result<Vec<f32>, Error> {
        let mut z = g.up(z)?;
        if let Some(c) = &self.post_quant {
            z = g.conv2d(&z, h, w, c)?.0;
        }
        let (mut x, mut hh, mut ww) = g.conv2d(&z, h, w, &self.conv_in)?;
        let mut c = self.conv_in.w.rows;
        x = self.mid.0.run(g, &x, hh, ww, c, None)?;
        let t = g.group_norm(&x, hh * ww, c, 32, &self.mid.1 .0, &self.mid.1 .1, 1e-6, false)?;
        self.mid.2.run_add(g, &t, hh * ww, None, 1, &mut x)?;
        x = self.mid.3.run(g, &x, hh, ww, c, None)?;
        for (resnets, u) in &self.up {
            for r in resnets {
                x = r.run(g, &x, hh, ww, c, None)?;
                c = r.c1.w.rows;
            }
            if let Some(u) = u {
                let upx = g.upsample2(&x, hh, ww, c)?;
                (hh, ww) = (hh * 2, ww * 2);
                x = g.conv2d(&upx, hh, ww, u)?.0;
            }
        }
        let t = g.group_norm(&x, hh * ww, c, 32, &self.norm_out.0, &self.norm_out.1, 1e-6, true)?;
        g.down(&g.conv2d(&t, hh, ww, &self.conv_out)?.0)
    }
}
