//! The speech encoder on the graphics card (`gtensor`): the same convolutions and transformer as
//! `AudioModel::encode_window`, with the weights uploaded as f16 on first use.

use super::{AudioModel, MelWindow, CHUNK, TOKENS_PER_CHUNK};
use crate::gtensor::{Act, DBuf, DConv, Gx, DW};
use crate::Error;

type Norm = (DBuf, DBuf);

pub(super) struct GAudio {
    conv: Vec<DConv>,
    conv_out: DW,
    blocks: Vec<(Norm, DW, DW, Norm, DW, DW)>,
    post: Norm,
    mm1: DW,
    mm2: DW,
}

fn norm(g: &Gx, n: &(Vec<f32>, Vec<f32>)) -> Result<Norm, Error> {
    Ok((g.up(&n.0)?, g.up(&n.1)?))
}

impl GAudio {
    pub(super) fn new(g: &Gx, m: &AudioModel) -> Result<Self, Error> {
        let mut ic = 1;
        let mut conv = Vec::new();
        for l in &m.conv {
            conv.push(g.conv(l, ic, 3, 2)?);
            ic = l.rows;
        }
        Ok(GAudio {
            conv,
            conv_out: g.weights(&m.conv_out)?,
            blocks: m.blocks.iter().map(|b| Ok((norm(g, &b.ln1)?, g.weights(&b.qkv)?, g.weights(&b.out)?, norm(g, &b.ln2)?, g.weights(&b.up)?, g.weights(&b.down)?))).collect::<Result<Vec<_>, Error>>()?,
            post: norm(g, &m.post)?,
            mm1: g.weights(&m.mm1)?,
            mm2: g.weights(&m.mm2)?,
        })
    }

    pub(super) fn encode_window(&self, g: &Gx, m: &AudioModel, w: &MelWindow) -> Result<Vec<f32>, Error> {
        let (n_mel, dim) = (m.n_mel, m.dim);
        let n_chunks = w.frames / CHUNK;
        let n = n_chunks * TOKENS_PER_CHUNK;
        let mut xs = vec![0f32; n * dim];
        for c in 0..n_chunks {
            // [mel][frame] of this chunk is [h][w][1].
            let mut act: Vec<f32> = Vec::with_capacity(n_mel * CHUNK);
            for mm in 0..n_mel {
                act.extend_from_slice(&w.data[mm * w.frames + c * CHUNK..mm * w.frames + (c + 1) * CHUNK]);
            }
            let (mut x, mut h, mut wd) = (g.up(&act)?, n_mel, CHUNK);
            let mut ch = 1;
            for cv in &self.conv {
                let (y, oh, ow) = g.conv2d(&x, h, wd, cv)?;
                x = y;
                (h, wd, ch) = (oh, ow, cv.w.rows);
                g.act(&mut x, h * wd * ch, Act::GeluErf)?;
            }
            // Tokens are the frame axis; features run channel-major over the mel axis (c * h + y).
            let act = g.down(&x)?;
            let feat = ch * h;
            if wd != TOKENS_PER_CHUNK || feat != self.conv_out.cols {
                return Err(Error::Format("audio convolutions gave an unexpected shape".into()));
            }
            let mut flat = vec![0f32; wd * feat];
            for t in 0..wd {
                for oc in 0..ch {
                    for y in 0..h {
                        flat[t * feat + oc * h + y] = act[(y * wd + t) * ch + oc];
                    }
                }
            }
            let mut dst = g.up(&m.pos[..wd * dim])?;
            g.linear(&g.up(&flat)?, wd, &self.conv_out, &mut dst, true)?;
            xs[c * TOKENS_PER_CHUNK * dim..(c + 1) * TOKENS_PER_CHUNK * dim].copy_from_slice(&g.down(&dst)?);
        }
        let hd = dim / m.heads;
        let mut x = g.up(&xs)?;
        let mut qkv = g.buf(n * 3 * dim)?;
        let mut att = g.buf(n * dim)?;
        let ffw = self.blocks.first().map_or(0, |b| b.4.rows);
        let mut ff = g.buf(n * ffw)?;
        for (ln1, wqkv, wout, ln2, up, down) in &self.blocks {
            let h = g.row_norm(&x, n, dim, &ln1.0, Some(&ln1.1), m.eps, false)?;
            g.linear(&h, n, wqkv, &mut qkv, false)?;
            g.attention(&qkv.slice(..), 3 * dim, &qkv.slice(dim..), 3 * dim, &qkv.slice(2 * dim..), 3 * dim, n, n, m.heads, hd, false, &mut att)?;
            g.linear(&att, n, wout, &mut x, true)?;
            let h = g.row_norm(&x, n, dim, &ln2.0, Some(&ln2.1), m.eps, false)?;
            g.linear(&h, n, up, &mut ff, false)?;
            g.act(&mut ff, n * ffw, Act::GeluErf)?;
            g.linear(&ff, n, down, &mut x, true)?;
        }
        let h = g.row_norm(&x, n, dim, &self.post.0, Some(&self.post.1), m.eps, false)?;
        let mut mid = g.buf(n * self.mm1.rows)?;
        g.linear(&h, n, &self.mm1, &mut mid, false)?;
        g.act(&mut mid, n * self.mm1.rows, Act::GeluErf)?;
        let mut out = g.buf(n * m.proj)?;
        g.linear(&mid, n, &self.mm2, &mut out, false)?;
        g.down(&out)
    }
}
