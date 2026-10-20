//! The vision encoder on the graphics card (`gtensor`): the same blocks as `VisionModel::encode`,
//! with the weights uploaded as f16 on first use and every activation kept on the card.

use super::{Lin, VisionModel};
use crate::gtensor::{Act, DBuf, Gx, DW};
use crate::Error;

type Norm = (DBuf, DBuf);

pub(super) struct GVision {
    patch: DW,
    blocks: Vec<(Norm, DW, DW, Norm, DW, DW)>,
    post: Norm,
    mm0: DW,
    mm2: DW,
}

fn norm(g: &Gx, n: &(Vec<f32>, Vec<f32>)) -> Result<Norm, Error> {
    Ok((g.up(&n.0)?, g.up(&n.1)?))
}

fn w(g: &Gx, l: &Lin) -> Result<DW, Error> {
    g.weights(l)
}

impl GVision {
    pub(super) fn new(g: &Gx, m: &VisionModel) -> Result<Self, Error> {
        Ok(GVision {
            patch: w(g, &m.patch)?,
            blocks: m.blocks.iter().map(|b| Ok((norm(g, &b.ln1)?, w(g, &b.qkv)?, w(g, &b.out)?, norm(g, &b.ln2)?, w(g, &b.up)?, w(g, &b.down)?))).collect::<Result<Vec<_>, Error>>()?,
            post: norm(g, &m.post)?,
            mm0: w(g, &m.mm0)?,
            mm2: w(g, &m.mm2)?,
        })
    }

    /// `patches` (n x kin, token order), `pos` (n x dim, the interpolated positions), `cs` (the
    /// rotary (cos, sin) per token and frequency): the projected tokens (n/4 x proj).
    pub(super) fn encode(&self, g: &Gx, m: &VisionModel, patches: &[f32], pos: &[f32], cs: &[(f32, f32)], n: usize) -> Result<Vec<f32>, Error> {
        let c = &m.cfg;
        let dim = c.dim;
        let hd = dim / c.heads;
        let mut x = g.up(pos)?;
        g.linear(&g.up(patches)?, n, &self.patch, &mut x, true)?;
        let flat: Vec<f32> = cs.iter().flat_map(|&(co, s)| [co, s]).collect();
        let cs = g.up(&flat)?;
        let mut qkv = g.buf(n * 3 * dim)?;
        let mut att = g.buf(n * dim)?;
        let mut ff = g.buf(n * c.ff)?;
        for (ln1, wqkv, wout, ln2, up, down) in &self.blocks {
            let h = g.row_norm(&x, n, dim, &ln1.0, Some(&ln1.1), c.eps, false)?;
            g.linear(&h, n, wqkv, &mut qkv, false)?;
            g.rope_halves(&mut qkv, &cs, n, c.heads, hd, 3 * dim, 2)?;
            g.attention(&qkv.slice(..), 3 * dim, &qkv.slice(dim..), 3 * dim, &qkv.slice(2 * dim..), 3 * dim, n, n, c.heads, hd, false, &mut att)?;
            g.linear(&att, n, wout, &mut x, true)?;
            let h = g.row_norm(&x, n, dim, &ln2.0, Some(&ln2.1), c.eps, false)?;
            g.linear(&h, n, up, &mut ff, false)?;
            g.act(&mut ff, n * c.ff, Act::GeluTanh)?;
            g.linear(&ff, n, down, &mut x, true)?;
        }
        let h = g.row_norm(&x, n, dim, &self.post.0, Some(&self.post.1), c.eps, false)?;
        // Four neighbours (consecutive in token order) are one row of the merger's input.
        let rows = n / 4;
        let mut mid = g.buf(rows * self.mm0.rows)?;
        g.linear(&h, rows, &self.mm0, &mut mid, false)?;
        g.act(&mut mid, rows * self.mm0.rows, Act::GeluTanh)?;
        let mut out = g.buf(rows * c.proj)?;
        g.linear(&mid, rows, &self.mm2, &mut out, false)?;
        g.down(&out)
    }
}
