//! Portable GPU backend: one set of WGSL compute shaders run through wgpu on Vulkan and DX12 (AMD,
//! Intel and NVIDIA GPUs on Windows and Linux) and Metal (Apple GPUs). It runs dense attention
//! models (Llama, Qwen3, Mistral-style) whole on the GPU.
//!
//! Weights of any GGUF format are repacked at load to int8 with one f32 scale per 32 values (the
//! CUDA int8 layout): one matrix kernel serves every format, and Q8_0 files repack exactly. The KV
//! cache is f16. Activations stay f32 throughout. Every token is one submission: the embedding is
//! written from the CPU, nine dispatches per layer run in one compute pass, and the logits are read
//! back. A prompt is read as several tokens per submission.

use wgpu::util::DeviceExt;

use crate::model::{Ffn, Mat, Model};
use crate::quant::dequant_row;
use crate::Error;

/// Every kernel reads the chunk's tokens from this record: n tokens at positions pos0.., and each
/// token's RoPE (cos, sin) pairs, `half` per token (computed on the CPU, as the CPU path does).
const STEP: &str = r#"
struct Step { n: u32, pos0: u32, half: u32, _b: u32, cs: array<vec2<f32>> }
@group(0) @binding(1) var<storage, read> st: Step;
"#;

/// One workgroup per token: o[t] = x[t] / rms(x[t]) * w. `last` reads only the chunk's last token
/// (into row 0), for the output head.
const RMSNORM: &str = r#"
struct P { n: u32, eps: f32, last: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let i0 = l.x;
    var ti = g.x;
    var to = g.x;
    if (p.last == 1u) { ti = st.n - 1u; to = 0u; }
    let xb = ti * p.n;
    let ob = to * p.n;
    var s = 0.0;
    for (var i = i0; i < p.n; i += 256u) { let v = x[xb + i]; s += v * v; }
    red[i0] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (i0 < k) { red[i0] += red[i0 + k]; }
        workgroupBarrier();
    }
    let r = 1.0 / sqrt(red[0] / f32(p.n) + p.eps);
    for (var i = i0; i < p.n; i += 256u) { o[ob + i] = x[xb + i] * r * w[i]; }
}
"#;

/// o[t * out_stride + off + r] (+)= scale-per-32 int8 row r . x[t]. Four rows per workgroup, 64
/// threads per row, 16 weights per load; each weight is read once for TB tokens. Rows past 65535*4
/// continue in y; token groups are z. `fixed` > 0 overrides the chunk's token count (the head).
const MATVEC: &str = r#"
struct P { rows: u32, cols: u32, add: u32, off: u32, out_stride: u32, in_stride: u32, fixed: u32, _a: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> s: array<f32>;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
var<workgroup> red: array<f32, TB * 256u>;
fn unpack(q: u32) -> vec4<f32> {
    let i = i32(q);
    return vec4<f32>(f32(extractBits(i, 0u, 8u)), f32(extractBits(i, 8u, 8u)), f32(extractBits(i, 16u, 8u)), f32(extractBits(i, 24u, 8u)));
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let lane = l.x % 64u;
    let r = (g.x + g.y * 65535u) * 4u + l.x / 64u;
    let valid = r < p.rows;
    var nt = st.n;
    if (p.fixed > 0u) { nt = p.fixed; }
    let t0 = g.z * TB;
    var acc: array<f32, TB>;
    if (valid) {
        let w16 = p.cols / 16u;
        let base = r * w16;
        let xs = p.in_stride / 4u;
        for (var j = lane; j < w16; j += 64u) {
            let q = w[base + j];
            let sc = s[(base + j) / 2u];
            let a0 = unpack(q.x);
            let a1 = unpack(q.y);
            let a2 = unpack(q.z);
            let a3 = unpack(q.w);
            for (var t = 0u; t < TB; t++) {
                if (t0 + t < nt) {
                    let xb = (t0 + t) * xs + 4u * j;
                    acc[t] += sc * (dot(a0, x[xb]) + dot(a1, x[xb + 1u]) + dot(a2, x[xb + 2u]) + dot(a3, x[xb + 3u]));
                }
            }
        }
    }
    for (var t = 0u; t < TB; t++) { red[t * 256u + l.x] = acc[t]; }
    workgroupBarrier();
    for (var k = 32u; k > 0u; k >>= 1u) {
        if (lane < k) {
            for (var t = 0u; t < TB; t++) { red[t * 256u + l.x] += red[t * 256u + l.x + k]; }
        }
        workgroupBarrier();
    }
    if (valid && lane == 0u) {
        for (var t = 0u; t < TB; t++) {
            if (t0 + t < nt) {
                let oi = (t0 + t) * p.out_stride + p.off + r;
                let v = red[t * 256u + l.x];
                if (p.add == 1u) { o[oi] += v; } else { o[oi] = v; }
            }
        }
    }
}
"#;

/// The same product, one row per workgroup of WG threads and four weights per load: the faster
/// kernel on some GPUs (Intel's, measured), slower on others; `load` times both families.
const MATVEC_ROW: &str = r#"
struct P { rows: u32, cols: u32, add: u32, off: u32, out_stride: u32, in_stride: u32, fixed: u32, _a: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> s: array<f32>;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
var<workgroup> red: array<f32, TB * WG>;
fn unpack(q: u32) -> vec4<f32> {
    let i = i32(q);
    return vec4<f32>(f32(extractBits(i, 0u, 8u)), f32(extractBits(i, 8u, 8u)), f32(extractBits(i, 16u, 8u)), f32(extractBits(i, 24u, 8u)));
}
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let r = g.x + g.y * 65535u;
    let valid = r < p.rows;
    var nt = st.n;
    if (p.fixed > 0u) { nt = p.fixed; }
    let t0 = g.z * TB;
    var acc: array<f32, TB>;
    if (valid) {
        let wpr = p.cols / 4u;
        let base = r * wpr;
        let xs = p.in_stride / 4u;
        for (var i = l.x; i < wpr; i += WG) {
            let a = unpack(w[base + i]);
            let sc = s[(base + i) / 8u];
            for (var t = 0u; t < TB; t++) {
                if (t0 + t < nt) { acc[t] += sc * dot(a, x[(t0 + t) * xs + i]); }
            }
        }
    }
    for (var t = 0u; t < TB; t++) { red[t * WG + l.x] = acc[t]; }
    workgroupBarrier();
    for (var k = WG / 2u; k > 0u; k >>= 1u) {
        if (l.x < k) {
            for (var t = 0u; t < TB; t++) { red[t * WG + l.x] += red[t * WG + l.x + k]; }
        }
        workgroupBarrier();
    }
    if (valid && l.x == 0u) {
        for (var t = 0u; t < TB; t++) {
            if (t0 + t < nt) {
                let oi = (t0 + t) * p.out_stride + p.off + r;
                let v = red[t * WG];
                if (p.add == 1u) { o[oi] += v; } else { o[oi] = v; }
            }
        }
    }
}
"#;

/// The product straight from the file's Q4_K blocks (256 weights in 144 bytes: f16 d and dmin,
/// twelve bytes of 6-bit scales and mins, 128 bytes of nibbles), one row per workgroup of 64
/// threads. A thread takes 8 bytes of nibbles: 8 weights of sub-block 2j and 8 of 2j+1. Half the
/// memory of the int8 layout, so a Q4_K_M model fits a card the int8 copy would not.
const MATVEC_Q4K: &str = r#"
struct P { rows: u32, cols: u32, add: u32, off: u32, out_stride: u32, in_stride: u32, fixed: u32, _a: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
var<workgroup> red: array<f32, TB * WG>;
fn sb(bw: u32, k: u32) -> u32 { return (w[bw + 1u + k / 4u] >> (8u * (k % 4u))) & 0xFFu; }
fn lo(q: u32) -> vec4<f32> { return vec4<f32>(f32(q & 15u), f32((q >> 8u) & 15u), f32((q >> 16u) & 15u), f32((q >> 24u) & 15u)); }
fn hi(q: u32) -> vec4<f32> { return vec4<f32>(f32((q >> 4u) & 15u), f32((q >> 12u) & 15u), f32((q >> 20u) & 15u), f32((q >> 28u) & 15u)); }
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let r = g.x + g.y * 65535u;
    let valid = r < p.rows;
    var nt = st.n;
    if (p.fixed > 0u) { nt = p.fixed; }
    let t0 = g.z * TB;
    var acc: array<f32, TB>;
    if (valid) {
        let nb = p.cols / 256u;
        let rw = r * nb * 36u;
        let xs = p.in_stride / 4u;
        let one = vec4<f32>(1.0, 1.0, 1.0, 1.0);
        for (var u = l.x; u < nb * 16u; u += WG) {
            let b = u / 16u;
            let j = (u % 16u) / 4u;
            let qq = u % 4u;
            let bw = rw + b * 36u;
            let dd = unpack2x16float(w[bw]);
            let is = 2u * j;
            var sc1 = 0u;
            var m1 = 0u;
            var sc2 = 0u;
            var m2 = 0u;
            if (j < 2u) {
                sc1 = sb(bw, is) & 63u;
                m1 = sb(bw, is + 4u) & 63u;
                sc2 = sb(bw, is + 1u) & 63u;
                m2 = sb(bw, is + 5u) & 63u;
            } else {
                sc1 = (sb(bw, is + 4u) & 15u) | ((sb(bw, is - 4u) >> 6u) << 4u);
                m1 = (sb(bw, is + 4u) >> 4u) | ((sb(bw, is) >> 6u) << 4u);
                sc2 = (sb(bw, is + 5u) & 15u) | ((sb(bw, is - 3u) >> 6u) << 4u);
                m2 = (sb(bw, is + 5u) >> 4u) | ((sb(bw, is + 1u) >> 6u) << 4u);
            }
            let d1 = dd.x * f32(sc1);
            let n1 = dd.y * f32(m1);
            let d2 = dd.x * f32(sc2);
            let n2 = dd.y * f32(m2);
            let q0 = w[bw + 4u + j * 8u + qq * 2u];
            let q1 = w[bw + 5u + j * 8u + qq * 2u];
            let l0 = lo(q0);
            let l1 = lo(q1);
            let h0 = hi(q0);
            let h1 = hi(q1);
            let xl = (b * 256u + j * 64u + qq * 8u) / 4u;
            for (var t = 0u; t < TB; t++) {
                if (t0 + t < nt) {
                    let xb = (t0 + t) * xs + xl;
                    let a0 = x[xb];
                    let a1 = x[xb + 1u];
                    let c0 = x[xb + 8u];
                    let c1 = x[xb + 9u];
                    acc[t] += d1 * (dot(l0, a0) + dot(l1, a1)) - n1 * (dot(a0, one) + dot(a1, one)) + d2 * (dot(h0, c0) + dot(h1, c1)) - n2 * (dot(c0, one) + dot(c1, one));
                }
            }
        }
    }
    for (var t = 0u; t < TB; t++) { red[t * WG + l.x] = acc[t]; }
    workgroupBarrier();
    for (var k = WG / 2u; k > 0u; k >>= 1u) {
        if (l.x < k) {
            for (var t = 0u; t < TB; t++) { red[t * WG + l.x] += red[t * WG + l.x + k]; }
        }
        workgroupBarrier();
    }
    if (valid && l.x == 0u) {
        for (var t = 0u; t < TB; t++) {
            if (t0 + t < nt) {
                let oi = (t0 + t) * p.out_stride + p.off + r;
                let v = red[t * WG];
                if (p.add == 1u) { o[oi] += v; } else { o[oi] = v; }
            }
        }
    }
}
"#;

/// Q4_K, two rows per workgroup of 64 threads: a thread takes a whole 64-weight group (32
/// contiguous bytes of nibbles, its two scales decoded once) and keeps its eight words in registers
/// across tokens. More bytes in flight per thread than MATVEC_Q4K; `load` times both.
const MATVEC_Q4K2: &str = r#"
struct P { rows: u32, cols: u32, add: u32, off: u32, out_stride: u32, in_stride: u32, fixed: u32, _a: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
var<workgroup> red: array<f32, TB * 64u>;
fn sb(bw: u32, k: u32) -> u32 { return (w[bw + 1u + k / 4u] >> (8u * (k % 4u))) & 0xFFu; }
fn lo(q: u32) -> vec4<f32> { return vec4<f32>(f32(q & 15u), f32((q >> 8u) & 15u), f32((q >> 16u) & 15u), f32((q >> 24u) & 15u)); }
fn hi(q: u32) -> vec4<f32> { return vec4<f32>(f32((q >> 4u) & 15u), f32((q >> 12u) & 15u), f32((q >> 20u) & 15u), f32((q >> 28u) & 15u)); }
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let lane = l.x % 32u;
    let r = (g.x + g.y * 65535u) * 2u + l.x / 32u;
    let valid = r < p.rows;
    var nt = st.n;
    if (p.fixed > 0u) { nt = p.fixed; }
    let t0 = g.z * TB;
    var acc: array<f32, TB>;
    if (valid) {
        let nb = p.cols / 256u;
        let rw = r * nb * 36u;
        let xs = p.in_stride / 4u;
        let one = vec4<f32>(1.0, 1.0, 1.0, 1.0);
        for (var u = lane; u < nb * 4u; u += 32u) {
            let b = u / 4u;
            let j = u % 4u;
            let bw = rw + b * 36u;
            let dd = unpack2x16float(w[bw]);
            let is = 2u * j;
            var sc1 = 0u;
            var m1 = 0u;
            var sc2 = 0u;
            var m2 = 0u;
            if (j < 2u) {
                sc1 = sb(bw, is) & 63u;
                m1 = sb(bw, is + 4u) & 63u;
                sc2 = sb(bw, is + 1u) & 63u;
                m2 = sb(bw, is + 5u) & 63u;
            } else {
                sc1 = (sb(bw, is + 4u) & 15u) | ((sb(bw, is - 4u) >> 6u) << 4u);
                m1 = (sb(bw, is + 4u) >> 4u) | ((sb(bw, is) >> 6u) << 4u);
                sc2 = (sb(bw, is + 5u) & 15u) | ((sb(bw, is - 3u) >> 6u) << 4u);
                m2 = (sb(bw, is + 5u) >> 4u) | ((sb(bw, is + 1u) >> 6u) << 4u);
            }
            let d1 = dd.x * f32(sc1);
            let n1 = dd.y * f32(m1);
            let d2 = dd.x * f32(sc2);
            let n2 = dd.y * f32(m2);
            let qb = bw + 4u + j * 8u;
            var q: array<u32, 8>;
            for (var k = 0u; k < 8u; k++) { q[k] = w[qb + k]; }
            let xl = (b * 256u + j * 64u) / 4u;
            for (var t = 0u; t < TB; t++) {
                if (t0 + t < nt) {
                    let xb = (t0 + t) * xs + xl;
                    var sl = 0.0;
                    var sxl = 0.0;
                    var sh = 0.0;
                    var sxh = 0.0;
                    for (var k = 0u; k < 8u; k++) {
                        let a = x[xb + k];
                        let c = x[xb + 8u + k];
                        sl += dot(lo(q[k]), a);
                        sxl += dot(a, one);
                        sh += dot(hi(q[k]), c);
                        sxh += dot(c, one);
                    }
                    acc[t] += d1 * sl - n1 * sxl + d2 * sh - n2 * sxh;
                }
            }
        }
    }
    for (var t = 0u; t < TB; t++) { red[t * 64u + l.x] = acc[t]; }
    workgroupBarrier();
    for (var k = 16u; k > 0u; k >>= 1u) {
        if (lane < k) {
            for (var t = 0u; t < TB; t++) { red[t * 64u + l.x] += red[t * 64u + l.x + k]; }
        }
        workgroupBarrier();
    }
    if (valid && lane == 0u) {
        for (var t = 0u; t < TB; t++) {
            if (t0 + t < nt) {
                let oi = (t0 + t) * p.out_stride + p.off + r;
                let v = red[t * 64u + l.x];
                if (p.add == 1u) { o[oi] += v; } else { o[oi] = v; }
            }
        }
    }
}
"#;

/// Per (head, token): optional RMS norm, RoPE on the first rd values, then K and V (f16) into the
/// cache at the token's position. Heads 0..n_head are queries, the rest KV heads.
const QK_POST: &str = r#"
struct P { n_head: u32, n_kv: u32, hd: u32, rd: u32, kvd: u32, q_norm: u32, k_norm: u32, eps: f32, qd: u32, row: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(3) var<storage, read> qn: array<f32>;
@group(0) @binding(4) var<storage, read> kn: array<f32>;
@group(0) @binding(5) var<storage, read_write> kc: array<u32>;
@group(0) @binding(6) var<storage, read_write> vc: array<u32>;
var<workgroup> hv: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let i = l.x;
    let t = g.y;
    let is_q = g.x < p.n_head;
    var head = g.x;
    var off = t * p.row + g.x * p.hd;
    var norm = p.q_norm;
    if (!is_q) { head = g.x - p.n_head; off = t * p.row + p.qd + head * p.hd; norm = p.k_norm; }
    var v = 0.0;
    if (i < p.hd) { v = qkv[off + i]; }
    red[i] = v * v;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (i < k) { red[i] += red[i + k]; }
        workgroupBarrier();
    }
    if (norm == 1u && i < p.hd) {
        var wv = 0.0;
        if (is_q) { wv = qn[i]; } else { wv = kn[i]; }
        v = v / sqrt(red[0] / f32(p.hd) + p.eps) * wv;
    }
    hv[i] = v;
    workgroupBarrier();
    let half = p.rd / 2u;
    let cb = t * st.half;
    var o = v;
    if (i < half) {
        let c = st.cs[cb + i];
        o = hv[i] * c.x - hv[i + half] * c.y;
    } else if (i < p.rd) {
        let c = st.cs[cb + i - half];
        o = hv[i - half] * c.y + hv[i] * c.x;
    }
    workgroupBarrier();
    hv[i] = o;
    workgroupBarrier();
    if (is_q) {
        if (i < p.hd) { qkv[off + i] = o; }
    } else if (i < p.hd / 2u) {
        let dst = ((st.pos0 + t) * p.kvd + head * p.hd) / 2u + i;
        kc[dst] = pack2x16float(vec2<f32>(hv[2u * i], hv[2u * i + 1u]));
        let vo = t * p.row + p.qd + p.kvd + head * p.hd;
        vc[dst] = pack2x16float(vec2<f32>(qkv[vo + 2u * i], qkv[vo + 2u * i + 1u]));
    }
}
"#;

/// Attention in spans of SPAN positions (split-K): one workgroup per (query head, span, token)
/// writes the span's max score, its sum of exponentials and its unnormalised weighted values; the
/// combine kernel merges the spans. Eight threads share each key's dot product, so neighbouring
/// threads read neighbouring bytes, and the span's scores stay in workgroup memory. One workgroup
/// per head (the first version) left most of a GPU idle on long contexts.
const ATTENTION: &str = r#"
const SPAN: u32 = 256u;
struct P { group: u32, hd: u32, kvd: u32, maxs: u32, scale: f32, row: u32, n_head: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> qkv: array<f32>;
@group(0) @binding(3) var<storage, read> kc: array<u32>;
@group(0) @binding(4) var<storage, read> vc: array<u32>;
@group(0) @binding(5) var<storage, read_write> part: array<f32>;
var<workgroup> qs: array<f32, 256>;
var<workgroup> sc: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
var<workgroup> acc2: array<vec2<f32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l3: vec3<u32>) {
    let h = g.x;
    let span = g.y;
    let t = g.z;
    let i = l3.x;
    let kvh = h / p.group;
    let len = st.pos0 + t + 1u;
    let pairs = p.hd / 2u;
    let u0 = span * SPAN;
    let pb = ((t * p.n_head + h) * p.maxs + span) * (p.hd + 2u);
    if (i < p.hd) { qs[i] = qkv[t * p.row + h * p.hd + i]; }
    workgroupBarrier();
    let lane = i % 8u;
    let slot = i / 8u;
    let per = pairs / 8u;
    for (var stp = 0u; stp < SPAN / 32u; stp++) {
        let u = u0 + stp * 32u + slot;
        var d = 0.0;
        if (u < len) {
            let kb = (u * p.kvd + kvh * p.hd) / 2u + lane * per;
            for (var j = 0u; j < per; j++) {
                let k = unpack2x16float(kc[kb + j]);
                let q = 2u * (lane * per + j);
                d += qs[q] * k.x + qs[q + 1u] * k.y;
            }
        }
        red[i] = d;
        workgroupBarrier();
        if (lane == 0u) {
            var s = 0.0;
            for (var k = 0u; k < 8u; k++) { s += red[i + k]; }
            if (u < len) { sc[stp * 32u + slot] = s * p.scale; } else { sc[stp * 32u + slot] = -3.0e38; }
        }
        workgroupBarrier();
    }
    red[i] = sc[i];
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (i < k) { red[i] = max(red[i], red[i + k]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var e = 0.0;
    if (u0 + i < len) { e = exp(sc[i] - m); }
    sc[i] = e;
    red[i] = e;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (i < k) { red[i] += red[i + k]; }
        workgroupBarrier();
    }
    let lsum = red[0];
    let ng = 256u / pairs;
    let pi = i % pairs;
    let gi = i / pairs;
    var a = vec2<f32>(0.0, 0.0);
    if (gi < ng) {
        for (var v = gi; v < SPAN; v += ng) {
            let u = u0 + v;
            if (u < len) { a += sc[v] * unpack2x16float(vc[(u * p.kvd + kvh * p.hd) / 2u + pi]); }
        }
    }
    acc2[i] = a;
    workgroupBarrier();
    if (gi == 0u) {
        var s2 = a;
        for (var k = 1u; k < ng; k++) { s2 += acc2[k * pairs + pi]; }
        part[pb + 2u + 2u * pi] = s2.x;
        part[pb + 3u + 2u * pi] = s2.y;
    }
    if (i == 0u) {
        part[pb] = m;
        part[pb + 1u] = lsum;
    }
}
"#;

/// Merges the spans of each (query head, token): out = sum(acc_s e^(m_s - M)) / sum(l_s e^(m_s - M)).
const ATTN_COMBINE: &str = r#"
const SPAN: u32 = 256u;
struct P { hd: u32, maxs: u32, n_head: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> part: array<f32>;
@group(0) @binding(3) var<storage, read_write> ao: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l3: vec3<u32>) {
    let h = g.x;
    let t = g.y;
    let i = l3.x;
    let len = st.pos0 + t + 1u;
    let ns = (len + SPAN - 1u) / SPAN;
    let base = (t * p.n_head + h) * p.maxs;
    let w = p.hd + 2u;
    var mm = -3.0e38;
    for (var s = 0u; s < ns; s++) { mm = max(mm, part[(base + s) * w]); }
    var tot = 0.0;
    for (var s = 0u; s < ns; s++) { tot += part[(base + s) * w + 1u] * exp(part[(base + s) * w] - mm); }
    if (i < p.hd) {
        var o = 0.0;
        for (var s = 0u; s < ns; s++) {
            let b = (base + s) * w;
            o += part[b + 2u + i] * exp(part[b] - mm);
        }
        ao[t * p.n_head * p.hd + h * p.hd + i] = o / tot;
    }
}
"#;

/// Per (query head, token): scores over positions 0..=pos, softmax, weighted values, one workgroup
/// per head with the scores in a scratch buffer. Used on Intel GPUs: the split kernel (ATTENTION)
/// hung an Iris Xe (driver 30.0.101.1660) when compiled by DXC, this one did not.
const ATTENTION_HEAD: &str = r#"
struct P { group: u32, hd: u32, kvd: u32, max_ctx: u32, scale: f32, row: u32, n_head: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> qkv: array<f32>;
@group(0) @binding(3) var<storage, read> kc: array<u32>;
@group(0) @binding(4) var<storage, read> vc: array<u32>;
@group(0) @binding(5) var<storage, read_write> att: array<f32>;
@group(0) @binding(6) var<storage, read_write> ao: array<f32>;
var<workgroup> qs: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
var<workgroup> acc2: array<vec2<f32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l3: vec3<u32>) {
    let h = g.x;
    let t = g.y;
    let i = l3.x;
    let kvh = h / p.group;
    let len = st.pos0 + t + 1u;
    let pairs = p.hd / 2u;
    if (i < p.hd) { qs[i] = qkv[t * p.row + h * p.hd + i]; }
    workgroupBarrier();
    let base = (t * p.n_head + h) * p.max_ctx;
    var mx = -3.0e38;
    for (var u = i; u < len; u += 256u) {
        let kb = (u * p.kvd + kvh * p.hd) / 2u;
        var s = 0.0;
        for (var j = 0u; j < pairs; j++) {
            let k = unpack2x16float(kc[kb + j]);
            s += qs[2u * j] * k.x + qs[2u * j + 1u] * k.y;
        }
        s *= p.scale;
        att[base + u] = s;
        mx = max(mx, s);
    }
    red[i] = mx;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (i < k) { red[i] = max(red[i], red[i + k]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var u = i; u < len; u += 256u) {
        let e = exp(att[base + u] - m);
        att[base + u] = e;
        sum += e;
    }
    storageBarrier();
    red[i] = sum;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (i < k) { red[i] += red[i + k]; }
        workgroupBarrier();
    }
    let total = red[0];
    let ng = 256u / pairs;
    let pi = i % pairs;
    let gi = i / pairs;
    var a = vec2<f32>(0.0, 0.0);
    if (gi < ng) {
        for (var u = gi; u < len; u += ng) {
            a += att[base + u] * unpack2x16float(vc[(u * p.kvd + kvh * p.hd) / 2u + pi]);
        }
    }
    acc2[i] = a;
    workgroupBarrier();
    if (gi == 0u) {
        var s2 = a;
        for (var k = 1u; k < ng; k++) { s2 += acc2[k * pairs + pi]; }
        let ob = t * p.n_head * p.hd + h * p.hd + 2u * pi;
        ao[ob] = s2.x / total;
        ao[ob + 1u] = s2.y / total;
    }
}
"#;

/// gu[t][i] = silu(gu[t][i]) * gu[t][n + i] for every token of the chunk.
const SILU: &str = r#"
struct P { n: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> gu: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k < st.n * p.n) {
        let t = k / p.n;
        let i = k % p.n;
        let b = t * 2u * p.n;
        let g = gu[b + i];
        gu[b + i] = g / (1.0 + exp(-g)) * gu[b + p.n + i];
    }
}
"#;

/// A GPU the portable backend can use.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PortableGpu {
    pub name: String,
    /// "Vulkan", "Dx12" or "Metal".
    pub backend: String,
    /// "DiscreteGpu", "IntegratedGpu", ...
    pub kind: String,
    pub vendor: u32,
    /// Largest single buffer, bytes.
    pub max_buffer: u64,
}

fn backends() -> wgpu::Backends {
    match std::env::var("CEREBRO_PORTABLE_BACKEND").ok().as_deref() {
        Some("vulkan") => wgpu::Backends::VULKAN,
        Some("dx12") => wgpu::Backends::DX12,
        Some("metal") => wgpu::Backends::METAL,
        _ => wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL,
    }
}

/// The DXC shader compiler for DX12, if one is found: `dxcompiler.dll` in CEREBRO_DXC_DIR, next to
/// the program or in its `dxc` folder (the app ships it), or in an installed Windows SDK. Without it wgpu uses FXC, whose
/// code hung an Intel Iris Xe in the one-row matrix kernels over several tokens (driver reset);
/// the same shaders compiled by DXC ran correctly.
fn dxc_path() -> Option<std::path::PathBuf> {
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    if let Some(d) = std::env::var_os("CEREBRO_DXC_DIR") {
        // "fxc": use FXC even when DXC is installed (to check the fallback).
        if d == "fxc" {
            return None;
        }
        dirs.push(d.into());
    }
    if let Some(d) = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.to_path_buf())) {
        dirs.push(d.join("dxc"));
        dirs.push(d.join("resources"));
        dirs.push(d);
    }
    if let Ok(rd) = std::fs::read_dir(r"C:\Program Files (x86)\Windows Kits\10\bin") {
        let mut sdk: Vec<_> = rd.flatten().map(|e| e.path().join("x64")).collect();
        sdk.sort();
        dirs.extend(sdk.into_iter().rev());
    }
    // DXC 1.8.2502 and later validates and signs shaders itself: dxcompiler.dll alone is enough.
    dirs.into_iter().map(|d| d.join("dxcompiler.dll")).find(|p| p.is_file())
}

/// Whether DX12 shaders are compiled by DXC (else FXC).
fn has_dxc() -> bool {
    static D: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    D.get_or_init(dxc_path).is_some()
}

fn instance() -> wgpu::Instance {
    let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
    d.backends = backends();
    if let Some(p) = dxc_path() {
        d.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::DynamicDxc { dxc_path: p.to_string_lossy().into_owned() };
    }
    wgpu::Instance::new(d)
}

/// Every adapter, one per GPU and API (a GPU appears once for Vulkan and once for DX12 on Windows).
pub fn adapters() -> Vec<PortableGpu> {
    let inst = instance();
    pollster::block_on(inst.enumerate_adapters(backends()))
        .into_iter()
        .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
        .map(|a| {
            let i = a.get_info();
            PortableGpu { name: i.name, backend: format!("{:?}", i.backend), kind: format!("{:?}", i.device_type), vendor: i.vendor, max_buffer: a.limits().max_buffer_size }
        })
        .collect()
}

const INTEL: u32 = 0x8086;

fn pick_adapter(inst: &wgpu::Instance) -> Result<wgpu::Adapter, Error> {
    let all: Vec<wgpu::Adapter> = pollster::block_on(inst.enumerate_adapters(backends())).into_iter().filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu).collect();
    // CEREBRO_PORTABLE_GPU picks by name (case-insensitive substring); otherwise a discrete GPU first,
    // then an integrated one; Vulkan before DX12, except on Intel GPUs under Windows, where DX12
    // comes first: Intel's Vulkan driver crashed in device creation on an Iris Xe (driver
    // 30.0.101.1660) while DX12 ran the same shaders correctly.
    // Integrated GPUs only when the switch allows it (or one is asked for by name).
    let integrated_ok = crate::switches::enabled(crate::switches::PORTABLE_INTEGRATED) || std::env::var_os("CEREBRO_PORTABLE_GPU").is_some();
    let all: Vec<wgpu::Adapter> = all.into_iter().filter(|a| integrated_ok || a.get_info().device_type != wgpu::DeviceType::IntegratedGpu).collect();
    let rank = |a: &wgpu::Adapter| {
        let i = a.get_info();
        let kind = match i.device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            _ => 2,
        };
        let preferred = if cfg!(windows) && i.vendor == INTEL { wgpu::Backend::Dx12 } else { wgpu::Backend::Vulkan };
        (kind, (i.backend != preferred) as u8)
    };
    if let Ok(want) = std::env::var("CEREBRO_PORTABLE_GPU") {
        let w = want.to_lowercase();
        return all.into_iter().filter(|a| a.get_info().name.to_lowercase().contains(&w)).min_by_key(rank).ok_or_else(|| Error::Unsupported(format!("no GPU named like {want}")));
    }
    all.into_iter().min_by_key(rank).ok_or_else(|| {
        Error::Unsupported(if integrated_ok { "no GPU found through Vulkan, DX12 or Metal".into() } else { "no separate graphics card found (integrated graphics is off: Turbo > Integrated graphics)".into() })
    })
}

fn no_room() -> Error {
    Error::Unsupported("the model does not fit in this graphics card's memory (the portable backend needs the whole model on the card)".into())
}

fn perr(e: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("portable GPU: {e}"))
}

/// One matrix on the GPU, in row parts that each fit a buffer binding.
struct GMat {
    parts: Vec<Part>,
    cols: usize,
}

impl GMat {
    /// Bytes on the GPU.
    fn size(&self) -> u64 {
        self.parts.iter().map(|p| p.bytes).sum()
    }
}

struct Part {
    w: wgpu::Buffer,
    /// int8 scales; a 16-byte placeholder for Q4_K parts (their scales are inside the blocks).
    s: wgpu::Buffer,
    /// First row of this part in the stacked output.
    row0: usize,
    rows: usize,
    /// The file's own Q4_K blocks rather than the int8 layout.
    q4k: bool,
    bytes: u64,
}

/// How a prepared dispatch is sized for a chunk of `nt` tokens.
#[derive(Clone, Copy)]
enum Dims {
    /// One workgroup per token (x).
    PerToken,
    /// Exactly one workgroup.
    One,
    /// `heads` workgroups in x, one per token in y.
    Heads(u32),
    /// A matrix of `rows` rows: four per workgroup (x, then y past 65535); token groups in z.
    /// `fixed`: the head, always one token.
    Rows { rows: u32, fixed: bool, q4k: bool },
    /// `n` values per token, 256 per workgroup.
    Elems(u32),
    /// Attention spans: `heads` in x, the chunk's spans (SPAN positions each) in y, tokens in z.
    Spans(u32),
}

/// One prepared dispatch.
struct Call {
    pipe: usize,
    bind: wgpu::BindGroup,
    dims: Dims,
    /// A matrix call's input and output buffers, and whether it adds into the output: the
    /// load-time kernel check runs candidates on them.
    io: Option<(wgpu::Buffer, wgpu::Buffer, bool)>,
}

pub struct PortableState {
    device: wgpu::Device,
    /// The first error the device reported outside any call (validation, device loss).
    error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    queue: wgpu::Queue,
    pub info: PortableGpu,
    max_ctx: usize,
    /// Tokens per pass (prompt reading); the attention scratch is sized for it.
    chunk: usize,
    x: wgpu::Buffer,
    step: wgpu::Buffer,
    logits: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// A split's activations after the GPU's layers, read back for the processor's layers.
    readback_x: wgpu::Buffer,
    pipes: Vec<wgpu::ComputePipeline>,
    /// Each layer's dispatches in order, then the output head's.
    layer_calls: Vec<Vec<Call>>,
    head_calls: Vec<Call>,
    /// Work per submission, adapted to measured time (see `run`).
    pass_tokens: usize,
    pass_layers: usize,
    /// The matrix kernels chosen at load for one token and for TB tokens: (pipeline, rows per workgroup).
    mv1: (usize, u32),
    mv8: (usize, u32),
    /// The Q4_K kernels chosen at load, likewise.
    qk1: (usize, u32),
    qk8: (usize, u32),
    /// What the load-time timing chose, in words.
    pub kernels: String,
    dim: usize,
    vocab: usize,
    pub bytes: u64,
}

const P_RMS: usize = 0;
const P_MATVEC: usize = 1;
const P_POST: usize = 2;
const P_ATTN: usize = 3;
const P_SILU: usize = 4;
const P_COMB: usize = 5;
/// The one-workgroup-per-head attention (ATTENTION_HEAD), for GPUs where the split one is unsafe.
const P_ATTN_HEAD: usize = 6;
/// Q4_K matrix kernels (P_MATVEC's layout): (label, rows per workgroup, tokens per weight read),
/// from pipeline P_Q4K on, in this order.
const Q4K_VARIANTS: [(&str, u32, usize); 8] = [
    ("Q4_K 1 row x 64 threads", 1, 1),
    ("Q4_K 1 row x 64 threads", 1, 8),
    ("Q4_K 2 rows x 32 threads", 2, 1),
    ("Q4_K 2 rows x 32 threads", 2, 8),
    ("Q4_K 1 row x 32 threads", 1, 1),
    ("Q4_K 1 row x 32 threads", 1, 8),
    ("Q4_K 1 row x 128 threads", 1, 1),
    ("Q4_K 1 row x 128 threads", 1, 8),
];
const P_Q4K: usize = 7;
/// Attention positions per workgroup (SPAN in the shader).
const SPAN: usize = 256;
/// Matrix kernel variants, all with P_MATVEC's layout: (label, family, threads, tokens per weight read).
const MV_VARIANTS: [(&str, &str, u32, usize); 6] = [
    ("4 rows x 64 threads", "quad", 256, 1),
    ("1 row x 64 threads", "row", 64, 1),
    ("1 row x 32 threads", "row", 32, 1),
    ("4 rows x 64 threads", "quad", 256, 8),
    ("1 row x 64 threads", "row", 64, 8),
    ("1 row x 32 threads", "row", 32, 8),
];
/// Pipelines from index MV0 on are the MV_VARIANTS, in order.
const MV0: usize = 15;
const TB: usize = 8;
/// A submission should take about this long: Windows resets a GPU driver whose work runs past two
/// seconds, and an integrated GPU shares itself with the desktop, which must stay responsive.
const PASS_SECONDS: f64 = 0.2;
/// Most tokens per pass; fewer when the attention scratch would exceed SCRATCH bytes.
const CHUNK: usize = 32;
const SCRATCH: usize = 256 << 20;

/// Repacks rows of `m` (any GGUF format) into int8 words and per-32 scales.
fn repack(model: &Model, m: &Mat, r0: usize, rows: usize, w: &mut [u8], s: &mut [u8]) {
    let g = model.gguf();
    let cols = m.cols;
    let all = g.bytes_at(m.offset, m.rows * m.row_bytes);
    let mut row = vec![0f32; cols];
    for r in 0..rows {
        dequant_row(m.dtype, &all[(r0 + r) * m.row_bytes..(r0 + r + 1) * m.row_bytes], &mut row);
        for b in 0..cols / 32 {
            let blk = &row[b * 32..b * 32 + 32];
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            let si = (r * cols / 32 + b) * 4;
            s[si..si + 4].copy_from_slice(&d.to_le_bytes());
            for (k, v) in blk.iter().enumerate() {
                w[r * cols + b * 32 + k] = ((v * inv).round().clamp(-127.0, 127.0) as i8) as u8;
            }
        }
    }
}

impl PortableState {
    /// Whether this backend runs `m`: dense attention layers, head size up to 256, rows of 32s.
    pub fn supports(m: &Model) -> Result<(), Error> {
        let c = &m.cfg;
        if c.ssm.is_some() || c.attn_gate {
            return Err(Error::Unsupported("the portable GPU backend does not run hybrid (DeltaNet) layers yet".into()));
        }
        if c.n_expert > 0 || m.layers().iter().any(|l| !matches!(l.ffn, Ffn::Dense { .. })) {
            return Err(Error::Unsupported("the portable GPU backend does not run mixture-of-experts models yet".into()));
        }
        if c.head_dim > 256 || c.head_dim % 16 != 0 || c.rope_dim > c.head_dim {
            return Err(Error::Unsupported(format!("head size {} is not supported on the portable GPU backend", c.head_dim)));
        }
        if c.dim % 32 != 0 || c.ffn % 32 != 0 || (c.n_head * c.head_dim) % 32 != 0 {
            return Err(Error::Unsupported("matrix widths must be multiples of 32".into()));
        }
        Ok(())
    }

    pub fn load(m: &Model, max_ctx: usize) -> Result<Self, Error> {
        Self::supports(m)?;
        let trace = |at: &str| if std::env::var_os("CEREBRO_PORTABLE_TRACE").is_some() { eprintln!("portable: {at}") };
        let inst = instance();
        let adapter = pick_adapter(&inst)?;
        trace("adapter");
        let ai = adapter.get_info();
        let limits = adapter.limits();
        let info = PortableGpu { name: ai.name.clone(), backend: format!("{:?}", ai.backend), kind: format!("{:?}", ai.device_type), vendor: ai.vendor, max_buffer: limits.max_buffer_size };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("cerebro"),
            required_features: wgpu::Features::empty(),
            // Only the limits this backend needs above the defaults.
            required_limits: wgpu::Limits { max_buffer_size: limits.max_buffer_size, max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size, ..wgpu::Limits::default() },
            ..Default::default()
        }))
        .map_err(perr)?;
        trace("device");
        let error: std::sync::Arc<std::sync::Mutex<Option<String>>> = Default::default();
        let sink = error.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
            let mut g = sink.lock().unwrap_or_else(|p| p.into_inner());
            if g.is_none() {
                *g = Some(e.to_string());
            }
        }));
        let sink = error.clone();
        device.set_device_lost_callback(move |reason, msg| {
            let mut g = sink.lock().unwrap_or_else(|p| p.into_inner());
            if g.is_none() {
                *g = Some(format!("the graphics driver reset the GPU ({reason:?}: {msg})"));
            }
        });
        let c = &m.cfg;
        let (dim, hd, nh, nkv, ffn) = (c.dim, c.head_dim, c.n_head, c.n_kv, c.ffn);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let row = qd + 2 * kvd;
        let max_bind = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size) as usize;
        // Span partials: per (token, head, span) the max, the sum and hd values; or, for the per-head
        // kernel, every score of every head.
        let per_head = ai.vendor == INTEL;
        let maxs = max_ctx.div_ceil(SPAN);
        let part_row = if per_head { nh * max_ctx * 4 } else { nh * maxs * (hd + 2) * 4 };
        let chunk = (SCRATCH / part_row).clamp(1, CHUNK);
        let mut bytes = 0u64;

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let buf = |n: usize, label: &str| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (n.max(16) as u64 + 15) & !15, usage: storage, mapped_at_creation: false });
        let f32s = |v: &[f32], label: &str| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(), usage: storage });
        let uni = |words: &[u32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &words.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(), usage: wgpu::BufferUsages::UNIFORM });

        // Q4_K matrices stay in the file's blocks on a separate graphics card, where memory is what
        // runs out: half the int8 copy, and as fast or faster (RTX 3070 Ti via Vulkan, Qwen3-0.6B
        // Q4_K_M: 49.0 vs 45.9 tok/s). An integrated GPU uses system memory and is short of
        // arithmetic, so it keeps int8 (Iris Xe: 24.2 native vs 33.8 tok/s int8).
        // CEREBRO_PORTABLE_INT8=1 / =0 forces either.
        let native = match std::env::var("CEREBRO_PORTABLE_INT8").ok().as_deref() {
            Some("1") => false,
            Some("0") => true,
            _ => ai.device_type != wgpu::DeviceType::IntegratedGpu,
        };
        let is_q4k = |mm: &Mat| native && mm.dtype == crate::quant::DType::Q4K && mm.cols % 256 == 0;
        let placeholder = device.create_buffer(&wgpu::BufferDescriptor { label: Some("no scales"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        // Stacks matrices of one width row-wise (q|k|v, gate|up) and uploads them in parts that each
        // fit one binding; neighbouring matrices of the same kind share parts.
        let upload = |mats: &[&Mat], label: &str| -> Result<GMat, Error> {
            let cols = mats[0].cols;
            if mats.iter().any(|x| x.cols != cols) {
                return Err(Error::Format("stacked matrices differ in width".into()));
            }
            let mut parts = Vec::new();
            let mut stack_row = 0;
            let mut k = 0;
            while k < mats.len() {
                let q4k = is_q4k(mats[k]);
                let mut e = k + 1;
                while e < mats.len() && is_q4k(mats[e]) == q4k {
                    e += 1;
                }
                let group = &mats[k..e];
                let total: usize = group.iter().map(|x| x.rows).sum();
                let row_bytes = if q4k { mats[k].row_bytes } else { cols };
                // A multiple of four rows (the 4-row kernel's share) per part.
                let per = ((max_bind / row_bytes) & !3).clamp(4, total.next_multiple_of(4));
                let mut row0 = 0;
                while row0 < total {
                    let rows = per.min(total - row0);
                    let mut wv = vec![0u8; rows * row_bytes];
                    let mut sv = if q4k { Vec::new() } else { vec![0u8; rows * cols / 32 * 4] };
                    // Rows [row0, row0 + rows) of the group, matrix by matrix.
                    let mut at = 0;
                    let mut start = 0;
                    for mm in group {
                        let (a, b) = (start.max(row0), (start + mm.rows).min(row0 + rows));
                        if a < b {
                            let n = b - a;
                            if q4k {
                                let src = m.gguf().bytes_at(mm.offset + (a - start) * row_bytes, n * row_bytes);
                                wv[at * row_bytes..(at + n) * row_bytes].copy_from_slice(src);
                            } else {
                                repack(m, mm, a - start, n, &mut wv[at * cols..(at + n) * cols], &mut sv[at * cols / 32 * 4..(at + n) * cols / 32 * 4]);
                            }
                            at += n;
                        }
                        start += mm.rows;
                    }
                    let w = device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: wv.len() as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: true });
                    // A buffer the GPU had no memory for comes back invalid and cannot be mapped.
                    w.slice(..).get_mapped_range_mut().map_err(|_| no_room())?.copy_from_slice(&wv);
                    w.unmap();
                    let s = if q4k {
                        placeholder.clone()
                    } else {
                        let s = device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: sv.len() as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: true });
                        s.slice(..).get_mapped_range_mut().map_err(|_| no_room())?.copy_from_slice(&sv);
                        s.unmap();
                        s
                    };
                    parts.push(Part { w, s, row0: stack_row + row0, rows, q4k, bytes: (wv.len() + sv.len()) as u64 });
                    row0 += rows;
                }
                stack_row += total;
                k = e;
            }
            Ok(GMat { parts, cols })
        };

        let x = buf(chunk * dim * 4, "x");
        let h = buf(chunk * dim * 4, "h");
        let qkv = buf(chunk * row * 4, "qkv");
        let ao = buf(chunk * qd * 4, "attn out");
        let gu = buf(chunk * 2 * ffn * 4, "gate up");
        let logits = buf(c.vocab * 4, "logits");
        let part = buf(chunk * part_row, "attention spans");
        let step = buf(16 + chunk * c.rope_dim / 2 * 8, "step");
        let readback = device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: (c.vocab * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let dummy = f32s(&vec![0.0; hd], "no norm");
        bytes += (chunk * (dim * 8 + row * 4 + qd * 4 + 8 * ffn + part_row) + c.vocab * 8) as u64;

        // Layouts: binding 0 the call's parameters, 1 the chunk's step record, then storage
        // buffers (true = read only).
        let layout = |rw: &[bool]| {
            let mut e = vec![
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ];
            for (k, &ro) in rw.iter().enumerate() {
                e.push(wgpu::BindGroupLayoutEntry { binding: 2 + k as u32, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None }, count: None });
            }
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &e })
        };
        let layouts = [
            layout(&[true, true, false]),
            layout(&[true, true, true, false]),
            layout(&[false, true, true, false, false]),
            layout(&[true, true, true, false]),
            layout(&[false]),
            layout(&[true, false]),
            layout(&[true, true, true, false, false]),
        ];
        let pipe = |src: String, li: usize, label: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layouts[li])], immediate_size: 0 });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        };
        trace("buffers");
        let pipes = vec![
            pipe(format!("{STEP}{RMSNORM}"), P_RMS, "rmsnorm"),
            pipe(format!("const TB: u32 = 1u;{STEP}{MATVEC}"), P_MATVEC, "matvec"),
            pipe(format!("{STEP}{QK_POST}"), P_POST, "qk post"),
            pipe(format!("{STEP}{ATTENTION}"), P_ATTN, "attention"),
            pipe(format!("{STEP}{SILU}"), P_SILU, "silu"),
            pipe(format!("{STEP}{ATTN_COMBINE}"), P_COMB, "attention combine"),
            pipe(format!("{STEP}{ATTENTION_HEAD}"), P_ATTN_HEAD, "attention per head"),
            pipe(format!("const TB: u32 = 1u; const WG: u32 = 64u;{STEP}{MATVEC_Q4K}"), P_MATVEC, "matvec q4k"),
            pipe(format!("const TB: u32 = {TB}u; const WG: u32 = 64u;{STEP}{MATVEC_Q4K}"), P_MATVEC, "matvec q4k batch"),
            pipe(format!("const TB: u32 = 1u;{STEP}{MATVEC_Q4K2}"), P_MATVEC, "matvec q4k 2 rows"),
            pipe(format!("const TB: u32 = {TB}u;{STEP}{MATVEC_Q4K2}"), P_MATVEC, "matvec q4k 2 rows batch"),
            pipe(format!("const TB: u32 = 1u; const WG: u32 = 32u;{STEP}{MATVEC_Q4K}"), P_MATVEC, "matvec q4k 32"),
            pipe(format!("const TB: u32 = {TB}u; const WG: u32 = 32u;{STEP}{MATVEC_Q4K}"), P_MATVEC, "matvec q4k 32 batch"),
            pipe(format!("const TB: u32 = 1u; const WG: u32 = 128u;{STEP}{MATVEC_Q4K}"), P_MATVEC, "matvec q4k 128"),
            pipe(format!("const TB: u32 = {TB}u; const WG: u32 = 128u;{STEP}{MATVEC_Q4K}"), P_MATVEC, "matvec q4k 128 batch"),
        ];
        let mut pipes = pipes;
        for (label, family, wg, tb) in MV_VARIANTS {
            let src = if family == "quad" { format!("const TB: u32 = {tb}u;{STEP}{MATVEC}") } else { format!("const TB: u32 = {tb}u; const WG: u32 = {wg}u;{STEP}{MATVEC_ROW}") };
            pipes.push(pipe(src, P_MATVEC, label));
        }
        trace("pipelines");
        let bind = |li: usize, bufs: &[&wgpu::Buffer]| {
            let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layouts[li], entries: &entries })
        };
        // out[t * out_stride + row0 + r] for each part; `inp` is read with a row stride of `in_stride`.
        let matvec = |g: &GMat, input: &wgpu::Buffer, in_stride: usize, out: &wgpu::Buffer, out_stride: usize, add: bool, fixed: bool, calls: &mut Vec<Call>| {
            for p in &g.parts {
                let u = uni(&[p.rows as u32, g.cols as u32, add as u32, p.row0 as u32, out_stride as u32, in_stride as u32, fixed as u32, 0]);
                calls.push(Call { pipe: P_MATVEC, bind: bind(P_MATVEC, &[&u, &step, &p.w, &p.s, input, out]), dims: Dims::Rows { rows: p.rows as u32, fixed, q4k: p.q4k }, io: Some((input.clone(), out.clone(), add)) });
            }
        };
        let eps = c.eps.to_bits();
        let rms_u = uni(&[dim as u32, eps, 0, 0]);
        let comb_u = uni(&[hd as u32, maxs as u32, nh as u32, 0]);
        // Layers go to the card until it is full (wgpu cannot say how much memory a card has, so
        // filling it is the measurement); the layer that did not fit and one more, as headroom for
        // the driver and the desktop, go back. The processor runs the rest and the output head.
        // A non-zero Layers on the graphics card switch (or CEREBRO_PORTABLE_LAYERS) caps the count.
        let n_layer = m.layers().len();
        let cap = std::env::var("CEREBRO_PORTABLE_LAYERS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(crate::switches::get(crate::switches::GPU_LAYERS) as usize);
        let limit = if cap == 0 { n_layer } else { cap.min(n_layer) };
        let is_oom = |e: &str| e.to_lowercase().contains("memory");
        let mut layer_calls = Vec::new();
        let mut ran_out = false;
        for l in m.layers().iter().take(limit) {
            let built = (|| -> Result<(Vec<Call>, u64), Error> {
                let mut calls = Vec::new();
                let Ffn::Dense { gate, up, down } = &l.ffn else { unreachable!("checked in supports") };
                let attn_norm = f32s(&l.attn_norm, "attn norm");
                let ffn_norm = f32s(&l.ffn_norm, "ffn norm");
                let wqkv = upload(&[&l.q, &l.k, &l.v], "qkv")?;
                let wo = upload(&[&l.o], "o")?;
                let wgu = upload(&[gate, up], "gate up")?;
                let wd = upload(&[down], "down")?;
                let mut lb = wqkv.size() + wo.size() + wgu.size() + wd.size();
                let kc = buf(max_ctx * kvd * 2, "k cache");
                let vc = buf(max_ctx * kvd * 2, "v cache");
                lb += (max_ctx * kvd * 4) as u64;
                let qn = l.q_norm.as_ref().map(|v| f32s(v, "q norm"));
                let kn = l.k_norm.as_ref().map(|v| f32s(v, "k norm"));
                calls.push(Call { pipe: P_RMS, bind: bind(P_RMS, &[&rms_u, &step, &x, &attn_norm, &h]), dims: Dims::PerToken, io: None });
                matvec(&wqkv, &h, dim, &qkv, row, false, false, &mut calls);
                let post_u = uni(&[nh as u32, nkv as u32, hd as u32, c.rope_dim as u32, kvd as u32, qn.is_some() as u32, kn.is_some() as u32, eps, qd as u32, row as u32, 0, 0]);
                calls.push(Call { pipe: P_POST, bind: bind(P_POST, &[&post_u, &step, &qkv, qn.as_ref().unwrap_or(&dummy), kn.as_ref().unwrap_or(&dummy), &kc, &vc]), dims: Dims::Heads((nh + nkv) as u32), io: None });
                let attn_u = uni(&[(nh / nkv) as u32, hd as u32, kvd as u32, maxs as u32, (1.0 / (hd as f32).sqrt()).to_bits(), row as u32, nh as u32, 0]);
                if per_head {
                    let head_u = uni(&[(nh / nkv) as u32, hd as u32, kvd as u32, max_ctx as u32, (1.0 / (hd as f32).sqrt()).to_bits(), row as u32, nh as u32, 0]);
                    calls.push(Call { pipe: P_ATTN_HEAD, bind: bind(P_ATTN_HEAD, &[&head_u, &step, &qkv, &kc, &vc, &part, &ao]), dims: Dims::Heads(nh as u32), io: None });
                } else {
                    calls.push(Call { pipe: P_ATTN, bind: bind(P_ATTN, &[&attn_u, &step, &qkv, &kc, &vc, &part]), dims: Dims::Spans(nh as u32), io: None });
                    calls.push(Call { pipe: P_COMB, bind: bind(P_COMB, &[&comb_u, &step, &part, &ao]), dims: Dims::Heads(nh as u32), io: None });
                }
                matvec(&wo, &ao, qd, &x, dim, true, false, &mut calls);
                calls.push(Call { pipe: P_RMS, bind: bind(P_RMS, &[&rms_u, &step, &x, &ffn_norm, &h]), dims: Dims::PerToken, io: None });
                matvec(&wgu, &h, dim, &gu, 2 * ffn, false, false, &mut calls);
                let silu_u = uni(&[ffn as u32, 0, 0, 0]);
                calls.push(Call { pipe: P_SILU, bind: bind(P_SILU, &[&silu_u, &step, &gu]), dims: Dims::Elems(ffn as u32), io: None });
                matvec(&wd, &gu, 2 * ffn, &x, dim, true, false, &mut calls);
                Ok((calls, lb))
            })();
            // Weights are written through staging copies that wgpu frees only after a submission:
            // without one per layer they pile up and run the upload pool out of memory long before
            // the card is full (an 8 GB card stopped at 3.2 GiB).
            queue.submit([]);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let reported = error.lock().unwrap_or_else(|p| p.into_inner()).take();
            match (built, reported) {
                (Ok((calls, lb)), None) => {
                    bytes += lb;
                    layer_calls.push(calls);
                }
                (_, Some(e)) if !is_oom(&e) => return Err(Error::Unsupported(format!("portable GPU: {e}"))),
                (Err(e), None) if !matches!(&e, Error::Unsupported(t) if t.starts_with("the model does not fit")) => return Err(e),
                (built, reported) => {
                    trace(&format!("layer {} did not fit: {}", layer_calls.len(), reported.unwrap_or_else(|| built.err().map_or_else(String::new, |e| e.to_string()))));
                    ran_out = true;
                    break;
                }
            }
        }
        trace("layers");
        let mut head_calls = Vec::new();
        if !ran_out && layer_calls.len() == n_layer {
            let out_norm = f32s(m.out_norm(), "output norm");
            let built = upload(&[m.output_mat()], "output");
            let reported = error.lock().unwrap_or_else(|p| p.into_inner()).take();
            match (built, reported) {
                (Ok(wout), None) => {
                    bytes += wout.size();
                    let last_u = uni(&[dim as u32, eps, 1, 0]);
                    head_calls.push(Call { pipe: P_RMS, bind: bind(P_RMS, &[&last_u, &step, &x, &out_norm, &h]), dims: Dims::One, io: None });
                    matvec(&wout, &h, dim, &logits, 0, false, true, &mut head_calls);
                }
                (_, Some(e)) if !is_oom(&e) => return Err(Error::Unsupported(format!("portable GPU: {e}"))),
                _ => ran_out = true,
            }
        }
        if ran_out {
            layer_calls.pop();
        }
        if layer_calls.is_empty() {
            return Err(no_room());
        }
        let readback_x = device.create_buffer(&wgpu::BufferDescriptor { label: Some("split readback"), size: (chunk * dim * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        // Buffers referenced by bind groups live as long as those.
        let n_layer = layer_calls.len();
        // Pick each matrix kernel for this GPU: every candidate runs the model's largest matrix on
        // the same pseudo-random input, the largest group that agrees is trusted (a variant that
        // computes wrong results on some GPU must never be chosen there: one did, on every GPU,
        // before a fix), and the fastest of that group is kept, for one token and for TB tokens.
        let fxc = ai.backend == wgpu::Backend::Dx12 && !has_dxc();
        let mut words = Vec::new();
        // (label, pipeline, rows per workgroup, tokens per weight read)
        let int8: Vec<(&str, usize, u32, usize)> = MV_VARIANTS
            .iter()
            .enumerate()
            // FXC-compiled one-row kernels hung over several tokens (see dxc_path).
            .filter(|(_, v)| !(fxc && v.3 > 1 && v.1 == "row"))
            .map(|(i, v)| (v.0, MV0 + i, if v.1 == "quad" { 4 } else { 1 }, v.3))
            .collect();
        let q4k: Vec<(&str, usize, u32, usize)> = Q4K_VARIANTS.iter().enumerate().map(|(i, v)| (v.0, P_Q4K + i, v.1, v.2)).collect();
        let mut warmed = false;
        let mut select = |family: &[(&str, usize, u32, usize)], want_q4k: bool, words: &mut Vec<String>| -> Result<Option<[(usize, u32); 2]>, Error> {
            let probe = layer_calls
                .iter()
                .flatten()
                .filter(|c| matches!(c.dims, Dims::Rows { q4k, .. } if q4k == want_q4k) && matches!(c.io, Some((_, _, false))))
                .max_by_key(|c| if let Dims::Rows { rows, .. } = c.dims { rows } else { 0 });
            let Some(Call { bind: bindg, dims: Dims::Rows { rows, .. }, io: Some((input, output, _)), .. }) = probe else { return Ok(None) };
            let rows = *rows;
            let run = |pipe: usize, per: u32, tb: usize, nt: usize, times: usize| -> Result<f64, Error> {
                let wg = rows.div_ceil(per);
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    pass.set_pipeline(&pipes[pipe]);
                    pass.set_bind_group(0, bindg, &[]);
                    for _ in 0..times {
                        pass.dispatch_workgroups(wg.min(65535), wg.div_ceil(65535), (nt as u32).div_ceil(tb as u32));
                    }
                }
                let t0 = std::time::Instant::now();
                let idx = queue.submit([enc.finish()]);
                device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).map_err(perr)?;
                Ok(t0.elapsed().as_secs_f64())
            };
            let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("kernel check"), size: output.size(), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let read_out = || -> Result<Vec<f32>, Error> {
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                enc.copy_buffer_to_buffer(output, 0, &staging, 0, output.size());
                queue.submit([enc.finish()]);
                let slice = staging.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::PollType::wait_indefinitely()).map_err(perr)?;
                let v = slice.get_mapped_range().map_err(perr)?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                staging.unmap();
                Ok(v)
            };
            // Pseudo-random input in [-1, 1).
            let n_in = (input.size() / 4) as usize;
            let mut seed = 0x9e37_79b9u32;
            let vals: Vec<u8> = (0..n_in)
                .flat_map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0).to_le_bytes()
                })
                .collect();
            if !warmed {
                // Raise the GPU's clocks first: an idle laptop GPU starts slow and would flatter
                // whichever variant runs last. About a quarter second of the first candidate.
                queue.write_buffer(&step, 0, &Self::step_record(m, 0, 1));
                let warm = std::time::Instant::now();
                while warm.elapsed().as_secs_f64() < 0.25 {
                    run(family[0].1, family[0].2, family[0].3, 1, 50)?;
                }
                warmed = true;
            }
            let mut picked = [(family[0].1, family[0].2); 2];
            for (k, nt) in [(0usize, 1usize), (1, TB)] {
                queue.write_buffer(&step, 0, &Self::step_record(m, 0, nt));
                let cands: Vec<_> = family.iter().filter(|c| (c.3 == 1) == (nt == 1)).collect();
                let mut outs = Vec::new();
                for c in &cands {
                    queue.write_buffer(input, 0, &vals);
                    run(c.1, c.2, c.3, nt, 1)?;
                    outs.push(read_out()?);
                }
                let agree = |a: &[f32], b: &[f32]| {
                    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
                    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-3 * scale)
                };
                let support: Vec<usize> = (0..cands.len()).map(|i| (0..cands.len()).filter(|&j| agree(&outs[i], &outs[j])).count()).collect();
                let top = support.iter().copied().max().unwrap_or(0);
                if top < 2 && cands.len() > 1 {
                    return Err(Error::Unsupported("this graphics card's matrix kernels disagree with each other; the model runs on the processor instead".into()));
                }
                let mut best = f64::MAX;
                for (ci, c) in cands.iter().enumerate() {
                    if support[ci] < top {
                        trace(&format!("{} ({} token(s)) disagrees with the other kernels: not used", c.0, nt));
                        continue;
                    }
                    let mut time = f64::MAX;
                    for _ in 0..2 {
                        time = time.min(run(c.1, c.2, c.3, nt, 20)? / 20.0);
                    }
                    trace(&format!("{}, {nt} token(s): {:.1} us", c.0, time * 1e6));
                    if time < best {
                        best = time;
                        picked[k] = (c.1, c.2);
                    }
                }
                let label = family.iter().find(|c| c.1 == picked[k].0).map_or("?", |c| c.0);
                words.push(format!("{label} for {}", if nt == 1 { "one token" } else { "prompts" }));
            }
            Ok(Some(picked))
        };
        let chosen = select(&int8, false, &mut words)?.unwrap_or([(MV0, 4), (MV0 + 3, 4)]);
        let qk = select(&q4k, true, &mut words)?.unwrap_or([(P_Q4K, 1), (P_Q4K + 1, 1)]);
        let (mut mv1, mut mv8, qk1, mut qk8) = (chosen[0], chosen[1], qk[0], qk[1]);
        // CEREBRO_PORTABLE_MV8 / _Q4K8 = i force variant i for prompts (diagnosis).
        if let Some(i) = std::env::var("CEREBRO_PORTABLE_MV8").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&i| i < MV_VARIANTS.len()) {
            mv8 = (MV0 + i, if MV_VARIANTS[i].1 == "quad" { 4 } else { 1 });
            words.push(format!("forced {} ({} per read) for prompts", MV_VARIANTS[i].0, MV_VARIANTS[i].3));
        }
        if let Some(i) = std::env::var("CEREBRO_PORTABLE_Q4K8").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&i| i < Q4K_VARIANTS.len()) {
            qk8 = (P_Q4K + i, Q4K_VARIANTS[i].1);
        }
        let _ = &mut mv1;
        words.push(if per_head { "attention one head per workgroup".into() } else { "attention in 256-position spans".into() });
        if ai.backend == wgpu::Backend::Dx12 {
            words.push(if fxc { "shaders compiled by FXC".into() } else { "shaders compiled by DXC".into() });
        }
        let kernels = words.join(", ");
        Ok(Self { device, error, queue, info, max_ctx, chunk, x, step, logits, readback, readback_x, pipes, layer_calls, head_calls, pass_tokens: 4.min(chunk), pass_layers: n_layer, mv1, mv8, qk1, qk8, kernels, dim, vocab: c.vocab, bytes })
    }

    /// Where the model went, for the app (the same shape as the CUDA planner's): every layer on this
    /// GPU. No speed estimate: the GPU's memory speed is not known here.
    pub fn plan(&self, m: &Model) -> crate::placement::Plan {
        let n = m.cfg.n_layer;
        let k = self.gpu_layers();
        let api = self.info.backend.replace("Dx12", "DirectX 12");
        // The processor's share, from the file: its layers' fraction of the weights plus the head.
        let file = m.gguf().len() as u64;
        let cpu = if self.is_split() { file * (n - k) as u64 / n as u64 + (m.output_mat().rows * m.output_mat().row_bytes) as u64 } else { 0 };
        let first = if self.is_split() {
            format!("The first {k} of {n} layers on the {} through {api}; the processor runs the other {} and the output head. The card filled up at layer {}.", self.info.name, n - k, k + 1)
        } else {
            format!("All {n} layers on the {} through {api}; the conversation cache as f16.", self.info.name)
        };
        crate::placement::Plan {
            layout: if self.is_split() { crate::placement::Layout::Split } else { crate::placement::Layout::GpuInt8 },
            gpu_layers: k,
            total_layers: n,
            vram_bytes: self.bytes,
            cpu_bytes: cpu,
            ram_resident: cpu,
            drive_bytes_per_token: 0,
            drive_layers: 0,
            est_decode_tps: 0.0,
            drive: crate::placement::drive_of(&m.path),
            notes: vec![first, format!("Kernels picked for this GPU when loading: {}.", self.kernels)],
        }
    }

    /// Layers on the GPU.
    pub fn gpu_layers(&self) -> usize {
        self.layer_calls.len()
    }

    /// The processor runs the layers from `gpu_layers()` on and the output head.
    pub fn is_split(&self) -> bool {
        self.head_calls.is_empty()
    }

    /// The chunk's step record: token count, first position, RoPE pairs per token.
    fn step_record(m: &Model, pos0: usize, n: usize) -> Vec<u8> {
        let half = m.inv_freq().len();
        let mut v = Vec::with_capacity(16 + n * half * 8);
        for w in [n as u32, pos0 as u32, half as u32, 0] {
            v.extend_from_slice(&w.to_le_bytes());
        }
        for t in 0..n {
            for f in m.inv_freq() {
                let (s, c) = ((pos0 + t) as f32 * f).sin_cos();
                v.extend_from_slice(&c.to_le_bytes());
                v.extend_from_slice(&s.to_le_bytes());
            }
        }
        v
    }

    /// Records `calls` for a chunk of `nt` tokens into `pass`.
    fn encode<'a>(&self, pass: &mut wgpu::ComputePass<'a>, calls: impl Iterator<Item = &'a Call>, nt: usize, pos0: usize) {
        let ntu = nt as u32;
        let spans = (pos0 + nt).div_ceil(SPAN) as u32;
        for call in calls {
            let (pipe, d) = match call.dims {
                Dims::PerToken => (call.pipe, (ntu, 1, 1)),
                Dims::One => (call.pipe, (1, 1, 1)),
                Dims::Heads(h) => (call.pipe, (h, ntu, 1)),
                Dims::Elems(n) => (call.pipe, ((ntu * n).div_ceil(256), 1, 1)),
                Dims::Spans(h) => (call.pipe, (h, spans, ntu)),
                Dims::Rows { rows, fixed, q4k: true } => {
                    let ((pipe, per), z) = if fixed || nt == 1 { (self.qk1, 1) } else { (self.qk8, ntu.div_ceil(Q4K_VARIANTS[self.qk8.0 - P_Q4K].2 as u32)) };
                    let wg = rows.div_ceil(per);
                    (pipe, (wg.min(65535), wg.div_ceil(65535), z))
                }
                Dims::Rows { rows, fixed, q4k: false } => {
                    let ((pipe, per), z) = if fixed || nt == 1 { (self.mv1, 1) } else { (self.mv8, ntu.div_ceil(MV_VARIANTS[self.mv8.0 - MV0].3 as u32)) };
                    let wg = rows.div_ceil(per);
                    (pipe, (wg.min(65535), wg.div_ceil(65535), z))
                }
            };
            pass.set_pipeline(&self.pipes[pipe]);
            pass.set_bind_group(0, &call.bind, &[]);
            pass.dispatch_workgroups(d.0, d.1, d.2);
        }
    }

    /// Submits and waits; returns the seconds the GPU took (wall time, including the wait).
    fn submit_wait(&self, enc: wgpu::CommandEncoder) -> Result<f64, Error> {
        let t0 = std::time::Instant::now();
        let idx = self.queue.submit([enc.finish()]);
        let waited = self.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
        if let Some(e) = self.error.lock().unwrap_or_else(|p| p.into_inner()).take() {
            return Err(Error::Unsupported(format!("portable GPU: {e}")));
        }
        waited.map_err(|e| Error::Unsupported(format!("the graphics driver stopped responding ({e})")))?;
        Ok(t0.elapsed().as_secs_f64())
    }

    /// Runs `toks` at positions pos0.. through every layer and the head; writes the last token's
    /// logits.
    pub fn run(&mut self, m: &Model, toks: &[u32], pos0: usize, logits: &mut Vec<f32>) -> Result<(), Error> {
        if self.is_split() {
            return Err(Error::Request("this model is split: the processor runs the head (run_split)".into()));
        }
        self.run_inner(m, toks, pos0, Some(logits), None)
    }

    /// A split: runs `toks` through the GPU's layers and returns their activations (tokens x width)
    /// in `xs`, for the processor's layers.
    pub fn run_split(&mut self, m: &Model, toks: &[u32], pos0: usize, xs: &mut Vec<f32>) -> Result<(), Error> {
        xs.clear();
        self.run_inner(m, toks, pos0, None, Some(xs))
    }

    /// Work goes to the GPU in submissions of `pass_tokens` tokens through `pass_layers` layers,
    /// both adapted so each takes about PASS_SECONDS: fewer tokens first, then fewer layers once a
    /// single token is too slow (long contexts on small GPUs), and back up when there is room.
    fn run_inner(&mut self, m: &Model, toks: &[u32], pos0: usize, logits: Option<&mut Vec<f32>>, mut xs: Option<&mut Vec<f32>>) -> Result<(), Error> {
        if pos0 + toks.len() > self.max_ctx {
            return Err(Error::Request("the context is full".into()));
        }
        let n_layer = self.layer_calls.len();
        let want_logits = logits.is_some();
        let mut emb = vec![0f32; self.dim];
        let mut i = 0;
        while i < toks.len() {
            let nt = self.pass_tokens.min(toks.len() - i);
            let mut xb = Vec::with_capacity(nt * self.dim * 4);
            for &t in &toks[i..i + nt] {
                m.embed_into(t, &mut emb);
                xb.extend(emb.iter().flat_map(|v| v.to_le_bytes()));
            }
            self.queue.write_buffer(&self.x, 0, &xb);
            self.queue.write_buffer(&self.step, 0, &Self::step_record(m, pos0 + i, nt));
            let last = i + nt == toks.len();
            let mut l0 = 0;
            while l0 < n_layer {
                let pl = self.pass_layers.min(n_layer - l0);
                let end = l0 + pl == n_layer;
                let head = want_logits && last && end;
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    let calls = self.layer_calls[l0..l0 + pl].iter().flatten().chain(if head { self.head_calls.iter() } else { [].iter() });
                    self.encode(&mut pass, calls, nt, pos0 + i);
                }
                if head {
                    enc.copy_buffer_to_buffer(&self.logits, 0, &self.readback, 0, (self.vocab * 4) as u64);
                }
                let xbytes = (nt * self.dim * 4) as u64;
                if end && xs.is_some() {
                    enc.copy_buffer_to_buffer(&self.x, 0, &self.readback_x, 0, xbytes);
                }
                let secs = self.submit_wait(enc)?;
                if std::env::var_os("CEREBRO_PORTABLE_TRACE").is_some() {
                    eprintln!("portable: {nt} tokens x {pl} layers at {}: {:.3} s", pos0 + i, secs);
                }
                if end {
                    if let Some(out) = xs.as_deref_mut() {
                        let slice = self.readback_x.slice(..xbytes);
                        slice.map_async(wgpu::MapMode::Read, |_| {});
                        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(perr)?;
                        {
                            let data = slice.get_mapped_range().map_err(|e| Error::Unsupported(format!("the graphics driver stopped responding ({e})")))?;
                            out.extend(data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
                        }
                        self.readback_x.unmap();
                    }
                }
                if secs > PASS_SECONDS {
                    if nt > 1 {
                        self.pass_tokens = ((nt as f64 * PASS_SECONDS / secs) as usize).max(1);
                    } else {
                        self.pass_layers = ((pl as f64 * PASS_SECONDS / secs) as usize).max(1);
                    }
                } else if secs < PASS_SECONDS / 4.0 && pl == self.pass_layers && nt == self.pass_tokens.min(toks.len() - i) {
                    if self.pass_layers < n_layer {
                        self.pass_layers = (self.pass_layers * 2).min(n_layer);
                    } else if nt == self.pass_tokens {
                        self.pass_tokens = (self.pass_tokens * 2).min(self.chunk);
                    }
                }
                l0 += pl;
            }
            i += nt;
        }
        if let Some(lg) = logits {
            let slice = self.readback.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(perr)?;
            {
                let data = slice.get_mapped_range().map_err(|e| Error::Unsupported(format!("the graphics driver stopped responding ({e})")))?;
                lg.clear();
                lg.extend(data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
            }
            self.readback.unmap();
        }
        Ok(())
    }
}
