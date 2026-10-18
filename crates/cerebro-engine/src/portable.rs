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

fn perr(e: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("portable GPU: {e}"))
}

/// One matrix on the GPU, in row parts that each fit a buffer binding.
struct GMat {
    parts: Vec<Part>,
    cols: usize,
}

impl GMat {
    /// Bytes on the GPU: one per weight plus a 4-byte scale per 32.
    fn size(&self) -> u64 {
        self.parts.iter().map(|p| (p.rows * self.cols * 9 / 8) as u64).sum()
    }
}

struct Part {
    w: wgpu::Buffer,
    s: wgpu::Buffer,
    row0: usize,
    rows: usize,
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
    Rows { rows: u32, fixed: bool },
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
const MV0: usize = 7;
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

        // Stacks matrices of one width row-wise (q|k|v, gate|up) and uploads them in parts.
        let upload = |mats: &[&Mat], label: &str| -> Result<GMat, Error> {
            let cols = mats[0].cols;
            if mats.iter().any(|x| x.cols != cols) {
                return Err(Error::Format("stacked matrices differ in width".into()));
            }
            let total: usize = mats.iter().map(|x| x.rows).sum();
            // Parts hold a multiple of four rows (a workgroup's share) and fit one binding.
            let per = ((max_bind / cols) & !3).clamp(4, total.next_multiple_of(4));
            let mut parts = Vec::new();
            let mut row0 = 0;
            while row0 < total {
                let rows = per.min(total - row0);
                let w = device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (rows * cols) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: true });
                let s = device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (rows * cols / 32 * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: true });
                {
                    let mut wv = vec![0u8; rows * cols];
                    let mut sv = vec![0u8; rows * cols / 32 * 4];
                    // Rows [row0, row0 + rows) of the stack, matrix by matrix.
                    let mut at = 0;
                    let mut start = 0;
                    for mm in mats {
                        let (a, b) = (start.max(row0), (start + mm.rows).min(row0 + rows));
                        if a < b {
                            let n = b - a;
                            repack(m, mm, a - start, n, &mut wv[at * cols..(at + n) * cols], &mut sv[at * cols / 32 * 4..(at + n) * cols / 32 * 4]);
                            at += n;
                        }
                        start += mm.rows;
                    }
                    w.slice(..).get_mapped_range_mut().map_err(perr)?.copy_from_slice(&wv);
                    s.slice(..).get_mapped_range_mut().map_err(perr)?.copy_from_slice(&sv);
                }
                w.unmap();
                s.unmap();
                parts.push(Part { w, s, row0, rows });
                row0 += rows;
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
                calls.push(Call { pipe: P_MATVEC, bind: bind(P_MATVEC, &[&u, &step, &p.w, &p.s, input, out]), dims: Dims::Rows { rows: p.rows as u32, fixed } });
            }
        };
        let eps = c.eps.to_bits();
        let rms_u = uni(&[dim as u32, eps, 0, 0]);
        let comb_u = uni(&[hd as u32, maxs as u32, nh as u32, 0]);
        let mut layer_calls = Vec::new();
        for l in m.layers() {
            let mut calls = Vec::new();
            let Ffn::Dense { gate, up, down } = &l.ffn else { unreachable!("checked in supports") };
            let attn_norm = f32s(&l.attn_norm, "attn norm");
            let ffn_norm = f32s(&l.ffn_norm, "ffn norm");
            let wqkv = upload(&[&l.q, &l.k, &l.v], "qkv")?;
            let wo = upload(&[&l.o], "o")?;
            let wgu = upload(&[gate, up], "gate up")?;
            let wd = upload(&[down], "down")?;
            bytes += wqkv.size() + wo.size() + wgu.size() + wd.size();
            let kc = buf(max_ctx * kvd * 2, "k cache");
            let vc = buf(max_ctx * kvd * 2, "v cache");
            bytes += (max_ctx * kvd * 4) as u64;
            let qn = l.q_norm.as_ref().map(|v| f32s(v, "q norm"));
            let kn = l.k_norm.as_ref().map(|v| f32s(v, "k norm"));
            calls.push(Call { pipe: P_RMS, bind: bind(P_RMS, &[&rms_u, &step, &x, &attn_norm, &h]), dims: Dims::PerToken });
            matvec(&wqkv, &h, dim, &qkv, row, false, false, &mut calls);
            let post_u = uni(&[nh as u32, nkv as u32, hd as u32, c.rope_dim as u32, kvd as u32, qn.is_some() as u32, kn.is_some() as u32, eps, qd as u32, row as u32, 0, 0]);
            calls.push(Call { pipe: P_POST, bind: bind(P_POST, &[&post_u, &step, &qkv, qn.as_ref().unwrap_or(&dummy), kn.as_ref().unwrap_or(&dummy), &kc, &vc]), dims: Dims::Heads((nh + nkv) as u32) });
            let attn_u = uni(&[(nh / nkv) as u32, hd as u32, kvd as u32, maxs as u32, (1.0 / (hd as f32).sqrt()).to_bits(), row as u32, nh as u32, 0]);
            if per_head {
                let head_u = uni(&[(nh / nkv) as u32, hd as u32, kvd as u32, max_ctx as u32, (1.0 / (hd as f32).sqrt()).to_bits(), row as u32, nh as u32, 0]);
                calls.push(Call { pipe: P_ATTN_HEAD, bind: bind(P_ATTN_HEAD, &[&head_u, &step, &qkv, &kc, &vc, &part, &ao]), dims: Dims::Heads(nh as u32) });
            } else {
                calls.push(Call { pipe: P_ATTN, bind: bind(P_ATTN, &[&attn_u, &step, &qkv, &kc, &vc, &part]), dims: Dims::Spans(nh as u32) });
                calls.push(Call { pipe: P_COMB, bind: bind(P_COMB, &[&comb_u, &step, &part, &ao]), dims: Dims::Heads(nh as u32) });
            }
            matvec(&wo, &ao, qd, &x, dim, true, false, &mut calls);
            calls.push(Call { pipe: P_RMS, bind: bind(P_RMS, &[&rms_u, &step, &x, &ffn_norm, &h]), dims: Dims::PerToken });
            matvec(&wgu, &h, dim, &gu, 2 * ffn, false, false, &mut calls);
            let silu_u = uni(&[ffn as u32, 0, 0, 0]);
            calls.push(Call { pipe: P_SILU, bind: bind(P_SILU, &[&silu_u, &step, &gu]), dims: Dims::Elems(ffn as u32) });
            matvec(&wd, &gu, 2 * ffn, &x, dim, true, false, &mut calls);
            layer_calls.push(calls);
        }
        trace("layers");
        let out_norm = f32s(m.out_norm(), "output norm");
        let wout = upload(&[m.output_mat()], "output")?;
        bytes += wout.size();
        let last_u = uni(&[dim as u32, eps, 1, 0]);
        let mut head_calls = vec![Call { pipe: P_RMS, bind: bind(P_RMS, &[&last_u, &step, &x, &out_norm, &h]), dims: Dims::One }];
        matvec(&wout, &h, dim, &logits, 0, false, true, &mut head_calls);
        // Buffers referenced by bind groups live as long as those.
        let n_layer = layer_calls.len();
        // Time every matrix kernel variant on this GPU with the first layer's gate|up product (the
        // largest matrix of a layer), for one token and for TB, and keep the fastest of each.
        let probe = layer_calls[0].iter().filter(|c| matches!(c.dims, Dims::Rows { .. })).max_by_key(|c| if let Dims::Rows { rows, .. } = c.dims { rows } else { 0 }).map(|c| (&c.bind, c.dims));
        let mut chosen = [(MV0, 4u32), (MV0 + 3, 4u32)];
        let fxc = ai.backend == wgpu::Backend::Dx12 && !has_dxc();
        let mut words = Vec::new();
        if let Some((bindg, Dims::Rows { rows, .. })) = probe {
            // Raise the GPU's clocks first: an idle laptop GPU starts slow and would flatter
            // whichever variant runs last. About a quarter second of the first variant.
            queue.write_buffer(&step, 0, &Self::step_record(m, 0, 1));
            let warm = std::time::Instant::now();
            while warm.elapsed().as_secs_f64() < 0.25 {
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    pass.set_pipeline(&pipes[MV0]);
                    pass.set_bind_group(0, bindg, &[]);
                    for _ in 0..50 {
                        pass.dispatch_workgroups(rows.div_ceil(4).min(65535), rows.div_ceil(4).div_ceil(65535), 1);
                    }
                }
                let idx = queue.submit([enc.finish()]);
                device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).map_err(perr)?;
            }
            for (k, nt) in [(0usize, 1usize), (1, TB)] {
                queue.write_buffer(&step, 0, &Self::step_record(m, 0, nt));
                let mut best = f64::MAX;
                for (vi, (label, family, _, tb)) in MV_VARIANTS.iter().enumerate() {
                    if (*tb == 1) != (nt == 1) {
                        continue;
                    }
                    // FXC-compiled one-row kernels hung over several tokens (see dxc_path).
                    if fxc && nt > 1 && *family == "row" {
                        continue;
                    }
                    let per = if *family == "quad" { 4 } else { 1 };
                    let wg = rows.div_ceil(per);
                    let d = (wg.min(65535), wg.div_ceil(65535), (nt as u32).div_ceil(*tb as u32));
                    let mut time = f64::MAX;
                    for rep in 0..3 {
                        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                        {
                            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                            pass.set_pipeline(&pipes[MV0 + vi]);
                            pass.set_bind_group(0, bindg, &[]);
                            for _ in 0..if rep == 0 { 1 } else { 20 } {
                                pass.dispatch_workgroups(d.0, d.1, d.2);
                            }
                        }
                        let t0 = std::time::Instant::now();
                        let idx = queue.submit([enc.finish()]);
                        device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).map_err(perr)?;
                        if rep > 0 {
                            time = time.min(t0.elapsed().as_secs_f64() / 20.0);
                        }
                    }
                    trace(&format!("matvec {label}, {nt} token(s): {:.1} us", time * 1e6));
                    if time < best {
                        best = time;
                        chosen[k] = (MV0 + vi, per);
                    }
                }
                words.push(format!("{} for {}", MV_VARIANTS[chosen[k].0 - MV0].0, if nt == 1 { "one token" } else { "prompts" }));
            }
        }
        // CEREBRO_PORTABLE_MV8=i forces variant i of MV_VARIANTS for prompts (diagnosis).
        if let Some(i) = std::env::var("CEREBRO_PORTABLE_MV8").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&i| i < MV_VARIANTS.len()) {
            chosen[1] = (MV0 + i, if MV_VARIANTS[i].1 == "quad" { 4 } else { 1 });
            words.push(format!("forced {} ({} per read) for prompts", MV_VARIANTS[i].0, MV_VARIANTS[i].3));
        }
        let (mv1, mv8) = (chosen[0], chosen[1]);
        words.push(if per_head { "attention one head per workgroup".into() } else { "attention in 256-position spans".into() });
        if ai.backend == wgpu::Backend::Dx12 {
            words.push(if fxc { "shaders compiled by FXC".into() } else { "shaders compiled by DXC".into() });
        }
        let kernels = words.join(", ");
        Ok(Self { device, error, queue, info, max_ctx, chunk, x, step, logits, readback, pipes, layer_calls, head_calls, pass_tokens: 4.min(chunk), pass_layers: n_layer, mv1, mv8, kernels, dim, vocab: c.vocab, bytes })
    }

    /// Where the model went, for the app (the same shape as the CUDA planner's): every layer on this
    /// GPU. No speed estimate: the GPU's memory speed is not known here.
    pub fn plan(&self, m: &Model) -> crate::placement::Plan {
        let n = m.cfg.n_layer;
        crate::placement::Plan {
            layout: crate::placement::Layout::GpuInt8,
            gpu_layers: n,
            total_layers: n,
            vram_bytes: self.bytes,
            cpu_bytes: 0,
            ram_resident: 0,
            drive_bytes_per_token: 0,
            est_decode_tps: 0.0,
            drive: crate::placement::drive_of(&m.path),
            notes: vec![
                format!("All {n} layers on the {} through {}: weights as int8, the conversation cache as f16.", self.info.name, self.info.backend.replace("Dx12", "DirectX 12")),
                format!("Kernels picked for this GPU when loading: {}.", self.kernels),
            ],
        }
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
                Dims::Rows { rows, fixed } => {
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

    /// Runs `toks` at positions pos0..; writes the last token's logits. Work goes to the GPU in
    /// submissions of `pass_tokens` tokens through `pass_layers` layers, both adapted so each takes
    /// about PASS_SECONDS: fewer tokens first, then fewer layers once a single token is too slow
    /// (long contexts on small GPUs), and back up when there is room.
    pub fn run(&mut self, m: &Model, toks: &[u32], pos0: usize, logits: &mut Vec<f32>) -> Result<(), Error> {
        if pos0 + toks.len() > self.max_ctx {
            return Err(Error::Request("the context is full".into()));
        }
        let n_layer = self.layer_calls.len();
        let mut emb = vec![0f32; self.dim];
        let mut i = 0;
        while i < toks.len() {
            let nt = self.pass_tokens.min(toks.len() - i);
            let mut xs = Vec::with_capacity(nt * self.dim * 4);
            for &t in &toks[i..i + nt] {
                m.embed_into(t, &mut emb);
                xs.extend(emb.iter().flat_map(|v| v.to_le_bytes()));
            }
            self.queue.write_buffer(&self.x, 0, &xs);
            self.queue.write_buffer(&self.step, 0, &Self::step_record(m, pos0 + i, nt));
            let last = i + nt == toks.len();
            let mut l0 = 0;
            while l0 < n_layer {
                let pl = self.pass_layers.min(n_layer - l0);
                let head = last && l0 + pl == n_layer;
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    let calls = self.layer_calls[l0..l0 + pl].iter().flatten().chain(if head { self.head_calls.iter() } else { [].iter() });
                    self.encode(&mut pass, calls, nt, pos0 + i);
                }
                if head {
                    enc.copy_buffer_to_buffer(&self.logits, 0, &self.readback, 0, (self.vocab * 4) as u64);
                }
                let secs = self.submit_wait(enc)?;
                if std::env::var_os("CEREBRO_PORTABLE_TRACE").is_some() {
                    eprintln!("portable: {nt} tokens x {pl} layers at {}: {:.3} s", pos0 + i, secs);
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
        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(perr)?;
        {
            let data = slice.get_mapped_range().map_err(|e| Error::Unsupported(format!("the graphics driver stopped responding ({e})")))?;
            logits.clear();
            logits.extend(data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
        }
        self.readback.unmap();
        Ok(())
    }
}
