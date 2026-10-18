//! Portable GPU backend: one set of WGSL compute shaders run through wgpu on Vulkan and DX12 (AMD,
//! Intel and NVIDIA GPUs on Windows and Linux) and Metal (Apple GPUs). It runs dense attention
//! models (Llama, Qwen3, Mistral-style) whole on the GPU.
//!
//! Weights of any GGUF format are repacked at load to int8 with one f32 scale per 32 values (the
//! CUDA int8 layout): one matrix kernel serves every format, and Q8_0 files repack exactly. The KV
//! cache is f16. Activations stay f32 throughout. Every token is one submission: the embedding is
//! written from the CPU, nine dispatches per layer run in one compute pass, and the logits are read
//! back. A prompt is read as several tokens per submission.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::model::{Ffn, Mat, Model};
use crate::quant::dequant_row;
use crate::Error;

const RMSNORM: &str = r#"
struct P { n: u32, eps: f32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
    let i0 = l.x;
    var s = 0.0;
    for (var i = i0; i < p.n; i += 256u) { let v = x[i]; s += v * v; }
    red[i0] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (i0 < k) { red[i0] += red[i0 + k]; }
        workgroupBarrier();
    }
    let r = 1.0 / sqrt(red[0] / f32(p.n) + p.eps);
    for (var i = i0; i < p.n; i += 256u) { o[i] = x[i] * r * w[i]; }
}
"#;

/// out[off + r] (+)= scale-per-32 int8 row r . x. One workgroup per row; rows past 65535 use y.
const MATVEC: &str = r#"
struct P { rows: u32, cols: u32, add: u32, off: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> s: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let r = g.x + g.y * 65535u;
    if (r >= p.rows) { return; }
    let wpr = p.cols / 4u;
    let base = r * wpr;
    var acc = 0.0;
    for (var i = l.x; i < wpr; i += 64u) {
        let q = i32(w[base + i]);
        let v = vec4<f32>(f32(extractBits(q, 0u, 8u)), f32(extractBits(q, 8u, 8u)), f32(extractBits(q, 16u, 8u)), f32(extractBits(q, 24u, 8u)));
        acc += s[(base + i) / 8u] * dot(v, x[i]);
    }
    red[l.x] = acc;
    workgroupBarrier();
    for (var k = 32u; k > 0u; k >>= 1u) {
        if (l.x < k) { red[l.x] += red[l.x + k]; }
        workgroupBarrier();
    }
    if (l.x == 0u) {
        if (p.add == 1u) { o[p.off + r] += red[0]; } else { o[p.off + r] = red[0]; }
    }
}
"#;

const STEP: &str = r#"
struct Step { pos: u32, len: u32, _a: u32, _b: u32, cs: array<vec2<f32>> }
"#;

/// Per head: optional RMS norm, RoPE on the first rd values, then K and V (f16) into the cache.
/// Workgroups 0..n_head are query heads, the rest KV heads.
const QK_POST: &str = r#"
struct P { n_head: u32, n_kv: u32, hd: u32, rd: u32, kvd: u32, q_norm: u32, k_norm: u32, eps: f32, qd: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
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
    let is_q = g.x < p.n_head;
    var head = g.x;
    var off = g.x * p.hd;
    var norm = p.q_norm;
    if (!is_q) { head = g.x - p.n_head; off = p.qd + head * p.hd; norm = p.k_norm; }
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
    var o = v;
    if (i < half) {
        let c = st.cs[i];
        o = hv[i] * c.x - hv[i + half] * c.y;
    } else if (i < p.rd) {
        let c = st.cs[i - half];
        o = hv[i - half] * c.y + hv[i] * c.x;
    }
    workgroupBarrier();
    hv[i] = o;
    workgroupBarrier();
    if (is_q) {
        if (i < p.hd) { qkv[off + i] = o; }
    } else if (i < p.hd / 2u) {
        let dst = (st.pos * p.kvd + head * p.hd) / 2u + i;
        kc[dst] = pack2x16float(vec2<f32>(hv[2u * i], hv[2u * i + 1u]));
        let vo = p.qd + p.kvd + head * p.hd;
        vc[dst] = pack2x16float(vec2<f32>(qkv[vo + 2u * i], qkv[vo + 2u * i + 1u]));
    }
}
"#;

/// One workgroup per query head over positions 0..len: scores, softmax, weighted values.
const ATTENTION: &str = r#"
struct P { group: u32, hd: u32, kvd: u32, max_ctx: u32, scale: f32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: Step;
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
    let i = l3.x;
    let kvh = h / p.group;
    let len = st.len;
    let pairs = p.hd / 2u;
    if (i < p.hd) { qs[i] = qkv[h * p.hd + i]; }
    workgroupBarrier();
    let base = h * p.max_ctx;
    var mx = -3.0e38;
    for (var t = i; t < len; t += 256u) {
        let kb = (t * p.kvd + kvh * p.hd) / 2u;
        var s = 0.0;
        for (var j = 0u; j < pairs; j++) {
            let k = unpack2x16float(kc[kb + j]);
            s += qs[2u * j] * k.x + qs[2u * j + 1u] * k.y;
        }
        s *= p.scale;
        att[base + t] = s;
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
    for (var t = i; t < len; t += 256u) {
        let e = exp(att[base + t] - m);
        att[base + t] = e;
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
        for (var t = gi; t < len; t += ng) {
            a += att[base + t] * unpack2x16float(vc[(t * p.kvd + kvh * p.hd) / 2u + pi]);
        }
    }
    acc2[i] = a;
    workgroupBarrier();
    if (gi == 0u) {
        var s2 = a;
        for (var k = 1u; k < ng; k++) { s2 += acc2[k * pairs + pi]; }
        ao[h * p.hd + 2u * pi] = s2.x / total;
        ao[h * p.hd + 2u * pi + 1u] = s2.y / total;
    }
}
"#;

/// gu[i] = silu(gu[i]) * gu[n + i] for i < n.
const SILU: &str = r#"
struct P { n: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> gu: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < p.n) {
        let g = gu[i];
        gu[i] = g / (1.0 + exp(-g)) * gu[p.n + i];
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

fn instance() -> wgpu::Instance {
    let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
    d.backends = backends();
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
    if let Ok(want) = std::env::var("CEREBRO_PORTABLE_GPU") {
        let w = want.to_lowercase();
        return all.into_iter().find(|a| a.get_info().name.to_lowercase().contains(&w)).ok_or_else(|| Error::Unsupported(format!("no GPU named like {want}")));
    }
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
    all.into_iter().min_by_key(rank).ok_or_else(|| Error::Unsupported("no GPU found through Vulkan, DX12 or Metal".into()))
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

/// One prepared dispatch.
struct Call {
    pipe: usize,
    bind: wgpu::BindGroup,
    groups: (u32, u32),
}

pub struct PortableState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub info: PortableGpu,
    max_ctx: usize,
    x: wgpu::Buffer,
    step: wgpu::Buffer,
    logits: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Staging for a chunk of prompt tokens: embeddings and step records.
    x_all: wgpu::Buffer,
    step_all: wgpu::Buffer,
    pipes: Vec<wgpu::ComputePipeline>,
    /// Every layer's dispatches in order, then the output head's.
    layer_calls: Vec<Call>,
    head_calls: Vec<Call>,
    step_bytes: usize,
    dim: usize,
    vocab: usize,
    pub bytes: u64,
    _keep: Vec<Arc<wgpu::Buffer>>,
}

const P_RMS: usize = 0;
const P_MATVEC: usize = 1;
const P_POST: usize = 2;
const P_ATTN: usize = 3;
const P_SILU: usize = 4;
/// Prompt tokens per submission.
const CHUNK: usize = 32;

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
        if c.head_dim > 256 || c.head_dim % 2 != 0 || c.rope_dim > c.head_dim {
            return Err(Error::Unsupported(format!("head size {} is not supported on the portable GPU backend", c.head_dim)));
        }
        if c.dim % 32 != 0 || c.ffn % 32 != 0 {
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
        let c = &m.cfg;
        let (dim, hd, nh, nkv, ffn) = (c.dim, c.head_dim, c.n_head, c.n_kv, c.ffn);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let max_bind = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size) as usize;
        let mut bytes = 0u64;

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let buf = |n: usize, label: &str| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (n.max(4) as u64 + 3) & !3, usage: storage, mapped_at_creation: false });
        let f32s = |v: &[f32], label: &str| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(), usage: storage });
        let uni = |words: &[u32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &words.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(), usage: wgpu::BufferUsages::UNIFORM });

        // Stacks matrices of one width row-wise (q|k|v, gate|up) and uploads them in parts.
        let upload = |mats: &[&Mat], label: &str| -> Result<GMat, Error> {
            let cols = mats[0].cols;
            if mats.iter().any(|x| x.cols != cols) {
                return Err(Error::Format("stacked matrices differ in width".into()));
            }
            let total: usize = mats.iter().map(|x| x.rows).sum();
            let per = (max_bind / cols).clamp(1, total);
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

        let x = buf(dim * 4, "x");
        let h = buf(dim * 4, "h");
        let qkv = buf((qd + 2 * kvd) * 4, "qkv");
        let ao = buf(qd * 4, "attn out");
        let gu = buf(2 * ffn * 4, "gate up");
        let logits = buf(c.vocab * 4, "logits");
        let att = buf(nh * max_ctx * 4, "scores");
        let step_bytes = 16 + c.rope_dim / 2 * 8;
        let step = buf(step_bytes, "step");
        let x_all = buf(CHUNK * dim * 4, "x chunk");
        let step_all = buf(CHUNK * step_bytes, "step chunk");
        let readback = device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: (c.vocab * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let dummy = f32s(&vec![0.0; hd], "no norm");
        bytes += (dim * 8 + (qd + 2 * kvd) * 4 + qd * 4 + 8 * ffn + c.vocab * 8 + nh * max_ctx * 4) as u64;

        let pipe = |src: &str, label: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        };
        trace("buffers");
        let pipes = vec![
            pipe(RMSNORM, "rmsnorm"),
            pipe(MATVEC, "matvec"),
            pipe(&format!("{STEP}{QK_POST}"), "qk post"),
            pipe(&format!("{STEP}{ATTENTION}"), "attention"),
            pipe(SILU, "silu"),
        ];
        trace("pipelines");
        let bind = |pi: usize, bufs: &[&wgpu::Buffer]| {
            let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipes[pi].get_bind_group_layout(0), entries: &entries })
        };
        let groups = |rows: usize| ((rows.min(65535)) as u32, rows.div_ceil(65535) as u32);
        let mut keep = Vec::new();
        let mut matvec = |g: &GMat, input: &wgpu::Buffer, out: &wgpu::Buffer, add: bool, calls: &mut Vec<Call>| {
            for p in &g.parts {
                let u = uni(&[p.rows as u32, g.cols as u32, add as u32, p.row0 as u32]);
                calls.push(Call { pipe: P_MATVEC, bind: bind(P_MATVEC, &[&u, &p.w, &p.s, input, out]), groups: groups(p.rows) });
                keep.push(Arc::new(u));
            }
        };
        let eps = c.eps.to_bits();
        let rms_u = uni(&[dim as u32, eps, 0, 0]);
        let mut layer_calls = Vec::new();
        let mut kv_bytes = 0u64;
        let mut mats_keep = Vec::new();
        for l in m.layers() {
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
            kv_bytes += (max_ctx * kvd * 4) as u64;
            let qn = l.q_norm.as_ref().map(|v| f32s(v, "q norm"));
            let kn = l.k_norm.as_ref().map(|v| f32s(v, "k norm"));
            layer_calls.push(Call { pipe: P_RMS, bind: bind(P_RMS, &[&rms_u, &x, &attn_norm, &h]), groups: (1, 1) });
            matvec(&wqkv, &h, &qkv, false, &mut layer_calls);
            let post_u = uni(&[nh as u32, nkv as u32, hd as u32, c.rope_dim as u32, kvd as u32, qn.is_some() as u32, kn.is_some() as u32, eps, qd as u32, 0, 0, 0]);
            layer_calls.push(Call { pipe: P_POST, bind: bind(P_POST, &[&post_u, &step, &qkv, qn.as_ref().unwrap_or(&dummy), kn.as_ref().unwrap_or(&dummy), &kc, &vc]), groups: ((nh + nkv) as u32, 1) });
            let attn_u = uni(&[(nh / nkv) as u32, hd as u32, kvd as u32, max_ctx as u32, (1.0 / (hd as f32).sqrt()).to_bits(), 0, 0, 0]);
            layer_calls.push(Call { pipe: P_ATTN, bind: bind(P_ATTN, &[&attn_u, &step, &qkv, &kc, &vc, &att, &ao]), groups: (nh as u32, 1) });
            matvec(&wo, &ao, &x, true, &mut layer_calls);
            layer_calls.push(Call { pipe: P_RMS, bind: bind(P_RMS, &[&rms_u, &x, &ffn_norm, &h]), groups: (1, 1) });
            matvec(&wgu, &h, &gu, false, &mut layer_calls);
            let silu_u = uni(&[ffn as u32, 0, 0, 0]);
            layer_calls.push(Call { pipe: P_SILU, bind: bind(P_SILU, &[&silu_u, &gu]), groups: (ffn.div_ceil(256) as u32, 1) });
            matvec(&wd, &gu, &x, true, &mut layer_calls);
            mats_keep.push((wqkv, wo, wgu, wd, kc, vc, attn_norm, ffn_norm, qn, kn, post_u, attn_u, silu_u));
        }
        trace("layers");
        let out_norm = f32s(m.out_norm(), "output norm");
        let wout = upload(&[m.output_mat()], "output")?;
        bytes += wout.size();
        let mut head_calls = vec![Call { pipe: P_RMS, bind: bind(P_RMS, &[&rms_u, &x, &out_norm, &h]), groups: (1, 1) }];
        matvec(&wout, &h, &logits, false, &mut head_calls);
        bytes += kv_bytes;
        // Buffers referenced only by bind groups stay alive through them; matrices are kept here so
        // their memory is accounted to this state.
        drop(mats_keep);
        drop(wout);
        Ok(Self { device, queue, info, max_ctx, x, step, logits, readback, x_all, step_all, pipes, layer_calls, head_calls, step_bytes, dim, vocab: c.vocab, bytes, _keep: keep })
    }

    fn step_record(&self, m: &Model, pos: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.step_bytes);
        v.extend_from_slice(&(pos as u32).to_le_bytes());
        v.extend_from_slice(&(pos as u32 + 1).to_le_bytes());
        v.extend_from_slice(&[0u8; 8]);
        for f in m.inv_freq() {
            let (s, c) = (pos as f32 * f).sin_cos();
            v.extend_from_slice(&c.to_le_bytes());
            v.extend_from_slice(&s.to_le_bytes());
        }
        v
    }

    /// Runs `toks` at positions pos0.. in one submission per chunk; writes the last token's logits.
    pub fn run(&mut self, m: &Model, toks: &[u32], pos0: usize, logits: &mut Vec<f32>) -> Result<(), Error> {
        if pos0 + toks.len() > self.max_ctx {
            return Err(Error::Request("the context is full".into()));
        }
        let mut emb = vec![0f32; self.dim];
        for (ci, chunk) in toks.chunks(CHUNK).enumerate() {
            let mut xs = Vec::with_capacity(chunk.len() * self.dim * 4);
            let mut steps = Vec::with_capacity(chunk.len() * self.step_bytes);
            for (i, &t) in chunk.iter().enumerate() {
                m.embed_into(t, &mut emb);
                xs.extend(emb.iter().flat_map(|v| v.to_le_bytes()));
                steps.extend(self.step_record(m, pos0 + ci * CHUNK + i));
            }
            self.queue.write_buffer(&self.x_all, 0, &xs);
            self.queue.write_buffer(&self.step_all, 0, &steps);
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let last_chunk = (ci + 1) * CHUNK >= toks.len();
            for i in 0..chunk.len() {
                let (db, sb) = (self.dim * 4, self.step_bytes);
                enc.copy_buffer_to_buffer(&self.x_all, (i * db) as u64, &self.x, 0, db as u64);
                enc.copy_buffer_to_buffer(&self.step_all, (i * sb) as u64, &self.step, 0, sb as u64);
                let head = last_chunk && i + 1 == chunk.len();
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                for call in self.layer_calls.iter().chain(if head { self.head_calls.iter() } else { [].iter() }) {
                    pass.set_pipeline(&self.pipes[call.pipe]);
                    pass.set_bind_group(0, &call.bind, &[]);
                    pass.dispatch_workgroups(call.groups.0, call.groups.1, 1);
                }
            }
            if last_chunk {
                enc.copy_buffer_to_buffer(&self.logits, 0, &self.readback, 0, (self.vocab * 4) as u64);
            }
            self.queue.submit([enc.finish()]);
        }
        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(perr)?;
        {
            let data = slice.get_mapped_range().map_err(perr)?;
            logits.clear();
            logits.extend(data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
        }
        self.readback.unmap();
        Ok(())
    }
}
