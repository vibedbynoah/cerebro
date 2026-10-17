//! Engine switches: every optimisation the user can turn on or off, in one registry. Each entry has a
//! plain-language explanation for beginners and a technical one for advanced users, its release
//! channel, and when a change takes effect. The app renders this list (Turbo screen) and calls `set`;
//! environment variables remain as start-up defaults for benchmarks.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    Stable,
    Beta,
    Experimental,
}

/// When a new value is used.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Applies {
    /// On the next request.
    Live,
    /// When a model is next loaded.
    Reload,
    /// After the app restarts.
    Restart,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Kind {
    Toggle,
    /// Integer in [min, max]; `zero_auto` means 0 lets the engine decide.
    Number { min: i64, max: i64, zero_auto: bool, unit: &'static str },
}

#[derive(Clone, Debug, Serialize)]
pub struct Switch {
    pub id: &'static str,
    pub name: &'static str,
    pub group: &'static str,
    pub kind: Kind,
    pub channel: Channel,
    pub applies: Applies,
    pub beginner: &'static str,
    pub advanced: &'static str,
    /// Environment variable that sets the start-up value (`NO_` variables turn a toggle off).
    pub env: &'static str,
    #[serde(skip)]
    default: fn() -> i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct State {
    #[serde(flatten)]
    pub info: Switch,
    pub value: i64,
    pub default: i64,
}

fn on() -> i64 {
    1
}
fn off() -> i64 {
    0
}
fn zero() -> i64 {
    0
}
fn threads_default() -> i64 {
    std::thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(4).max(1) as i64
}
fn four() -> i64 {
    4
}
fn logical() -> i64 {
    std::thread::available_parallelism().map_or(64, |n| n.get()) as i64
}

pub const GPU_MMA_MATMUL: &str = "gpu_mma_matmul";
pub const GPU_MMA_ATTENTION: &str = "gpu_mma_attention";
pub const GPU_DECODE_GQA: &str = "gpu_decode_gqa";
pub const CPU_BATCH_PREFILL: &str = "cpu_batch_prefill";
pub const CPU_ROWS4: &str = "cpu_rows4";
pub const CPU_THREADS: &str = "cpu_threads";
pub const CPU_CHUNK_DIV: &str = "cpu_chunk_div";
pub const PREFAULT: &str = "prefault";
pub const EXPERT_VRAM_MB: &str = "expert_vram_mb";
pub const GPU_ENABLED: &str = "gpu_enabled";
pub const GPU_LAYERS: &str = "gpu_layers";

fn registry() -> &'static [Switch] {
    static R: OnceLock<Vec<Switch>> = OnceLock::new();
    R.get_or_init(|| {
        vec![
            Switch {
                id: GPU_MMA_MATMUL,
                name: "Tensor-core prompt reading",
                group: "GPU",
                kind: Kind::Toggle,
                channel: Channel::Stable,
                applies: Applies::Reload,
                beginner: "Reads your prompt up to 12x faster on recent NVIDIA graphics cards.",
                advanced: "Prefill matmuls run on int8 tensor cores (mma m16n8k32, cp.async double buffer) instead of dp4a. Needs compute capability 8.0+; older GPUs ignore it.",
                env: "CEREBRO_NO_MMA",
                default: on,
            },
            Switch {
                id: GPU_MMA_ATTENTION,
                name: "Tensor-core attention",
                group: "GPU",
                kind: Kind::Toggle,
                channel: Channel::Stable,
                applies: Applies::Reload,
                beginner: "Keeps long prompts fast: the model compares every word with the ones before it using the graphics card's fastest units.",
                advanced: "Flash attention for prefill on f16 tensor cores (m16n8k16, ldmatrix.trans, online softmax in exp2). Head size 128, compute capability 8.0+.",
                env: "CEREBRO_NO_MMA_ATTN",
                default: on,
            },
            Switch {
                id: GPU_DECODE_GQA,
                name: "Shared-head decode attention",
                group: "GPU",
                kind: Kind::Toggle,
                channel: Channel::Beta,
                applies: Applies::Reload,
                beginner: "Keeps writing fast as a conversation gets long, by reading the model's memory of the conversation once for several heads at a time.",
                advanced: "Decode attention with one block per KV head serving its whole query group (K/V read once per group), one cached position per lane, splits sized to the GPU and derived from the position at run time. Group size up to 8, head size up to 256.",
                env: "CEREBRO_NO_DECODE_GQA",
                default: on,
            },
            Switch {
                id: CPU_ROWS4,
                name: "Four-row writing on the processor",
                group: "CPU",
                kind: Kind::Toggle,
                channel: Channel::Beta,
                applies: Applies::Live,
                beginner: "Works on four rows of the model at once on the processor. Off until it is measured faster on this machine.",
                advanced: "Q8_0 decode matvec computes four rows per call: one activation load per block for four rows, four independent FMA chains, block scales via F16C. Bit-identical to the one-row kernel. Needs AVX-VNNI and F16C.",
                env: "CEREBRO_ROWS4",
                default: off,
            },
            Switch {
                id: CPU_BATCH_PREFILL,
                name: "Batched prompt reading on the processor",
                group: "CPU",
                kind: Kind::Toggle,
                channel: Channel::Stable,
                applies: Applies::Live,
                beginner: "Reads prompts several times faster when the model runs on the processor.",
                advanced: "Prefill processes chunks of tokens per layer: each weight row is read once per chunk and dotted against every token's activations while it is in cache. Dense models.",
                env: "CEREBRO_NO_CPU_BATCH",
                default: on,
            },
            Switch {
                id: CPU_THREADS,
                name: "Processor threads",
                group: "CPU",
                kind: Kind::Number { min: 1, max: logical(), zero_auto: false, unit: "threads" },
                channel: Channel::Stable,
                applies: Applies::Restart,
                beginner: "How many processor threads the model may use. The default leaves room for your other apps.",
                advanced: "Size of the engine's worker pool (logical CPUs minus four by default). Measured on an i9-12900H: 6 -> 47.5, 12 -> 61.3, 16 -> 64.5 tok/s decode (0.6B Q8_0).",
                env: "CEREBRO_THREADS",
                default: threads_default,
            },
            Switch {
                id: CPU_CHUNK_DIV,
                name: "Work splitting",
                group: "CPU",
                kind: Kind::Number { min: 1, max: 16, zero_auto: false, unit: "pieces per thread" },
                channel: Channel::Experimental,
                applies: Applies::Restart,
                beginner: "Splits each step into smaller pieces so slower cores never hold up the fast ones.",
                advanced: "Work items per thread per parallel region. Higher values balance hybrid P/E-core CPUs better at the cost of more dispatch.",
                env: "CEREBRO_CHUNK_DIV",
                default: four,
            },
            Switch {
                id: PREFAULT,
                name: "Warm up memory at load",
                group: "Memory",
                kind: Kind::Toggle,
                channel: Channel::Stable,
                applies: Applies::Reload,
                beginner: "Loads the model into memory before your first message so the first answer starts at full speed.",
                advanced: "Touches weight pages (and CPU-side expert pages, most used first) up to 85-90% of available RAM at load instead of page-faulting during decode (measured 30.3 -> 36.2 tok/s on a 30B MoE).",
                env: "CEREBRO_PREFAULT",
                default: on,
            },
            Switch {
                id: EXPERT_VRAM_MB,
                name: "Graphics memory for experts",
                group: "Mixture of experts",
                kind: Kind::Number { min: 0, max: 262_144, zero_auto: true, unit: "MB" },
                channel: Channel::Beta,
                applies: Applies::Reload,
                beginner: "How much graphics memory holds the model's most-used parts. Automatic uses everything that is free.",
                advanced: "Caps the adaptive VRAM expert cache (0 = all free VRAM minus a 600 MB margin). Lower it to share the GPU with games or other apps.",
                env: "CEREBRO_EXPERT_VRAM_MB",
                default: zero,
            },
            Switch {
                id: GPU_ENABLED,
                name: "Use the graphics card",
                group: "Placement",
                kind: Kind::Toggle,
                channel: Channel::Stable,
                applies: Applies::Reload,
                beginner: "Runs the model on your NVIDIA graphics card when it has room. Off runs everything on the processor.",
                advanced: "Off skips CUDA entirely (CPU session). On, the placement planner puts as many layers on the GPU as fit (placement.rs).",
                env: "CEREBRO_NO_GPU",
                default: on,
            },
            Switch {
                id: GPU_LAYERS,
                name: "Layers on the graphics card",
                group: "Placement",
                kind: Kind::Number { min: 0, max: 1024, zero_auto: true, unit: "layers" },
                channel: Channel::Stable,
                applies: Applies::Reload,
                beginner: "How much of a model too big for your graphics card runs on it; the rest runs on the processor. Automatic puts as much there as fits.",
                advanced: "Dense models larger than free VRAM split by layer: the first N on the GPU in their own K-quant format, the rest and the output head on the CPU, one activation copy each way per token. 0 = as many as fit (600 MB margin). Larger values are clamped to what fits.",
                env: "CEREBRO_GPU_LAYERS",
                default: zero,
            },
        ]
    })
}

fn overrides() -> &'static RwLock<HashMap<&'static str, i64>> {
    static O: OnceLock<RwLock<HashMap<&'static str, i64>>> = OnceLock::new();
    O.get_or_init(|| RwLock::new(HashMap::new()))
}

fn from_env(s: &Switch) -> Option<i64> {
    let v = std::env::var(s.env).ok()?;
    if s.env.starts_with("CEREBRO_NO_") {
        return Some(0);
    }
    v.parse().ok()
}

fn info(id: &str) -> Option<&'static Switch> {
    registry().iter().find(|s| s.id == id)
}

/// Current value: the user's setting, else the environment, else the default. Unknown ids read 0.
pub fn get(id: &str) -> i64 {
    let Some(s) = info(id) else { return 0 };
    if let Some(v) = overrides().read().ok().and_then(|o| o.get(s.id).copied()) {
        return v;
    }
    from_env(s).unwrap_or_else(s.default)
}

pub fn enabled(id: &str) -> bool {
    get(id) != 0
}

/// Sets a switch, clamped to its range. Returns the stored value.
pub fn set(id: &str, value: i64) -> Result<i64, String> {
    let s = info(id).ok_or_else(|| format!("unknown switch {id}"))?;
    let v = match s.kind {
        Kind::Toggle => (value != 0) as i64,
        Kind::Number { min, max, zero_auto, .. } => {
            if zero_auto && value == 0 {
                0
            } else {
                value.clamp(min, max)
            }
        }
    };
    overrides().write().map_err(|_| "switch table poisoned".to_string())?.insert(s.id, v);
    Ok(v)
}

/// Restores every switch to its start-up value.
pub fn reset() {
    if let Ok(mut o) = overrides().write() {
        o.clear();
    }
}

/// Every switch with its current and default value, for the app.
pub fn catalogue() -> Vec<State> {
    registry()
        .iter()
        .map(|s| State { info: s.clone(), value: get(s.id), default: from_env(s).unwrap_or_else(s.default) })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_clamps_and_toggles_normalise() {
        assert_eq!(set(CPU_CHUNK_DIV, 99).unwrap(), 16);
        assert_eq!(set(GPU_MMA_MATMUL, 5).unwrap(), 1);
        assert_eq!(set(EXPERT_VRAM_MB, 0).unwrap(), 0);
        assert!(set("nope", 1).is_err());
        assert!(catalogue().iter().all(|s| !s.info.beginner.is_empty() && !s.info.advanced.is_empty()));
        reset();
    }
}
