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
fn spec_default() -> i64 {
    8
}

fn zero() -> i64 {
    0
}
fn threads_default() -> i64 {
    // Hybrid processors: every logical CPU, each thread pinned to its core class (pool.rs
    // apply_pinning), with work in small pieces so the efficiency cores never hold up the end.
    // i9-12900H, 512-token prompt, 12 against 20 threads: Qwen3-0.6B Q8_0 613 -> 671, Qwen3-8B
    // Q4_K_M 45 -> 58 tok/s. Otherwise every logical CPU but four, leaving room for the OS, the
    // GPU driver and the UI.
    let c = crate::hw::cores();
    if c.hybrid() {
        return c.logical as i64;
    }
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
pub const CPU_DECODE_THREADS: &str = "cpu_decode_threads";
pub const CPU_CHUNK_DIV: &str = "cpu_chunk_div";
pub const PREFAULT: &str = "prefault";
pub const EXPERT_VRAM_MB: &str = "expert_vram_mb";
pub const GPU_ENABLED: &str = "gpu_enabled";
pub const GPU_LAYERS: &str = "gpu_layers";
pub const SPEC_LOOKUP: &str = "spec_lookup";
pub const PORTABLE_GPU: &str = "portable_gpu";
pub const CPU_REPACK: &str = "cpu_repack";
pub const CPU_PIN: &str = "cpu_pin";
pub const PORTABLE_INTEGRATED: &str = "portable_integrated";
pub const STREAM_LAYERS: &str = "stream_layers";
pub const DRIVE_TIER: &str = "drive_tier";
pub const EXPERT_RAM_CACHE: &str = "expert_ram_cache";
pub const SPLIT_STREAM: &str = "split_stream";
pub const SPLIT_HEAD: &str = "split_head";
pub const LLAMA_SPLITS: &str = "llama_splits";

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
                beginner: "Reads prompts up to 12x faster on recent NVIDIA graphics cards.",
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
                applies: Applies::Live,
                beginner: "How many processor threads read prompts. The default uses the fast cores and leaves room for other apps.",
                advanced: "Threads for prompt reading and other batched work: the performance cores' threads on hybrid CPUs (pinned there), else logical CPUs minus four. Writing has its own count.",
                env: "CEREBRO_THREADS",
                default: threads_default,
            },
            Switch {
                id: CPU_DECODE_THREADS,
                name: "Threads while writing",
                group: "CPU",
                kind: Kind::Number { min: 0, max: logical(), zero_auto: true, unit: "threads" },
                channel: Channel::Stable,
                applies: Applies::Live,
                beginner: "How many processor threads write the reply. Writing waits on memory, so more cores help even when they are slower ones.",
                advanced: "Threads for token-by-token decode. Automatic on hybrid processors: every logical CPU while the conversation is under 256 tokens (writing streams weights), then the Processor threads on the performance cores (attention dominates).",
                // i9-12900H, 0.6B Q4_K_M: 91 vs 81 tok/s at 32 tokens, 44.6 vs 36.7 at 2048, in each
                // case for the automatic choice.
                env: "CEREBRO_DECODE_THREADS",
                default: zero,
            },
            Switch {
                id: CPU_CHUNK_DIV,
                name: "Work splitting",
                group: "CPU",
                kind: Kind::Number { min: 1, max: 16, zero_auto: false, unit: "pieces per thread" },
                channel: Channel::Experimental,
                applies: Applies::Live,
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
                beginner: "Loads the model into memory before the first message so the first answer starts at full speed.",
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
                advanced: "Caps the adaptive VRAM expert cache (0 = all free VRAM except a sixteenth of the card). Lower it to share the GPU with games or other apps.",
                env: "CEREBRO_EXPERT_VRAM_MB",
                default: zero,
            },
            Switch {
                id: SPEC_LOOKUP,
                name: "Draft from the conversation",
                group: "Speculation",
                kind: Kind::Number { min: 0, max: 15, zero_auto: false, unit: "tokens" },
                channel: Channel::Beta,
                applies: Applies::Live,
                beginner: "When the reply repeats something already in the chat (code, names, quotes), the model checks several words at once instead of one by one. Its word choices keep the same likelihoods; exact wording can differ slightly.",
                advanced: "Prompt-lookup speculation: the last 1-3 tokens are matched earlier in the context and the tokens that followed are drafted (up to this many), then verified in one batched pass with speculative sampling (accepted with the sampler's probability, resampled otherwise), so the output distribution is unchanged. Batched checks can flip near-ties, so text is not guaranteed identical. Drafts need a match of 2+ tokens, are sized by recent acceptance and pause after a full rejection. Whole-GPU attention models only. 0 turns it off.",
                env: "CEREBRO_SPEC_LOOKUP",
                default: spec_default,
            },
            Switch {
                id: GPU_ENABLED,
                name: "Use the graphics card",
                group: "Placement",
                kind: Kind::Toggle,
                channel: Channel::Stable,
                applies: Applies::Reload,
                beginner: "Runs the model on the NVIDIA graphics card when it has room. Off runs everything on the processor.",
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
                beginner: "How much of a model too big for the graphics card runs on it; the rest runs on the processor. Automatic puts as much there as fits.",
                advanced: "Dense models larger than free VRAM split by layer: the first N on the GPU in their own K-quant format, the rest and the output head on the CPU, one activation copy each way per token. 0 = as many as fit (a sixteenth of the card stays free). Larger values are clamped to what fits.",
                env: "CEREBRO_GPU_LAYERS",
                default: zero,
            },
            Switch {
                id: STREAM_LAYERS,
                name: "Layers read from the drive",
                group: "Placement",
                kind: Kind::Number { min: 0, max: 1024, zero_auto: true, unit: "layers" },
                channel: Channel::Beta,
                applies: Applies::Reload,
                beginner: "How many of a model's last layers are read from the drive as it writes instead of kept in memory. Automatic does it only for a model that does not fit in memory, and only as many layers as needed. A higher number leaves memory free for other apps.",
                advanced: "The last N processor layers (and the output head when N covers every layer) are read with unbuffered I/O on every pass, 16 parallel 4 MB requests into a ring of buffers (about 512 MB), so the next layers are read while this one computes. 0 = automatic: placement.rs streams the fewest trailing layers that let the rest plus the ring fit in available RAM minus 1.5 GB and the processor's KV cache. Dense and recurrent models; mixtures of experts read only their active experts already.",
                env: "CEREBRO_STREAM_LAYERS",
                default: zero,
            },
            Switch {
                id: LLAMA_SPLITS,
                name: "Use llama.cpp where it is faster",
                group: "Placement",
                kind: Kind::Toggle,
                channel: Channel::Beta,
                applies: Applies::Reload,
                beginner: "AMD or Intel graphics, including graphics built into the processor, run llama.cpp's Vulkan build, which is measured faster there than Cerebro's portable backend today. NVIDIA cards always run Cerebro's own engine, whole or split. The first time it downloads the official release (about 35 MB).",
                advanced: "Split plans on NVIDIA run the CUDA build with the planner's GPU layer count; with no NVIDIA card, a graphics card the portable backend would use runs the Vulkan build (all layers it holds). The app starts llama-server on 127.0.0.1 and streams over its OpenAI-compatible API; downloads are checked against the SHA-256 GitHub publishes. Off: Cerebro's engine runs everything.",
                // Measured on an i9-12900H: split prompt reading 0.05-0.12x and Iris Xe prompt reading
                // 0.05-0.14x of llama.cpp.
                env: "CEREBRO_NO_LLAMA_SPLITS",
                default: on,
            },
            Switch {
                id: SPLIT_HEAD,
                name: "Vocabulary layer on the graphics card for split models",
                group: "Placement",
                kind: Kind::Toggle,
                channel: Channel::Beta,
                applies: Applies::Reload,
                beginner: "When a model is split between the graphics card and the processor, its last layer (the one that scores every word) goes on the card. It is read once per word written, so the processor has that much less to read and writes faster.",
                advanced: "Split plans reserve VRAM for the output matrix before fitting layers; each decode step uploads the residual after the CPU's layers, runs the final norm and the head on the GPU and copies the logits back. Not for hyper-connection models. Off: the head stays on the processor and the card holds one or two more layers.",
                env: "CEREBRO_NO_SPLIT_HEAD",
                default: on,
            },
            Switch {
                id: SPLIT_STREAM,
                name: "Read prompts on the graphics card for split models",
                group: "Placement",
                kind: Kind::Toggle,
                channel: Channel::Beta,
                applies: Applies::Live,
                beginner: "When a model is split between the graphics card and the processor, Cerebro sends the processor's share of the model to the card, one layer at a time, while it reads the prompt. Long prompts are read many times faster.",
                advanced: "Split prefill: after the GPU's resident layers, each CPU layer's weights are copied to a GPU slot (native K-quants, or Q8_0 split on the GPU), its KV rows for earlier positions uploaded, the batched layer kernel run, and the new KV rows copied back to the processor's cache. Plain dense attention layers only; others fall back to the processor. Off: the processor runs its layers.",
                env: "CEREBRO_NO_SPLIT_STREAM",
                default: on,
            },
            Switch {
                id: EXPERT_RAM_CACHE,
                name: "Keep the most-used experts in memory",
                group: "Placement",
                kind: Kind::Toggle,
                channel: Channel::Beta,
                applies: Applies::Reload,
                beginner: "For a mixture-of-experts model bigger than memory: Cerebro keeps the experts the conversation uses in memory and reads the others from the drive at full speed, instead of leaving it to Windows.",
                advanced: "When a MoE model's experts do not fit in available RAM (minus 1.5 GB, the KV cache and the weights read every token), experts.rs keeps an arena of whole-expert slots, filled on demand with parallel unbuffered reads and kept by decayed use counts. Off: the experts stay memory-mapped and the OS pages them in 4 KB at a time.",
                env: "CEREBRO_NO_EXPERT_CACHE",
                default: on,
            },
            Switch {
                id: DRIVE_TIER,
                name: "Read from the drive when memory runs short",
                group: "Placement",
                kind: Kind::Toggle,
                channel: Channel::Beta,
                applies: Applies::Reload,
                beginner: "When a model does not fit in memory, reads the part that does not fit straight from the drive at its full speed. Off leaves it to Windows, which reads it in small pieces and slows the whole computer down.",
                advanced: "Off: no layer is streamed unless \"Layers read from the drive\" names a number; the overflow is paged in through the memory map 4 KB at a time and competes with every other program for the file cache.",
                env: "CEREBRO_NO_DRIVE_TIER",
                default: on,
            },
            Switch {
                id: CPU_PIN,
                name: "Performance cores only",
                group: "CPU",
                kind: Kind::Toggle,
                channel: Channel::Stable,
                applies: Applies::Live,
                beginner: "Keeps the model on the processor's fast cores. Without it Windows moves the work to the slow cores after a pause.",
                advanced: "On hybrid CPUs, while the thread count fits the performance cores, every pool thread is pinned to them (SetThreadAffinityMask). More threads than performance cores unpins.",
                // Measured on an i9-12900H: reading a 512-token prompt after a pause 220 -> 339 tok/s,
                // writing 41 -> 52 tok/s.
                env: "CEREBRO_NO_PIN",
                default: on,
            },
            Switch {
                id: CPU_REPACK,
                name: "Interleaved weights on the processor",
                group: "CPU",
                kind: Kind::Toggle,
                channel: Channel::Beta,
                applies: Applies::Reload,
                beginner: "Rearranges Q8_0 and Q4_K weights once when a model loads so the processor reads prompts and writes faster.",
                advanced: "Q8_0 and Q4_K rows interleaved eight at a time (Q8_0 offset to unsigned; scales as f32; Q4_K 6-bit scales decoded): one VNNI instruction gives eight rows' partial sums. The original file pages are released from the working set, so RAM use stays about the same.",
                env: "CEREBRO_NO_REPACK",
                default: on,
            },
            Switch {
                id: PORTABLE_GPU,
                name: "Other graphics cards",
                group: "Placement",
                kind: Kind::Toggle,
                channel: Channel::Beta,
                applies: Applies::Reload,
                beginner: "Runs models on AMD, Intel and Apple graphics cards too, not only NVIDIA ones.",
                advanced: "The portable backend: WGSL kernels through Vulkan, DX12 or Metal, used when no NVIDIA GPU takes the model. Dense attention models only, whole on the GPU, weights repacked to int8. Kernels are timed per GPU at load.",
                env: "CEREBRO_NO_PORTABLE",
                default: on,
            },
            Switch {
                id: PORTABLE_INTEGRATED,
                name: "Integrated graphics",
                group: "Placement",
                kind: Kind::Toggle,
                channel: Channel::Beta,
                applies: Applies::Reload,
                beginner: "Uses the graphics built into the processor. It shares the computer's memory, so it is rarely faster than the processor, but it leaves the processor free.",
                advanced: "Lets the portable backend pick an integrated GPU.",
                // Measured on an Iris Xe (DX12, Qwen3-0.6B Q8_0): 43-47 tok/s writing against ~47
                // sustained on the i9-12900H itself, 58 vs 28 tok/s reading a 3k-token prompt.
                env: "CEREBRO_PORTABLE_INTEGRATED",
                default: off,
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
    if s.id == CPU_THREADS {
        crate::pool::apply_threads(v as usize);
    }
    if s.id == CPU_PIN {
        crate::pool::apply_pin();
    }
    Ok(v)
}

/// Restores every switch to its start-up value.
pub fn reset() {
    if let Ok(mut o) = overrides().write() {
        o.clear();
    }
    crate::pool::apply_threads(get(CPU_THREADS) as usize);
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
