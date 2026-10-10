//! What a candidate is, and which candidates a model on a machine gets.
//!
//! A candidate is three independent choices ([`CandidateConfig`]): where the weights go
//! ([`Placement`]), the KV cache type ([`KvCache`]) and the CPU threads ([`ThreadPlan`]). The
//! tuner screens them in three stages, each starting from the leader so far, so the set stays
//! small (6 to 10 runs) instead of the full cross product:
//!
//! 1. **Placement** ([`Planner::placements`]): llama.cpp's own automatic settings (the baseline,
//!    always measured first), then full GPU offload when the fit estimate says it fits, or else
//!    `-ncmoe` steps for a mixture of experts and dense FFN offload steps for a dense model,
//!    centred on the lightest offload the estimate says fits.
//! 2. **KV cache** ([`Planner::kv_variants`]): the leader with a q8_0 KV cache (flash attention
//!    on), and, for an offload step, the next lighter step with q8_0, which the smaller cache may
//!    make room for.
//! 3. **Threads** ([`Planner::thread_variants`]): the leader pinned to one thread per performance
//!    core, to half of those, and to every performance-core thread, from `cerebro-probe`'s masks.
//!
//! The fit estimate only chooses which candidates to measure; it never chooses the winner. It is
//! deliberately rough (weights from the GGUF tensor table, the KV cache from the attention shape,
//! a fixed compute reserve, llama-server's 1024 MiB margin per device), which is why the offload
//! steps straddle it on both sides. Its KV term is biased high for models whose layers do not all
//! hold a full KV cache (sliding-window layers, hybrid linear-attention layers), so full offload is
//! also measured when the estimate misses by at most a quarter of the budget.
//!
//! A configuration is emitted for both tools from one mapping: [`CandidateConfig::bench_flags`]
//! for the screening runs and [`CandidateConfig::server_flags`] for serving the winner, where
//! every setting the tuner varies is stated explicitly rather than inherited from llama-server.
//!
//! Dense FFN offload is expressed as `-ot` rules (`blk\.(0|1|..)\.ffn_(up|down|gate)\.=CPU`):
//! llama-bench has no `-ncffn`, and the rules place exactly the tensors `-ncffn` would, in both
//! dialects.

use std::fmt;
use std::num::NonZeroU32;

use cerebro_bench::run::SERVER_DEFAULT_FIT_TARGET_MIB;
use cerebro_flags::value::{BufferType, CacheType, CpuMask, FlashAttn, OtRule, OtRuleSet};
use cerebro_flags::{Flag, ValueError};
use cerebro_probe::CpuTopology;

use crate::gguf::ModelFacts;

/// What the model will mostly be used for, which sets the context depth decode is judged at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Workload {
    /// Conversations: decode judged at 4,096 tokens of context.
    Chat,
    /// Coding with files in context: 16,384 tokens.
    Coding,
    /// Long documents: 32,768 tokens.
    LongDocuments,
}

impl Workload {
    /// The context depth the winner is chosen at (`tg` at this depth), before clamping to the
    /// model's own context.
    pub fn target_depth(self) -> u32 {
        match self {
            Workload::Chat => 4096,
            Workload::Coding => 16_384,
            Workload::LongDocuments => 32_768,
        }
    }

    /// Short lowercase name, for logs and progress.
    pub fn name(self) -> &'static str {
        match self {
            Workload::Chat => "chat",
            Workload::Coding => "coding",
            Workload::LongDocuments => "long-documents",
        }
    }
}

/// Where the weights go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Placement {
    /// llama.cpp's own automatic placement: llama-bench fits the model to free device memory the
    /// way llama-server does by default (`-fitt 1024`). The baseline.
    Auto,
    /// Every block on the GPU (`-ngl 999`).
    FullOffload,
    /// Every block on the GPU except the routed experts of the first `n` blocks
    /// (`-ngl 999 -ncmoe n`).
    ExpertsOnCpu(u32),
    /// Every block on the GPU except the dense FFN weights of the first `n` blocks
    /// (`-ngl 999 -ot ...=CPU`).
    DenseFfnOnCpu(u32),
}

/// The KV cache type, with flash attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvCache {
    /// llama.cpp's defaults (f16 cache, flash attention auto); emits nothing.
    Default,
    /// f16 K and V with flash attention on.
    F16,
    /// q8_0 K and V with flash attention on (quantized V needs it): half the cache memory.
    Q8,
}

/// CPU threads for generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThreadPlan {
    /// llama.cpp's default thread count; emits nothing.
    Default,
    /// `threads` threads pinned (`--cpu-strict 1`) to `mask`.
    Pinned {
        /// `-t`.
        threads: u32,
        /// `-C`.
        mask: CpuMask,
    },
}

/// One configuration to measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CandidateConfig {
    /// Where the weights go.
    pub placement: Placement,
    /// KV cache type.
    pub kv: KvCache,
    /// CPU threads.
    pub threads: ThreadPlan,
}

impl CandidateConfig {
    /// The baseline every other candidate must beat: llama.cpp's automatic settings, nothing else.
    pub const BASELINE: Self = Self {
        placement: Placement::Auto,
        kv: KvCache::Default,
        threads: ThreadPlan::Default,
    };

    /// Whether this is [`CandidateConfig::BASELINE`].
    pub fn is_baseline(&self) -> bool {
        *self == Self::BASELINE
    }

    /// The pinned thread count, when this candidate sets one.
    pub fn threads(&self) -> Option<u32> {
        match self.threads {
            ThreadPlan::Default => None,
            ThreadPlan::Pinned { threads, .. } => Some(threads),
        }
    }

    /// The llama-bench flags (Bench dialect, without `-m`): placement, then KV, then threads.
    /// What llama-bench's defaults already match llama-server's (f16 KV with flash attention
    /// auto, the default thread count) is left unset, so the baseline measures exactly the
    /// automatic settings.
    ///
    /// # Errors
    /// [`ValueError`] only if an `-ot` pattern were built with a reserved character, which the
    /// fixed pattern here never contains.
    pub fn bench_flags(&self) -> Result<Vec<Flag>, ValueError> {
        let mut flags = Vec::new();
        if self.placement == Placement::Auto {
            flags.push(Flag::FitTarget(fit_target()));
        } else {
            self.push_offload_plan(&mut flags)?;
        }
        if let Some(t) = self.explicit_cache_type() {
            flags.extend([Flag::FlashAttn(FlashAttn::On), Flag::CacheTypeK(t), Flag::CacheTypeV(t)]);
        }
        self.push_pinned_threads(&mut flags);
        Ok(flags)
    }

    /// The llama-server flags for this configuration (Server dialect, without `-m`, the context
    /// size, the network or the key, which the serving launch adds), so the winner, or the
    /// baseline when nothing beat it, is served exactly as it was measured.
    ///
    /// Nothing the tuner varies is left to a llama-server default: [`Placement::Auto`] states
    /// the fitter and its parameters (`--fit on --fit-target 1024 --fit-ctx 4096`), an explicit
    /// placement turns the fitter off (llama-bench measured it with the fitter off), the default
    /// KV cache is stated as `-fa auto -ctk f16 -ctv f16`, and the default thread count is stated
    /// as `-t measured_threads` when llama-bench reported one (it uses the same default as
    /// llama-server, so this is the count that was measured).
    ///
    /// # Errors
    /// As [`CandidateConfig::bench_flags`].
    pub fn server_flags(&self, measured_threads: Option<u32>) -> Result<Vec<Flag>, ValueError> {
        let mut flags = Vec::new();
        if self.placement == Placement::Auto {
            // A non-zero literal; the fallback is unreachable and still a valid minimum.
            let ctx = NonZeroU32::new(SERVER_DEFAULT_FIT_CTX).unwrap_or(NonZeroU32::MIN);
            flags.extend([Flag::Fit(true), Flag::FitTarget(fit_target()), Flag::FitCtx(ctx)]);
        } else {
            flags.push(Flag::Fit(false));
            self.push_offload_plan(&mut flags)?;
        }
        match self.explicit_cache_type() {
            Some(t) => {
                flags.extend([Flag::FlashAttn(FlashAttn::On), Flag::CacheTypeK(t), Flag::CacheTypeV(t)]);
            }
            None => flags.extend([
                Flag::FlashAttn(FlashAttn::Auto),
                Flag::CacheTypeK(CacheType::F16),
                Flag::CacheTypeV(CacheType::F16),
            ]),
        }
        match (self.threads, measured_threads) {
            (ThreadPlan::Pinned { .. }, _) => self.push_pinned_threads(&mut flags),
            (ThreadPlan::Default, Some(n)) => flags.push(Flag::Threads(n)),
            (ThreadPlan::Default, None) => {}
        }
        Ok(flags)
    }

    /// `-ngl` and the tensor overrides of an explicit placement (nothing for [`Placement::Auto`]).
    fn push_offload_plan(&self, flags: &mut Vec<Flag>) -> Result<(), ValueError> {
        match self.placement {
            Placement::Auto => {}
            Placement::FullOffload => flags.push(Flag::GpuLayers(FULL_OFFLOAD_LAYERS)),
            Placement::ExpertsOnCpu(n) => {
                flags.push(Flag::GpuLayers(FULL_OFFLOAD_LAYERS));
                flags.push(Flag::NCpuMoe(n));
            }
            Placement::DenseFfnOnCpu(n) => {
                flags.push(Flag::GpuLayers(FULL_OFFLOAD_LAYERS));
                flags.push(Flag::OverrideTensor(dense_ffn_rules(n)?));
            }
        }
        Ok(())
    }

    /// The cache type this candidate sets with flash attention on; `None` for the defaults.
    fn explicit_cache_type(&self) -> Option<CacheType> {
        match self.kv {
            KvCache::Default => None,
            KvCache::F16 => Some(CacheType::F16),
            KvCache::Q8 => Some(CacheType::Q8_0),
        }
    }

    fn push_pinned_threads(&self, flags: &mut Vec<Flag>) {
        if let ThreadPlan::Pinned { threads, mask } = self.threads {
            flags.extend([Flag::Threads(threads), Flag::CpuMask(mask), Flag::CpuStrict(true)]);
        }
    }
}

/// llama-server's default fit margin as a flag value.
fn fit_target() -> NonZeroU32 {
    // A non-zero literal; the fallback is unreachable and still a valid margin.
    NonZeroU32::new(SERVER_DEFAULT_FIT_TARGET_MIB).unwrap_or(NonZeroU32::MIN)
}

impl fmt::Display for CandidateConfig {
    /// A short human description, e.g. `experts of the first 32 blocks on CPU, q8_0 KV, 6 threads
    /// on 0x555`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_baseline() {
            return f.write_str("llama.cpp automatic settings (baseline)");
        }
        match self.placement {
            Placement::Auto => f.write_str("llama.cpp automatic placement")?,
            Placement::FullOffload => f.write_str("all layers on GPU")?,
            Placement::ExpertsOnCpu(n) => write!(f, "experts of the first {n} blocks on CPU")?,
            Placement::DenseFfnOnCpu(n) => write!(f, "FFN of the first {n} blocks on CPU")?,
        }
        match self.kv {
            KvCache::Default => {}
            KvCache::F16 => f.write_str(", f16 KV, flash attention")?,
            KvCache::Q8 => f.write_str(", q8_0 KV, flash attention")?,
        }
        if let ThreadPlan::Pinned { threads, mask } = self.threads {
            write!(f, ", {threads} threads on 0x{mask}")?;
        }
        Ok(())
    }
}

/// `-ngl` for "every layer": llama.cpp clamps it to the model's layer count plus the output layer.
pub const FULL_OFFLOAD_LAYERS: u32 = 999;

/// llama-server's default `--fit-ctx` (`llama-server --help`, and `llama-bench --help`: "minimum
/// ctx size for --fit-target (default: 4096)"), stated explicitly when serving the baseline.
pub const SERVER_DEFAULT_FIT_CTX: u32 = 4096;

/// How far over the budget the full-offload estimate may be and full offload still be measured:
/// a quarter. The KV estimate counts a full cache in every block, which sliding-window and
/// KV-less layers do not hold.
const NEAR_MISS_DIVISOR: u64 = 4;

/// VRAM kept free for llama.cpp's compute buffers, beyond llama-server's own fit margin.
pub const COMPUTE_RESERVE_MIB: u64 = 512;

/// Offload steps per model: the step size is the block count over this, at least 1.
const OFFLOAD_STEP_DIVISOR: u32 = 12;

const MIB: u64 = 1024 * 1024;

/// One `-ot` rule sending the dense FFN weights of blocks `0..n` to host memory; no rule (so no
/// `-ot` at all) for `n == 0`, rather than an empty alternation that matches nothing.
fn dense_ffn_rules(n: u32) -> Result<OtRuleSet, ValueError> {
    if n == 0 {
        return Ok(OtRuleSet::new());
    }
    let blocks = (0..n).map(|i| i.to_string()).collect::<Vec<_>>().join("|");
    let mut rules = OtRuleSet::new();
    rules.push(OtRule::new(format!(r"blk\.({blocks})\.ffn_(up|down|gate)\."), BufferType::Cpu)?);
    Ok(rules)
}

/// Free GPU memory as the tuner sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Vram {
    /// No GPU will be used (CPU backend, or no GPU present): no placement candidates beyond the
    /// baseline.
    None,
    /// A GPU may be used but its free memory is unknown (a non-NVIDIA GPU, which `nvidia-smi`
    /// cannot see): full offload is measured and the baseline protects against it being slower.
    Unknown,
    /// Free MiB per device, sampled just before tuning.
    FreeMib(Vec<u64>),
}

/// The machine facts candidate generation uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineFacts {
    /// Free GPU memory.
    pub vram: Vram,
    /// CPU topology for the thread variants; `None` skips them.
    pub cpu: Option<CpuTopology>,
}

impl MachineFacts {
    /// Probe this machine: CPU topology, and free VRAM of every NVIDIA GPU (none found reads as
    /// [`Vram::Unknown`], since other vendors' GPUs are invisible to `nvidia-smi`; a caller using a
    /// CPU backend should set [`Vram::None`]). A probe failure leaves the part unknown rather than
    /// failing the tune.
    pub fn probe() -> Self {
        let cpu = cerebro_probe::cpu::probe().ok();
        let vram = match cerebro_probe::gpu::probe() {
            Ok(gpus) if !gpus.is_empty() => match gpus.sample_free_vram() {
                Ok(free) if !free.is_empty() => Vram::FreeMib(free),
                _ => Vram::Unknown,
            },
            _ => Vram::Unknown,
        };
        Self { vram, cpu }
    }
}

/// Generates each stage's candidates for one model on one machine at one context size.
#[derive(Debug, Clone)]
pub struct Planner<'a> {
    model: &'a ModelFacts,
    machine: &'a MachineFacts,
    /// Tokens of KV cache the deepest test holds: depth plus the larger test size.
    kv_tokens: u64,
}

impl<'a> Planner<'a> {
    /// A planner sizing the KV cache for `kv_tokens` tokens of context.
    pub fn new(model: &'a ModelFacts, machine: &'a MachineFacts, kv_tokens: u64) -> Self {
        Self {
            model,
            machine,
            kv_tokens,
        }
    }

    /// Bytes available for weights and KV cache: free VRAM, less llama-server's fit margin per
    /// device and [`COMPUTE_RESERVE_MIB`]. `None` when free VRAM is unknown or there is no GPU.
    pub fn budget_bytes(&self) -> Option<u64> {
        let Vram::FreeMib(free) = &self.machine.vram else {
            return None;
        };
        if free.is_empty() {
            return None;
        }
        let total = free.iter().fold(0u64, |s, &f| s.saturating_add(f));
        let devices = u64::try_from(free.len()).unwrap_or(u64::MAX);
        let margin = u64::from(SERVER_DEFAULT_FIT_TARGET_MIB).saturating_mul(devices);
        Some(
            total
                .saturating_sub(margin)
                .saturating_sub(COMPUTE_RESERVE_MIB)
                .saturating_mul(MIB),
        )
    }

    /// Estimated VRAM `placement` needs with `kv`: GPU-resident weights plus the KV cache (none
    /// counted when the attention shape is unknown; the offload steps straddle the estimate
    /// anyway). `None` for [`Placement::Auto`], where llama.cpp decides.
    pub fn needed_bytes(&self, placement: Placement, kv: KvCache) -> Option<u64> {
        let full = self.model.full_offload_bytes();
        let weights = match placement {
            Placement::Auto => return None,
            Placement::FullOffload => full,
            Placement::ExpertsOnCpu(n) => full.saturating_sub(self.model.expert_bytes_of_first(n)),
            Placement::DenseFfnOnCpu(n) => {
                full.saturating_sub(self.model.dense_ffn_bytes_of_first(n))
            }
        };
        // q8_0 stores 32 values in 34 bytes; f16 is 2 bytes a value.
        let (num, den) = if kv == KvCache::Q8 { (34, 32) } else { (2, 1) };
        let kv_bytes = match self.model.kv_bytes_per_token(num, den) {
            Some(per_token) => per_token.checked_mul(self.kv_tokens)?,
            None => 0,
        };
        weights.checked_add(kv_bytes)
    }

    /// Whether the estimate says `placement` fits; `None` when it cannot say.
    pub fn fits(&self, placement: Placement, kv: KvCache) -> Option<bool> {
        Some(self.needed_bytes(placement, kv)? <= self.budget_bytes()?)
    }

    /// Whether the estimate for `placement` exceeds the budget by at most a quarter of it, close
    /// enough that the estimate's high bias may be all that stands between it and fitting.
    fn near_miss(&self, placement: Placement, kv: KvCache) -> bool {
        match (self.needed_bytes(placement, kv), self.budget_bytes()) {
            (Some(need), Some(budget)) => {
                need <= budget.saturating_add(budget / NEAR_MISS_DIVISOR)
            }
            _ => false,
        }
    }

    /// Offload step size for this model.
    fn step(&self) -> u32 {
        self.model.block_count.div_ceil(OFFLOAD_STEP_DIVISOR).max(1)
    }

    /// Stage 1, after the baseline: the placements to measure, each with f16 KV and flash
    /// attention.
    pub fn placements(&self) -> Vec<CandidateConfig> {
        let with = |placement| CandidateConfig {
            placement,
            kv: KvCache::F16,
            threads: ThreadPlan::Default,
        };
        let placements: Vec<Placement> = match (&self.machine.vram, self.fits(Placement::FullOffload, KvCache::F16)) {
            (Vram::None, _) => Vec::new(),
            // A sample listing no device is no GPU, not an unknown one.
            (Vram::FreeMib(free), _) if free.is_empty() => Vec::new(),
            // Free memory unknown: measure full offload and let the baseline guard against it.
            (Vram::Unknown, _) | (_, None) | (_, Some(true)) => vec![Placement::FullOffload],
            (Vram::FreeMib(_), Some(false)) => {
                let blocks = self.model.block_count;
                let kind: fn(u32) -> Placement = if self.model.is_moe() {
                    Placement::ExpertsOnCpu
                } else if self.model.dense_ffn_bytes_of_first(blocks) > 0 {
                    Placement::DenseFfnOnCpu
                } else {
                    return Vec::new();
                };
                if blocks == 0 {
                    return Vec::new();
                }
                // The lightest offload the estimate says fits, or every block when even that does
                // not; the steps below it are measured either way, since the estimate is biased
                // high and a lighter offload may still fit.
                let lightest = (1..=blocks)
                    .find(|&n| self.fits(kind(n), KvCache::F16) == Some(true))
                    .unwrap_or(blocks);
                let s = self.step();
                let mut steps: Vec<u32> = [
                    lightest.saturating_sub(s.saturating_mul(2)),
                    lightest.saturating_sub(s),
                    lightest,
                    lightest.saturating_add(s),
                ]
                .into_iter()
                .map(|n| n.clamp(1, blocks))
                .collect();
                steps.dedup();
                // A near miss on full offload is measured too: one extra run, and an out-of-memory
                // failure is only a recorded candidate failure.
                let near = self.near_miss(Placement::FullOffload, KvCache::F16);
                near.then_some(Placement::FullOffload)
                    .into_iter()
                    .chain(steps.into_iter().map(kind))
                    .collect()
            }
        };
        placements.into_iter().map(with).collect()
    }

    /// Stage 2: KV cache variants of `leader`.
    pub fn kv_variants(&self, leader: &CandidateConfig) -> Vec<CandidateConfig> {
        if leader.kv == KvCache::Q8 {
            return Vec::new();
        }
        let mut out = vec![CandidateConfig { kv: KvCache::Q8, ..*leader }];
        let lighter = match leader.placement {
            Placement::ExpertsOnCpu(n) if n > self.step() => {
                Some(Placement::ExpertsOnCpu(n - self.step()))
            }
            Placement::DenseFfnOnCpu(n) if n > self.step() => {
                Some(Placement::DenseFfnOnCpu(n - self.step()))
            }
            _ => None,
        };
        if let Some(placement) = lighter {
            out.push(CandidateConfig {
                placement,
                kv: KvCache::Q8,
                ..*leader
            });
        }
        out
    }

    /// Stage 3: thread variants of `leader`, from the CPU's performance-core masks.
    pub fn thread_variants(&self, leader: &CandidateConfig) -> Vec<CandidateConfig> {
        let Some(cpu) = &self.machine.cpu else {
            return Vec::new();
        };
        let one = cpu.p_core_mask_one_per_core().get();
        let all = cpu.p_core_mask_all_logical().get();
        let mut masks = vec![one];
        let half = lowest_bits(one, one.count_ones() / 2);
        if half != 0 {
            masks.push(half);
        }
        if all != one {
            masks.push(all);
        }
        let mut out: Vec<CandidateConfig> = Vec::new();
        for bits in masks {
            let Ok(mask) = CpuMask::new(bits) else { continue };
            let candidate = CandidateConfig {
                threads: ThreadPlan::Pinned {
                    threads: mask.count(),
                    mask,
                },
                ..*leader
            };
            if candidate != *leader && !out.contains(&candidate) {
                out.push(candidate);
            }
        }
        out
    }
}

/// The lowest `count` set bits of `mask`.
fn lowest_bits(mask: u64, count: u32) -> u64 {
    let mut out = 0u64;
    let mut rest = mask;
    for _ in 0..count {
        let bit = rest & rest.wrapping_neg();
        if bit == 0 {
            break;
        }
        out |= bit;
        rest &= !bit;
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::gguf::BlockBytes;
    use cerebro_flags::{ArgvBuilder, Dialect};
    use cerebro_probe::cpu::PhysicalCore;

    /// A Qwen3-30B-A3B-shaped MoE: 48 blocks, about 17 GiB, experts about 95% of it.
    pub(crate) fn big_moe() -> ModelFacts {
        ModelFacts {
            architecture: "qwen3moe".into(),
            block_count: 48,
            expert_count: Some(128),
            context_length: Some(40960),
            kv_heads: Some(4),
            key_length: Some(128),
            value_length: Some(128),
            input_bytes: 300 * MIB,
            other_bytes: 300 * MIB,
            blocks: vec![BlockBytes { total: 360 * MIB, experts: 340 * MIB, dense_ffn: 0 }; 48],
        }
    }

    /// A 0.5B dense model: 24 blocks, about 1 GiB.
    pub(crate) fn small_dense() -> ModelFacts {
        ModelFacts {
            architecture: "qwen2".into(),
            block_count: 24,
            expert_count: None,
            context_length: Some(32768),
            kv_heads: Some(2),
            key_length: Some(64),
            value_length: Some(64),
            input_bytes: 260 * MIB,
            other_bytes: MIB,
            blocks: vec![BlockBytes { total: 30 * MIB, experts: 0, dense_ffn: 20 * MIB }; 24],
        }
    }

    /// The development laptop's i9-12900H: 6 SMT P-cores, 8 E-cores.
    pub(crate) fn laptop_cpu() -> CpuTopology {
        let mut cores = Vec::new();
        for i in 0..6 {
            cores.push(PhysicalCore { efficiency_class: 1, smt: true, mask: 0b11 << (2 * i), group: 0 });
        }
        for i in 0..8 {
            cores.push(PhysicalCore { efficiency_class: 0, smt: false, mask: 1 << (12 + i), group: 0 });
        }
        CpuTopology::from_cores(&cores, 1).unwrap()
    }

    fn machine(free_mib: u64) -> MachineFacts {
        MachineFacts { vram: Vram::FreeMib(vec![free_mib]), cpu: Some(laptop_cpu()) }
    }

    /// `c` builds in both dialects: llama-bench for screening, llama-server for serving.
    pub(crate) fn assert_bench_valid(c: &CandidateConfig) {
        let mut b = ArgvBuilder::new(Dialect::Bench);
        b.push(Flag::Model(r"C:\models\m.gguf".into())).unwrap();
        for f in c.bench_flags().unwrap() {
            b.push(f).unwrap();
        }
        b.build().unwrap_or_else(|e| panic!("bench {c}: {e}"));
        for threads in [None, Some(14)] {
            server_argv(c, threads);
        }
    }

    /// The llama-server argv for `c` (after `-m`), panicking if the Server dialect refuses it.
    pub(crate) fn server_argv(c: &CandidateConfig, threads: Option<u32>) -> Vec<String> {
        let mut b = ArgvBuilder::new(Dialect::Server);
        b.push(Flag::Model(r"C:\models\m.gguf".into())).unwrap();
        for f in c.server_flags(threads).unwrap() {
            b.push(f).unwrap();
        }
        let argv = b.build().unwrap_or_else(|e| panic!("server {c}: {e}"));
        argv[2..].to_vec()
    }

    #[test]
    fn a_model_that_fits_gets_full_offload_only() {
        let model = small_dense();
        let m = machine(7000);
        let p = Planner::new(&model, &m, 4096 + 512);
        assert_eq!(p.fits(Placement::FullOffload, KvCache::F16), Some(true));
        let placements = p.placements();
        assert_eq!(placements.len(), 1);
        assert_eq!(placements[0].placement, Placement::FullOffload);
        assert_eq!(placements[0].kv, KvCache::F16);
    }

    #[test]
    fn a_moe_that_does_not_fit_gets_expert_offload_steps_straddling_the_estimate() {
        let model = big_moe();
        let m = machine(7300);
        let tokens = 16_384 + 512;
        let p = Planner::new(&model, &m, tokens);
        assert_eq!(p.fits(Placement::FullOffload, KvCache::F16), Some(false));
        let steps: Vec<u32> = p
            .placements()
            .iter()
            .map(|c| match c.placement {
                Placement::ExpertsOnCpu(n) => n,
                other => panic!("expected expert offload, got {other:?}"),
            })
            .collect();
        // Budget: 7300 - 1024 - 512 = 5764 MiB. Non-expert weights 300 + 48 x 20 = 1260 MiB, KV
        // 48 x 4 x 256 x 2 B x 16896 tokens = 1584 MiB, leaving 2920 MiB = 8 blocks of experts on
        // the GPU, so the lightest fitting offload keeps 40 blocks' experts on the CPU.
        let lightest = 40;
        assert_eq!(p.fits(Placement::ExpertsOnCpu(lightest), KvCache::F16), Some(true));
        assert_eq!(p.fits(Placement::ExpertsOnCpu(lightest - 1), KvCache::F16), Some(false));
        assert_eq!(steps, [32, 36, 40, 44], "step 4 = 48 blocks / 12");
        for c in p.placements() {
            assert_bench_valid(&c);
        }
    }

    #[test]
    fn a_dense_model_that_does_not_fit_gets_ffn_offload_steps() {
        let mut model = small_dense();
        model.blocks.fill(BlockBytes { total: 400 * MIB, experts: 0, dense_ffn: 260 * MIB });
        let m = machine(6000);
        let p = Planner::new(&model, &m, 4096 + 512);
        let placements = p.placements();
        assert!(!placements.is_empty());
        for c in &placements {
            assert!(matches!(c.placement, Placement::DenseFfnOnCpu(n) if (1..=24).contains(&n)), "{c}");
            assert_bench_valid(c);
        }
        let flags = CandidateConfig { placement: Placement::DenseFfnOnCpu(3), ..placements[0] }
            .bench_flags()
            .unwrap();
        let mut b = ArgvBuilder::new(Dialect::Bench);
        for f in flags {
            b.push(f).unwrap();
        }
        let argv = b.build().unwrap();
        let ot = argv.iter().position(|a| a == "-ot").unwrap();
        assert_eq!(argv[ot + 1], r"blk\.(0|1|2)\.ffn_(up|down|gate)\.=CPU");
        let none = CandidateConfig { placement: Placement::DenseFfnOnCpu(0), ..placements[0] };
        assert_eq!(none.bench_flags().unwrap()[..2], [Flag::GpuLayers(999), Flag::OverrideTensor(OtRuleSet::new())]);
        assert_bench_valid(&none);
    }

    #[test]
    fn no_gpu_means_no_placement_candidates_and_unknown_vram_means_full_offload() {
        let model = big_moe();
        let none = MachineFacts { vram: Vram::None, cpu: None };
        assert!(Planner::new(&model, &none, 4608).placements().is_empty());
        let empty = MachineFacts { vram: Vram::FreeMib(Vec::new()), cpu: None };
        assert!(Planner::new(&model, &empty, 4608).placements().is_empty());
        let unknown = MachineFacts { vram: Vram::Unknown, cpu: None };
        let p = Planner::new(&model, &unknown, 4608);
        assert_eq!(p.placements()[0].placement, Placement::FullOffload);
        assert_eq!(p.fits(Placement::FullOffload, KvCache::F16), None);

        // An unknown attention shape still gets offload steps, sized from the weights alone.
        let mut shapeless = big_moe();
        shapeless.kv_heads = None;
        let m = machine(7300);
        let p = Planner::new(&shapeless, &m, 16_896);
        assert!(p.placements().iter().all(|c| matches!(c.placement, Placement::ExpertsOnCpu(_))));
    }

    #[test]
    fn q8_kv_halves_the_cache_estimate_and_kv_variants_try_a_lighter_step() {
        let model = big_moe();
        let m = machine(7300);
        let p = Planner::new(&model, &m, 16_896);
        let f16 = p.needed_bytes(Placement::FullOffload, KvCache::F16).unwrap();
        let q8 = p.needed_bytes(Placement::FullOffload, KvCache::Q8).unwrap();
        let kv_f16 = model.kv_bytes_per_token(2, 1).unwrap() * 16_896;
        assert_eq!(f16 - q8, kv_f16 - kv_f16 * 17 / 32);

        let leader = CandidateConfig { placement: Placement::ExpertsOnCpu(40), kv: KvCache::F16, threads: ThreadPlan::Default };
        let v = p.kv_variants(&leader);
        assert_eq!(v[0], CandidateConfig { kv: KvCache::Q8, ..leader });
        assert_eq!(v[1].placement, Placement::ExpertsOnCpu(36));
        assert!(p.kv_variants(&v[0]).is_empty(), "q8_0 is already the smallest cache tried");
        let base = p.kv_variants(&CandidateConfig::BASELINE);
        assert_eq!(base, [CandidateConfig { kv: KvCache::Q8, ..CandidateConfig::BASELINE }]);
        for c in v.iter().chain(&base) {
            assert_bench_valid(c);
        }
    }

    #[test]
    fn thread_variants_come_from_the_performance_core_masks() {
        let model = small_dense();
        let m = machine(7000);
        let p = Planner::new(&model, &m, 4608);
        let v = p.thread_variants(&CandidateConfig::BASELINE);
        let pinned: Vec<(u32, u64)> = v
            .iter()
            .map(|c| match c.threads {
                ThreadPlan::Pinned { threads, mask } => (threads, mask.bits()),
                ThreadPlan::Default => panic!("not pinned"),
            })
            .collect();
        assert_eq!(pinned, [(6, 0x555), (3, 0x015), (12, 0xFFF)]);
        for c in &v {
            assert_eq!(c.placement, Placement::Auto, "only the threads change");
            assert_bench_valid(c);
        }
        let no_cpu = MachineFacts { vram: Vram::None, cpu: None };
        assert!(Planner::new(&model, &no_cpu, 4608).thread_variants(&CandidateConfig::BASELINE).is_empty());
    }

    #[test]
    fn a_near_miss_on_full_offload_measures_full_offload_too() {
        let model = big_moe();
        let tokens = 16_384 + 512;
        // Full offload needs 17580 MiB of weights + 1584 MiB of KV = 19164 MiB. 18000 MiB free
        // leaves a 16464 MiB budget: over by 2700 MiB, within a quarter (4116 MiB).
        let m = machine(18_000);
        let p = Planner::new(&model, &m, tokens);
        assert_eq!(p.fits(Placement::FullOffload, KvCache::F16), Some(false));
        let placements: Vec<Placement> = p.placements().iter().map(|c| c.placement).collect();
        assert_eq!(
            placements,
            [
                Placement::FullOffload,
                Placement::ExpertsOnCpu(1),
                Placement::ExpertsOnCpu(4),
                Placement::ExpertsOnCpu(8),
                Placement::ExpertsOnCpu(12),
            ]
        );
        for c in p.placements() {
            assert_bench_valid(&c);
        }
        // 7300 MiB free is far from fitting: no full offload.
        let far = machine(7300);
        let p = Planner::new(&model, &far, tokens);
        assert!(p.placements().iter().all(|c| c.placement != Placement::FullOffload));
    }

    #[test]
    fn every_candidate_is_served_as_measured_with_nothing_left_to_a_server_default() {
        // The baseline states llama-server's automatic settings rather than inheriting them.
        assert_eq!(
            server_argv(&CandidateConfig::BASELINE, Some(14)),
            ["--fit", "on", "--fit-target", "1024", "--fit-ctx", "4096", "-fa", "auto", "-ctk", "f16", "-ctv", "f16", "-t", "14"]
        );
        // An explicit placement turns the fitter off, as llama-bench measured it.
        let tuned = CandidateConfig {
            placement: Placement::ExpertsOnCpu(32),
            kv: KvCache::Q8,
            threads: ThreadPlan::Pinned { threads: 6, mask: CpuMask::new(0x555).unwrap() },
        };
        assert_eq!(
            server_argv(&tuned, Some(14)),
            ["--fit", "off", "-ngl", "999", "-ncmoe", "32", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-t", "6", "-C", "555", "--cpu-strict", "1"]
        );
        // The bench flags of the same candidate differ only by what llama-bench lacks (`--fit`).
        let bench: Vec<Flag> = tuned.bench_flags().unwrap();
        let server: Vec<Flag> = tuned.server_flags(Some(14)).unwrap();
        assert_eq!(server[0], Flag::Fit(false));
        assert_eq!(server[1..], bench[..]);

        // Every candidate every stage can generate, on several machines, builds in both dialects.
        for (model, free) in [(big_moe(), 7300), (big_moe(), 18_000), (small_dense(), 7000), (small_dense(), 1200)] {
            for vram in [Vram::FreeMib(vec![free]), Vram::Unknown, Vram::None] {
                let m = MachineFacts { vram, cpu: Some(laptop_cpu()) };
                let p = Planner::new(&model, &m, 16_896);
                let mut all = vec![CandidateConfig::BASELINE];
                all.extend(p.placements());
                let leaders = all.clone();
                for leader in &leaders {
                    let kv = p.kv_variants(leader);
                    for c in kv.iter().chain(std::iter::once(leader)) {
                        all.extend(p.thread_variants(c));
                    }
                    all.extend(kv);
                }
                assert!(all.len() > 1);
                for c in &all {
                    assert_bench_valid(c);
                }
            }
        }
    }

    #[test]
    fn labels_say_what_changed() {
        assert_eq!(CandidateConfig::BASELINE.to_string(), "llama.cpp automatic settings (baseline)");
        let c = CandidateConfig {
            placement: Placement::ExpertsOnCpu(32),
            kv: KvCache::Q8,
            threads: ThreadPlan::Pinned { threads: 6, mask: CpuMask::new(0x555).unwrap() },
        };
        assert_eq!(c.to_string(), "experts of the first 32 blocks on CPU, q8_0 KV, flash attention, 6 threads on 0x555");
    }
}
