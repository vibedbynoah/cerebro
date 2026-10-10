//! The closed flag allowlist and dialect-aware argv emission.
//!
//! # Why this is a closed enum
//!
//! The autotuner generates flag combinations and persists them; a results file or config that
//! Cerebro later reads back can therefore influence what argv gets constructed. If the emittable
//! flag space were open — a `HashMap<String, String>` spliced into argv — a poisoned file could
//! steer the tuner into emitting flags that reach the network, load remote models, or enable tool
//! execution. [`Flag`] is a closed enum precisely so that no such flag is *representable*. Adding
//! one requires a source change and a review, not a config edit.
//!
//! No shell is involved in spawning (arguments go to `Command::arg` directly), so classic
//! metacharacter injection does not apply. The live risk is *argument* injection — a value that
//! terminates its own flag and begins another — which the typed values in [`crate::value`] prevent
//! structurally.
//!
//! # Why there are two dialects
//!
//! `llama-server` and `llama-bench` disagree about separators, because `llama-bench` reserves the
//! comma for its multi-value sweep syntax. Reusing one serializer across both is a silent
//! correctness bug, not a formatting nit: `-dev CUDA0,RPC0` means *two devices* to the server and
//! *two separate benchmark runs* to bench.
//!
//! | Value | `llama-server` / `llama-cli` | `llama-bench` |
//! |---|---|---|
//! | device list | `CUDA0,RPC0` | `CUDA0/RPC0` |
//! | tensor split | `0.5,0.5` | `0.5/0.5` |
//! | `-ot` rules | `a=CPU,b=CPU` | `a=CPU;b=CPU` |
//!
//! Every flag name below was verified against b10751 (commit 3466812d1) by reading its `--help`
//! output and, where behavior was in question, by running the binary.

use std::collections::BTreeSet;
use std::path::Path;

use crate::constraints::{Severity, Violation};
use crate::error::EmitError;
use crate::value::{
    CacheType, CpuMask, DeviceList, FlashAttn, LoadMode, OtRuleSet, SpecType, SplitMode, TensorSplit,
};

/// Which backend tool an argv is being built for.
///
/// Dialects differ in separator syntax and in which flags exist at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dialect {
    /// `llama-server` and `llama-cli`, which share an argument parser.
    Server,
    /// `llama-bench`, which has a reduced flag set and different separators.
    Bench,
}

impl Dialect {
    /// Short, lowercase name used in error messages (`"server"` / `"bench"`).
    pub fn name(self) -> &'static str {
        match self {
            Dialect::Server => "server",
            Dialect::Bench => "bench",
        }
    }

    fn device_sep(self) -> &'static str {
        match self {
            Dialect::Server => ",",
            Dialect::Bench => "/",
        }
    }

    fn tensor_split_sep(self) -> &'static str {
        match self {
            Dialect::Server => ",",
            Dialect::Bench => "/",
        }
    }

    fn ot_sep(self) -> &'static str {
        match self {
            Dialect::Server => ",",
            Dialect::Bench => ";",
        }
    }
}

/// Where a flag is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Availability {
    Both,
    ServerOnly(&'static str),
}

/// The complete set of flags Cerebro is permitted to emit.
///
/// Ordering of variants mirrors the architecture's search axes so that a reviewer can check the
/// allowlist against the design document section by section.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Flag {
    // ---- model selection ----
    /// `-m` — the target model. Must be an absolute path.
    Model(String),
    /// `-md` — the draft model, for the `draft-*` speculation family.
    DraftModel(String),
    /// `-c` — context size.
    CtxSize(u32),

    // ---- axis 2: offload plan ----
    /// `-ngl` — offloads the TOP N layers; the output layer counts as one.
    GpuLayers(u32),
    /// `-ncmoe` — keep routed MoE expert weights of the first N layers in host memory.
    NCpuMoe(u32),
    /// `-cmoe` — keep all routed MoE expert weights in host memory.
    CpuMoe,
    /// `-ncffn` — keep dense FFN weights of the first N layers in host memory.
    NCpuFfn(u32),
    /// `-ot` — per-tensor buffer overrides, first-match-wins in the given order.
    OverrideTensor(OtRuleSet),
    /// `-dev` — the addressable device list for the target model.
    Device(DeviceList),
    /// `-sm` — multi-device split strategy.
    SplitMode(SplitMode),
    /// `-ts` — per-device proportions.
    TensorSplit(TensorSplit),
    /// `-mg` — main device index.
    MainGpu(u32),
    /// `-lm` — model loading mode. Emitted in place of the deprecated `--mlock`/`--mmap`/`-dio`.
    LoadMode(LoadMode),

    // ---- axis 3: KV and context ----
    /// `-ctk` — K cache element type.
    CacheTypeK(CacheType),
    /// `-ctv` — V cache element type.
    CacheTypeV(CacheType),
    /// `-fa` — flash attention mode.
    FlashAttn(FlashAttn),

    // ---- axis 4: CPU placement ----
    /// `-t` — generation thread count.
    Threads(u32),
    /// `-tb` — prompt-processing thread count.
    ThreadsBatch(u32),
    /// `-C` — generation affinity mask.
    CpuMask(CpuMask),
    /// `--cpu-strict` — whether affinity is binding rather than a hint.
    CpuStrict(bool),
    /// `--poll` — busy-wait level, 0..=100.
    Poll(u8),
    /// `-b` — logical maximum batch size.
    BatchSize(u32),
    /// `-ub` — physical maximum batch size.
    UBatchSize(u32),

    // ---- axis 1: speculation (server dialect only) ----
    /// `--spec-type` — the speculation strategy.
    SpecType(SpecType),
    /// `--spec-draft-n-max` — maximum tokens drafted per step (upstream default 3).
    SpecDraftNMax(u32),
    /// `--spec-draft-n-min` — minimum tokens drafted per step.
    SpecDraftNMin(u32),
    /// `--spec-draft-p-min` — minimum draft acceptance probability (upstream default 0.00).
    SpecDraftPMin(f32),
    /// `-devd` — device list for the draft model. Has no environment binding upstream, which is
    /// one reason Cerebro's config format is argv-based rather than environment-based.
    SpecDraftDevice(DeviceList),
    /// `-ngld` — layers offloaded for the draft model.
    SpecDraftGpuLayers(u32),
    /// `-Cd` — draft-model affinity mask, e.g. pinning the draft to E-cores.
    SpecDraftCpuMask(CpuMask),
    /// `--cpu-strict-draft` — whether the draft affinity is binding.
    SpecDraftCpuStrict(bool),
    /// `-ctkd` — draft K cache type.
    SpecDraftCacheTypeK(CacheType),
    /// `-ctvd` — draft V cache type.
    SpecDraftCacheTypeV(CacheType),
    /// `--spec-ngram-simple-size-n` — lookup length for `ngram-simple`.
    SpecNgramSimpleSizeN(u32),
    /// `--spec-ngram-simple-size-m` — draft length for `ngram-simple`.
    SpecNgramSimpleSizeM(u32),
    /// `--spec-ngram-simple-min-hits` — minimum hits before `ngram-simple` drafts.
    SpecNgramSimpleMinHits(u32),
    /// `--spec-draft-p-split` — split probability (upstream default 0.10).
    SpecDraftPSplit(f32),
    /// `--spec-draft-backend-sampling` / `--no-...` — upstream default is ENABLED. Offloads draft
    /// sampling to the backend; mutually exclusive with grammar-constrained output.
    SpecDraftBackendSampling(bool),
    /// `--spec-synth-len` — synthetic mean acceptance length. Benchmarking only; lets Cerebro find
    /// this machine's break-even acceptance rate without obtaining any real draft model.
    SpecSynthLen(u32),
    /// `--spec-synth-rates` — synthetic per-position acceptance probabilities. Benchmarking only.
    SpecSynthRates(Vec<f32>),

    // ---- axis 0: sampler entropy ----
    // Speculation acceptance is a direct function of how peaked the target distribution is, so
    // these are not user preferences here — they are the conditions under which every Axis 1
    // measurement is valid, and belong in every result record.
    /// `--temp` — sampling temperature (upstream default 0.80).
    Temp(f32),
    /// `--top-k` — top-k truncation (upstream default 40).
    TopK(u32),
    /// `--top-p` — nucleus truncation (upstream default 0.95).
    TopP(f32),
    /// `--min-p` — min-p truncation (upstream default 0.05, already an upstream default and
    /// therefore not something Cerebro may claim credit for).
    MinP(f32),
    /// `-bs` — enable backend sampling (upstream default DISABLED, experimental).
    BackendSampling,

    // ---- structured output ----
    /// `--grammar` — GBNF grammar constraining generation.
    Grammar(String),
    /// `-j` — JSON schema constraining generation.
    JsonSchema(String),

    // ---- additional KV / context axes ----
    /// `-nkvo` / `-kvo` — whether the KV cache is offloaded. Directly moves the KV-versus-weights
    /// VRAM boundary, which is the binding constraint on a small card.
    KvOffload(bool),
    /// `--cache-reuse` — minimum chunk size reusable from cache via KV shifting.
    CacheReuse(u32),
    /// `-cram` — maximum host cache size in MiB (upstream default 8192; -1 means no limit).
    CacheRam(i64),
    /// `--swa-full` — use a full-size sliding-window cache.
    SwaFull,
    /// `-ctxcp` — number of context checkpoints.
    CtxCheckpoints(u32),
    /// `-cms` — minimum spacing between context checkpoints (upstream default 8192).
    CheckpointMinStep(u32),
    /// `--slot-save-path` — directory for slot KV persistence.
    SlotSavePath(String),

    // ---- reproducibility ----
    /// `--fit` — upstream's memory autofitter. Cerebro emits `off` for measured runs so that the
    /// configuration under test is exactly the configuration recorded.
    Fit(bool),

    // ---- server hardening (server dialect only) ----
    /// `--host` — bind address.
    Host(String),
    /// `--port` — bind port.
    Port(u16),
    /// `--no-webui` / `--webui` — upstream default is ENABLED.
    WebUi(bool),
    /// `--no-slots` / `--slots` — upstream default is ENABLED, exposing slot state on loopback.
    Slots(bool),
    /// `--cors-origins` — upstream default is `*`, fully permissive.
    CorsOrigins(String),
    /// `--api-key-file` — path to a file of bearer tokens, one per line.
    ApiKeyFile(String),
}

impl Flag {
    /// The canonical flag name Cerebro emits.
    ///
    /// Short forms are used where upstream's help text leads with them, long forms otherwise. The
    /// choice is fixed so that recorded argvs are diffable across runs.
    pub fn name(&self) -> &'static str {
        match self {
            Flag::Model(_) => "-m",
            Flag::DraftModel(_) => "-md",
            Flag::CtxSize(_) => "-c",
            Flag::GpuLayers(_) => "-ngl",
            Flag::NCpuMoe(_) => "-ncmoe",
            Flag::CpuMoe => "-cmoe",
            Flag::NCpuFfn(_) => "-ncffn",
            Flag::OverrideTensor(_) => "-ot",
            Flag::Device(_) => "-dev",
            Flag::SplitMode(_) => "-sm",
            Flag::TensorSplit(_) => "-ts",
            Flag::MainGpu(_) => "-mg",
            Flag::LoadMode(_) => "-lm",
            Flag::CacheTypeK(_) => "-ctk",
            Flag::CacheTypeV(_) => "-ctv",
            Flag::FlashAttn(_) => "-fa",
            Flag::Threads(_) => "-t",
            Flag::ThreadsBatch(_) => "-tb",
            Flag::CpuMask(_) => "-C",
            Flag::CpuStrict(_) => "--cpu-strict",
            Flag::Poll(_) => "--poll",
            Flag::BatchSize(_) => "-b",
            Flag::UBatchSize(_) => "-ub",
            Flag::SpecType(_) => "--spec-type",
            Flag::SpecDraftNMax(_) => "--spec-draft-n-max",
            Flag::SpecDraftNMin(_) => "--spec-draft-n-min",
            Flag::SpecDraftPMin(_) => "--spec-draft-p-min",
            Flag::SpecDraftDevice(_) => "-devd",
            Flag::SpecDraftGpuLayers(_) => "-ngld",
            Flag::SpecDraftCpuMask(_) => "-Cd",
            Flag::SpecDraftCpuStrict(_) => "--cpu-strict-draft",
            Flag::SpecDraftCacheTypeK(_) => "-ctkd",
            Flag::SpecDraftCacheTypeV(_) => "-ctvd",
            Flag::SpecNgramSimpleSizeN(_) => "--spec-ngram-simple-size-n",
            Flag::SpecNgramSimpleSizeM(_) => "--spec-ngram-simple-size-m",
            Flag::SpecNgramSimpleMinHits(_) => "--spec-ngram-simple-min-hits",
            Flag::SpecDraftPSplit(_) => "--spec-draft-p-split",
            Flag::SpecDraftBackendSampling(_) => "--spec-draft-backend-sampling",
            Flag::SpecSynthLen(_) => "--spec-synth-len",
            Flag::SpecSynthRates(_) => "--spec-synth-rates",
            Flag::Temp(_) => "--temp",
            Flag::TopK(_) => "--top-k",
            Flag::TopP(_) => "--top-p",
            Flag::MinP(_) => "--min-p",
            Flag::BackendSampling => "-bs",
            Flag::Grammar(_) => "--grammar",
            Flag::JsonSchema(_) => "-j",
            Flag::KvOffload(_) => "--kv-offload",
            Flag::CacheReuse(_) => "--cache-reuse",
            Flag::CacheRam(_) => "-cram",
            Flag::SwaFull => "--swa-full",
            Flag::CtxCheckpoints(_) => "-ctxcp",
            Flag::CheckpointMinStep(_) => "-cms",
            Flag::SlotSavePath(_) => "--slot-save-path",
            Flag::Fit(_) => "--fit",
            Flag::Host(_) => "--host",
            Flag::Port(_) => "--port",
            Flag::WebUi(_) => "--webui",
            Flag::Slots(_) => "--slots",
            Flag::CorsOrigins(_) => "--cors-origins",
            Flag::ApiKeyFile(_) => "--api-key-file",
        }
    }

    fn availability(&self) -> Availability {
        match self {
            // llama-bench has no speculative surface whatsoever. This is the single most important
            // dialect difference: it is why the entire speculation axis needs a second harness.
            Flag::SpecType(_)
            | Flag::SpecDraftNMax(_)
            | Flag::SpecDraftNMin(_)
            | Flag::SpecDraftPMin(_)
            | Flag::SpecDraftDevice(_)
            | Flag::SpecDraftGpuLayers(_)
            | Flag::SpecDraftCpuMask(_)
            | Flag::SpecDraftCpuStrict(_)
            | Flag::SpecDraftCacheTypeK(_)
            | Flag::SpecDraftCacheTypeV(_)
            | Flag::SpecNgramSimpleSizeN(_)
            | Flag::SpecNgramSimpleSizeM(_)
            | Flag::SpecNgramSimpleMinHits(_)
            | Flag::SpecDraftPSplit(_)
            | Flag::SpecDraftBackendSampling(_)
            | Flag::SpecSynthLen(_)
            | Flag::SpecSynthRates(_)
            | Flag::DraftModel(_) => Availability::ServerOnly(
                "llama-bench has no speculative decoding flags; drive speculation through llama-server",
            ),

            Flag::CpuMoe | Flag::NCpuFfn(_) => {
                Availability::ServerOnly("not present in the llama-bench flag set")
            }

            // Samplers do not exist in llama-bench, which measures raw throughput and never
            // samples. They are recorded as measurement CONDITIONS for Axis 1 rather than swept
            // there; see ARCHITECTURE.md Axis 0.
            Flag::Temp(_)
            | Flag::TopK(_)
            | Flag::TopP(_)
            | Flag::MinP(_)
            | Flag::BackendSampling
            | Flag::Grammar(_)
            | Flag::JsonSchema(_) => {
                Availability::ServerOnly("sampling and constrained-output flags are server-only")
            }

            Flag::Host(_)
            | Flag::Port(_)
            | Flag::WebUi(_)
            | Flag::Slots(_)
            | Flag::CorsOrigins(_)
            | Flag::ApiKeyFile(_)
            | Flag::Fit(_)
            | Flag::ThreadsBatch(_)
            | Flag::LoadMode(_)
            | Flag::CacheReuse(_)
            | Flag::CacheRam(_)
            | Flag::SwaFull
            | Flag::CtxCheckpoints(_)
            | Flag::CheckpointMinStep(_)
            | Flag::SlotSavePath(_)
            | Flag::CtxSize(_) => Availability::ServerOnly("server-only flag"),

            _ => Availability::Both,
        }
    }

    /// Render this flag into argv tokens for the given dialect.
    pub fn render(&self, dialect: Dialect) -> Result<Vec<String>, EmitError> {
        if let Availability::ServerOnly(reason) = self.availability() {
            if dialect == Dialect::Bench {
                return Err(EmitError::NotInDialect {
                    flag: self.name(),
                    dialect: dialect.name(),
                    reason,
                });
            }
        }

        let name = self.name().to_string();
        let one = |v: String| -> Result<Vec<String>, EmitError> { Ok(vec![name.clone(), v]) };

        match self {
            Flag::Model(p) | Flag::DraftModel(p) | Flag::ApiKeyFile(p) | Flag::SlotSavePath(p) => {
                one(absolute(p)?)
            }

            Flag::CtxSize(n)
            | Flag::GpuLayers(n)
            | Flag::NCpuMoe(n)
            | Flag::NCpuFfn(n)
            | Flag::MainGpu(n)
            | Flag::Threads(n)
            | Flag::ThreadsBatch(n)
            | Flag::BatchSize(n)
            | Flag::UBatchSize(n)
            | Flag::SpecDraftNMax(n)
            | Flag::SpecDraftNMin(n)
            | Flag::SpecDraftGpuLayers(n)
            | Flag::SpecNgramSimpleSizeN(n)
            | Flag::SpecNgramSimpleSizeM(n)
            | Flag::SpecNgramSimpleMinHits(n)
            | Flag::SpecSynthLen(n)
            | Flag::CacheReuse(n)
            | Flag::CtxCheckpoints(n)
            | Flag::CheckpointMinStep(n)
            | Flag::TopK(n) => one(n.to_string()),

            Flag::Port(p) => one(p.to_string()),
            Flag::Poll(p) => one(p.to_string()),
            Flag::CacheRam(n) => one(n.to_string()),
            Flag::SpecDraftPMin(p)
            | Flag::SpecDraftPSplit(p)
            | Flag::Temp(p)
            | Flag::TopP(p)
            | Flag::MinP(p) => one(format!("{p}")),

            Flag::SpecSynthRates(rates) => {
                // Server dialect only, so the separator is always a comma.
                let joined = rates
                    .iter()
                    .map(|r| format!("{r}"))
                    .collect::<Vec<_>>()
                    .join(",");
                one(joined)
            }

            Flag::Grammar(g) => one(g.clone()),
            Flag::JsonSchema(s) => one(s.clone()),

            // Valueless switches.
            Flag::CpuMoe | Flag::SwaFull | Flag::BackendSampling => Ok(vec![name]),

            // Boolean pairs where upstream spells the negation as a distinct flag rather than a
            // value. Emitting the wrong member here silently keeps an insecure default.
            Flag::WebUi(on) => Ok(vec![if *on { "--webui" } else { "--no-webui" }.to_string()]),
            Flag::Slots(on) => Ok(vec![if *on { "--slots" } else { "--no-slots" }.to_string()]),
            Flag::KvOffload(on) => Ok(vec![if *on {
                "--kv-offload"
            } else {
                "--no-kv-offload"
            }
            .to_string()]),
            Flag::SpecDraftBackendSampling(on) => Ok(vec![if *on {
                "--spec-draft-backend-sampling"
            } else {
                "--no-spec-draft-backend-sampling"
            }
            .to_string()]),

            // Booleans upstream takes as a 0|1 value.
            Flag::CpuStrict(b) | Flag::SpecDraftCpuStrict(b) => {
                one(if *b { "1" } else { "0" }.to_string())
            }

            // `--fit` takes on|off, not a --no- form.
            Flag::Fit(on) => one(if *on { "on" } else { "off" }.to_string()),

            Flag::OverrideTensor(rules) => {
                if rules.is_empty() {
                    return Ok(Vec::new());
                }
                let joined = rules
                    .rules()
                    .iter()
                    .map(|r| r.to_string())
                    .collect::<Vec<_>>()
                    .join(dialect.ot_sep());
                one(joined)
            }

            Flag::Device(list) | Flag::SpecDraftDevice(list) => match list {
                // The sentinel is lowercase-only and cannot be mixed into a list; both rules are
                // enforced by DeviceList being an enum rather than a magic string.
                DeviceList::None => one("none".to_string()),
                DeviceList::Devices(ids) => {
                    let joined = ids
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(dialect.device_sep());
                    one(joined)
                }
            },

            Flag::TensorSplit(ts) => {
                let joined = ts
                    .parts()
                    .iter()
                    .map(|p| format!("{p}"))
                    .collect::<Vec<_>>()
                    .join(dialect.tensor_split_sep());
                one(joined)
            }

            Flag::SplitMode(v) => one(v.to_string()),
            Flag::LoadMode(v) => one(v.to_string()),
            Flag::FlashAttn(v) => one(v.to_string()),
            Flag::CacheTypeK(v)
            | Flag::CacheTypeV(v)
            | Flag::SpecDraftCacheTypeK(v)
            | Flag::SpecDraftCacheTypeV(v) => one(v.to_string()),
            Flag::SpecType(v) => one(v.to_string()),
            Flag::CpuMask(v) | Flag::SpecDraftCpuMask(v) => one(v.to_string()),
            Flag::Host(h) => one(h.clone()),
            Flag::CorsOrigins(o) => one(o.clone()),
        }
    }
}

fn absolute(p: &str) -> Result<String, EmitError> {
    let path = Path::new(p);
    if !path.is_absolute() {
        return Err(EmitError::RelativePath(p.to_string()));
    }
    Ok(p.to_string())
}

/// Builds a validated argv for one backend tool invocation.
///
/// Duplicate flags are rejected rather than last-wins, because a duplicate in a generated
/// configuration means two parts of the planner disagreed — and silently taking one of them would
/// record a benchmark result against a configuration nobody chose.
#[derive(Debug, Clone)]
pub struct ArgvBuilder {
    dialect: Dialect,
    flags: Vec<Flag>,
    seen: BTreeSet<&'static str>,
}

impl ArgvBuilder {
    /// Start an argv for the given dialect.
    pub fn new(dialect: Dialect) -> Self {
        Self {
            dialect,
            flags: Vec::new(),
            seen: BTreeSet::new(),
        }
    }

    /// Add a flag, rejecting duplicates.
    pub fn push(&mut self, flag: Flag) -> Result<&mut Self, EmitError> {
        let name = flag.name();
        // The boolean-pair flags share a variant but emit two different tokens; key the duplicate
        // check on the variant's canonical name so `--webui` and `--no-webui` still collide.
        if !self.seen.insert(name) {
            return Err(EmitError::Duplicate(name));
        }
        self.flags.push(flag);
        Ok(self)
    }

    /// Apply the mandatory loopback hardening set for a server spawn.
    ///
    /// Every one of these overrides an upstream default that is wrong for a privacy-first product:
    /// `--slots` defaults to enabled (exposing slot state to any local process), `--webui` defaults
    /// to enabled (a second UI Cerebro does not control), and `--cors-origins` defaults to `*`
    /// (every web origin). Cerebro never inherits a server default; it states all of them.
    pub fn harden_server(
        &mut self,
        host: &str,
        port: u16,
        api_key_file: &str,
    ) -> Result<&mut Self, EmitError> {
        debug_assert_eq!(self.dialect, Dialect::Server);
        self.push(Flag::Host(host.to_string()))?;
        self.push(Flag::Port(port))?;
        self.push(Flag::Slots(false))?;
        self.push(Flag::WebUi(false))?;
        // Empty origin list: Cerebro serves its own UI and no browser origin needs to reach the
        // backend directly.
        self.push(Flag::CorsOrigins(String::new()))?;
        self.push(Flag::ApiKeyFile(api_key_file.to_string()))?;
        Ok(self)
    }

    /// Pin reproducibility for a measured run.
    ///
    /// `--fit` is ON by default and rewrites any argument Cerebro left unset, which would mean the
    /// configuration recorded in a result is not the configuration that ran. Turning it off makes
    /// runs reproducible — and transfers OOM safety to Cerebro, which is why the planner must model
    /// VRAM itself rather than relying on upstream's fitter.
    pub fn pin_for_measurement(&mut self) -> Result<&mut Self, EmitError> {
        self.push(Flag::Fit(false))?;
        Ok(self)
    }

    /// Check cross-flag rules without emitting.
    ///
    /// Returns every violation rather than the first, so a generated configuration can be repaired
    /// in one pass rather than by iterative rejection.
    pub fn validate(&self) -> Vec<Violation> {
        crate::constraints::validate(&self.flags)
    }

    /// Emit the argv tokens, refusing any configuration that violates a hard cross-flag rule.
    ///
    /// Warnings do not block emission — the tuner prunes those from its search space, but a user
    /// may still deliberately measure one. Errors do block, because they represent either a crash
    /// or a flag that is silently inert, and an inert flag would cause a measurement to be recorded
    /// against a setting that was never active.
    pub fn build(&self) -> Result<Vec<String>, EmitError> {
        let violations = self.validate();
        if let Some(v) = violations
            .iter()
            .find(|v| v.severity == Severity::Error)
        {
            return Err(EmitError::ConstraintViolated {
                rule: v.rule,
                detail: v.detail.clone(),
                remedy: v.remedy.clone(),
            });
        }
        self.build_unchecked()
    }

    /// Emit the argv tokens without cross-flag validation.
    ///
    /// For tests and for deliberately probing a combination upstream might reject.
    pub fn build_unchecked(&self) -> Result<Vec<String>, EmitError> {
        let mut out = Vec::new();
        for flag in &self.flags {
            out.extend(flag.render(self.dialect)?);
        }
        Ok(out)
    }

    /// The flags added so far, in insertion order.
    pub fn flags(&self) -> &[Flag] {
        &self.flags
    }

    /// Which dialect this builder was constructed for.
    ///
    /// Lets a function that accepts a caller-supplied builder (e.g. a server-hardening helper)
    /// check it is being handed the right kind before mutating it, instead of relying on the
    /// `debug_assert_eq!` inside [`ArgvBuilder::harden_server`], which compiles out of a release
    /// build.
    pub fn dialect(&self) -> Dialect {
        self.dialect
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{BufferType, DeviceId, OtRule};

    fn dev(s: &str) -> DeviceId {
        DeviceId::parse(s).unwrap()
    }

    #[test]
    fn device_list_separator_differs_by_dialect() {
        let list = DeviceList::devices([dev("CUDA0"), dev("RPC0")]).unwrap();
        let f = Flag::Device(list);
        assert_eq!(
            f.render(Dialect::Server).unwrap(),
            vec!["-dev", "CUDA0,RPC0"]
        );
        // Comma here would mean two benchmark runs, not two devices.
        assert_eq!(f.render(Dialect::Bench).unwrap(), vec!["-dev", "CUDA0/RPC0"]);
    }

    #[test]
    fn ot_separator_differs_by_dialect() {
        let mut rules = OtRuleSet::new();
        rules.push(OtRule::layer(0, BufferType::Cpu));
        rules.push(OtRule::layer(1, BufferType::Cpu));
        let f = Flag::OverrideTensor(rules);
        assert_eq!(
            f.render(Dialect::Server).unwrap(),
            vec!["-ot", r"blk\.0\.=CPU,blk\.1\.=CPU"]
        );
        assert_eq!(
            f.render(Dialect::Bench).unwrap(),
            vec!["-ot", r"blk\.0\.=CPU;blk\.1\.=CPU"]
        );
    }

    #[test]
    fn tensor_split_separator_differs_by_dialect() {
        let f = Flag::TensorSplit(TensorSplit::new([0.5, 0.5]).unwrap());
        assert_eq!(f.render(Dialect::Server).unwrap(), vec!["-ts", "0.5,0.5"]);
        assert_eq!(f.render(Dialect::Bench).unwrap(), vec!["-ts", "0.5/0.5"]);
    }

    #[test]
    fn speculation_is_rejected_in_bench_dialect() {
        // llama-bench silently has no such flag; making this a hard error stops the tuner from
        // recording a "speculative" measurement that never enabled speculation.
        let f = Flag::SpecType(SpecType::NgramSimple);
        assert!(matches!(
            f.render(Dialect::Bench),
            Err(EmitError::NotInDialect { flag: "--spec-type", .. })
        ));
        assert_eq!(
            f.render(Dialect::Server).unwrap(),
            vec!["--spec-type", "ngram-simple"]
        );
    }

    #[test]
    fn none_sentinel_is_lowercase_and_alone() {
        let f = Flag::SpecDraftDevice(DeviceList::None);
        assert_eq!(f.render(Dialect::Server).unwrap(), vec!["-devd", "none"]);
    }

    #[test]
    fn negated_switches_emit_the_right_token() {
        assert_eq!(
            Flag::Slots(false).render(Dialect::Server).unwrap(),
            vec!["--no-slots"]
        );
        assert_eq!(
            Flag::WebUi(false).render(Dialect::Server).unwrap(),
            vec!["--no-webui"]
        );
        // --fit takes a value rather than a --no- form.
        assert_eq!(
            Flag::Fit(false).render(Dialect::Server).unwrap(),
            vec!["--fit", "off"]
        );
    }

    #[test]
    fn relative_model_paths_are_refused() {
        assert!(matches!(
            Flag::Model("models/x.gguf".into()).render(Dialect::Server),
            Err(EmitError::RelativePath(_))
        ));
        assert!(Flag::Model(r"C:\models\x.gguf".into())
            .render(Dialect::Server)
            .is_ok());
    }

    #[test]
    fn duplicate_flags_are_refused() {
        let mut b = ArgvBuilder::new(Dialect::Server);
        b.push(Flag::GpuLayers(999)).unwrap();
        assert!(matches!(
            b.push(Flag::GpuLayers(40)),
            Err(EmitError::Duplicate("-ngl"))
        ));
    }

    #[test]
    fn hardening_overrides_every_wrong_upstream_default() {
        let mut b = ArgvBuilder::new(Dialect::Server);
        b.harden_server("127.0.0.1", 8127, r"C:\Users\x\.cerebro\api.key")
            .unwrap();
        let argv = b.build().unwrap();
        // Upstream defaults: slots enabled, webui enabled, cors-origins "*".
        assert!(argv.contains(&"--no-slots".to_string()));
        assert!(argv.contains(&"--no-webui".to_string()));
        assert!(argv.contains(&"--cors-origins".to_string()));
        assert!(argv.contains(&"127.0.0.1".to_string()));
    }

    #[test]
    fn reference_machine_heterogeneous_plan_emits_correctly() {
        // The plan from ARCHITECTURE.md section 5: target on dGPU, MoE experts in host RAM,
        // draft model on the loopback CPU RPC device with its threads pinned to E-cores.
        let mut b = ArgvBuilder::new(Dialect::Server);
        b.push(Flag::Model(r"C:\models\target.gguf".into())).unwrap();
        b.push(Flag::GpuLayers(999)).unwrap();
        b.push(Flag::NCpuMoe(32)).unwrap();
        b.push(Flag::CacheTypeK(CacheType::Q8_0)).unwrap();
        b.push(Flag::CacheTypeV(CacheType::Q8_0)).unwrap();
        b.push(Flag::CpuMask(CpuMask::new(0x555).unwrap())).unwrap();
        b.push(Flag::CpuStrict(true)).unwrap();
        b.push(Flag::Threads(6)).unwrap();
        b.push(Flag::SpecType(SpecType::DraftSimple)).unwrap();
        b.push(Flag::DraftModel(r"C:\models\draft.gguf".into())).unwrap();
        b.push(Flag::SpecDraftDevice(
            DeviceList::devices([dev("RPC0")]).unwrap(),
        ))
        .unwrap();
        b.push(Flag::SpecDraftCpuMask(CpuMask::new(0xFF000).unwrap()))
            .unwrap();
        b.push(Flag::SpecDraftCpuStrict(true)).unwrap();
        b.pin_for_measurement().unwrap();

        let argv = b.build().unwrap();
        let joined = argv.join(" ");
        assert!(joined.contains("-ngl 999"));
        assert!(joined.contains("-ncmoe 32"));
        assert!(joined.contains("-ctk q8_0"));
        assert!(joined.contains("-C 555"));
        assert!(joined.contains("-Cd ff000"));
        assert!(joined.contains("--spec-type draft-simple"));
        assert!(joined.contains("-devd RPC0"));
        assert!(joined.contains("--fit off"));
    }
}
