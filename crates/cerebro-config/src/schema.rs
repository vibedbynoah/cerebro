//! `Settings`: the one schema-validated shape of `settings.json`.
//!
//! # Why `runtime` is made of newtypes and re-exported flag enums, not raw numbers and strings
//!
//! [`cerebro_flags`] already closed the argv space: [`cerebro_flags::Flag`] is the complete list of
//! flags Cerebro may ever emit, and its value types make an invalid value unrepresentable rather
//! than merely rejected. If [`RuntimeSection`] stored `n_gpu_layers` as a bare `u32` and
//! `cache_type_k` as a bare `String`, a config file could carry a value that is well-typed *here*
//! but meaningless once handed to `cerebro-flags` (or, worse, a string that means something to
//! `cerebro-flags` only by coincidence of spelling). So every runtime field is either a newtype with
//! a `to_flag()` that produces exactly the [`cerebro_flags::Flag`] variant it stands for
//! ([`CtxSize`], [`GpuLayerCount`], [`CpuMoeLayers`]), or the flag crate's own enum, serialized
//! through its own `as_str()`/`parse()` round trip ([`cerebro_flags::CacheType`],
//! [`cerebro_flags::FlashAttn`]). There is no field here that can hold a value `cerebro-flags`
//! cannot also hold, and no field that needs its own separate validation to stay in sync with it.
//! [`RuntimeSection::to_flags`] and [`SamplerSection::to_flags`] are the entire legal surface
//! between this crate and argv -- a caller should never assemble a [`cerebro_flags::Flag`] from a
//! raw accessor by hand; the per-field `to_flag()`s on [`CtxSize`], [`GpuLayerCount`] and
//! [`CpuMoeLayers`] exist as building blocks `RuntimeSection::to_flags` calls, not as a `pub`
//! surface meant to be composed by anything else.
//!
//! # Why there is no `allow_unsafe_dacl`
//!
//! An earlier draft of this schema carried `security.allow_unsafe_dacl: bool = false`. That key
//! would have let *any* settings source disable the refusal to spawn from a backend directory
//! writable by another principal — which is not a hypothetical on the reference machine (see
//! `docs/ARCHITECTURE.md` §8). Locking the key (see [`crate::profile::SECURITY_LOCKED`]) controls
//! *who may write* it, but [`Source::File`](crate::profile::Source::File) and
//! [`Source::Gui`](crate::profile::Source::Gui) are never locked out of it, and a poisoned
//! `settings.json` is exactly the attack this schema must refuse. So the key has no representation
//! at all: there is no field, of any type, anywhere in [`Settings`], whose value can widen a spawn
//! decision. [`crate::profile::no_settings_value_from_any_source_permits_a_spawn_from_an_unsafe_directory`]
//! is the regression test for that property.

use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize};

use crate::error::ConfigError;

/// The only settings-document version this build understands.
///
/// Named and checked, rather than merely present and typed, for the same reason
/// `cerebro_store::record::SCHEMA_VERSION` is: `#[serde(deny_unknown_fields)]` catches a future
/// version that *adds* a key, but not one that silently changes what an existing key *means* (e.g.
/// `security.model_roots` becoming glob patterns). [`Settings::validate`] refuses any document
/// whose `version` differs from this constant, before any other check runs.
pub const SCHEMA_VERSION: u32 = 1;

/// The full, schema-validated contents of `settings.json`.
///
/// `#[serde(deny_unknown_fields)]` is load-bearing, not a style preference: it is what turns a
/// config carrying a resurrected `allow_unsafe_dacl` (or any other field nobody reviewed) into a
/// typed parse failure instead of a silently-ignored key. See [`crate::file::ConfigFile::reload`]
/// for what happens to that failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// Schema version of this document. Checked against [`SCHEMA_VERSION`] by [`Settings::validate`]
    /// before anything else: a document declaring any other version is refused outright rather than
    /// interpreted under an assumption about its shape that may no longer hold.
    pub version: u32,
    /// Where Cerebro looks for models, and which one loads by default.
    pub models: ModelsSection,
    /// Where the llama.cpp backend build lives, and its integrity manifest.
    pub backend: BackendSection,
    /// The offload/KV/context settings that map directly onto `cerebro-flags`.
    pub runtime: RuntimeSection,
    /// Sampler entropy settings. See `docs/ARCHITECTURE.md` §4 (Axis 0) for why these are not mere
    /// preferences: they are a precondition for every speculative-decoding measurement.
    pub sampler: SamplerSection,
    /// Cerebro's own local HTTP API.
    pub api: ApiSection,
    /// Display-only preferences with no security relevance.
    pub ui: UiSection,
    /// The security-locked settings: approved model roots, LAN exposure, and the backend path used
    /// for the pre-spawn integrity/DACL audit.
    pub security: SecuritySection,
}

impl Settings {
    /// Structural consistency checks beyond what `serde` alone expresses.
    ///
    /// This is the crate's invariant hub: every path that produces a `Settings` from bytes or from
    /// a patch ends here -- [`crate::file::ConfigFile`]'s load path and the tail of
    /// [`Settings::apply_patch`](crate::profile) both call it, so any future cross-field or
    /// document-level rule belongs here too. It stays filesystem-free by design; everything that
    /// touches the filesystem is [`crate::profile::DefaultPreflighter::decide`]'s job instead.
    ///
    /// Checks, in order (house rule S6: fail on the first thing that does not hold):
    ///
    /// 1. [`Settings::version`] equals [`SCHEMA_VERSION`] -- a document from a version this build
    ///    does not understand is refused outright, never interpreted under a shape assumption that
    ///    may no longer hold.
    /// 2. [`BackendSection::dir`] and [`SecuritySection::backend_path`] name the literal same path.
    ///    This is the cheap, filesystem-free half of that guarantee, so the two keys can never
    ///    drift apart in a document that is merely loaded or patched, not spawned from. The
    ///    expensive, filesystem-aware half -- that they *canonicalize* to the same real directory,
    ///    which is what actually matters at spawn time -- is
    ///    [`crate::profile::DefaultPreflighter::decide`]'s job, and it re-checks this independently
    ///    rather than trusting that `validate` already ran.
    ///
    /// # Errors
    /// [`ConfigError::UnsupportedVersion`] or [`ConfigError::BackendPathDisagreement`].
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.version != SCHEMA_VERSION {
            return Err(ConfigError::UnsupportedVersion {
                found: self.version,
                supported: SCHEMA_VERSION,
            });
        }
        if self.backend.dir != self.security.backend_path {
            return Err(ConfigError::BackendPathDisagreement {
                backend_dir: self.backend.dir.clone(),
                security_backend_path: self.security.backend_path.clone(),
            });
        }
        Ok(())
    }
}

impl Default for Settings {
    /// A settings document that is valid on its own terms but trusts nothing: no approved model
    /// root, LAN mode off, the local API off. A fresh install must fail closed, not open.
    fn default() -> Self {
        Self {
            version: 1,
            models: ModelsSection::default(),
            backend: BackendSection::default(),
            runtime: RuntimeSection::default(),
            sampler: SamplerSection::default(),
            api: ApiSection::default(),
            ui: UiSection::default(),
            security: SecuritySection::default(),
        }
    }
}

/// Where Cerebro discovers models for the UI to list, distinct from
/// [`SecuritySection::model_roots`], which is the enforced boundary
/// [`crate::profile::resolve_model_path`] checks against. A discovery root is a convenience; a
/// security root is a promise, and only [`Source::File`](crate::profile::Source::File) or
/// [`Source::Gui`](crate::profile::Source::Gui) may move the latter (see
/// [`crate::profile::SECURITY_LOCKED`]).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsSection {
    /// Directories to scan when listing available models.
    pub roots: Vec<PathBuf>,
    /// Alias of the model to load when none is specified. `None` until the user picks one.
    #[serde(default)]
    pub default_alias: Option<String>,
}

/// The llama.cpp build Cerebro spawns.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendSection {
    /// Directory containing the backend executables and `ggml-*` DLLs.
    ///
    /// Security-locked (see [`crate::profile::SECURITY_LOCKED`]): ggml loads backends as DLLs from
    /// this directory at spawn time, so a live source (the API) redirecting it would be a DLL-plant
    /// vector, not a preference change.
    pub dir: PathBuf,
    /// Path to the SHA-256 manifest checked before every spawn.
    pub manifest: PathBuf,
}

/// Context length in tokens, mapped 1:1 onto llama.cpp's `-c` / [`cerebro_flags::Flag::CtxSize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CtxSize(pub u32);

impl CtxSize {
    /// The flag this value maps onto.
    pub fn to_flag(self) -> cerebro_flags::Flag {
        cerebro_flags::Flag::CtxSize(self.0)
    }
}

/// GPU offload depth, mapped 1:1 onto `-ngl` / [`cerebro_flags::Flag::GpuLayers`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GpuLayerCount(pub u32);

impl GpuLayerCount {
    /// The flag this value maps onto.
    pub fn to_flag(self) -> cerebro_flags::Flag {
        cerebro_flags::Flag::GpuLayers(self.0)
    }
}

/// MoE expert host-memory depth, mapped 1:1 onto `-ncmoe` / [`cerebro_flags::Flag::NCpuMoe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CpuMoeLayers(pub u32);

impl CpuMoeLayers {
    /// The flag this value maps onto.
    pub fn to_flag(self) -> cerebro_flags::Flag {
        cerebro_flags::Flag::NCpuMoe(self.0)
    }
}

/// Offload, KV, and context settings that map directly onto `cerebro-flags` values.
///
/// See the module documentation above for why every field here is a newtype or a re-exported flag
/// enum rather than a bare number or string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSection {
    /// `-c` context size.
    pub n_ctx: CtxSize,
    /// `-ngl` GPU offload depth.
    pub n_gpu_layers: GpuLayerCount,
    /// `-ncmoe` MoE-expert host-memory depth.
    pub n_cpu_moe: CpuMoeLayers,
    /// `-ctk` K cache element type.
    #[serde(with = "cache_type_serde")]
    pub cache_type_k: cerebro_flags::CacheType,
    /// `-ctv` V cache element type.
    #[serde(with = "cache_type_serde")]
    pub cache_type_v: cerebro_flags::CacheType,
    /// `-fa` flash attention mode.
    #[serde(with = "flash_attn_serde")]
    pub flash_attn: cerebro_flags::FlashAttn,
}

impl RuntimeSection {
    /// The six [`cerebro_flags::Flag`] entries this section stands for, in emission order.
    ///
    /// The one sanctioned mapping from this section to argv, for the same reason
    /// [`SamplerSection::to_flags`] is: a caller assembling `Flag::CacheTypeK`/`Flag::CacheTypeV`
    /// by hand from [`RuntimeSection::cache_type_k`] and [`RuntimeSection::cache_type_v`] can
    /// transpose K and V and nothing would fail to compile or to run -- calling this instead makes
    /// that mistake structurally impossible.
    pub fn to_flags(&self) -> [cerebro_flags::Flag; 6] {
        [
            self.n_ctx.to_flag(),
            self.n_gpu_layers.to_flag(),
            self.n_cpu_moe.to_flag(),
            cerebro_flags::Flag::CacheTypeK(self.cache_type_k),
            cerebro_flags::Flag::CacheTypeV(self.cache_type_v),
            cerebro_flags::Flag::FlashAttn(self.flash_attn),
        ]
    }
}

impl Default for RuntimeSection {
    /// Upstream's own defaults where cerebro-flags has one (`f16` KV, `-fa auto`), zero offload
    /// otherwise -- a fresh install must not assume the machine has a GPU worth using yet.
    fn default() -> Self {
        Self {
            n_ctx: CtxSize(4096),
            n_gpu_layers: GpuLayerCount(0),
            n_cpu_moe: CpuMoeLayers(0),
            cache_type_k: cerebro_flags::CacheType::F16,
            cache_type_v: cerebro_flags::CacheType::F16,
            flash_attn: cerebro_flags::FlashAttn::Auto,
        }
    }
}

/// Sampler entropy. Not user taste: a correctness precondition for every speculative-decoding
/// measurement (`docs/ARCHITECTURE.md` §4, Axis 0).
///
/// Fields are private and only constructible through [`SamplerSection::new`] (or, for `serde`,
/// through the hand-written [`Deserialize`] impl below that routes through the same constructor):
/// `cerebro_store::SamplerProfile::new` -- the type that actually records a measurement's sampler
/// settings -- rejects a non-finite or negative `temp` and a `top_p`/`min_p` outside `0.0..=1.0`,
/// with four tests pinning exactly those bounds. A `Settings` this crate accepts but the results
/// store then refuses would produce a run that is measured but unloggable, so the same bounds are
/// enforced here, at construction, rather than left to be discovered later. `f64` (not `f32`) to
/// match `SamplerProfile`'s field width exactly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SamplerSection {
    temp: f64,
    top_k: u32,
    top_p: f64,
    min_p: f64,
}

fn default_min_p() -> f64 {
    0.05
}

impl SamplerSection {
    /// Upper bound on `temp`: [`cerebro_flags::Flag::Temp`] is an `f32`, so any larger value is
    /// unrepresentable downstream and would silently saturate to `inf` in an argv token rather than
    /// fail loudly. Checked here, at construction, so the bound lives next to the field it protects
    /// rather than at the one call site that would otherwise need to remember `as f32` can saturate.
    pub const MAX_TEMP: f64 = f32::MAX as f64;

    /// Build a sampler section, validating that every field is a value llama.cpp -- and
    /// `cerebro_store::SamplerProfile`, which records the run -- could actually accept.
    ///
    /// # Errors
    /// [`ConfigError::SamplerOutOfRange`] if `temp` is not finite or outside
    /// `0.0..=`[`Self::MAX_TEMP`], or if `top_p` or `min_p` is not finite or outside `0.0..=1.0`.
    pub fn new(temp: f64, top_k: u32, top_p: f64, min_p: f64) -> Result<Self, ConfigError> {
        if !(temp.is_finite() && (0.0..=Self::MAX_TEMP).contains(&temp)) {
            return Err(ConfigError::SamplerOutOfRange { field: "temp", value: temp });
        }
        for (field, value) in [("top_p", top_p), ("min_p", min_p)] {
            if !(value.is_finite() && (0.0..=1.0).contains(&value)) {
                return Err(ConfigError::SamplerOutOfRange { field, value });
            }
        }
        Ok(Self { temp, top_k, top_p, min_p })
    }

    /// `--temp` sampling temperature.
    pub fn temp(&self) -> f64 {
        self.temp
    }

    /// `--top-k` truncation.
    pub fn top_k(&self) -> u32 {
        self.top_k
    }

    /// `--top-p` nucleus truncation.
    pub fn top_p(&self) -> f64 {
        self.top_p
    }

    /// `--min-p` truncation. Defaults to upstream's own default of `0.05`, which
    /// `docs/ARCHITECTURE.md` §2 is explicit Cerebro may not claim credit for.
    pub fn min_p(&self) -> f64 {
        self.min_p
    }

    /// The four [`cerebro_flags::Flag`] entries this section stands for, in emission order.
    ///
    /// This is the crate's only sanctioned `f64` -> `f32` narrowing: [`Self::new`] has already
    /// refused every value outside `f32`'s finite range (see [`Self::MAX_TEMP`] for `temp`; `top_p`
    /// and `min_p` are bounded to `0.0..=1.0`, well inside it), so none of these four casts can ever
    /// produce `inf` or `NaN`. No caller should assemble these flags by hand from the accessors
    /// above -- this is the one place the mapping is written.
    pub fn to_flags(&self) -> [cerebro_flags::Flag; 4] {
        [
            cerebro_flags::Flag::Temp(self.temp as f32),
            cerebro_flags::Flag::TopK(self.top_k),
            cerebro_flags::Flag::TopP(self.top_p as f32),
            cerebro_flags::Flag::MinP(self.min_p as f32),
        ]
    }
}

impl Default for SamplerSection {
    fn default() -> Self {
        // Direct construction, not `new()`: these literals are known-valid at compile time (all
        // within bounds), so a fallible call here would only add a dead error branch that "no
        // unwrap/expect/panic outside tests" would then force into an `unreachable!()`. `Default`
        // is defined in the same module as the private fields it sets, which is what makes this
        // legal without weakening the constructor's guarantee for every other caller.
        Self {
            temp: 0.80,
            top_k: 40,
            top_p: 0.95,
            min_p: default_min_p(),
        }
    }
}

impl<'de> Deserialize<'de> for SamplerSection {
    /// Deserializes through [`SamplerSection::new`] so a document can never construct an
    /// out-of-range sampler section -- there is no field-by-field derived path that would bypass
    /// the checks `new` performs.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // `deny_unknown_fields` lives on this private `Raw` shape, not on `SamplerSection` itself
        // (which has no derived `Deserialize` to attach it to) -- otherwise a poisoned document
        // could add an unknown key here for free while the rest of the schema still refuses it.
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            temp: f64,
            top_k: u32,
            top_p: f64,
            #[serde(default = "default_min_p")]
            min_p: f64,
        }
        let raw = Raw::deserialize(deserializer)?;
        SamplerSection::new(raw.temp, raw.top_k, raw.top_p, raw.min_p).map_err(serde::de::Error::custom)
    }
}

/// Bind address for Cerebro's own local HTTP API.
///
/// Exactly two variants, both loopback. There is no representation for `0.0.0.0`, a LAN interface
/// address, or any other host: an operator who wants LAN exposure sets
/// [`SecuritySection::lan_mode`] (security-locked, see [`crate::profile::SECURITY_LOCKED`]), which
/// a future surface may interpret as permission to run a *separate*, explicitly-audited listener --
/// but that is a decision this type cannot make silently, because it cannot express it at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindAddr {
    /// `127.0.0.1`.
    #[default]
    Loopback4,
    /// `::1`.
    Loopback6,
}

impl BindAddr {
    /// The literal host string a server surface should bind, e.g. for `cerebro_flags::Flag::Host`.
    pub fn host_str(self) -> &'static str {
        match self {
            BindAddr::Loopback4 => "127.0.0.1",
            BindAddr::Loopback6 => "::1",
        }
    }
}

/// Cerebro's own local HTTP API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiSection {
    /// Whether the API listens at all. Defaults to off: a fresh install exposes nothing.
    #[serde(default)]
    pub enabled: bool,
    /// TCP port to bind.
    pub port: u16,
    /// Which loopback address to bind. Security-locked (see
    /// [`crate::profile::SECURITY_LOCKED`]): a live source (the API itself) redirecting its own
    /// bind address is not a preference change.
    pub bind: BindAddr,
}

impl Default for ApiSection {
    fn default() -> Self {
        Self {
            enabled: false,
            port: 8127,
            bind: BindAddr::default(),
        }
    }
}

/// UI density preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Density {
    /// More whitespace between elements.
    #[default]
    Comfortable,
    /// Less whitespace between elements.
    Compact,
}

/// Display-only preferences. None of these fields participate in any security or spawn decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UiSection {
    /// Base theme identifier, e.g. `"dark"`.
    pub theme_base: String,
    /// Accent color identifier, e.g. `"violet"`.
    pub theme_accent: String,
    /// Contrast level, 0-255.
    pub theme_contrast: u8,
    /// Layout density.
    pub density: Density,
    /// Whether to suppress non-essential animation.
    pub reduce_motion: bool,
}

impl Default for UiSection {
    fn default() -> Self {
        Self {
            theme_base: "dark".to_string(),
            theme_accent: "blue".to_string(),
            theme_contrast: 100,
            density: Density::default(),
            reduce_motion: false,
        }
    }
}

/// The security-locked settings.
///
/// Every field here, plus everything under `/backend` and `/api/bind`, is in
/// [`crate::profile::SECURITY_LOCKED`]: [`Source::File`](crate::profile::Source::File) and
/// [`Source::Gui`](crate::profile::Source::Gui) may set them, [`Source::Api`](
/// crate::profile::Source::Api) may not, ever, regardless of which pointer or value is asked for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecuritySection {
    /// The only roots [`crate::profile::resolve_model_path`] will resolve a path under.
    pub model_roots: Vec<PathBuf>,
    /// Whether Cerebro's API may be reached from beyond loopback. Off by default: a fresh install
    /// exposes nothing to the LAN.
    #[serde(default)]
    pub lan_mode: bool,
    /// The backend directory as audited for DACL/integrity purposes before every spawn. See
    /// [`crate::profile::Preflighter`].
    pub backend_path: PathBuf,
}

impl Default for SecuritySection {
    fn default() -> Self {
        Self {
            model_roots: Vec::new(),
            lan_mode: false,
            backend_path: PathBuf::new(),
        }
    }
}

/// Bridges [`cerebro_flags::CacheType`] through its own `as_str()`/`parse()` for serde, rather than
/// deriving `Serialize`/`Deserialize` on it (it lives in a different crate, and cerebro-flags's own
/// `serde` feature is unused/reserved -- see its `Cargo.toml`). Using the flag crate's own string
/// form, rather than re-deriving one here, means the two can never drift out of sync with each
/// other.
mod cache_type_serde {
    use cerebro_flags::CacheType;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S: Serializer>(value: &CacheType, s: S) -> Result<S::Ok, S::Error> {
        value.as_str().serialize(s)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<CacheType, D::Error> {
        let raw = String::deserialize(d)?;
        CacheType::parse(&raw).map_err(serde::de::Error::custom)
    }
}

/// See [`cache_type_serde`]; the same bridge for [`cerebro_flags::FlashAttn`].
mod flash_attn_serde {
    use cerebro_flags::FlashAttn;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S: Serializer>(value: &FlashAttn, s: S) -> Result<S::Ok, S::Error> {
        value.as_str().serialize(s)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<FlashAttn, D::Error> {
        let raw = String::deserialize(d)?;
        FlashAttn::parse(&raw).map_err(serde::de::Error::custom)
    }
}

/// A JSON Schema (2020-12) description of [`Settings`].
///
/// Written to `docs/schema/settings.schema.json` (see the `schema_file_matches_generated_schema`
/// test below); kept in sync with the struct definitions by
/// `schema_covers_every_key_of_a_fully_populated_settings`, which walks a fully-populated
/// `Settings` value and fails if any serialized key has no matching schema node.
pub fn json_schema() -> serde_json::Value {
    serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Cerebro Settings",
        "type": "object",
        "additionalProperties": false,
        "required": ["version", "models", "backend", "runtime", "sampler", "api", "ui", "security"],
        "properties": {
            "version": { "type": "integer", "minimum": 0, "maximum": u64::from(u32::MAX), "const": SCHEMA_VERSION },
            "models": models_schema(),
            "backend": backend_schema(),
            "runtime": runtime_schema(),
            "sampler": sampler_schema(),
            "api": api_schema(),
            "ui": ui_schema(),
            "security": security_schema(),
        }
    })
}

fn models_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["roots"],
        "properties": {
            "roots": { "type": "array", "items": { "type": "string" } },
            "default_alias": { "type": ["string", "null"] },
        }
    })
}

fn backend_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["dir", "manifest"],
        "properties": {
            "dir": { "type": "string" },
            "manifest": { "type": "string" },
        }
    })
}

fn cache_type_enum_schema() -> serde_json::Value {
    let values: Vec<&str> = cerebro_flags::CacheType::ALL.iter().map(|v| v.as_str()).collect();
    serde_json::json!({ "type": "string", "enum": values })
}

fn flash_attn_enum_schema() -> serde_json::Value {
    let values: Vec<&str> = cerebro_flags::FlashAttn::ALL.iter().map(|v| v.as_str()).collect();
    serde_json::json!({ "type": "string", "enum": values })
}

fn runtime_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["n_ctx", "n_gpu_layers", "n_cpu_moe", "cache_type_k", "cache_type_v", "flash_attn"],
        "properties": {
            "n_ctx": { "type": "integer", "minimum": 0, "maximum": u64::from(u32::MAX) },
            "n_gpu_layers": { "type": "integer", "minimum": 0, "maximum": u64::from(u32::MAX) },
            "n_cpu_moe": { "type": "integer", "minimum": 0, "maximum": u64::from(u32::MAX) },
            "cache_type_k": cache_type_enum_schema(),
            "cache_type_v": cache_type_enum_schema(),
            "flash_attn": flash_attn_enum_schema(),
        }
    })
}

fn sampler_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["temp", "top_k", "top_p"],
        "properties": {
            // Bounds mirror `SamplerSection::new` exactly (see its doc comment): a document the
            // schema accepts must also be a document the constructor accepts.
            "temp": { "type": "number", "minimum": 0.0, "maximum": SamplerSection::MAX_TEMP },
            "top_k": { "type": "integer", "minimum": 0, "maximum": u64::from(u32::MAX) },
            "top_p": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
            "min_p": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.05 },
        }
    })
}

fn api_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["port", "bind"],
        "properties": {
            "enabled": { "type": "boolean", "default": false },
            "port": { "type": "integer", "minimum": 0, "maximum": 65535 },
            "bind": { "type": "string", "enum": ["loopback4", "loopback6"] },
        }
    })
}

fn ui_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["theme_base", "theme_accent", "theme_contrast", "density", "reduce_motion"],
        "properties": {
            "theme_base": { "type": "string" },
            "theme_accent": { "type": "string" },
            "theme_contrast": { "type": "integer", "minimum": 0, "maximum": 255 },
            "density": { "type": "string", "enum": ["comfortable", "compact"] },
            "reduce_motion": { "type": "boolean" },
        }
    })
}

fn security_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["model_roots", "backend_path"],
        "properties": {
            "model_roots": { "type": "array", "items": { "type": "string" } },
            "lan_mode": { "type": "boolean", "default": false },
            "backend_path": { "type": "string" },
        }
    })
}

/// A `Settings` value with every field set to a distinctive, non-default value.
///
/// Used by the round-trip test (which must prove every field actually survives serialization, not
/// just that defaults survive) and by the schema-coverage test (which must see every key at least
/// once, including the optional ones).
#[cfg(test)]
pub(crate) fn full_settings_fixture() -> Settings {
    Settings {
        version: SCHEMA_VERSION,
        models: ModelsSection {
            roots: vec![PathBuf::from(r"C:\models"), PathBuf::from(r"D:\more-models")],
            default_alias: Some("bastion-control-0.5b".to_string()),
        },
        backend: BackendSection {
            dir: PathBuf::from(r"C:\Users\noahm\Projects\qwen-flash-next\llama.cpp-b10751"),
            manifest: PathBuf::from(r"C:\Users\noahm\.cerebro\backend-manifest.json"),
        },
        runtime: RuntimeSection {
            n_ctx: CtxSize(8192),
            n_gpu_layers: GpuLayerCount(999),
            n_cpu_moe: CpuMoeLayers(32),
            cache_type_k: cerebro_flags::CacheType::Q8_0,
            cache_type_v: cerebro_flags::CacheType::Q8_0,
            flash_attn: cerebro_flags::FlashAttn::On,
        },
        sampler: SamplerSection::new(0.7, 64, 0.9, 0.05).expect("fixture sampler values are in range"),
        api: ApiSection {
            enabled: true,
            port: 8127,
            bind: BindAddr::Loopback6,
        },
        ui: UiSection {
            theme_base: "dark".to_string(),
            theme_accent: "violet".to_string(),
            theme_contrast: 90,
            density: Density::Compact,
            reduce_motion: true,
        },
        security: SecuritySection {
            model_roots: vec![PathBuf::from(r"C:\models")],
            lan_mode: false,
            backend_path: PathBuf::from(r"C:\Users\noahm\Projects\qwen-flash-next\llama.cpp-b10751"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newtypes_map_onto_the_flags_they_stand_for() {
        assert_eq!(CtxSize(4096).to_flag(), cerebro_flags::Flag::CtxSize(4096));
        assert_eq!(GpuLayerCount(999).to_flag(), cerebro_flags::Flag::GpuLayers(999));
        assert_eq!(CpuMoeLayers(32).to_flag(), cerebro_flags::Flag::NCpuMoe(32));
    }

    #[test]
    fn default_settings_is_valid_and_trusts_nothing() {
        let settings = Settings::default();
        assert!(settings.security.model_roots.is_empty());
        assert!(!settings.security.lan_mode);
        assert!(!settings.api.enabled);
        assert_eq!(settings.sampler.min_p, 0.05);
    }

    #[test]
    fn every_schema_key_round_trips_through_json() {
        let original = full_settings_fixture();
        let text = serde_json::to_string(&original).expect("fixture serializes");
        let restored: Settings = serde_json::from_str(&text).expect("fixture round-trips");
        assert_eq!(original, restored);
    }

    #[test]
    fn min_p_defaults_when_absent_from_the_document() {
        let mut value = serde_json::to_value(full_settings_fixture()).expect("serializes");
        value["sampler"]
            .as_object_mut()
            .expect("sampler is an object")
            .remove("min_p");
        let settings: Settings = serde_json::from_value(value).expect("min_p is optional");
        assert_eq!(settings.sampler.min_p, 0.05);
    }

    #[test]
    fn cache_type_and_flash_attn_use_the_flag_crate_string_form() {
        let value = serde_json::to_value(full_settings_fixture()).expect("serializes");
        assert_eq!(value["runtime"]["cache_type_k"], "q8_0");
        assert_eq!(value["runtime"]["flash_attn"], "on");
    }

    #[test]
    fn sampler_section_constructor_rejects_values_cerebro_store_would_reject() {
        // Same table as `cerebro_store::record::sampler_profile_rejects_*`: a value this
        // constructor accepts must always be one `SamplerProfile::new` also accepts, so the two
        // are tested against the identical boundary values here.
        let cases: [(&str, f64, f64, f64, f64); 5] = [
            ("negative temp", -0.1, 40.0, 0.9, 0.05),
            ("non-finite temp", f64::NAN, 40.0, 0.9, 0.05),
            ("top_p above 1.0", 0.7, 40.0, 1.5, 0.05),
            ("min_p below 0.0", 0.7, 40.0, 0.9, -0.01),
            ("non-finite min_p", 0.7, 40.0, 0.9, f64::INFINITY),
        ];
        for (label, temp, top_k, top_p, min_p) in cases {
            let err = SamplerSection::new(temp, top_k as u32, top_p, min_p).unwrap_err();
            assert!(matches!(err, ConfigError::SamplerOutOfRange { .. }), "{label}: got {err:?}");
        }
    }

    #[test]
    fn sampler_flags_are_never_infinite_at_the_representable_boundary() {
        // The major finding this test would have caught: `temp` far beyond `f32::MAX` must be
        // refused at construction, not silently saturated to `inf` inside `to_flags()`.
        let err = SamplerSection::new(1e300, 40, 0.9, 0.05).unwrap_err();
        assert!(
            matches!(&err, ConfigError::SamplerOutOfRange { field, .. } if *field == "temp"),
            "got {err:?}"
        );

        // Exactly at the representable boundary must succeed, and every emitted flag must format
        // to something finite -- `cerebro_flags::emit` formats these with `format!("{p}")`.
        let at_boundary = SamplerSection::new(f64::from(f32::MAX), 40, 0.9, 0.05)
            .expect("f32::MAX is in range");
        for flag in at_boundary.to_flags() {
            let formatted = match flag {
                cerebro_flags::Flag::Temp(v) | cerebro_flags::Flag::TopP(v) | cerebro_flags::Flag::MinP(v) => {
                    format!("{v}")
                }
                cerebro_flags::Flag::TopK(v) => format!("{v}"),
                other => panic!("unexpected flag {other:?}"),
            };
            assert!(!formatted.to_ascii_lowercase().contains("inf"), "flag {flag:?} formatted as {formatted}");
            assert!(!formatted.to_ascii_lowercase().contains("nan"), "flag {flag:?} formatted as {formatted}");
        }

        // A hand-written document carrying a temp beyond the boundary must fail to deserialize at
        // all, with "temp" visible in the error.
        let mut value = serde_json::to_value(full_settings_fixture()).expect("serializes");
        value["sampler"]["temp"] = serde_json::json!(1e300);
        let err = serde_json::from_value::<Settings>(value).unwrap_err();
        assert!(err.to_string().contains("temp"), "error: {err}");
    }

    #[test]
    fn schema_temp_bound_matches_the_flag_width() {
        // Pins the advertised JSON Schema maximum to the same constant `SamplerSection::new`
        // enforces, so the two cannot drift apart the way they did before this bound existed.
        let schema = json_schema();
        let advertised = schema["properties"]["sampler"]["properties"]["temp"]["maximum"]
            .as_f64()
            .expect("temp has a numeric maximum");
        assert_eq!(advertised, SamplerSection::MAX_TEMP);
    }

    #[test]
    fn a_hand_written_settings_json_with_an_out_of_range_sampler_field_is_refused() {
        let mut value = serde_json::to_value(full_settings_fixture()).expect("serializes");
        value["sampler"]["top_p"] = serde_json::json!(7.5);
        let err = serde_json::from_value::<Settings>(value).unwrap_err();
        // The hand-written `Deserialize` routes through `SamplerSection::new`, whose error
        // `Display`s as "sampler.top_p = 7.5 is out of range" via `serde::de::Error::custom`, so
        // the out-of-range value is still visible in a parse failure, not just a silent clamp.
        assert!(err.to_string().contains("top_p"), "error: {err}");
    }

    #[test]
    fn a_sampler_from_settings_is_always_recordable() {
        // Pins `SamplerSection`'s bounds to `cerebro_store::SamplerProfile`'s: if either crate's
        // bounds ever drift, this fails rather than producing a run whose sampler the results
        // store then refuses to log.
        let sampler = full_settings_fixture().sampler;
        let recordable = cerebro_store::SamplerProfile::new(sampler.temp(), sampler.top_k(), sampler.top_p(), sampler.min_p());
        assert!(recordable.is_ok(), "a Settings-constructible sampler must always be recordable: {recordable:?}");
    }

    #[test]
    fn runtime_section_maps_onto_exactly_the_flags_it_stands_for() {
        let runtime = full_settings_fixture().runtime;
        assert_eq!(
            runtime.to_flags(),
            [
                cerebro_flags::Flag::CtxSize(8192),
                cerebro_flags::Flag::GpuLayers(999),
                cerebro_flags::Flag::NCpuMoe(32),
                cerebro_flags::Flag::CacheTypeK(cerebro_flags::CacheType::Q8_0),
                cerebro_flags::Flag::CacheTypeV(cerebro_flags::CacheType::Q8_0),
                cerebro_flags::Flag::FlashAttn(cerebro_flags::FlashAttn::On),
            ]
        );
    }

    #[test]
    fn every_integer_node_in_the_schema_is_bounded_on_both_sides() {
        // An unbounded "integer" in the published contract is a document an external validator
        // calls valid and this crate then refuses to load -- the schema file is the API, not a
        // comment next to it.
        fn walk(node: &serde_json::Value, path: &str) {
            if node.get("type") == Some(&serde_json::json!("integer")) {
                assert!(node.get("minimum").is_some(), "{path} has no minimum");
                assert!(node.get("maximum").is_some(), "{path} has no maximum");
            }
            if let Some(props) = node.get("properties").and_then(|p| p.as_object()) {
                for (key, child) in props {
                    walk(child, &format!("{path}/{key}"));
                }
            }
        }
        walk(&json_schema(), "");
    }

    #[test]
    fn a_document_declaring_an_unsupported_schema_version_is_refused() {
        let mut settings = full_settings_fixture();
        settings.version = SCHEMA_VERSION + 1;
        let err = settings.validate().unwrap_err();
        assert_eq!(
            err,
            ConfigError::UnsupportedVersion { found: SCHEMA_VERSION + 1, supported: SCHEMA_VERSION }
        );
    }

    #[test]
    fn settings_with_disagreeing_backend_paths_is_refused() {
        let mut settings = full_settings_fixture();
        settings.security.backend_path = PathBuf::from(r"C:\somewhere\else");
        let err = settings.validate().unwrap_err();
        assert!(matches!(err, ConfigError::BackendPathDisagreement { .. }), "got {err:?}");
    }

    fn assert_schema_covers(value: &serde_json::Value, schema: &serde_json::Value, path: &str) {
        match value {
            serde_json::Value::Object(map) => {
                let properties = schema
                    .get("properties")
                    .and_then(|p| p.as_object())
                    .unwrap_or_else(|| panic!("schema at {path:?} has no \"properties\" object"));
                for (key, child_value) in map {
                    let child_path = format!("{path}/{key}");
                    let child_schema = properties.get(key).unwrap_or_else(|| {
                        panic!("schema at {path:?} does not describe key {key:?}; add it to json_schema()")
                    });
                    assert_schema_covers(child_value, child_schema, &child_path);
                }
            }
            serde_json::Value::Array(items) => {
                let item_schema = schema.get("items");
                for (index, item) in items.iter().enumerate() {
                    let item_schema = item_schema
                        .unwrap_or_else(|| panic!("schema at {path:?} has no \"items\" for its array value"));
                    assert_schema_covers(item, item_schema, &format!("{path}/{index}"));
                }
            }
            _ => {
                // Scalars only need a schema node to exist. This walk's job is to catch a field
                // added to `Settings` with no matching schema entry, not to re-implement a general
                // JSON Schema validator.
                assert!(!schema.is_null(), "schema at {path:?} is missing for value {value:?}");
            }
        }
    }

    #[test]
    fn schema_covers_every_key_of_a_fully_populated_settings() {
        let value = serde_json::to_value(full_settings_fixture()).expect("fixture serializes");
        assert_schema_covers(&value, &json_schema(), "");
    }

    #[test]
    fn schema_file_matches_generated_schema() {
        let generated = serde_json::to_string_pretty(&json_schema()).expect("schema serializes") + "\n";
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/schema/settings.schema.json");

        // Regenerating is opt-in via an env var so an ordinary `cargo test` never rewrites a
        // checked-in file as a side effect; it only ever compares.
        if std::env::var_os("CEREBRO_UPDATE_SCHEMA").is_some() {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("create docs/schema");
            }
            std::fs::write(&path, &generated).expect("write settings.schema.json");
        }

        let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{} missing or unreadable ({e}); run `CEREBRO_UPDATE_SCHEMA=1 cargo test -p cerebro-config schema_file_matches_generated_schema` to generate it",
                path.display()
            )
        });
        assert_eq!(
            committed, generated,
            "docs/schema/settings.schema.json is stale; regenerate with \
             `CEREBRO_UPDATE_SCHEMA=1 cargo test -p cerebro-config schema_file_matches_generated_schema`"
        );
    }
}
