//! `Settings`: the one schema-validated shape of `settings.json`.
//!
//! # Sections
//!
//! | Key | Type | What it holds |
//! |---|---|---|
//! | `models` | [`ModelsSection`] | Model folders (scanned, and the only roots a model path may resolve under) and the default model |
//! | `backend` | [`BackendSection`] | The llama.cpp build directory and its SHA-256 manifest |
//! | `load` | [`crate::load::LoadSection`] | Default load parameters and named presets; unset means llama.cpp's own automatic value |
//! | `sampler` | [`SamplerSection`] | Sampler entropy |
//! | `effort` | [`crate::assistant::EffortSection`] | Effort dial: default level, per-level instructions and budgets |
//! | `verbosity` | [`crate::assistant::VerbositySection`] | Verbosity dial: Brief by default, editable instructions |
//! | `api` | [`ApiSection`] | Cerebro's local HTTP API |
//! | `ui` | [`UiSection`] | Display preferences |
//! | `network` | [`crate::policy::NetworkSection`] | Offline / on-request / online, and LAN exposure |
//! | `protections` | [`crate::policy::ProtectionsSection`] | Every user-switchable protection, all on by default |
//! | `plugins` | [`crate::policy::PluginsSection`] | Plugin permission defaults |
//!
//! Only `version` is required; a missing section takes its defaults, and a missing protection key
//! means that protection is on. `#[serde(deny_unknown_fields)]` on every section turns a typo (or
//! a key nobody reviewed) into a typed parse failure instead of a silently ignored setting.
//!
//! # Protections are settings, not unrepresentable values
//!
//! An earlier draft had no way at all to express "launch from a backend folder others can write",
//! on the theory that a poisoned `settings.json` must not be able to widen a spawn decision. The
//! owner overruled that: every protection is the user's to switch off, safe by default, with one
//! neutral recorded note. The folder audit is therefore
//! [`crate::policy::ProtectionsSection::audit_folder_permissions`], enforced by
//! `cerebro-supervisor`'s preflight like every other launch check ([`Settings::launch_spec`]).
//! Who may write it is still restricted: the local HTTP API may not (see
//! [`crate::profile::SECURITY_LOCKED`]).

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize};

use crate::assistant::{EffortSection, VerbositySection};
use crate::error::ConfigError;
use crate::load::LoadSection;
use crate::policy::{NetworkSection, PluginsSection, ProtectionsSection};

/// The only settings-document version this build understands.
///
/// Named and checked, rather than merely present and typed, for the same reason
/// `cerebro_store::record::SCHEMA_VERSION` is: `#[serde(deny_unknown_fields)]` catches a future
/// version that *adds* a key, but not one that silently changes what an existing key *means*.
/// [`Settings::validate`] refuses any document whose `version` differs from this constant, before
/// any other check runs. Version 2 replaced version 1's `security` section with `protections`,
/// `network` and `models.folders`, and its `runtime` section with `load`.
pub const SCHEMA_VERSION: u32 = 2;

/// `llama-server`'s file name inside [`BackendSection::dir`].
pub const SERVER_TOOL_NAME: &str = "llama-server.exe";

/// Most model folders one document may list.
pub const MAX_MODEL_FOLDERS: usize = 256;

/// The full, schema-validated contents of `settings.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// Schema version of this document; must equal [`SCHEMA_VERSION`].
    pub version: u32,
    /// Model folders and the default model.
    #[serde(default)]
    pub models: ModelsSection,
    /// Where the llama.cpp backend build lives, and its integrity manifest.
    #[serde(default)]
    pub backend: BackendSection,
    /// Default model load parameters and named presets.
    #[serde(default)]
    pub load: LoadSection,
    /// Sampler entropy settings. See `docs/ARCHITECTURE.md` §4 (Axis 0) for why these are not mere
    /// preferences: they are a precondition for every speculative-decoding measurement.
    #[serde(default)]
    pub sampler: SamplerSection,
    /// The effort dial.
    #[serde(default)]
    pub effort: EffortSection,
    /// The verbosity dial.
    #[serde(default)]
    pub verbosity: VerbositySection,
    /// Cerebro's own local HTTP API.
    #[serde(default)]
    pub api: ApiSection,
    /// Display-only preferences.
    #[serde(default)]
    pub ui: UiSection,
    /// Network mode and LAN exposure.
    #[serde(default)]
    pub network: NetworkSection,
    /// Every user-switchable protection.
    #[serde(default)]
    pub protections: ProtectionsSection,
    /// Plugin permission defaults.
    #[serde(default)]
    pub plugins: PluginsSection,
}

impl Settings {
    /// Structural consistency checks beyond what `serde` alone expresses.
    ///
    /// Every path that produces a `Settings` from bytes or from a patch ends here:
    /// [`crate::file::ConfigFile`]'s load path and the tail of [`Settings::apply_patch`] both call
    /// it. Filesystem-free by design.
    ///
    /// Checks, in order, failing on the first that does not hold:
    ///
    /// 1. [`Settings::version`] equals [`SCHEMA_VERSION`].
    /// 2. At most [`MAX_MODEL_FOLDERS`] model folders, each an absolute path: a relative folder
    ///    would resolve against the process's working directory, so the CLI and the GUI would
    ///    approve different trees from the same file.
    /// 3. LAN mode is off while the network mode is offline (the two contradict each other).
    /// 4. Load parameters: preset names, the active preset exists, `ubatch_size <= batch_size`
    ///    for the defaults and for every preset's overlay on them.
    /// 5. Both dials: at least one level, legal ids, unique ranks, a default that names a level,
    ///    bounded instructions and passes.
    ///
    /// # Errors
    /// [`ConfigError::UnsupportedVersion`] or [`ConfigError::SettingRejected`].
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.version != SCHEMA_VERSION {
            return Err(ConfigError::UnsupportedVersion {
                found: self.version,
                supported: SCHEMA_VERSION,
            });
        }
        if self.models.folders.len() > MAX_MODEL_FOLDERS {
            return Err(ConfigError::SettingRejected {
                pointer: "/models/folders".to_string(),
                reason: format!("more than {MAX_MODEL_FOLDERS} folders"),
            });
        }
        for (i, folder) in self.models.folders.iter().enumerate() {
            // `is_absolute` on Windows needs a prefix and a root (`C:\x`, `\\server\share\x`), so
            // the empty path, `x\y`, `\x` and the drive-relative `C:x` are all refused here.
            if !folder.is_absolute() {
                return Err(ConfigError::SettingRejected {
                    pointer: format!("/models/folders/{i}"),
                    reason: "a model folder must be an absolute path".to_string(),
                });
            }
        }
        if self.network.lan_mode && self.network.mode == crate::policy::NetworkMode::Offline {
            return Err(ConfigError::SettingRejected {
                pointer: "/network/lan_mode".to_string(),
                reason: "LAN exposure contradicts offline mode".to_string(),
            });
        }
        self.load.validate()?;
        crate::assistant::validate(&self.effort, &self.verbosity)
    }

    /// The [`cerebro_supervisor::LaunchSpec`] `cerebro-supervisor`'s preflight decides on, with the
    /// backend directory, tool name and protections taken from these settings.
    ///
    /// This is how a launch decision is routed: settings never decide whether a spawn is safe
    /// themselves. The caller supplies what is per-launch (the model, the port, the key file, and the
    /// manifest read from [`BackendSection::manifest`]) and hands the result to
    /// [`cerebro_supervisor::preflight()`], which audits, hashes and refuses, honouring
    /// [`ProtectionsSection::launch`] and recording one warning per switch that is off.
    /// `allow_unsafe_dacl` is always `None`: turning the folder audit off is the settings-level
    /// way to accept an unaudited folder.
    pub fn launch_spec(&self, launch: LaunchRequest) -> cerebro_supervisor::LaunchSpec {
        cerebro_supervisor::LaunchSpec {
            backend_dir: self.backend.dir.clone(),
            tool_name: SERVER_TOOL_NAME.to_string(),
            manifest: launch.manifest,
            model_path: launch.model_path,
            draft_model_path: launch.draft_model_path,
            port: launch.port,
            api_key_file: launch.api_key_file,
            allow_unsafe_dacl: None,
            protections: self.protections.launch(),
        }
    }
}

/// The per-launch half of a [`cerebro_supervisor::LaunchSpec`]; [`Settings::launch_spec`] fills in
/// the rest from settings.
#[derive(Debug, Clone)]
pub struct LaunchRequest {
    /// Target model path.
    pub model_path: PathBuf,
    /// Draft model path for speculative decoding, if any.
    pub draft_model_path: Option<PathBuf>,
    /// Expected backend files, name to lowercase hex SHA-256.
    pub manifest: BTreeMap<String, String>,
    /// How the listening port is chosen.
    pub port: cerebro_supervisor::PortRequest,
    /// The `--api-key-file` path (see [`crate::secrets::KeyFile`]).
    pub api_key_file: PathBuf,
}

impl Default for Settings {
    /// A valid document that exposes nothing: no model folder, the API off, LAN off, network use
    /// only on request, every protection on, every plugin capability `ask`, and no load parameter
    /// set (llama.cpp's own automatic settings).
    fn default() -> Self {
        Self {
            version: SCHEMA_VERSION,
            models: ModelsSection::default(),
            backend: BackendSection::default(),
            load: LoadSection::default(),
            sampler: SamplerSection::default(),
            effort: EffortSection::default(),
            verbosity: VerbositySection::default(),
            api: ApiSection::default(),
            ui: UiSection::default(),
            network: NetworkSection::default(),
            protections: ProtectionsSection::default(),
            plugins: PluginsSection::default(),
        }
    }
}

/// Model folders and the default model.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsSection {
    /// Folders scanned for models (Cerebro's own, LM Studio's, a Hugging Face cache, a network
    /// share such as `\\nas\models`, ...). Each must be absolute. The local HTTP API may not change
    /// this list (see [`crate::profile::SECURITY_LOCKED`]).
    ///
    /// These are also the only roots [`crate::profile::resolve_model_path`] resolves a model path
    /// under. The rule that always holds: a model's literal location (after `.`/`..` are resolved)
    /// must be inside a listed folder, and a UNC model is only ever looked at when it is inside a
    /// listed UNC folder. By default the real target must be inside a folder too, so a link that
    /// leads out is refused; switching
    /// [`crate::policy::ProtectionsSection::refuse_redirected_paths`] off lifts that second
    /// requirement.
    #[serde(default)]
    pub folders: Vec<PathBuf>,
    /// The model loaded when none is named. `None` until the user picks one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
}

/// The llama.cpp build Cerebro spawns. The local HTTP API may not change it: ggml loads DLLs from
/// `dir` at spawn time, so a live source redirecting it is a DLL-plant vector, not a preference.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendSection {
    /// Directory containing the backend executables and `ggml-*` DLLs. Empty until configured, in
    /// which case preflight refuses (it must be absolute).
    #[serde(default)]
    pub dir: PathBuf,
    /// Path to the SHA-256 manifest checked before every launch.
    #[serde(default)]
    pub manifest: PathBuf,
}

/// Sampler entropy. Not user taste: a correctness precondition for every speculative-decoding
/// measurement (`docs/ARCHITECTURE.md` §4, Axis 0).
///
/// Fields are private and only constructible through [`SamplerSection::new`] (or, for `serde`,
/// through the hand-written [`Deserialize`] impl below that routes through the same constructor):
/// `cerebro_store::SamplerProfile::new` -- the type that actually records a measurement's sampler
/// settings -- rejects a non-finite or negative `temp` and a `top_p`/`min_p` outside `0.0..=1.0`.
/// A `Settings` this crate accepts but the results store then refuses would produce a run that is
/// measured but unloggable, so the same bounds are enforced here, at construction. `f64` (not
/// `f32`) to match `SamplerProfile`'s field width exactly. The defaults are llama.cpp's own.
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
    /// fail loudly.
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
    /// refused every value outside `f32`'s finite range, so none of these casts can produce `inf`
    /// or `NaN`.
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
        // Direct construction, not `new()`: these literals are known-valid (all within bounds), so
        // a fallible call here would only add a dead error branch.
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
    /// out-of-range sampler section.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // `deny_unknown_fields` lives on this private `Raw` shape, not on `SamplerSection` itself
        // (which has no derived `Deserialize` to attach it to).
        // A missing key takes llama.cpp's default, like every other section.
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            #[serde(default = "default_temp")]
            temp: f64,
            #[serde(default = "default_top_k")]
            top_k: u32,
            #[serde(default = "default_top_p")]
            top_p: f64,
            #[serde(default = "default_min_p")]
            min_p: f64,
        }
        fn default_temp() -> f64 {
            SamplerSection::default().temp
        }
        fn default_top_k() -> u32 {
            SamplerSection::default().top_k
        }
        fn default_top_p() -> f64 {
            SamplerSection::default().top_p
        }
        let raw = Raw::deserialize(deserializer)?;
        SamplerSection::new(raw.temp, raw.top_k, raw.top_p, raw.min_p).map_err(serde::de::Error::custom)
    }
}

/// Bind address for Cerebro's own local HTTP API.
///
/// Exactly two variants, both loopback. LAN exposure is [`crate::policy::NetworkSection::lan_mode`],
/// which a future surface may interpret as permission to run a separate, explicitly configured
/// listener; this type cannot express a non-loopback host at all.
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

/// Cerebro's own local HTTP API. A missing key takes its default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ApiSection {
    /// Whether the API listens at all. Defaults to off: a fresh install exposes nothing.
    pub enabled: bool,
    /// TCP port to bind.
    pub port: u16,
    /// Which loopback address to bind. The API may not change this itself (see
    /// [`crate::profile::SECURITY_LOCKED`]).
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
/// A missing key takes its default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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
        "required": ["version"],
        "properties": {
            "version": { "type": "integer", "minimum": 0, "maximum": u64::from(u32::MAX), "const": SCHEMA_VERSION },
            "models": models_schema(),
            "backend": backend_schema(),
            "load": crate::load::load_schema(),
            "sampler": sampler_schema(),
            "effort": crate::assistant::effort_schema(),
            "verbosity": crate::assistant::verbosity_schema(),
            "api": api_schema(),
            "ui": ui_schema(),
            "network": crate::policy::network_schema(),
            "protections": crate::policy::protections_schema(),
            "plugins": crate::policy::plugins_schema(),
        }
    })
}

fn models_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "folders": {
                "type": "array",
                "maxItems": MAX_MODEL_FOLDERS,
                "items": { "type": "string", "minLength": 1, "description": "An absolute path." },
            },
            "default_model": { "type": "string" },
        }
    })
}

fn backend_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "dir": { "type": "string" },
            "manifest": { "type": "string" },
        }
    })
}

fn sampler_schema() -> serde_json::Value {
    let d = SamplerSection::default();
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            // Bounds mirror `SamplerSection::new` exactly: a document the schema accepts must also
            // be a document the constructor accepts.
            "temp": { "type": "number", "minimum": 0.0, "maximum": SamplerSection::MAX_TEMP, "default": d.temp },
            "top_k": { "type": "integer", "minimum": 0, "maximum": u64::from(u32::MAX), "default": d.top_k },
            "top_p": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": d.top_p },
            "min_p": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": d.min_p },
        }
    })
}

fn api_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "enabled": { "type": "boolean", "default": false },
            "port": { "type": "integer", "minimum": 0, "maximum": 65535, "default": ApiSection::default().port },
            "bind": { "type": "string", "enum": ["loopback4", "loopback6"], "default": "loopback4" },
        }
    })
}

fn ui_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "theme_base": { "type": "string" },
            "theme_accent": { "type": "string" },
            "theme_contrast": { "type": "integer", "minimum": 0, "maximum": 255 },
            "density": { "type": "string", "enum": ["comfortable", "compact"] },
            "reduce_motion": { "type": "boolean" },
        }
    })
}

/// A `Settings` value with every field set to a distinctive, non-default value.
///
/// Used by the round-trip test (which must prove every field survives serialization, not just that
/// defaults survive) and by the schema-coverage test (which must see every key at least once,
/// including the optional ones).
#[cfg(test)]
pub(crate) fn full_settings_fixture() -> Settings {
    use crate::load::{BatchSize, CpuMoeLayers, CtxSize, GpuLayerCount, LoadParams, ThreadCount};
    use crate::policy::{Capability, NetworkMode, Permission};

    let params = LoadParams {
        n_ctx: Some(CtxSize(8192)),
        n_gpu_layers: Some(GpuLayerCount(999)),
        n_cpu_moe: Some(CpuMoeLayers(32)),
        cache_type_k: Some(cerebro_flags::CacheType::Q8_0),
        cache_type_v: Some(cerebro_flags::CacheType::Q8_0),
        flash_attn: Some(cerebro_flags::FlashAttn::On),
        threads: Some(ThreadCount::new(6).expect("in range")),
        threads_batch: Some(ThreadCount::new(12).expect("in range")),
        batch_size: Some(BatchSize::new(2048).expect("in range")),
        ubatch_size: Some(BatchSize::new(512).expect("in range")),
        load_mode: Some(cerebro_flags::LoadMode::Mmap),
        kv_offload: Some(true),
    };
    let mut effort = EffortSection {
        default_level: "high".to_string(),
        answer_reserve_tokens: 4096,
        ..EffortSection::default()
    };
    if let Some(level) = effort.levels.get_mut("high") {
        level.instruction = "Custom high-effort instruction.".to_string();
        level.thinking_budget = crate::assistant::ThinkingBudget::Tokens(8192);
    }
    let verbosity = VerbositySection {
        default_level: "detailed".to_string(),
        ..VerbositySection::default()
    };

    Settings {
        version: SCHEMA_VERSION,
        models: ModelsSection {
            folders: vec![PathBuf::from(r"C:\models"), PathBuf::from(r"D:\more-models")],
            default_model: Some("bastion-control-0.5b".to_string()),
        },
        backend: BackendSection {
            dir: PathBuf::from(r"C:\llama-cuda"),
            manifest: PathBuf::from(r"C:\Users\noahm\.cerebro\backend-manifest.json"),
        },
        load: LoadSection {
            defaults: params.clone(),
            presets: BTreeMap::from([(
                "long context".to_string(),
                LoadParams {
                    n_ctx: Some(CtxSize(32768)),
                    ..LoadParams::default()
                },
            )]),
            active_preset: Some("long context".to_string()),
        },
        sampler: SamplerSection::new(0.7, 64, 0.9, 0.05).expect("fixture sampler values are in range"),
        effort,
        verbosity,
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
        network: NetworkSection {
            mode: NetworkMode::Online,
            lan_mode: true,
        },
        protections: ProtectionsSection {
            audit_folder_permissions: false,
            permission_prompts: false,
            ..ProtectionsSection::default()
        },
        plugins: PluginsSection {
            default_permission: Permission::Deny,
            capabilities: BTreeMap::from([
                (Capability::ReadFiles, Permission::Always),
                (Capability::WriteFiles, Permission::Session),
                (Capability::RunCommands, Permission::Ask),
                (Capability::Network, Permission::Deny),
            ]),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_validate_and_expose_nothing() {
        let settings = Settings::default();
        settings.validate().expect("defaults are valid");
        assert!(settings.models.folders.is_empty());
        assert!(!settings.network.lan_mode);
        assert!(!settings.api.enabled);
        assert_eq!(settings.protections.launch(), cerebro_supervisor::Protections::default());
        assert!(settings.load.effective().to_flags().is_empty(), "llama.cpp decides by default");
        assert_eq!(settings.verbosity.default_level, "brief");
        assert_eq!(settings.sampler.min_p, 0.05);
    }

    #[test]
    fn a_document_with_only_a_version_is_the_defaults() {
        let settings: Settings =
            serde_json::from_value(serde_json::json!({"version": SCHEMA_VERSION})).expect("parses");
        assert_eq!(settings, Settings::default());
    }

    #[test]
    fn every_schema_key_round_trips_through_json() {
        let original = full_settings_fixture();
        original.validate().expect("fixture is valid");
        let text = serde_json::to_string(&original).expect("fixture serializes");
        let restored: Settings = serde_json::from_str(&text).expect("fixture round-trips");
        assert_eq!(original, restored);
    }

    #[test]
    fn a_resurrected_version_1_key_is_an_unknown_field() {
        for (section, key) in [("security", "model_roots"), ("runtime", "n_ctx")] {
            let mut value = serde_json::to_value(Settings::default()).expect("serializes");
            value[section] = serde_json::json!({ key: 1 });
            let err = serde_json::from_value::<Settings>(value).unwrap_err();
            assert!(err.to_string().contains("unknown field"), "{section}: {err}");
        }
    }

    #[test]
    fn launch_spec_carries_the_backend_and_every_protection_switch() {
        let mut settings = full_settings_fixture();
        settings.protections.refuse_elevation = false;
        let request = LaunchRequest {
            model_path: PathBuf::from(r"C:\models\m.gguf"),
            draft_model_path: None,
            manifest: BTreeMap::from([("llama-server.exe".to_string(), "ab".repeat(32))]),
            port: cerebro_supervisor::PortRequest::Any,
            api_key_file: PathBuf::from(r"C:\tmp\k.key"),
        };
        let spec = settings.launch_spec(request);
        assert_eq!(spec.backend_dir, settings.backend.dir);
        assert_eq!(spec.tool_name, SERVER_TOOL_NAME);
        assert_eq!(spec.protections, settings.protections.launch());
        assert!(!spec.protections.audit_folder_permissions);
        assert!(!spec.protections.refuse_elevation);
        assert!(spec.protections.verify_backend_integrity);
        assert!(spec.allow_unsafe_dacl.is_none());
        assert_eq!(spec.manifest.len(), 1);
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
    fn sampler_section_constructor_rejects_values_cerebro_store_would_reject() {
        let cases: [(&str, f64, u32, f64, f64); 5] = [
            ("negative temp", -0.1, 40, 0.9, 0.05),
            ("non-finite temp", f64::NAN, 40, 0.9, 0.05),
            ("top_p above 1.0", 0.7, 40, 1.5, 0.05),
            ("min_p below 0.0", 0.7, 40, 0.9, -0.01),
            ("non-finite min_p", 0.7, 40, 0.9, f64::INFINITY),
        ];
        for (label, temp, top_k, top_p, min_p) in cases {
            let err = SamplerSection::new(temp, top_k, top_p, min_p).unwrap_err();
            assert!(matches!(err, ConfigError::SamplerOutOfRange { .. }), "{label}: got {err:?}");
        }
    }

    #[test]
    fn sampler_flags_are_never_infinite_at_the_representable_boundary() {
        let err = SamplerSection::new(1e300, 40, 0.9, 0.05).unwrap_err();
        assert!(
            matches!(&err, ConfigError::SamplerOutOfRange { field, .. } if *field == "temp"),
            "got {err:?}"
        );
        let at_boundary = SamplerSection::new(f64::from(f32::MAX), 40, 0.9, 0.05).expect("f32::MAX is in range");
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
        let mut value = serde_json::to_value(full_settings_fixture()).expect("serializes");
        value["sampler"]["temp"] = serde_json::json!(1e300);
        let err = serde_json::from_value::<Settings>(value).unwrap_err();
        assert!(err.to_string().contains("temp"), "error: {err}");
    }

    #[test]
    fn schema_temp_bound_matches_the_flag_width() {
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
        assert!(err.to_string().contains("top_p"), "error: {err}");
    }

    #[test]
    fn a_sampler_from_settings_is_always_recordable() {
        let sampler = full_settings_fixture().sampler;
        let recordable =
            cerebro_store::SamplerProfile::new(sampler.temp(), sampler.top_k(), sampler.top_p(), sampler.min_p());
        assert!(recordable.is_ok(), "a Settings-constructible sampler must always be recordable: {recordable:?}");
    }

    /// Every child schema node of `node`, with the path segment it describes: object `properties`,
    /// a map's `additionalProperties`, array `items`, and each `oneOf` branch.
    fn children(node: &serde_json::Value) -> Vec<(String, &serde_json::Value)> {
        let mut out = Vec::new();
        if let Some(props) = node.get("properties").and_then(|p| p.as_object()) {
            out.extend(props.iter().map(|(k, v)| (k.clone(), v)));
        }
        if let Some(extra) = node.get("additionalProperties").filter(|v| v.is_object()) {
            out.push(("*".to_string(), extra));
        }
        if let Some(items) = node.get("items") {
            out.push(("[]".to_string(), items));
        }
        if let Some(branches) = node.get("oneOf").and_then(|b| b.as_array()) {
            out.extend(branches.iter().enumerate().map(|(i, b)| (format!("oneOf{i}"), b)));
        }
        out
    }

    #[test]
    fn every_integer_node_in_the_schema_is_bounded_on_both_sides() {
        // An unbounded "integer" in the published contract is a document an external validator
        // calls valid and this crate then refuses to load.
        fn walk(node: &serde_json::Value, path: &str, seen: &mut usize) {
            if node.get("type") == Some(&serde_json::json!("integer")) {
                *seen += 1;
                assert!(node.get("minimum").is_some(), "{path} has no minimum");
                assert!(node.get("maximum").is_some(), "{path} has no maximum");
            }
            for (segment, child) in children(node) {
                walk(child, &format!("{path}/{segment}"), seen);
            }
        }
        let mut seen = 0;
        walk(&json_schema(), "", &mut seen);
        // The walk must actually reach nested maps (load presets, effort levels), not stop at the
        // top level: there are well over twenty integer nodes.
        assert!(seen > 20, "only {seen} integer nodes visited");
    }

    #[test]
    fn a_document_declaring_an_unsupported_schema_version_is_refused() {
        let mut settings = full_settings_fixture();
        settings.version = 1;
        let err = settings.validate().unwrap_err();
        assert_eq!(err, ConfigError::UnsupportedVersion { found: 1, supported: SCHEMA_VERSION });
    }

    #[test]
    fn too_many_model_folders_is_refused() {
        let mut settings = Settings::default();
        settings.models.folders = vec![PathBuf::from(r"C:\m"); MAX_MODEL_FOLDERS + 1];
        assert!(matches!(settings.validate(), Err(ConfigError::SettingRejected { .. })));
    }

    #[test]
    fn a_relative_or_empty_model_folder_is_refused() {
        for bad in ["relative/dir", "", r"\rooted-no-drive", "C:drive-relative"] {
            let mut settings = full_settings_fixture();
            settings.models.folders.push(PathBuf::from(bad));
            let err = settings.validate().unwrap_err();
            assert_eq!(
                err,
                ConfigError::SettingRejected {
                    pointer: "/models/folders/2".to_string(),
                    reason: "a model folder must be an absolute path".to_string(),
                },
                "{bad:?}"
            );
        }
        let mut settings = full_settings_fixture();
        settings.models.folders.push(PathBuf::from(r"\\nas\models"));
        settings.validate().expect("a UNC folder is absolute");
    }

    #[test]
    fn offline_with_lan_mode_is_refused() {
        let mut settings = Settings::default();
        settings.network.mode = crate::policy::NetworkMode::Offline;
        settings.validate().expect("offline alone is fine");
        settings.network.lan_mode = true;
        let err = settings.validate().unwrap_err();
        assert!(
            matches!(&err, ConfigError::SettingRejected { pointer, .. } if pointer == "/network/lan_mode"),
            "{err:?}"
        );
        settings.network.mode = crate::policy::NetworkMode::OnRequest;
        settings.validate().expect("LAN mode while not offline is representable");
    }

    #[test]
    fn a_partial_api_ui_sampler_or_plugins_section_takes_defaults() {
        let settings: Settings = serde_json::from_value(serde_json::json!({
            "version": SCHEMA_VERSION,
            "api": {"enabled": true},
            "ui": {"density": "compact"},
            "sampler": {"temp": 0.2},
            "plugins": {"capabilities": {"network": "deny"}},
        }))
        .expect("partial sections parse");
        settings.validate().expect("and validate");
        let d = Settings::default();
        assert!(settings.api.enabled);
        assert_eq!((settings.api.port, settings.api.bind), (d.api.port, d.api.bind));
        assert_eq!(settings.ui.density, Density::Compact);
        assert_eq!(settings.ui.theme_base, d.ui.theme_base);
        assert_eq!(settings.sampler.temp(), 0.2);
        assert_eq!(settings.sampler.top_k(), d.sampler.top_k());
        assert_eq!(settings.sampler.top_p(), d.sampler.top_p());
        assert_eq!(settings.plugins.default_permission, d.plugins.default_permission);
        assert_eq!(settings.plugins.capabilities.len(), 1, "a written map replaces the default map");
    }

    #[test]
    fn a_duplicate_plugin_capability_key_is_refused() {
        let text = format!(
            r#"{{"version": {SCHEMA_VERSION}, "plugins": {{"capabilities": {{"network": "deny", "network": "always"}}}}}}"#
        );
        let err = serde_json::from_str::<Settings>(&text).unwrap_err();
        assert!(err.to_string().contains("duplicate key"), "{err}");
    }

    #[test]
    fn every_schema_default_matches_the_rust_default() {
        // A `default` the published schema advertises must be what a missing key actually becomes.
        fn walk(schema: &serde_json::Value, value: &serde_json::Value, path: &str, seen: &mut usize) {
            let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
                return;
            };
            for (key, child) in props {
                let child_value = value.get(key).unwrap_or(&serde_json::Value::Null);
                if let Some(advertised) = child.get("default") {
                    *seen += 1;
                    assert_eq!(advertised, child_value, "{path}/{key}");
                }
                walk(child, child_value, &format!("{path}/{key}"), seen);
            }
        }
        let mut seen = 0;
        let defaults = serde_json::to_value(Settings::default()).expect("serializes");
        walk(&json_schema(), &defaults, "", &mut seen);
        assert!(seen >= 15, "only {seen} defaults checked");
    }

    fn assert_schema_covers(value: &serde_json::Value, schema: &serde_json::Value, path: &str) {
        match value {
            serde_json::Value::Object(map) => {
                let properties = schema.get("properties").and_then(|p| p.as_object());
                let extra = schema.get("additionalProperties").filter(|v| v.is_object());
                assert!(
                    properties.is_some() || extra.is_some(),
                    "schema at {path:?} describes no keys"
                );
                for (key, child_value) in map {
                    let child_path = format!("{path}/{key}");
                    let child_schema = properties
                        .and_then(|p| p.get(key))
                        .or(extra)
                        .unwrap_or_else(|| {
                            panic!("schema at {path:?} does not describe key {key:?}; add it to json_schema()")
                        });
                    assert_schema_covers(child_value, child_schema, &child_path);
                }
            }
            serde_json::Value::Array(items) => {
                let item_schema = schema
                    .get("items")
                    .unwrap_or_else(|| panic!("schema at {path:?} has no \"items\" for its array value"));
                for (index, item) in items.iter().enumerate() {
                    assert_schema_covers(item, item_schema, &format!("{path}/{index}"));
                }
            }
            _ => assert!(!schema.is_null(), "schema at {path:?} is missing for value {value:?}"),
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
        // Compared with line endings normalized: a checkout may turn the committed file's LF into
        // CRLF, which changes no byte of the schema itself.
        assert_eq!(
            committed.replace("\r\n", "\n"),
            generated,
            "docs/schema/settings.schema.json is stale; regenerate with \
             `CEREBRO_UPDATE_SCHEMA=1 cargo test -p cerebro-config schema_file_matches_generated_schema`"
        );
    }
}
