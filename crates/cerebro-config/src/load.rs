//! Model load parameters: the defaults every load starts from, and named presets over them.
//!
//! # `None` means llama.cpp decides, never a Cerebro guess
//!
//! Measured on the development machine: tuned flags beat llama.cpp's own automatic settings about
//! 6x for a 30B MoE that does not fit in VRAM, but llama.cpp's defaults won for a small model that
//! fits. So no field here carries a hardcoded number. Every one is optional, and an unset field
//! emits no flag at all, which leaves llama.cpp's own automatic choice (its memory fit, its thread
//! count, its KV type) in charge. A value appears only when the user set it or a measurement
//! (`cerebro tune`) wrote it, which is how a configuration slower than llama.cpp's own can never
//! become the default by accident. "Reset" is deleting the key: a JSON Merge Patch `null` on
//! `/load/defaults/n_gpu_layers` returns that parameter to automatic.
//!
//! # Why every value is a `cerebro-flags` type
//!
//! [`cerebro_flags`] closed the argv space, and its value types make an invalid value
//! unrepresentable. Each field here is either one of those enums, serialized through its own
//! `as_str()`/`parse()` round trip, or a newtype whose `to_flag()` produces exactly the
//! [`cerebro_flags::Flag`] it stands for, so a document this crate accepts is always one
//! `cerebro-flags` can emit. [`LoadParams::to_flags`] is the one mapping to argv.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{cap_echo, ConfigError};

/// Context length in tokens, mapped 1:1 onto llama.cpp's `-c`. `0` means "the model's own trained
/// context", which is llama.cpp's meaning, not Cerebro's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CtxSize(pub u32);

/// GPU offload depth, mapped 1:1 onto `-ngl` (`999` offloads every layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GpuLayerCount(pub u32);

/// MoE expert host-memory depth, mapped 1:1 onto `-ncmoe`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CpuMoeLayers(pub u32);

/// Largest thread count accepted for `-t`/`-tb`. Far above any consumer CPU; bounds a typo like
/// `40000` before it reaches llama.cpp.
pub const MAX_THREADS: u32 = 1024;

/// Largest batch accepted for `-b`/`-ub`, in tokens.
pub const MAX_BATCH: u32 = 1 << 20;

/// A thread count for `-t`/`-tb`, `1..=`[`MAX_THREADS`]. Zero is not a thread count; leaving the
/// field unset is how llama.cpp picks its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct ThreadCount(u32);

impl ThreadCount {
    /// Build a thread count, refusing `0` and anything above [`MAX_THREADS`].
    ///
    /// # Errors
    /// [`ConfigError::SettingRejected`] naming the bound.
    pub fn new(n: u32) -> Result<Self, ConfigError> {
        if (1..=MAX_THREADS).contains(&n) {
            Ok(Self(n))
        } else {
            Err(ConfigError::SettingRejected {
                pointer: "thread count".to_string(),
                reason: format!("{n} is outside 1..={MAX_THREADS}"),
            })
        }
    }

    /// The count.
    pub fn get(self) -> u32 {
        self.0
    }
}

impl TryFrom<u32> for ThreadCount {
    type Error = ConfigError;
    fn try_from(n: u32) -> Result<Self, Self::Error> {
        Self::new(n)
    }
}

impl From<ThreadCount> for u32 {
    fn from(t: ThreadCount) -> Self {
        t.0
    }
}

/// A batch size in tokens for `-b`/`-ub`, `1..=`[`MAX_BATCH`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct BatchSize(u32);

impl BatchSize {
    /// Build a batch size, refusing `0` and anything above [`MAX_BATCH`].
    ///
    /// # Errors
    /// [`ConfigError::SettingRejected`] naming the bound.
    pub fn new(n: u32) -> Result<Self, ConfigError> {
        if (1..=MAX_BATCH).contains(&n) {
            Ok(Self(n))
        } else {
            Err(ConfigError::SettingRejected {
                pointer: "batch size".to_string(),
                reason: format!("{n} is outside 1..={MAX_BATCH}"),
            })
        }
    }

    /// The size in tokens.
    pub fn get(self) -> u32 {
        self.0
    }
}

impl TryFrom<u32> for BatchSize {
    type Error = ConfigError;
    fn try_from(n: u32) -> Result<Self, Self::Error> {
        Self::new(n)
    }
}

impl From<BatchSize> for u32 {
    fn from(b: BatchSize) -> Self {
        b.0
    }
}

/// Every model load parameter a user can set. Each unset field emits no flag, so llama.cpp's own
/// automatic value applies (see the module documentation).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadParams {
    /// `-c` context size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_ctx: Option<CtxSize>,
    /// `-ngl` GPU offload depth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_gpu_layers: Option<GpuLayerCount>,
    /// `-ncmoe` MoE expert layers kept in host memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_cpu_moe: Option<CpuMoeLayers>,
    /// `-ctk` K cache element type.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_token")]
    pub cache_type_k: Option<cerebro_flags::CacheType>,
    /// `-ctv` V cache element type.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_token")]
    pub cache_type_v: Option<cerebro_flags::CacheType>,
    /// `-fa` flash attention mode.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_token")]
    pub flash_attn: Option<cerebro_flags::FlashAttn>,
    /// `-t` generation threads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<ThreadCount>,
    /// `-tb` prompt-processing threads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads_batch: Option<ThreadCount>,
    /// `-b` logical batch size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<BatchSize>,
    /// `-ub` physical batch size. Must not exceed `batch_size` when both are set: llama.cpp would
    /// silently clamp it, and a setting that does nothing must not be recorded as if it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ubatch_size: Option<BatchSize>,
    /// `-lm` model loading mode (mmap, mlock, direct I/O).
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_token")]
    pub load_mode: Option<cerebro_flags::LoadMode>,
    /// `-kvo`/`-nkvo`: whether the KV cache lives in VRAM.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_offload: Option<bool>,
}

impl LoadParams {
    /// The flags for every field that is set, in [`cerebro_flags::Flag`]'s declaration order.
    ///
    /// The one sanctioned mapping from load parameters to argv: assembling `CacheTypeK`/`CacheTypeV`
    /// by hand can transpose K and V without anything failing to compile.
    pub fn to_flags(&self) -> Vec<cerebro_flags::Flag> {
        use cerebro_flags::Flag;
        let mut flags = Vec::new();
        if let Some(v) = self.n_ctx {
            flags.push(Flag::CtxSize(v.0));
        }
        if let Some(v) = self.n_gpu_layers {
            flags.push(Flag::GpuLayers(v.0));
        }
        if let Some(v) = self.n_cpu_moe {
            flags.push(Flag::NCpuMoe(v.0));
        }
        if let Some(v) = self.load_mode {
            flags.push(Flag::LoadMode(v));
        }
        if let Some(v) = self.cache_type_k {
            flags.push(Flag::CacheTypeK(v));
        }
        if let Some(v) = self.cache_type_v {
            flags.push(Flag::CacheTypeV(v));
        }
        if let Some(v) = self.flash_attn {
            flags.push(Flag::FlashAttn(v));
        }
        if let Some(v) = self.threads {
            flags.push(Flag::Threads(v.0));
        }
        if let Some(v) = self.threads_batch {
            flags.push(Flag::ThreadsBatch(v.0));
        }
        if let Some(v) = self.batch_size {
            flags.push(Flag::BatchSize(v.0));
        }
        if let Some(v) = self.ubatch_size {
            flags.push(Flag::UBatchSize(v.0));
        }
        if let Some(v) = self.kv_offload {
            flags.push(Flag::KvOffload(v));
        }
        flags
    }

    /// `self` with every field `over` sets replacing the same field here. How a preset applies to
    /// the defaults.
    pub fn overlaid_with(&self, over: &LoadParams) -> LoadParams {
        LoadParams {
            n_ctx: over.n_ctx.or(self.n_ctx),
            n_gpu_layers: over.n_gpu_layers.or(self.n_gpu_layers),
            n_cpu_moe: over.n_cpu_moe.or(self.n_cpu_moe),
            cache_type_k: over.cache_type_k.or(self.cache_type_k),
            cache_type_v: over.cache_type_v.or(self.cache_type_v),
            flash_attn: over.flash_attn.or(self.flash_attn),
            threads: over.threads.or(self.threads),
            threads_batch: over.threads_batch.or(self.threads_batch),
            batch_size: over.batch_size.or(self.batch_size),
            ubatch_size: over.ubatch_size.or(self.ubatch_size),
            load_mode: over.load_mode.or(self.load_mode),
            kv_offload: over.kv_offload.or(self.kv_offload),
        }
    }

    /// Cross-field checks: `ubatch_size <= batch_size` when both are set.
    ///
    /// # Errors
    /// [`ConfigError::SettingRejected`] at `pointer`.
    pub fn validate(&self, pointer: &str) -> Result<(), ConfigError> {
        if let (Some(b), Some(ub)) = (self.batch_size, self.ubatch_size) {
            if ub.0 > b.0 {
                return Err(ConfigError::SettingRejected {
                    pointer: format!("{pointer}/ubatch_size"),
                    reason: format!(
                        "{} exceeds batch_size {}; llama.cpp would clamp it silently",
                        ub.0, b.0
                    ),
                });
            }
        }
        Ok(())
    }
}

/// Longest preset name, in characters.
pub const MAX_PRESET_NAME_CHARS: usize = 64;

/// Most presets one document may hold. Bounds a hostile file's memory without limiting any real
/// use.
pub const MAX_PRESETS: usize = 256;

/// The defaults every model load starts from, plus named presets layered over them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadSection {
    /// Parameters applied to every load. Empty by default: llama.cpp's own automatic settings.
    #[serde(default)]
    pub defaults: LoadParams,
    /// Named presets, e.g. `"long context"`, each overriding only the fields it sets. A name
    /// repeated in the file is refused rather than silently letting the last one win.
    #[serde(default, deserialize_with = "unique_map")]
    pub presets: BTreeMap<String, LoadParams>,
    /// The preset in effect, or `None` for the defaults alone. Must name an entry in `presets`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_preset: Option<String>,
}

impl LoadSection {
    /// The parameters in effect: `defaults`, overlaid with the active preset when one is set.
    ///
    /// A dangling `active_preset` cannot reach here through [`crate::Settings`] (validation refuses
    /// it); if one is constructed directly, the defaults alone apply.
    pub fn effective(&self) -> LoadParams {
        match self.active_preset.as_ref().and_then(|name| self.presets.get(name)) {
            Some(preset) => self.defaults.overlaid_with(preset),
            None => self.defaults.clone(),
        }
    }

    /// Preset names non-blank and at most [`MAX_PRESET_NAME_CHARS`], at most [`MAX_PRESETS`]
    /// presets, `active_preset` naming one of them, and every parameter set internally consistent,
    /// both on its own and as the effective overlay (defaults plus preset) it would produce.
    ///
    /// # Errors
    /// [`ConfigError::SettingRejected`] naming the first offending pointer.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.defaults.validate("/load/defaults")?;
        if self.presets.len() > MAX_PRESETS {
            return Err(ConfigError::SettingRejected {
                pointer: "/load/presets".to_string(),
                reason: format!("{} presets exceed the limit of {MAX_PRESETS}", self.presets.len()),
            });
        }
        for (name, params) in &self.presets {
            let at = format!("/load/presets/{}", cap_echo(name));
            check_name(name, &at)?;
            params.validate(&at)?;
            // Every preset can become the active one, so the parameters that would actually reach
            // argv (the defaults with this preset laid over them) must hold too: defaults `-b 512`
            // plus a preset `-ub 4096` is a clamp llama.cpp would apply silently.
            self.defaults.overlaid_with(params).validate(&at)?;
        }
        if let Some(active) = &self.active_preset {
            check_name(active, "/load/active_preset")?;
        }
        if let Some(active) = &self.active_preset {
            if !self.presets.contains_key(active) {
                return Err(ConfigError::SettingRejected {
                    pointer: "/load/active_preset".to_string(),
                    reason: format!("no preset named {:?}", cap_echo(active)),
                });
            }
        }
        Ok(())
    }
}

/// A display name: not blank, no control characters, at most [`MAX_PRESET_NAME_CHARS`]. `pointer`
/// is where the name lives, reported as-is on refusal.
pub(crate) fn check_name(name: &str, pointer: &str) -> Result<(), ConfigError> {
    // U+FEFF counts as blank too: JSON Schema's `\s` (ECMA-262) includes it, and a name made only
    // of byte-order marks shows as nothing. See `NAME_PATTERN`.
    let reason = if name.chars().all(|c| c.is_whitespace() || c == '\u{FEFF}') {
        Some("a name must not be blank".to_string())
    } else if name.chars().count() > MAX_PRESET_NAME_CHARS {
        Some(format!("a name must be at most {MAX_PRESET_NAME_CHARS} characters"))
    } else if name.chars().any(char::is_control) {
        Some("a name must not contain control characters".to_string())
    } else {
        None
    };
    match reason {
        Some(reason) => Err(ConfigError::SettingRejected {
            pointer: pointer.to_string(),
            reason,
        }),
        None => Ok(()),
    }
}

/// [`check_name`]'s rule as a JSON Schema (ECMA-262) pattern: at least one character that is
/// neither whitespace nor a control character, and no C0/C1 control character anywhere
/// (`char::is_control` is exactly the `Cc` category, U+0000-U+001F and U+007F-U+009F).
///
/// Whitespace is spelled out rather than written `\s`, because `\s` differs between regex engines
/// (ECMA-262 includes U+FEFF, Rust's `char::is_whitespace` does not); the class below is every
/// non-control character either one treats as whitespace, and [`check_name`] counts the same set
/// as blank. Every such character is written as a `\uXXXX` escape, never literally, so the source
/// holds no invisible characters. No lookaround, so any validator's engine accepts it. Pair it
/// with `maxLength` [`MAX_PRESET_NAME_CHARS`] (JSON Schema counts code points, as `chars().count()` does).
pub(crate) const NAME_PATTERN: &str = concat!(
    r"^[^\u0000-\u001F\u007F-\u009F]*",
    r"[^\u0000-\u001F\u007F-\u009F\u0020\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000\uFEFF]",
    r"[^\u0000-\u001F\u007F-\u009F]*$",
);

/// JSON Schema for a display name [`check_name`] accepts.
pub(crate) fn name_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "string",
        "minLength": 1,
        "maxLength": MAX_PRESET_NAME_CHARS,
        "pattern": NAME_PATTERN,
    })
}

/// Deserialize a map, refusing a key that appears twice.
///
/// `serde_json` keeps the last value of a repeated key in a map silently, so a hand-edited file
/// could show a preset or level in its text that never takes effect. Struct fields already refuse
/// duplicates; this gives the map-backed sections the same rule. Keys and values still go through
/// their own `Deserialize` (and so their own `deny_unknown_fields` and range checks).
pub(crate) fn unique_map<'de, D, K, V>(d: D) -> Result<BTreeMap<K, V>, D::Error>
where
    D: Deserializer<'de>,
    K: Deserialize<'de> + Ord + std::fmt::Debug,
    V: Deserialize<'de>,
{
    struct UniqueMap<K, V>(std::marker::PhantomData<(K, V)>);

    impl<'de, K, V> serde::de::Visitor<'de> for UniqueMap<K, V>
    where
        K: Deserialize<'de> + Ord + std::fmt::Debug,
        V: Deserialize<'de>,
    {
        type Value = BTreeMap<K, V>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a map with unique keys")
        }

        fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some(key) = map.next_key::<K>()? {
                if out.contains_key(&key) {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate key {}",
                        cap_echo(&format!("{key:?}"))
                    )));
                }
                let value = map.next_value::<V>()?;
                out.insert(key, value);
            }
            Ok(out)
        }
    }

    d.deserialize_map(UniqueMap(std::marker::PhantomData))
}

/// A `cerebro-flags` string enum, bridged to serde through its own token spelling.
pub(crate) trait FlagToken: Copy + 'static {
    /// Every legal value.
    const ALL: &'static [Self];
    /// The exact token llama.cpp expects.
    fn token(self) -> &'static str;
}

impl FlagToken for cerebro_flags::CacheType {
    const ALL: &'static [Self] = cerebro_flags::CacheType::ALL;
    fn token(self) -> &'static str {
        self.as_str()
    }
}

impl FlagToken for cerebro_flags::FlashAttn {
    const ALL: &'static [Self] = cerebro_flags::FlashAttn::ALL;
    fn token(self) -> &'static str {
        self.as_str()
    }
}

impl FlagToken for cerebro_flags::LoadMode {
    const ALL: &'static [Self] = cerebro_flags::LoadMode::ALL;
    fn token(self) -> &'static str {
        self.as_str()
    }
}

/// Every token of `T`, for the JSON Schema `enum`.
pub(crate) fn tokens<T: FlagToken>() -> Vec<&'static str> {
    T::ALL.iter().map(|v| v.token()).collect()
}

/// Serde bridge for `Option<T: FlagToken>`: the flag crate's own spelling, so the two can never
/// drift. Paired with `skip_serializing_if = "Option::is_none"`, so `serialize` only ever sees
/// `Some`.
mod opt_token {
    use super::*;

    pub(super) fn serialize<T: FlagToken, S: Serializer>(
        value: &Option<T>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(v) => s.serialize_str(v.token()),
            None => s.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, T: FlagToken, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<T>, D::Error> {
        let Some(raw) = Option::<String>::deserialize(d)? else {
            return Ok(None);
        };
        T::ALL
            .iter()
            .copied()
            .find(|v| v.token() == raw)
            .map(Some)
            .ok_or_else(|| {
                serde::de::Error::custom(format!(
                    "unknown value {:?}, expected one of {}",
                    cap_echo(&raw),
                    tokens::<T>().join(", ")
                ))
            })
    }
}

/// JSON Schema for [`LoadParams`].
pub(crate) fn load_params_schema() -> serde_json::Value {
    let u32_max = u64::from(u32::MAX);
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "description": "Unset keys emit no flag, so llama.cpp's own automatic value applies.",
        "properties": {
            "n_ctx": { "type": "integer", "minimum": 0, "maximum": u32_max },
            "n_gpu_layers": { "type": "integer", "minimum": 0, "maximum": u32_max },
            "n_cpu_moe": { "type": "integer", "minimum": 0, "maximum": u32_max },
            "cache_type_k": { "type": "string", "enum": tokens::<cerebro_flags::CacheType>() },
            "cache_type_v": { "type": "string", "enum": tokens::<cerebro_flags::CacheType>() },
            "flash_attn": { "type": "string", "enum": tokens::<cerebro_flags::FlashAttn>() },
            "threads": { "type": "integer", "minimum": 1, "maximum": MAX_THREADS },
            "threads_batch": { "type": "integer", "minimum": 1, "maximum": MAX_THREADS },
            "batch_size": { "type": "integer", "minimum": 1, "maximum": MAX_BATCH },
            "ubatch_size": { "type": "integer", "minimum": 1, "maximum": MAX_BATCH },
            "load_mode": { "type": "string", "enum": tokens::<cerebro_flags::LoadMode>() },
            "kv_offload": { "type": "boolean" },
        }
    })
}

/// JSON Schema for [`LoadSection`].
pub(crate) fn load_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "defaults": load_params_schema(),
            "presets": {
                "type": "object",
                "maxProperties": MAX_PRESETS,
                "propertyNames": name_schema(),
                "additionalProperties": load_params_schema(),
            },
            "active_preset": name_schema(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerebro_flags::{CacheType, Flag, FlashAttn, LoadMode};

    fn every_field() -> LoadParams {
        LoadParams {
            n_ctx: Some(CtxSize(8192)),
            n_gpu_layers: Some(GpuLayerCount(999)),
            n_cpu_moe: Some(CpuMoeLayers(32)),
            cache_type_k: Some(CacheType::Q8_0),
            cache_type_v: Some(CacheType::Q4_0),
            flash_attn: Some(FlashAttn::On),
            threads: Some(ThreadCount(6)),
            threads_batch: Some(ThreadCount(12)),
            batch_size: Some(BatchSize(2048)),
            ubatch_size: Some(BatchSize(512)),
            load_mode: Some(LoadMode::Mmap),
            kv_offload: Some(false),
        }
    }

    #[test]
    fn unset_parameters_emit_no_flag_so_llama_cpp_decides() {
        assert!(LoadParams::default().to_flags().is_empty());
        let only_ctx = LoadParams {
            n_ctx: Some(CtxSize(4096)),
            ..LoadParams::default()
        };
        assert_eq!(only_ctx.to_flags(), vec![Flag::CtxSize(4096)]);
    }

    #[test]
    fn every_field_maps_to_its_own_flag_with_k_and_v_not_transposed() {
        let flags = every_field().to_flags();
        assert_eq!(
            flags,
            vec![
                Flag::CtxSize(8192),
                Flag::GpuLayers(999),
                Flag::NCpuMoe(32),
                Flag::LoadMode(LoadMode::Mmap),
                Flag::CacheTypeK(CacheType::Q8_0),
                Flag::CacheTypeV(CacheType::Q4_0),
                Flag::FlashAttn(FlashAttn::On),
                Flag::Threads(6),
                Flag::ThreadsBatch(12),
                Flag::BatchSize(2048),
                Flag::UBatchSize(512),
                Flag::KvOffload(false),
            ]
        );
    }

    #[test]
    fn emitted_flags_are_accepted_by_the_server_argv_builder() {
        // The whole point of using cerebro-flags types: whatever settings hold, the argv builder
        // accepts, so a load never fails on a value the settings screen allowed.
        let mut builder = cerebro_flags::ArgvBuilder::new(cerebro_flags::Dialect::Server);
        for flag in every_field().to_flags() {
            builder.push(flag).expect("settings-emitted flag is legal");
        }
        let argv = builder.build().expect("settings-emitted flags pass cross-flag validation");
        assert!(argv.windows(2).any(|w| w == ["-ctv", "q4_0"]), "{argv:?}");
    }

    #[test]
    fn a_preset_overrides_only_the_fields_it_sets() {
        let defaults = LoadParams {
            n_ctx: Some(CtxSize(4096)),
            threads: Some(ThreadCount(6)),
            ..LoadParams::default()
        };
        let preset = LoadParams {
            n_ctx: Some(CtxSize(32768)),
            ..LoadParams::default()
        };
        let mut section = LoadSection {
            defaults,
            presets: BTreeMap::from([("long".to_string(), preset)]),
            active_preset: None,
        };
        assert_eq!(section.effective().n_ctx, Some(CtxSize(4096)));
        section.active_preset = Some("long".to_string());
        let effective = section.effective();
        assert_eq!(effective.n_ctx, Some(CtxSize(32768)));
        assert_eq!(effective.threads, Some(ThreadCount(6)));
    }

    #[test]
    fn a_dangling_active_preset_is_refused() {
        let section = LoadSection {
            active_preset: Some("missing".to_string()),
            ..LoadSection::default()
        };
        let err = section.validate().unwrap_err();
        assert!(
            matches!(&err, ConfigError::SettingRejected { pointer, .. } if pointer == "/load/active_preset"),
            "{err:?}"
        );
    }

    #[test]
    fn a_blank_or_control_character_preset_name_is_refused() {
        for bad in ["", "   ", "tab\there", &"x".repeat(65)] {
            let section = LoadSection {
                presets: BTreeMap::from([(bad.to_string(), LoadParams::default())]),
                ..LoadSection::default()
            };
            assert!(section.validate().is_err(), "{bad:?} accepted");
        }
        let fine = LoadSection {
            presets: BTreeMap::from([("Long context".to_string(), LoadParams::default())]),
            ..LoadSection::default()
        };
        fine.validate().expect("an ordinary name is accepted");
    }

    #[test]
    fn ubatch_larger_than_batch_is_refused_and_equal_is_fine() {
        let mut p = LoadParams {
            batch_size: Some(BatchSize(512)),
            ubatch_size: Some(BatchSize(1024)),
            ..LoadParams::default()
        };
        assert!(p.validate("/load/defaults").is_err());
        p.ubatch_size = Some(BatchSize(512));
        p.validate("/load/defaults").expect("ubatch == batch is legal");
    }

    #[test]
    fn a_preset_whose_overlay_breaks_ubatch_le_batch_is_refused() {
        // Each set is consistent alone; together they would emit `-b 512 -ub 4096`.
        let mut section = LoadSection {
            defaults: LoadParams {
                batch_size: Some(BatchSize(512)),
                ..LoadParams::default()
            },
            presets: BTreeMap::from([(
                "big ubatch".to_string(),
                LoadParams {
                    ubatch_size: Some(BatchSize(4096)),
                    ..LoadParams::default()
                },
            )]),
            active_preset: None,
        };
        // Refused whether or not the preset is active: any preset can be activated later.
        for active in [None, Some("big ubatch".to_string())] {
            section.active_preset = active;
            let err = section.validate().unwrap_err();
            assert!(
                matches!(&err, ConfigError::SettingRejected { pointer, .. }
                    if pointer == "/load/presets/big ubatch/ubatch_size"),
                "{err:?}"
            );
        }
        // A preset that raises the batch alongside the ubatch is fine.
        if let Some(p) = section.presets.get_mut("big ubatch") {
            p.batch_size = Some(BatchSize(4096));
        }
        section.validate().expect("overlay b=4096 ub=4096 is consistent");
        assert_eq!(section.effective().to_flags(), vec![Flag::BatchSize(4096), Flag::UBatchSize(4096)]);
    }

    #[test]
    fn a_duplicate_preset_key_is_refused() {
        let text = r#"{"presets": {"a": {"n_ctx": 1}, "a": {"n_ctx": 2}}}"#;
        let err = serde_json::from_str::<LoadSection>(text).unwrap_err();
        assert!(err.to_string().contains("duplicate key"), "{err}");
        // Distinct keys still load, and element validation still runs inside the map.
        let ok: LoadSection = serde_json::from_str(r#"{"presets": {"a": {"n_ctx": 1}, "b": {}}}"#).expect("unique");
        assert_eq!(ok.presets.len(), 2);
        let bad = serde_json::from_str::<LoadSection>(r#"{"presets": {"a": {"typo": 1}}}"#).unwrap_err();
        assert!(bad.to_string().contains("unknown field"), "{bad}");
    }

    #[test]
    fn schema_name_pattern_agrees_with_check_name() {
        // An external validator and this crate must give the same verdict on every name: compile
        // the published pattern and run it beside `check_name` (plus the schema's length bounds).
        let pattern = regex::Regex::new(NAME_PATTERN).expect("the pattern compiles");
        let schema = name_schema();
        let max = schema["maxLength"].as_u64().expect("maxLength") as usize;
        let min = schema["minLength"].as_u64().expect("minLength") as usize;
        let schema_accepts = |n: &str| {
            let len = n.chars().count();
            (min..=max).contains(&len) && pattern.is_match(n)
        };
        let long = "x".repeat(MAX_PRESET_NAME_CHARS + 1);
        let longest = "x".repeat(MAX_PRESET_NAME_CHARS);
        let cases: [(&str, bool); 16] = [
            ("", false),
            ("   ", false),
            ("\t", false),
            ("tab\there", false),
            ("line\nbreak", false),
            ("del\u{7f}", false),
            ("c1\u{85}", false),
            ("\u{a0}\u{2003}\u{3000}", false),
            ("\u{feff}", false),
            ("\u{2028}", false),
            (&long, false),
            ("Long context", true),
            (" padded ", true),
            ("\u{2028}x", true),
            ("日本語", true),
            (&longest, true),
        ];
        for (name, expected) in cases {
            assert_eq!(check_name(name, "/x").is_ok(), expected, "check_name on {name:?}");
            assert_eq!(schema_accepts(name), expected, "schema on {name:?}");
        }
    }

    #[test]
    fn zero_threads_and_zero_batch_do_not_deserialize() {
        for doc in [
            serde_json::json!({"threads": 0}),
            serde_json::json!({"batch_size": 0}),
            serde_json::json!({"threads": MAX_THREADS + 1}),
        ] {
            assert!(serde_json::from_value::<LoadParams>(doc.clone()).is_err(), "{doc}");
        }
        let ok: LoadParams = serde_json::from_value(serde_json::json!({"threads": 1})).expect("1 thread");
        assert_eq!(ok.threads.map(ThreadCount::get), Some(1));
    }

    #[test]
    fn flag_enums_use_the_flag_crate_spelling_and_reject_anything_else() {
        let v = serde_json::to_value(every_field()).expect("serializes");
        assert_eq!(v["cache_type_k"], "q8_0");
        assert_eq!(v["load_mode"], "mmap");
        let err = serde_json::from_value::<LoadParams>(serde_json::json!({"cache_type_k": "Q8_0"}))
            .unwrap_err();
        assert!(err.to_string().contains("expected one of"), "{err}");
    }

    #[test]
    fn unset_fields_are_omitted_and_round_trip_as_unset() {
        let text = serde_json::to_string(&LoadParams::default()).expect("serializes");
        assert_eq!(text, "{}");
        let back: LoadParams = serde_json::from_str(&text).expect("parses");
        assert_eq!(back, LoadParams::default());
        let full = every_field();
        let back: LoadParams =
            serde_json::from_value(serde_json::to_value(&full).expect("serializes")).expect("parses");
        assert_eq!(back, full);
    }
}
