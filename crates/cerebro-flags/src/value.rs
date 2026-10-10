//! Typed, self-validating values for every llama.cpp flag Cerebro is permitted to emit.
//!
//! The autotuner is a machine that generates flag combinations and persists them. If the values it
//! can construct are only validated at emission time, a poisoned config or results file can steer
//! it somewhere unintended. So validation lives in the constructors: an invalid value cannot be
//! represented, and `Display` is therefore always safe to splice into an argv slot.
//!
//! Every rule below was verified against llama.cpp b10751 (commit 3466812d1). Where upstream itself
//! does not validate (`-mg`, `-ts`), Cerebro validates anyway — upstream accepts those at parse time
//! and fails later at model load, which would otherwise waste an entire soak test.

use std::fmt;

use crate::error::ValueError;

/// A backend device as llama.cpp addresses it, e.g. `CUDA0`, `RPC0`.
///
/// Canonicalized to uppercase-backend + ordinal on construction. Upstream matching is
/// case-insensitive, but emitting a canonical form keeps result keys comparable and logs greppable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceId {
    backend: String,
    ordinal: u32,
}

impl DeviceId {
    /// Parse a device identifier such as `CUDA0` or `rpc1`.
    ///
    /// `CPU` is rejected: verified that `-dev CPU`, `-devd CPU` and `-mmdev CPU` all fail at parse
    /// time with `invalid device: CPU`. The CPU is the implicit fallback, not an addressable
    /// offload target. CPU placement routes through [`OtRule`], `-cmoe`/`-ncmoe`, or `-dev none`.
    pub fn parse(s: &str) -> Result<Self, ValueError> {
        let s = s.trim();
        // Split the alphabetic backend name from its numeric ordinal. The ordinal may be absent
        // here — `CPU` with no ordinal is the single most likely caller mistake, and it must reach
        // the CpuNotAddressable branch below rather than being rejected as generic bad syntax.
        let (backend, ordinal) = match s.find(|c: char| c.is_ascii_digit()) {
            Some(i) => s.split_at(i),
            None => (s, ""),
        };
        if backend.is_empty() {
            return Err(ValueError::DeviceSyntax(s.to_string()));
        }
        let backend = backend.to_ascii_uppercase();
        if backend == "CPU" {
            return Err(ValueError::CpuNotAddressable);
        }
        if !backend.chars().all(|c| c.is_ascii_alphabetic() || c == '_') {
            return Err(ValueError::DeviceSyntax(s.to_string()));
        }
        let ordinal = ordinal
            .parse::<u32>()
            .map_err(|_| ValueError::DeviceSyntax(s.to_string()))?;
        Ok(Self { backend, ordinal })
    }

    /// The backend name, canonicalized to uppercase (`CUDA`, `RPC`, `VULKAN`).
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// The zero-based device ordinal within its backend.
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.backend, self.ordinal)
    }
}

/// A value for `-dev` / `-devd` / `-mmdev`.
///
/// The `none` sentinel is deliberately a separate variant rather than a magic device name, because
/// it behaves differently in two verified ways: it is case-SENSITIVE where device names are not
/// (`-dev None` hard-fails), and it cannot be mixed into a list (`-dev CUDA0/none` is rejected by
/// llama-bench). Modelling it as a variant makes both mistakes unrepresentable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceList {
    /// Do not offload. Emitted as lowercase `none`, always alone.
    None,
    /// An explicit, non-empty device list.
    Devices(Vec<DeviceId>),
}

impl DeviceList {
    /// Build a device list from identifiers. Empty lists are rejected — the way to say "no
    /// offload" is [`DeviceList::None`], not an empty list, which upstream would read as a
    /// malformed value.
    pub fn devices(ids: impl IntoIterator<Item = DeviceId>) -> Result<Self, ValueError> {
        let ids: Vec<_> = ids.into_iter().collect();
        if ids.is_empty() {
            return Err(ValueError::EmptyDeviceList);
        }
        Ok(Self::Devices(ids))
    }

    /// The devices named, or an empty slice for the `none` sentinel.
    pub fn as_slice(&self) -> &[DeviceId] {
        match self {
            Self::None => &[],
            Self::Devices(d) => d,
        }
    }
}

/// A CPU affinity mask for `--cpu-mask` / `-Cd`, emitted as bare lowercase hex without `0x`.
///
/// llama.cpp documents this as "arbitrarily long hex"; a `u64` covers 64 logical processors, which
/// is every machine Cerebro targets. A wider mask would need a different representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CpuMask(u64);

impl CpuMask {
    /// Wrap a raw affinity bitmask. Zero is rejected: an all-zero mask pins the process to no
    /// processor at all, which upstream accepts and which then behaves unpredictably.
    pub fn new(bits: u64) -> Result<Self, ValueError> {
        if bits == 0 {
            return Err(ValueError::EmptyCpuMask);
        }
        Ok(Self(bits))
    }

    /// The raw bitmask.
    pub fn bits(self) -> u64 {
        self.0
    }

    /// Number of logical processors selected — the natural companion to `-t`.
    pub fn count(self) -> u32 {
        self.0.count_ones()
    }
}

impl fmt::Display for CpuMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Bare hex, no 0x prefix: `--cpu-mask 555`, matching llama.cpp's own documented form.
        write!(f, "{:x}", self.0)
    }
}

macro_rules! str_enum {
    (
        $(#[$meta:meta])*
        $name:ident, $err:ident {
            $( $variant:ident => $text:literal ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum $name {
            $(
                #[doc = concat!("`", $text, "`")]
                $variant
            ),+
        }

        impl $name {
            /// Every legal value, in upstream's documented order.
            pub const ALL: &'static [$name] = &[ $( $name::$variant ),+ ];

            /// The exact token llama.cpp expects.
            pub fn as_str(self) -> &'static str {
                match self { $( $name::$variant => $text ),+ }
            }

            /// Parse from the upstream token. Case-sensitive, matching upstream.
            pub fn parse(s: &str) -> Result<Self, ValueError> {
                match s {
                    $( $text => Ok($name::$variant), )+
                    other => Err(ValueError::$err(other.to_string())),
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

str_enum! {
    /// KV cache element type for `-ctk` / `-ctv` (and the `-ctkd` / `-ctvd` draft equivalents).
    ///
    /// Exactly these nine, verified by probing `-ctk bogus`, which prints the allowed set.
    /// Default upstream is `f16`.
    CacheType, BadCacheType {
        F32 => "f32",
        F16 => "f16",
        Bf16 => "bf16",
        Q8_0 => "q8_0",
        Q4_0 => "q4_0",
        Q4_1 => "q4_1",
        Iq4Nl => "iq4_nl",
        Q5_0 => "q5_0",
        Q5_1 => "q5_1",
    }
}

str_enum! {
    /// Speculative decoding strategy for `--spec-type`.
    ///
    /// This is Cerebro's primary search axis. Note that the `ngram-*` family requires no draft model
    /// at all — it speculates from context — which is what makes speculation reachable for a user
    /// who owns exactly one model. `llama-bench` cannot measure any of these; they must be driven
    /// through `llama-server`.
    SpecType, BadSpecType {
        None => "none",
        DraftSimple => "draft-simple",
        DraftEagle3 => "draft-eagle3",
        DraftMtp => "draft-mtp",
        DraftDflash => "draft-dflash",
        DraftDspark => "draft-dspark",
        NgramSimple => "ngram-simple",
        NgramMapK => "ngram-map-k",
        NgramMapK4v => "ngram-map-k4v",
        NgramMod => "ngram-mod",
        NgramCache => "ngram-cache",
    }
}

impl SpecType {
    /// Whether this strategy requires a separate draft model via `-md`.
    ///
    /// The `ngram-*` family drafts from the context itself and needs no second GGUF — which also
    /// means it does not double the untrusted-file surface the way the `draft-*` family does.
    ///
    /// `draft-mtp` is deliberately excluded: multi-token-prediction heads may be embedded in the
    /// target GGUF itself (in which case no `-md` is needed) or supplied as a separate file. It is
    /// therefore *optional* rather than required, and forcing `-md` would reject the embedded case.
    /// Use [`SpecType::accepts_draft_model`] when asking whether `-md` is meaningful at all.
    pub fn requires_draft_model(self) -> bool {
        matches!(
            self,
            SpecType::DraftSimple
                | SpecType::DraftEagle3
                | SpecType::DraftDflash
                | SpecType::DraftDspark
        )
    }

    /// Whether a separate draft model is meaningful for this strategy, whether or not required.
    ///
    /// False for the entire `ngram-*` family and for `none`, where every draft-placement flag is
    /// silently inert.
    pub fn accepts_draft_model(self) -> bool {
        self.requires_draft_model() || self == SpecType::DraftMtp
    }

    /// Whether this strategy is a context-drafting n-gram variant.
    pub fn is_ngram(self) -> bool {
        matches!(
            self,
            SpecType::NgramSimple
                | SpecType::NgramMapK
                | SpecType::NgramMapK4v
                | SpecType::NgramMod
                | SpecType::NgramCache
        )
    }
}

str_enum! {
    /// Model loading strategy for `-lm` / `--load-mode`.
    ///
    /// Cerebro emits this and never the deprecated `--mlock` / `--mmap` / `--no-mmap` / `-dio`,
    /// which produce deprecation noise that the log parser would have to work around.
    LoadMode, BadLoadMode {
        Auto => "auto",
        Mmap => "mmap",
        Mlock => "mlock",
        MmapMlock => "mmap+mlock",
        None => "none",
        Dio => "dio",
    }
}

str_enum! {
    /// Multi-device split strategy for `-sm` / `--split-mode`. Validated at parse time upstream;
    /// `auto` is NOT accepted despite being a natural guess.
    SplitMode, BadSplitMode {
        None => "none",
        Layer => "layer",
        Row => "row",
        Tensor => "tensor",
    }
}

str_enum! {
    /// Flash-attention selection for `-fa` / `--flash-attn`.
    FlashAttn, BadFlashAttn {
        On => "on",
        Off => "off",
        Auto => "auto",
    }
}

/// A buffer type usable as the right-hand side of an `-ot` rule.
///
/// Verified accepted on b10751: `CPU` and `CUDA0`. Verified REJECTED as "unknown buffer type":
/// `CUDA_Host`, `CUDA0_Split`, `Vulkan0`. Note the asymmetry with [`DeviceId`] — `CPU` is legal
/// here and illegal there, which is exactly why the two are distinct types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BufferType {
    /// Host memory.
    Cpu,
    /// A specific backend device's memory.
    Device(DeviceId),
}

impl fmt::Display for BufferType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BufferType::Cpu => f.write_str("CPU"),
            BufferType::Device(d) => write!(f, "{d}"),
        }
    }
}

/// One `pattern=buffer_type` rule for `-ot` / `--override-tensor`.
///
/// Three verified hazards are enforced here:
///
/// 1. `=` and `,` are structurally reserved inside the value, so a pattern may contain neither.
///    That forbids regex bounded repetition (`{1,2}`) outright — enumerate per-layer patterns
///    instead, exactly as llama.cpp's own examples do.
/// 2. Matching is unanchored `regex_search` with first-match-wins, so rule ORDER is semantic.
///    [`OtRuleSet`] preserves it and refuses to reorder.
/// 3. An unescaped `.` is a regex wildcard. `blk.3.` matches `blk.13.` and `blk.30.`. Use
///    [`OtRule::layer`] to build correctly escaped per-layer patterns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtRule {
    pattern: String,
    buft: BufferType,
}

impl OtRule {
    /// Build a rule from a raw regex pattern, rejecting the structurally reserved characters.
    pub fn new(pattern: impl Into<String>, buft: BufferType) -> Result<Self, ValueError> {
        let pattern = pattern.into();
        if pattern.is_empty() {
            return Err(ValueError::EmptyOtPattern);
        }
        if let Some(bad) = pattern.chars().find(|&c| c == ',' || c == '=') {
            return Err(ValueError::ReservedInOtPattern { pattern, ch: bad });
        }
        Ok(Self { pattern, buft })
    }

    /// Build a correctly escaped rule targeting a single transformer block, e.g. layer 17 becomes
    /// the pattern `blk\.17\.`. This is the constructor the planner should use; it cannot produce
    /// the `blk.3.` wildcard bug.
    pub fn layer(index: u32, buft: BufferType) -> Self {
        Self {
            pattern: format!(r"blk\.{index}\."),
            buft,
        }
    }

    /// Build an escaped rule targeting one tensor group within one block, e.g.
    /// `(17, "ffn_gate_exps")` becomes `blk\.17\.ffn_gate_exps`.
    pub fn layer_tensor(index: u32, tensor: &str, buft: BufferType) -> Result<Self, ValueError> {
        if let Some(bad) = tensor.chars().find(|&c| c == ',' || c == '=') {
            return Err(ValueError::ReservedInOtPattern {
                pattern: tensor.to_string(),
                ch: bad,
            });
        }
        Ok(Self {
            pattern: format!(r"blk\.{index}\.{tensor}"),
            buft,
        })
    }

    /// The regex pattern.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// The destination buffer type.
    pub fn buffer_type(&self) -> &BufferType {
        &self.buft
    }
}

impl fmt::Display for OtRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.pattern, self.buft)
    }
}

/// An ordered set of `-ot` rules.
///
/// Order is preserved exactly, because upstream resolves first-match-wins. A set that sorted or
/// deduplicated its rules would silently change which tensors land where.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OtRuleSet(Vec<OtRule>);

impl OtRuleSet {
    /// An empty rule set.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Append a rule. Later rules are lower priority than earlier ones, so callers must add
    /// narrow rules before broad ones.
    pub fn push(&mut self, rule: OtRule) -> &mut Self {
        self.0.push(rule);
        self
    }

    /// The rules, in priority order.
    pub fn rules(&self) -> &[OtRule] {
        &self.0
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Detect rules that can never match because an earlier, broader rule shadows them.
    ///
    /// This is a conservative prefix check rather than full regex containment analysis: it reports
    /// a later rule whose pattern starts with an earlier rule's pattern. That catches the common
    /// planner mistake (broad `blk\.` before narrow `blk\.17\.ffn`) without pulling in a regex
    /// engine. It cannot catch every shadowing case and does not claim to.
    pub fn shadowed(&self) -> Vec<ShadowWarning> {
        let mut out = Vec::new();
        for (i, later) in self.0.iter().enumerate() {
            for (j, earlier) in self.0[..i].iter().enumerate() {
                if later.pattern.starts_with(&earlier.pattern) {
                    out.push(ShadowWarning {
                        shadowed_index: i,
                        shadowed_by_index: j,
                        shadowed: later.pattern.clone(),
                        shadowed_by: earlier.pattern.clone(),
                    });
                    break;
                }
            }
        }
        out
    }
}

/// A report that one `-ot` rule is unreachable because an earlier rule matches first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowWarning {
    /// Index of the rule that will never match.
    pub shadowed_index: usize,
    /// Index of the earlier rule that captures it.
    pub shadowed_by_index: usize,
    /// Pattern that will never match.
    pub shadowed: String,
    /// Pattern that captures it first.
    pub shadowed_by: String,
}

/// A `-ts` / `--tensor-split` proportion vector.
///
/// Upstream does NOT validate this at parse time — `-ts 1;1` with the wrong separator and `-ts 1`
/// on a two-device system both parse cleanly and fail later at model load. Cerebro validates here
/// so a malformed split is caught in milliseconds instead of after a soak test.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSplit(Vec<f32>);

impl TensorSplit {
    /// Build a split vector. Rejects empty vectors, negative or non-finite proportions, and
    /// all-zero vectors (which would assign the model to no device).
    pub fn new(parts: impl IntoIterator<Item = f32>) -> Result<Self, ValueError> {
        let parts: Vec<f32> = parts.into_iter().collect();
        if parts.is_empty() {
            return Err(ValueError::EmptyTensorSplit);
        }
        if parts.iter().any(|p| !p.is_finite() || *p < 0.0) {
            return Err(ValueError::BadTensorSplit);
        }
        if parts.iter().all(|p| *p == 0.0) {
            return Err(ValueError::BadTensorSplit);
        }
        Ok(Self(parts))
    }

    /// The proportions.
    pub fn parts(&self) -> &[f32] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_parses_and_canonicalizes() {
        let d = DeviceId::parse("cuda0").unwrap();
        assert_eq!(d.to_string(), "CUDA0");
        assert_eq!(d.backend(), "CUDA");
        assert_eq!(d.ordinal(), 0);
        assert_eq!(DeviceId::parse("RPC1").unwrap().to_string(), "RPC1");
    }

    #[test]
    fn cpu_is_not_an_addressable_device() {
        // Verified upstream: `-dev CPU` -> "invalid device: CPU".
        assert!(matches!(
            DeviceId::parse("CPU"),
            Err(ValueError::CpuNotAddressable)
        ));
        assert!(matches!(
            DeviceId::parse("cpu0"),
            Err(ValueError::CpuNotAddressable)
        ));
        // A backend name with no ordinal is bad syntax, not the CPU case.
        assert!(matches!(
            DeviceId::parse("CUDA"),
            Err(ValueError::DeviceSyntax(_))
        ));
        assert!(matches!(DeviceId::parse(""), Err(ValueError::DeviceSyntax(_))));
    }

    #[test]
    fn cpu_is_a_legal_ot_buffer_type() {
        // The asymmetry that motivates two distinct types.
        let rule = OtRule::new(r"blk\..*ffn.*", BufferType::Cpu).unwrap();
        assert_eq!(rule.to_string(), r"blk\..*ffn.*=CPU");
    }

    #[test]
    fn device_list_rejects_empty() {
        assert!(matches!(
            DeviceList::devices(vec![]),
            Err(ValueError::EmptyDeviceList)
        ));
    }

    #[test]
    fn cpu_mask_renders_as_bare_hex() {
        // The reference machine's physical-only P-core mask and its E-core mask.
        assert_eq!(CpuMask::new(0x555).unwrap().to_string(), "555");
        assert_eq!(CpuMask::new(0xFF000).unwrap().to_string(), "ff000");
        assert_eq!(CpuMask::new(0x555).unwrap().count(), 6);
        assert_eq!(CpuMask::new(0xFFF).unwrap().count(), 12);
        assert!(matches!(CpuMask::new(0), Err(ValueError::EmptyCpuMask)));
    }

    #[test]
    fn cache_types_are_exactly_the_nine_upstream_accepts() {
        assert_eq!(CacheType::ALL.len(), 9);
        for t in ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"] {
            assert_eq!(CacheType::parse(t).unwrap().as_str(), t);
        }
        assert!(CacheType::parse("q6_K").is_err());
        assert!(CacheType::parse("F16").is_err(), "upstream is case-sensitive here");
    }

    #[test]
    fn spec_types_are_exactly_the_eleven_upstream_accepts() {
        assert_eq!(SpecType::ALL.len(), 11);
        // Four strictly require a separate draft model; draft-mtp may instead use heads embedded
        // in the target GGUF, so it accepts one without requiring it.
        assert_eq!(
            SpecType::ALL.iter().filter(|s| s.requires_draft_model()).count(),
            4
        );
        assert_eq!(
            SpecType::ALL.iter().filter(|s| s.accepts_draft_model()).count(),
            5
        );
        assert!(SpecType::DraftMtp.accepts_draft_model());
        assert!(!SpecType::DraftMtp.requires_draft_model());
        assert_eq!(SpecType::ALL.iter().filter(|s| s.is_ngram()).count(), 5);
        // `none` is neither; every other variant is exactly one of the two families.
        assert!(!SpecType::None.accepts_draft_model() && !SpecType::None.is_ngram());
    }

    #[test]
    fn split_mode_rejects_auto() {
        // A natural guess that upstream rejects at parse time.
        assert!(SplitMode::parse("auto").is_err());
        assert!(SplitMode::parse("layer").is_ok());
    }

    #[test]
    fn ot_pattern_rejects_structurally_reserved_chars() {
        // Bounded repetition contains a comma and would silently truncate the value.
        assert!(matches!(
            OtRule::new(r"blk\.\d{1,2}\.", BufferType::Cpu),
            Err(ValueError::ReservedInOtPattern { ch: ',', .. })
        ));
        assert!(matches!(
            OtRule::new("a=b", BufferType::Cpu),
            Err(ValueError::ReservedInOtPattern { ch: '=', .. })
        ));
    }

    #[test]
    fn layer_constructor_escapes_dots() {
        // Unescaped `blk.3.` would also match blk.13. and blk.30.
        let r = OtRule::layer(3, BufferType::Cpu);
        assert_eq!(r.pattern(), r"blk\.3\.");
        let r = OtRule::layer_tensor(17, "ffn_gate_exps", BufferType::Cpu).unwrap();
        assert_eq!(r.pattern(), r"blk\.17\.ffn_gate_exps");
    }

    #[test]
    fn shadowing_is_detected_in_priority_order() {
        let mut set = OtRuleSet::new();
        set.push(OtRule::new(r"blk\.", BufferType::Cpu).unwrap());
        set.push(OtRule::layer(17, BufferType::Cpu));
        let warns = set.shadowed();
        assert_eq!(warns.len(), 1);
        assert_eq!(warns[0].shadowed_index, 1);
        assert_eq!(warns[0].shadowed_by_index, 0);

        // Narrow-before-broad is the correct order and raises nothing.
        let mut ok = OtRuleSet::new();
        ok.push(OtRule::layer(17, BufferType::Cpu));
        ok.push(OtRule::new(r"blk\.", BufferType::Cpu).unwrap());
        assert!(ok.shadowed().is_empty());
    }

    #[test]
    fn tensor_split_validates_what_upstream_does_not() {
        assert!(TensorSplit::new([1.0, 1.0]).is_ok());
        assert!(TensorSplit::new([]).is_err());
        assert!(TensorSplit::new([0.0, 0.0]).is_err());
        assert!(TensorSplit::new([1.0, -1.0]).is_err());
        assert!(TensorSplit::new([1.0, f32::NAN]).is_err());
    }
}
