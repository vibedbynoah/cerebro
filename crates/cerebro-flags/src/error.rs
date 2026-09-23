//! Error types for value construction and argv emission.

use thiserror::Error;

/// A value that cannot be represented because llama.cpp would reject it, or because upstream would
/// silently accept it and fail later at model load.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ValueError {
    /// A device identifier that is not `BACKEND` + ordinal.
    #[error("not a valid device identifier: {0:?} (expected a backend name followed by an ordinal, e.g. CUDA0)")]
    DeviceSyntax(String),

    /// `CPU` was used where an addressable device was required.
    ///
    /// Verified: `-dev CPU` fails with `invalid device: CPU`. Route CPU placement through `-ot`
    /// with buffer type `CPU`, through `-cmoe`/`-ncmoe`, or through `-dev none`.
    #[error("CPU is not an addressable offload device; use an -ot rule with buffer type CPU, -ncmoe, or -dev none")]
    CpuNotAddressable,

    /// An explicit device list with no devices in it.
    #[error("device list is empty; use the `none` sentinel to disable offload rather than an empty list")]
    EmptyDeviceList,

    /// An affinity mask selecting no logical processor.
    #[error("CPU affinity mask is zero, which would pin the process to no processor")]
    EmptyCpuMask,

    /// An `-ot` pattern containing a character that is structurally reserved in the flag value.
    #[error("-ot pattern {pattern:?} contains {ch:?}, which is reserved as a separator in the flag value; enumerate per-layer patterns instead of using bounded repetition")]
    ReservedInOtPattern {
        /// The offending pattern.
        pattern: String,
        /// The reserved character found in it.
        ch: char,
    },

    /// An empty `-ot` pattern, which would match every tensor.
    #[error("-ot pattern is empty, which would match every tensor")]
    EmptyOtPattern,

    /// An unrecognized KV cache type.
    #[error("unsupported cache type: {0:?} (allowed: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1)")]
    BadCacheType(String),

    /// An unrecognized speculative decoding strategy.
    #[error("unsupported spec type: {0:?}")]
    BadSpecType(String),

    /// An unrecognized load mode.
    #[error("unsupported load mode: {0:?} (allowed: auto, mmap, mlock, mmap+mlock, none, dio)")]
    BadLoadMode(String),

    /// An unrecognized split mode. `auto` is a common wrong guess.
    #[error("unsupported split mode: {0:?} (allowed: none, layer, row, tensor — note that `auto` is not accepted)")]
    BadSplitMode(String),

    /// An unrecognized flash-attention mode.
    #[error("unsupported flash-attn mode: {0:?} (allowed: on, off, auto)")]
    BadFlashAttn(String),

    /// An empty tensor-split vector.
    #[error("tensor split is empty")]
    EmptyTensorSplit,

    /// A tensor-split vector that is negative, non-finite, or entirely zero.
    #[error("tensor split must contain finite non-negative proportions with at least one non-zero")]
    BadTensorSplit,
}

/// A failure to emit an argv for a particular backend tool.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EmitError {
    /// A flag was requested for a tool whose dialect has no such flag.
    ///
    /// The motivating case is speculation: `llama-bench` has no `--spec-*` surface at all, so the
    /// entire speculation axis must be driven through `llama-server`. Catching this at emission
    /// time turns a silently-ignored flag into a hard error.
    #[error("flag {flag} is not available in the {dialect} dialect: {reason}")]
    NotInDialect {
        /// The flag that was requested.
        flag: &'static str,
        /// The dialect it was requested for.
        dialect: &'static str,
        /// Why it is unavailable.
        reason: &'static str,
    },

    /// A flag was emitted more than once where upstream takes only the last occurrence.
    #[error("flag {0} was set more than once; the duplicate would silently override the first")]
    Duplicate(&'static str),

    /// A path argument that cannot be represented as UTF-8 argv.
    #[error("path is not valid UTF-8 and cannot be placed in an argv: {0}")]
    NonUtf8Path(String),

    /// A combination of individually-valid flags that violates a hard cross-flag rule.
    #[error("cross-flag rule {rule} violated: {detail} -- {remedy}")]
    ConstraintViolated {
        /// Stable rule identifier.
        rule: &'static str,
        /// What is wrong with this configuration.
        detail: String,
        /// What to do about it.
        remedy: String,
    },

    /// A path argument that is relative, where an absolute path is required.
    ///
    /// Relative paths in a persisted configuration resolve against whatever the working directory
    /// happens to be when the config is later replayed, which makes a recorded benchmark result
    /// non-reproducible and can point a model load somewhere unintended.
    #[error("path must be absolute, got {0:?}")]
    RelativePath(String),

    /// A flag whose value must come from a verified, out-of-band source (e.g. a preflighted launch)
    /// was already present on the builder handed to the function that owns emitting it.
    ///
    /// The motivating case is `cerebro_supervisor::build_hardened_server_argv`: every path flag it
    /// emits (`-m`, `-md`, `--api-key-file`, `--host`, `--port`) was independently verified by
    /// preflight, so that function must be the only place those flags are set. `ArgvBuilder::push`'s
    /// ordinary duplicate check would still catch a second, later push of the same flag, but it
    /// cannot stop a caller from pre-loading the builder with an unverified value before handing it
    /// over — this variant is for that upstream refusal instead.
    #[error("flag {0} must be set by the caller that owns it, not pre-set on the builder")]
    PreflightOwned(&'static str),

    /// A caller-supplied [`crate::ArgvBuilder`] was built for the wrong dialect.
    ///
    /// The motivating case is a hardening helper that always emits server-only flags (`--host`,
    /// `--api-key-file`, ...): those calls would themselves fail with [`EmitError::NotInDialect`]
    /// against a `Bench` builder, but only once `render` ran, and a `debug_assert_eq!` guarding the
    /// same mistake compiles out of a release build. Checking [`crate::ArgvBuilder::dialect`] up
    /// front turns that into a typed, always-enforced error.
    #[error("expected an ArgvBuilder for the {expected} dialect, got one for {actual}")]
    WrongDialect {
        /// The dialect the caller required.
        expected: &'static str,
        /// The dialect the supplied builder was actually constructed for.
        actual: &'static str,
    },
}
