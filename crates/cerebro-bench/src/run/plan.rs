//! What one run measures, and the exact `llama-bench` argv it becomes.
//!
//! The argv has two halves. The configuration under test (`-m` plus the caller's flags) goes
//! through [`cerebro_flags::ArgvBuilder`] in the Bench dialect, so it is the closed allowlist, with
//! llama-bench's own separators, duplicate refusal and cross-flag rules. The measurement protocol
//! (`-p`, `-n`, `-d`, `-r`, `--delay`, `-o json`) is not configuration and not tuner-steerable: it
//! is built here from a validated [`RunPlan`] of integers only, so no value in it can be read as a
//! flag or a second sweep entry beyond the depth list, which is the sweep by design.
//!
//! An empty configuration measures *llama-bench's* defaults, which are not llama.cpp's served
//! automatic settings: llama-bench's memory fitter is off (`-fitt` "default: off", `fit_target: 0`
//! in its JSON) and `-ngl -1` offloads every layer, so a model that does not fit VRAM fails or
//! spills to system memory. llama-server fits by default (`--fit on`, `--fit-target 1024`). The
//! baseline every tuned configuration must beat is the latter:
//! [`super::BenchRequest::server_default_baseline`].

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

use cerebro_flags::{ArgvBuilder, Dialect, Flag};

use super::error::{PlanError, RunError};
use crate::stats::{MAX_REPETITIONS, MIN_KEPT_SAMPLES};

/// Most distinct depths one run may sweep. Each depth adds a prefill and a decode test; a list
/// longer than this is a mistake, not a measurement plan.
pub const MAX_DEPTHS: usize = 16;

/// Timeout used when the caller sets none: generous enough for a large MoE that does not fit
/// VRAM at a deep context, short enough that a hung run cannot hold the machine all day.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60 * 60);

/// Longest timeout a plan accepts.
pub const MAX_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);

/// Seed for the bootstrap confidence interval when the caller sets none, so the same records
/// always summarize to the same numbers.
pub const DEFAULT_SEED: u64 = 0x0cee_b00b_5eed_0001;

/// A validated measurement plan: how many repetitions, which test sizes, which depths.
///
/// Fields are private and every constructor validates, so a plan that exists is runnable: enough
/// kept repetitions for a summary, a prefill and a decode test, depth 0 and at least one deeper
/// depth (the three-number rule in `docs/CLAIMS.md`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunPlan {
    kept_repetitions: u32,
    prompt_tokens: u32,
    gen_tokens: u32,
    depths: Vec<u32>,
    delay_secs: u32,
    timeout: Duration,
    seed: u64,
}

impl RunPlan {
    /// A plan keeping `kept_repetitions` samples per test (llama-bench runs one more; the first
    /// is always discarded), with `-p prompt_tokens`, `-n gen_tokens` and the depth list `depths`
    /// (sorted and de-duplicated here). Timeout [`DEFAULT_TIMEOUT`], no delay, seed
    /// [`DEFAULT_SEED`].
    ///
    /// The depth list is consumed lazily and refused as soon as it holds more than
    /// [`MAX_DEPTHS`] distinct values, so an unbounded iterator costs nothing.
    pub fn new(
        kept_repetitions: u32,
        prompt_tokens: u32,
        gen_tokens: u32,
        depths: impl IntoIterator<Item = u32>,
    ) -> Result<Self, PlanError> {
        let min = u32::try_from(MIN_KEPT_SAMPLES).unwrap_or(u32::MAX);
        if kept_repetitions < min {
            return Err(PlanError::TooFewRepetitions {
                kept: kept_repetitions,
                min,
            });
        }
        let total = u64::from(kept_repetitions) + 1;
        let max = u64::try_from(MAX_REPETITIONS).unwrap_or(u64::MAX);
        if total > max {
            return Err(PlanError::TooManyRepetitions { total, max });
        }
        if prompt_tokens == 0 {
            return Err(PlanError::ZeroPromptTokens);
        }
        if gen_tokens == 0 {
            return Err(PlanError::ZeroGenTokens);
        }
        let mut distinct = BTreeSet::new();
        for depth in depths {
            distinct.insert(depth);
            if distinct.len() > MAX_DEPTHS {
                return Err(PlanError::TooManyDepths {
                    count: distinct.len(),
                    max: MAX_DEPTHS,
                });
            }
        }
        // Ascending and de-duplicated already.
        let depths: Vec<u32> = distinct.into_iter().collect();
        if depths.first() != Some(&0) {
            return Err(PlanError::NoDepthZero);
        }
        if depths.len() < 2 {
            return Err(PlanError::NoDepthSweep);
        }
        Ok(Self {
            kept_repetitions,
            prompt_tokens,
            gen_tokens,
            depths,
            delay_secs: 0,
            timeout: DEFAULT_TIMEOUT,
            seed: DEFAULT_SEED,
        })
    }

    /// Pause `secs` seconds before each test (`--delay`), letting a thermally limited laptop cool
    /// so later tests are not measured hotter than earlier ones. 0 (the default) emits no flag.
    ///
    /// # Errors
    /// [`PlanError::DelayExceedsTimeout`] when the delays alone (`secs` before each of the
    /// [`RunPlan::expected_records`] tests) would use up the timeout, which guarantees a timeout.
    pub fn with_delay_secs(mut self, secs: u32) -> Result<Self, PlanError> {
        check_delay(secs, self.expected_records(), self.timeout)?;
        self.delay_secs = secs;
        Ok(self)
    }

    /// Kill the run's process tree if it has not finished after `timeout`.
    ///
    /// # Errors
    /// [`PlanError::TimeoutOutOfRange`] for zero or above [`MAX_TIMEOUT`];
    /// [`PlanError::DelayExceedsTimeout`] when the plan's delays would use it up.
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self, PlanError> {
        if timeout.is_zero() || timeout > MAX_TIMEOUT {
            return Err(PlanError::TimeoutOutOfRange {
                got: timeout,
                max: MAX_TIMEOUT,
            });
        }
        check_delay(self.delay_secs, self.expected_records(), timeout)?;
        self.timeout = timeout;
        Ok(self)
    }

    /// Seed the bootstrap confidence interval.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Samples kept per test after the first is discarded.
    pub fn kept_repetitions(&self) -> u32 {
        self.kept_repetitions
    }

    /// What `-r` carries: the kept repetitions plus the discarded first one.
    pub fn total_repetitions(&self) -> u32 {
        self.kept_repetitions.saturating_add(1)
    }

    /// `-p`: prompt tokens per prefill test.
    pub fn prompt_tokens(&self) -> u32 {
        self.prompt_tokens
    }

    /// `-n`: tokens generated per decode test.
    pub fn gen_tokens(&self) -> u32 {
        self.gen_tokens
    }

    /// `-d`: the depth list, ascending, starting at 0, at least two entries.
    pub fn depths(&self) -> &[u32] {
        &self.depths
    }

    /// The deepest depth, where the third triple member is measured.
    pub fn deepest_depth(&self) -> u32 {
        self.depths.last().copied().unwrap_or(0)
    }

    /// `--delay` seconds between tests; 0 emits no flag.
    pub fn delay_secs(&self) -> u32 {
        self.delay_secs
    }

    /// The run's timeout.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// The bootstrap seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// How many records a successful run returns: one prefill and one decode test per depth.
    pub fn expected_records(&self) -> usize {
        self.depths.len().saturating_mul(2)
    }
}

/// Refuse a delay whose total (llama-bench sleeps `secs` before each of `tests` tests) is not
/// below `timeout`: such a run cannot finish.
fn check_delay(secs: u32, tests: usize, timeout: Duration) -> Result<(), PlanError> {
    let total = Duration::from_secs(u64::from(secs))
        .saturating_mul(u32::try_from(tests).unwrap_or(u32::MAX));
    if secs > 0 && total >= timeout {
        return Err(PlanError::DelayExceedsTimeout {
            delay_total: total,
            timeout,
        });
    }
    Ok(())
}

/// The exact `llama-bench` argv for measuring `config` on `model` under `plan`.
///
/// Order: `-m <model>`, the configuration flags in the order given, then
/// `-p P -n N -d 0,D1,.. -r R [--delay S] -o json`. `model` should be the preflighted path
/// ([`cerebro_supervisor::BenchPreflighted::model_path`]), never the caller's own spelling.
///
/// # Errors
/// [`RunError::ModelPathNotUtf8`] for a non-UTF-8 model path; [`RunError::Argv`] for a flag
/// llama-bench does not have, a duplicate (including a `Flag::Model` in `config`), a relative model
/// path, or a hard cross-flag rule violation.
pub fn build_argv(model: &Path, config: &[Flag], plan: &RunPlan) -> Result<Vec<String>, RunError> {
    let model = model
        .to_str()
        .ok_or_else(|| RunError::ModelPathNotUtf8(model.to_path_buf()))?;
    let mut builder = ArgvBuilder::new(Dialect::Bench);
    builder.push(Flag::Model(model.to_string()))?;
    for flag in config {
        builder.push(flag.clone())?;
    }
    let mut argv = builder.build()?;

    let depths = plan
        .depths()
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    argv.extend([
        "-p".to_string(),
        plan.prompt_tokens().to_string(),
        "-n".to_string(),
        plan.gen_tokens().to_string(),
        "-d".to_string(),
        depths,
        "-r".to_string(),
        plan.total_repetitions().to_string(),
    ]);
    if plan.delay_secs() > 0 {
        argv.extend(["--delay".to_string(), plan.delay_secs().to_string()]);
    }
    argv.extend(["-o".to_string(), "json".to_string()]);
    Ok(argv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerebro_flags::value::{CacheType, CpuMask, DeviceId, DeviceList, FlashAttn};
    use cerebro_flags::EmitError;

    const MODEL: &str = r"C:\models\qwen3-coder-30b-a3b-q4_k_m.gguf";

    fn plan() -> RunPlan {
        RunPlan::new(3, 64, 32, [128, 0]).expect("valid plan")
    }

    fn argv(config: &[Flag], plan: &RunPlan) -> Vec<String> {
        build_argv(Path::new(MODEL), config, plan).expect("valid argv")
    }

    #[test]
    fn llama_cpp_defaults_emit_only_the_model_and_the_protocol() {
        assert_eq!(
            argv(&[], &plan()),
            [
                "-m", MODEL, "-p", "64", "-n", "32", "-d", "0,128", "-r", "4", "-o", "json"
            ]
        );
    }

    #[test]
    fn the_cpu_only_live_test_configuration_emits_exactly() {
        let plan = plan().with_delay_secs(2).unwrap();
        assert_eq!(
            argv(&[Flag::Device(DeviceList::None)], &plan),
            [
                "-m", MODEL, "-dev", "none", "-p", "64", "-n", "32", "-d", "0,128", "-r", "4",
                "--delay", "2", "-o", "json"
            ]
        );
    }

    #[test]
    fn a_tuned_moe_configuration_uses_bench_separators_and_keeps_flag_order() {
        let gpu = DeviceList::devices([DeviceId::parse("CUDA0").unwrap()]).unwrap();
        let config = [
            Flag::GpuLayers(999),
            Flag::NCpuMoe(32),
            Flag::CacheTypeK(CacheType::Q8_0),
            Flag::CacheTypeV(CacheType::Q8_0),
            Flag::FlashAttn(FlashAttn::On),
            Flag::Threads(6),
            Flag::CpuMask(CpuMask::new(0x555).unwrap()),
            Flag::CpuStrict(true),
            Flag::KvOffload(true),
            Flag::Device(gpu),
        ];
        let plan = RunPlan::new(7, 512, 128, [0, 4096, 16384]).unwrap();
        assert_eq!(
            argv(&config, &plan),
            [
                "-m", MODEL, "-ngl", "999", "-ncmoe", "32", "-ctk", "q8_0", "-ctv", "q8_0", "-fa",
                "on", "-t", "6", "-C", "555", "--cpu-strict", "1", "-nkvo", "0", "-dev", "CUDA0",
                "-p", "512", "-n", "128", "-d", "0,4096,16384", "-r", "8", "-o", "json"
            ]
        );
    }

    #[test]
    fn the_server_default_baseline_fits_like_llama_server() {
        let baseline = super::super::server_default_baseline_config();
        assert_eq!(
            argv(&baseline, &plan()),
            [
                "-m", MODEL, "-fitt", "1024", "-p", "64", "-n", "32", "-d", "0,128", "-r", "4",
                "-o", "json"
            ]
        );
        // A tuned configuration places the model itself; the fitter beside it is refused.
        let mut tuned = baseline.clone();
        tuned.push(Flag::GpuLayers(999));
        let err = build_argv(Path::new(MODEL), &tuned, &plan()).expect_err("fit and -ngl fight");
        assert!(
            matches!(
                err,
                RunError::Argv(EmitError::ConstraintViolated {
                    rule: "fit_with_explicit_offload_plan",
                    ..
                })
            ),
            "{err:?}"
        );
    }

    #[test]
    fn depths_are_sorted_and_deduplicated() {
        let plan = RunPlan::new(3, 64, 32, [4096, 0, 512, 4096, 0]).unwrap();
        assert_eq!(plan.depths(), &[0, 512, 4096]);
        assert_eq!(plan.deepest_depth(), 4096);
        assert_eq!(plan.expected_records(), 6);
    }

    #[test]
    fn server_only_and_duplicate_flags_are_refused_before_anything_runs() {
        let err = build_argv(Path::new(MODEL), &[Flag::CtxSize(4096)], &plan())
            .expect_err("-c does not exist in llama-bench");
        assert!(
            matches!(err, RunError::Argv(EmitError::NotInDialect { flag: "-c", .. })),
            "{err:?}"
        );
        let err = build_argv(Path::new(MODEL), &[Flag::Model(MODEL.into())], &plan())
            .expect_err("-m comes from preflight only");
        assert!(matches!(err, RunError::Argv(EmitError::Duplicate("-m"))), "{err:?}");
        let err = build_argv(Path::new("models/x.gguf"), &[], &plan())
            .expect_err("a relative model path is refused");
        assert!(matches!(err, RunError::Argv(EmitError::RelativePath(_))), "{err:?}");
    }

    #[test]
    fn invalid_plans_are_unrepresentable() {
        assert_eq!(
            RunPlan::new(2, 64, 32, [0, 128]),
            Err(PlanError::TooFewRepetitions { kept: 2, min: 3 })
        );
        assert_eq!(
            RunPlan::new(1000, 64, 32, [0, 128]),
            Err(PlanError::TooManyRepetitions { total: 1001, max: 1000 })
        );
        assert!(RunPlan::new(999, 64, 32, [0, 128]).is_ok());
        assert_eq!(RunPlan::new(3, 0, 32, [0, 128]), Err(PlanError::ZeroPromptTokens));
        assert_eq!(RunPlan::new(3, 64, 0, [0, 128]), Err(PlanError::ZeroGenTokens));
        assert_eq!(RunPlan::new(3, 64, 32, [128, 256]), Err(PlanError::NoDepthZero));
        assert_eq!(RunPlan::new(3, 64, 32, [0, 0]), Err(PlanError::NoDepthSweep));
        assert_eq!(
            RunPlan::new(3, 64, 32, 0..17),
            Err(PlanError::TooManyDepths { count: 17, max: 16 })
        );
        // An unbounded depth iterator is refused at the 17th distinct value, without collecting.
        assert_eq!(
            RunPlan::new(3, 64, 32, 0..u32::MAX),
            Err(PlanError::TooManyDepths { count: 17, max: 16 })
        );
        // 4 tests x 30 s of delay is 2 minutes: refused under a 2-minute timeout, either order.
        let two_min = Duration::from_secs(120);
        let short = plan().with_timeout(two_min).unwrap();
        assert_eq!(
            short.clone().with_delay_secs(30),
            Err(PlanError::DelayExceedsTimeout { delay_total: two_min, timeout: two_min })
        );
        assert_eq!(short.clone().with_delay_secs(29).unwrap().delay_secs(), 29);
        assert!(short.with_delay_secs(u32::MAX).is_err(), "no overflow on a huge delay");
        let delayed = plan().with_delay_secs(30).unwrap();
        assert_eq!(
            delayed.clone().with_timeout(two_min),
            Err(PlanError::DelayExceedsTimeout { delay_total: two_min, timeout: two_min })
        );
        assert!(delayed.with_timeout(Duration::from_secs(121)).is_ok());
        assert!(plan().with_timeout(Duration::ZERO).is_err());
        assert!(plan().with_timeout(MAX_TIMEOUT + Duration::from_secs(1)).is_err());
        assert_eq!(
            plan().with_timeout(Duration::from_secs(5)).unwrap().timeout(),
            Duration::from_secs(5)
        );
    }
}
