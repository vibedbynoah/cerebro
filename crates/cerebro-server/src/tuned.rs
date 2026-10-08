//! Binding a tuned recommendation to exactly what is about to be served.
//!
//! A measurement is only evidence about the configuration that was measured: the same model file,
//! the same backend build, the same machine and the same flags. [`verify`] checks each of those
//! against the store record the recommendation names, in this order, and refuses with
//! [`ServerError::TunedMismatch`] at the first difference:
//!
//! 1. the model file's SHA-256 against the record's `model_sha256`;
//! 2. the backend manifest digest ([`cerebro_bench::run::backend_manifest_digest`] of the pinned
//!    manifest) against the record's `backend_manifest_digest`;
//! 3. the flags against the record's measured llama-bench argv: the server flags are mapped to the
//!    llama-bench flags that measure the same configuration ([`bench_equivalent`]), rendered through
//!    the closed allowlist in the Bench dialect, and must equal the record's argv token for token
//!    (between its `-m <model>` and its run plan). The record's argv is only compared, never run;
//! 4. this machine's profile id against the record's `machine_profile_id` (probed last, so the
//!    cheaper checks refuse first).
//!
//! The flags served are the caller's minus a bare `-t <n>` (a thread count with no CPU mask): the
//! record cannot show which count llama-bench's default picked, so llama-server's own default is
//! served, which is the same default llama-bench measured with.

use std::collections::BTreeMap;
use std::path::Path;

use cerebro_bench::run::{backend_manifest_digest, gpu_summaries, machine_profile_id, CpuSummary, ModelHashCache};
use cerebro_flags::value::{CacheType, FlashAttn};
use cerebro_flags::{ArgvBuilder, Dialect, Flag};
use cerebro_store::Record;

use crate::error::ServerError;

/// llama-bench's and llama-server's default `--fit-ctx` (both `--help`: "minimum ctx size for
/// --fit-target (default: 4096)"). A served `--fit-ctx 4096` therefore matches a measurement that
/// did not pass `-fitc`.
pub const LLAMA_DEFAULT_FIT_CTX: u32 = 4096;

fn mismatch(what: impl Into<String>) -> ServerError {
    ServerError::TunedMismatch { what: what.into() }
}

/// The llama-bench flags that measure the configuration `flags` serves, and the flags to serve.
///
/// Mapping (the inverse of `cerebro_tune::CandidateConfig::{bench_flags, server_flags}`):
/// `--fit on/off` and `--fit-ctx 4096` emit nothing (llama-bench fits only when given `-fitt`, with
/// a 4096 fit context); `-fitt`, `-ngl`, `-ncmoe`, `-ot`, `-C` and `--cpu-strict` are the same in
/// both tools; `-fa auto` with an f16 KV cache is both tools' default and emits nothing, any other
/// KV setting is the same in both; `-t` emits only together with a CPU mask (see the module docs).
/// When the measurement fitted nothing (no `-fitt`), the served flags must turn llama-server's
/// fitter off, as llama-bench's was.
///
/// # Errors
/// [`ServerError::TunedMismatch`] for a flag that is not part of any measured configuration, or a
/// fitter left on for an explicit placement.
pub fn bench_equivalent(flags: &[Flag]) -> Result<(Vec<Flag>, Vec<Flag>), ServerError> {
    let flash_auto = !flags.iter().any(|f| matches!(f, Flag::FlashAttn(fa) if *fa != FlashAttn::Auto));
    let pinned = flags.iter().any(|f| matches!(f, Flag::CpuMask(_)));
    let mut bench = Vec::new();
    let mut served = Vec::new();
    for flag in flags {
        let (to_bench, to_serve) = match flag {
            Flag::Fit(_) => (false, true),
            Flag::FitCtx(c) if c.get() == LLAMA_DEFAULT_FIT_CTX => (false, true),
            Flag::FitCtx(_) => (true, true),
            Flag::FitTarget(_) | Flag::GpuLayers(_) | Flag::NCpuMoe(_) | Flag::OverrideTensor(_) => (true, true),
            Flag::CpuMask(_) | Flag::CpuStrict(_) => (true, true),
            Flag::FlashAttn(FlashAttn::Auto) => (false, true),
            Flag::FlashAttn(_) => (true, true),
            Flag::CacheTypeK(CacheType::F16) | Flag::CacheTypeV(CacheType::F16) if flash_auto => (false, true),
            Flag::CacheTypeK(_) | Flag::CacheTypeV(_) => (true, true),
            Flag::Threads(_) if pinned => (true, true),
            Flag::Threads(_) => (false, false),
            other => return Err(mismatch(format!("{} is not part of any measured configuration", other.name()))),
        };
        if to_bench {
            bench.push(flag.clone());
        }
        if to_serve {
            served.push(flag.clone());
        }
    }
    let fitted = bench.iter().any(|f| matches!(f, Flag::FitTarget(_)));
    if !fitted && !served.contains(&Flag::Fit(false)) {
        return Err(mismatch("the flags leave llama-server's fitter on, but the measurement placed the model without it"));
    }
    Ok((bench, served))
}

/// Whether `bench` rendered in the Bench dialect is exactly the configuration part of the record's
/// measured argv: `-m <model>`, then those tokens, then the run plan (`-p ...`).
fn argv_matches(bench: &[Flag], measured: &[String]) -> Result<bool, ServerError> {
    let (Some("-m"), Some(model)) = (measured.first().map(String::as_str), measured.get(1)) else {
        return Ok(false);
    };
    let mut builder = ArgvBuilder::new(Dialect::Bench);
    for flag in std::iter::once(Flag::Model(model.clone())).chain(bench.iter().cloned()) {
        builder.push(flag).map_err(|e| mismatch(format!("the flags are not a valid llama-bench configuration: {e}")))?;
    }
    let expected = builder.build().map_err(|e| mismatch(format!("the flags are not a valid llama-bench configuration: {e}")))?;
    Ok(measured.len() > expected.len()
        && measured[..expected.len()] == expected[..]
        && measured[expected.len()] == "-p")
}

/// Check `record` against what is about to be served (see the module docs for the order) and
/// return the flags to serve. `machine_id` is called only when every cheaper check passed.
///
/// # Errors
/// [`ServerError::TunedMismatch`] naming the first difference, or whatever `machine_id` returns.
pub fn verify(
    record: &Record,
    model_sha256: &str,
    backend_digest: &str,
    flags: &[Flag],
    machine_id: impl FnOnce() -> Result<String, ServerError>,
) -> Result<Vec<Flag>, ServerError> {
    let provenance = record.provenance();
    if provenance.model_sha256().as_str() != model_sha256 {
        return Err(mismatch("the record measured a different model file"));
    }
    if provenance.backend_manifest_digest().as_str() != backend_digest {
        return Err(mismatch("the record measured a different backend build"));
    }
    let (bench, served) = bench_equivalent(flags)?;
    if !argv_matches(&bench, record.measured_argv_untrusted())? {
        return Err(mismatch("the flags differ from the configuration the record measured"));
    }
    if provenance.machine_profile_id().as_str() != machine_id()? {
        return Err(mismatch("the record was measured on another machine profile"));
    }
    Ok(served)
}

/// SHA-256 of the model at `path` (cached by file version in `hashes`). Blocking.
///
/// # Errors
/// [`ServerError::TunedUnverifiable`] when the file cannot be read or changes while it is hashed.
pub fn model_sha256(path: &Path, hashes: &ModelHashCache) -> Result<String, ServerError> {
    hashes
        .sha256(path)
        .map(|(sha, _)| sha)
        .map_err(|e| ServerError::TunedUnverifiable { what: format!("the model could not be hashed: {e}") })
}

/// The digest of the backend build the launch will verify against.
pub fn manifest_digest(manifest: &BTreeMap<String, String>) -> String {
    backend_manifest_digest(manifest)
}

/// This machine's profile id, as `cerebro-bench` keys measurements. Blocking (probes the CPU and
/// queries the GPU vendor tool).
///
/// # Errors
/// [`ServerError::TunedUnverifiable`] when the CPU or GPU cannot be probed.
pub fn current_machine_profile_id() -> Result<String, ServerError> {
    let unverifiable = |what: String| ServerError::TunedUnverifiable { what };
    let cpu = cerebro_probe::cpu::probe().map_err(|e| unverifiable(format!("the CPU could not be probed: {e}")))?;
    let gpus = cerebro_probe::gpu::probe().map_err(|e| unverifiable(format!("the GPUs could not be probed: {e}")))?;
    let summary = CpuSummary::new(&cpu, cerebro_probe::cpu::brand_string());
    Ok(machine_profile_id(&summary, &gpu_summaries(&gpus)))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use cerebro_flags::value::CpuMask;
    use cerebro_store::{MeasurementSummary, MeasurementTriple, ProvenanceKey, SamplerProfile};
    use std::num::NonZeroU32;

    pub(crate) fn hex(c: char) -> String {
        std::iter::repeat_n(c, 64).collect()
    }

    fn nz(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).expect("non-zero")
    }

    /// A record whose measured argv is `bench` in the Bench dialect plus a run plan.
    pub(crate) fn record(bench: &[Flag]) -> Record {
        record_with_tg(bench, 40.0, 4096)
    }

    /// [`record`] with decode at depth `depth` measured at a median of `tg` tok/s.
    pub(crate) fn record_with_tg(bench: &[Flag], tg: f64, depth: u64) -> Record {
        let mut builder = ArgvBuilder::new(Dialect::Bench);
        builder.push(Flag::Model(r"C:\models\m.gguf".into())).expect("model");
        for f in bench {
            builder.push(f.clone()).expect("flag");
        }
        let mut argv = builder.build().expect("argv");
        argv.extend(["-p", "512", "-n", "128", "-d", "0,4096", "-r", "4", "-o", "json"].map(String::from));
        let s = MeasurementSummary::new(3, 40.0, 39.0, 41.0, 1.0, 39.5, 40.5).expect("summary");
        let deep = MeasurementSummary::new(3, tg, tg - 1.0, tg + 1.0, 1.0, tg - 0.5, tg + 0.5).expect("summary");
        let triple = MeasurementTriple::new(s, s, deep, depth).expect("triple");
        let key = ProvenanceKey::new(hex('a'), hex('b'), hex('c'), Some(1), Some("abc".into()), None, None, Some(4000)).expect("key");
        let sampler = SamplerProfile::new(0.7, 40, 0.9, 0.05).expect("sampler");
        Record::new(key, sampler, argv, triple, None, None, 1).expect("record")
    }

    fn baseline_server() -> Vec<Flag> {
        vec![
            Flag::Fit(true),
            Flag::FitTarget(nz(1024)),
            Flag::FitCtx(nz(LLAMA_DEFAULT_FIT_CTX)),
            Flag::FlashAttn(FlashAttn::Auto),
            Flag::CacheTypeK(CacheType::F16),
            Flag::CacheTypeV(CacheType::F16),
            Flag::Threads(14),
        ]
    }

    fn moe_server() -> Vec<Flag> {
        let mask = CpuMask::new(0x555).expect("mask");
        vec![
            Flag::Fit(false),
            Flag::GpuLayers(999),
            Flag::NCpuMoe(32),
            Flag::FlashAttn(FlashAttn::On),
            Flag::CacheTypeK(CacheType::Q8_0),
            Flag::CacheTypeV(CacheType::Q8_0),
            Flag::Threads(6),
            Flag::CpuMask(mask),
            Flag::CpuStrict(true),
        ]
    }

    fn moe_bench() -> Vec<Flag> {
        moe_server().into_iter().filter(|f| !matches!(f, Flag::Fit(_))).collect()
    }

    fn machine() -> Result<String, ServerError> {
        Ok(hex('a'))
    }

    #[test]
    fn the_baseline_and_a_tuned_configuration_verify_against_what_they_measured() {
        let served = verify(&record(&[Flag::FitTarget(nz(1024))]), &hex('b'), &hex('c'), &baseline_server(), machine).expect("baseline");
        // The bare thread count is served as llama-server's default, which llama-bench measured.
        assert_eq!(served, baseline_server()[..6].to_vec());
        let served = verify(&record(&moe_bench()), &hex('b'), &hex('c'), &moe_server(), machine).expect("moe");
        assert_eq!(served, moe_server());
    }

    #[test]
    fn a_different_model_backend_or_machine_is_refused() {
        let r = record(&moe_bench());
        let refused = |err: ServerError, expected: &str| {
            assert!(matches!(&err, ServerError::TunedMismatch { what } if what.contains(expected)), "{err}");
        };
        refused(verify(&r, &hex('e'), &hex('c'), &moe_server(), machine).unwrap_err(), "different model");
        refused(verify(&r, &hex('b'), &hex('f'), &moe_server(), machine).unwrap_err(), "different backend");
        refused(verify(&r, &hex('b'), &hex('c'), &moe_server(), || Ok(hex('d'))).unwrap_err(), "another machine");
    }

    #[test]
    fn flags_that_were_not_measured_are_refused() {
        // The MoE record served with a different expert split, with the baseline's flags, or
        // with an extra flag.
        let r = record(&moe_bench());
        let mut fewer_experts = moe_server();
        fewer_experts[2] = Flag::NCpuMoe(30);
        let mut extra = moe_server();
        extra.push(Flag::BatchSize(4096));
        for flags in [fewer_experts, baseline_server(), extra] {
            let err = verify(&r, &hex('b'), &hex('c'), &flags, machine).unwrap_err();
            assert!(matches!(err, ServerError::TunedMismatch { .. }), "{err}");
        }
        // An explicit placement with llama-server's fitter left on is not what was measured.
        let fitter_on: Vec<Flag> = moe_server().into_iter().filter(|f| !matches!(f, Flag::Fit(_))).collect();
        assert!(verify(&r, &hex('b'), &hex('c'), &fitter_on, machine).is_err());
        // The machine is probed only once everything cheaper matched.
        let probed = std::cell::Cell::new(false);
        let _ = verify(&r, &hex('e'), &hex('c'), &moe_server(), || {
            probed.set(true);
            machine()
        });
        assert!(!probed.get());
    }

    #[test]
    fn a_record_whose_argv_is_not_a_bench_argv_never_matches() {
        let good = record(&[Flag::FitTarget(nz(1024))]);
        let argv_without_model = vec!["-fitt".to_string(), "1024".to_string(), "-p".to_string(), "512".to_string()];
        let odd = Record::new(good.provenance().clone(), *good.sampler(), argv_without_model, *good.measurement(), None, None, 1)
            .expect("record");
        assert!(verify(&odd, &hex('b'), &hex('c'), &baseline_server(), machine).is_err());
    }
}
