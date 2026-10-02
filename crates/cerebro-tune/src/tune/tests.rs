//! The tuner against a fake bench runner: staging, winner selection, the tie rule, the
//! never-slower rule and its guards, comparability, the drift check, failures, cancellation and
//! persistence. The live counterpart is `tests/live_tune.rs`.

use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use cerebro_bench::RunError;
use cerebro_flags::value::CacheType;
use cerebro_flags::{ArgvBuilder, Dialect, Flag};
use cerebro_probe::{PowerSource, PowerState};
use cerebro_store::ProvenanceKey;

use super::*;
use crate::candidate::tests::{big_moe, laptop_cpu, server_argv, small_dense};
use crate::candidate::{KvCache, Placement, ThreadPlan, Vram};
use crate::runner::TestSamples;
use crate::select::tests::summary;

/// llama-bench's own default thread count on the development laptop.
const DEFAULT_THREADS: u32 = 14;

/// What a fake candidate measures: decode at depth `tg` +/- `half`, prefill `pp` (default
/// `10 x tg`), under conditions `tweak` edits.
#[derive(Clone, Copy)]
struct Speed {
    tg: f64,
    half: f64,
    pp: Option<f64>,
    tweak: Option<fn(&mut Conditions)>,
}

fn sp(tg: f64, half: f64) -> Speed {
    Speed { tg, half, pp: None, tweak: None }
}

type Respond = dyn Fn(&Parsed, usize, &CancelToken) -> Result<Speed, MeasureError>;

/// The configuration a fake call was asked for, recovered from its flags.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Parsed {
    baseline_fit: bool,
    ngl: Option<u32>,
    ncmoe: Option<u32>,
    q8: bool,
    threads: Option<u32>,
}

fn parse(flags: &[Flag]) -> Parsed {
    let mut p = Parsed { baseline_fit: false, ngl: None, ncmoe: None, q8: false, threads: None };
    for f in flags {
        match f {
            Flag::FitTarget(_) => p.baseline_fit = true,
            Flag::GpuLayers(n) => p.ngl = Some(*n),
            Flag::NCpuMoe(n) => p.ncmoe = Some(*n),
            Flag::CacheTypeK(CacheType::Q8_0) => p.q8 = true,
            Flag::Threads(n) => p.threads = Some(*n),
            _ => {}
        }
    }
    p
}

struct Fake {
    respond: Box<Respond>,
    calls: Mutex<Vec<Parsed>>,
}

impl Fake {
    fn new(respond: impl Fn(&Parsed, usize, &CancelToken) -> Result<Speed, MeasureError> + 'static) -> Self {
        Self { respond: Box::new(respond), calls: Mutex::new(Vec::new()) }
    }

    fn calls(&self) -> Vec<Parsed> {
        self.calls.lock().unwrap().clone()
    }
}

fn hex(c: char) -> String {
    std::iter::repeat_n(c, 64).collect()
}

/// Quiet, plugged-in conditions matching the fake provenance key.
fn quiet() -> Conditions {
    Conditions {
        model_sha256: hex('b'),
        backend_manifest_digest: hex('c'),
        machine_profile_id: hex('a'),
        power: Some(PowerState { source: PowerSource::Ac, battery_saver: false, battery_percent: None }),
        other_llama_processes: Some(0),
        gpu_utilization_pct: Some(3),
    }
}

/// A bench failure as llama-bench reports an out-of-memory exit.
fn oom() -> MeasureError {
    MeasureError::CandidateFailed(RunError::BackendExited {
        code: Some(1),
        stderr_tail: vec!["cudaMalloc failed: out of memory".into()],
    })
}

impl CandidateRunner for Fake {
    fn measure(&self, flags: &[Flag], plan: &RunPlan, cancel: &CancelToken) -> Result<Measurement, MeasureError> {
        // Every candidate the tuner generates must be a valid llama-bench configuration...
        let mut builder = ArgvBuilder::new(Dialect::Bench);
        builder.push(Flag::Model(r"C:\models\m.gguf".into())).unwrap();
        for f in flags {
            builder.push(f.clone()).unwrap();
        }
        let argv = builder.build().expect("a valid bench configuration");
        let parsed = parse(flags);
        // ...and servable as measured.
        let config = CandidateConfig {
            placement: match (parsed.baseline_fit, parsed.ncmoe, parsed.ngl) {
                (true, ..) => Placement::Auto,
                (false, Some(n), _) => Placement::ExpertsOnCpu(n),
                _ => Placement::FullOffload,
            },
            kv: KvCache::Default,
            threads: ThreadPlan::Default,
        };
        server_argv(&config, Some(DEFAULT_THREADS));

        let call = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(parsed);
            calls.len() - 1
        };
        let speed = (self.respond)(&parsed, call, cancel)?;
        let depth = u64::from(plan.deepest_depth());
        let tg = summary(speed.tg, speed.half);
        let pp = speed.pp.unwrap_or(speed.tg * 10.0);
        let mut conditions = quiet();
        if let Some(tweak) = speed.tweak {
            tweak(&mut conditions);
        }
        Ok(Measurement {
            provenance: ProvenanceKey::new(hex('a'), hex('b'), hex('c'), Some(10751), Some("3466812d1".into()), None, None, Some(7000)).unwrap(),
            conditions,
            argv,
            triple: Triple {
                pp_tok_s: summary(pp, 1.0),
                tg_tok_s_depth0: summary(speed.tg * 1.2, speed.half),
                tg_tok_s_deepest: tg,
                deepest_depth: NonZeroU64::new(depth).unwrap(),
            },
            threads: Some(parsed.threads.unwrap_or(DEFAULT_THREADS)),
            samples: vec![
                TestSamples { kind: TestKind::Prefill, depth: 0, kept_tok_s: vec![pp; 3] },
                TestSamples { kind: TestKind::Decode, depth: 0, kept_tok_s: vec![speed.tg * 1.2; 3] },
                TestSamples { kind: TestKind::Prefill, depth, kept_tok_s: vec![pp * 0.8; 3] },
                TestSamples { kind: TestKind::Decode, depth, kept_tok_s: vec![speed.tg; 3] },
            ],
            warnings: vec!["folder permission audit is off".into()],
        })
    }
}

/// A store file in a fresh scratch directory, removed on drop.
struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "cerebro-tune-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }

    fn store(&self) -> Store {
        Store::new(self.dir.join("results.jsonl"))
    }

    fn records(&self) -> Vec<Record> {
        self.store().load_verified().unwrap().records
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn sampler() -> SamplerProfile {
    SamplerProfile::new(0.7, 20, 0.8, 0.05).unwrap()
}

fn request(model: ModelFacts, free_mib: u64, workload: Workload) -> TuneRequest {
    TuneRequest {
        workload,
        model,
        machine: MachineFacts { vram: Vram::FreeMib(vec![free_mib]), cpu: Some(laptop_cpu()) },
        screen: ScreenSettings::default(),
        sampler: sampler(),
    }
}

fn run(req: &TuneRequest, fake: &Fake, scratch: &Scratch, cancel: &CancelToken) -> (Result<TuneOutcome, TuneError>, Vec<TuneEvent>) {
    let mut events = Vec::new();
    let result = tune(req, fake, &scratch.store(), cancel, &mut |e| events.push(e.clone()));
    (result, events)
}

/// Every `CandidateStarted` is followed by exactly one `CandidateFinished` for it, with only its
/// own `Sample`s between, and nothing else interleaved.
fn assert_paired(events: &[TuneEvent]) {
    let mut open: Option<usize> = None;
    for e in events {
        match (e, open) {
            (TuneEvent::CandidateStarted { index, .. }, None) => open = Some(*index),
            (TuneEvent::Sample { index, .. }, Some(o)) => assert_eq!(*index, o),
            (TuneEvent::CandidateFinished { index, .. }, Some(o)) => {
                assert_eq!(*index, o);
                open = None;
            }
            (other, None) if !matches!(other, TuneEvent::Sample { .. } | TuneEvent::CandidateFinished { .. }) => {}
            (other, state) => panic!("{other:?} while candidate {state:?} is open"),
        }
    }
    assert_eq!(open, None, "a started candidate never finished");
}

/// The MoE story from the development laptop: llama.cpp's automatic settings 3.1 tok/s, the best
/// expert offload about 18, q8_0 KV a little better, and thread counts that tie.
fn moe_speeds(p: &Parsed) -> Speed {
    if p.baseline_fit {
        return sp(3.1, 0.1);
    }
    let ncmoe = p.ncmoe.unwrap_or(0);
    let mut tg = 18.0 - f64::from(ncmoe.abs_diff(32)) * 0.25;
    if p.q8 {
        tg += 1.0;
    }
    tg += match p.threads {
        None => 0.0,
        Some(6) => 0.1,
        Some(3) => -4.0,
        Some(12) => 0.2,
        Some(_) => -1.0,
    };
    sp(tg, if p.threads.is_some() || p.q8 { 0.5 } else { 0.3 })
}

fn six_thread_winner() -> CandidateConfig {
    CandidateConfig {
        placement: Placement::ExpertsOnCpu(32),
        kv: KvCache::Q8,
        threads: ThreadPlan::Pinned { threads: 6, mask: cerebro_flags::CpuMask::new(0x555).unwrap() },
    }
}

#[test]
fn a_moe_tune_screens_every_stage_picks_the_winner_by_the_tie_rule_and_persists_everything() {
    let scratch = Scratch::new("moe");
    let req = request(big_moe(), 7300, Workload::Coding);
    let fake = Fake::new(|p, _, _| Ok(moe_speeds(p)));
    let (result, events) = run(&req, &fake, &scratch, &CancelToken::new());
    let outcome = result.expect("tune completes");

    let calls = fake.calls();
    assert!(calls[0].baseline_fit, "the baseline is always measured first");
    assert_eq!(
        calls[1..5].iter().map(|c| c.ncmoe).collect::<Vec<_>>(),
        [Some(32), Some(36), Some(40), Some(44)],
        "expert offload steps around the estimate"
    );
    // KV stage from the placement leader (32): q8_0 there, and q8_0 one step lighter.
    assert_eq!((calls[5].ncmoe, calls[5].q8), (Some(32), true));
    assert_eq!((calls[6].ncmoe, calls[6].q8), (Some(28), true));
    // Thread stage from the KV leader (32, q8_0).
    assert_eq!(calls[7..10].iter().map(|c| c.threads).collect::<Vec<_>>(), [Some(6), Some(3), Some(12)]);
    assert!(calls[7..10].iter().all(|c| c.ncmoe == Some(32) && c.q8));
    // A candidate won, so the baseline is measured once more at the end.
    assert_eq!(calls.len(), 11);
    assert!(calls[10].baseline_fit && calls[10].threads.is_none() && !calls[10].q8);

    // 12 threads has the highest median (19.2), but 6 threads (19.1) and the default 14 (19.0)
    // overlap it; the fewest threads among the tied wins.
    let best = outcome.recommended().expect("a recommendation");
    assert_eq!(best.config, six_thread_winner());
    // Served exactly as measured.
    assert_eq!(
        outcome.recommended_server_flags().unwrap(),
        Some(six_thread_winner().server_flags(Some(6)).unwrap())
    );
    assert_eq!(outcome.verdict, Verdict::Improved { winner: 7 });
    assert_eq!(outcome.depth, 16_384);
    assert!((outcome.speedup().unwrap() - 19.1 / 3.1).abs() < 1e-9);
    assert_eq!(outcome.warnings, ["folder permission audit is off"], "each warning once, no contention");
    assert!(outcome.excluded.is_empty());
    let recheck = outcome.baseline_recheck.as_ref().expect("the drift check ran");
    assert_eq!((recheck.index, recheck.stage), (10, Stage::Recheck));
    assert!(recheck.config.is_baseline());

    // Every measurement is on disk, the re-measure last, with the sampler profile and its argv.
    let records = scratch.records();
    assert_eq!(records.len(), 11);
    for (i, (record, measured)) in records.iter().zip(outcome.measured.iter().chain([recheck])).enumerate() {
        assert_eq!(measured.record_index, i);
        assert_eq!(*record.sampler(), sampler());
        assert_eq!(record.measured_argv_untrusted(), measured.measurement.argv.as_slice());
        assert_eq!(record.measurement().tg_tok_s_deepest().median(), measured.measurement.triple.tg_tok_s_deepest.median);
        assert_eq!(record.measurement().deepest_depth().get(), 16_384);
    }

    // Progress: four stages, eleven starts and finishes, four samples each, then the winner.
    assert_paired(&events);
    let count = |f: fn(&TuneEvent) -> bool| events.iter().filter(|e| f(e)).count();
    assert_eq!(count(|e| matches!(e, TuneEvent::StageStarted { .. })), 4);
    assert_eq!(count(|e| matches!(e, TuneEvent::CandidateStarted { .. })), 11);
    assert_eq!(count(|e| matches!(e, TuneEvent::Sample { .. })), 44);
    assert_eq!(count(|e| matches!(e, TuneEvent::CandidateFinished { result: CandidateResult::Measured { contention: None, .. }, .. })), 11);
    assert!(matches!(events.first(), Some(TuneEvent::StageStarted { stage: Stage::Placement, candidates: 5 })));
    match events.last() {
        Some(TuneEvent::Winner { index, verdict, tg_at_depth, baseline_tg_at_depth, .. }) => {
            assert_eq!(*index, 7);
            assert_eq!(*verdict, Verdict::Improved { winner: 7 });
            assert!((tg_at_depth - 19.1).abs() < 1e-9);
            assert!((baseline_tg_at_depth - 3.1).abs() < 1e-9);
        }
        other => panic!("expected the winner last, got {other:?}"),
    }
}

#[test]
fn when_nothing_is_demonstrably_faster_the_baseline_wins_and_the_outcome_says_so() {
    // The small-model story: llama.cpp's defaults 80.6 beat hand-tuned flags at 74.8. One thread
    // variant has a higher median (81.0) but its interval overlaps the baseline's: not selected.
    let scratch = Scratch::new("baseline");
    let req = request(small_dense(), 7000, Workload::Chat);
    let fake = Fake::new(|p, _, _| {
        Ok(match (p.baseline_fit, p.threads, p.q8) {
            (true, None, false) => sp(80.5, 0.5),
            (true, Some(6), _) => sp(81.0, 1.0),
            (true, Some(_), _) => sp(70.0, 1.0),
            (true, None, true) => sp(79.0, 0.5),
            (false, ..) => sp(74.8, 0.4),
        })
    });
    let (result, events) = run(&req, &fake, &scratch, &CancelToken::new());
    let outcome = result.unwrap();
    assert_eq!(outcome.verdict, Verdict::BaselineWins);
    let rec = outcome.recommended().unwrap();
    assert!(rec.config.is_baseline());
    assert_eq!(rec.index, 0);
    assert_eq!(outcome.speedup(), Some(1.0));
    assert!(outcome.excluded.is_empty(), "slower or tied is not an exclusion");
    // The baseline is served with llama.cpp's automatic settings stated, never inherited.
    let served = outcome.recommended_server_flags().unwrap().unwrap();
    assert_eq!(served, CandidateConfig::BASELINE.server_flags(Some(DEFAULT_THREADS)).unwrap());
    assert!(served.contains(&Flag::Fit(true)) && served.contains(&Flag::Threads(DEFAULT_THREADS)));

    let calls = fake.calls();
    assert_eq!(calls[1].ngl, Some(999), "full offload, because the model fits");
    // Full offload lost, so the next stages vary the baseline itself.
    assert!(calls[2].baseline_fit && calls[2].q8);
    assert!(calls[3..].iter().all(|c| c.baseline_fit && c.threads.is_some() && !c.q8));
    assert_eq!(calls.len(), 6, "no re-measure when the baseline already wins");
    assert!(outcome.baseline_recheck.is_none());
    assert_eq!(scratch.records().len(), 6, "losers are persisted too");
    assert_paired(&events);
    assert!(matches!(
        events.last(),
        Some(TuneEvent::Winner { index: 0, verdict: Verdict::BaselineWins, .. })
    ));
}

#[test]
fn faster_decode_with_slower_prefill_is_not_chosen_and_the_outcome_says_why() {
    // Full offload decodes faster at depth than the baseline but prefills at well under half its
    // speed: for a long prompt that is slower for the user.
    let scratch = Scratch::new("prefill");
    let req = request(small_dense(), 7000, Workload::Chat);
    let fake = Fake::new(|p, _, _| {
        Ok(match (p.baseline_fit, p.ngl, p.q8, p.threads) {
            (true, _, false, None) => Speed { pp: Some(7641.0), ..sp(80.5, 0.5) },
            (false, Some(999), ..) => Speed { pp: Some(2866.0), ..sp(90.0, 0.5) },
            _ => sp(70.0, 1.0),
        })
    });
    let (result, events) = run(&req, &fake, &scratch, &CancelToken::new());
    let outcome = result.unwrap();
    assert_eq!(outcome.verdict, Verdict::BaselineWins);
    assert_eq!(outcome.excluded, [Excluded { index: 1, reason: Exclusion::Guard(Guard::Prefill) }]);
    // The excluded candidate is not the leader either: the later stages vary the baseline.
    let calls = fake.calls();
    assert!(calls[2].baseline_fit && calls[2].q8);
    assert_eq!(calls.len(), 6);
    assert_paired(&events);
}

#[test]
fn a_candidate_slower_than_the_baseline_never_wins_even_as_the_only_success() {
    let scratch = Scratch::new("slower");
    // Coding depth: offload steps 32, 36, 40 and 44.
    let req = request(big_moe(), 7300, Workload::Coding);
    let fake = Fake::new(|p, _, _| {
        if p.baseline_fit && p.threads.is_none() && !p.q8 {
            Ok(sp(10.0, 0.2))
        } else if p.ncmoe == Some(40) || p.ncmoe == Some(44) {
            Ok(sp(9.0, 0.2))
        } else {
            Err(oom())
        }
    });
    let (result, events) = run(&req, &fake, &scratch, &CancelToken::new());
    let outcome = result.unwrap();
    assert_eq!(outcome.verdict, Verdict::BaselineWins);
    assert!(!outcome.failed.is_empty(), "failures are recorded, and the tune carries on");
    for f in &outcome.failed {
        assert!(f.reason.contains("out of memory"), "{}", f.reason);
        assert_eq!(f.kind, FailureKind::Exited { code: Some(1) });
    }
    assert_eq!(outcome.measured.len(), 3, "the baseline and the two slower steps");
    assert_eq!(scratch.records().len(), 3, "only measurements are records");
    assert_paired(&events);
    assert!(events.iter().any(|e| matches!(e, TuneEvent::CandidateFinished { result: CandidateResult::Failed { .. }, .. })));
}

#[test]
fn the_winner_index_survives_failed_candidates() {
    let scratch = Scratch::new("index");
    let req = request(big_moe(), 7300, Workload::Coding);
    let fake = Fake::new(|p, _, _| if p.ncmoe == Some(36) { Err(oom()) } else { Ok(moe_speeds(p)) });
    let (result, events) = run(&req, &fake, &scratch, &CancelToken::new());
    let outcome = result.unwrap();
    assert_eq!(outcome.failed.len(), 1);
    assert_eq!(outcome.failed[0].index, 2);
    // Start order: 0 baseline, 1 ncmoe 32, 2 ncmoe 36 (failed), ..., 7 the 6-thread winner; it is
    // the seventh measurement, so `measured[6]`.
    let Verdict::Improved { winner } = outcome.verdict else { panic!("{:?}", outcome.verdict) };
    assert_eq!(winner, 6);
    assert_eq!(outcome.measured[winner].index, 7);
    assert_eq!(outcome.measured[winner].config, six_thread_winner());
    assert_eq!(outcome.recommended().map(|m| m.index), Some(7));
    let Some(TuneEvent::Winner { index, .. }) = events.last() else { panic!("{:?}", events.last()) };
    assert_eq!(*index, 7, "the event carries the start-order index");
    assert_paired(&events);
}

#[test]
fn a_store_failure_stops_the_tune_after_finishing_the_candidate_and_streams_no_samples() {
    let scratch = Scratch::new("store-fail");
    // A store path under a regular file cannot be opened.
    std::fs::write(scratch.dir.join("not-a-dir"), b"x").unwrap();
    let store = Store::new(scratch.dir.join("not-a-dir").join("results.jsonl"));
    let req = request(small_dense(), 7000, Workload::Chat);
    let fake = Fake::new(|_, _, _| Ok(sp(50.0, 0.5)));
    let mut events = Vec::new();
    let result = tune(&req, &fake, &store, &CancelToken::new(), &mut |e| events.push(e.clone()));
    assert!(matches!(result, Err(TuneError::Store(_))), "{result:?}");
    assert_eq!(fake.calls().len(), 1);
    assert!(!events.iter().any(|e| matches!(e, TuneEvent::Sample { .. })), "nothing unpersisted is streamed");
    assert!(matches!(
        events.as_slice(),
        [
            TuneEvent::StageStarted { .. },
            TuneEvent::CandidateStarted { index: 0, .. },
            TuneEvent::CandidateFinished { index: 0, result: CandidateResult::Failed { .. } },
        ]
    ), "{events:?}");
}

#[test]
fn a_measurement_of_a_different_model_stops_the_tune() {
    let scratch = Scratch::new("model-changed");
    let req = request(big_moe(), 7300, Workload::Coding);
    let fake = Fake::new(|p, call, _| {
        let mut s = moe_speeds(p);
        if call == 1 {
            s.tweak = Some(|c| c.model_sha256 = hex('d'));
        }
        Ok(s)
    });
    let (result, events) = run(&req, &fake, &scratch, &CancelToken::new());
    assert!(matches!(result, Err(TuneError::ConditionsChanged(Identity::Model))), "{result:?}");
    assert_eq!(fake.calls().len(), 2, "nothing runs after the change");
    assert_eq!(scratch.records().len(), 1, "the incomparable measurement is not recorded");
    assert_paired(&events);
    assert!(matches!(
        events.last(),
        Some(TuneEvent::CandidateFinished { index: 1, result: CandidateResult::Failed { .. } })
    ));

    let fake = Fake::new(|p, call, _| {
        let mut s = moe_speeds(p);
        if call == 1 {
            s.tweak = Some(|c| c.backend_manifest_digest = hex('e'));
        }
        Ok(s)
    });
    let (result, _) = run(&req, &fake, &Scratch::new("backend-changed"), &CancelToken::new());
    assert!(matches!(result, Err(TuneError::ConditionsChanged(Identity::Backend))), "{result:?}");
}

#[test]
fn a_contended_baseline_compares_nothing_unless_the_user_allows_it() {
    let req = request(big_moe(), 7300, Workload::Coding);
    let respond = |p: &Parsed, call: usize, _: &CancelToken| {
        let mut s = moe_speeds(p);
        if call == 0 {
            s.tweak = Some(|c| c.other_llama_processes = Some(1));
        }
        Ok(s)
    };
    let scratch = Scratch::new("contended");
    let fake = Fake::new(respond);
    let (result, events) = run(&req, &fake, &scratch, &CancelToken::new());
    let outcome = result.unwrap();
    assert_eq!(outcome.verdict, Verdict::BaselineWins, "a contended baseline is not a fair bar");
    assert_eq!(
        outcome.excluded,
        [Excluded { index: 0, reason: Exclusion::Contended(Contention::OtherLlamaProcesses(1)) }]
    );
    let warnings: Vec<&String> = outcome.warnings.iter().filter(|w| w.contains("contention")).collect();
    assert_eq!(warnings.len(), 1, "one neutral warning: {:?}", outcome.warnings);
    assert!(warnings[0].contains("1 other llama.cpp process"), "{}", warnings[0]);
    // Nothing became the leader, so every later stage varied the baseline; no re-measure.
    let calls = fake.calls();
    assert_eq!(calls.len(), 9);
    assert!(calls[5..].iter().all(|c| c.baseline_fit));
    assert_eq!(scratch.records().len(), 9, "contended measurements are still persisted");
    assert!(events.iter().any(|e| matches!(
        e,
        TuneEvent::CandidateFinished { index: 0, result: CandidateResult::Measured { contention: Some(_), .. } }
    )));

    let mut allowed = req.clone();
    allowed.screen.allow_contended = true;
    let fake = Fake::new(respond);
    let (result, _) = run(&allowed, &fake, &Scratch::new("contended-allowed"), &CancelToken::new());
    let outcome = result.unwrap();
    assert_eq!(outcome.verdict, Verdict::Improved { winner: 7 });
    assert!(outcome.excluded.is_empty());
    assert!(outcome.warnings.iter().any(|w| w.contains("allow_contended is on")), "{:?}", outcome.warnings);
}

#[test]
fn a_candidate_measured_on_battery_is_set_aside_and_the_next_best_wins() {
    let scratch = Scratch::new("battery");
    let req = request(big_moe(), 7300, Workload::Coding);
    let fake = Fake::new(|p, _, _| {
        let mut s = moe_speeds(p);
        if p.threads == Some(6) {
            s.tweak = Some(|c| {
                c.power = Some(PowerState { source: PowerSource::Battery, battery_saver: false, battery_percent: Some(80) });
            });
        }
        Ok(s)
    });
    let (result, _) = run(&req, &fake, &scratch, &CancelToken::new());
    let outcome = result.unwrap();
    assert_eq!(outcome.excluded, [Excluded { index: 7, reason: Exclusion::Contended(Contention::PowerChanged) }]);
    // Without it, 12 threads (19.2) leads and ties only the default 14 (19.0): 12 is fewer.
    let best = outcome.recommended().unwrap();
    assert_eq!(best.config.threads(), Some(12));
    assert!(outcome.warnings.iter().any(|w| w.contains("power source")), "{:?}", outcome.warnings);

    // A busy GPU at launch is contention too.
    let fake = Fake::new(|p, _, _| {
        let mut s = moe_speeds(p);
        if p.threads == Some(6) {
            s.tweak = Some(|c| c.gpu_utilization_pct = Some(CONTENDED_GPU_UTILIZATION_PCT));
        }
        Ok(s)
    });
    let (result, _) = run(&req, &fake, &Scratch::new("gpu-busy"), &CancelToken::new());
    let outcome = result.unwrap();
    assert_eq!(outcome.excluded, [Excluded { index: 7, reason: Exclusion::Contended(Contention::GpuBusy(50)) }]);
}

#[test]
fn a_baseline_that_drifts_by_more_than_the_margin_withdraws_the_winner() {
    let req = request(big_moe(), 7300, Workload::Coding);
    // The re-measured baseline (call 10) runs at 12.0 instead of 3.1: the machine changed by
    // more than enough to explain the winner's 19.1, though 19.1 still beats 12.0 outright.
    let fake = Fake::new(|p, call, _| if call == 10 { Ok(sp(12.0, 0.1)) } else { Ok(moe_speeds(p)) });
    let (result, events) = run(&req, &fake, &Scratch::new("drift"), &CancelToken::new());
    let outcome = result.unwrap();
    assert_eq!(outcome.verdict, Verdict::BaselineWins);
    assert_eq!(outcome.excluded, [Excluded { index: 7, reason: Exclusion::BaselineDrifted }]);
    assert!(outcome.baseline_recheck.is_some());
    assert_paired(&events);
    assert!(matches!(events.last(), Some(TuneEvent::Winner { index: 0, verdict: Verdict::BaselineWins, .. })));

    // A drift far smaller than the margin (3.1 -> 5.0 against a 19.1 winner) keeps the winner,
    // with one neutral warning naming both baseline figures.
    let fake = Fake::new(|p, call, _| if call == 10 { Ok(sp(5.0, 0.1)) } else { Ok(moe_speeds(p)) });
    let (result, _) = run(&req, &fake, &Scratch::new("small-drift"), &CancelToken::new());
    let outcome = result.unwrap();
    assert_eq!(outcome.verdict, Verdict::Improved { winner: 7 });
    assert!(outcome.excluded.is_empty());
    let drift: Vec<&String> = outcome.warnings.iter().filter(|w| w.contains("at the start of the tune")).collect();
    assert_eq!(drift.len(), 1, "{:?}", outcome.warnings);
    assert!(drift[0].contains("3.1") && drift[0].contains("5.0"), "{}", drift[0]);

    // Agreeing on decode, but the re-measure prefills far faster than the winner: not eligible.
    let fake = Fake::new(|p, call, _| {
        if call == 10 {
            Ok(Speed { pp: Some(500.0), ..sp(3.15, 0.1) })
        } else {
            Ok(moe_speeds(p))
        }
    });
    let (result, _) = run(&req, &fake, &Scratch::new("recheck-guard"), &CancelToken::new());
    let outcome = result.unwrap();
    assert_eq!(outcome.verdict, Verdict::BaselineWins);
    assert_eq!(outcome.excluded, [Excluded { index: 7, reason: Exclusion::NotFasterThanRecheck }]);

    // A re-measure that fails leaves the winner unconfirmed.
    let fake = Fake::new(|p, call, _| if call == 10 { Err(oom()) } else { Ok(moe_speeds(p)) });
    let (result, events) = run(&req, &fake, &Scratch::new("recheck-fail"), &CancelToken::new());
    let outcome = result.unwrap();
    assert_eq!(outcome.verdict, Verdict::BaselineWins);
    assert_eq!(outcome.excluded, [Excluded { index: 7, reason: Exclusion::Unconfirmed }]);
    assert_eq!(outcome.failed.last().map(|f| (f.index, f.stage)), Some((10, Stage::Recheck)));
    assert_paired(&events);

    // Switched off: no re-measure, the winner stands.
    let mut off = req.clone();
    off.screen.recheck_baseline = false;
    let fake = Fake::new(|p, call, _| if call == 10 { Ok(sp(12.0, 0.1)) } else { Ok(moe_speeds(p)) });
    let (result, _) = run(&off, &fake, &Scratch::new("recheck-off"), &CancelToken::new());
    assert_eq!(result.unwrap().verdict, Verdict::Improved { winner: 7 });
    assert_eq!(fake.calls().len(), 10);
}

#[test]
fn a_killed_llama_bench_that_is_not_confirmed_gone_stops_the_tune() {
    let req = request(big_moe(), 7300, Workload::Coding);
    let fake = Fake::new(|p, call, _| {
        if call == 1 {
            Err(MeasureError::CandidateFailed(RunError::TimedOut {
                timeout: Duration::from_secs(60),
                confirmed_gone: false,
                stderr_tail: Vec::new(),
            }))
        } else {
            Ok(moe_speeds(p))
        }
    });
    let (result, events) = run(&req, &fake, &Scratch::new("orphan"), &CancelToken::new());
    assert!(matches!(result, Err(TuneError::ProcessNotConfirmedGone(_))), "{result:?}");
    assert_eq!(fake.calls().len(), 2, "nothing is launched beside a possible orphan");
    assert_paired(&events);

    // A confirmed timeout is an ordinary candidate failure.
    let fake = Fake::new(|p, call, _| {
        if call == 1 {
            Err(MeasureError::CandidateFailed(RunError::TimedOut {
                timeout: Duration::from_secs(60),
                confirmed_gone: true,
                stderr_tail: Vec::new(),
            }))
        } else {
            Ok(moe_speeds(p))
        }
    });
    let (result, _) = run(&req, &fake, &Scratch::new("timeout"), &CancelToken::new());
    let outcome = result.unwrap();
    assert_eq!(outcome.failed[0].kind, FailureKind::TimedOut);

    // So is a cancel whose kill was not confirmed.
    let fake = Fake::new(|p, call, _| {
        if call == 1 {
            Err(MeasureError::Cancelled { confirmed_gone: false })
        } else {
            Ok(moe_speeds(p))
        }
    });
    let (result, _) = run(&req, &fake, &Scratch::new("cancel-orphan"), &CancelToken::new());
    assert!(matches!(result, Err(TuneError::ProcessNotConfirmedGone(_))), "{result:?}");
    assert_eq!(fake.calls().len(), 2);
}

#[test]
fn cancelling_during_a_run_stops_the_tune_keeps_what_was_measured_and_recommends_nothing() {
    let scratch = Scratch::new("cancel-run");
    let req = request(big_moe(), 7300, Workload::Coding);
    let fake = Fake::new(|p, call, cancel| {
        if call == 2 {
            // The real runner kills llama-bench when the token is set and reports a cancel.
            cancel.cancel();
            return Err(MeasureError::Cancelled { confirmed_gone: true });
        }
        Ok(moe_speeds(p))
    });
    let token = CancelToken::new();
    let (result, events) = run(&req, &fake, &scratch, &token);
    let outcome = result.expect("a cancel is an outcome, not an error");
    assert_eq!(outcome.verdict, Verdict::Cancelled);
    assert!(outcome.recommended().is_none());
    assert_eq!(outcome.recommended_server_flags().unwrap(), None);
    assert_eq!(outcome.measured.len(), 2);
    assert_eq!(fake.calls().len(), 3, "nothing starts after the cancel");
    assert_eq!(scratch.records().len(), 2, "what was measured stays persisted");
    assert_paired(&events);
    assert_eq!(
        events[events.len() - 2..],
        [
            TuneEvent::CandidateFinished { index: 2, result: CandidateResult::Cancelled },
            TuneEvent::Cancelled { measured: 2 },
        ]
    );
    assert!(!events.iter().any(|e| matches!(e, TuneEvent::Winner { .. })));
}

#[test]
fn cancelling_between_candidates_starts_nothing_more() {
    let scratch = Scratch::new("cancel-between");
    let req = request(small_dense(), 7000, Workload::Chat);
    let fake = Fake::new(|_, _, _| Ok(sp(50.0, 0.5)));
    let token = CancelToken::new();
    let mut events = Vec::new();
    let outcome = tune(&req, &fake, &scratch.store(), &token, &mut |e| {
        if matches!(e, TuneEvent::CandidateFinished { index: 0, .. }) {
            token.cancel();
        }
        events.push(e.clone());
    })
    .unwrap();
    assert_eq!(outcome.verdict, Verdict::Cancelled);
    assert_eq!(fake.calls().len(), 1, "only the baseline ran");
    assert_eq!(outcome.baseline().map(|b| b.index), Some(0));
    assert_eq!(events.last(), Some(&TuneEvent::Cancelled { measured: 1 }));
}

#[test]
fn a_baseline_that_cannot_be_measured_stops_the_tune() {
    let scratch = Scratch::new("no-baseline");
    let req = request(small_dense(), 7000, Workload::Chat);
    let fake = Fake::new(|_, _, _| {
        Err(MeasureError::CandidateFailed(RunError::BackendExited { code: None, stderr_tail: vec!["crashed".into()] }))
    });
    let (result, events) = run(&req, &fake, &scratch, &CancelToken::new());
    assert!(
        matches!(&result, Err(TuneError::BaselineFailed(RunError::BackendExited { stderr_tail, .. })) if stderr_tail == &["crashed"]),
        "{result:?}"
    );
    assert_eq!(fake.calls().len(), 1, "no candidate runs without a baseline to beat");
    assert!(scratch.records().is_empty());
    assert_paired(&events);
}

#[test]
fn a_refusal_that_would_stop_every_candidate_stops_the_tune() {
    let scratch = Scratch::new("refused");
    let req = request(small_dense(), 7000, Workload::Chat);
    let fake = Fake::new(|p, _, _| {
        if p.baseline_fit {
            Ok(sp(50.0, 0.5))
        } else {
            Err(MeasureError::Run(RunError::ModelPathNotUtf8(PathBuf::from("x"))))
        }
    });
    let (result, events) = run(&req, &fake, &scratch, &CancelToken::new());
    assert!(matches!(result, Err(TuneError::Measure(MeasureError::Run(_)))));
    assert_eq!(fake.calls().len(), 2);
    assert_paired(&events);
}

#[test]
fn the_depth_follows_the_workload_and_is_clamped_to_the_model_context() {
    let mut req = request(small_dense(), 7000, Workload::LongDocuments);
    assert_eq!(resolve_depth(&req).unwrap(), 32_768 - 512, "32k context less the prompt test");
    req.workload = Workload::Chat;
    assert_eq!(resolve_depth(&req).unwrap(), 4096);
    req.screen.depth_override = Some(256);
    assert_eq!(resolve_depth(&req).unwrap(), 256);
    req.model.context_length = Some(512);
    assert!(matches!(resolve_depth(&req), Err(TuneError::ContextTooShort { context: 512, .. })));
    req.model.context_length = None;
    req.screen.depth_override = Some(0);
    assert!(matches!(resolve_depth(&req), Err(TuneError::ZeroDepth)));
}
