//! Tuning for this machine: measures the processor settings that depend on the hardware (how many
//! threads take part, how finely work is split, four-row Q8_0 dots) on the loaded model and keeps
//! the fastest. Only the writing (decode) speed is measured: it is what the processor share limits.
//!
//! Method, so a result is believable on a laptop whose clocks move with temperature:
//! - every setting is timed as the median of three repetitions after a warm-up, each repetition
//!   sized to about a second of writing;
//! - one setting at a time (threads, then work splitting, then four-row dots), each starting from
//!   the best so far;
//! - among settings within 2% of the fastest, the one with fewer threads (or the current value) wins:
//!   the same speed with fewer busy cores runs cooler and leaves room for other apps;
//! - finally the starting settings and the winner are timed alternately twice (A B A B) and the
//!   winner is kept only if it is at least 3% faster, or as fast (within 1%) with fewer threads.
//!   Otherwise the starting settings are restored.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use serde::Serialize;

use crate::model::{model_fingerprint, Model, Session};
use crate::sample::argmax;
use crate::switches::{self, CPU_CHUNK_DIV, CPU_DECODE_THREADS as CPU_THREADS, CPU_ROWS4};
use crate::Error;

/// One switch value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Setting {
    pub id: &'static str,
    pub value: i64,
}

/// One timed setting.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Trial {
    /// "threads", "splitting", "rows4" or "confirm".
    pub phase: &'static str,
    /// Plain words, e.g. "12 threads".
    pub label: String,
    pub settings: Vec<Setting>,
    /// Median writing speed, tokens per second.
    pub tps: f64,
    pub reps: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Outcome {
    /// `hw::machine_key()`: the result belongs to this processor, GPU and core layout.
    pub machine: String,
    /// `model_fingerprint`, as hex.
    pub model: String,
    /// Layers on the GPU when measured (0 = processor only); a different split needs a new tune.
    pub gpu_layers: usize,
    /// The settings in force afterwards (the winner if kept, else the starting ones).
    pub settings: Vec<Setting>,
    pub start: Vec<Setting>,
    pub start_tps: f64,
    pub tuned_tps: f64,
    /// tuned / start - 1, from the alternating confirmation.
    pub gain: f64,
    pub kept: bool,
    /// How busy the processor was (0..1) with other programs just before measuring (this one's own
    /// threads not counted).
    pub busy: Option<f64>,
    pub note: String,
    pub trials: Vec<Trial>,
    pub seconds: f64,
}

const PROMPT: &str = "Write a short paragraph about how lighthouses guided ships along dangerous coasts before radio.";
const WARM: usize = 6;
const REPS: usize = 3;
const HEAT_SECONDS: f64 = 5.0;
/// Above this share of the processor in use by other programs, a result is reported but not kept.
const BUSY_LIMIT: f64 = 0.25;

fn current() -> Vec<Setting> {
    // Writing threads as in effect (the switch's 0 means automatic).
    [CPU_THREADS, CPU_CHUNK_DIV, CPU_ROWS4]
        .into_iter()
        .map(|id| Setting { id, value: if id == CPU_THREADS { crate::pool::writing_threads() as i64 } else { switches::get(id) } })
        .collect()
}

fn apply(set: &[Setting]) -> Result<(), Error> {
    for s in set {
        switches::set(s.id, s.value).map_err(Error::Request)?;
    }
    Ok(())
}

fn with(base: &[Setting], id: &'static str, value: i64) -> Vec<Setting> {
    base.iter().map(|s| if s.id == id { Setting { id, value } } else { *s }).collect()
}

fn value(set: &[Setting], id: &str) -> i64 {
    set.iter().find(|s| s.id == id).map_or(0, |s| s.value)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() {
        0.0
    } else if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2.0
    }
}

/// Thread counts worth timing on this processor: the performance cores (one thread each, then
/// both), halfway to every thread, and every thread with and without headroom.
pub fn thread_candidates(c: crate::hw::Cores, current: usize, capacity: usize) -> Vec<usize> {
    let l = c.logical;
    let mut v = vec![c.performance_physical, c.performance_logical, (c.performance_logical + l) / 2, l.saturating_sub(4), l.saturating_sub(2), l, current];
    v.retain(|&t| t >= 1 && t <= capacity);
    v.sort_unstable();
    v.dedup();
    v
}

/// The pick among measured (value, tps): the fastest, unless a preferred value is within `tie` of
/// it. `prefer` orders the candidates within the tie (first wins).
pub fn pick(results: &[(i64, f64)], tie: f64, prefer: &dyn Fn(i64, i64) -> std::cmp::Ordering) -> Option<i64> {
    let best = results.iter().map(|r| r.1).fold(f64::MIN, f64::max);
    let mut near: Vec<_> = results.iter().filter(|r| r.1 >= best * (1.0 - tie)).collect();
    near.sort_by(|a, b| prefer(a.0, b.0));
    near.first().map(|r| r.0)
}

/// Whether the winner replaces the starting settings (see the module notes).
pub fn keep(gain: f64, tuned_threads: i64, start_threads: i64) -> bool {
    gain >= 0.03 || (gain >= -0.01 && tuned_threads < start_threads)
}

struct Bench<'a> {
    m: &'a Model,
    prompt: Vec<u32>,
    rep_tokens: usize,
}

impl Bench<'_> {
    /// One timed repetition at `set`: the prompt, a warm-up, then `rep_tokens` written tokens.
    fn once(&self, s: &mut Session, set: &[Setting]) -> Result<f64, Error> {
        apply(set)?;
        s.feed(self.m, &self.prompt)?;
        let mut t = argmax(&s.logits);
        for _ in 0..WARM {
            s.step(self.m, t);
            t = argmax(&s.logits);
        }
        let t0 = Instant::now();
        for _ in 0..self.rep_tokens {
            s.step(self.m, t);
            t = argmax(&s.logits);
        }
        Ok(self.rep_tokens as f64 / t0.elapsed().as_secs_f64())
    }

    /// Times every set `REPS` times in rounds, the order reversed each round: a laptop slows as it
    /// heats, and timing each setting back to back would favour whichever ran first.
    fn rounds(&self, s: &mut Session, sets: &[Vec<Setting>], cancel: &AtomicBool) -> Result<Vec<Vec<f64>>, Error> {
        let mut reps = vec![Vec::with_capacity(REPS); sets.len()];
        for i in round_order(sets.len()) {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::Request("tuning was cancelled".into()));
            }
            reps[i].push(self.once(s, &sets[i])?);
        }
        Ok(reps)
    }
}

/// The order `rounds` times n settings in: forward, backward, forward, ...
fn round_order(n: usize) -> Vec<usize> {
    (0..REPS).flat_map(|r| (0..n).map(move |k| if r % 2 == 0 { k } else { n - 1 - k })).collect()
}

/// Tunes the processor settings for `m` as loaded in `s` (its conversation cache is replaced by
/// the measurement prompt). `on_trial` sees every timed setting as its phase finishes; `cancel`
/// stops between repetitions and restores the starting settings.
pub fn tune(m: &Model, s: &mut Session, cancel: &AtomicBool, on_trial: &mut dyn FnMut(&Trial)) -> Result<Outcome, Error> {
    let t_start = Instant::now();
    let start = current();
    #[cfg(feature = "cuda")]
    let (gpu_layers, whole_gpu) = s.gpu.as_ref().map_or((0, false), |g| (g.gpu_layers(), !g.is_split()));
    #[cfg(not(feature = "cuda"))]
    let (gpu_layers, whole_gpu) = (0, false);
    // The portable backend runs every layer on its GPU too.
    #[cfg(feature = "portable")]
    let (gpu_layers, whole_gpu) = match s.portable.as_ref() {
        Some(p) => (p.gpu_layers(), !p.is_split()),
        None => (gpu_layers, whole_gpu),
    };
    let outcome = Outcome {
        machine: crate::hw::machine_key(),
        model: format!("{:016x}", model_fingerprint(m)),
        gpu_layers,
        settings: start.clone(),
        start: start.clone(),
        start_tps: 0.0,
        tuned_tps: 0.0,
        gain: 0.0,
        kept: false,
        busy: None,
        note: String::new(),
        trials: Vec::new(),
        seconds: 0.0,
    };
    if whole_gpu && m.cfg.n_expert == 0 {
        return Ok(Outcome { note: "The whole model runs on the graphics card, so the processor settings do not change its speed.".into(), ..outcome });
    }
    // Other programs' load makes every timing noisy; measured before the engine's own threads spin.
    let busy = crate::hw::cpu_busy(std::time::Duration::from_millis(1000));
    let tok = crate::tokenizer::Tokenizer::from_gguf(m.gguf())?;
    let prompt = tok.encode(PROMPT);
    let mut bench = Bench { m, prompt, rep_tokens: 4 };
    let result = (|| -> Result<Outcome, Error> {
        // Heat the machine to its working temperature first: a cold laptop is faster for a few
        // seconds, which would flatter whichever setting is timed first.
        let heat = Instant::now();
        let mut probe = bench.once(s, &start)?;
        while heat.elapsed().as_secs_f64() < HEAT_SECONDS {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::Request("tuning was cancelled".into()));
            }
            probe = bench.once(s, &start)?;
        }
        // Size each repetition to about a second of writing at the starting settings.
        bench.rep_tokens = (probe.round() as usize).clamp(4, 48);
        let mut trials = Vec::new();
        let mut phase = |name: &'static str, cands: Vec<(i64, String, Vec<Setting>)>, trials: &mut Vec<Trial>| -> Result<Vec<(i64, f64)>, Error> {
            let sets: Vec<Vec<Setting>> = cands.iter().map(|c| c.2.clone()).collect();
            let reps = bench.rounds(s, &sets, cancel)?;
            let mut out = Vec::new();
            for ((v, label, set), reps) in cands.into_iter().zip(reps) {
                let t = Trial { phase: name, label, settings: set, tps: median(&reps), reps };
                on_trial(&t);
                out.push((v, t.tps));
                trials.push(t);
            }
            Ok(out)
        };
        let mut best = start.clone();

        let capacity = crate::pool::global().capacity();
        let threads_now = value(&best, CPU_THREADS);
        let cands = thread_candidates(crate::hw::cores(), threads_now as usize, capacity)
            .into_iter()
            .map(|t| (t as i64, format!("{t} threads"), with(&best, CPU_THREADS, t as i64)))
            .collect();
        let r = phase("threads", cands, &mut trials)?;
        // Fewer threads win a tie: cooler, quieter, more room for other apps.
        best = with(&best, CPU_THREADS, pick(&r, 0.02, &|a, b| a.cmp(&b)).unwrap_or(threads_now));

        let div_now = value(&best, CPU_CHUNK_DIV);
        let mut divs = vec![1, 2, 4, 8, div_now];
        divs.sort_unstable();
        divs.dedup();
        let cands = divs.into_iter().map(|d| (d, format!("{d} pieces per thread"), with(&best, CPU_CHUNK_DIV, d))).collect();
        let r = phase("splitting", cands, &mut trials)?;
        best = with(&best, CPU_CHUNK_DIV, pick(&r, 0.02, &|a, b| (a != div_now).cmp(&(b != div_now)).then(a.cmp(&b))).unwrap_or(div_now));

        if m.uses_q8_0() {
            let rows_now = value(&best, CPU_ROWS4);
            let cands = [0, 1].into_iter().map(|v| (v, if v == 1 { "Four-row dots on" } else { "Four-row dots off" }.to_owned(), with(&best, CPU_ROWS4, v))).collect();
            let r = phase("rows4", cands, &mut trials)?;
            best = with(&best, CPU_ROWS4, pick(&r, 0.02, &|a, b| (a != rows_now).cmp(&(b != rows_now))).unwrap_or(rows_now));
        }

        // Confirm: the starting settings and the winner, timed in alternating rounds.
        let (start_tps, tuned_tps) = if best == start {
            let t = trials.iter().filter(|t| t.settings == start).map(|t| t.tps).fold(0.0, f64::max);
            (t, t)
        } else {
            let mut r = bench.rounds(s, &[start.clone(), best.clone()], cancel)?;
            let b = r.pop().unwrap_or_default();
            let a = r.pop().unwrap_or_default();
            let confirm = Trial { phase: "confirm", label: "The winner against the starting settings".into(), settings: best.clone(), tps: median(&b), reps: b };
            on_trial(&confirm);
            let tuned = confirm.tps;
            trials.push(confirm);
            (median(&a), tuned)
        };
        let gain = if start_tps > 0.0 { tuned_tps / start_tps - 1.0 } else { 0.0 };
        let loaded = busy.filter(|&b| b > BUSY_LIMIT);
        let kept = loaded.is_none() && best != start && keep(gain, value(&best, CPU_THREADS), value(&start, CPU_THREADS));
        let note = if let Some(b) = loaded {
            format!("Other programs were using {:.0}% of the processor, so the timings are not reliable and your settings were kept. Close them and tune again.", b * 100.0)
        } else if best == start {
            "Your current settings were already the fastest measured.".to_owned()
        } else if kept && gain >= 0.03 {
            format!("{:.0}% faster writing with the tuned settings.", gain * 100.0)
        } else if kept {
            format!("As fast with {} fewer threads, so it runs cooler.", value(&start, CPU_THREADS) - value(&best, CPU_THREADS))
        } else if gain > 0.0 {
            format!("The best setting found was only {:.1}% faster than yours, within measurement noise, so yours were kept.", gain * 100.0)
        } else {
            "No setting measured faster than yours, so yours were kept.".to_owned()
        };
        Ok(Outcome { settings: if kept { best } else { start.clone() }, start_tps, tuned_tps, gain, kept, busy, note, trials, ..outcome.clone() })
    })();
    let mut out = match result {
        Ok(o) => o,
        Err(e) => {
            let _ = apply(&start);
            return Err(e);
        }
    };
    apply(&out.settings)?;
    out.seconds = t_start.elapsed().as_secs_f64();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw::Cores;

    #[test]
    fn thread_candidates_follow_the_core_layout() {
        let hybrid = Cores { physical: 14, logical: 20, performance_physical: 6, performance_logical: 12 };
        assert_eq!(thread_candidates(hybrid, 16, 20), vec![6, 12, 16, 18, 20]);
        let flat = Cores { physical: 8, logical: 16, performance_physical: 8, performance_logical: 16 };
        assert_eq!(thread_candidates(flat, 12, 16), vec![8, 12, 14, 16]);
        let tiny = Cores { physical: 2, logical: 2, performance_physical: 2, performance_logical: 2 };
        assert_eq!(thread_candidates(tiny, 1, 2), vec![1, 2]);
        assert_eq!(thread_candidates(hybrid, 16, 8), vec![6]);
    }

    #[test]
    fn ties_prefer_fewer_threads_and_the_current_value() {
        let fewer = |a: i64, b: i64| a.cmp(&b);
        assert_eq!(pick(&[(6, 50.0), (12, 61.0), (16, 62.0), (20, 60.0)], 0.02, &fewer), Some(12));
        assert_eq!(pick(&[(6, 50.0), (12, 58.0), (16, 62.0)], 0.02, &fewer), Some(16));
        let cur = 4;
        let keep_cur = |a: i64, b: i64| (a != cur).cmp(&(b != cur)).then(a.cmp(&b));
        assert_eq!(pick(&[(1, 60.5), (2, 60.0), (4, 60.2), (8, 59.0)], 0.02, &keep_cur), Some(4));
        assert_eq!(pick(&[(1, 70.0), (4, 60.0)], 0.02, &keep_cur), Some(1));
        assert_eq!(pick(&[], 0.02, &fewer), None);
    }

    #[test]
    fn a_winner_must_be_faster_or_cooler() {
        assert!(keep(0.05, 16, 16));
        assert!(!keep(0.02, 16, 16));
        assert!(keep(-0.005, 12, 16));
        assert!(!keep(-0.02, 12, 16));
        assert!(!keep(0.0, 18, 16));
    }

    #[test]
    fn rounds_alternate_direction() {
        assert_eq!(round_order(3), vec![0, 1, 2, 2, 1, 0, 0, 1, 2]);
        // Over a forward and a backward round every setting's positions sum the same, so a steady
        // slowdown weighs on each equally.
        let o = &round_order(4)[..8];
        let pos = |i: usize| o.iter().enumerate().filter(|p| *p.1 == i).map(|p| p.0).sum::<usize>();
        assert!((0..4).all(|i| pos(i) == pos(0)));
    }

    #[test]
    fn median_of_odd_even_and_empty() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&[]), 0.0);
    }
}
