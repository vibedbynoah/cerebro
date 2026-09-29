//! Deterministic exponential backoff with jitter, and the sliding window the restart policy counts
//! restarts over.
//!
//! Both pieces are pure functions of their inputs — no wall-clock read, no OS RNG — so the state
//! machine's tests in [`super`] can assert an exact delay and an exact restart count without a real
//! clock or a real child process. The driver loop supplies the only impure inputs: how much
//! [`tokio::time`] (paused in tests) has advanced, and the caller-provided jitter seed.

use std::collections::VecDeque;
use std::time::Duration;

/// How restarts back off after a failure, and when the state machine gives up on them entirely.
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    /// Delay before the first restart attempt (before jitter is applied).
    pub base_backoff: Duration,
    /// However many attempts have accumulated, the backoff delay never exceeds this.
    pub max_backoff: Duration,
    /// Once more than this many restarts have happened inside `window`, the state machine gives up
    /// and moves to [`super::LifecycleState::Failed`] instead of backing off again.
    pub max_restarts: u32,
    /// The sliding window `max_restarts` is counted over. A restart older than `window` no longer
    /// counts against the limit — and, since [`backoff_delay`] is keyed off the same in-window
    /// count (see [`RestartWindow::record`]), the backoff exponent relaxes back toward
    /// `base_backoff` once a run has been stable for longer than `window`, rather than staying
    /// inflated by restarts from an unrelated, long-past incident.
    pub window: Duration,
    /// Seed for this policy's jitter. Two runs seeded identically compute identical delays for
    /// identical in-window restart counts — the property this module's own tests rely on, and what
    /// makes a caller's own test of code built on [`crate::lifecycle`] reproducible too.
    pub jitter_seed: u64,
}

/// A small, fixed, non-cryptographic PRNG step (SplitMix64), used only to turn `(seed, attempt)`
/// into a reproducible fraction in `[0, 1)`. Nothing security-relevant ever derives from it — no
/// key, token, or path — so a full PRNG crate would be more machinery than this one call site needs.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The delay to wait before the restart attempt that brings the in-window restart count to
/// `attempt` (so `attempt == 1` is the delay after the first failure).
///
/// Uses "full jitter" — a duration chosen uniformly in `[0, capped_exponential]` — which the AWS
/// architecture blog's own backoff writeup recommends over "equal" or "decorrelated" jitter for
/// avoiding synchronized retry storms across independently restarting supervisors. Computed
/// deterministically from `policy.jitter_seed` and `attempt`, never from the OS RNG, so the same
/// pair always yields the same delay.
pub(crate) fn backoff_delay(policy: &RestartPolicy, attempt: u32) -> Duration {
    // Capped well short of 32: `1u32 << shift` must not shift by an amount >= the type's bit width,
    // and no real restart count ever needs the delay to keep doubling past this — it is already
    // pinned at `max_backoff` long before `shift` gets anywhere near this cap.
    let shift = attempt.saturating_sub(1).min(20);
    let exponential = policy
        .base_backoff
        .checked_mul(1u32 << shift)
        .unwrap_or(policy.max_backoff);
    let capped = exponential.min(policy.max_backoff);

    let mixed = splitmix64(policy.jitter_seed ^ u64::from(attempt));
    // Top 53 bits give a fraction with full `f64` mantissa precision, the standard way to turn a
    // `u64` into a uniform value in `[0, 1)`.
    let fraction = (mixed >> 11) as f64 / (1u64 << 53) as f64;
    // `fraction` is always in `[0, 1)` and `capped` is finite and bounded by `max_backoff`, so the
    // product can never exceed `capped` — `Duration::mul_f64` cannot overflow here.
    capped.mul_f64(fraction)
}

/// The sliding window of recent restarts a [`RestartPolicy`] counts against its limit.
///
/// Timestamps are caller-supplied `Duration`s since an arbitrary reference point (the driver uses
/// time since the supervisor started), not a wall clock read internally — the same "inject the
/// impure input" discipline [`mod@crate::preflight`] and [`cerebro_probe::child_env::build_child_env`]
/// already use for their own host state, and what lets this type's tests run without a
/// real clock at all.
#[derive(Debug, Default)]
pub(crate) struct RestartWindow {
    history: VecDeque<Duration>,
}

impl RestartWindow {
    /// A window with no restarts recorded yet.
    pub(crate) fn new() -> Self {
        Self {
            history: VecDeque::new(),
        }
    }

    /// Record a restart at `now`, evict every entry older than `policy.window`, and return how many
    /// restarts — including this one — remain counted inside the window.
    ///
    /// This count is deliberately the single value both the restart-limit check and
    /// [`backoff_delay`]'s exponent are keyed off: a supervisor that has been stable for longer than
    /// `window` sees old restarts age out of *both* at once, rather than the limit resetting while
    /// the backoff delay stays inflated by history the limit itself has already forgotten.
    pub(crate) fn record(&mut self, policy: &RestartPolicy, now: Duration) -> u32 {
        while let Some(&front) = self.history.front() {
            if now.saturating_sub(front) > policy.window {
                self.history.pop_front();
            } else {
                break;
            }
        }
        self.history.push_back(now);
        self.history.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> RestartPolicy {
        RestartPolicy {
            base_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            max_restarts: 5,
            window: Duration::from_secs(60),
            jitter_seed: 42,
        }
    }

    #[test]
    fn backoff_delay_is_deterministic_for_the_same_seed_and_attempt() {
        let p = policy();
        assert_eq!(backoff_delay(&p, 3), backoff_delay(&p, 3));
    }

    #[test]
    fn backoff_delay_differs_across_seeds() {
        // Not a mathematical guarantee for every possible pair, but SplitMix64 mixes its input well
        // enough that two arbitrary, unrelated seeds never collide in practice — if this ever did
        // fail, it would mean the jitter derivation stopped depending on the seed at all, which is
        // the actual property being protected here.
        let mut a = policy();
        let mut b = policy();
        a.jitter_seed = 1;
        b.jitter_seed = 2;
        assert_ne!(backoff_delay(&a, 3), backoff_delay(&b, 3));
    }

    #[test]
    fn backoff_delay_never_exceeds_the_cap() {
        let p = policy();
        for attempt in 1..=64 {
            assert!(backoff_delay(&p, attempt) <= p.max_backoff, "attempt {attempt}");
        }
    }

    #[test]
    fn backoff_delay_grows_with_attempt_before_hitting_the_cap() {
        // Full jitter is randomized, so a single pair of attempts could tie or invert by chance;
        // averaging several jitter seeds at each attempt is what actually distinguishes "roughly
        // doubling" from "flat" or "capped already".
        let p = policy();
        let avg = |attempt: u32| -> Duration {
            let total: Duration = (0..64)
                .map(|seed| {
                    let mut trial = p.clone();
                    trial.max_backoff = Duration::from_secs(3600); // clear of the cap for this comparison
                    trial.jitter_seed = seed;
                    backoff_delay(&trial, attempt)
                })
                .sum();
            total / 64
        };
        assert!(avg(2) > avg(1), "{:?} should exceed {:?}", avg(2), avg(1));
        assert!(avg(4) > avg(2), "{:?} should exceed {:?}", avg(4), avg(2));
    }

    #[test]
    fn restart_window_counts_up_within_the_window() {
        let p = policy();
        let mut w = RestartWindow::new();
        assert_eq!(w.record(&p, Duration::from_secs(0)), 1);
        assert_eq!(w.record(&p, Duration::from_secs(1)), 2);
        assert_eq!(w.record(&p, Duration::from_secs(2)), 3);
    }

    #[test]
    fn restart_window_evicts_entries_older_than_the_window() {
        let p = policy();
        let mut w = RestartWindow::new();
        assert_eq!(w.record(&p, Duration::from_secs(0)), 1);
        assert_eq!(w.record(&p, Duration::from_secs(10)), 2);
        // 90s later, both prior restarts (at 0s and 10s) are outside the 60s window.
        assert_eq!(w.record(&p, Duration::from_secs(100)), 1);
    }

    #[test]
    fn restart_window_can_exceed_max_restarts() {
        let p = policy();
        let mut w = RestartWindow::new();
        let mut last = 0;
        for i in 0..10 {
            last = w.record(&p, Duration::from_millis(i));
        }
        assert!(last > p.max_restarts, "{last} should exceed {}", p.max_restarts);
    }
}
