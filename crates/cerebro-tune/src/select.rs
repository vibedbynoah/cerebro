//! Choosing the winner: decode throughput at the workload's depth, with the never-slower rule,
//! never-worse guards on prefill and depth-0 decode, and a confidence-interval tie rule.
//!
//! The rules, in order:
//!
//! 1. **Never slower than the baseline.** A candidate is eligible only if
//!    - its 95% bootstrap confidence interval for median decode tok/s at depth lies entirely
//!      above the baseline's (`candidate.ci95_low > baseline.ci95_high`). An interval that
//!      overlaps the baseline's is a tie with llama.cpp's own automatic settings, and a tie goes to
//!      the baseline: recommending a configuration that is not demonstrably faster could
//!      recommend one that is slower; and
//!    - neither its prefill nor its depth-0 decode is demonstrably slower than the baseline's
//!      (its interval entirely below the baseline's, [`Guard`]). These are guards, never
//!      objectives: they exclude a candidate, they never rank one. A long prompt is mostly
//!      prefill, so a winner with half the baseline's prefill is slower for the user however fast
//!      it decodes. "Demonstrably below" rather than "overlapping" keeps noise from excluding
//!      everything.
//! 2. **No eligible candidate: the baseline wins**, and the outcome says so.
//! 3. **Fastest, then fewest threads among the statistically tied.** The leader is the eligible
//!    candidate with the highest median; every eligible candidate whose interval overlaps the
//!    leader's is tied with it. Among the tied, the one with the fewest threads wins (cooler, and
//!    leaves cores for everything else), then the higher median, then the earlier candidate. The
//!    tie rule only ever chooses among eligible candidates, so a fewer-thread candidate cannot
//!    bring back a prefill regression the guards excluded.

use cerebro_bench::Summary;

/// What selection needs to know about one measured candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored {
    /// Decode tok/s at the workload's depth: what the winner is chosen by.
    pub tg_at_depth: Summary,
    /// Prefill tok/s at depth 0: guarded, never chosen by.
    pub pp: Summary,
    /// Decode tok/s at depth 0: guarded, never chosen by.
    pub tg_depth0: Summary,
    /// Generation threads as llama-bench reported them (`None`: not reported).
    pub threads: Option<u32>,
}

/// A metric a candidate may not be demonstrably slower at than the baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Guard {
    /// Prefill tok/s at depth 0.
    Prefill,
    /// Decode tok/s at depth 0.
    DecodeAtDepth0,
}

/// The selection's verdict, by index into the candidate slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    /// Candidate `index` is demonstrably faster than the baseline and won.
    Candidate(usize),
    /// No candidate is eligible; the baseline wins.
    Baseline,
}

/// Whether two intervals overlap (touching counts).
fn overlaps(a: &Summary, b: &Summary) -> bool {
    a.ci95_low <= b.ci95_high && b.ci95_low <= a.ci95_high
}

/// `a` is demonstrably below `b`: its whole interval lies under `b`'s.
fn below(a: &Summary, b: &Summary) -> bool {
    a.ci95_high < b.ci95_low
}

/// Whether two decode-at-depth intervals agree (overlap), e.g. two measurements of one baseline.
pub fn agree_at_depth(a: &Scored, b: &Scored) -> bool {
    overlaps(&a.tg_at_depth, &b.tg_at_depth)
}

/// Whether `candidate` stays demonstrably faster at depth than both baseline measurements even if
/// its own measurement was flattered by as much as the baseline moved between them: the low end of
/// its interval, scaled by the slower baseline median over the faster, still lies above both
/// baselines' intervals. Drift that could account for the whole margin fails this; drift far
/// smaller than the margin (a 6x winner, a baseline that moved a few percent) does not.
pub fn faster_despite_drift(candidate: &Scored, first: &Scored, second: &Scored) -> bool {
    let (a, b) = (first.tg_at_depth.median, second.tg_at_depth.median);
    let (slow, fast) = if a <= b { (a, b) } else { (b, a) };
    if !(slow.is_finite() && fast.is_finite() && slow > 0.0) {
        return false;
    }
    let bar = first.tg_at_depth.ci95_high.max(second.tg_at_depth.ci95_high);
    candidate.tg_at_depth.ci95_low * (slow / fast) > bar
}

/// Whether `candidate`'s decode at depth is demonstrably faster than `baseline`'s.
pub fn faster_at_depth(candidate: &Scored, baseline: &Scored) -> bool {
    candidate.tg_at_depth.ci95_low > baseline.tg_at_depth.ci95_high
}

/// The first guard `candidate` breaks against `baseline`, if any.
pub fn broken_guard(candidate: &Scored, baseline: &Scored) -> Option<Guard> {
    if below(&candidate.pp, &baseline.pp) {
        Some(Guard::Prefill)
    } else if below(&candidate.tg_depth0, &baseline.tg_depth0) {
        Some(Guard::DecodeAtDepth0)
    } else {
        None
    }
}

/// Rule 1: demonstrably faster at depth, and no guard broken.
pub fn eligible(candidate: &Scored, baseline: &Scored) -> bool {
    faster_at_depth(candidate, baseline) && broken_guard(candidate, baseline).is_none()
}

/// Choose among `candidates` (the baseline excluded) against `baseline`; see the module docs.
pub fn choose(baseline: &Scored, candidates: &[Scored]) -> Choice {
    let eligible: Vec<(usize, &Scored)> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| eligible(c, baseline))
        .collect();
    // Highest median; on an exact tie the earlier candidate. `total_cmp` orders NaN deterministically
    // (the store refuses non-finite summaries, so none reach here in practice).
    let Some(&(_, leader)) = eligible.iter().reduce(|best, next| {
        if next.1.tg_at_depth.median.total_cmp(&best.1.tg_at_depth.median).is_gt() {
            next
        } else {
            best
        }
    }) else {
        return Choice::Baseline;
    };
    let winner = eligible
        .iter()
        .filter(|(_, c)| overlaps(&c.tg_at_depth, &leader.tg_at_depth))
        .reduce(|best, next| {
            // A thread count llama-bench did not report sorts after every known one: the tie rule
            // can only prefer fewer threads it can see.
            let fewer = next.1.threads.unwrap_or(u32::MAX).cmp(&best.1.threads.unwrap_or(u32::MAX));
            let faster = next.1.tg_at_depth.median.total_cmp(&best.1.tg_at_depth.median);
            if fewer.is_lt() || (fewer.is_eq() && faster.is_gt()) {
                next
            } else {
                best
            }
        });
    winner.map_or(Choice::Baseline, |&(i, _)| Choice::Candidate(i))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A summary with median `m` and interval `m +/- half`.
    pub(crate) fn summary(m: f64, half: f64) -> Summary {
        Summary {
            kept: 3,
            median: m,
            min: m - half,
            max: m + half,
            iqr: half,
            ci95_low: m - half,
            ci95_high: m + half,
        }
    }

    /// A candidate with decode at depth `m +/- half` and guard metrics that never trip.
    fn scored(m: f64, half: f64, threads: Option<u32>) -> Scored {
        Scored { tg_at_depth: summary(m, half), pp: summary(1000.0, 10.0), tg_depth0: summary(100.0, 1.0), threads }
    }

    #[test]
    fn the_fastest_candidate_above_the_baseline_wins() {
        let baseline = scored(3.1, 0.1, Some(14));
        let c = [scored(10.0, 0.2, Some(14)), scored(17.9, 0.3, Some(14)), scored(12.0, 0.2, Some(14))];
        assert_eq!(choose(&baseline, &c), Choice::Candidate(1));
    }

    #[test]
    fn a_candidate_overlapping_the_baseline_never_wins_even_with_a_higher_median() {
        // 81.0 +/- 1.0 overlaps 80.5 +/- 0.5: not demonstrably faster, so the baseline keeps it.
        // (Binary-exact values, so "touching" below is exact.)
        let baseline = scored(80.5, 0.5, Some(14));
        assert_eq!(choose(&baseline, &[scored(81.0, 1.0, Some(4))]), Choice::Baseline);
        // Touching intervals (81.0 = 81.0) are a tie too.
        assert_eq!(choose(&baseline, &[scored(82.0, 1.0, Some(4))]), Choice::Baseline);
        // Strictly above wins.
        assert_eq!(choose(&baseline, &[scored(82.25, 1.0, Some(4))]), Choice::Candidate(0));
    }

    #[test]
    fn when_everything_is_slower_the_baseline_wins() {
        let baseline = scored(80.6, 0.5, Some(14));
        let slower = [scored(74.8, 0.4, Some(6)), scored(60.0, 1.0, Some(3))];
        assert_eq!(choose(&baseline, &slower), Choice::Baseline);
        assert_eq!(choose(&baseline, &[]), Choice::Baseline);
    }

    #[test]
    fn among_statistically_tied_candidates_fewer_threads_wins() {
        let baseline = scored(3.0, 0.1, Some(14));
        let c = [
            scored(18.0, 0.5, Some(12)),
            scored(17.8, 0.5, Some(6)),  // overlaps the leader: tied, fewer threads
            scored(17.6, 0.5, Some(3)),  // overlaps the leader (17.1..18.1 vs 17.5..18.5): fewest
            scored(15.0, 0.5, Some(1)),  // eligible but clearly slower: not tied
        ];
        assert_eq!(choose(&baseline, &c), Choice::Candidate(2));
    }

    #[test]
    fn the_tie_rule_breaks_equal_threads_by_median_then_order_and_unknown_threads_sort_last() {
        let baseline = scored(3.0, 0.1, None);
        let c = [scored(17.8, 0.5, Some(6)), scored(18.0, 0.5, Some(6))];
        assert_eq!(choose(&baseline, &c), Choice::Candidate(1));
        let same = [scored(18.0, 0.5, Some(6)), scored(18.0, 0.5, Some(6))];
        assert_eq!(choose(&baseline, &same), Choice::Candidate(0));
        let unknown = [scored(18.0, 0.5, None), scored(17.9, 0.5, Some(8))];
        assert_eq!(choose(&baseline, &unknown), Choice::Candidate(1));
    }

    #[test]
    fn faster_decode_with_demonstrably_slower_prefill_is_not_eligible() {
        // The live run: 6 threads decoded faster at depth but prefilled at 2866 tok/s against the
        // baseline's 7641.
        let baseline = Scored { pp: summary(7641.0, 50.0), ..scored(80.0, 0.5, Some(14)) };
        let slow_pp = Scored { pp: summary(2866.0, 40.0), ..scored(90.0, 0.5, Some(6)) };
        assert!(faster_at_depth(&slow_pp, &baseline));
        assert_eq!(broken_guard(&slow_pp, &baseline), Some(Guard::Prefill));
        assert_eq!(choose(&baseline, &[slow_pp]), Choice::Baseline);

        let slow_tg0 = Scored { tg_depth0: summary(50.0, 1.0), pp: baseline.pp, ..scored(90.0, 0.5, Some(6)) };
        assert_eq!(broken_guard(&slow_tg0, &baseline), Some(Guard::DecodeAtDepth0));
        assert_eq!(choose(&baseline, &[slow_tg0]), Choice::Baseline);

        // Overlapping, even with a lower median, is not demonstrably slower: still eligible.
        let close = Scored { pp: summary(7600.0, 50.0), ..scored(90.0, 0.5, Some(6)) };
        assert_eq!(broken_guard(&close, &baseline), None);
        assert_eq!(choose(&baseline, &[close]), Choice::Candidate(0));
    }

    #[test]
    fn drift_only_withdraws_a_winner_when_it_could_account_for_the_margin() {
        let first = scored(3.1, 0.1, Some(14));
        let winner = scored(19.1, 0.5, Some(6));
        // The baseline moved 3.1 -> 5.0: 18.6 x 3.1 / 5.0 = 11.5, still above 5.1.
        assert!(faster_despite_drift(&winner, &first, &scored(5.0, 0.1, Some(14))));
        // 3.1 -> 12.0: 18.6 x 3.1 / 12.0 = 4.8, below 12.1, though 18.6 beats 12.1 outright.
        let moved = scored(12.0, 0.1, Some(14));
        assert!(faster_at_depth(&winner, &moved));
        assert!(!faster_despite_drift(&winner, &first, &moved));
        // Symmetric in which measurement was the faster one.
        assert!(!faster_despite_drift(&winner, &moved, &first));
        // A degenerate baseline median proves nothing.
        assert!(!faster_despite_drift(&winner, &scored(0.0, 0.0, None), &first));
    }

    #[test]
    fn the_tie_rule_cannot_pick_a_fewer_thread_candidate_whose_prefill_regressed() {
        let baseline = Scored { pp: summary(7641.0, 50.0), ..scored(3.0, 0.1, Some(14)) };
        let c = [
            Scored { pp: summary(7700.0, 50.0), ..scored(18.0, 0.5, Some(14)) },
            // Tied with the leader and fewer threads, but its prefill is demonstrably slower.
            Scored { pp: summary(2866.0, 40.0), ..scored(17.9, 0.5, Some(6)) },
        ];
        assert_eq!(choose(&baseline, &c), Choice::Candidate(0));
        // Without the regression the fewer-thread candidate wins the tie.
        let fixed = [c[0], Scored { pp: summary(7650.0, 50.0), ..c[1] }];
        assert_eq!(choose(&baseline, &fixed), Choice::Candidate(1));
    }
}
