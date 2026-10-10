//! The one definition of "a bare, unambiguous `.exe` file name that cannot escape its directory".
//!
//! [`preflight::verify_tool_name`](crate::preflight) and [`spawn::verify_spawn_target`](crate::spawn)
//! both need this exact predicate — preflight to decide whether a launch may proceed at all, spawn
//! to re-check the same fact at its own, `Preflighted`-free entry points (see `spawn`'s module docs
//! on why it cannot simply trust preflight already ran) — and so does `cerebro_probe::backend::probe_with`,
//! which runs the same tool for its `--version` and `--list-devices` queries. Independently
//! maintained copies of a security predicate drift: a future hardening applied to one could silently
//! leave another permissive. So the predicate and its table of rejected shapes live once, in
//! `cerebro-probe` (the lower crate), and this module only re-exports them; each call site maps the
//! same `bool` to its own typed error.

pub(crate) use cerebro_probe::backend::is_bare_exe_name;

/// Every known shape that must be rejected: `cerebro_probe::backend::REJECTED_TOOL_NAMES`, the one
/// table shared with `probe_with`'s own tests. `preflight::verify_tool_name`'s and
/// `spawn::verify_spawn_target`'s test suites point back at this name.
#[cfg(test)]
pub(crate) use cerebro_probe::backend::REJECTED_TOOL_NAMES as BAD_NAMES;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_every_known_escape_shape() {
        for bad in BAD_NAMES {
            assert!(!is_bare_exe_name(bad), "{bad:?} should have been rejected");
        }
    }

    #[test]
    fn accepts_a_bare_exe_name() {
        assert!(is_bare_exe_name("llama-server.exe"));
    }
}
