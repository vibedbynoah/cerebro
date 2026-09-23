//! The one definition of "a bare, unambiguous `.exe` file name that cannot escape its directory".
//!
//! [`preflight::verify_tool_name`](crate::preflight) and [`spawn::verify_spawn_target`](crate::spawn)
//! both need this exact predicate — preflight to decide whether a launch may proceed at all, spawn
//! to re-check the same fact at its own, `Preflighted`-free entry points (see `spawn`'s module docs
//! on why it cannot simply trust preflight already ran). Two independently maintained copies of a
//! security predicate drift: a future hardening applied to one (say, rejecting an 8.3 short name
//! like `PROGRA~1`) could silently leave the other permissive, and two separate `fn` bodies give a
//! compiler-assisted rename nothing to catch. Sharing one function, crate-private, makes that
//! impossible — both call sites map the same `bool` to their own typed error instead.

use std::path::{Component, Path};

/// Whether `name` is a single, bare `.exe` file name that stays inside whatever directory it is
/// joined to.
///
/// Exactly one [`Component::Normal`] (rules out `..`, `.`, a bare drive letter, an absolute path
/// that would make `Path::join` discard the directory it was meant to join with, and a path with
/// any separator in it at all), no trailing `.` or ` ` (Windows silently strips both, so
/// `"llama-server.exe."` and `"llama-server.exe"` would otherwise name the same file), no `:`
/// (which would address an alternate data stream on an otherwise-legitimate file name), and a
/// `.exe` suffix (which also stops a `.dll` from being accepted as the thing to launch).
pub(crate) fn is_bare_exe_name(name: &str) -> bool {
    let mut components = Path::new(name).components();
    let is_single_normal_component = matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(_)), None)
    );
    is_single_normal_component
        && !name.ends_with('.')
        && !name.ends_with(' ')
        && !name.contains(':')
        && name.to_ascii_lowercase().ends_with(".exe")
}

/// Every known shape that must be rejected, in one place: `preflight::verify_tool_name`'s and
/// `spawn::verify_spawn_target`'s own test suites both point back at this table (as thin wrappers
/// asserting the predicate's `bool` maps to each caller's own typed error) instead of each keeping
/// an independently maintained copy — see this module's top-level docs on why a second copy would
/// defeat the point of sharing the predicate at all.
#[cfg(test)]
pub(crate) const BAD_NAMES: &[&str] = &[
    r"..\evil.exe",
    r"C:\Windows\System32\cmd.exe",
    r"\\?\C:\evil.exe",
    r"sub\tool.exe",
    "llama-server.exe:ads",
    "llama-server.exe.",
    "llama-server.exe ",
    "llama-server.dll",
    "..",
];

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
