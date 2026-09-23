//! Deciding exactly what environment block a spawned `llama-server` child receives.
//!
//! # Why this is a closed allowlist, not an inherited environment
//!
//! `llama.cpp` binds several flags to environment variable names (`LLAMA_ARG_PORT`,
//! `LLAMA_ARG_DEVICE`, `-dev` -> `LLAMA_ARG_DEVICE`, …) — see ARCHITECTURE.md section 9. If the
//! child inherited Cerebro's own process environment, any `LLAMA_ARG_*` variable left over from a
//! previous run, or set by an unrelated program on the same machine, would silently override an
//! argv value [`crate::preflight::build_hardened_server_argv`] already proved safe. The only way to
//! make that impossible is to never let it happen: `env_clear` on the [`std::process::Command`],
//! then rebuild the block from nothing, one name at a time, from a fixed list, exactly as
//! [`cerebro_flags::Flag`] closes the argv surface for the same reason.
//!
//! # Provenance, not just naming
//!
//! A closed allowlist is not enough by itself: it only decides which *names* survive, and says
//! nothing about where the *value* behind a surviving name came from. Two of this child's most
//! security-relevant values — `PATH` and `SystemRoot`/`windir` — are directory paths that steer
//! ggml's DLL search (ARCHITECTURE.md section 8 item 3), so the rule here is deliberately stronger
//! than "is the name on the list": a variable whose value will be interpreted as a path, a
//! directory, or a command is never *copied* from the parent, only *constructed* from a source this
//! process trusts — [`system_directory`]'s call to `GetSystemDirectoryW`, or the caller-supplied
//! `backend_dir`/`temp_dir`. Copying `SystemRoot`'s *value* out of the parent environment and
//! interpolating it into `PATH`, as an earlier version of this module did, would let a parent
//! launched with a poisoned `SystemRoot` plant an attacker-controlled `PATH` entry even though
//! `SystemRoot` is itself an allowlisted *name* — the allowlist is for values the child merely
//! reads, never for values that steer where the child looks.
//!
//! [`build_child_env`] is a pure function of an injected parent-environment iterator and an injected
//! system directory, not of `std::env::vars_os()` or `GetSystemDirectoryW` directly, for the same
//! reason [`crate::preflight`] injects [`crate::preflight::ElevationCheck`] and
//! [`crate::preflight::DaclAuditor`] rather than reading ambient host state inline: a test proving
//! "a `LLAMA_ARG_*` variable in the parent does not reach the child", or "a poisoned `SystemRoot` in
//! the parent does not reach `PATH`", must not depend on mutating this test binary's real
//! process-wide environment, which every other test thread in the same binary would also observe.

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::Path;

/// Names copied verbatim from Cerebro's own environment into the child's, when present there.
///
/// Shrunk deliberately to the two values that cannot themselves name a path, a directory, or a
/// command — `NUMBER_OF_PROCESSORS` and `PROCESSOR_ARCHITECTURE` are plain scalars the CRT reads
/// during startup on some builds, and copying either one's *value* verbatim carries none of the
/// DLL-search-path hazard a copied `SystemRoot` would (see this module's top-level docs). Everything
/// else Cerebro's own process happens to have set — including any stray `LLAMA_ARG_*` variable, a
/// proxy setting, or an unrelated tool's configuration — is dropped.
const COPIED_ALLOWLIST: &[&str] = &["NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE"];

/// Resolve the real Windows system directory (ordinarily `C:\Windows\System32`) from the kernel,
/// never from an inherited `SystemRoot`/`windir` variable — see this module's top-level docs on why
/// a directory-valued variable is never trusted merely because its *name* is allowlisted.
#[cfg(windows)]
pub(super) fn system_directory() -> Result<std::path::PathBuf, super::SpawnError> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;

    // A system directory has never come close to `MAX_PATH` (260 `u16`s) on any supported Windows
    // release; this is sized generously above that as a fixed, stack-allocated buffer rather than a
    // two-call "ask the required size, then allocate" dance this one-shot call does not need.
    let mut buf = [0u16; 1024];
    // SAFETY: `buf` is a live array of exactly `buf.len()` `u16`s, and that same length is the count
    // passed as the buffer size, so the call cannot write past it. The return value is checked
    // below before any of `buf` is read: `0` means failure (`GetLastError` is read immediately
    // afterward, before any other API call can overwrite it), and a value `>= buf.len()` means the
    // buffer was too small and nothing was written (the value is then the required size including
    // the terminating NUL) — both are rejected rather than used to index `buf`.
    let len = unsafe { GetSystemDirectoryW(buf.as_mut_ptr(), buf.len() as u32) } as usize;
    if len == 0 {
        // SAFETY: reads the calling thread's last-error value immediately after the failed call
        // above, before any other API call can overwrite it.
        let code = unsafe { GetLastError() };
        return Err(super::SpawnError::SystemDirectoryUnavailable(format!(
            "GetSystemDirectoryW failed with Win32 error {code}"
        )));
    }
    if len >= buf.len() {
        return Err(super::SpawnError::SystemDirectoryUnavailable(format!(
            "GetSystemDirectoryW needs {len} UTF-16 units, more than this fixed {}-unit buffer",
            buf.len()
        )));
    }
    Ok(std::path::PathBuf::from(OsString::from_wide(&buf[..len])))
}

/// Build the exact environment block a `llama-server` child receives.
///
/// `parent_env` stands in for `std::env::vars_os()` (the real caller in [`super::spawn_windows_at`]
/// passes exactly that); every name not in [`COPIED_ALLOWLIST`] is dropped regardless of what it is
/// set to, `LLAMA_ARG_*` included. `system32` stands in for [`system_directory`]'s result (the real
/// caller resolves it via `GetSystemDirectoryW`, never from the parent environment); `SystemRoot`,
/// `windir` and `SystemDrive` are derived from it rather than copied, and `PATH` is built fresh as
/// exactly `backend_dir` followed by `system32` — so the child's DLL and executable search path is
/// the audited backend directory and the OS's own answer for its system directory, and nothing an
/// ambient `PATH` or `SystemRoot` entry could have added. `TEMP`/`TMP` are likewise never copied:
/// they are pinned to `temp_dir`, which the caller must already own and have created for Cerebro's
/// use (see [`super::SpawnOptions::temp_dir`]).
///
/// Every path value is concatenated with [`OsString::push`], never through `Path::display` or
/// `to_string_lossy`: a Windows path component is UTF-16 and can contain an unpaired surrogate that
/// a lossy conversion would silently replace with `U+FFFD`, turning a valid directory into a
/// different, wrong one rather than an error.
///
/// Returns a sorted map (not a `Vec`) so both the emitted `PATH`/`TEMP`/`TMP` overrides and the
/// allowlisted names can never appear twice — `Command::envs` would otherwise happily accept
/// duplicate keys and let the second silently win, which would defeat the point of a function whose
/// entire job is to make the resulting block exhaustively enumerable by a reviewer.
pub(super) fn build_child_env(
    parent_env: impl Iterator<Item = (OsString, OsString)>,
    system32: &Path,
    backend_dir: &Path,
    temp_dir: &Path,
    cuda_visible_devices: Option<&str>,
) -> BTreeMap<OsString, OsString> {
    let parent: BTreeMap<OsString, OsString> = parent_env.collect();
    let mut env = BTreeMap::new();

    for &name in COPIED_ALLOWLIST {
        if let Some(value) = parent.get(OsStr::new(name)) {
            env.insert(OsString::from(name), value.clone());
        }
    }

    // `system32` is ordinarily `<SystemRoot>\System32`; its parent is `SystemRoot`/`windir` (the
    // same directory under two historical names), and that directory's own root component is
    // `SystemDrive`. All three are derived from the OS-resolved path, never from the parent's own
    // `SystemRoot`/`windir`/`SystemDrive` — see this module's top-level docs.
    if let Some(system_root) = system32.parent() {
        env.insert(OsString::from("SystemRoot"), system_root.as_os_str().to_owned());
        env.insert(OsString::from("windir"), system_root.as_os_str().to_owned());
        if let Some(std::path::Component::Prefix(prefix)) = system_root.components().next() {
            env.insert(
                OsString::from("SystemDrive"),
                prefix.as_os_str().to_owned(),
            );
        }
    }

    let mut path = backend_dir.as_os_str().to_owned();
    path.push(";");
    path.push(system32.as_os_str());
    env.insert(OsString::from("PATH"), path);

    env.insert(OsString::from("TEMP"), temp_dir.as_os_str().to_owned());
    env.insert(OsString::from("TMP"), temp_dir.as_os_str().to_owned());

    if let Some(cuda) = cuda_visible_devices {
        env.insert(OsString::from("CUDA_VISIBLE_DEVICES"), OsString::from(cuda));
    }

    env
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(pairs: &[(&str, &str)]) -> impl Iterator<Item = (OsString, OsString)> {
        pairs
            .iter()
            .map(|(k, v)| (OsString::from(*k), OsString::from(*v)))
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// A realistic `system32` fixture, standing in for [`system_directory`]'s real result without
    /// this test binary calling `GetSystemDirectoryW` itself.
    const SYSTEM32: &str = r"C:\Windows\System32";

    #[test]
    fn llama_arg_variable_in_the_parent_never_reaches_the_child() {
        // The exact hazard ARCHITECTURE.md section 9 names: `LLAMA_ARG_PORT` bypassing the closed
        // argv allowlist by overriding `--port` through the environment instead.
        let parent = vars(&[("LLAMA_ARG_PORT", "9999"), ("SystemRoot", r"C:\Windows")]);
        let env = build_child_env(
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            Path::new(r"C:\cerebro-temp"),
            None,
        );
        assert!(
            !env.keys().any(|k| k.to_string_lossy().starts_with("LLAMA_ARG_")),
            "{env:?}"
        );
    }

    #[test]
    fn only_allowlisted_and_constructed_names_survive() {
        // Proves the allowlist is closed, not merely that one named variable is absent: an
        // unrelated variable (here `USERPROFILE`, `HTTP_PROXY`) must be dropped exactly as
        // thoroughly as an `LLAMA_ARG_*` one would be.
        let parent = vars(&[
            ("USERPROFILE", r"C:\Users\noahm"),
            ("HTTP_PROXY", "http://example.invalid:8080"),
            ("SystemRoot", r"C:\Windows"),
            ("SystemDrive", "C:"),
            ("windir", r"C:\Windows"),
            ("NUMBER_OF_PROCESSORS", "8"),
            ("PROCESSOR_ARCHITECTURE", "AMD64"),
        ]);
        let env = build_child_env(
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            Path::new(r"C:\cerebro-temp"),
            None,
        );
        let mut names: Vec<String> = env.keys().map(|k| k.to_string_lossy().into_owned()).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "NUMBER_OF_PROCESSORS",
                "PATH",
                "PROCESSOR_ARCHITECTURE",
                "SystemDrive",
                "SystemRoot",
                "TEMP",
                "TMP",
                "windir",
            ]
        );
    }

    #[test]
    fn path_is_backend_dir_then_system32_and_nothing_else() {
        let parent = vars(&[("SystemRoot", r"C:\Windows"), ("PATH", r"C:\evil;C:\also-evil")]);
        let env = build_child_env(
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            Path::new(r"C:\cerebro-temp"),
            None,
        );
        assert_eq!(
            env.get(OsStr::new("PATH")).map(|v| v.to_string_lossy().into_owned()),
            Some(format!(r"C:\backend;{SYSTEM32}"))
        );
    }

    #[test]
    fn path_system32_component_comes_from_the_os_not_the_parent_environment() {
        // This is the test that would have caught the finding: a parent whose own `SystemRoot`
        // points somewhere attacker-controlled must have zero influence on the child's `PATH` once
        // `system32` is supplied from a trusted source instead.
        let parent = vars(&[
            ("SystemRoot", r"\\attacker\share"),
            ("windir", r"\\attacker\share"),
            ("SystemDrive", r"\\attacker"),
        ]);
        let env = build_child_env(
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            Path::new(r"C:\cerebro-temp"),
            None,
        );
        let path = env
            .get(OsStr::new("PATH"))
            .map(|v| v.to_string_lossy().into_owned())
            .unwrap_or_default();
        assert_eq!(path, format!(r"C:\backend;{SYSTEM32}"));
        assert!(!path.contains("attacker"), "{path}");
    }

    #[test]
    fn system_root_and_windir_are_derived_from_the_resolved_system_directory() {
        let parent = vars(&[
            ("SystemRoot", r"\\attacker\share"),
            ("windir", r"\\attacker\share"),
            ("SystemDrive", r"\\attacker"),
        ]);
        let env = build_child_env(
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            Path::new(r"C:\cerebro-temp"),
            None,
        );
        assert_eq!(env.get(OsStr::new("SystemRoot")), Some(&OsString::from(r"C:\Windows")));
        assert_eq!(env.get(OsStr::new("windir")), Some(&OsString::from(r"C:\Windows")));
        assert_eq!(env.get(OsStr::new("SystemDrive")), Some(&OsString::from("C:")));
        for value in env.values() {
            assert!(!value.to_string_lossy().contains("attacker"), "{env:?}");
        }
    }

    #[test]
    fn path_is_byte_exact_for_a_non_utf8_backend_dir() {
        // An unpaired UTF-16 surrogate is not valid Unicode and has no lossless UTF-8
        // representation; `format!`/`Path::display` would silently turn it into `U+FFFD`; only a
        // byte-exact `OsString` concatenation preserves it.
        use std::ffi::OsString;
        use std::os::windows::ffi::OsStringExt;
        let weird = OsString::from_wide(&[0x0043, 0x003A, 0x005C, 0xD800]); // "C:\" + lone surrogate
        let backend_dir = std::path::PathBuf::from(&weird);

        let env = build_child_env(
            vars(&[]),
            Path::new(SYSTEM32),
            &backend_dir,
            Path::new(r"C:\cerebro-temp"),
            None,
        );
        let path = env.get(OsStr::new("PATH")).expect("PATH must be set");
        let mut expected = weird.clone();
        expected.push(";");
        expected.push(SYSTEM32);
        assert_eq!(path, &expected);

        // `to_string_lossy` is *supposed* to replace an unpaired surrogate with `U+FFFD` — that is
        // what "lossy" means, and asserting its absence from a lossy conversion of a genuinely
        // invalid sequence would fail on any correct implementation, not just a buggy one. The
        // property this test actually needs is that the lone surrogate code unit itself survived
        // byte-exact through `build_child_env`, which `encode_wide` (no lossy step at all) can
        // check directly: it must still be `0xD800`, never replaced by `0xFFFD`.
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = path.encode_wide().collect();
        assert!(
            wide.contains(&0xD800),
            "the lone surrogate did not survive byte-exact: {wide:?}"
        );
        assert!(
            !wide.contains(&(0xFFFDu32 as u16)),
            "a byte-exact concatenation must never introduce a replacement character: {wide:?}"
        );
    }

    #[test]
    fn temp_and_tmp_are_pinned_to_the_caller_directory_never_the_parents() {
        let parent = vars(&[("TEMP", r"C:\Users\someone\AppData\Local\Temp"), ("TMP", r"D:\other")]);
        let env = build_child_env(
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            Path::new(r"C:\cerebro-temp"),
            None,
        );
        assert_eq!(env.get(OsStr::new("TEMP")), Some(&OsString::from(r"C:\cerebro-temp")));
        assert_eq!(env.get(OsStr::new("TMP")), Some(&OsString::from(r"C:\cerebro-temp")));
    }

    #[test]
    fn a_semicolon_in_the_backend_dir_would_split_path_into_two_entries() {
        // This function has no reason to know about `;` at all — `build_child_env` is a pure
        // string-concatenation helper, and rejecting an unsafe `backend_dir` is deliberately
        // `spawn_windows_at`'s job, the one choke point both `spawn_unaudited` and
        // `spawn_preflighted` cross (see `SpawnError::BackendDirNotPathSafe`'s own docs). This test
        // documents *why* that refusal has to happen upstream of here: a `backend_dir` containing
        // `;` splits `PATH` into three entries instead of two, the middle one a directory nobody
        // audited.
        let env = build_child_env(
            vars(&[]),
            Path::new(SYSTEM32),
            Path::new(r"C:\a;C:\evil"),
            Path::new(r"C:\t"),
            None,
        );
        let path = env
            .get(OsStr::new("PATH"))
            .expect("PATH must be set")
            .to_string_lossy()
            .into_owned();
        assert_eq!(path.split(';').count(), 3, "{path}");
    }

    #[test]
    fn cuda_visible_devices_is_omitted_when_not_given_and_passed_through_verbatim_when_given() {
        let without = build_child_env(
            vars(&[]),
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            Path::new(r"C:\t"),
            None,
        );
        assert!(!without.contains_key(OsStr::new("CUDA_VISIBLE_DEVICES")));

        let with = build_child_env(
            vars(&[]),
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            Path::new(r"C:\t"),
            Some("0"),
        );
        assert_eq!(with.get(OsStr::new("CUDA_VISIBLE_DEVICES")), Some(&OsString::from("0")));
    }
}
