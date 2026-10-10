//! The one definition of the environment block a spawned llama.cpp process receives.
//!
//! Cerebro starts backend binaries from two places: [`crate::backend::probe_with`]'s `--version` and
//! `--list-devices` queries, and `cerebro-supervisor`'s supervised spawn of `llama-server`. Both
//! load every ggml DLL, so both must build the child's environment the same way, under the same user
//! setting. This module is that single definition; neither call site builds an environment itself.
//!
//! # Why this is a closed allowlist, not an inherited environment
//!
//! Two classes of variable steer a llama.cpp child through its environment rather than its argv:
//!
//! - `GGML_BACKEND_PATH`: the shipped `ggml.dll` reads it and `LoadLibrary`s the named file. An
//!   inherited value therefore loads a DLL from outside the backend directory that the folder audit
//!   examined and the integrity check hashed, which is code execution around both protections.
//! - `LLAMA_ARG_*`: llama.cpp binds several flags to environment names (`LLAMA_ARG_PORT`,
//!   `LLAMA_ARG_DEVICE`, ...). An inherited one would silently override an argv value
//!   `cerebro-supervisor` already proved safe.
//!
//! The only way to make both impossible is to never let them through: `env_clear` on the
//! [`std::process::Command`], then rebuild the block from nothing, one name at a time, from a fixed
//! list ([`build_child_env`]).
//!
//! # Provenance, not just naming
//!
//! A closed allowlist decides which *names* survive, not where the *value* behind a surviving name
//! came from. `PATH` and `SystemRoot`/`windir` steer the DLL search, so a variable whose value is a
//! path, a directory, or a command is never *copied* from the parent, only *constructed* from a
//! source this process trusts: [`system_directory`]'s call to `GetSystemDirectoryW`, or the
//! caller-supplied backend and temp directories. Copying `SystemRoot`'s value out of the parent and
//! interpolating it into `PATH` would let a parent launched with a poisoned `SystemRoot` plant an
//! attacker-controlled `PATH` entry even though `SystemRoot` is an allowlisted name.
//!
//! # Driver caches
//!
//! GPU drivers persist what they compile: the CUDA driver JIT-compiles PTX into the ComputeCache
//! (`%APPDATA%\NVIDIA\ComputeCache` unless `CUDA_CACHE_PATH` says otherwise) whenever the build
//! carries no native code for the GPU (a newer architecture than the build targets), and the
//! Vulkan/DirectX drivers of NVIDIA, AMD and Intel keep pipeline caches under `%LOCALAPPDATA%`. The
//! closed block drops both folders, so without a replacement every launch, restart and probe query
//! recompiles. Measured 2026-09-29 on the official CUDA build b10603 (RTX 3070 Ti) with the 0.5B
//! model, fully offloaded, and PTX JIT forced (`CUDA_FORCE_PTX_JIT=1`), three launches each:
//! load-to-ready took 19.0-22.0 s on every launch with the folders dropped, 20.5 s on the first
//! launch with a pinned cache and 1.21-1.39 s on every launch after it (28 MB cached), against
//! 1.25-1.53 s for llama-server run directly with a warm ComputeCache and 1.33 s with no JIT at all.
//! `tests/live_driver_cache.rs` in `cerebro-supervisor` repeats the measurement.
//!
//! So a caller passes a `cache_dir` it owns and has checked, and [`build_child_env`] points
//! `CUDA_CACHE_PATH` (`<cache_dir>\cuda`), `LOCALAPPDATA` and `APPDATA` at it. The values are
//! constructed from that directory, never copied from the parent, for the reason in "Provenance"
//! above. Keeping one cache directory per backend build stops one build's compiled kernels from
//! being offered to another and keeps the folder's contents attributable. The probe's queries get
//! the same directory as the server, so identifying a backend warms the cache the server then uses.
//!
//! [`build_child_env`] and [`effective_child_env`] are pure functions of an injected parent
//! environment and an injected system directory, not of `std::env::vars_os()` or
//! `GetSystemDirectoryW` directly, so a test proving "a `GGML_BACKEND_PATH` in the parent does not
//! reach the child" never has to mutate the test binary's real, process-wide environment.

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::Path;

#[cfg(windows)]
use crate::error::ProbeError;

/// Names copied verbatim from Cerebro's own environment into the child's, when present there.
///
/// Only values that cannot name a path, a directory, or a command: `NUMBER_OF_PROCESSORS` and
/// `PROCESSOR_ARCHITECTURE` are plain scalars the CRT reads during startup on some builds. Everything
/// else Cerebro's own process happens to have set (`GGML_BACKEND_PATH`, any `LLAMA_ARG_*`, a proxy
/// setting, an unrelated tool's configuration) is dropped while the allowlist is enforced.
///
/// `APPDATA` and `LOCALAPPDATA` are deliberately absent although GPU drivers keep their compile
/// caches under them: they are directory-valued, so they are constructed from the caller's verified
/// cache directory instead (see "Driver caches" in this module's docs and [`CACHE_DIR_NAMES`]).
pub const COPIED_ALLOWLIST: &[&str] = &["NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE"];

/// Names [`build_child_env`] sets to the caller's `cache_dir` itself (alongside `CUDA_CACHE_PATH`,
/// which gets its `cuda` subdirectory): the per-user data folders NVIDIA, AMD and Intel drivers put
/// their Vulkan/DirectX pipeline caches under, and the default parent of the CUDA ComputeCache.
pub const CACHE_DIR_NAMES: &[&str] = &["APPDATA", "LOCALAPPDATA"];

/// Resolve the real Windows system directory (ordinarily `C:\Windows\System32`) from the kernel,
/// never from an inherited `SystemRoot`/`windir` variable; see this module's docs on why a
/// directory-valued variable is never trusted merely because its name is allowlisted.
///
/// # Platform
///
/// Windows only: there is no system directory to resolve anywhere else.
#[cfg(windows)]
pub fn system_directory() -> Result<std::path::PathBuf, ProbeError> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;

    // A system directory has never come close to `MAX_PATH` (260 `u16`s) on any supported Windows
    // release; this fixed buffer is sized generously above that.
    let mut buf = [0u16; 1024];
    // SAFETY: `buf` is a live array of exactly `buf.len()` `u16`s and that same length is passed as
    // its size, so the call cannot write past it. The return value is checked before any of `buf` is
    // read: `0` means failure, and a value `>= buf.len()` means the buffer was too small and nothing
    // was written. Both are rejected rather than used to index `buf`.
    let len = unsafe { GetSystemDirectoryW(buf.as_mut_ptr(), buf.len() as u32) } as usize;
    if len == 0 {
        // SAFETY: reads the calling thread's last-error value immediately after the failed call
        // above, before any other API call can overwrite it.
        let code = unsafe { GetLastError() };
        return Err(ProbeError::SystemDirectoryUnavailable {
            cause: format!("GetSystemDirectoryW failed with Win32 error {code}"),
        });
    }
    // `>= buf.len()` is the too-small answer (the required size, including the NUL); `get` then
    // cannot fail, but costs nothing and cannot panic if the API ever reported otherwise.
    let wide = buf.get(..len).filter(|_| len < buf.len()).ok_or_else(|| {
        ProbeError::SystemDirectoryUnavailable {
            cause: format!(
                "GetSystemDirectoryW needs {len} UTF-16 units, more than this fixed {}-unit buffer",
                buf.len()
            ),
        }
    })?;
    Ok(std::path::PathBuf::from(OsString::from_wide(wide)))
}

/// Build the exact, closed environment block a backend child receives while the environment
/// allowlist is enforced.
///
/// `parent_env` stands in for `std::env::vars_os()`; every name not in [`COPIED_ALLOWLIST`] is
/// dropped regardless of its value, `GGML_BACKEND_PATH` and `LLAMA_ARG_*` included. `system32`
/// stands in for [`system_directory`]'s result; `SystemRoot`, `windir` and `SystemDrive` are derived
/// from it rather than copied, and `PATH` is built fresh as exactly `backend_dir` followed by
/// `system32`. So the child's DLL and executable search path is the audited backend directory and
/// the OS's own answer for its system directory, and nothing an ambient `PATH` or `SystemRoot` could
/// have added.
///
/// `temp_dir`, when given, pins `TEMP`/`TMP` to a directory the caller owns (the supervisor's spawn
/// passes its verified `SpawnOptions::temp_dir`); `None` omits both, which is what the probe's short
/// queries use, since they write no temp files. `cuda_visible_devices`, when given, is passed
/// through verbatim; `None` omits the variable, so every device is visible.
///
/// `cache_dir`, when given, is where GPU drivers may keep compiled kernels and pipelines for this
/// child: `CUDA_CACHE_PATH` is set to `<cache_dir>\cuda` and `LOCALAPPDATA`/`APPDATA` to `cache_dir`
/// itself. See "Driver caches" in this module's docs for why; `None` omits all three.
///
/// Every path value is concatenated with [`OsString::push`], never through `Path::display` or
/// `to_string_lossy`: a Windows path is UTF-16 and can contain an unpaired surrogate that a lossy
/// conversion would silently replace with `U+FFFD`, turning a valid directory into a different one.
///
/// Returns a sorted map so no name can appear twice: `Command::envs` would otherwise accept
/// duplicate keys and let the second silently win.
pub fn build_child_env(
    parent_env: impl Iterator<Item = (OsString, OsString)>,
    system32: &Path,
    backend_dir: &Path,
    temp_dir: Option<&Path>,
    cuda_visible_devices: Option<&str>,
    cache_dir: Option<&Path>,
) -> BTreeMap<OsString, OsString> {
    let parent: BTreeMap<OsString, OsString> = parent_env.collect();
    let mut env = BTreeMap::new();

    for &name in COPIED_ALLOWLIST {
        if let Some(value) = parent.get(OsStr::new(name)) {
            env.insert(OsString::from(name), value.clone());
        }
    }

    // `system32` is ordinarily `<SystemRoot>\System32`; its parent is `SystemRoot`/`windir` (one
    // directory under two historical names), and that directory's root component is `SystemDrive`.
    // All three come from the OS-resolved path, never from the parent's own variables.
    if let Some(system_root) = system32.parent() {
        env.insert(
            OsString::from("SystemRoot"),
            system_root.as_os_str().to_owned(),
        );
        env.insert(OsString::from("windir"), system_root.as_os_str().to_owned());
        if let Some(std::path::Component::Prefix(prefix)) = system_root.components().next() {
            env.insert(OsString::from("SystemDrive"), prefix.as_os_str().to_owned());
        }
    }

    let mut path = backend_dir.as_os_str().to_owned();
    path.push(";");
    path.push(system32.as_os_str());
    env.insert(OsString::from("PATH"), path);

    if let Some(temp_dir) = temp_dir {
        env.insert(OsString::from("TEMP"), temp_dir.as_os_str().to_owned());
        env.insert(OsString::from("TMP"), temp_dir.as_os_str().to_owned());
    }

    if let Some(cuda) = cuda_visible_devices {
        env.insert(OsString::from("CUDA_VISIBLE_DEVICES"), OsString::from(cuda));
    }

    // Constructed from the caller's verified directory, never copied from the parent: all three are
    // directory-valued, which this module's provenance rule forbids inheriting.
    if let Some(cache) = cache_dir {
        for &name in CACHE_DIR_NAMES {
            env.insert(OsString::from(name), cache.as_os_str().to_owned());
        }
        let mut cuda = cache.as_os_str().to_owned();
        cuda.push(r"\cuda");
        env.insert(OsString::from("CUDA_CACHE_PATH"), cuda);
    }

    env
}

/// The environment block a backend child receives under the user's environment-allowlist setting:
/// [`build_child_env`] when `enforce_allowlist` is `true`, the inherited environment with the launch
/// settings re-pinned when it is `false`. Both Cerebro call sites that start a backend
/// (`probe_with`'s queries and the supervisor's spawn) call this, so they cannot honour the one
/// setting differently.
///
/// Off means the child sees the environment it would see if Cerebro did not filter it, apart from
/// values that are explicit launch settings. It does not simply hand back `parent_env`, because
/// turning one protection off must never change a different setting's effect:
///
/// - `cuda_visible_devices` would stop reaching the child, which can land it on the wrong GPU and
///   silently invalidate whatever was measured with the allowlist on; and
/// - `TEMP`/`TMP` would stop being pinned to the caller's verified `temp_dir`; and
/// - the driver caches would move back to the user's own folders, away from the verified
///   `cache_dir` the allowlist-on launches warmed.
///
/// `PATH` is different: it is the core of what the allowlist controls (which directories the loader
/// searches for DLLs), so with the allowlist off the inherited `PATH` reaches the child. A
/// CUDA/HIP/SYCL build that relies on the toolkit's `bin` directory being on `PATH` then loads as it
/// would outside Cerebro. The pinned `backend_dir;System32` entries still come first, so a stray
/// `ggml-*.dll` elsewhere on the user's `PATH` cannot shadow the backend's own build. An empty
/// inherited `PATH` adds nothing: a trailing `;` is an empty entry, which some loaders read as the
/// current directory.
///
/// Pinned names replace inherited ones matched case-insensitively, since Windows environment names
/// are, and a raw `BTreeMap<OsString, _>` is not: an inherited `Path` would otherwise sort after a
/// pinned `PATH` and silently win. `CUDA_VISIBLE_DEVICES` is pinned even when
/// `cuda_visible_devices` is `None`: that means "omit it", a decision the launch settings already
/// made, not one an ambient value should override. `TEMP`/`TMP` are pinned only when `temp_dir` is
/// given, and `CUDA_CACHE_PATH`/`LOCALAPPDATA`/`APPDATA` only when `cache_dir` is; otherwise the
/// inherited values pass through, like any other unpinned name.
pub fn effective_child_env(
    enforce_allowlist: bool,
    parent_env: impl Iterator<Item = (OsString, OsString)>,
    system32: &Path,
    backend_dir: &Path,
    temp_dir: Option<&Path>,
    cuda_visible_devices: Option<&str>,
    cache_dir: Option<&Path>,
) -> BTreeMap<OsString, OsString> {
    if enforce_allowlist {
        return build_child_env(
            parent_env,
            system32,
            backend_dir,
            temp_dir,
            cuda_visible_devices,
            cache_dir,
        );
    }

    let parent: Vec<(OsString, OsString)> = parent_env.collect();
    // The settings half of `build_child_env`'s output, computed against an empty parent so it holds
    // exactly the pinned names and nothing a `COPIED_ALLOWLIST` entry would add.
    let mut pinned = build_child_env(
        std::iter::empty(),
        system32,
        backend_dir,
        temp_dir,
        cuda_visible_devices,
        cache_dir,
    );
    let inherited_path = parent
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("PATH"))
        .map(|(_, v)| v)
        .filter(|v| !v.is_empty());
    if let Some(inherited) = inherited_path {
        if let Some((_, path)) = pinned
            .iter_mut()
            .find(|(k, _)| k.eq_ignore_ascii_case("PATH"))
        {
            path.push(";");
            path.push(inherited);
        }
    }
    let is_pinned = |k: &OsStr| {
        pinned.keys().any(|p| p.eq_ignore_ascii_case(k))
            || k.eq_ignore_ascii_case("CUDA_VISIBLE_DEVICES")
    };
    let mut out: BTreeMap<OsString, OsString> =
        parent.into_iter().filter(|(k, _)| !is_pinned(k)).collect();
    out.extend(pinned);
    out
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

    /// A realistic `system32`, standing in for [`system_directory`]'s real result.
    const SYSTEM32: &str = r"C:\Windows\System32";

    fn get<'a>(env: &'a BTreeMap<OsString, OsString>, name: &str) -> Option<&'a OsString> {
        env.get(OsStr::new(name))
    }

    #[test]
    fn llama_arg_and_ggml_backend_path_in_the_parent_never_reach_the_child() {
        // The two hazards this module names: `LLAMA_ARG_PORT` overriding `--port`, and
        // `GGML_BACKEND_PATH` making ggml load a DLL from outside the audited directory.
        let parent = vars(&[
            ("LLAMA_ARG_PORT", "9999"),
            ("GGML_BACKEND_PATH", r"C:\elsewhere\evil.dll"),
            ("SystemRoot", r"C:\Windows"),
        ]);
        let env = build_child_env(
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            Some(Path::new(r"C:\cerebro-temp")),
            None,
            None,
        );
        assert!(
            !env.keys()
                .any(|k| k.to_string_lossy().starts_with("LLAMA_ARG_")),
            "{env:?}"
        );
        assert!(
            !env.keys()
                .any(|k| k.eq_ignore_ascii_case("GGML_BACKEND_PATH")),
            "{env:?}"
        );
    }

    #[test]
    fn only_allowlisted_and_constructed_names_survive() {
        // The allowlist is closed, not merely missing one named variable: unrelated names must be
        // dropped as thoroughly as an `LLAMA_ARG_*` one.
        let parent = vars(&[
            ("USERPROFILE", r"C:\Users\someone"),
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
            Some(Path::new(r"C:\cerebro-temp")),
            None,
            None,
        );
        let names: Vec<String> = env
            .keys()
            .map(|k| k.to_string_lossy().into_owned())
            .collect();
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
    fn without_a_temp_dir_temp_and_tmp_are_omitted_rather_than_inherited() {
        // The probe's queries pass no temp dir; the closed block must then carry no TEMP/TMP at all,
        // never the parent's.
        let parent = vars(&[
            ("TEMP", r"C:\Users\someone\AppData\Local\Temp"),
            ("TMP", r"D:\other"),
        ]);
        let env = build_child_env(
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            None,
            None,
            None,
        );
        assert_eq!(get(&env, "TEMP"), None);
        assert_eq!(get(&env, "TMP"), None);
        assert_eq!(
            get(&env, "PATH"),
            Some(&OsString::from(format!(r"C:\backend;{SYSTEM32}")))
        );
    }

    #[test]
    fn path_is_backend_dir_then_system32_and_nothing_else() {
        let parent = vars(&[
            ("SystemRoot", r"C:\Windows"),
            ("PATH", r"C:\evil;C:\also-evil"),
        ]);
        let env = build_child_env(
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            Some(Path::new(r"C:\cerebro-temp")),
            None,
            None,
        );
        assert_eq!(
            get(&env, "PATH"),
            Some(&OsString::from(format!(r"C:\backend;{SYSTEM32}")))
        );
    }

    #[test]
    fn system_values_come_from_the_os_not_the_parent_environment() {
        // A parent whose own `SystemRoot` points somewhere attacker-controlled must have no
        // influence on the child's `PATH`, `SystemRoot`, `windir` or `SystemDrive`.
        let parent = vars(&[
            ("SystemRoot", r"\\attacker\share"),
            ("windir", r"\\attacker\share"),
            ("SystemDrive", r"\\attacker"),
        ]);
        let env = build_child_env(
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            Some(Path::new(r"C:\cerebro-temp")),
            None,
            None,
        );
        assert_eq!(
            get(&env, "PATH"),
            Some(&OsString::from(format!(r"C:\backend;{SYSTEM32}")))
        );
        assert_eq!(
            get(&env, "SystemRoot"),
            Some(&OsString::from(r"C:\Windows"))
        );
        assert_eq!(get(&env, "windir"), Some(&OsString::from(r"C:\Windows")));
        assert_eq!(get(&env, "SystemDrive"), Some(&OsString::from("C:")));
        for value in env.values() {
            assert!(!value.to_string_lossy().contains("attacker"), "{env:?}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn path_is_byte_exact_for_a_non_utf8_backend_dir() {
        // An unpaired UTF-16 surrogate has no lossless UTF-8 form; only a byte-exact `OsString`
        // concatenation preserves it.
        use std::os::windows::ffi::{OsStrExt, OsStringExt};
        let weird = OsString::from_wide(&[0x0043, 0x003A, 0x005C, 0xD800]); // "C:\" + lone surrogate
        let backend_dir = std::path::PathBuf::from(&weird);

        let env = build_child_env(vars(&[]), Path::new(SYSTEM32), &backend_dir, None, None, None);
        let path = get(&env, "PATH").expect("PATH must be set");
        let mut expected = weird.clone();
        expected.push(";");
        expected.push(SYSTEM32);
        assert_eq!(path, &expected);
        let wide: Vec<u16> = path.encode_wide().collect();
        assert!(
            wide.contains(&0xD800),
            "the lone surrogate did not survive byte-exact: {wide:?}"
        );
        assert!(
            !wide.contains(&0xFFFD),
            "no replacement character may appear: {wide:?}"
        );
    }

    #[test]
    fn temp_and_tmp_are_pinned_to_the_caller_directory_never_the_parents() {
        let parent = vars(&[
            ("TEMP", r"C:\Users\someone\AppData\Local\Temp"),
            ("TMP", r"D:\other"),
        ]);
        let env = build_child_env(
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            Some(Path::new(r"C:\cerebro-temp")),
            None,
            None,
        );
        assert_eq!(get(&env, "TEMP"), Some(&OsString::from(r"C:\cerebro-temp")));
        assert_eq!(get(&env, "TMP"), Some(&OsString::from(r"C:\cerebro-temp")));
    }

    #[test]
    fn a_semicolon_in_the_backend_dir_would_split_path_into_two_entries() {
        // Documents why the supervisor refuses a `;` in `backend_dir` upstream of here: this is a
        // plain concatenation, and `C:\a;C:\evil` becomes three PATH entries, one never audited.
        let env = build_child_env(
            vars(&[]),
            Path::new(SYSTEM32),
            Path::new(r"C:\a;C:\evil"),
            None,
            None,
            None,
        );
        let path = get(&env, "PATH")
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
            None,
            None,
            None,
        );
        assert!(!without.contains_key(OsStr::new("CUDA_VISIBLE_DEVICES")));
        let with = build_child_env(
            vars(&[]),
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            None,
            Some("0"),
            None,
        );
        assert_eq!(
            get(&with, "CUDA_VISIBLE_DEVICES"),
            Some(&OsString::from("0"))
        );
    }

    #[test]
    fn effective_env_honours_the_allowlist_switch_for_ggml_backend_path_in_both_states() {
        let parent = || {
            vars(&[
                ("GGML_BACKEND_PATH", r"C:\elsewhere\evil.dll"),
                ("LLAMA_ARG_PORT", "9999"),
                ("PATH", r"C:\user\bin"),
            ])
        };
        let on = effective_child_env(
            true,
            parent(),
            Path::new(SYSTEM32),
            Path::new(r"C:\b"),
            None,
            None,
            None,
        );
        assert_eq!(get(&on, "GGML_BACKEND_PATH"), None, "allowlist on: {on:?}");
        assert_eq!(get(&on, "LLAMA_ARG_PORT"), None, "allowlist on: {on:?}");
        assert_eq!(
            get(&on, "PATH"),
            Some(&OsString::from(format!(r"C:\b;{SYSTEM32}")))
        );

        let off = effective_child_env(
            false,
            parent(),
            Path::new(SYSTEM32),
            Path::new(r"C:\b"),
            None,
            None,
            None,
        );
        assert_eq!(
            get(&off, "GGML_BACKEND_PATH"),
            Some(&OsString::from(r"C:\elsewhere\evil.dll")),
            "allowlist off: the inherited environment must reach the child: {off:?}"
        );
        assert_eq!(get(&off, "LLAMA_ARG_PORT"), Some(&OsString::from("9999")));
        assert_eq!(
            get(&off, "PATH"),
            Some(&OsString::from(format!(r"C:\b;{SYSTEM32};C:\user\bin"))),
            "the pinned entries must stay first"
        );
    }

    #[test]
    fn effective_env_off_without_a_temp_dir_passes_the_inherited_temp_through() {
        let parent = vars(&[("TEMP", r"C:\t-user"), ("CUDA_VISIBLE_DEVICES", "1")]);
        let off = effective_child_env(
            false,
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\b"),
            None,
            None,
            None,
        );
        assert_eq!(get(&off, "TEMP"), Some(&OsString::from(r"C:\t-user")));
        // `None` is a launch decision ("omit"), which an ambient value must not override.
        assert_eq!(get(&off, "CUDA_VISIBLE_DEVICES"), None, "{off:?}");
    }

    #[test]
    fn effective_env_off_replaces_a_differently_cased_inherited_path() {
        // `Path` sorts after `PATH` in byte order; if both survived, the inherited one would win.
        let parent = vars(&[("Path", r"C:\user\bin")]);
        let off = effective_child_env(
            false,
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\b"),
            None,
            None,
            None,
        );
        let paths: Vec<_> = off
            .keys()
            .filter(|k| k.eq_ignore_ascii_case("PATH"))
            .collect();
        assert_eq!(
            paths.len(),
            1,
            "exactly one PATH-named key must survive: {off:?}"
        );
    }

    #[test]
    fn a_cache_dir_pins_every_driver_cache_location_and_opens_nothing_else() {
        // With the parent's own folders dropped, the pinned cache is the only place a driver can
        // keep compiled kernels; the values must be the caller's directory, never the parent's.
        let parent = vars(&[
            ("APPDATA", r"C:\Users\someone\AppData\Roaming"),
            ("LOCALAPPDATA", r"C:\Users\someone\AppData\Local"),
            ("CUDA_CACHE_PATH", r"\\attacker\share\cache"),
            ("GGML_BACKEND_PATH", r"C:\elsewhere\evil.dll"),
            ("LLAMA_ARG_PORT", "9999"),
        ]);
        let env = build_child_env(
            parent,
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            None,
            None,
            Some(Path::new(r"C:\cerebro\cache\b1")),
        );
        assert_eq!(
            get(&env, "CUDA_CACHE_PATH"),
            Some(&OsString::from(r"C:\cerebro\cache\b1\cuda"))
        );
        assert_eq!(
            get(&env, "LOCALAPPDATA"),
            Some(&OsString::from(r"C:\cerebro\cache\b1"))
        );
        assert_eq!(
            get(&env, "APPDATA"),
            Some(&OsString::from(r"C:\cerebro\cache\b1"))
        );
        assert_eq!(get(&env, "GGML_BACKEND_PATH"), None, "{env:?}");
        assert!(
            !env.keys()
                .any(|k| k.to_string_lossy().starts_with("LLAMA_ARG_")),
            "{env:?}"
        );
        let names: Vec<String> = env
            .keys()
            .map(|k| k.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec![
                "APPDATA",
                "CUDA_CACHE_PATH",
                "LOCALAPPDATA",
                "PATH",
                "SystemDrive",
                "SystemRoot",
                "windir",
            ]
        );

        // No cache dir: none of the three exists, inherited or otherwise.
        let without = build_child_env(
            vars(&[("APPDATA", r"C:\u"), ("CUDA_CACHE_PATH", r"C:\c")]),
            Path::new(SYSTEM32),
            Path::new(r"C:\backend"),
            None,
            None,
            None,
        );
        for name in ["APPDATA", "LOCALAPPDATA", "CUDA_CACHE_PATH"] {
            assert_eq!(get(&without, name), None, "{name}: {without:?}");
        }
    }

    #[test]
    fn with_the_allowlist_off_the_pinned_cache_replaces_inherited_values_whatever_their_case() {
        // `AppData` sorts after `APPDATA` in byte order; if both survived, the inherited one could
        // win, and the launch would lose the cache the allowlist-on launches warmed.
        let parent = || {
            vars(&[
                ("AppData", r"C:\Users\someone\AppData\Roaming"),
                ("localappdata", r"C:\Users\someone\AppData\Local"),
                ("Cuda_Cache_Path", r"D:\elsewhere"),
                ("PATH", r"C:\user\bin"),
            ])
        };
        let off = effective_child_env(
            false,
            parent(),
            Path::new(SYSTEM32),
            Path::new(r"C:\b"),
            None,
            None,
            Some(Path::new(r"C:\cache")),
        );
        for (name, expected) in [
            ("APPDATA", r"C:\cache"),
            ("LOCALAPPDATA", r"C:\cache"),
            ("CUDA_CACHE_PATH", r"C:\cache\cuda"),
        ] {
            let matching: Vec<_> = off
                .iter()
                .filter(|(k, _)| k.eq_ignore_ascii_case(name))
                .collect();
            assert_eq!(
                matching,
                vec![(&OsString::from(name), &OsString::from(expected))],
                "exactly the pinned {name} must survive: {off:?}"
            );
        }

        // Off without a cache dir: nothing to pin, so the user's own folders pass through.
        let off_no_cache = effective_child_env(
            false,
            parent(),
            Path::new(SYSTEM32),
            Path::new(r"C:\b"),
            None,
            None,
            None,
        );
        assert_eq!(
            get(&off_no_cache, "AppData"),
            Some(&OsString::from(r"C:\Users\someone\AppData\Roaming"))
        );
    }

    #[cfg(windows)]
    #[test]
    fn system_directory_is_an_absolute_existing_directory() {
        let dir = system_directory().expect("GetSystemDirectoryW must succeed");
        assert!(dir.is_absolute() && dir.is_dir(), "{}", dir.display());
    }
}
