//! Environment-mutating live tests, deliberately alone in their own process.
//!
//! `std::env::set_var` is a process-global mutation, and `spawn` reads the environment through
//! `std::env::vars_os` (see `spawn::env`'s module docs on why that is injected rather than read
//! inline). Running a test like this in the same binary as any other test that spawns or reads the
//! environment — as an earlier version of it did, inside `cerebro-supervisor`'s own unit tests —
//! races that mutation against every other test thread in the process. `cargo test` gives each
//! `tests/*.rs` file its own process, which is the cheap, structural fix: every test that needs to
//! mutate the real process environment lives in this one file, and nothing else in the crate shares
//! its address space.
//!
//! This file itself now holds two such tests (the allowlist enforced, and
//! [`Protections::enforce_environment_allowlist`] turned off), and `cargo test` still runs multiple
//! tests in the same binary on separate threads by default — so both acquire [`ENV_LOCK`] for their
//! whole body before touching `std::env::set_var`, serializing them against each other. Any further
//! test added to this file that mutates process environment must acquire it too.

// Everything below is Windows-only: `spawn_unaudited` returns `SpawnError::UnsupportedPlatform` on
// every other platform (see `spawn`'s own module docs), and this file links `windows-sys` directly
// for `GetSystemDirectoryW`, which is a `[target.'cfg(windows)'.dependencies]` dependency -- using
// it unconditionally here would fail to *compile* `cargo test --workspace` on a non-Windows host
// rather than compiling and simply skipping, unlike every other cfg-gated fallback in this crate
// (the `SpawnError::UnsupportedPlatform` arms, the `#[cfg(not(windows))]` bodies of
// `open_append_no_follow` and `ChildHandle::kill`). Gating the whole file, rather than only the one
// test in it, keeps that property.
#![cfg(windows)]

use cerebro_supervisor::{spawn_unaudited, AcceptUnauditedBackend, Protections, SpawnOptions};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// A scratch directory under the OS temp dir, unique to this test process, cleaned up on drop —
/// mirrors `spawn`'s own live-test helper of the same shape.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "cerebro-supervisor-live-env-allowlist-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir for a live test");
        Self(dir)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The driver cache directory [`options`] pins: distinct from the temp directory, so a value the
/// child prints can only have come from the cache pin.
fn cache_dir(scratch: &ScratchDir) -> PathBuf {
    let dir = scratch.0.join("driver-cache");
    std::fs::create_dir_all(&dir).expect("create the driver cache dir");
    dir
}

fn options(scratch: &ScratchDir) -> SpawnOptions {
    SpawnOptions {
        temp_dir: scratch.0.clone(),
        cache_dir: cache_dir(scratch),
        cuda_visible_devices: None,
        log_path: scratch.0.join("spawn.log"),
        log_max_bytes: 1024 * 1024,
    }
}

/// Poll `f` until it returns `Some`, or `timeout` elapses (in which case this panics — acceptable
/// in `#[ignore]`d test code, never in the library).
fn poll_until<T>(timeout: Duration, mut f: impl FnMut() -> Option<T>) -> T {
    let start = Instant::now();
    loop {
        if let Some(v) = f() {
            return v;
        }
        if start.elapsed() > timeout {
            panic!("condition did not become true within {timeout:?}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Serializes every test in this file that mutates the real process environment — see this file's
/// own module docs.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Sets process environment variables for one test and restores every one of them (to its old
/// value, or removed if it had none) when dropped — on panic too, so a failed assertion cannot leave
/// a variable behind that makes the next test in this binary pass for the wrong reason. Only ever
/// created while [`ENV_LOCK`] is held.
struct EnvGuard(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl EnvGuard {
    fn new() -> Self {
        Self(Vec::new())
    }

    fn set(&mut self, name: &'static str, value: impl AsRef<std::ffi::OsStr>) {
        if !self.0.iter().any(|(n, _)| *n == name) {
            self.0.push((name, std::env::var_os(name)));
        }
        std::env::set_var(name, value);
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, old) in self.0.drain(..).rev() {
            match old {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
        }
    }
}

#[test]
#[ignore = "spawns a real child process (this crate's cerebro-print-env fixture) and mutates this \
            test process's own environment — must acquire ENV_LOCK, see this file's module docs"]
fn llama_arg_and_non_allowlisted_variables_never_reach_the_child() {
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // Prove the hazard this module exists to close: something left over in *this test process's
    // own* environment must not reach the child, `LLAMA_ARG_*` included. Mutating the current
    // process's real environment is safe here only because this file is the sole test in its own
    // process (see this file's module docs) — never do this in a binary any other test shares.
    let mut env = EnvGuard::new();
    env.set("LLAMA_ARG_PORT", "31337");
    env.set("CEREBRO_LIVE_TEST_UNRELATED_VAR", "should-not-leak");

    // Populated at compile time for a `[[bin]]` target used from an integration test — resolved
    // by Cargo, never by shelling out to `cargo build` ourselves (which would re-lock the target
    // directory and break under `--offline`/`--frozen`, a vendored registry, or a custom
    // `CARGO_TARGET_DIR`).
    let exe = Path::new(env!("CARGO_BIN_EXE_cerebro-print-env"));
    let dir = exe
        .parent()
        .expect("the fixture binary's path has a parent directory");

    let scratch = ScratchDir::new("env-allowlist");
    let opts = options(&scratch);
    let accept = AcceptUnauditedBackend::new(
        "live test fixture: cerebro-print-env has no manifest to preflight against",
    )
    .expect("a non-blank literal reason is always accepted");
    let handle = spawn_unaudited(
        dir,
        "cerebro-print-env.exe",
        &[],
        &opts,
        Protections::default(),
        &accept,
    )
    .expect("spawn the cerebro-print-env fixture");

    let status = poll_until(Duration::from_secs(10), || {
        handle.try_wait().expect("poll child exit status")
    });
    assert!(status.success(), "cerebro-print-env exited with {status:?}");

    // See `spawn`'s own live tests for why this polls rather than reading `recent_lines()` once.
    let printed: Vec<(String, String)> = poll_until(Duration::from_secs(2), || {
        let pairs: Vec<(String, String)> = handle
            .recent_lines()
            .into_iter()
            .filter_map(|line| line.strip_prefix("stdout: ").map(str::to_string))
            .filter_map(|kv| kv.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
            .collect();
        (!pairs.is_empty()).then_some(pairs)
    });
    assert!(!printed.is_empty(), "cerebro-print-env produced no output");
    let printed_names: Vec<&str> = printed.iter().map(|(k, _)| k.as_str()).collect();

    assert!(
        !printed_names.iter().any(|name| name.starts_with("LLAMA_ARG_")),
        "a LLAMA_ARG_* variable reached the child: {printed_names:?}"
    );
    assert!(
        !printed_names.contains(&"CEREBRO_LIVE_TEST_UNRELATED_VAR"),
        "an unrelated ambient variable reached the child: {printed_names:?}"
    );

    // `cerebro-print-env` also prints three process facts ahead of the environment dump (see its
    // own module docs) that are not environment variables at all, so the closed-allowlist check
    // below must not hold them to the same list.
    const PROCESS_FACTS: &[&str] = &["CWD", "HAS_CONSOLE_WINDOW", "STDIN_READ"];
    const ALLOWED: &[&str] = &[
        "SystemRoot",
        "SystemDrive",
        "windir",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "PATH",
        "TEMP",
        "TMP",
        "APPDATA",
        "LOCALAPPDATA",
        "CUDA_CACHE_PATH",
    ];
    for name in &printed_names {
        if PROCESS_FACTS.contains(name) {
            continue;
        }
        assert!(
            ALLOWED.contains(name),
            "child environment contained {name:?}, which is not in the closed allowlist: \
             {printed_names:?}"
        );
    }

    // Beyond which *names* survived, prove the *values* the child actually received: this is the
    // difference between "the allowlist's shape is closed" (checked above) and "the constructed
    // values actually reached the child intact" — a bug that emitted the right variable names with
    // the wrong values would still pass every assertion above it.
    let value_of = |name: &str| -> String {
        printed
            .iter()
            .find(|(k, _)| k == name)
            .unwrap_or_else(|| panic!("child did not print {name}: {printed:?}"))
            .1
            .clone()
    };
    let expected_path = format!("{};{}", dir.display(), real_system_directory());
    assert_eq!(value_of("PATH"), expected_path);
    assert_eq!(
        value_of("PATH").split(';').count(),
        2,
        "PATH must be exactly backend_dir followed by system32, nothing else"
    );
    assert_eq!(value_of("TEMP"), scratch.0.display().to_string());
    assert_eq!(value_of("TMP"), scratch.0.display().to_string());
    // Driver caches land in the checked cache directory, never the parent's own folders.
    let cache = cache_dir(&scratch).display().to_string();
    assert_eq!(value_of("APPDATA"), cache);
    assert_eq!(value_of("LOCALAPPDATA"), cache);
    assert_eq!(value_of("CUDA_CACHE_PATH"), format!(r"{cache}\cuda"));

    // Three of `spawn_windows_at`'s four containment properties (`current_dir`, `CREATE_NO_WINDOW`,
    // `stdin(Stdio::null())`) live as plain builder calls with no dedicated helper module, and before
    // `cerebro-print-env` reported them, no test in this crate asserted any of the three — deleting
    // any one left every unit and live test green. `current_dir` is not cosmetic: ARCHITECTURE.md
    // section 9 names Cerebro's own current directory as a live DLL-plant vector for ggml's backend
    // loader, so losing that call would silently reintroduce exactly the hazard the backend
    // directory's DACL audit exists to close.
    assert_eq!(
        value_of("CWD"),
        dir.display().to_string(),
        "the child's working directory must be the audited backend directory"
    );
    assert_eq!(
        value_of("HAS_CONSOLE_WINDOW"),
        "false",
        "CREATE_NO_WINDOW must leave the child with no console window"
    );
    assert_eq!(
        value_of("STDIN_READ"),
        "Ok(0)",
        "a null stdin must read back as immediate EOF, not block or inherit a real console"
    );

    // `env` restores both variables when dropped, before `_guard` releases `ENV_LOCK` (locals drop
    // in reverse order): leaving `LLAMA_ARG_PORT` set would make the allowlist-off test's own "it
    // leaked" assertion pass for the wrong reason.
}

#[test]
#[ignore = "spawns a real child process (this crate's cerebro-print-env fixture) and mutates this \
            test process's own environment — must acquire ENV_LOCK, see this file's module docs"]
fn llama_arg_variables_leak_through_when_the_allowlist_is_off() {
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // The other half of the property the test above proves: with
    // `Protections::enforce_environment_allowlist` turned off, the same ambient `LLAMA_ARG_*`
    // variable this crate otherwise strips must actually reach the child — proving the switch has a
    // real effect, not merely that it compiles.
    let mut env = EnvGuard::new();
    env.set("LLAMA_ARG_PORT", "31337");
    // An inherited driver cache location must be replaced by the pinned one even with the
    // allowlist off: the cache is a launch setting, not part of the protection.
    env.set("APPDATA", r"C:\cerebro-live-test-inherited-appdata");
    env.set("CUDA_CACHE_PATH", r"C:\cerebro-live-test-inherited-cuda-cache");

    let exe = Path::new(env!("CARGO_BIN_EXE_cerebro-print-env"));
    let dir = exe
        .parent()
        .expect("the fixture binary's path has a parent directory");

    let scratch = ScratchDir::new("env-allowlist-off");
    // A directory put on this process's PATH must reach the child's PATH once the allowlist is off
    // (a CUDA toolkit's bin directory, in real use), after the pinned backend_dir and System32.
    let on_path = scratch.0.join("inherited-path-marker");
    std::fs::create_dir_all(&on_path).expect("create the PATH marker directory");
    let mut inherited = std::ffi::OsString::from(on_path.as_os_str());
    if let Some(old) = std::env::var_os("PATH").filter(|p| !p.is_empty()) {
        inherited.push(";");
        inherited.push(old);
    }
    env.set("PATH", &inherited);
    let opts = options(&scratch);
    let protections = Protections {
        enforce_environment_allowlist: false,
        ..Protections::default()
    };
    let accept = AcceptUnauditedBackend::new(
        "live test fixture: cerebro-print-env has no manifest to preflight against",
    )
    .expect("a non-blank literal reason is always accepted");
    let handle = spawn_unaudited(
        dir,
        "cerebro-print-env.exe",
        &[],
        &opts,
        protections,
        &accept,
    )
    .expect("spawn the cerebro-print-env fixture");

    let status = poll_until(Duration::from_secs(10), || {
        handle.try_wait().expect("poll child exit status")
    });
    assert!(status.success(), "cerebro-print-env exited with {status:?}");

    let printed: Vec<(String, String)> = poll_until(Duration::from_secs(2), || {
        let pairs: Vec<(String, String)> = handle
            .recent_lines()
            .into_iter()
            .filter_map(|line| line.strip_prefix("stdout: ").map(str::to_string))
            .filter_map(|kv| {
                kv.split_once('=')
                    .map(|(k, v)| (k.to_string(), v.to_string()))
            })
            .collect();
        (!pairs.is_empty()).then_some(pairs)
    });
    assert!(!printed.is_empty(), "cerebro-print-env produced no output");

    assert_eq!(
        printed
            .iter()
            .find(|(k, _)| k == "LLAMA_ARG_PORT")
            .map(|(_, v)| v.as_str()),
        Some("31337"),
        "with the environment allowlist turned off, an ambient LLAMA_ARG_PORT must reach the \
         child unfiltered: {printed:?}"
    );
    // TEMP/TMP and (had one been set) CUDA_VISIBLE_DEVICES are launch settings, not part of the
    // allowlist protection, and must stay pinned exactly as they are with the allowlist on — see
    // `effective_child_env`'s own documentation.
    assert_eq!(
        printed
            .iter()
            .find(|(k, _)| k == "TEMP")
            .map(|(_, v)| v.clone()),
        Some(scratch.0.display().to_string()),
        "TEMP must stay pinned to the caller's temp_dir even with the environment allowlist off: \
         {printed:?}"
    );
    assert_eq!(
        printed
            .iter()
            .filter(|(k, _)| k.eq_ignore_ascii_case("PATH"))
            .count(),
        1,
        "exactly one PATH-named key must reach the child even with the allowlist off: {printed:?}"
    );
    let cache = cache_dir(&scratch).display().to_string();
    for (name, expected) in [
        ("APPDATA", cache.clone()),
        ("CUDA_CACHE_PATH", format!(r"{cache}\cuda")),
    ] {
        let seen: Vec<&String> = printed
            .iter()
            .filter(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v)
            .collect();
        assert_eq!(
            seen,
            vec![&expected],
            "allowlist off: {name} must be the pinned cache, not the inherited value: {printed:?}"
        );
    }
    let child_path = printed
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("PATH"))
        .map(|(_, v)| v.clone())
        .unwrap_or_default();
    let expected_path = format!(
        "{};{};{}",
        dir.display(),
        real_system_directory(),
        inherited.to_string_lossy()
    );
    assert_eq!(
        child_path, expected_path,
        "allowlist off: the child's PATH must be backend_dir;System32 followed by this process's own \
         PATH (which starts with the marker directory)"
    );
    let entries: Vec<&str> = child_path.split(';').collect();
    let marker = on_path.display().to_string();
    let marker_at = entries.iter().position(|e| *e == marker);
    assert!(
        marker_at.is_some_and(|i| i >= 2) && entries[0] == dir.display().to_string(),
        "the inherited PATH directory must reach the child, after backend_dir: {entries:?}"
    );
    assert_eq!(
        handle.protection_warnings().len(),
        1,
        "exactly one protection was turned off for this launch: {:?}",
        handle.protection_warnings()
    );
    assert_eq!(
        handle.protection_warnings()[0].protection(),
        cerebro_supervisor::Protection::EnvironmentAllowlist
    );
    // `env` restores LLAMA_ARG_PORT and PATH when dropped, before `_guard` releases ENV_LOCK.
}

#[test]
#[ignore = "spawns a real child process (this crate's cerebro-print-env fixture, as a probe query \
            tool) and mutates this test process's own environment — must acquire ENV_LOCK, see \
            this file's module docs"]
fn probe_queries_honour_the_environment_allowlist_in_both_states() {
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // `probe_with`'s `--version`/`--list-devices` queries load every ggml DLL, and ggml.dll honours
    // GGML_BACKEND_PATH. With the allowlist on, neither it nor LLAMA_ARG_* may reach the query
    // child; with it off, both must, and the setting must be recorded on the BackendProbe either way.
    let mut env = EnvGuard::new();
    env.set(
        "GGML_BACKEND_PATH",
        r"C:\cerebro-live-test-not-a-real-dir\evil.dll",
    );
    env.set("LLAMA_ARG_PORT", "31337");

    let scratch = ScratchDir::new("probe-env");
    let backend = scratch.0.join("backend");
    std::fs::create_dir_all(&backend).expect("create the fixture backend dir");
    std::fs::copy(
        Path::new(env!("CARGO_BIN_EXE_cerebro-print-env")),
        backend.join("cerebro-print-env.exe"),
    )
    .expect("copy the print-env fixture into its own backend dir");

    for allowlist_on in [true, false] {
        // The folder audit is off only because %TEMP%'s ancestors are writable by other principals
        // on this machine; it is orthogonal to what this test checks. Elevation refusal is off for
        // the same reason: these live tests also run from an elevated shell, where the probe would
        // otherwise refuse before any query (that refusal has its own tests in cerebro-probe).
        let protections = Protections {
            audit_folder_permissions: false,
            enforce_environment_allowlist: allowlist_on,
            refuse_elevation: false,
            ..Protections::default()
        };
        let settings = protections.probe_settings();
        let probed =
            cerebro_probe::backend::probe_with(&backend, "cerebro-print-env.exe", settings)
                .expect("probe the print-env fixture");
        assert_eq!(probed.environment_allowlist, settings.environment_allowlist);
        assert_eq!(
            probed.environment_allowlist == cerebro_probe::backend::EnvironmentAllowlist::Enforce,
            allowlist_on,
            "the BackendProbe must record the switch as set"
        );
        assert_eq!(
            probed.folder_audit,
            cerebro_probe::backend::FolderAudit::SkippedByUserSetting
        );

        // The probe's queries get the same checked driver cache as a launch would, in both states.
        let cache = cache_dir(&scratch);
        let verified = cerebro_supervisor::verify_cache_dir(&cache, &protections)
            .expect("the scratch cache dir passes the launch's own check");
        let with_cache = cerebro_probe::backend::probe_with_cache(
            &backend,
            "cerebro-print-env.exe",
            settings,
            &verified,
        )
        .expect("probe with a pinned driver cache");
        assert_eq!(with_cache.identity.dir(), probed.identity.dir());

        let dir = probed.identity.dir();
        let out = cerebro_probe::backend::query_tool_for_tests(
            &dir.join("cerebro-print-env.exe"),
            dir,
            &["--list-devices"],
            settings.environment_allowlist,
            Some(&verified),
        )
        .expect("run the query exactly as probe_with_cache does");
        let printed: Vec<(&str, &str)> = out.lines().filter_map(|l| l.split_once('=')).collect();
        let value_of = |name: &str| {
            printed
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v.to_string())
        };

        let leaked = value_of("GGML_BACKEND_PATH").is_some();
        assert_eq!(
            leaked, !allowlist_on,
            "GGML_BACKEND_PATH must reach the query child exactly when the allowlist is off \
             (allowlist on = {allowlist_on}): {printed:?}"
        );
        assert_eq!(
            value_of("LLAMA_ARG_PORT").is_some(),
            !allowlist_on,
            "LLAMA_ARG_PORT (allowlist on = {allowlist_on}): {printed:?}"
        );
        // Hardening that holds in both states: pinned working directory, no console window, and
        // the backend directory then System32 first on PATH.
        assert_eq!(value_of("CWD"), Some(dir.display().to_string()));
        let cache_text = verified.display().to_string();
        assert_eq!(value_of("APPDATA"), Some(cache_text.clone()));
        assert_eq!(value_of("LOCALAPPDATA"), Some(cache_text.clone()));
        assert_eq!(
            value_of("CUDA_CACHE_PATH"),
            Some(format!(r"{cache_text}\cuda"))
        );
        assert_eq!(value_of("HAS_CONSOLE_WINDOW").as_deref(), Some("false"));
        let path = value_of("PATH").expect("PATH is always pinned");
        let pinned = format!("{};{}", dir.display(), real_system_directory());
        if allowlist_on {
            assert_eq!(
                path, pinned,
                "allowlist on: PATH must be exactly backend_dir;System32"
            );
        } else {
            assert!(
                path.starts_with(&format!("{pinned};")),
                "allowlist off: {path}"
            );
        }
    }
    // `env` restores both variables when dropped, before `_guard` releases ENV_LOCK.
}

/// The real Windows system directory, resolved independently of
/// `cerebro_probe::child_env::system_directory` — this test needs its own, trusted expectation to
/// compare the child's printed `PATH` against, not a call into the very function under test.
fn real_system_directory() -> String {
    use std::os::windows::ffi::OsStringExt;

    let mut buf = [0u16; 1024];
    // SAFETY: `buf` is a live array of exactly `buf.len()` `u16`s, and that length is the buffer
    // size passed, so the call cannot write past it. The return value is checked below before any
    // of `buf` is read.
    let len = unsafe {
        windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW(
            buf.as_mut_ptr(),
            buf.len() as u32,
        )
    } as usize;
    assert!(len != 0 && len < buf.len(), "GetSystemDirectoryW failed or needed a larger buffer");
    std::ffi::OsString::from_wide(&buf[..len])
        .into_string()
        .expect("the system directory path is valid Unicode on this machine")
}
