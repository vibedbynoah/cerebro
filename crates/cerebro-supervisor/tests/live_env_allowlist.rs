//! One test, deliberately alone in its own process.
//!
//! `std::env::set_var` is a process-global mutation, and `spawn` reads the environment through
//! `std::env::vars_os` (see `spawn::env`'s module docs on why that is injected rather than read
//! inline). Running this test in the same binary as any other test that spawns or reads the
//! environment — as an earlier version of it did, inside `cerebro-supervisor`'s own unit tests —
//! races that mutation against every other test thread in the process. `cargo test` gives each
//! `tests/*.rs` file its own process, which is the cheap, structural fix: this file contains
//! exactly one test, and nothing else in the crate shares its address space.

// Everything below is Windows-only: `spawn_unaudited` returns `SpawnError::UnsupportedPlatform` on
// every other platform (see `spawn`'s own module docs), and this file links `windows-sys` directly
// for `GetSystemDirectoryW`, which is a `[target.'cfg(windows)'.dependencies]` dependency -- using
// it unconditionally here would fail to *compile* `cargo test --workspace` on a non-Windows host
// rather than compiling and simply skipping, unlike every other cfg-gated fallback in this crate
// (the `SpawnError::UnsupportedPlatform` arms, the `#[cfg(not(windows))]` bodies of
// `open_append_no_follow` and `ChildHandle::kill`). Gating the whole file, rather than only the one
// test in it, keeps that property.
#![cfg(windows)]

use cerebro_supervisor::{spawn_unaudited, AcceptUnauditedBackend, SpawnOptions};
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

fn options(scratch: &ScratchDir) -> SpawnOptions {
    SpawnOptions {
        temp_dir: scratch.0.clone(),
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

#[test]
#[ignore = "spawns a real child process (this crate's cerebro-print-env fixture) and mutates this \
            test process's own environment — must stay the only test in this binary"]
fn llama_arg_and_non_allowlisted_variables_never_reach_the_child() {
    // Prove the hazard this module exists to close: something left over in *this test process's
    // own* environment must not reach the child, `LLAMA_ARG_*` included. Mutating the current
    // process's real environment is safe here only because this file is the sole test in its own
    // process (see this file's module docs) — never do this in a binary any other test shares.
    std::env::set_var("LLAMA_ARG_PORT", "31337");
    std::env::set_var("CEREBRO_LIVE_TEST_UNRELATED_VAR", "should-not-leak");

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
    let handle = spawn_unaudited(dir, "cerebro-print-env.exe", &[], &opts, &accept)
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

    // No `remove_var` here: a panic between `set_var` above and here would otherwise leave
    // `LLAMA_ARG_PORT` set for whatever ran next in this process — but since this test is the only
    // one in this binary, the whole process exits right after it either way, so there is nothing
    // left to leak into.
}

/// The real Windows system directory, resolved independently of `cerebro_supervisor::spawn::env`'s
/// own (private) `system_directory` — this test needs its own, trusted expectation to compare the
/// child's printed `PATH` against, not a call into the very function under test.
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
