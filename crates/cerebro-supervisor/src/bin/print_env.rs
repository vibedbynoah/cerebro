//! Fixture binary for `spawn`'s environment-allowlist and containment tests.
//!
//! A `[[bin]]` target rather than an `examples/` one: `tests/live_env_allowlist.rs` resolves it at
//! compile time via `env!("CARGO_BIN_EXE_cerebro-print-env")`, which Cargo populates only for
//! `[[bin]]` targets used from an integration test — never for `examples/`, and never for a unit
//! test compiled into the library itself (see that test file's module docs for why the mutation it
//! performs must not share a process with any other test).
//!
//! Gated behind the default-off `test-fixtures` feature (see `Cargo.toml`'s `[[bin]]` entry and its
//! self dev-dependency), so this target is built for this crate's own `cargo test` and never reaches
//! a plain `cargo build`/`cargo install` of the published library, or a consumer's build of it.
//! Every exit path returns [`ExitCode`] rather than panicking, consistent with the rest of this
//! product's house style: a fixture is still code that ships (even if only into this crate's own
//! `target/`), not a place to relax "no `panic!`/`unwrap`/`expect` outside `#[cfg(test)]`".
//!
//! Two modes, selected by argv (never by an environment variable — the whole point of this fixture
//! is to report what environment it was actually given, so it must not also consume one to decide
//! what to do):
//!
//! - No arguments (the default): print every environment variable this process can see, one
//!   `KEY=VALUE` per line, to stdout, then exit immediately. Used by the closed-environment-allowlist
//!   test and, incidentally, by the "child exits before job assignment" race test — this mode is
//!   deliberately fast (single-digit milliseconds) so it reliably wins that race.
//! - `spawn-grandchild`: spawn a long-running `ping.exe -t 127.0.0.1` from `%SystemRoot%\System32`
//!   (argv only, not a shell — consistent with the rest of this product), print its PID as
//!   `GRANDCHILD_PID=<pid>` to stdout, flush, then park forever. Used by the Job Object regression
//!   test: the immediate child (this process) and the grandchild (`ping.exe`) must both die when the
//!   `ChildHandle` above them is dropped, which only the Job Object — not a plain `TerminateProcess`
//!   of the immediate child — can guarantee.

use std::io::{Read, Write};
use std::process::ExitCode;

fn main() -> ExitCode {
    let mode = std::env::args().nth(1);
    match mode.as_deref() {
        Some("spawn-grandchild") => spawn_grandchild_and_park(),
        _ => {
            print_environment();
            ExitCode::SUCCESS
        }
    }
}

/// Default mode: report this process's own environment (and a few other containment-relevant
/// process facts) and exit.
///
/// A `println!` that hits a closed pipe (the test harness having already stopped reading, say)
/// panics by default; this crate's house style forbids `panic!`/`unwrap`/`expect` outside tests, so
/// every line is written explicitly and a write failure ends the loop rather than panicking — the
/// process still exits cleanly, just with less output than it meant to produce.
///
/// The three `CWD=`/`HAS_CONSOLE_WINDOW=`/`STDIN_READ=` lines exist because three of the four
/// containment properties `spawn_windows_at` configures on its `Command` — `.current_dir`,
/// `CREATE_NO_WINDOW`, `.stdin(Stdio::null())` — were, before this fixture printed them, asserted by
/// no test anywhere in this crate: deleting any of the three left every unit and live test green.
/// `current_dir` is not cosmetic either — ARCHITECTURE.md section 9 names Cerebro's own current
/// directory as a live DLL-plant vector for ggml's backend loader, so losing that call silently
/// reintroduces exactly the hazard the backend-directory DACL audit exists to close.
fn print_environment() {
    let mut stdout = std::io::stdout().lock();
    print_process_facts(&mut stdout);
    for (key, value) in std::env::vars() {
        if writeln!(stdout, "{key}={value}").is_err() {
            break;
        }
    }
}

/// Print `CWD=`, `HAS_CONSOLE_WINDOW=`, and `STDIN_READ=`, in that order, before the environment
/// dump — see [`print_environment`]'s own docs for why these three exist. Every write is checked and
/// a failure simply skips the rest, consistent with this fixture's no-panic discipline.
fn print_process_facts(stdout: &mut impl Write) {
    if let Ok(cwd) = std::env::current_dir() {
        if writeln!(stdout, "CWD={}", cwd.display()).is_err() {
            return;
        }
    }
    if writeln!(stdout, "HAS_CONSOLE_WINDOW={}", has_console_window()).is_err() {
        return;
    }
    // A null stdin (what `spawn_windows_at` sets via `Stdio::null()`) is the NUL device: a read
    // from it returns `Ok(0)` (immediate EOF) rather than blocking, which is exactly the contrast
    // this line is meant to surface against a child that inherited a real, readable stdin.
    let mut probe = [0u8; 1];
    let read_result = std::io::stdin().lock().read(&mut probe);
    let _ = writeln!(stdout, "STDIN_READ={read_result:?}");
}

/// Whether this process has a console window at all. `GetConsoleWindow` returns null exactly when
/// there is none — what `CREATE_NO_WINDOW` on the parent's `Command` produces; a child spawned
/// without that flag from a console-attached parent inherits one instead.
#[cfg(windows)]
fn has_console_window() -> bool {
    // SAFETY: takes no arguments and dereferences nothing; the only thing done with the return
    // value is a null check, which is exactly how this API reports "no console".
    let handle = unsafe { windows_sys::Win32::System::Console::GetConsoleWindow() };
    !handle.is_null()
}

/// Non-Windows stub: this fixture (and the whole `spawn` module it exists to test) is Windows-only,
/// but the binary must still compile elsewhere, consistent with the rest of this crate's
/// `#[cfg(not(windows))]` fallbacks.
#[cfg(not(windows))]
fn has_console_window() -> bool {
    false
}

/// `spawn-grandchild` mode: spawn `ping -t`, report its PID, then never exit on its own.
fn spawn_grandchild_and_park() -> ExitCode {
    let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".to_string());
    let ping = std::path::Path::new(&system_root)
        .join("System32")
        .join("ping.exe");

    // Argv only, no shell: `ping -t 127.0.0.1` never terminates on its own, which is exactly what
    // the drop-kills-the-tree test needs from a grandchild it did not have to build itself.
    let mut child = match std::process::Command::new(&ping)
        .args(["-t", "127.0.0.1"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            eprintln!("fixture could not spawn {}: {e}", ping.display());
            return ExitCode::FAILURE;
        }
    };

    println!("GRANDCHILD_PID={}", child.id());
    let _ = std::io::stdout().flush();

    // Block on the grandchild rather than parking this thread separately: `ping -t` never exits on
    // its own, so this call never returns in practice, which is exactly what keeps this process
    // (and the grandchild under it) alive until the test drops the `ChildHandle` above both of
    // them — at which point the kernel kills this whole assigned tree without either process ever
    // running another instruction. Reaping it this way (instead of leaving `child` to be dropped
    // unwaited, which clippy's `zombie_processes` lint flags) also means this fixture never itself
    // leaves an un-`wait()`ed handle behind on whatever short-lived path might reach here.
    match child.wait() {
        Ok(status) => {
            if status.success() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("fixture could not wait on its own grandchild: {e}");
            ExitCode::FAILURE
        }
    }
}
