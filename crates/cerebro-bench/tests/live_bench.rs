//! Live integration test: run the real `llama-bench.exe` against the tiny fixture model and feed
//! its output straight through `cerebro_bench::parse` and `cerebro_bench::stats`.
//!
//! `#[ignore]`d because it depends on this machine's local llama.cpp build and GGUF fixture, which
//! are not present in CI and must never be fetched over the network (offline-first product; no
//! network access beyond loopback per the machine rules this crate was built under). Run
//! explicitly with `cargo test -p cerebro-bench -- --ignored`.
//!
//! The child is spawned with an explicit argv (`Command::arg`, one flag per call) and no shell, in
//! keeping with the house rule that llama.cpp binaries are only ever invoked that way. Unlike
//! `Command::output` (which blocks forever if the child never exits — a stuck backend init, or a
//! CUDA probe despite `-dev none`), this test polls the child with a deadline and `kill()`s it on
//! timeout, so nothing from this test can be left running afterwards even if llama-bench hangs.

use std::io::Read as _;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use cerebro_bench::{assemble_triples, parse_records};

/// How long the live run is allowed before this test kills it and fails. The real run takes a few
/// seconds (`-dev none`, `-r 4`, a 0.5B model); this is generous headroom, not an expected time.
const TIMEOUT: Duration = Duration::from_secs(120);

/// Reaps the wrapped child on every exit path out of the scope holding it, including a panic from
/// an `expect`/`assert` elsewhere in the test.
///
/// `wait_with_deadline`'s own timeout branch already killed and waited on the child before this
/// existed, but that was the only path that did: a `try_wait` `Err` (or any other panic between
/// spawn and the end of the test) unwound straight past it and left `llama-bench.exe` running,
/// which is exactly the machine rule this test's own doc comment promises to uphold.
struct KillOnDrop(std::process::Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        // `try_wait` returning `Ok(None)` (still running) or `Err` (unknown state -- treat as
        // "might still be running") both call for a kill attempt; `Ok(Some(_))` means it already
        // exited and there is nothing to do. Every result here is best-effort: a `kill()` racing
        // an already-exited process, or a `wait()` on one already reaped, is not this destructor's
        // problem to report, only to not panic over.
        if matches!(self.0.try_wait(), Ok(None) | Err(_)) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

/// Run `child` to completion or kill it at `TIMEOUT`, returning its exit status and captured
/// stdout. stdout is drained on a background thread while we poll, so a full pipe buffer can never
/// deadlock the child against us. The child is held in a [`KillOnDrop`] guard so it is reaped on
/// every exit from this function, including a panic.
fn wait_with_deadline(child: std::process::Child) -> (std::process::ExitStatus, String) {
    let mut guard = KillOnDrop(child);
    let mut stdout = guard.0.stdout.take().expect("child was spawned with Stdio::piped() stdout");
    let reader = std::thread::spawn(move || {
        let mut buf = String::new();
        stdout.read_to_string(&mut buf).map(|_| buf)
    });

    let deadline = Instant::now() + TIMEOUT;
    let status = loop {
        if let Some(status) = guard.0.try_wait().expect("try_wait on llama-bench.exe") {
            break status;
        }
        if Instant::now() > deadline {
            // KillOnDrop::drop also reaps on the way out, but doing it here too gives a status to
            // report before the panic unwinds.
            let _ = guard.0.kill();
            let _ = guard.0.wait();
            panic!("llama-bench.exe exceeded {TIMEOUT:?} and was killed; machine rule requires no process left running");
        }
        std::thread::sleep(Duration::from_millis(50));
    };

    let stdout = reader.join().expect("stdout reader thread panicked").expect("llama-bench stdout was not UTF-8");
    (status, stdout)
}

/// b10751 CPU+CUDA+RPC build directory this workspace's live tests are pinned to.
const LLAMA_BENCH_EXE: &str =
    r"C:\Users\noahm\Projects\qwen-flash-next\llama.cpp-b10751\llama-bench.exe";

/// The tiny model every live test in this workspace is restricted to.
const TINY_MODEL: &str =
    r"C:\Users\noahm\.lmstudio\models\skiesoffire\bastion-control\bastion-control-0.5b-f16.gguf";

#[test]
#[ignore = "spawns the real llama-bench.exe against a local model; not available in CI"]
fn real_llama_bench_output_parses_and_assembles_a_complete_triple() {
    // -dev none pins this to the CPU backend: fast (a few seconds), no VRAM residency to leak
    // across test runs, and no risk of colliding with anything else on the GPU.
    let child = Command::new(LLAMA_BENCH_EXE)
        .arg("-m")
        .arg(TINY_MODEL)
        .arg("-o")
        .arg("json")
        .arg("-r")
        .arg("4")
        .arg("-p")
        .arg("64")
        .arg("-n")
        .arg("32")
        .arg("-d")
        .arg("0,128")
        .arg("-dev")
        .arg("none")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn llama-bench.exe -- is the b10751 fixture build present?");

    let (status, stdout) = wait_with_deadline(child);
    assert!(status.success(), "llama-bench exited with {:?}", status.code());

    let records = parse_records(&stdout).expect("real llama-bench -o json output must parse");

    // -p 64 -n 32 -d 0,128 sweeps to exactly 4 (test, depth) combinations: pp@0, tg@0, pp@128,
    // tg@128, all for one configuration (CPU backend, no other axis varied).
    assert_eq!(records.len(), 4);

    let triples = assemble_triples(&records, 1).expect("every record classifies cleanly");
    assert_eq!(triples.len(), 1, "one configuration was measured");

    let (_key, result) = triples.into_iter().next().expect("checked len == 1 above");
    let triple = result.expect("a full -p/-n/-d 0,128 sweep must yield a complete triple");

    assert_eq!(triple.deepest_depth, std::num::NonZeroU64::new(128).unwrap());
    for summary in [triple.pp_tok_s, triple.tg_tok_s_depth0, triple.tg_tok_s_deepest] {
        // -r 4 discards 1 -> exactly 3 kept, the minimum this crate will summarize at all.
        assert_eq!(summary.kept, 3);
        assert!(summary.min > 0.0, "tok/s must be positive");
        assert!(summary.min <= summary.median && summary.median <= summary.max);
        assert!(summary.ci95_low <= summary.ci95_high);
    }
}
