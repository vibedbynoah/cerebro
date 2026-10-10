//! Job-object containment switch, in both states, in the default `cargo test` run.
//!
//! Nothing here depends on this machine: the child is this crate's own `cerebro-print-env` fixture
//! (resolved by Cargo through `CARGO_BIN_EXE_*`) in `spawn-grandchild` mode, and its grandchild is
//! `ping.exe -t 127.0.0.1` from System32, present on every Windows install. No model is loaded, no
//! window opens, and every process started is killed before the test returns, on the failure path
//! too (see [`ProcessGuard`], which holds a handle rather than a raw PID).
//!
//! Each state is checked by behaviour, not only by the flag: with containment on, killing the
//! handle must take the grandchild down with it (the job's kill-on-close), and with it off, the
//! grandchild must survive that same kill (only the immediate PID is ever killed), which is
//! exactly the risk `Protections::contain_in_job_object`'s docs name.
//!
//! Its own file, so its processes share nothing with `live_env_allowlist.rs`, which mutates this
//! process's environment.

// `spawn_unaudited` returns `SpawnError::UnsupportedPlatform` everywhere else, and this file calls
// `windows-sys` directly.
#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cerebro_supervisor::{
    spawn_unaudited, AcceptUnauditedBackend, ChildHandle, Protection, Protections, SpawnOptions,
};

/// A scratch directory unique to this test process and state, removed on drop with a bounded retry:
/// with containment off, the grandchild inherited the output pipe, so the log reader can still be
/// finishing its last write for a moment after both processes are gone.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "cerebro-supervisor-job-containment-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self(dir)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let started = Instant::now();
        while std::fs::remove_dir_all(&self.0).is_err()
            && self.0.exists()
            && started.elapsed() < Duration::from_secs(5)
        {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

/// Holds a handle to a process from the moment its PID is read, so neither the liveness checks nor
/// the cleanup kill can reach a different process that reused the PID after this one exited.
/// Terminates the process on drop, so a failed assertion cannot leave `ping -t` running.
struct ProcessGuard(windows_sys::Win32::Foundation::HANDLE);

impl ProcessGuard {
    /// `None` if the process could not be opened, which means it was already gone.
    fn open(pid: u32) -> Option<Self> {
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE,
        };
        // SAFETY: `OpenProcess` takes no pointers; the result is checked for null before use and
        // closed exactly once, in `Drop`.
        let handle = unsafe {
            OpenProcess(
                PROCESS_TERMINATE | PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                0,
                pid,
            )
        };
        (!handle.is_null()).then_some(Self(handle))
    }

    /// Whether the process exits within `timeout` (a zero timeout only polls).
    fn exited_within(&self, timeout: Duration) -> bool {
        use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
        use windows_sys::Win32::System::Threading::WaitForSingleObject;
        let millis = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX - 1);
        // SAFETY: `self.0` is a live process handle opened with `PROCESS_SYNCHRONIZE`.
        unsafe { WaitForSingleObject(self.0, millis) == WAIT_OBJECT_0 }
    }

    /// Whether the process is still running right now.
    fn is_alive(&self) -> bool {
        !self.exited_within(Duration::ZERO)
    }

    /// The PID the held handle refers to, read back from the handle itself.
    fn pid(&self) -> u32 {
        // SAFETY: `self.0` is a live process handle opened with `PROCESS_QUERY_LIMITED_INFORMATION`.
        unsafe { windows_sys::Win32::System::Threading::GetProcessId(self.0) }
    }

    /// End the process. Harmless once it has exited.
    fn terminate(&self) {
        // SAFETY: `self.0` is a live process handle opened with `PROCESS_TERMINATE`.
        unsafe {
            windows_sys::Win32::System::Threading::TerminateProcess(self.0, 1);
        }
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        self.terminate();
        // SAFETY: opened in `open`, closed only here.
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

fn poll_until<T>(timeout: Duration, mut f: impl FnMut() -> Option<T>) -> Option<T> {
    let start = Instant::now();
    loop {
        if let Some(v) = f() {
            return Some(v);
        }
        if start.elapsed() > timeout {
            return None;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn spawn_fixture(scratch: &ScratchDir, protections: Protections) -> ChildHandle {
    let exe = Path::new(env!("CARGO_BIN_EXE_cerebro-print-env"));
    let dir = exe
        .parent()
        .expect("the fixture binary's path has a parent directory");
    let opts = SpawnOptions {
        temp_dir: scratch.0.clone(),
        cache_dir: scratch.0.clone(),
        cuda_visible_devices: None,
        log_path: scratch.0.join("spawn.log"),
        log_max_bytes: 1024 * 1024,
    };
    let accept = AcceptUnauditedBackend::new(
        "test fixture: cerebro-print-env has no manifest to preflight against",
    )
    .expect("a non-blank literal reason is always accepted");
    spawn_unaudited(
        dir,
        "cerebro-print-env.exe",
        &["spawn-grandchild".to_string()],
        &opts,
        protections,
        &accept,
    )
    .expect("spawn the cerebro-print-env fixture")
}

/// Read the grandchild's PID from the fixture's output and open a handle to it at once, while the
/// grandchild is certainly alive (the fixture is parked waiting on it).
fn grandchild(handle: &ChildHandle) -> ProcessGuard {
    let pid: u32 = poll_until(Duration::from_secs(10), || {
        handle.recent_lines().iter().find_map(|line| {
            line.strip_prefix("stdout: GRANDCHILD_PID=")
                .and_then(|pid| pid.trim().parse().ok())
        })
    })
    .expect("the fixture must report its grandchild's PID");
    let guard = ProcessGuard::open(pid).expect("open the grandchild while it is alive");
    // The held handle, not the PID, is what every later check uses: prove it names this process.
    assert_eq!(
        guard.pid(),
        pid,
        "the guard's handle must refer to the grandchild just read"
    );
    assert!(
        guard.is_alive(),
        "the grandchild must be running when its handle is opened"
    );
    guard
}

#[test]
fn job_containment_on_contains_the_tree_and_records_no_warning() {
    let scratch = ScratchDir::new("on");
    let handle = spawn_fixture(&scratch, Protections::default());
    let gc = grandchild(&handle);

    // The fixture is parked on `ping -t` by now, so assignment cannot have lost the
    // assign-after-exit race: a contained launch must report `true` here.
    assert!(
        handle.is_contained(),
        "containment is on, so a running child must be contained"
    );
    assert!(
        !handle
            .protection_warnings()
            .iter()
            .any(|w| w.protection() == Protection::JobObjectContainment),
        "no containment warning while it is on: {:?}",
        handle.protection_warnings()
    );

    let status = handle
        .kill_and_wait(Duration::from_secs(5))
        .expect("kill the fixture");
    assert!(
        status.is_some(),
        "the fixture must be gone within the timeout"
    );
    // Only the job's kill-on-close reaches the grandchild: `kill` itself ends the immediate PID.
    assert!(
        gc.exited_within(Duration::from_secs(5)),
        "the grandchild must die with the job when containment is on"
    );
}

#[test]
fn job_containment_off_is_honoured_and_recorded_as_one_warning() {
    let scratch = ScratchDir::new("off");
    let protections = Protections {
        contain_in_job_object: false,
        ..Protections::default()
    };
    let handle = spawn_fixture(&scratch, protections);
    let gc = grandchild(&handle);

    assert!(
        !handle.is_contained(),
        "containment is off, so the launch must not be contained"
    );
    let recorded: Vec<Protection> = handle
        .protection_warnings()
        .iter()
        .map(|w| w.protection())
        .collect();
    assert_eq!(
        recorded,
        vec![Protection::JobObjectContainment],
        "exactly the one protection turned off must be recorded"
    );

    let status = handle
        .kill_and_wait(Duration::from_secs(5))
        .expect("kill the fixture");
    assert!(
        status.is_some(),
        "the immediate child must still be killed with containment off"
    );
    // Nothing outside a job ends the grandchild, so it outlives its parent: the switch really did
    // skip the Job Object rather than only reporting that it did. The guard ends it afterwards.
    assert!(
        gc.is_alive(),
        "with containment off the grandchild must survive the parent's kill"
    );
    gc.terminate();
    assert!(
        gc.exited_within(Duration::from_secs(5)),
        "cleanup must end the grandchild"
    );
}
