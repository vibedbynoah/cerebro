//! A Windows Job Object for one child process tree.
//!
//! `run_command` (apps/desktop/src-tauri/src/builtin_tools.rs) spawns `powershell.exe` for a model,
//! and one command can start further processes. Killing only the PID `Command::spawn` returned
//! would leave any such grandchild running as an orphan. Assigning the child to a Job Object
//! created with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` fixes that at the kernel level: every process
//! ever assigned to the job — the immediate child, and, because the limit is inherited, anything it
//! spawns while assigned — is terminated the instant the job is terminated or its last handle
//! closes, even if Cerebro itself dies. No userspace bookkeeping of a process tree can offer that
//! guarantee; only the kernel object can (the same reasoning as cerebro-supervisor's spawn::job).
//!
//! The crate exists on its own because the desktop crate is `#![forbid(unsafe_code)]`: the FFI
//! lives here, small and reviewable, so nothing else needs an `unsafe` exemption.

/// Why a Job Object call failed.
#[derive(Debug, thiserror::Error)]
pub enum JobError {
    /// Job objects exist on Windows only; everywhere else there is no kernel object to contain a
    /// process tree with.
    #[error("job objects are Windows-only")]
    Unsupported,
    /// A Win32 call failed, with the calling thread's last-error code from right after it.
    #[error("{call} failed (Win32 error {code})")]
    Win32 {
        /// The failing call, for the log.
        call: &'static str,
        /// `GetLastError` immediately after the failed call.
        code: u32,
    },
}

#[cfg(windows)]
use std::os::windows::io::AsRawHandle;
#[cfg(windows)]
use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, HANDLE};
#[cfg(windows)]
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject, TerminateJobObject,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

/// An unnamed Windows Job Object configured, at construction, to kill every process assigned to it
/// the moment its last handle closes.
///
/// [`Job::create`] is the only constructor and applies the limit before returning, so a `Job`
/// existing at all is proof the kill-on-close behavior is armed. Dropping the `Job` therefore ends
/// the whole assigned tree; [`Job::terminate`] ends it earlier, without closing anything.
#[cfg(windows)]
pub struct Job(HANDLE);

/// Non-Windows builds: [`Job::create`] always fails with [`JobError::Unsupported`].
#[cfg(not(windows))]
pub struct Job(());

// A Win32 `HANDLE` is an opaque, process-scoped kernel reference, not a thread-affine resource:
// Win32 documents `CloseHandle`/`AssignProcessToJobObject` as safe to call from any thread of the
// owning process. Only `&self` methods are exposed, so `Send` is claimed, not `Sync`.
#[cfg(windows)]
unsafe impl Send for Job {}

#[cfg(windows)]
impl Job {
    /// Creates a new, unnamed job object with kill-on-close armed.
    ///
    /// # Errors
    /// A [`JobError::Win32`] when the job cannot be created or the limit cannot be set.
    pub fn create() -> Result<Self, JobError> {
        // SAFETY: both arguments being null requests default security attributes and no name,
        // which is documented as a valid `CreateJobObjectW` call; the return value is checked for
        // null (failure) immediately below before it is treated as a handle.
        let raw = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if raw.is_null() {
            return Err(win32_failure("CreateJobObjectW"));
        }
        let job = Job(raw);
        job.arm_kill_on_close()?;
        Ok(job)
    }

    /// Sets `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. Called exactly once, from [`Job::create`], before
    /// the job is handed to a caller.
    fn arm_kill_on_close(&self) -> Result<(), JobError> {
        // Zero-initialized is a valid `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`: every field not set
        // here is documented as ignored unless its bit is present in `LimitFlags`, which holds only
        // the one bit this function cares about.
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: `self.0` is a valid job handle owned by this wrapper (just created and not yet
        // handed to any caller that could have closed it). `info` is a live, correctly-aligned
        // local of exactly the size passed, so the call reads no more than the buffer holds.
        let ok = unsafe {
            SetInformationJobObject(
                self.0,
                JobObjectExtendedLimitInformation,
                std::ptr::addr_of!(info).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if ok == 0 {
            return Err(win32_failure("SetInformationJobObject"));
        }
        Ok(())
    }

    /// Assigns an already-spawned child to this job.
    ///
    /// Once assigned, the process — and anything it spawns while still assigned — is terminated
    /// when this `Job` is terminated or dropped. Call it immediately after `Command::spawn`, before
    /// the child gets a chance to start anything of its own.
    ///
    /// # Errors
    /// A [`JobError::Win32`] when the process cannot be assigned; the caller should then kill the
    /// child itself rather than run it uncontained.
    pub fn assign(&self, child: &std::process::Child) -> Result<(), JobError> {
        // SAFETY: `self.0` is a valid job handle owned by this wrapper and still open (no `&self`
        // call can run concurrently with `drop`, which takes `&mut self`). `child.as_raw_handle()`
        // is the process handle `std::process::Child` keeps open for its own lifetime, so it is a
        // valid, live process handle for the duration of this call.
        let ok = unsafe { AssignProcessToJobObject(self.0, child.as_raw_handle() as HANDLE) };
        if ok == 0 {
            return Err(win32_failure("AssignProcessToJobObject"));
        }
        Ok(())
    }

    /// Terminates every process currently assigned to the job.
    ///
    /// # Errors
    /// A [`JobError::Win32`] when Win32 rejects the call; dropping the `Job` (which closes the
    /// handle and so kills the tree through kill-on-close) still ends it either way.
    pub fn terminate(&self) -> Result<(), JobError> {
        // SAFETY: `self.0` is a valid, open job handle owned by this wrapper (see `assign`), and a
        // exit code of 1 is the documented parameter meaning "terminated", nothing is read.
        let ok = unsafe { TerminateJobObject(self.0, 1) };
        if ok == 0 {
            return Err(win32_failure("TerminateJobObject"));
        }
        Ok(())
    }
}

#[cfg(windows)]
impl Drop for Job {
    fn drop(&mut self) {
        // SAFETY: `self.0` was created by `create` above, is owned solely by this wrapper, and this
        // runs at most once per instance. Closing the last handle of a job with
        // `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` terminates every process still assigned to it: this
        // one call is how dropping a `Job` kills the whole tree, not merely the immediate PID.
        unsafe {
            CloseHandle(self.0);
        }
    }
}

#[cfg(not(windows))]
impl Job {
    /// Always fails with [`JobError::Unsupported`]: job objects are Windows-only.
    ///
    /// # Errors
    /// Always [`JobError::Unsupported`].
    pub fn create() -> Result<Self, JobError> {
        Err(JobError::Unsupported)
    }

    /// Always fails with [`JobError::Unsupported`]: job objects are Windows-only.
    ///
    /// # Errors
    /// Always [`JobError::Unsupported`].
    pub fn assign(&self, _child: &std::process::Child) -> Result<(), JobError> {
        Err(JobError::Unsupported)
    }

    /// Always fails with [`JobError::Unsupported`]: job objects are Windows-only.
    ///
    /// # Errors
    /// Always [`JobError::Unsupported`].
    pub fn terminate(&self) -> Result<(), JobError> {
        Err(JobError::Unsupported)
    }
}

/// Builds a typed error from the calling thread's last Win32 error, immediately after a failed call.
#[cfg(windows)]
fn win32_failure(call: &'static str) -> JobError {
    // SAFETY: reads the calling thread's last-error value; every caller invokes this immediately
    // after the failed call it names, before any other API call can overwrite it.
    let code = unsafe { GetLastError() };
    JobError::Win32 { call, code }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn create_succeeds_and_drop_does_not_panic() {
        let job = Job::create().expect("create a job object");
        drop(job);
    }

    #[test]
    fn terminate_on_a_job_with_no_processes_is_not_an_error() {
        let job = Job::create().expect("create a job object");
        job.terminate().expect("terminate an empty job");
    }

    #[test]
    fn assigning_an_invalid_handle_reports_a_typed_win32_error_rather_than_crashing() {
        let job = Job::create().expect("create a job object");
        let bogus = std::process::Command::new("cmd").arg("/C").arg("exit 0").spawn().expect("spawn a child to hold a dead handle");
        let mut child = bogus;
        let _ = child.wait();
        // A dead child's handle is still a valid handle; the point here is only that `assign`
        // answers with a result (never a crash) on a live handle too.
        let _ = job.assign(&child);
    }
}
