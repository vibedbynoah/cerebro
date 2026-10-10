//! Win32 half of child containment: the Job Object that makes a spawned tree unable to outlive
//! Cerebro.
//!
//! Split out for the same reason `preflight::elevation` and `preflight::win_path` are: a small,
//! reviewable `unsafe` surface kept apart from the process-lifecycle logic in `super`.
//!
//! # Why a Job Object, and not just killing the one child PID
//!
//! `llama-server` can itself spawn further processes (an `--rpc` worker it launches, for instance —
//! see ARCHITECTURE.md section 5's loopback RPC pattern). Killing only the PID `Command::spawn`
//! returned would leave any such grandchild running as an orphan the moment Cerebro exits or the
//! parent handle is dropped. Assigning the child to a Job Object created with
//! `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` fixes this at the kernel level: every process ever assigned
//! to the job (the immediate child, and — because that limit is inherited — anything *it* spawns
//! while assigned) is terminated the instant the job's last handle closes, whether that close comes
//! from an explicit [`JobHandle::drop`] or from Cerebro's own process dying unexpectedly. No
//! userspace bookkeeping of a process tree can offer that guarantee; only the kernel object can.

use std::ptr::{addr_of, null_mut};

use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, HANDLE};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

use super::SpawnError;

/// An unnamed Windows Job Object configured, at construction, to kill every process assigned to it
/// the moment its last handle closes.
///
/// There is deliberately no way to construct one of these without
/// [`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`] already set: [`JobHandle::create`] is the only
/// constructor, and it applies the limit before returning, so a `JobHandle` existing at all is
/// proof the kill-on-close behavior is armed.
pub(super) struct JobHandle(HANDLE);

// A Win32 `HANDLE` is an opaque, process-scoped kernel reference, not a thread-affine resource
// (unlike, say, a GDI device context) — Win32 documents `CloseHandle`/`AssignProcessToJobObject` as
// safe to call from any thread of the owning process. `JobHandle`'s API only ever hands out shared
// access (`&self`) to the wrapped handle, so `Sync` is not claimed, only `Send`.
unsafe impl Send for JobHandle {}

impl JobHandle {
    /// Create a new, unnamed job object with kill-on-close armed.
    pub(super) fn create() -> Result<Self, SpawnError> {
        // SAFETY: both arguments being null requests default security attributes and no name,
        // which is documented as a valid `CreateJobObjectW` call; the return value is checked for
        // null (failure) immediately below before it is treated as a handle.
        let raw = unsafe { CreateJobObjectW(null_mut(), null_mut()) };
        if raw.is_null() {
            return Err(win32_failure("CreateJobObjectW"));
        }
        let job = JobHandle(raw);
        job.arm_kill_on_close()?;
        Ok(job)
    }

    /// Set `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` on this job. Called exactly once, from
    /// [`JobHandle::create`], before the job is ever handed back to a caller.
    fn arm_kill_on_close(&self) -> Result<(), SpawnError> {
        // Zero-initialized is a valid `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`: every field this call
        // does not explicitly set (the I/O counters, the process/memory limits, `IoInfo`) is
        // documented as ignored unless its corresponding bit is present in `LimitFlags`, which is
        // set to nothing but the one bit this function cares about.
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

        // SAFETY: `self.0` is a valid job handle owned by this wrapper (just created, and not yet
        // returned to any caller who could have closed it). `info` is a live, correctly-aligned
        // local of exactly `size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>()` bytes, which is the
        // size passed, so the call can read no more than the buffer holds.
        let ok = unsafe {
            SetInformationJobObject(
                self.0,
                JobObjectExtendedLimitInformation,
                addr_of!(info).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if ok == 0 {
            return Err(win32_failure("SetInformationJobObject"));
        }
        Ok(())
    }

    /// Assign `process` to this job.
    ///
    /// `process` must be a valid, open, live process handle; this function does not take ownership
    /// of it (the caller — [`super::spawn_windows_at`] — keeps and eventually closes it via its own
    /// `std::process::Child`). Once assigned, the process (and anything it spawns while still
    /// assigned) is terminated the moment this `JobHandle` is dropped, or immediately if the
    /// process has already exceeded a limit this job enforces — kill-on-close is the only limit set
    /// here, so assignment alone never kills anything by itself.
    pub(super) fn assign(&self, process: HANDLE) -> Result<(), SpawnError> {
        // SAFETY: `self.0` is a valid job handle owned by this wrapper and still open (no `&self`
        // call can run concurrently with `drop`, which takes `&mut self`). `process` is required by
        // this function's own contract (see above) to be a valid, live process handle for the
        // duration of this call, which the sole caller satisfies by passing a handle it holds open
        // via its own `Child`.
        let ok = unsafe { AssignProcessToJobObject(self.0, process) };
        if ok == 0 {
            return Err(win32_failure("AssignProcessToJobObject"));
        }
        Ok(())
    }
}

impl Drop for JobHandle {
    fn drop(&mut self) {
        // SAFETY: `self.0` was created by `create` above, is owned solely by this wrapper, and this
        // runs at most once per instance (ordinary `Drop`). Closing the last handle to a job
        // configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` terminates every process still
        // assigned to it — this one call is the entire mechanism by which dropping a
        // [`super::ChildHandle`] kills the whole child tree, not merely the immediate PID.
        unsafe {
            CloseHandle(self.0);
        }
    }
}

/// Build a typed error from the calling thread's last Win32 error, immediately after a failed call.
fn win32_failure(call: &'static str) -> SpawnError {
    // SAFETY: reads the calling thread's last-error value; every caller invokes this immediately
    // after the failed call it names, before any other API call can overwrite it.
    let code = unsafe { GetLastError() };
    SpawnError::Win32 { call, code }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_succeeds_and_drop_does_not_panic() {
        // The FFI call sequence itself, exercised without a real child process: `create` must
        // succeed and `arm_kill_on_close` must not reject a freshly created job.
        let job = JobHandle::create().expect("create a job object");
        drop(job);
    }

    #[test]
    fn assigning_an_invalid_handle_reports_a_typed_win32_error_rather_than_crashing() {
        let job = JobHandle::create().expect("create a job object");
        // `HANDLE` is a plain, process-scoped integer-ish value on this platform; `-2` (the
        // "pseudo current process" style sentinel some Win32 APIs recognize elsewhere) is not a
        // registered handle from `AssignProcessToJobObject`'s point of view and is guaranteed to be
        // rejected rather than accidentally aliasing a real open handle.
        let bogus: HANDLE = -2isize as HANDLE;
        let err = job.assign(bogus).expect_err("a bogus handle must not be silently accepted");
        assert!(matches!(err, SpawnError::Win32 { call: "AssignProcessToJobObject", .. }), "{err:?}");
    }
}
