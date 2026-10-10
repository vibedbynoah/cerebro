//! Facts about this process's own access token.
//!
//! One implementation of "is this process elevated", shared by [`crate::backend::probe_with`]
//! (which refuses to run its backend queries elevated unless the user turned that protection off)
//! and `cerebro-supervisor`'s preflight gate (which refuses a launch the same way). Both start the
//! same backend binaries, so both must ask the OS the same question the same way; keeping a second
//! copy of the Win32 query in the supervisor would let the two drift.
//!
//! The `unsafe` surface is two FFI calls, each carrying a `SAFETY` comment.

use crate::error::ProbeError;

/// Whether the current process's primary token carries the elevated bit (`TokenElevation`).
///
/// Nothing in the inference path needs admin rights, so callers treat `Ok(true)` as a reason to
/// refuse, and an `Err` as "cannot tell", which they also refuse on rather than assume "not
/// elevated".
///
/// # Platform
///
/// Windows only; every other platform returns [`ProbeError::UnsupportedPlatform`], which callers
/// treat as the check failing (fail closed).
pub fn is_elevated() -> Result<bool, ProbeError> {
    #[cfg(windows)]
    {
        win::query()
    }
    #[cfg(not(windows))]
    {
        Err(ProbeError::UnsupportedPlatform)
    }
}

#[cfg(windows)]
mod win {
    use std::ptr::null_mut;

    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, HANDLE};
    use windows_sys::Win32::Security::{
        GetTokenInformation, TokenElevation, TOKEN_ELEVATION, TOKEN_QUERY,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    use crate::error::ProbeError;

    /// `TOKEN_ELEVATION` is one `u32`; a compile-time fact, so no fallible cast is needed to hand
    /// its size to `GetTokenInformation`.
    const TOKEN_ELEVATION_SIZE: u32 = std::mem::size_of::<TOKEN_ELEVATION>() as u32;

    /// A token handle opened by [`query`] and owned exclusively; closed exactly once on drop.
    struct OwnedHandle(HANDLE);

    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            // SAFETY: `self.0` was opened by `query`, is owned solely by this wrapper, and drop runs
            // at most once per instance.
            unsafe {
                CloseHandle(self.0);
            }
        }
    }

    pub(super) fn query() -> Result<bool, ProbeError> {
        let mut raw: HANDLE = null_mut();
        // SAFETY: `GetCurrentProcess` returns a pseudo-handle that needs no closing and cannot
        // fail. `raw` is a live local that receives a process-owned token handle on success; it is
        // wrapped for guaranteed closure on the next line, before any early return.
        if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut raw) } == 0 {
            return Err(last_error("OpenProcessToken"));
        }
        let token = OwnedHandle(raw);

        let mut elevation = TOKEN_ELEVATION { TokenIsElevated: 0 };
        let mut returned: u32 = 0;
        // SAFETY: `elevation` is a live, aligned local of exactly `TOKEN_ELEVATION_SIZE` bytes (the
        // struct's own `size_of`), so the API writes no more than the buffer holds. `returned` is a
        // live local for the fixed-size output length.
        let ok = unsafe {
            GetTokenInformation(
                token.0,
                TokenElevation,
                std::ptr::addr_of_mut!(elevation).cast(),
                TOKEN_ELEVATION_SIZE,
                &mut returned,
            )
        };
        if ok == 0 {
            return Err(last_error("GetTokenInformation"));
        }
        Ok(elevation.TokenIsElevated != 0)
    }

    fn last_error(call: &'static str) -> ProbeError {
        // SAFETY: reads the calling thread's last-error value; every caller invokes this
        // immediately after the failed call it names.
        let code = unsafe { GetLastError() };
        ProbeError::Win32 { call, code }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn query_completes_without_leaking_or_panicking() {
        // Whether this run is elevated is a fact about the environment, not the code; the refusal
        // branches are proven with injected answers in `backend` and in the supervisor's preflight.
        // This proves only that the FFI sequence itself is sound.
        let _ = is_elevated().expect("querying this process's own token must succeed");
    }
}
