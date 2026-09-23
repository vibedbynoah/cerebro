//! Win32 half of the elevation check.
//!
//! Split out for the same reason `cerebro_probe::backend::dacl` is: the `unsafe` surface is one
//! small, reviewable FFI function, kept apart from the pure decision logic in `super`.

use std::ptr::null_mut;

use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, HANDLE};
use windows_sys::Win32::Security::{
    GetTokenInformation, TokenElevation, TOKEN_ELEVATION, TOKEN_QUERY,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

use super::PreflightError;

/// `TOKEN_ELEVATION` is one `u32`; this is a compile-time fact, not a runtime conversion, so it
/// needs no fallible cast (and therefore no `expect`) to hand to `GetTokenInformation`, which wants
/// the size as a `u32`.
const TOKEN_ELEVATION_SIZE: u32 = std::mem::size_of::<TOKEN_ELEVATION>() as u32;

/// A token handle this function opened and owns exclusively; closed exactly once on drop.
struct OwnedHandle(HANDLE);

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        // SAFETY: `self.0` was opened by `query` below, is owned solely by this wrapper, and drop
        // runs at most once per instance.
        unsafe {
            CloseHandle(self.0);
        }
    }
}

/// Whether the current process's primary token carries the elevated bit.
///
/// ARCHITECTURE.md section 8: "Cerebro refuses to run elevated" — nothing in the inference path
/// needs admin. This is therefore a refusal gate, not informational: [`super::preflight`] treats
/// `Ok(true)` as a hard stop.
pub(crate) fn query() -> Result<bool, PreflightError> {
    let mut raw: HANDLE = null_mut();
    // SAFETY: `GetCurrentProcess` returns a pseudo-handle that needs no closing and cannot fail.
    // `raw` is a live local that receives a real, process-owned token handle on success; that
    // handle is wrapped for guaranteed closure on the next line before any early return.
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut raw) } == 0 {
        return Err(win32_failure("OpenProcessToken"));
    }
    let token = OwnedHandle(raw);

    let mut elevation = TOKEN_ELEVATION { TokenIsElevated: 0 };
    let mut returned: u32 = 0;
    // SAFETY: `elevation` is a live, correctly-aligned local of exactly `TOKEN_ELEVATION_SIZE`
    // bytes, and that constant is that struct's own `size_of`, so the API can write no more than
    // the buffer holds. `returned` is a live local for the (unused, fixed-size) output length.
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
        return Err(win32_failure("GetTokenInformation"));
    }
    Ok(elevation.TokenIsElevated != 0)
}

/// Build a typed error from the calling thread's last Win32 error, immediately after a failed call.
fn win32_failure(call: &'static str) -> PreflightError {
    // SAFETY: reads the calling thread's last-error value; every caller invokes this immediately
    // after the failed call it names, before any other API call can overwrite it.
    let code = unsafe { GetLastError() };
    PreflightError::ElevationCheckFailed(format!("{call} failed with Win32 error {code}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_completes_without_leaking_or_panicking() {
        // Whether *this particular* test run happens to be elevated is an environment fact, not a
        // property of the code under test, and is exercised instead by the injectable-trait tests
        // in `preflight`, which fake the answer rather than depending on the ambient token. This
        // test only proves the FFI call sequence itself is sound.
        let _ = query().expect("querying this process's own token must succeed");
    }
}
