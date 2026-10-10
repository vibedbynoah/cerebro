//! Which other processes are running, as a measurement condition.
//!
//! A benchmark measured while another `llama-server` or `llama-bench` is computing on the same CPU
//! and GPU is a measurement of contention, not of the configuration: a review run with a local
//! `llama-server` busy reported decode at depth 0 slower than decode at depth 128, which is
//! physically implausible, and nothing flagged it. One process snapshot before launch costs about
//! a millisecond and needs no privilege (image names only; no process is opened).
//!
//! The `unsafe` surface is four FFI calls, each carrying a `SAFETY` comment.

use crate::error::ProbeError;

/// One running process whose image name matched.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RunningProcess {
    /// Process id.
    pub pid: u32,
    /// Image file name as the kernel reports it, e.g. `llama-server.exe`.
    pub image_name: String,
}

/// Every running process whose image name equals one of `image_names` (ASCII case-insensitive),
/// in snapshot order.
///
/// # Platform
///
/// Windows only; every other platform returns [`ProbeError::UnsupportedPlatform`].
pub fn running_with_image_names(image_names: &[&str]) -> Result<Vec<RunningProcess>, ProbeError> {
    #[cfg(windows)]
    {
        Ok(win::snapshot()?
            .into_iter()
            .filter(|p| image_names.iter().any(|n| p.image_name.eq_ignore_ascii_case(n)))
            .collect())
    }
    #[cfg(not(windows))]
    {
        let _ = image_names;
        Err(ProbeError::UnsupportedPlatform)
    }
}

#[cfg(windows)]
mod win {
    use windows_sys::Win32::Foundation::{
        CloseHandle, GetLastError, ERROR_NO_MORE_FILES, HANDLE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W,
        TH32CS_SNAPPROCESS,
    };

    use super::RunningProcess;
    use crate::error::ProbeError;

    /// Closes the snapshot handle on every path out of [`snapshot`].
    struct Snapshot(HANDLE);

    impl Drop for Snapshot {
        fn drop(&mut self) {
            // SAFETY: `self.0` is a valid snapshot handle this guard owns exclusively; it is
            // closed exactly once, here.
            unsafe { CloseHandle(self.0) };
        }
    }

    fn last_error(call: &'static str) -> ProbeError {
        // SAFETY: reads the calling thread's last-error value immediately after the failed call.
        let code = unsafe { GetLastError() };
        ProbeError::Win32 { call, code }
    }

    pub(super) fn snapshot() -> Result<Vec<RunningProcess>, ProbeError> {
        // SAFETY: plain value arguments; the returned handle is checked before use and owned by
        // the guard below.
        let handle = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
        if handle == INVALID_HANDLE_VALUE {
            return Err(last_error("CreateToolhelp32Snapshot"));
        }
        let snap = Snapshot(handle);
        let mut entry = PROCESSENTRY32W {
            dwSize: u32::try_from(std::mem::size_of::<PROCESSENTRY32W>()).unwrap_or(u32::MAX),
            ..PROCESSENTRY32W::default()
        };
        let mut out = Vec::new();
        // SAFETY: `snap.0` is a live snapshot handle and `entry` a live, aligned local of the
        // exact type the API writes, with `dwSize` set as it requires; nothing is retained.
        let mut ok = unsafe { Process32FirstW(snap.0, &mut entry) };
        while ok != 0 {
            let len = entry
                .szExeFile
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(entry.szExeFile.len());
            let name = entry.szExeFile.get(..len).unwrap_or_default();
            out.push(RunningProcess {
                pid: entry.th32ProcessID,
                image_name: String::from_utf16_lossy(name),
            });
            // SAFETY: as for `Process32FirstW` above.
            ok = unsafe { Process32NextW(snap.0, &mut entry) };
        }
        // SAFETY: reads the last-error value immediately after the call that ended the loop.
        let code = unsafe { GetLastError() };
        if code != ERROR_NO_MORE_FILES {
            return Err(ProbeError::Win32 {
                call: "Process32NextW",
                code,
            });
        }
        Ok(out)
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn the_snapshot_finds_this_test_process_by_its_own_image_name() {
        let exe = std::env::current_exe().unwrap();
        let name = exe.file_name().unwrap().to_str().unwrap().to_ascii_uppercase();
        let found = running_with_image_names(&[&name]).unwrap();
        assert!(
            found.iter().any(|p| p.pid == std::process::id()),
            "{name} not found in {found:?}"
        );
        assert!(running_with_image_names(&["no-such-image-7f3e.exe"]).unwrap().is_empty());
        assert!(running_with_image_names(&[]).unwrap().is_empty());
    }
}
