//! Which file an open handle refers to, independent of its name, size or timestamps.
//!
//! A cache keyed by (path, size, modification time) cannot see a file replaced by another of the
//! same size whose modification time was set back (`SetFileTime` needs only write access). The
//! volume serial number plus the file index identify the file object itself: a replacement is a
//! different file and gets a different index. `std` exposes these only on nightly
//! (`windows_by_handle`), so this is one `GetFileInformationByHandle` call.

use std::fs::File;

use crate::error::ProbeError;

/// The identity of the file object behind a handle: same value, same file (until it is deleted
/// and its index reused, which an open handle prevents).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileIdentity {
    /// Serial number of the volume holding the file.
    pub volume_serial: u32,
    /// The file's index on that volume (the NTFS file reference number).
    pub file_index: u64,
}

/// The identity of the file `file` is open on.
///
/// # Platform
///
/// Windows only; every other platform returns [`ProbeError::UnsupportedPlatform`].
pub fn identity(file: &File) -> Result<FileIdentity, ProbeError> {
    #[cfg(windows)]
    {
        win::identity(file)
    }
    #[cfg(not(windows))]
    {
        let _ = file;
        Err(ProbeError::UnsupportedPlatform)
    }
}

#[cfg(windows)]
mod win {
    use std::fs::File;
    use std::os::windows::io::AsRawHandle;

    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
    };

    use super::FileIdentity;
    use crate::error::ProbeError;

    pub(super) fn identity(file: &File) -> Result<FileIdentity, ProbeError> {
        // SAFETY: an all-zero bit pattern is a valid value of this plain-data struct.
        let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `file` owns a live handle for the whole call, and `info` is a live, aligned
        // local of the exact type the API writes; nothing is retained after it returns.
        if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) } == 0 {
            // SAFETY: reads the calling thread's last-error value immediately after the failure.
            let code = unsafe { GetLastError() };
            return Err(ProbeError::Win32 {
                call: "GetFileInformationByHandle",
                code,
            });
        }
        Ok(FileIdentity {
            volume_serial: info.dwVolumeSerialNumber,
            file_index: (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow),
        })
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn a_replaced_file_has_a_new_identity_and_the_same_file_keeps_its_own() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("cerebro-probe-file-id-{}.bin", std::process::id()));
        let other = dir.join(format!("cerebro-probe-file-id-{}-b.bin", std::process::id()));
        std::fs::write(&path, b"same size").unwrap();
        std::fs::write(&other, b"same size").unwrap();
        let first = identity(&File::open(&path).unwrap()).unwrap();
        assert_eq!(first, identity(&File::open(&path).unwrap()).unwrap());
        // Same name, same size, same bytes: a different file object all the same.
        std::fs::rename(&other, &path).unwrap();
        let replaced = identity(&File::open(&path).unwrap()).unwrap();
        assert_eq!(replaced.volume_serial, first.volume_serial);
        assert_ne!(replaced.file_index, first.file_index);
        let _ = std::fs::remove_file(&path);
    }
}
