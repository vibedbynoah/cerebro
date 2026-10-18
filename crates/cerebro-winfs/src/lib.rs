//! Two filesystem facts Windows does not expose through `std`, for the Models screen's disk tools
//! (apps/desktop/src-tauri/src/dedup.rs):
//!
//! - [`identify`]: which file a path *is*, as the volume's serial number plus the file's index on
//!   that volume. Every path to the same file — its hard links included — reports the same
//!   [`FileId`], so a scan can tell "the same file kept under two names" from "the same bytes
//!   stored twice".
//! - [`free_bytes`]: room left on the volume holding a path, so restoring a copy can refuse
//!   before it starts writing instead of failing halfway.
//!
//! The crate exists on its own because the desktop crate is `#![forbid(unsafe_code)]`: the two
//! Win32 calls live here, small and reviewable, like `cerebro-job`'s Job Object. On platforms
//! other than Windows both facts are "unknown": [`identify`] reports `Ok(None)` (a caller then
//! treats the files as separate, the safe direction) and [`free_bytes`] reports an error.

use std::path::Path;

/// Which file on a volume something is: the volume's serial number and the file's index there.
///
/// Two paths with equal ids are hard links to the same file (one file on disk, two names). The id
/// is meaningful on the machine that produced it (the same NTFS volume seen over two mounts can
/// report differently); that is all callers use it for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileId {
    /// The serial number Windows assigns the volume when it is formatted.
    pub volume: u32,
    /// The file's index within that volume.
    pub index: u64,
}

/// The identity of the file at `path`, or `Ok(None)` where the platform cannot tell (never on
/// Windows: there a failure to open or read the file is an error instead).
///
/// # Errors
/// The OS error when the file cannot be opened, or its handle information cannot be read. Opening
/// reads nothing: the file is not touched beyond asking the kernel about it.
#[cfg(windows)]
pub fn identify(path: &Path) -> std::io::Result<Option<FileId>> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::Storage::FileSystem::{GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION};

    let file = std::fs::File::open(path)?;
    let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `file` owns a valid handle for the whole call (it outlives it), and `info` is a
    // live, correctly-aligned local of exactly the struct the call fills. The return value is
    // checked before `info` is read, and a failed call is documented to leave it unspecified.
    let ok = unsafe { GetFileInformationByHandle(file.as_raw_handle() as HANDLE, &mut info) };
    if ok == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(Some(FileId {
        volume: info.dwVolumeSerialNumber,
        index: ((info.nFileIndexHigh as u64) << 32) | info.nFileIndexLow as u64,
    }))
}

/// Non-Windows builds cannot tell: hard links are reported as unknown, so callers treat every path
/// as its own file. That only loses the "already linked" grouping, never any data.
///
/// # Errors
/// Never; the signature mirrors the Windows one.
#[cfg(not(windows))]
pub fn identify(_path: &Path) -> std::io::Result<Option<FileId>> {
    Ok(None)
}

/// Free bytes on the volume that holds `path`. The folder that holds the path is what is asked
/// about — Windows' call wants a folder, and a file always lives on its folder's volume — so the
/// volume, not the folder, is what is measured.
///
/// # Errors
/// The OS error when the volume cannot be queried.
#[cfg(windows)]
pub fn free_bytes(path: &Path) -> std::io::Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let dir: &Path = if path.is_dir() { path } else { path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new(".")) };
    let mut wide: Vec<u16> = dir.as_os_str().encode_wide().collect();
    wide.push(0);
    let mut free = 0u64;
    // SAFETY: `wide` is NUL-terminated and lives for the call; `free` is a live local the call
    // only writes to; the two null pointers are documented as "not wanted".
    let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut free, std::ptr::null_mut(), std::ptr::null_mut()) };
    if ok == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(free)
}

/// Non-Windows builds cannot tell how much room is left.
///
/// # Errors
/// Always [`std::io::ErrorKind::Unsupported`]; callers there skip the check and let a copy fail at
/// the write instead.
#[cfg(not(windows))]
pub fn free_bytes(_path: &Path) -> std::io::Result<u64> {
    Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "free space is known on Windows only"))
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    /// A fresh folder under the system temp directory, removed on drop.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
            let dir = std::env::temp_dir().join(format!("cerebro-winfs-{:016x}", stamp ^ (N.fetch_add(1, Ordering::Relaxed) << 40)));
            std::fs::create_dir_all(&dir).expect("temp dir");
            TempDir(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn identifies_hard_links_as_one_file() {
        let dir = TempDir::new();
        let a = dir.0.join("a.bin");
        let b = dir.0.join("b.bin");
        let other = dir.0.join("other.bin");
        std::fs::write(&a, b"same").unwrap();
        std::fs::hard_link(&a, &b).unwrap();
        std::fs::write(&other, b"same").unwrap();

        let id_a = identify(&a).unwrap().expect("a file id");
        let id_b = identify(&b).unwrap().expect("a file id");
        let id_other = identify(&other).unwrap().expect("a file id");
        assert_eq!(id_a, id_b);
        assert_ne!(id_a, id_other);
        assert_eq!(id_a.volume, id_other.volume);
    }

    #[test]
    fn reports_free_space_and_the_missing_file_error() {
        let dir = TempDir::new();
        let a = dir.0.join("a.bin");
        std::fs::write(&a, b"x").unwrap();
        assert!(free_bytes(&a).unwrap() > 0);
        assert!(identify(&dir.0.join("missing.bin")).is_err());
    }
}
