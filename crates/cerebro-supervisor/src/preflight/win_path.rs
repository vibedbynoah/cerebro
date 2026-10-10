//! Win32 handle-to-path resolution used by [`super::verify_model_file`], `verify_api_key_file`, and
//! `crate::spawn` (for a temp or log directory reached through a link while
//! `Protections::refuse_redirected_paths` is off).
//!
//! Split out as a small, reviewable `unsafe` surface kept apart from the pure decision logic in
//! `super`.

use std::ffi::OsString;
use std::fs::File;
use std::io;
use std::os::windows::ffi::OsStringExt;
use std::os::windows::io::AsRawHandle;
use std::path::PathBuf;

use windows_sys::Win32::Storage::FileSystem::{
    GetFinalPathNameByHandleW, FILE_NAME_NORMALIZED, VOLUME_NAME_DOS,
};

/// Resolve the canonical, `\\?\`-prefixed path of an already-open file, straight from its handle.
///
/// # Why this exists instead of `std::fs::canonicalize(path)`
///
/// `verify_model_file` opens `path`, then runs the reparse-point, regular-file and GGUF-magic
/// checks against that one handle. The version this replaced then called
/// `fs::canonicalize(path)` — which re-resolves the name *from scratch*, by string, and follows a
/// reparse point in the last path component while doing so. If `path`'s last component were
/// replaced with a symlink in the instant between the magic read and that call, `canonicalize`
/// would silently follow it and hand back the *target's* path: a path that was never opened,
/// never checked for the GGUF magic, and never checked for being a reparse point at all. Its own
/// doc comment claimed this window was closed; it had only moved.
///
/// Asking the handle already in hand for its own final path cannot exhibit that problem: there is
/// no second name resolution for anything to race against. The one gap this still leaves — the
/// spawn reopening the model file by name, some time after preflight returns — belongs to the
/// lifecycle layer, not to preflight; see this module's top-level docs.
///
/// The returned path keeps its `\\?\` prefix. Verified on this machine against the real b10751
/// build: `llama-server -m \\?\C:\...\model.gguf` loads the model without complaint, so nothing
/// here strips the prefix — doing so would be string surgery that would then need the comma and
/// absolute-path checks re-run on whatever the stripped form turned out to be, for no benefit.
pub(crate) fn final_path(file: &File) -> io::Result<PathBuf> {
    // Large enough that the common case needs exactly one call; not `MAX_PATH`, because a
    // `\\?\`-prefixed name is long-path-capable and can legitimately exceed it. Split out as
    // `final_path_with_capacity` so the grow-and-retry branch below can be exercised directly with
    // a deliberately too-small capacity, independent of how long any real path on the test machine
    // happens to be.
    final_path_with_capacity(file, 512)
}

/// [`final_path`] with the initial buffer capacity (in UTF-16 units) exposed, so the too-small
/// retry branch can be forced by a test without depending on a coincidentally long fixture path.
fn final_path_with_capacity(file: &File, initial: usize) -> io::Result<PathBuf> {
    let mut buf: Vec<u16> = vec![0; initial.max(1)];
    loop {
        let cap = u32::try_from(buf.len())
            .map_err(|_| io::Error::other("resolved path exceeds u32::MAX UTF-16 units"))?;
        // SAFETY: `file` is a live `File` borrowed for this whole call, so its handle is valid.
        // `buf` is a writable buffer of exactly `cap` `u16` elements; `GetFinalPathNameByHandleW`
        // writes at most `cap` elements including its terminator, so it can never write past
        // `buf`'s allocation. `buf.as_mut_ptr()` is used only for the duration of this call.
        let written = unsafe {
            GetFinalPathNameByHandleW(
                file.as_raw_handle().cast(),
                buf.as_mut_ptr(),
                cap,
                FILE_NAME_NORMALIZED | VOLUME_NAME_DOS,
            )
        };
        if written == 0 {
            return Err(io::Error::last_os_error());
        }
        let written = written as usize;
        if written < buf.len() {
            // Success: on success the return value is the length actually written, excluding the
            // terminator — distinct from the too-small case below, where it is the required size
            // *including* the terminator. Confusing the two would either truncate a long path or
            // include a trailing NUL in the returned `PathBuf`.
            buf.truncate(written);
            return Ok(PathBuf::from(OsString::from_wide(&buf)));
        }
        // The buffer was too small; grow to exactly the size the API reported and retry, rather
        // than looping on a fixed increment that could take many calls for a very long path.
        buf.resize(written, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A uniquely named temp file, removed on drop, mirroring `preflight`'s own `TempDir` helper.
    struct TempFile(PathBuf);

    /// A temp-file path unique per process and per call, made only of characters valid in an NTFS
    /// file name. An earlier version formatted `SystemTime` with `{:?}`, whose `SystemTime {
    /// intervals: N }` contains a `:`, which NTFS reads as a stream separator: the test wrote an
    /// alternate data stream on a file named up to the colon, and `TempFile` removed the stream,
    /// leaving that file behind in %TEMP% on every run.
    fn unique_temp_path(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        std::env::temp_dir().join(format!(
            "cerebro-supervisor-{tag}-{}-{nanos}-{n}.txt",
            std::process::id()
        ))
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn final_path_of_a_real_open_file_agrees_with_canonicalize() {
        let path = unique_temp_path("win-path-test");
        let mut created = File::create(&path).expect("create fixture file");
        created.write_all(b"x").expect("write fixture file");
        drop(created);
        let guard = TempFile(path.clone());

        let file = File::open(&guard.0).expect("open fixture file");
        let resolved = final_path(&file).expect("resolve final path from handle");
        // `std::fs::canonicalize` is itself implemented on top of `GetFinalPathNameByHandleW` with
        // the same flags, so comparing against it is a real cross-check of this function's Win32
        // call, not merely a restatement of it: a bug in the buffer-growth loop above (an off-by-one
        // in the too-small/success distinction, for instance) would produce a truncated or
        // NUL-suffixed string that this comparison would catch.
        let expected = std::fs::canonicalize(&guard.0).expect("canonicalize fixture path");
        assert_eq!(resolved, expected);
    }

    #[test]
    fn a_too_small_initial_buffer_still_resolves_correctly() {
        // Calls `final_path_with_capacity` with a 1-element initial buffer — deliberately too small
        // for any real path — so the grow-and-retry branch inside it actually runs. `final_path`
        // itself always starts at 512 units, so calling it directly here (as an earlier version of
        // this test did) would only exercise that branch on a machine whose temp path happens to
        // exceed 512 UTF-16 units, which this development machine's does not; an off-by-one in the
        // `written < buf.len()` success/retry distinction, or in `buf.resize(written, 0)`, would
        // still pass such a test.
        let path = unique_temp_path("win-path-grow-test");
        std::fs::write(&path, b"x").expect("write fixture file");
        let guard = TempFile(path.clone());

        let file = File::open(&guard.0).expect("open fixture file");
        let resolved = final_path_with_capacity(&file, 1)
            .expect("final_path_with_capacity must still succeed via its own retry");
        assert_eq!(resolved, std::fs::canonicalize(&guard.0).unwrap());
    }
}
