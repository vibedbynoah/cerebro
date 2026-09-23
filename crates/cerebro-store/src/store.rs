//! The append-only, hash-chained JSON Lines file.
//!
//! # What the hash chain proves, and what it does not
//!
//! Each stored line carries `prev_hash` (the previous line's link hash) and its own
//! `record_hash` (`sha256(prev_hash_or_empty || 0x00 || raw record bytes)`, see [`crate::line`]).
//! [`Store::load_verified`] recomputes both for every line it reads.
//!
//! **Detected**: a dropped, duplicated, or reordered line anywhere in the file; a bit flip or a
//! hand edit inside *any* record, including the last one (`record_hash` covers the record's own
//! bytes, not only the link to it).
//!
//! **Not detected**: deleting whole records from the *end* of the file. The remaining prefix is,
//! by construction, still a valid chain on its own -- there is nothing after the last surviving
//! record to notice it is missing. Detecting that would need an external anchor (e.g. the last
//! `record_hash` and record count kept in separate profile metadata); this crate does not keep one.
//! Also not detected: a text editor's CRLF line-ending conversion. `scan_reader` splits lines on
//! `read_until(b'\n')`, which leaves a trailing `\r` inside `content`; `serde_json::from_slice`
//! treats that `\r` as insignificant JSON whitespace outside any string, so the line still parses;
//! and `record_hash` covers only the `record` value's own bytes, which [`serde_json::value::RawValue`]
//! borrows verbatim from the input -- the `\r` sits outside every hashed region on every line.
//! Confirmed against this crate: a file rewritten LF -> CRLF after the fact still loads, still
//! verifies, and still accepts a further (LF-terminated) append. Harmless -- llama.cpp's argv and
//! this crate's own JSON round-trip neither produce nor care about the byte -- but this crate does
//! not notice it, and `docs/CLAIMS.md` does not let this module say that it does.
//!
//! **This is not a defense against an attacker with write access to the file.** Anyone who can
//! modify a line can also recompute every hash after it and produce a chain that verifies
//! cleanly; there is no key, signature, or secret anywhere in this scheme. `docs/CLAIMS.md`
//! forbids shipping a control like this as if it were security it is not, so this crate does not
//! claim tamper-*proofing*, only tamper-*evidence* against corruption that does not also rewrite
//! the rest of the file.
//!
//! # Crash safety
//!
//! [`Store::append`] writes with exactly one [`std::io::Write::write_all`] call followed by
//! [`std::fs::File::sync_all`]. On real Windows file I/O a crash mid-write can still leave more
//! than a clean prefix behind -- cached pages flush independently, and valid-data-length
//! zero-fill can leave a final line ending in a real `\n` with NULs or garbage inside it -- so a
//! torn tail is classified by *content*, not only by the presence of a trailing newline:
//!
//! - A final line that fully parses, verifies its hashes, and passes schema and semantic
//!   validation is a **valid record** and is kept, whether or not it ends in `\n`. If it is
//!   missing the newline, [`Store::append`] writes one before its own new line.
//! - A final line that fails to parse, or exceeds the size limit before a terminator is found, is
//!   an excluded [`TornTail`] -- reported, never silently dropped.
//! - A chain, hash, schema, or semantic failure on the last line is treated the same as anywhere
//!   else in the file: a hard [`crate::error::StoreError`], not a torn tail. A crash does not
//!   produce a record that parses cleanly but claims the wrong `prev_hash` or an invalid sampler
//!   value; a hostile or hand-edited file might, and this crate refuses to build on top of it or
//!   discard it silently.
//!
//! [`Store::append`] also repairs a torn tail it finds before writing a new record, since a torn
//! line was never confirmed durable and appending on top of it without removing it first would
//! bury it mid-file where it could never again be recognized as a crash artifact.
//!
//! One gap this crate does not close: `sync_all` makes the file's *contents* durable, but nothing
//! here syncs the parent *directory*, so a crash immediately after the very first append to a new
//! machine profile can leave the file's data durable and its directory entry naming it not. Every
//! subsequent append to an already-existing file is unaffected.
//!
//! **Measured, not assumed** (`docs/CLAIMS.md`'s evidence standard): opening the parent as a
//! directory handle needs no Win32 FFI at all -- `OpenOptions::new().custom_flags(0x0200_0000
//! /* FILE_FLAG_BACKUP_SEMANTICS */)` is the exact safe-std `OpenOptionsExt` trait this module
//! already uses for `share_mode`, and the `open()` call succeeds. An earlier version of this
//! module claimed the gap was left open because closing it needed FFI this
//! `#![forbid(unsafe_code)]` crate does not carry; that claim was never run and was wrong. Running
//! it against this reference machine's NTFS volume shows the real obstacle one step later:
//! `File::sync_all` on that directory handle (`FlushFileBuffers`) fails with
//! `ERROR_ACCESS_DENIED` (raw OS error 5) for an ordinary, non-elevated process --
//! `FlushFileBuffers` on a directory requires a privilege (`SeBackupPrivilege`/administrator) this
//! crate cannot assume the caller has, and Cerebro is meant to run unelevated. Demanding elevation
//! to close a one-record-wide crash window would be a worse trade than leaving it documented and
//! open, so this crate does not attempt it.
//!
//! # Concurrency
//!
//! **One writer at a time; readers are unrestricted.** [`Store::append`] and [`Store::append_all`]
//! open the file with a Windows share mode that permits any number of concurrent readers *and* a
//! concurrent writer to *open* the file, but the writer's own read+write access mask combined with
//! the reader's declared tolerance for it is what makes this work: [`Store::load_verified`] shares
//! both read and write access, so it never contends with an append in progress, while a *second*
//! appender's share mode (matching the first's) omits write sharing and is refused with
//! [`crate::error::StoreError::Busy`] -- two appenders racing (the tuner and the CLI, or two tuner
//! workers) fail one of them instead of both computing the same `prev_hash` from the same last line
//! and forking the chain.
//!
//! Windows checks share-mode compatibility in **both directions**: the appender's handle and the
//! reader's handle each declare what they tolerate concurrently, and both declarations must agree.
//! An earlier version of this module gave the reader only `FILE_SHARE_READ`, which reads like "any
//! reader may open this," but Windows still refused it while an appender's read+write handle was
//! open (`ERROR_SHARING_VIOLATION`, raw OS error 32) -- confirmed with a standalone probe using the
//! exact `OpenOptions` both functions use, not merely inferred from the flag's name. The fix is for
//! the reader to also share write access; that grants the reader no write capability of its own; it
//! only declares "I tolerate a concurrent writer."
//!
//! One consequence a reader must accept: it may observe a torn tail mid-append (the appender's
//! in-progress final line, not yet fsynced), which [`scan_reader`] already classifies as a
//! [`TornTail`] -- this is the intended, documented behavior for a concurrent reader, not a new
//! hazard introduced by allowing the read to proceed.
//!
//! [`open_read_locked`] also shares delete access (`FILE_SHARE_DELETE`), unlike
//! [`open_write_locked`]'s exclusive handle: a reader has no stake in the file's *name*, and
//! withholding that share flag (Rust's `OpenOptions` grants it by default, but naming any explicit
//! `share_mode` drops it) blocks [`Store::rotate_to`] for as long as any concurrent
//! [`Store::load_verified`] is in flight -- confirmed with a standalone probe: `std::fs::rename`
//! failed with `ERROR_SHARING_VIOLATION` against a reader missing this flag, and succeeded once it
//! was added. Granting it gives the reader no ability to delete or rename anything itself.
//!
//! This exclusion mechanism has no portable equivalent: Cerebro is Windows-first
//! (`docs/ARCHITECTURE.md`), and off Windows every write path
//! ([`Store::append`], [`Store::append_all`], [`Store::rotate_to`]) refuses outright with
//! [`crate::error::StoreError::UnsupportedPlatform`] rather than silently writing with no exclusion
//! at all -- the same convention `cerebro_probe::cpu::probe` uses for a platform-specific guarantee
//! with no portable substitute. Reading remains available on every platform.
//!
//! # Path trust
//!
//! `docs/ARCHITECTURE.md` section 8 adversary 3 -- a hostile local principal who can create a name
//! in a directory this crate writes to -- is **live on the reference machine**: a llama.cpp
//! directory this crate might share a parent with is Modify-writable by
//! `ASUS\CodexSandboxUsers`. Every opener in this module ([`open_write_locked`],
//! [`open_exclusive_for_rotate`], [`open_read_locked`]) therefore opens the final path component
//! itself with `FILE_FLAG_OPEN_REPARSE_POINT` and refuses with
//! [`crate::error::StoreError::PathIsReparsePoint`] when it turns out to be a symlink or an NTFS
//! junction, rather than transparently following it into wherever it points.
//!
//! **What this closes**: a planted reparse point at the store path silently redirecting
//! `set_len`/`write_all`/`rename` into a file this crate never chose.
//!
//! **What this does not close**: a directory another local principal can already write to. Refusing
//! a reparse point stops the one redirection trick that content-side validation (size caps, schema
//! checks, the hash chain) cannot see coming; it does not make a world- or group-writable results
//! directory a safe place to keep a durable log. The caller is responsible for placing the results
//! file in a directory no untrusted principal can write to -- the same precondition
//! `cerebro_probe::backend::ProbeError::BackendToolEscapesDir` documents for a caller-supplied tool
//! path.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use crate::error::StoreError;
use crate::line::{decode_line, encode_line};
use crate::record::Record;

/// Largest single serialized record this crate will write or accept while reading, in bytes.
///
/// `argv` and free-form `notes` fields are the only unbounded-looking inputs in [`Record`]; this
/// bounds what a corrupt or hostile results file can make [`Store::load_verified`] allocate for a
/// single line (reading is streamed and bounded per-line, so this limit is enforced before a
/// single oversized line is ever read in full), and what [`Store::append`] will ever commit to
/// disk.
pub const MAX_RECORD_BYTES: usize = 1024 * 1024;

/// Largest whole-file size this crate will scan, in bytes.
///
/// Checked against the file's metadata before a single byte is read, so a hostile multi-gigabyte
/// results file is rejected in one `stat` call. 256 MiB comfortably covers realistic
/// per-machine-profile result logs (each record is at most [`MAX_RECORD_BYTES`]) while remaining
/// small enough that a legitimate load never approaches the operating system's own file-size
/// limits.
pub const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// Largest number of records [`Store::load_verified`] will decode into memory in one call.
///
/// [`MAX_FILE_BYTES`] bounds bytes *on disk*; this bounds the heap those bytes expand into once
/// loaded, which is several times larger than the on-disk size -- each [`Record`] owns its `argv`
/// strings and several other heap allocations, so a file of minimal ~400-byte records near
/// [`MAX_FILE_BYTES`] would otherwise expand to roughly a gigabyte of live `Record`s without
/// tripping any limit this crate advertises. Checked inside [`scan_reader`]'s loop, so a hostile
/// file with more than this many lines is refused partway through rather than after it is fully
/// loaded.
pub const MAX_RECORDS: usize = 200_000;

/// `FILE_SHARE_READ` from `winnt.h`: any handle may also open the file for reading.
#[cfg(windows)]
const FILE_SHARE_READ: u32 = 0x1;

/// `FILE_SHARE_WRITE` from `winnt.h`: any handle may also open the file for writing.
///
/// [`open_read_locked`] sets this alongside [`FILE_SHARE_READ`] -- see the [module
/// documentation](self)'s concurrency section for why a *reader* declaring tolerance for a
/// concurrent *writer* is required, not merely permissive, on Windows.
#[cfg(windows)]
const FILE_SHARE_WRITE: u32 = 0x2;

/// `FILE_SHARE_DELETE` from `winnt.h`: any handle may also rename or delete the file.
///
/// Windows requires every open handle to a file -- including the caller's own -- to share delete
/// access before that file can be renamed, which [`Store::rotate_to`] needs to do while holding its
/// own exclusive writer handle open across the rename (to keep another appender from landing in the
/// gap). Confirmed with a standalone probe: a handle opened with only [`FILE_SHARE_READ`] (what
/// [`open_write_locked`] uses) makes `std::fs::rename` on the very path it has open fail with
/// `ERROR_SHARING_VIOLATION`, even from the same process and the same handle; adding this flag does
/// not weaken the single-writer guarantee, since exclusivity comes entirely from omitting
/// [`FILE_SHARE_WRITE`].
#[cfg(windows)]
const FILE_SHARE_DELETE: u32 = 0x4;

/// `ERROR_SHARING_VIOLATION`, the raw OS error Windows returns for a conflicting share mode.
///
/// Windows-only: off Windows this numeric code has an unrelated meaning (`32` is `EPIPE` on most
/// Unix-like platforms), so treating it as a sharing violation there would misclassify a broken
/// pipe as [`StoreError::Busy`].
#[cfg(windows)]
const ERROR_SHARING_VIOLATION: i32 = 32;

/// `FILE_FLAG_OPEN_REPARSE_POINT` from `winbase.h`: open the reparse point itself rather than
/// transparently following it to whatever it points at.
///
/// See the [module documentation](self)'s [`# Path trust`](self#path-trust) section. This flag
/// changes what a *successful* open returns -- it does not make the open fail against a junction --
/// so it is only load-bearing together with the [`FILE_ATTRIBUTE_REPARSE_POINT`] check every opener
/// performs immediately afterwards.
#[cfg(windows)]
const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;

/// `FILE_ATTRIBUTE_REPARSE_POINT` from `winnt.h`: set on a handle's metadata when the path it was
/// opened with (via [`FILE_FLAG_OPEN_REPARSE_POINT`]) named a symlink or an NTFS junction rather
/// than a regular file.
#[cfg(windows)]
const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;

fn io_err(operation: &'static str, path: &Path, e: &std::io::Error) -> StoreError {
    StoreError::Io { operation, path: path.display().to_string(), detail: e.to_string() }
}

/// Refuse `file` (opened at `path` with [`FILE_FLAG_OPEN_REPARSE_POINT`]) if it turns out to name a
/// reparse point rather than a regular file. See the [module documentation](self)'s
/// [`# Path trust`](self#path-trust) section.
///
/// `MetadataExt` (for `file_attributes`) is a different trait from `OpenOptionsExt` (for
/// `custom_flags`/`share_mode`), so it is imported here rather than at module scope -- pulling it
/// in unconditionally would be an unused import on a non-Windows build under `-D warnings`.
#[cfg(windows)]
fn refuse_if_reparse_point(file: &File, path: &Path) -> Result<(), StoreError> {
    use std::os::windows::fs::MetadataExt;
    let attrs = file.metadata().map_err(|e| io_err("metadata", path, &e))?.file_attributes();
    if attrs & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(StoreError::PathIsReparsePoint { path: path.display().to_string() });
    }
    Ok(())
}

/// Turn an `OpenOptions::open` failure into [`StoreError::Busy`] when it is a sharing conflict,
/// or a generic [`StoreError::Io`] otherwise.
#[cfg(windows)]
fn classify_open_err(operation: &'static str, path: &Path, e: std::io::Error) -> StoreError {
    if e.raw_os_error() == Some(ERROR_SHARING_VIOLATION) {
        StoreError::Busy { path: path.display().to_string() }
    } else {
        io_err(operation, path, &e)
    }
}

/// Off Windows there is no share-mode equivalent, so no OS error code here means "a sharing
/// violation" -- everything is reported as a generic [`StoreError::Io`].
#[cfg(not(windows))]
fn classify_open_err(operation: &'static str, path: &Path, e: std::io::Error) -> StoreError {
    io_err(operation, path, &e)
}

/// Open the store file for appending: read+write, created if absent, sharing only with readers.
///
/// This is the *only* handle [`Store::append`]/[`Store::append_all`] ever open for their whole
/// operation -- the read (via [`scan_reader`]), the torn-tail truncation, and the write all go
/// through it, so there is no window between "read the last hash" and "write the new line" for
/// another writer to land in.
///
/// # Errors
/// [`StoreError::UnsupportedPlatform`] off Windows: see the [module documentation](self)'s
/// concurrency section.
#[cfg(windows)]
fn open_write_locked(path: &Path) -> Result<File, StoreError> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    options.share_mode(FILE_SHARE_READ);
    // Open the name itself, never what it points at: see the module docs' `# Path trust` section.
    options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    let file = options.open(path).map_err(|e| classify_open_err("open", path, e))?;
    refuse_if_reparse_point(&file, path)?;
    Ok(file)
}

/// Off Windows, refuses outright rather than opening the file with no exclusion at all: see the
/// [module documentation](self)'s concurrency section.
#[cfg(not(windows))]
fn open_write_locked(_path: &Path) -> Result<File, StoreError> {
    Err(StoreError::UnsupportedPlatform { operation: "Store::append" })
}

/// Open an exclusive writer handle for [`Store::rotate_to`], additionally sharing delete access so
/// the rename it performs while still holding this handle does not fail against itself. See
/// [`FILE_SHARE_DELETE`]'s docs for why plain [`open_write_locked`] cannot be reused here.
///
/// # Errors
/// [`StoreError::UnsupportedPlatform`] off Windows.
#[cfg(windows)]
fn open_exclusive_for_rotate(path: &Path) -> Result<File, StoreError> {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    options.share_mode(FILE_SHARE_READ | FILE_SHARE_DELETE);
    // Open the name itself, never what it points at: see the module docs' `# Path trust` section.
    options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    let file = options.open(path).map_err(|e| classify_open_err("open", path, e))?;
    refuse_if_reparse_point(&file, path)?;
    Ok(file)
}

/// Off Windows, refuses outright: see [`open_write_locked`]'s non-Windows counterpart.
#[cfg(not(windows))]
fn open_exclusive_for_rotate(_path: &Path) -> Result<File, StoreError> {
    Err(StoreError::UnsupportedPlatform { operation: "Store::rotate_to" })
}

/// Open the store file for reading only. `Ok(None)` means the file does not exist yet -- the
/// natural state of a machine profile that has never recorded a measurement -- which is not an
/// error.
///
/// Shares read, write, *and* delete access: see the [module documentation](self)'s concurrency
/// section for why a reader must tolerate a concurrent writer, not merely other readers, on
/// Windows, and why it must also tolerate a concurrent rename. This grants the reader no write or
/// delete capability of its own; it only declares that it will not object to one being open at the
/// same time (and may then observe a torn tail, which is handled).
fn open_read_locked(path: &Path) -> Result<Option<File>, StoreError> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        options.share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE);
        // Open the name itself, never what it points at: see the module docs' `# Path trust`
        // section. Off Windows this crate has no reparse-point defense at all (every write path
        // already refuses outright there), so there is nothing to additionally check on the read
        // path either.
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    match options.open(path) {
        Ok(f) => {
            #[cfg(windows)]
            refuse_if_reparse_point(&f, path)?;
            Ok(Some(f))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(classify_open_err("open", path, e)),
    }
}

/// Why a torn tail could not be accepted as a record.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TornReason {
    /// The segment's bytes did not parse as a valid stored line, or the line exceeded
    /// [`MAX_RECORD_BYTES`] before a terminator was found.
    Unparseable {
        /// Description of the parse or size failure.
        detail: String,
    },
}

/// A partial record found at the very end of the file, excluded from the loaded records.
///
/// See the [module documentation](self) for exactly which final-line shapes this is, and are not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TornTail {
    /// Zero-based position the torn record would have occupied.
    pub index: usize,
    /// Number of bytes the partial line's content held (excluding any trailing newline).
    /// A lower bound when the line was rejected for being oversized, since reading stops as soon
    /// as the limit is crossed.
    pub bytes: usize,
    /// Why this segment could not be accepted.
    pub reason: TornReason,
}

/// Result of a verified load.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadOutcome {
    /// Every record that passed size, shape, schema, semantic, and hash-chain verification, in
    /// file order.
    pub records: Vec<Record>,
    /// The excluded torn tail, if the file's last line could not be accepted as a record.
    pub torn_tail: Option<TornTail>,
}

/// Result of one [`Store::append`] or [`Store::append_all`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendOutcome {
    /// Zero-based position the new record now occupies -- or, for [`Store::append_all`], the
    /// position the *first* of the newly appended records now occupies.
    pub index: usize,
    /// The torn tail this append found and discarded before writing, if any -- so a repair is
    /// never silent.
    pub repaired_torn_tail: Option<TornTail>,
}

/// Outcome of scanning the file through an already-open handle.
#[derive(Debug)]
struct Scan {
    records: Vec<Record>,
    /// Byte offset of the end of the last accepted record's bytes (including its trailing `\n`
    /// when it has one). Where [`Store::append`] seeks before writing, and what it truncates a
    /// torn tail down to.
    clean_end: u64,
    /// Whether the last accepted record is missing its trailing newline, so [`Store::append`]
    /// must write one before its own new line.
    needs_newline_repair: bool,
    /// Link hash of the last accepted record, to chain the next append onto.
    last_hash: Option<String>,
    torn_tail: Option<TornTail>,
}

/// Scan and verify a file through an already-open handle, per the [module documentation](self)'s
/// crash-safety and tamper-evidence contract.
///
/// Bounded on two independent axes, neither of which alone bounds the other:
/// - **Per line**: streamed through a bounded [`BufReader`] rather than reading the file whole,
///   each line's buffer capped at [`MAX_RECORD_BYTES`] + 2 bytes, so a single hostile line cannot
///   allocate without bound.
/// - **Per load**: the number of decoded records kept in memory is capped at [`MAX_RECORDS`]
///   ([`StoreError::TooManyRecords`]) -- bounding a single line's *bytes* does not bound how many
///   *lines*, and therefore how many owned [`Record`]s (each with its own heap-allocated `argv`),
///   a load ends up holding all at once.
///
/// Leaves the handle's position unspecified on return; a caller that will write afterwards must
/// `seek` first.
///
/// `max_records` is a parameter purely so tests can exercise [`StoreError::TooManyRecords`]
/// against a small number instead of synthesizing (and then fully hashing and re-verifying) a
/// file with the real, 200,000-record [`MAX_RECORDS`] in it; every production call site passes
/// [`MAX_RECORDS`] itself.
///
/// # Errors
/// [`StoreError::FileTooLarge`], [`StoreError::TooManyRecords`], [`StoreError::RecordTooLarge`],
/// [`StoreError::MalformedLine`], [`StoreError::UnsupportedSchemaVersion`],
/// [`StoreError::InvalidRecord`], or [`StoreError::ChainBroken`] for the first line (other than an
/// excluded torn final line) that fails that check. A problem anywhere *before* the end is real
/// corruption or tampering, never silently skipped in favor of the records after it.
fn scan_reader(file: &mut File, path: &Path, max_records: usize) -> Result<Scan, StoreError> {
    let len = file.metadata().map_err(|e| io_err("metadata", path, &e))?.len();
    if len > MAX_FILE_BYTES {
        return Err(StoreError::FileTooLarge { bytes: len, max: MAX_FILE_BYTES });
    }
    file.seek(SeekFrom::Start(0)).map_err(|e| io_err("seek", path, &e))?;

    let mut reader = BufReader::new(file);
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let mut records = Vec::new();
    let mut expected_prev: Option<String> = None;
    let mut clean_end: u64 = 0;
    let mut needs_newline_repair = false;
    let mut torn_tail = None;
    let mut index = 0usize;

    loop {
        buf.clear();
        // Bound this read to just over the largest legal record: a hostile line with no
        // terminator can never make this allocate past that, no matter how long the file is.
        let n = (&mut reader)
            .take(MAX_RECORD_BYTES as u64 + 2)
            .read_until(b'\n', &mut buf)
            .map_err(|e| io_err("read", path, &e))?;
        if n == 0 {
            break; // True EOF: nothing pending.
        }

        let has_newline = buf.last() == Some(&b'\n');
        let content: &[u8] = if has_newline { &buf[..buf.len() - 1] } else { &buf[..] };
        // A cheap, bounded peek: `fill_buf` refills at most one more internal buffer and never
        // consumes, so this never reads unboundedly ahead just to answer "is there more file?"
        let is_last = reader.fill_buf().map_err(|e| io_err("read", path, &e))?.is_empty();

        if content.len() > MAX_RECORD_BYTES {
            if has_newline {
                // A complete, delimited line that is simply too large is always a hard failure,
                // wherever it sits in the file.
                return Err(StoreError::RecordTooLarge { index, bytes: content.len(), max: MAX_RECORD_BYTES });
            }
            if is_last {
                torn_tail = Some(TornTail {
                    index,
                    bytes: content.len(),
                    reason: TornReason::Unparseable {
                        detail: "line exceeds the maximum record size before a terminator was found".to_string(),
                    },
                });
                break;
            }
            // Hit the per-read cap without finding `\n` and without being at EOF: an oversize
            // line that keeps going. Refuse rather than read further to find out how long it is.
            return Err(StoreError::RecordTooLarge { index, bytes: content.len(), max: MAX_RECORD_BYTES });
        }

        if content.is_empty() {
            if is_last {
                // A stray trailing newline (a text editor "ensuring a trailing newline" on a file
                // whose lines already end in one) is the least-damaged tail there is -- less
                // damaged than the garbage-but-newline-terminated line already handled as a torn
                // tail below. Treating it as a hard error would strand the whole profile's history
                // behind one byte with no way back: `append` can only ever truncate a torn TAIL,
                // never repair a hard `MalformedLine`, so this has to be one.
                torn_tail = Some(TornTail {
                    index,
                    bytes: 0,
                    reason: TornReason::Unparseable { detail: "empty final line".to_string() },
                });
                break;
            }
            // A blank line with real records after it is structural corruption, not a crash
            // artifact -- a crash truncates a file, it does not insert a blank line in the middle.
            return Err(StoreError::MalformedLine { index, detail: "empty line is not a valid JSON object".to_string() });
        }

        match decode_line(content, index, expected_prev.as_deref()) {
            Ok(decoded) => {
                if records.len() == max_records {
                    // Refuse before growing the vector past the cap, not after: this crate never
                    // holds more than `max_records` decoded records at once, even transiently.
                    return Err(StoreError::TooManyRecords { path: path.display().to_string(), max: max_records });
                }
                expected_prev = Some(decoded.record_hash);
                clean_end += buf.len() as u64;
                needs_newline_repair = !has_newline;
                records.push(decoded.record);
                index += 1;
            }
            // Only a parse/size failure on the LAST line is a crash artifact. The same failure
            // anywhere earlier, or ANY other failure (chain break, bad schema, invalid value) even
            // on the last line, is real corruption or tampering and stays a hard error.
            Err(e @ StoreError::MalformedLine { .. }) if is_last => {
                torn_tail = Some(TornTail { index, bytes: content.len(), reason: TornReason::Unparseable { detail: e.to_string() } });
                break;
            }
            Err(e) => return Err(e),
        }
    }

    Ok(Scan { records, clean_end, needs_newline_repair, last_hash: expected_prev, torn_tail })
}

/// One machine profile's append-only, hash-chained measurement log.
///
/// A thin handle around a path: all state lives on disk, so a `Store` is cheap to construct and
/// carries no in-memory cache to go stale -- and cheap to `Clone` for each worker that appends to
/// the same profile.
#[derive(Debug, Clone)]
pub struct Store {
    path: PathBuf,
}

impl Store {
    /// Open a handle to the JSON Lines file at `path`. The file need not exist yet; [`Store::append`]
    /// creates it on first use.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The path this store reads and writes.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Load and verify every record in the file, per the [module documentation](self)'s
    /// crash-safety and tamper-evidence contract.
    ///
    /// # Errors
    /// See [`scan_reader`]. [`StoreError::Busy`] if another process holds the file open for
    /// writing.
    pub fn load_verified(&self) -> Result<LoadOutcome, StoreError> {
        let Some(mut file) = open_read_locked(&self.path)? else {
            return Ok(LoadOutcome { records: Vec::new(), torn_tail: None });
        };
        let scanned = scan_reader(&mut file, &self.path, MAX_RECORDS)?;
        Ok(LoadOutcome { records: scanned.records, torn_tail: scanned.torn_tail })
    }

    /// Append one record, chaining it onto the last valid record already in the file.
    ///
    /// Opens, scans, repairs, and writes through a single exclusively-held handle (see the
    /// [module documentation](self)'s concurrency section), so this call is atomic with respect
    /// to any other [`Store::append`]/[`Store::append_all`] on the same path. Repairs a torn tail
    /// left by a previous crash before writing, so the file is always left torn-tail-free and
    /// chain-consistent when this call returns `Ok`.
    ///
    /// Every append re-scans and re-verifies the whole existing file first (see [`scan_reader`]),
    /// which is the right default -- this crate never chains a new record onto a file it has not
    /// just checked -- but makes `n` single-record appends cost `O(n^2)` overall. A caller writing
    /// many records at once (a tuner sweep, one measurement per configuration) should use
    /// [`Store::append_all`] instead, which pays that scan once for the whole batch.
    ///
    /// # Errors
    /// See [`Store::append_all`] (this is `append_all(std::slice::from_ref(record))`).
    pub fn append(&self, record: &Record) -> Result<AppendOutcome, StoreError> {
        self.append_all(std::slice::from_ref(record))
    }

    /// Append several records under one open, one scan, one write, and one fsync.
    ///
    /// Verification is unchanged from [`Store::append`] -- the existing file is fully checked
    /// once -- but a sweep of many measurements no longer re-parses its entire history once per
    /// record. An empty `records` is a documented no-op: an existing file (including any pending
    /// torn tail) is left byte-for-byte untouched, and a path with no file yet is left with no file
    /// -- this call never creates one on its own.
    ///
    /// # Errors
    /// - [`StoreError::Busy`] if another process already has the file open for writing.
    /// - [`StoreError::UnsupportedPlatform`] off Windows: see the [module documentation](self)'s
    ///   concurrency section.
    /// - Any [`StoreError`] [`scan_reader`] would report for the existing file's non-tail content.
    /// - [`StoreError::MachineProfileMismatch`] if `records` (or the file's existing first record)
    ///   disagree on [`crate::record::ProvenanceKey::machine_profile_id`] -- the file is one
    ///   machine profile, never a mix.
    /// - [`StoreError::RecordTooLarge`] if a serialized record exceeds [`MAX_RECORD_BYTES`].
    /// - [`StoreError::FileWouldExceedLimit`] if committing the batch would take the file past
    ///   [`MAX_FILE_BYTES`] -- checked before a single byte is written, so the file this call
    ///   would have produced is never left half-written; call [`Store::rotate_to`] first.
    /// - [`StoreError::RecordCountWouldExceedLimit`] if committing the batch would take the file
    ///   past [`MAX_RECORDS`] -- the identical projection on the other axis a file can exceed,
    ///   checked before a single byte is written for the same reason.
    /// - [`StoreError::Unserializable`] if a record cannot be serialized to JSON.
    /// - [`StoreError::Io`] for a filesystem failure truncating, seeking, writing, or syncing.
    pub fn append_all(&self, records: &[Record]) -> Result<AppendOutcome, StoreError> {
        self.commit(records, MAX_FILE_BYTES, MAX_RECORDS)
    }

    /// Move the current log aside so appends can resume once [`MAX_FILE_BYTES`] or [`MAX_RECORDS`]
    /// has been reached (see [`StoreError::FileWouldExceedLimit`] and
    /// [`StoreError::RecordCountWouldExceedLimit`]).
    ///
    /// `archive` keeps its own independently valid, independently verifiable hash chain; the
    /// original path starts fresh on the next append, beginning a new genesis record (`prev_hash:
    /// null`). **The chain never anchors across a rotation** -- a caller wanting the full history
    /// loads both files separately; this crate does not stitch rotated files back into one logical
    /// sequence, the same documented limitation as the undetectable trailing-deletion gap in the
    /// [module documentation](self).
    ///
    /// Holds an exclusive writer handle across the rename, so a concurrent append cannot land in
    /// the gap between "decide to rotate" and "the rename completing."
    ///
    /// Never overwrites an existing archive. `std::fs::rename` is `MoveFileExW` with
    /// `MOVEFILE_REPLACE_EXISTING` on Windows -- reads like "move," behaves like "move, destroying
    /// whatever was already there" -- so the archive name is reserved with `create_new(true)`
    /// before the rename is attempted; a path that already exists is refused with
    /// [`StoreError::ArchiveExists`] rather than silently destroying a previous rotation (two
    /// rotations landing on the same timestamped name inside one second, or a caller reusing a
    /// fixed `results.jsonl.1` name).
    ///
    /// # Errors
    /// - [`StoreError::Busy`] if another process already has the file open for writing, or the
    ///   rename is blocked by a handle (this crate's own or a third party's, e.g. an antivirus
    ///   scanner or an editor) that has not shared delete access.
    /// - [`StoreError::ArchiveExists`] if `archive` already exists.
    /// - [`StoreError::UnsupportedPlatform`] off Windows.
    /// - [`StoreError::Io`] if the file does not exist or the rename otherwise fails.
    pub fn rotate_to(&self, archive: &Path) -> Result<(), StoreError> {
        // Take the exclusive writer handle on the SOURCE first: if this is refused as `Busy`, no
        // archive placeholder must be left behind to squat the name.
        let _writer = open_exclusive_for_rotate(&self.path)?;
        match OpenOptions::new().write(true).create_new(true).open(archive) {
            Ok(placeholder) => drop(placeholder),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(StoreError::ArchiveExists { path: archive.display().to_string() });
            }
            Err(e) => return Err(io_err("create archive", archive, &e)),
        }
        std::fs::rename(&self.path, archive).map_err(|e| {
            // Do not leave the reserved placeholder behind on a failed rotation.
            let _ = std::fs::remove_file(archive);
            classify_open_err("rename", &self.path, e)
        })
    }

    /// Shared implementation behind [`Store::append`]/[`Store::append_all`]. Takes the whole-file
    /// size bound and the record-count bound as parameters purely so tests can exercise
    /// [`StoreError::FileWouldExceedLimit`] and [`StoreError::RecordCountWouldExceedLimit`] without
    /// literally growing a fixture file to the real, 256 MiB / 200,000-record limits; every public
    /// entry point always passes [`MAX_FILE_BYTES`] and [`MAX_RECORDS`] themselves.
    fn commit(&self, records: &[Record], max_file_bytes: u64, max_records: usize) -> Result<AppendOutcome, StoreError> {
        if records.is_empty() {
            // A no-op has to be a no-op on a path that does not exist yet, too: `open_write_locked`
            // carries `create(true)`, so opening it unconditionally would turn "append nothing"
            // into "create a 0-byte file". Route the empty batch through the read-only open
            // instead, whose `Ok(None)` already means exactly "no file yet" -- the index-0 answer
            // this case needs, with the file left untouched.
            let Some(mut file) = open_read_locked(&self.path)? else {
                return Ok(AppendOutcome { index: 0, repaired_torn_tail: None });
            };
            let scanned = scan_reader(&mut file, &self.path, max_records)?;
            return Ok(AppendOutcome { index: scanned.records.len(), repaired_torn_tail: None });
        }

        // Sampled before `open_write_locked`'s `create(true)` runs, so this reflects whether THIS
        // call is the one bringing the file into existence.
        let created = !self.path.exists();
        if created {
            // If no file exists yet, this IS the scan a real append would see -- not a guess --
            // so validating the batch against it now, before `create(true)` can materialize
            // anything, means a batch that cannot possibly succeed (a batch-internal machine
            // mismatch, a record over the size limit, a batch alone over the record/byte caps)
            // never leaves a 0-byte file behind. This costs nothing but CPU: no file is touched.
            // If a file DOES exist, this branch is skipped entirely and creation is not in
            // question, so there is nothing to protect here -- the authoritative call below,
            // against the real scan taken under the exclusive write lock, decides the outcome
            // either way and is unaffected by whether this speculative check ran.
            let empty = Scan { records: Vec::new(), clean_end: 0, needs_newline_repair: false, last_hash: None, torn_tail: None };
            validate_and_encode(&self.path, records, &empty, max_file_bytes, max_records)?;
        }

        let mut file = open_write_locked(&self.path)?;
        let scanned = scan_reader(&mut file, &self.path, max_records)?;
        let (start_index, out) = validate_and_encode(&self.path, records, &scanned, max_file_bytes, max_records)?;

        if scanned.torn_tail.is_some() {
            // Discard exactly the excluded tail: `scan_reader` already refused to reach here if
            // anything before it was broken.
            file.set_len(scanned.clean_end).map_err(|e| io_err("truncate", &self.path, &e))?;
        }
        // `scan_reader` streamed through a `BufReader`, which may have buffered the underlying
        // file position ahead of `clean_end`; always seek explicitly rather than assuming the
        // cursor is where the last accepted byte left off.
        file.seek(SeekFrom::Start(scanned.clean_end)).map_err(|e| io_err("seek", &self.path, &e))?;
        // A single write call, per the module documentation's crash-safety contract.
        file.write_all(&out).map_err(|e| io_err("write", &self.path, &e))?;
        file.sync_all().map_err(|e| io_err("fsync", &self.path, &e))?;

        Ok(AppendOutcome { index: start_index, repaired_torn_tail: scanned.torn_tail })
    }
}

/// Validate `records` against an already-scanned state and encode the batch to the exact bytes
/// [`Store::commit`] would write, returning the index the batch starts at alongside those bytes.
///
/// Called twice on a path with no file yet: once, speculatively, against a synthetic empty
/// [`Scan`] (which is not an approximation in that case -- it is exactly what a real scan of a
/// nonexistent file would report), before [`open_write_locked`] can create anything; and once,
/// authoritatively, against the real [`Scan`] taken under the exclusive write lock, which is what
/// actually decides whether anything is written. On a path that already has a file, only the
/// second call happens. `records` must not be empty -- [`Store::commit`] handles that case itself,
/// before this function or [`open_write_locked`] is ever reached.
///
/// # Errors
/// - [`StoreError::MachineProfileMismatch`] if `records` (or the established first record in
///   `scanned`) disagree on machine profile.
/// - [`StoreError::RecordCountWouldExceedLimit`] if the batch would take the record count past
///   `max_records`.
/// - [`StoreError::RecordTooLarge`] if any single encoded record exceeds [`MAX_RECORD_BYTES`].
/// - [`StoreError::FileWouldExceedLimit`] if the batch would take the file past `max_file_bytes`.
/// - [`StoreError::Unserializable`] if a record cannot be serialized to JSON.
fn validate_and_encode(
    path: &Path,
    records: &[Record],
    scanned: &Scan,
    max_file_bytes: u64,
    max_records: usize,
) -> Result<(usize, Vec<u8>), StoreError> {
    // The file is "one per machine profile" (see the crate's top-level docs): refuse a batch that
    // disagrees with what is already on disk, or that mixes machines within itself, rather than
    // silently merging two profiles into one chain that would go on to verify cleanly (see
    // `docs/ARCHITECTURE.md` section 6).
    let established = scanned.records.first().map(|r| r.provenance().machine_profile_id());
    let batch_machine = established.unwrap_or_else(|| records[0].provenance().machine_profile_id());
    if let Some(mismatch) = records.iter().find(|r| r.provenance().machine_profile_id() != batch_machine) {
        return Err(StoreError::MachineProfileMismatch {
            path: path.display().to_string(),
            expected: batch_machine.to_string(),
            found: mismatch.provenance().machine_profile_id().to_string(),
        });
    }

    // Same invariant as `FileWouldExceedLimit`, projected on the axis bytes do not bound:
    // `MAX_FILE_BYTES` leaves room for well over `MAX_RECORDS` minimal records, so without this an
    // append can produce a file `load_verified` would then refuse to read at all. Checked before
    // any byte of the batch is written. `saturating_add` rather than `+`: a hostile or merely huge
    // `records` slice must never wrap this check into passing.
    let projected_records = scanned.records.len().saturating_add(records.len());
    if projected_records > max_records {
        return Err(StoreError::RecordCountWouldExceedLimit { path: path.display().to_string(), records: projected_records, max: max_records });
    }

    let start_index = scanned.records.len();
    let mut prev = scanned.last_hash.clone();
    let mut out = Vec::new();
    if scanned.needs_newline_repair {
        // The previous record on disk never got its trailing `\n` (a crash, or a hand-trimmed
        // file); give it one now rather than gluing this line onto the end of that one.
        out.push(b'\n');
    }
    for (offset, record) in records.iter().enumerate() {
        let (line, hash) = encode_line(prev.as_deref(), record)?;
        if line.len() > MAX_RECORD_BYTES {
            return Err(StoreError::RecordTooLarge { index: start_index + offset, bytes: line.len(), max: MAX_RECORD_BYTES });
        }
        out.extend_from_slice(&line);
        out.push(b'\n');
        prev = Some(hash);
    }

    // Never write a file this crate would later refuse to read (`scan_reader` enforces the same
    // `max_file_bytes` on load): that would strand the rest of the profile's history behind a
    // limit with no way back. Checked before any of `out` reaches disk.
    let projected = scanned.clean_end + out.len() as u64;
    if projected > max_file_bytes {
        return Err(StoreError::FileWouldExceedLimit { path: path.display().to_string(), bytes: projected, max: max_file_bytes });
    }

    Ok((start_index, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{MeasurementSummary, MeasurementTriple, ProvenanceKey, SamplerProfile, SCHEMA_VERSION};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A unique temp file path per test invocation, so parallel test runs never collide.
    fn temp_path(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("cerebro_store_test_{tag}_{}_{n}.jsonl", std::process::id()))
    }

    /// Removes the temp file on drop, best-effort, so repeated local test runs do not accumulate
    /// files in the OS temp directory.
    struct TempFile(PathBuf);
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn summary(median: f64) -> MeasurementSummary {
        MeasurementSummary::new(3, median, median - 1.0, median + 1.0, 1.0, median - 0.5, median + 0.5).unwrap()
    }

    /// The machine profile every fixture in this module shares, unless a test is specifically
    /// about machine identity.
    fn machine() -> String {
        "c".repeat(64)
    }

    fn record(model_hash: &str, backend_digest: &str, decode_median: f64, quality_passed: Option<bool>) -> Record {
        record_on(&machine(), model_hash, backend_digest, decode_median, quality_passed)
    }

    fn record_on(machine_hash: &str, model_hash: &str, backend_digest: &str, decode_median: f64, quality_passed: Option<bool>) -> Record {
        let provenance =
            ProvenanceKey::new(machine_hash, model_hash, backend_digest, Some(10751), Some("3466812d1".into()), None, None, Some(7113))
                .unwrap();
        let sampler = SamplerProfile::new(0.7, 40, 0.9, 0.05).unwrap();
        let triple = MeasurementTriple::new(summary(40.0), summary(3.0), summary(decode_median), 128).unwrap();
        let quality_gate = quality_passed
            .map(|passed| {
                crate::record::QualityGateOutcome::new(crate::record::QualityGateKind::Nmse, if passed { 1e-7 } else { 1e-2 }, 3.3e-3)
            })
            .transpose()
            .unwrap();
        Record::new(provenance, sampler, vec!["llama-server".into(), "-m".into(), "model.gguf".into()], triple, None, quality_gate, 1_726_500_000_000)
            .unwrap()
    }

    #[test]
    fn round_trip_preserves_the_record() {
        let path = temp_path("round_trip");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);

        let r = record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true));
        store.append(&r).unwrap();

        let loaded = store.load_verified().unwrap();
        assert_eq!(loaded.records, vec![r]);
        assert!(loaded.torn_tail.is_none());
    }

    #[test]
    fn empty_file_loads_as_no_records() {
        let path = temp_path("empty");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        // Never appended to, and never created: the natural state of a fresh machine profile.
        let loaded = store.load_verified().unwrap();
        assert!(loaded.records.is_empty());
        assert!(loaded.torn_tail.is_none());
    }

    #[test]
    fn chain_links_successive_records() {
        let path = temp_path("chain_links");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);

        for i in 0..4 {
            store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0 + i as f64, Some(true))).unwrap();
        }

        let loaded = store.load_verified().unwrap();
        assert_eq!(loaded.records.len(), 4);

        // Verify the chain by hand from the raw bytes, independent of `scan_reader`'s own
        // bookkeeping: each line's prev_hash must equal the previous line's record_hash.
        let bytes = std::fs::read(&path).unwrap();
        let lines: Vec<&[u8]> = bytes.split_inclusive(|&b| b == b'\n').collect();
        let mut expected_prev: Option<String> = None;
        for line in &lines {
            let content = &line[..line.len() - 1];
            let stored: serde_json::Value = serde_json::from_slice(content).unwrap();
            assert_eq!(stored["prev_hash"].as_str().map(str::to_string), expected_prev);
            expected_prev = Some(stored["record_hash"].as_str().unwrap().to_string());
        }
    }

    #[test]
    fn deleting_a_middle_line_breaks_the_chain_at_that_index() {
        let path = temp_path("delete_middle");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        for i in 0..3 {
            store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0 + i as f64, Some(true))).unwrap();
        }

        let bytes = std::fs::read(&path).unwrap();
        let lines: Vec<&[u8]> = bytes.split_inclusive(|&b| b == b'\n').collect();
        assert_eq!(lines.len(), 3);
        let mut without_middle = Vec::new();
        without_middle.extend_from_slice(lines[0]);
        without_middle.extend_from_slice(lines[2]);
        std::fs::write(&path, &without_middle).unwrap();

        let err = store.load_verified().unwrap_err();
        assert!(matches!(err, StoreError::ChainBroken { index: 1, .. }), "got {err:?}");
    }

    #[test]
    fn editing_the_last_record_is_detected() {
        let path = temp_path("edit_last");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        for i in 0..3 {
            store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0 + i as f64, Some(true))).unwrap();
        }

        // Flip a digit inside the LAST record's timestamp. Nothing hashes "the whole file", so
        // the old (link-only) chain design would miss this entirely; `record_hash` covering the
        // record's own bytes catches it.
        let mut bytes = std::fs::read(&path).unwrap();
        let needle = b"1726500000000";
        let pos = bytes.windows(needle.len()).rposition(|w| w == needle).expect("timestamp must appear literally");
        bytes[pos] = b'9';
        std::fs::write(&path, &bytes).unwrap();

        let err = store.load_verified().unwrap_err();
        assert!(matches!(err, StoreError::ChainBroken { index: 2, .. }), "got {err:?}");
    }

    #[test]
    fn deleting_trailing_records_is_documented_as_undetected() {
        // Pinning the documented limitation: the hash chain covers what came before a record, and
        // each record's own bytes, but nothing anchors "how many records SHOULD be here" -- so
        // truncating whole records off the end leaves a chain that verifies cleanly.
        let path = temp_path("delete_trailing");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        for i in 0..3 {
            store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0 + i as f64, Some(true))).unwrap();
        }
        let bytes = std::fs::read(&path).unwrap();
        let lines: Vec<&[u8]> = bytes.split_inclusive(|&b| b == b'\n').collect();
        assert_eq!(lines.len(), 3);
        let mut without_last = Vec::new();
        without_last.extend_from_slice(lines[0]);
        without_last.extend_from_slice(lines[1]);
        std::fs::write(&path, &without_last).unwrap();

        let loaded = store.load_verified().unwrap();
        assert_eq!(loaded.records.len(), 2, "the prefix must still load cleanly; this is the documented gap, not a bug");
    }

    #[test]
    fn torn_tail_is_excluded_and_earlier_records_remain_valid() {
        let path = temp_path("torn");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        for i in 0..3 {
            store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0 + i as f64, Some(true))).unwrap();
        }

        // Simulate a crash mid-write of what would have been a 4th record: chop off the trailing
        // newline and a run of bytes before it.
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 99.0, Some(true))).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let truncated_len = bytes.len() - 12;
        bytes.truncate(truncated_len);
        std::fs::write(&path, &bytes).unwrap();
        assert_ne!(bytes.last(), Some(&b'\n'), "the torn tail must not end in a newline");

        let loaded = store.load_verified().unwrap();
        assert_eq!(loaded.records.len(), 3, "the three earlier, fully-written records must still load");
        let torn = loaded.torn_tail.expect("the truncated 4th record must be reported as a torn tail");
        assert_eq!(torn.index, 3);
        assert!(torn.bytes > 0);
    }

    #[test]
    fn torn_tail_that_is_a_complete_valid_record_is_kept_not_truncated() {
        let path = temp_path("kept_no_newline");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        for i in 0..3 {
            store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0 + i as f64, Some(true))).unwrap();
        }

        // Strip ONLY the final newline -- not the fix's job to guess, but a fully intact, fully
        // verifying record must never be thrown away just because an editor or a hand-trim
        // dropped its trailing byte.
        let mut bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.pop(), Some(b'\n'));
        std::fs::write(&path, &bytes).unwrap();

        let loaded = store.load_verified().unwrap();
        assert_eq!(loaded.records.len(), 3, "a complete, verifying final record must be kept even without a trailing newline");
        assert!(loaded.torn_tail.is_none());

        // A further append must still work and leave a clean, 4-record file.
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 44.0, Some(true))).unwrap();
        let loaded = store.load_verified().unwrap();
        assert_eq!(loaded.records.len(), 4);
        assert_eq!(loaded.records[3].measurement().tg_tok_s_deepest().median(), 44.0);
    }

    #[test]
    fn newline_terminated_garbage_final_line_is_torn_not_fatal() {
        let path = temp_path("garbage_tail");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        for i in 0..2 {
            store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0 + i as f64, Some(true))).unwrap();
        }
        // Simulate NTFS valid-data-length zero-fill: a final line that DOES end in `\n` but whose
        // content is garbage, not merely truncated.
        let mut file = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"{\"prev\x00\x00\x00\n").unwrap();
        drop(file);

        let loaded = store.load_verified().unwrap();
        assert_eq!(loaded.records.len(), 2);
        let torn = loaded.torn_tail.expect("garbage final line must be reported as torn, not a hard error");
        assert_eq!(torn.index, 2);

        // The store must still be writable afterwards.
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 50.0, Some(true))).unwrap();
        assert_eq!(store.load_verified().unwrap().records.len(), 3);
    }

    #[test]
    fn append_reports_the_torn_tail_it_repaired() {
        let path = temp_path("repair_reports");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 41.0, Some(true))).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let truncated_len = bytes.len() - 12;
        bytes.truncate(truncated_len);
        std::fs::write(&path, &bytes).unwrap();

        let outcome = store.append(&record(&"a".repeat(64), &"b".repeat(64), 42.0, Some(true))).unwrap();
        let torn = outcome.repaired_torn_tail.expect("append must report the tail it repaired");
        assert_eq!(torn.index, 1);
        assert!(torn.bytes > 0);
        assert_eq!(outcome.index, 1, "the repaired append lands at the position the torn record would have occupied");
    }

    #[test]
    fn append_repairs_a_torn_tail_before_writing() {
        let path = temp_path("repair");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();

        store.append(&record(&"a".repeat(64), &"b".repeat(64), 41.0, Some(true))).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let truncated_len = bytes.len() - 12;
        bytes.truncate(truncated_len);
        std::fs::write(&path, &bytes).unwrap();

        store.append(&record(&"a".repeat(64), &"b".repeat(64), 42.0, Some(true))).unwrap();

        let loaded = store.load_verified().unwrap();
        assert!(loaded.torn_tail.is_none(), "repaired append must leave no torn tail behind");
        assert_eq!(loaded.records.len(), 2);
        assert_eq!(loaded.records[1].measurement().tg_tok_s_deepest().median(), 42.0);
    }

    #[test]
    fn stray_trailing_newline_is_a_torn_tail_and_append_repairs_it() {
        // What a text editor "ensuring a trailing newline" produces on a file whose lines already
        // end in one: a single extra `\n` and nothing else. This is the LEAST damaged corruption
        // this crate can see -- strictly less than the garbage-but-terminated line already handled
        // as a torn tail -- so it must be recoverable exactly the same way, not a permanent brick.
        let path = temp_path("stray_newline");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        for i in 0..3 {
            store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0 + i as f64, Some(true))).unwrap();
        }
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.push(b'\n');
        std::fs::write(&path, &bytes).unwrap();

        let loaded = store.load_verified().unwrap();
        assert_eq!(loaded.records.len(), 3, "the three real records must still load");
        let torn = loaded.torn_tail.expect("a stray trailing newline must be reported as a torn tail, not silently dropped");
        assert_eq!(torn.index, 3);
        assert_eq!(torn.bytes, 0, "the stray line held no content at all");

        // And, unlike a hard error, this must be recoverable through the normal append path.
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 44.0, Some(true))).unwrap();
        let reloaded = store.load_verified().unwrap();
        assert_eq!(reloaded.records.len(), 4);
        assert!(reloaded.torn_tail.is_none());
    }

    #[test]
    fn crlf_converted_file_still_loads_and_appends() {
        // Pins the corrected module documentation: CRLF conversion is a documented LIMITATION, not
        // a guarantee this crate detects it. If this test ever starts failing because CRLF became
        // detected, the module docs' Not-detected list must be updated in the same change.
        let path = temp_path("crlf");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        for i in 0..3 {
            store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0 + i as f64, Some(true))).unwrap();
        }
        let bytes = std::fs::read(&path).unwrap();
        let mut crlf = Vec::new();
        for &b in &bytes {
            if b == b'\n' {
                crlf.push(b'\r');
            }
            crlf.push(b);
        }
        std::fs::write(&path, &crlf).unwrap();

        let loaded = store.load_verified().unwrap();
        assert_eq!(loaded.records.len(), 3);
        assert!(loaded.torn_tail.is_none());

        store.append(&record(&"a".repeat(64), &"b".repeat(64), 44.0, Some(true))).unwrap();
        assert_eq!(store.load_verified().unwrap().records.len(), 4);
    }

    #[test]
    fn duplicated_line_breaks_the_chain() {
        let path = temp_path("duplicated");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        for i in 0..3 {
            store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0 + i as f64, Some(true))).unwrap();
        }
        let bytes = std::fs::read(&path).unwrap();
        let lines: Vec<&[u8]> = bytes.split_inclusive(|&b| b == b'\n').collect();
        assert_eq!(lines.len(), 3);
        // Duplicate line 0: line 1 (originally chained onto the real line 0) now sees the same
        // `prev_hash` twice in a row instead of chaining onto what is really line 0's successor.
        let mut duplicated = Vec::new();
        duplicated.extend_from_slice(lines[0]);
        duplicated.extend_from_slice(lines[0]);
        duplicated.extend_from_slice(lines[1]);
        duplicated.extend_from_slice(lines[2]);
        std::fs::write(&path, &duplicated).unwrap();

        let err = store.load_verified().unwrap_err();
        assert!(matches!(err, StoreError::ChainBroken { index: 1, .. }), "got {err:?}");
    }

    #[test]
    fn reordered_lines_break_the_chain() {
        let path = temp_path("reordered");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        for i in 0..3 {
            store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0 + i as f64, Some(true))).unwrap();
        }
        let bytes = std::fs::read(&path).unwrap();
        let lines: Vec<&[u8]> = bytes.split_inclusive(|&b| b == b'\n').collect();
        assert_eq!(lines.len(), 3);
        // Swap lines 0 and 1: line 0's own bytes are untouched (its self-hash still verifies), but
        // nothing established a `prev_hash: null` genesis anymore, so the chain link fails at the
        // first line.
        let mut reordered = Vec::new();
        reordered.extend_from_slice(lines[1]);
        reordered.extend_from_slice(lines[0]);
        reordered.extend_from_slice(lines[2]);
        std::fs::write(&path, &reordered).unwrap();

        let err = store.load_verified().unwrap_err();
        assert!(matches!(err, StoreError::ChainBroken { index: 0, .. }), "got {err:?}");
    }

    #[test]
    fn concurrent_appender_is_refused_with_busy() {
        let path = temp_path("busy");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();
        let before = std::fs::read(&path).unwrap();

        // Hold a second writer handle open on the same path, exactly as `Store::append` would.
        let _held = open_write_locked(&path).unwrap();
        let err = store.append(&record(&"a".repeat(64), &"b".repeat(64), 41.0, Some(true))).unwrap_err();
        assert!(matches!(err, StoreError::Busy { .. }), "got {err:?}");

        let after = std::fs::read(&path).unwrap();
        assert_eq!(before, after, "a refused append must not touch the file");
    }

    #[test]
    fn a_reader_can_load_while_a_writer_holds_the_file() {
        // Regression for the share-mode bug: `open_read_locked` used to request only
        // `FILE_SHARE_READ`, and Windows checks share compatibility in BOTH directions, so this
        // failed with `StoreError::Busy` (raw OS error 32) even though nothing was actually
        // writing yet -- confirmed with a standalone probe before this fix.
        let path = temp_path("reader_during_write");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();

        let _held = open_write_locked(&path).unwrap();
        let loaded = store.load_verified().unwrap();
        assert_eq!(loaded.records.len(), 1);
    }

    #[test]
    fn an_append_succeeds_while_a_reader_holds_the_file() {
        // The direction that actually costs a measurement in production: a results-UI process
        // scanning the file must not block the tuner's append.
        let path = temp_path("append_during_read");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();

        let _reader = open_read_locked(&path).unwrap().unwrap();
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 41.0, Some(true))).unwrap();

        drop(_reader);
        assert_eq!(store.load_verified().unwrap().records.len(), 2);
    }

    #[test]
    fn record_over_the_size_limit_is_rejected_on_append() {
        let path = temp_path("too_large");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);

        let huge_argv = vec!["x".repeat(MAX_RECORD_BYTES + 1)];
        let provenance = ProvenanceKey::new("c".repeat(64), "a".repeat(64), "b".repeat(64), None, None, None, None, None).unwrap();
        let sampler = SamplerProfile::new(0.7, 40, 0.9, 0.05).unwrap();
        let triple = MeasurementTriple::new(summary(40.0), summary(3.0), summary(2.5), 128).unwrap();
        let r = Record::new(provenance, sampler, huge_argv, triple, None, None, 0).unwrap();

        let err = store.append(&r).unwrap_err();
        assert!(matches!(err, StoreError::RecordTooLarge { .. }), "got {err:?}");
        // Nothing should have been written at all -- not even a 0-byte file created and left
        // behind by `open_write_locked`'s `create(true)` before the size check ran.
        assert!(!path.exists());
    }

    #[test]
    fn oversize_line_on_load_is_record_too_large_without_reading_it_all() {
        let path = temp_path("oversize_load");
        let _guard = TempFile(path.clone());
        let mut line = vec![b'x'; MAX_RECORD_BYTES + 10];
        line.push(b'\n');
        std::fs::write(&path, &line).unwrap();

        let store = Store::new(&path);
        let err = store.load_verified().unwrap_err();
        match err {
            StoreError::RecordTooLarge { index, bytes, .. } => {
                assert_eq!(index, 0);
                assert!(bytes > MAX_RECORD_BYTES);
            }
            other => panic!("expected RecordTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn oversize_torn_tail_is_bounded() {
        let path = temp_path("oversize_torn");
        let _guard = TempFile(path.clone());
        // A final line with NO newline, well over the limit: must not panic, must not hang, and
        // must be reported (either as a torn tail or as a hard RecordTooLarge), never silently
        // accepted or read in full. Bounded reading means this crate cannot always tell "this
        // huge line is genuinely the last one" from "it just hasn't found EOF within the read
        // cap yet" -- when it cannot tell, it refuses (RecordTooLarge) rather than guessing, and
        // refusing unrecoverable leading corruption rather than auto-repairing it is correct: only
        // a torn TAIL (this crate's crash-artifact case) is something `append` may discard.
        let line = vec![b'x'; 2 * 1024 * 1024];
        std::fs::write(&path, &line).unwrap();

        let store = Store::new(&path);
        match store.load_verified() {
            Ok(outcome) => {
                let torn = outcome.torn_tail.expect("an unparseable oversize tail must be reported, not silently accepted");
                assert_eq!(torn.index, 0);
                // Only in this branch is the file crash-artifact-shaped, so only here must a
                // fresh append succeed and repair it.
                store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();
            }
            Err(StoreError::RecordTooLarge { index: 0, .. }) => {}
            Err(other) => panic!("expected a torn tail or RecordTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn file_over_the_whole_file_limit_is_rejected() {
        let path = temp_path("file_too_large");
        let _guard = TempFile(path.clone());
        // A sparse file: logically huge, but allocates almost no real disk space.
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MAX_FILE_BYTES + 1).unwrap();
        drop(file);

        let store = Store::new(&path);
        let err = store.load_verified().unwrap_err();
        assert!(matches!(err, StoreError::FileTooLarge { .. }), "got {err:?}");
    }

    /// Build one raw stored line by hand, with a correctly-computed `record_hash`, from a
    /// `record` JSON value the test controls directly -- used to construct lines a validating
    /// `Record::new`/`Record::from_wire` call could never produce, so scanning's typed handling of
    /// each failure mode can be exercised directly.
    fn raw_line(prev_hash: Option<&str>, record_value: &serde_json::Value) -> Vec<u8> {
        let raw_record = serde_json::to_vec(record_value).unwrap();
        let hash = crate::line::link_hash(prev_hash, &raw_record);
        let mut out = Vec::new();
        out.extend_from_slice(b"{\"prev_hash\":");
        match prev_hash {
            // Every prev_hash used in these tests is plain lowercase hex, so no JSON escaping
            // is needed (matching `crate::line::encode_line`'s own assumption).
            Some(h) => {
                out.push(b'"');
                out.extend_from_slice(h.as_bytes());
                out.push(b'"');
            }
            None => out.extend_from_slice(b"null"),
        }
        out.extend_from_slice(b",\"record\":");
        out.extend_from_slice(&raw_record);
        out.extend_from_slice(b",\"record_hash\":\"");
        out.extend_from_slice(hash.as_bytes());
        out.extend_from_slice(b"\"}\n");
        out
    }

    fn valid_record_value() -> serde_json::Value {
        serde_json::json!({
            "schema_version": SCHEMA_VERSION,
            "provenance": {
                "machine_profile_id": "c".repeat(64),
                "model_sha256": "a".repeat(64),
                "backend_build_number": null,
                "backend_commit": null,
                "backend_compiler": null,
                "backend_cpu_variant": null,
                "backend_manifest_digest": "b".repeat(64),
                "free_vram_mib_at_start": null,
            },
            "sampler": {"temp": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.05},
            "argv": ["llama-server"],
            "measurement": {
                "pp_tok_s": {"kept": 3, "median": 40.0, "min": 39.0, "max": 41.0, "iqr": 1.0, "ci95_low": 39.5, "ci95_high": 40.5},
                "tg_tok_s_depth0": {"kept": 3, "median": 3.0, "min": 2.0, "max": 4.0, "iqr": 1.0, "ci95_low": 2.5, "ci95_high": 3.5},
                "tg_tok_s_deepest": {"kept": 3, "median": 2.5, "min": 2.0, "max": 3.0, "iqr": 1.0, "ci95_low": 2.2, "ci95_high": 2.8},
                "deepest_depth": 128,
            },
            "soak": null,
            "quality_gate": null,
            "recorded_at_unix_ms": 0,
        })
    }

    #[test]
    fn unsupported_schema_version_is_a_typed_error() {
        let path = temp_path("schema_version");
        let _guard = TempFile(path.clone());
        let mut value = valid_record_value();
        value["schema_version"] = serde_json::json!(SCHEMA_VERSION + 1);
        std::fs::write(&path, raw_line(None, &value)).unwrap();

        let store = Store::new(&path);
        let err = store.load_verified().unwrap_err();
        assert!(
            matches!(err, StoreError::UnsupportedSchemaVersion { index: 0, found, supported: SCHEMA_VERSION } if found == SCHEMA_VERSION + 1),
            "got {err:?}"
        );
    }

    #[test]
    fn schema_version_1_record_is_rejected_as_unsupported_not_malformed() {
        // A genuine old-shape record: schema_version 1 predates `machine_profile_id`, so its
        // `provenance` object structurally cannot satisfy today's strict `RecordWire` at all. The
        // permissive schema-version peek in `line::decode_line` must catch this BEFORE the strict
        // parse runs, so the result is `UnsupportedSchemaVersion` (the version field earning its
        // place), never a generic `MalformedLine` that looks like ordinary corruption.
        let path = temp_path("schema_v1");
        let _guard = TempFile(path.clone());
        let v1_record = serde_json::json!({
            "schema_version": 1,
            "provenance": {
                "model_sha256": "a".repeat(64),
                "backend_build_number": null,
                "backend_commit": null,
                "backend_compiler": null,
                "backend_cpu_variant": null,
                "backend_manifest_digest": "b".repeat(64),
                "free_vram_mib_at_start": null,
            },
            "sampler": {"temp": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.05},
            "argv": ["llama-server"],
            "measurement": {
                "pp_tok_s": {"kept": 3, "median": 40.0, "min": 39.0, "max": 41.0, "iqr": 1.0, "ci95_low": 39.5, "ci95_high": 40.5},
                "tg_tok_s_depth0": {"kept": 3, "median": 3.0, "min": 2.0, "max": 4.0, "iqr": 1.0, "ci95_low": 2.5, "ci95_high": 3.5},
                "tg_tok_s_deepest": {"kept": 3, "median": 2.5, "min": 2.0, "max": 3.0, "iqr": 1.0, "ci95_low": 2.2, "ci95_high": 2.8},
                "deepest_depth": 128,
            },
            "soak": null,
            "quality_gate": null,
            "recorded_at_unix_ms": 0,
        });
        std::fs::write(&path, raw_line(None, &v1_record)).unwrap();

        let store = Store::new(&path);
        let err = store.load_verified().unwrap_err();
        assert!(matches!(err, StoreError::UnsupportedSchemaVersion { index: 0, found: 1, supported: SCHEMA_VERSION }), "got {err:?}");
    }

    #[test]
    fn hostile_record_with_invalid_fields_is_rejected_on_load() {
        type Mutator = fn(&mut serde_json::Value);
        let cases: [(Mutator, &str); 4] = [
            (|v: &mut serde_json::Value| v["provenance"]["model_sha256"] = serde_json::json!("xyz"), "bad model hash"),
            (|v: &mut serde_json::Value| v["measurement"]["deepest_depth"] = serde_json::json!(0), "zero deepest_depth"),
            (|v: &mut serde_json::Value| v["sampler"]["top_p"] = serde_json::json!(7.0), "top_p out of range"),
            (|v: &mut serde_json::Value| v["argv"] = serde_json::json!([]), "empty argv"),
        ];
        for (mutate, description) in cases {
            let path = temp_path("hostile");
            let _guard = TempFile(path.clone());
            let mut value = valid_record_value();
            mutate(&mut value);
            std::fs::write(&path, raw_line(None, &value)).unwrap();

            let store = Store::new(&path);
            let err = store.load_verified().unwrap_err();
            assert!(matches!(err, StoreError::InvalidRecord { index: 0, .. }), "{description}: got {err:?}");
        }
    }

    #[test]
    fn unknown_field_inside_record_is_rejected() {
        let path = temp_path("unknown_field");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        // A real, valid first record, so the hostile line below is NOT the file's last line --
        // otherwise a structural parse failure at the true end of the file is (correctly) a torn
        // tail rather than a hard error, and this test wants to exercise the hard-error path.
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();

        let first_record_hash = store.load_verified().unwrap().records.len(); // sanity: exactly 1 so far
        assert_eq!(first_record_hash, 1);
        let prev_hash = {
            let bytes = std::fs::read(&path).unwrap();
            let content = &bytes[..bytes.len() - 1];
            let stored: serde_json::Value = serde_json::from_slice(content).unwrap();
            stored["record_hash"].as_str().unwrap().to_string()
        };

        let mut value = valid_record_value();
        value["evil"] = serde_json::json!(true);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(&raw_line(Some(&prev_hash), &value));
        bytes.extend_from_slice(b"trailing content so the hostile line above is not EOF\n");
        std::fs::write(&path, &bytes).unwrap();

        let err = store.load_verified().unwrap_err();
        assert!(matches!(err, StoreError::MalformedLine { index: 1, .. }), "got {err:?}");
    }

    #[test]
    fn argv_with_newline_and_quotes_round_trips_on_one_line() {
        let path = temp_path("argv_escaping");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);

        let provenance = ProvenanceKey::new("c".repeat(64), "a".repeat(64), "b".repeat(64), None, None, None, None, None).unwrap();
        let sampler = SamplerProfile::new(0.7, 40, 0.9, 0.05).unwrap();
        let triple = MeasurementTriple::new(summary(40.0), summary(3.0), summary(2.5), 128).unwrap();
        let argv = vec!["a\nb".to_string(), "\"q\"".to_string(), "\u{65e5}\u{672c}".to_string()];
        let r = Record::new(provenance, sampler, argv.clone(), triple, None, None, 0).unwrap();
        store.append(&r).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.iter().filter(|&&b| b == b'\n').count(), 1, "the whole record must stay on one physical line");

        let loaded = store.load_verified().unwrap();
        assert_eq!(loaded.records[0].measured_argv_untrusted(), argv.as_slice());
    }

    /// A small, local, deterministic PRNG (splitmix64), matching the convention
    /// `cerebro_bench::stats::Rng` uses.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_f64_unit(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn next_hex64(&mut self) -> String {
            (0..64).map(|_| char::from_digit((self.next_u64() % 16) as u32, 16).unwrap()).collect()
        }
        fn next_below(&mut self, bound: usize) -> usize {
            (self.next_u64() % bound as u64) as usize
        }
    }

    /// Like [`record`], but with a caller-chosen timestamp -- used so the property test's 200
    /// appended records are all distinguishable by `recorded_at_unix_ms` alone.
    fn record_at(model_hash: &str, backend_digest: &str, decode_median: f64, passed: bool, recorded_at_unix_ms: u64) -> Record {
        let provenance =
            ProvenanceKey::new(machine(), model_hash, backend_digest, Some(10751), Some("3466812d1".into()), None, None, Some(7113))
                .unwrap();
        let sampler = SamplerProfile::new(0.7, 40, 0.9, 0.05).unwrap();
        let triple = MeasurementTriple::new(summary(40.0), summary(3.0), summary(decode_median), 128).unwrap();
        let quality_gate = crate::record::QualityGateOutcome::new(
            crate::record::QualityGateKind::Nmse,
            if passed { 1e-7 } else { 1e-2 },
            3.3e-3,
        )
        .unwrap();
        Record::new(
            provenance,
            sampler,
            vec!["llama-server".into(), "-m".into(), "model.gguf".into()],
            triple,
            None,
            Some(quality_gate),
            recorded_at_unix_ms,
        )
        .unwrap()
    }

    #[test]
    fn property_many_random_appends_round_trip_and_verify() {
        let path = temp_path("property");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        let mut rng = Rng::new(0xC0FFEE);

        let mut expected = Vec::new();
        for i in 0..200u64 {
            let model_hash = rng.next_hex64();
            let backend_digest = rng.next_hex64();
            let decode_median = 1.0 + rng.next_f64_unit() * 100.0;
            let passed = rng.next_u64() % 2 == 0;
            let r = record_at(&model_hash, &backend_digest, decode_median, passed, i);
            store.append(&r).unwrap();
            expected.push(r);
        }

        let loaded = store.load_verified().unwrap();
        assert!(loaded.torn_tail.is_none());
        assert_eq!(loaded.records.len(), expected.len());
        assert_eq!(loaded.records, expected, "every appended record must round-trip in order, unchanged");
    }

    #[test]
    fn property_random_truncation_is_classified_exactly() {
        let mut rng = Rng::new(0xDEC0DE);
        for case in 0..50 {
            let path = temp_path(&format!("prop_trunc_{case}"));
            let _guard = TempFile(path.clone());
            let store = Store::new(&path);
            let n = 1 + rng.next_below(20);
            for i in 0..n {
                store.append(&record(&"a".repeat(64), &"b".repeat(64), 1.0 + i as f64, Some(true))).unwrap();
            }
            let bytes = std::fs::read(&path).unwrap();
            if bytes.is_empty() {
                continue;
            }
            let ends: Vec<usize> = bytes.iter().enumerate().filter(|(_, &b)| b == b'\n').map(|(i, _)| i + 1).collect();
            let cut = rng.next_below(bytes.len());
            std::fs::write(&path, &bytes[..cut]).unwrap();

            let intact = ends.iter().filter(|&&e| e <= cut).count();
            let store = Store::new(&path);
            let outcome = store.load_verified().unwrap();
            assert_eq!(outcome.records.len(), intact, "cut={cut} of {}", bytes.len());
            // A cut exactly on a line boundary (or at 0) leaves nothing torn; any other cut point
            // leaves a partial final line that must be reported (whether excluded as a torn tail
            // or, for a record ending mid-write past its `\n`... not reachable here since we only
            // ever cut, never insert bytes after the last real newline).
            let cut_is_clean = cut == 0 || ends.contains(&cut);
            assert_eq!(outcome.torn_tail.is_some(), !cut_is_clean, "cut={cut}");
        }
    }

    #[test]
    fn property_random_byte_flip_in_non_final_line_is_hard_error() {
        let mut rng = Rng::new(0xBADC0DE);
        for case in 0..50 {
            let path = temp_path(&format!("prop_flip_{case}"));
            let _guard = TempFile(path.clone());
            let store = Store::new(&path);
            let n = 2 + rng.next_below(19); // at least 2 lines, so a non-final line exists
            for i in 0..n {
                store.append(&record(&"a".repeat(64), &"b".repeat(64), 1.0 + i as f64, Some(true))).unwrap();
            }
            let mut bytes = std::fs::read(&path).unwrap();
            let ends: Vec<usize> = bytes.iter().enumerate().filter(|(_, &b)| b == b'\n').map(|(i, _)| i + 1).collect();
            let flip_line = rng.next_below(n - 1); // never the last line
            let start = if flip_line == 0 { 0 } else { ends[flip_line - 1] };
            let end = ends[flip_line];
            let pos = start + rng.next_below(end - start - 1); // stay off the newline itself
            bytes[pos] ^= 0xFF;
            std::fs::write(&path, &bytes).unwrap();

            let store = Store::new(&path);
            let err = store.load_verified().unwrap_err();
            // A byte flip inside a non-final line's JSON is either a structural break (malformed
            // UTF-8/JSON) or a semantic one (chain/hash mismatch downstream); either is a hard
            // error, and it must never be misreported as a torn tail this far from EOF.
            assert!(
                matches!(err, StoreError::ChainBroken { .. } | StoreError::MalformedLine { .. } | StoreError::InvalidRecord { .. }),
                "case={case} got {err:?}"
            );
        }
    }

    #[test]
    fn append_refuses_to_push_the_file_past_the_load_limit() {
        // Reproducing this against the REAL 256 MiB `MAX_FILE_BYTES` would need either ~870,000
        // minimal records (which would trip `MAX_RECORDS` first) or ~256 near-`MAX_RECORD_BYTES`
        // records (multiple minutes of hashing); neither is a reasonable test. `commit` takes the
        // bound as a parameter for exactly this reason: this test exercises the real formula
        // (`clean_end + encoded_batch_len > max`) and the real byte-identity guarantee, just
        // against a bound the test controls, while every public entry point always passes the
        // real `MAX_FILE_BYTES`.
        let path = temp_path("would_exceed");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();
        let before = std::fs::read(&path).unwrap();

        // The file is already exactly at this (fake, test-only) limit: no further byte may land.
        let tiny_limit = before.len() as u64;
        let r = record(&"a".repeat(64), &"b".repeat(64), 41.0, Some(true));
        let err = store.commit(std::slice::from_ref(&r), tiny_limit, MAX_RECORDS).unwrap_err();
        assert!(matches!(err, StoreError::FileWouldExceedLimit { .. }), "got {err:?}");

        let after = std::fs::read(&path).unwrap();
        assert_eq!(before, after, "a refused commit must not touch the file");
    }

    #[test]
    fn rotate_then_append_starts_a_fresh_chain() {
        let path = temp_path("rotate_src");
        let archive = temp_path("rotate_archive");
        let _guard = TempFile(path.clone());
        let _archive_guard = TempFile(archive.clone());
        let store = Store::new(&path);
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 41.0, Some(true))).unwrap();

        store.rotate_to(&archive).unwrap();
        assert!(!path.exists(), "the original path must be gone after rotation");

        // The archive keeps its own complete, independently verifiable chain.
        let archived = Store::new(&archive).load_verified().unwrap();
        assert_eq!(archived.records.len(), 2);

        // Appending recreates the original path and starts a fresh genesis record -- the chain
        // never anchors across a rotation.
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 42.0, Some(true))).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let first_line_end = bytes.iter().position(|&b| b == b'\n').unwrap();
        let stored: serde_json::Value = serde_json::from_slice(&bytes[..first_line_end]).unwrap();
        assert!(stored["prev_hash"].is_null(), "a rotated-away file must start a fresh genesis record");
        assert_eq!(store.load_verified().unwrap().records.len(), 1);
    }

    #[test]
    fn rotate_to_refuses_an_existing_archive() {
        // `std::fs::rename` is `MoveFileExW` + `MOVEFILE_REPLACE_EXISTING` on Windows: without this
        // check, rotating onto an occupied name silently destroys whatever archive was there.
        let path = temp_path("rotate_no_overwrite_src");
        let archive = temp_path("rotate_no_overwrite_archive");
        let _guard = TempFile(path.clone());
        let _archive_guard = TempFile(archive.clone());
        let store = Store::new(&path);
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();

        let prior_archive_bytes = b"a previous archive's bytes, not this crate's format at all".to_vec();
        std::fs::write(&archive, &prior_archive_bytes).unwrap();

        let err = store.rotate_to(&archive).unwrap_err();
        assert!(matches!(err, StoreError::ArchiveExists { .. }), "got {err:?}");

        assert_eq!(std::fs::read(&archive).unwrap(), prior_archive_bytes, "the existing archive must be untouched");
        assert!(path.exists(), "a refused rotation must leave the source file in place");
        assert_eq!(store.load_verified().unwrap().records.len(), 1, "the source's records must still load");
    }

    #[test]
    fn load_verified_does_not_block_rotate_to() {
        // Regression for the dropped `FILE_SHARE_DELETE` on the reader: a `load_verified` in
        // flight must not block this crate's own `rotate_to`.
        let path = temp_path("rotate_not_blocked_by_reader");
        let archive = temp_path("rotate_not_blocked_by_reader_archive");
        let _guard = TempFile(path.clone());
        let _archive_guard = TempFile(archive.clone());
        let store = Store::new(&path);
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();

        let _reader = open_read_locked(&path).unwrap().unwrap();
        store.rotate_to(&archive).unwrap();

        drop(_reader);
        assert_eq!(Store::new(&archive).load_verified().unwrap().records.len(), 1);
    }

    #[test]
    fn rotate_to_reports_busy_when_the_rename_is_blocked() {
        // A handle that permits concurrent read+write (so `rotate_to`'s own exclusive-writer open
        // succeeds) but withholds delete sharing -- a third party this crate does not control, e.g.
        // an antivirus scanner or an editor -- must surface as `Busy`, not a generic `Io`.
        let path = temp_path("rotate_busy");
        let archive = temp_path("rotate_busy_archive");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();

        #[cfg(windows)]
        {
            let mut options = OpenOptions::new();
            options.read(true).write(true);
            options.share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE); // no FILE_SHARE_DELETE
            let _blocker = options.open(&path).unwrap();

            let err = store.rotate_to(&archive).unwrap_err();
            assert!(matches!(err, StoreError::Busy { .. }), "got {err:?}");
            assert!(!archive.exists(), "a busy rotation must not leave a placeholder archive behind");
        }
    }

    #[test]
    fn load_refuses_a_file_with_more_than_max_records() {
        // Reproducing this against the real 200,000-record `MAX_RECORDS` would mean writing,
        // hashing, and fully re-verifying 200,001 records -- correct, but far too slow for a unit
        // test (minutes, dominated by re-hashing on load). `scan_reader` takes the cap as a
        // parameter for exactly this reason: this test drives the real enforcement loop directly
        // against a small cap, while every production call site (`load_verified`, `commit`)
        // always passes the real `MAX_RECORDS`.
        let path = temp_path("too_many_records");
        let _guard = TempFile(path.clone());
        let small_cap = 3usize;
        let r = record(&"a".repeat(64), &"b".repeat(64), 1.0, Some(true));
        let mut bytes = Vec::new();
        let mut prev: Option<String> = None;
        for _ in 0..=small_cap {
            let (line, hash) = encode_line(prev.as_deref(), &r).unwrap();
            bytes.extend_from_slice(&line);
            bytes.push(b'\n');
            prev = Some(hash);
        }
        std::fs::write(&path, &bytes).unwrap();

        let mut file = std::fs::File::open(&path).unwrap();
        let err = scan_reader(&mut file, &path, small_cap).unwrap_err();
        assert!(matches!(err, StoreError::TooManyRecords { max, .. } if max == small_cap), "got {err:?}");
    }

    #[test]
    fn append_refuses_to_push_the_file_past_the_record_limit() {
        // Mirrors `append_refuses_to_push_the_file_past_the_load_limit`, but for the axis bytes do
        // not bound: `commit` must refuse a batch that would take the file's record count past the
        // cap, exactly as it already refuses one that would take its byte count past its own cap.
        // Driven against a small cap for the same reason that test drives a small byte limit --
        // reproducing this against the real 200,000-record `MAX_RECORDS` would take minutes.
        let path = temp_path("would_exceed_records");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        let small_cap = 3usize;
        for i in 0..small_cap {
            store.append(&record(&"a".repeat(64), &"b".repeat(64), 1.0 + i as f64, Some(true))).unwrap();
        }
        let before = std::fs::read(&path).unwrap();

        let r = record(&"a".repeat(64), &"b".repeat(64), 99.0, Some(true));
        let err = store.commit(std::slice::from_ref(&r), MAX_FILE_BYTES, small_cap).unwrap_err();
        assert!(
            matches!(err, StoreError::RecordCountWouldExceedLimit { records, max, .. } if records == small_cap + 1 && max == small_cap),
            "got {err:?}"
        );

        let after = std::fs::read(&path).unwrap();
        assert_eq!(before, after, "a refused commit must not touch the file");

        // The real entry point must still pass the real, large `MAX_RECORDS` -- confirmed by the
        // same three records committing without issue through the public API.
        store.append(&r).unwrap();
        assert_eq!(store.load_verified().unwrap().records.len(), small_cap + 1);
    }

    #[test]
    fn append_all_is_chain_equivalent_to_repeated_append() {
        let path_a = temp_path("append_all_a");
        let path_b = temp_path("append_all_b");
        let _guard_a = TempFile(path_a.clone());
        let _guard_b = TempFile(path_b.clone());

        let records: Vec<Record> = (0..20).map(|i| record(&"a".repeat(64), &"b".repeat(64), 1.0 + i as f64, Some(true))).collect();

        let store_a = Store::new(&path_a);
        store_a.append_all(&records).unwrap();

        let store_b = Store::new(&path_b);
        for r in &records {
            store_b.append(r).unwrap();
        }

        let bytes_a = std::fs::read(&path_a).unwrap();
        let bytes_b = std::fs::read(&path_b).unwrap();
        assert_eq!(bytes_a, bytes_b, "append_all and N calls to append must produce byte-identical files");

        let loaded_a = store_a.load_verified().unwrap();
        assert_eq!(loaded_a.records, records);
    }

    #[test]
    fn append_all_with_an_empty_slice_leaves_the_file_untouched() {
        let path = temp_path("append_all_empty");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();

        // Also cover the torn-tail case: an empty batch must not repair a pending torn tail --
        // that is not "no new record justifies a rewrite", it is a real, if small, rewrite.
        let mut bytes = std::fs::read(&path).unwrap();
        let truncated_len = bytes.len() - 12;
        bytes.truncate(truncated_len);
        std::fs::write(&path, &bytes).unwrap();
        let before = std::fs::read(&path).unwrap();

        let outcome = store.append_all(&[]).unwrap();
        assert!(outcome.repaired_torn_tail.is_none(), "an empty batch must not report a repair it never performed");

        let after = std::fs::read(&path).unwrap();
        assert_eq!(before, after, "an empty batch must leave the file, including a pending torn tail, byte-for-byte untouched");
    }

    #[test]
    fn append_refuses_a_record_from_another_machine_profile() {
        let path = temp_path("machine_mismatch");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap();
        let before = std::fs::read(&path).unwrap();

        let other_machine_record = record_on(&"9".repeat(64), &"a".repeat(64), &"b".repeat(64), 41.0, Some(true));
        let err = store.append(&other_machine_record).unwrap_err();
        assert!(matches!(err, StoreError::MachineProfileMismatch { .. }), "got {err:?}");

        let after = std::fs::read(&path).unwrap();
        assert_eq!(before, after, "a refused append must not touch the file");
    }

    #[test]
    fn append_all_refuses_a_batch_that_mixes_machine_profiles() {
        // Even on a brand-new file with no established machine yet, a batch that disagrees with
        // ITSELF must be refused rather than silently adopting its first record's machine.
        let path = temp_path("machine_mismatch_batch");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        let a = record_on(&"c".repeat(64), &"a".repeat(64), &"b".repeat(64), 40.0, Some(true));
        let b = record_on(&"9".repeat(64), &"a".repeat(64), &"b".repeat(64), 41.0, Some(true));

        let err = store.append_all(&[a, b]).unwrap_err();
        assert!(matches!(err, StoreError::MachineProfileMismatch { .. }), "got {err:?}");
        assert!(!path.exists(), "nothing should have been written, not even a 0-byte file");
    }

    #[test]
    fn empty_batch_does_not_create_a_missing_file() {
        let path = temp_path("empty_batch_no_create");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);

        let outcome = store.append_all(&[]).unwrap();
        assert_eq!(outcome, AppendOutcome { index: 0, repaired_torn_tail: None });
        assert!(!path.exists(), "an empty batch must never bring a file into existence");
    }

    #[test]
    fn a_refused_first_append_leaves_no_file_behind() {
        // The tightened form of `append_all_refuses_a_batch_that_mixes_machine_profiles`'s
        // assertion, pinned as its own test: a brand-new path plus a batch that trips
        // `MachineProfileMismatch` must leave no file at all, not a 0-byte one.
        let path = temp_path("refused_first_append_no_file");
        let _guard = TempFile(path.clone());
        let store = Store::new(&path);
        let a = record_on(&"c".repeat(64), &"a".repeat(64), &"b".repeat(64), 40.0, Some(true));
        let b = record_on(&"9".repeat(64), &"a".repeat(64), &"b".repeat(64), 41.0, Some(true));

        let err = store.append_all(&[a, b]).unwrap_err();
        assert!(matches!(err, StoreError::MachineProfileMismatch { .. }), "got {err:?}");
        assert!(!path.exists());
    }

    #[test]
    fn store_is_debug_and_clone() {
        // A caller (a tuner holding one `Store` per machine profile) needs to put one in a
        // `#[derive(Debug)]` struct of its own and hand a copy to each worker.
        fn assert_debug_and_clone<T: std::fmt::Debug + Clone>() {}
        assert_debug_and_clone::<Store>();
        let store = Store::new(temp_path("debug_clone"));
        let cloned = store.clone();
        assert_eq!(store.path(), cloned.path());
        let _ = format!("{store:?}");
    }

    #[test]
    #[cfg(windows)]
    #[ignore = "creating a Windows file symlink needs SeCreateSymbolicLinkPrivilege (Developer \
                Mode or an elevated shell); this sandboxed run does not have it, so run this \
                explicitly (cargo test -p cerebro-store -- --ignored) on a machine that does"]
    fn append_refuses_a_reparse_point_at_the_store_path() {
        // ARCHITECTURE.md section 8 adversary 3: a hostile local principal who can create a name
        // in the results directory plants a reparse point at the store path. This must be refused
        // rather than transparently followed into the decoy file.
        let path = temp_path("reparse_append");
        let decoy = temp_path("reparse_append_decoy");
        let _guard = TempFile(path.clone());
        let _decoy_guard = TempFile(decoy.clone());
        let decoy_bytes = b"a decoy file's bytes, which this crate must never touch".to_vec();
        std::fs::write(&decoy, &decoy_bytes).unwrap();
        std::os::windows::fs::symlink_file(&decoy, &path).expect("test setup: create the symlink");

        let store = Store::new(&path);
        let err = store.append(&record(&"a".repeat(64), &"b".repeat(64), 40.0, Some(true))).unwrap_err();
        assert!(matches!(err, StoreError::PathIsReparsePoint { .. }), "got {err:?}");
        // The load-bearing assertion: a redirected write would have silently corrupted the decoy.
        assert_eq!(std::fs::read(&decoy).unwrap(), decoy_bytes, "the decoy file must be untouched");
    }

    #[test]
    #[cfg(windows)]
    #[ignore = "creating a Windows file symlink needs SeCreateSymbolicLinkPrivilege (Developer \
                Mode or an elevated shell); this sandboxed run does not have it, so run this \
                explicitly (cargo test -p cerebro-store -- --ignored) on a machine that does"]
    fn load_refuses_a_reparse_point_at_the_store_path() {
        // Pairs with `append_refuses_a_reparse_point_at_the_store_path` so the guard cannot be
        // added to the write opener and forgotten on the read one.
        let path = temp_path("reparse_load");
        let decoy = temp_path("reparse_load_decoy");
        let _guard = TempFile(path.clone());
        let _decoy_guard = TempFile(decoy.clone());
        std::fs::write(&decoy, b"").unwrap();
        std::os::windows::fs::symlink_file(&decoy, &path).expect("test setup: create the symlink");

        let store = Store::new(&path);
        let err = store.load_verified().unwrap_err();
        assert!(matches!(err, StoreError::PathIsReparsePoint { .. }), "got {err:?}");
    }

}
