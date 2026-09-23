//! Typed failures for building result records and for the append-only store.
//!
//! Split into two enums for the same reason `cerebro-flags` splits `ValueError` from `EmitError`:
//! [`RecordError`] is a construction-time refusal to represent an invalid value at all (a bad
//! hash, an out-of-range sampler parameter), while [`StoreError`] is a failure of the durable log
//! itself (I/O, size limits, a broken hash chain, or a syntactically valid record that failed
//! semantic validation on the way in from disk). Keeping them separate means a caller can tell,
//! from the type alone, whether a failure means "the data you gave me is wrong" or "the file on
//! disk is wrong".

use thiserror::Error;

/// A value that cannot be represented in a [`crate::record::Record`] because it is not a value
/// the caller could plausibly have measured.
///
/// `docs/ARCHITECTURE.md` section 6 rule: refuse rather than guess. These checks exist so that a
/// record which passed construction is already safe to hash, chain, and query. Every field of
/// every type in [`crate::record`] is private specifically so that this is the *only* way to
/// produce one -- including a record read back from disk, which [`crate::store::Store`] always
/// routes through these same constructors (see [`StoreError::InvalidRecord`]) rather than trusting
/// whatever shape a results file happens to contain.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum RecordError {
    /// A field documented as a SHA-256 hex digest is not exactly 64 lowercase hex characters.
    ///
    /// `sha2::Sha256`'s `{:x}` formatting always produces lowercase hex, so a value in any other
    /// case did not come from this crate's own hashing and is either hand-edited or was computed
    /// by something using a different convention -- either way, not safe to compare byte-for-byte
    /// against a digest this crate produced.
    #[error("{field} must be a 64-character lowercase hex SHA-256 digest, got {value:?}")]
    InvalidHash {
        /// Name of the offending field.
        field: &'static str,
        /// The value that was rejected, truncated to 80 characters: this field echoes untrusted
        /// input (a results file is a hostile input per `docs/ARCHITECTURE.md` section 8), and an
        /// unbounded echo would let a single oversized value inflate an error message and any log
        /// that captures it.
        value: String,
    },

    /// A sampler parameter that must be finite and non-negative (`temp`) was not.
    #[error("{field} must be finite and >= 0.0, got {value}")]
    NonFiniteOrNegative {
        /// Name of the offending field.
        field: &'static str,
        /// The value that was rejected.
        value: f64,
    },

    /// A sampler parameter documented as a probability/proportion (`top_p`, `min_p`) fell outside
    /// `0.0..=1.0`, or was not finite.
    #[error("{field} must be finite and in 0.0..=1.0, got {value}")]
    OutOfUnitRange {
        /// Name of the offending field.
        field: &'static str,
        /// The value that was rejected.
        value: f64,
    },

    /// `argv` was empty.
    ///
    /// A record exists to answer "what exact command line produced this measurement"; an empty
    /// argv cannot have spawned anything and would make the record meaningless before it is even
    /// written.
    #[error("argv must not be empty; a record must name the command line it measured")]
    EmptyArgv,

    /// An `argv` element contained an embedded NUL byte.
    ///
    /// A NUL cannot appear in anything that was actually spawned -- every OS argv representation
    /// this product targets is NUL-terminated or NUL-delimited -- so its presence here means the
    /// value did not come from a real measurement and is either corrupted or hand-crafted. Other
    /// control characters (a literal newline from a multi-line `--grammar` or system-prompt value,
    /// for instance) are legitimate and are not rejected; see
    /// [`crate::record::Record::measured_argv_untrusted`] for why that is safe.
    #[error("argv[{position}] contains an embedded NUL byte, which cannot appear in a real spawned argument")]
    NulByteInArgv {
        /// Index into `argv` of the offending element.
        position: usize,
    },

    /// [`crate::record::MeasurementTriple::deepest_depth`] was zero.
    ///
    /// `docs/ARCHITECTURE.md` section 6: a "deepest" decode number measured at depth 0 is the
    /// depth-0 number again under a different name, not a real context-decay measurement.
    #[error("deepest_depth must be greater than 0; a depth-0-only run has no context-decay member")]
    ZeroDeepestDepth,

    /// [`crate::record::MeasurementSummary::kept`] was zero.
    ///
    /// A summary describes a set of kept samples; zero samples is not a measurement, it is the
    /// absence of one.
    #[error("kept must be greater than 0; a summary needs at least one retained sample")]
    ZeroKeptSamples,

    /// A [`crate::record::MeasurementSummary`]'s `min`/`ci95_low`/`median`/`ci95_high`/`max` were
    /// not in non-decreasing order.
    ///
    /// A confidence interval that does not bracket its own median, or a median outside
    /// `[min, max]`, cannot have come from a real sample set and would silently corrupt
    /// [`crate::query::best_config`]'s ranking.
    #[error(
        "measurement summary values are out of order: expected min <= ci95_low <= median <= ci95_high <= max, got min={min}, ci95_low={ci95_low}, median={median}, ci95_high={ci95_high}, max={max}"
    )]
    SummaryOutOfOrder {
        /// The rejected `min`.
        min: f64,
        /// The rejected `ci95_low`.
        ci95_low: f64,
        /// The rejected `median`.
        median: f64,
        /// The rejected `ci95_high`.
        ci95_high: f64,
        /// The rejected `max`.
        max: f64,
    },

    /// A [`crate::record::SoakOutcome`] claimed `passed: true` while
    /// `vram_trend_non_negative: false`.
    ///
    /// `docs/ARCHITECTURE.md` section 6: "a soak passes only if free VRAM has a non-negative trend
    /// over its final third." A soak that claims to have passed despite a negative VRAM trend
    /// contradicts the one pass criterion this crate knows about and cannot be represented.
    #[error("soak reports passed=true but vram_trend_non_negative=false, which docs/ARCHITECTURE.md section 6 makes a required pass criterion")]
    InconsistentSoakOutcome,
}

/// A failure of the append-only store itself, as distinct from a bad input value ([`RecordError`]).
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum StoreError {
    /// A filesystem operation failed.
    ///
    /// `std::io::Error` is not `Clone`/`Eq`, so its message is captured as text -- the same
    /// convention `cerebro-probe::ProbeError` uses for foreign OS errors.
    #[error("{operation} failed on {path}: {detail}")]
    Io {
        /// What was being attempted, e.g. `"read"`, `"open"`, `"write"`, `"fsync"`, `"truncate"`.
        operation: &'static str,
        /// The path involved.
        path: String,
        /// The underlying `std::io::Error` message.
        detail: String,
    },

    /// Another handle already has the file open for writing.
    ///
    /// [`crate::store::Store::append`] opens the file with Windows share flags that allow
    /// concurrent readers but refuse a second writer (`ERROR_SHARING_VIOLATION`), so two
    /// appenders racing (the tuner and the CLI, or two tuner workers) fail one of them cleanly
    /// instead of forking the hash chain or truncating a sibling's just-committed record.
    #[error("{path} is open for writing by another process; try again")]
    Busy {
        /// The path that could not be opened.
        path: String,
    },

    /// The file is larger than [`crate::store::MAX_FILE_BYTES`].
    ///
    /// Checked against the file's metadata *before* any of its bytes are read, so a hostile
    /// multi-gigabyte results file is rejected in one `stat` call rather than read into memory
    /// first.
    #[error("file is {bytes} bytes, exceeding the {max}-byte whole-file limit")]
    FileTooLarge {
        /// The size of the oversized file, in bytes.
        bytes: u64,
        /// The largest whole-file size this crate accepts.
        max: u64,
    },

    /// A record, once serialized, exceeds [`crate::store::MAX_RECORD_BYTES`].
    ///
    /// Bounds what a corrupt or hostile results file (or a caller building an unreasonably large
    /// `argv`/notes field) can make this crate allocate while parsing a single line. Reading is
    /// streamed and bounded per-line, so this is reported without the offending line ever being
    /// read in full.
    #[error("record at line {index} is {bytes} bytes, exceeding the {max}-byte limit")]
    RecordTooLarge {
        /// Zero-based position of the offending line in the file.
        index: usize,
        /// Size of the offending line observed so far, in bytes. May be a lower bound: reading
        /// stops as soon as the limit is crossed, before the line's true length is known.
        bytes: usize,
        /// The largest single-record size this crate accepts.
        max: usize,
    },

    /// A line is not valid JSON, or is valid JSON that does not match this crate's stored-line
    /// envelope or the nested record's shape (wrong top-level shape, a required field missing, an
    /// unrecognized field, or a field with the wrong type).
    #[error("line {index} is not a valid stored-record JSON object: {detail}")]
    MalformedLine {
        /// Zero-based position of the offending line in the file.
        index: usize,
        /// The underlying parse error message, or a description of the structural problem.
        detail: String,
    },

    /// A record's `schema_version` is not one this build of the crate understands.
    #[error("line {index} has schema_version {found}, but this build only reads schema_version {supported}")]
    UnsupportedSchemaVersion {
        /// Zero-based position of the offending line in the file.
        index: usize,
        /// The schema version the line actually carries.
        found: u32,
        /// The schema version this build supports.
        supported: u32,
    },

    /// A line's nested record is well-formed JSON but fails [`RecordError`] validation.
    ///
    /// A results file is a hostile input (`docs/ARCHITECTURE.md` section 8): scanning routes
    /// every record read back from disk through the exact same constructors a caller building one
    /// in memory would use, so a hand-edited or corrupted-but-still-parseable record (an
    /// out-of-range sampler value, a malformed hash, an empty argv) is refused here rather than
    /// silently accepted and later recommended by [`crate::query::best_config`].
    #[error("line {index} holds an invalid record: {source}")]
    InvalidRecord {
        /// Zero-based position of the offending line in the file.
        index: usize,
        /// The specific validation failure.
        #[source]
        source: RecordError,
    },

    /// A record's `prev_hash` does not match the previous record's link hash, or a record's own
    /// `record_hash` does not match the hash recomputed from its bytes.
    ///
    /// This crate does not distinguish "this record does not chain onto the one before it" from
    /// "this record's own bytes were altered after it was written" -- both surface as the same
    /// broken link, reported rather than silently skipped. Everything *before* `index` verified;
    /// [`crate::store::Store::load_verified`] returns `Err` here rather than the verified prefix,
    /// so no records at all are returned to a caller that does not specifically want partial
    /// recovery. See the [`crate::store`] module documentation for exactly what this chain does
    /// and does not prove -- in particular, deleting whole records from the *end* of the file is
    /// **not** detected by this check, since the remaining prefix is a valid chain on its own.
    #[error("hash chain broken at line {index}: expected {expected:?}, found {found:?}")]
    ChainBroken {
        /// Zero-based position of the first record whose link does not verify.
        index: usize,
        /// The hash this crate expected to find (the previous line's link hash, or the
        /// recomputed `record_hash` for a self-hash mismatch). This is always this crate's own
        /// computation, never echoed from the file, so it is not length-capped.
        expected: Option<String>,
        /// The hash the line actually claims, truncated to 80 characters: this field echoes
        /// untrusted input (a results file is a hostile input per `docs/ARCHITECTURE.md`
        /// section 8), and an unbounded echo would let a single oversized value inflate an error
        /// message and any log that captures it -- the same reasoning as
        /// [`RecordError::InvalidHash`]'s `value` field.
        found: Option<String>,
    },

    /// A [`crate::record::Record`] could not be serialized to JSON.
    ///
    /// Every field type here is either a plain scalar, a `String`, or one of this crate's own
    /// structs, all of which are validated finite at construction, so in practice this variant is
    /// unreachable through the public API. It is still reported as a typed error rather than
    /// letting `serde_json::to_vec` panic, on the chance a future field type reintroduces the
    /// possibility.
    #[error("record could not be serialized: {detail}")]
    Unserializable {
        /// The underlying `serde_json` error message.
        detail: String,
    },

    /// Appending this record would take the file past [`crate::store::MAX_FILE_BYTES`], which
    /// [`crate::store::Store::load_verified`] would then refuse to read at all.
    ///
    /// [`crate::store::Store::append`] checks the projected size *before* writing a single byte, so
    /// this crate can never create a file it would later refuse to open -- that would strand the
    /// rest of the machine profile's history behind a limit with no way back. Use
    /// [`crate::store::Store::rotate_to`] to move the current log aside and resume appending.
    #[error("appending to {path} would take it to {bytes} bytes, past the {max}-byte load limit; call Store::rotate_to first")]
    FileWouldExceedLimit {
        /// The file that would be pushed over the limit.
        path: String,
        /// The size, in bytes, the file would reach after this append.
        bytes: u64,
        /// [`crate::store::MAX_FILE_BYTES`].
        max: u64,
    },

    /// The file holds more records than [`crate::store::MAX_RECORDS`] this crate will load into
    /// memory in one [`crate::store::Store::load_verified`] call.
    ///
    /// [`crate::store::MAX_FILE_BYTES`] bounds bytes on disk; this bounds the heap those bytes
    /// expand into once each line becomes an owned [`crate::record::Record`] (argv strings and all)
    /// sitting in a `Vec` -- several times larger than the on-disk bytes, and otherwise unbounded
    /// by anything this crate advertises. Use [`crate::store::Store::rotate_to`] to move the
    /// current log aside and resume appending under a fresh, empty file.
    #[error("{path} holds more than {max} records; this crate will not load them all into memory at once")]
    TooManyRecords {
        /// The file that holds too many records.
        path: String,
        /// [`crate::store::MAX_RECORDS`].
        max: usize,
    },

    /// Appending this batch would take the file past [`crate::store::MAX_RECORDS`], which
    /// [`crate::store::Store::load_verified`] would then refuse to read at all.
    ///
    /// [`crate::store::MAX_FILE_BYTES`] already gets this projection ([`StoreError::FileWouldExceedLimit`]);
    /// this is the identical guard on the other axis a file can exceed. Checked before any byte of
    /// the batch is written, for the same reason: never produce a file this crate would later
    /// refuse to load at all. Use [`crate::store::Store::rotate_to`] to move the current log aside
    /// and resume appending.
    #[error("appending to {path} would take it to {records} records, past the {max}-record load limit; call Store::rotate_to first")]
    RecordCountWouldExceedLimit {
        /// The file that would be pushed over the limit.
        path: String,
        /// The number of records the file would hold after this append.
        records: usize,
        /// [`crate::store::MAX_RECORDS`].
        max: usize,
    },

    /// A record's [`crate::record::ProvenanceKey::machine_profile_id`] does not match the machine
    /// profile already recorded in this file's first record.
    ///
    /// The file is documented as "one per machine profile" (see the crate's top-level docs); this
    /// is the check that actually enforces it, so a results file copied between machines, restored
    /// from a backup onto different hardware, or concatenated from two profiles is refused at
    /// append time rather than silently accepted into a chain that would go on to verify cleanly.
    #[error("record is from machine profile {found}, but {path} holds machine profile {expected}")]
    MachineProfileMismatch {
        /// The path whose existing records disagree with the new one.
        path: String,
        /// The machine profile id already recorded in the file.
        expected: String,
        /// The machine profile id the new record carries.
        found: String,
    },

    /// A [`crate::store::Store::rotate_to`] archive path already exists.
    ///
    /// `std::fs::rename` is `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING` on Windows: without this
    /// check, rotating onto a path a caller reused (a fixed `results.jsonl.1` name, or two
    /// rotations landing on the same timestamped name inside one second) would silently destroy
    /// whatever archive was already there. Refusing is the only option this crate has: there is no
    /// safe merge of two independent hash chains.
    #[error("archive {path} already exists; rotation will not overwrite a previous archive")]
    ArchiveExists {
        /// The occupied archive path.
        path: String,
    },

    /// This operation has no correct implementation on the current platform and was refused rather
    /// than silently running without the guarantee it documents.
    ///
    /// Cerebro is Windows-first (`docs/ARCHITECTURE.md`); the single-writer guarantee documented on
    /// [`crate::store::Store::append`] is enforced through a Windows share-mode check, which has no
    /// portable equivalent. Rather than append with no exclusion at all -- which would silently
    /// drop the documented guarantee the moment two writers raced off Windows -- this crate follows
    /// `cerebro_probe::cpu::probe`'s convention: an admitted absence instead of a guess.
    #[error("{operation} needs a write-exclusion guarantee this crate only implements on Windows")]
    UnsupportedPlatform {
        /// The operation that was refused, e.g. `"Store::append"`.
        operation: &'static str,
    },

    /// The store path names a reparse point (a symlink or an NTFS junction), not a regular file.
    ///
    /// `docs/ARCHITECTURE.md` section 8 adversary 3 is live on the reference machine (a llama.cpp
    /// directory this crate might share a parent with is Modify-writable by
    /// `ASUS\CodexSandboxUsers`): a hostile local principal who can create a *name* in the results
    /// directory could plant a junction at the store path and silently redirect this crate's
    /// truncate-and-write into a file it never chose. Every opener in [`crate::store`] opens the
    /// final path component itself (`FILE_FLAG_OPEN_REPARSE_POINT`) and refuses when it turns out to
    /// be a reparse point, rather than transparently following it.
    ///
    /// **This closes the planted-link redirect only. It does not make a directory an attacker
    /// already has write access to safe.** The path in this error comes from the caller, not from
    /// the file's contents, so it needs no [`crate::record::cap_echo`]-style truncation the way a
    /// value read off disk would.
    #[error("{path} is a reparse point (symlink or junction); the results file must be a regular file")]
    PathIsReparsePoint {
        /// The offending path, exactly as the caller supplied it.
        path: String,
    },
}
