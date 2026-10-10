//! DPAPI storage for Cerebro's only two real secrets, plus the per-launch child key file.
//!
//! # What this protects against, and what it does not
//!
//! Every blob here is protected with `CryptProtectData` under the *current Windows user's* DPAPI
//! master key (`CRYPTPROTECT_UI_FORBIDDEN`; the `CRYPTPROTECT_LOCAL_MACHINE` flag is never passed).
//! That defends against exactly two things:
//!
//! * **another Windows user account** on the same machine trying to read the blob, and
//! * **offline theft of the file** -- a copied `.dpapi` file, a stolen disk, a backup -- because
//!   decrypting it needs this user's logon-derived master key, which does not travel with the file.
//!
//! It defends against **nothing else**, and in particular it does **not** protect against malware
//! running as this same user: anything that can execute code in this account can call
//! `CryptUnprotectData` itself, exactly as this module does. `docs/CLAIMS.md` forbids shipping
//! security theater, and `docs/ARCHITECTURE.md` §8 says plainly that "encrypting a config file with
//! a key stored beside it is theater" -- DPAPI is not that, but it is also not a defense against a
//! same-user attacker, and this paragraph is the honest statement of exactly where the line falls.
//!
//! # Why every blob also carries id-specific entropy
//!
//! DPAPI's protection is scoped to the *user*, not to the caller or to which secret a blob holds.
//! Passing no entropy at all would mean any process running as this user that also calls
//! `CryptUnprotectData` with no entropy could decrypt Cerebro's blobs, and -- more subtly -- a blob
//! written for [`SecretId::ApiBearer`] would decrypt just as happily if it were ever handed to
//! [`SecretId::HfToken`]'s slot by a bug or a poisoned file. [`entropy_for`] mixes a constant,
//! versioned prefix with the id's own stable name into the `poptionalentropy` blob on every call,
//! so a ciphertext produced for one id fails to decrypt under another's entropy -- see
//! `blob_protected_under_one_id_does_not_decrypt_as_another` below, which is the regression test
//! for exactly that property.
//!
//! # Why the on-disk ACL matters as much as the encryption
//!
//! DPAPI answers "who can decrypt this," not "who can overwrite or delete it." A principal who can
//! write inside the secrets directory could replace a `.dpapi` blob, delete it, or (if the
//! directory itself were writable by others) plant an entirely different file there. Every file and
//! directory this module creates is therefore given the SDDL `D:P(A;;FA;;;<user SID>)(A;;FA;;;SY)`
//! -- File-All-Access to the current user and to `SYSTEM` only, `P` marking the ACL protected so
//! nothing above it in the tree can inject an inherited ACE that would widen it. The audit methods
//! ([`SecretStore::audit_dir`], [`SecretStore::audit_secret`], [`KeyFile::audit`]) read that ACL
//! back through `cerebro-probe`'s single-object audit, the workspace's one DACL reader, so "could not
//! verify" is never mistaken for "verified, fine" here either.
//!
//! # KeyFile and process-kill
//!
//! [`KeyFile::create`] writes a per-launch secret to an owner-only file and deletes it on
//! [`Drop`]. That covers every *normal* exit path -- the value returned by `create`, an error
//! return, an early `?`, an explicit [`KeyFile::close`] -- because all of them run destructors.
//! It does **not** cover the process being killed (`TerminateProcess`, a crash, a forced shutdown):
//! Windows does not run `Drop` for a process that no longer exists, so a killed Cerebro can leave a
//! stale key file behind. This is stated here, plainly, rather than left to be discovered: nothing
//! in this module's public contract claims otherwise, and a caller that needs guaranteed cleanup
//! (e.g. sweeping the directory for stale `cerebro-child-key-*` files on startup) must implement
//! that itself.
//!
//! # Why this module, alone in this crate, allows `unsafe`
//!
//! See `src/lib.rs`'s crate documentation for why this crate is `#![deny(unsafe_code)]` rather than
//! `#![forbid(unsafe_code)]`. Every `unsafe` block below is a single Win32 FFI call (or the
//! smallest possible wrapper around one) carrying a `SAFETY` comment; no path reachable from a
//! `pub` function here panics, unwraps, or indexes unchecked -- a malformed system response becomes
//! a [`SecretsError`], never a crash.

#![allow(unsafe_code)]

use std::path::{Path, PathBuf};

use rand_core::CryptoRng;
use thiserror::Error;
use zeroize::Zeroize as _;

/// Which real secret a [`SecretStore`] operation names.
///
/// Exactly two variants, matching `docs/ARCHITECTURE.md` §8's "secrets are nearly nonexistent."
/// Adding a third is a decision to widen the entropy-separated secret set this module manages, not
/// a mechanical storage-layer change, and should not be made by editing this enum alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SecretId {
    /// The bearer token Cerebro's own local HTTP API requires on every route, including reads (see
    /// `docs/ARCHITECTURE.md` §8).
    ApiBearer,
    /// An optional Hugging Face access token, used only to download a gated model the user has
    /// already agreed to that model's license for.
    HfToken,
}

impl SecretId {
    /// Stable, filesystem-safe name for this id.
    ///
    /// Used both as the `.dpapi` file stem ([`SecretStore::path_for`]) and, mixed with
    /// [`ENTROPY_PREFIX`], as the DPAPI entropy that keeps one id's blob from decrypting as
    /// another's ([`entropy_for`]). Never change an existing variant's stem: doing so silently
    /// orphans every blob already written under the old name and changes what entropy a stored
    /// blob needs to decrypt.
    fn file_stem(self) -> &'static str {
        match self {
            SecretId::ApiBearer => "api-bearer",
            SecretId::HfToken => "hf-token",
        }
    }
}

/// A secret byte string that is never printed and is zeroed when dropped.
///
/// [`std::fmt::Debug`], [`std::fmt::Display`], and [`serde::Serialize`] all render the fixed string
/// `"<redacted>"` regardless of content, so a `Secret` that ends up embedded in a log line, an
/// error message, or a JSON document by accident -- a `#[derive(Debug)]` on a struct that holds
/// one, a `tracing` span, a settings document that should never have carried one in the first place
/// -- cannot leak its bytes through any of those paths. See
/// `secret_debug_display_and_serde_never_contain_plaintext` below for the test that pins this.
///
/// There is deliberately no [`serde::Deserialize`] impl: nothing in this crate ever needs to parse
/// a `Secret` back out of an untrusted document, and adding one would be an invitation to embed a
/// real secret inside `settings.json` -- whose invalid-edit path
/// ([`crate::file::strip_comments`], [`crate::error::ConfigError::Invalid`]) echoes attacker- or
/// typo-chosen *content* into error messages -- instead of routing it through [`SecretStore`].
pub struct Secret(Vec<u8>);

impl Secret {
    /// Wrap `bytes` as a secret.
    ///
    /// Takes ownership so the caller's own copy, if any, is the only copy left un-managed by this
    /// type's [`Drop`] -- callers that need to keep a second copy around are choosing to keep an
    /// unzeroed one, which this constructor cannot prevent but also does not encourage.
    pub fn new(bytes: Vec<u8>) -> Self {
        Secret(bytes)
    }

    /// The secret's raw bytes.
    ///
    /// Named `expose_secret`, not `as_bytes` or [`std::ops::Deref`], so every call site reads, at
    /// the call site, as a deliberate decision to look at the plaintext -- the same convention the
    /// `secrecy` crate established for the same reason.
    pub fn expose_secret(&self) -> &[u8] {
        &self.0
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the secret is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

impl std::fmt::Display for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

impl serde::Serialize for Secret {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str("<redacted>")
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        // `Zeroize` writes through a volatile barrier the optimizer cannot remove, unlike a plain
        // loop or `slice::fill(0)` -- see the Cargo.toml pin's rationale for why that distinction
        // is load-bearing here rather than cosmetic.
        self.0.zeroize();
    }
}

/// Anything that can go wrong storing, retrieving, or auditing a secret, or creating the
/// per-launch child key file.
///
/// No variant here ever carries secret plaintext or ciphertext bytes -- only paths, Win32 error
/// codes, API names, and [`SecretId`]s, all of which are safe to log. See
/// `no_error_path_ever_echoes_the_secret_bytes` below for the test that drives every reachable
/// variant and checks exactly that.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum SecretsError {
    /// A Win32 call failed for a reason not covered by a more specific variant below. `call` names
    /// the API so the failure is actionable without a debugger.
    #[error("{call} failed with Win32 error {code}")]
    Win32 {
        /// The API that failed, e.g. `"CreateFileW"`.
        call: &'static str,
        /// The value `GetLastError` returned immediately afterwards.
        code: u32,
    },

    /// `CryptProtectData` failed while encrypting a secret for storage.
    #[error("could not encrypt secret {id:?}: Win32 error {code}")]
    Encrypt {
        /// Which secret was being encrypted.
        id: SecretId,
        /// The value `GetLastError` returned immediately afterwards.
        code: u32,
    },

    /// `CryptUnprotectData` failed while decrypting a stored secret.
    ///
    /// Reached identically by a genuinely tampered blob and by a blob protected under a
    /// *different* [`SecretId`]'s entropy -- DPAPI reports both as the same failure, and this
    /// module does not attempt to tell them apart. Doing so would need a plaintext oracle (try
    /// decrypting, inspect the result, decide) that a decrypt-failure path must not have.
    #[error("could not decrypt secret {id:?}: Win32 error {code}")]
    Decrypt {
        /// Which secret was being decrypted.
        id: SecretId,
        /// The value `GetLastError` returned immediately afterwards.
        code: u32,
    },

    /// A file or directory this module manages could not be read, written, or deleted.
    #[error("could not access {path}: {cause}")]
    Io {
        /// The path involved.
        path: PathBuf,
        /// Underlying OS error, rendered as text so this stays `Clone` + `PartialEq`.
        cause: String,
    },

    /// DPAPI and the owner-only ACLs this module relies on have no equivalent outside Windows.
    #[error("secret storage is implemented for Windows only")]
    UnsupportedPlatform,

    /// The current user's SID, which every owner-only ACL here names, could not be read from the
    /// process token. Carries the probe's own error text.
    #[error("could not identify the current user for an owner-only ACL: {0}")]
    UserIdentity(String),
}

/// Constant, versioned prefix mixed into every entropy value.
///
/// Guards against a future change to this module's entropy scheme (or a coincidental match with
/// some *other* DPAPI consumer's entropy on this machine) silently letting one kind of blob decrypt
/// as another. See [`entropy_for`].
const ENTROPY_PREFIX: &[u8] = b"cerebro-config/secrets/v1/";

/// Application- and id-specific entropy for `id`, mixed into every `CryptProtectData`/
/// `CryptUnprotectData` call. See the module documentation's "Why every blob also carries
/// id-specific entropy" section for what this defends against.
fn entropy_for(id: SecretId) -> Vec<u8> {
    let mut entropy = ENTROPY_PREFIX.to_vec();
    entropy.extend_from_slice(id.file_stem().as_bytes());
    entropy
}

/// SDDL granting File-All-Access to the current user and to `SYSTEM` only. `P` marks the ACL
/// protected, so nothing above the object in the directory tree can inject an inherited ACE that
/// would widen this -- see the module documentation's "Why the on-disk ACL matters" section.
///
/// The user is named by SID (from [`cerebro_probe::backend::current_user_sid`]), not by the SDDL
/// alias `OW`: `OW` stores the OWNER RIGHTS SID (S-1-3-4), which
/// [`cerebro_probe::backend::audit_object_dacl`] decides on SIDs alone and does not trust, so the
/// audit of a freshly written file would report it. Naming the exact principal the audit trusts
/// keeps writer and auditor in agreement by construction.
#[cfg(windows)]
fn owner_only_sddl() -> Result<String, SecretsError> {
    let sid = cerebro_probe::backend::current_user_sid()
        .map_err(|e| SecretsError::UserIdentity(e.to_string()))?;
    Ok(format!("D:P(A;;FA;;;{sid})(A;;FA;;;SY)"))
}

/// UTF-16, NUL-terminated, suitable for any Win32 `PCWSTR`/`PWSTR` parameter this module passes a
/// path or a literal string through.
#[cfg(windows)]
fn to_wide(s: impl AsRef<std::ffi::OsStr>) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt as _;
    s.as_ref().encode_wide().chain(std::iter::once(0)).collect()
}

/// Reads the calling thread's last-error value. A thin, named wrapper purely so every call site
/// below reads as "fetch the code for the Win32 call that just failed" rather than a bare `unsafe`
/// block repeated at each site.
#[cfg(windows)]
fn last_error() -> u32 {
    // SAFETY: `GetLastError` has no preconditions; it reads thread-local state Win32 itself
    // maintains. Callers of this function are required (by convention, like every use of
    // `GetLastError` in this workspace -- see `cerebro-probe`) to call it immediately after the
    // failed call it is meant to explain, before any other API call can overwrite that state.
    unsafe { windows_sys::Win32::Foundation::GetLastError() }
}

/// An owner-only `SECURITY_ATTRIBUTES`, built once from [`owner_only_sddl`] and freed exactly once
/// on drop.
#[cfg(windows)]
struct OwnerOnlySecurityDescriptor {
    descriptor: windows_sys::Win32::Security::PSECURITY_DESCRIPTOR,
}

#[cfg(windows)]
impl OwnerOnlySecurityDescriptor {
    fn build() -> Result<Self, SecretsError> {
        use windows_sys::Win32::Security::Authorization::{
            ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
        };

        let wide = to_wide(owner_only_sddl()?);
        let mut descriptor: windows_sys::Win32::Security::PSECURITY_DESCRIPTOR = std::ptr::null_mut();
        // SAFETY: `wide` is a live, NUL-terminated UTF-16 string for the duration of this call.
        // `descriptor` starts null and receives, on success, a `PSECURITY_DESCRIPTOR` Windows
        // allocates via `LocalAlloc`; this struct becomes its sole owner and frees it with
        // `LocalFree` exactly once, in `Drop` below. The final `securitydescriptorsize` parameter
        // is documented as optional and is not needed here, so it is left null.
        let ok = unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                wide.as_ptr(),
                SDDL_REVISION_1,
                &mut descriptor,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(SecretsError::Win32 {
                call: "ConvertStringSecurityDescriptorToSecurityDescriptorW",
                code: last_error(),
            });
        }
        Ok(Self { descriptor })
    }

    fn as_security_attributes(&self) -> windows_sys::Win32::Security::SECURITY_ATTRIBUTES {
        windows_sys::Win32::Security::SECURITY_ATTRIBUTES {
            nLength: std::mem::size_of::<windows_sys::Win32::Security::SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: self.descriptor,
            bInheritHandle: 0,
        }
    }
}

#[cfg(windows)]
impl Drop for OwnerOnlySecurityDescriptor {
    fn drop(&mut self) {
        if !self.descriptor.is_null() {
            // SAFETY: `self.descriptor` was allocated by
            // `ConvertStringSecurityDescriptorToSecurityDescriptorW` via `LocalAlloc` and has not
            // been freed yet -- this is the only place that frees it, and `Drop::drop` runs at
            // most once per value, so this satisfies "freed exactly once."
            unsafe {
                windows_sys::Win32::Foundation::LocalFree(
                    self.descriptor as windows_sys::Win32::Foundation::HLOCAL,
                )
            };
        }
    }
}

/// Create `dir` with an owner-only ACL if it does not already exist.
///
/// If it already exists (from an earlier run, or created concurrently), its ACL is left exactly as
/// found: Windows only honors `SECURITY_ATTRIBUTES` at creation time, and this function does not
/// attempt to retroactively tighten a directory it did not itself create. [`SecretStore::audit_dir`]
/// exists so a caller can check rather than silently trust that case.
#[cfg(windows)]
fn create_owner_only_dir_if_missing(dir: &Path) -> Result<(), SecretsError> {
    use windows_sys::Win32::Foundation::ERROR_ALREADY_EXISTS;
    use windows_sys::Win32::Storage::FileSystem::CreateDirectoryW;

    let wide = to_wide(dir);
    let sd = OwnerOnlySecurityDescriptor::build()?;
    let attrs = sd.as_security_attributes();
    // SAFETY: `wide` is a live NUL-terminated wide path. `attrs.lpSecurityDescriptor` is a live
    // `PSECURITY_DESCRIPTOR` owned by `sd`, which outlives this call and is freed exactly once when
    // `sd` drops at the end of this function.
    let ok = unsafe { CreateDirectoryW(wide.as_ptr(), &attrs) };
    if ok != 0 {
        return Ok(());
    }
    let code = last_error();
    if code == ERROR_ALREADY_EXISTS {
        return Ok(());
    }
    Err(SecretsError::Win32 { call: "CreateDirectoryW", code })
}

/// Create `path` fresh (failing if it already exists) as an owner-only file and hand back a
/// [`std::fs::File`] open for reading and writing.
#[cfg(windows)]
fn create_owner_only_file(path: &Path) -> Result<std::fs::File, SecretsError> {
    use std::os::windows::io::FromRawHandle as _;
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, FILE_GENERIC_READ, FILE_GENERIC_WRITE, FILE_SHARE_READ,
    };

    let wide = to_wide(path);
    let sd = OwnerOnlySecurityDescriptor::build()?;
    let attrs = sd.as_security_attributes();
    // SAFETY: `wide` is a live NUL-terminated wide path; `attrs.lpSecurityDescriptor` is live for
    // the duration of this call (owned by `sd`, dropped after it). `CREATE_NEW` means this call
    // either fails (handled below) or returns a handle to a file it alone just created, so the
    // `File::from_raw_handle` wrap immediately below is the handle's sole owner from the moment it
    // exists.
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            FILE_GENERIC_READ | FILE_GENERIC_WRITE,
            FILE_SHARE_READ,
            &attrs,
            CREATE_NEW,
            FILE_ATTRIBUTE_NORMAL,
            std::ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(SecretsError::Win32 { call: "CreateFileW", code: last_error() });
    }
    // SAFETY: `handle` is not `INVALID_HANDLE_VALUE` (checked above), is freshly opened by the call
    // immediately above, and is not owned by anything else -- wrapping it hands sole ownership to
    // the returned `File`, whose own `Drop` closes it exactly once.
    Ok(unsafe { std::fs::File::from_raw_handle(handle as std::os::windows::io::RawHandle) })
}

/// Unique per-call suffix for [`write_owner_only_file`]'s temp names (process id plus this
/// counter), the same scheme as `crate::file`'s `TMP_COUNTER`.
#[cfg(windows)]
static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Replace whatever is at `path` with an owner-only file containing `bytes`, atomically.
///
/// The bytes go to a fresh owner-only temp file beside `path` (`CREATE_NEW`, so nothing planted
/// at that name can be written through), are `fsync`ed, and the temp file is then renamed over
/// `path` (`MoveFileExW` with `MOVEFILE_REPLACE_EXISTING`, same volume). A same-volume rename moves
/// the file object together with its own protected DACL, so the result carries *this* call's
/// owner-only ACL, never one an earlier write or a hostile pre-plant left at `path`. And a crash
/// or a failure at any step leaves the previous secret intact: there is no moment at which
/// neither the old nor the new blob exists, as there was with delete-then-create.
#[cfg(windows)]
fn write_owner_only_file(path: &Path, bytes: &[u8]) -> Result<(), SecretsError> {
    use std::io::Write as _;

    let io_err = |e: std::io::Error| SecretsError::Io { path: path.to_path_buf(), cause: e.to_string() };
    let unique = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.tmp-{}-{unique}", std::process::id()));

    let written = create_owner_only_file(&tmp).and_then(|mut file| {
        file.write_all(bytes).map_err(io_err)?;
        file.sync_all().map_err(io_err)
    });
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        io_err(e)
    })
}

/// Largest `.dpapi` blob [`SecretStore::get`] reads. DPAPI output for a 32-byte bearer token or a
/// Hugging Face token is well under 1 KiB; the cap only stops a planted multi-GB file from being
/// read into memory whole.
#[cfg(windows)]
const MAX_SECRET_BLOB_BYTES: u64 = 64 * 1024;

/// `FILE_FLAG_OPEN_REPARSE_POINT` (winbase.h): open a link itself, never what it points at.
#[cfg(windows)]
const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;

/// `FILE_FLAG_BACKUP_SEMANTICS` (winbase.h): required to open a directory handle at all.
#[cfg(windows)]
const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;

/// `FILE_ATTRIBUTE_REPARSE_POINT` (winnt.h).
#[cfg(windows)]
const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;

/// Read the blob at `path`: `None` when absent, a typed error for a reparse point (this module
/// never creates one, so a link here was planted) or a file over [`MAX_SECRET_BLOB_BYTES`].
#[cfg(windows)]
fn read_blob(path: &Path) -> Result<Option<Vec<u8>>, SecretsError> {
    use std::io::Read as _;
    use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};

    let io_err = |cause: String| SecretsError::Io { path: path.to_path_buf(), cause };
    let file = match std::fs::OpenOptions::new()
        .read(true)
        // `FILE_FLAG_BACKUP_SEMANTICS` so a planted *directory* junction opens too (it grants no
        // privilege by itself) and is refused by the attribute check below, like a file symlink,
        // instead of surfacing as an unexplained "access denied".
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)
    {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io_err(e.to_string())),
    };
    let meta = file.metadata().map_err(|e| io_err(e.to_string()))?;
    if meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(io_err("the secret blob is a reparse point (symlink or junction)".to_string()));
    }
    // Sized from the metadata when it is plausible, so a normal read makes one allocation; the
    // `take` bound, not the metadata, is what enforces the cap.
    let mut ciphertext = Vec::with_capacity(meta.len().min(MAX_SECRET_BLOB_BYTES) as usize);
    file.take(MAX_SECRET_BLOB_BYTES + 1)
        .read_to_end(&mut ciphertext)
        .map_err(|e| io_err(e.to_string()))?;
    if ciphertext.len() as u64 > MAX_SECRET_BLOB_BYTES {
        return Err(io_err(format!("the secret blob exceeds {MAX_SECRET_BLOB_BYTES} bytes")));
    }
    Ok(Some(ciphertext))
}

/// Owns one Windows-allocated `CRYPT_INTEGER_BLOB` and frees it with `LocalFree` exactly once, on
/// drop. When `sensitive` is set, the buffer is zeroed first -- see the module documentation's
/// honesty note: this is what keeps a decrypted secret from lingering in this process's heap (and
/// potentially in a swap file or a crash dump) any longer than the single `Vec<u8>` copy this
/// module hands back inside a [`Secret`].
#[cfg(windows)]
struct FreeBlob {
    blob: windows_sys::Win32::Security::Cryptography::CRYPT_INTEGER_BLOB,
    sensitive: bool,
}

#[cfg(windows)]
impl Drop for FreeBlob {
    fn drop(&mut self) {
        if self.blob.pbData.is_null() {
            return;
        }
        if self.sensitive && self.blob.cbData > 0 {
            // SAFETY: `pbData` is a live buffer of exactly `cbData` bytes that this struct
            // uniquely owns until the `LocalFree` call immediately below; `Zeroize` (unlike
            // `ptr::write_bytes`) writes through a volatile barrier the optimizer cannot remove,
            // which matters here because the only thing done with this buffer afterward is an
            // opaque FFI call (`LocalFree`) whose effect on the buffer's *contents* the optimizer
            // cannot reason about -- it could otherwise conclude these writes are dead.
            let slice = unsafe { std::slice::from_raw_parts_mut(self.blob.pbData, self.blob.cbData as usize) };
            slice.zeroize();
        }
        // SAFETY: `pbData` was allocated by Windows (via `LocalAlloc`, per the documented
        // `CryptProtectData`/`CryptUnprotectData` contract for their output blob) and is freed
        // here exactly once -- this is the only place that frees it, and `Drop::drop` runs at most
        // once per value.
        unsafe {
            windows_sys::Win32::Foundation::LocalFree(self.blob.pbData as windows_sys::Win32::Foundation::HLOCAL)
        };
    }
}

/// Encrypt `plaintext` for `id` via `CryptProtectData`, with `CRYPTPROTECT_UI_FORBIDDEN` (this is a
/// headless service; a credential prompt must never appear) and `id`'s own entropy.
#[cfg(windows)]
fn protect(id: SecretId, plaintext: &[u8]) -> Result<Vec<u8>, SecretsError> {
    use windows_sys::Win32::Security::Cryptography::{CryptProtectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB};

    let entropy = entropy_for(id);
    let data_in = CRYPT_INTEGER_BLOB {
        cbData: plaintext.len() as u32,
        pbData: plaintext.as_ptr() as *mut u8,
    };
    let entropy_blob = CRYPT_INTEGER_BLOB {
        cbData: entropy.len() as u32,
        pbData: entropy.as_ptr() as *mut u8,
    };
    let mut data_out = CRYPT_INTEGER_BLOB::default();

    // SAFETY: `data_in`/`entropy_blob` point at `plaintext`/`entropy`, both live for the whole
    // call. `data_out` is zero-initialized and receives, on success, a Windows-allocated output
    // blob; `szdatadescr`, `pvreserved` and `ppromptstruct` are all documented as optional and are
    // not needed here, so each is null. `CRYPTPROTECT_UI_FORBIDDEN` means a failure that would
    // otherwise show a UI prompt instead returns an error, which is required for a headless
    // service and is exactly what the typed `Err` below turns it into.
    let ok = unsafe {
        CryptProtectData(
            &data_in,
            std::ptr::null(),
            &entropy_blob,
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut data_out,
        )
    };
    if ok == 0 {
        return Err(SecretsError::Encrypt { id, code: last_error() });
    }
    // This is the *encrypted* output, not plaintext, so `FreeBlob` need not zero it before freeing.
    let guard = FreeBlob { blob: data_out, sensitive: false };
    // SAFETY: `guard.blob.pbData` is non-null (the call above succeeded) and `guard.blob.cbData`
    // bytes were written by Windows; copied out here, before `guard`'s `Drop` frees the buffer.
    let ciphertext = unsafe { std::slice::from_raw_parts(guard.blob.pbData, guard.blob.cbData as usize) }.to_vec();
    Ok(ciphertext)
}

/// Decrypt `ciphertext` for `id` via `CryptUnprotectData`, with the same entropy [`protect`] used
/// to encrypt it.
#[cfg(windows)]
fn unprotect(id: SecretId, ciphertext: &[u8]) -> Result<Vec<u8>, SecretsError> {
    use windows_sys::Win32::Security::Cryptography::{CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB};

    let entropy = entropy_for(id);
    let data_in = CRYPT_INTEGER_BLOB {
        cbData: ciphertext.len() as u32,
        pbData: ciphertext.as_ptr() as *mut u8,
    };
    let entropy_blob = CRYPT_INTEGER_BLOB {
        cbData: entropy.len() as u32,
        pbData: entropy.as_ptr() as *mut u8,
    };
    let mut data_out = CRYPT_INTEGER_BLOB::default();

    // SAFETY: as `protect`'s call above. `ppszdatadescr` is null, which the API documents as legal
    // when the caller does not want the optional description string it would otherwise allocate
    // and hand back (a second buffer this function would then also have to free).
    let ok = unsafe {
        CryptUnprotectData(
            &data_in,
            std::ptr::null_mut(),
            &entropy_blob,
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut data_out,
        )
    };
    if ok == 0 {
        // Reached identically by a tampered blob and by the wrong id's entropy -- see
        // `SecretsError::Decrypt`'s own doc comment for why this module does not try to tell them
        // apart.
        return Err(SecretsError::Decrypt { id, code: last_error() });
    }
    // This is the secret plaintext DPAPI just handed back, so `FreeBlob` zeroes it before freeing
    // -- once it has been copied into the `Vec<u8>` below, that copy (inside the `Secret` a caller
    // constructs from it) is the only one left, and it is zeroed in turn by `Secret::drop`.
    let guard = FreeBlob { blob: data_out, sensitive: true };
    // SAFETY: `guard.blob.pbData` is non-null (the call above succeeded) and `guard.blob.cbData`
    // bytes were written by Windows; copied out here, before `guard`'s `Drop` zeroes and frees the
    // buffer.
    let plaintext = unsafe { std::slice::from_raw_parts(guard.blob.pbData, guard.blob.cbData as usize) }.to_vec();
    Ok(plaintext)
}

/// DPAPI storage for [`SecretId::ApiBearer`] and [`SecretId::HfToken`].
///
/// See the module documentation for what DPAPI does and does not protect against, and for why the
/// on-disk ACL is audited independently of the encryption.
pub struct SecretStore {
    dir: PathBuf,
}

impl SecretStore {
    /// Open (creating if absent) the `secrets` subdirectory of `dir`.
    #[cfg(windows)]
    pub fn open(dir: &Path) -> Result<Self, SecretsError> {
        let secrets_dir = dir.join("secrets");
        create_owner_only_dir_if_missing(&secrets_dir)?;
        Ok(Self { dir: secrets_dir })
    }

    /// Always [`SecretsError::UnsupportedPlatform`] off Windows -- DPAPI has no portable
    /// equivalent, and a substitute that merely obfuscated would contradict the module's own
    /// honesty requirement.
    #[cfg(not(windows))]
    pub fn open(_dir: &Path) -> Result<Self, SecretsError> {
        Err(SecretsError::UnsupportedPlatform)
    }

    /// Path of one id's `.dpapi` blob inside this store's directory.
    fn path_for(&self, id: SecretId) -> PathBuf {
        self.dir.join(format!("{}.dpapi", id.file_stem()))
    }

    /// Encrypt and store `secret` under `id`, replacing whatever was there.
    #[cfg(windows)]
    pub fn set(&self, id: SecretId, secret: &Secret) -> Result<(), SecretsError> {
        let ciphertext = protect(id, secret.expose_secret())?;
        write_owner_only_file(&self.path_for(id), &ciphertext)
    }

    /// Read and decrypt the secret stored under `id`, or `None` if nothing has been set yet.
    ///
    /// # Errors
    /// [`SecretsError::Decrypt`] if the stored blob does not decrypt under `id`'s entropy --
    /// either it was tampered with, or it was written under a *different* [`SecretId`]'s entropy.
    /// See the module documentation's entropy-separation section. [`SecretsError::Io`] if the blob
    /// cannot be read, is a reparse point, or is larger than any real blob (64 KiB).
    #[cfg(windows)]
    pub fn get(&self, id: SecretId) -> Result<Option<Secret>, SecretsError> {
        let Some(ciphertext) = read_blob(&self.path_for(id))? else {
            return Ok(None);
        };
        let plaintext = unprotect(id, &ciphertext)?;
        Ok(Some(Secret::new(plaintext)))
    }

    /// Remove the stored secret for `id`, if any. Not having one is not an error.
    #[cfg(windows)]
    pub fn delete(&self, id: SecretId) -> Result<(), SecretsError> {
        let path = self.path_for(id);
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(SecretsError::Io { path, cause: e.to_string() }),
        }
    }

    /// Generate a fresh 256-bit API bearer token from `rng` and store it, replacing any previous
    /// one.
    ///
    /// `rng` is supplied by the caller rather than instantiated here. A [`CryptoRng`] is the
    /// *caller's* assertion that their generator is cryptographically secure; this function has no
    /// way to check that claim, so it does not weaken it by reaching for some other, unaudited
    /// source of randomness (a static, a thread-local, `std::process::id()`-seeded state) on its
    /// own -- see the Cargo.toml pin's rationale for `rand_core` for why this crate depends on the
    /// trait alone and carries no RNG implementation, and no `static`, of its own.
    #[cfg(windows)]
    pub fn rotate_api_bearer(&self, rng: &mut impl CryptoRng) -> Result<Secret, SecretsError> {
        let mut bytes = vec![0u8; 32]; // 256 bits
        rng.fill_bytes(&mut bytes);
        let secret = Secret::new(bytes);
        self.set(SecretId::ApiBearer, &secret)?;
        Ok(secret)
    }

    /// Always [`SecretsError::UnsupportedPlatform`] off Windows.
    #[cfg(not(windows))]
    pub fn set(&self, _id: SecretId, _secret: &Secret) -> Result<(), SecretsError> {
        Err(SecretsError::UnsupportedPlatform)
    }

    /// Always [`SecretsError::UnsupportedPlatform`] off Windows.
    #[cfg(not(windows))]
    pub fn get(&self, _id: SecretId) -> Result<Option<Secret>, SecretsError> {
        Err(SecretsError::UnsupportedPlatform)
    }

    /// Always [`SecretsError::UnsupportedPlatform`] off Windows.
    #[cfg(not(windows))]
    pub fn delete(&self, _id: SecretId) -> Result<(), SecretsError> {
        Err(SecretsError::UnsupportedPlatform)
    }

    /// Always [`SecretsError::UnsupportedPlatform`] off Windows.
    #[cfg(not(windows))]
    pub fn rotate_api_bearer(&self, _rng: &mut impl CryptoRng) -> Result<Secret, SecretsError> {
        Err(SecretsError::UnsupportedPlatform)
    }

    /// Audit this store's own `secrets` directory: is it owner-only, as [`open`](Self::open)
    /// created it (or requires a caller's own check if it already existed)?
    ///
    /// This is `cerebro-probe`'s single-object audit
    /// ([`cerebro_probe::backend::audit_object_dacl`]), the same trusted principals and
    /// write-capable rights as the backend-directory audit, so the workspace has one DACL reader.
    /// A caller should treat anything other than [`cerebro_probe::DaclAudit::OwnerOnly`] as
    /// untrusted: "could not check" is never "checked, fine".
    pub fn audit_dir(&self) -> cerebro_probe::DaclAudit {
        cerebro_probe::backend::audit_object_dacl(&self.dir)
    }

    /// Audit one id's `.dpapi` file. See [`audit_dir`](Self::audit_dir).
    pub fn audit_secret(&self, id: SecretId) -> cerebro_probe::DaclAudit {
        cerebro_probe::backend::audit_object_dacl(&self.path_for(id))
    }
}

/// Unique per-call suffix for [`KeyFile`] file names, so two overlapping launches never collide.
/// Mirrors `crate::file`'s identical `TMP_COUNTER` pattern (process id plus a monotonically
/// increasing counter) for the identical reason: a genuine collision would mean this exact process
/// reused a name it had already used, not an innocent coincidence.
#[cfg(windows)]
static KEY_FILE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The per-launch `llama-server` API key file `cerebro-supervisor` passes as `--api-key-file`.
///
/// The key is written once, to an owner-only file, and is never placed in argv, an environment
/// variable, a log line, or an error message -- see the module documentation's "KeyFile and
/// process-kill" section for what its [`Drop`] does and does not guarantee.
pub struct KeyFile {
    path: PathBuf,
}

impl KeyFile {
    /// Write `secret` to a fresh, uniquely named, owner-only file inside `dir`.
    #[cfg(windows)]
    pub fn create(dir: &Path, secret: &Secret) -> Result<Self, SecretsError> {
        let unique = KEY_FILE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = dir.join(format!("cerebro-child-key-{}-{unique}.key", std::process::id()));
        write_owner_only_file(&path, secret.expose_secret())?;
        Ok(Self { path })
    }

    /// Always [`SecretsError::UnsupportedPlatform`] off Windows.
    #[cfg(not(windows))]
    pub fn create(_dir: &Path, _secret: &Secret) -> Result<Self, SecretsError> {
        Err(SecretsError::UnsupportedPlatform)
    }

    /// The file's path, for a caller to pass as `--api-key-file`.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Audit this file's own ACL. See [`SecretStore::audit_dir`].
    pub fn audit(&self) -> cerebro_probe::DaclAudit {
        cerebro_probe::backend::audit_object_dacl(&self.path)
    }

    /// Delete the key file now, propagating a real failure instead of best-effort-ing it away.
    ///
    /// Equivalent to letting this value drop, except the result is observable. Safe to call and
    /// then let the value drop anyway: the file is already gone, so the second removal attempt
    /// [`Drop`] performs sees [`std::io::ErrorKind::NotFound`] and does nothing.
    pub fn close(self) -> Result<(), SecretsError> {
        match std::fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(SecretsError::Io { path: self.path.clone(), cause: e.to_string() }),
        }
    }
}

impl Drop for KeyFile {
    fn drop(&mut self) {
        match std::fs::remove_file(&self.path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                // `Drop::drop` cannot return a `Result`, and this crate carries no logging
                // dependency, so stderr is the only channel available to satisfy "a failed delete
                // is recorded, not swallowed silently" (the brief's own words for this exact
                // requirement) rather than the alternative, `let _ = std::fs::remove_file(...)`,
                // which really would swallow it. Only the path is named here; this struct never
                // holds the key's bytes (they were written once, in `create`, and never stored),
                // so there is nothing in scope that could leak it even by accident.
                eprintln!("cerebro-config: could not delete child key file {}: {e}", self.path.display());
            }
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use rand::SeedableRng as _;
    use std::os::windows::fs::OpenOptionsExt as _;

    use super::*;

    #[test]
    fn round_trip_set_and_get_for_both_ids() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");
        for (id, bytes) in [
            (SecretId::ApiBearer, b"api-bearer-secret-value".to_vec()),
            (SecretId::HfToken, b"hf_abcdefgh12345".to_vec()),
        ] {
            assert!(store.get(id).expect("get before set").is_none(), "{id:?} must start unset");
            store.set(id, &Secret::new(bytes.clone())).expect("set");
            let got = store.get(id).expect("get after set").expect("secret present");
            assert_eq!(got.expose_secret(), bytes.as_slice());
        }
    }

    #[test]
    fn set_replaces_a_previously_stored_secret() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");
        store.set(SecretId::HfToken, &Secret::new(b"first".to_vec())).expect("set first");
        store.set(SecretId::HfToken, &Secret::new(b"second".to_vec())).expect("set second");
        let got = store.get(SecretId::HfToken).expect("get").expect("present");
        assert_eq!(got.expose_secret(), b"second");
    }

    #[test]
    fn blob_protected_under_one_id_does_not_decrypt_as_another() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");
        store
            .set(SecretId::ApiBearer, &Secret::new(b"only-for-api-bearer".to_vec()))
            .expect("set");

        let ciphertext = std::fs::read(store.path_for(SecretId::ApiBearer)).expect("read blob");
        write_owner_only_file(&store.path_for(SecretId::HfToken), &ciphertext).expect("plant blob");

        let err = store.get(SecretId::HfToken).expect_err("cross-id blob must not decrypt");
        assert!(matches!(err, SecretsError::Decrypt { id: SecretId::HfToken, .. }), "got {err:?}");
    }

    #[test]
    fn tampered_ciphertext_is_a_typed_error_not_a_panic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");
        store.set(SecretId::ApiBearer, &Secret::new(b"tamper-me".to_vec())).expect("set");

        let path = store.path_for(SecretId::ApiBearer);
        let mut bytes = std::fs::read(&path).expect("read blob");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write tampered blob");

        let err = store.get(SecretId::ApiBearer).expect_err("tampered blob must not decrypt");
        assert!(matches!(err, SecretsError::Decrypt { id: SecretId::ApiBearer, .. }), "got {err:?}");
    }

    #[test]
    fn delete_is_idempotent_and_removes_the_stored_secret() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");
        store.set(SecretId::HfToken, &Secret::new(b"gone-soon".to_vec())).expect("set");
        store.delete(SecretId::HfToken).expect("delete");
        assert!(store.get(SecretId::HfToken).expect("get after delete").is_none());
        store
            .delete(SecretId::HfToken)
            .expect("deleting an already-absent secret must not be an error");
    }

    #[test]
    fn rotate_api_bearer_generates_256_bits_and_stores_them() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FF_EE01);
        let secret = store.rotate_api_bearer(&mut rng).expect("rotate");
        assert_eq!(secret.len(), 32, "256 bits must be 32 bytes");

        let stored = store.get(SecretId::ApiBearer).expect("get").expect("present");
        assert_eq!(stored.expose_secret(), secret.expose_secret());
    }

    #[test]
    fn successive_rotations_from_the_same_generator_differ() {
        // Regression test for "the generator was accidentally re-seeded, or its state discarded,
        // between calls" -- two draws from one live generator must not coincide.
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let first = store.rotate_api_bearer(&mut rng).expect("rotate 1");
        let second = store.rotate_api_bearer(&mut rng).expect("rotate 2");
        assert_ne!(first.expose_secret(), second.expose_secret());
    }

    #[test]
    fn secret_debug_display_and_serde_never_contain_plaintext() {
        let marker = "PLAINTEXT-MARKER-should-never-appear";
        let secret = Secret::new(marker.as_bytes().to_vec());
        assert_eq!(format!("{secret:?}"), "<redacted>");
        assert_eq!(format!("{secret}"), "<redacted>");
        let json = serde_json::to_string(&secret).expect("serialize");
        assert_eq!(json, "\"<redacted>\"");
        assert!(!json.contains(marker), "serialized form leaked the secret: {json}");
    }

    fn assert_no_marker(err: &SecretsError, marker: &str) {
        let display = err.to_string();
        let debug = format!("{err:?}");
        assert!(!display.contains(marker), "Display leaked the secret: {display}");
        assert!(!debug.contains(marker), "Debug leaked the secret: {debug}");
    }

    #[test]
    fn no_error_path_ever_echoes_the_secret_bytes() {
        let marker = "ULTRA-SECRET-MARKER-7f3c9a";
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");

        // Decrypt: cross-id.
        store
            .set(SecretId::ApiBearer, &Secret::new(marker.as_bytes().to_vec()))
            .expect("set");
        let ciphertext = std::fs::read(store.path_for(SecretId::ApiBearer)).expect("read blob");
        write_owner_only_file(&store.path_for(SecretId::HfToken), &ciphertext).expect("plant");
        let cross_id_err = store.get(SecretId::HfToken).expect_err("cross-id must fail");
        assert_no_marker(&cross_id_err, marker);

        // Decrypt: tampered.
        let path = store.path_for(SecretId::ApiBearer);
        let mut bytes = std::fs::read(&path).expect("read");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write tampered");
        let tampered_err = store.get(SecretId::ApiBearer).expect_err("tampered must fail");
        assert_no_marker(&tampered_err, marker);

        // Win32 (CreateFileW/CREATE_NEW): deny creating files in the secrets directory, so `set`
        // fails creating its temp file while the value it was asked to write carries the marker.
        let win32_err = with_file_creation_denied(&store.dir, || {
            store
                .set(SecretId::ApiBearer, &Secret::new(marker.as_bytes().to_vec()))
                .expect_err("a directory that refuses new files must make CreateFileW fail")
        });
        assert!(matches!(win32_err, SecretsError::Win32 { call: "CreateFileW", .. }), "got {win32_err:?}");
        assert_no_marker(&win32_err, marker);

        // Io (rename): lock the target so the final rename over it fails.
        std::fs::write(&path, b"placeholder").expect("seed a real file to lock");
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0) // no sharing at all: blocks replacing it, exactly like `file.rs`'s own tests.
            .open(&path)
            .expect("exclusively lock the blob");
        let rename_err = store
            .set(SecretId::ApiBearer, &Secret::new(marker.as_bytes().to_vec()))
            .expect_err("a locked target must make the rename fail");
        assert!(matches!(rename_err, SecretsError::Io { .. }), "got {rename_err:?}");
        assert_no_marker(&rename_err, marker);
        drop(lock);

        // Io (std::fs::read): lock the file so a concurrent `get` fails at the read step with a
        // sharing violation.
        std::fs::write(&path, marker.as_bytes()).expect("reseed with the marker itself");
        let lock = std::fs::OpenOptions::new()
            .write(true)
            .share_mode(0)
            .open(&path)
            .expect("exclusively lock the blob");
        let io_err = store.get(SecretId::ApiBearer).expect_err("locked file must fail to read");
        assert!(matches!(io_err, SecretsError::Io { .. }), "got {io_err:?}");
        assert_no_marker(&io_err, marker);
        drop(lock);

        // `Encrypt` (CryptProtectData) and `UnsupportedPlatform` are not reachable through a live
        // call on this platform without mocking the OS. Constructed directly so their Display/Debug
        // are still proven never to embed the marker, without pretending this is a live failure.
        assert_no_marker(&SecretsError::Encrypt { id: SecretId::ApiBearer, code: 8 }, marker);
        assert_no_marker(&SecretsError::UnsupportedPlatform, marker);
    }

    /// Run `f` while `dir` grants this user read and traverse only (no file creation), restoring
    /// the owner-only ACL afterwards so the tempdir can be cleaned up.
    fn with_file_creation_denied<T>(dir: &Path, f: impl FnOnce() -> T) -> T {
        let me = cerebro_probe::backend::current_user_sid().expect("current user");
        cerebro_probe::backend::set_dacl_from_sddl_for_tests(dir, &format!("D:P(A;;FRFX;;;{me})(A;;FA;;;SY)"))
            .expect("make the directory read-only for this user");
        let out = f();
        cerebro_probe::backend::set_dacl_from_sddl_for_tests(dir, &format!("D:P(A;;FA;;;{me})(A;;FA;;;SY)"))
            .expect("restore the owner-only ACL");
        out
    }

    /// Names in `dir` other than the store's own blobs: a failed write must leave no temp file.
    fn stray_files(dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .expect("list")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .filter(|n| !n.ends_with(".dpapi"))
            .collect()
    }

    #[test]
    fn a_failed_set_leaves_the_previous_secret_intact() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");
        store.set(SecretId::HfToken, &Secret::new(b"old-token".to_vec())).expect("set old");

        // The create fails: delete-then-create would already have destroyed the old blob here.
        let err = with_file_creation_denied(&store.dir, || {
            store.set(SecretId::HfToken, &Secret::new(b"new-token".to_vec())).expect_err("create denied")
        });
        assert!(matches!(err, SecretsError::Win32 { call: "CreateFileW", .. }), "got {err:?}");
        let kept = store.get(SecretId::HfToken).expect("get").expect("the old secret survives");
        assert_eq!(kept.expose_secret(), b"old-token");

        // The rename fails (target locked): the old blob stays, and the temp file is cleaned up.
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(1) // FILE_SHARE_READ only: `get` can still read, but nothing can replace it.
            .open(store.path_for(SecretId::HfToken))
            .expect("lock the blob against replacement");
        let err = store.set(SecretId::HfToken, &Secret::new(b"new-token".to_vec())).expect_err("rename blocked");
        assert!(matches!(err, SecretsError::Io { .. }), "got {err:?}");
        let kept = store.get(SecretId::HfToken).expect("get").expect("the old secret survives");
        assert_eq!(kept.expose_secret(), b"old-token");
        drop(lock);
        assert!(stray_files(&store.dir).is_empty(), "{:?}", stray_files(&store.dir));

        // And with nothing in the way the replace goes through, still owner-only.
        store.set(SecretId::HfToken, &Secret::new(b"new-token".to_vec())).expect("set new");
        let got = store.get(SecretId::HfToken).expect("get").expect("present");
        assert_eq!(got.expose_secret(), b"new-token");
        assert_eq!(store.audit_secret(SecretId::HfToken), cerebro_probe::DaclAudit::OwnerOnly);
        assert!(stray_files(&store.dir).is_empty(), "{:?}", stray_files(&store.dir));
    }

    #[test]
    fn secret_get_refuses_an_oversized_blob() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");
        let path = store.path_for(SecretId::ApiBearer);
        std::fs::write(&path, vec![0u8; MAX_SECRET_BLOB_BYTES as usize + 1]).expect("plant an oversized blob");
        let err = store.get(SecretId::ApiBearer).expect_err("an oversized blob is refused");
        assert!(
            matches!(&err, SecretsError::Io { cause, .. } if cause.contains("exceeds")),
            "got {err:?}"
        );
        // Exactly at the cap it is read (and then fails to decrypt, as junk must).
        std::fs::write(&path, vec![0u8; MAX_SECRET_BLOB_BYTES as usize]).expect("plant a blob at the cap");
        let err = store.get(SecretId::ApiBearer).expect_err("junk does not decrypt");
        assert!(matches!(err, SecretsError::Decrypt { .. }), "got {err:?}");
    }

    #[test]
    fn secret_get_refuses_a_blob_that_is_a_reparse_point() {
        // A junction needs no admin rights; this module never creates one, so one here was planted.
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");
        let elsewhere = dir.path().join("elsewhere");
        std::fs::create_dir_all(&elsewhere).expect("create target");
        cerebro_probe::backend::create_junction_for_tests(&store.path_for(SecretId::HfToken), &elsewhere)
            .expect("plant a junction at the blob's name");
        let err = store.get(SecretId::HfToken).expect_err("a reparse point is refused");
        assert!(
            matches!(&err, SecretsError::Io { cause, .. } if cause.contains("reparse point")),
            "got {err:?}"
        );
    }

    #[test]
    fn secrets_directory_and_stored_files_are_owner_only_per_dacl_audit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");
        let dir_audit = store.audit_dir();
        assert_eq!(dir_audit, cerebro_probe::DaclAudit::OwnerOnly, "got {dir_audit:?}");
        assert!(dir_audit.is_safe_to_spawn());

        store.set(SecretId::ApiBearer, &Secret::new(b"x".to_vec())).expect("set");
        let file_audit = store.audit_secret(SecretId::ApiBearer);
        assert_eq!(file_audit, cerebro_probe::DaclAudit::OwnerOnly, "got {file_audit:?}");
        assert!(file_audit.is_safe_to_spawn());
    }

    #[test]
    fn a_widened_secret_or_key_file_acl_is_reported_by_the_shared_audit() {
        // The audit must be a real read of the on-disk ACL, not an echo of what `set` wrote: widen
        // each file after the fact and the verdict must change.
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SecretStore::open(dir.path()).expect("open store");
        store.set(SecretId::HfToken, &Secret::new(b"x".to_vec())).expect("set");
        let me = cerebro_probe::backend::current_user_sid().expect("current user");
        let widened = format!("D:P(A;;FA;;;{me})(A;;FW;;;BU)");
        cerebro_probe::backend::set_dacl_from_sddl_for_tests(&store.path_for(SecretId::HfToken), &widened)
            .expect("widen the blob's ACL");
        let audit = store.audit_secret(SecretId::HfToken);
        assert_eq!(audit.principals().len(), 1, "got {audit:?}");
        assert!(!audit.is_safe_to_spawn());

        let key = KeyFile::create(dir.path(), &Secret::new(b"child-key".to_vec())).expect("create");
        cerebro_probe::backend::set_dacl_from_sddl_for_tests(key.path(), &widened).expect("widen the key file");
        assert!(matches!(key.audit(), cerebro_probe::DaclAudit::WritableByOthers(_)), "{:?}", key.audit());
    }

    #[test]
    fn key_file_is_owner_only_per_dacl_audit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let key = KeyFile::create(dir.path(), &Secret::new(b"child-key".to_vec())).expect("create");
        let audit = key.audit();
        assert_eq!(audit, cerebro_probe::DaclAudit::OwnerOnly, "got {audit:?}");
        assert!(audit.is_safe_to_spawn());
    }

    #[test]
    fn key_file_is_deleted_when_dropped() {
        let dir = tempfile::tempdir().expect("tempdir");
        let secret = Secret::new(b"drop-me".to_vec());
        let key = KeyFile::create(dir.path(), &secret).expect("create");
        let path = key.path().to_path_buf();
        assert!(path.is_file());
        drop(key);
        assert!(!path.exists(), "key file must be deleted on drop");
    }

    #[test]
    fn key_file_close_deletes_and_propagates_a_real_failure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let secret = Secret::new(b"close-me".to_vec());
        let key = KeyFile::create(dir.path(), &secret).expect("create");
        let path = key.path().to_path_buf();
        key.close().expect("close deletes the file");
        assert!(!path.exists());

        // A second `close`, on a fresh `KeyFile` whose file is then made undeletable, must
        // propagate the real failure rather than swallow it.
        let key2 = KeyFile::create(dir.path(), &Secret::new(b"undeletable".to_vec())).expect("create");
        let path2 = key2.path().to_path_buf();
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(&path2)
            .expect("exclusively lock the key file");
        let err = key2.close().expect_err("locked file must fail to delete");
        assert!(matches!(err, SecretsError::Io { .. }), "got {err:?}");
        drop(lock);
        let _ = std::fs::remove_file(&path2);
    }

    #[test]
    fn key_file_never_holds_the_secret_bytes_in_its_own_fields() {
        // `KeyFile` writes the secret once, in `create`, and its only field afterward is a
        // `PathBuf`. This is a structural check, not a behavioral one: it exists so a future
        // edit that adds a `secret: Secret` field for convenience gets caught by a compile-time
        // size assumption rather than relying on every future test remembering to check for it.
        assert_eq!(
            std::mem::size_of::<KeyFile>(),
            std::mem::size_of::<PathBuf>(),
            "KeyFile must hold nothing beyond its path"
        );
    }
}
