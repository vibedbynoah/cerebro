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
//! directory this module creates is therefore given the SDDL `D:P(A;;FA;;;OW)(A;;FA;;;SY)` --
//! File-All-Access to the owner and to `SYSTEM` only, `P` marking the ACL protected so nothing
//! above it in the tree can inject an inherited ACE that would widen it. [`audit_owner_only`] reads
//! that ACL back and reports a [`cerebro_probe::DaclAudit`], the same three-state type
//! `cerebro-probe` uses for the backend-directory check, so "could not verify" is never mistaken
//! for "verified, fine" here either.
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

/// SDDL granting File-All-Access to the object's owner and to `SYSTEM` only. `P` marks the ACL
/// protected, so nothing above the object in the directory tree can inject an inherited ACE that
/// would widen this -- see the module documentation's "Why the on-disk ACL matters" section.
#[cfg(windows)]
const OWNER_ONLY_SDDL: &str = "D:P(A;;FA;;;OW)(A;;FA;;;SY)";

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

/// An owner-only `SECURITY_ATTRIBUTES`, built once from [`OWNER_ONLY_SDDL`] and freed exactly once
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

        let wide = to_wide(OWNER_ONLY_SDDL);
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
/// attempt to retroactively tighten a directory it did not itself create. [`audit_owner_only`]
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

/// Replace whatever is at `path` with a fresh owner-only file containing `bytes`.
///
/// Deletes any pre-existing file at `path` first (best-effort; absence is not an error) so the
/// `CREATE_NEW` this performs always reflects *this* call's ACL, never one an earlier write -- or a
/// hostile pre-plant -- left behind.
#[cfg(windows)]
fn write_owner_only_file(path: &Path, bytes: &[u8]) -> Result<(), SecretsError> {
    use std::io::Write as _;

    let _ = std::fs::remove_file(path);
    let mut file = create_owner_only_file(path)?;
    file.write_all(bytes)
        .map_err(|e| SecretsError::Io { path: path.to_path_buf(), cause: e.to_string() })?;
    file.sync_all()
        .map_err(|e| SecretsError::Io { path: path.to_path_buf(), cause: e.to_string() })
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

/// `ACCESS_ALLOWED_ACE_TYPE` (winnt.h). Defined locally, rather than by adding the
/// `Win32_System_SystemServices` feature for two constants, matching
/// `crate::file`/`crate::profile`'s own local copies of `FILE_FLAG_OPEN_REPARSE_POINT` and
/// `FILE_ATTRIBUTE_REPARSE_POINT` for the identical reason.
#[cfg(windows)]
const ACCESS_ALLOWED_ACE_TYPE: u8 = 0;

/// `ACCESS_DENIED_ACE_TYPE` (winnt.h). See [`ACCESS_ALLOWED_ACE_TYPE`]'s doc comment.
#[cfg(windows)]
const ACCESS_DENIED_ACE_TYPE: u8 = 1;

/// Read the DACL of `path` and report whether any principal other than its owner, `SYSTEM`, or
/// `Administrators` holds a write-capable right on it.
///
/// This is [`cerebro_probe::DaclAudit`], the same three-state type `cerebro-probe` audits backend
/// directories with, so "could not check" is never mistaken for "checked, fine" here either --
/// house rule S6. `cerebro-probe`'s own `audit_dacl` is not reused directly: it is a private
/// implementation detail of `cerebro_probe::backend::probe` (a backend-directory probe that shells
/// out to `llama-server --version`), not a general-purpose "audit this path" function, and it is
/// not yet implemented there (it returns [`cerebro_probe::DaclAudit::Unknown`] unconditionally).
/// This function is that implementation, written against the one type both crates already share,
/// so nothing here depends on `cerebro-probe`'s internals changing.
///
/// Only `ACCESS_ALLOWED_ACE` entries are inspected. An `ACCESS_DENIED_ACE` is skipped (it can only
/// narrow access, and this function asks the narrower, sufficient question "does any *allow* ACE
/// grant a write-capable right to an unexpected trustee" rather than attempting full
/// effective-permissions evaluation across ACE ordering and inheritance). Any other ACE type --
/// object ACEs, callback ACEs, conditional ACEs -- could in principle still grant access, so
/// encountering one reports [`cerebro_probe::DaclAudit::Unknown`] rather than silently skipping it,
/// per house rule S6.
#[cfg(windows)]
fn audit_owner_only(path: &Path) -> cerebro_probe::DaclAudit {
    audit_owner_only_inner(path)
        .unwrap_or(cerebro_probe::DaclAudit::Unknown("owner-only DACL audit could not complete"))
}

#[cfg(windows)]
fn audit_owner_only_inner(path: &Path) -> Result<cerebro_probe::DaclAudit, SecretsError> {
    use windows_sys::Win32::Foundation::{GENERIC_ALL, GENERIC_WRITE, LocalFree};
    use windows_sys::Win32::Security::Authorization::{GetNamedSecurityInfoW, SE_FILE_OBJECT};
    use windows_sys::Win32::Security::{
        GetAce, WinBuiltinAdministratorsSid, WinCreatorOwnerRightsSid, WinLocalSystemSid, ACCESS_ALLOWED_ACE,
        ACE_HEADER, ACL, DACL_SECURITY_INFORMATION, OWNER_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, PSID,
    };
    use windows_sys::Win32::Storage::FileSystem::{DELETE, FILE_APPEND_DATA, FILE_WRITE_DATA, WRITE_DAC, WRITE_OWNER};

    /// Every access bit this audit treats as "can modify the object," combined once so the loop
    /// below is a single mask test rather than a chain of comparisons that could individually be
    /// forgotten to update. `FILE_APPEND_DATA` doubles as `FILE_ADD_SUBDIRECTORY` (same bit) and
    /// `FILE_WRITE_DATA` as `FILE_ADD_FILE` (same bit); both readings are write-capable, so the
    /// single import covers files and directories alike.
    const WRITE_CAPABLE_MASK: u32 = GENERIC_ALL | GENERIC_WRITE | FILE_WRITE_DATA | FILE_APPEND_DATA | DELETE | WRITE_DAC | WRITE_OWNER;

    let wide = to_wide(path);
    let mut owner_sid: PSID = std::ptr::null_mut();
    let mut dacl: *mut ACL = std::ptr::null_mut();
    let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();

    // SAFETY: `wide` is a live NUL-terminated path. The group-SID and SACL out-parameters are left
    // null, which is documented as legal when the corresponding bit is not set in `securityinfo`
    // (only `OWNER_SECURITY_INFORMATION` and `DACL_SECURITY_INFORMATION` are requested).
    // `descriptor` receives, on success, ownership of a `LocalAlloc`'d buffer that `owner_sid` and
    // `dacl` merely point *into* -- freeing `descriptor` exactly once (`_free_descriptor` below) is
    // therefore correct and sufficient; `owner_sid`/`dacl` must never be freed separately.
    let status = unsafe {
        GetNamedSecurityInfoW(
            wide.as_ptr(),
            SE_FILE_OBJECT,
            OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            &mut owner_sid,
            std::ptr::null_mut(),
            &mut dacl,
            std::ptr::null_mut(),
            &mut descriptor,
        )
    };
    if status != 0 {
        return Err(SecretsError::Win32 { call: "GetNamedSecurityInfoW", code: status });
    }

    struct FreeDescriptor(PSECURITY_DESCRIPTOR);
    impl Drop for FreeDescriptor {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: allocated by `GetNamedSecurityInfoW` via `LocalAlloc`; freed here exactly
                // once, for the same reason as every other `LocalFree` in this module.
                unsafe { LocalFree(self.0 as windows_sys::Win32::Foundation::HLOCAL) };
            }
        }
    }
    let _free_descriptor = FreeDescriptor(descriptor);

    if dacl.is_null() {
        // A null DACL means "no protection at all": every principal has every right. Never
        // possible for a file or directory this module created (it always supplies the explicit
        // owner-only SDDL), so this is only reachable auditing something else -- and it is
        // unambiguously unsafe, not merely unknown.
        return Ok(cerebro_probe::DaclAudit::WritableByOthers(vec![
            "(no DACL at all: every principal has every right)".to_string(),
        ]));
    }

    let system_sid = well_known_sid(WinLocalSystemSid)?;
    let admins_sid = well_known_sid(WinBuiltinAdministratorsSid)?;
    // The SDDL alias `OW` this module writes (`OWNER_ONLY_SDDL`) does not expand to the object's
    // *current* owner SID at rest -- it stores the literal, special "OWNER RIGHTS" SID (S-1-3-4),
    // which Windows resolves to whatever principal owns the object *at access-check time*. That is
    // precisely why `OW` is the idiomatic way to write an owner-only ACE (it keeps working if
    // ownership is ever transferred, unlike embedding a literal user SID would), but it also means
    // this audit must accept this specific well-known SID as expected, not just the literal
    // `owner_sid` `GetNamedSecurityInfoW` reported -- the two are related but are not the same SID.
    let owner_rights_sid = well_known_sid(WinCreatorOwnerRightsSid)?;

    // SAFETY: `dacl` is non-null and was populated by the successful call above, and remains valid
    // through this whole function (kept alive by `_free_descriptor`, which drops only at the end of
    // this function's scope). This reads only the ACL's own fixed header field.
    let ace_count = unsafe { (*dacl).AceCount };

    let mut others = Vec::new();
    for index in 0..u32::from(ace_count) {
        let mut ace_ptr: *mut core::ffi::c_void = std::ptr::null_mut();
        // SAFETY: `dacl` is valid (see above) and `index < ace_count`, which is exactly what
        // `GetAce` requires of `dwaceindex`.
        let ok = unsafe { GetAce(dacl, index, &mut ace_ptr) };
        if ok == 0 {
            return Err(SecretsError::Win32 { call: "GetAce", code: last_error() });
        }
        // SAFETY: `ace_ptr` is non-null (`GetAce` succeeded) and every ACE, of any type, begins
        // with an `ACE_HEADER` -- reading only that common prefix is in bounds regardless of the
        // ACE's actual type.
        let ace_type = unsafe { (*ace_ptr.cast::<ACE_HEADER>()).AceType };

        if ace_type == ACCESS_DENIED_ACE_TYPE {
            continue;
        }
        if ace_type != ACCESS_ALLOWED_ACE_TYPE {
            return Ok(cerebro_probe::DaclAudit::Unknown(
                "DACL contains an ACE type this audit does not model",
            ));
        }

        // SAFETY: `ace_type == ACCESS_ALLOWED_ACE_TYPE` was just confirmed, so `ace_ptr` points to
        // an `ACCESS_ALLOWED_ACE` (`Header`, then `Mask`, then the inline SID starting at
        // `SidStart`); Windows guarantees `Header.AceSize` covers at least this fixed prefix for
        // this ACE type.
        let ace = unsafe { &*ace_ptr.cast::<ACCESS_ALLOWED_ACE>() };
        if ace.Mask & WRITE_CAPABLE_MASK == 0 {
            continue;
        }

        // SAFETY: `SidStart` is the address of the ACE's inline, variable-length SID, which is
        // valid for as many bytes as the SID's own sub-authority count declares -- exactly the
        // memory `EqualSid`/`LookupAccountSidW` read, and well within `Header.AceSize`.
        let sid: PSID = std::ptr::addr_of!(ace.SidStart) as PSID;

        let is_expected = sid_equals(sid, owner_sid)
            || sid_equals(sid, system_sid.as_ptr() as PSID)
            || sid_equals(sid, admins_sid.as_ptr() as PSID)
            || sid_equals(sid, owner_rights_sid.as_ptr() as PSID);
        if !is_expected {
            others.push(describe_sid(sid));
        }
    }

    if others.is_empty() {
        Ok(cerebro_probe::DaclAudit::OwnerOnly)
    } else {
        Ok(cerebro_probe::DaclAudit::WritableByOthers(others))
    }
}

/// Build one of the fixed well-known SIDs (`SYSTEM`, `Administrators`) into a stack buffer sized to
/// [`windows_sys::Win32::Security::SECURITY_MAX_SID_SIZE`], the documented maximum size of any SID.
#[cfg(windows)]
fn well_known_sid(
    sid_type: windows_sys::Win32::Security::WELL_KNOWN_SID_TYPE,
) -> Result<[u8; windows_sys::Win32::Security::SECURITY_MAX_SID_SIZE as usize], SecretsError> {
    use windows_sys::Win32::Security::{CreateWellKnownSid, SECURITY_MAX_SID_SIZE};

    let mut buf = [0u8; SECURITY_MAX_SID_SIZE as usize];
    let mut size = buf.len() as u32;
    // SAFETY: `buf` is exactly `SECURITY_MAX_SID_SIZE` bytes and `size` communicates that capacity;
    // on success the API writes the SID's actual length back into `size`, which this function does
    // not need further (the SID itself, in `buf`, is what callers compare against).
    let ok = unsafe { CreateWellKnownSid(sid_type, std::ptr::null_mut(), buf.as_mut_ptr().cast(), &mut size) };
    if ok == 0 {
        return Err(SecretsError::Win32 { call: "CreateWellKnownSid", code: last_error() });
    }
    Ok(buf)
}

/// Whether two SIDs identify the same trustee.
#[cfg(windows)]
fn sid_equals(a: windows_sys::Win32::Security::PSID, b: windows_sys::Win32::Security::PSID) -> bool {
    use windows_sys::Win32::Security::EqualSid;
    if a.is_null() || b.is_null() {
        return false;
    }
    // SAFETY: both `a` and `b` are non-null and point at initialized SIDs, live for the duration of
    // this call -- either an ACE's own inline SID (kept alive by the audit's `descriptor`) or one
    // of `well_known_sid`'s stack buffers (kept alive by its caller's local variable).
    unsafe { EqualSid(a, b) != 0 }
}

/// Best-effort human-readable name for a trustee SID, for
/// [`cerebro_probe::DaclAudit::WritableByOthers`]'s finding list. Falls back to the SID's string
/// form (`S-1-5-...`) when the account cannot be looked up (an orphaned SID from a deleted
/// account), rather than losing the finding entirely.
#[cfg(windows)]
fn describe_sid(sid: windows_sys::Win32::Security::PSID) -> String {
    use windows_sys::Win32::Security::LookupAccountSidW;

    let mut name = [0u16; 256];
    let mut name_len = name.len() as u32;
    let mut domain = [0u16; 256];
    let mut domain_len = domain.len() as u32;
    let mut use_: i32 = 0;
    // SAFETY: `sid` is a live, valid SID for the duration of this call. `name`/`domain` are
    // fixed-size stack buffers, and `name_len`/`domain_len` communicate their capacity in `u16`
    // elements (what this API expects) and receive the lengths actually written back.
    let ok = unsafe {
        LookupAccountSidW(
            std::ptr::null(),
            sid,
            name.as_mut_ptr(),
            &mut name_len,
            domain.as_mut_ptr(),
            &mut domain_len,
            &mut use_,
        )
    };
    if ok != 0 {
        let name = String::from_utf16_lossy(&name[..name_len as usize]);
        let domain = String::from_utf16_lossy(&domain[..domain_len as usize]);
        if domain.is_empty() {
            return name;
        }
        return format!("{domain}\\{name}");
    }
    string_sid(sid).unwrap_or_else(|| "(unknown trustee)".to_string())
}

/// The `S-1-5-...` string form of a SID, via `ConvertSidToStringSidW`.
#[cfg(windows)]
fn string_sid(sid: windows_sys::Win32::Security::PSID) -> Option<String> {
    use windows_sys::Win32::Security::Authorization::ConvertSidToStringSidW;

    let mut ptr: *mut u16 = std::ptr::null_mut();
    // SAFETY: `sid` is valid for the duration of this call; on success Windows allocates the
    // output string via `LocalAlloc`, freed exactly once below.
    let ok = unsafe { ConvertSidToStringSidW(sid, &mut ptr) };
    if ok == 0 || ptr.is_null() {
        return None;
    }
    // SAFETY: `ptr` is a NUL-terminated wide string Windows just allocated; scanning for the
    // terminator is the documented way to discover its length before copying it out.
    let len = unsafe {
        let mut i = 0usize;
        while *ptr.add(i) != 0 {
            i += 1;
        }
        i
    };
    // SAFETY: `ptr` is valid for `len` initialized `u16` elements per the scan just performed.
    let text = unsafe { std::slice::from_raw_parts(ptr, len) };
    let s = String::from_utf16_lossy(text);
    // SAFETY: `ptr` was allocated by `ConvertSidToStringSidW` via `LocalAlloc`; freed here exactly
    // once.
    unsafe { windows_sys::Win32::Foundation::LocalFree(ptr as windows_sys::Win32::Foundation::HLOCAL) };
    Some(s)
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
    /// See the module documentation's entropy-separation section.
    #[cfg(windows)]
    pub fn get(&self, id: SecretId) -> Result<Option<Secret>, SecretsError> {
        let path = self.path_for(id);
        let ciphertext = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(SecretsError::Io { path, cause: e.to_string() }),
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
    /// A caller (`cerebro-supervisor`, before ever trusting a secret this store hands back) should
    /// treat anything other than [`cerebro_probe::DaclAudit::OwnerOnly`] as untrusted -- house rule
    /// S6, and the same rule [`crate::profile::DefaultPreflighter`] applies to the backend
    /// directory.
    #[cfg(windows)]
    pub fn audit_dir(&self) -> cerebro_probe::DaclAudit {
        audit_owner_only(&self.dir)
    }

    /// Always [`cerebro_probe::DaclAudit::Unknown`] off Windows: there is no DACL to audit.
    #[cfg(not(windows))]
    pub fn audit_dir(&self) -> cerebro_probe::DaclAudit {
        cerebro_probe::DaclAudit::Unknown("DACL auditing is implemented for Windows only")
    }

    /// Audit one id's `.dpapi` file. See [`audit_dir`](Self::audit_dir).
    #[cfg(windows)]
    pub fn audit_secret(&self, id: SecretId) -> cerebro_probe::DaclAudit {
        audit_owner_only(&self.path_for(id))
    }

    /// Always [`cerebro_probe::DaclAudit::Unknown`] off Windows.
    #[cfg(not(windows))]
    pub fn audit_secret(&self, _id: SecretId) -> cerebro_probe::DaclAudit {
        cerebro_probe::DaclAudit::Unknown("DACL auditing is implemented for Windows only")
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
    #[cfg(windows)]
    pub fn audit(&self) -> cerebro_probe::DaclAudit {
        audit_owner_only(&self.path)
    }

    /// Always [`cerebro_probe::DaclAudit::Unknown`] off Windows.
    #[cfg(not(windows))]
    pub fn audit(&self) -> cerebro_probe::DaclAudit {
        cerebro_probe::DaclAudit::Unknown("DACL auditing is implemented for Windows only")
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

        // Win32 (CreateFileW/CREATE_NEW): lock the target file so `set`'s best-effort
        // remove-then-create-new sequence cannot remove it first, forcing `CreateFileW` itself to
        // fail with a real Win32 error while the value it was asked to write still carries the
        // marker.
        std::fs::write(&path, b"placeholder").expect("seed a real file to lock");
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0) // no sharing at all: blocks delete, exactly like `file.rs`'s own tests.
            .open(&path)
            .expect("exclusively lock the blob");
        let win32_err = store
            .set(SecretId::ApiBearer, &Secret::new(marker.as_bytes().to_vec()))
            .expect_err("locked file must make CreateFileW fail");
        assert!(matches!(win32_err, SecretsError::Win32 { call: "CreateFileW", .. }), "got {win32_err:?}");
        assert_no_marker(&win32_err, marker);
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
