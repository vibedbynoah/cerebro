//! Win32 half of [`super::audit_dacl`].
//!
//! Split in two on purpose. Everything that can be decided without the kernel — how an ACE's bytes
//! parse, which bits are write-capable, which SIDs are trusted — is safe Rust over owned values and
//! is tested from SDDL without touching disk. The `unsafe` surface is confined to fetching a
//! security descriptor and copying its owner and ACEs out of it, one small FFI function at a time.
//!
//! Every allocation the system hands back (`GetNamedSecurityInfoW`, the SDDL conversion,
//! `ConvertStringSidToSidW`) is owned by a [`LocalMemory`] the instant the call returns, so it is
//! freed exactly once on every path, including the early error returns.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::fs::MetadataExt;
use std::path::{Component, Path, PathBuf, Prefix};
use std::ptr::{null, null_mut};

use windows_sys::Win32::Foundation::{
    CloseHandle, GetLastError, LocalFree, ERROR_INSUFFICIENT_BUFFER, ERROR_SUCCESS, GENERIC_ALL,
    GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertStringSidToSidW, GetNamedSecurityInfoW, SE_FILE_OBJECT,
};
use windows_sys::Win32::Security::{
    GetAce, GetLengthSid, GetSecurityDescriptorDacl, GetSecurityDescriptorOwner,
    GetTokenInformation, IsValidAcl, IsValidSid, LookupAccountSidW, TokenUser, ACE_HEADER,
    ACE_INHERITED_OBJECT_TYPE_PRESENT, ACE_OBJECT_TYPE_PRESENT, ACL, DACL_SECURITY_INFORMATION,
    INHERIT_ONLY_ACE, OBJECT_INHERIT_ACE, OWNER_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, PSID,
    SID_NAME_USE, TOKEN_QUERY, TOKEN_USER,
};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, GetDriveTypeW, GetFinalPathNameByHandleW, GetVolumePathNameW, DELETE,
    FILE_ADD_FILE, FILE_ADD_SUBDIRECTORY, FILE_APPEND_DATA, FILE_ATTRIBUTE_REPARSE_POINT,
    FILE_DELETE_CHILD, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
    FILE_NAME_NORMALIZED, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, FILE_WRITE_DATA,
    OPEN_EXISTING, VOLUME_NAME_DOS, WRITE_DAC, WRITE_OWNER,
};
use windows_sys::Win32::System::SystemServices::{
    ACCESS_ALLOWED_ACE_TYPE, ACCESS_ALLOWED_CALLBACK_ACE_TYPE,
    ACCESS_ALLOWED_CALLBACK_OBJECT_ACE_TYPE, ACCESS_ALLOWED_COMPOUND_ACE_TYPE,
    ACCESS_ALLOWED_OBJECT_ACE_TYPE, ACCESS_DENIED_ACE_TYPE, ACCESS_DENIED_CALLBACK_ACE_TYPE,
    ACCESS_DENIED_CALLBACK_OBJECT_ACE_TYPE, ACCESS_DENIED_OBJECT_ACE_TYPE, MAXIMUM_ALLOWED,
    SYSTEM_ACCESS_FILTER_ACE_TYPE, SYSTEM_ALARM_ACE_TYPE, SYSTEM_ALARM_CALLBACK_ACE_TYPE,
    SYSTEM_ALARM_CALLBACK_OBJECT_ACE_TYPE, SYSTEM_ALARM_OBJECT_ACE_TYPE, SYSTEM_AUDIT_ACE_TYPE,
    SYSTEM_AUDIT_CALLBACK_ACE_TYPE, SYSTEM_AUDIT_CALLBACK_OBJECT_ACE_TYPE,
    SYSTEM_AUDIT_OBJECT_ACE_TYPE, SYSTEM_MANDATORY_LABEL_ACE_TYPE,
    SYSTEM_PROCESS_TRUST_LABEL_ACE_TYPE, SYSTEM_RESOURCE_ATTRIBUTE_ACE_TYPE,
    SYSTEM_SCOPED_POLICY_ID_ACE_TYPE,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

use super::{AuditedObject, GrantVia, WritableGrant};
use crate::error::ProbeError;

const SYSTEM: &str = "S-1-5-18";
const ADMINISTRATORS: &str = "S-1-5-32-544";
const TRUSTED_INSTALLER: &str = "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464";
const CREATOR_OWNER: &str = "S-1-3-0";
const EVERYONE: &str = "S-1-1-0";

/// Rights that let a principal replace code or take control of who can, on any object type.
const COMMON_WRITE: u32 =
    DELETE | WRITE_DAC | WRITE_OWNER | GENERIC_WRITE | GENERIC_ALL | MAXIMUM_ALLOWED;
/// `FILE_WRITE_EA` and `FILE_WRITE_ATTRIBUTES` are absent on purpose: neither alters contents or
/// the DACL, so neither can substitute a DLL.
const FILE_WRITE_MASK: u32 = FILE_WRITE_DATA | FILE_APPEND_DATA | COMMON_WRITE;
/// `FILE_DELETE_CHILD` is present on purpose: it deletes a DLL whatever that file's DACL says.
const DIRECTORY_WRITE_MASK: u32 =
    FILE_ADD_FILE | FILE_ADD_SUBDIRECTORY | FILE_DELETE_CHILD | COMMON_WRITE;

/// Rights on an ancestor directory that let a principal delete or re-ACL the path component
/// directly beneath it — which lets that principal delete-and-recreate that component (and
/// everything under it, including the backend directory itself) even though the component's own
/// DACL denies them everything.
///
/// `FILE_ADD_FILE` and `FILE_ADD_SUBDIRECTORY` are deliberately absent: `C:\` grants
/// `FILE_ADD_SUBDIRECTORY` to `Authenticated Users` by default on a stock Windows install, and the
/// ability to create a *new*, differently-named sibling cannot by itself replace an *existing*
/// component. Only rights that remove, rename, or re-ACL something already there count here.
const ANCESTOR_REPLACE_MASK: u32 =
    FILE_DELETE_CHILD | WRITE_DAC | WRITE_OWNER | GENERIC_ALL | MAXIMUM_ALLOWED;
/// Rights on an intermediate path component's own security descriptor that let it be renamed or
/// deleted out from under the backend directory it contains, regardless of what its parent's DACL
/// says. `FILE_DELETE_CHILD` is absent here (unlike [`ANCESTOR_REPLACE_MASK`]): it governs deleting
/// this component's *children*, not deleting or re-ACLing the component itself.
const COMPONENT_REPLACE_MASK: u32 =
    DELETE | WRITE_DAC | WRITE_OWNER | GENERIC_ALL | MAXIMUM_ALLOWED;

/// A SID is at most 8 header bytes plus 15 sub-authorities (`SID_MAX_SUB_AUTHORITIES`).
const SID_MAX_SUB_AUTHORITIES: usize = 15;

/// `GetDriveTypeW`'s "fixed volume" return value (`winbase.h`). Not exported by the
/// `Win32_Storage_FileSystem` feature already enabled for this crate; defined locally rather than
/// pulling in `Win32_System_WindowsProgramming` for two integers.
const DRIVE_FIXED: u32 = 3;
/// `GetDriveTypeW`'s "RAM disk" return value (`winbase.h`). Treated the same as [`DRIVE_FIXED`]: a
/// RAM disk is opaque to the network and its DACL is issued by this machine's own kernel, so
/// [`Trust`]'s well-known SIDs mean what it assumes they mean.
const DRIVE_RAMDISK: u32 = 6;

/// Which kind of object this audit itself is the subject of — as opposed to
/// [`AuditedObject::Ancestor`], which is never passed to [`untrusted_writes`] and has its own
/// dedicated mask parameter on [`ancestor_grants`] instead.
///
/// A separate, two-variant type rather than [`AuditedObject`] itself, so a call site cannot
/// construct the impossible state of auditing directory/file "ownership" through an ancestor path:
/// there is no `OwnKind::Ancestor` arm that could silently fall through and report nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OwnKind {
    /// The backend directory itself.
    Directory,
    /// A regular file directly inside it.
    File,
}

impl OwnKind {
    /// The write-capable rights mask for this kind of object.
    fn mask(self) -> u32 {
        match self {
            OwnKind::Directory => DIRECTORY_WRITE_MASK,
            OwnKind::File => FILE_WRITE_MASK,
        }
    }

    /// The [`AuditedObject`] a grant against this kind of object is reported as.
    fn audited(self) -> AuditedObject {
        match self {
            OwnKind::Directory => AuditedObject::Directory,
            OwnKind::File => AuditedObject::File,
        }
    }
}

/// An access-allowed ACE reduced to what the trust decision needs.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AllowAce {
    flags: u8,
    mask: u32,
    sid: String,
}

/// A DACL as the audit sees it. Deny and audit entries were already dropped while parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Dacl {
    /// No DACL at all: everyone has full control.
    Null,
    /// A DACL, possibly with zero entries (which grants nobody anything).
    Present(Vec<AllowAce>),
}

/// Owner and DACL of one object, copied out of its security descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ObjectSecurity {
    owner: Option<String>,
    dacl: Dacl,
}

/// The principals whose write access is not a finding.
#[derive(Debug, Clone)]
struct Trust {
    current_user: String,
    /// `Some` only when the directory's real owner is already one of the fixed trusted principals
    /// or the current user — see [`Trust::new`]. Any other owner, including one this audit could
    /// not resolve at all, is deliberately not carried in here, so [`Trust::trusts`] treats it like
    /// any other untrusted SID and [`allow_capable`]'s implicit-`WRITE_DAC`-from-ownership check
    /// reports it as [`GrantVia::UntrustedOwner`] — the same path already used for an untrusted
    /// file owner.
    directory_owner: Option<String>,
}

impl Trust {
    /// Build a trust set for one audit run.
    ///
    /// The directory owner is trusted only when it is already one of the fixed principals or the
    /// current user — never on the strength of ownership alone. Ownership implies `WRITE_DAC`
    /// regardless of what the DACL says, so any other owner (a local group, a domain group, another
    /// standard user account, or a SID that does not resolve at all) is reported as
    /// [`GrantVia::UntrustedOwner`] by [`allow_capable`]. This is a deliberate narrowing of "the
    /// object owner" as a trusted principal: an owner outside this fixed set can re-ACL the
    /// directory at will, so ownership by *some* account is not, on its own, a reason to trust it.
    ///
    /// Decided purely on SIDs, with no name-service or domain-controller round trip: a trust
    /// decision must be deterministic and available offline, never dependent on network state.
    fn new(current_user: String, owner: Option<&str>) -> Self {
        let directory_owner = owner
            .filter(|sid| {
                *sid == current_user
                    || *sid == SYSTEM
                    || *sid == ADMINISTRATORS
                    || *sid == TRUSTED_INSTALLER
            })
            .map(str::to_owned);
        Trust {
            current_user,
            directory_owner,
        }
    }

    /// `for_new_files` is true only for inheritable ACEs, where `CREATOR OWNER` means the future
    /// file's creator rather than a SID that can appear in anyone's token.
    fn trusts(&self, sid: &str, for_new_files: bool) -> bool {
        sid == SYSTEM
            || sid == ADMINISTRATORS
            || sid == TRUSTED_INSTALLER
            || sid == self.current_user
            || self.directory_owner.as_deref() == Some(sid)
            || (for_new_files && sid == CREATOR_OWNER)
    }
}

/// A finding before its SID is resolved to a name and attached to a path.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RawGrant {
    sid: String,
    mask: u32,
    applies_to: AuditedObject,
    via: GrantVia,
}

/// Run the audit.
pub(super) fn audit(dir: &Path) -> Result<Vec<WritableGrant>, ProbeError> {
    // Made absolute exactly once, at the very top: every Win32 call below, the directory
    // enumeration, and every reported grant path are derived from this one `PathBuf`. Two spellings
    // of the caller's `dir` (a relative path plus a `set_current_dir` race, or a literal `..`) would
    // otherwise let the security descriptor read and the reported grant describe different things.
    let dir = std::path::absolute(dir).map_err(|e| {
        unauditable(
            &dir.display().to_string(),
            format!("could not make path absolute: {e}"),
        )
    })?;

    // Local-fixed-volume precondition, before a single security descriptor is read. Every trust
    // decision below assumes `SYSTEM`, `Administrators`, and the rest of `Trust`'s fixed set mean
    // *this machine's* principals — true only when this machine issued the descriptor.
    require_local_volume(&dir)?;

    // Resolve `dir`'s DOS-form canonical path before anything else touches it: a `subst` mapping or
    // a `DefineDosDevice` alias is not a reparse point and would otherwise pass `check_path_chain`
    // untouched while every ancestor check below walked the alias's own (unrelated) components
    // instead of the real ones. Every remaining step — the ancestor chain, the directory's own
    // audit, every enumerated file, and every reported grant path — uses this resolved value.
    let dir = canonical_path(&dir)?;
    let dir: &Path = &dir;

    let current_user = current_user_sid()?;
    // `object_security(dir)` before `check_path_chain`, deliberately: it is what makes a genuinely
    // missing `dir` fail with the plain `GetNamedSecurityInfoW` "not found" error this crate has
    // always reported for that case, rather than a differently-shaped `symlink_metadata` error from
    // walking the chain first for no benefit.
    let directory = object_security(dir)?;

    // Path pinning. `dir`'s own DACL, checked just below, is worthless if the *path used to reach
    // it* can be swapped out from under it: a junction or symlink anywhere in the chain would let
    // this audit read one object's security descriptor while ggml loads DLLs from another, and an
    // untrusted principal holding delete/re-ACL rights on an ancestor (or on an intermediate
    // component itself) can rename the real directory away and put an attacker-owned one at the
    // same path. `check_path_chain` rules out the first; the loop after `trust` is built rules out
    // the second.
    let chain = check_path_chain(dir)?;

    let trust = Trust::new(current_user, directory.owner.as_deref());

    let mut raw: Vec<(PathBuf, RawGrant)> = untrusted_writes(&directory, OwnKind::Directory, &trust)
        .into_iter()
        .map(|g| (dir.to_path_buf(), g))
        .collect();

    // `chain` is root-first and its last element is `dir`. For every (parent, child) pair, the
    // parent's own rights over that child are audited; for every child that is *not* `dir` itself
    // (which is already fully audited above and below), that child's own rights over itself are
    // audited too, since a principal need not touch the parent at all to rename or delete a
    // component it directly controls. `windows(2)` rather than indexing `chain[i]`/`chain[i + 1]`:
    // panic-free by construction, matching what this crate promises everywhere else.
    let last_pair = chain.len().saturating_sub(2);
    for (i, pair) in chain.windows(2).enumerate() {
        let [parent, child] = pair else {
            continue; // `windows(2)` never yields a slice of any other length.
        };
        let parent_sec = object_security(parent)?;
        raw.extend(
            ancestor_grants(&parent_sec, &trust, ANCESTOR_REPLACE_MASK)
                .into_iter()
                .map(|g| (parent.clone(), g)),
        );
        if i != last_pair {
            let child_sec = object_security(child)?;
            raw.extend(
                ancestor_grants(&child_sec, &trust, COMPONENT_REPLACE_MASK)
                    .into_iter()
                    .map(|g| (child.clone(), g)),
            );
        }
    }

    // Enumeration errors are fatal rather than skipped: a file the audit could not see is exactly
    // the file it cannot vouch for.
    let entries = std::fs::read_dir(dir).map_err(|e| ProbeError::BackendDirUnreadable {
        dir: dir.display().to_string(),
        cause: e.to_string(),
    })?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| ProbeError::BackendDirUnreadable {
            dir: dir.display().to_string(),
            cause: e.to_string(),
        })?;
        let path = entry.path();
        // Checked with `symlink_metadata` (which does not follow the link) before anything that
        // would resolve it: this crate audits only `dir` and the link's target file, never the
        // target's *parent* directory, so a principal who can delete-and-recreate the target via
        // that other directory's DACL would go unaudited while this function still reported clean.
        // Rather than make a claim it cannot back, any reparse point fails the whole audit closed.
        let link_meta = std::fs::symlink_metadata(&path).map_err(|e| {
            ProbeError::BackendFileUnreadable {
                file: path.display().to_string(),
                cause: e.to_string(),
            }
        })?;
        match classify_entry(link_meta.file_attributes(), link_meta.is_file()) {
            EntryKind::ReparsePoint => {
                return Err(unauditable(
                    &path.display().to_string(),
                    "is a reparse point (symbolic link, junction, or similar); its target's \
                     containing directory is not audited by this crate, so resolving it here \
                     would understate what can be planted",
                ));
            }
            EntryKind::RegularFile => files.push(path),
            EntryKind::Other => {}
        }
    }
    files.sort();

    for file in files {
        let security = object_security(&file)?;
        raw.extend(
            untrusted_writes(&security, OwnKind::File, &trust)
                .into_iter()
                .map(|g| (file.clone(), g)),
        );
    }

    // One lookup per distinct SID: the development build has 55 files carrying the same two
    // untrusted ACEs, and each lookup can reach the account database.
    let mut names: BTreeMap<String, String> = BTreeMap::new();
    Ok(raw
        .into_iter()
        .map(|(path, g)| {
            let principal = names
                .entry(g.sid.clone())
                .or_insert_with(|| account_name(&g.sid).unwrap_or_else(|| g.sid.clone()))
                .clone();
            WritableGrant {
                path,
                applies_to: g.applies_to,
                principal,
                rights: right_names(g.mask, g.applies_to),
                sid: g.sid,
                access_mask: g.mask,
                via: g.via,
            }
        })
        .collect())
}

/// What kind of directory entry counts as, for deciding whether to audit it as a regular file, skip
/// it, or fail the whole audit closed.
///
/// A pure function over the bits Windows already reported, so it is unit-tested without touching
/// disk — the disk-backed reparse-point test only proves that creating a real symlink sets the
/// attribute, not that this decision itself is made correctly.
#[derive(Debug, PartialEq, Eq)]
enum EntryKind {
    /// An ordinary file, safe to audit like any other.
    RegularFile,
    /// Not a reparse point and not a file — a subdirectory, which is out of scope (see
    /// [`super::audit_dacl`]'s "not recursive" note).
    Other,
    /// A symbolic link, junction, or similar.
    ReparsePoint,
}

/// Classify one directory entry from its raw attributes and whether it is a file.
///
/// The reparse attribute wins over the file/directory bit on purpose: a directory junction carries
/// `FILE_ATTRIBUTE_DIRECTORY` right alongside `FILE_ATTRIBUTE_REPARSE_POINT`, and treating it as
/// `Other` because it "looks like a directory" would silently skip exactly the entry this check
/// exists to catch.
fn classify_entry(attributes: u32, is_file: bool) -> EntryKind {
    if attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        EntryKind::ReparsePoint
    } else if is_file {
        EntryKind::RegularFile
    } else {
        EntryKind::Other
    }
}

/// Every path component from the filesystem root down to, and including, `dir` — confirming along
/// the way that none of them is a reparse point.
///
/// `dir` must already be absolute and canonical (`audit` computes that exactly once, at the top,
/// and passes the same value into every function that needs it); this function does no further
/// normalization.
///
/// Called only after [`object_security`] has already confirmed `dir` itself exists, so a genuinely
/// missing directory keeps failing with the plain `GetNamedSecurityInfoW` "not found" error this
/// crate has always reported for that case, rather than a differently-worded `symlink_metadata`
/// error.
///
/// A false positive is possible on a machine using OneDrive Files On-Demand, dedup, or an
/// AppExecLink shim: placeholder files under those carry `FILE_ATTRIBUTE_REPARSE_POINT` too, and
/// this audit fails closed on them exactly as it does on a symlink, since it cannot tell the
/// difference from the attribute alone.
fn check_path_chain(dir: &Path) -> Result<Vec<PathBuf>, ProbeError> {
    let mut chain: Vec<PathBuf> = dir.ancestors().map(Path::to_path_buf).collect();
    chain.reverse(); // Root first, `dir` last.

    for component in &chain {
        // `symlink_metadata` never follows a reparse point, unlike `metadata` or
        // `GetNamedSecurityInfoW` — exactly the property this check needs.
        let meta = std::fs::symlink_metadata(component).map_err(|e| {
            ProbeError::BackendDirUnreadable {
                dir: component.display().to_string(),
                cause: e.to_string(),
            }
        })?;
        if meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(unauditable(
                &component.display().to_string(),
                "is a reparse point (symbolic link, junction, or similar); every component of \
                 the path used to reach the backend directory must be the real object, not a \
                 redirect whose target this audit has not examined",
            ));
        }
    }
    Ok(chain)
}

/// Whether `abs`'s leading path-prefix component names a local disk drive letter — the only prefix
/// shape this module treats as possibly local. Purely syntactic: no filesystem or network access,
/// so this can never itself time out or hang.
///
/// An allowlist, not a denylist. Naming every prefix shape this audit cannot vouch for was tried
/// first and missed `\\?\GLOBALROOT\...` and other verbatim forms that name no drive letter at all.
/// Only `Prefix::Disk` (`C:\...`) and `Prefix::VerbatimDisk` (`\\?\C:\...`) name a local drive
/// letter; a UNC path, a verbatim UNC path (which also covers `\\localhost\C$\...`, a loopback share
/// that merely happens to resolve back to this machine), a `\\.\...` device-namespace path, and
/// every other verbatim prefix are all rejected — the share's or device's own authority decides what
/// the DACL's well-known SIDs mean, and this machine cannot vouch for that.
fn has_local_disk_prefix(abs: &Path) -> bool {
    matches!(
        abs.components().next(),
        Some(Component::Prefix(p))
            if matches!(p.kind(), Prefix::Disk(_) | Prefix::VerbatimDisk(_))
    )
}

/// Refuse to audit any path that is not on a local fixed volume (or RAM disk).
///
/// `abs` must already be absolute. This is the precondition every trust decision in this module
/// depends on: [`Trust`]'s fixed SIDs (`SYSTEM`, `Administrators`, `TrustedInstaller`, the current
/// user) name principals on *this* machine only when this machine's security subsystem issued the
/// descriptor being read. A mapped network drive resolves to the same story as a UNC path — it is
/// still someone else's volume under a local-looking letter — which is why this checks the volume's
/// actual drive type rather than trusting the path's own spelling past the prefix stage.
///
/// A path is rejected here, and the whole audit reported as [`crate::backend::DaclAudit::Unknown`],
/// without a single security descriptor having been read: "could not check" must never be the shape
/// that quietly falls back to a permissive answer.
fn require_local_volume(abs: &Path) -> Result<(), ProbeError> {
    let label = abs.display().to_string();
    if !has_local_disk_prefix(abs) {
        return Err(unauditable(
            &label,
            "is not a local disk path (a network share, device-namespace path, or other \
             non-drive-letter path); its DACL's well-known SIDs are not necessarily this \
             machine's principals, so this audit cannot vouch for them",
        ));
    }

    let wide: Vec<u16> = abs.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    // `GetVolumePathNameW` writes only the drive root (e.g. `C:\`), always far shorter than the
    // input path, so the input's own length (plus a little) is generous headroom for the output.
    let mut root = vec![0u16; wide.len().max(261)];
    let cap =
        u32::try_from(root.len()).map_err(|_| unauditable(&label, "path too long to audit"))?;
    // SAFETY: `wide` is NUL-terminated and outlives the call; `root` owns `cap` writable u16s that
    // the call fills in place.
    if unsafe { GetVolumePathNameW(wide.as_ptr(), root.as_mut_ptr(), cap) } == 0 {
        return Err(with_object(&label, last_error("GetVolumePathNameW")));
    }
    // SAFETY: the successful call above left `root` NUL-terminated.
    let kind = unsafe { GetDriveTypeW(root.as_ptr()) };
    if kind != DRIVE_FIXED && kind != DRIVE_RAMDISK {
        // Covers `DRIVE_REMOTE` (a mapped network drive), `DRIVE_UNKNOWN`, and `DRIVE_NO_ROOT_DIR`
        // alike: none of them is a volume this machine is authoritative for.
        return Err(unauditable(
            &label,
            format!(
                "is on a drive of type {kind} (GetDriveTypeW), not a local fixed volume or RAM disk"
            ),
        ));
    }
    Ok(())
}

/// Resolve `abs` to the real path Windows would use to reach the same file system object: the
/// DOS-form target of `GetFinalPathNameByHandleW`, opened without following any reparse point on
/// `abs` itself.
///
/// A `subst` mapping, a `DefineDosDevice` alias, or an 8.3 short-name spelling can all make a
/// path's own syntax name a different location than the one it actually resolves to — none of
/// those is a reparse point, so [`check_path_chain`] would pass an aliased drive root straight
/// through and every ancestor check after it would walk the alias's own (unrelated, often
/// nonexistent) components instead of the real ones. Resolving once, here, and using the result for
/// every remaining step closes that gap without special-casing any one alias mechanism.
///
/// `FILE_FLAG_OPEN_REPARSE_POINT` is passed so this open cannot itself silently resolve a reparse
/// point on `abs`; [`check_path_chain`], called afterward with this function's result, is what
/// rejects that case explicitly.
fn canonical_path(abs: &Path) -> Result<PathBuf, ProbeError> {
    let label = abs.display().to_string();
    let wide: Vec<u16> = abs.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    // SAFETY: `wide` is NUL-terminated and outlives the call. Desired access 0 only reads metadata;
    // `FILE_FLAG_BACKUP_SEMANTICS` is required to open a directory at all.
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            0,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            null(),
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
            null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(with_object(&label, last_error("CreateFileW")));
    }
    let handle = OwnedHandle(handle);

    // 32,768 UTF-16 units: the documented ceiling a `\\?\`-prefixed path can reach, far more than
    // any resolved form of `abs` could need.
    let mut buf = vec![0u16; 32_768];
    let cap = u32::try_from(buf.len())
        .map_err(|_| unauditable(&label, "path too long to canonicalize"))?;
    // SAFETY: `handle` is open and live for the duration of this call; `buf` owns `cap` writable
    // u16s.
    let n = unsafe {
        GetFinalPathNameByHandleW(handle.0, buf.as_mut_ptr(), cap, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS)
    };
    if n == 0 {
        return Err(with_object(&label, last_error("GetFinalPathNameByHandleW")));
    }
    if n >= cap {
        return Err(unauditable(
            &label,
            "resolved path is longer than this audit can canonicalize",
        ));
    }
    // SAFETY: the call above reported exactly `n` valid u16s written into `buf`.
    let resolved = String::from_utf16_lossy(&buf[..n as usize]);
    let resolved = resolved.strip_prefix(r"\\?\").unwrap_or(&resolved);
    Ok(PathBuf::from(resolved))
}

/// Every grant of `mask` on one object's own security descriptor to a principal outside `trust`,
/// labeled `applies_to` in the result.
///
/// The shared core of [`untrusted_writes`] (directory/file audits, whose mask is fixed by the kind
/// of object) and [`ancestor_grants`] (the path-pinning audit, whose mask is chosen per question —
/// "can this principal replace what's below" is different from "can this principal write into
/// this"). Inherit-only ACEs are always skipped here: they describe what a *future* file created in
/// this container would receive, never a right on the container itself, so a caller that also cares
/// about that (only [`untrusted_writes`] on a directory does) checks it separately.
fn allow_capable(
    sec: &ObjectSecurity,
    trust: &Trust,
    mask: u32,
    applies_to: AuditedObject,
) -> Vec<RawGrant> {
    let mut out = Vec::new();

    // An owner holds WRITE_DAC implicitly, so an untrusted owner is a finding before any ACE is
    // read.
    if let Some(owner) = &sec.owner {
        if !trust.trusts(owner, false) {
            out.push(RawGrant {
                sid: owner.clone(),
                mask: WRITE_DAC,
                applies_to,
                via: GrantVia::UntrustedOwner,
            });
        }
    }

    let aces = match &sec.dacl {
        Dacl::Null => {
            out.push(RawGrant {
                sid: EVERYONE.to_string(),
                mask: GENERIC_ALL,
                applies_to,
                via: GrantVia::NullDacl,
            });
            return out;
        }
        Dacl::Present(aces) => aces,
    };

    for ace in aces {
        if u32::from(ace.flags) & INHERIT_ONLY_ACE != 0 {
            continue;
        }
        let bits = ace.mask & mask;
        if bits != 0 && !trust.trusts(&ace.sid, false) {
            out.push(RawGrant {
                sid: ace.sid.clone(),
                mask: bits,
                applies_to,
                via: GrantVia::Ace,
            });
        }
    }
    out
}

/// Every write-capable grant on one directory or file to a principal outside `trust`.
fn untrusted_writes(sec: &ObjectSecurity, kind: OwnKind, trust: &Trust) -> Vec<RawGrant> {
    let applies_to = kind.audited();
    let mut out = allow_capable(sec, trust, kind.mask(), applies_to);

    // Inherit-only, object-inherit ACEs on a directory are not a right on the directory itself, but
    // every file created there later receives them — including a planted DLL — so they are a
    // directory finding of a different shape than `allow_capable` reports.
    if kind == OwnKind::Directory {
        if let Dacl::Present(aces) = &sec.dacl {
            for ace in aces {
                let flags = u32::from(ace.flags);
                if flags & INHERIT_ONLY_ACE != 0 && flags & OBJECT_INHERIT_ACE != 0 {
                    // `allow_capable` skips every inherit-only ACE outright, so it never evaluated
                    // this one against `DIRECTORY_WRITE_MASK`; filtered against `FILE_WRITE_MASK`
                    // here instead, because what this ACE grants lands on a *future file*, not the
                    // directory that carries it.
                    let bits = ace.mask & FILE_WRITE_MASK;
                    if bits != 0 && !trust.trusts(&ace.sid, true) {
                        out.push(RawGrant {
                            sid: ace.sid.clone(),
                            mask: bits,
                            applies_to: AuditedObject::File,
                            via: GrantVia::InheritableByNewFiles,
                        });
                    }
                }
            }
        }
    }
    out
}

/// Every grant of `mask` on one ancestor directory's or intermediate path component's own security
/// descriptor to a principal outside `trust`, reported as [`AuditedObject::Ancestor`].
///
/// Unlike [`untrusted_writes`], there is no separate inherit-only pass: this function answers "can
/// this principal act on the object itself right now", which inherit-only ACEs never grant, and it
/// is never asked about files created later underneath an ancestor.
fn ancestor_grants(sec: &ObjectSecurity, trust: &Trust, mask: u32) -> Vec<RawGrant> {
    allow_capable(sec, trust, mask, AuditedObject::Ancestor)
}

/// Names of the set bits, spelled for the object kind they apply to.
fn right_names(mask: u32, kind: AuditedObject) -> Vec<&'static str> {
    let specific: &[(u32, &'static str)] = match kind {
        AuditedObject::Directory => &[
            (FILE_ADD_FILE, "FILE_ADD_FILE"),
            (FILE_ADD_SUBDIRECTORY, "FILE_ADD_SUBDIRECTORY"),
            (FILE_DELETE_CHILD, "FILE_DELETE_CHILD"),
        ],
        AuditedObject::File => &[
            (FILE_WRITE_DATA, "FILE_WRITE_DATA"),
            (FILE_APPEND_DATA, "FILE_APPEND_DATA"),
        ],
        // `ANCESTOR_REPLACE_MASK` is the only ancestor mask that carries a bit outside `COMMON`
        // below (`FILE_DELETE_CHILD`); `COMPONENT_REPLACE_MASK`'s bits are all in `COMMON` already.
        AuditedObject::Ancestor => &[(FILE_DELETE_CHILD, "FILE_DELETE_CHILD")],
    };
    const COMMON: [(u32, &str); 6] = [
        (DELETE, "DELETE"),
        (WRITE_DAC, "WRITE_DAC"),
        (WRITE_OWNER, "WRITE_OWNER"),
        (GENERIC_WRITE, "GENERIC_WRITE"),
        (GENERIC_ALL, "GENERIC_ALL"),
        (MAXIMUM_ALLOWED, "MAXIMUM_ALLOWED"),
    ];
    specific
        .iter()
        .chain(COMMON.iter())
        .filter(|(bit, _)| mask & bit != 0)
        .map(|&(_, name)| name)
        .collect()
}

/// Interpret one ACE's bytes. `Ok(None)` means the ACE cannot grant access and is ignored.
///
/// Errors on anything that could grant access but is not understood, so an unfamiliar ACE makes
/// the audit `Unknown` instead of silently not counting.
fn parse_ace(bytes: &[u8]) -> Result<Option<AllowAce>, String> {
    let Some(&[ace_type, flags, _, _]) = bytes.first_chunk::<4>() else {
        return Err(format!("ACE of {} bytes is shorter than its header", bytes.len()));
    };
    let sid_offset = match u32::from(ace_type) {
        ACCESS_ALLOWED_ACE_TYPE | ACCESS_ALLOWED_CALLBACK_ACE_TYPE => 8,
        ACCESS_ALLOWED_OBJECT_ACE_TYPE | ACCESS_ALLOWED_CALLBACK_OBJECT_ACE_TYPE => {
            // The two GUIDs are optional and the SID moves up when either is absent; reading at a
            // fixed offset would attribute the grant to garbage.
            let object_flags = read_u32(bytes, 8)
                .ok_or_else(|| "object ACE truncated before its flags".to_string())?;
            let mut offset = 12;
            if object_flags & ACE_OBJECT_TYPE_PRESENT != 0 {
                offset += 16;
            }
            if object_flags & ACE_INHERITED_OBJECT_TYPE_PRESENT != 0 {
                offset += 16;
            }
            offset
        }
        ACCESS_ALLOWED_COMPOUND_ACE_TYPE => {
            return Err("compound (impersonation) ACEs are not interpreted".to_string());
        }
        // Deny entries only restrict; ignoring them can over-report but never under-report.
        // Audit, alarm, label, attribute, policy, trust-label and filter entries grant nothing.
        ACCESS_DENIED_ACE_TYPE
        | ACCESS_DENIED_OBJECT_ACE_TYPE
        | ACCESS_DENIED_CALLBACK_ACE_TYPE
        | ACCESS_DENIED_CALLBACK_OBJECT_ACE_TYPE
        | SYSTEM_AUDIT_ACE_TYPE
        | SYSTEM_ALARM_ACE_TYPE
        | SYSTEM_AUDIT_OBJECT_ACE_TYPE
        | SYSTEM_ALARM_OBJECT_ACE_TYPE
        | SYSTEM_AUDIT_CALLBACK_ACE_TYPE
        | SYSTEM_ALARM_CALLBACK_ACE_TYPE
        | SYSTEM_AUDIT_CALLBACK_OBJECT_ACE_TYPE
        | SYSTEM_ALARM_CALLBACK_OBJECT_ACE_TYPE
        | SYSTEM_MANDATORY_LABEL_ACE_TYPE
        | SYSTEM_RESOURCE_ATTRIBUTE_ACE_TYPE
        | SYSTEM_SCOPED_POLICY_ID_ACE_TYPE
        | SYSTEM_PROCESS_TRUST_LABEL_ACE_TYPE
        | SYSTEM_ACCESS_FILTER_ACE_TYPE => return Ok(None),
        other => return Err(format!("unrecognized ACE type {other:#x}")),
    };
    let mask = read_u32(bytes, 4).ok_or_else(|| "ACE truncated before its mask".to_string())?;
    let sid_bytes = bytes
        .get(sid_offset..)
        .ok_or_else(|| "ACE truncated before its SID".to_string())?;
    Ok(Some(AllowAce {
        flags,
        mask,
        sid: sid_string(sid_bytes)?,
    }))
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let chunk = bytes.get(offset..)?.first_chunk::<4>()?;
    Some(u32::from_le_bytes(*chunk))
}

/// Render a binary SID in `S-1-...` form, bounds-checked against `bytes`.
///
/// Done in Rust rather than with `ConvertSidToStringSidW` so trust comparisons need no allocation
/// to free and so a SID claiming more sub-authorities than the ACE holds is caught here.
fn sid_string(bytes: &[u8]) -> Result<String, String> {
    let Some(&[revision, count, a0, a1, a2, a3, a4, a5]) = bytes.first_chunk::<8>() else {
        return Err("SID truncated before its header".to_string());
    };
    if revision != 1 {
        return Err(format!("SID revision {revision} is not 1"));
    }
    let count = usize::from(count);
    if count > SID_MAX_SUB_AUTHORITIES {
        return Err(format!("SID claims {count} sub-authorities, above the maximum of 15"));
    }
    let subs = bytes
        .get(8..8 + 4 * count)
        .ok_or_else(|| format!("SID claims {count} sub-authorities but is truncated"))?;
    // The authority is a 48-bit big-endian value; Windows prints it in hex once it exceeds 32 bits —
    // uppercase and *without* zero-padding, exactly as `ConvertSidToStringSidW` renders it (verified
    // against that API in `sid_rendering_matches_convert_sid_to_string_sid`), since
    // `WritableGrant::sid` is documented as the unambiguous form and must match what icacls and
    // every other Windows tool prints for the same bytes.
    let authority = u64::from_be_bytes([0, 0, a0, a1, a2, a3, a4, a5]);
    let mut out = if authority >> 32 == 0 {
        format!("S-1-{authority}")
    } else {
        format!("S-1-0x{authority:X}")
    };
    for sub in subs.chunks_exact(4) {
        let value = u32::from_le_bytes([sub[0], sub[1], sub[2], sub[3]]);
        out.push('-');
        out.push_str(&value.to_string());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Win32
// ---------------------------------------------------------------------------------------------

/// Memory the system allocated for the caller with `LocalAlloc`, freed exactly once on drop.
struct LocalMemory(*mut c_void);

impl Drop for LocalMemory {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the pointer came from an API documented to allocate with LocalAlloc and hand
            // ownership to the caller, this wrapper is its only owner, and drop runs once.
            unsafe { LocalFree(self.0) };
        }
    }
}

/// A kernel handle closed exactly once on drop.
struct OwnedHandle(HANDLE);

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        // SAFETY: the handle was opened by this module, is owned solely by this wrapper, and drop
        // runs once. A close failure leaves nothing to recover.
        unsafe { CloseHandle(self.0) };
    }
}

fn last_error(call: &'static str) -> ProbeError {
    // SAFETY: reads the calling thread's last-error value; every caller invokes this immediately
    // after the failed call it reports.
    let code = unsafe { GetLastError() };
    ProbeError::Win32 { call, code }
}

fn unauditable(object: &str, detail: impl Into<String>) -> ProbeError {
    ProbeError::UnauditableSecurityDescriptor {
        object: object.to_string(),
        detail: detail.into(),
    }
}

/// Attach `object` to a failure that does not otherwise name what it was examining — a bare
/// [`ProbeError::Win32`], in every call site that uses this.
///
/// Folded into a single [`ProbeError::UnauditableSecurityDescriptor`] rather than a nested
/// container: every error this module ultimately surfaces is rendered once, as one line in
/// [`crate::backend::DaclAudit::Unknown`], so there is no structured consumer that would need the
/// original variant preserved underneath. Callers apply this only to errors that do not already
/// name an object — wrapping one that does (e.g. the result of another call in this module) would
/// print the object twice.
fn with_object(object: &str, err: ProbeError) -> ProbeError {
    unauditable(object, err.to_string())
}

/// Owner and DACL of a file or directory on disk.
fn object_security(path: &Path) -> Result<ObjectSecurity, ProbeError> {
    let label = path.display().to_string();
    let wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    if wide.contains(&0) {
        return Err(unauditable(&label, "path contains an interior NUL"));
    }
    let wide: Vec<u16> = wide.into_iter().chain(std::iter::once(0)).collect();

    let mut owner: PSID = null_mut();
    let mut dacl: *mut ACL = null_mut();
    let mut descriptor: PSECURITY_DESCRIPTOR = null_mut();
    // SAFETY: `wide` is NUL-terminated and outlives the call; every out-pointer is a live local of
    // the type the API writes. `owner` and `dacl` point into `descriptor` and are not used after
    // it is freed — they are re-derived from it below instead.
    let status = unsafe {
        GetNamedSecurityInfoW(
            wide.as_ptr(),
            SE_FILE_OBJECT,
            OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            &mut owner,
            null_mut(),
            &mut dacl,
            null_mut(),
            &mut descriptor,
        )
    };
    // Owned before the status check so a failure that still allocated cannot leak.
    let descriptor = LocalMemory(descriptor);
    if status != ERROR_SUCCESS {
        return Err(with_object(
            &label,
            ProbeError::Win32 {
                call: "GetNamedSecurityInfoW",
                code: status,
            },
        ));
    }
    if descriptor.0.is_null() {
        return Err(unauditable(&label, "no security descriptor was returned"));
    }
    read_descriptor(&descriptor, &label)
}

/// Build a security descriptor from SDDL, for tests that must not depend on any file's ACL.
#[cfg(test)]
fn sddl_descriptor(sddl: &str) -> Result<LocalMemory, ProbeError> {
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
    };
    let wide: Vec<u16> = sddl.encode_utf16().chain(std::iter::once(0)).collect();
    let mut descriptor: PSECURITY_DESCRIPTOR = null_mut();
    // SAFETY: `wide` is NUL-terminated and outlives the call; `descriptor` is a live local, and the
    // optional size out-parameter is documented as nullable.
    let ok = unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            wide.as_ptr(),
            SDDL_REVISION_1,
            &mut descriptor,
            null_mut(),
        )
    };
    let descriptor = LocalMemory(descriptor);
    if ok == 0 {
        return Err(last_error("ConvertStringSecurityDescriptorToSecurityDescriptorW"));
    }
    Ok(descriptor)
}

/// Replace a real path's DACL with one built from `sddl`, for tests that must exercise the
/// on-disk read path (`GetNamedSecurityInfoW` in [`object_security`]) rather than only the
/// in-memory ACE walk that [`sddl_descriptor`] feeds directly.
///
/// Sets the DACL only — never the owner — so a test directory or file stays owned by whoever
/// created it (the current user, which the audit already trusts) no matter what `sddl` contains.
/// Passes `PROTECTED_DACL_SECURITY_INFORMATION` so `sddl` becomes the object's *entire* DACL:
/// without it, `SetNamedSecurityInfoW` keeps every existing inherited ACE and only adds `sddl`'s
/// entries as new explicit ones, which under an inherited ACE this development machine's own
/// `%TEMP%` carries would leave old grants in place underneath a supposedly clean baseline.
#[cfg(test)]
pub(super) fn set_dacl_from_sddl(path: &Path, sddl: &str) -> Result<(), ProbeError> {
    use windows_sys::Win32::Security::Authorization::SetNamedSecurityInfoW;
    use windows_sys::Win32::Security::PROTECTED_DACL_SECURITY_INFORMATION;

    let descriptor = sddl_descriptor(sddl)?;
    let mut present = 0;
    let mut acl: *mut ACL = null_mut();
    let mut defaulted = 0;
    // SAFETY: `descriptor` is a live, valid security descriptor built above for this call only.
    if unsafe { GetSecurityDescriptorDacl(descriptor.0, &mut present, &mut acl, &mut defaulted) }
        == 0
    {
        return Err(last_error("GetSecurityDescriptorDacl"));
    }

    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    // SAFETY: `wide` is NUL-terminated and outlives the call. `acl` points into `descriptor`, which
    // is still alive (dropped only at the end of this function); `SetNamedSecurityInfoW` reads and
    // copies the ACL into the object's security descriptor rather than retaining the pointer.
    let status = unsafe {
        SetNamedSecurityInfoW(
            wide.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            null_mut(),
            null_mut(),
            acl,
            null_mut(),
        )
    };
    if status != ERROR_SUCCESS {
        return Err(ProbeError::Win32 {
            call: "SetNamedSecurityInfoW",
            code: status,
        });
    }
    Ok(())
}

/// Create a directory junction (an NTFS mount-point reparse point) at `link`, pointing at `target`.
///
/// Test-only: production code in this module only ever *detects* reparse points
/// ([`check_path_chain`], [`classify_entry`]), never creates one. This exists because a junction —
/// unlike a symbolic link — needs no privilege and no Developer Mode to create, which is exactly why
/// it is the reparse type an unprivileged attacker can actually plant, and why the path-pinning
/// audit has to be tested against a real one rather than only a symlink that may not be creatable at
/// all on the machine running the test.
#[cfg(test)]
pub(super) fn create_junction(link: &Path, target: &Path) -> Result<(), ProbeError> {
    use windows_sys::Win32::System::Ioctl::FSCTL_SET_REPARSE_POINT;
    use windows_sys::Win32::System::SystemServices::IO_REPARSE_TAG_MOUNT_POINT;
    use windows_sys::Win32::System::IO::DeviceIoControl;

    let label = link.display().to_string();
    std::fs::create_dir(link)
        .map_err(|e| unauditable(&label, format!("could not create the link's own directory: {e}")))?;

    // The substitute name inside a reparse point must be an NT path (`\??\C:\...`), not the Win32
    // path this function otherwise deals in.
    let target_abs = std::path::absolute(target)
        .map_err(|e| unauditable(&label, format!("could not make junction target absolute: {e}")))?;
    let sub: Vec<u16> = format!(r"\??\{}", target_abs.display()).encode_utf16().collect();
    let sub_bytes = u16::try_from(sub.len() * 2)
        .map_err(|_| unauditable(&label, "junction target path is too long"))?;

    // A minimal MOUNT_POINT_REPARSE_BUFFER: a substitute name, a NUL terminator pair, and an empty
    // print name (offset past the substitute name and terminator, length zero — Explorer never sees
    // this junction, so there is nothing useful to print). Lengths throughout are in bytes, not
    // UTF-16 code units, and `ReparseDataLength` excludes the 8-byte `REPARSE_DATA_BUFFER` header.
    let mut buf = Vec::<u8>::new();
    buf.extend_from_slice(&IO_REPARSE_TAG_MOUNT_POINT.to_le_bytes());
    buf.extend_from_slice(&(sub_bytes + 12).to_le_bytes()); // ReparseDataLength
    buf.extend_from_slice(&0u16.to_le_bytes()); // Reserved
    for field in [0u16, sub_bytes, sub_bytes + 2, 0u16] {
        // SubstituteNameOffset, SubstituteNameLength, PrintNameOffset, PrintNameLength.
        buf.extend_from_slice(&field.to_le_bytes());
    }
    for unit in &sub {
        buf.extend_from_slice(&unit.to_le_bytes());
    }
    buf.extend_from_slice(&[0, 0, 0, 0]); // Substitute- and print-name NUL terminators.

    let wide_link: Vec<u16> = link.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    // SAFETY: `wide_link` is NUL-terminated and outlives the call. `FILE_FLAG_OPEN_REPARSE_POINT |
    // FILE_FLAG_BACKUP_SEMANTICS` is required for `FSCTL_SET_REPARSE_POINT` to succeed on a
    // directory handle; without `FILE_FLAG_BACKUP_SEMANTICS`, opening a directory at all fails.
    let handle = unsafe {
        CreateFileW(
            wide_link.as_ptr(),
            GENERIC_WRITE,
            0,
            null(),
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
            null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(last_error("CreateFileW"));
    }
    let handle = OwnedHandle(handle);

    let mut returned: u32 = 0;
    let len = u32::try_from(buf.len()).map_err(|_| unauditable(&label, "reparse buffer too long"))?;
    // SAFETY: `handle` was just opened with the flags this FSCTL requires; `buf` is a valid,
    // initialized input buffer of `len` bytes that outlives the call; no output buffer is requested,
    // so the null/0 output arguments are exactly what the API documents as acceptable.
    let ok = unsafe {
        DeviceIoControl(
            handle.0,
            FSCTL_SET_REPARSE_POINT,
            buf.as_ptr().cast(),
            len,
            null_mut(),
            0,
            &mut returned,
            null_mut(),
        )
    };
    if ok == 0 {
        return Err(last_error("DeviceIoControl(FSCTL_SET_REPARSE_POINT)"));
    }
    Ok(())
}

/// Copy the owner and the access-allowed ACEs out of a security descriptor.
fn read_descriptor(descriptor: &LocalMemory, label: &str) -> Result<ObjectSecurity, ProbeError> {
    let mut owner: PSID = null_mut();
    let mut defaulted = 0;
    // SAFETY: `descriptor` owns a valid security descriptor for the duration of this borrow; the
    // out-pointers are live locals. The returned SID pointer aliases the descriptor's memory.
    if unsafe { GetSecurityDescriptorOwner(descriptor.0, &mut owner, &mut defaulted) } == 0 {
        return Err(with_object(label, last_error("GetSecurityDescriptorOwner")));
    }
    let owner = if owner.is_null() {
        None
    } else {
        Some(sid_at(owner, label)?)
    };

    let mut present = 0;
    let mut acl: *mut ACL = null_mut();
    // SAFETY: as above; the returned ACL pointer aliases the descriptor's memory, which stays alive
    // because `descriptor` is borrowed for this whole function.
    if unsafe { GetSecurityDescriptorDacl(descriptor.0, &mut present, &mut acl, &mut defaulted) }
        == 0
    {
        return Err(with_object(label, last_error("GetSecurityDescriptorDacl")));
    }
    // No DACL present and a present-but-NULL DACL mean the same thing to an access check.
    let dacl = if present == 0 || acl.is_null() {
        Dacl::Null
    } else {
        Dacl::Present(allow_aces(acl, label)?)
    };
    Ok(ObjectSecurity { owner, dacl })
}

/// Walk a DACL with `GetAce`, keeping the entries that can grant access.
fn allow_aces(acl: *const ACL, label: &str) -> Result<Vec<AllowAce>, ProbeError> {
    // SAFETY: `acl` is non-null and points into a live security descriptor; IsValidAcl reads only
    // the ACL it validates.
    if unsafe { IsValidAcl(acl) } == 0 {
        return Err(unauditable(label, "DACL failed IsValidAcl"));
    }
    // SAFETY: a valid ACL starts with an ACL header. `read_unaligned` makes no assumption about
    // where inside the descriptor buffer the ACL was placed.
    let header = unsafe { acl.read_unaligned() };
    let start = acl as usize;
    let end = start + usize::from(header.AclSize);

    let mut aces = Vec::with_capacity(usize::from(header.AceCount));
    for index in 0..u32::from(header.AceCount) {
        let mut ace: *mut c_void = null_mut();
        // SAFETY: `acl` is valid and `index` is below its AceCount; `ace` is a live local.
        if unsafe { GetAce(acl, index, &mut ace) } == 0 {
            return Err(with_object(label, last_error("GetAce")));
        }
        let at = ace as usize;
        // Never trust AceSize beyond the ACL that contains it: a slice past AclSize would read
        // whatever follows the DACL in the descriptor buffer.
        if at < start || at + std::mem::size_of::<ACE_HEADER>() > end {
            return Err(unauditable(label, format!("ACE {index} lies outside its ACL")));
        }
        // SAFETY: the header's four bytes were just bounds-checked to lie inside the ACL.
        let ace_header = unsafe { ace.cast::<ACE_HEADER>().read_unaligned() };
        let size = usize::from(ace_header.AceSize);
        if size < std::mem::size_of::<ACE_HEADER>() || at + size > end {
            return Err(unauditable(
                label,
                format!("ACE {index} declares size {size}, which does not fit its ACL"),
            ));
        }
        // SAFETY: `size` bytes from `ace` were just checked to lie inside the ACL, which is
        // initialized memory owned by the borrowed descriptor; u8 has no alignment requirement.
        let bytes = unsafe { std::slice::from_raw_parts(ace.cast::<u8>(), size) };
        match parse_ace(bytes) {
            Ok(Some(allow)) => aces.push(allow),
            Ok(None) => {}
            Err(detail) => return Err(unauditable(label, format!("ACE {index}: {detail}"))),
        }
    }
    Ok(aces)
}

/// Copy a SID the system handed back by pointer into its string form.
fn sid_at(sid: PSID, label: &str) -> Result<String, ProbeError> {
    // SAFETY: `sid` is non-null and points into memory the caller keeps alive; IsValidSid reads
    // only the structure it validates.
    if unsafe { IsValidSid(sid) } == 0 {
        return Err(unauditable(label, "SID failed IsValidSid"));
    }
    // SAFETY: the SID was just validated, so GetLengthSid returns its exact length.
    let len = unsafe { GetLengthSid(sid) } as usize;
    // SAFETY: a valid SID of `len` bytes lives at `sid` for the caller's borrow; u8 has no
    // alignment requirement.
    let bytes = unsafe { std::slice::from_raw_parts(sid.cast::<u8>(), len) };
    sid_string(bytes).map_err(|detail| unauditable(label, detail))
}

/// SID of the user the current process token represents.
///
/// The process token rather than a thread token: Cerebro spawns backends from its own identity,
/// and the grant that matters is the one that user already holds.
fn current_user_sid() -> Result<String, ProbeError> {
    let mut raw: HANDLE = null_mut();
    // SAFETY: GetCurrentProcess returns a pseudo-handle that needs no closing; `raw` is a live
    // local that receives a real token handle, owned immediately below.
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut raw) } == 0 {
        return Err(with_object("process token", last_error("OpenProcessToken")));
    }
    let token = OwnedHandle(raw);

    let mut needed: u32 = 0;
    // SAFETY: the documented sizing call; a null buffer with length 0 is allowed and the API writes
    // the required size into `needed`, a live local.
    let sized = unsafe { GetTokenInformation(token.0, TokenUser, null_mut(), 0, &mut needed) };
    if sized == 0 {
        // SAFETY: reading last-error immediately after the failed sizing call.
        let code = unsafe { GetLastError() };
        if code != ERROR_INSUFFICIENT_BUFFER {
            return Err(with_object(
                "process token",
                ProbeError::Win32 {
                    call: "GetTokenInformation",
                    code,
                },
            ));
        }
    }
    // u64 storage so the TOKEN_USER (which begins with a pointer) is correctly aligned.
    let words = (needed as usize)
        .max(std::mem::size_of::<TOKEN_USER>())
        .div_ceil(std::mem::size_of::<u64>());
    let mut storage = vec![0u64; words];
    let capacity = u32::try_from(words * std::mem::size_of::<u64>())
        .map_err(|_| unauditable("process token", "TokenUser size overflows u32"))?;
    // SAFETY: `storage` owns `capacity` writable, 8-byte-aligned bytes and the length tells the
    // API exactly that; `needed` is a live local.
    let ok = unsafe {
        GetTokenInformation(
            token.0,
            TokenUser,
            storage.as_mut_ptr().cast(),
            capacity,
            &mut needed,
        )
    };
    if ok == 0 {
        return Err(with_object("process token", last_error("GetTokenInformation")));
    }
    // SAFETY: on success the buffer begins with a TOKEN_USER, and the storage is aligned for it.
    let user = unsafe { storage.as_ptr().cast::<TOKEN_USER>().read() };
    if user.User.Sid.is_null() {
        return Err(unauditable("process token", "TokenUser carried no SID"));
    }
    // The SID points into `storage`, which is still alive here.
    sid_at(user.User.Sid, "process token")
}

/// Exposes [`current_user_sid`] to `backend`'s on-disk test module, which needs the real value to
/// pin a baseline or restoration ACL to an account that can still list, read, and delete what it
/// creates — a baseline pinned to `SYSTEM`/`Administrators` only would leave a non-elevated test
/// runner with no access to its own temp directory the moment it applied it.
#[cfg(test)]
pub(super) fn test_current_user_sid() -> Result<String, ProbeError> {
    current_user_sid()
}

/// `DOMAIN\name` for a SID string, or `None` when it does not resolve — a deleted account, a
/// capability SID, or a foreign domain SID this machine cannot reach.
///
/// For display only. Every trust decision in this module is made on the SID string alone; nothing
/// here feeds back into [`Trust`].
fn account_name(sid: &str) -> Option<String> {
    let wide: Vec<u16> = sid.encode_utf16().chain(std::iter::once(0)).collect();
    let mut raw: PSID = null_mut();
    // SAFETY: `wide` is NUL-terminated and outlives the call; `raw` is a live local that receives a
    // LocalAlloc'd SID, owned immediately below.
    let ok = unsafe { ConvertStringSidToSidW(wide.as_ptr(), &mut raw) };
    let binary = LocalMemory(raw);
    if ok == 0 || binary.0.is_null() {
        return None;
    }

    let mut name_len: u32 = 0;
    let mut domain_len: u32 = 0;
    let mut sizing_use: SID_NAME_USE = 0;
    // SAFETY: the documented sizing call; null buffers with zero lengths are allowed and the API
    // writes the required lengths (including the terminator) into the live locals. `sizing_use` is
    // never read: some documented behavior lets the account-type out-param come back unset when the
    // call's only job is reporting buffer sizes, so only the second call's value below is trusted.
    unsafe {
        LookupAccountSidW(
            null(),
            binary.0,
            null_mut(),
            &mut name_len,
            null_mut(),
            &mut domain_len,
            &mut sizing_use,
        )
    };
    if name_len == 0 {
        return None;
    }
    let mut name = vec![0u16; name_len as usize];
    let mut domain = vec![0u16; (domain_len as usize).max(1)];
    let mut domain_cap = u32::try_from(domain.len()).ok()?;
    let mut use_kind: SID_NAME_USE = 0;
    // SAFETY: each buffer owns exactly as many u16s as the length passed alongside it, and every
    // out-pointer is a live local.
    let ok = unsafe {
        LookupAccountSidW(
            null(),
            binary.0,
            name.as_mut_ptr(),
            &mut name_len,
            domain.as_mut_ptr(),
            &mut domain_cap,
            &mut use_kind,
        )
    };
    if ok == 0 {
        return None;
    }
    // On success the lengths exclude the terminator.
    let name = String::from_utf16_lossy(name.get(..name_len as usize)?);
    let domain = String::from_utf16_lossy(domain.get(..domain_cap as usize)?);
    Some(if domain.is_empty() { name } else { format!("{domain}\\{name}") })
}

#[cfg(test)]
mod tests {
    use super::*;

    const USER: &str = "S-1-5-21-111-222-333-1001";
    const OTHER_USER: &str = "S-1-5-21-111-222-333-1002";
    const USERS: &str = "S-1-5-32-545";

    fn security(sddl: &str) -> ObjectSecurity {
        let descriptor = sddl_descriptor(sddl).expect("test SDDL must parse");
        read_descriptor(&descriptor, "<sddl>").expect("test descriptor must be readable")
    }

    /// Trust as if the process ran as `USER` and the directory were owned by `USER`, so no test
    /// depends on who runs it.
    fn trust() -> Trust {
        Trust::new(USER.to_string(), Some(USER))
    }

    fn findings(sddl: &str, kind: OwnKind) -> Vec<RawGrant> {
        untrusted_writes(&security(sddl), kind, &trust())
    }

    fn findings_masked(sddl: &str, mask: u32) -> Vec<RawGrant> {
        ancestor_grants(&security(sddl), &trust(), mask)
    }

    #[test]
    fn null_dacl_is_everyone_full_control() {
        let sec = security(&format!("O:{USER}D:NO_ACCESS_CONTROL"));
        assert_eq!(sec.dacl, Dacl::Null);
        let got = untrusted_writes(&sec, OwnKind::Directory, &trust());
        assert_eq!(
            got,
            vec![RawGrant {
                sid: EVERYONE.into(),
                mask: GENERIC_ALL,
                applies_to: AuditedObject::Directory,
                via: GrantVia::NullDacl,
            }]
        );
    }

    #[test]
    fn empty_dacl_grants_nobody_and_is_not_confused_with_null() {
        let sec = security(&format!("O:{USER}D:"));
        // The distinction that matters: NULL means everyone, empty means no one.
        assert_eq!(sec.dacl, Dacl::Present(Vec::new()));
        assert!(untrusted_writes(&sec, OwnKind::Directory, &trust()).is_empty());
    }

    #[test]
    fn owner_system_admins_and_current_user_are_trusted() {
        let sddl = format!("O:{USER}D:PAI(A;OICI;FA;;;{USER})(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)");
        assert!(findings(&sddl, OwnKind::Directory).is_empty());
        assert!(findings(&sddl, OwnKind::File).is_empty());
    }

    #[test]
    fn owner_is_trusted_only_while_it_is_a_fixed_principal() {
        // Same full-control ACE for a SID that is neither the process user nor anything well known.
        // It is trusted only while it is one of the fixed principals or the current user — never
        // merely because it happens to own the object.
        let sddl = format!("O:{OTHER_USER}D:(A;;FA;;;{OTHER_USER})");
        let as_owner = Trust::new(USER.into(), Some(OTHER_USER));
        let got = untrusted_writes(&security(&sddl), OwnKind::Directory, &as_owner);
        assert!(got.iter().any(|g| g.sid == OTHER_USER && g.via == GrantVia::Ace));
        assert!(got
            .iter()
            .any(|g| g.sid == OTHER_USER && g.via == GrantVia::UntrustedOwner && g.mask == WRITE_DAC));

        let not_owner = Trust::new(USER.into(), Some(USER));
        let got = untrusted_writes(&security(&sddl), OwnKind::File, &not_owner);
        assert!(got.iter().any(|g| g.sid == OTHER_USER && g.via == GrantVia::Ace));
        assert!(got
            .iter()
            .any(|g| g.sid == OTHER_USER && g.via == GrantVia::UntrustedOwner && g.mask == WRITE_DAC));
    }

    #[test]
    fn another_user_owning_the_directory_is_reported_as_owner() {
        // Ownership by *some* single account is not trust: an owner outside the fixed set holds
        // implicit WRITE_DAC and can re-ACL the directory at will, regardless of how "ordinary" the
        // account looks.
        let sddl = format!("O:{OTHER_USER}D:(A;;FA;;;SY)");
        let sec = security(&sddl);
        let t = Trust::new(USER.into(), sec.owner.as_deref());
        assert_eq!(t.directory_owner, None);
        let got = untrusted_writes(&sec, OwnKind::Directory, &t);
        assert!(
            got.iter().any(|g| g.sid == OTHER_USER && g.via == GrantVia::UntrustedOwner),
            "{got:?}"
        );
    }

    #[test]
    fn group_owner_ace_is_reported_even_though_it_names_the_owner() {
        // Regression guard: a group that owns the directory and also holds an explicit ACE must
        // produce *two* findings (the ownership itself, and the ACE), never just one — trusting the
        // owner's own ACE because "it's the owner's ACE" would silently drop the second.
        let group = "S-1-5-21-1-2-3-3080";
        let sddl = format!("O:{group}D:(A;;FA;;;{group})");
        let sec = security(&sddl);
        let t = Trust::new(USER.into(), sec.owner.as_deref());
        let got = untrusted_writes(&sec, OwnKind::Directory, &t);
        assert!(got.iter().any(|g| g.sid == group && g.via == GrantVia::UntrustedOwner), "{got:?}");
        assert!(got.iter().any(|g| g.sid == group && g.via == GrantVia::Ace), "{got:?}");
    }

    #[test]
    fn users_write_is_reported_with_only_the_write_capable_bits() {
        // icacls "(W)": SYNCHRONIZE | FILE_WRITE_DATA | FILE_APPEND_DATA | FILE_WRITE_EA |
        // FILE_WRITE_ATTRIBUTES.
        let got = findings(&format!("O:{USER}D:(A;;0x100116;;;BU)"), OwnKind::File);
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].sid, USERS);
        assert_eq!(got[0].mask, FILE_WRITE_DATA | FILE_APPEND_DATA);
        assert_eq!(
            right_names(got[0].mask, AuditedObject::File),
            vec!["FILE_WRITE_DATA", "FILE_APPEND_DATA"]
        );
        // The same bits on a directory are spelled as what they allow there.
        assert_eq!(
            right_names(got[0].mask, AuditedObject::Directory),
            vec!["FILE_ADD_FILE", "FILE_ADD_SUBDIRECTORY"]
        );
    }

    #[test]
    fn modify_rights_as_granted_on_the_development_build_are_reported() {
        // 0x1301ff is exactly the mask CodexSandboxUsers holds on llama.cpp-b10751.
        let got = findings(&format!("O:{USER}D:AI(A;OICIID;0x1301ff;;;BU)"), OwnKind::Directory);
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(
            right_names(got[0].mask, AuditedObject::Directory),
            vec!["FILE_ADD_FILE", "FILE_ADD_SUBDIRECTORY", "FILE_DELETE_CHILD", "DELETE"]
        );
    }

    #[test]
    fn write_ea_and_write_attributes_alone_are_not_write_access() {
        // FILE_GENERIC_READ (0x120089) | FILE_GENERIC_EXECUTE (0x1200a0) | FILE_WRITE_EA (0x10) |
        // FILE_WRITE_ATTRIBUTES (0x100) = 0x1201b9. Neither write bit is FILE_WRITE_DATA or
        // FILE_APPEND_DATA, so this must not be reported as write access.
        let sddl = format!("O:{USER}D:(A;;0x1201b9;;;BU)");
        assert!(findings(&sddl, OwnKind::File).is_empty());
        assert!(findings(&sddl, OwnKind::Directory).is_empty());
    }

    #[test]
    fn delete_child_counts_on_a_directory_but_means_nothing_on_a_file() {
        let sddl = format!("O:{USER}D:(A;;0x40;;;BU)");
        let dir = findings(&sddl, OwnKind::Directory);
        assert_eq!(dir.len(), 1);
        assert_eq!(right_names(dir[0].mask, AuditedObject::Directory), vec!["FILE_DELETE_CHILD"]);
        assert!(findings(&sddl, OwnKind::File).is_empty());
    }

    #[test]
    fn inherit_only_object_inherit_ace_is_unsafe_for_the_directory() {
        let sddl = format!("O:{USER}D:(A;OIIO;0x100116;;;BU)");
        let got = findings(&sddl, OwnKind::Directory);
        assert_eq!(
            got,
            vec![RawGrant {
                sid: USERS.into(),
                mask: FILE_WRITE_DATA | FILE_APPEND_DATA,
                applies_to: AuditedObject::File,
                via: GrantVia::InheritableByNewFiles,
            }]
        );
        // On a file an inherit-only ACE applies to nothing.
        assert!(findings(&sddl, OwnKind::File).is_empty());
    }

    #[test]
    fn container_inherit_only_ace_does_not_reach_files() {
        let sddl = format!("O:{USER}D:(A;CIIO;FA;;;BU)");
        assert!(findings(&sddl, OwnKind::Directory).is_empty());
    }

    #[test]
    fn creator_owner_is_trusted_only_on_inheritable_aces() {
        let inheritable = format!("O:{USER}D:(A;OICIIO;FA;;;CO)");
        assert!(findings(&inheritable, OwnKind::Directory).is_empty());
        let effective = format!("O:{USER}D:(A;;FA;;;CO)");
        let got = findings(&effective, OwnKind::Directory);
        assert!(got.iter().any(|g| g.sid == CREATOR_OWNER), "{got:?}");
    }

    #[test]
    fn trusted_installer_is_trusted_but_a_neighbouring_service_sid_is_not() {
        let ti = format!("O:{USER}D:(A;;FA;;;{TRUSTED_INSTALLER})");
        assert!(findings(&ti, OwnKind::File).is_empty());
        // Last sub-authority off by one: exact SID comparison, not prefix matching.
        let near = "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478465";
        let got = findings(&format!("O:{USER}D:(A;;FA;;;{near})"), OwnKind::File);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].sid, near);
    }

    #[test]
    fn deny_aces_never_become_grants_and_do_not_mask_an_allow() {
        // A deny for Everyone must not be reported as though it granted Everyone anything.
        let deny_only = format!("O:{USER}D:(D;;FA;;;WD)(A;;FA;;;SY)");
        assert!(findings(&deny_only, OwnKind::Directory).is_empty());
        // Denies are not subtracted, so an allow behind a deny is still reported (conservative).
        let both = format!("O:{USER}D:(D;;FA;;;BU)(A;;FA;;;BU)");
        let got = findings(&both, OwnKind::File);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].sid, USERS);
    }

    #[test]
    fn generic_rights_are_write_capable_only_when_they_write() {
        let all = findings(&format!("O:{USER}D:(A;OICIIO;GA;;;AU)"), OwnKind::Directory);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].sid, "S-1-5-11");
        assert_eq!(right_names(all[0].mask, AuditedObject::File), vec!["GENERIC_ALL"]);

        let write = findings(&format!("O:{USER}D:(A;;GW;;;WD)"), OwnKind::File);
        assert_eq!(write.len(), 1);
        assert_eq!(write[0].mask, GENERIC_WRITE);

        assert!(findings(&format!("O:{USER}D:(A;;GRGX;;;WD)"), OwnKind::File).is_empty());
    }

    #[test]
    fn capability_sids_are_untrusted() {
        // ALL APPLICATION PACKAGES.
        let got = findings(&format!("O:{USER}D:(A;;FA;;;AC)"), OwnKind::File);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].sid, "S-1-15-2-1");
    }

    #[test]
    fn object_ace_sid_is_read_after_the_guids_it_carries() {
        // With an object-type GUID present the SID starts 16 bytes later. Reading it at the fixed
        // offset would yield a different or invalid SID.
        let sddl = format!(
            "O:{USER}D:(OA;;FA;bf967aba-0de6-11d0-a285-00aa003049e2;;BU)(OA;;FA;;;BG)"
        );
        let got = findings(&sddl, OwnKind::File);
        let sids: Vec<&str> = got.iter().map(|g| g.sid.as_str()).collect();
        assert_eq!(sids, vec![USERS, "S-1-5-32-546"]);
    }

    #[test]
    fn callback_ace_counts_whatever_its_condition() {
        let sddl = format!("O:{USER}D:(XA;;FA;;;BU;(Member_of {{SID(BA)}}))");
        let got = findings(&sddl, OwnKind::File);
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].sid, USERS);
    }

    #[test]
    fn untrusted_file_owner_is_reported_even_with_a_clean_dacl() {
        let got = findings("O:BUD:(A;;FA;;;SY)", OwnKind::File);
        assert_eq!(
            got,
            vec![RawGrant {
                sid: USERS.into(),
                mask: WRITE_DAC,
                applies_to: AuditedObject::File,
                via: GrantVia::UntrustedOwner,
            }]
        );
    }

    #[test]
    fn directory_owned_by_a_broad_group_is_untrusted_as_owner_and_as_grantee() {
        let sddl = "O:BUD:(A;;FA;;;BU)";
        let sec = security(sddl);
        assert_eq!(sec.owner.as_deref(), Some(USERS));
        let trust = Trust::new(USER.to_string(), sec.owner.as_deref());
        assert_eq!(trust.directory_owner, None, "a broad-group owner must not be trusted");

        let got = untrusted_writes(&sec, OwnKind::Directory, &trust);
        assert!(
            got.iter()
                .any(|g| g.sid == USERS && g.via == GrantVia::UntrustedOwner && g.mask == WRITE_DAC),
            "{got:?}"
        );
        assert!(
            got.iter().any(|g| g.sid == USERS && g.via == GrantVia::Ace),
            "the explicit ACE for BU must still be reported alongside the ownership finding: {got:?}"
        );
    }

    #[test]
    fn group_owned_directory_is_not_trusted_as_owner() {
        // A SID of this shape (S-1-5-21-<domain>-<RID>) is exactly as likely to be a local group —
        // `ASUS\CodexSandboxUsers` on the development machine is one — as it is a user.
        let group = "S-1-5-21-1-2-3-3080";
        let sddl = format!("O:{group}D:(A;;FA;;;{group})");
        let sec = security(&sddl);
        let trust = Trust::new(USER.to_string(), sec.owner.as_deref());
        assert_eq!(trust.directory_owner, None, "{trust:?}");

        let got = untrusted_writes(&sec, OwnKind::Directory, &trust);
        assert!(
            got.iter().any(|g| g.sid == group && g.via == GrantVia::UntrustedOwner),
            "{got:?}"
        );
        assert!(got.iter().any(|g| g.sid == group && g.via == GrantVia::Ace), "{got:?}");
    }

    #[test]
    fn unresolvable_owner_fails_closed() {
        // Failing to positively identify an owner is not the same as identifying it as safe.
        let unresolvable = "S-1-5-21-9-9-9-9999";
        let sddl = format!("O:{unresolvable}D:(A;;FA;;;{unresolvable})");
        let sec = security(&sddl);
        let trust = Trust::new(USER.to_string(), sec.owner.as_deref());
        assert_eq!(trust.directory_owner, None);
        let got = untrusted_writes(&sec, OwnKind::Directory, &trust);
        assert!(
            got.iter().any(|g| g.sid == unresolvable && g.via == GrantVia::UntrustedOwner),
            "{got:?}"
        );
    }

    #[test]
    fn ancestor_with_delete_child_for_users_is_reported() {
        // 0x40 = FILE_DELETE_CHILD: enough on its own to delete-and-recreate the component beneath
        // this ancestor, regardless of that component's own DACL.
        let sddl = format!("O:{USER}D:(A;;0x40;;;BU)");
        let got = ancestor_grants(&security(&sddl), &trust(), ANCESTOR_REPLACE_MASK);
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].sid, USERS);
        assert_eq!(got[0].applies_to, AuditedObject::Ancestor);
        assert_eq!(right_names(got[0].mask, AuditedObject::Ancestor), vec!["FILE_DELETE_CHILD"]);
    }

    #[test]
    fn ancestor_add_subdirectory_alone_is_not_reported() {
        // AU = Authenticated Users; 0x4 = FILE_ADD_SUBDIRECTORY, which `C:\` grants by default on a
        // stock Windows install and which cannot, on its own, replace an existing component.
        let sddl = format!("O:{USER}D:(A;;0x4;;;AU)");
        let got = ancestor_grants(&security(&sddl), &trust(), ANCESTOR_REPLACE_MASK);
        assert!(got.is_empty(), "{got:?}");
    }

    #[test]
    fn ancestor_null_dacl_is_everyone_full_control() {
        let sec = security(&format!("O:{USER}D:NO_ACCESS_CONTROL"));
        let got = ancestor_grants(&sec, &trust(), ANCESTOR_REPLACE_MASK);
        assert_eq!(
            got,
            vec![RawGrant {
                sid: EVERYONE.into(),
                mask: GENERIC_ALL,
                applies_to: AuditedObject::Ancestor,
                via: GrantVia::NullDacl,
            }]
        );
    }

    #[test]
    fn component_replace_mask_catches_delete_but_not_delete_child() {
        // COMPONENT_REPLACE_MASK is about deleting or re-ACLing the component itself (`DELETE`),
        // never about deleting *its own children* (`FILE_DELETE_CHILD`) — that bit belongs to
        // ANCESTOR_REPLACE_MASK instead, checked against the *parent's* descriptor.
        let delete = findings_masked(&format!("O:{USER}D:(A;;0x10000;;;BU)"), COMPONENT_REPLACE_MASK);
        assert_eq!(delete.len(), 1, "{delete:?}");
        assert_eq!(right_names(delete[0].mask, AuditedObject::Ancestor), vec!["DELETE"]);

        let delete_child =
            findings_masked(&format!("O:{USER}D:(A;;0x40;;;BU)"), COMPONENT_REPLACE_MASK);
        assert!(delete_child.is_empty(), "{delete_child:?}");
    }

    #[test]
    fn sid_rendering_matches_convert_sid_to_string_sid() {
        // `sid_string` is hand-written (rather than delegating to `ConvertSidToStringSidW`)
        // specifically so a trust comparison needs no allocation to free — checked here against
        // Windows itself on the same bytes, including a 48-bit authority, which Windows renders as
        // bare uppercase hex with no zero-padding.
        use windows_sys::Win32::Security::Authorization::ConvertSidToStringSidW;

        // S-1-5-32-544 (Administrators): a small, 32-bit authority.
        let admins: &[u8] = &[1, 2, 0, 0, 0, 0, 0, 5, 32, 0, 0, 0, 0x20, 0x02, 0, 0];
        // A 48-bit authority, which Windows switches to hex for.
        let big: &[u8] = &[1, 1, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 7, 0, 0, 0];

        for bytes in [admins, big] {
            let mut raw: *mut u16 = null_mut();
            // SAFETY: `bytes` is the same well-formed SID `sid_string` parses below, and it outlives
            // this call; `raw` is a live local that receives a LocalAlloc'd string, owned by
            // `LocalMemory` immediately below so it is freed exactly once regardless of what happens
            // next in this loop iteration.
            let ok = unsafe { ConvertSidToStringSidW(bytes.as_ptr().cast_mut().cast(), &mut raw) };
            assert_ne!(ok, 0, "ConvertSidToStringSidW must accept a well-formed SID");
            let owned = LocalMemory(raw.cast());
            // SAFETY: `raw` is non-null (the call above succeeded) and NUL-terminated; each offset
            // up to the terminator lies within the allocation `owned` keeps alive for this loop.
            let len = (0..).take_while(|&i| unsafe { *raw.add(i) != 0 }).count();
            // SAFETY: `len` u16s precede the terminator just measured, inside the live allocation.
            let windows = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(raw, len) });
            drop(owned);
            assert_eq!(sid_string(bytes).as_deref(), Ok(windows.as_str()));
        }
    }

    #[test]
    fn malformed_sids_and_aces_are_errors_not_skips() {
        // Claims two sub-authorities, carries one.
        assert!(sid_string(&[1, 2, 0, 0, 0, 0, 0, 5, 32, 0, 0, 0]).is_err());
        assert!(sid_string(&[1, 16, 0, 0, 0, 0, 0, 5]).is_err());
        assert!(sid_string(&[2, 0, 0, 0, 0, 0, 0, 5]).is_err());
        // Allowed ACE truncated inside its SID.
        assert!(parse_ace(&[0, 0, 12, 0, 0xff, 0x01, 0x1f, 0x00, 1, 1, 0, 0]).is_err());
        // Compound and unknown types could grant access, so they must not be silently ignored.
        assert!(parse_ace(&[4, 0, 8, 0, 0, 0, 0, 0]).is_err());
        assert!(parse_ace(&[0x7f, 0, 8, 0, 0, 0, 0, 0]).is_err());
        // A deny ACE is ignorable even if its body is not parsed.
        assert_eq!(parse_ace(&[1, 0, 8, 0, 0, 0, 0, 0]), Ok(None));
    }

    #[test]
    fn each_control_right_alone_is_write_capable() {
        // Regression guard for `COMMON_WRITE`: every existing test that grants `WRITE_DAC`,
        // `WRITE_OWNER`, or `MAXIMUM_ALLOWED` does so via `FA` (full control), which also carries
        // `FILE_WRITE_DATA` — so deleting any one of these three bits from the constant would still
        // pass every one of those tests. Isolating each bit on its own, and asserting the exact
        // right name rather than just "one grant", is what makes deleting the line fail here.
        for (mask, name) in [
            ("0x40000", "WRITE_DAC"),
            ("0x80000", "WRITE_OWNER"),
            ("0x2000000", "MAXIMUM_ALLOWED"),
        ] {
            let got = findings(&format!("O:{USER}D:(A;;{mask};;;BU)"), OwnKind::File);
            assert_eq!(got.len(), 1, "{name}: {got:?}");
            assert_eq!(right_names(got[0].mask, AuditedObject::File), vec![name]);
        }
    }

    #[test]
    fn object_ace_sid_offset_handles_inherited_type_only_and_both_guids() {
        // `object_ace_sid_is_read_after_the_guids_it_carries` already covers "object-type GUID
        // present, inherited-object-type GUID absent". This covers the two shapes it does not:
        // inherited-object-type present alone (only the `ACE_INHERITED_OBJECT_TYPE_PRESENT` branch,
        // never exercised on its own elsewhere) and both GUIDs present together — plus a callback
        // object ACE (`ZA`, with a condition), which no other test parses at all despite
        // `parse_ace` treating it identically to a plain object ACE for offset purposes.
        let g = "bf967aba-0de6-11d0-a285-00aa003049e2";
        let sddl = format!(
            "O:{USER}D:(OA;;FA;;{g};BU)(OA;;FA;{g};{g};BG)(ZA;;FA;{g};;AU;(Member_of {{SID(BA)}}))"
        );
        let sids: Vec<String> = findings(&sddl, OwnKind::File).into_iter().map(|g| g.sid).collect();
        assert_eq!(sids, vec![USERS, "S-1-5-32-546", "S-1-5-11"]);
    }

    #[test]
    fn inherit_only_delete_child_is_not_a_new_file_grant() {
        // `untrusted_writes`'s inherit-only pass filters against `FILE_WRITE_MASK`, which does not
        // include `FILE_DELETE_CHILD` — a future file inheriting only the right to delete its own
        // children is not, by itself, a way to plant a DLL there. Untested before this: nothing
        // exercised that filter with a mask outside `FILE_WRITE_MASK` at all.
        assert!(findings(&format!("O:{USER}D:(A;OIIO;0x40;;;BU)"), OwnKind::Directory).is_empty());
    }

    #[test]
    fn verbatim_globalroot_and_device_prefixes_are_rejected() {
        // Pure prefix classification, no filesystem or network access: a rejection here must never
        // depend on I/O succeeding, failing, or timing out, since it runs before any descriptor read
        // and has to stay fast and reliable even when the machine is offline.
        for (path, is_local) in [
            (r"\\server\share\dir", false),
            (r"\\?\UNC\server\share\dir", false),
            (r"\\.\C:\dir", false),
            (r"\\?\GLOBALROOT\Device\HarddiskVolume1\dir", false),
            (r"\\?\C:\dir", true),
            (r"C:\dir", true),
        ] {
            assert_eq!(has_local_disk_prefix(Path::new(path)), is_local, "{path}");
        }
    }

    #[test]
    #[cfg(windows)]
    fn local_fixed_volume_passes_precondition() {
        // The one test in this module that performs real I/O for `require_local_volume`
        // (`GetVolumePathNameW` + `GetDriveTypeW`) rather than only the prefix stage. `%TEMP%` is
        // always a local fixed volume (or RAM disk) in this environment.
        let abs = std::path::absolute(std::env::temp_dir()).expect("temp dir must be absolute-able");
        require_local_volume(&abs).expect("the local temp volume must pass the precondition");
    }

    #[test]
    fn current_user_sid_is_a_real_account_sid() {
        let sid = current_user_sid().expect("the process token is always queryable");
        assert!(sid.starts_with("S-1-5-"), "{sid}");
        // An ACE for the real process user must be trusted through the real trust path.
        let t = Trust::new(sid.clone(), None);
        let sddl = format!("O:SYD:(A;;FA;;;{sid})");
        assert!(untrusted_writes(&security(&sddl), OwnKind::File, &t).is_empty());
    }

    #[test]
    fn reparse_attribute_wins_over_file_type() {
        use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_ARCHIVE;

        // A directory junction carries FILE_ATTRIBUTE_DIRECTORY alongside the reparse bit; the
        // reparse classification must win regardless of what the file/directory bit says.
        assert_eq!(
            classify_entry(FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_ARCHIVE, true),
            EntryKind::ReparsePoint
        );
        assert_eq!(classify_entry(FILE_ATTRIBUTE_ARCHIVE, true), EntryKind::RegularFile);
        assert_eq!(classify_entry(FILE_ATTRIBUTE_ARCHIVE, false), EntryKind::Other);
    }

    /// Look a name back up to a SID via `LookupAccountNameW`, independent of `account_name`'s own
    /// `LookupAccountSidW` call — so a test that compares the two is not just checking the function
    /// against itself. Used only to assert that a resolved name round-trips; never used for a trust
    /// decision.
    fn name_round_trips_to_sid(name: &str) -> Option<String> {
        use windows_sys::Win32::Security::LookupAccountNameW;

        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        let mut sid_len: u32 = 0;
        let mut domain_len: u32 = 0;
        let mut use_kind: SID_NAME_USE = 0;
        // SAFETY: the documented sizing call; null buffers with zero lengths are allowed.
        unsafe {
            LookupAccountNameW(
                null(),
                wide.as_ptr(),
                null_mut(),
                &mut sid_len,
                null_mut(),
                &mut domain_len,
                &mut use_kind,
            );
        }
        if sid_len == 0 {
            return None;
        }
        let mut sid_buf = vec![0u8; sid_len as usize];
        let mut domain = vec![0u16; (domain_len as usize).max(1)];
        let mut domain_cap = u32::try_from(domain.len()).ok()?;
        let mut sid_cap = sid_len;
        // SAFETY: `sid_buf` owns `sid_cap` bytes and `domain` owns `domain_cap` u16s, matching the
        // lengths passed alongside each; every out-pointer is a live local.
        let ok = unsafe {
            LookupAccountNameW(
                null(),
                wide.as_ptr(),
                sid_buf.as_mut_ptr().cast(),
                &mut sid_cap,
                domain.as_mut_ptr(),
                &mut domain_cap,
                &mut use_kind,
            )
        };
        if ok == 0 {
            return None;
        }
        sid_string(&sid_buf[..sid_cap as usize]).ok()
    }

    #[test]
    fn well_known_sids_resolve_to_names_that_round_trip() {
        // Not hard-coded English names: this machine may be running a localized build of Windows
        // (e.g. `VORDEFINIERT\Benutzer` for BUILTIN\Users under German), so the only assertion that
        // holds everywhere is that a name comes back and that name resolves to the same SID again.
        for sid in [USERS, SYSTEM] {
            let name = account_name(sid).expect("a well-known SID must resolve to some name");
            assert!(!name.is_empty());
            assert_eq!(
                name_round_trips_to_sid(&name).as_deref(),
                Some(sid),
                "LookupAccountNameW({name:?}) did not round-trip to {sid}"
            );
        }
    }

    #[test]
    fn a_string_that_is_not_a_sid_does_not_resolve() {
        assert_eq!(account_name("not a sid"), None);
    }

    #[test]
    #[ignore = "a SID from a domain this machine does not belong to can make LookupAccountSidW \
                query a domain controller, which adds network latency (or a long timeout when \
                offline) to what should be a fast unit test"]
    fn unresolvable_domain_sid_does_not_resolve() {
        assert_eq!(account_name("S-1-5-21-1-2-3-4242"), None);
    }
}
