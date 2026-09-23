//! The security profile: which settings pointers are locked, which [`Source`] may write them, path
//! canonicalization against the approved model roots, and the spawn-decision surface that
//! everything else in this crate exists to protect.
//!
//! # House rule S6
//!
//! "An unknown state is a refusal, never a warning that proceeds." Every function here follows it:
//! [`resolve_model_path`] fails closed on a path it cannot fully account for,
//! [`Settings::apply_patch`] validates an entire patch before committing any of it, and
//! [`DefaultPreflighter`] refuses on [`DaclAudit::Unknown`] exactly as it refuses on
//! [`DaclAudit::WritableByOthers`] -- "could not check" is never treated as "checked, fine".

use std::path::{Component, Path, PathBuf};

use cerebro_probe::DaclAudit;

use crate::error::ConfigError;
use crate::schema::Settings;

/// JSON pointers no [`Source::Api`] patch may ever touch, regardless of value.
///
/// A pointer is locked if it equals one of these exactly or falls beneath one of them (`/security`
/// covers `/security/lan_mode`, `/security/model_roots`, ...). This is the entire security surface
/// of [`Settings::apply_patch`]: everything under `/security` (the model-root allowlist, LAN
/// exposure, the audited backend path), everything under `/backend` (ggml loads DLLs from
/// `backend.dir` at spawn time -- a live source redirecting it is a DLL-plant vector, not a
/// preference change), and `/api/bind` (a live source redirecting its own bind address).
///
/// Locking a pointer controls *who may write it*, not *what value it may hold* -- that is a
/// separate, stronger guarantee. See the module documentation on [`crate::schema`] for why
/// `security.allow_unsafe_dacl` has no representation in [`Settings`] at all, rather than merely
/// being locked.
pub const SECURITY_LOCKED: &[&str] = &["/security", "/backend", "/api/bind"];

/// Whether a JSON pointer ([RFC 6901](https://www.rfc-editor.org/rfc/rfc6901)) falls under
/// [`SECURITY_LOCKED`].
///
/// `pointer` is taken in **escaped** form, exactly as an external JSON Patch document carries it
/// (`~` as `~0`, `/` as `~1`) -- this is a `pub` function the HTTP API surface is expected to call
/// with pointers it parsed itself, not only with pointers this crate built via
/// [`touched_pointers`], so its contract has to hold for either caller. Comparison never decodes:
/// decoding both sides before comparing would reopen exactly the ambiguity escaping exists to
/// remove (a key literally named `security/x` would decode to a segment matching `/security`).
pub fn is_security_locked(pointer: &str) -> bool {
    SECURITY_LOCKED
        .iter()
        .any(|locked| pointer == *locked || pointer.strip_prefix(locked).is_some_and(|rest| rest.starts_with('/')))
}

/// Where a settings patch claims to originate.
///
/// This is the entire input to the lock check in [`Settings::apply_patch`]: only [`Source::Api`] is
/// ever refused a locked pointer. [`Source::File`] (the config file on disk) and [`Source::Gui`]
/// (a local, already-privileged UI surface) may set any key that exists in the schema at all --
/// which, for `allow_unsafe_dacl`, is none, by construction, for every source uniformly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    /// The settings file on disk, as read by [`crate::file::ConfigFile`].
    File,
    /// Cerebro's own local UI (e.g. a Tauri shell), running as the same user.
    Gui,
    /// Cerebro's local HTTP API.
    Api,
}

/// Escape one JSON Patch reference token per [RFC 6901 §3](https://www.rfc-editor.org/rfc/rfc6901#section-3):
/// `~` becomes `~0` and `/` becomes `~1`. Order matters -- escaping `/` first would turn a literal
/// `~` that preceded it into part of a bogus `~1` escape once `~` was escaped second.
fn escape_token(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

/// Recursively collect every JSON pointer a merge-patch object touches, including intermediate
/// object nodes (so that a patch touching `/security` as a whole, not just a leaf beneath it, is
/// still caught). Each key is escaped per RFC 6901 as it is appended, so a pointer built here
/// compares equal to the pointer an external JSON Patch parser would produce for the same key --
/// see [`is_security_locked`].
fn touched_pointers(patch: &serde_json::Value, prefix: &str, out: &mut Vec<String>) {
    let serde_json::Value::Object(map) = patch else {
        return;
    };
    for (key, value) in map {
        let pointer = format!("{prefix}/{}", escape_token(key));
        out.push(pointer.clone());
        touched_pointers(value, &pointer, out);
    }
}

/// Apply one [RFC 7396](https://www.rfc-editor.org/rfc/rfc7396) JSON Merge Patch operation in place.
/// A `null` in the patch deletes the corresponding key; anything else recurses for objects and
/// replaces wholesale otherwise. No `unwrap`/`expect`: the "target just became an object" branch is
/// structurally guaranteed by the assignment immediately above it, but is written to no-op rather
/// than panic if that ever stops being true.
fn merge_patch(target: &mut serde_json::Value, patch: &serde_json::Value) {
    let serde_json::Value::Object(patch_map) = patch else {
        *target = patch.clone();
        return;
    };
    if !target.is_object() {
        *target = serde_json::Value::Object(serde_json::Map::new());
    }
    let serde_json::Value::Object(target_map) = target else {
        return;
    };
    for (key, value) in patch_map {
        if value.is_null() {
            target_map.remove(key);
        } else {
            let entry = target_map.entry(key.clone()).or_insert(serde_json::Value::Null);
            merge_patch(entry, value);
        }
    }
}

impl Settings {
    /// Apply a JSON Merge Patch to these settings, from `source`.
    ///
    /// The whole patch is validated before anything is committed:
    ///
    /// 1. If `source` is [`Source::Api`] and the patch touches any [`SECURITY_LOCKED`] pointer, the
    ///    whole patch is refused with [`ConfigError::LockedPointerRefused`] naming the first such
    ///    pointer (in sorted order, since `serde_json::Map` is `BTreeMap`-backed here) -- nothing is
    ///    applied, not even the unlocked parts.
    /// 2. Otherwise the patch is merged onto a serialized copy of the current settings and the
    ///    result is deserialized back into [`Settings`]. `#[serde(deny_unknown_fields)]` on every
    ///    section means an unknown key (an attempted `allow_unsafe_dacl`, a typo) fails this step
    ///    exactly as it would fail loading a file, from every [`Source`] -- only step 1
    ///    distinguishes sources at all, and it distinguishes them by pointer, never by key name.
    /// 3. The merged result must also pass [`Settings::validate`] (currently: `backend.dir` and
    ///    `security.backend_path` must still agree) -- a patch that pulls them apart is refused
    ///    exactly as a hand-edited file that did the same would be.
    ///
    /// Only on success is `self` replaced.
    pub fn apply_patch(&mut self, patch: &serde_json::Value, source: Source) -> Result<(), ConfigError> {
        if source == Source::Api {
            let mut touched = Vec::new();
            touched_pointers(patch, "", &mut touched);
            if let Some(pointer) = touched.into_iter().find(|p| is_security_locked(p)) {
                // The pointer is attacker-chosen (an `Api` caller supplies the patch this was
                // extracted from), so it is capped before it leaves this function -- see
                // `ConfigError::LockedPointerRefused::pointer`'s doc comment.
                return Err(ConfigError::LockedPointerRefused {
                    pointer: crate::error::cap_echo(&pointer),
                    by: source,
                });
            }
        }

        let mut merged = serde_json::to_value(&*self).map_err(|e| ConfigError::PatchRejected(e.to_string()))?;
        merge_patch(&mut merged, patch);
        let next: Settings = serde_json::from_value(merged).map_err(|e| ConfigError::PatchRejected(e.to_string()))?;
        // Structural checks beyond what `serde` alone expresses (currently: `backend.dir` and
        // `security.backend_path` must agree) apply here too, not just on load -- a patch is just
        // as capable of introducing the disagreement as a hand-edited file is.
        next.validate()?;
        *self = next;
        Ok(())
    }
}

/// Canonicalize `path` and require the result to be under one of `settings`'s approved model roots
/// ([`crate::schema::SecuritySection::model_roots`]).
///
/// Every check below returns the single [`ConfigError::ModelPathNotApproved`] variant, with no
/// distinguishing detail, so this function is not an oracle a caller could use to enumerate the
/// filesystem: "the file does not exist" and "the file exists but is outside every approved root"
/// look identical from the outside.
///
/// Rejected outright, on the literal input and again on the canonicalized result (so a reparse
/// point *inside* an approved root that resolves to one of these on the far side is still caught):
///
/// * any path segment that would make the string ambiguous as a plain local path -- a comma
///   (reserved as list syntax in several `llama.cpp` flags, per `cerebro-supervisor::preflight`'s
///   existing rule), or a colon after the drive letter (an NTFS alternate data stream reference);
/// * a UNC path (`\\server\share\...`), a verbatim UNC path (`\\?\UNC\...`), or a `GLOBALROOT`/
///   `Device`/`DosDevices` object-namespace path -- none of these name a location under a local
///   drive letter, which is the only thing an approved root can be.
///
/// A reparse point (symlink or directory junction) that leaves the root is caught structurally:
/// `std::fs::canonicalize` resolves it to its real target before the root check runs, so the check
/// below is comparing real, final path components, not the reparse point's apparent location.
///
/// # This function only checks; it does not close the window between checking and using
///
/// Every guarantee above is true of the filesystem at the instant `canonicalize` ran. A principal
/// who can write inside an approved root can replace the leaf with a reparse point pointing anywhere
/// between this function returning and a caller's eventual `File::open` of the path it approved --
/// this function cannot see that happen, because it does not hold anything open. Prefer
/// [`open_approved_model`] for any path that will actually be read; keep this function for
/// display/listing, where nothing downstream is ever opened.
pub fn resolve_model_path(settings: &Settings, path: &Path) -> Result<PathBuf, ConfigError> {
    reject_path_hazards(path)?;

    let canonical = std::fs::canonicalize(path).map_err(|_| ConfigError::ModelPathNotApproved)?;
    reject_path_hazards(&canonical)?;

    let canonical_components: Vec<Component> = canonical.components().collect();
    for root in &settings.security.model_roots {
        let Ok(root_canonical) = std::fs::canonicalize(root) else {
            continue;
        };
        let root_components: Vec<Component> = root_canonical.components().collect();
        if root_components.len() <= canonical_components.len()
            && canonical_components[..root_components.len()] == root_components[..]
        {
            return Ok(canonical);
        }
    }

    Err(ConfigError::ModelPathNotApproved)
}

/// `FILE_FLAG_OPEN_REPARSE_POINT` (winbase.h). Same guard, same rationale, as
/// [`crate::file`]'s copy for `settings.json`: a principal who can write inside an approved model
/// root must not be able to swap the leaf for a reparse point between [`open_approved_model`]
/// checking it and the caller reading it.
#[cfg(windows)]
const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;

/// `FILE_ATTRIBUTE_REPARSE_POINT` (winnt.h).
#[cfg(windows)]
const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;

/// Resolve `path` against the approved roots and return it together with an open read handle.
///
/// Prefer this over [`resolve_model_path`] for any path that will actually be loaded: the caller
/// reads through the handle returned here, so the leaf cannot be swapped between the approval and
/// the read. `MetadataExt`/`OpenOptionsExt` are imported inside this function, not at module scope,
/// so a non-Windows build does not carry an unused import under `-D warnings`; both are safe std
/// APIs, so this stays under the crate's `#![forbid(unsafe_code)]`.
///
/// # Residual window
///
/// `FILE_FLAG_OPEN_REPARSE_POINT` constrains only the *final* component. The canonical path is
/// re-traversed from the volume root when this handle is opened, so a principal who can write an
/// *intermediate* directory inside an approved root can still replace that directory with a
/// junction between [`resolve_model_path`] canonicalizing and this open. Closing that window needs
/// a component-by-component walk that opens each directory with `FILE_FLAG_BACKUP_SEMANTICS |
/// FILE_FLAG_OPEN_REPARSE_POINT` and reopens the next component relative to the parent handle; that
/// walk is not implemented here, and no caller may assume it happens.
///
/// # Errors
/// The same single [`ConfigError::ModelPathNotApproved`] variant [`resolve_model_path`] returns,
/// for the same reason: a caller must not be able to tell "not approved" apart from "approved, but
/// the handle-open step then failed" by the shape of the error.
#[cfg(windows)]
pub fn open_approved_model(settings: &Settings, path: &Path) -> Result<(PathBuf, std::fs::File), ConfigError> {
    use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};

    let canonical = resolve_model_path(settings, path)?;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(&canonical)
        .map_err(|_| ConfigError::ModelPathNotApproved)?;
    let attrs = file.metadata().map_err(|_| ConfigError::ModelPathNotApproved)?.file_attributes();
    if attrs & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(ConfigError::ModelPathNotApproved);
    }
    Ok((canonical, file))
}

/// Off Windows there is no reparse-point defense anywhere in this crate (see [`crate::file`]'s
/// equivalent split), so this just resolves and opens normally.
#[cfg(not(windows))]
pub fn open_approved_model(settings: &Settings, path: &Path) -> Result<(PathBuf, std::fs::File), ConfigError> {
    let canonical = resolve_model_path(settings, path)?;
    let file = std::fs::File::open(&canonical).map_err(|_| ConfigError::ModelPathNotApproved)?;
    Ok((canonical, file))
}

/// The literal-string hazards checked by [`resolve_model_path`]; see its documentation.
fn reject_path_hazards(path: &Path) -> Result<(), ConfigError> {
    let raw = path.to_string_lossy();
    if raw.contains(',') {
        return Err(ConfigError::ModelPathNotApproved);
    }
    if is_unc_or_device_path(&raw) {
        return Err(ConfigError::ModelPathNotApproved);
    }
    if has_alternate_data_stream(&raw) {
        return Err(ConfigError::ModelPathNotApproved);
    }
    Ok(())
}

/// A path body starting with an ASCII drive letter and colon, e.g. `C:\models\x.gguf`.
fn is_drive_letter_root(body: &str) -> bool {
    let bytes = body.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

/// Whether `raw` is a UNC path, a verbatim UNC/`GLOBALROOT` path, or an NT native-namespace path --
/// anything that is not, in the end, a plain local drive-letter path.
fn is_unc_or_device_path(raw: &str) -> bool {
    if let Some(rest) = raw.strip_prefix(r"\\?\") {
        // Verbatim local disk paths (`\\?\C:\...`) are exactly what `std::fs::canonicalize`
        // itself produces on Windows and must be accepted; verbatim UNC (`\\?\UNC\...`) and
        // `GLOBALROOT` must not be.
        let upper = rest.to_ascii_uppercase();
        return upper.starts_with("UNC") || upper.starts_with("GLOBALROOT") || !is_drive_letter_root(rest);
    }
    if raw.starts_with(r"\\") {
        return true; // plain UNC, e.g. \\server\share\...
    }
    if raw.starts_with(r"\??\") || raw.starts_with(r"\Device\") || raw.starts_with(r"\DosDevices\") {
        return true; // NT native namespace
    }
    false
}

/// Whether `raw` contains a colon anywhere other than the drive-letter position, i.e. an alternate
/// data stream reference such as `C:\file.txt:secret`.
fn has_alternate_data_stream(raw: &str) -> bool {
    let body = raw.strip_prefix(r"\\?\").unwrap_or(raw);
    let rest = if is_drive_letter_root(body) { &body[2..] } else { body };
    rest.contains(':')
}

/// Outcome of a preflight decision for a prospective backend spawn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpawnDecision {
    /// Nothing observed forbids the spawn.
    Allow,
    /// The spawn must not proceed, and why (safe to log or show a user).
    Refuse {
        /// Human-readable reason.
        reason: String,
    },
}

/// Decides whether a backend spawn may proceed.
///
/// `audited_dir` is in the signature, distinct from whatever [`Settings`] happens to name, on
/// purpose: a decision about the write-permissions of *one* directory must never be usable to
/// authorise a spawn from a *different* one. [`DefaultPreflighter::decide`] enforces that by
/// refusing unless `audited_dir`, [`BackendSection::dir`] and [`SecuritySection::backend_path`] all
/// canonicalize to the same real directory -- see the
/// `preflight_refuses_when_the_audited_directory_is_not_the_spawn_directory` test below for the
/// attack this closes: a poisoned file that points the DACL audit at a clean directory while
/// `backend.dir` (the one a spawn would actually use) names a hostile one.
pub trait Preflighter {
    /// Decide whether to spawn, given the settings in effect, the directory a [`DaclAudit`] was
    /// actually performed on, and its result.
    fn decide(&self, settings: &Settings, audited_dir: &Path, dacl: &DaclAudit) -> SpawnDecision;
}

/// The one [`Preflighter`] Cerebro ships.
///
/// Refuses unless all of the following hold, in order (house rule S6: a check this function cannot
/// complete -- including a path that fails to canonicalize, e.g. because nothing has configured a
/// backend directory yet -- is a refusal, never treated as safe):
///
/// 1. `audited_dir`, `settings.backend.dir` and `settings.security.backend_path` each
///    canonicalize successfully.
/// 2. `settings.backend.dir` and `settings.security.backend_path` canonicalize to the same
///    directory (see [`crate::schema::Settings::validate`], which already checks the literal-path
///    form of this on every load and patch; this re-checks the filesystem-resolved form, which is
///    what actually matters here).
/// 3. `audited_dir` canonicalizes to that same directory -- the audit that produced `dacl` was
///    actually performed on the directory a spawn would use, not some other one.
/// 4. `dacl` is [`DaclAudit::OwnerOnly`].
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultPreflighter;

impl Preflighter for DefaultPreflighter {
    fn decide(&self, settings: &Settings, audited_dir: &Path, dacl: &DaclAudit) -> SpawnDecision {
        fn refuse(reason: &str) -> SpawnDecision {
            SpawnDecision::Refuse { reason: reason.to_string() }
        }

        let Ok(audited) = std::fs::canonicalize(audited_dir) else {
            return refuse("the audited directory could not be resolved");
        };
        let Ok(spawn_dir) = std::fs::canonicalize(&settings.backend.dir) else {
            return refuse("backend.dir could not be resolved (not configured, or does not exist)");
        };
        let Ok(declared) = std::fs::canonicalize(&settings.security.backend_path) else {
            return refuse("security.backend_path could not be resolved (not configured, or does not exist)");
        };

        if spawn_dir != declared {
            return refuse("backend.dir and security.backend_path resolve to different directories");
        }
        if audited != spawn_dir {
            return refuse("the audited directory is not the one a spawn would use");
        }
        if !dacl.is_safe_to_spawn() {
            return refuse("backend directory audit did not establish owner-only write access");
        }
        SpawnDecision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::full_settings_fixture;

    #[test]
    fn security_backend_and_api_bind_pointers_are_locked() {
        assert!(is_security_locked("/security"));
        assert!(is_security_locked("/security/lan_mode"));
        assert!(is_security_locked("/backend"));
        assert!(is_security_locked("/backend/dir"));
        assert!(is_security_locked("/api/bind"));
    }

    #[test]
    fn sibling_and_prefix_lookalike_pointers_are_not_locked() {
        assert!(!is_security_locked("/models/roots"));
        assert!(!is_security_locked("/api/enabled"));
        assert!(!is_security_locked("/api/port"));
        // A pointer that merely starts with the same characters as a locked one, without a `/`
        // boundary, must not match -- `/securityish` is not `/security`.
        assert!(!is_security_locked("/securityish"));
        assert!(!is_security_locked("/backendish/x"));
    }

    #[test]
    fn escaped_pointers_round_trip_and_are_not_decoded_before_comparison() {
        // A key literally named `bind/x` under `/api` escapes to `/api/bind~1x`, which is a
        // sibling of `/api/bind`, not the same pointer -- decoding before comparing would collapse
        // that distinction and lock a pointer that was never meant to be.
        assert!(!is_security_locked("/api/bind~1x"));
        assert!(is_security_locked("/api/bind"));
        assert!(is_security_locked("/security/model_roots/0"));

        // `touched_pointers` must produce exactly the escaped form `is_security_locked` expects.
        let mut out = Vec::new();
        touched_pointers(&serde_json::json!({"api": {"bind/x": 1}}), "", &mut out);
        assert!(out.contains(&"/api/bind~1x".to_string()), "{out:?}");
        assert!(!out.iter().any(|p| is_security_locked(p)), "{out:?} must not lock /api/bind~1x");
    }

    #[test]
    fn api_source_is_refused_on_every_locked_pointer() {
        let attempts = [
            serde_json::json!({"security": {"lan_mode": true}}),
            serde_json::json!({"security": {"model_roots": ["C:\\"]}}),
            serde_json::json!({"backend": {"dir": "C:\\elsewhere"}}),
            serde_json::json!({"api": {"bind": "loopback6"}}),
        ];
        for patch in attempts {
            let mut settings = full_settings_fixture();
            let before = settings.clone();
            let err = settings.apply_patch(&patch, Source::Api).unwrap_err();
            assert!(
                matches!(err, ConfigError::LockedPointerRefused { .. }),
                "patch {patch} got {err:?}"
            );
            assert_eq!(settings, before, "a refused patch must not be partially applied");
        }
    }

    #[test]
    fn gui_and_file_sources_may_write_the_same_locked_pointers() {
        let patch = serde_json::json!({"security": {"lan_mode": true}});
        for source in [Source::Gui, Source::File] {
            let mut settings = full_settings_fixture();
            settings.apply_patch(&patch, source).unwrap_or_else(|e| panic!("{source:?} refused: {e}"));
            assert!(settings.security.lan_mode);
        }
    }

    #[test]
    fn api_source_may_still_write_unlocked_pointers() {
        let mut settings = full_settings_fixture();
        let patch = serde_json::json!({"sampler": {"temp": 0.42}});
        settings.apply_patch(&patch, Source::Api).expect("unlocked pointer is not refused");
        assert_eq!(settings.sampler.temp(), 0.42);
    }

    #[test]
    fn api_patch_touching_a_locked_pointer_with_an_escaped_key_is_still_refused() {
        // The genuinely locked pointer is refused specifically as a lock violation.
        let mut settings = full_settings_fixture();
        let locked = serde_json::json!({"security": {"model_roots": ["C:\\"]}});
        let err = settings.apply_patch(&locked, Source::Api).unwrap_err();
        assert!(matches!(err, ConfigError::LockedPointerRefused { .. }), "got {err:?}");

        // A key that merely *looks* like it could escape to a locked sibling (`bind/x` under
        // `/api`, escaping to `/api/bind~1x`, a sibling of `/api/bind`) is not itself locked -- it
        // is refused only because `ApiSection` denies unknown fields, proving escaping
        // discriminates rather than merely runs without ever mattering.
        let mut settings = full_settings_fixture();
        let unknown = serde_json::json!({"api": {"bind/x": true}});
        let err = settings.apply_patch(&unknown, Source::Api).unwrap_err();
        assert!(matches!(err, ConfigError::PatchRejected(_)), "got {err:?}");
    }

    #[test]
    fn a_partial_patch_leaves_untouched_sibling_fields_alone() {
        let mut settings = full_settings_fixture();
        let original_top_k = settings.sampler.top_k();
        settings
            .apply_patch(&serde_json::json!({"sampler": {"temp": 1.0}}), Source::Gui)
            .expect("valid partial patch");
        assert_eq!(settings.sampler.temp(), 1.0);
        assert_eq!(settings.sampler.top_k(), original_top_k);
    }

    #[test]
    fn a_patch_declaring_an_unsupported_schema_version_is_refused_from_every_source() {
        // `version` is not itself a `SECURITY_LOCKED` pointer, so this proves the refusal comes
        // from `Settings::validate` (which every `apply_patch` call ends with), not from the lock
        // check -- it must hold identically for every `Source`.
        for source in [Source::File, Source::Gui, Source::Api] {
            let mut settings = full_settings_fixture();
            let before = settings.clone();
            let patch = serde_json::json!({"version": crate::schema::SCHEMA_VERSION + 1});
            let err = settings.apply_patch(&patch, source).unwrap_err();
            assert!(matches!(err, ConfigError::UnsupportedVersion { .. }), "source {source:?} got {err:?}");
            assert_eq!(settings, before, "a refused patch must not be partially applied");
        }
    }

    #[test]
    fn attempting_to_smuggle_allow_unsafe_dacl_through_a_patch_is_an_unknown_field_refusal() {
        for source in [Source::File, Source::Gui, Source::Api] {
            let mut settings = full_settings_fixture();
            let before = settings.clone();
            let patch = serde_json::json!({"security": {"allow_unsafe_dacl": true}});
            let err = settings.apply_patch(&patch, source).unwrap_err();
            // From Source::Api this could in principle be caught by the lock check first (the
            // pointer is under /security either way); the property that must hold regardless is
            // that it is refused and nothing is applied.
            assert!(
                matches!(err, ConfigError::LockedPointerRefused { .. } | ConfigError::PatchRejected(_)),
                "source {source:?} got {err:?}"
            );
            assert_eq!(settings, before);
        }
    }

    fn approved_root_with_file() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("models");
        std::fs::create_dir_all(&root).expect("create root");
        let file = root.join("target.gguf");
        std::fs::write(&file, b"not a real gguf").expect("write model file");
        (dir, root, file)
    }

    fn settings_with_root(root: &Path) -> Settings {
        let mut settings = full_settings_fixture();
        settings.security.model_roots = vec![root.to_path_buf()];
        settings
    }

    #[test]
    fn a_file_under_an_approved_root_resolves() {
        let (_dir, root, file) = approved_root_with_file();
        let settings = settings_with_root(&root);
        let resolved = resolve_model_path(&settings, &file).expect("under an approved root");
        assert_eq!(resolved, std::fs::canonicalize(&file).unwrap());
    }

    #[test]
    fn open_approved_model_returns_a_readable_handle_to_the_approved_file() {
        let (_dir, root, file) = approved_root_with_file();
        let settings = settings_with_root(&root);
        let (resolved, mut handle) = open_approved_model(&settings, &file).expect("under an approved root");
        assert_eq!(resolved, std::fs::canonicalize(&file).unwrap());

        let mut contents = Vec::new();
        std::io::Read::read_to_end(&mut handle, &mut contents).expect("read through the returned handle");
        assert_eq!(contents, b"not a real gguf");
    }

    #[test]
    fn open_approved_model_refuses_a_path_outside_every_root() {
        let (dir, root, _file) = approved_root_with_file();
        let settings = settings_with_root(&root);
        let outside = dir.path().join("outside.gguf");
        std::fs::write(&outside, b"x").expect("write outside file");
        let err = open_approved_model(&settings, &outside).unwrap_err();
        assert_eq!(err, ConfigError::ModelPathNotApproved);
    }

    #[test]
    fn missing_and_outside_root_produce_the_identical_error() {
        let (dir, root, _file) = approved_root_with_file();
        let settings = settings_with_root(&root);

        let missing = root.join("does-not-exist.gguf");
        let missing_err = resolve_model_path(&settings, &missing).unwrap_err();

        let outside = dir.path().join("outside.gguf");
        std::fs::write(&outside, b"x").expect("write outside file");
        let outside_err = resolve_model_path(&settings, &outside).unwrap_err();

        assert_eq!(missing_err, ConfigError::ModelPathNotApproved);
        assert_eq!(outside_err, ConfigError::ModelPathNotApproved);
        assert_eq!(missing_err, outside_err, "the two failure causes must be indistinguishable");
    }

    #[test]
    fn a_sibling_directory_whose_name_extends_the_root_is_rejected() {
        // The regression test for component comparison: `...\models-evil` has `...\models` as a
        // *string* prefix but is not under it. A `starts_with` implementation would approve this;
        // comparing `Component`s cannot, because `models` and `models-evil` are different single
        // components at the same depth.
        let (dir, root, _file) = approved_root_with_file();
        let lookalike = dir.path().join(format!("{}-evil", root.file_name().expect("root has a name").to_string_lossy()));
        std::fs::create_dir_all(&lookalike).expect("create lookalike dir");
        let evil = lookalike.join("evil.gguf");
        std::fs::write(&evil, b"x").expect("write evil file");

        let settings = settings_with_root(&root);
        assert_eq!(resolve_model_path(&settings, &evil).unwrap_err(), ConfigError::ModelPathNotApproved);
        assert_eq!(open_approved_model(&settings, &evil).unwrap_err(), ConfigError::ModelPathNotApproved);
    }

    #[test]
    fn dot_dot_escape_from_an_approved_root_is_rejected() {
        let (dir, root, _file) = approved_root_with_file();
        let sibling = dir.path().join("sibling");
        std::fs::create_dir_all(&sibling).expect("create sibling dir");
        let evil = sibling.join("evil.gguf");
        std::fs::write(&evil, b"x").expect("write evil file");

        let settings = settings_with_root(&root);
        let escape_path = root.join("..").join("sibling").join("evil.gguf");
        let err = resolve_model_path(&settings, &escape_path).unwrap_err();
        assert_eq!(err, ConfigError::ModelPathNotApproved);
    }

    #[test]
    fn alternate_data_stream_is_rejected() {
        let (_dir, root, file) = approved_root_with_file();
        let settings = settings_with_root(&root);
        let ads = PathBuf::from(format!("{}:secret", file.display()));
        let err = resolve_model_path(&settings, &ads).unwrap_err();
        assert_eq!(err, ConfigError::ModelPathNotApproved);
    }

    #[test]
    fn unc_path_is_rejected() {
        let settings = full_settings_fixture();
        let unc = PathBuf::from(r"\\server\share\model.gguf");
        let err = resolve_model_path(&settings, &unc).unwrap_err();
        assert_eq!(err, ConfigError::ModelPathNotApproved);
    }

    #[test]
    fn verbatim_globalroot_path_is_rejected() {
        let settings = full_settings_fixture();
        let device = PathBuf::from(r"\\?\GLOBALROOT\Device\Something\model.gguf");
        let err = resolve_model_path(&settings, &device).unwrap_err();
        assert_eq!(err, ConfigError::ModelPathNotApproved);
    }

    #[test]
    fn comma_in_path_is_rejected() {
        let (_dir, root, _file) = approved_root_with_file();
        let settings = settings_with_root(&root);
        let commaed = root.join("model,evil.gguf");
        std::fs::write(&commaed, b"x").expect("write file with comma in name");
        let err = resolve_model_path(&settings, &commaed).unwrap_err();
        assert_eq!(err, ConfigError::ModelPathNotApproved);
    }

    /// Needs no admin: unlike a symlink, a directory junction (`mklink /J`) can be created by any
    /// local user. Ignored because it touches the real filesystem with a Windows-specific tool
    /// outside the sandboxed unit-test environment.
    #[test]
    #[ignore = "creates a real directory junction with mklink /J; run explicitly on Windows"]
    fn directory_junction_escaping_the_root_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("models");
        std::fs::create_dir_all(&root).expect("create root");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&outside).expect("create outside dir");
        std::fs::write(outside.join("evil.gguf"), b"x").expect("write evil file");

        let link = root.join("junction");
        let status = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(&link)
            .arg(&outside)
            .status()
            .expect("run mklink");
        assert!(status.success(), "mklink /J failed; junctions need no admin rights but do need dev mode off is fine");

        let settings = settings_with_root(&root);
        let via_junction = link.join("evil.gguf");
        let err = resolve_model_path(&settings, &via_junction).unwrap_err();
        assert_eq!(err, ConfigError::ModelPathNotApproved);
    }

    /// A real, on-disk directory whose `backend.dir` and `security.backend_path` already agree,
    /// suitable as the "clean" side of every [`Preflighter`] test below. Kept separate from
    /// [`full_settings_fixture`]'s hardcoded real machine path so these tests are self-contained
    /// and do not depend on anything but a fresh [`tempfile::tempdir`].
    fn settings_pointing_at(dir: &Path) -> Settings {
        let mut settings = full_settings_fixture();
        settings.backend.dir = dir.to_path_buf();
        settings.security.backend_path = dir.to_path_buf();
        settings
    }

    #[test]
    fn no_settings_value_from_any_source_permits_a_spawn_from_an_unsafe_directory() {
        let clean = tempfile::tempdir().expect("tempdir");
        let hostile = tempfile::tempdir().expect("tempdir");
        let hostile_dir = hostile.path().display().to_string();
        let unsafe_dacl = DaclAudit::WritableByOthers(vec!["ASUS\\CodexSandboxUsers".to_string()]);
        let unknown_dacl = DaclAudit::Unknown("audit not run");
        let preflighter = DefaultPreflighter;

        // Every security-relevant key an attacker might try to flip in the attempt to widen a
        // spawn decision, from every Source, including the removed key itself (which must fail to
        // apply at all, but the property under test holds either way -- see below).
        let attempts: Vec<serde_json::Value> = vec![
            serde_json::json!({"security": {"lan_mode": true}}),
            serde_json::json!({"security": {"model_roots": ["C:\\"]}}),
            serde_json::json!({"security": {"backend_path": "C:\\elsewhere"}}),
            serde_json::json!({"security": {"allow_unsafe_dacl": true}}),
            serde_json::json!({"backend": {"dir": "C:\\elsewhere", "manifest": "C:\\elsewhere\\m.json"}}),
            serde_json::json!({"api": {"enabled": true, "bind": "loopback6"}}),
            // The only shape that survives `Settings::validate`: both coupled backend keys moved
            // to a real hostile directory at once, which is exactly what a poisoned file would
            // write, and exactly the attack `BackendPathDisagreement` alone cannot catch because
            // nothing here disagrees. This must still be refused -- by the audited-directory check
            // instead, since `decide` re-verifies `audited_dir` against `backend.dir` itself.
            serde_json::json!({
                "backend": {"dir": hostile_dir, "manifest": format!("{hostile_dir}\\m.json")},
                "security": {"backend_path": hostile_dir},
            }),
        ];

        for source in [Source::File, Source::Gui, Source::Api] {
            for patch in &attempts {
                let mut settings = settings_pointing_at(clean.path());
                // Whether the patch is accepted or refused is not the property under test here
                // (that is covered above); what must hold regardless is that the resulting
                // settings, whatever they are, cannot turn either audited-unsafe decision into
                // `Allow` -- audited *against the one real directory this test controls*, `clean`,
                // which is the axis the pre-fix version of this test never varied. The reason
                // string is deliberately not pinned here (it now legitimately differs between the
                // ordinary attempts and the coupled-hostile-backend one); the specific reason for
                // the writable-by-others case is pinned separately, by
                // `owner_only_matching_directory_is_allowed`'s sibling below and by
                // `preflight_refuses_when_the_audited_directory_is_not_the_spawn_directory`.
                let _ = settings.apply_patch(patch, source);

                let unsafe_decision = preflighter.decide(&settings, clean.path(), &unsafe_dacl);
                assert!(
                    matches!(unsafe_decision, SpawnDecision::Refuse { .. }),
                    "source {source:?} patch {patch} allowed a spawn for a writable-by-others directory: {unsafe_decision:?}"
                );
                let unknown_decision = preflighter.decide(&settings, clean.path(), &unknown_dacl);
                assert!(
                    matches!(unknown_decision, SpawnDecision::Refuse { .. }),
                    "source {source:?} patch {patch} allowed a spawn for an unaudited directory: {unknown_decision:?}"
                );
            }
        }
    }

    #[test]
    fn writable_by_others_directory_is_refused_with_the_dacl_specific_reason() {
        // The reason string pinned separately from the sweep above: when `audited_dir` really is
        // the spawn directory and only the DACL audit itself is unsafe, `decide` must name that
        // specific cause rather than a generic refusal.
        let clean = tempfile::tempdir().expect("tempdir");
        let preflighter = DefaultPreflighter;
        let settings = settings_pointing_at(clean.path());
        let unsafe_dacl = DaclAudit::WritableByOthers(vec!["ASUS\\CodexSandboxUsers".to_string()]);
        assert_eq!(
            preflighter.decide(&settings, clean.path(), &unsafe_dacl),
            SpawnDecision::Refuse {
                reason: "backend directory audit did not establish owner-only write access".to_string(),
            }
        );
    }

    #[test]
    fn owner_only_matching_directory_is_allowed() {
        let clean = tempfile::tempdir().expect("tempdir");
        let preflighter = DefaultPreflighter;
        let settings = settings_pointing_at(clean.path());
        assert_eq!(
            preflighter.decide(&settings, clean.path(), &DaclAudit::OwnerOnly),
            SpawnDecision::Allow
        );
    }

    #[test]
    fn preflight_refuses_when_the_audited_directory_is_not_the_spawn_directory() {
        // This is the case the pre-fix `decide` could not fail: the DACL audit is a clean
        // `OwnerOnly` result, but it was performed on a *different* directory from the one
        // `backend.dir` names. A caller who audits `security.backend_path` (the documented audit
        // subject) and trusts the result must not thereby authorise a spawn from `backend.dir`.
        let clean = tempfile::tempdir().expect("tempdir");
        let hostile = tempfile::tempdir().expect("tempdir");
        let mut settings = full_settings_fixture();
        settings.backend.dir = hostile.path().to_path_buf();
        settings.security.backend_path = clean.path().to_path_buf();

        let preflighter = DefaultPreflighter;
        let decision = preflighter.decide(&settings, clean.path(), &DaclAudit::OwnerOnly);
        assert!(matches!(decision, SpawnDecision::Refuse { .. }), "got {decision:?}");
    }

    #[test]
    fn preflight_refuses_when_backend_dir_and_security_backend_path_disagree() {
        // Both directories are real and both audits would pass individually; only the fact that
        // the two schema keys disagree makes this unsafe. `Settings::validate` already refuses
        // this on load/patch (see `settings_with_disagreeing_backend_paths_is_refused` in
        // `schema.rs`) -- this test is `decide`'s own, independent re-check of the same property,
        // in case a `Settings` value ever reaches it without having gone through `validate` first.
        let dir_a = tempfile::tempdir().expect("tempdir");
        let dir_b = tempfile::tempdir().expect("tempdir");
        let mut settings = full_settings_fixture();
        settings.backend.dir = dir_a.path().to_path_buf();
        settings.security.backend_path = dir_b.path().to_path_buf();

        let preflighter = DefaultPreflighter;
        let decision = preflighter.decide(&settings, dir_a.path(), &DaclAudit::OwnerOnly);
        assert!(matches!(decision, SpawnDecision::Refuse { .. }), "got {decision:?}");
    }

    /// The EVAL_PLAN §6.4 poisoned-config scenario end to end, against the real machine fixture
    /// rather than a synthetic [`DaclAudit`]: a settings document whose `backend.dir` (and, kept
    /// consistent so [`Settings::validate`] does not intercept it first) `security.backend_path`
    /// point at the real `llama.cpp-b10751` build directory, which this development machine's
    /// `ASUS\CodexSandboxUsers` holds Modify rights on. Ignored because it depends on this
    /// specific machine's filesystem layout and DACLs, neither of which a general test environment
    /// has.
    #[test]
    #[ignore = "depends on this machine's real llama.cpp-b10751 fixture directory and its DACLs; run explicitly"]
    fn poisoned_file_cannot_point_the_spawn_at_the_writable_llama_cpp_fixture() {
        let fixture_dir =
            PathBuf::from(r"C:\Users\noahm\Projects\qwen-flash-next\llama.cpp-b10751");
        let mut settings = full_settings_fixture();
        settings.backend.dir = fixture_dir.clone();
        settings.security.backend_path = fixture_dir.clone();

        let probe = cerebro_probe::backend::probe(&fixture_dir, "llama-server.exe")
            .expect("probe the real fixture directory");

        let preflighter = DefaultPreflighter;
        let decision = preflighter.decide(&settings, &fixture_dir, &probe.dacl);
        // `DaclAudit::is_safe_to_spawn` is false for both `WritableByOthers` (the DACL audit is
        // implemented) and `Unknown` (it is not yet, per `cerebro_probe::backend::audit_dacl`'s own
        // doc comment) -- either way, house rule S6 means this must refuse, never allow.
        assert!(matches!(decision, SpawnDecision::Refuse { .. }), "got {decision:?}");
    }

    #[test]
    fn preflight_refuses_on_an_unconfigured_backend_directory() {
        // A fresh install's `Settings::default()` has `backend.dir == security.backend_path ==
        // PathBuf::new()`, which cannot canonicalize. House rule S6 says that is a refusal, not a
        // pass -- there is nothing to audit, so there is nothing to trust.
        let clean = tempfile::tempdir().expect("tempdir");
        let settings = Settings::default();
        let preflighter = DefaultPreflighter;
        let decision = preflighter.decide(&settings, clean.path(), &DaclAudit::OwnerOnly);
        assert!(matches!(decision, SpawnDecision::Refuse { .. }), "got {decision:?}");
    }
}
