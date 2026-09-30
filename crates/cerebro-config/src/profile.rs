//! Who may write which settings, and model-path resolution against the model folders.
//!
//! Launch decisions are not made here: [`Settings::launch_spec`] hands the backend directory and
//! [`crate::policy::ProtectionsSection::launch`] to `cerebro-supervisor`'s preflight, which owns
//! the folder audit, the integrity check and every other refusal.
//!
//! # House rule S6
//!
//! "An unknown state is a refusal, never a warning that proceeds." [`resolve_model_path`] fails
//! closed on a path it cannot fully account for, and [`Settings::apply_patch`] validates an entire
//! patch before committing any of it.

use std::path::{Component, Path, PathBuf, Prefix};

use crate::error::ConfigError;
use crate::schema::Settings;

/// JSON pointers no [`Source::Api`] patch may ever touch, regardless of value.
///
/// A pointer is locked if it equals one of these exactly or falls beneath one of them
/// (`/protections` covers `/protections/refuse_elevation`, ...):
///
/// * `/backend`: ggml loads DLLs from `backend.dir` at spawn time, so a live source redirecting it
///   is a DLL-plant vector, not a preference change.
/// * `/api/bind`: a live source redirecting its own bind address.
/// * `/models/folders`: the only roots a model path may resolve under.
/// * `/network`: offline mode and LAN exposure.
/// * `/protections`: a client of the API turning off the guards on its own launches.
/// * `/plugins`: a plugin granting itself permissions.
///
/// Locking controls *who may write* a key, never *what value* it may hold: [`Source::File`] and
/// [`Source::Gui`] (the user) may set every key, including turning any protection off.
pub const SECURITY_LOCKED: &[&str] = &[
    "/backend",
    "/api/bind",
    "/models/folders",
    "/network",
    "/protections",
    "/plugins",
];

/// Whether a JSON pointer ([RFC 6901](https://www.rfc-editor.org/rfc/rfc6901)) falls under
/// [`SECURITY_LOCKED`].
///
/// `pointer` is taken in **escaped** form, exactly as an external JSON Patch document carries it
/// (`~` as `~0`, `/` as `~1`) -- this is a `pub` function the HTTP API surface is expected to call
/// with pointers it parsed itself, not only with pointers this crate built via
/// [`touched_pointers`], so its contract has to hold for either caller. Comparison never decodes:
/// decoding both sides before comparing would reopen exactly the ambiguity escaping exists to
/// remove (a key literally named `network/x` would decode to a segment matching `/network`).
pub fn is_security_locked(pointer: &str) -> bool {
    SECURITY_LOCKED
        .iter()
        .any(|locked| pointer == *locked || pointer.strip_prefix(locked).is_some_and(|rest| rest.starts_with('/')))
}

/// Where a settings patch claims to originate.
///
/// This is the entire input to the lock check in [`Settings::apply_patch`]: only [`Source::Api`] is
/// ever refused a locked pointer. [`Source::File`] (the config file on disk) and [`Source::Gui`]
/// (Cerebro's own UI, i.e. the user) may set any key the schema has.
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

/// Recursively collect every JSON pointer a merge-patch object names, including intermediate
/// object nodes (so a patch naming a fully locked section such as `/protections` as a whole, not
/// just a leaf beneath it, is caught by name). An *ancestor* of a partially locked key
/// (`/models` above `/models/folders`) is not locked by name; [`Settings::apply_patch`] catches a
/// change made through one by comparing effective values after the merge. Each key is escaped per
/// RFC 6901 as it is appended, so a pointer built here
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
    ///    section means an unknown key (a typo, a version-1 key) fails this step exactly as it
    ///    would fail loading a file, from every [`Source`]. A `null` deletes a key, which for an
    ///    optional load parameter means "back to llama.cpp's automatic value".
    /// 3. The merged result must also pass [`Settings::validate`] -- a patch that, say, deletes
    ///    the level `effort.default_level` names is refused exactly as a hand-edited file would be.
    /// 4. For [`Source::Api`], the effective value at every [`SECURITY_LOCKED`] pointer must be
    ///    unchanged after the merge. This catches what step 1 cannot see: a `null` on an ancestor
    ///    (`{"models": null}`) that deletes a locked key without naming it and lets serde fill in
    ///    its default.
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

        let before = serde_json::to_value(&*self).map_err(|e| ConfigError::PatchRejected(e.to_string()))?;
        let mut merged = before.clone();
        merge_patch(&mut merged, patch);
        let next: Settings =
            serde_json::from_value(merged).map_err(|e| ConfigError::PatchRejected(crate::error::cap_echo(&e.to_string())))?;
        // Structural checks beyond what `serde` alone expresses apply here too, not just on load.
        next.validate()?;
        if source == Source::Api {
            // The key check above only sees names. A `null` on an ancestor of a partially locked
            // section (`{"models": null}` above `/models/folders`, `{"api": null}` above
            // `/api/bind`) deletes the locked key without naming it, and serde then fills in its
            // default. So compare the *effective* value at every locked pointer, after deserialize
            // and re-serialize, which also covers any locked key a section gains later.
            let after = serde_json::to_value(&next).map_err(|e| ConfigError::PatchRejected(e.to_string()))?;
            if let Some(locked) = SECURITY_LOCKED.iter().find(|p| before.pointer(p) != after.pointer(p)) {
                return Err(ConfigError::LockedPointerRefused {
                    pointer: (*locked).to_string(),
                    by: source,
                });
            }
        }
        *self = next;
        Ok(())
    }
}

/// Canonicalize `path` and require the result to be under one of `settings`'s model folders
/// ([`crate::schema::ModelsSection::folders`]).
///
/// Every check below returns the single [`ConfigError::ModelPathNotApproved`] variant, with no
/// distinguishing detail, so this function is not an oracle a caller could use to enumerate the
/// filesystem: "the file does not exist" and "the file exists but is outside every approved root"
/// look identical from the outside.
///
/// Rejected outright, on the literal input and again on the canonicalized result (so a reparse
/// point *inside* an approved root that resolves to one of these on the far side is still caught):
///
/// * any path segment that would make the string ambiguous as a plain path -- a comma (reserved
///   as list syntax in several `llama.cpp` flags, per `cerebro-supervisor::preflight`'s existing
///   rule), or a colon after the drive letter (an NTFS alternate data stream reference);
/// * a device or object-namespace path (`\\.\...`, `\\?\GLOBALROOT\...`, `\??\...`,
///   `\Device\...`, `\DosDevices\...`), which names no folder a user can list.
///
/// A network share is a folder like any other: a UNC model (`\\nas\models\x.gguf`, or its
/// verbatim form) resolves when it is under a UNC folder the user listed. A UNC input is checked
/// against the listed folders *lexically, before any filesystem access*, because resolving an
/// arbitrary UNC path makes Windows connect to that server and offer this account's credentials.
///
/// A reparse point (symlink or directory junction) that leaves every folder is caught
/// structurally: `std::fs::canonicalize` resolves it to its real target before the root check
/// runs, so the check compares real, final path components, not the link's apparent location.
/// That refusal is the user's to switch off: with
/// [`crate::policy::ProtectionsSection::refuse_redirected_paths`] off (the setting LM Studio and
/// Ollama users with shared GGUFs need), a path whose *literal* location is under a folder is
/// accepted wherever its links lead, and the canonical target is returned. The literal location is
/// compared after lexical normalization (`GetFullPathNameW`); a verbatim path containing `.` or
/// `..` is never accepted that way.
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

    let literal = Lexical::of(path);
    let literally_under_a_folder = || {
        literal.as_ref().is_some_and(|l| {
            settings.models.folders.iter().filter_map(|f| Lexical::of(f)).any(|root| l.is_under(&root))
        })
    };
    if is_unc(path) && !literally_under_a_folder() {
        return Err(ConfigError::ModelPathNotApproved);
    }

    let canonical = std::fs::canonicalize(path).map_err(|_| ConfigError::ModelPathNotApproved)?;
    reject_path_hazards(&canonical)?;

    let canonical_components: Vec<Component> = canonical.components().collect();
    for root in &settings.models.folders {
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

    // The target left every folder through a link. That is refused unless the user switched the
    // redirect refusal off, in which case the literal location decides.
    if !settings.protections.refuse_redirected_paths && literally_under_a_folder() {
        return Ok(canonical);
    }

    Err(ConfigError::ModelPathNotApproved)
}

/// Whether `path` is a UNC path, plain (`\\server\share`) or verbatim (`\\?\UNC\server\share`).
fn is_unc(path: &Path) -> bool {
    matches!(
        path.components().next(),
        Some(Component::Prefix(p)) if matches!(p.kind(), Prefix::UNC(..) | Prefix::VerbatimUNC(..))
    )
}

/// Where a path points, worked out from its text alone (no filesystem access), for comparing a
/// literal location against a folder.
#[derive(Debug, PartialEq, Eq)]
struct Lexical {
    root: LexicalRoot,
    parts: Vec<String>,
}

/// The volume a [`Lexical`] location is on. Plain and verbatim spellings of the same volume are
/// the same root; letters and server/share names compare ASCII-case-insensitively, as Windows does.
#[derive(Debug, PartialEq, Eq)]
enum LexicalRoot {
    /// A drive letter, upper-cased.
    Disk(u8),
    /// A server and share, ASCII-lower-cased.
    Unc(String, String),
    /// `/` (non-Windows).
    Posix,
}

impl Lexical {
    /// `path` made absolute with `std::path::absolute` (on Windows `GetFullPathNameW`, the same
    /// lexical normalization Win32 applies when it opens a path: `.` and `..` resolved, the
    /// working directory applied to a relative path). `None`, never a guess, for anything that is
    /// not a plain disk, UNC or POSIX path, for a non-UTF-8 component, and for a `.`/`..` left in a
    /// verbatim path (which Win32 does not normalize).
    fn of(path: &Path) -> Option<Lexical> {
        let absolute = std::path::absolute(path).ok()?;
        let mut components = absolute.components().peekable();
        let root = match components.peek() {
            Some(Component::Prefix(p)) => {
                let root = match p.kind() {
                    Prefix::Disk(d) | Prefix::VerbatimDisk(d) => LexicalRoot::Disk(d.to_ascii_uppercase()),
                    Prefix::UNC(server, share) | Prefix::VerbatimUNC(server, share) => LexicalRoot::Unc(
                        server.to_str()?.to_ascii_lowercase(),
                        share.to_str()?.to_ascii_lowercase(),
                    ),
                    _ => return None,
                };
                components.next();
                root
            }
            Some(Component::RootDir) => LexicalRoot::Posix,
            _ => return None,
        };
        let mut parts = Vec::new();
        for component in components {
            match component {
                Component::RootDir => {}
                Component::Normal(name) => {
                    let name = name.to_str()?;
                    if name == "." || name == ".." {
                        return None;
                    }
                    parts.push(name.to_owned());
                }
                Component::CurDir | Component::ParentDir | Component::Prefix(_) => return None,
            }
        }
        Some(Lexical { root, parts })
    }

    /// Whether `self` is `folder` or lies beneath it, component by component.
    fn is_under(&self, folder: &Lexical) -> bool {
        self.root == folder.root
            && folder.parts.len() <= self.parts.len()
            && folder.parts.iter().zip(&self.parts).all(|(a, b)| a.eq_ignore_ascii_case(b))
    }
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
/// APIs, so this module stays free of `unsafe`.
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
    if is_device_path(&raw) {
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

/// Whether `raw` is a device or object-namespace path (`\\.\...`, a verbatim path that is neither
/// a disk nor UNC such as `\\?\GLOBALROOT\...` or `\\?\Volume{...}\...`, or an NT native path),
/// i.e. anything that is neither a drive-letter path nor a UNC share. UNC shares are left to
/// [`resolve_model_path`]'s folder check.
fn is_device_path(raw: &str) -> bool {
    // Win32 accepts `/` as a separator in `\\.\` and UNC prefixes (`//./pipe/x`), so compare with
    // every separator spelled `\`. Borrowed, not copied, in the common case of no `/` at all.
    let normalized: std::borrow::Cow<'_, str> =
        if raw.contains('/') { raw.replace('/', r"\").into() } else { raw.into() };
    let raw = normalized.as_ref();
    if let Some(rest) = raw.strip_prefix(r"\\?\") {
        // Verbatim local disk paths (`\\?\C:\...`) are exactly what `std::fs::canonicalize`
        // itself produces on Windows, and `\\?\UNC\server\share\...` is its spelling of a share.
        let is_verbatim_unc = rest.get(..4).is_some_and(|p| p.eq_ignore_ascii_case(r"UNC\"));
        return !(is_drive_letter_root(rest) || is_verbatim_unc);
    }
    if raw.starts_with(r"\\.\") || raw.starts_with(r"\\??\") {
        return true; // Win32 device namespace
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::full_settings_fixture;

    #[test]
    fn every_guarding_section_is_locked_with_its_descendants() {
        for pointer in [
            "/backend",
            "/backend/dir",
            "/api/bind",
            "/models/folders",
            "/models/folders/0",
            "/network",
            "/network/lan_mode",
            "/protections",
            "/protections/audit_folder_permissions",
            "/plugins/capabilities/run_commands",
        ] {
            assert!(is_security_locked(pointer), "{pointer} must be locked");
        }
        // And every entry of the constant itself, with a descendant, so an entry added to the list
        // is covered without editing this test.
        for locked in SECURITY_LOCKED {
            assert!(is_security_locked(locked), "{locked}");
            assert!(is_security_locked(&format!("{locked}/child")), "{locked}/child");
        }
    }

    #[test]
    fn sibling_and_prefix_lookalike_pointers_are_not_locked() {
        for pointer in [
            "/models/default_model",
            "/api/enabled",
            "/api/port",
            "/load/defaults/n_ctx",
            "/effort/levels/high/instruction",
            "/verbosity/default_level",
            // A pointer that merely starts with the same characters as a locked one, without a `/`
            // boundary, must not match.
            "/protectionsish",
            "/backendish/x",
            "/models/folders2",
        ] {
            assert!(!is_security_locked(pointer), "{pointer} must not be locked");
        }
    }

    #[test]
    fn escaped_pointers_round_trip_and_are_not_decoded_before_comparison() {
        // A key literally named `bind/x` under `/api` escapes to `/api/bind~1x`, a sibling of
        // `/api/bind`, not the same pointer.
        assert!(!is_security_locked("/api/bind~1x"));
        assert!(is_security_locked("/api/bind"));

        let mut out = Vec::new();
        touched_pointers(&serde_json::json!({"api": {"bind/x": 1}}), "", &mut out);
        assert!(out.contains(&"/api/bind~1x".to_string()), "{out:?}");
        assert!(!out.iter().any(|p| is_security_locked(p)), "{out:?} must not lock /api/bind~1x");
    }

    #[test]
    fn api_source_is_refused_on_every_locked_pointer() {
        let attempts = [
            serde_json::json!({"network": {"lan_mode": true}}),
            serde_json::json!({"network": {"mode": "online"}}),
            serde_json::json!({"models": {"folders": ["C:\\"]}}),
            serde_json::json!({"backend": {"dir": "C:\\elsewhere"}}),
            serde_json::json!({"api": {"bind": "loopback6"}}),
            serde_json::json!({"protections": {"verify_backend_integrity": false}}),
            serde_json::json!({"plugins": {"default_permission": "always"}}),
        ];
        for patch in attempts {
            let mut settings = full_settings_fixture();
            let before = settings.clone();
            let err = settings.apply_patch(&patch, Source::Api).unwrap_err();
            assert!(matches!(err, ConfigError::LockedPointerRefused { .. }), "patch {patch} got {err:?}");
            assert_eq!(settings, before, "a refused patch must not be partially applied");
        }
    }

    #[test]
    fn api_null_on_an_ancestor_of_a_locked_key_is_refused() {
        // `{"models": null}` names only `/models`, which is not locked, yet deleting it resets
        // `models.folders` to empty; `{"api": null}` resets `api.bind` from loopback6 to loopback4.
        for (patch, locked) in [
            (serde_json::json!({"models": null}), "/models/folders"),
            (serde_json::json!({"api": null}), "/api/bind"),
        ] {
            let mut settings = full_settings_fixture();
            assert!(!settings.models.folders.is_empty() && settings.api.bind == crate::schema::BindAddr::Loopback6);
            let before = settings.clone();
            let err = settings.apply_patch(&patch, Source::Api).unwrap_err();
            assert_eq!(
                err,
                ConfigError::LockedPointerRefused { pointer: locked.to_string(), by: Source::Api },
                "{patch}"
            );
            assert_eq!(settings, before, "{patch} must not be partially applied");
            // The same patch from the user is an ordinary reset to defaults.
            settings.apply_patch(&patch, Source::Gui).expect("the user may reset a section");
        }
        // Siblings of a partially locked key stay writable, including by an ancestor `null` when
        // the locked value does not change (the defaults already hold it).
        let mut settings = full_settings_fixture();
        settings
            .apply_patch(&serde_json::json!({"models": {"default_model": "x"}}), Source::Api)
            .expect("a sibling of /models/folders stays writable");
        assert_eq!(settings.models.default_model.as_deref(), Some("x"));
        let mut defaults = Settings::default();
        defaults
            .apply_patch(&serde_json::json!({"api": null}), Source::Api)
            .expect("resetting /api changes no locked value when bind is already the default");
    }

    #[test]
    fn the_user_may_turn_any_protection_off_from_the_gui_or_the_file() {
        // Owner rule: every protection is an ordinary setting for the user. Each switch, from both
        // user sources, lands in the supervisor mapping.
        let keys = [
            "verify_backend_integrity",
            "audit_folder_permissions",
            "enforce_environment_allowlist",
            "contain_in_job_object",
            "refuse_elevation",
            "refuse_redirected_paths",
            "sandbox_plugins",
            "permission_prompts",
        ];
        for source in [Source::Gui, Source::File] {
            for key in keys {
                let mut settings = Settings::default();
                settings
                    .apply_patch(&serde_json::json!({"protections": {key: false}}), source)
                    .unwrap_or_else(|e| panic!("{source:?} could not turn {key} off: {e}"));
                assert_eq!(settings.disabled_protections().len(), 1, "{source:?} {key}");
                let value = serde_json::to_value(settings.protections).expect("serializes");
                assert_eq!(value[key], false, "{key}");
            }
        }
    }

    #[test]
    fn api_source_may_still_write_unlocked_pointers() {
        let mut settings = full_settings_fixture();
        settings
            .apply_patch(&serde_json::json!({"sampler": {"temp": 0.42}}), Source::Api)
            .expect("unlocked pointer is not refused");
        assert_eq!(settings.sampler.temp(), 0.42);
        settings
            .apply_patch(
                &serde_json::json!({"effort": {"levels": {"high": {"instruction": "Edited."}}}}),
                Source::Api,
            )
            .expect("a level instruction is not locked");
        assert_eq!(settings.effort.levels["high"].instruction, "Edited.");
    }

    #[test]
    fn a_null_in_a_patch_resets_a_load_parameter_to_automatic() {
        let mut settings = full_settings_fixture();
        assert!(settings.load.defaults.n_gpu_layers.is_some());
        settings
            .apply_patch(&serde_json::json!({"load": {"defaults": {"n_gpu_layers": null}}}), Source::Gui)
            .expect("reset applies");
        assert_eq!(settings.load.defaults.n_gpu_layers, None);
        assert!(settings.load.defaults.n_ctx.is_some(), "siblings untouched");
    }

    #[test]
    fn a_patch_that_breaks_a_cross_field_rule_is_refused_whole() {
        for patch in [
            serde_json::json!({"effort": {"levels": {"high": null}}}),
            serde_json::json!({"load": {"active_preset": "missing"}, "sampler": {"temp": 0.1}}),
        ] {
            let mut settings = full_settings_fixture();
            let before = settings.clone();
            let err = settings.apply_patch(&patch, Source::Gui).unwrap_err();
            assert!(matches!(err, ConfigError::SettingRejected { .. }), "{patch}: {err:?}");
            assert_eq!(settings, before, "nothing from {patch} may be applied");
        }
    }

    #[test]
    fn api_patch_with_an_escaped_lookalike_key_is_an_unknown_field_not_a_lock() {
        let mut settings = full_settings_fixture();
        let err = settings
            .apply_patch(&serde_json::json!({"api": {"bind/x": true}}), Source::Api)
            .unwrap_err();
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
        for source in [Source::File, Source::Gui, Source::Api] {
            let mut settings = full_settings_fixture();
            let before = settings.clone();
            let patch = serde_json::json!({"version": crate::schema::SCHEMA_VERSION + 1});
            let err = settings.apply_patch(&patch, source).unwrap_err();
            assert!(matches!(err, ConfigError::UnsupportedVersion { .. }), "source {source:?} got {err:?}");
            assert_eq!(settings, before, "a refused patch must not be partially applied");
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
        settings.models.folders = vec![root.to_path_buf()];
        settings
    }

    #[test]
    fn a_file_under_a_model_folder_resolves() {
        let (_dir, root, file) = approved_root_with_file();
        let settings = settings_with_root(&root);
        let resolved = resolve_model_path(&settings, &file).expect("under a model folder");
        assert_eq!(resolved, std::fs::canonicalize(&file).expect("canonicalize"));
    }

    #[test]
    fn open_approved_model_returns_a_readable_handle_to_the_approved_file() {
        let (_dir, root, file) = approved_root_with_file();
        let settings = settings_with_root(&root);
        let (resolved, mut handle) = open_approved_model(&settings, &file).expect("under a model folder");
        assert_eq!(resolved, std::fs::canonicalize(&file).expect("canonicalize"));
        let mut contents = Vec::new();
        std::io::Read::read_to_end(&mut handle, &mut contents).expect("read through the returned handle");
        assert_eq!(contents, b"not a real gguf");
    }

    #[test]
    fn open_approved_model_refuses_a_path_outside_every_folder() {
        let (dir, root, _file) = approved_root_with_file();
        let settings = settings_with_root(&root);
        let outside = dir.path().join("outside.gguf");
        std::fs::write(&outside, b"x").expect("write outside file");
        assert_eq!(open_approved_model(&settings, &outside).unwrap_err(), ConfigError::ModelPathNotApproved);
    }

    #[test]
    fn missing_and_outside_root_produce_the_identical_error() {
        let (dir, root, _file) = approved_root_with_file();
        let settings = settings_with_root(&root);
        let missing_err = resolve_model_path(&settings, &root.join("does-not-exist.gguf")).unwrap_err();
        let outside = dir.path().join("outside.gguf");
        std::fs::write(&outside, b"x").expect("write outside file");
        let outside_err = resolve_model_path(&settings, &outside).unwrap_err();
        assert_eq!(missing_err, ConfigError::ModelPathNotApproved);
        assert_eq!(missing_err, outside_err, "the two failure causes must be indistinguishable");
    }

    #[test]
    fn a_sibling_directory_whose_name_extends_the_root_is_rejected() {
        // `...\models-evil` has `...\models` as a *string* prefix but is not under it; comparing
        // `Component`s cannot confuse the two.
        let (dir, root, _file) = approved_root_with_file();
        let lookalike = dir.path().join("models-evil");
        std::fs::create_dir_all(&lookalike).expect("create lookalike dir");
        let evil = lookalike.join("evil.gguf");
        std::fs::write(&evil, b"x").expect("write evil file");
        let settings = settings_with_root(&root);
        assert_eq!(resolve_model_path(&settings, &evil).unwrap_err(), ConfigError::ModelPathNotApproved);
        assert_eq!(open_approved_model(&settings, &evil).unwrap_err(), ConfigError::ModelPathNotApproved);
    }

    #[test]
    fn dot_dot_escape_from_a_model_folder_is_rejected() {
        let (dir, root, _file) = approved_root_with_file();
        let sibling = dir.path().join("sibling");
        std::fs::create_dir_all(&sibling).expect("create sibling dir");
        std::fs::write(sibling.join("evil.gguf"), b"x").expect("write evil file");
        let settings = settings_with_root(&root);
        let escape_path = root.join("..").join("sibling").join("evil.gguf");
        assert_eq!(resolve_model_path(&settings, &escape_path).unwrap_err(), ConfigError::ModelPathNotApproved);
    }

    #[test]
    fn alternate_data_stream_unc_globalroot_and_comma_paths_are_rejected() {
        let (_dir, root, file) = approved_root_with_file();
        let settings = settings_with_root(&root);
        let commaed = root.join("model,evil.gguf");
        std::fs::write(&commaed, b"x").expect("write file with comma in name");
        for path in [
            PathBuf::from(format!("{}:secret", file.display())),
            PathBuf::from(r"\\server\share\model.gguf"),
            PathBuf::from(r"\\?\GLOBALROOT\Device\Something\model.gguf"),
            commaed,
        ] {
            assert_eq!(
                resolve_model_path(&settings, &path).unwrap_err(),
                ConfigError::ModelPathNotApproved,
                "{}",
                path.display()
            );
        }
    }

    #[test]
    fn no_model_folder_means_nothing_resolves() {
        let (_dir, _root, file) = approved_root_with_file();
        assert_eq!(
            resolve_model_path(&Settings::default(), &file).unwrap_err(),
            ConfigError::ModelPathNotApproved
        );
    }

    /// Needs no admin: unlike a symlink, a directory junction can be created by any local user.
    #[cfg(windows)]
    #[test]
    fn directory_junction_escaping_the_folder_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("models");
        std::fs::create_dir_all(&root).expect("create root");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&outside).expect("create outside dir");
        std::fs::write(outside.join("evil.gguf"), b"x").expect("write evil file");

        let link = root.join("junction");
        cerebro_probe::backend::create_junction_for_tests(&link, &outside).expect("create junction");

        let settings = settings_with_root(&root);
        let err = resolve_model_path(&settings, &link.join("evil.gguf")).unwrap_err();
        assert_eq!(err, ConfigError::ModelPathNotApproved);
    }

    /// Owner rule: the redirect refusal is a protection, so the user can switch it off (shared
    /// LM Studio / Ollama GGUFs reached through links). Off, the literal location decides.
    #[cfg(windows)]
    #[test]
    fn a_linked_model_resolves_when_the_user_turns_the_redirect_refusal_off() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("models");
        std::fs::create_dir_all(&root).expect("create root");
        let shared = dir.path().join("shared");
        std::fs::create_dir_all(&shared).expect("create shared dir");
        std::fs::write(shared.join("m.gguf"), b"gguf").expect("write shared model");
        let link = root.join("lmstudio");
        cerebro_probe::backend::create_junction_for_tests(&link, &shared).expect("create junction");

        let mut settings = settings_with_root(&root);
        let via_link = link.join("m.gguf");
        assert_eq!(resolve_model_path(&settings, &via_link).unwrap_err(), ConfigError::ModelPathNotApproved);

        settings.protections.refuse_redirected_paths = false;
        let target = std::fs::canonicalize(shared.join("m.gguf")).expect("canonicalize target");
        assert_eq!(resolve_model_path(&settings, &via_link).expect("literal path is under a folder"), target);
        let (opened, mut handle) = open_approved_model(&settings, &via_link).expect("opens the real target");
        assert_eq!(opened, target);
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut handle, &mut bytes).expect("read");
        assert_eq!(bytes, b"gguf");

        // The literal location must itself be under a folder: the switch does not open `..` escapes
        // or a path that names the shared folder directly.
        let dotdot = root.join("..").join("shared").join("m.gguf");
        assert_eq!(resolve_model_path(&settings, &dotdot).unwrap_err(), ConfigError::ModelPathNotApproved);
        assert_eq!(
            resolve_model_path(&settings, &shared.join("m.gguf")).unwrap_err(),
            ConfigError::ModelPathNotApproved
        );
    }

    #[test]
    fn device_namespace_paths_are_refused_in_every_spelling() {
        let (_dir, root, _file) = approved_root_with_file();
        let mut settings = settings_with_root(&root);
        settings.protections.refuse_redirected_paths = false;
        for raw in [
            r"\\.\pipe\model.gguf",
            "//./pipe/model.gguf",
            r"\\?\GLOBALROOT\Device\HarddiskVolume1\model.gguf",
            r"\\?\Volume{00000000-0000-0000-0000-000000000000}\model.gguf",
            r"\??\C:\model.gguf",
            r"\Device\HarddiskVolume1\model.gguf",
        ] {
            assert!(is_device_path(raw), "{raw}");
            assert_eq!(
                resolve_model_path(&settings, Path::new(raw)).unwrap_err(),
                ConfigError::ModelPathNotApproved,
                "{raw}"
            );
        }
        for raw in [r"\\?\C:\models\m.gguf", r"\\?\UNC\nas\models\m.gguf", r"\\nas\models\m.gguf", r"C:\m.gguf"] {
            assert!(!is_device_path(raw), "{raw}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn lexical_locations_match_across_spellings_and_fail_closed() {
        let of = |s: &str| Lexical::of(Path::new(s));
        let share = of(r"\\NAS\Models").expect("plain UNC");
        for inside in [r"\\nas\models\a.gguf", r"\\?\UNC\nas\MODELS\sub\a.gguf", r"\\nas\models\x\..\a.gguf"] {
            assert!(of(inside).expect(inside).is_under(&share), "{inside}");
        }
        for outside in [r"\\nas\other\a.gguf", r"\\evil\models\a.gguf", r"C:\models\a.gguf"] {
            assert!(!of(outside).expect(outside).is_under(&share), "{outside}");
        }
        let disk = of(r"C:\Models").expect("disk");
        assert!(of(r"\\?\c:\models\a.gguf").expect("verbatim disk").is_under(&disk));
        assert!(!of(r"C:\models-evil\a.gguf").expect("lookalike").is_under(&disk));
        // Verbatim paths are not normalized by Win32, so a `..` left in one is never guessed at.
        assert_eq!(of(r"\\?\C:\models\..\a.gguf"), None);
        assert_eq!(of(r"\\.\pipe\x"), None);
    }

    #[cfg(windows)]
    #[test]
    fn a_unc_model_outside_every_unc_folder_is_refused_before_any_filesystem_access() {
        // The folder check for a UNC input is lexical, so this refusal never contacts the named
        // server (`.invalid` cannot resolve even if it did). A share the user listed is accepted
        // by the same check (see `lexical_locations_match_across_spellings_and_fail_closed`).
        let (_dir, root, _file) = approved_root_with_file();
        let mut settings = settings_with_root(&root);
        settings.models.folders.push(PathBuf::from(r"\\nas.invalid\models"));
        settings.protections.refuse_redirected_paths = false;
        for raw in [r"\\attacker.invalid\share\m.gguf", r"\\?\UNC\attacker.invalid\share\m.gguf", r"\\nas.invalid\other\m.gguf"] {
            assert_eq!(
                resolve_model_path(&settings, Path::new(raw)).unwrap_err(),
                ConfigError::ModelPathNotApproved,
                "{raw}"
            );
        }
    }
}