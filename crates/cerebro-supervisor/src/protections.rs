//! User-switchable safety protections.
//!
//! Owner rule: speed and full user control are core values, so every protection this crate can
//! enforce — the backend integrity check, the folder-permission (DACL) audit, the spawn-time
//! environment allowlist, Job Object process containment, refusing to run elevated, and refusing a
//! model, key, temp or log path that is a symlink or junction — can be turned off in settings.
//! Every other refusal is a correctness constraint no setting could make work (a comma in a model
//! path, a `;` in the backend path, a missing file, a key file llama.cpp would parse as empty). [`Protections`] defaults every one of them on (safe by default); turning
//! one off is honoured immediately and unconditionally by whichever module owns that check — it is
//! never refused, and never asks twice. The one thing it always does is leave behind exactly one
//! neutral, structured [`ProtectionWarning`] in the launch record, so a switch left off is
//! discoverable later without being a nag repeated on every single launch.
//!
//! Two modules each own part of the launch record this type feeds:
//! [`crate::preflight::Preflighted::warnings`] carries a [`crate::preflight::PreflightWarning::ProtectionDisabled`]
//! for each of the four checks [`crate::preflight::preflight`] itself is responsible for
//! ([`Protections::refuse_elevation`], [`Protections::audit_folder_permissions`],
//! [`Protections::verify_backend_integrity`], [`Protections::refuse_redirected_paths`]);
//! [`crate::spawn::ChildHandle::protection_warnings`] carries the two [`crate::spawn`] is
//! responsible for ([`Protections::enforce_environment_allowlist`],
//! [`Protections::contain_in_job_object`]).
//!
//! [`Protections::refuse_redirected_paths`] is enforced in both places (model and key files in
//! preflight, temp and log directories in spawn) but recorded once: by preflight for a preflighted
//! launch, and by [`crate::spawn::spawn_unaudited`] (which has no preflight record) otherwise.
//!
//! `cerebro_probe::backend::probe_with` also starts the backend (its `--version`/`--list-devices`
//! queries load every ggml DLL), so the three switches it can honour, the folder audit, the
//! environment allowlist and the elevation refusal, reach it through
//! [`Protections::probe_settings`], the one mapping, and are recorded on the `BackendProbe` it
//! returns. The probe does not check the backend against a pinned manifest: it computes the identity
//! a manifest is built from, and [`crate::preflight::preflight`] verifies it before every launch.

use std::fmt;

/// Which safety protection a [`ProtectionWarning`] is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Protection {
    /// [`Protections::verify_backend_integrity`]: the SHA-256 manifest check
    /// [`crate::preflight::preflight`] runs against every `.exe`/`.dll` in the backend directory.
    BackendIntegrityVerification,
    /// [`Protections::audit_folder_permissions`]: [`crate::preflight::preflight`]'s DACL audit of
    /// the backend directory and the path used to reach it.
    FolderPermissionAudit,
    /// [`Protections::enforce_environment_allowlist`]: [`crate::spawn`]'s closed,
    /// rebuilt-from-nothing child environment (see [`cerebro_probe::child_env`]).
    EnvironmentAllowlist,
    /// [`Protections::contain_in_job_object`]: [`crate::spawn`]'s Windows Job Object containment.
    JobObjectContainment,
    /// [`Protections::refuse_elevation`]: [`crate::preflight::preflight`]'s refusal to launch while
    /// this process itself is running elevated.
    ElevationRefusal,
    /// [`Protections::refuse_redirected_paths`]: the refusal of a model, draft, API-key file, temp
    /// directory or log directory that is (or is reached through) a symlink or junction.
    RedirectedPathRefusal,
}

impl Protection {
    /// Stable, machine-readable identifier for this protection, in snake_case. Never changes
    /// across releases — unlike [`Protection`]'s `Display` text (meant for a human to read) or its
    /// variant name (which Rust source, not a persisted record, is free to rename), this is what a
    /// hash-chained store record or any other durable log should key a disabled protection on.
    pub fn key(self) -> &'static str {
        match self {
            Protection::BackendIntegrityVerification => "backend_integrity_verification",
            Protection::FolderPermissionAudit => "folder_permission_audit",
            Protection::EnvironmentAllowlist => "environment_allowlist",
            Protection::JobObjectContainment => "job_object_containment",
            Protection::ElevationRefusal => "elevation_refusal",
            Protection::RedirectedPathRefusal => "redirected_path_refusal",
        }
    }
}

impl Protection {
    /// The neutral clause a refusal this protection governs ends with: "turn the <protection> off in
    /// settings to allow this". Built from [`Protection`]'s `Display`, so every message names the
    /// switch with the same words a settings UI shows. See
    /// [`crate::preflight::PreflightError::governing_protection`].
    pub fn settings_hint(self) -> String {
        format!("turn the {self} off in settings to allow this")
    }
}

impl fmt::Display for Protection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Protection::BackendIntegrityVerification => "backend integrity verification",
            Protection::FolderPermissionAudit => "backend folder permission audit",
            Protection::EnvironmentAllowlist => "child environment allowlist",
            Protection::JobObjectContainment => "job-object process containment",
            Protection::ElevationRefusal => "elevation refusal",
            Protection::RedirectedPathRefusal => "redirected path refusal",
        })
    }
}

/// One protection the user turned off, plus the risk that switch accepts.
///
/// A structured value, not a plain string, so a caller — or a settings UI listing what is currently
/// disabled — can group, filter, or re-render these by [`Protection`] without re-parsing text meant
/// for a human. Only ever constructed inside this crate (by [`Protections`]'s own `*_warning`
/// methods): every instance that exists names a real, specific risk one of those methods wrote down,
/// never an ad hoc message assembled at a call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectionWarning {
    protection: Protection,
    risk: &'static str,
}

impl ProtectionWarning {
    fn new(protection: Protection, risk: &'static str) -> Self {
        Self { protection, risk }
    }

    /// Which protection this warning is about.
    pub fn protection(&self) -> Protection {
        self.protection
    }

    /// The risk this protection being off accepts, in one plain sentence.
    pub fn risk(&self) -> &str {
        self.risk
    }
}

impl fmt::Display for ProtectionWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} is turned off in settings: {}",
            self.protection, self.risk
        )
    }
}

/// Explicit, per-protection on/off switches for every safety check
/// [`crate::preflight`](mod@crate::preflight) and [`crate::spawn`] can enforce.
///
/// Every field defaults to `true` ([`Protections::default`]) — the safe-by-default half of the
/// owner rule this type exists for. The other half: the user can turn off any one of them, and doing
/// so is always honoured, never refused, and never asked about twice; see each field's own docs for
/// exactly what turning it off changes, and this module's top-level docs for where the resulting
/// [`ProtectionWarning`] ends up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Protections {
    /// Verify every `.exe`/`.dll` in the backend directory against the pinned manifest before
    /// spawning. Enforced by [`crate::preflight::preflight`]; off skips the hash check entirely (no
    /// file is even read), and every backend file already there is trusted as-is.
    pub verify_backend_integrity: bool,
    /// Audit the backend directory's, and its path's, DACL before spawning. Enforced by
    /// [`crate::preflight::preflight`]; off skips the audit entirely. Distinct from
    /// [`crate::preflight::AllowUnsafeDacl`], which accepts one *specific, already-audited* unsafe
    /// result with a recorded reason, rather than turning the audit off altogether.
    pub audit_folder_permissions: bool,
    /// Rebuild the environment of every backend process Cerebro starts from a closed allowlist
    /// instead of letting it inherit Cerebro's own. Enforced by [`crate::spawn`] for the server and,
    /// through [`Protections::probe_settings`], by `cerebro_probe::backend::probe_with` for its
    /// `--version`/`--list-devices` queries; both build the block with
    /// [`cerebro_probe::child_env::effective_child_env`]. Off lets the child inherit this process's
    /// ambient environment. That includes `GGML_BACKEND_PATH`, which makes ggml `LoadLibrary` a DLL
    /// from outside the audited, hash-verified backend directory (a code-execution path around both
    /// the folder audit and the integrity check), any stray `LLAMA_ARG_*` variable, and the inherited
    /// `PATH` (placed after the backend directory and System32, so the backend's own DLLs still win).
    /// `TEMP`/`TMP` and `CUDA_VISIBLE_DEVICES` stay pinned either way: they are launch settings, not
    /// part of this protection.
    pub enforce_environment_allowlist: bool,
    /// Assign the spawned child to a Windows Job Object, so its whole process tree is killed with
    /// it. Enforced by [`crate::spawn`]; off leaves the child (and anything it spawns) outside any
    /// job, so only the immediate PID is ever guaranteed to stop when Cerebro does.
    pub contain_in_job_object: bool,
    /// Refuse to start backend code while this process itself is running elevated. Enforced by
    /// [`crate::preflight::preflight`] before every launch and, through
    /// [`Protections::probe_settings`], by `cerebro_probe::backend::probe_with` before its queries;
    /// both ask [`cerebro_probe::token::is_elevated`]. Off allows either to proceed from an elevated
    /// process.
    pub refuse_elevation: bool,
    /// Refuse a model, draft model or API-key file that is itself a symlink or junction, and a temp
    /// or crash-log directory with a symlink or junction anywhere on its path. Enforced by
    /// [`crate::preflight::preflight`] (the files) and [`crate::spawn`] (the directories). Off
    /// follows the link, which a GGUF shared with LM Studio or Ollama, a Hugging Face cache snapshot
    /// (symlinks to blobs) or a profile folder moved to another drive needs. Even then the child is
    /// never handed the link: the path in argv, `TEMP`/`TMP` and the log path is the target,
    /// resolved from the handle that was opened and checked, so repointing the link later changes
    /// nothing for this launch. Distinct from [`Protections::audit_folder_permissions`], which is
    /// about the backend directory.
    pub refuse_redirected_paths: bool,
}

impl Default for Protections {
    /// Every protection on — the safe-by-default half of the owner rule this type exists for.
    fn default() -> Self {
        Self {
            verify_backend_integrity: true,
            audit_folder_permissions: true,
            enforce_environment_allowlist: true,
            contain_in_job_object: true,
            refuse_elevation: true,
            refuse_redirected_paths: true,
        }
    }
}

impl Protections {
    /// The one supported mapping from [`Protections::audit_folder_permissions`] to
    /// [`cerebro_probe::backend::probe_with`]'s switch.
    ///
    /// Both crates enforce the folder-permission audit: preflight before a launch, `probe_with`
    /// before it identifies a backend. Deriving probe's argument here, from the same value
    /// preflight reads, means the two can never honour different answers for one user setting.
    /// Pass `protections.folder_audit()` to `probe_with`; do not build a
    /// [`cerebro_probe::backend::FolderAudit`] separately from the settings.
    pub fn folder_audit(&self) -> cerebro_probe::backend::FolderAudit {
        if self.audit_folder_permissions {
            cerebro_probe::backend::FolderAudit::Enforce
        } else {
            cerebro_probe::backend::FolderAudit::SkippedByUserSetting
        }
    }

    /// The one supported mapping from these settings to every switch
    /// [`cerebro_probe::backend::probe_with`] honours: the folder audit (via
    /// [`Protections::folder_audit`]), the environment allowlist for its query children, and the
    /// elevation refusal.
    ///
    /// The probe's queries load every ggml DLL just like the supervised server, so the user's
    /// environment-allowlist answer must reach them too; deriving both here means the probe and the
    /// launch can never honour different answers for one setting. Pass
    /// `protections.probe_settings()` to `probe_with`; do not build a
    /// [`cerebro_probe::backend::ProbeSettings`] separately. The resulting
    /// [`cerebro_probe::backend::BackendProbe`] records all three switches.
    pub fn probe_settings(&self) -> cerebro_probe::backend::ProbeSettings {
        use cerebro_probe::backend::{ElevationRefusal, EnvironmentAllowlist, ProbeSettings};
        ProbeSettings {
            folder_audit: self.folder_audit(),
            environment_allowlist: if self.enforce_environment_allowlist {
                EnvironmentAllowlist::Enforce
            } else {
                EnvironmentAllowlist::SkippedByUserSetting
            },
            elevation_refusal: if self.refuse_elevation {
                ElevationRefusal::Enforce
            } else {
                ElevationRefusal::SkippedByUserSetting
            },
        }
    }
}

/// One `*_warning` method per field, each returning the fixed, structured warning
/// [`crate::preflight`](mod@crate::preflight) or [`crate::spawn`] records when that specific protection is off, or `None`
/// while it is still on. Kept here, next to the field each one describes, rather than scattered
/// across whichever call site turns a protection off — a single source of truth for both the
/// decision ("is this one off") and the wording of the risk it accepts.
impl Protections {
    /// See [`Protections::refuse_elevation`].
    pub fn elevation_warning(&self) -> Option<ProtectionWarning> {
        (!self.refuse_elevation).then(|| {
            ProtectionWarning::new(
                Protection::ElevationRefusal,
                "this launch is allowed to proceed even while Cerebro is running elevated, \
                 though nothing in the inference path needs admin rights",
            )
        })
    }

    /// See [`Protections::audit_folder_permissions`].
    pub fn folder_permission_warning(&self) -> Option<ProtectionWarning> {
        (!self.audit_folder_permissions).then(|| {
            ProtectionWarning::new(
                Protection::FolderPermissionAudit,
                "the backend directory's DACL, and the path used to reach it, are not being \
                 checked; a principal other than this account could plant a DLL there unnoticed",
            )
        })
    }

    /// See [`Protections::verify_backend_integrity`].
    pub fn integrity_warning(&self) -> Option<ProtectionWarning> {
        (!self.verify_backend_integrity).then(|| {
            ProtectionWarning::new(
                Protection::BackendIntegrityVerification,
                "backend .exe/.dll files are not checked against the pinned manifest before \
                 spawning; a replaced or tampered file would load unnoticed",
            )
        })
    }

    /// See [`Protections::enforce_environment_allowlist`].
    pub fn environment_allowlist_warning(&self) -> Option<ProtectionWarning> {
        (!self.enforce_environment_allowlist).then(|| {
            ProtectionWarning::new(
                Protection::EnvironmentAllowlist,
                "the child inherits this process's full environment instead of a closed \
                 allowlist; GGML_BACKEND_PATH can make it load a DLL from outside the checked \
                 backend folder, and a LLAMA_ARG_* variable can override a hardened flag",
            )
        })
    }

    /// See [`Protections::refuse_redirected_paths`].
    pub fn redirect_warning(&self) -> Option<ProtectionWarning> {
        (!self.refuse_redirected_paths).then(|| {
            ProtectionWarning::new(
                Protection::RedirectedPathRefusal,
                "model, key, temp and log paths may be symlinks or junctions; whoever can repoint \
                 one chooses the file or folder the next launch uses",
            )
        })
    }

    /// See [`Protections::contain_in_job_object`].
    pub fn job_containment_warning(&self) -> Option<ProtectionWarning> {
        (!self.contain_in_job_object).then(|| {
            ProtectionWarning::new(
                Protection::JobObjectContainment,
                "the child is not assigned to a Job Object; if Cerebro is killed or crashes, the \
                 server and anything it spawned keep running",
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_protections_are_all_on() {
        let p = Protections::default();
        assert!(p.verify_backend_integrity);
        assert!(p.audit_folder_permissions);
        assert!(p.enforce_environment_allowlist);
        assert!(p.contain_in_job_object);
        assert!(p.refuse_elevation);
        assert!(p.refuse_redirected_paths);
    }

    #[test]
    fn every_warning_is_none_while_its_protection_is_on() {
        let p = Protections::default();
        assert_eq!(p.elevation_warning(), None);
        assert_eq!(p.folder_permission_warning(), None);
        assert_eq!(p.integrity_warning(), None);
        assert_eq!(p.environment_allowlist_warning(), None);
        assert_eq!(p.job_containment_warning(), None);
        assert_eq!(p.redirect_warning(), None);
    }

    #[test]
    fn turning_off_elevation_refusal_produces_its_own_tagged_warning() {
        let p = Protections {
            refuse_elevation: false,
            ..Protections::default()
        };
        let w = p.elevation_warning().expect("elevation refusal is off");
        assert_eq!(w.protection(), Protection::ElevationRefusal);
        assert!(!w.risk().is_empty());
        // Every other protection is still on and must stay silent.
        assert_eq!(p.folder_permission_warning(), None);
        assert_eq!(p.integrity_warning(), None);
        assert_eq!(p.environment_allowlist_warning(), None);
        assert_eq!(p.job_containment_warning(), None);
    }

    #[test]
    fn turning_off_folder_permission_audit_produces_its_own_tagged_warning() {
        let p = Protections {
            audit_folder_permissions: false,
            ..Protections::default()
        };
        let w = p.folder_permission_warning().expect("folder audit is off");
        assert_eq!(w.protection(), Protection::FolderPermissionAudit);
        assert!(!w.risk().is_empty());
        assert_eq!(p.elevation_warning(), None);
    }

    #[test]
    fn turning_off_integrity_verification_produces_its_own_tagged_warning() {
        let p = Protections {
            verify_backend_integrity: false,
            ..Protections::default()
        };
        let w = p
            .integrity_warning()
            .expect("integrity verification is off");
        assert_eq!(w.protection(), Protection::BackendIntegrityVerification);
        assert!(!w.risk().is_empty());
        assert_eq!(p.elevation_warning(), None);
    }

    #[test]
    fn turning_off_the_environment_allowlist_produces_its_own_tagged_warning() {
        let p = Protections {
            enforce_environment_allowlist: false,
            ..Protections::default()
        };
        let w = p
            .environment_allowlist_warning()
            .expect("environment allowlist is off");
        assert_eq!(w.protection(), Protection::EnvironmentAllowlist);
        assert!(!w.risk().is_empty());
        assert_eq!(p.job_containment_warning(), None);
    }

    #[test]
    fn turning_off_job_containment_produces_its_own_tagged_warning() {
        let p = Protections {
            contain_in_job_object: false,
            ..Protections::default()
        };
        let w = p.job_containment_warning().expect("job containment is off");
        assert_eq!(w.protection(), Protection::JobObjectContainment);
        assert!(!w.risk().is_empty());
        assert_eq!(p.environment_allowlist_warning(), None);
    }

    #[test]
    fn turning_off_redirected_path_refusal_produces_its_own_tagged_warning() {
        let p = Protections {
            refuse_redirected_paths: false,
            ..Protections::default()
        };
        let w = p
            .redirect_warning()
            .expect("redirected-path refusal is off");
        assert_eq!(w.protection(), Protection::RedirectedPathRefusal);
        assert!(w.risk().contains("symlink"), "{}", w.risk());
        // A separate switch from the folder audit: neither moves the other.
        assert_eq!(p.folder_permission_warning(), None);
        assert_eq!(
            p.folder_audit(),
            cerebro_probe::backend::FolderAudit::Enforce
        );
        let audit_off = Protections {
            audit_folder_permissions: false,
            ..Protections::default()
        };
        assert_eq!(audit_off.redirect_warning(), None);
    }

    #[test]
    fn folder_audit_follows_the_one_setting_in_both_states() {
        use cerebro_probe::backend::FolderAudit;
        assert_eq!(Protections::default().folder_audit(), FolderAudit::Enforce);
        let off = Protections {
            audit_folder_permissions: false,
            ..Protections::default()
        };
        assert_eq!(off.folder_audit(), FolderAudit::SkippedByUserSetting);
        // Only the folder switch moves it: every other protection off leaves the audit enforced.
        let others_off = Protections {
            verify_backend_integrity: false,
            enforce_environment_allowlist: false,
            contain_in_job_object: false,
            refuse_elevation: false,
            refuse_redirected_paths: false,
            ..Protections::default()
        };
        assert_eq!(others_off.folder_audit(), FolderAudit::Enforce);
    }

    #[test]
    fn the_environment_allowlist_risk_names_ggml_backend_path() {
        // The one recorded warning is the user's only notice, so it must name the real risk: an
        // inherited GGML_BACKEND_PATH loads a DLL around both the folder audit and the hash check.
        let p = Protections {
            enforce_environment_allowlist: false,
            ..Protections::default()
        };
        let w = p.environment_allowlist_warning().expect("allowlist is off");
        assert!(w.risk().contains("GGML_BACKEND_PATH"), "{}", w.risk());
        assert!(w.risk().contains("LLAMA_ARG_"), "{}", w.risk());
    }

    #[test]
    fn probe_settings_follow_the_same_three_switches_in_both_states() {
        use cerebro_probe::backend::{
            ElevationRefusal, EnvironmentAllowlist, FolderAudit, ProbeSettings,
        };
        assert_eq!(
            Protections::default().probe_settings(),
            ProbeSettings::default()
        );
        let env_off = Protections {
            enforce_environment_allowlist: false,
            ..Protections::default()
        };
        assert_eq!(
            env_off.probe_settings(),
            ProbeSettings {
                folder_audit: FolderAudit::Enforce,
                environment_allowlist: EnvironmentAllowlist::SkippedByUserSetting,
                elevation_refusal: ElevationRefusal::Enforce,
            }
        );
        let audit_off = Protections {
            audit_folder_permissions: false,
            ..Protections::default()
        };
        assert_eq!(
            audit_off.probe_settings(),
            ProbeSettings {
                folder_audit: FolderAudit::SkippedByUserSetting,
                environment_allowlist: EnvironmentAllowlist::Enforce,
                elevation_refusal: ElevationRefusal::Enforce,
            }
        );
        let elevation_off = Protections {
            refuse_elevation: false,
            ..Protections::default()
        };
        assert_eq!(
            elevation_off.probe_settings(),
            ProbeSettings {
                folder_audit: FolderAudit::Enforce,
                environment_allowlist: EnvironmentAllowlist::Enforce,
                elevation_refusal: ElevationRefusal::SkippedByUserSetting,
            }
        );
        // Switches the probe does not own leave all three of its switches on.
        let others_off = Protections {
            verify_backend_integrity: false,
            contain_in_job_object: false,
            refuse_redirected_paths: false,
            ..Protections::default()
        };
        assert_eq!(others_off.probe_settings(), ProbeSettings::default());
    }

    #[test]
    fn every_protection_key_is_distinct_and_snake_case() {
        let all = [
            Protection::BackendIntegrityVerification,
            Protection::FolderPermissionAudit,
            Protection::EnvironmentAllowlist,
            Protection::JobObjectContainment,
            Protection::ElevationRefusal,
            Protection::RedirectedPathRefusal,
        ];
        let mut keys: Vec<&str> = all.iter().map(|p| p.key()).collect();
        let before = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(
            keys.len(),
            before,
            "every Protection must have a distinct key: {keys:?}"
        );
        for key in keys {
            assert!(
                key.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "{key:?} is not snake_case"
            );
        }
    }

    #[test]
    fn display_names_the_protection_and_the_risk() {
        let p = Protections {
            refuse_elevation: false,
            ..Protections::default()
        };
        let rendered = p.elevation_warning().unwrap().to_string();
        assert!(rendered.contains("elevation refusal"), "{rendered}");
        assert!(rendered.contains("turned off in settings"), "{rendered}");
    }
}
