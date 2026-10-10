//! Safety and privacy settings: protections, network mode, and plugin permission defaults.
//!
//! # Owner rule: every protection is an ordinary setting
//!
//! Every protection defaults on, and the user can turn any of them off from the settings file or
//! the GUI. Turning one off is honoured, never refused and never asked about twice; what remains is
//! one neutral, structured note per switch ([`Settings::disabled_protections`]) that a settings
//! screen can show once, plus the supervisor's own one-per-launch record. An earlier design made an
//! unsafe backend-folder override unrepresentable; the owner overruled it (docs/RESTART.md, "Product
//! requirements added 2026-09-22"), so the folder audit is now
//! [`ProtectionsSection::audit_folder_permissions`], a switch like the rest.
//!
//! What *is* restricted is who may write these keys: Cerebro's local HTTP API may not (see
//! [`crate::profile::SECURITY_LOCKED`]), because a client of the API is not the user at the
//! keyboard, and a live source turning its own guards off is not a preference change.
//!
//! # One mapping to the supervisor
//!
//! The six launch switches are enforced by `cerebro-supervisor`'s preflight and spawn, whose
//! [`cerebro_supervisor::Protections`] is the only type they read. [`ProtectionsSection::launch`]
//! is the one mapping from these settings to it, so the settings screen and the launch can never
//! honour different answers.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::schema::Settings;

fn on() -> bool {
    true
}

/// Every user-switchable protection. A missing key means on (protected).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectionsSection {
    /// Check every backend `.exe`/`.dll` against the pinned SHA-256 manifest before each launch.
    /// See [`cerebro_supervisor::Protections::verify_backend_integrity`].
    #[serde(default = "on")]
    pub verify_backend_integrity: bool,
    /// Audit who can write the backend folder and the path to it before each launch. See
    /// [`cerebro_supervisor::Protections::audit_folder_permissions`].
    #[serde(default = "on")]
    pub audit_folder_permissions: bool,
    /// Start backend processes with a closed environment instead of inheriting Cerebro's. See
    /// [`cerebro_supervisor::Protections::enforce_environment_allowlist`].
    #[serde(default = "on")]
    pub enforce_environment_allowlist: bool,
    /// Keep the backend's whole process tree in a Job Object so it dies with Cerebro. See
    /// [`cerebro_supervisor::Protections::contain_in_job_object`].
    #[serde(default = "on")]
    pub contain_in_job_object: bool,
    /// Refuse to start backend code while Cerebro runs elevated. See
    /// [`cerebro_supervisor::Protections::refuse_elevation`].
    #[serde(default = "on")]
    pub refuse_elevation: bool,
    /// Refuse model, key, temp and log paths that are symlinks or junctions. See
    /// [`cerebro_supervisor::Protections::refuse_redirected_paths`].
    #[serde(default = "on")]
    pub refuse_redirected_paths: bool,
    /// Run plugin and tool processes inside Cerebro's sandbox.
    #[serde(default = "on")]
    pub sandbox_plugins: bool,
    /// Ask before a plugin or tool uses a capability set to `ask` or `session`. Off treats those as
    /// `always`; an explicit `deny` still denies (see [`Settings::permission_for`]).
    #[serde(default = "on")]
    pub permission_prompts: bool,
}

impl Default for ProtectionsSection {
    /// Every protection on.
    fn default() -> Self {
        Self {
            verify_backend_integrity: true,
            audit_folder_permissions: true,
            enforce_environment_allowlist: true,
            contain_in_job_object: true,
            refuse_elevation: true,
            refuse_redirected_paths: true,
            sandbox_plugins: true,
            permission_prompts: true,
        }
    }
}

impl ProtectionsSection {
    /// The supervisor's launch protections, the one mapping from these settings. Pass it as
    /// `LaunchSpec::protections` (or use [`Settings::launch_spec`], which does).
    pub fn launch(&self) -> cerebro_supervisor::Protections {
        cerebro_supervisor::Protections {
            verify_backend_integrity: self.verify_backend_integrity,
            audit_folder_permissions: self.audit_folder_permissions,
            enforce_environment_allowlist: self.enforce_environment_allowlist,
            contain_in_job_object: self.contain_in_job_object,
            refuse_elevation: self.refuse_elevation,
            refuse_redirected_paths: self.refuse_redirected_paths,
        }
    }
}

/// One protection the user turned off, for a settings screen's single neutral note.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisabledProtection {
    /// Stable snake_case key; for launch protections, the supervisor's own
    /// [`cerebro_supervisor::Protection::key`].
    pub key: &'static str,
    /// Human-readable name, e.g. `"backend integrity verification"`.
    pub name: String,
    /// The risk accepted, in one plain sentence.
    pub risk: String,
}

/// Whether Cerebro may use the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkMode {
    /// Nothing uses the network, not even an explicit download.
    Offline,
    /// Only actions the user starts (a model download, a plugin call they allowed); nothing in the
    /// background: no update checks, no catalog refresh, no telemetry. The default.
    #[default]
    OnRequest,
    /// Background features may also use the network (update check, catalog refresh).
    Online,
}

impl NetworkMode {
    /// Whether an action the user explicitly started may use the network.
    pub fn allows_user_initiated(self) -> bool {
        !matches!(self, NetworkMode::Offline)
    }

    /// Whether anything may use the network without a user action.
    pub fn allows_background(self) -> bool {
        matches!(self, NetworkMode::Online)
    }
}

/// Network and exposure settings. Everything Cerebro does works with no network at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkSection {
    /// See [`NetworkMode`].
    #[serde(default)]
    pub mode: NetworkMode,
    /// Whether Cerebro's API may be reached from beyond loopback. Off by default, and refused
    /// while [`NetworkMode::Offline`].
    ///
    /// Recorded intent only, for now: [`crate::schema::BindAddr`] can express loopback alone, and
    /// no listener in this workspace reads this switch yet. A future LAN listener must honour it;
    /// until then turning it on exposes nothing.
    #[serde(default)]
    pub lan_mode: bool,
}

/// What happens when a plugin or tool asks to use a capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    /// Ask every time.
    Ask,
    /// Ask once, then allow for the rest of the session.
    Session,
    /// Allow without asking.
    Always,
    /// Refuse without asking.
    Deny,
}

/// A capability a plugin or tool can request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Read files.
    ReadFiles,
    /// Create, change or delete files.
    WriteFiles,
    /// Run programs or shell commands.
    RunCommands,
    /// Use the network (also bounded by [`NetworkSection::mode`]).
    Network,
}

impl Capability {
    /// Every capability, in display order.
    pub const ALL: [Capability; 4] = [
        Capability::ReadFiles,
        Capability::WriteFiles,
        Capability::RunCommands,
        Capability::Network,
    ];

    fn key(self) -> &'static str {
        match self {
            Capability::ReadFiles => "read_files",
            Capability::WriteFiles => "write_files",
            Capability::RunCommands => "run_commands",
            Capability::Network => "network",
        }
    }
}

/// Plugin permission defaults: one fallback plus a per-capability override. A missing key takes
/// its default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PluginsSection {
    /// Used for a capability with no entry in `capabilities`. `ask` by default.
    pub default_permission: Permission,
    /// Per-capability defaults. Every capability is listed as `ask` by default, so a settings
    /// screen shows each one. A capability listed twice in the file is refused.
    #[serde(deserialize_with = "crate::load::unique_map")]
    pub capabilities: BTreeMap<Capability, Permission>,
}

impl Default for PluginsSection {
    fn default() -> Self {
        Self {
            default_permission: Permission::Ask,
            capabilities: Capability::ALL.iter().map(|c| (*c, Permission::Ask)).collect(),
        }
    }
}

impl Settings {
    /// The permission in effect for `capability`, after every setting that bears on it:
    ///
    /// 1. [`Capability::Network`] is [`Permission::Deny`] while [`NetworkMode::Offline`].
    /// 2. Otherwise the capability's own entry, or [`PluginsSection::default_permission`].
    /// 3. With [`ProtectionsSection::permission_prompts`] off, `ask` and `session` become
    ///    [`Permission::Always`]. An explicit `deny` stays: it is the user's choice, not a
    ///    protection.
    pub fn permission_for(&self, capability: Capability) -> Permission {
        if capability == Capability::Network && !self.network.mode.allows_user_initiated() {
            return Permission::Deny;
        }
        let set = self
            .plugins
            .capabilities
            .get(&capability)
            .copied()
            .unwrap_or(self.plugins.default_permission);
        match set {
            Permission::Ask | Permission::Session if !self.protections.permission_prompts => {
                Permission::Always
            }
            other => other,
        }
    }

    /// Every protection currently off, one entry each, for a settings screen's single neutral note.
    /// Empty when everything is on. Launch protections carry the supervisor's own key, name and
    /// risk wording, so the settings screen and the launch record say the same thing.
    pub fn disabled_protections(&self) -> Vec<DisabledProtection> {
        let launch = self.protections.launch();
        let mut out: Vec<DisabledProtection> = [
            launch.integrity_warning(),
            launch.folder_permission_warning(),
            launch.environment_allowlist_warning(),
            launch.job_containment_warning(),
            launch.elevation_warning(),
            launch.redirect_warning(),
        ]
        .into_iter()
        .flatten()
        .map(|w| DisabledProtection {
            key: w.protection().key(),
            name: w.protection().to_string(),
            risk: w.risk().to_string(),
        })
        .collect();
        if !self.protections.sandbox_plugins {
            out.push(DisabledProtection {
                key: "plugin_sandbox",
                name: "plugin sandbox".to_string(),
                risk: "plugins and tools run with this account's full access to files, programs \
                       and the network their permissions allow"
                    .to_string(),
            });
        }
        if !self.protections.permission_prompts {
            out.push(DisabledProtection {
                key: "permission_prompts",
                name: "permission prompts".to_string(),
                risk: "capabilities set to ask or session are used without asking; only an \
                       explicit deny still stops them"
                    .to_string(),
            });
        }
        out
    }
}

/// JSON Schema for [`ProtectionsSection`].
pub(crate) fn protections_schema() -> serde_json::Value {
    let flag = serde_json::json!({ "type": "boolean", "default": true });
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "description": "Every protection defaults on; a missing key means on.",
        "properties": {
            "verify_backend_integrity": flag,
            "audit_folder_permissions": flag,
            "enforce_environment_allowlist": flag,
            "contain_in_job_object": flag,
            "refuse_elevation": flag,
            "refuse_redirected_paths": flag,
            "sandbox_plugins": flag,
            "permission_prompts": flag,
        }
    })
}

/// JSON Schema for [`NetworkSection`].
pub(crate) fn network_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "mode": { "type": "string", "enum": ["offline", "on_request", "online"], "default": "on_request" },
            "lan_mode": {
                "type": "boolean",
                "default": false,
                "description": "Refused while mode is offline. No listener honours it yet; the API binds loopback only.",
            },
        },
        // `Settings::validate`'s offline/LAN rule, so an external validator refuses the same file.
        "if": { "properties": { "mode": { "const": "offline" } }, "required": ["mode"] },
        "then": { "properties": { "lan_mode": { "const": false } } },
    })
}

/// JSON Schema for [`PluginsSection`].
pub(crate) fn plugins_schema() -> serde_json::Value {
    let permission = serde_json::json!({ "type": "string", "enum": ["ask", "session", "always", "deny"] });
    let capabilities: serde_json::Map<String, serde_json::Value> = Capability::ALL
        .iter()
        .map(|c| (c.key().to_string(), permission.clone()))
        .collect();
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "default_permission": permission,
            "capabilities": {
                "type": "object",
                "additionalProperties": false,
                "properties": capabilities,
            },
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_protection_defaults_on_and_a_missing_key_means_on() {
        let d = ProtectionsSection::default();
        assert_eq!(d.launch(), cerebro_supervisor::Protections::default());
        assert!(d.sandbox_plugins && d.permission_prompts);
        let parsed: ProtectionsSection = serde_json::from_value(serde_json::json!({})).expect("empty");
        assert_eq!(parsed, d);
        let one_off: ProtectionsSection =
            serde_json::from_value(serde_json::json!({"refuse_elevation": false})).expect("one key");
        assert!(!one_off.refuse_elevation);
        assert!(one_off.verify_backend_integrity && one_off.permission_prompts);
    }

    #[test]
    fn launch_maps_each_switch_to_its_own_supervisor_field() {
        // Destructured without `..`: a seventh supervisor switch fails to compile here until the
        // settings grow a matching key, so the two cannot silently drift apart.
        let cerebro_supervisor::Protections {
            verify_backend_integrity,
            audit_folder_permissions,
            enforce_environment_allowlist,
            contain_in_job_object,
            refuse_elevation,
            refuse_redirected_paths,
        } = cerebro_supervisor::Protections::default();
        assert!(
            verify_backend_integrity
                && audit_folder_permissions
                && enforce_environment_allowlist
                && contain_in_job_object
                && refuse_elevation
                && refuse_redirected_paths
        );

        type Setter = fn(&mut ProtectionsSection);
        type Getter = fn(&cerebro_supervisor::Protections) -> bool;
        let cases: [(Setter, Getter); 6] = [
            (|p| p.verify_backend_integrity = false, |l| l.verify_backend_integrity),
            (|p| p.audit_folder_permissions = false, |l| l.audit_folder_permissions),
            (|p| p.enforce_environment_allowlist = false, |l| l.enforce_environment_allowlist),
            (|p| p.contain_in_job_object = false, |l| l.contain_in_job_object),
            (|p| p.refuse_elevation = false, |l| l.refuse_elevation),
            (|p| p.refuse_redirected_paths = false, |l| l.refuse_redirected_paths),
        ];
        for (i, (set, get)) in cases.iter().enumerate() {
            let mut p = ProtectionsSection::default();
            set(&mut p);
            let launch = p.launch();
            assert!(!get(&launch), "case {i}: the switch did not reach the supervisor");
            // Exactly one supervisor switch moved.
            let moved = [
                launch.verify_backend_integrity,
                launch.audit_folder_permissions,
                launch.enforce_environment_allowlist,
                launch.contain_in_job_object,
                launch.refuse_elevation,
                launch.refuse_redirected_paths,
            ]
            .iter()
            .filter(|on| !**on)
            .count();
            assert_eq!(moved, 1, "case {i}");
        }
    }

    #[test]
    fn disabled_protections_lists_exactly_what_is_off_with_supervisor_wording() {
        let mut s = Settings::default();
        assert!(s.disabled_protections().is_empty());

        s.protections.audit_folder_permissions = false;
        s.protections.permission_prompts = false;
        let off = s.disabled_protections();
        let keys: Vec<&str> = off.iter().map(|d| d.key).collect();
        assert_eq!(keys, ["folder_permission_audit", "permission_prompts"]);
        let expected_risk = s
            .protections
            .launch()
            .folder_permission_warning()
            .map(|w| w.risk().to_string());
        assert_eq!(Some(off[0].risk.clone()), expected_risk);
        assert!(off.iter().all(|d| !d.risk.is_empty() && !d.name.is_empty()));
    }

    #[test]
    fn permission_resolution_honours_network_mode_prompts_and_explicit_deny() {
        let mut s = Settings::default();
        for c in Capability::ALL {
            assert_eq!(s.permission_for(c), Permission::Ask, "{c:?}");
        }

        s.network.mode = NetworkMode::Offline;
        s.plugins.capabilities.insert(Capability::Network, Permission::Always);
        assert_eq!(s.permission_for(Capability::Network), Permission::Deny);
        s.network.mode = NetworkMode::OnRequest;
        assert_eq!(s.permission_for(Capability::Network), Permission::Always);

        s.protections.permission_prompts = false;
        s.plugins.capabilities.insert(Capability::WriteFiles, Permission::Session);
        s.plugins.capabilities.insert(Capability::RunCommands, Permission::Deny);
        assert_eq!(s.permission_for(Capability::ReadFiles), Permission::Always);
        assert_eq!(s.permission_for(Capability::WriteFiles), Permission::Always);
        assert_eq!(s.permission_for(Capability::RunCommands), Permission::Deny);

        s.plugins.capabilities.clear();
        s.plugins.default_permission = Permission::Deny;
        assert_eq!(s.permission_for(Capability::ReadFiles), Permission::Deny);
    }

    #[test]
    fn network_modes_gate_user_and_background_use() {
        assert!(!NetworkMode::Offline.allows_user_initiated());
        assert!(!NetworkMode::Offline.allows_background());
        assert!(NetworkMode::OnRequest.allows_user_initiated());
        assert!(!NetworkMode::OnRequest.allows_background());
        assert!(NetworkMode::Online.allows_background());
        assert_eq!(NetworkMode::default(), NetworkMode::OnRequest);
    }

    #[test]
    fn an_unknown_capability_or_permission_is_refused() {
        assert!(serde_json::from_value::<PluginsSection>(serde_json::json!({
            "default_permission": "ask", "capabilities": {"root": "always"}
        }))
        .is_err());
        assert!(serde_json::from_value::<PluginsSection>(serde_json::json!({
            "default_permission": "maybe"
        }))
        .is_err());
    }
}
