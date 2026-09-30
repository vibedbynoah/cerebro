//! Generates the Tauri context and the ACL for the app's own commands.
//!
//! Declaring the commands in an app manifest makes them deny-by-default: each gets an
//! `allow-<command>` permission, and a window may call it only if one of its capabilities
//! (capabilities/*.json) grants that permission. Without the manifest, every registered command
//! would be callable from every window.

/// Must match `tauri::generate_handler!` in src/lib.rs, the grants in capabilities/main-window.json
/// and the isolation allowlist in ../isolation/index.js (tests/unit/tauri-config.test.ts checks all four).
const COMMANDS: &[&str] = &["window_minimize", "window_toggle_maximize", "window_is_maximized", "window_close", "app_info"];

fn main() {
    let attributes = tauri_build::Attributes::new().app_manifest(tauri_build::AppManifest::new().commands(COMMANDS));
    if let Err(error) = tauri_build::try_build(attributes) {
        eprintln!("cerebro-desktop build script failed: {error:#}");
        std::process::exit(1);
    }
}
