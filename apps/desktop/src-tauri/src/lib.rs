//! Cerebro desktop shell: one Tauri v2 window around the SolidJS UI in `../ui`.
//!
//! Security posture (the config half is asserted by tests/unit/tauri-config.test.ts):
//! - Isolation pattern: every IPC message passes through `../isolation`, which drops any command
//!   that is not on its allowlist before Tauri sees it.
//! - Deny-by-default ACL: the app's commands are declared in build.rs and granted only to the main
//!   window (capabilities/main-window.json); no shell, fs or `core:default` permissions exist.
//! - Strict CSP (tauri.conf.json), prototype freezing, no asset protocol, no global Tauri object.
//! - The webview may only show the bundled app: navigation anywhere else is refused
//!   ([`navigation::is_app_url`]) and `window.open` never creates a window.
//! - Devtools are compiled out of release builds (the `devtools` cargo feature is never enabled).
//! - Vetted crypto only: 256-bit secrets from the OS CSPRNG compared in constant time
//!   ([`secret`]), and Ed25519 `verify_strict` for update packages ([`update_sig`]).

#![forbid(unsafe_code)]

mod commands;
pub mod navigation;
pub mod secret;
pub mod update_sig;

use tauri::webview::NewWindowResponse;
use tauri::WebviewWindowBuilder;

/// Label of the one window; capabilities/main-window.json grants its permissions to this label.
pub const MAIN_WINDOW: &str = "main";

/// Builds and runs the app until the last window closes.
///
/// # Errors
/// Returns the Tauri error if the runtime or the main window cannot be created.
pub fn run() -> Result<(), tauri::Error> {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            commands::window_minimize,
            commands::window_toggle_maximize,
            commands::window_is_maximized,
            commands::window_close,
            commands::app_info,
        ])
        .setup(|app| {
            // Declared in tauri.conf.json with `create: false` and built here, so the navigation and
            // new-window guards are attached before the window loads anything.
            let config = app
                .config()
                .app
                .windows
                .iter()
                .find(|w| w.label == MAIN_WINDOW)
                .cloned()
                .ok_or("tauri.conf.json has no window labelled \"main\"")?;
            WebviewWindowBuilder::from_config(app.handle(), &config)?
                .on_navigation(navigation::is_app_url)
                .on_new_window(|_url, _features| NewWindowResponse::Deny)
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
}
