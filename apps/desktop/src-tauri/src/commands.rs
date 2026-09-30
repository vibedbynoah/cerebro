//! The app's own IPC commands. Each is granted to the main window only (capabilities/main-window.json)
//! and allowlisted in the isolation hook. None accepts an argument from the webview: each acts on
//! the window that called it, which Tauri resolves on the Rust side, so there is nothing to forge
//! or validate beyond the caller's identity.

use serde::Serialize;
use tauri::{Runtime, WebviewWindow};

use crate::MAIN_WINDOW;

/// What the UI may learn about the app. Mirrors `AppInfo` in ui/src/service/types.ts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AppInfo {
    /// Product name from tauri.conf.json.
    pub name: String,
    /// App version from tauri.conf.json.
    pub version: String,
    /// Always "desktop" here; the browser preview reports "web".
    pub mode: &'static str,
}

/// Errors cross the IPC boundary as one fixed sentence: no paths, OS messages or internals.
fn fail<E: std::fmt::Display>(action: &'static str) -> impl FnOnce(E) -> String {
    move |error| {
        eprintln!("window command could not {action}: {error}");
        format!("The window could not {action}.")
    }
}

/// Defence in depth behind the capability file: only the main window may drive window state.
fn require_main<R: Runtime>(window: &WebviewWindow<R>) -> Result<(), String> {
    if window.label() == MAIN_WINDOW {
        Ok(())
    } else {
        Err("This window may not use window controls.".to_owned())
    }
}

/// Minimizes the calling window.
#[tauri::command]
pub fn window_minimize<R: Runtime>(window: WebviewWindow<R>) -> Result<(), String> {
    require_main(&window)?;
    window.minimize().map_err(fail("minimize"))
}

/// Maximizes the calling window, or restores it if it is maximized.
#[tauri::command]
pub fn window_toggle_maximize<R: Runtime>(window: WebviewWindow<R>) -> Result<(), String> {
    require_main(&window)?;
    if window.is_maximized().map_err(fail("read its size"))? {
        window.unmaximize().map_err(fail("restore"))
    } else {
        window.maximize().map_err(fail("maximize"))
    }
}

/// Whether the calling window is maximized, so the title bar can show Maximize or Restore.
#[tauri::command]
pub fn window_is_maximized<R: Runtime>(window: WebviewWindow<R>) -> Result<bool, String> {
    require_main(&window)?;
    window.is_maximized().map_err(fail("read its size"))
}

/// Closes the calling window (the app exits when its last window closes).
#[tauri::command]
pub fn window_close<R: Runtime>(window: WebviewWindow<R>) -> Result<(), String> {
    require_main(&window)?;
    window.close().map_err(fail("close"))
}

/// Name and version of the running app.
#[tauri::command]
pub fn app_info<R: Runtime>(app: tauri::AppHandle<R>) -> AppInfo {
    let info = app.package_info();
    AppInfo { name: info.name.clone(), version: info.version.to_string(), mode: "desktop" }
}
