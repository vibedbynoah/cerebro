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

mod mcp;
mod remote;
mod mcp_server;
mod toolcall;
mod orchestrate;
mod commands;
mod engine;
mod hub;
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
            commands::hw_detect,
            commands::library_list,
            commands::model_load,
            commands::model_unload,
            commands::models_loaded,
            commands::chat_send,
            commands::chat_stop,
            commands::live_stats,
            commands::metrics,
            commands::hub_search,
            commands::hub_files,
            commands::hub_file_meta,
            commands::hub_download,
            commands::hub_cancel,
            commands::switches_list,
            commands::switch_set,
            commands::switches_reset,
            mcp::mcp_config_get,
            mcp::mcp_config_save,
            mcp::mcp_status,
            mcp::mcp_restart,
            mcp::mcp_tools,
            mcp::mcp_call,
            commands::remote_get,
            commands::remote_set_password,
            commands::remote_enable,
            commands::remote_options,
            commands::remote_pair,
            commands::remote_revoke,
            commands::chat_tool_decide,
            commands::mcp_server_get,
            commands::mcp_server_enable,
            commands::mcp_server_rotate_token,
        ])
        .manage(engine::Engine::start())
        .manage(hub::Downloads::default())
        .manage(mcp::Mcp::default())
        .manage(commands::Telemetry::default())
        .setup(|app| {
            commands::load_switches(app.handle());
            {
                use tauri::Manager;
                mcp::mcp_start_all(app.handle().clone(), app.state::<mcp::Mcp>());
                app.state::<engine::Engine>().set_tool_host(std::sync::Arc::new(commands::McpToolHost(app.handle().clone())));
                if let Ok(dir) = app.path().app_data_dir() {
                    app.state::<engine::Engine>().set_ledger_root(dir);
                }
                // Phone access: saved settings, written atomically (temp file + rename) on change.
                let path = commands::remote_file(app.handle());
                let settings = path.as_ref().and_then(|p| std::fs::read(p).ok()).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
                let save = move |s: &remote::Settings| {
                    let Some(p) = &path else { return };
                    let _ = std::fs::create_dir_all(p.parent().unwrap_or(p));
                    let tmp = p.with_extension("json.tmp");
                    if std::fs::write(&tmp, serde_json::to_vec_pretty(s).unwrap_or_default()).is_ok() {
                        let _ = std::fs::rename(&tmp, p);
                    }
                };
                let phone = remote::Remote::new(settings, std::sync::Arc::new(commands::PhoneBackend(app.handle().clone())), Box::new(save));
                app.manage(std::sync::Arc::clone(&phone));
                phone.resume();
                // Cerebro as an MCP server for other apps on this computer (off by default).
                let path = commands::mcp_server_file(app.handle());
                let settings = path.as_ref().and_then(|p| std::fs::read(p).ok()).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
                let save = move |s: &mcp_server::Settings| {
                    let Some(p) = &path else { return };
                    let _ = std::fs::create_dir_all(p.parent().unwrap_or(p));
                    let tmp = p.with_extension("json.tmp");
                    if std::fs::write(&tmp, serde_json::to_vec_pretty(s).unwrap_or_default()).is_ok() {
                        let _ = std::fs::rename(&tmp, p);
                    }
                };
                let mcp_srv = mcp_server::McpServer::new(settings, std::sync::Arc::new(commands::McpEngineBackend(app.handle().clone())), Box::new(save));
                app.manage(std::sync::Arc::clone(&mcp_srv));
                mcp_srv.resume();
            }
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
            let builder = WebviewWindowBuilder::from_config(app.handle(), &config)?;
            // Test builds only (`--features e2e`, never in a release bundle): WebView2 remote
            // debugging so scripts/e2e-real.mjs can drive the real app.
            #[cfg(feature = "e2e")]
            let builder = builder.additional_browser_args("--remote-debugging-port=9333 --disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection");
            builder
                .on_navigation(navigation::is_app_url)
                .on_new_window(|_url, _features| NewWindowResponse::Deny)
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())?
        .run(|app, event| {
            // Phone access closes its port as the app exits, not when the process is torn down.
            if let tauri::RunEvent::Exit = event {
                use tauri::Manager;
                if let Some(phone) = app.try_state::<std::sync::Arc<remote::Remote>>() {
                    phone.shutdown();
                }
                if let Some(srv) = app.try_state::<std::sync::Arc<mcp_server::McpServer>>() {
                    srv.stop();
                }
            }
        });
    Ok(())
}
