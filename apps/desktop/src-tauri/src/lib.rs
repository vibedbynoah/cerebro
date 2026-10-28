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
mod tunnel;
mod mcp_server;
mod local_api;
pub mod llama_backend;
mod toolcall;
mod orchestrate;
mod commands;
mod builtin_tools;
mod agent_tools;
mod trainer;
mod engine;
mod bench;
mod dedup;
mod gguf_edit;
mod tune_screen;
mod stt;
mod tts;
mod images;
mod cloud;
mod cluster;
mod mcp_registry;
mod vision_input;
mod hub;
mod kv_store;
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
    let Some(instance) = instance::claim() else {
        // Cerebro is already open: that window was brought forward.
        return Ok(());
    };
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
            commands::placement_preview,
            commands::disk_scan,
            commands::disk_deduplicate,
            commands::disk_unlink,
            commands::model_reveal,
            commands::tune_start,
            commands::mic_start,
            commands::tts_sample,
            commands::image_generate,
            commands::cloud_presets,
            commands::cluster_share_get,
            commands::mcp_registry_search,
            commands::cluster_share_start,
            commands::cluster_share_stop,
            commands::cluster_discover,
            commands::cluster_load,
            commands::cloud_providers,
            commands::cloud_provider_save,
            commands::cloud_provider_remove,
            commands::cloud_models_list,
            commands::image_reveal,
            commands::tts_read,
            commands::tts_stop,
            commands::mic_level,
            commands::stt_transcribe,
            commands::stt_transcribe_file,
            commands::model_describe_image,
            commands::tune_subscribe,
            commands::tune_cancel,
            commands::tune_resume,
            commands::model_detail,
            commands::gguf_copy_write,
            commands::gguf_copy_revert,
            commands::chat_send,
            commands::chat_stop,
            commands::kv_store_usage,
            commands::kv_store_clear,
            commands::live_stats,
            commands::metrics,
            commands::metrics_history,
            commands::request_detail,
            commands::hub_search,
            commands::hub_files,
            commands::hub_avatar,
            commands::image_edit,
            commands::train_dataset_save,
            commands::hub_latest,
            commands::remote_tunnel_start,
            commands::remote_tunnel_stop,
            commands::remote_tunnel_get,
            commands::image_video,
            commands::moe_activity,
            commands::context_check,
            commands::hub_pipeline_files,
            commands::hub_file_meta,
            commands::hub_download,
            commands::hub_cancel,
            commands::switches_list,
            commands::switch_set,
            commands::network_offline_set,
            commands::switches_reset,
            commands::tune_machine_start,
            commands::tune_machine_last,
            commands::bench_run,
            commands::bench_history,
            commands::bench_stop,
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
            commands::chat_tool_answer,
            commands::builtin_tools,
            commands::builtin_folders_get,
            commands::open_external,
            commands::documents_dir,
            commands::builtin_folders_set,
            commands::mcp_server_get,
            commands::mcp_server_enable,
            commands::mcp_server_rotate_token,
            commands::local_api_get,
            commands::local_api_enable,
            commands::local_api_options,
            commands::local_api_rotate_key,
            trainer::trainer_status,
            trainer::trainer_setup,
            trainer::trainer_start,
            trainer::trainer_cancel,
        ])
        .manage(engine::Engine::start())
        .manage(hub::Downloads::default())
        .manage(mcp::Mcp::default())
        .manage(commands::Telemetry::default())
        .manage(trainer::Trainer::default())
        // Disk space (dedup.rs): the last scan's groups, so the calls that follow re-check the
        // copies the user was shown.
        .manage(std::sync::Arc::new(dedup::Disk::default()))
        // The Tune screen's run (tune_screen.rs): its event log, listeners and finished steps.
        .manage(std::sync::Arc::new(tune_screen::TuneHub::default()))
        // Speech to text (stt.rs): the microphone and the speech model, beside the chat model.
        .manage(std::sync::Arc::new(stt::Mic::default()))
        .manage(std::sync::Arc::new(stt::Stt::default()))
        // Text to speech (tts.rs): the speech model and the speaker.
        .manage(std::sync::Arc::new(tts::Tts::default()))
        .manage(std::sync::Arc::new(tts::Player::default()))
        // Image Studio (images.rs).
        .manage(std::sync::Arc::new(images::Images::default()))
        .setup(|app| {
            commands::load_switches(app.handle());
            {
                use tauri::Manager;
                mcp::mcp_start_all(app.handle().clone(), app.state::<mcp::Mcp>());
                // Built-in tools (builtin_tools.rs) run only inside the folders the user added on the
                // Tools screen; the composite host lists them under "cerebro" beside the MCP tools.
                let workspace = std::sync::Arc::new(builtin_tools::WorkspaceFolders::load(app.handle()));
                app.manage(std::sync::Arc::clone(&workspace));
                // Long tool results are kept whole under <app data>\spills (a week at most).
                let mut tools = builtin_tools::BuiltinTools::new(std::sync::Arc::new(commands::McpToolHost(app.handle().clone())), std::sync::Arc::clone(&workspace));
                if let Ok(dir) = app.path().app_data_dir() {
                    let spills = dir.join("spills");
                    builtin_tools::prune_spills(&spills, std::time::Duration::from_secs(7 * 24 * 3600));
                    tools = tools.with_spill_dir(spills);
                }
                app.state::<engine::Engine>().set_tool_host(std::sync::Arc::new(tools));
                if let Ok(dir) = app.path().app_config_dir() {
                    app.state::<std::sync::Arc<tune_screen::TuneHub>>().set_results_path(dir.join("tune-results.json"));
                    cloud::set_file(dir.join("cloud.json"));
                }
                if let Ok(dir) = app.path().app_data_dir() {
                    app.state::<std::sync::Arc<images::Images>>().set_out_dir(dir.join("images"));
                }
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
                app.manage(std::sync::Arc::new(tunnel::Tunnel::default()));
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
                // The local OpenAI/Anthropic-compatible API (off by default).
                let path = commands::local_api_file(app.handle());
                let settings = path.as_ref().and_then(|p| std::fs::read(p).ok()).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
                let save = move |s: &local_api::Settings| {
                    let Some(p) = &path else { return };
                    let _ = std::fs::create_dir_all(p.parent().unwrap_or(p));
                    let tmp = p.with_extension("json.tmp");
                    if std::fs::write(&tmp, serde_json::to_vec_pretty(s).unwrap_or_default()).is_ok() {
                        let _ = std::fs::rename(&tmp, p);
                    }
                };
                let api = local_api::LocalApi::new(settings, std::sync::Arc::new(commands::ApiEngineBackend(app.state::<engine::Engine>().handle())), Box::new(save));
                app.manage(std::sync::Arc::clone(&api));
                api.resume();
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
            if let Some(listener) = instance {
                instance::serve(listener, app.handle().clone());
            }
            // A window that was just closed can hold WebView2's profile for a moment ("The requested
            // resource is in use"): wait for it rather than fail to start.
            for attempt in 0.. {
                match main_window(app, &config) {
                    Err(e) if attempt < 20 && e.to_string().contains("0x800700AA") => std::thread::sleep(std::time::Duration::from_millis(500)),
                    r => {
                        r?;
                        break;
                    }
                }
            }
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
                if let Some(t) = app.try_state::<std::sync::Arc<tunnel::Tunnel>>() {
                    t.stop();
                }
                if let Some(srv) = app.try_state::<std::sync::Arc<mcp_server::McpServer>>() {
                    srv.stop();
                }
            }
        });
    Ok(())
}

/// The main window, with the navigation and new-window guards attached before it loads anything.
fn main_window(app: &tauri::App, config: &tauri::utils::config::WindowConfig) -> Result<(), tauri::Error> {
    let builder = WebviewWindowBuilder::from_config(app.handle(), config)?;
    // Test builds only (`--features e2e`, never in a release bundle): WebView2 remote
    // debugging so scripts/e2e-real.mjs can drive the real app, on CEREBRO_E2E_PORT (default
    // 9333) so a test run can pick a port no leftover WebView holds.
    #[cfg(feature = "e2e")]
    let e2e_args = format!(
        "--remote-debugging-port={} --disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection",
        std::env::var("CEREBRO_E2E_PORT").ok().and_then(|p| p.parse::<u16>().ok()).unwrap_or(9333)
    );
    #[cfg(feature = "e2e")]
    let builder = builder.additional_browser_args(&e2e_args);
    builder
        .on_navigation(navigation::is_app_url)
        .on_new_window(|_url, _features| NewWindowResponse::Deny)
        .build()?;
    Ok(())
}

/// One Cerebro at a time. The first instance listens on a fixed loopback port; a second launch
/// connects, asks it to show its window and exits, instead of starting a second engine and
/// colliding with the first window's WebView2 profile. Test builds skip this so a test can run
/// next to the user's app. Nothing here is reachable from another computer.
mod instance {
    use std::io::{BufRead, BufReader, Write};
    use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
    use std::time::Duration;
    use tauri::Manager;

    const PORT: u16 = 47_613;
    const HELLO: &str = "cerebro-show";
    const ANSWER: &str = "cerebro-shown";

    /// Some(listener) for the first instance (None inside it when the port belongs to another
    /// program: Cerebro then runs without the check); None when Cerebro is already open.
    pub fn claim() -> Option<Option<TcpListener>> {
        if cfg!(feature = "e2e") {
            return Some(None);
        }
        let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, PORT));
        if let Ok(l) = TcpListener::bind(addr) {
            return Some(Some(l));
        }
        let asked = TcpStream::connect_timeout(&addr, Duration::from_millis(500)).and_then(|mut s| {
            s.set_read_timeout(Some(Duration::from_secs(2)))?;
            writeln!(s, "{HELLO}")?;
            let mut line = String::new();
            BufReader::new(s).read_line(&mut line)?;
            Ok(line.trim() == ANSWER)
        });
        if asked.unwrap_or(false) {
            None
        } else {
            Some(None)
        }
    }

    /// Answers later launches by bringing the main window forward.
    pub fn serve(listener: TcpListener, app: tauri::AppHandle) {
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                let mut line = String::new();
                if BufReader::new(&stream).read_line(&mut line).is_err() || line.trim() != HELLO {
                    continue;
                }
                let _ = (&stream).write_all(format!("{ANSWER}
").as_bytes());
                if let Some(w) = app.get_webview_window(super::MAIN_WINDOW) {
                    let _ = w.unminimize();
                    let _ = w.show();
                    let _ = w.set_focus();
                }
            }
        });
    }
}
