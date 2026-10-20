//! Cerebro desktop entry point. Everything lives in the library so it can be unit-tested.

// No console window behind the app in release builds on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // A release build has no console, so a failure to start would otherwise look like nothing
    // happening: every startup error and panic is written to a log and shown in a message box.
    std::panic::set_hook(Box::new(|info| {
        let msg = format!("Cerebro stopped unexpectedly.\n\n{info}");
        report(&msg);
    }));
    if let Err(error) = cerebro_desktop::run() {
        report(&format!("Cerebro could not start.\n\n{error}"));
        std::process::exit(1);
    }
}

/// Writes `msg` to %LOCALAPPDATA%\Cerebro\startup-error.log and shows it.
fn report(msg: &str) {
    eprintln!("{msg}");
    let log = std::env::var_os("LOCALAPPDATA").map(|d| std::path::PathBuf::from(d).join("Cerebro"));
    let mut shown = msg.to_owned();
    if let Some(dir) = log {
        let file = dir.join("startup-error.log");
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        if std::fs::create_dir_all(&dir).is_ok() && std::fs::write(&file, format!("{stamp}\n{msg}\n")).is_ok() {
            shown.push_str(&format!("\n\nThis message was saved to {}.", file.display()));
        }
    }
    alert(&shown);
}

#[cfg(windows)]
fn alert(msg: &str) {
    #[link(name = "user32")]
    extern "system" {
        fn MessageBoxW(hwnd: *mut core::ffi::c_void, text: *const u16, caption: *const u16, kind: u32) -> i32;
    }
    let wide = |s: &str| s.encode_utf16().chain(Some(0)).collect::<Vec<u16>>();
    let (text, caption) = (wide(msg), wide("Cerebro"));
    // MB_OK | MB_ICONERROR | MB_SETFOREGROUND
    // SAFETY: both strings are NUL-terminated UTF-16 that outlive the call.
    unsafe { MessageBoxW(std::ptr::null_mut(), text.as_ptr(), caption.as_ptr(), 0x10 | 0x0001_0000) };
}

#[cfg(not(windows))]
fn alert(_msg: &str) {}
