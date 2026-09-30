//! Cerebro desktop entry point. Everything lives in the library so it can be unit-tested.

// No console window behind the app in release builds on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    if let Err(error) = cerebro_desktop::run() {
        eprintln!("Cerebro could not start: {error}");
        std::process::exit(1);
    }
}
