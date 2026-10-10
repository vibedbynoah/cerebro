//! Where the machine-dependent live tests find their fixtures, in one place for every crate.
//!
//! The live tests (`#[ignore]`d, run explicitly) need an official llama.cpp build and a tiny GGUF.
//! Their locations differ between machines and change over time on one machine, so each is read
//! from an environment variable, falling back to the development machine's path. A missing fixture
//! is reported with the variable that points elsewhere; tests never recreate it.
//!
//! Test support only: compiled for this crate's own tests and, through the `test-support` feature
//! (enabled only by dev-dependencies), for other crates' tests. Never part of a release build.

use std::path::PathBuf;

/// Names the directory of an official llama.cpp build (`llama-server.exe`, `llama-bench.exe`).
pub const OFFICIAL_BACKEND_ENV: &str = "CEREBRO_LLAMACPP_OFFICIAL";

/// Names the tiny GGUF model the live tests may load.
pub const TINY_GGUF_ENV: &str = "CEREBRO_TINY_GGUF";

/// The development machine's official build, used when [`OFFICIAL_BACKEND_ENV`] is unset.
const DEFAULT_OFFICIAL_BACKEND: &str = "C:/bench/llamacpp-official";

/// The development machine's tiny model, used when [`TINY_GGUF_ENV`] is unset.
const DEFAULT_TINY_GGUF: &str =
    "C:/Users/noahm/.lmstudio/models/skiesoffire/bastion-control/bastion-control-0.5b-f16.gguf";

/// The official llama.cpp build directory: [`OFFICIAL_BACKEND_ENV`] if set, else the default.
/// `Err` names the directory and the variable when it holds no `llama-server.exe`.
pub fn official_backend_dir() -> Result<PathBuf, String> {
    let dir = from_env_or(OFFICIAL_BACKEND_ENV, DEFAULT_OFFICIAL_BACKEND);
    if dir.join("llama-server.exe").is_file() {
        Ok(dir)
    } else {
        Err(format!(
            "official llama.cpp build not found at {} (no llama-server.exe); set {} to the \
             directory of an official build",
            dir.display(),
            OFFICIAL_BACKEND_ENV
        ))
    }
}

/// The tiny GGUF fixture: [`TINY_GGUF_ENV`] if set, else the default. `Err` names the path and the
/// variable when it is not a file.
pub fn tiny_gguf() -> Result<PathBuf, String> {
    let path = from_env_or(TINY_GGUF_ENV, DEFAULT_TINY_GGUF);
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "tiny GGUF fixture not found at {}; set {} to a small .gguf model",
            path.display(),
            TINY_GGUF_ENV
        ))
    }
}

/// `name`'s value as a path when it is set and non-empty, else `default`.
fn from_env_or(name: &str, default: &str) -> PathBuf {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map_or_else(|| PathBuf::from(default), PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_fixture_names_the_variable_that_points_elsewhere() {
        // Not the env override itself (mutating this process's environment would race other
        // tests): the default resolution and the message a missing fixture produces.
        let dir = from_env_or("CEREBRO_FIXTURE_TEST_VAR_NEVER_SET_7c1", "C:/nowhere/x");
        assert_eq!(dir, PathBuf::from("C:/nowhere/x"));
        if let Err(message) = official_backend_dir() {
            assert!(message.contains(OFFICIAL_BACKEND_ENV), "{message}");
        }
        if let Err(message) = tiny_gguf() {
            assert!(message.contains(TINY_GGUF_ENV), "{message}");
        }
    }
}
