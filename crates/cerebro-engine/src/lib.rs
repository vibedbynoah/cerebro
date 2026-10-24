//! Cerebro Engine: Cerebro's own inference engine (docs/specs/engine.md). No inference library is
//! used: GGUF parsing, tokenisation, the model graph, quantised kernels, sampling and the
//! conversation cache are all implemented here.

/// The engine's version and the commit it was built from ("+changes" when its source differed).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const COMMIT: &str = env!("CEREBRO_ENGINE_COMMIT");

pub mod audio;
pub mod chat;
pub mod cluster;
pub mod diffusion;
pub mod experts;
pub mod gguf;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod gtensor;
pub mod hc;
pub mod hw;
mod iq_tables;
pub mod model;
pub mod placement;
#[cfg(feature = "portable")]
pub mod portable;
pub mod pool;
pub mod quant;
mod quant_more;
pub mod sample;
pub mod stream;
pub mod switches;
pub mod telemetry;
pub mod tokenizer;
pub mod topic;
pub mod tune;
pub mod tts;
pub mod vision;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("model file: {0}")]
    Format(String),
    #[error("not supported yet: {0}")]
    Unsupported(String),
    #[error("request: {0}")]
    Request(String),
}

/// Formats a chat for Qwen-family models (ChatML). `think: false` adds the empty think block that
/// tells Qwen3 hybrid-reasoning models to answer directly; `think: true` opens the think block, as
/// the models' own templates do (left open, Qwen3.5 measured closing it at once: 1 of 8 right).
pub fn chatml(system: Option<&str>, turns: &[(&str, &str)], think: bool) -> String {
    let mut s = String::new();
    if let Some(sys) = system {
        s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
    }
    for (role, text) in turns {
        s.push_str(&format!("<|im_start|>{role}\n{text}<|im_end|>\n"));
    }
    s.push_str("<|im_start|>assistant\n");
    s.push_str(if think { "<think>\n" } else { "<think>\n\n</think>\n\n" });
    s
}
