//! Cerebro Engine: Cerebro's own inference engine (docs/specs/engine.md). No inference library is
//! used: GGUF parsing, tokenisation, the model graph, quantised kernels, sampling and the
//! conversation cache are all implemented here.

pub mod chat;
pub mod gguf;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod hw;
pub mod model;
pub mod pool;
pub mod quant;
pub mod sample;
pub mod tokenizer;

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
/// tells Qwen3 hybrid-reasoning models to answer directly.
pub fn chatml(system: Option<&str>, turns: &[(&str, &str)], think: bool) -> String {
    let mut s = String::new();
    if let Some(sys) = system {
        s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
    }
    for (role, text) in turns {
        s.push_str(&format!("<|im_start|>{role}\n{text}<|im_end|>\n"));
    }
    s.push_str("<|im_start|>assistant\n");
    if !think {
        s.push_str("<think>\n\n</think>\n\n");
    }
    s
}
