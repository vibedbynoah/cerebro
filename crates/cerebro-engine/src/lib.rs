//! Cerebro Engine: Cerebro's own inference engine (docs/specs/engine.md). No inference library is
//! used: GGUF parsing, tokenisation, the model graph, quantised kernels, sampling and the
//! conversation cache are all implemented here.

/// The engine's version and the commit it was built from ("+changes" when its source differed).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const COMMIT: &str = env!("CEREBRO_ENGINE_COMMIT");

pub mod activity;
pub mod audio;
pub mod chat;
pub mod cluster;
pub mod diffusion;
pub mod experts;
pub mod gemma4;
pub mod gguf;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod gpu_g4;
#[cfg(feature = "cuda")]
pub mod gpu_probe;
#[cfg(feature = "cuda")]
pub mod gtensor;
pub mod hc;
pub mod hw;
pub mod longctx;
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
    /// The host asked the work to stop (`STOP`); what finished before it is kept.
    #[error("stopped")]
    Stopped,
}

/// Set by the host to stop a model load or a prompt read at the next layer or chunk boundary
/// (the stop button, Unload, Cancel load). The host clears it before it starts new work.
pub static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the host asked the current work to stop.
pub fn stop_requested() -> bool {
    STOP.load(std::sync::atomic::Ordering::Relaxed)
}

/// `Err(Error::Stopped)` once the host asked the current work to stop.
pub fn check_stop() -> Result<(), Error> {
    if stop_requested() { Err(Error::Stopped) } else { Ok(()) }
}

/// The chat format a model's own template uses, so prompts read the way the model was trained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatFormat {
    /// Qwen and most others: <|im_start|>role ... <|im_end|>, with Qwen3's think block.
    ChatMl,
    /// Llama 3: <|start_header_id|>role<|end_header_id|> ... <|eot_id|>.
    Llama3,
    /// Gemma 4: <|turn>role ... <turn|>, the model's turns called "model"; <|think|> in the system
    /// turn turns its thinking on.
    Gemma4,
}

impl ChatFormat {
    /// From the file's `tokenizer.chat_template`.
    pub fn of(g: &gguf::Gguf) -> Self {
        let t = g.str("tokenizer.chat_template").unwrap_or("");
        if t.contains("<|turn>") {
            Self::Gemma4
        } else if t.contains("<|start_header_id|>") {
            Self::Llama3
        } else {
            Self::ChatMl
        }
    }

    /// Whether replies carry Qwen-style <think> blocks (the other formats are rendered without
    /// thinking for now).
    pub fn think_tags(self) -> bool {
        self == Self::ChatMl
    }

    /// The prompt for `turns` ("user" / "assistant"), ending where the model writes its reply.
    pub fn render(self, system: Option<&str>, turns: &[(&str, &str)], think: bool) -> String {
        match self {
            Self::ChatMl => chatml(system, turns, think),
            Self::Llama3 => {
                let mut s = String::from("<|begin_of_text|>");
                let mut push = |role: &str, text: &str| s.push_str(&format!("<|start_header_id|>{role}<|end_header_id|>\n\n{}<|eot_id|>", text.trim()));
                if let Some(sys) = system {
                    push("system", sys);
                }
                for (role, text) in turns {
                    push(role, text);
                }
                s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                s
            }
            Self::Gemma4 => {
                // <bos> comes from the tokenizer.
                let mut s = String::new();
                if let Some(sys) = system {
                    s.push_str(&format!("<|turn>system\n{}<turn|>\n", sys.trim()));
                }
                for (role, text) in turns {
                    let role = if *role == "assistant" { "model" } else { role };
                    s.push_str(&format!("<|turn>{role}\n{}<turn|>\n", text.trim()));
                }
                s.push_str("<|turn>model\n");
                s
            }
        }
    }
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
