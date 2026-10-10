//! Streaming generation for apps: prompt in, text pieces out through a callback, with the
//! conversation cache, the GPU greedy fast path and cancellation handled here.

use std::time::Instant;

use crate::model::{Model, Session};
use crate::sample::{argmax, Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;
use crate::Error;

/// What one generation measured.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GenStats {
    pub prompt_tokens: usize,
    /// Prompt tokens served from the conversation cache.
    pub reused_tokens: usize,
    pub output_tokens: usize,
    /// Time to the first output token, ms (includes prefill).
    pub ttft_ms: f64,
    pub prefill_tps: f64,
    pub decode_tps: f64,
    pub total_ms: f64,
    /// True when the callback stopped generation before an end token or the limit.
    pub stopped: bool,
}

pub struct GenConfig {
    pub sampler: SamplerConfig,
    pub max_tokens: usize,
}

/// Feeds `prompt` (reusing the cached prefix) and generates until an end-of-turn token,
/// `max_tokens`, the context limit, or `on_text` returning false. `on_text` receives complete UTF-8
/// pieces in order.
pub fn generate(model: &Model, session: &mut Session, tok: &Tokenizer, prompt: &[u32], cfg: &GenConfig, mut on_text: impl FnMut(&str) -> bool) -> Result<GenStats, Error> {
    let t0 = Instant::now();
    let reused = session.feed(model, prompt)?;
    let prefill_tps = session.stats.prefill_tps();
    let room = session.max_ctx.saturating_sub(session.tokens.len() + 1);
    let max_new = cfg.max_tokens.min(room);
    let mut out: Vec<u32> = Vec::new();
    let mut printed = 0usize;
    let mut stopped = false;
    let mut first_at: Option<f64> = None;
    // Emits newly completed UTF-8 text; returns false to stop.
    let mut push = |out: &mut Vec<u32>, t: u32, printed: &mut usize, first_at: &mut Option<f64>| -> bool {
        out.push(t);
        first_at.get_or_insert_with(|| t0.elapsed().as_secs_f64() * 1000.0);
        let bytes = tok.decode_bytes(out);
        let mut end = bytes.len();
        // Hold back an incomplete multi-byte character until its remaining bytes arrive.
        while end > *printed && std::str::from_utf8(&bytes[*printed..end]).is_err() {
            end -= 1;
        }
        if end > *printed {
            let piece = std::str::from_utf8(&bytes[*printed..end]).unwrap_or_default().to_owned();
            *printed = end;
            return on_text(&piece);
        }
        true
    };
    let td = Instant::now();
    #[cfg(feature = "cuda")]
    if cfg.sampler.temperature <= 0.0 {
        if let Some(g) = session.gpu.as_mut() {
            let first = argmax(&session.logits);
            if max_new > 0 && !tok.eot.contains(&first) {
                if !push(&mut out, first, &mut printed, &mut first_at) {
                    stopped = true;
                } else {
                    let pos = session.tokens.len();
                    let mut emitted: Vec<u32> = Vec::new();
                    let steps = g.generate_greedy(model, first, pos, max_new - 1, &tok.eot, |t| {
                        emitted.push(t);
                        let go = push(&mut out, t, &mut printed, &mut first_at);
                        stopped |= !go;
                        go
                    })?;
                    // Step i fed `first` or the token emitted before it: those are in the KV cache now.
                    session.tokens.push(first);
                    session.tokens.extend_from_slice(&emitted[..steps.saturating_sub(1).min(emitted.len())]);
                }
            }
            let dsecs = td.elapsed().as_secs_f64();
            return Ok(GenStats {
                prompt_tokens: prompt.len(),
                reused_tokens: reused,
                output_tokens: out.len(),
                ttft_ms: first_at.unwrap_or(0.0),
                prefill_tps,
                decode_tps: if dsecs > 0.0 { out.len() as f64 / dsecs } else { 0.0 },
                total_ms: t0.elapsed().as_secs_f64() * 1000.0,
                stopped,
            });
        }
    }
    let mut sampler = Sampler::new(SamplerConfig { ..cfg.sampler });
    for _ in 0..max_new {
        let next = if cfg.sampler.temperature <= 0.0 { argmax(&session.logits) } else { sampler.sample(&session.logits, &session.tokens) };
        if tok.eot.contains(&next) {
            break;
        }
        if !push(&mut out, next, &mut printed, &mut first_at) {
            stopped = true;
            break;
        }
        session.step(model, next);
    }
    let dsecs = td.elapsed().as_secs_f64();
    Ok(GenStats {
        prompt_tokens: prompt.len(),
        reused_tokens: reused,
        output_tokens: out.len(),
        ttft_ms: first_at.unwrap_or(0.0),
        prefill_tps,
        decode_tps: if dsecs > 0.0 { out.len() as f64 / dsecs } else { 0.0 },
        total_ms: t0.elapsed().as_secs_f64() * 1000.0,
        stopped,
    })
}
