//! Streaming generation for apps: prompt in, text pieces out through a callback, with the
//! conversation cache, the GPU greedy fast path and cancellation handled here.

use std::time::Instant;

use crate::model::{Model, Session};

/// Draft tokens for prompt-lookup speculation: the tokens that followed the most recent earlier
/// occurrence of the context's last 3 (then 2, then 1) tokens, up to `max`. Searches the last 8192
/// tokens only, so the cost stays bounded on long contexts.
pub fn lookup_draft(seq: &[u32], max: usize) -> Vec<u32> {
    lookup_draft_min(seq, max, 1)
}

/// `lookup_draft` with the shortest tail that counts as a match (single tokens repeat by chance).
pub fn lookup_draft_min(seq: &[u32], max: usize, min_n: usize) -> Vec<u32> {
    if max == 0 || seq.len() < 2 {
        return Vec::new();
    }
    let lo = seq.len().saturating_sub(8192);
    for n in (min_n.max(1)..=3usize.min(seq.len() - 1)).rev() {
        let tail = &seq[seq.len() - n..];
        // Most recent earlier match first; what followed it (up to the end, for repeating text).
        let mut i = seq.len() - n;
        while i > lo {
            i -= 1;
            if &seq[i..i + n] == tail {
                let start = i + n;
                return seq[start..(start + max).min(seq.len())].to_vec();
            }
        }
    }
    Vec::new()
}
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
    let speculating = crate::switches::get(crate::switches::SPEC_LOOKUP) > 0 && session.gpu.as_ref().is_some_and(|g| g.can_verify());
    #[cfg(not(feature = "cuda"))]
    let speculating = false;
    #[cfg(feature = "cuda")]
    if cfg.sampler.temperature <= 0.0 && !speculating {
        if let Some(g) = session.gpu.as_mut().filter(|g| !g.is_split()) {
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
    #[cfg(feature = "cuda")]
    {
        let k = (crate::switches::get(crate::switches::SPEC_LOOKUP).max(0) as usize).min(crate::gpu::SPEC_MAX - 1);
        if k > 0 && session.gpu.as_ref().is_some_and(|g| g.can_verify()) {
            let mut cur = sampler.sample(&session.logits, &session.tokens);
            let mut row = Vec::new();
            let mut drafted = 0usize;
            let mut accepted_total = 0usize;
            // Drafts cost a verify pass; they are sized by recent acceptance and paused after a
            // draft that was rejected outright, so text with nothing to copy decodes at full speed.
            let mut rate = 0.5f32;
            let mut cooldown = 0usize;
            let mut backoff = 8usize;
            // Greedy without a repetition penalty: plain steps pick on the GPU (4-byte read).
            let gpu_greedy = cfg.sampler.temperature <= 0.0 && cfg.sampler.repeat_penalty == 1.0;
            'gen: while out.len() < max_new {
                if tok.eot.contains(&cur) {
                    break;
                }
                if !push(&mut out, cur, &mut printed, &mut first_at) {
                    stopped = true;
                    break;
                }
                let room = max_new - out.len();
                let pos = session.tokens.len();
                let mut ctx = session.tokens.clone();
                ctx.push(cur);
                let want = ((rate * k as f32 * 1.5).round() as usize).clamp(1, k);
                let draft = if cooldown > 0 {
                    cooldown -= 1;
                    Vec::new()
                } else {
                    lookup_draft_min(&ctx, want.min(room).min(session.max_ctx.saturating_sub(pos + 2)), 2)
                };
                if draft.is_empty() {
                    if gpu_greedy {
                        let nxt = session.gpu.as_mut().expect("gpu").step(model, cur, pos, None)?;
                        session.tokens.push(cur);
                        cur = nxt;
                    } else {
                        session.step(model, cur);
                        cur = sampler.sample(&session.logits, &session.tokens);
                    }
                    continue;
                }
                let mut input = vec![cur];
                input.extend_from_slice(&draft);
                session.gpu.as_mut().expect("gpu").verify(model, &input, pos, &mut row)?;
                let v = session.logits.len();
                session.tokens.push(cur);
                drafted += draft.len();
                let mut next = None;
                for (i, &d) in draft.iter().enumerate() {
                    match sampler.speculate(&row[i * v..(i + 1) * v], &session.tokens, d) {
                        Ok(()) => {
                            accepted_total += 1;
                            session.tokens.push(d);
                            if tok.eot.contains(&d) {
                                break 'gen;
                            }
                            if out.len() >= max_new || !push(&mut out, d, &mut printed, &mut first_at) {
                                stopped = out.len() < max_new;
                                break 'gen;
                            }
                        }
                        Err(t) => {
                            next = Some(t);
                            break;
                        }
                    }
                }
                let i = session.tokens.len() - pos - 1;
                rate = 0.7 * rate + 0.3 * (i as f32 / draft.len() as f32);
                if i == 0 {
                    cooldown = backoff;
                    backoff = (backoff * 2).min(64);
                } else {
                    backoff = 8;
                }
                cur = next.unwrap_or_else(|| sampler.sample(&row[i * v..(i + 1) * v], &session.tokens));
                // The logits after the last accepted token, for anything that reads them next.
                session.logits.copy_from_slice(&row[i * v..(i + 1) * v]);
            }
            if std::env::var_os("CEREBRO_SPEC_STATS").is_some() && drafted > 0 {
                eprintln!("speculation: {accepted_total} of {drafted} drafted tokens accepted");
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

#[cfg(test)]
mod tests {
    use super::lookup_draft;

    #[test]
    fn drafts_follow_the_latest_match_of_the_longest_tail() {
        // ... 5 6 7 8 9 ... 5 6 7 -> drafts 8 9
        let seq = [1, 5, 6, 7, 8, 9, 2, 3, 5, 6, 7];
        assert_eq!(lookup_draft(&seq, 4), vec![8, 9, 2, 3]);
        assert_eq!(lookup_draft(&seq, 2), vec![8, 9]);
        // The most recent match wins: 4 x A then 4 y; tail 4 -> y follows the later one.
        assert_eq!(lookup_draft(&[4, 10, 4, 20, 4], 1), vec![20]);
        // No match, or nothing after it.
        assert!(lookup_draft(&[1, 2, 3], 4).is_empty());
        assert!(lookup_draft(&[7, 7], 0).is_empty());
        // A draft never runs into the tail itself.
        assert_eq!(lookup_draft(&[1, 2, 1, 2], 8), vec![1, 2]);
    }
}
