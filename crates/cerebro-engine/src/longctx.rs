//! Does the model still work at a long context? A "needle in a haystack" check: a passcode is
//! hidden at a chosen depth of a long document of unrelated filler, and the model is asked for it.
//! A model that reads its whole context answers with the passcode at every depth; one past the
//! length it handles (or with a broken context extension) misses the far ones or answers nonsense.
//!
//! The filler is varied sentences from a fixed vocabulary (deterministic per seed), so the passcode
//! is the only number-shaped fact in the document.

use crate::model::{Model, Session};
use crate::tokenizer::Tokenizer;
use crate::Error;

const SUBJECTS: [&str; 16] = ["The river", "A quiet village", "The old library", "Every spring the orchard", "The night market", "A small fishing boat", "The mountain road", "The weaver's workshop", "An autumn festival", "The lighthouse keeper", "A field of barley", "The train station", "The museum garden", "A winter storm", "The city clock", "A family of herons"];
const VERBS: [&str; 16] = ["drifted past", "welcomed", "was quiet beside", "slowly filled", "looked out over", "waited near", "wound around", "smelled of", "brightened", "rested under", "echoed through", "was painted with", "leaned toward", "carried", "surrounded", "remembered"];
const OBJECTS: [&str; 16] = ["the morning fog", "travellers from the coast", "rows of linden trees", "the sound of bells", "a long stone wall", "the evening light", "baskets of apples", "the edge of the forest", "a narrow wooden bridge", "children playing", "the harbour", "fresh bread and tea", "a field of poppies", "the distant hills", "an empty courtyard", "the first snow"];

/// One prepared check: the prompt's tokens and the passcode it hides.
pub struct Needle {
    pub tokens: Vec<u32>,
    pub passcode: String,
    /// Where the passcode sits, 0 = start of the document, 1 = end.
    pub depth: f32,
}

/// The outcome of one check.
#[derive(Clone, Debug)]
pub struct Probe {
    /// Prompt length, tokens.
    pub tokens: usize,
    pub depth: f32,
    pub found: bool,
    pub reply: String,
    pub seconds: f64,
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 33
}

/// A document of about `tokens` tokens with the passcode at `depth`, as a chat prompt asking for it.
pub fn build(tok: &Tokenizer, tokens: usize, depth: f32, seed: u64) -> Needle {
    let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
    let passcode = format!("{}", 100_000 + lcg(&mut s) % 900_000);
    let needle = format!(" The passcode for the blue vault is {passcode}. Remember it.");
    let question = "\n\nWhat is the passcode for the blue vault? Reply with the number only.";
    let overhead = tok.encode(&crate::chatml(None, &[("user", question)], false)).len() + tok.encode(&needle).len() + 8;
    let budget = tokens.saturating_sub(overhead).max(16);
    let mut sentences: Vec<String> = Vec::new();
    let mut used = 0usize;
    while used < budget {
        let st = format!(" {} {} {}.", SUBJECTS[(lcg(&mut s) % 16) as usize], VERBS[(lcg(&mut s) % 16) as usize], OBJECTS[(lcg(&mut s) % 16) as usize]);
        used += tok.encode(&st).len();
        sentences.push(st);
    }
    let at = ((sentences.len() as f32) * depth.clamp(0.0, 1.0)).round() as usize;
    sentences.insert(at.min(sentences.len()), needle);
    let doc: String = sentences.concat();
    let user = format!("{}{question}", doc.trim_start());
    Needle { tokens: tok.encode(&crate::chatml(None, &[("user", &user)], false)), passcode, depth }
}

/// Runs one check in `s` (its context is replaced): reads the prompt, then answers greedily.
pub fn probe(m: &Model, s: &mut Session, tok: &Tokenizer, n: &Needle) -> Result<Probe, Error> {
    let t = std::time::Instant::now();
    s.clear()?;
    s.feed(m, &n.tokens)?;
    let mut out = Vec::new();
    for _ in 0..16 {
        let next = crate::sample::argmax(&s.logits);
        if tok.eot.contains(&next) {
            break;
        }
        out.push(next);
        s.step(m, next);
    }
    let reply = tok.decode(&out);
    let digits: String = reply.chars().filter(|c| c.is_ascii_digit()).collect();
    Ok(Probe { tokens: n.tokens.len(), depth: n.depth, found: digits.contains(&n.passcode), reply: reply.trim().to_owned(), seconds: t.elapsed().as_secs_f64() })
}
