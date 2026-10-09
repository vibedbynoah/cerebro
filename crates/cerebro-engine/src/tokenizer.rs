//! Byte-level BPE tokenizer built from GGUF metadata (`tokenizer.ggml.*`), as used by the Qwen,
//! Llama 3 and GPT families: text is split by the model's pre-tokenizer pattern, each piece is mapped
//! to byte symbols, then merged by rank. Special tokens (`<|im_start|>` …) are matched first.

use std::collections::HashMap;

use fancy_regex::Regex;

use crate::gguf::{Gguf, Value};
use crate::Error;

const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    vocab: Vec<String>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    specials: Vec<(String, u32)>,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    pattern: Regex,
    pub eos: u32,
    pub eot: Vec<u32>,
}

fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        out[*b as usize] = char::from_u32(c).unwrap();
    }
    out
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self, Error> {
        let model = g.str("tokenizer.ggml.model").unwrap_or("gpt2");
        if model != "gpt2" {
            return Err(Error::Unsupported(format!("tokenizer type {model}")));
        }
        let arr = |k: &str| g.meta.get(k).and_then(Value::as_array).ok_or_else(|| Error::Format(format!("missing {k}")));
        let vocab: Vec<String> = arr("tokenizer.ggml.tokens")?.iter().map(|v| v.as_str().unwrap_or("").to_owned()).collect();
        let types: Vec<u64> = g.meta.get("tokenizer.ggml.token_type").and_then(Value::as_array).map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1)).collect()).unwrap_or_default();
        let mut ranks = HashMap::new();
        for (i, m) in arr("tokenizer.ggml.merges")?.iter().enumerate() {
            if let Some((a, b)) = m.as_str().and_then(|s| s.split_once(' ')) {
                ranks.insert((a.to_owned(), b.to_owned()), i as u32);
            }
        }
        let ids: HashMap<String, u32> = vocab.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        // Control (3) and user-defined (4) tokens are matched verbatim before BPE.
        let mut specials: Vec<(String, u32)> = vocab
            .iter()
            .enumerate()
            .filter(|(i, t)| matches!(types.get(*i), Some(3) | Some(4)) && !t.is_empty())
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));
        let byte_enc = bytes_to_unicode();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        let eos = g.u64("tokenizer.ggml.eos_token_id").unwrap_or(0) as u32;
        let mut eot = vec![eos];
        for name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>"] {
            if let Some(&id) = ids.get(name) {
                if !eot.contains(&id) {
                    eot.push(id);
                }
            }
        }
        let pattern = Regex::new(QWEN2_PATTERN).map_err(|e| Error::Format(e.to_string()))?;
        Ok(Self { vocab, ids, ranks, specials, byte_enc, byte_dec, pattern, eos, eot })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // Next special token, if any.
            let next = self.specials.iter().filter_map(|(s, id)| rest.find(s.as_str()).map(|p| (p, s.len(), *id))).min_by_key(|(p, len, _)| (*p, std::cmp::Reverse(*len)));
            let (plain, special) = match next {
                Some((p, len, id)) => (&rest[..p], Some((id, p + len))),
                None => (rest, None),
            };
            self.encode_plain(plain, &mut out);
            match special {
                Some((id, end)) => {
                    out.push(id);
                    rest = &rest[end..];
                }
                None => break,
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for m in self.pattern.find_iter(text).flatten() {
            let piece: String = m.as_str().bytes().map(|b| self.byte_enc[b as usize]).collect();
            if let Some(&id) = self.ids.get(&piece) {
                out.push(id);
                continue;
            }
            let mut parts: Vec<String> = piece.chars().map(|c| c.to_string()).collect();
            loop {
                let mut best: Option<(usize, u32)> = None;
                for i in 0..parts.len().saturating_sub(1) {
                    if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                        if best.is_none_or(|(_, br)| r < br) {
                            best = Some((i, r));
                        }
                    }
                }
                let Some((i, _)) = best else { break };
                let merged = format!("{}{}", parts[i], parts[i + 1]);
                parts.splice(i..i + 2, [merged]);
            }
            for p in parts {
                if let Some(&id) = self.ids.get(&p) {
                    out.push(id);
                } else {
                    // Fall back to single bytes (always in a byte-level vocabulary).
                    for c in p.chars() {
                        if let Some(&id) = self.ids.get(&c.to_string()) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// Decodes token ids to bytes (may end mid-UTF-8 sequence while streaming).
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            let Some(t) = self.vocab.get(id as usize) else { continue };
            if self.specials.iter().any(|(_, s)| *s == id) {
                out.extend_from_slice(t.as_bytes());
                continue;
            }
            for c in t.chars() {
                match self.byte_dec.get(&c) {
                    Some(&b) => out.push(b),
                    None => out.extend_from_slice(c.to_string().as_bytes()),
                }
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.ids.get(s).copied()
    }
}
