//! Topic-aware expert caching for mixture-of-experts models.
//!
//! Different subjects use different experts: a coding prompt and a French one light up largely
//! separate sets. The experts the router picks while reading a prompt are therefore the prompt's
//! fingerprint, in any language and without guessing from keywords. Each topic keeps its own
//! expert-use profile, learned as the user chats; when a new prompt's fingerprint matches a topic,
//! that topic's hot experts are what the caches should hold (VRAM first, then RAM) and its cold
//! ones can stay on the drive. Switching from code to French swaps what the caches prefer instead
//! of letting them drift over hundreds of tokens.
//!
//! This module is the bookkeeping only: fingerprints, matching, profiles and a plain label for the
//! Monitor. The caches (gpu.rs, experts.rs) take `hot` as their preference.

/// One learned topic.
#[derive(Clone, Debug)]
pub struct Topic {
    /// A short name for the Monitor ("Code", "French", ...), from the prompts that formed it.
    pub label: String,
    /// The normalised expert-use histogram of the prompts seen (one value per layer x expert).
    pub centroid: Vec<f32>,
    /// Expert uses while answering, decayed: what the caches should keep for this topic.
    pub counts: Vec<f32>,
    /// Prompts matched, for replacing the least used topic.
    pub uses: u32,
}

/// The topics of one model.
#[derive(Clone, Debug, Default)]
pub struct Topics {
    pub topics: Vec<Topic>,
    /// Experts per layer times layers: the length of every histogram.
    pub width: usize,
}

/// Most topics kept per model.
pub const MAX_TOPICS: usize = 8;
/// Cosine similarity above which a prompt belongs to an existing topic.
pub const MATCH: f32 = 0.55;

fn normalise(h: &[f32]) -> Vec<f32> {
    let n = h.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n > 0.0 { h.iter().map(|v| v / n).collect() } else { h.to_vec() }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// A plain label for a prompt, for display only (matching uses the experts, not this).
pub fn label_of(text: &str) -> String {
    let t = text.to_lowercase();
    let code = ["```", "fn ", "def ", "class ", "import ", "#include", "{", "};", "=>", "console.", "select ", "function"];
    if code.iter().filter(|k| t.contains(*k)).count() >= 2 {
        return "Code".into();
    }
    let words: Vec<&str> = t.split(|c: char| !c.is_alphanumeric() && c != '\'').filter(|w| !w.is_empty()).collect();
    let share = |list: &[&str]| words.iter().filter(|w| list.contains(w)).count() as f32 / words.len().max(1) as f32;
    let langs: [(&str, &[&str]); 5] = [
        ("French", &["le", "la", "les", "est", "et", "une", "des", "pour", "que", "je", "vous", "pas", "dans", "c'est"]),
        ("Spanish", &["el", "los", "las", "es", "y", "una", "por", "que", "para", "con", "como", "pero"]),
        ("German", &["der", "die", "das", "und", "ist", "nicht", "ein", "eine", "ich", "mit", "für", "auf"]),
        ("Italian", &["il", "lo", "gli", "è", "e", "una", "per", "che", "non", "con", "sono"]),
        ("Portuguese", &["o", "os", "as", "é", "e", "uma", "para", "que", "não", "com", "por"]),
    ];
    if let Some((name, s)) = langs.iter().map(|(n, l)| (*n, share(l))).max_by(|a, b| a.1.total_cmp(&b.1)) {
        if s >= 0.12 {
            return name.into();
        }
    }
    if t.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)) {
        return "Chinese".into();
    }
    if t.chars().any(|c| ('\u{3040}'..='\u{30ff}').contains(&c)) {
        return "Japanese".into();
    }
    let digits = t.chars().filter(|c| c.is_ascii_digit() || "+-*/=^".contains(*c)).count();
    if digits * 6 >= t.len().max(1) {
        return "Math".into();
    }
    "General".into()
}

impl Topics {
    pub fn new(width: usize) -> Self {
        Self { topics: Vec::new(), width }
    }

    /// The topic a prompt belongs to, from its routing histogram (expert uses while reading it):
    /// the closest one above MATCH, else a new topic (replacing the least used when full).
    /// Returns its index.
    pub fn assign(&mut self, prompt_hist: &[f32], text: &str) -> usize {
        assert_eq!(prompt_hist.len(), self.width, "histogram width");
        let h = normalise(prompt_hist);
        let best = self.topics.iter().enumerate().map(|(i, t)| (i, cosine(&h, &t.centroid))).max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, sim)) = best.filter(|b| b.1 >= MATCH) {
            let t = &mut self.topics[i];
            // The centroid moves towards the new prompt, more slowly as the topic grows.
            let w = 1.0 / (t.uses as f32 + 1.0).min(16.0);
            let merged: Vec<f32> = t.centroid.iter().zip(&h).map(|(c, x)| c * (1.0 - w) + x * w).collect();
            t.centroid = normalise(&merged);
            t.uses += 1;
            let _ = sim;
            return i;
        }
        let topic = Topic { label: label_of(text), centroid: h, counts: prompt_hist.to_vec(), uses: 1 };
        if self.topics.len() < MAX_TOPICS {
            self.topics.push(topic);
            self.topics.len() - 1
        } else {
            let i = self.topics.iter().enumerate().min_by_key(|(_, t)| t.uses).map_or(0, |(i, _)| i);
            self.topics[i] = topic;
            i
        }
    }

    /// Adds the experts used while answering to topic `i` (older uses fade by `decay`).
    pub fn learn(&mut self, i: usize, reply_hist: &[f32], decay: f32) {
        if let Some(t) = self.topics.get_mut(i) {
            for (c, x) in t.counts.iter_mut().zip(reply_hist) {
                *c = *c * decay + x;
            }
        }
    }

    /// Topic `i`'s experts, most used first: what the caches should hold for it.
    pub fn hot(&self, i: usize) -> Vec<usize> {
        let Some(t) = self.topics.get(i) else { return Vec::new() };
        let mut keys: Vec<usize> = (0..t.counts.len()).filter(|&k| t.counts[k] > 0.0).collect();
        keys.sort_by(|&a, &b| t.counts[b].total_cmp(&t.counts[a]));
        keys
    }

    /// Saved as text next to the model's expert profile: one topic per line,
    /// "label|uses|centroid values|count values".
    pub fn to_text(&self) -> String {
        self.topics
            .iter()
            .map(|t| {
                let f = |v: &[f32]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(" ");
                format!("{}|{}|{}|{}", t.label.replace('|', " "), t.uses, f(&t.centroid), f(&t.counts))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn from_text(s: &str, width: usize) -> Self {
        let mut out = Self::new(width);
        for line in s.lines() {
            let p: Vec<&str> = line.split('|').collect();
            if p.len() != 4 {
                continue;
            }
            let nums = |x: &str| x.split_whitespace().filter_map(|v| v.parse::<f32>().ok()).collect::<Vec<f32>>();
            let (c, k) = (nums(p[2]), nums(p[3]));
            if c.len() == width && k.len() == width && out.topics.len() < MAX_TOPICS {
                out.topics.push(Topic { label: p[0].to_owned(), centroid: c, counts: k, uses: p[1].parse().unwrap_or(1) });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A histogram using experts `set` (width 64).
    fn hist(set: &[usize]) -> Vec<f32> {
        let mut h = vec![0.0; 64];
        for &k in set {
            h[k] += 3.0;
        }
        h
    }

    #[test]
    fn opposite_topics_get_separate_profiles_and_come_back() {
        let mut t = Topics::new(64);
        let code = t.assign(&hist(&[1, 2, 3, 4, 5]), "fn main() { println!(\"hi\"); } ```rust");
        let french = t.assign(&hist(&[40, 41, 42, 43, 44]), "Bonjour, c'est une question pour vous: est-ce que le train part?");
        assert_ne!(code, french);
        assert_eq!(t.topics[code].label, "Code");
        assert_eq!(t.topics[french].label, "French");
        // A new coding prompt with mostly the same experts lands in the coding topic.
        assert_eq!(t.assign(&hist(&[1, 2, 3, 4, 6]), "def f(x): return x"), code);
        t.learn(code, &hist(&[2, 2, 9]), 0.9);
        let hot = t.hot(code);
        assert_eq!(hot[0], 2);
        assert!(!hot.contains(&40));
    }

    #[test]
    fn profiles_survive_a_save() {
        let mut t = Topics::new(64);
        t.assign(&hist(&[7, 8]), "2 + 2 * 3 = ?");
        let back = Topics::from_text(&t.to_text(), 64);
        assert_eq!(back.topics.len(), 1);
        assert_eq!(back.topics[0].label, "Math");
        assert_eq!(back.hot(0)[..2].iter().copied().collect::<std::collections::BTreeSet<_>>(), [7, 8].into_iter().collect());
    }

    #[test]
    fn the_least_used_topic_makes_room() {
        let mut t = Topics::new(64);
        for i in 0..MAX_TOPICS {
            t.assign(&hist(&[i * 7, i * 7 + 1]), "general text");
        }
        t.assign(&hist(&[0, 1]), "again");
        let n = t.assign(&hist(&[60, 61, 62, 63]), "new subject");
        assert!(n < MAX_TOPICS);
        assert_eq!(t.topics.len(), MAX_TOPICS);
    }
}
