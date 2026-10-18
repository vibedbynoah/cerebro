//! Token sampling: temperature, top-k, top-p, min-p and a repetition penalty. Temperature 0 is
//! greedy. The generator is a small seeded PRNG so runs are reproducible.

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self { temperature: 0.7, top_k: 40, top_p: 0.95, min_p: 0.05, repeat_penalty: 1.05, seed: 0x5eed }
    }
}

pub struct Sampler {
    pub cfg: SamplerConfig,
    state: u64,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let state = cfg.seed | 1;
        Self { cfg, state }
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        ((self.state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / (1u64 << 24) as f32
    }

    /// The distribution `sample` draws from, after the repetition penalty, temperature, top-k, min-p
    /// and top-p: (token, probability) pairs, most likely first, summing to 1. Greedy is one token.
    pub fn distribution(&self, logits: &[f32], recent: &[u32]) -> Vec<(u32, f32)> {
        let c = &self.cfg;
        let mut l: Vec<f32> = logits.to_vec();
        if c.repeat_penalty != 1.0 {
            for &t in recent.iter().rev().take(64) {
                if let Some(v) = l.get_mut(t as usize) {
                    *v = if *v > 0.0 { *v / c.repeat_penalty } else { *v * c.repeat_penalty };
                }
            }
        }
        if c.temperature <= 0.0 {
            return vec![(argmax(&l), 1.0)];
        }
        let mut idx: Vec<u32> = (0..l.len() as u32).collect();
        let k = if c.top_k == 0 { l.len() } else { c.top_k.min(l.len()) };
        idx.select_nth_unstable_by(k - 1, |&a, &b| l[b as usize].total_cmp(&l[a as usize]));
        idx.truncate(k);
        idx.sort_unstable_by(|&a, &b| l[b as usize].total_cmp(&l[a as usize]));
        let mx = l[idx[0] as usize];
        let mut p: Vec<f32> = idx.iter().map(|&i| ((l[i as usize] - mx) / c.temperature).exp()).collect();
        let sum: f32 = p.iter().sum();
        p.iter_mut().for_each(|v| *v /= sum);
        // min-p: drop tokens below min_p times the best probability.
        let floor = c.min_p * p[0];
        let mut keep = p.iter().take_while(|&&v| v >= floor).count().max(1);
        // top-p
        let mut acc = 0.0;
        for (i, v) in p.iter().enumerate().take(keep) {
            acc += v;
            if acc >= c.top_p {
                keep = i + 1;
                break;
            }
        }
        let total: f32 = p[..keep].iter().sum();
        idx[..keep].iter().zip(&p[..keep]).map(|(&t, &v)| (t, v / total)).collect()
    }

    fn draw(&mut self, dist: &[(u32, f32)]) -> u32 {
        let total: f32 = dist.iter().map(|d| d.1).sum();
        let mut r = self.next_f32() * total;
        for &(t, v) in dist {
            r -= v;
            if r <= 0.0 {
                return t;
            }
        }
        dist.last().map_or(0, |d| d.0)
    }

    pub fn sample(&mut self, logits: &[f32], recent: &[u32]) -> u32 {
        let d = self.distribution(logits, recent);
        self.draw(&d)
    }

    /// Speculative sampling against a deterministic draft token: accepted with the probability the
    /// sampler gives it; on rejection a token is drawn from the distribution without it. The tokens
    /// that come out follow the sampler's distribution exactly. Ok = accepted, Err = the replacement.
    pub fn speculate(&mut self, logits: &[f32], recent: &[u32], draft: u32) -> Result<(), u32> {
        let d = self.distribution(logits, recent);
        let pd = d.iter().find(|x| x.0 == draft).map_or(0.0, |x| x.1);
        if pd >= 1.0 || (pd > 0.0 && self.next_f32() < pd) {
            return Ok(());
        }
        let rest: Vec<(u32, f32)> = d.into_iter().filter(|x| x.0 != draft).collect();
        Err(if rest.is_empty() { draft } else { self.draw(&rest) })
    }
}

pub fn argmax(l: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..l.len() {
        if l[i] > l[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_the_largest() {
        let mut s = Sampler::new(SamplerConfig { temperature: 0.0, ..Default::default() });
        assert_eq!(s.sample(&[0.1, 3.0, 2.0], &[]), 1);
    }

    #[test]
    fn speculation_keeps_the_distribution() {
        // Target distribution over 3 tokens; the draft always proposes token 1.
        let logits = [1.0f32, 1.5, 0.2];
        let cfg = SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0, repeat_penalty: 1.0, seed: 11 };
        let want = Sampler::new(cfg.clone()).distribution(&logits, &[]);
        let mut s = Sampler::new(cfg);
        let mut n = [0usize; 3];
        let trials = 200_000;
        for _ in 0..trials {
            let t = match s.speculate(&logits, &[], 1) {
                Ok(()) => 1,
                Err(t) => t,
            };
            n[t as usize] += 1;
        }
        for &(t, p) in &want {
            let got = n[t as usize] as f32 / trials as f32;
            assert!((got - p).abs() < 0.006, "token {t}: {got} vs {p}");
        }
        // Greedy: accepted exactly when the draft is the argmax.
        let mut g = Sampler::new(SamplerConfig { temperature: 0.0, ..Default::default() });
        assert_eq!(g.speculate(&logits, &[], 1), Ok(()));
        assert_eq!(g.speculate(&logits, &[], 0), Err(1));
    }

    #[test]
    fn min_p_never_picks_a_tiny_token() {
        let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.5, repeat_penalty: 1.0, seed: 7 });
        for _ in 0..200 {
            assert_ne!(s.sample(&[10.0, 9.9, -10.0], &[]), 2);
        }
    }
}
