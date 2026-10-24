//! Which experts of a mixture-of-experts model are firing: a heat per (layer, expert) that every
//! routing decision adds to and every token cools, plus the experts the latest token picked in
//! each layer. The app draws it as a heatmap and moves its logo's route with it.
//!
//! One model runs at a time, so this is process-wide; `reset` starts it for a newly loaded model.
//! Recording is a few additions per layer and a cooling pass per token (n_layer * n_expert
//! multiplies), far below the cost of the token itself.

use std::sync::Mutex;

/// How much of the heat a token keeps: about the last 30 tokens show.
const COOL: f32 = 0.93;

struct Activity {
    layers: usize,
    experts: usize,
    heat: Vec<f32>,
    last: Vec<Vec<u16>>,
    tokens: u64,
}

static ACT: Mutex<Option<Activity>> = Mutex::new(None);

/// What the app reads: heat scaled to 0..=255 by the hottest cell, layer-major.
pub struct Snapshot {
    pub layers: usize,
    pub experts: usize,
    pub heat: Vec<u8>,
    /// The experts the latest token picked, per layer.
    pub last: Vec<Vec<u16>>,
    /// Tokens recorded since the model was loaded.
    pub tokens: u64,
}

/// Starts recording for a model with `layers` x `experts` (0 experts: a dense model, nothing kept).
pub fn reset(layers: usize, experts: usize) {
    let mut g = ACT.lock().unwrap_or_else(|e| e.into_inner());
    *g = (experts > 0).then(|| Activity { layers, experts, heat: vec![0.0; layers * experts], last: vec![Vec::new(); layers], tokens: 0 });
}

/// One token's routing in `layer`: the experts it picked. Layer 0 also cools the map, once per token.
pub fn fire(layer: usize, picked: impl IntoIterator<Item = usize>) {
    let mut g = ACT.lock().unwrap_or_else(|e| e.into_inner());
    let Some(a) = g.as_mut() else { return };
    if layer >= a.layers {
        return;
    }
    if layer == 0 {
        a.heat.iter_mut().for_each(|h| *h *= COOL);
        a.tokens += 1;
    }
    let row = &mut a.heat[layer * a.experts..(layer + 1) * a.experts];
    let last = &mut a.last[layer];
    last.clear();
    for e in picked {
        if e < row.len() {
            row[e] += 1.0;
            last.push(e as u16);
        }
    }
}

/// A prompt chunk's routing in `layer`: how many of its tokens picked each expert. The map cools
/// as if those tokens had gone one by one, so a long prompt does not swamp what follows.
pub fn fire_counts(layer: usize, counts: &[(usize, usize)], tokens: usize) {
    let mut g = ACT.lock().unwrap_or_else(|e| e.into_inner());
    let Some(a) = g.as_mut() else { return };
    if layer >= a.layers || tokens == 0 {
        return;
    }
    if layer == 0 {
        let k = COOL.powi(tokens.min(1000) as i32);
        a.heat.iter_mut().for_each(|h| *h *= k);
        a.tokens += tokens as u64;
    }
    // A chunk's uses are spread over its tokens; weight them like the last ~1/(1-COOL) of them.
    let w = (1.0 / (1.0 - COOL)).min(tokens as f32) / tokens as f32;
    let row = &mut a.heat[layer * a.experts..(layer + 1) * a.experts];
    for &(e, n) in counts {
        if e < row.len() {
            row[e] += n as f32 * w;
        }
    }
}

/// The current map, or None for a dense model (or before any MoE model loaded).
pub fn snapshot() -> Option<Snapshot> {
    let g = ACT.lock().unwrap_or_else(|e| e.into_inner());
    let a = g.as_ref()?;
    let max = a.heat.iter().copied().fold(0.0f32, f32::max);
    let heat = a.heat.iter().map(|&h| if max > 0.0 { (h / max * 255.0).round() as u8 } else { 0 }).collect();
    Some(Snapshot { layers: a.layers, experts: a.experts, heat, last: a.last.clone(), tokens: a.tokens })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_follows_the_picks_and_cools() {
        reset(2, 4);
        fire(0, [1, 2]);
        fire(1, [3]);
        let s = snapshot().expect("moe");
        assert_eq!((s.layers, s.experts, s.tokens), (2, 4, 1));
        assert_eq!(s.heat[1], 255);
        assert_eq!(s.heat[4 + 3], 255);
        assert_eq!(s.heat[0], 0);
        assert_eq!(s.last[0], vec![1, 2]);
        // Later tokens on expert 0 overtake the cooled expert 1.
        for _ in 0..5 {
            fire(0, [0]);
        }
        let s = snapshot().expect("moe");
        assert_eq!(s.heat[0], 255);
        assert!(s.heat[1] < 80);
        reset(2, 0);
        assert!(snapshot().is_none());
    }
}
