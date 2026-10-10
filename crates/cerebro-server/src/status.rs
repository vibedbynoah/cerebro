//! What `/cerebro/status` reports: the model, the last measured speeds, VRAM, and the launch's
//! disabled protections.
//!
//! Speeds are llama-server's own `timings` from the most recent completion the gateway relayed
//! (decode `predicted_per_second`, prefill `prompt_per_second`); nothing is estimated. VRAM comes
//! from a [`VramSource`], by default `nvidia-smi` through `cerebro-probe`, sampled at most once per
//! [`VRAM_CACHE`] so polling the status route never spawns a process per request.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

/// How long one VRAM sample is reused.
pub const VRAM_CACHE: Duration = Duration::from_secs(2);

/// One GPU's memory, in MiB.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuMemory {
    /// Device name as the driver reports it.
    pub name: String,
    /// Total memory, MiB.
    pub total_mib: u64,
    /// Free memory, MiB.
    pub free_mib: u64,
}

/// Where VRAM figures come from. Called on a blocking thread.
pub type VramSource = Box<dyn Fn() -> Result<Vec<GpuMemory>, String> + Send + Sync>;

/// The production source: NVIDIA GPUs through `cerebro-probe` (an empty list when there are none).
pub fn probe_vram() -> VramSource {
    Box::new(|| {
        let inventory = cerebro_probe::gpu::probe().map_err(|e| e.to_string())?;
        Ok(inventory
            .nvidia()
            .iter()
            .map(|g| GpuMemory { name: g.name().to_string(), total_mib: g.total_mib(), free_mib: g.free_mib() })
            .collect())
    })
}

/// Decode and prefill speed of one completion, tokens per second.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Speed {
    /// `timings.predicted_per_second`.
    pub decode: Option<f64>,
    /// `timings.prompt_per_second`.
    pub prefill: Option<f64>,
}

impl Speed {
    /// Speeds from a parsed llama-server response, if it carried `timings`.
    pub fn from_response(response: &Value) -> Option<Self> {
        let timings = response.get("timings")?;
        let speed = Self {
            decode: timings.get("predicted_per_second").and_then(Value::as_f64),
            prefill: timings.get("prompt_per_second").and_then(Value::as_f64),
        };
        (speed.decode.is_some() || speed.prefill.is_some()).then_some(speed)
    }

    /// Speeds from the tail of a streamed response: the last `"predicted_per_second":` and
    /// `"prompt_per_second":` numbers in it (llama-server puts `timings` in its final chunk).
    pub fn from_stream_tail(tail: &[u8]) -> Option<Self> {
        let text = String::from_utf8_lossy(tail);
        let speed = Self {
            decode: last_number_after(&text, "\"predicted_per_second\":"),
            prefill: last_number_after(&text, "\"prompt_per_second\":"),
        };
        (speed.decode.is_some() || speed.prefill.is_some()).then_some(speed)
    }
}

fn last_number_after(text: &str, key: &str) -> Option<f64> {
    let start = text.rfind(key)? + key.len();
    let rest = text[start..].trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(rest.len());
    rest[..end].parse().ok().filter(|v: &f64| v.is_finite())
}

/// One VRAM sample (or why it failed) and when it was taken.
type VramSample = (Result<Vec<GpuMemory>, String>, Instant);

/// Shared, mutable status: last speed, cached VRAM sample, cached model properties.
pub struct Telemetry {
    last_speed: Mutex<Option<(Speed, Instant)>>,
    vram: Mutex<Option<VramSample>>,
    source: VramSource,
}

impl std::fmt::Debug for Telemetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Telemetry").field("last_speed", &self.last_speed()).finish_non_exhaustive()
    }
}

impl Telemetry {
    /// Telemetry reading VRAM from `source`.
    pub fn new(source: VramSource) -> Self {
        Self { last_speed: Mutex::new(None), vram: Mutex::new(None), source }
    }

    /// Record the speed of a completion that just finished.
    pub fn record(&self, speed: Speed) {
        if let Ok(mut slot) = self.last_speed.lock() {
            *slot = Some((speed, Instant::now()));
        }
    }

    /// The most recent completion's speed, if any.
    pub fn last_speed(&self) -> Option<Speed> {
        self.last_speed.lock().ok().and_then(|s| s.map(|(speed, _)| speed))
    }

    /// The VRAM sample, refreshed when older than [`VRAM_CACHE`]. Blocking: call from a blocking
    /// thread.
    pub fn vram_blocking(&self) -> Result<Vec<GpuMemory>, String> {
        let Ok(mut slot) = self.vram.lock() else {
            return Err("VRAM cache unavailable".to_string());
        };
        if let Some((sample, at)) = slot.as_ref() {
            if at.elapsed() < VRAM_CACHE {
                return sample.clone();
            }
        }
        let sample = (self.source)();
        *slot = Some((sample.clone(), Instant::now()));
        sample
    }

    /// The `speed` part of the status body.
    pub fn speed_json(&self) -> Value {
        match self.last_speed.lock().ok().and_then(|s| *s) {
            Some((speed, at)) => json!({
                "decode_tokens_per_second": speed.decode,
                "prefill_tokens_per_second": speed.prefill,
                "measured_seconds_ago": at.elapsed().as_secs(),
                "source": "llama-server timings of the last completion",
            }),
            None => Value::Null,
        }
    }
}

/// The `vram` part of the status body.
pub fn vram_json(sample: &Result<Vec<GpuMemory>, String>) -> Value {
    match sample {
        Ok(gpus) => Value::Array(
            gpus.iter()
                .map(|g| json!({
                    "name": g.name,
                    "total_mib": g.total_mib,
                    "free_mib": g.free_mib,
                    "used_mib": g.total_mib.saturating_sub(g.free_mib),
                }))
                .collect(),
        ),
        Err(e) => json!({ "error": e }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn speeds_come_from_timings_only() {
        let r = json!({"timings": {"predicted_per_second": 41.5, "prompt_per_second": 900.0}});
        assert_eq!(Speed::from_response(&r), Some(Speed { decode: Some(41.5), prefill: Some(900.0) }));
        assert_eq!(Speed::from_response(&json!({"choices": []})), None);
    }

    #[test]
    fn stream_tail_takes_the_last_timings() {
        let tail = br#"data: {"timings":{"prompt_per_second":10.0,"predicted_per_second":1.5}}

data: {"choices":[],"timings":{"prompt_per_second":812.25,"predicted_per_second":63.75e0}}

data: [DONE]
"#;
        assert_eq!(Speed::from_stream_tail(tail), Some(Speed { decode: Some(63.75), prefill: Some(812.25) }));
        assert_eq!(Speed::from_stream_tail(b"data: [DONE]"), None);
    }

    #[test]
    fn vram_is_sampled_at_most_once_per_cache_window() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let t = Telemetry::new(Box::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(vec![GpuMemory { name: "GPU".into(), total_mib: 8192, free_mib: 1024 }])
        }));
        let first = t.vram_blocking().expect("sample");
        let second = t.vram_blocking().expect("sample");
        assert_eq!(first, second);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(vram_json(&Ok(first))[0]["used_mib"], 7168);
    }
}
