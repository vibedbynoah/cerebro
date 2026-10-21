//! Cluster: run one model across several computers (cerebro_engine::cluster). This computer can
//! share itself as a worker (another Cerebro then runs some of a model's layers here), and can load
//! a model with some of its layers on paired workers. Pairing is a 6-digit code shown on the worker;
//! both sides derive the same key from it, and the worker refuses any connection that cannot prove
//! it. Nothing listens until the user turns sharing on.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::sync::Mutex;
use std::time::Duration;

use cerebro_engine::cluster as ec;

/// The key both sides derive from a pairing code.
pub fn key_from_code(code: &str) -> Vec<u8> {
    let digits: String = code.chars().filter(|c| c.is_ascii_digit()).collect();
    Sha256::digest(format!("cerebro-pair-v1:{digits}")).to_vec()
}

fn new_code() -> String {
    let mut h = Sha256::new();
    h.update(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()).to_le_bytes());
    h.update(std::process::id().to_le_bytes());
    let v = u32::from_le_bytes(h.finalize()[..4].try_into().expect("4 bytes")) % 1_000_000;
    format!("{v:06}")
}

/// This computer's addresses other computers on the network can reach.
pub fn local_addresses() -> Vec<String> {
    let mut out = Vec::new();
    // The address the default route uses (no packet is sent by connect on UDP).
    if let Ok(s) = std::net::UdpSocket::bind("0.0.0.0:0") {
        if s.connect("192.0.2.1:9").is_ok() {
            if let Ok(a) = s.local_addr() {
                if !a.ip().is_loopback() && !a.ip().is_unspecified() {
                    out.push(a.ip().to_string());
                }
            }
        }
    }
    out
}

struct Sharing {
    worker: ec::Worker,
    code: String,
    model: String,
    port: u16,
}

static SHARING: Mutex<Option<Sharing>> = Mutex::new(None);

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShareView {
    pub sharing: bool,
    pub model: Option<String>,
    pub code: Option<String>,
    pub port: Option<u16>,
    pub addresses: Vec<String>,
    pub computer: String,
}

pub fn computer_name() -> String {
    std::env::var("COMPUTERNAME").unwrap_or_else(|_| "This computer".to_owned())
}

pub fn share_view() -> ShareView {
    let g = SHARING.lock().unwrap_or_else(|e| e.into_inner());
    ShareView {
        sharing: g.is_some(),
        model: g.as_ref().map(|s| s.model.clone()),
        code: g.as_ref().map(|s| s.code.clone()),
        port: g.as_ref().map(|s| s.port),
        addresses: local_addresses(),
        computer: computer_name(),
    }
}

/// Lets other computers run layers of `model_path` here, behind a new pairing code.
pub fn share_start(model_path: &std::path::Path, model_name: &str) -> Result<ShareView, String> {
    share_stop();
    let code = new_code();
    let key = key_from_code(&code);
    let cfg = |port: u16| ec::WorkerConfig { model_path: model_path.to_path_buf(), key: key.clone(), addr: std::net::SocketAddr::from(([0, 0, 0, 0], port)), name: computer_name(), announce: true };
    let worker = ec::serve(cfg(ec::DEFAULT_PORT)).or_else(|_| ec::serve(cfg(0))).map_err(|e| format!("Could not share this computer: {e}"))?;
    let port = worker.addr.port();
    *SHARING.lock().unwrap_or_else(|e| e.into_inner()) = Some(Sharing { worker, code, model: model_name.to_owned(), port });
    Ok(share_view())
}

pub fn share_stop() {
    if let Some(s) = SHARING.lock().unwrap_or_else(|e| e.into_inner()).take() {
        s.worker.stop();
    }
}

/// Workers announcing themselves on the local network (listens for about 2.5 seconds).
pub fn discover() -> Vec<ec::Announced> {
    ec::discover(Duration::from_millis(2500))
}

/// One worker to use: where it is, its pairing code, and the layers it runs.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerChoice {
    pub addr: String,
    pub code: String,
    /// This worker's share of the model, 0..1 (the planner turns shares into layer ranges).
    pub share: f64,
}

/// How one link measured when the model loaded across computers.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LinkReport {
    pub addr: String,
    pub lo: usize,
    pub hi: usize,
    pub rtt_ms: f64,
    pub mb_per_s: f64,
}

/// Connects a session's remote ranges: this computer keeps the first share of the layers, each
/// worker the next contiguous range in order.
pub fn connect(model_path: &std::path::Path, n_layer: usize, ctx: usize, local_share: f64, workers: &[WorkerChoice]) -> Result<(ec::Remote, Vec<LinkReport>), String> {
    if workers.is_empty() || workers.len() > 8 {
        return Err("Choose from one to eight other computers.".to_owned());
    }
    if n_layer < workers.len() + 1 {
        return Err("This model has too few layers to share between that many computers.".to_owned());
    }
    let fp = ec::fingerprint(model_path).map_err(|e| format!("Could not read the model file: {e}"))?;
    let plan = ec::plan(n_layer, local_share.max(0.01), &workers.iter().map(|w| w.share.max(0.01)).collect::<Vec<_>>());
    let mut ranges = Vec::new();
    let mut reports = Vec::new();
    for (w, (lo, hi)) in workers.iter().zip(plan) {
        let mut r = ec::RemoteRange::connect(&w.addr, &key_from_code(&w.code), &fp, lo, hi, ctx).map_err(|e| e.to_string())?;
        let (rtt_ms, mb_per_s) = r.probe().map_err(|e| e.to_string())?;
        reports.push(LinkReport { addr: w.addr.clone(), lo, hi, rtt_ms, mb_per_s });
        ranges.push(r);
    }
    Ok((ec::Remote { ranges }, reports))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_code_gives_the_same_key_however_it_is_typed() {
        assert_eq!(key_from_code("123456"), key_from_code("123 456"));
        assert_eq!(key_from_code("123-456"), key_from_code("123456"));
        assert_ne!(key_from_code("123456"), key_from_code("123457"));
        let c = new_code();
        assert_eq!(c.len(), 6);
        assert!(c.chars().all(|x| x.is_ascii_digit()));
    }
}
