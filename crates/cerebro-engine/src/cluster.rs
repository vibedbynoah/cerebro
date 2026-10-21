//! One model across several computers: contiguous layer ranges run on other machines on the local
//! network (Wi-Fi, Ethernet, a USB network link: any IP connection). A worker loads the same GGUF
//! file and runs its layers for the hidden states a client sends; the client runs the rest. Each
//! token's activation crosses the link once per remote range, so a split costs one round trip per
//! range per token (and prompt tokens go in batches).
//!
//! Protocol (TCP, little-endian, length-prefixed frames `u32 len | u8 kind | payload`):
//! 1. The worker sends `Challenge { nonce: [u8; 16] }`.
//! 2. The client sends `Hello { version, fingerprint: [u8; 32], lo, hi, max_ctx, proof: [u8; 32] }`,
//!    `proof = HMAC-SHA256(pairing key, nonce || fingerprint)`. The worker checks the proof against
//!    its own pairing key, that `fingerprint` is its own model file's, and the range; it answers
//!    `Ready { dim }` or `Error { message }`.
//! 3. `Forward { pos0, rows }` -> `Rows { rows }`: the hidden states after layers lo..hi for the
//!    tokens at pos0, pos0 + 1, ... (f32). The worker keeps that connection's KV cache, indexed by
//!    position, so a client that rewinds simply sends earlier positions again.
//! 4. `Ping { payload }` -> `Pong { payload }` (round-trip time and bandwidth probes).
//!
//! Discovery: a worker announces itself every two seconds by UDP broadcast on port 50081
//! (`CEREBRO-WORKER v1 <name> <tcp port> <fingerprint hex> <free memory GiB>`).

use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::model::{Model, Session};
use crate::Error;

pub const VERSION: u32 = 1;
pub const DEFAULT_PORT: u16 = 50080;
pub const DISCOVERY_PORT: u16 = 50081;

const CHALLENGE: u8 = 1;
const HELLO: u8 = 2;
const READY: u8 = 3;
const FORWARD: u8 = 4;
const ROWS: u8 = 5;
const PING: u8 = 6;
const PONG: u8 = 7;
const ERROR: u8 = 8;
/// Frames above this are refused (a 1,024-token batch of an 8,192-wide model is 32 MiB).
const MAX_FRAME: usize = 64 << 20;

fn io(e: std::io::Error) -> Error {
    Error::Request(format!("cluster link: {e}"))
}

/// HMAC-SHA256 (RFC 2104).
pub fn hmac(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 64];
    if key.len() > 64 {
        k[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(k.iter().map(|b| b ^ 0x36).collect::<Vec<u8>>());
    inner.update(msg);
    let ih = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(k.iter().map(|b| b ^ 0x5c).collect::<Vec<u8>>());
    outer.update(ih);
    outer.finalize().into()
}

/// A model file's identity: SHA-256 over its size and its first and last 4 MiB (the header, the
/// tensor table and the ends of the data), fast even for a 100 GB file.
pub fn fingerprint(path: &Path) -> Result<[u8; 32], Error> {
    use std::io::{Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    let mut h = Sha256::new();
    h.update(len.to_le_bytes());
    let chunk = 4u64 << 20;
    let mut buf = vec![0u8; chunk.min(len) as usize];
    f.read_exact(&mut buf)?;
    h.update(&buf);
    if len > chunk {
        let start = len.saturating_sub(chunk).max(chunk);
        f.seek(SeekFrom::Start(start))?;
        let mut tail = Vec::new();
        f.read_to_end(&mut tail)?;
        h.update(&tail);
    }
    Ok(h.finalize().into())
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn write_frame(w: &mut impl Write, kind: u8, payload: &[u8]) -> Result<(), Error> {
    w.write_all(&((payload.len() + 1) as u32).to_le_bytes()).map_err(io)?;
    w.write_all(&[kind]).map_err(io)?;
    w.write_all(payload).map_err(io)?;
    w.flush().map_err(io)
}

fn read_frame(r: &mut impl Read) -> Result<(u8, Vec<u8>), Error> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len).map_err(io)?;
    let len = u32::from_le_bytes(len) as usize;
    if len == 0 || len > MAX_FRAME {
        return Err(Error::Request(format!("cluster link: a frame of {len} bytes")));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(io)?;
    let kind = buf[0];
    buf.remove(0);
    Ok((kind, buf))
}

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

fn nonce() -> [u8; 16] {
    let mut h = Sha256::new();
    h.update(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()).to_le_bytes());
    h.update(std::process::id().to_le_bytes());
    let mut n = [0u8; 16];
    n.copy_from_slice(&h.finalize()[..16]);
    n
}

/// How a worker runs: its model, the pairing key clients must prove, and where it listens.
pub struct WorkerConfig {
    pub model_path: std::path::PathBuf,
    pub key: Vec<u8>,
    pub addr: SocketAddr,
    /// A name for discovery (the computer's name by default).
    pub name: String,
    /// Announce on the LAN by UDP broadcast.
    pub announce: bool,
}

/// A running worker; dropping it does not stop it, `stop` does.
pub struct Worker {
    pub addr: SocketAddr,
    stop: Arc<AtomicBool>,
}

impl Worker {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        // Unblock accept().
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_millis(200));
    }
}

/// Starts a worker: loads the model once and serves each connection on its own thread.
pub fn serve(cfg: WorkerConfig) -> Result<Worker, Error> {
    let model = Arc::new(Model::load(&cfg.model_path)?);
    let fp = fingerprint(&cfg.model_path)?;
    let listener = TcpListener::bind(cfg.addr).map_err(io)?;
    let addr = listener.local_addr().map_err(io)?;
    let stop = Arc::new(AtomicBool::new(false));
    let key = Arc::new(cfg.key.clone());
    if cfg.announce {
        let (stop, name) = (stop.clone(), cfg.name.clone());
        std::thread::spawn(move || announce(&name, addr.port(), &fp, &stop));
    }
    let stop2 = stop.clone();
    std::thread::Builder::new()
        .name("cluster-worker".into())
        .spawn(move || {
            for conn in listener.incoming() {
                if stop2.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(s) = conn else { continue };
                let (m, k) = (model.clone(), key.clone());
                std::thread::spawn(move || {
                    let _ = s.set_nodelay(true);
                    if let Err(e) = handle(s, &m, &fp, &k) {
                        eprintln!("cluster worker: {e}");
                    }
                });
            }
        })
        .map_err(io)?;
    Ok(Worker { addr, stop })
}

fn handle(s: TcpStream, m: &Model, fp: &[u8; 32], key: &[u8]) -> Result<(), Error> {
    let mut r = BufReader::new(s.try_clone().map_err(io)?);
    let mut w = BufWriter::new(s);
    let n = nonce();
    write_frame(&mut w, CHALLENGE, &n)?;
    let (kind, hello) = read_frame(&mut r)?;
    if kind != HELLO || hello.len() != 4 + 32 + 12 + 32 {
        return write_frame(&mut w, ERROR, b"expected a hello");
    }
    let version = u32_at(&hello, 0);
    let their_fp = &hello[4..36];
    let (lo, hi, max_ctx) = (u32_at(&hello, 36) as usize, u32_at(&hello, 40) as usize, u32_at(&hello, 44) as usize);
    let proof = &hello[48..80];
    let mut msg = n.to_vec();
    msg.extend_from_slice(their_fp);
    if hmac(key, &msg) != proof {
        return write_frame(&mut w, ERROR, b"this computer is not paired with the worker (wrong pairing code)");
    }
    if version != VERSION {
        return write_frame(&mut w, ERROR, format!("protocol version {version}; this worker speaks {VERSION}").as_bytes());
    }
    if their_fp != fp {
        return write_frame(&mut w, ERROR, b"the worker has a different model file loaded");
    }
    if lo >= hi || hi > m.cfg.n_layer || max_ctx == 0 || max_ctx > 1 << 20 {
        return write_frame(&mut w, ERROR, format!("layers {lo}..{hi} of a {}-layer model", m.cfg.n_layer).as_bytes());
    }
    let mut session = Session::new(m, max_ctx);
    write_frame(&mut w, READY, &(m.cfg.dim as u32).to_le_bytes())?;
    let dim = m.cfg.dim;
    loop {
        let (kind, body) = match read_frame(&mut r) {
            Ok(f) => f,
            Err(_) => return Ok(()), // the client went away
        };
        match kind {
            FORWARD if body.len() >= 4 && (body.len() - 4) % (dim * 4) == 0 => {
                let pos0 = u32_at(&body, 0) as usize;
                let mut rows = bytes_to_f32s(&body[4..]);
                if pos0 + rows.len() / dim > max_ctx {
                    write_frame(&mut w, ERROR, b"the context is full")?;
                    continue;
                }
                session.forward_range(m, lo, hi, pos0, &mut rows);
                write_frame(&mut w, ROWS, &f32s_to_bytes(&rows))?;
            }
            PING => write_frame(&mut w, PONG, &body)?,
            _ => write_frame(&mut w, ERROR, b"unexpected message")?,
        }
    }
}

fn announce(name: &str, port: u16, fp: &[u8; 32], stop: &AtomicBool) {
    let Ok(sock) = UdpSocket::bind("0.0.0.0:0") else { return };
    let _ = sock.set_broadcast(true);
    let line = format!("CEREBRO-WORKER v{VERSION} {} {port} {}", name.replace(' ', "_"), hex(fp));
    while !stop.load(Ordering::SeqCst) {
        let _ = sock.send_to(line.as_bytes(), ("255.255.255.255", DISCOVERY_PORT));
        std::thread::sleep(Duration::from_secs(2));
    }
}

/// A worker heard on the LAN.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Announced {
    pub name: String,
    pub addr: String,
    pub fingerprint: String,
}

/// Listens for worker announcements for `wait`, returning each worker once.
pub fn discover(wait: Duration) -> Vec<Announced> {
    let Ok(sock) = UdpSocket::bind(("0.0.0.0", DISCOVERY_PORT)) else { return Vec::new() };
    let _ = sock.set_read_timeout(Some(Duration::from_millis(250)));
    let end = Instant::now() + wait;
    let mut out: Vec<Announced> = Vec::new();
    let mut buf = [0u8; 512];
    while Instant::now() < end {
        if let Ok((n, from)) = sock.recv_from(&mut buf) {
            if let Some(a) = parse_announce(&String::from_utf8_lossy(&buf[..n]), from) {
                if !out.contains(&a) {
                    out.push(a);
                }
            }
        }
    }
    out
}

pub fn parse_announce(line: &str, from: SocketAddr) -> Option<Announced> {
    let mut it = line.split_whitespace();
    if it.next()? != "CEREBRO-WORKER" || it.next()? != format!("v{VERSION}") {
        return None;
    }
    let name = it.next()?.to_owned();
    let port: u16 = it.next()?.parse().ok()?;
    let fingerprint = it.next()?.to_owned();
    (fingerprint.len() == 64).then(|| Announced { name, addr: SocketAddr::new(from.ip(), port).to_string(), fingerprint })
}

/// A connection to a worker running layers lo..hi.
pub struct RemoteRange {
    pub lo: usize,
    pub hi: usize,
    pub addr: String,
    r: BufReader<TcpStream>,
    w: BufWriter<TcpStream>,
    dim: usize,
}

impl RemoteRange {
    /// Connects and proves the pairing key; the worker checks it has the same model file.
    pub fn connect(addr: &str, key: &[u8], fp: &[u8; 32], lo: usize, hi: usize, max_ctx: usize) -> Result<Self, Error> {
        let sa: SocketAddr = addr.parse().map_err(|_| Error::Request(format!("{addr} is not an address:port")))?;
        let s = TcpStream::connect_timeout(&sa, Duration::from_secs(5)).map_err(|e| Error::Request(format!("could not reach the worker at {addr}: {e}")))?;
        let _ = s.set_nodelay(true);
        let mut r = BufReader::new(s.try_clone().map_err(io)?);
        let mut w = BufWriter::new(s);
        let (kind, n) = read_frame(&mut r)?;
        if kind != CHALLENGE || n.len() != 16 {
            return Err(Error::Request("the worker did not start the handshake".into()));
        }
        let mut msg = n.clone();
        msg.extend_from_slice(fp);
        let mut hello = VERSION.to_le_bytes().to_vec();
        hello.extend_from_slice(fp);
        for v in [lo as u32, hi as u32, max_ctx as u32] {
            hello.extend_from_slice(&v.to_le_bytes());
        }
        hello.extend_from_slice(&hmac(key, &msg));
        write_frame(&mut w, HELLO, &hello)?;
        let (kind, body) = read_frame(&mut r)?;
        match kind {
            READY if body.len() == 4 => Ok(RemoteRange { lo, hi, addr: addr.to_owned(), r, w, dim: u32_at(&body, 0) as usize }),
            ERROR => Err(Error::Request(format!("the worker at {addr} refused: {}", String::from_utf8_lossy(&body)))),
            _ => Err(Error::Request("the worker answered the handshake with something else".into())),
        }
    }

    /// Hidden states after this range's layers, for the rows at pos0, pos0 + 1, ...
    pub fn forward(&mut self, pos0: usize, rows: &mut [f32]) -> Result<(), Error> {
        if rows.len() % self.dim != 0 {
            return Err(Error::Request("rows of the wrong width".into()));
        }
        let mut body = (pos0 as u32).to_le_bytes().to_vec();
        body.extend_from_slice(&f32s_to_bytes(rows));
        write_frame(&mut self.w, FORWARD, &body)?;
        let (kind, out) = read_frame(&mut self.r)?;
        match kind {
            ROWS if out.len() == rows.len() * 4 => {
                rows.copy_from_slice(&bytes_to_f32s(&out));
                Ok(())
            }
            ERROR => Err(Error::Request(format!("the worker at {}: {}", self.addr, String::from_utf8_lossy(&out)))),
            _ => Err(Error::Request("the worker sent an unexpected answer".into())),
        }
    }

    /// Round-trip time (ms) and bandwidth (MB/s, both ways) of the link.
    pub fn probe(&mut self) -> Result<(f64, f64), Error> {
        let mut rtts = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            write_frame(&mut self.w, PING, &[0u8; 8])?;
            read_frame(&mut self.r)?;
            rtts.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        rtts.sort_by(f64::total_cmp);
        let big = vec![0u8; 4 << 20];
        let t = Instant::now();
        write_frame(&mut self.w, PING, &big)?;
        read_frame(&mut self.r)?;
        let mbs = 2.0 * big.len() as f64 / 1e6 / t.elapsed().as_secs_f64().max(1e-6);
        Ok((rtts[rtts.len() / 2], mbs))
    }
}

/// The remote layer ranges of a session (sorted, not overlapping).
#[derive(Default)]
pub struct Remote {
    pub ranges: Vec<RemoteRange>,
}

impl Remote {
    /// The range starting at layer `li`, if any.
    pub fn starting_at(&mut self, li: usize) -> Option<&mut RemoteRange> {
        self.ranges.iter_mut().find(|r| r.lo == li)
    }
}

/// Picks the split: layers in proportion to each machine's share of the combined free memory
/// (this computer first, then the workers in order), at least one layer each. Returns, per worker,
/// its layer range.
pub fn plan(n_layer: usize, local_gib: f64, workers_gib: &[f64]) -> Vec<(usize, usize)> {
    let total: f64 = local_gib.max(0.0) + workers_gib.iter().map(|g| g.max(0.0)).sum::<f64>();
    if workers_gib.is_empty() || total <= 0.0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = ((local_gib.max(0.0) / total) * n_layer as f64).round() as usize;
    start = start.clamp(1, n_layer.saturating_sub(workers_gib.len()));
    let mut acc = local_gib.max(0.0);
    for (i, g) in workers_gib.iter().enumerate() {
        acc += g.max(0.0);
        let mut end = if i + 1 == workers_gib.len() { n_layer } else { ((acc / total) * n_layer as f64).round() as usize };
        end = end.clamp(start + 1, n_layer - (workers_gib.len() - 1 - i));
        out.push((start, end));
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_matches_rfc_4231_case_2() {
        assert_eq!(hex(&hmac(b"Jefe", b"what do ya want for nothing?")), "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    }

    #[test]
    fn frames_round_trip() {
        let mut buf = Vec::new();
        write_frame(&mut buf, FORWARD, &[1, 2, 3]).unwrap();
        let (k, p) = read_frame(&mut buf.as_slice()).unwrap();
        assert_eq!((k, p), (FORWARD, vec![1, 2, 3]));
        assert_eq!(bytes_to_f32s(&f32s_to_bytes(&[1.5, -2.25])), vec![1.5, -2.25]);
        let mut huge = ((MAX_FRAME + 2) as u32).to_le_bytes().to_vec();
        huge.push(1);
        assert!(read_frame(&mut huge.as_slice()).is_err());
    }

    #[test]
    fn announcements_parse() {
        let from: SocketAddr = "192.168.1.34:9999".parse().unwrap();
        let fp = "ab".repeat(32);
        let a = parse_announce(&format!("CEREBRO-WORKER v1 BOSGAME 50080 {fp}"), from).unwrap();
        assert_eq!(a.addr, "192.168.1.34:50080");
        assert_eq!(a.name, "BOSGAME");
        assert!(parse_announce("HELLO v1 x 1 y", from).is_none());
        assert!(parse_announce(&format!("CEREBRO-WORKER v2 x 50080 {fp}"), from).is_none());
    }

    #[test]
    fn the_plan_shares_layers_by_memory() {
        assert_eq!(plan(28, 8.0, &[8.0]), vec![(14, 28)]);
        assert_eq!(plan(40, 8.0, &[24.0]), vec![(10, 40)]);
        let p = plan(36, 4.0, &[8.0, 4.0]);
        assert_eq!(p, vec![(9, 27), (27, 36)]);
        // Every machine keeps at least one layer, ranges are contiguous and end at the last layer.
        let p = plan(4, 100.0, &[0.1, 0.1]);
        assert_eq!(p.first().unwrap().0, 2);
        assert_eq!(p.last().unwrap().1, 4);
    }

    /// Two workers on this computer run the middle and the end of Qwen3-0.6B; greedy tokens must be
    /// exactly those of the model run here alone. Prints the speed of both.
    /// `cargo test --release -p cerebro-engine --lib cluster::tests::split -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn split_over_two_workers_matches_local() {
        let path = std::path::PathBuf::from(std::env::var("CEREBRO_TEST_MODEL").unwrap_or_else(|_| r"C:\models\dev\Qwen3-0.6B-Q8_0.gguf".into()));
        let key = b"test pairing key".to_vec();
        let mk = || serve(WorkerConfig { model_path: path.clone(), key: key.clone(), addr: "127.0.0.1:0".parse().unwrap(), name: "test".into(), announce: false }).unwrap();
        let (w1, w2) = (mk(), mk());
        let m = Model::load(&path).unwrap();
        let fp = fingerprint(&path).unwrap();
        let tok = crate::tokenizer::Tokenizer::from_gguf(m.gguf()).unwrap();
        let prompt = tok.encode(&crate::chatml(None, &[("user", "Write three sentences about the moon.")], false));
        let n = m.cfg.n_layer;
        let greedy = |s: &mut Session| -> (Vec<u32>, f64) {
            s.feed(&m, &prompt).unwrap();
            let mut out = Vec::new();
            let t = Instant::now();
            for _ in 0..64 {
                let next = s.logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
                out.push(next);
                s.step(&m, next);
            }
            (out, 64.0 / t.elapsed().as_secs_f64())
        };
        let mut local = Session::new(&m, 2048);
        let (want, tps_local) = greedy(&mut local);
        let mut split = Session::new(&m, 2048);
        let a = RemoteRange::connect(&w1.addr.to_string(), &key, &fp, n / 3, 2 * n / 3, 2048).unwrap();
        let mut b = RemoteRange::connect(&w2.addr.to_string(), &key, &fp, 2 * n / 3, n, 2048).unwrap();
        let (rtt, mbs) = b.probe().unwrap();
        split.remote = Some(Remote { ranges: vec![a, b] });
        let (got, tps_split) = greedy(&mut split);
        println!("local {tps_local:.1} tok/s, split over two workers {tps_split:.1} tok/s; link {rtt:.3} ms, {mbs:.0} MB/s");
        assert_eq!(got, want, "the split must write exactly what the model writes alone");
        // A wrong pairing key is refused.
        let bad = RemoteRange::connect(&w1.addr.to_string(), b"other key", &fp, 0, 1, 64);
        assert!(bad.err().map(|e| e.to_string()).unwrap_or_default().contains("not paired"));
        w1.stop();
        w2.stop();
    }
}
