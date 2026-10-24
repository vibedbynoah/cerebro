//! Phone access: the owner checks on the local model from their phone over their own network.
//!
//! A small HTTPS server, off by default, that binds to one private LAN address of this computer (never
//! 0.0.0.0, never a public address, no port mapping). A phone gets in only by scanning a one-time QR
//! code shown on this screen AND typing the owner's password; it then holds a revocable session
//! cookie. Everything here exists so that nobody else on the network, and nobody on the internet,
//! can use this computer's GPU.
//!
//! Hand-written HTTP/1.1 on std threads keeps the surface small: GET and POST, bounded headers and
//! bodies, JSON only. TLS is rustls with a certificate generated on this machine.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::Argon2;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

// ---- What the app provides -------------------------------------------------------------------

/// The engine as the phone sees it. Implemented over the real engine in commands.rs and by a fake
/// in the tests.
pub trait Backend: Send + Sync + 'static {
    /// Live status for the phone, already shaped (see remote/app.js).
    fn status(&self) -> Value;
    /// Stops the reply being generated, if any.
    fn stop(&self);
    /// Runs one reply to `turns` (role, text), blocking, reporting through `sink`.
    fn chat(&self, turns: Vec<(String, String)>, sink: Box<dyn Fn(PhoneEvent) + Send + Sync>) -> Result<(), String>;
}

pub enum PhoneEvent {
    Token(String),
    Thinking(String),
    Done { tokens: u64, tps: f64 },
    Error(String),
}

// ---- Saved settings --------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub enabled: bool,
    /// Argon2id PHC string; the password itself is never stored.
    pub password_hash: Option<String>,
    /// Off by default: the phone sees status and can stop a reply, nothing more.
    pub allow_chat: bool,
    /// Also accept clients in 100.64.0.0/10 (the owner's Tailscale network).
    pub allow_tailscale: bool,
    /// The LAN address to bind; None = the address of the default route.
    pub address: Option<String>,
    /// 0 = pick a random high port once, then keep it so a bookmarked page keeps working.
    pub port: u16,
    /// Phones idle this long are signed out; 0 = never.
    pub session_days: u32,
    pub devices: Vec<Device>,
    pub cert_der_b64: Option<String>,
    pub key_der_b64: Option<String>,
    /// The addresses the certificate was made for; a new address means a new certificate.
    pub cert_ips: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Device {
    pub id: String,
    pub name: String,
    /// sha256 of the session token: a copy of the settings file cannot sign anyone in.
    pub token_sha256: String,
    pub created_at: u64,
    pub last_seen_at: u64,
    pub ip: String,
}

const DEFAULT_SESSION_DAYS: u32 = 30;
const PAIR_TTL: Duration = Duration::from_secs(10 * 60);
const MAX_PAIR_CODES: usize = 3;
const MAX_CONNECTIONS: usize = 16;
const MAX_HEAD: usize = 16 * 1024;
const MAX_BODY: usize = 64 * 1024;
const MAX_REQUESTS_PER_CONNECTION: usize = 1000;
const IO_TIMEOUT: Duration = Duration::from_secs(30);
const CHAT_KEEP: Duration = Duration::from_secs(5 * 60);
const MIN_PASSWORD: usize = 8;
const MAX_PASSWORD: usize = 256;

// ---- Small pure helpers (tested) -------------------------------------------------------------

pub(crate) fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

pub(crate) fn random_bytes<const N: usize>() -> Result<[u8; N], String> {
    let mut b = [0u8; N];
    getrandom::fill(&mut b).map_err(|_| "The system random number generator failed.".to_owned())?;
    Ok(b)
}

fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

/// 10/8, 172.16/12, 192.168/16, 169.254/16 link-local, loopback; 100.64/10 (CGNAT, used by
/// Tailscale) only when allowed. IPv6: unique-local fc00::/7 and link-local fe80::/10.
pub fn is_allowed_peer(ip: IpAddr, tailscale: bool) -> bool {
    match ip {
        IpAddr::V4(v4) => is_private_v4(v4) || v4.is_loopback() || (tailscale && is_cgnat(v4)),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_allowed_peer(IpAddr::V4(v4), tailscale);
            }
            let s = v6.segments()[0];
            (s & 0xfe00) == 0xfc00 || (s & 0xffc0) == 0xfe80 || v6.is_loopback()
        }
    }
}

fn is_private_v4(ip: Ipv4Addr) -> bool {
    ip.is_private() || ip.is_link_local()
}

fn is_cgnat(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    o[0] == 100 && (o[1] & 0xc0) == 64
}

/// Sign-in lockout after `failures` wrong attempts in a row: none for the first four, then 30 s
/// doubling per further failure, at most an hour.
pub fn lockout(failures: u32) -> Duration {
    if failures < 5 {
        return Duration::ZERO;
    }
    let exp = (failures - 5).min(7);
    Duration::from_secs((30u64 << exp).min(3600))
}

/// This computer's private IPv4 addresses that can serve the phone: the one on the default route
/// (the Wi-Fi or Ethernet a phone shares), plus the Tailscale address when allowed. No packets are
/// sent: connecting a UDP socket only asks the OS which local address it would use.
pub fn local_addresses(tailscale: bool) -> Vec<Ipv4Addr> {
    let probe = |target: &str| -> Option<Ipv4Addr> {
        let s = UdpSocket::bind("0.0.0.0:0").ok()?;
        s.connect(target).ok()?;
        match s.local_addr().ok()?.ip() {
            IpAddr::V4(v4) => Some(v4),
            IpAddr::V6(_) => None,
        }
    };
    let mut out = Vec::new();
    for (target, cgnat) in [("192.0.2.1:9", false), ("10.255.255.255:9", false), ("172.31.255.255:9", false), ("192.168.255.255:9", false), ("100.100.100.100:53", true)] {
        if let Some(ip) = probe(target) {
            let ok = if cgnat { tailscale && is_cgnat(ip) } else { is_private_v4(ip) };
            if ok && !out.contains(&ip) {
                out.push(ip);
            }
        }
    }
    out
}

pub fn hash_password(pw: &str) -> Result<String, String> {
    let salt = SaltString::encode_b64(&random_bytes::<16>()?).map_err(|_| "Could not make a salt.".to_owned())?;
    Argon2::default().hash_password(pw.as_bytes(), &salt).map(|h| h.to_string()).map_err(|_| "Could not hash the password.".to_owned())
}

/// A valid Argon2id hash of a random string, verified against when no password is set so that a
/// sign-in attempt costs the same either way.
fn dummy_hash() -> &'static str {
    static H: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    H.get_or_init(|| hash_password(&URL_SAFE_NO_PAD.encode(random_bytes::<16>().unwrap_or([7; 16]))).unwrap_or_default())
}

pub fn verify_password(pw: &str, phc: &str) -> bool {
    PasswordHash::new(phc).is_ok_and(|h| Argon2::default().verify_password(pw.as_bytes(), &h).is_ok())
}

/// Device names come from the phone: printable, short, no markup.
fn clean_name(raw: &str) -> String {
    let s: String = raw.chars().filter(|c| c.is_alphanumeric() || " -_.'()".contains(*c)).take(40).collect();
    let s = s.trim().to_owned();
    if s.is_empty() { "Phone".to_owned() } else { s }
}

/// One-time pairing codes, stored as hashes with their expiry.
#[derive(Default)]
struct PairCodes(Vec<(String, Instant)>);

impl PairCodes {
    fn add(&mut self, code: &str, now: Instant) {
        self.0.retain(|(_, exp)| *exp > now);
        while self.0.len() >= MAX_PAIR_CODES {
            self.0.remove(0);
        }
        self.0.push((sha256_hex(code.as_bytes()), now + PAIR_TTL));
    }
    fn valid(&self, code: &str, now: Instant) -> bool {
        let h = sha256_hex(code.as_bytes());
        self.0.iter().any(|(k, exp)| *exp > now && bool::from(k.as_bytes().ct_eq(h.as_bytes())))
    }
    fn consume(&mut self, code: &str, now: Instant) -> bool {
        let ok = self.valid(code, now);
        let h = sha256_hex(code.as_bytes());
        self.0.retain(|(k, exp)| *exp > now && *k != h);
        ok
    }
}

// ---- HTTP ------------------------------------------------------------------------------------

#[derive(Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub query: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
    fn cookie(&self, name: &str) -> Option<&str> {
        self.header("cookie")?.split(';').filter_map(|p| p.trim().split_once('=')).find(|(k, _)| *k == name).map(|(_, v)| v)
    }
    pub(crate) fn keep_alive(&self) -> bool {
        !self.header("connection").is_some_and(|c| c.eq_ignore_ascii_case("close"))
    }
}

#[derive(Debug, PartialEq)]
pub enum HttpError {
    /// The client closed the connection between requests.
    Closed,
    Bad(u16),
}

/// Reads one HTTP/1.1 request: at most MAX_HEAD bytes of request line and headers, a body only by
/// Content-Length and at most MAX_BODY. Anything else is refused with a status code.
pub fn read_request<R: BufRead>(r: &mut R) -> Result<Request, HttpError> {
    read_request_limited(r, MAX_BODY)
}

/// [`read_request`] with a caller-chosen body limit (the local API takes whole conversations).
pub fn read_request_limited<R: BufRead>(r: &mut R, max_body: usize) -> Result<Request, HttpError> {
    let mut head = 0usize;
    let mut line = Vec::new();
    let mut read_line = |r: &mut R, line: &mut Vec<u8>, first: bool| -> Result<String, HttpError> {
        line.clear();
        let n = r.by_ref().take((MAX_HEAD - head + 1) as u64).read_until(b'\n', line).map_err(|_| HttpError::Closed)?;
        if n == 0 {
            return Err(if first { HttpError::Closed } else { HttpError::Bad(400) });
        }
        head += n;
        if head > MAX_HEAD || !line.ends_with(b"\n") {
            return Err(HttpError::Bad(431));
        }
        let s = std::str::from_utf8(line).map_err(|_| HttpError::Bad(400))?;
        Ok(s.trim_end_matches(['\r', '\n']).to_owned())
    };
    let start = read_line(r, &mut line, true)?;
    let mut parts = start.split(' ');
    let (Some(method), Some(target), Some(version), None) = (parts.next(), parts.next(), parts.next(), parts.next()) else {
        return Err(HttpError::Bad(400));
    };
    if version != "HTTP/1.1" && version != "HTTP/1.0" {
        return Err(HttpError::Bad(505));
    }
    if method != "GET" && method != "POST" {
        return Err(HttpError::Bad(405));
    }
    if !target.starts_with('/') {
        return Err(HttpError::Bad(400));
    }
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let mut headers = Vec::new();
    loop {
        let l = read_line(r, &mut line, false)?;
        if l.is_empty() {
            break;
        }
        let (k, v) = l.split_once(':').ok_or(HttpError::Bad(400))?;
        if k.is_empty() || k.contains(' ') || headers.len() >= 64 {
            return Err(HttpError::Bad(400));
        }
        headers.push((k.to_owned(), v.trim().to_owned()));
    }
    let mut req = Request { method: method.to_owned(), path: path.to_owned(), query: query.to_owned(), headers, body: Vec::new() };
    if req.header("transfer-encoding").is_some() {
        return Err(HttpError::Bad(411));
    }
    let len = match req.header("content-length") {
        None => 0,
        Some(v) => v.parse::<usize>().map_err(|_| HttpError::Bad(400))?,
    };
    if len > max_body {
        return Err(HttpError::Bad(413));
    }
    req.body = vec![0; len];
    r.read_exact(&mut req.body).map_err(|_| HttpError::Bad(400))?;
    Ok(req)
}

pub struct Response {
    pub(crate) status: u16,
    pub(crate) content_type: &'static str,
    pub(crate) body: Vec<u8>,
    pub(crate) extra: Vec<(&'static str, String)>,
}

impl Response {
    pub(crate) fn json(status: u16, v: Value) -> Self {
        Response { status, content_type: "application/json", body: v.to_string().into_bytes(), extra: Vec::new() }
    }
    pub(crate) fn error(status: u16, msg: &str) -> Self {
        Self::json(status, json!({ "error": msg }))
    }
    fn file(content_type: &'static str, body: &'static str) -> Self {
        Response { status: 200, content_type, body: body.as_bytes().to_vec(), extra: Vec::new() }
    }
    pub(crate) fn with(mut self, k: &'static str, v: String) -> Self {
        self.extra.push((k, v));
        self
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Payload Too Large",
        421 => "Misdirected Request",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        _ => "Error",
    }
}

const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; manifest-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

pub(crate) fn write_response<W: Write + ?Sized>(w: &mut W, r: &Response, keep_alive: bool) -> std::io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nReferrer-Policy: no-referrer\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Resource-Policy: same-origin\r\nPermissions-Policy: camera=(), microphone=(), geolocation=()\r\nContent-Security-Policy: {CSP}\r\n",
        r.status,
        reason(r.status),
        r.content_type,
        r.body.len(),
        if keep_alive { "keep-alive" } else { "close" },
    );
    for (k, v) in &r.extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    w.write_all(head.as_bytes())?;
    w.write_all(&r.body)?;
    w.flush()
}

// ---- The service -----------------------------------------------------------------------------

#[derive(Default)]
struct ChatJob {
    text: String,
    thinking: String,
    done: bool,
    error: Option<String>,
    tokens: u64,
    tps: f64,
    finished_at: Option<Instant>,
}

struct Running {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    addr: SocketAddr,
}

#[derive(Default)]
struct Lockout {
    failures: u32,
    until: Option<Instant>,
}

pub struct Remote {
    settings: Mutex<Settings>,
    backend: Arc<dyn Backend>,
    save: Box<dyn Fn(&Settings) + Send + Sync>,
    running: Mutex<Option<Running>>,
    error: Mutex<Option<String>>,
    codes: Mutex<PairCodes>,
    lock: Mutex<Lockout>,
    /// Sign-ins are checked one at a time: Argon2 is deliberately expensive.
    login_gate: Mutex<()>,
    chats: Mutex<HashMap<String, Arc<Mutex<ChatJob>>>>,
    chat_busy: Arc<AtomicBool>,
    connections: Arc<AtomicUsize>,
    tls: Mutex<Option<Arc<rustls::ServerConfig>>>,
    last_seen_saved: Mutex<Option<Instant>>,
}

impl Remote {
    pub fn new(mut settings: Settings, backend: Arc<dyn Backend>, save: Box<dyn Fn(&Settings) + Send + Sync>) -> Arc<Self> {
        if settings.session_days == 0 && settings.devices.is_empty() && settings.password_hash.is_none() {
            settings.session_days = DEFAULT_SESSION_DAYS;
        }
        Arc::new(Remote {
            settings: Mutex::new(settings),
            backend,
            save,
            running: Mutex::new(None),
            error: Mutex::new(None),
            codes: Mutex::new(PairCodes::default()),
            lock: Mutex::new(Lockout::default()),
            login_gate: Mutex::new(()),
            chats: Mutex::new(HashMap::new()),
            chat_busy: Arc::new(AtomicBool::new(false)),
            connections: Arc::new(AtomicUsize::new(0)),
            tls: Mutex::new(None),
            last_seen_saved: Mutex::new(None),
        })
    }

    fn settings(&self) -> std::sync::MutexGuard<'_, Settings> {
        self.settings.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn persist(&self) {
        let snapshot = self.settings().clone();
        (self.save)(&snapshot);
    }

    fn set_error(&self, e: Option<String>) {
        *self.error.lock().unwrap_or_else(|e| e.into_inner()) = e;
    }

    /// Starts the server if the saved settings say it was on (called once at app start).
    pub fn resume(self: &Arc<Self>) {
        if self.settings().enabled {
            if let Err(e) = self.start() {
                self.set_error(Some(e));
            }
        }
    }

    /// Where phone access listens now (the public tunnel's origin).
    pub fn bound_addr(&self) -> Option<SocketAddr> {
        self.bound()
    }

    /// Whether a password is set (the public link requires one).
    pub fn has_password(&self) -> bool {
        self.settings().password_hash.is_some()
    }

    fn bound(&self) -> Option<SocketAddr> {
        self.running.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|r| r.addr)
    }

    fn url(&self) -> Option<String> {
        self.bound().map(|a| format!("https://{a}/"))
    }

    pub fn view(&self) -> Value {
        let s = self.settings().clone();
        let fingerprint = s.cert_der_b64.as_deref().and_then(|b| STANDARD.decode(b).ok()).map(|der| {
            Sha256::digest(&der).iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(":")
        });
        let running = self.bound();
        json!({
            "enabled": s.enabled,
            "running": running.is_some(),
            "url": self.url(),
            "address": running.map(|a| a.ip().to_string()).or(s.address.clone()),
            // The saved choice (None = automatic), distinct from where it is listening now.
            "addressChoice": s.address,
            "port": running.map_or(s.port, |a| a.port()),
            "addresses": local_addresses(s.allow_tailscale).iter().map(|ip| ip.to_string()).collect::<Vec<_>>(),
            "passwordSet": s.password_hash.is_some(),
            "allowChat": s.allow_chat,
            "allowTailscale": s.allow_tailscale,
            "sessionDays": s.session_days,
            "fingerprint": fingerprint,
            "devices": s.devices.iter().map(|d| json!({ "id": d.id, "name": d.name, "createdAt": d.created_at, "lastSeenAt": d.last_seen_at, "ip": d.ip })).collect::<Vec<_>>(),
            "error": self.error.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        })
    }

    /// Sets or changes the password. Every paired phone is signed out, since whoever knew the old
    /// password may have paired one.
    pub fn set_password(&self, pw: &str) -> Result<(), String> {
        let n = pw.chars().count();
        if n < MIN_PASSWORD {
            return Err(format!("Use at least {MIN_PASSWORD} characters."));
        }
        if n > MAX_PASSWORD {
            return Err("That password is too long.".into());
        }
        let hash = hash_password(pw)?;
        {
            let mut s = self.settings();
            s.password_hash = Some(hash);
            s.devices.clear();
        }
        *self.lock.lock().unwrap_or_else(|e| e.into_inner()) = Lockout::default();
        self.persist();
        Ok(())
    }

    pub fn set_enabled(self: &Arc<Self>, on: bool) -> Result<(), String> {
        if on && self.settings().password_hash.is_none() {
            return Err("Set a password first.".into());
        }
        self.settings().enabled = on;
        self.persist();
        self.stop_server();
        self.set_error(None);
        if on {
            if let Err(e) = self.start() {
                self.set_error(Some(e.clone()));
                return Err(e);
            }
        }
        Ok(())
    }

    pub fn set_options(self: &Arc<Self>, allow_chat: bool, allow_tailscale: bool, address: Option<String>, session_days: u32) -> Result<(), String> {
        if let Some(a) = &address {
            let ip: Ipv4Addr = a.parse().map_err(|_| "That is not an IPv4 address.".to_owned())?;
            if !(is_private_v4(ip) || (allow_tailscale && is_cgnat(ip))) {
                return Err("Phone access only listens on a private network address.".into());
            }
        }
        if session_days > 3650 {
            return Err("At most 3650 days.".into());
        }
        let restart = {
            let mut s = self.settings();
            let restart = s.address != address || s.allow_tailscale != allow_tailscale;
            s.allow_chat = allow_chat;
            s.allow_tailscale = allow_tailscale;
            s.address = address;
            s.session_days = session_days;
            restart && s.enabled
        };
        self.persist();
        if restart {
            self.stop_server();
            if let Err(e) = self.start() {
                self.set_error(Some(e.clone()));
                return Err(e);
            }
            self.set_error(None);
        }
        Ok(())
    }

    /// A new one-time code and its QR. Also lifts a sign-in lockout: the owner is at the computer.
    pub fn pair(&self) -> Result<Value, String> {
        let Some(addr) = self.bound() else { return Err("Turn phone access on first.".into()) };
        let code = URL_SAFE_NO_PAD.encode(random_bytes::<32>()?);
        self.codes.lock().unwrap_or_else(|e| e.into_inner()).add(&code, Instant::now());
        *self.lock.lock().unwrap_or_else(|e| e.into_inner()) = Lockout::default();
        // The code rides in the fragment: browsers never send it in the request line or Referer.
        let url = format!("https://{addr}/#pair={code}");
        let svg = qrcode::QrCode::with_error_correction_level(url.as_bytes(), qrcode::EcLevel::M)
            .map_err(|_| "Could not make the QR code.".to_owned())?
            .render::<qrcode::render::svg::Color>()
            .min_dimensions(240, 240)
            .dark_color(qrcode::render::svg::Color("#000000"))
            .light_color(qrcode::render::svg::Color("#ffffff"))
            .build();
        Ok(json!({ "url": url, "qrSvg": svg, "expiresAt": now_ms() + PAIR_TTL.as_millis() as u64 }))
    }

    pub fn revoke(&self, id: &str) {
        {
            let mut s = self.settings();
            if id == "*" {
                s.devices.clear();
            } else {
                s.devices.retain(|d| d.id != id);
            }
        }
        self.persist();
    }

    pub fn shutdown(&self) {
        self.stop_server();
    }

    fn stop_server(&self) {
        let r = self.running.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(mut r) = r {
            r.stop.store(true, Ordering::SeqCst);
            if let Some(t) = r.thread.take() {
                let _ = t.join();
            }
        }
    }

    fn tls_config(&self, ip: Ipv4Addr) -> Result<Arc<rustls::ServerConfig>, String> {
        let ip_s = ip.to_string();
        let (cert, key) = {
            let mut s = self.settings();
            let stale = !s.cert_ips.contains(&ip_s) || s.cert_der_b64.is_none() || s.key_der_b64.is_none();
            if stale {
                let mut names = vec!["cerebro.local".to_owned(), ip_s.clone()];
                names.extend(s.cert_ips.iter().filter(|x| **x != ip_s).take(4).cloned());
                let ck = rcgen::generate_simple_self_signed(names.clone()).map_err(|_| "Could not make the certificate.".to_owned())?;
                s.cert_der_b64 = Some(STANDARD.encode(ck.cert.der()));
                s.key_der_b64 = Some(STANDARD.encode(ck.key_pair.serialize_der()));
                s.cert_ips = names.into_iter().skip(1).collect();
            }
            let dec = |b: &Option<String>| b.as_deref().and_then(|b| STANDARD.decode(b).ok()).ok_or_else(|| "The saved certificate is damaged.".to_owned());
            (dec(&s.cert_der_b64)?, dec(&s.key_der_b64)?)
        };
        self.persist();
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let cfg = rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|_| "TLS setup failed.".to_owned())?
            .with_no_client_auth()
            .with_single_cert(vec![rustls::pki_types::CertificateDer::from(cert)], rustls::pki_types::PrivateKeyDer::Pkcs8(key.into()))
            .map_err(|_| "The certificate could not be used.".to_owned())?;
        Ok(Arc::new(cfg))
    }

    fn start(self: &Arc<Self>) -> Result<(), String> {
        let (address, tailscale, port) = {
            let s = self.settings();
            (s.address.clone(), s.allow_tailscale, s.port)
        };
        let candidates = local_addresses(tailscale);
        let ip = match address {
            Some(a) => {
                let ip: Ipv4Addr = a.parse().map_err(|_| "The saved address is not valid.".to_owned())?;
                if !candidates.contains(&ip) {
                    return Err(format!("This computer is not on {ip} right now. Pick another address under Options."));
                }
                ip
            }
            None => *candidates.first().ok_or_else(|| "No private network found. Connect to your Wi-Fi or Ethernet first.".to_owned())?,
        };
        let listener = if port != 0 {
            TcpListener::bind((ip, port)).map_err(|e| format!("Could not listen on {ip}:{port} ({e})."))?
        } else {
            let mut found = None;
            for _ in 0..32 {
                let p = 49152 + u16::from_le_bytes(random_bytes::<2>()?) % 16000;
                if let Ok(l) = TcpListener::bind((ip, p)) {
                    found = Some(l);
                    break;
                }
            }
            let l = found.ok_or_else(|| "Could not find a free port.".to_owned())?;
            if let Ok(a) = l.local_addr() {
                self.settings().port = a.port();
                self.persist();
            }
            l
        };
        listener.set_nonblocking(true).map_err(|_| "Could not start the server.".to_owned())?;
        let addr = listener.local_addr().map_err(|_| "Could not start the server.".to_owned())?;
        *self.tls.lock().unwrap_or_else(|e| e.into_inner()) = Some(self.tls_config(ip)?);
        let stop = Arc::new(AtomicBool::new(false));
        let me = Arc::clone(self);
        let st = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("cerebro-phone".into())
            .spawn(move || me.accept_loop(listener, st))
            .map_err(|_| "Could not start the server.".to_owned())?;
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = Some(Running { stop, thread: Some(thread), addr });
        Ok(())
    }

    fn accept_loop(self: Arc<Self>, listener: TcpListener, stop: Arc<AtomicBool>) {
        while !stop.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((stream, peer)) => {
                    let tailscale = self.settings().allow_tailscale;
                    if !is_allowed_peer(peer.ip(), tailscale) {
                        continue; // dropped: closes the socket without a byte
                    }
                    if self.connections.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
                        self.connections.fetch_sub(1, Ordering::SeqCst);
                        continue;
                    }
                    let me = Arc::clone(&self);
                    let st = Arc::clone(&stop);
                    let spawned = std::thread::Builder::new().name("cerebro-phone-conn".into()).spawn(move || {
                        me.serve_connection(stream, peer, &st);
                        me.connections.fetch_sub(1, Ordering::SeqCst);
                    });
                    if spawned.is_err() {
                        self.connections.fetch_sub(1, Ordering::SeqCst);
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(100)),
                Err(_) => std::thread::sleep(Duration::from_millis(100)),
            }
        }
    }

    fn serve_connection(&self, stream: TcpStream, peer: SocketAddr, stop: &AtomicBool) {
        let Some(tls) = self.tls.lock().unwrap_or_else(|e| e.into_inner()).clone() else { return };
        let _ = stream.set_nonblocking(false);
        let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
        let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
        let _ = stream.set_nodelay(true);
        let Ok(conn) = rustls::ServerConnection::new(tls) else { return };
        let mut io = BufReader::new(rustls::StreamOwned::new(conn, stream));
        let Some(host) = self.bound().map(|a| a.to_string()) else { return };
        for _ in 0..MAX_REQUESTS_PER_CONNECTION {
            if stop.load(Ordering::SeqCst) {
                return;
            }
            let req = match read_request(&mut io) {
                Ok(r) => r,
                Err(HttpError::Closed) => return,
                Err(HttpError::Bad(code)) => {
                    let _ = write_response(io.get_mut(), &Response::error(code, "Bad request"), false);
                    return;
                }
            };
            // Turned off while this connection waited: answer nothing, close.
            if stop.load(Ordering::SeqCst) || self.bound().is_none() {
                return;
            }
            let keep = req.keep_alive();
            // The Host must be this server's own address: a page elsewhere that re-points its name at
            // this computer (DNS rebinding) is refused before anything else runs.
            let resp = if req.header("host") != Some(host.as_str()) { Response::error(421, "Wrong host") } else { self.route(&req, peer) };
            if write_response(io.get_mut(), &resp, keep).is_err() || !keep {
                return;
            }
        }
    }

    /// The session's device id, refreshing its last-seen time. Expired devices are removed.
    fn session(&self, req: &Request, peer: SocketAddr) -> Option<(String, String)> {
        let token = req.cookie("cerebro_s")?;
        let raw = URL_SAFE_NO_PAD.decode(token).ok()?;
        if raw.len() != 32 {
            return None;
        }
        let h = sha256_hex(&raw);
        let now = now_ms();
        let (found, changed) = {
            let mut s = self.settings();
            let days = s.session_days;
            let before = s.devices.len();
            if days > 0 {
                let idle = u64::from(days) * 86_400_000;
                s.devices.retain(|d| now.saturating_sub(d.last_seen_at) < idle);
            }
            let removed = s.devices.len() != before;
            let found = s.devices.iter_mut().find(|d| bool::from(d.token_sha256.as_bytes().ct_eq(h.as_bytes()))).map(|d| {
                d.last_seen_at = now;
                d.ip = peer.ip().to_string();
                (d.id.clone(), d.name.clone())
            });
            (found, removed)
        };
        // Last-seen is saved at most once a minute; a removal is saved at once.
        let due = {
            let mut g = self.last_seen_saved.lock().unwrap_or_else(|e| e.into_inner());
            let due = changed || g.is_none_or(|t| t.elapsed() > Duration::from_secs(60));
            if due {
                *g = Some(Instant::now());
            }
            due
        };
        if due {
            self.persist();
        }
        found
    }

    fn route(&self, req: &Request, peer: SocketAddr) -> Response {
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/") => return Response::file("text/html; charset=utf-8", include_str!("remote/index.html")),
            ("GET", "/app.js") => return Response::file("text/javascript; charset=utf-8", include_str!("remote/app.js")),
            ("GET", "/app.css") => return Response::file("text/css; charset=utf-8", include_str!("remote/app.css")),
            ("GET", "/icon.svg") => return Response::file("image/svg+xml", include_str!("remote/icon.svg")),
            ("GET", "/manifest.webmanifest") => return Response::file("application/manifest+json", include_str!("remote/manifest.webmanifest")),
            _ => {}
        }
        if !req.path.starts_with("/api/") {
            return Response::error(404, "Not found");
        }
        // CSRF: a form or a cross-site fetch cannot set this header or a JSON content type without
        // a preflight this server never answers.
        if req.method == "POST" && (req.header("x-cerebro") != Some("1") || !req.header("content-type").is_some_and(|c| c.starts_with("application/json"))) {
            return Response::error(403, "Forbidden");
        }
        if req.method == "POST" && req.path == "/api/login" {
            return self.login(req, peer);
        }
        let Some((device_id, device_name)) = self.session(req, peer) else { return Response::error(401, "Sign in") };
        match (req.method.as_str(), req.path.as_str()) {
            ("POST", "/api/logout") => {
                self.revoke(&device_id);
                Response::json(200, json!({ "ok": true })).with("Set-Cookie", "cerebro_s=; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=0".into())
            }
            ("GET", "/api/status") => {
                let mut v = self.backend.status();
                if let Some(o) = v.as_object_mut() {
                    o.insert("chatAllowed".into(), json!(self.settings().allow_chat));
                    o.insert("device".into(), json!(device_name));
                }
                Response::json(200, v)
            }
            ("POST", "/api/stop") => {
                self.backend.stop();
                Response::json(200, json!({ "ok": true }))
            }
            ("POST", "/api/chat") => self.chat_start(req),
            ("GET", p) if p.starts_with("/api/chat/") => self.chat_poll(&p["/api/chat/".len()..], &req.query),
            _ => Response::error(404, "Not found"),
        }
    }

    fn login(&self, req: &Request, peer: SocketAddr) -> Response {
        #[derive(Deserialize)]
        struct Body {
            password: String,
            #[serde(default)]
            pair: Option<String>,
            #[serde(default)]
            device: Option<String>,
        }
        let locked_for = |l: &Lockout| l.until.map(|u| u.saturating_duration_since(Instant::now())).filter(|d| !d.is_zero());
        if let Some(d) = locked_for(&self.lock.lock().unwrap_or_else(|e| e.into_inner())) {
            return Response::json(429, json!({ "error": "Too many attempts. Wait, then try again.", "retryAfterS": d.as_secs().max(1) }));
        }
        let Ok(body) = serde_json::from_slice::<Body>(&req.body) else { return Response::error(400, "Bad request") };
        if body.password.len() > MAX_PASSWORD * 4 {
            return Response::error(400, "Bad request");
        }
        let _gate = self.login_gate.lock().unwrap_or_else(|e| e.into_inner());
        let hash = self.settings().password_hash.clone();
        // Always run Argon2, so a missing password or code costs the same time as a wrong one.
        let pw_ok = verify_password(&body.password, hash.as_deref().unwrap_or_else(|| dummy_hash())) && hash.is_some();
        let now = Instant::now();
        let code_ok = body.pair.as_deref().is_some_and(|c| c.len() <= 64 && self.codes.lock().unwrap_or_else(|e| e.into_inner()).valid(c, now));
        if !(pw_ok && code_ok && self.settings().enabled) {
            let mut l = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            l.failures = l.failures.saturating_add(1);
            let wait = lockout(l.failures);
            l.until = (!wait.is_zero()).then(|| now + wait);
            return Response::error(401, "Wrong password, or the QR code has expired. Scan a new one.");
        }
        // Single use: a code that raced another sign-in is refused here.
        if !body.pair.as_deref().is_some_and(|c| self.codes.lock().unwrap_or_else(|e| e.into_inner()).consume(c, now)) {
            return Response::error(401, "That QR code was already used. Scan a new one.");
        }
        *self.lock.lock().unwrap_or_else(|e| e.into_inner()) = Lockout::default();
        let (Ok(token), Ok(id)) = (random_bytes::<32>(), random_bytes::<9>()) else { return Response::error(500, "Try again") };
        let days = {
            let mut s = self.settings();
            let t = now_ms();
            s.devices.push(Device {
                id: URL_SAFE_NO_PAD.encode(id),
                name: clean_name(body.device.as_deref().unwrap_or("Phone")),
                token_sha256: sha256_hex(&token),
                created_at: t,
                last_seen_at: t,
                ip: peer.ip().to_string(),
            });
            while s.devices.len() > 32 {
                s.devices.remove(0);
            }
            s.session_days
        };
        self.persist();
        let max_age = if days == 0 { 400 * 86_400 } else { u64::from(days) * 86_400 };
        Response::json(200, json!({ "ok": true }))
            .with("Set-Cookie", format!("cerebro_s={}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age={max_age}", URL_SAFE_NO_PAD.encode(token)))
    }

    fn chat_start(&self, req: &Request) -> Response {
        #[derive(Deserialize)]
        struct Turn {
            role: String,
            text: String,
        }
        #[derive(Deserialize)]
        struct Body {
            turns: Vec<Turn>,
        }
        if !self.settings().allow_chat {
            return Response::error(403, "Chat from the phone is off. Turn it on in Settings on the computer.");
        }
        let Ok(body) = serde_json::from_slice::<Body>(&req.body) else { return Response::error(400, "Bad request") };
        let total: usize = body.turns.iter().map(|t| t.text.len()).sum();
        if body.turns.is_empty() || body.turns.len() > 64 || total > 48 * 1024 || body.turns.iter().any(|t| t.role != "user" && t.role != "assistant") {
            return Response::error(400, "Bad request");
        }
        if self.chat_busy.swap(true, Ordering::SeqCst) {
            return Response::error(409, "A reply is already being written.");
        }
        {
            let mut chats = self.chats.lock().unwrap_or_else(|e| e.into_inner());
            chats.retain(|_, j| j.lock().unwrap_or_else(|e| e.into_inner()).finished_at.is_none_or(|t| t.elapsed() < CHAT_KEEP));
        }
        let Ok(idb) = random_bytes::<12>() else {
            self.chat_busy.store(false, Ordering::SeqCst);
            return Response::error(500, "Try again");
        };
        let id = URL_SAFE_NO_PAD.encode(idb);
        let job = Arc::new(Mutex::new(ChatJob::default()));
        self.chats.lock().unwrap_or_else(|e| e.into_inner()).insert(id.clone(), Arc::clone(&job));
        let backend = Arc::clone(&self.backend);
        let busy = Arc::clone(&self.chat_busy);
        let turns: Vec<(String, String)> = body.turns.into_iter().map(|t| (t.role, t.text)).collect();
        let spawned = std::thread::Builder::new().name("cerebro-phone-chat".into()).spawn(move || {
            let j = Arc::clone(&job);
            let sink = Box::new(move |e: PhoneEvent| {
                let mut g = j.lock().unwrap_or_else(|e| e.into_inner());
                match e {
                    PhoneEvent::Token(t) => g.text.push_str(&t),
                    PhoneEvent::Thinking(t) => g.thinking.push_str(&t),
                    PhoneEvent::Done { tokens, tps } => {
                        g.tokens = tokens;
                        g.tps = tps;
                    }
                    PhoneEvent::Error(m) => g.error = Some(m),
                }
            });
            let r = backend.chat(turns, sink);
            let mut g = job.lock().unwrap_or_else(|e| e.into_inner());
            if let Err(e) = r {
                g.error.get_or_insert(e);
            }
            g.done = true;
            g.finished_at = Some(Instant::now());
            drop(g);
            busy.store(false, Ordering::SeqCst);
        });
        if spawned.is_err() {
            self.chat_busy.store(false, Ordering::SeqCst);
            return Response::error(500, "Try again");
        }
        Response::json(200, json!({ "id": id }))
    }

    fn chat_poll(&self, id: &str, query: &str) -> Response {
        let Some(job) = self.chats.lock().unwrap_or_else(|e| e.into_inner()).get(id).cloned() else { return Response::error(404, "Not found") };
        let from = query.split('&').find_map(|kv| kv.strip_prefix("from=")).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let g = job.lock().unwrap_or_else(|e| e.into_inner());
        let mut start = from.min(g.text.len());
        while !g.text.is_char_boundary(start) {
            start -= 1;
        }
        Response::json(
            200,
            json!({ "text": &g.text[start..], "thinking": g.thinking, "done": g.done, "error": g.error, "tokens": g.tokens, "tps": g.tps, "next": g.text.len() }),
        )
    }
}

impl Drop for Remote {
    fn drop(&mut self) {
        self.stop_server();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn req(raw: &str) -> Result<Request, HttpError> {
        read_request(&mut Cursor::new(raw.as_bytes().to_vec()))
    }

    #[test]
    fn parses_a_request_with_a_body() {
        let r = req("POST /api/login?x=1 HTTP/1.1\r\nHost: 192.168.1.5:50000\r\nContent-Length: 2\r\nCookie: a=1; cerebro_s=abc\r\n\r\n{}").unwrap();
        assert_eq!((r.method.as_str(), r.path.as_str(), r.query.as_str()), ("POST", "/api/login", "x=1"));
        assert_eq!(r.body, b"{}");
        assert_eq!(r.header("HOST"), Some("192.168.1.5:50000"));
        assert_eq!(r.cookie("cerebro_s"), Some("abc"));
    }

    #[test]
    fn refuses_bad_requests() {
        assert_eq!(req("").unwrap_err(), HttpError::Closed);
        assert_eq!(req("GET / HTTP/1.1\r\nNoColon\r\n\r\n").unwrap_err(), HttpError::Bad(400));
        assert_eq!(req("PUT / HTTP/1.1\r\n\r\n").unwrap_err(), HttpError::Bad(405));
        assert_eq!(req("GET / HTTP/2\r\n\r\n").unwrap_err(), HttpError::Bad(505));
        assert_eq!(req("GET http://x/ HTTP/1.1\r\n\r\n").unwrap_err(), HttpError::Bad(400));
        assert_eq!(req("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap_err(), HttpError::Bad(411));
        assert_eq!(req(&format!("POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n", MAX_BODY + 1)).unwrap_err(), HttpError::Bad(413));
        assert_eq!(req("POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\nshort").unwrap_err(), HttpError::Bad(400));
        let huge = format!("GET / HTTP/1.1\r\nX: {}\r\n\r\n", "a".repeat(MAX_HEAD));
        assert_eq!(req(&huge).unwrap_err(), HttpError::Bad(431));
    }

    #[test]
    fn only_private_peers_are_allowed() {
        let ip = |s: &str| s.parse::<IpAddr>().unwrap();
        for ok in ["10.0.0.2", "172.16.4.1", "172.31.255.1", "192.168.1.20", "169.254.3.3", "127.0.0.1", "fd00::1", "fe80::1", "::ffff:192.168.1.2"] {
            assert!(is_allowed_peer(ip(ok), false), "{ok}");
        }
        for no in ["8.8.8.8", "172.32.0.1", "100.64.0.1", "1.1.1.1", "2001:4860::1", "::ffff:8.8.8.8"] {
            assert!(!is_allowed_peer(ip(no), false), "{no}");
        }
        assert!(is_allowed_peer(ip("100.101.102.103"), true));
        assert!(!is_allowed_peer(ip("100.128.0.1"), true));
    }

    #[test]
    fn lockout_doubles_and_caps() {
        assert_eq!(lockout(4), Duration::ZERO);
        assert_eq!(lockout(5), Duration::from_secs(30));
        assert_eq!(lockout(6), Duration::from_secs(60));
        assert_eq!(lockout(11), Duration::from_secs(1920));
        assert_eq!(lockout(12), Duration::from_secs(3600));
        assert_eq!(lockout(1000), Duration::from_secs(3600));
    }

    #[test]
    fn pairing_codes_are_single_use_and_expire() {
        let t = Instant::now();
        let mut c = PairCodes::default();
        c.add("one", t);
        assert!(c.valid("one", t) && !c.valid("two", t));
        assert!(c.consume("one", t));
        assert!(!c.consume("one", t), "single use");
        c.add("late", t);
        assert!(!c.valid("late", t + PAIR_TTL + Duration::from_secs(1)), "expires");
        for k in ["a", "b", "c", "d"] {
            c.add(k, t);
        }
        assert!(!c.valid("a", t) && c.valid("d", t), "at most three outstanding");
    }

    #[test]
    fn password_round_trip() {
        let h = hash_password("correct horse").unwrap();
        assert!(h.starts_with("$argon2id$"));
        assert!(verify_password("correct horse", &h));
        assert!(!verify_password("wrong horse", &h));
        assert!(!verify_password("x", "not a hash"));
    }

    #[test]
    fn device_names_are_cleaned() {
        assert_eq!(clean_name("<script>Noah's iPhone</script>"), "scriptNoah's iPhonescript");
        assert_eq!(clean_name("   "), "Phone");
        assert_eq!(clean_name(&"x".repeat(100)).len(), 40);
    }

    struct Fake;
    impl Backend for Fake {
        fn status(&self) -> Value {
            json!({ "engine": { "state": "idle" } })
        }
        fn stop(&self) {}
        fn chat(&self, _: Vec<(String, String)>, sink: Box<dyn Fn(PhoneEvent) + Send + Sync>) -> Result<(), String> {
            sink(PhoneEvent::Token("héllo".into()));
            sink(PhoneEvent::Done { tokens: 1, tps: 9.0 });
            Ok(())
        }
    }

    fn post(path: &str, body: &str, cookie: Option<&str>) -> Request {
        let mut headers = vec![("X-Cerebro".to_owned(), "1".to_owned()), ("Content-Type".to_owned(), "application/json".to_owned())];
        if let Some(c) = cookie {
            headers.push(("Cookie".into(), c.into()));
        }
        Request { method: "POST".into(), path: path.into(), query: String::new(), headers, body: body.as_bytes().to_vec() }
    }

    /// The whole sign-in path without sockets: no code, wrong password, then a good pairing.
    #[test]
    fn sign_in_needs_password_and_code() {
        let remote = Remote::new(Settings::default(), Arc::new(Fake), Box::new(|_| {}));
        remote.set_password("correct horse").unwrap();
        remote.settings().enabled = true;
        let peer: SocketAddr = "192.168.1.9:5000".parse().unwrap();
        let t = Instant::now();
        remote.codes.lock().unwrap().add("CODE", t);

        let r = remote.route(&post("/api/login", r#"{"password":"correct horse"}"#, None), peer);
        assert_eq!(r.status, 401, "password alone is not enough");
        let r = remote.route(&post("/api/login", r#"{"password":"nope nope","pair":"CODE"}"#, None), peer);
        assert_eq!(r.status, 401, "code alone is not enough");
        let mut no_header = post("/api/login", r#"{"password":"correct horse","pair":"CODE"}"#, None);
        no_header.headers.remove(0);
        assert_eq!(remote.route(&no_header, peer).status, 403, "CSRF header required");

        let r = remote.route(&post("/api/login", r#"{"password":"correct horse","pair":"CODE","device":"Pixel"}"#, None), peer);
        assert_eq!(r.status, 200);
        let cookie = r.extra.iter().find(|(k, _)| *k == "Set-Cookie").map(|(_, v)| v.split(';').next().unwrap().to_owned()).unwrap();
        assert!(remote.settings().devices.iter().all(|d| !cookie.contains(&d.token_sha256)), "only the hash is stored");

        let again = remote.route(&post("/api/login", r#"{"password":"correct horse","pair":"CODE"}"#, None), peer);
        assert_eq!(again.status, 401, "the code was used up");

        let status = Request { method: "GET".into(), path: "/api/status".into(), query: String::new(), headers: vec![("Cookie".into(), cookie.clone())], body: vec![] };
        let r = remote.route(&status, peer);
        assert_eq!(r.status, 200);
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["device"], "Pixel");
        assert_eq!(v["chatAllowed"], false);

        assert_eq!(remote.route(&post("/api/chat", r#"{"turns":[{"role":"user","text":"hi"}]}"#, Some(&cookie)), peer).status, 403, "chat is off by default");

        remote.settings().allow_chat = true;
        let r = remote.route(&post("/api/chat", r#"{"turns":[{"role":"user","text":"hi"}]}"#, Some(&cookie)), peer);
        assert_eq!(r.status, 200);
        let id = serde_json::from_slice::<Value>(&r.body).unwrap()["id"].as_str().unwrap().to_owned();
        let mut v = Value::Null;
        for _ in 0..100 {
            let poll = Request { method: "GET".into(), path: format!("/api/chat/{id}"), query: "from=2".into(), headers: vec![("Cookie".into(), cookie.clone())], body: vec![] };
            v = serde_json::from_slice(&remote.route(&poll, peer).body).unwrap();
            if v["done"] == true {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(v["done"], true);
        assert_eq!(v["text"], "éllo", "a byte offset inside a character backs up to its start");

        remote.set_password("another password").unwrap();
        assert_eq!(remote.route(&status, peer).status, 401, "a new password signs every phone out");
    }

    /// The real server over real TLS on this computer's LAN address: page, security headers, a
    /// wrong Host, sign-in with the QR code, status with the cookie, and shutdown. Skipped on a
    /// machine with no private network.
    #[test]
    fn https_end_to_end_on_the_lan() {
        if local_addresses(false).is_empty() {
            eprintln!("no private network; skipped");
            return;
        }
        let remote = Remote::new(Settings::default(), Arc::new(Fake), Box::new(|_| {}));
        remote.set_password("correct horse").unwrap();
        remote.set_enabled(true).unwrap();
        let pairing = remote.pair().unwrap();
        let url = pairing["url"].as_str().unwrap().to_owned();
        let (base, code) = url.split_once("#pair=").unwrap();
        assert!(pairing["qrSvg"].as_str().unwrap().starts_with("<?xml") || pairing["qrSvg"].as_str().unwrap().contains("<svg"));
        let client = reqwest::blocking::Client::builder().tls_danger_accept_invalid_certs(true).build().unwrap();

        let page = client.get(base).send().unwrap();
        assert_eq!(page.status(), 200);
        assert!(page.headers()["content-security-policy"].to_str().unwrap().contains("script-src 'self'"));
        assert_eq!(page.headers()["x-frame-options"], "DENY");
        assert!(page.text().unwrap().contains("/app.js"));

        let addr = remote.bound().unwrap();
        let wrong_host = client.get(base).header("Host", format!("evil.example:{}", addr.port())).send().unwrap();
        assert_eq!(wrong_host.status(), 421, "DNS rebinding is refused");

        let login = |body: Value| client.post(format!("{base}api/login")).header("X-Cerebro", "1").json(&body).send().unwrap();
        assert_eq!(login(json!({ "password": "correct horse" })).status(), 401);
        let ok = login(json!({ "password": "correct horse", "pair": code, "device": "Test phone" }));
        assert_eq!(ok.status(), 200);
        let cookie = ok.headers()["set-cookie"].to_str().unwrap().to_owned();
        assert!(cookie.contains("HttpOnly") && cookie.contains("Secure") && cookie.contains("SameSite=Strict"));
        let cookie = cookie.split(';').next().unwrap().to_owned();

        let status = client.get(format!("{base}api/status")).header("Cookie", &cookie).send().unwrap();
        assert_eq!(status.status(), 200);
        assert_eq!(status.json::<Value>().unwrap()["device"], "Test phone");
        assert_eq!(client.get(format!("{base}api/status")).send().unwrap().status(), 401, "no cookie, no status");

        remote.set_enabled(false).unwrap();
        assert!(client.get(base).send().is_err(), "off means nothing is listening");
    }

    /// A live server for the phone-page browser check (scripts/phone-check.mjs): writes the pairing
    /// URL to $PHONE_URL_FILE, serves a fake engine with chat on for $PHONE_SERVE_S seconds.
    /// `cargo test -p cerebro-desktop serve_for_browser_check -- --ignored`
    #[test]
    #[ignore]
    fn serve_for_browser_check() {
        struct Demo;
        impl Backend for Demo {
            fn status(&self) -> Value {
                json!({
                    "at": now_ms(), "engine": { "state": "generating" },
                    "model": { "name": "Qwen3 0.6B", "quant": "Q8_0", "contextUsed": 1834, "contextMax": 32768 },
                    "decodeTps": 241.7, "prefillTps": 23215.0,
                    "gpu": { "name": "RTX 3070 Ti Laptop", "utilPct": 87.0, "tempC": 71.0, "powerW": 96.0, "vramUsedGiB": 3.4, "vramTotalGiB": 8.0 },
                    "cpuPct": 12.0, "ramUsedGiB": 14.2, "ramTotalGiB": 31.7,
                    "requests": [{ "id": "req-12", "state": "generating", "startedAt": now_ms() - 4000, "outputTokens": 512, "decodeTps": 241.7 }],
                })
            }
            fn stop(&self) {}
            fn chat(&self, _: Vec<(String, String)>, sink: Box<dyn Fn(PhoneEvent) + Send + Sync>) -> Result<(), String> {
                for w in "Your model is running on the GPU at about 240 tokens a second.".split_inclusive(' ') {
                    sink(PhoneEvent::Token(w.to_owned()));
                    std::thread::sleep(Duration::from_millis(40));
                }
                sink(PhoneEvent::Done { tokens: 14, tps: 240.0 });
                Ok(())
            }
        }
        let remote = Remote::new(Settings::default(), Arc::new(Demo), Box::new(|_| {}));
        remote.set_password("correct horse").unwrap();
        remote.set_options(true, false, None, 30).unwrap();
        remote.set_enabled(true).unwrap();
        let mut urls = Vec::new();
        for _ in 0..3 {
            urls.push(remote.pair().unwrap()["url"].as_str().unwrap().to_owned());
        }
        std::fs::write(std::env::var("PHONE_URL_FILE").unwrap_or("phone-url.txt".into()), urls.join("\n")).unwrap();
        let secs = std::env::var("PHONE_SERVE_S").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
        std::thread::sleep(Duration::from_secs(secs));
    }

    #[test]
    fn lockout_after_repeated_failures() {
        let remote = Remote::new(Settings::default(), Arc::new(Fake), Box::new(|_| {}));
        remote.set_password("correct horse").unwrap();
        let peer: SocketAddr = "192.168.1.9:5000".parse().unwrap();
        for _ in 0..5 {
            remote.route(&post("/api/login", r#"{"password":"bad","pair":"x"}"#, None), peer);
        }
        assert_eq!(remote.route(&post("/api/login", r#"{"password":"bad","pair":"x"}"#, None), peer).status, 429);
    }
}
