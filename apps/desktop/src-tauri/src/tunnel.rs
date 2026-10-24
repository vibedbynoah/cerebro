//! Sharing phone access over the internet: a Cloudflare quick tunnel to the phone-access server.
//!
//! Off until the owner asks for a public link. `cloudflared` runs on this computer and connects to
//! the phone-access address (a private LAN address, so the server's peer check holds), with the
//! Host header set to that address (so its host check holds too). Whoever opens the public link
//! still needs a one-time pairing code and the owner's password, with the server's lockout after
//! wrong passwords; nothing is reachable without both. The link is new each time and ends when
//! sharing stops or the app closes.
//!
//! `cloudflared` is the official release from Cloudflare's GitHub, pinned to one version and
//! checked against the SHA-256 Cloudflare published for it before it is ever run.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

/// cloudflared 2026.10.3, Windows x64: (URL, SHA-256, bytes), from the release notes and the
/// asset's own digest (they agree).
const RELEASE: (&str, &str, u64) = (
    "https://github.com/cloudflare/cloudflared/releases/download/2026.10.3/cloudflared-windows-amd64.exe",
    "76b2805253d0d6ffacb092a7acfbee99e7af6917835c0c5410e23f3ef5052da2",
    55_329_336,
);

/// The running tunnel and its public address.
#[derive(Default)]
pub struct Tunnel {
    child: Mutex<Option<Child>>,
    url: Mutex<Option<String>>,
}

impl Drop for Tunnel {
    fn drop(&mut self) {
        self.stop();
    }
}

fn exe_path() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("Cerebro").join("cloudflared").join("cloudflared-2026.10.3.exe"))
}

/// The checked cloudflared, downloaded on first use.
fn ensure_cloudflared() -> Result<PathBuf, String> {
    let path = exe_path().ok_or("There is no local app data folder.")?;
    let ok = |p: &PathBuf| std::fs::read(p).is_ok_and(|b| b.len() as u64 == RELEASE.2 && hex(&Sha256::digest(&b)) == RELEASE.1);
    if path.is_file() && ok(&path) {
        return Ok(path);
    }
    if crate::llama_backend::offline() {
        return Err("Offline mode is on, so the tunnel program was not downloaded.".into());
    }
    std::fs::create_dir_all(path.parent().expect("has a parent")).map_err(|e| e.to_string())?;
    let mut r = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .map_err(|e| e.to_string())?
        .get(RELEASE.0)
        .send()
        .map_err(|e| format!("Could not download cloudflared: {e}"))?;
    if !r.status().is_success() {
        return Err(format!("Could not download cloudflared: HTTP {}", r.status()));
    }
    let mut bytes = Vec::with_capacity(RELEASE.2 as usize);
    r.read_to_end(&mut bytes).map_err(|e| format!("The download of cloudflared broke off: {e}"))?;
    if bytes.len() as u64 != RELEASE.2 || hex(&Sha256::digest(&bytes)) != RELEASE.1 {
        return Err("cloudflared did not match the checksum Cloudflare published; it was not saved.".into());
    }
    let tmp = path.with_extension("part");
    std::fs::File::create(&tmp).and_then(|mut f| f.write_all(&bytes)).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    Ok(path)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The quick-tunnel address in one line of cloudflared's log, if it has one.
pub fn url_in(line: &str) -> Option<String> {
    let i = line.find("https://")?;
    let rest = &line[i..];
    let end = rest.find(|c: char| c.is_whitespace() || c == '|' || c == '"').unwrap_or(rest.len());
    let url = &rest[..end];
    url.ends_with(".trycloudflare.com").then(|| url.to_owned())
}

impl Tunnel {
    /// The public address while sharing.
    pub fn url(&self) -> Option<String> {
        self.url.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Starts a tunnel to `origin` (the phone-access server) and waits for its public address.
    pub fn start(&self, origin: std::net::SocketAddr) -> Result<String, String> {
        self.stop();
        let exe = ensure_cloudflared()?;
        let mut cmd = Command::new(exe);
        cmd.args(["tunnel", "--no-autoupdate", "--url", &format!("https://{origin}"), "--no-tls-verify", "--http-host-header", &origin.to_string()]);
        cmd.stdout(Stdio::null()).stderr(Stdio::piped()).stdin(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000);
        }
        let mut child = cmd.spawn().map_err(|e| format!("Could not start cloudflared: {e}"))?;
        let err = child.stderr.take().ok_or("cloudflared has no log.")?;
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                if let Some(u) = url_in(&line) {
                    let _ = tx.send(u);
                }
            }
        });
        let t0 = Instant::now();
        let url = loop {
            if let Ok(u) = rx.recv_timeout(Duration::from_millis(250)) {
                break u;
            }
            if let Ok(Some(status)) = child.try_wait() {
                return Err(format!("cloudflared stopped before the link was ready ({status})."));
            }
            if t0.elapsed() > Duration::from_secs(60) {
                let _ = child.kill();
                return Err("Cloudflare did not give a link within a minute. Check the connection and try again.".into());
            }
        };
        *self.child.lock().unwrap_or_else(|e| e.into_inner()) = Some(child);
        *self.url.lock().unwrap_or_else(|e| e.into_inner()) = Some(url.clone());
        Ok(url)
    }

    pub fn stop(&self) {
        if let Some(mut c) = self.child.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        *self.url.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_quick_tunnel_address_in_the_log() {
        let line = "2026-10-24T15:00:00Z INF |  https://bright-river-1234.trycloudflare.com                                |";
        assert_eq!(url_in(line).as_deref(), Some("https://bright-river-1234.trycloudflare.com"));
        assert_eq!(url_in("INF Requesting new quick Tunnel on trycloudflare.com..."), None);
        assert_eq!(url_in("see https://developers.cloudflare.com/ for help"), None);
    }
}
