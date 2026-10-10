//! End to end with a real llama-server: `serve()` launches the official build with the tiny model
//! through preflight and the supervisor, and the gateway answers in front of it.
//!
//! Machine-dependent: needs an official llama.cpp build (`CEREBRO_LLAMACPP_OFFICIAL`, for example
//! `C:\llama-cuda`) and the tiny GGUF (`CEREBRO_TINY_GGUF`). Each test loads the 0.5B model once,
//! opens no window, and checks the server process is gone afterwards.

#![cfg(windows)]

use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use cerebro_config::{KeyFile, LaunchRequest, Secret, SecretStore, Settings, Source};
use cerebro_server::{effort, serve, LoadFlags, ServeRequest, Serving};
use cerebro_supervisor::{preflight, PortRequest, Protection};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpStream;

struct Reply {
    status: u16,
    body: String,
}

impl Reply {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|_| panic!("not JSON ({}): {}", self.status, self.body))
    }
}

async fn send(addr: SocketAddr, method: &str, path: &str, bearer: Option<&str>, body: Option<&Value>) -> Reply {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let body = body.map(Value::to_string).unwrap_or_default();
    let auth = bearer.map(|t| format!("Authorization: Bearer {t}\r\n")).unwrap_or_default();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        addr.port(),
        body.len()
    );
    stream.write_all(head.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    tokio::time::timeout(Duration::from_secs(120), stream.read_to_end(&mut raw)).await.expect("reply within 120 s").unwrap();
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, rest) = text.split_once("\r\n\r\n").expect("header end");
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") { dechunk(rest) } else { rest.to_string() };
    Reply { status, body }
}

fn dechunk(mut data: &str) -> String {
    let mut out = String::new();
    while let Some((size, rest)) = data.split_once("\r\n") {
        let n = usize::from_str_radix(size.trim(), 16).unwrap_or(0);
        if n == 0 || rest.len() < n {
            break;
        }
        out.push_str(&rest[..n]);
        data = rest[n..].trim_start_matches("\r\n");
    }
    out
}

/// Turn off, in `settings`, each machine-state protection a real preflight refuses on (an
/// elevated shell; a build under `C:\` inheriting `Authenticated Users: Modify`), as a user would.
/// Returns the setting keys turned off.
fn settle_machine_protections(settings: &mut Settings, model: &Path, manifest: &std::collections::BTreeMap<String, String>) -> Vec<&'static str> {
    let keys = tempfile::tempdir().unwrap();
    let key = KeyFile::create(keys.path(), &Secret::new(b"0123456789abcdef".repeat(4))).unwrap();
    let mut off = Vec::new();
    loop {
        let spec = settings.launch_spec(LaunchRequest {
            model_path: model.to_path_buf(),
            draft_model_path: None,
            manifest: manifest.clone(),
            port: PortRequest::Any,
            api_key_file: key.path().to_path_buf(),
        });
        match preflight(&spec) {
            Ok(_) => return off,
            Err(err) => {
                let key = match err.governing_protection() {
                    Some(Protection::ElevationRefusal) => "refuse_elevation",
                    Some(Protection::FolderPermissionAudit) => "audit_folder_permissions",
                    _ => panic!("an intact build must pass or name a machine-state setting: {err}"),
                };
                assert!(!off.contains(&key), "{key} refused again after it was turned off: {err}");
                settings.apply_patch(&json!({"protections": {key: false}}), Source::Gui).unwrap();
                off.push(key);
            }
        }
    }
}

/// `serve()` the tiny model on the official build with the machine-state protections this machine
/// refuses on turned off, as a user would. Returns the server, the outcome of waiting for it, the
/// tail of llama-server's log when it never became ready, the data directory and the settings
/// turned off.
async fn serve_tiny() -> (Serving, Result<(), cerebro_supervisor::LifecycleState>, Option<String>, tempfile::TempDir, Vec<&'static str>) {
    let backend = cerebro_probe::fixtures::official_backend_dir().unwrap_or_else(|e| panic!("{e}"));
    let model = cerebro_probe::fixtures::tiny_gguf().unwrap_or_else(|e| panic!("{e}"));
    let (manifest, _) = cerebro_probe::backend::hash_directory(&backend).unwrap();

    let mut settings = Settings::default();
    let folder = model.parent().unwrap().to_path_buf();
    settings.apply_patch(&json!({"backend": {"dir": backend}, "models": {"folders": [folder]}}), Source::Gui).unwrap();
    let turned_off = settle_machine_protections(&mut settings, &model, &manifest);

    let data = tempfile::tempdir().unwrap();
    let mut request = ServeRequest::new(settings, model.clone(), manifest, LoadFlags::Settings, data.path().to_path_buf());
    request.gateway.port = 0;
    request.readiness_deadline = Duration::from_secs(180);
    let serving = serve(request).await.unwrap();
    let ready = serving.wait_ready(Duration::from_secs(200)).await;
    let log_tail = ready.is_err().then(|| {
        let log = std::fs::read(data.path().join("logs").join("llama-server.log")).unwrap_or_default();
        String::from_utf8_lossy(&log[log.len().saturating_sub(4096)..]).into_owned()
    });
    (serving, ready, log_tail, data, turned_off)
}

/// Whether a llama-server process with `pid` is running.
fn llama_server_running(pid: u64) -> bool {
    let still = cerebro_probe::process::running_with_image_names(&["llama-server.exe"]).unwrap();
    still.iter().any(|p| u64::from(p.pid) == pid)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs an official llama.cpp build (CEREBRO_LLAMACPP_OFFICIAL, e.g. C:\\llama-cuda) and the tiny \
            GGUF (CEREBRO_TINY_GGUF); loads the 0.5B model once"]
async fn serve_launches_the_tiny_model_and_the_gateway_answers_end_to_end() {
    let (mut serving, ready, log_tail, data, turned_off) = serve_tiny().await;
    let addr = serving.addr();
    let token = serving.token().reveal().to_string();
    let keys_dir = data.path().join("keys");

    // Everything below runs inside this block so the server is shut down whatever it asserts.
    let outcome = tokio::spawn(async move {
        ready.unwrap_or_else(|state| panic!("the server did not become ready: {state:?}\nllama-server log tail:\n{}", log_tail.unwrap_or_default()));

        assert_eq!(send(addr, "GET", "/v1/models", None, None).await.status, 401);
        let wrong = "0".repeat(64);
        assert_eq!(send(addr, "GET", "/cerebro/status", Some(&wrong), None).await.status, 401);

        let models = send(addr, "GET", "/v1/models", Some(&token), None).await;
        assert_eq!(models.status, 200, "{}", models.body);
        assert!(models.json()["data"].as_array().is_some_and(|d| !d.is_empty()), "{}", models.body);

        let low = json!({"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16, "temperature": 0, "cerebro_effort": "low"});
        let reply = send(addr, "POST", "/v1/chat/completions", Some(&token), Some(&low)).await;
        assert_eq!(reply.status, 200, "{}", reply.body);
        let v = reply.json();
        assert!(v["choices"][0]["message"]["content"].as_str().is_some_and(|c| !c.trim().is_empty()), "{}", reply.body);
        assert_eq!(v["cerebro"]["effort"], "low");
        assert_eq!(v["cerebro"]["verbosity"], "brief");
        assert_eq!(v["cerebro"]["max_tokens"], 16, "Low thinks 0 tokens, so the answer budget is the whole limit");
        let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap();
        assert!(prompt_tokens > 40, "the effort and verbosity instructions reached the model ({prompt_tokens} prompt tokens)");

        let mut stream_body = low.clone();
        stream_body["stream"] = json!(true);
        let streamed = send(addr, "POST", "/v1/chat/completions", Some(&token), Some(&stream_body)).await;
        assert_eq!(streamed.status, 200);
        assert!(streamed.body.contains("data: ") && streamed.body.contains("[DONE]"), "{}", streamed.body);
        // The gateway adds one chunk carrying the cerebro field just before [DONE], with the
        // stream's own decode speed.
        let report_at = streamed.body.find("\"cerebro\"").unwrap_or_else(|| panic!("no cerebro chunk: {}", streamed.body));
        assert!(report_at < streamed.body.find("data: [DONE]").unwrap(), "{}", streamed.body);
        let report_line = streamed.body.lines().find(|l| l.contains("\"cerebro\"")).unwrap();
        let report: Value = serde_json::from_str(report_line.trim_start_matches("data: ")).unwrap();
        assert_eq!(report["cerebro"]["effort"], "low");
        assert!(report["cerebro"]["passes"][0]["tokens_per_second"].as_f64().is_some_and(|t| t > 0.0), "{report}");

        let xhigh = json!({"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 24, "temperature": 0, "cerebro_effort": "xhigh"});
        let reviewed = send(addr, "POST", "/v1/chat/completions", Some(&token), Some(&xhigh)).await;
        assert_eq!(reviewed.status, 200, "{}", reviewed.body);
        let r = reviewed.json();
        let passes: Vec<&str> = r["cerebro"]["passes"].as_array().unwrap().iter().map(|p| p["pass"].as_str().unwrap()).collect();
        assert_eq!(passes, ["draft", "self_review"], "{}", reviewed.body);
        // This model's template has no thinking mode, so nothing is reserved for thinking and the
        // client's 24 tokens stay the whole limit.
        assert_eq!(r["cerebro"]["thinking_supported"], false, "{}", reviewed.body);
        assert_eq!(r["cerebro"]["max_tokens"], 24, "{}", reviewed.body);
        assert!(r["usage"]["completion_tokens"].as_u64().is_some_and(|n| n <= 48), "two passes of at most 24 tokens: {}", reviewed.body);
        println!("xhigh cerebro field: {}", r["cerebro"]);

        let completion = send(addr, "POST", "/v1/completions", Some(&token), Some(&json!({"prompt": "One, two,", "max_tokens": 4}))).await;
        assert_eq!(completion.status, 200, "{}", completion.body);

        // llama-server was not started with --embeddings; its own answer is relayed unchanged.
        let embeddings = send(addr, "POST", "/v1/embeddings", Some(&token), Some(&json!({"input": "x"}))).await;
        println!("embeddings passthrough: {} {}", embeddings.status, embeddings.body);
        assert_ne!(embeddings.status, 401);
        assert!(embeddings.json().get("error").is_some() || embeddings.status == 200);

        let health = send(addr, "GET", "/cerebro/health", Some(&token), None).await;
        assert_eq!(health.status, 200);
        let status = send(addr, "GET", "/cerebro/status", Some(&token), None).await.json();
        println!("status: {status}");
        assert!(status["speed"]["decode_tokens_per_second"].as_f64().is_some_and(|t| t > 0.0), "{status}");
        assert!(status["model"]["n_ctx"].as_u64().is_some_and(|n| n > 0), "{status}");
        assert!(status["vram"].is_array(), "{status}");
        assert_eq!(status["launch"]["load"]["source"], "settings");
        let disabled: Vec<&str> = status["launch"]["disabled_protections"].as_array().unwrap().iter().map(|d| d["setting"].as_str().unwrap()).collect();
        assert_eq!(status["gateway"]["disabled_protections"], json!([]), "every gateway check is on by default");

        // Gateway overhead, measured rather than assumed. 20 small requests each through the
        // gateway and straight to llama-server (with its per-launch key, read from the key file).
        // The direct request carries exactly the body the gateway forwards (instructions injected,
        // budget applied, built with the crate's own public functions), so llama-server does the
        // same work on both paths and only the gateway differs; which path goes first alternates,
        // so a prompt-cache hit on the second of an identical pair favours neither. Each request's
        // wall time minus llama-server's own prompt + decode time is HTTP and request handling; the
        // gateway's cost is the difference of the two medians.
        let upstream = SocketAddr::from(([127, 0, 0, 1], status["launch"]["upstream_port"].as_u64().unwrap() as u16));
        let upstream_key = std::fs::read_dir(keys_dir).unwrap().flatten().find_map(|e| std::fs::read_to_string(e.path()).ok()).unwrap();
        let upstream_key = upstream_key.trim().to_string();
        let defaults = Settings::default();
        let dials = cerebro_server::Dials { effort: defaults.effort.clone(), verbosity: defaults.verbosity.clone() };
        let forwarded = |client: &Value| {
            let mut body = client.as_object().unwrap().clone();
            let mut plan = effort::resolve(&dials, &mut body).unwrap();
            effort::inject_system(body["messages"].as_array_mut().unwrap(), &effort::instruction_text(&plan)).unwrap();
            // This model's template has no thinking mode, so the gateway sends no thinking budget.
            plan.effort.thinking_budget = cerebro_config::ThinkingBudget::Off;
            let client_max = effort::client_max_tokens(&body).unwrap();
            effort::apply_budget(&mut body, &effort::plan_budget(&plan, client_max, None));
            Value::Object(body)
        };
        let outside_timings = |reply: &Reply, started: std::time::Instant| {
            let wall = started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(reply.status, 200, "{}", reply.body);
            let v = reply.json();
            wall - v["timings"]["prompt_ms"].as_f64().unwrap() - v["timings"]["predicted_ms"].as_f64().unwrap()
        };
        let (mut via_gateway, mut direct) = (Vec::new(), Vec::new());
        for i in 0..20 {
            let client = json!({"messages": [{"role": "user", "content": format!("Count to {}.", i % 5 + 2)}], "max_tokens": 8, "temperature": 0, "cerebro_effort": "low"});
            let upstream_body = forwarded(&client);
            for gateway_turn in [i % 2 == 0, i % 2 != 0] {
                let started = std::time::Instant::now();
                if gateway_turn {
                    let reply = send(addr, "POST", "/v1/chat/completions", Some(&token), Some(&client)).await;
                    via_gateway.push(outside_timings(&reply, started));
                } else {
                    let reply = send(upstream, "POST", "/v1/chat/completions", Some(&upstream_key), Some(&upstream_body)).await;
                    direct.push(outside_timings(&reply, started));
                }
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (gateway_ms, direct_ms) = (median(&mut via_gateway), median(&mut direct));
        let added = gateway_ms - direct_ms;
        println!("outside llama-server's timings, median of 20: {gateway_ms:.2} ms through the gateway, {direct_ms:.2} ms direct; the gateway adds {added:.2} ms");
        assert!(added < 5.0, "the gateway adds {added:.2} ms per request: gateway {via_gateway:?}, direct {direct:?}");
        status["launch"]["pid"].as_u64().map(|p| (p, disabled.len()))
    })
    .await;

    // Rotating the token on the running server: the old one is refused at once, the new one works.
    let rotated = async {
        let store = SecretStore::open(data.path()).unwrap();
        let old = serving.token().reveal().to_string();
        serving.rotate_token(&store).unwrap();
        let new = serving.token().reveal().to_string();
        assert_ne!(old, new);
        assert_eq!(send(addr, "GET", "/v1/models", Some(&old), None).await.status, 401, "old token refused after rotation");
        assert_eq!(send(addr, "GET", "/v1/models", Some(&new), None).await.status, 200, "new token accepted after rotation");
    };
    let rotated = tokio::time::timeout(Duration::from_secs(60), rotated).await;

    let shutdown = serving.shutdown().await;
    let (pid, disabled) = match outcome {
        Ok(Some(pair)) => pair,
        Ok(None) => panic!("status reported no pid"),
        Err(panic) => std::panic::resume_unwind(panic.into_panic()),
    };
    shutdown.unwrap();
    rotated.expect("rotation checks within 60 s");
    assert_eq!(disabled, turned_off.len(), "each protection turned off is reported once ({turned_off:?})");
    assert!(!llama_server_running(pid), "llama-server {pid} still running after shutdown");
}

/// Dropping a `Serving` without calling `shutdown` (a `?` after `serve`, a panic, the app starting
/// another one) still stops llama-server: nothing else holds the supervisor strongly.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs an official llama.cpp build (CEREBRO_LLAMACPP_OFFICIAL, e.g. C:\\llama-cuda) and the tiny \
            GGUF (CEREBRO_TINY_GGUF); loads the 0.5B model once"]
async fn dropping_serving_without_shutdown_stops_llama_server() {
    let (serving, ready, log_tail, _data, _) = serve_tiny().await;
    let addr = serving.addr();
    let token = serving.token().reveal().to_string();
    let pid = if ready.is_ok() {
        send(addr, "GET", "/cerebro/status", Some(&token), None).await.json()["launch"]["pid"].as_u64()
    } else {
        None
    };
    drop(serving);
    if let Err(state) = ready {
        panic!("the server did not become ready: {state:?}\nllama-server log tail:\n{}", log_tail.unwrap_or_default());
    }
    let pid = pid.expect("status reported the launch pid");
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while llama_server_running(pid) {
        assert!(std::time::Instant::now() < deadline, "llama-server {pid} still running 20 s after Serving was dropped");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    // The gateway stopped listening too.
    assert!(TcpStream::connect(addr).await.is_err(), "the gateway still accepts connections after the drop");
}
