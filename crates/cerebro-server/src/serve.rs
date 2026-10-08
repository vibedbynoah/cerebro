//! `cerebro serve`: launch one llama-server through the supervisor and put the gateway in front of
//! it.
//!
//! [`serve`] does, in order:
//!
//! 1. checks the model is inside a configured model folder;
//! 2. binds the gateway's listener on settings `api.bind`/`api.port`: the cheap step and the one
//!    most likely to fail (another instance on the port), so it fails before any key file is
//!    swept or any model is hashed or loaded;
//! 3. turns the load flags into the supervisor's extra flags: a tuned recommendation
//!    ([`LoadFlags::Tuned`], whose record in a verified store must describe this model file,
//!    backend build, machine and flags; see [`crate::tuned`]), or the user's load defaults from
//!    settings ([`LoadFlags::Settings`]; every unset parameter emits no flag, so llama.cpp's own
//!    automatic value applies), plus the settings sampler where the load flags do not set it.
//!    Flags preflight owns (host, port, key file, model, web UI, slots, CORS) are refused;
//! 4. writes a fresh per-launch key for llama-server to an owner-only key file (key files a killed
//!    Cerebro left behind are swept first; a file whose process is still running is kept);
//! 5. starts the [`Supervisor`] with the user's [`cerebro_supervisor::Protections`] from settings:
//!    preflight runs before every launch and restart, and every protection switched off is
//!    recorded in the launch record and shown under `launch.disabled_protections` in status;
//! 6. attaches the stored gateway token and the upstream to the listener and starts serving.
//!
//! `api.enabled` is not consulted: calling [`serve`] is the explicit request to listen (the desktop
//! app checks the setting before it calls).
//!
//! [`Serving::shutdown`] stops the gateway first, then the supervisor, and reports whether the OS
//! confirmed llama-server gone. Dropping a [`Serving`] without it still stops both, fire and
//! forget: [`Serving`] holds the only strong reference to the [`Supervisor`] (the status watcher
//! and the upstream's target lookup hold weak ones), so the drop reaches the supervisor's own
//! stop-on-drop, and the gateway's accept loop ends when its shutdown sender goes.
//!
//! A tuned configuration is served only when its record is at least as fast at decode depth as the
//! store's measurement of llama.cpp's automatic settings on the same model, build and machine.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use cerebro_bench::run::ModelHashCache;
use cerebro_config::{KeyFile, LaunchRequest, Secret, SecretStore, Settings};
use cerebro_store::Record;
use cerebro_flags::Flag;
use cerebro_supervisor::{LifecycleConfig, LifecycleState, PortRequest, RestartPolicy, SpawnOptions, Supervisor};
use serde_json::{json, Value};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::error::ServerError;
use crate::gateway::{GatewayConfig, GatewayHandle, GatewayListener, LaunchSink};
use crate::status::{probe_vram, Telemetry};
use crate::token::BearerToken;
use crate::tuned;
use crate::upstream::{TargetSource, Upstream, UpstreamTarget};

/// Which load flags the server is launched with.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadFlags {
    /// The user's load defaults and active preset from settings. With nothing set this emits no
    /// load flag at all: llama.cpp's automatic settings.
    Settings,
    /// A tuned recommendation (`cerebro_tune::TuneOutcome::recommended_server_flags`) and the
    /// store record its winning measurement was appended as. Used only when that record measured
    /// this model file, backend build and machine with exactly these flags ([`crate::tuned`]).
    Tuned {
        /// The recommended server flags, as measured.
        flags: Vec<Flag>,
        /// The results store the tune wrote.
        store: PathBuf,
        /// The winning measurement's record index in that store.
        record_index: usize,
    },
}

/// Everything [`serve`] needs.
#[derive(Debug, Clone)]
pub struct ServeRequest {
    /// The user's settings (backend, protections, load defaults, sampler, dials, API port).
    pub settings: Settings,
    /// The model to serve; must be inside a configured model folder.
    pub model_path: PathBuf,
    /// Expected backend files, name to lowercase hex SHA-256 (from `backend.manifest`).
    pub manifest: BTreeMap<String, String>,
    /// Which load flags to launch with.
    pub load: LoadFlags,
    /// Cerebro's data directory: the secret store, key files, the child's temp directory, driver
    /// cache and crash log live under it.
    pub data_dir: PathBuf,
    /// The port llama-server listens on (loopback); [`PortRequest::Any`] unless a client must reach
    /// it directly, which it should not.
    pub upstream_port: PortRequest,
    /// Gateway settings; [`GatewayConfig::from_settings`] plus the desktop app's origin, if any.
    pub gateway: GatewayConfig,
    /// How long one launch may take to load the model and answer `/health`.
    pub readiness_deadline: Duration,
    /// Model digests by file version, for checking a tuned record; share one across `serve` calls
    /// so serving the same model again does not hash it again.
    pub model_hashes: Arc<ModelHashCache>,
}

impl ServeRequest {
    /// A request with the gateway taken from settings, any upstream port and a 10-minute load
    /// deadline (a large model from a slow disk).
    pub fn new(settings: Settings, model_path: PathBuf, manifest: BTreeMap<String, String>, load: LoadFlags, data_dir: PathBuf) -> Self {
        let gateway = GatewayConfig::from_settings(&settings);
        Self {
            settings,
            model_path,
            manifest,
            load,
            data_dir,
            upstream_port: PortRequest::Any,
            gateway,
            readiness_deadline: Duration::from_secs(600),
            model_hashes: Arc::new(ModelHashCache::new()),
        }
    }
}

/// The supervisor as `serve` shares it: [`Serving`] holds the only strong reference.
type SharedSupervisor = Arc<Mutex<Option<Supervisor>>>;

/// A running gateway and its supervised llama-server.
///
/// Call [`Serving::shutdown`] to stop both and learn whether the OS confirmed llama-server gone.
/// Dropping it instead still stops both, without waiting or confirming (see the module docs).
pub struct Serving {
    gateway: GatewayHandle,
    /// The only strong reference: dropping it drops the [`Supervisor`], whose `Drop` asks the
    /// driver to stop the child.
    supervisor: SharedSupervisor,
    watcher: JoinHandle<()>,
    state: watch::Receiver<LifecycleState>,
    token: BearerToken,
    /// llama-server reads its key only at launch. [`Serving::shutdown`] deletes it after the
    /// supervisor confirmed the child gone; a plain drop deletes it while the stop is still in
    /// flight, which is harmless because a stopping driver never relaunches.
    key_file: Option<KeyFile>,
}

impl std::fmt::Debug for Serving {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Serving").field("addr", &self.gateway.addr()).field("state", &*self.state.borrow()).finish_non_exhaustive()
    }
}

impl Serving {
    /// The gateway's loopback address.
    pub fn addr(&self) -> std::net::SocketAddr {
        self.gateway.addr()
    }

    /// The gateway token clients must present (show it to the user; never log it).
    pub fn token(&self) -> &BearerToken {
        &self.token
    }

    /// Store a fresh gateway token and require it at once: from the next request on the old token
    /// is refused and the new one ([`Serving::token`]) accepted.
    ///
    /// # Errors
    /// [`ServerError::Random`] or [`ServerError::Secrets`]; on error nothing changed and the old
    /// token stays valid.
    pub fn rotate_token(&mut self, store: &SecretStore) -> Result<(), ServerError> {
        let fresh = BearerToken::rotate(store)?;
        self.gateway.set_token(fresh.duplicate());
        self.token = fresh;
        Ok(())
    }

    /// The supervised server's lifecycle state, updated as it changes.
    pub fn state(&self) -> watch::Receiver<LifecycleState> {
        self.state.clone()
    }

    /// Wait until the server is ready, or return the terminal state it ended in instead
    /// (`Failed`/`Stopped`), or the current state when `timeout` passes first.
    ///
    /// # Errors
    /// The state that ended the wait, when it is not `Ready`.
    pub async fn wait_ready(&self, timeout: Duration) -> Result<(), LifecycleState> {
        let mut rx = self.state.clone();
        let wait = async {
            loop {
                let current = rx.borrow_and_update().clone();
                match current {
                    LifecycleState::Ready => return Ok(()),
                    LifecycleState::Failed(_) | LifecycleState::Stopped => return Err(current),
                    _ => {}
                }
                if rx.changed().await.is_err() {
                    return Err(rx.borrow().clone());
                }
            }
        };
        match tokio::time::timeout(timeout, wait).await {
            Ok(result) => result,
            Err(_) => Err(self.state.borrow().clone()),
        }
    }

    /// Stop the gateway, then the supervised server.
    ///
    /// # Errors
    /// [`ServerError::Supervisor`] when the OS did not confirm llama-server gone (its port and
    /// VRAM may still be held).
    pub async fn shutdown(mut self) -> Result<(), ServerError> {
        self.gateway.shutdown().await;
        self.watcher.abort();
        let supervisor = self.supervisor.lock().ok().and_then(|mut s| s.take());
        let result = match supervisor {
            Some(s) => s.stop().await.map_err(ServerError::from),
            None => Ok(()),
        };
        self.key_file.take();
        result
    }
}

/// Launch llama-server under the supervisor and start the gateway in front of it. Returns once
/// both are started; the model may still be loading (see [`Serving::wait_ready`]).
///
/// # Errors
/// A model outside the model folders, a gateway bind failure (checked before anything else is
/// touched), a reserved, unverifiable or mismatched load flag, a secret-store or key-file failure,
/// or a supervisor refusal. Nothing is left running on error.
pub async fn serve(request: ServeRequest) -> Result<Serving, ServerError> {
    let ServeRequest { settings, model_path, manifest, load, data_dir, upstream_port, gateway, readiness_deadline, model_hashes } =
        request;
    let model_path = cerebro_config::resolve_model_path(&settings, &model_path)?;
    let connect_timeout = gateway.timeouts.connect;
    // Cheap and the likeliest failure (another instance on the port): before any key file is
    // swept, any model hashed or any model loaded.
    let listener = GatewayListener::bind(gateway).await?;
    let (extra_flags, load_info) = launch_flags(&settings, &load, &model_path, &manifest, model_hashes).await?;

    let dirs = Dirs::create(&data_dir)?;
    sweep_stale_key_files(&dirs.keys, live_process_ids().as_ref())?;
    let secrets = SecretStore::open(&data_dir)?;
    let token = BearerToken::load_or_create(&secrets)?;

    let upstream_key = BearerToken::generate()?;
    let key_file = KeyFile::create(&dirs.keys, &Secret::new(format!("{}\n", upstream_key.reveal()).into_bytes()))?;
    let launch_spec = settings.launch_spec(LaunchRequest {
        model_path,
        draft_model_path: None,
        manifest,
        port: upstream_port,
        api_key_file: key_file.path().to_path_buf(),
    });
    let config = LifecycleConfig {
        launch_spec,
        extra_flags,
        spawn_options: SpawnOptions {
            temp_dir: dirs.temp.clone(),
            cache_dir: dirs.cache.clone(),
            cuda_visible_devices: None,
            log_path: dirs.logs.join("llama-server.log"),
            log_max_bytes: 8 * 1024 * 1024,
        },
        api_key: upstream_key.reveal().to_string(),
        readiness_deadline,
        readiness_poll_interval: Duration::from_millis(200),
        child_poll_interval: Duration::from_millis(500),
        restart_policy: RestartPolicy {
            base_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(30),
            max_restarts: 3,
            window: Duration::from_secs(600),
            jitter_seed: u64::from(std::process::id()),
        },
        stop_timeout: Duration::from_secs(15),
    };

    let (supervisor, state_rx, _log) = Supervisor::spawn(config)?;
    let supervisor: SharedSupervisor = Arc::new(Mutex::new(Some(supervisor)));
    // The supervisor replaces its launch record before it publishes Ready for that launch, so the
    // upstream reads its target from there, never from a copy updated later. Weak: the gateway
    // must not keep the supervisor (and its child) alive once `Serving` is gone.
    let source = Arc::downgrade(&supervisor);
    let target: TargetSource = Arc::new(move || {
        let record = last_launch(&source)?;
        Some(UpstreamTarget { pid: record.pid, port: record.port })
    });
    let upstream = match Upstream::new(target, &upstream_key, state_rx.clone(), connect_timeout) {
        Ok(upstream) => upstream,
        Err(e) => {
            let s = supervisor.lock().ok().and_then(|mut s| s.take());
            if let Some(s) = s {
                let _ = s.stop().await;
            }
            return Err(e);
        }
    };
    let handle = listener.attach(token.duplicate(), upstream, Telemetry::new(probe_vram())).spawn();
    let sink = handle.launch_sink();
    sink.publish(json!({ "load": load_info }));
    let watcher = tokio::spawn(watch_launches(state_rx.clone(), Arc::downgrade(&supervisor), sink, load_info));
    Ok(Serving { gateway: handle, supervisor, watcher, state: state_rx, token, key_file: Some(key_file) })
}

/// The supervisor's latest launch record, while the supervisor still exists.
fn last_launch(supervisor: &Weak<Mutex<Option<Supervisor>>>) -> Option<Arc<cerebro_supervisor::LaunchRecord>> {
    let strong = supervisor.upgrade()?;
    let guard = strong.lock().ok()?;
    guard.as_ref().and_then(Supervisor::last_launch)
}

/// Keep the status launch facts in step with the supervisor: published from each new launch
/// record once the server is ready. Holds the supervisor weakly, so it never keeps it alive; it
/// ends when the supervisor's state channel closes.
async fn watch_launches(
    mut state: watch::Receiver<LifecycleState>,
    supervisor: Weak<Mutex<Option<Supervisor>>>,
    sink: LaunchSink,
    load_info: Value,
) {
    loop {
        let ready = *state.borrow_and_update() == LifecycleState::Ready;
        if ready {
            if let Some(record) = last_launch(&supervisor) {
                let disabled: Vec<Value> = record
                    .disabled_protections()
                    .iter()
                    .map(|w| json!({ "setting": w.protection().key(), "risk": w.risk() }))
                    .collect();
                sink.publish(json!({
                    "load": load_info,
                    "upstream_port": record.port,
                    "pid": record.pid,
                    "backend_dir": record.backend_dir,
                    "disabled_protections": disabled,
                }));
            }
        }
        if state.changed().await.is_err() {
            return;
        }
    }
}

/// The supervisor's extra flags and a description of where they came from. A tuned
/// recommendation is checked against `model_path` and `manifest` on a blocking thread
/// ([`crate::tuned`]); the flags served are the ones that check returns.
async fn launch_flags(
    settings: &Settings,
    load: &LoadFlags,
    model_path: &Path,
    manifest: &BTreeMap<String, String>,
    model_hashes: Arc<ModelHashCache>,
) -> Result<(Vec<Flag>, Value), ServerError> {
    let (mut flags, mut info) = match load {
        LoadFlags::Settings => (
            settings.load.effective().to_flags(),
            json!({ "source": "settings", "preset": settings.load.active_preset }),
        ),
        LoadFlags::Tuned { flags, store, record_index } => {
            if let Some(reserved) = flags.iter().find(|f| is_preflight_owned(f)) {
                return Err(ServerError::ReservedFlag(reserved.name()));
            }
            let (flags, store, index) = (flags.clone(), store.clone(), *record_index);
            let (model_path, manifest) = (model_path.to_path_buf(), manifest.clone());
            tokio::task::spawn_blocking(move || verify_tuned(&flags, &store, index, &model_path, &manifest, &model_hashes))
                .await
                .map_err(|_| ServerError::TunedUnverifiable { what: "the verification task did not finish".to_string() })??
        }
    };
    if let Some(reserved) = flags.iter().find(|f| is_preflight_owned(f)) {
        return Err(ServerError::ReservedFlag(reserved.name()));
    }
    for sampler in settings.sampler.to_flags() {
        if !flags.iter().any(|f| f.name() == sampler.name()) {
            flags.push(sampler);
        }
    }
    let rendered: Vec<&'static str> = flags.iter().map(Flag::name).collect();
    if let Some(map) = info.as_object_mut() {
        map.insert("flags".to_string(), json!(rendered));
    }
    Ok((flags, info))
}

/// Load record `index` from the verified store and check it against the model, backend and
/// machine about to be served (blocking: hashes the model and probes the machine). Returns the
/// flags to serve and the status description.
fn verify_tuned(
    flags: &[Flag],
    store: &Path,
    index: usize,
    model_path: &Path,
    manifest: &BTreeMap<String, String>,
    model_hashes: &ModelHashCache,
) -> Result<(Vec<Flag>, Value), ServerError> {
    let outcome = cerebro_store::Store::new(store).load_verified()?;
    let record = outcome
        .records
        .get(index)
        .ok_or(ServerError::TunedRecordMissing { index, len: outcome.records.len() })?;
    let model_sha = tuned::model_sha256(model_path, model_hashes)?;
    let served = tuned::verify(record, &model_sha, &tuned::manifest_digest(manifest), flags, tuned::current_machine_profile_id)?;
    let baseline = check_not_slower_than_baseline(&outcome.records, record)?;
    let tg = record.measurement().tg_tok_s_deepest();
    let info = json!({
        "source": "tuned",
        "record_index": index,
        "verified": ["model_sha256", "backend_manifest_digest", "flags", "machine_profile_id", "not_slower_than_automatic"],
        "measured_decode_tokens_per_second_at_depth": tg.median(),
        "baseline_decode_tokens_per_second_at_depth": baseline,
        "measured_depth": record.measurement().deepest_depth().get(),
    });
    Ok((served, info))
}

/// Whether a measured llama-bench argv is llama.cpp's automatic settings as `cerebro-tune`
/// measures them: `-m <model> -fitt 1024`, then the run plan (`-p ...`).
fn is_baseline_argv(argv: &[String]) -> bool {
    matches!(argv, [m, _, fitt, target, p, ..] if m == "-m" && fitt == "-fitt" && target == "1024" && p == "-p")
}

/// The owner rule "never serve a configuration slower than llama.cpp's automatic settings",
/// checked against the store itself rather than trusted from the caller's record index: the most
/// recent baseline record with the same model, backend build and machine, measured at the same
/// decode depth, must not decode faster (median tg at depth, the statistic `cerebro-tune` chooses
/// by) than `tuned`. Returns that baseline's median. Serving the baseline record itself passes.
///
/// # Errors
/// [`ServerError::TunedUnverifiable`] when the store holds no such baseline;
/// [`ServerError::TunedMismatch`] when `tuned` decoded slower than it.
fn check_not_slower_than_baseline(records: &[Record], tuned: &Record) -> Result<f64, ServerError> {
    let p = tuned.provenance();
    let depth = tuned.measurement().deepest_depth();
    let baseline = records.iter().rev().find(|r| {
        let q = r.provenance();
        q.model_sha256() == p.model_sha256()
            && q.backend_manifest_digest() == p.backend_manifest_digest()
            && q.machine_profile_id() == p.machine_profile_id()
            && r.measurement().deepest_depth() == depth
            && is_baseline_argv(r.measured_argv_untrusted())
    });
    let Some(baseline) = baseline else {
        return Err(ServerError::TunedUnverifiable {
            what: format!(
                "the store holds no measurement of llama.cpp's automatic settings at depth {depth} for this model, build and machine to compare against"
            ),
        });
    };
    let tuned_tg = tuned.measurement().tg_tok_s_deepest().median();
    let base_tg = baseline.measurement().tg_tok_s_deepest().median();
    if tuned_tg < base_tg {
        return Err(ServerError::TunedMismatch {
            what: format!(
                "the record decoded {tuned_tg:.1} tok/s at depth {depth}, slower than llama.cpp's automatic settings ({base_tg:.1})"
            ),
        });
    }
    Ok(base_tg)
}

/// Flags preflight and the hardening set own; the supervisor sets them on every launch.
fn is_preflight_owned(flag: &Flag) -> bool {
    matches!(
        flag,
        Flag::Model(_)
            | Flag::DraftModel(_)
            | Flag::Host(_)
            | Flag::Port(_)
            | Flag::ApiKeyFile(_)
            | Flag::WebUi(_)
            | Flag::Slots(_)
            | Flag::CorsOrigins(_)
    )
}

/// Directories under the data directory a launch uses.
struct Dirs {
    keys: PathBuf,
    temp: PathBuf,
    cache: PathBuf,
    logs: PathBuf,
}

impl Dirs {
    fn create(data_dir: &Path) -> Result<Self, ServerError> {
        let dirs = Self {
            keys: data_dir.join("keys"),
            temp: data_dir.join("run").join("tmp"),
            cache: data_dir.join("driver-cache"),
            logs: data_dir.join("logs"),
        };
        for dir in [&dirs.keys, &dirs.temp, &dirs.cache, &dirs.logs] {
            std::fs::create_dir_all(dir).map_err(|e| ServerError::Io { path: dir.display().to_string(), cause: e.to_string() })?;
        }
        Ok(dirs)
    }
}

/// The ids of every running process, or `None` when the process list cannot be read.
fn live_process_ids() -> Option<HashSet<u32>> {
    cerebro_probe::process::running().ok().map(|ps| ps.into_iter().map(|p| p.pid).collect())
}

/// The process id a key file's name carries (`cerebro-child-key-<pid>-<n>.key`).
fn key_file_pid(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("cerebro-child-key-")?.strip_suffix(".key")?;
    let (pid, counter) = rest.split_once('-')?;
    counter.parse::<u64>().ok()?;
    pid.parse().ok()
}

/// Delete `cerebro-child-key-<pid>-*` files a killed Cerebro left behind (a process kill skips
/// [`KeyFile`]'s `Drop`): only this directory, only that name pattern, regular files only, and only
/// when `<pid>` is neither this process nor any running one, so another Cerebro instance's live key
/// survives. With `live` unknown (`None`, the process list could not be read) nothing is deleted:
/// a stale file costs nothing, a deleted live key breaks that instance's next restart.
fn sweep_stale_key_files(keys: &Path, live: Option<&HashSet<u32>>) -> Result<(), ServerError> {
    let Some(live) = live else {
        return Ok(());
    };
    let own = std::process::id();
    let entries = std::fs::read_dir(keys).map_err(|e| ServerError::Io { path: keys.display().to_string(), cause: e.to_string() })?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name.to_str().and_then(key_file_pid) else {
            continue;
        };
        if pid == own || live.contains(&pid) || !entry.file_type().is_ok_and(|t| t.is_file()) {
            continue;
        }
        let _ = std::fs::remove_file(entry.path());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerebro_config::Source;
    use cerebro_flags::value::{CacheType, FlashAttn};
    use std::num::NonZeroU32;

    fn hashes() -> Arc<ModelHashCache> {
        Arc::new(ModelHashCache::new())
    }

    #[tokio::test]
    async fn settings_load_flags_add_the_sampler_and_nothing_else_by_default() {
        let settings = Settings::default();
        let (flags, info) = launch_flags(&settings, &LoadFlags::Settings, Path::new("m.gguf"), &BTreeMap::new(), hashes())
            .await
            .expect("flags");
        // No load parameter set: no load flag, only the sampler (llama.cpp's automatic settings).
        assert_eq!(flags.iter().map(Flag::name).collect::<Vec<_>>(), settings.sampler.to_flags().iter().map(Flag::name).collect::<Vec<_>>());
        assert_eq!(info["source"], "settings");
    }

    #[tokio::test]
    async fn preflight_owned_flags_are_refused_before_the_store_is_read() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = dir.path().join("results.jsonl");
        let load = LoadFlags::Tuned { flags: vec![Flag::Port(1)], store: store.clone(), record_index: 0 };
        let err = launch_flags(&Settings::default(), &load, Path::new("m.gguf"), &BTreeMap::new(), hashes()).await.unwrap_err();
        assert!(matches!(err, ServerError::ReservedFlag("--port")), "{err}");
        // An empty store reports the missing record.
        let load = LoadFlags::Tuned { flags: vec![Flag::GpuLayers(99)], store, record_index: 0 };
        let err = launch_flags(&Settings::default(), &load, Path::new("m.gguf"), &BTreeMap::new(), hashes()).await.unwrap_err();
        assert!(matches!(err, ServerError::TunedRecordMissing { index: 0, len: 0 }), "{err}");
        assert!(is_preflight_owned(&Flag::ApiKeyFile("k".into())));
        assert!(is_preflight_owned(&Flag::Host("0.0.0.0".into())));
        assert!(!is_preflight_owned(&Flag::GpuLayers(99)));
    }

    /// A store holding one record measured for a model whose SHA-256 is all `b`s.
    fn store_with_record(dir: &Path) -> PathBuf {
        let path = dir.join("results.jsonl");
        let target = NonZeroU32::new(1024).expect("non-zero");
        cerebro_store::Store::new(&path).append(&crate::tuned::tests::record(&[Flag::FitTarget(target)])).expect("append");
        path
    }

    fn baseline_flags() -> Vec<Flag> {
        let nz = |n| NonZeroU32::new(n).expect("non-zero");
        vec![
            Flag::Fit(true),
            Flag::FitTarget(nz(1024)),
            Flag::FitCtx(nz(4096)),
            Flag::FlashAttn(FlashAttn::Auto),
            Flag::CacheTypeK(CacheType::F16),
            Flag::CacheTypeV(CacheType::F16),
        ]
    }

    #[tokio::test]
    async fn a_tuned_record_for_another_model_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = store_with_record(dir.path());
        let model = dir.path().join("other.gguf");
        std::fs::write(&model, b"not the measured model").expect("write");
        let load = LoadFlags::Tuned { flags: baseline_flags(), store, record_index: 0 };
        let err = launch_flags(&Settings::default(), &load, &model, &BTreeMap::new(), hashes()).await.unwrap_err();
        assert!(matches!(&err, ServerError::TunedMismatch { what } if what.contains("different model")), "{err}");
        // A model that cannot be read is unverifiable, never assumed to match.
        let load = LoadFlags::Tuned { flags: baseline_flags(), store: store_with_record(dir.path()), record_index: 0 };
        let err = launch_flags(&Settings::default(), &load, &dir.path().join("missing.gguf"), &BTreeMap::new(), hashes()).await;
        assert!(matches!(err, Err(ServerError::TunedUnverifiable { .. }) | Err(ServerError::Store(_))), "{err:?}");
    }

    /// Settings whose model folder is `dir`, holding a placeholder model file.
    fn settings_with_model(dir: &Path) -> (Settings, PathBuf) {
        let model = dir.join("m.gguf");
        std::fs::write(&model, b"placeholder").expect("write");
        let mut settings = Settings::default();
        settings.apply_patch(&json!({"models": {"folders": [dir]}}), Source::Gui).expect("patch");
        (settings, model)
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn a_taken_gateway_port_fails_before_anything_is_swept_or_launched() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (settings, model) = settings_with_model(dir.path());
        let data = dir.path().join("data");
        let keys = data.join("keys");
        std::fs::create_dir_all(&keys).expect("keys dir");
        // A key file named for a pid that is not running: the sweep would delete it.
        let stale = keys.join("cerebro-child-key-4294967290-0.key");
        std::fs::write(&stale, "x").expect("write");

        let taken = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let mut request = ServeRequest::new(settings, model, BTreeMap::new(), LoadFlags::Settings, data.clone());
        request.gateway.port = taken.local_addr().expect("addr").port();
        let err = serve(request).await.unwrap_err();
        assert!(matches!(err, ServerError::Bind { .. }), "{err}");
        assert!(stale.exists(), "nothing is swept before the port is held");
        assert!(!data.join("secrets").exists(), "nothing is created before the port is held");
    }

    #[tokio::test]
    async fn a_tuned_mismatch_fails_serve_before_any_key_or_launch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (settings, model) = settings_with_model(dir.path());
        let data = dir.path().join("data");
        let load = LoadFlags::Tuned { flags: baseline_flags(), store: store_with_record(dir.path()), record_index: 0 };
        let mut request = ServeRequest::new(settings, model, BTreeMap::new(), load, data.clone());
        request.gateway.port = 0;
        let err = serve(request).await.unwrap_err();
        assert!(matches!(&err, ServerError::TunedMismatch { what } if what.contains("different model")), "{err}");
        assert!(!data.join("keys").exists(), "no key file directory, so no key and no launch");
    }

    fn moe_bench() -> Vec<Flag> {
        vec![Flag::GpuLayers(999), Flag::NCpuMoe(32)]
    }

    #[test]
    fn a_tuned_record_slower_than_automatic_settings_is_refused() {
        use crate::tuned::tests::{record, record_with_tg};
        let target = NonZeroU32::new(1024).expect("non-zero");
        let baseline = record_with_tg(&[Flag::FitTarget(target)], 50.0, 4096);
        let slower = record_with_tg(&moe_bench(), 45.0, 4096);
        let faster = record_with_tg(&moe_bench(), 55.0, 4096);

        let err = check_not_slower_than_baseline(&[baseline.clone(), slower.clone()], &slower).unwrap_err();
        assert!(matches!(&err, ServerError::TunedMismatch { what } if what.contains("slower than llama.cpp's automatic settings")), "{err}");
        assert_eq!(check_not_slower_than_baseline(&[baseline.clone(), faster.clone()], &faster).expect("faster"), 50.0);
        // The baseline record itself is never slower than itself.
        assert_eq!(check_not_slower_than_baseline(std::slice::from_ref(&baseline), &baseline).expect("baseline"), 50.0);
        // The latest baseline counts: re-measured slower after the winner, the winner stands.
        let remeasured = record_with_tg(&[Flag::FitTarget(target)], 44.0, 4096);
        check_not_slower_than_baseline(&[baseline.clone(), slower.clone(), remeasured], &slower).expect("against the latest baseline");

        // No baseline at all, or only one measured at another depth: unverifiable.
        let err = check_not_slower_than_baseline(std::slice::from_ref(&faster), &faster).unwrap_err();
        assert!(matches!(err, ServerError::TunedUnverifiable { .. }), "{err}");
        let shallow = record_with_tg(&[Flag::FitTarget(target)], 10.0, 1024);
        let err = check_not_slower_than_baseline(&[shallow, faster.clone()], &faster).unwrap_err();
        assert!(matches!(err, ServerError::TunedUnverifiable { .. }), "{err}");
        // A record fitted to another target is not the baseline.
        let not_baseline = record(&[Flag::FitTarget(NonZeroU32::new(2048).expect("non-zero"))]);
        assert!(!is_baseline_argv(not_baseline.measured_argv_untrusted()));
        assert!(is_baseline_argv(baseline.measured_argv_untrusted()));
    }

    #[test]
    fn key_file_names_carry_their_pid() {
        assert_eq!(key_file_pid("cerebro-child-key-1234-0.key"), Some(1234));
        assert_eq!(key_file_pid("cerebro-child-key-1234-17.key"), Some(1234));
        for bad in ["cerebro-child-key-x-0.key", "cerebro-child-key-1234.key", "cerebro-child-key-1-0.txt", "other-1-0.key"] {
            assert_eq!(key_file_pid(bad), None, "{bad}");
        }
    }

    #[test]
    fn only_dead_instances_key_files_are_swept() {
        let dir = tempfile::tempdir().expect("tempdir");
        let own = std::process::id();
        let dead = dir.path().join("cerebro-child-key-4294967290-0.key");
        let live_other = dir.path().join("cerebro-child-key-77-3.key");
        let mine = dir.path().join(format!("cerebro-child-key-{own}-0.key"));
        let other = dir.path().join("notes.txt");
        for f in [&dead, &live_other, &mine, &other] {
            std::fs::write(f, "x").expect("write");
        }
        // Unknown liveness: nothing is deleted.
        sweep_stale_key_files(dir.path(), None).expect("sweep");
        assert!(dead.exists() && live_other.exists() && mine.exists());

        let live: HashSet<u32> = [77].into_iter().collect();
        sweep_stale_key_files(dir.path(), Some(&live)).expect("sweep");
        assert!(!dead.exists(), "a dead pid's key file is swept");
        assert!(live_other.exists(), "another running instance's key file survives");
        assert!(mine.exists(), "this process's own key file survives");
        assert!(other.exists());
    }

    #[cfg(windows)]
    #[test]
    fn the_live_process_list_is_read_and_holds_this_process() {
        let live = live_process_ids().expect("process list");
        assert!(live.contains(&std::process::id()));
    }
}
