//! Optional backends, each enabled only where its hardware and OS requirements hold: vLLM and
//! SGLang (Linux or WSL 2), ExLlamaV3 (NVIDIA), MLX (Apple Silicon), OpenVINO (Intel CPU/GPU/NPU),
//! Ryzen AI (AMD XDNA 2 NPU) and Qualcomm QNN (Hexagon NPU).
//!
//! None of these has been run by Cerebro yet ([`Verification::Unverified`]): their requirements,
//! argv and health checks are modelled from upstream documentation as of 2026-10 and must be
//! confirmed by running each binary before any claim is made about them (docs/CLAIMS.md, "run the
//! binary to make the claim"). Every launch here follows the same rules as llama.cpp's: loopback
//! only, no shell, the model given as a local path, and for every Python runtime the Hugging Face
//! hub forced offline and every documented telemetry opt-out set ([`OFFLINE_ENV`]).
//!
//! **API keys.** Every backend that can take a key is keyed: by environment (vLLM), by a key file
//! the supervisor writes (TabbyAPI's `api_tokens.yml`), or, where the runtime takes it no other
//! way, on its command line (SGLang `--api-key`, with a note that same-user processes can read an
//! argv). A backend that cannot take a key at all (MLX, OpenVINO Model Server, Lemonade) refuses to
//! build a launch while the user's `require_api_key` protection is on
//! ([`BackendError::KeylessRefused`]); with it off it runs keyless on loopback and the launch
//! carries one neutral note. OVMS documents `--api_key_file` and Lemonade an API-key variable, but
//! neither binary is installed here to confirm them against (docs/CLAIMS.md), so both stay keyless
//! until a run does.

use std::path::{Path, PathBuf};

use cerebro_probe::hardware::{AmdArch, CpuArch, NpuArch, NvidiaArch, OsFamily};

use crate::backend::{
    path_text, to_wsl_path, validate_common, validate_context_length, validate_distro,
    validate_linux_path, ApiKeyDelivery, Backend, BackendId, EnvValue, HealthCheck, InstallKind,
    InstallReport, KeyFileFormat, LaunchCommand, LaunchRecipe, LaunchRequest, ListenerOwner,
    Location, ModelFormat, OpenVinoDevice, PythonHost, RuntimeQuant, Verification, LOOPBACK,
};
use crate::error::BackendError;
use crate::llamacpp::NVIDIA_TURING_PLUS;
use crate::requirement::Requirement;

/// Volta and newer (vLLM's documented floor, compute capability 7.0).
const NVIDIA_VOLTA_PLUS: &[NvidiaArch] = &[
    NvidiaArch::Volta,
    NvidiaArch::Turing,
    NvidiaArch::Ampere,
    NvidiaArch::Ada,
    NvidiaArch::Hopper,
    NvidiaArch::BlackwellDatacenter,
    NvidiaArch::BlackwellClient,
];

/// Ampere and newer (FlashAttention 2's floor).
pub(crate) const NVIDIA_AMPERE_PLUS: &[NvidiaArch] = &[
    NvidiaArch::Ampere,
    NvidiaArch::Ada,
    NvidiaArch::Hopper,
    NvidiaArch::BlackwellDatacenter,
    NvidiaArch::BlackwellClient,
];

const REQ_VLLM: &[Requirement] = &[
    Requirement::LinuxOrWsl2,
    Requirement::AnyOf(&[
        Requirement::NvidiaArch(NVIDIA_VOLTA_PLUS),
        Requirement::AmdArch(&[AmdArch::Cdna2, AmdArch::Cdna3, AmdArch::Rdna3, AmdArch::Rdna4]),
    ]),
];
const REQ_SGLANG: &[Requirement] = &[
    Requirement::LinuxOrWsl2,
    Requirement::AnyOf(&[
        Requirement::NvidiaArch(NVIDIA_TURING_PLUS),
        Requirement::AmdArch(&[AmdArch::Cdna3]),
    ]),
];
const REQ_EXLLAMAV3: &[Requirement] = &[
    Requirement::Os(&[OsFamily::Windows, OsFamily::Linux]),
    Requirement::NvidiaArch(NVIDIA_AMPERE_PLUS),
];
const REQ_MLX: &[Requirement] = &[
    Requirement::Os(&[OsFamily::MacOs]),
    Requirement::NativeArch(&[CpuArch::Aarch64]),
];
const REQ_OPENVINO: &[Requirement] = &[
    Requirement::Os(&[OsFamily::Windows, OsFamily::Linux]),
    Requirement::NativeArch(&[CpuArch::X86_64]),
];
const REQ_RYZEN_AI: &[Requirement] = &[
    Requirement::Os(&[OsFamily::Windows]),
    Requirement::Npu(&[NpuArch::AmdXdna2]),
];
const REQ_QNN: &[Requirement] = &[
    Requirement::Os(&[OsFamily::Windows]),
    Requirement::NativeArch(&[CpuArch::Aarch64]),
    Requirement::Npu(&[NpuArch::QualcommHexagon]),
];

/// Recorded once per launch of a keyless backend, which only happens with the user's
/// `require_api_key` protection off.
pub const NOTE_KEYLESS: &str = "the API key requirement is turned off in settings and this backend \
                                cannot take a key, so it runs keyless on loopback; any local \
                                process can reach it while it runs";

/// Recorded with every SGLang launch: its key travels in the argv.
pub const NOTE_ARGV_KEY: &str = "SGLang takes its API key only on its command line, which other \
                                 processes of this user can read (under WSL also through \
                                 \\\\wsl.localhost\\<distro>\\proc)";

/// Recorded with every WSL launch whose model is on a Windows drive.
pub const NOTE_WSL_MOUNT: &str = "the model is read through WSL's /mnt drive mount (9P), not the \
                                  distribution's own disk; Cerebro has not measured that \
                                  load-time cost";

/// Why Qualcomm QNN has no launch recipe.
const QNN_MISSING: &str = "the QNN SDK has no OpenAI-compatible server; a serving shim is not built yet";

/// Every Python backend's environment: the hub offline (owner rule: everything works with no
/// network, and a typo'd local path must not become a download) and every documented telemetry
/// opt-out set (nothing leaves the machine unless the user asks).
pub const OFFLINE_ENV: &[(&str, &str)] = &[
    ("HF_HUB_OFFLINE", "1"),
    // Older transformers releases read their own switch.
    ("TRANSFORMERS_OFFLINE", "1"),
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
    // The cross-tool opt-out (vLLM and others honour it).
    ("DO_NOT_TRACK", "1"),
    // vLLM's anonymous usage reporting to stats.vllm.ai.
    ("VLLM_NO_USAGE_STATS", "1"),
];

/// The note list of a backend that cannot take a key: refused while the protection is on.
fn keyless(backend: BackendId, request: &LaunchRequest) -> Result<Vec<String>, BackendError> {
    if request.require_api_key {
        Err(BackendError::KeylessRefused { backend })
    } else {
        Ok(vec![NOTE_KEYLESS.into()])
    }
}

/// The program for a console script in a Python environment root on this machine.
fn native_script(env: &Path, script: &str) -> PathBuf {
    if cfg!(windows) {
        env.join("Scripts").join(format!("{script}.exe"))
    } else {
        env.join("bin").join(script)
    }
}

/// How a Linux-or-WSL backend is invoked: the program and leading args, the model path as the
/// Linux side sees it, who owns the listener and the notes the location implies.
struct LinuxInvocation {
    program: PathBuf,
    prefix: Vec<String>,
    model: String,
    wsl: bool,
    listener: ListenerOwner,
    notes: Vec<String>,
}

/// Resolve `script` (relative to the environment's `bin/`) for a Linux-or-WSL backend. Under WSL
/// the command runs with `wsl.exe --exec`, which executes it directly, without the distribution's
/// shell, so no argument is ever shell-interpreted.
fn linux_invocation(request: &LaunchRequest, script: &str) -> Result<LinuxInvocation, BackendError> {
    match &request.location {
        Location::Native(env) => {
            if cfg!(windows) {
                return Err(BackendError::InvalidRequest(
                    "this backend runs on Linux; on Windows give a WSL location".into(),
                ));
            }
            Ok(LinuxInvocation {
                program: env.join("bin").join(script),
                prefix: Vec::new(),
                model: path_text(&request.model)?.to_string(),
                wsl: false,
                listener: ListenerOwner::Child,
                notes: Vec::new(),
            })
        }
        Location::Wsl { distro, env_path } => {
            validate_distro(distro)?;
            validate_linux_path(env_path)?;
            Ok(LinuxInvocation {
                program: wsl_exe()?,
                prefix: vec![
                    "-d".into(),
                    distro.clone(),
                    "--exec".into(),
                    format!("{}/bin/{script}", env_path.trim_end_matches('/')),
                ],
                model: to_wsl_path(&request.model)?,
                wsl: true,
                // WSL 2 localhost forwarding: Windows' side of the port belongs to WSL's relay.
                listener: ListenerOwner::WslRelay,
                notes: vec![NOTE_WSL_MOUNT.into()],
            })
        }
    }
}

#[cfg(windows)]
fn wsl_exe() -> Result<PathBuf, BackendError> {
    cerebro_probe::child_env::system_directory()
        .map(|d| d.join("wsl.exe"))
        .map_err(|e| BackendError::InvalidRequest(format!("system directory unknown: {e}")))
}

#[cfg(not(windows))]
fn wsl_exe() -> Result<PathBuf, BackendError> {
    Err(BackendError::InvalidRequest("WSL exists only on Windows".into()))
}

/// The environment every Python backend gets: [`OFFLINE_ENV`], plus the API key variable when the
/// backend takes one. Under WSL, `WSLENV` forwards every one of them across the boundary; it is
/// built from the final list, so nothing can be added without being forwarded.
fn python_env(api_key_var: Option<&'static str>, wsl: bool) -> Vec<(String, EnvValue)> {
    let mut env: Vec<(String, EnvValue)> = OFFLINE_ENV
        .iter()
        .map(|(k, v)| ((*k).to_string(), EnvValue::Literal((*v).to_string())))
        .collect();
    if let Some(var) = api_key_var {
        env.push((var.to_string(), EnvValue::ApiKey));
    }
    if wsl {
        let forwarded: Vec<String> = env.iter().map(|(name, _)| format!("{name}/u")).collect();
        env.push(("WSLENV".to_string(), EnvValue::Literal(forwarded.join(":"))));
    }
    env
}

fn refuse_llama_flags(backend: BackendId, request: &LaunchRequest) -> Result<(), BackendError> {
    if request.llama_flags.is_empty() {
        Ok(())
    } else {
        Err(BackendError::OptionUnsupported {
            backend,
            option: "llama.cpp flags",
            hint: "llama.cpp flags apply only to llama.cpp backends",
        })
    }
}

fn refuse(backend: BackendId, option: &'static str, set: bool) -> Result<(), BackendError> {
    if set {
        Err(BackendError::OptionUnsupported {
            backend,
            option,
            hint: "not available on this backend",
        })
    } else {
        Ok(())
    }
}

/// A Python-environment install has no fixed file list to check; the entry script is the check.
fn check_python_entry(backend: BackendId, env: &Path, script: &str) -> Result<InstallReport, BackendError> {
    check_entry(backend, env, native_script(env, script))
}

/// The same check for a Linux environment, which keeps scripts in `bin/` even when examined from
/// Windows through its `\\wsl.localhost\<distro>\...` path.
fn check_linux_entry(backend: BackendId, env: &Path, script: &str) -> Result<InstallReport, BackendError> {
    check_entry(backend, env, env.join("bin").join(script))
}

fn check_entry(backend: BackendId, env: &Path, entry: PathBuf) -> Result<InstallReport, BackendError> {
    if entry.is_file() {
        Ok(InstallReport { entry, build: None })
    } else {
        Err(BackendError::MissingFiles {
            backend,
            dir: env.to_path_buf(),
            missing: vec![entry.display().to_string()],
        })
    }
}

/// vLLM's OpenAI-compatible server.
#[derive(Debug, Clone, Copy, Default)]
pub struct Vllm;

impl Backend for Vllm {
    fn id(&self) -> BackendId {
        BackendId::Vllm
    }
    fn model_formats(&self) -> &'static [ModelFormat] {
        &[ModelFormat::Safetensors]
    }
    fn requirements(&self) -> &'static [Requirement] {
        REQ_VLLM
    }
    fn install(&self) -> InstallKind {
        InstallKind::PythonEnv {
            package: "vllm",
            runs_in: PythonHost::LinuxOrWsl2,
        }
    }
    fn check_install(&self, dir: &Path) -> Result<InstallReport, BackendError> {
        check_linux_entry(self.id(), dir, "vllm")
    }
    fn build_argv(&self, request: &LaunchRequest) -> Result<LaunchCommand, BackendError> {
        validate_common(request)?;
        validate_context_length(request)?;
        refuse_llama_flags(self.id(), request)?;
        refuse(self.id(), "openvino_device", request.openvino_device.is_some())?;
        let inv = linux_invocation(request, "vllm")?;
        let mut args = inv.prefix;
        args.extend([
            "serve".into(),
            inv.model,
            "--host".into(),
            LOOPBACK.into(),
            "--port".into(),
            request.port.to_string(),
        ]);
        if let Some(ctx) = request.context {
            args.extend(["--max-model-len".into(), ctx.to_string()]);
        }
        if let Some(tp) = request.tensor_parallel {
            args.extend(["--tensor-parallel-size".into(), tp.to_string()]);
        }
        if let Some(RuntimeQuant::Fp8) = request.quantization {
            args.extend(["--quantization".into(), "fp8".into()]);
        }
        Ok(LaunchCommand {
            program: inv.program,
            args,
            env: python_env(Some("VLLM_API_KEY"), inv.wsl),
            api_key: ApiKeyDelivery::Env("VLLM_API_KEY"),
            listener: inv.listener,
            notes: inv.notes,
        })
    }
    fn health_check(&self) -> HealthCheck {
        HealthCheck {
            path: "/health",
            ready_status: 200,
            loading_status: None,
            needs_api_key: false,
        }
    }
    fn verification(&self) -> Verification {
        Verification::Unverified
    }
}

/// SGLang's server (`python -m sglang.launch_server`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Sglang;

impl Backend for Sglang {
    fn id(&self) -> BackendId {
        BackendId::Sglang
    }
    fn model_formats(&self) -> &'static [ModelFormat] {
        &[ModelFormat::Safetensors]
    }
    fn requirements(&self) -> &'static [Requirement] {
        REQ_SGLANG
    }
    fn install(&self) -> InstallKind {
        InstallKind::PythonEnv {
            package: "sglang",
            runs_in: PythonHost::LinuxOrWsl2,
        }
    }
    fn check_install(&self, dir: &Path) -> Result<InstallReport, BackendError> {
        check_linux_entry(self.id(), dir, "python3")
    }
    fn build_argv(&self, request: &LaunchRequest) -> Result<LaunchCommand, BackendError> {
        validate_common(request)?;
        validate_context_length(request)?;
        refuse_llama_flags(self.id(), request)?;
        refuse(self.id(), "openvino_device", request.openvino_device.is_some())?;
        let inv = linux_invocation(request, "python3")?;
        let mut args = inv.prefix;
        args.extend([
            "-m".into(),
            "sglang.launch_server".into(),
            "--model-path".into(),
            inv.model,
            "--host".into(),
            LOOPBACK.into(),
            "--port".into(),
            request.port.to_string(),
        ]);
        if let Some(ctx) = request.context {
            args.extend(["--context-length".into(), ctx.to_string()]);
        }
        if let Some(tp) = request.tensor_parallel {
            args.extend(["--tp".into(), tp.to_string()]);
        }
        if let Some(RuntimeQuant::Fp8) = request.quantization {
            args.extend(["--quantization".into(), "fp8".into()]);
        }
        let mut notes = vec![NOTE_ARGV_KEY.to_string()];
        notes.extend(inv.notes);
        Ok(LaunchCommand {
            program: inv.program,
            args,
            env: python_env(None, inv.wsl),
            // SGLang reads its key from `--api-key` only; the supervisor appends it.
            api_key: ApiKeyDelivery::Argv("--api-key"),
            listener: inv.listener,
            notes,
        })
    }
    fn health_check(&self) -> HealthCheck {
        HealthCheck {
            path: "/health",
            ready_status: 200,
            loading_status: None,
            needs_api_key: false,
        }
    }
    fn verification(&self) -> Verification {
        Verification::Unverified
    }
}

/// ExLlamaV3 through TabbyAPI (a Python app with its own venv in `<install>/venv`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ExLlamaV3;

impl Backend for ExLlamaV3 {
    fn id(&self) -> BackendId {
        BackendId::ExLlamaV3
    }
    fn model_formats(&self) -> &'static [ModelFormat] {
        &[ModelFormat::Exl3]
    }
    fn requirements(&self) -> &'static [Requirement] {
        REQ_EXLLAMAV3
    }
    fn install(&self) -> InstallKind {
        InstallKind::PythonEnv {
            package: "exllamav3",
            runs_in: PythonHost::Native,
        }
    }
    fn check_install(&self, dir: &Path) -> Result<InstallReport, BackendError> {
        let main = dir.join("main.py");
        if !main.is_file() {
            return Err(BackendError::MissingFiles {
                backend: self.id(),
                dir: dir.to_path_buf(),
                missing: vec!["main.py".into()],
            });
        }
        check_python_entry(self.id(), &dir.join("venv"), "python")
    }
    fn build_argv(&self, request: &LaunchRequest) -> Result<LaunchCommand, BackendError> {
        validate_common(request)?;
        validate_context_length(request)?;
        refuse_llama_flags(self.id(), request)?;
        refuse(self.id(), "tensor_parallel", request.tensor_parallel.is_some())?;
        refuse(self.id(), "load-time quantization", request.quantization.is_some())?;
        refuse(self.id(), "openvino_device", request.openvino_device.is_some())?;
        let Location::Native(app) = &request.location else {
            return Err(BackendError::InvalidRequest("ExLlamaV3 runs natively".into()));
        };
        // TabbyAPI takes a model folder as a parent directory plus a folder name.
        let (Some(parent), Some(name)) = (request.model.parent(), request.model.file_name()) else {
            return Err(BackendError::InvalidRequest("model must be a folder path".into()));
        };
        let name = name.to_str().ok_or_else(|| {
            BackendError::InvalidRequest("model folder name is not valid UTF-8".into())
        })?;
        let mut args = vec![
            path_text(&app.join("main.py"))?.to_string(),
            "--host".into(),
            LOOPBACK.into(),
            "--port".into(),
            request.port.to_string(),
            "--model-dir".into(),
            path_text(parent)?.to_string(),
            "--model-name".into(),
            name.to_string(),
        ];
        if let Some(ctx) = request.context {
            args.extend(["--max-seq-len".into(), ctx.to_string()]);
        }
        Ok(LaunchCommand {
            program: native_script(&app.join("venv"), "python"),
            args,
            env: python_env(None, false),
            // TabbyAPI authenticates by default from `api_tokens.yml` in its folder (its working
            // directory when started from it); the supervisor writes the launch key there.
            api_key: ApiKeyDelivery::KeyFile {
                path: app.join("api_tokens.yml"),
                format: KeyFileFormat::TabbyApiTokens,
            },
            listener: ListenerOwner::Child,
            notes: Vec::new(),
        })
    }
    fn health_check(&self) -> HealthCheck {
        HealthCheck {
            path: "/health",
            ready_status: 200,
            loading_status: None,
            needs_api_key: false,
        }
    }
    fn verification(&self) -> Verification {
        Verification::Unverified
    }
}

/// MLX's `mlx_lm.server` (Apple Silicon; for the future macOS build).
#[derive(Debug, Clone, Copy, Default)]
pub struct Mlx;

impl Backend for Mlx {
    fn id(&self) -> BackendId {
        BackendId::Mlx
    }
    fn model_formats(&self) -> &'static [ModelFormat] {
        &[ModelFormat::Mlx]
    }
    fn requirements(&self) -> &'static [Requirement] {
        REQ_MLX
    }
    fn install(&self) -> InstallKind {
        InstallKind::PythonEnv {
            package: "mlx-lm",
            runs_in: PythonHost::Native,
        }
    }
    fn check_install(&self, dir: &Path) -> Result<InstallReport, BackendError> {
        check_python_entry(self.id(), dir, "mlx_lm.server")
    }
    fn build_argv(&self, request: &LaunchRequest) -> Result<LaunchCommand, BackendError> {
        validate_common(request)?;
        refuse_llama_flags(self.id(), request)?;
        refuse(self.id(), "tensor_parallel", request.tensor_parallel.is_some())?;
        refuse(self.id(), "load-time quantization", request.quantization.is_some())?;
        refuse(self.id(), "openvino_device", request.openvino_device.is_some())?;
        refuse(self.id(), "context", request.context.is_some())?;
        let Location::Native(env) = &request.location else {
            return Err(BackendError::InvalidRequest("MLX runs natively on macOS".into()));
        };
        // mlx_lm.server has no authentication at all.
        let notes = keyless(self.id(), request)?;
        Ok(LaunchCommand {
            program: native_script(env, "mlx_lm.server"),
            args: vec![
                "--model".into(),
                path_text(&request.model)?.to_string(),
                "--host".into(),
                LOOPBACK.into(),
                "--port".into(),
                request.port.to_string(),
            ],
            env: python_env(None, false),
            api_key: ApiKeyDelivery::None,
            listener: ListenerOwner::Child,
            notes,
        })
    }
    fn health_check(&self) -> HealthCheck {
        HealthCheck {
            path: "/v1/models",
            ready_status: 200,
            loading_status: None,
            needs_api_key: false,
        }
    }
    fn verification(&self) -> Verification {
        Verification::Unverified
    }
}

/// OpenVINO Model Server (`ovms.exe`), for Intel CPUs, GPUs and NPUs.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenVino;

const OVMS_FILES: &[&str] = &["ovms.exe"];

impl Backend for OpenVino {
    fn id(&self) -> BackendId {
        BackendId::OpenVino
    }
    fn model_formats(&self) -> &'static [ModelFormat] {
        &[ModelFormat::OpenVinoIr]
    }
    fn requirements(&self) -> &'static [Requirement] {
        REQ_OPENVINO
    }
    fn install(&self) -> InstallKind {
        InstallKind::Folder {
            dir_name: "openvino-model-server",
            entry: "ovms.exe",
            required: OVMS_FILES,
        }
    }
    fn check_install(&self, dir: &Path) -> Result<InstallReport, BackendError> {
        let entry = dir.join("ovms.exe");
        if entry.is_file() {
            Ok(InstallReport { entry, build: None })
        } else {
            Err(BackendError::MissingFiles {
                backend: self.id(),
                dir: dir.to_path_buf(),
                missing: vec!["ovms.exe".into()],
            })
        }
    }
    fn build_argv(&self, request: &LaunchRequest) -> Result<LaunchCommand, BackendError> {
        validate_common(request)?;
        refuse_llama_flags(self.id(), request)?;
        refuse(self.id(), "tensor_parallel", request.tensor_parallel.is_some())?;
        refuse(self.id(), "load-time quantization", request.quantization.is_some())?;
        refuse(self.id(), "context", request.context.is_some())?;
        let Location::Native(dir) = &request.location else {
            return Err(BackendError::InvalidRequest("OpenVINO runs natively".into()));
        };
        let notes = keyless(self.id(), request)?;
        let name = request
            .model
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| BackendError::InvalidRequest("model folder name is not valid UTF-8".into()))?;
        let device = match request.openvino_device.unwrap_or(OpenVinoDevice::Cpu) {
            OpenVinoDevice::Cpu => "CPU",
            OpenVinoDevice::Gpu => "GPU",
            OpenVinoDevice::Npu => "NPU",
        };
        Ok(LaunchCommand {
            program: dir.join("ovms.exe"),
            args: vec![
                "--rest_port".into(),
                request.port.to_string(),
                "--rest_bind_address".into(),
                LOOPBACK.into(),
                "--model_path".into(),
                path_text(&request.model)?.to_string(),
                "--model_name".into(),
                name.to_string(),
                "--target_device".into(),
                device.into(),
                // Without a task OVMS serves the folder as a classic model over KServe/TFS, not
                // the OpenAI `/v3/chat/completions` endpoint (modelled from the OVMS docs; not run).
                "--task".into(),
                "text_generation".into(),
            ],
            env: Vec::new(),
            api_key: ApiKeyDelivery::None,
            listener: ListenerOwner::Child,
            notes,
        })
    }
    fn health_check(&self) -> HealthCheck {
        HealthCheck {
            path: "/v2/health/ready",
            ready_status: 200,
            loading_status: None,
            needs_api_key: false,
        }
    }
    fn verification(&self) -> Verification {
        Verification::Unverified
    }
}

/// AMD Ryzen AI through Lemonade Server (pip `lemonade-sdk`); models are loaded through its API
/// after start, so the request's model is not on the command line.
#[derive(Debug, Clone, Copy, Default)]
pub struct RyzenAi;

impl Backend for RyzenAi {
    fn id(&self) -> BackendId {
        BackendId::RyzenAi
    }
    fn model_formats(&self) -> &'static [ModelFormat] {
        &[ModelFormat::OnnxGenAi]
    }
    fn requirements(&self) -> &'static [Requirement] {
        REQ_RYZEN_AI
    }
    fn install(&self) -> InstallKind {
        InstallKind::PythonEnv {
            package: "lemonade-sdk",
            runs_in: PythonHost::Native,
        }
    }
    fn check_install(&self, dir: &Path) -> Result<InstallReport, BackendError> {
        check_python_entry(self.id(), dir, "lemonade-server-dev")
    }
    fn build_argv(&self, request: &LaunchRequest) -> Result<LaunchCommand, BackendError> {
        validate_common(request)?;
        refuse_llama_flags(self.id(), request)?;
        refuse(self.id(), "tensor_parallel", request.tensor_parallel.is_some())?;
        refuse(self.id(), "load-time quantization", request.quantization.is_some())?;
        refuse(self.id(), "openvino_device", request.openvino_device.is_some())?;
        refuse(self.id(), "context", request.context.is_some())?;
        let Location::Native(env) = &request.location else {
            return Err(BackendError::InvalidRequest("Ryzen AI runs natively on Windows".into()));
        };
        let mut notes = keyless(self.id(), request)?;
        notes.push(
            "the model is loaded through the server's API after start, not on its command line".into(),
        );
        Ok(LaunchCommand {
            program: native_script(env, "lemonade-server-dev"),
            args: vec![
                "serve".into(),
                "--host".into(),
                LOOPBACK.into(),
                "--port".into(),
                request.port.to_string(),
            ],
            env: python_env(None, false),
            api_key: ApiKeyDelivery::None,
            listener: ListenerOwner::Child,
            notes,
        })
    }
    fn health_check(&self) -> HealthCheck {
        HealthCheck {
            path: "/api/v1/health",
            ready_status: 200,
            loading_status: None,
            needs_api_key: false,
        }
    }
    fn verification(&self) -> Verification {
        Verification::Unverified
    }
}

/// Qualcomm AI Engine Direct (QNN) on the Hexagon NPU. Modelled for the capability matrix; the
/// SDK ships no OpenAI-compatible server, so there is no launch recipe yet.
#[derive(Debug, Clone, Copy, Default)]
pub struct Qnn;

impl Backend for Qnn {
    fn id(&self) -> BackendId {
        BackendId::Qnn
    }
    fn model_formats(&self) -> &'static [ModelFormat] {
        &[ModelFormat::QnnContext]
    }
    fn requirements(&self) -> &'static [Requirement] {
        REQ_QNN
    }
    fn install(&self) -> InstallKind {
        InstallKind::Folder {
            dir_name: "qualcomm-qnn",
            entry: "genie-t2t-run.exe",
            required: &["genie-t2t-run.exe"],
        }
    }
    fn check_install(&self, dir: &Path) -> Result<InstallReport, BackendError> {
        let entry = dir.join("genie-t2t-run.exe");
        if entry.is_file() {
            Ok(InstallReport { entry, build: None })
        } else {
            Err(BackendError::MissingFiles {
                backend: self.id(),
                dir: dir.to_path_buf(),
                missing: vec!["genie-t2t-run.exe".into()],
            })
        }
    }
    fn launch_recipe(&self) -> LaunchRecipe {
        LaunchRecipe::Missing(QNN_MISSING)
    }
    fn build_argv(&self, _request: &LaunchRequest) -> Result<LaunchCommand, BackendError> {
        Err(BackendError::LaunchUnsupported {
            backend: self.id(),
            reason: QNN_MISSING,
        })
    }
    fn health_check(&self) -> HealthCheck {
        HealthCheck {
            path: "/health",
            ready_status: 200,
            loading_status: None,
            needs_api_key: false,
        }
    }
    fn verification(&self) -> Verification {
        Verification::Unverified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wsl_request() -> LaunchRequest {
        LaunchRequest::new(
            Location::Wsl { distro: "Ubuntu-24.04".into(), env_path: "/home/me/vllm/".into() },
            r"C:\models\Qwen3-8B-FP8",
            8100,
        )
    }

    #[cfg(windows)]
    #[test]
    fn vllm_under_wsl_runs_without_a_shell_and_offline() {
        let mut req = wsl_request();
        req.tensor_parallel = Some(2);
        req.quantization = Some(RuntimeQuant::Fp8);
        req.context = Some(32768);
        let cmd = Vllm.build_argv(&req).unwrap();
        assert!(cmd.program.ends_with("wsl.exe"), "{}", cmd.program.display());
        assert_eq!(
            cmd.args,
            [
                "-d", "Ubuntu-24.04", "--exec", "/home/me/vllm/bin/vllm", "serve",
                "/mnt/c/models/Qwen3-8B-FP8", "--host", "127.0.0.1", "--port", "8100",
                "--max-model-len", "32768", "--tensor-parallel-size", "2", "--quantization", "fp8",
            ]
        );
        assert_eq!(cmd.api_key, ApiKeyDelivery::Env("VLLM_API_KEY"));
        assert!(cmd.env.contains(&("VLLM_API_KEY".into(), EnvValue::ApiKey)));
        assert!(cmd.env.contains(&(
            "WSLENV".into(),
            EnvValue::Literal(
                "HF_HUB_OFFLINE/u:TRANSFORMERS_OFFLINE/u:HF_HUB_DISABLE_TELEMETRY/u:DO_NOT_TRACK/u:\
                 VLLM_NO_USAGE_STATS/u:VLLM_API_KEY/u"
                    .into()
            )
        )));
        // The key is never in the argv.
        assert!(!cmd.args.iter().any(|a| a.contains("api-key")));
        // Under WSL the listener is the relay's, and the /mnt read is noted.
        assert_eq!(cmd.listener, ListenerOwner::WslRelay);
        assert_eq!(cmd.notes, [NOTE_WSL_MOUNT]);
    }

    /// Every telemetry and network opt-out reaches every Python backend, and under WSL each one is
    /// forwarded across the boundary. Dropping any one entry fails here.
    #[cfg(windows)]
    #[test]
    fn python_backends_never_report_usage_or_reach_the_hub() {
        let mut keyless = LaunchRequest::new(Location::Native(r"C:\env".into()), r"C:\models\m", 8000);
        keyless.require_api_key = false;
        let tabby = LaunchRequest::new(Location::Native(r"C:\tabbyAPI".into()), r"C:\models\m", 8000);
        let commands = [
            ("vllm", Vllm.build_argv(&wsl_request()).unwrap()),
            ("sglang", Sglang.build_argv(&wsl_request()).unwrap()),
            ("exllamav3", ExLlamaV3.build_argv(&tabby).unwrap()),
            ("mlx", Mlx.build_argv(&keyless).unwrap()),
            ("ryzen-ai", RyzenAi.build_argv(&keyless).unwrap()),
        ];
        assert_eq!(OFFLINE_ENV.len(), 5);
        for (name, cmd) in &commands {
            for (k, v) in OFFLINE_ENV {
                assert!(
                    cmd.env.contains(&((*k).to_string(), EnvValue::Literal((*v).to_string()))),
                    "{name}: {k} missing"
                );
            }
            let wslenv = cmd.env.iter().find(|(k, _)| k == "WSLENV");
            if let Some((_, EnvValue::Literal(fwd))) = wslenv {
                for (k, _) in &cmd.env {
                    if k != "WSLENV" {
                        assert!(fwd.split(':').any(|f| f == format!("{k}/u")), "{name}: {k} not forwarded");
                    }
                }
            }
        }
        assert!(commands[..2].iter().all(|(_, c)| c.env.iter().any(|(k, _)| k == "WSLENV")));
    }

    #[cfg(windows)]
    #[test]
    fn linux_backends_refuse_a_native_windows_location_and_bad_wsl_fields() {
        let native = LaunchRequest::new(Location::Native(r"C:\venv".into()), r"C:\m", 8100);
        assert!(Vllm.build_argv(&native).is_err());
        let mut bad = wsl_request();
        bad.location = Location::Wsl { distro: "Ubuntu && calc".into(), env_path: "/x".into() };
        assert!(Sglang.build_argv(&bad).is_err());
        let mut unc = wsl_request();
        unc.model = r"\\nas\models\m".into();
        assert!(Vllm.build_argv(&unc).is_err());
        let mut flags = wsl_request();
        flags.llama_flags = vec![cerebro_flags::Flag::GpuLayers(1)];
        assert!(matches!(Vllm.build_argv(&flags), Err(BackendError::OptionUnsupported { .. })));
        // The exec target may not climb out of the environment, and zero sizes are refused.
        for env_path in ["/home/me/../../usr", "/home/me/venv\n"] {
            let mut escape = wsl_request();
            escape.location = Location::Wsl { distro: "Ubuntu-24.04".into(), env_path: env_path.into() };
            assert!(matches!(Vllm.build_argv(&escape), Err(BackendError::InvalidRequest(_))), "{env_path:?}");
        }
        let mut zero_ctx = wsl_request();
        zero_ctx.context = Some(0);
        assert!(matches!(Vllm.build_argv(&zero_ctx), Err(BackendError::InvalidRequest(_))));
        assert!(matches!(Sglang.build_argv(&zero_ctx), Err(BackendError::InvalidRequest(_))));
        let mut zero_tp = wsl_request();
        zero_tp.tensor_parallel = Some(0);
        assert!(matches!(Sglang.build_argv(&zero_tp), Err(BackendError::InvalidRequest(_))));
    }

    /// SGLang takes its key only as `--api-key`: the supervisor appends it (the key is never in
    /// the built argv) and the launch says the argv is readable by same-user processes.
    #[cfg(windows)]
    #[test]
    fn sglang_is_keyed_on_its_command_line_with_a_visibility_note() {
        let cmd = Sglang.build_argv(&wsl_request()).unwrap();
        assert_eq!(&cmd.args[4..6], ["-m", "sglang.launch_server"]);
        assert_eq!(cmd.api_key, ApiKeyDelivery::Argv("--api-key"));
        assert_eq!(cmd.notes, [NOTE_ARGV_KEY, NOTE_WSL_MOUNT]);
        assert!(!cmd.args.iter().any(|a| a.contains("api-key")));
    }

    /// MLX, OpenVINO and Lemonade cannot take a key: refused while the protection is on (the
    /// default), and with it off they build with exactly one keyless note.
    #[test]
    fn keyless_backends_refuse_unless_the_protection_is_off() {
        let env = std::env::temp_dir();
        let mut req = LaunchRequest::new(Location::Native(env.clone()), env.join("model"), 8000);
        let keyless: [&dyn Backend; 3] = [&Mlx, &OpenVino, &RyzenAi];
        for backend in keyless {
            let err = backend.build_argv(&req).unwrap_err();
            assert_eq!(err, BackendError::KeylessRefused { backend: backend.id() });
            assert!(err.to_string().contains("protections.require_api_key"), "{err}");
        }
        req.require_api_key = false;
        for backend in keyless {
            let cmd = backend.build_argv(&req).unwrap();
            assert_eq!(cmd.api_key, ApiKeyDelivery::None);
            assert_eq!(cmd.notes.iter().filter(|n| n.as_str() == NOTE_KEYLESS).count(), 1, "{}", backend.id());
        }
    }

    #[cfg(windows)]
    #[test]
    fn openvino_targets_the_requested_device_on_loopback() {
        let mut req = LaunchRequest::new(
            Location::Native(r"C:\Cerebro\backends\openvino-model-server".into()),
            r"C:\models\phi-4-mini-int4-ov",
            9000,
        );
        req.openvino_device = Some(OpenVinoDevice::Npu);
        req.require_api_key = false;
        let cmd = OpenVino.build_argv(&req).unwrap();
        let argv = cmd.args.join(" ");
        assert!(argv.contains("--rest_bind_address 127.0.0.1"), "{argv}");
        assert!(argv.contains("--target_device NPU"), "{argv}");
        assert!(argv.contains("--model_name phi-4-mini-int4-ov"), "{argv}");
        assert!(argv.ends_with("--task text_generation"), "{argv}");
        let mut ctx = req.clone();
        ctx.context = Some(4096);
        assert!(matches!(OpenVino.build_argv(&ctx), Err(BackendError::OptionUnsupported { .. })));
        req.tensor_parallel = Some(2);
        assert!(OpenVino.build_argv(&req).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn exllamav3_and_ryzen_ai_build_native_commands() {
        let req = LaunchRequest::new(Location::Native(r"C:\tabbyAPI".into()), r"C:\models\Qwen3-8B-exl3", 5000);
        let cmd = ExLlamaV3.build_argv(&req).unwrap();
        assert_eq!(cmd.program, PathBuf::from(r"C:\tabbyAPI\venv\Scripts\python.exe"));
        assert!(cmd.args.join(" ").contains("--model-dir C:\\models --model-name Qwen3-8B-exl3"));
        // Authentication stays on: the supervisor writes the key into TabbyAPI's own token file.
        assert!(!cmd.args.iter().any(|a| a.contains("disable-auth")), "{:?}", cmd.args);
        assert_eq!(
            cmd.api_key,
            ApiKeyDelivery::KeyFile {
                path: PathBuf::from(r"C:\tabbyAPI\api_tokens.yml"),
                format: KeyFileFormat::TabbyApiTokens,
            }
        );
        assert!(cmd.notes.is_empty());
        let mut req = LaunchRequest::new(Location::Native(r"C:\lemonade".into()), r"C:\models\m", 8000);
        req.require_api_key = false;
        let cmd = RyzenAi.build_argv(&req).unwrap();
        assert_eq!(cmd.program, PathBuf::from(r"C:\lemonade\Scripts\lemonade-server-dev.exe"));
        assert_eq!(cmd.notes.len(), 2);
    }

    #[test]
    fn linux_environments_are_checked_in_bin_even_from_windows() {
        let env = std::env::temp_dir().join(format!("cerebro-backends-venv-{}", std::process::id()));
        std::fs::create_dir_all(env.join("bin")).unwrap();
        assert!(matches!(Vllm.check_install(&env), Err(BackendError::MissingFiles { .. })));
        std::fs::write(env.join("bin").join("vllm"), b"#!/usr/bin/env python3
").unwrap();
        assert_eq!(Vllm.check_install(&env).unwrap().entry, env.join("bin").join("vllm"));
        std::fs::remove_dir_all(&env).unwrap();
    }

    #[test]
    fn qnn_has_no_launch_recipe_and_python_backends_have_no_manifest() {
        let req = LaunchRequest::new(Location::Native("/x".into()), "/m", 1);
        assert!(matches!(Qnn.build_argv(&req), Err(BackendError::LaunchUnsupported { .. })));
        assert_eq!(Qnn.launch_recipe(), LaunchRecipe::Missing(QNN_MISSING));
        assert_eq!(Vllm.launch_recipe(), LaunchRecipe::Ready);
        assert!(matches!(
            Vllm.integrity_manifest(Path::new("/nowhere")),
            Err(BackendError::ManifestUnsupported { .. })
        ));
        assert_eq!(Vllm.default_install_dir(Path::new("/data")), None);
        assert_eq!(
            OpenVino.default_install_dir(Path::new("/data")),
            Some(Path::new("/data").join("backends").join("openvino-model-server"))
        );
    }
}
