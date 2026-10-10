//! The one trait every inference backend implements, and the types its methods exchange.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use cerebro_probe::hardware::HardwareProfile;

use crate::error::BackendError;
use crate::feature::Feature;
use crate::requirement::{check_all, Check, Requirement};

/// The loopback address every backend binds. Backends are never exposed beyond the machine;
/// Cerebro's own API is what a LAN setting opens.
pub const LOOPBACK: &str = "127.0.0.1";

/// A llama.cpp build flavour: which ggml GPU backend its binaries carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LlamaCppBuild {
    /// CUDA 12 runtime (`ggml-cuda.dll` + `cudart64_12.dll`): Maxwell and newer.
    Cuda12,
    /// CUDA 13 runtime (`cudart64_13.dll`): Turing and newer, driver 580+; native Blackwell code.
    Cuda13,
    /// Vulkan (`ggml-vulkan.dll`): any GPU with a Vulkan driver.
    Vulkan,
    /// ROCm/HIP (`ggml-hip.dll`): AMD RDNA 2 and newer.
    Hip,
    /// SYCL (`ggml-sycl.dll`): Intel Arc and Xe integrated graphics.
    Sycl,
    /// OpenCL for Adreno (`ggml-opencl.dll`): Snapdragon X on Windows on Arm.
    OpenClAdreno,
    /// Metal: Apple Silicon (future macOS build).
    Metal,
    /// CPU only (`ggml-cpu-*.dll`).
    Cpu,
}

impl LlamaCppBuild {
    /// Every build, in the registry's order.
    pub const ALL: &'static [LlamaCppBuild] = &[
        Self::Cuda12,
        Self::Cuda13,
        Self::Vulkan,
        Self::Hip,
        Self::Sycl,
        Self::OpenClAdreno,
        Self::Metal,
        Self::Cpu,
    ];

    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cuda12 => "CUDA 12",
            Self::Cuda13 => "CUDA 13",
            Self::Vulkan => "Vulkan",
            Self::Hip => "ROCm/HIP",
            Self::Sycl => "SYCL",
            Self::OpenClAdreno => "OpenCL (Adreno)",
            Self::Metal => "Metal",
            Self::Cpu => "CPU",
        }
    }

    /// The device-name prefix this build's `--list-devices` uses (`CUDA0`, `Vulkan0`, `ROCm0`),
    /// upper-cased as `cerebro_flags::DeviceId` canonicalizes it. `None` for the CPU build, which
    /// has no addressable devices.
    pub fn device_prefix(self) -> Option<&'static str> {
        match self {
            Self::Cuda12 | Self::Cuda13 => Some("CUDA"),
            Self::Vulkan => Some("VULKAN"),
            Self::Hip => Some("ROCM"),
            Self::Sycl => Some("SYCL"),
            Self::OpenClAdreno => Some("GPUOPENCL"),
            Self::Metal => Some("MTL"),
            Self::Cpu => None,
        }
    }

    /// Folder name under Cerebro's backend root.
    pub fn dir_name(self) -> &'static str {
        match self {
            Self::Cuda12 => "llama.cpp-cuda12",
            Self::Cuda13 => "llama.cpp-cuda13",
            Self::Vulkan => "llama.cpp-vulkan",
            Self::Hip => "llama.cpp-hip",
            Self::Sycl => "llama.cpp-sycl",
            Self::OpenClAdreno => "llama.cpp-opencl-adreno",
            Self::Metal => "llama.cpp-metal",
            Self::Cpu => "llama.cpp-cpu",
        }
    }
}

impl fmt::Display for LlamaCppBuild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Every backend Cerebro models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BackendId {
    /// llama.cpp `llama-server`, one id per build flavour. The default for GGUF.
    LlamaCpp(LlamaCppBuild),
    /// vLLM (Linux or WSL 2).
    Vllm,
    /// SGLang (Linux or WSL 2).
    Sglang,
    /// ExLlamaV3, served through TabbyAPI (NVIDIA).
    ExLlamaV3,
    /// MLX `mlx_lm.server` (Apple Silicon).
    Mlx,
    /// OpenVINO Model Server (Intel CPU, GPU and NPU).
    OpenVino,
    /// AMD Ryzen AI through Lemonade Server (XDNA 2 NPU).
    RyzenAi,
    /// Qualcomm AI Engine Direct (QNN) on the Hexagon NPU.
    Qnn,
}

impl BackendId {
    /// Display name, e.g. `llama.cpp (CUDA 12)`.
    pub fn name(self) -> String {
        match self {
            Self::LlamaCpp(b) => format!("llama.cpp ({b})"),
            Self::Vllm => "vLLM".into(),
            Self::Sglang => "SGLang".into(),
            Self::ExLlamaV3 => "ExLlamaV3".into(),
            Self::Mlx => "MLX".into(),
            Self::OpenVino => "OpenVINO".into(),
            Self::RyzenAi => "Ryzen AI".into(),
            Self::Qnn => "Qualcomm QNN".into(),
        }
    }
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

/// A model file or folder format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ModelFormat {
    /// A `.gguf` file.
    Gguf,
    /// A Hugging Face safetensors folder (FP16/BF16, FP8, NVFP4, AWQ, GPTQ).
    Safetensors,
    /// An EXL3 folder.
    Exl3,
    /// An MLX folder.
    Mlx,
    /// An OpenVINO IR folder.
    OpenVinoIr,
    /// An ONNX Runtime GenAI folder (Ryzen AI).
    OnnxGenAi,
    /// A QNN context-binary bundle.
    QnnContext,
}

/// How a backend is installed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallKind {
    /// A self-contained folder of native binaries Cerebro manages and can pin by hash.
    Folder {
        /// Folder name under Cerebro's backend root.
        dir_name: &'static str,
        /// The executable launched (Windows name).
        entry: &'static str,
        /// Files that must be present (Windows names).
        required: &'static [&'static str],
    },
    /// A Python environment (a venv on this machine, or one inside a WSL 2 distribution).
    PythonEnv {
        /// The pip package.
        package: &'static str,
        /// Where the environment must live.
        runs_in: PythonHost,
    },
}

/// Where a Python-environment backend runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PythonHost {
    /// On this OS directly.
    Native,
    /// On native Linux, or inside a WSL 2 distribution on Windows.
    LinuxOrWsl2,
}

/// Where an installed backend lives, as passed to [`Backend::build_argv`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    /// A folder (or Python environment root) on this machine.
    Native(PathBuf),
    /// A Python environment inside a WSL 2 distribution.
    Wsl {
        /// Distribution name (`Ubuntu-24.04`).
        distro: String,
        /// The environment root as a Linux path (`/home/me/.venvs/vllm`).
        env_path: String,
    },
}

/// SHA-256 of every `.exe`/`.dll` in a backend folder, keyed by file name: the same shape the
/// supervisor's preflight pins and verifies before every launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrityManifest {
    /// The canonical directory the hashes cover: every junction or symlink on the given path
    /// resolved once, with the same probe function preflight resolves through, so the pin records
    /// the directory it describes (a scoop-style `current` junction repointed later does not
    /// silently carry the pin to another folder).
    pub resolved_dir: PathBuf,
    /// File name to lowercase hex SHA-256.
    pub files: BTreeMap<String, String>,
}

/// What [`Backend::check_install`] established.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallReport {
    /// The executable that will be launched.
    pub entry: PathBuf,
    /// For llama.cpp folders, the build the files identify.
    pub build: Option<LlamaCppBuild>,
}

/// The HTTP request that tells whether a launched backend is ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HealthCheck {
    /// Path to `GET`.
    pub path: &'static str,
    /// Status meaning ready.
    pub ready_status: u16,
    /// Status meaning still loading, when the backend distinguishes it.
    pub loading_status: Option<u16>,
    /// Whether the request must carry the API key.
    pub needs_api_key: bool,
}

/// Whether Cerebro has run a backend end to end, as CLAIMS.md requires before claiming it works.
///
/// This is about the backend launching with Cerebro's argv and passing its health check, nothing
/// more: which *features* have been run on it is per (backend, feature) pair
/// ([`crate::matrix::VERIFIED_FEATURES`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verification {
    /// Launched and health-checked by Cerebro's live tests with this crate's argv.
    Verified,
    /// Modelled from upstream documentation; its argv and health check have not been run.
    Unverified,
}

/// Whether Cerebro can build a launch for a backend at all, independently of the hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchRecipe {
    /// [`Backend::build_argv`] can produce a launch.
    Ready,
    /// No launch recipe exists yet; the backend is modelled for the capability matrix only and is
    /// never offered as a candidate or shown as available.
    Missing(&'static str),
}

/// A runtime quantization a backend applies at load (not a pre-quantized checkpoint).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeQuant {
    /// Dynamic FP8 weight quantization (vLLM/SGLang `--quantization fp8`).
    Fp8,
}

/// The accelerator an OpenVINO model is compiled for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenVinoDevice {
    /// The CPU.
    Cpu,
    /// The Intel GPU.
    Gpu,
    /// The Intel NPU.
    Npu,
}

/// Everything needed to build one launch command.
#[derive(Debug, Clone, PartialEq)]
pub struct LaunchRequest {
    /// Where the backend is installed.
    pub location: Location,
    /// The model: an absolute path on this machine (offline; never a hub id).
    pub model: PathBuf,
    /// Loopback port.
    pub port: u16,
    /// API key file (llama.cpp reads the key from it; other backends get it by environment).
    pub api_key_file: Option<PathBuf>,
    /// Context length, when set (unset leaves the backend's own default).
    pub context: Option<u32>,
    /// Tensor-parallel GPU count (vLLM, SGLang).
    pub tensor_parallel: Option<u32>,
    /// Load-time quantization (vLLM, SGLang).
    pub quantization: Option<RuntimeQuant>,
    /// OpenVINO target device.
    pub openvino_device: Option<OpenVinoDevice>,
    /// Tuned llama.cpp flags (llama.cpp only), through the closed allowlist.
    pub llama_flags: Vec<cerebro_flags::Flag>,
    /// The user's `require_api_key` protection (on by default; settings key
    /// `protections.require_api_key`, mapped to `cerebro_supervisor::Protections::require_api_key`).
    /// On, a backend that cannot take an API key refuses to build a launch
    /// ([`BackendError::KeylessRefused`]); off lets it run keyless on loopback, recorded once as a
    /// note on the launch.
    pub require_api_key: bool,
}

impl LaunchRequest {
    /// A request with only the required fields set (and the API-key protection on).
    pub fn new(location: Location, model: impl Into<PathBuf>, port: u16) -> Self {
        Self {
            location,
            model: model.into(),
            port,
            api_key_file: None,
            context: None,
            tensor_parallel: None,
            quantization: None,
            openvino_device: None,
            llama_flags: Vec::new(),
            require_api_key: true,
        }
    }
}

/// An environment variable the launch needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvValue {
    /// A fixed value.
    Literal(String),
    /// The API key: the supervisor reads it from the key file and sets it, so it never appears in
    /// an argv other local processes can read.
    ApiKey,
}

/// A key file a backend reads in its own format, which the supervisor writes before launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyFileFormat {
    /// TabbyAPI's `api_tokens.yml` (`api_key` and `admin_key`). Both are the launch key: Cerebro
    /// is the only client and loads models through the admin endpoints.
    TabbyApiTokens,
}

impl KeyFileFormat {
    /// The file's contents for `key`. The key must be printable ASCII without quotes or
    /// backslashes, so it can never break out of its scalar (the supervisor's generated keys are
    /// URL-safe base64).
    pub fn render(self, key: &str) -> Result<String, BackendError> {
        let safe = !key.is_empty()
            && key.bytes().all(|b| b.is_ascii_graphic() && !matches!(b, b'\'' | b'"' | b'\\'));
        if !safe {
            return Err(BackendError::InvalidRequest(
                "an API key for a key file must be printable ASCII without quotes or backslashes".into(),
            ));
        }
        match self {
            Self::TabbyApiTokens => Ok(format!("api_key: '{key}'\nadmin_key: '{key}'\n")),
        }
    }
}

/// How the backend receives the API key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiKeyDelivery {
    /// A key-file path in the argv (llama.cpp `--api-key-file`).
    File,
    /// An environment variable ([`EnvValue::ApiKey`] in [`LaunchCommand::env`]).
    Env(&'static str),
    /// The supervisor appends this flag and the key to the argv. Other processes of the same user
    /// can read an argv (on WSL also through `\\wsl.localhost\<distro>\proc`), so this is used
    /// only where the backend takes a key no other way, and [`LaunchCommand::notes`] says so. Still
    /// far less exposure than no key.
    Argv(&'static str),
    /// The supervisor writes the key to `path` in `format`, with an owner-only ACL, before launch
    /// (and removes it after the child exits).
    KeyFile {
        /// Where the backend reads it.
        path: PathBuf,
        /// Its format.
        format: KeyFileFormat,
    },
    /// The backend runs keyless on loopback: only when the user turned the `require_api_key`
    /// protection off ([`LaunchRequest::require_api_key`]); [`LaunchCommand::notes`] records it.
    None,
}

/// Which process owns the backend's loopback listener, for the supervisor's PID-attributed
/// readiness check (it sends nothing until every listener on the port belongs to the expected
/// process).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListenerOwner {
    /// The launched child itself (or a process in its job).
    Child,
    /// WSL's localhost relay: under WSL 2 the server listens inside the VM and Windows' side of the
    /// port is owned by WSL's relay process, never by the `wsl.exe` child. The supervisor must
    /// attribute the listener to the relay (or check readiness another way) for these launches.
    WslRelay,
}

/// A fully built launch: program, arguments and the environment it needs on top of the
/// supervisor's allowlisted environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchCommand {
    /// Absolute path of the program.
    pub program: PathBuf,
    /// Arguments, each one argv slot (no shell is involved).
    pub args: Vec<String>,
    /// Extra environment variables.
    pub env: Vec<(String, EnvValue)>,
    /// How the API key reaches the backend.
    pub api_key: ApiKeyDelivery,
    /// Who owns the loopback listener once it is up.
    pub listener: ListenerOwner,
    /// Neutral notes to record with the launch (keyless backend, ignored model, ...).
    pub notes: Vec<String>,
}

/// One inference backend: its gate, install, integrity, launch and readiness check.
pub trait Backend: fmt::Debug + Send + Sync {
    /// Which backend.
    fn id(&self) -> BackendId;

    /// Model formats it loads.
    fn model_formats(&self) -> &'static [ModelFormat];

    /// Hardware and OS requirements (all must hold).
    fn requirements(&self) -> &'static [Requirement];

    /// Whether this machine can run it.
    fn support(&self, hw: &HardwareProfile) -> Check {
        check_all(self.requirements(), hw)
    }

    /// Features it can provide, before their own hardware conditions (from the capability table).
    fn features(&self) -> Vec<Feature> {
        crate::matrix::features_of(self.id())
    }

    /// How it is installed.
    fn install(&self) -> InstallKind;

    /// Whether Cerebro can build a launch for it at all. [`LaunchRecipe::Missing`] keeps it out of
    /// selection and out of the matrix's "available", whatever the hardware.
    fn launch_recipe(&self) -> LaunchRecipe {
        LaunchRecipe::Ready
    }

    /// Default install folder under Cerebro's data root, for folder installs.
    fn default_install_dir(&self, data_root: &Path) -> Option<PathBuf> {
        match self.install() {
            InstallKind::Folder { dir_name, .. } => Some(data_root.join("backends").join(dir_name)),
            InstallKind::PythonEnv { .. } => None,
        }
    }

    /// Check that `dir` holds a usable install of this backend (files present, right build).
    fn check_install(&self, dir: &Path) -> Result<InstallReport, BackendError>;

    /// Compute the integrity manifest of the install in `dir`, to pin it: `dir` is resolved once
    /// (every junction or symlink followed, as preflight does) and the resolved directory is what
    /// is hashed and recorded in [`IntegrityManifest::resolved_dir`].
    fn integrity_manifest(&self, dir: &Path) -> Result<IntegrityManifest, BackendError> {
        match self.install() {
            InstallKind::Folder { .. } => {
                let hash_error = |cause: String| BackendError::Hash {
                    backend: self.id(),
                    cause,
                };
                let resolved = resolve_backend_dir(dir).map_err(hash_error)?;
                let (files, _) = cerebro_probe::backend::hash_directory(&resolved)
                    .map_err(|e| hash_error(e.to_string()))?;
                Ok(IntegrityManifest {
                    resolved_dir: resolved,
                    files,
                })
            }
            InstallKind::PythonEnv { .. } => Err(BackendError::ManifestUnsupported {
                backend: self.id(),
                reason: "a Python environment changes on every package upgrade and loads code \
                         from many directories; integrity verification is unavailable for it",
            }),
        }
    }

    /// Build the launch command.
    fn build_argv(&self, request: &LaunchRequest) -> Result<LaunchCommand, BackendError>;

    /// How readiness is checked once launched.
    fn health_check(&self) -> HealthCheck;

    /// Whether Cerebro has run it end to end.
    fn verification(&self) -> Verification;
}

/// Resolve a backend folder once, through the same probe function preflight uses when it does not
/// audit (`canonical_backend_dir_unaudited`: every reparse point followed, no volume
/// precondition), so a pinned manifest and preflight's later verification name one directory.
#[cfg(windows)]
pub(crate) fn resolve_backend_dir(dir: &Path) -> Result<PathBuf, String> {
    cerebro_probe::backend::canonical_backend_dir_unaudited(dir).map_err(|e| e.to_string())
}

/// Elsewhere there are no junctions; symlinks are resolved by the OS.
#[cfg(not(windows))]
pub(crate) fn resolve_backend_dir(dir: &Path) -> Result<PathBuf, String> {
    std::fs::canonicalize(dir).map_err(|e| format!("{}: {e}", dir.display()))
}

/// Shared request checks: a non-zero port, an absolute model path with no NUL, and a tensor-parallel
/// size of at least one.
pub(crate) fn validate_common(request: &LaunchRequest) -> Result<(), BackendError> {
    if request.port == 0 {
        return Err(BackendError::InvalidRequest("port 0 is not a fixed loopback port".into()));
    }
    if request.tensor_parallel == Some(0) {
        return Err(BackendError::InvalidRequest("tensor_parallel must be at least 1".into()));
    }
    if !request.model.is_absolute() {
        return Err(BackendError::InvalidRequest(format!(
            "model path {} is not absolute",
            request.model.display()
        )));
    }
    let text = path_text(&request.model)?;
    if text.contains('\0') {
        return Err(BackendError::InvalidRequest("model path contains NUL".into()));
    }
    Ok(())
}

/// For backends where a context length of 0 is not a "use the model's own" sentinel (every one but
/// llama.cpp, whose `-c 0` is): refuse it instead of letting the backend fail after its load.
pub(crate) fn validate_context_length(request: &LaunchRequest) -> Result<(), BackendError> {
    if request.context == Some(0) {
        Err(BackendError::InvalidRequest(
            "context 0 is not a length; leave it unset for the backend's default".into(),
        ))
    } else {
        Ok(())
    }
}

/// A path as UTF-8 text for an argv slot.
pub(crate) fn path_text(path: &Path) -> Result<&str, BackendError> {
    path.to_str().ok_or_else(|| {
        BackendError::InvalidRequest(format!("path {} is not valid UTF-8", path.display()))
    })
}

/// Translate an absolute Windows path (`C:\models\m`) to its WSL mount path (`/mnt/c/models/m`).
/// UNC and relative paths are refused: WSL mounts only drive letters.
pub fn to_wsl_path(path: &Path) -> Result<String, BackendError> {
    let text = path_text(path)?;
    let bytes = text.as_bytes();
    match bytes {
        [drive, b':', b'\\' | b'/', ..] if drive.is_ascii_alphabetic() => {
            let rest = text.get(3..).unwrap_or_default().replace('\\', "/");
            let drive = char::from(drive.to_ascii_lowercase());
            Ok(if rest.is_empty() {
                format!("/mnt/{drive}")
            } else {
                format!("/mnt/{drive}/{rest}")
            })
        }
        _ => Err(BackendError::InvalidRequest(format!(
            "{text} is not a drive-letter path WSL can mount"
        ))),
    }
}

/// A WSL distribution name as `wsl.exe -d` accepts it: letters, digits, `.`, `_`, `-`.
pub(crate) fn validate_distro(distro: &str) -> Result<(), BackendError> {
    let ok = !distro.is_empty()
        && distro
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if ok {
        Ok(())
    } else {
        Err(BackendError::InvalidRequest(format!("{distro:?} is not a WSL distribution name")))
    }
}

/// A normalized absolute Linux path for an environment root inside WSL: no `.` or `..` segment
/// (the exec target must stay inside the environment) and no control character (NUL included).
pub(crate) fn validate_linux_path(path: &str) -> Result<(), BackendError> {
    let ok = path.starts_with('/')
        && !path.chars().any(char::is_control)
        && !path.split('/').any(|seg| seg == ".." || seg == ".");
    if ok {
        Ok(())
    } else {
        Err(BackendError::InvalidRequest(format!(
            "{path:?} is not a normalized absolute Linux path"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_paths_map_to_wsl_mounts() {
        assert_eq!(to_wsl_path(Path::new(r"C:\models\Qwen3 8B")).unwrap(), "/mnt/c/models/Qwen3 8B");
        assert_eq!(to_wsl_path(Path::new("D:/m")).unwrap(), "/mnt/d/m");
        assert_eq!(to_wsl_path(Path::new(r"E:\")).unwrap(), "/mnt/e");
        for bad in [r"\\server\share\m", r"models\m", "/home/m", r"C:models"] {
            assert!(to_wsl_path(Path::new(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn distro_names_and_linux_paths_are_validated() {
        assert!(validate_distro("Ubuntu-24.04").is_ok());
        assert!(validate_distro("").is_err());
        assert!(validate_distro("Ubuntu; rm -rf /").is_err());
        assert!(validate_linux_path("/home/me/venv").is_ok());
        assert!(validate_linux_path("/home/me/venv/").is_ok());
        assert!(validate_linux_path("home/me").is_err());
        for bad in ["/home/me/../../usr", "/home/./me", "/home/me/..", "/home/me\n/x", "/x\0y", "/x\u{1b}[0m"] {
            assert!(validate_linux_path(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn zero_tensor_parallel_and_zero_context_are_refused() {
        let mut req = LaunchRequest::new(Location::Native("/env".into()), std::env::temp_dir(), 8000);
        assert!(validate_common(&req).is_ok());
        req.tensor_parallel = Some(0);
        assert!(matches!(validate_common(&req), Err(BackendError::InvalidRequest(_))));
        req.tensor_parallel = Some(1);
        req.context = Some(0);
        assert!(validate_common(&req).is_ok(), "context 0 is llama.cpp's sentinel, refused per backend");
        assert!(matches!(validate_context_length(&req), Err(BackendError::InvalidRequest(_))));
        req.context = Some(4096);
        assert!(validate_context_length(&req).is_ok());
    }

    #[test]
    fn tabby_key_files_carry_the_key_and_refuse_unsafe_ones() {
        let text = KeyFileFormat::TabbyApiTokens.render("abc-DEF_123").unwrap();
        assert_eq!(text, "api_key: 'abc-DEF_123'\nadmin_key: 'abc-DEF_123'\n");
        for bad in ["", "a'b", "a\"b", "a\\b", "a b", "a\nadmin_key: x"] {
            assert!(KeyFileFormat::TabbyApiTokens.render(bad).is_err(), "{bad:?}");
        }
    }

    /// The manifest hashes the resolved directory and records it; a missing folder is a typed
    /// hash error, not a panic.
    #[test]
    fn manifests_record_the_resolved_directory() {
        let backend = crate::llamacpp::LlamaCpp::new(LlamaCppBuild::Vulkan);
        let dir = std::env::temp_dir().join(format!("cerebro-backends-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("llama-server.exe"), b"server").unwrap();
        let manifest = backend.integrity_manifest(&dir).unwrap();
        assert_eq!(manifest.resolved_dir, resolve_backend_dir(&dir).unwrap());
        assert!(manifest.files.contains_key("llama-server.exe"), "{:?}", manifest.files);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(matches!(backend.integrity_manifest(&dir), Err(BackendError::Hash { .. })));
    }

    #[test]
    fn device_prefixes_match_the_flag_crate_canonical_form() {
        for build in LlamaCppBuild::ALL {
            if let Some(prefix) = build.device_prefix() {
                let id = cerebro_flags::DeviceId::parse(&format!("{prefix}0")).unwrap();
                assert_eq!(id.backend(), prefix);
            }
        }
        // `ROCm0` as llama.cpp prints it canonicalizes to the HIP prefix.
        assert_eq!(cerebro_flags::DeviceId::parse("ROCm0").unwrap().backend(), "ROCM");
    }
}
