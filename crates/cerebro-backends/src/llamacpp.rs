//! llama.cpp `llama-server`, one backend per official build flavour: the default for GGUF.
//!
//! The flavour is a property of the files in the folder (`ggml-cuda.dll` + `cudart64_12.dll` is
//! the CUDA 12 build), so [`identify_build`] reads it from the file names and
//! [`LlamaCpp::check_install`] refuses a folder whose files name a different build than the one
//! it is registered as: pointing the CUDA 13 entry at a Vulkan folder would otherwise launch,
//! benchmark and record results against the wrong binary.

use std::path::Path;

use cerebro_flags::{ArgvBuilder, Dialect, DeviceList, Flag};
use cerebro_probe::hardware::{AmdArch, CpuArch, GpuVendor, IntelArch, NvidiaArch, OsFamily};

use crate::backend::{
    path_text, resolve_backend_dir, validate_common, ApiKeyDelivery, Backend, BackendId,
    HealthCheck, InstallKind, InstallReport, LaunchCommand, LaunchRequest, ListenerOwner,
    LlamaCppBuild, Location, ModelFormat, Verification, LOOPBACK,
};
use crate::error::BackendError;
use crate::requirement::Requirement;

/// Every NVIDIA architecture in the table: CUDA 12 still builds for Maxwell.
const NVIDIA_ALL: &[NvidiaArch] = &[
    NvidiaArch::Maxwell,
    NvidiaArch::Pascal,
    NvidiaArch::Volta,
    NvidiaArch::Turing,
    NvidiaArch::Ampere,
    NvidiaArch::Ada,
    NvidiaArch::Hopper,
    NvidiaArch::BlackwellDatacenter,
    NvidiaArch::BlackwellClient,
];

/// CUDA 13 dropped Maxwell, Pascal and Volta.
pub(crate) const NVIDIA_TURING_PLUS: &[NvidiaArch] = &[
    NvidiaArch::Turing,
    NvidiaArch::Ampere,
    NvidiaArch::Ada,
    NvidiaArch::Hopper,
    NvidiaArch::BlackwellDatacenter,
    NvidiaArch::BlackwellClient,
];

const WINDOWS_LINUX: &[OsFamily] = &[OsFamily::Windows, OsFamily::Linux];

const REQ_CUDA12: &[Requirement] = &[
    Requirement::Os(WINDOWS_LINUX),
    Requirement::NvidiaArch(NVIDIA_ALL),
    // CUDA 12.x minor-version compatibility floor (527.41 on Windows, 525.60 on Linux).
    Requirement::NvidiaDriverAtLeast { windows: (527, 41), linux: (525, 60) },
];
const REQ_CUDA13: &[Requirement] = &[
    Requirement::Os(WINDOWS_LINUX),
    Requirement::NvidiaArch(NVIDIA_TURING_PLUS),
    // CUDA 13.x minor-version compatibility floor: the r580 driver branch on both OSes.
    Requirement::NvidiaDriverAtLeast { windows: (580, 0), linux: (580, 0) },
];
const REQ_VULKAN: &[Requirement] = &[
    Requirement::Os(WINDOWS_LINUX),
    Requirement::Gpus {
        vendors: &[],
        min: 1,
        discrete_only: false,
    },
    Requirement::VulkanLoader,
];
const REQ_HIP: &[Requirement] = &[
    Requirement::Os(WINDOWS_LINUX),
    Requirement::AmdArch(&[
        AmdArch::Rdna2,
        AmdArch::Rdna3,
        AmdArch::Rdna35,
        AmdArch::Rdna4,
        AmdArch::Cdna2,
        AmdArch::Cdna3,
    ]),
    Requirement::HipRuntime,
];
const REQ_SYCL: &[Requirement] = &[
    Requirement::Os(WINDOWS_LINUX),
    Requirement::NativeArch(&[CpuArch::X86_64]),
    Requirement::IntelArch(&[
        IntelArch::XeLp,
        IntelArch::XeLpg,
        IntelArch::Xe2Lpg,
        IntelArch::Alchemist,
        IntelArch::Battlemage,
    ]),
];
const REQ_OPENCL_ADRENO: &[Requirement] = &[
    Requirement::Os(&[OsFamily::Windows]),
    Requirement::NativeArch(&[CpuArch::Aarch64]),
    Requirement::Gpus {
        vendors: &[GpuVendor::Qualcomm],
        min: 1,
        discrete_only: false,
    },
];
const REQ_METAL: &[Requirement] = &[
    Requirement::Os(&[OsFamily::MacOs]),
    Requirement::NativeArch(&[CpuArch::Aarch64]),
    Requirement::Gpus {
        vendors: &[GpuVendor::Apple],
        min: 1,
        discrete_only: false,
    },
];
const REQ_CPU: &[Requirement] = &[Requirement::NativeArch(&[CpuArch::X86_64, CpuArch::Aarch64])];

/// The server executable's file name on this platform.
pub const SERVER_EXE: &str = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };

/// Files every Windows build folder holds besides its ggml GPU library.
const COMMON_FILES: &[&str] = &["llama-server.exe", "llama.dll", "ggml.dll", "ggml-base.dll"];

/// A file name reduced for comparison: lower case, `lib` prefix and library suffix removed
/// (`libggml-vulkan.so` and `ggml-vulkan.dll` both become `ggml-vulkan`).
fn stem(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    let lower = lower.strip_prefix("lib").unwrap_or(&lower);
    let cut = [".dll", ".so", ".dylib"]
        .iter()
        .filter_map(|ext| lower.find(ext))
        .min()
        .unwrap_or(lower.len());
    lower.get(..cut).unwrap_or(lower).to_string()
}

/// Identify a build from the file names in its folder.
///
/// Exactly one ggml GPU library may be present (a folder with two is a custom build Cerebro does
/// not guess about). A CUDA library needs its runtime beside it, and the runtime's major decides
/// 12 versus 13: `cudart64_12.dll` on Windows, `libcudart.so.12*` on Linux (matched on the raw
/// lower-cased name, because the versioned `.so.12.4.127` suffix is what carries the major). No GPU
/// library and at least one `ggml-cpu*` library is the CPU build.
pub fn identify_build(file_names: &[String]) -> Result<LlamaCppBuild, String> {
    let stems: Vec<String> = file_names.iter().map(|n| stem(n)).collect();
    let lower: Vec<String> = file_names.iter().map(|n| n.to_ascii_lowercase()).collect();
    let has = |s: &str| stems.iter().any(|x| x == s);
    let gpu: Vec<&str> = [
        "ggml-cuda",
        "ggml-vulkan",
        "ggml-hip",
        "ggml-sycl",
        "ggml-opencl",
        "ggml-metal",
    ]
    .into_iter()
    .filter(|lib| has(lib))
    .collect();
    match gpu.as_slice() {
        [] if stems.iter().any(|s| s.starts_with("ggml-cpu")) => Ok(LlamaCppBuild::Cpu),
        [] => Err("no ggml backend library (ggml-cpu*, ggml-cuda, ggml-vulkan, ...) found".into()),
        ["ggml-cuda"] => {
            let runtime = |major: &str| {
                let dll = format!("cudart64_{major}.dll");
                let so = format!("libcudart.so.{major}");
                lower.iter().any(|n| {
                    *n == dll
                        || n.strip_prefix(&so)
                            .is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
                })
            };
            let (r12, r13) = (runtime("12"), runtime("13"));
            match (r12, r13) {
                (true, false) => Ok(LlamaCppBuild::Cuda12),
                (false, true) => Ok(LlamaCppBuild::Cuda13),
                (true, true) => Err("both CUDA 12 and CUDA 13 runtimes are present".into()),
                (false, false) => Err(
                    "ggml-cuda is present but no CUDA runtime (cudart64_12/13.dll, \
                     libcudart.so.12/13) is beside it; the runtime ships as a separate cudart \
                     download"
                        .into(),
                ),
            }
        }
        ["ggml-vulkan"] => Ok(LlamaCppBuild::Vulkan),
        ["ggml-hip"] => Ok(LlamaCppBuild::Hip),
        ["ggml-sycl"] => Ok(LlamaCppBuild::Sycl),
        ["ggml-opencl"] => Ok(LlamaCppBuild::OpenClAdreno),
        ["ggml-metal"] => Ok(LlamaCppBuild::Metal),
        several => Err(format!("several GPU backends present: {}", several.join(", "))),
    }
}

/// One llama.cpp build flavour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaCpp {
    build: LlamaCppBuild,
}

impl LlamaCpp {
    /// The backend for one build flavour.
    pub fn new(build: LlamaCppBuild) -> Self {
        Self { build }
    }

    /// Which flavour.
    pub fn build(&self) -> LlamaCppBuild {
        self.build
    }

    /// Refuse `-dev`/`-devd` devices this build cannot address (a `Vulkan0` handed to the CUDA
    /// build fails at load, after the model read). RPC devices are addressable from every build.
    fn check_devices(&self, flag: &Flag) -> Result<(), BackendError> {
        let list = match flag {
            Flag::Device(list) | Flag::SpecDraftDevice(list) => list,
            _ => return Ok(()),
        };
        let DeviceList::Devices(ids) = list else {
            return Ok(());
        };
        for id in ids {
            let ok = id.backend() == "RPC" || Some(id.backend()) == self.build.device_prefix();
            if !ok {
                return Err(BackendError::InvalidRequest(format!(
                    "device {id} is not addressable by the llama.cpp {} build",
                    self.build
                )));
            }
        }
        Ok(())
    }
}

impl Backend for LlamaCpp {
    fn id(&self) -> BackendId {
        BackendId::LlamaCpp(self.build)
    }

    fn model_formats(&self) -> &'static [ModelFormat] {
        &[ModelFormat::Gguf]
    }

    fn requirements(&self) -> &'static [Requirement] {
        match self.build {
            LlamaCppBuild::Cuda12 => REQ_CUDA12,
            LlamaCppBuild::Cuda13 => REQ_CUDA13,
            LlamaCppBuild::Vulkan => REQ_VULKAN,
            LlamaCppBuild::Hip => REQ_HIP,
            LlamaCppBuild::Sycl => REQ_SYCL,
            LlamaCppBuild::OpenClAdreno => REQ_OPENCL_ADRENO,
            LlamaCppBuild::Metal => REQ_METAL,
            LlamaCppBuild::Cpu => REQ_CPU,
        }
    }

    fn install(&self) -> InstallKind {
        InstallKind::Folder {
            dir_name: self.build.dir_name(),
            entry: SERVER_EXE,
            required: COMMON_FILES,
        }
    }

    fn check_install(&self, dir: &Path) -> Result<InstallReport, BackendError> {
        // The resolved directory, as the manifest and preflight see it: the files identified and
        // the entry reported are the ones a launch from it would use.
        let resolved = resolve_backend_dir(dir).map_err(|cause| BackendError::DirUnreadable {
            dir: dir.to_path_buf(),
            cause,
        })?;
        let entries = std::fs::read_dir(&resolved).map_err(|e| BackendError::DirUnreadable {
            dir: dir.to_path_buf(),
            cause: e.to_string(),
        })?;
        let mut names = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| BackendError::DirUnreadable {
                dir: dir.to_path_buf(),
                cause: e.to_string(),
            })?;
            if let Some(name) = entry.file_name().to_str() {
                names.push(name.to_string());
            }
        }
        // Windows folders carry the server and its import libraries side by side; elsewhere the
        // server binary is the one fixed name (shared-library names vary by packaging).
        let required: &[&str] = if cfg!(windows) { COMMON_FILES } else { &[SERVER_EXE] };
        let missing: Vec<String> = required
            .iter()
            .filter(|f| !names.iter().any(|n| n.eq_ignore_ascii_case(f)))
            .map(|f| (*f).to_string())
            .collect();
        if !missing.is_empty() {
            return Err(BackendError::MissingFiles {
                backend: self.id(),
                dir: dir.to_path_buf(),
                missing,
            });
        }
        let found = identify_build(&names).map_err(|detail| BackendError::UnidentifiedBuild {
            dir: dir.to_path_buf(),
            detail,
        })?;
        if found != self.build {
            return Err(BackendError::WrongBuild {
                dir: dir.to_path_buf(),
                expected: self.build,
                found,
            });
        }
        Ok(InstallReport {
            entry: resolved.join(SERVER_EXE),
            build: Some(found),
        })
    }

    fn build_argv(&self, request: &LaunchRequest) -> Result<LaunchCommand, BackendError> {
        validate_common(request)?;
        let Location::Native(dir) = &request.location else {
            return Err(BackendError::OptionUnsupported {
                backend: self.id(),
                option: "a WSL location",
                hint: "llama.cpp runs natively; install the Windows build",
            });
        };
        if request.tensor_parallel.is_some() {
            return Err(BackendError::OptionUnsupported {
                backend: self.id(),
                option: "tensor_parallel",
                hint: "split across GPUs with Flag::SplitMode and Flag::TensorSplit",
            });
        }
        if request.quantization.is_some() {
            return Err(BackendError::OptionUnsupported {
                backend: self.id(),
                option: "load-time quantization",
                hint: "GGUF files are already quantized; pick a different GGUF",
            });
        }
        if request.openvino_device.is_some() {
            return Err(BackendError::OptionUnsupported {
                backend: self.id(),
                option: "openvino_device",
                hint: "choose devices with Flag::Device",
            });
        }
        let key_file = request.api_key_file.as_deref().ok_or_else(|| {
            BackendError::InvalidRequest("llama.cpp is always launched with an API key file".into())
        })?;

        let mut builder = ArgvBuilder::new(Dialect::Server);
        builder.push(Flag::Model(path_text(&request.model)?.to_string()))?;
        builder.harden_server(LOOPBACK, request.port, path_text(key_file)?)?;
        if let Some(ctx) = request.context {
            builder.push(Flag::CtxSize(ctx))?;
        }
        for flag in &request.llama_flags {
            self.check_devices(flag)?;
            builder.push(flag.clone())?;
        }
        Ok(LaunchCommand {
            program: dir.join(SERVER_EXE),
            args: builder.build()?,
            env: Vec::new(),
            api_key: ApiKeyDelivery::File,
            listener: ListenerOwner::Child,
            notes: Vec::new(),
        })
    }

    fn health_check(&self) -> HealthCheck {
        HealthCheck {
            path: "/health",
            ready_status: 200,
            loading_status: Some(503),
            needs_api_key: false,
        }
    }

    fn verification(&self) -> Verification {
        match self.build {
            // The official CUDA 12 build is what the supervisor and bench live tests run.
            LlamaCppBuild::Cuda12 => Verification::Verified,
            _ => Verification::Unverified,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerebro_flags::value::{CacheType, DeviceId};
    use std::path::PathBuf;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn builds_are_identified_from_their_files() {
        let cuda12 = names(&["llama-server.exe", "ggml-cuda.dll", "cudart64_12.dll", "ggml-cpu-haswell.dll"]);
        assert_eq!(identify_build(&cuda12), Ok(LlamaCppBuild::Cuda12));
        let cuda13 = names(&["ggml-cuda.dll", "cudart64_13.dll", "cublas64_13.dll"]);
        assert_eq!(identify_build(&cuda13), Ok(LlamaCppBuild::Cuda13));
        assert_eq!(identify_build(&names(&["GGML-VULKAN.DLL", "ggml-cpu-x64.dll"])), Ok(LlamaCppBuild::Vulkan));
        assert_eq!(identify_build(&names(&["ggml-hip.dll"])), Ok(LlamaCppBuild::Hip));
        assert_eq!(identify_build(&names(&["ggml-sycl.dll"])), Ok(LlamaCppBuild::Sycl));
        assert_eq!(identify_build(&names(&["ggml-opencl.dll"])), Ok(LlamaCppBuild::OpenClAdreno));
        assert_eq!(identify_build(&names(&["libggml-metal.dylib"])), Ok(LlamaCppBuild::Metal));
        assert_eq!(identify_build(&names(&["ggml-cpu-alderlake.dll", "llama.dll"])), Ok(LlamaCppBuild::Cpu));
        assert_eq!(identify_build(&names(&["libggml-vulkan.so", "libggml-cpu.so"])), Ok(LlamaCppBuild::Vulkan));
    }

    #[test]
    fn linux_cuda_builds_are_identified_by_the_runtime_soname() {
        let n = names(&["llama-server", "libggml-cuda.so", "libcudart.so.12.4.127", "libggml-cpu.so"]);
        assert_eq!(identify_build(&n), Ok(LlamaCppBuild::Cuda12));
        assert_eq!(identify_build(&names(&["libggml-cuda.so", "libcudart.so.13"])), Ok(LlamaCppBuild::Cuda13));
        assert_eq!(identify_build(&names(&["libggml-cuda.so", "libcudart.so.12"])), Ok(LlamaCppBuild::Cuda12));
        // `libcudart.so.120` is not major 12, and a bare `libcudart.so` names no major at all.
        assert!(identify_build(&names(&["libggml-cuda.so", "libcudart.so.120"])).is_err());
        assert!(identify_build(&names(&["libggml-cuda.so", "libcudart.so"])).is_err());
        assert!(identify_build(&names(&["libggml-cuda.so", "libcudart.so.12.4", "libcudart.so.13.0"])).is_err());
    }

    #[test]
    fn ambiguous_or_incomplete_folders_are_not_guessed() {
        assert!(identify_build(&names(&["ggml-cuda.dll"])).unwrap_err().contains("cudart"));
        assert!(identify_build(&names(&["ggml-cuda.dll", "cudart64_12.dll", "cudart64_13.dll"])).is_err());
        assert!(identify_build(&names(&["ggml-cuda.dll", "cudart64_12.dll", "ggml-vulkan.dll"])).unwrap_err().contains("several"));
        assert!(identify_build(&names(&["llama-server.exe"])).is_err());
        assert!(identify_build(&[]).is_err());
    }

    fn request(dir: &str) -> LaunchRequest {
        let mut r = LaunchRequest::new(Location::Native(dir.into()), r"C:\models\m.gguf", 8091);
        r.api_key_file = Some(r"C:\Cerebro\key.txt".into());
        r
    }

    #[cfg(windows)]
    #[test]
    fn argv_is_hardened_and_goes_through_the_allowlist() {
        let cuda = LlamaCpp::new(LlamaCppBuild::Cuda12);
        let mut req = request(r"C:\Cerebro\backends\llama.cpp-cuda12");
        req.context = Some(8192);
        req.llama_flags = vec![
            Flag::GpuLayers(999),
            Flag::CacheTypeK(CacheType::Q8_0),
            Flag::Device(DeviceList::devices([DeviceId::parse("CUDA0").unwrap()]).unwrap()),
        ];
        let cmd = cuda.build_argv(&req).unwrap();
        assert_eq!(cmd.program, PathBuf::from(r"C:\Cerebro\backends\llama.cpp-cuda12\llama-server.exe"));
        let argv = cmd.args.join(" ");
        for expected in ["-m C:\\models\\m.gguf", "--host 127.0.0.1", "--port 8091", "--no-webui", "-c 8192", "-ngl 999", "-dev CUDA0"] {
            assert!(argv.contains(expected), "{expected:?} missing from {argv}");
        }
        assert!(argv.contains("--api-key-file"), "{argv}");
        assert_eq!(cmd.api_key, ApiKeyDelivery::File);
    }

    #[cfg(windows)]
    #[test]
    fn devices_from_another_build_and_misplaced_options_are_refused() {
        let cuda = LlamaCpp::new(LlamaCppBuild::Cuda12);
        let mut req = request(r"C:\b");
        req.llama_flags = vec![Flag::Device(DeviceList::devices([DeviceId::parse("Vulkan0").unwrap()]).unwrap())];
        let err = cuda.build_argv(&req).unwrap_err();
        assert!(err.to_string().contains("VULKAN0") && err.to_string().contains("CUDA 12"), "{err}");
        // The Vulkan build accepts it, and every build accepts RPC devices.
        assert!(LlamaCpp::new(LlamaCppBuild::Vulkan).build_argv(&req).is_ok());
        req.llama_flags = vec![Flag::Device(DeviceList::devices([DeviceId::parse("RPC0").unwrap()]).unwrap())];
        assert!(LlamaCpp::new(LlamaCppBuild::Cpu).build_argv(&req).is_ok());

        let mut tp = request(r"C:\b");
        tp.tensor_parallel = Some(2);
        assert!(matches!(cuda.build_argv(&tp), Err(BackendError::OptionUnsupported { .. })));
        let mut no_key = request(r"C:\b");
        no_key.api_key_file = None;
        assert!(matches!(cuda.build_argv(&no_key), Err(BackendError::InvalidRequest(_))));
        let mut dup = request(r"C:\b");
        dup.llama_flags = vec![Flag::Port(1)];
        assert!(matches!(cuda.build_argv(&dup), Err(BackendError::Argv(_))));
        let mut relative = request(r"C:\b");
        relative.model = "m.gguf".into();
        assert!(matches!(cuda.build_argv(&relative), Err(BackendError::InvalidRequest(_))));
        let wsl = LaunchRequest {
            location: Location::Wsl { distro: "Ubuntu".into(), env_path: "/x".into() },
            ..request(r"C:\b")
        };
        assert!(matches!(cuda.build_argv(&wsl), Err(BackendError::OptionUnsupported { .. })));
    }

    #[test]
    fn a_folder_of_another_build_is_refused() {
        let dir = std::env::temp_dir().join(format!("cerebro-backends-wrongbuild-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["llama-server.exe", "llama.dll", "ggml.dll", "ggml-base.dll", "ggml-vulkan.dll"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        let err = LlamaCpp::new(LlamaCppBuild::Cuda12).check_install(&dir).unwrap_err();
        assert!(matches!(err, BackendError::WrongBuild { found: LlamaCppBuild::Vulkan, .. }), "{err}");
        let ok = LlamaCpp::new(LlamaCppBuild::Vulkan).check_install(&dir).unwrap();
        assert_eq!(ok.build, Some(LlamaCppBuild::Vulkan));
        std::fs::remove_file(dir.join("llama.dll")).unwrap();
        if cfg!(windows) {
            let err = LlamaCpp::new(LlamaCppBuild::Vulkan).check_install(&dir).unwrap_err();
            assert!(matches!(&err, BackendError::MissingFiles { missing, .. } if missing == &["llama.dll"]), "{err}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// On every platform a folder with the right libraries but no server binary is refused, rather
    /// than reported with an `entry` that does not exist.
    #[test]
    fn a_folder_without_the_server_is_refused() {
        let dir = std::env::temp_dir().join(format!("cerebro-backends-noserver-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["llama.dll", "ggml.dll", "ggml-base.dll", "ggml-vulkan.dll", "libggml-vulkan.so"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        let err = LlamaCpp::new(LlamaCppBuild::Vulkan).check_install(&dir).unwrap_err();
        assert!(
            matches!(&err, BackendError::MissingFiles { missing, .. } if missing.iter().any(|m| m == SERVER_EXE)),
            "{err}"
        );
        assert!(matches!(
            LlamaCpp::new(LlamaCppBuild::Vulkan).install(),
            InstallKind::Folder { entry: SERVER_EXE, .. }
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The official CUDA build identifies as CUDA 12, passes its install check, and hashes into a
    /// manifest covering its server and CUDA library.
    #[test]
    #[ignore = "requires an official llama.cpp CUDA 12 build (CEREBRO_LLAMACPP_OFFICIAL)"]
    fn live_official_build_is_identified_and_hashed() {
        let dir = cerebro_probe::fixtures::official_backend_dir().unwrap();
        let backend = LlamaCpp::new(LlamaCppBuild::Cuda12);
        let report = backend.check_install(&dir).unwrap();
        assert_eq!(report.build, Some(LlamaCppBuild::Cuda12));
        assert!(report.entry.is_file());
        let manifest = backend.integrity_manifest(&dir).unwrap();
        for f in ["llama-server.exe", "ggml-cuda.dll", "cudart64_12.dll"] {
            let hash = manifest.files.get(f).unwrap_or_else(|| panic!("{f} not hashed"));
            assert_eq!(hash.len(), 64);
        }
        assert!(matches!(
            LlamaCpp::new(LlamaCppBuild::Vulkan).check_install(&dir),
            Err(BackendError::WrongBuild { found: LlamaCppBuild::Cuda12, .. })
        ));
    }
}
