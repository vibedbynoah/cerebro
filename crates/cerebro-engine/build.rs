//! Compiles src/kernels.cu to PTX with nvcc when the `cuda` feature is on. The PTX targets compute 7.5
//! (dp4a) and newer; the driver JIT-compiles it for the installed GPU.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-changed=src/kernels_mma.cu");
    println!("cargo:rerun-if-changed=src/tensor.cu");
    // The exact source a build came from, for measurements (the Tune screen keys results to it).
    let commit = Command::new("git").args(["rev-parse", "--short=10", "HEAD"]).output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned()).unwrap_or_default();
    let dirty = Command::new("git").args(["status", "--porcelain", "--untracked-files=no", "--", "src"]).output().ok().is_some_and(|o| !o.stdout.is_empty());
    println!("cargo:rustc-env=CEREBRO_ENGINE_COMMIT={}{}", if commit.is_empty() { "unknown" } else { &commit }, if dirty { "+changes" } else { "" });
    // Rebuilt when the source or the checked-out commit changes (HEAD and its branch ref, wherever
    // git keeps them: this may be a worktree).
    println!("cargo:rerun-if-changed=src");
    let git_path = |p: &str| Command::new("git").args(["rev-parse", "--git-path", p]).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned()).filter(|s| !s.is_empty());
    let branch = Command::new("git").args(["symbolic-ref", "-q", "HEAD"]).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned()).filter(|s| !s.is_empty());
    for p in [git_path("HEAD"), branch.as_deref().and_then(git_path), git_path("packed-refs")].into_iter().flatten() {
        println!("cargo:rerun-if-changed={p}");
    }
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }
    let dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    // Base kernels target sm_75 (dp4a); tensor-core kernels a separate sm_80 module loaded when present.
    for (src, ptx, arch) in [("src/kernels.cu", "kernels.ptx", "compute_75"), ("src/kernels_mma.cu", "kernels_mma.ptx", "compute_80"), ("src/tensor.cu", "tensor.ptx", "compute_80")] {
        let status = Command::new("nvcc")
            .args(["--ptx", "-O3", "--use_fast_math", &format!("-arch={arch}"), "-o"])
            .arg(dir.join(ptx))
            .arg(src)
            .status()
            .expect("nvcc not found: install the CUDA toolkit to build the GPU backend, or build without the cuda feature");
        assert!(status.success(), "nvcc failed to compile {src}");
    }
}
