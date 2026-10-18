//! Compiles src/kernels.cu to PTX with nvcc when the `cuda` feature is on. The PTX targets compute 7.5
//! (dp4a) and newer; the driver JIT-compiles it for the installed GPU.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-changed=src/kernels_mma.cu");
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }
    let dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    // Base kernels target sm_75 (dp4a); tensor-core kernels a separate sm_80 module loaded when present.
    for (src, ptx, arch) in [("src/kernels.cu", "kernels.ptx", "compute_75"), ("src/kernels_mma.cu", "kernels_mma.ptx", "compute_80")] {
        let status = Command::new("nvcc")
            .args(["--ptx", "-O3", "--use_fast_math", &format!("-arch={arch}"), "-o"])
            .arg(dir.join(ptx))
            .arg(src)
            .status()
            .expect("nvcc not found: install the CUDA toolkit to build the GPU backend, or build without the cuda feature");
        assert!(status.success(), "nvcc failed to compile {src}");
    }
}
