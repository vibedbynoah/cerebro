//! Compiles src/kernels.cu to PTX with nvcc when the `cuda` feature is on. The PTX targets compute 7.5
//! (dp4a) and newer; the driver JIT-compiles it for the installed GPU.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels.cu");
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("kernels.ptx");
    let status = Command::new("nvcc")
        .args(["--ptx", "-O3", "--use_fast_math", "-arch=compute_75", "-o"])
        .arg(&out)
        .arg("src/kernels.cu")
        .status()
        .expect("nvcc not found: install the CUDA toolkit to build the GPU backend, or build without the cuda feature");
    assert!(status.success(), "nvcc failed to compile src/kernels.cu");
}
