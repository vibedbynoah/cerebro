//! Inference backends for every hardware setup, the capability matrix that says which features a
//! machine supports and why not, and backend selection.
//!
//! Cerebro ships to machines worldwide, not only to the development laptop. This crate models
//! every backend behind one [`Backend`] trait (gate, install, integrity manifest, launch argv,
//! health check):
//!
//! * **llama.cpp**, the default for GGUF, in its CUDA 12, CUDA 13, Vulkan, ROCm/HIP, SYCL,
//!   OpenCL (Adreno), Metal and CPU builds ([`llamacpp`]);
//! * **optional backends**, offered only where hardware and OS allow: vLLM and SGLang (Linux or
//!   WSL 2), ExLlamaV3 (NVIDIA Ampere+), MLX (Apple Silicon), OpenVINO (Intel CPU/GPU/NPU),
//!   Ryzen AI (AMD XDNA 2) and Qualcomm QNN (Hexagon) ([`external`]).
//!
//! [`CapabilityMatrix`] maps a [`cerebro_probe::HardwareProfile`] to every [`Feature`] (quant
//! formats, attention kernels, multi-GPU split, speculative decoding, NPU offload, CPU kernels)
//! with the reason each unavailable one is unavailable, so the app can show "available on your
//! hardware" or "needs an NVIDIA Hopper GPU". Every feature and every backend is switchable by the
//! user (safe default, one neutral recorded warning when forced past detection).
//!
//! Three rules hold throughout:
//!
//! 1. **Detection that could not check is never a yes or a no.** It is [`Check::Unknown`], shown
//!    as "could not check", and still switchable.
//! 2. **Selection proposes, measurement decides.** [`BackendRegistry::select`] orders candidates;
//!    the tuner measures them. llama.cpp's automatic baseline is each candidate build run with no
//!    tuning flags (on a GPU machine, the GPU build with defaults); the tuner measures that per
//!    build and never stores a configuration that loses to it. The CPU build is always a GGUF
//!    candidate when supported, as the fallback that runs everywhere.
//! 3. **Modelled is not verified.** [`Backend::verification`] says whether Cerebro has launched a
//!    backend end to end; [`matrix::VERIFIED_FEATURES`] says which (backend, feature) pairs it has
//!    run. Only those pairs may be claimed as working (docs/CLAIMS.md). A backend with no launch
//!    recipe ([`LaunchRecipe::Missing`]) is never offered or shown as available.

#![deny(missing_docs)]

pub mod backend;
pub mod error;
pub mod external;
pub mod feature;
pub mod llamacpp;
pub mod matrix;
pub mod registry;
pub mod requirement;

pub use backend::{
    ApiKeyDelivery, Backend, BackendId, EnvValue, HealthCheck, InstallKind, InstallReport,
    IntegrityManifest, KeyFileFormat, LaunchCommand, LaunchRecipe, LaunchRequest, ListenerOwner,
    LlamaCppBuild, Location, ModelFormat, Verification,
};
pub use error::BackendError;
pub use feature::{Feature, FeatureCategory, Override};
pub use matrix::{Availability, CapabilityMatrix, FeatureState, VERIFIED_FEATURES};
pub use registry::{BackendRegistry, Candidate, CandidateStatus, Selection, SelectionPolicy};
pub use requirement::{Check, Reason, Requirement};
