//! The hardware-gated features the app can show and the user can switch.

use std::fmt;

/// Which part of the settings a feature belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FeatureCategory {
    /// Weight formats.
    Quantization,
    /// Attention kernels.
    Attention,
    /// Splitting one model across GPUs.
    MultiGpu,
    /// Speculative decoding.
    Speculative,
    /// Offloading work to an NPU.
    Npu,
    /// CPU kernel families llama.cpp selects by instruction set.
    CpuKernels,
}

/// One hardware-gated feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Feature {
    /// GGUF K-quants and I-quants (llama.cpp, every build).
    GgufQuants,
    /// FP8 weights and activations on FP8 tensor/matrix cores (vLLM, SGLang).
    Fp8,
    /// NVFP4 checkpoints on Blackwell FP4 tensor cores (vLLM, SGLang).
    Nvfp4,
    /// AWQ and GPTQ checkpoints (vLLM, SGLang).
    AwqGptq,
    /// EXL3 checkpoints (ExLlamaV3).
    Exl3,
    /// MLX 4/8-bit checkpoints (MLX on Apple Silicon).
    MlxQuant,
    /// OpenVINO IR int4/int8 models (OpenVINO).
    OpenVinoInt4,
    /// llama.cpp's own flash attention (`-fa`), in every build.
    LlamaFlashAttention,
    /// FlashAttention 2 kernels (vLLM, SGLang, ExLlamaV3; Ampere and newer).
    FlashAttention2,
    /// FlashAttention 3 kernels (vLLM, SGLang; Hopper only).
    FlashAttention3,
    /// llama.cpp layer split across GPUs (`-sm layer -ts ...`).
    LayerSplit,
    /// llama.cpp row split across GPUs (`-sm row`, CUDA/HIP).
    RowSplit,
    /// Tensor parallelism across GPUs (vLLM, SGLang).
    TensorParallel,
    /// Direct GPU-to-GPU transfers over NVLink.
    NvlinkPeerToPeer,
    /// Draft-free n-gram speculation (llama.cpp `ngram-*`, vLLM n-gram).
    SpecNgram,
    /// Speculation with a separate draft model.
    SpecDraftModel,
    /// EAGLE-3 speculation heads.
    SpecEagle3,
    /// Multi-token-prediction heads.
    SpecMtp,
    /// Intel NPU offload (OpenVINO).
    NpuOffloadIntel,
    /// AMD XDNA NPU offload (Ryzen AI).
    NpuOffloadAmd,
    /// Qualcomm Hexagon NPU offload (QNN).
    NpuOffloadQualcomm,
    /// llama.cpp AVX-512 CPU kernels.
    CpuAvx512Kernels,
    /// llama.cpp AMX CPU kernels (int8 tile matmul).
    CpuAmxKernels,
    /// llama.cpp Arm I8MM CPU kernels.
    CpuArmI8mmKernels,
}

impl Feature {
    /// Every feature, in display order.
    pub const ALL: &'static [Feature] = &[
        Self::GgufQuants,
        Self::Fp8,
        Self::Nvfp4,
        Self::AwqGptq,
        Self::Exl3,
        Self::MlxQuant,
        Self::OpenVinoInt4,
        Self::LlamaFlashAttention,
        Self::FlashAttention2,
        Self::FlashAttention3,
        Self::LayerSplit,
        Self::RowSplit,
        Self::TensorParallel,
        Self::NvlinkPeerToPeer,
        Self::SpecNgram,
        Self::SpecDraftModel,
        Self::SpecEagle3,
        Self::SpecMtp,
        Self::NpuOffloadIntel,
        Self::NpuOffloadAmd,
        Self::NpuOffloadQualcomm,
        Self::CpuAvx512Kernels,
        Self::CpuAmxKernels,
        Self::CpuArmI8mmKernels,
    ];

    /// Settings section.
    pub fn category(self) -> FeatureCategory {
        use Feature::*;
        match self {
            GgufQuants | Fp8 | Nvfp4 | AwqGptq | Exl3 | MlxQuant | OpenVinoInt4 => {
                FeatureCategory::Quantization
            }
            LlamaFlashAttention | FlashAttention2 | FlashAttention3 => FeatureCategory::Attention,
            LayerSplit | RowSplit | TensorParallel | NvlinkPeerToPeer => FeatureCategory::MultiGpu,
            SpecNgram | SpecDraftModel | SpecEagle3 | SpecMtp => FeatureCategory::Speculative,
            NpuOffloadIntel | NpuOffloadAmd | NpuOffloadQualcomm => FeatureCategory::Npu,
            CpuAvx512Kernels | CpuAmxKernels | CpuArmI8mmKernels => FeatureCategory::CpuKernels,
        }
    }

    /// Display name.
    pub fn name(self) -> &'static str {
        use Feature::*;
        match self {
            GgufQuants => "GGUF quantization",
            Fp8 => "FP8",
            Nvfp4 => "NVFP4",
            AwqGptq => "AWQ / GPTQ",
            Exl3 => "EXL3",
            MlxQuant => "MLX quantization",
            OpenVinoInt4 => "OpenVINO int4",
            LlamaFlashAttention => "Flash attention (llama.cpp)",
            FlashAttention2 => "FlashAttention 2",
            FlashAttention3 => "FlashAttention 3",
            LayerSplit => "Split layers across GPUs",
            RowSplit => "Split rows across GPUs",
            TensorParallel => "Tensor parallelism",
            NvlinkPeerToPeer => "NVLink peer-to-peer",
            SpecNgram => "N-gram speculation (no draft model)",
            SpecDraftModel => "Draft-model speculation",
            SpecEagle3 => "EAGLE-3 speculation",
            SpecMtp => "Multi-token prediction",
            NpuOffloadIntel => "Intel NPU offload",
            NpuOffloadAmd => "AMD NPU offload",
            NpuOffloadQualcomm => "Qualcomm NPU offload",
            CpuAvx512Kernels => "AVX-512 CPU kernels",
            CpuAmxKernels => "AMX CPU kernels",
            CpuArmI8mmKernels => "Arm I8MM CPU kernels",
        }
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The user's setting for one feature. `Auto` (the default) follows detection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Override {
    /// On exactly when the hardware supports it.
    #[default]
    Auto,
    /// On even if detection says unavailable or could not check (the user has the hardware, or
    /// wants to try). Recorded with one neutral warning.
    ForceOn,
    /// Off even if available.
    ForceOff,
}
