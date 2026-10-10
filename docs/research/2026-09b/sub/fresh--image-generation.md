# Local image generation backends — research notes (2026-09-22)

Focus: what changed since 2026-09-01, for a Windows-first Rust harness
(spawn-backend-as-child-process, like llama.cpp) on the reference laptop
(i9-12900H, 31.7 GB RAM, RTX 3070 Ti 8 GB Ampere, no NPU).
All URLs below were opened during this pass. Anything not verified is marked
"unverified".

## 1. stable-diffusion.cpp is the obvious embeddable core

Repo: https://github.com/leejet/stable-diffusion.cpp (MIT, pure C/C++, ggml-based,
"working in the same way as llama.cpp"). News entries run through 2026-09-20
(Day-0 Qwen-Image-2.1), i.e. actively maintained within the focus window.
Supported image models now include SD1.x/2.x/Turbo, SDXL/Turbo, SD3/SD3.5,
FLUX.1-dev/schnell, FLUX.2-dev/klein, Qwen-Image (+2.1), Z-Image, Chroma,
Chroma1-Radiance, Lens, PiD, LongCat, MiniT2I, Ovis, Anima, ERNIE-Image, Krea2,
Mage-Flow, HiDream-O1, Ideogram4, LLaDA-Image; edit models (FLUX Kontext,
Qwen-Image-Edit series); video (Wan2.1/2.2, LTX-2.3/2.5, MiniMax-H3,
HunyuanVideo 1.5, LingBot-Video). Backends: CPU (AVX/AVX2/AVX512), CUDA,
Vulkan, Metal, OpenCL, SYCL. Weight formats: ckpt, safetensors, GGUF, plus a
convert mode (weights to GGUF/safetensors). Prebuilt Windows binaries
(CUDA/Vulkan/CPU) are published per-release, so Cerebro can redistribute or
sidecar them like it does llama.cpp.
Source: https://github.com/leejet/stable-diffusion.cpp

## 2. Server/API mode exists and is the harness seam

`sd-server` exposes three families
(source: https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/api.md):
OpenAI-compatible (`POST /v1/images/generations`, `POST /v1/images/edits`
multipart incl. mask, `GET /v1/models`); WebUI-compatible
(`POST /sdapi/v1/txt2img`, `/sdapi/v1/img2img`, plus loras/upscalers/samplers/
schedulers/sd-models/options discovery); and a native async `sdcpp API`
(`GET /sdcpp/v1/capabilities`, `POST /sdcpp/v1/img_gen`, `POST
/sdcpp/v1/vid_gen`, `GET /sdcpp/v1/jobs/{id}`, cancel endpoint). Native-only
controls can be smuggled through the compat APIs inside
`<sd_cpp_extra_args>{...}</sd_cpp_extra_args>` in the prompt. Prompt-embedded
`<lora:...>` tags are intentionally unsupported server-side; LoRA must go
through structured fields. Since 2026-04-11 there is also a brand-new embedded
web UI served by the binary itself
(source: https://github.com/leejet/stable-diffusion.cpp). Caveat from a
community server-mode write-up: model is fixed at server start (restart to
swap) — unverified in this pass beyond that report.

## 3. Speed per image at 1024px: sd.cpp trails ComfyUI, leads on footprint

Hard numbers all come from one sd.cpp issue thread (batch 1; Linux where
stated; samplers mostly unstated — treat as indicative, not transferable):
https://github.com/leejet/stable-diffusion.cpp/issues/1215
- Reporter: FLUX.2-klein-4B, 1024x1024, 4 steps, cfg 1 — ComfyUI ~5 s vs
  sd.cpp ~25 s (Vulkan and ROCm builds).
- RX 7800XT, Arch, mesa-git, same thread: SDXL (cfg 4, 30 steps) ComfyUI
  12.36 s warm (16.86 s cold) vs sd.cpp Vulkan 29.63 s (~2.4x slower);
  Z-Image-Turbo (cfg 1, 9 steps) 20.84 s vs 34.57 s (~1.65x); FLUX.2-klein-4B
  6.24 s vs 23.32 s (~3.7x), then after upstream optimization PR ~31.8%
  faster (15.90 s); klein-9B ~1.9x slower, ~29% faster after the fix.
- RTX 4090, CUDA backend, same thread: sd.cpp ~6.7 it/s vs ComfyUI ~10.5
  it/s (~1.6x gap).
No RTX 3070 Ti-class 1024px timing was verified in this pass (unverified).
Rule of thumb for Cerebro's benchmark table: sd.cpp ≈ 1.5–2.5x slower per
image than ComfyUI/PyTorch on the same discrete GPU, but with far lower
VRAM and zero Python/CUDA-toolkit dependencies.

## 4. VRAM: what fits the 8 GB Ampere card

- FLUX.1 diffusion-weights-only table
  (source: https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux.md):
  q8_0 12068 MB / q4_0 6394 MB / q4_k 6395 MB / q3_k 4888 MB / q2_k 3735 MB —
  plus VAE + CLIP-L + T5-XXL on top. Doc says 6 GB, "even 4 GB" GPUs can run
  FLUX with offload; cfg-scale 1.0 recommended; schnell needs only 4 steps.
  Practical 8 GB recipe: schnell q4/q2_k GGUF + `--clip-on-cpu`
  (+ `--offload-to-cpu` if needed), T5 fp8 variant.
- Qwen-Image (20B MMDiT; Apache-2.0; source: https://github.com/QwenLM/Qwen-Image):
  Unsloth's sd.cpp tutorial
  (source: https://unsloth.ai/docs/models/tutorials/qwen-image-2512/stable-diffusion.cpp)
  uses Q4_K_M diffusion (~13.1 GB for the Edit-2512 GGUF) + Qwen2.5-VL-7B
  Q4 text encoder + safetensors VAE, rule of thumb "combined RAM+VRAM >
  GGUF size", example 1024px/40 steps/euler/cfg 2.5/flow-shift 3 with
  `--diffusion-fa --offload-to-cpu`. Fits the reference laptop only with
  CPU offload (slow); not the default pick.
- SDXL/SDXL-Turbo remains the safe 1024px default on 8 GB (few-GB
  footprint); SD3.5-Medium/Large-Turbo is the quality step-up. Exact
  sd.cpp VRAM figures for these were not re-verified this pass (unverified).

## 5. Feature coverage: img2img, inpainting, upscaling all present

sd.cpp API supports init image, mask image (incl. inpainting invert flag),
denoising strength, control/IP-adapter/reference images, tiled VAE, ESRGAN
upscale models, TAESD fast decode, hi-res fix (upscaler incl. latent modes),
LoRA, LCM/LCM-LoRA, PhotoMaker, IP-Adapter (SD1.5/SDXL), ControlNet (SD1.5
only), ADetailer, Flash Attention, negative prompt
(sources: https://github.com/leejet/stable-diffusion.cpp and
https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/api.md).
KoboldCpp's image-gen settings page confirms the same surface from a
shipper's view (quantize/compress weights, TAE fallback VAE, no-VAE-tiling
toggle, per-model CLIP/T5/VAE slots):
https://koboldai.com/Guides/Kobold_CPP/KoboldCpp_ImageGen/

## 6. Licences (all primary sources opened)

- FLUX.1-schnell: Apache-2.0, commercial OK
  (https://huggingface.co/black-forest-labs/FLUX.1-schnell).
- FLUX.1-dev (and Kontext etc.): Non-Commercial License v1.1.1 — no
  commercial/production use without a BFL deal; outputs are yours but can't
  train competing models
  (https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
- FLUX.2-klein is described as Apache-2.0 and FLUX.2-dev as non-commercial
  only in a secondary listing (https://lmstudio.ai/maticulous75/image) —
  primary BFL terms unverified this pass.
- SD3.5 suite: Stability Community License — free incl. commercial use under
  $1M annual revenue, Enterprise above; attribution/"Powered by Stability AI"
  required (https://stability.ai/license).
- Qwen-Image (v1): Apache-2.0 (https://github.com/QwenLM/Qwen-Image).
- Qwen-Image-2.1: Qwen Research License, NON-COMMERCIAL ONLY, commercial
  licence via model-business@notice.qwencloud.com
  (https://huggingface.co/Qwen/Qwen-Image-2.1/raw/main/LICENSE). Licence
  regime changed between v1 and 2.x — Cerebro must gate these separately.
- sd.cpp itself is MIT; KoboldCpp is AGPL-3.0 (careful with bundling:
  https://github.com/LostRuins/koboldcpp/); Comfy Desktop is AGPL/commercial
  dual (FAQ below); diffusion-rs is MIT (its repo page).

## 7. ComfyUI: headless API yes, Desktop app no

Comfy Desktop "does not support headless, server-only, or remote operation
modes… For headless or server deployments, use a manual ComfyUI installation
or the Windows Portable build instead"
(source: https://docs.comfy.org/installation/desktop/faq). The installed
`C:/Users/noahm/AppData/Local/Comfy-Desktop` copy therefore cannot be driven
as Cerebro's backend; the portable/manual install exposes the `/prompt` +
`/history` (+ websocket) API that wrappers like comfy-headless POST
workflow JSON to (workflow API format described in search results —
unverified in this pass). ComfyUI is the speed leader (§3) but drags a full
Python + PyTorch + CUDA stack and model-folder management — an optional
"bring your own ComfyUI" target, not the bundled default.

## 8. Other embeddable engines

- KoboldCpp (single-file exe, AGPL): bundles sd.cpp image generation
  (SD1.5/SDXL/SD3/FLUX/Qwen-Image/Z-Image/Klein per its readme) with an
  A1111-compatible API and bundled StableUI; proves the "llama.cpp-style
  single binary + image tab" pattern Cerebro wants
  (sources: https://github.com/LostRuins/koboldcpp/ and
  https://koboldai.com/Guides/Kobold_CPP/KoboldCpp_ImageGen/).
- diffusion-rs: Rust bindings to sd.cpp on crates.io; CUDA+Vulkan marked
  working on Windows (source: https://github.com/newfla/diffusion-rs) —
  the fastest path if Cerebro wants in-process instead of argv-spawned
  `sd-server` (F FI risk vs process isolation trade-off).
- sd.cpp readme also lists Python/C#/Go/Dart bindings and UIs
  (sd.cpp-webui, LocalAI, GIMP plugins)
  (source: https://github.com/leejet/stable-diffusion.cpp) — not evaluated
  further (unverified beyond the listing).

## 9. Competitors: LM Studio and Jan do NOT do local image generation

- LM Studio: no native local diffusion runtime found. Image generation
  happens only via cloud: an MCP workaround calling Hugging Face Spaces, or
  a Hub plugin that calls the Hugging Face Inference API (FLUX.1-schnell
  default, dev models need Pro credits, cold starts 20–60 s)
  (source: https://lmstudio.ai/maticulous75/image). Fully offline local
  image gen is a genuine Cerebro differentiator.
- Jan: v0.6.9 "image support" is image *input* (multimodal understanding:
  auto-detected vision for local Gemma3/Qwen3 etc.), not image synthesis
  (source: https://jan.ai/changelog/2025-08-28-image-support). A Jan server
  doc snippet suggests provider-routed `/v1/images/generations|edits`
  proxying to an external image provider — unverified (doc URL 404'd when
  opened). Either way, no bundled local diffusion.

## 10. Cutting-edge 2025–2026 deltas for the harness

Day-0/1 model support is now a release-note race (Qwen-Image-2.1 Day-0
2026-09-20; FLUX.2-dev/klein, Z-Image, Wan2.1/2.2, LTX-2.3/2.5, MiniMax-H3
all added since Sep 2025: https://github.com/leejet/stable-diffusion.cpp).
Notable: sd.cpp docs list an inference-acceleration-via-caching guide and
INT8-convrot safetensors (listing only — unverified); Wan/LTX video jobs
run through the same async `vid_gen` API but are VRAM-heavy and not a sane
default on the 8 GB card (unverified timings). Qwen-Image-2.x's licence flip
to non-commercial is the sharpest gotcha of the month (§6).

## 11. Recommendation for Cerebro

Bundle `sd-server` (prebuilt CUDA Windows binary, pinned version + SHA) as
a second argv-spawned child beside llama.cpp; talk to it over
`sdcpp API` (async jobs) with OpenAI-compat fallback. Defaults for the
reference machine: SDXL-Turbo (fastest) / FLUX.1-schnell-Q4 (quality), both
Apache/Community-clean; hide dev/Qwen-2.x weights behind a non-commercial
acknowledgement. Keep ComfyUI portable as an optional external engine via
its `/prompt` API. Benchmark table must state hardware/model/quant/steps/
backend per image, and re-measure on the 3070 Ti — the 5 s vs 25 s class of
gaps above do not transfer across GPUs.
