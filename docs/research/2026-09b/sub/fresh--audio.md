# Local speech stack for Cerebro — late Sept 2026 (fresh--audio)

Scope: best fully-offline Windows desktop STT + TTS for a Rust harness that spawns
`llama-server`/`llama.cpp` with argv only. Reference machine: i9-12900H, 32 GB RAM,
RTX 3070 Ti Laptop 8 GB (Ampere sm_86, ~80–115 W), no NPU. All figures are batch-1
laptop-relevant unless stated; datacenter batch-128 numbers are flagged non-transferable.
Cutoff focus: changes since 2026-09-01, preferring 2025–2026 primary sources.

## 1. Speech-to-text

### 1.1 whisper.cpp (C/C++, GGML) + server mode — default offline STT
- Current: v1.9.4 (2026-09-11) fixes decoder reseeding, adds language in `/inference`
detect response; v1.9.3 (2026-08-20) nightly b4938; v1.8.5+ hardened server
(token_timestamps flag, carry_initial_prompt, speaker JSON, params-leak fixes).
Sources: https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.3 ,
https://freedom.tech/posts/2026-09-11-whisper-cpp-1-9-4/ ,
https://github.com/ggml-org/whisper.cpp/releases/tag/v1.8.5
- Server (`whisper-server`) is file-POST `/inference`, not chunked streaming; one
community thread prototypes `new_segment_callback` chunked streaming but crashes on
segment 2 (mutex/lifetime bug, unverified for production).
Source: https://github.com/ggml-org/whisper.cpp/discussions/3306
- Live mic path is `whisper-stream` (SDL2, `--step/--length`, sliding-window `--step 0`
+ `-vth ~0.6` VAD). Source: https://github.com/ggerganov/whisper.cpp/blob/master/examples/stream/README.md
- whisper.cpp vs faster-whisper (Aug 2026 comparison): same weights (tiny 39M … large-v3
1550M); RTX 4070 large-v3 faster-whisper int8 ~12xRT/2.5 GB vs whisper.cpp CUDA ~8xRT/~3 GB;
M-series whisper.cpp Metal ~10xRT wins on Mac; x86 CPU tiny/base faster-whisper int8 ~20xRT
vs whisper.cpp ~15xRT. No RTX 3070 Ti laptop figure found (unverified); treat 4070 ratios
as upper bounds (power-limited Ampere will be lower).
Source: https://www.promptquorum.com/power-local-llm/local-whisper-stt-comparison-2026
- Weights choice: large-v3-turbo (809M, decoder 32→4 layers, Sep 2024) is ~5x Apple-Silicon
/ ~2.7x GPU faster than large-v3 at +0.39pp mean WER (7.83 vs 7.44); faster-whisper 13-min
GPU run: turbo fp16 19.2 s/2.5 GB vs large-v3 fp16 52.0 s/4.5 GB; turbo int8 1.5 GB.
Sources: https://huggingface.co/openai/whisper-large-v3-turbo ,
https://github.com/SYSTRAN/faster-whisper/issues/1030 ,
https://whispernotes.app/blog/introducing-whisper-large-v3-turbo
- distil-large-v3 (756M, English-only, 6.3x relative latency, short-form WER 9.7 vs 8.4
large-v3) is fastest for English-only; needs GGML conversion for whisper.cpp, native in
faster-whisper. Source: https://weesperneonflow.ai/en/blog/2026-08-27-whisper-v3-turbo-vs-distil-whisper-comparison-2026/
- faster-whisper itself: CTranslate2 port, up to ~4x vs openai/whisper, int8 ~2x vs fp16
at <0.1pp WER cost; needs CUDA 12 + cuDNN 9 on current ctranslate2; 2026 fork
`faster-whisper2` v2.1.1 (2026-06-18) same claim. Python-only, CUDA-only GPU path.
Sources: https://github.com/SYSTRAN/faster-whisper , https://pypi.org/project/faster-whisper2 ,
https://localaimaster.com/blog/faster-whisper-guide

### 1.2 NVIDIA Parakeet TDT 0.6B v2/v3 — fastest accurate EN/European STT
- 600M FastConformer+TDT; v2 English-only, v3 25 European langs + auto-detect; native
PnC/caps, word timestamps, up to 24-min single pass (full attention, A100 80 GB).
License CC-BY-4.0. Sources: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 ,
https://catalog.ngc.nvidia.com/orgs/nvidia/collections/parakeet-tdt-0.6b ,
https://docs.nvidia.com/nemo/speech/3.0.0/asr/asr_checkpoints.html
- Local C++ path: `NeMo-Speech.cpp` + `parakeet-tdt-0.6b-v3.q8_0.gguf`
(`hf download …; nemo-speech transcribe`). Python path is NeMo 2.2+/Riva; HF Transformers
also supported. Streaming = chunked-inference script (`chunk_secs=2`, unverified latency).
Source: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/main/README.md
- Speed claims (non-transferable as stated): RTFx 3380 at batch 128 on
HF-Open-ASR leaderboard; community i7 CPU INT8 2.4 s clip → 307 ms (~8xRT), M5 Neural
Engine 33xRT STT. Sources: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 ,
https://www.reddit.com/r/artificial/comments/1u3r85g/i_built_a_100_local_cpuonly_voice_loop_for_any/
- Caveat: NeMo/Riva docs target Linux + H100/A100/L40; Windows is community/Docker only
(unverified). 8 GB Ampere fits q8 GGUF easily (model ~0.6–1.2 GB), but plan CPU fallback.

### 1.3 Moonshine (Useful Sensors) — tiny streaming-first STT
- Offline tiny/base: tiny 27M/~190 MB, base 61M/~400 MB, ONNX, 5x faster than Whisper per
10-s segment at same-or-better WER (vendor claim, hardware unspecified — treat as
unverified). WebRTC/VAD-ready plugin API. Sources: https://pypi.org/project/getstream-plugins-moonshine/ ,
https://github.com/moonshine-ai/moonshine-js
- Streaming family: tiny 34M / small 123M / medium 245M, 50 Hz frontend, sliding-window
encoder, 80 ms lookahead; Transformers support exists but "not yet fully efficient
streaming" (uses flash-attn sliding window). Live browser demo via MoonshineJS+WebRTC.
Sources: https://huggingface.co/UsefulSensors/moonshine-streaming-tiny ,
https://huggingface.co/docs/transformers/v5.9.0/model_doc/moonshine_streaming ,
https://webrtc.moonshine.ai/
- 2026 direction is the `moonshine-ai/moonshine` "Moonshine Voice" toolkit (Python/JS/WASM,
1 MB–size options, unverified benchmarks). Source: https://github.com/moonshine-ai/moonshine

### 1.4 Kyutai STT (delayed-streams) — low-latency streaming baseline
- `stt-1b-en_fr` (~1B, EN+FR, 0.5 s delay + semantic VAD) and `stt-2.6b-en` (~2.6B,
EN-only, 2.5 s delay); decoder-only over Mimi tokens (12.5 Hz, 32 tokens/frame);
word timestamps = frame offset − delay. Weights CC-BY-4.0; code MIT/Apache.
Sources: https://kyutai.org/stt/ , https://huggingface.co/kyutai/stt-2.6b-en ,
https://github.com/kyutai-labs/delayed-streams-modeling/blob/main/README.md
- Runtimes: `moshi.run_inference`, `moshi-server` (CUDA, TOML configs), MLX (Apple).
Paper: https://arxiv.org/abs/2509.08753 (preprint Sep 2025, unverified peer review).
No Windows-native server documented (unverified); run as sidecar process.

### 1.5 llama.cpp audio input (mtmd) — one-process audio+text+vision
- Since b8769 (2026-04-12): Qwen3-Omni-30B-A3B (MoE 30B/3B active, text+image+audio+video)
and Qwen3-ASR supported in mtmd; run via `llama-mtmd-cli`. Earlier: Qwen2-Audio,
SeaLLM-Audio (poor quant results noted), Qwen2.5-Omni 3B/7B, Ultravox, Voxtral, GLM.
Sources: https://www.creativeainews.com/blog/llama-cpp-audio-qwen3-omni-multimodal/ ,
https://github.com/ggml-org/llama.cpp/pull/19441 ,
https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- Gemma 3n (E2B/E4B, USM audio encoder, 160 ms chunks, 140 text / 35 multimodal langs)
is text-only in llama.cpp as of Jun 2026; audio needs MLX/transformers.
Sources: https://huggingface.co/blog/gemma3n , https://github.com/ggml-org/llama.cpp/discussions/15194
- Fit: Qwen3-ASR/Omni GGUFs are large for 8 GB VRAM alongside an LLM; prefer dedicated
STT sidecar and reserve mtmd audio for "audio understanding" turns. Quality/latency on
3070 Ti unverified.

## 2. Text-to-speech

### 2.1 Kokoro-82M — default local TTS (quality + licence + size)
- 82M StyleTTS2-class, Apache-2.0 weights+code, 24 kHz, 54 voices/8 langs (v1.0),
int8 ONNX ~92 MB + voices ~28 MB; q-variants listed on HF. Sources:
https://huggingface.co/hexgrad/Kokoro-82M , https://github.com/hexgrad/kokoro ,
https://huggingface.co/onnx-community/Kokoro-82M-ONNX
- Windows-ready ONNX: `kokoro-onnx` (MIT runtime, CPU/CUDA/DirectML providers),
`pykokoro` (DirectML extra), NVIDIA `kokoro-82M-onnx-opt` (win-x64 CUDA13, Ampere
listed). CPU RTF ~0.45–0.51 (~2xRT) on 4-core EPYC; quality rated above Supertonic.
Sources: http://www.github.com/thewh1teagle/kokoro-onnx , https://pypi.org/project/pykokoro/ ,
https://huggingface.co/nvidia/kokoro-82M-onnx-opt , https://heyneo.com/blog/kokoro-tts-vs-supertonic-3-tts

### 2.2 Piper — cheapest CPU voice prompts
- VITS voices as ONNX (tens of MB), tiers x_low/low 16 kHz, medium/high 22.05 kHz;
100+ voices/30+ langs; ~10xRT desktop CPU, RT on Pi 5 (guide claim, board-dependent).
Maintained fork `OHF-Voice/piper1-gpl` v1.8.0 (2026-09-04); legacy standalone
`rhasspy/piper` Windows zip is 2023 MIT but stale. Source: https://localaimaster.com/blog/piper-tts-setup-guide
- Streaming: `streaming-vits` v0.1.0 (2026-09-01) reuses Piper/VITS graphs, TTFA ~163 ms
vs 701 ms non-streaming on 15.2 s utterance (M-series Mac, 4 threads; unverified on
Windows). Sherpa-ONNX SAPI5 engine reports ~0.5 s latency/250 MB peak (unverified).
Sources: https://pypi.org/project/streaming-vits/ , https://github.com/willwade/sherpaonnxazuresapi-installer/
- Licence caution: `piper1-gpl` is GPL-licensed (name + repo); verify before bundling.

### 2.3 Orpheus 3B — expressive LLM-TTS over llama.cpp (experimental)
- Llama-3.2-3B finetune emitting SNAC 24 kHz tokens as text tokens; GGUF floats in
community (`mmwillet2/Orpheus_GGUF`, isaiahbjork local server recipe `-ngl 29 -c 8192`);
`orpheus-cpp` streams via llama-cpp-python + WebRTC/FastRTC demo. Sources:
https://github.com/canopyai/Orpheus-TTS/blob/main/additional_inference_options/no_gpu/README.md ,
https://github.com/markuryy/orpheus-cpp , https://news.ycombinator.com/item?id=43419983
- Upstream llama.cpp support is OPEN: issue #12476 (SNAC decoder missing), draft PR
#12487 stalled; TTS.cpp PR #90 is the working C++ reference. Sources:
https://github.com/ggml-org/llama.cpp/issues/12476 ,
https://github.com/ggml-org/llama.cpp/pull/12487 ,
https://github.com/mmwillet/TTS.cpp/pull/90
- Fit: 3B Q4 fits 8 GB but contends with the LLM; quality/streaming on 3070 Ti unverified.

### 2.4 Dia 1.6B / Dia2 — dialogue-native, heavy
- Dia 1.6B: 1.6B + Descript codec, `[S1]/[S2]` dialogue + laugh/cough tokens, EN-only,
Apache-2.0, ~10 GB VRAM full, ~40 tok/s on A4000 (~0.46xRT; 86 tok ≈ 1 s audio).
Sources: https://github.com/nari-labs/dia , https://huggingface.co/nari-labs/Dia-1.6B-0626
- Dia2 (2025-11-19, 1B/2B, streaming from first tokens, ≤2 min) is CUDA 12.8+ only as of
Apr 2026; no MLX port. 8 GB laptop cannot hold Dia 1.6B + LLM together; batch-only.
Source: https://codersera.com/blog/run-nari-dia-16b-on-mac-installation-guide/

### 2.5 Kyutai TTS 1.6B (+Pocket TTS) — true text-streaming TTS
- 1.6B-name/1.8B-actual, Mimi 12.5 Hz, audio delayed 16 steps (1.28 s), text-streaming:
220 ms token→audio, batched 32-stream 350 ms on L40S, 16 streams >2xRTF; voices via
precomputed embeddings; custom-voice weights private (cloning issue open).
CC-BY-4.0 weights. Sources: https://kyutai.org/blog/2025-07-03-kyutai-tts-1-6b/ ,
https://huggingface.co/kyutai/tts-1.6b-en_fr , https://github.com/kyutai-labs/moshi/issues/404
- Runtimes: PyTorch streaming script, Rust `moshi-server` + `configs/config-tts.toml`,
MLX `--quantize 8/4`. Pocket TTS (Jan 2026 per thread) is tiny CPU 5-s cloner but
non-streaming batch. Source: https://github.com/kyutai-labs/delayed-streams-modeling
- No Windows-native Rust server docs (unverified); sidecar on WSL2/Linux or CPU PyTorch.

### 2.6 Chatterbox family (Resemble AI, MIT + watermark) — clone + control
- Turbo 350M EN: 1-step distilled decoder, ~6xRT GPU/~75 ms, paralinguistic tags;
Nano 110M: 3xRT on 8 CPU cores; Multilingual v3 0.5B Llama: 23–25 langs, better
similarity/fewer hallucinations; Single-Language Pack (zh/es-mx/es-es/pt-br/pt-pt/hi).
Every clip carries PerTh watermark. Sources: https://www.resemble.ai/learn/models/chatterbox-turbo ,
https://github.com/resemble-ai/chatterbox , https://huggingface.co/ResembleAI/chatterbox ,
https://www.resemble.ai/resources/chatterbox-multilingual-v3-tts-with-embedded-watermarking-for-25-languages
- Caveat: v3 Korean/Vietnamese CER ~71–75% (data-starved, not production); Czech/Japanese/
Finnish 18–32% CER need per-content eval. NVIDIA NIM gives 2–39x vs raw PyTorch
(datacenter, non-transferable). Same v3 source.

### 2.7 Qwen3-TTS (Alibaba, Apache-2.0) — lowest first-packet latency
- 0.6B/1.7B × 12 Hz multi-codebook + 25 Hz semantic variants; 10 langs; 3-s clone;
97 ms first packet (0.6B) / 101 ms (1.7B) at concurrency 1 (paper Table, GPU
unspecified — treat as lab figure); EN WER 1.24 (1.7B-12Hz). Sources:
https://arxiv.org/html/2601.15621v1 , https://github.com/QwenLM/Qwen3-TTS ,
https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
- vLLM-Omni day-0 offline support; streaming Python "coming soon" per FAQ; Windows/3070 Ti
numbers unverified. Source: https://qwenlm-qwen3-tts.mintlify.app/resources/faq

### 2.8 Supertonic 3 — tiny multilingual CPU TTS
- ~99M flow-matching ONNX, 404 MB assets, 31 langs, 44.1 kHz; 5-step CPU RTF ~0.31
(~3xRT, "good/clear"), 2-step ~0.165 but robotic; Kokoro wins quality, Apache wins
licence (Supertonic is OpenRAIL-M). Fixed-voice open weights; cloning via Voice
Builder JSON only. Sources: https://heyneo.com/blog/kokoro-tts-vs-supertonic-3-tts ,
https://github.com/supertone-inc/supertonic , https://www.kiadev.net/news/2026-05-15-supertonic-v3-release

## 3. Voice-mode design (Windows harness)

- VAD: Silero v4 1.72 MB / v5 2.22 MB (16/8 kHz), ~1 ms per 30 ms chunk on 1 CPU thread;
sherpa-onnx bundles Silero + ASR/TTS/VAD C++ with DTMF/endpointing knobs.
Sources: https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html , https://github.com/k2-fsa/sherpa-onnx/
- Barge-in recipe (2026 production bar <150 ms): dual-pass WebRTC-GMM instant trigger
(~15–20 ms) + Silero confirm (~30 ms + 150–300 ms smoothing); streaming TTS buffer
200–300 ms with <20 ms flush; 250 ms post-flush cooldown against echo re-trigger;
browser/Windows AEC before VAD; record only heard words on interrupt.
Sources: https://www.syncsoft.ai/en/blog/voice-agent-barge-in-vad-tuning-2026 ,
https://rajatpandit.com/agentic-ai/real-time-audio-vad
- Modes: push-to-talk (simplest, no VAD tuning, best for office/noise) → VAD-gated
auto-turn (500 ms→dynamic semantic endpointing) → full-duplex barge-in (needs AEC +
streaming TTS). Local CPU loop proof (Silero + Parakeet INT8 + Supertonic, 2.4 s→307 ms
STT on i7) shows office-viable CPU-only chain. Source: https://www.reddit.com/r/artificial/comments/1u3r85g/i_built_a_100_local_cpuonly_voice_loop_for_any/
- Recommended Cerebro default: whisper.cpp large-v3-turbo-q5_0 (or Parakeet q8 GGUF where
EN/European-only) + Silero VAD + Kokoro-int8 ONNX; push-to-talk first, VAD auto-turn
second, barge-in behind a flag. All-spawned-as-child-process (argv only, no shell),
loopback-only ports, opt-in update check only (matches offline policy).

## 4. Gaps / what to verify on the reference laptop
- No published RTX 3070 Ti Laptop batch-1 STT/TTS latencies found (unverified); measure:
turbo-q5_0 whisper.cpp CUDA, Parakeet q8 NeMo-speech.cpp, Kokoro-int8 DirectML vs CPU,
Qwen3-TTS-0.6B CUDA.
- Whisper short-bench overstatement (~3x vs soak, per brief) applies to Parakeet/Chatterbox
RTFx too — soak-test before claiming wins.
- Licences to re-check at bundle time: Piper GPL fork, Supertonic OpenRAIL-M, Kyutai/Parakeet
CC-BY-4.0 attribution, Chatterbox PerTh disclosure (EU AI Act Art. 50).
