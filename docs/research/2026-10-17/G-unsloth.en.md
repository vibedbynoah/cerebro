> English translation of G-unsloth.md, sections 0 to 5.4 only (the translation run stopped there). Sections 6 onward, including the build checklist and the sources, are in the original G-unsloth.md.

# Unsloth (Core + Studio + Desktop) — A Practical Handbook for Training, Memory, Quantization, Multimodality, Permissions, and the Desktop App, plus a Plan for Building a Cerebro Counterpart

*Research snapshot: October 17, 2026. Primary sources are the official Unsloth documentation (unsloth.ai/docs), the GitHub repository `unslothai/unsloth`, the llama.cpp documentation, and Hugging Face. Any information that cannot be confirmed from a first-hand source is marked **UNVERIFIED**. All sources and a 25-item build checklist are appended at the end of this document.*

---

## 0. Executive Summary

Unsloth has evolved from "a faster LoRA fine-tuning library" into a complete local AI workspace. It now comes in three forms: **Unsloth Core** (a Python package), **Unsloth Studio** (a local browser web interface, `unsloth studio`, FastAPI backend + frontend), and **Unsloth Desktop** (which the official docs explicitly describe as "a Tauri-based app," supporting macOS/Windows/Linux/WSL). It claims to unlock 500+ models, covering text, vision, TTS/audio, embeddings, diffusion image/video, and RL; 2× faster training, 70% VRAM savings, lossless accuracy, support for ultra-long context (500K-token context for training gpt-oss-20b on a single H100), and support for QLoRA/LoRA/FFT/pretraining/GRPO/DPO/FP8 as well as export to GGUF/NVFP4/FP8, and more.

For Cerebro (Tauri 2 + SolidJS, a self-built Rust inference engine, CUDA + AVX2/VNNI CPU, GGUF, MoE offloading — target hardware being an i9-12900H / 32 GB RAM / RTX 3070 Ti Laptop 8 GB), there are three main takeaways:

1. **Training mode is entirely achievable on the desktop**, but Unsloth itself is a Python/Triton stack. The most pragmatic short-term approach is for Cerebro to manage a stub process maintained by `uv` (Python 3.11–3.13, `unsloth` with `transformers` v5 pinned, `trl`, etc.) orchestrated through its own task runner. For model loading, quantization, export, and most inference, Cerebro already has native Rust capabilities.
2. **Unsloth's VRAM estimation formulas are public**, coming from the Studio backend's `VRAM_ESTIMATION.md` and the frontend's `vram.ts`. Cerebro should port them into a shared Rust crate (`cerebro-mem`) so that they can uniformly drive the Model Hub's fit badges (recommended / tight / out-of-VRAM), the training page, and chat.
3. **Unsloth's strongest facet is actually its release engineering — "Day Zero" support** — pre-release quantization work, Triton kernels, upstream fixes for llama.cpp/transformers/vLLM, and evaluations covering quantization quality (KL divergence, Divergence-300@32, top-1 accuracy). Cerebro cannot replicate the scale of their partnerships, but it can build an architecture registry + conversion CI + public metrics, so that its own GGUF and quantization schemes become "day-one ready."

---

## 1. Unsloth's Current Form

- The docs describe Unsloth as "an open-source framework for running and training LLMs" — supporting macOS, Linux, Windows, NVIDIA, AMD, Intel, and pure-CPU setups. ([docs index](https://unsloth.ai/docs))
- Three forms: **Unsloth Desktop** (native app), **Unsloth Studio** (browser interface, `unsloth studio [-p 8888] [-H 0.0.0.0] [--secure]`), **Unsloth Core** (Python package). ([install](https://unsloth.ai/docs/get-started/install), [requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements))
- Desktop is built on **Tauri**: "Unsloth is a Tauri-based app. Install it, download a model, and start chatting!" Supports GGUF, MLX, diffusion image/video and audio models; native multi-GPU inference. ([desktop](https://unsloth.ai/docs/desktop))
- Inside Studio: `studio/backend` (FastAPI, `auth/` JWT, `routes/{training,models,inference,datasets,auth}`, `core/` training engine, `utils/hardware`) and `studio/frontend`, with feature modules (auth, training, studio, chat, export, onboarding) — the frontend uses React/shadcn/Zustand. ([Studio start](https://unsloth.ai/docs/new/studio/start))
- Key figures: LoRA/QLoRA training is "2× faster, 70% less VRAM, lossless accuracy"; MoE training is 12× faster; 30% less VRAM comes from `"unsloth"` gradient checkpointing; gpt-oss-20b QLoRA uses 14 GB VRAM; Qwen3.8-27B QLoRA runs in 24 GB VRAM. ([requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements), [MoE](https://unsloth.ai/docs/basics/faster-moe), [Qwen3.8 train](https://unsloth.ai/docs/models/qwen3.8/train))

---

## 2. Training Modes

### 2.1 Full LoRA / QLoRA Workflow and Defaults

Standard invocation (Unsloth Core):

```python
from unsloth import FastModel            # or FastLanguageModel / FastVisionModel
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit",  # use -unsloth-bnb-4bit weights for QLoRA
    max_seq_length = 2048,
    load_in_4bit = True,      # QLoRA; False = bf16
    load_in_16bit = True,     # 16-bit LoRA (dense models)
    full_finetuning = False,
    offload_embedding = True,
)
model = FastModel.get_peft_model(
    model,
    r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
    use_gradient_checkpointing = "unsloth",   # saves an extra 30% VRAM over True
    random_state = 3407, use_rslora = False, loftq_config = None,
)
```
([Qwen3.8 train](https://unsloth.ai/docs/models/qwen3.8/train), [gpt-oss fine-tune](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss))

Verified hyperparameter guidance (from the hyperparameter guide):

- Rank `r` ∈ {8, 16, 32, 64, 128}; 16 or 32 recommended. `lora_alpha = r` or `2r`; RS-LoRA changes the scaling factor to α/√r.
- `lora_dropout = 0` is already optimized; the Studio UI default is 0.05. Recommend applying LoRA to all seven linear layers (`q,k,v,o,gate,up,down`); skipping the MLP reduces the ability to approximate FFT on large models.
- Effective batch size = batch size × gradient accumulation steps.
- Recommended training steps/epochs: LoRA learning rate 2e-4 (range 1e-4 … 2e-5), typical value `max_steps=60` for testing, `num_epochs=1–3`; FFT uses a lower learning rate.
- Training on completions only (input masking via `train_on_responses_only`) improves accuracy by about 1%, especially in multi-turn tasks.
  ([LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide))

- QLoRA vs LoRA cost: QLoRA reduces VRAM by 4×; for a 70B dense model, Unsloth docs say QLoRA can run under 48 GB.
- One-stop examples: **[gpt-oss-20b](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss)** QLoRA needs 14 GB, bf16 LoRA needs 44 GB; **gpt-oss-120b** QLoRA is 65 GB, LoRA needs 210 GB; **Qwen3.5** bf16 LoRA figures: 0.8B→3 GB, 2B→5 GB, 4B→10 GB, 9B→22 GB, 27B→56 GB (MoE 35B-A3B needs 74 GB); **Qwen3.8-27B** QLoRA fits in 24 GB, LoRA needs >36 GB.

### 2.2 Datasets and Data Recipes

- Format table: raw corpus (continued pretraining), instruction (SFT), conversation (SFT), RLHF (RL). Alpaca/ShareGPT/ChatML are all supported. Practical rule: ≥1000 rows → fine-tune the base model; 300–1000 rows → either works; <300 rows → fine-tune the instruction model. Synthetic data is encouraged; Unsloth advocates "mixing in more general datasets" to increase diversity. ([datasets guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide))
- Studio accepts datasets from the Hub or locally (PDF/DOCX/JSONL/JSON/CSV/Parquet), supports automatic format detection (`auto`/`alpaca`/`chatml`/`sharegpt`), subset/split/slice selection, and a manual column-mapping dialog/preview when automatic mapping fails. ([Studio start](https://unsloth.ai/docs/new/studio/start))
- **Data Recipes** is a graph-node-based synthetic data builder (Seed → LLM text/structured/code/judge blocks → expression (Jinja2) → validators (Python/SQL/TS code checks) → sampler → tool config), with a validate/preview/full-run workflow, and results land in Studio's dataset selector. Under the hood it's powered by NVIDIA NeMo DataDesigner; recipes are stored locally in the browser and can be exported/imported. ([Data Recipes](https://unsloth.ai/docs/new/studio/data-recipe))

### 2.3 Chat Templates

- Built-in registry `unsloth.chat_templates.CHAT_TEMPLATES`, containing templates for llama 3.x, qwen2.5, mistral, gemma 1/2/3, phi-3.5/4, chatml, zephyr, vicuna, alpaca, and more. Unsloth deliberately re-publishes checked, fixed versions of the `chat_template` for every repository it uploads: "Templates uploaded by the original model owners sometimes contain errors."
- Key APIs: `get_chat_template(tokenizer, chat_template=..., mapping=..., map_eos_token=True)`; use `standardize_sharegpt()` first when needed; `conversation_extension` to merge single-turn data into multi-turn; `add_new_tokens(model, tokenizer, [...])` **must** be called before `get_peft_model`; `train_on_responses_only(trainer, instruction_part=..., response_part=...)`.
- Vision models: `UnslothVisionDataCollator(model, processor, ...)` supports `train_on_responses_only`, `completion_only_loss=True`, `resize` strategy, and `snap_to_patch_size`. ([chat templates](https://unsloth.ai/docs/basics/chat-templates), [vision fine-tuning](https://unsloth.ai/docs/basics/vision-fine-tuning))

### 2.4 GRPO / RL and Preference Alignment

- GRPO removes the value/reward model; each prompt with N generations, each completion scored by a reward function/verifier, updated using group-normalized advantages. GSM8K reward functions are provided (correctness, integer, strict format, soft format, XML count) plus proximity-based rewards; care must be taken to avoid reward hacking and to prevent it by sandboxing code-reward generation (e.g., blocking non-standard-library imports, restricting `locals/globals` to `{}`). ([RL guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), [gpt-oss RL tutorial](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/gpt-oss-reinforcement-learning/tutorial-how-to-train-gpt-oss-with-rl))
- Algorithms supported via `GRPOConfig`: `loss_type` ∈ `grpo | dr_grpo | dapo | bnpo` (default `dapo`), GSPO using `importance_sampling_level="sequence"`, plus `epsilon`/`epsilon_high`, `mask_truncated_completions`, `scale_rewards` ∈ {group, batch, none}. ([advanced RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation), [GSPO](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/gspo-reinforcement-learning))
- Throughput: vLLM generation speeds up to 11× faster, and in Unsloth's RL runs, "training time <4%, with pure vLLM inference accounting for 96%." The docs give a typical task-2048 environment GRPO example: learning rate 5e-5, temperature 1.0, `num_generations=2`, `optim="adamw_8bit"`, with gradient accumulation steps raised to 4 for stability.
- **FP8 RL and Unsloth Standby** are RL's VRAM tricks:
  - `load_in_fp8=True` → FP8 frozen weights + a shared FP8 buffer as the vLLM weights (a single copy), BF16 LoRA adapters, dequantized during backprop; claimed ~1.4× speedup, 60% less VRAM, and 10× the context length of other FP8 implementations; Qwen3-1.7B FP8 GRPO fits in 5 GB VRAM; works on consumer GPUs after the 4090.
  - `UNSLOTH_VLLM_STANDBY=1` → shares model weights between inference and training; the KV cache and training activations/optimizer/gradients can reuse the same space; safely raises `gpu_memory_utilization` to 0.95; cites an H100 experiment: Qwen3-32B 16-bit LoRA context length goes from 3,600 → 6,144 (1.7×), with no more OOMs, whereas previously TRL+LoRA with gradient accumulation 2 would crash on a 4B model beyond about 1–2K.
  ([FP8 RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning), [Memory efficient RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl))
- Preference optimization: DPO/ORPO/KTO/SimPO/PPO all run under Unsloth. ([DPO/ORPO/KTO](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto))

### 2.5 VRAM Tricks (Common to SFT and RL) — Unsloth's VRAM tricks stack as follows:

1. `use_gradient_checkpointing="unsloth"`: offloads activations to CPU memory (introduced April 2024, now an industry standard); newer versions use CUDA Streams to reduce the cost, with ≤0.1% overhead.
2. **Fused and chunked cross-entropy** loss: computes chunked logits after flattening `(batch × seq)`, automatically adjusting the chunk size to free VRAM; claimed to reduce VRAM by 60% and support 3.2× context length.
3. **Tiled MLP** (derived from Snowflake Arctic long-sequence training): `unsloth_tiled_mlp=True`; shards by `num_shards = ceil(seq_len/hidden_size)` at the cost of about 1.3× step time; can automatically patch any MLP module; a single H100 supports gpt-oss-20b QLoRA at 500K context (up to 750K on a 192 GB B200).
4. **MoE Split LoRA**: avoids materializing `B·A` for all experts (reorders the matrix multiplication to exploit associativity); claimed to save about 35% VRAM and be 2× faster than Transformers v5 + `torch._grouped_mm`, and 12–30× faster than the v4 curve; the backend is switchable via `UNSLOTH_MOE_BACKEND` (default `grouped_mm`, or `unsloth_triton` / `native_torch`); router-layer fine-tuning is disabled by default for stability reasons; **QLoRA 4-bit on MoE is currently not recommended** (a BitsAndBytes limitation). gpt-oss-20b fine-tunes in 12.8 GB VRAM; Qwen3-30B-A3B 16-bit LoRA uses 63 GB.
   ([Long context blog](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning), [MoE](https://unsloth.ai/docs/basics/faster-moe))

### 2.6 Local Trainer Interface (Unsloth Studio) — Exact Behavior

The Studio main page contains four cards, **Model / Dataset / Parameters / Train & Configure**:

- **Model type** selector: Text / Vision / Audio / Embedding; **Method** pill selector: QLoRA (lowest VRAM) / LoRA (medium) / Full fine-tuning (highest). Searchable HF Hub or local `~/.unsloth/studio/models` directory. **GGUF models are excluded from training — inference only.** Default hyperparameters are auto-filled after model selection; for gated models, the HF token is validated in real time.
- **Dataset** tab: HF Hub or local (drag & drop); auto/alpaca/chatml/sharegpt format selection; subset/train split/eval split/slice; column-mapping preview dialog (`instruction`, `input`, `output`, `image`, etc.).
- **Default hyperparameters** (table): context length 2048 (512→32768); learning rate 2e-4; max steps 0 (= computed by epochs); rank 16 (slider 4–128); α 32 (slider 4–256); dropout 0.05; LoRA variant LoRA / RS-LoRA / LoftQ; all seven target modules on; vision models add four checkboxes (vision layers / language layers / attention modules / MLP modules). Training hyperparameters: 3 epochs, batch size 4, gradient accumulation 8, weight decay 0.01, optimizer AdamW 8-bit; scheduler linear, warmup 5 steps, gradient checkpointing `unsloth`, random seed 3407, save/eval steps 0, **packing false**, **train on completions only false**; WandB/TensorBoard off, logging frequency 10.
- **Configuration management**: upload / save / reset YAML; auto-naming `{model}_{method}_{dataset}_{timestamp}.yaml`; structure split into `training:` / `lora:` / `logging:`.
- **Training interface**: the start-training button is disabled until both model and dataset are present; inline validation (setting eval steps when there's no eval split, a vision dataset paired with a text-only model, etc.); a full-page transition shows terminal stages in blue (downloading) / amber (loading) / blue (configuring) / green (training), cancellable via ×; the live view shows epoch (1.23), a step-based progress bar, loss to 4 decimal places, learning rate in scientific notation, gradient norm, model/method, elapsed time / estimated remaining time / steps per second / total tokens; GPU monitoring (utilization %, temperature in °C, VRAM used/total GB, power watts/limit), polled every few seconds; four charts (loss + EMA smoothing + running mean, learning rate, gradient norm, eval loss optional) each with gear settings (window, EMA 0.6, raw/smooth/mean toggle, log scale, outlier clipping p95/p99); **Stop and Save** and **Cancel (without saving checkpoint)** dialogs.
  ([Studio start](https://unsloth.ai/docs/new/studio/start))

### 2.7 Exporting to GGUF and Other Formats

- Core API: `model.save_pretrained_gguf("dir", tokenizer, quantization_method="q4_k_m" | "q8_0" | "f16" ...)`; `save_pretrained_merged(..., save_method="merged_16bit")`; for gpt-oss use the native `save_method="mxfp4"` format; LoRA adapters can be pushed via `save_pretrained` / `push_to_hub`. The Qwen3.8 guide also mentions NVFP4 / FP8 export options. ([saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf), [Qwen3.8 train](https://unsloth.ai/docs/models/qwen3.8/train))
- Studio export interface: select training run → checkpoint → export method (**Merge model** 16-bit / **LoRA only** / **GGUF for llama.cpp/Ollama/LM Studio/Unsloth**) → save locally or push to the Hub with an HF write token; the detailed tutorial is at `/docs/new/studio/export`.
- Deployment narrative: dedicated guides are provided for GGUF, vLLM, SGLang, Ollama, LM Studio, llama-server, and smartphones (ExecuTorch); Chat offers a **comparison mode** that runs the same prompt side by side on the original model and the LoRA fine-tuned model.

### 2.8 Native Rust/Desktop vs. Interfacing with Python

Unsloth is entirely Python-based (PyTorch + Triton kernels + Transformers/TRL/PEFT + bitsandbytes + vLLM, with llama.cpp involved in GGUF conversion/quantization). There is no official Rust API. For Cerebro:

- **Short term**, interface with a `uv`-managed sidecar process, pinning Python 3.11–3.13 (a Studio requirement), pre-installing unsloth + trl + transformers v5 (the Qwen3.5 series requires "please use `transformers v5`"), and on Windows additionally requiring the CUDA toolkit/CMake/VS Build Tools (Studio installs these components via `winget`) plus model weights cached from HF.
- **Medium term**, migrate the large non-training chunks back to the Rust side: dataset preprocessing (the tokenizer is portable to Rust: render chat templates directly using `tokenizer.json` straight from the GGUF metadata), GGUF quantization/conversion (Cerebro already parses GGUF; quantization can carry out the same kinds of k-quant/i-quant math as `llama.cpp`, or call the bundled `llama-quantize` directly), and the manager/monitoring/UI (NVML, CUDA-visible devices).
- **Long term**, rewrite the hot kernels in CUDA/cudarc: chunked fused cross-entropy (Unsloth's `chunked_hidden_states_selective_log_softmax` is exactly the portable part), activation-offloading checkpointing, sharded MLP, and fused 8-bit AdamW for MoE expert routing. These are proprietary techniques that can bring Cerebro's engine on par with Unsloth without relying on Python.
- Minimum viable parameters: trainer task = subprocess + process group: set `CUDA_VISIBLE_DEVICES` (Studio's implementation treats this as a precondition), prefer the GPU with the most free VRAM (`free[0] + 0.85×free[others]` as the available capacity), stream step metrics (Studio uses SSE `/api/train/stream`). In Rust, use `tokio::process` with `kill_on_drop` configured, and place a JSONL protocol between the sidecar process and Cerebro.

### 2.9 Cerebro Counterpart/Surpass Plan

- Ship a **training tab** created under the app data directory on first run, install a pinned Python environment (`uv venv --python 3.12`, `uv pip install unsloth`, etc.) and treat it as an implementation detail; show versions and provide a "reinstall/repair" action.
- Design the trainer as tasks: queue, snapshot/YAML config, resume, cancel, checkpoint. All numbers displayed live in the UI (see the field list in §2.6).
- Because Cerebro controls its engine, it has opportunities Studio lacks: (a) train the same GGUF concurrently so it can run immediately after training finishes, with no export needed (a "train, then chat" loop); (b) load the same adapter into the Rust side for inference (LoRA hot-swap); (c) present real data instead of predictions: compare the memory estimator's predictions against NVML-measured values and calibrate correction factors per model (Unsloth itself calibrates its estimator in exactly this way).

---

## 3. Memory Usage Estimator — Exact Formulas

### 3.1 Training VRAM Model (Unsloth Studio Backend, Primary Source)

`studio/backend/utils/hardware/VRAM_ESTIMATION.md` (from the `unslothai/unsloth` repository): total VRAM = weights + LoRA adapters + optimizer + gradients + activations + CUDA overhead.

Notation: `H` hidden size, `L` number of layers, `V` vocabulary, `K = (H/num_attention_heads)×num_kv_heads`, `M` intermediate/expert intermediate size, `E` number of experts (1 for dense models), `r` LoRA rank, `B` per-device batch size, `S` max sequence length.

**1. Model Weights**
```
QKVO = (H + K + K + H) * H
MLP  = H * M * 3 * E  +  (E * H if E > 1 else 0)
Quantizable     = (QKVO + MLP) * L
Non-quantizable = 2*H*L + V*H + (V*H if not tie_embeddings else 0)

QLoRA 4-bit:  Quantizable * 2 / 3.2 + Non_quantizable * 2   # 3.2 ≈ 16/5, for NF4 block-wise scales
              (use 3.6 when double-quant is enabled)
LoRA/FFT fp16: (Quantizable + Non_quantizable) * 2
```
Embedding layers, the LM head (unless tied), and normalization parameters stay fp16 (`Non-quantizable`). A 4-bit config with `llm_int8_skip_modules` pointing to the language layers bills those parts at fp16. The estimator respects per-layer config fields (`layer_types`, `head_dim`, `num_kv_shared_layers`, `use_double_wide_mlp`, etc.) — when such information is present, weights are derived from per-layer shapes.

**2. LoRA Adapters** — For each selected module, the A/B shapes are: `q: H×r, r×H; k: H×r, r×K; v: H×r, r×K; o: H×r, r×H; gate: H×r, r×M; up: H×r, r×M; down: M×r, r×H`; MLP modules are multiplied by `E` to accommodate MoE.
```
LoRA_bytes = sum(A + B per selected module) * L * 2
```
`"all-linear"` = all text linear layers listed in the table above. Vision tower modules are deliberately excluded from the estimate (they vary too much between models).

**3. Optimizer (calibrated)** — `adamw_8bit` 4 bytes/parameter (upcast to fp32 during BNB updates); `adamw_torch` 6; `paged_adamw_32bit` 8; `sgd` 4. Trainable parameters = LoRA only, or all parameters for full fine-tuning.

**4. Gradients** — `Gradient_bytes = trainable_params * 2`.

**5. Activations** — per layer:
```
Per_layer = (S*B*(H+K+K) + S*B*2 + S*B*(M+M)) * 2 * 1.25
```
If the attention implementation is neither `flash_attention_2` nor `sdpa`/`flex_attention`, use the quadratic term as a lower bound:
```
Non_flash = B * num_attention_heads * S^2 * 2 * 12.0 * effective_layers
Activations = max(Per_layer_with_gc, Non_flash)
```
Gradient checkpointing effective-layer multiplier: no checkpointing `L`; HF `True` → full fine-tuning 2.0 / LoRA 1.0; `"unsloth"` → 1.5 / 1.0.

**6. Floor** — Full fine-tuning: `gradient_bytes = max(computed, weights * 0.15)` (autograd/NCCL/sharding overhead). LoRA/QLoRA:
```
raw = trainable_params * 2
floor = min(weights * 0.15, max(activation_bytes, optimizer_bytes))
gradient_bytes = max(raw, floor)
```

**7. CUDA Overhead** — fixed **1.4 GB** (CUDA driver + PyTorch runtime, calibrated on an RTX 5070 Ti).

**8. Multi-GPU** — `usable = free[gpu0] + Σ free[gpui] × 0.85` (a 15% discount per additional GPU, for NCCL/PCIe/fragmentation; calibrated on 2–8 GPU configurations).

Parameters that can trigger re-estimation: `batch_size`, `max_seq_length`, `lora_r`, `target_modules`, `gradient_checkpointing`, `optim`.
([VRAM_ESTIMATION.md](https://github.com/unslothai/unsloth/blob/744b59f0/studio/backend/utils/hardware/VRAM_ESTIMATION.md))

### 3.2 Inference/Loading Estimator (Model Hub Fit Badges)

`studio/frontend/src/lib/vram.ts` computes the VRAM required to load a 4-bit model:
```
estimateLoadingVram(params, method) = params * bytesPerParam + 1.4 GB
   QLoRA: 0.90 bytes/parameter (calibrated from the raw bnb 0.5, with embeddings/lm_head in fp16 + block metadata)
   LoRA/FFT: 2.0 bytes/parameter (theoretical)
Fits = need/available ≤ 0.75; Tight = ≤1.0; Exceeds >1.0.
Calibration examples: Qwen2.5-0.5B estimated 1.8 vs actual 1.86 GB; Llama-3.2-1B estimated 2.5 vs actual 2.54; 3B estimated 4.3 vs actual 4.40; Llama-3.1-8B estimated 8.6 vs actual 8.14 (±6%).
```
([vram.ts](https://github.com/unslothai/unsloth/blob/9a261aec/studio/frontend/src/lib/vram.ts))

For GGUF inference (Cerebro's scenario), the industry rule is simple but effective: **compare file size + KV cache + compute buffer against RAM+VRAM**; Unsloth's guidance for each GGUF is "RAM+VRAM ≈ quantized file size, otherwise it can still run but will be slower due to offloading" ([Kimi K2.6 guide](https://unsloth.ai/docs/models/kimi-k2.6)). `llama-server` reports the exact segment breakdown at runtime (model buffer, KV buffer, compute buffer, and expert offloading via `-ot ".ffn_.*_exps.=CPU"`), which Cerebro can simulate at load time.

### 3.3 KV Cache

The standard inference formula for GQA/MHA models:
```
kv_bytes/token = 2 (K and V) × n_layers × n_kv_heads × head_dim × bytes_per_element
kv_total = kv_bytes/token × seq_len × batch
```
An fp16 element = 2 bytes; a q8_0 KV cache roughly halves it (llama.cpp: `--cache-type-k q8_0 --cache-type-v q8_0`). Notorious special cases: **MLA** (DeepSeek style) compresses to `(kv_lora_rank + qk_rope_head_dim) × layers × 2`; **hybrid attention/linear** models (Mamba/SSM class) count only full-attention layers — the SSM state has its own fixed-size recurrent cache (e.g., the llama.cpp log "llama_memory_recurrent: RS buffer size ... R (f32)/S (f32)"). A quantized example: Qwen3.8-Flash-Next, with 262,144 cells and 12 KV layers at q8_0 → llama.cpp reports 3,264 MiB (1,632 K + 1,632 V) plus a second 1,224 MiB cache (the indexer/dummy KV shown in the docs). ([llama.cpp multimodal](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md), HF discussion log from [Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/8); third-party KV formula cross-check: [llm-vram-calculator](https://github.com/collindjohnson/llm-vram-calculator) — a third-party formula; the official Unsloth estimator uses the per-layer activation values in §3.1 above, not the inference KV model.)

### 3.4 MoE: Activated Parameters vs. Total Parameters

- **Weights/VRAM**: All experts must be resident; use the total parameters in the §3.1 formulas, where `E` multiplies the MLP. For quantization baselines, the same rule applies: MoE quantized size is driven by *total parameters*, while speed is driven by *activated parameters*.
- **Training**: Unsloth's estimator's activation formula does not multiply by `E` (only activated experts produce layer activations), and the router layer is frozen by default. MoE's "Split LoRA" separates the typical LoRA parameters `r ≪ n` from the expert-matrix routing parameters `k(m+n)`; the docs derive that applying LoRA to the expert matrices is better when the number of experts is large — but the printed threshold inequality was truncated at capture time, so treat the exact threshold as **UNVERIFIED**, although the guidance to "apply LoRA to the MoE layers (`gate_up_proj`, `down_proj`)" and "never fine-tune the router unless you want to" is clear.
- **Inference**: MoE architectures support offloading experts to CPU while attention/KV stays on the GPU (which is exactly what Cerebro does; the llama.cpp equivalent is `-ot ".ffn_.*_exps.=CPU"`). The best token rate is achieved when RAM+VRAM ≥ file size.

### 3.5 Applied Example for the Target Hardware: i9-12900H, 32 GB, RTX 3070 Ti 8 GB

- QLoRA on an 8B language model at r=32, batch 1: Unsloth's table gives a minimum VRAM of 6 GB, but the **measured maximum context length on an 8 GB card is about 2,972 tokens** (HF+FA2: out of VRAM); 12 GB → 21,848 tokens; 16 GB → 40,724. So on an 8 GB laptop GPU, 8B QLoRA training is functionally feasible, but only for short contexts, and it leaves no VRAM headroom to accommodate a second model.
- bf16 LoRA on 8B (can't use 4-bit): about 22 GB → never viable for 8 GB.
- LoRA parameters for an 8B model at r=32 under `adamw_8bit`: roughly `2·r·(in+out)` parameters per adapter × 7 modules × 32 layers ⇒ tens of millions of parameters → the optimizer at ~4 bytes/parameter is only a few hundred MB; weights dominate (about 4.5 GB at 4-bit + embeddings/head at fp16).
- MoE training (Qwen3-30B-A3B bf16 LoRA 63 GB; gpt-oss-20b 12.8 GB) — not feasible on a laptop with 8 GB; MoE QLoRA isn't recommended anyway.
- Inference: dense models at the UD-Q4_K_XL level ≤8B fit easily; Gemma 4 12B 4-bit "runs on 8 GB VRAM" and Qwen3.5 9B "runs on 12 GB RAM/VRAM" are public reference points; MoE-class models (Qwen3.5-35B-A3B about 22 GB) need to leverage 32 GB RAM for expert offloading. (Sources: Requirements and Pablo Caro's notes, a third party summarizing Unsloth's per-model pages — **UNVERIFIED** as an independent source, but consistent with Unsloth's own tables.)

### 3.6 Cerebro Plan

- Port §3.1 and §3.2 into a Rust crate (`cerebro-mem`), using the same symbol-typed struct populated from `config.json`/GGUF metadata (`h, l, v, kv_heads, heads, m, e, tie, per-layer overrides`). Pure functions: weights/adapters/optimizer/gradients/activations/overhead, plus `fits/tight/exceeds` thresholds 0.75/1.0.
- The Model Hub badge is driven by the same estimator for every quantization, using the measured GGUF byte size for the inference skeleton, and using the KV formula when it's MoE with offloaded experts.
- Add a Rust version of the MoE multi-GPU discount (0.85), preparing for future bundled/multi-GPU modes.
- Collect real data: record NVML-measured values at each training step, store a calibration ratio per model (Cerebro's estimator error table, just as Unsloth does for its 0.90 bytes/parameter calibration), and shrink the error over time.

---

## 4. Model Recommendations by Hardware

### 4.1 Unsloth Official Data

Fine-tuning VRAM table (absolute minimums): 3B 3.5/8 GB (QLoRA/LoRA), 7B 5/19, 8B 6/22, 9B 6.5/24, 11B 7.5/29, 14B 8.5/33, 27B 22/64, 32B 26/76, 40B 30/96, 70B 41/164, 81B 48/192, 90B 53/212, 405B 237/950 GB. ([requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements))

Context length (Llama 3.1 8B QLoRA r32, batch 1; and 70B): see the table in §3.5. The same page also recommends "keep the batch size at 1, 2, or 3." ([benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks))

Run guides by family (excerpted from the individual model pages):

- **Qwen3.8**: 27B runs in 17 GB RAM/VRAM (4-bit); on 24 GB VRAM NVFP4 is about 1.5× faster than BF16; a calibrated 8-bit KV cache enables 2× context. ([Qwen3.8](https://unsloth.ai/docs/models/qwen3.8))
- **Qwen3.5**: small models ≤9B run in 12 GB; 35B-A3B about 22 GB; 122B-A10B needs 256 GB VRAM for LoRA.
- **Gemma 4**: 12B on 8 GB (4-bit), 26B-A4B about 18 GB, 31B about 20 GB. ([Pablo Caro's notes on the Unsloth docs](https://pablocaro.es/en/unsloth-docs) — an **UNVERIFIED** summary, but its content is consistent with Unsloth's pages; verify with the individual model pages when writing.)
- **gpt-oss**: 20b inference about 14 GB 4-bit; GRPO in the 12.8–14 GB range; 20b QLoRA training 14 GB / bf16 LoRA 44 GB.
- **Kimi K2.6**: recommends UD-Q2_K_XL about 350 GB; lossless UD-Q8_K_XL is 10 GB larger than Q4; needs ">40 tokens/s" on a B200.
- **DeepSeek-V4-Flash**: UD-Q8_K_XL 162 GB lossless (MXFP4 expert repackaging); UD-IQ3_XXS fits 128 GB RAM machines.

### 4.2 Target Hardware (8 GB NVIDIA Laptop GPU + 32 GB System Memory)

**Training (QLoRA/LoRA/FFT):**
- Comfort zone: 1.5B–4B QLoRA (table: 3B ≈ 3.5 GB), with context up to several K tokens; the model ceiling is around 14B QLoRA (8.5 GB), feasible only with short context and strictly VRAM-saving settings.
- Not recommended: 8B QLoRA beyond ~3K context; any bf16 LoRA ≥7B; any MoE LoRA/QLoRA; FFT beyond ~1B.
- Target techniques: `gradient_checkpointing="unsloth"`, batch 1–2, gradient accumulation to make up the effective batch size, `adamw_8bit`, train on completions only, embedding offloading where possible (`offload_embedding`).
**Inference:**
- Comfort zone: ≤8B UD-Q4_K_XL (about 4–5 GB file + 1–3 GB KV, depending on context); the Gemma 4 12B-class 4-bit "fits 8 GB" claim; 14B 4-bit under a compact context; 30B-class MoE (A3B) by offloading experts to 32 GB of memory (Cerebro's MoE offloading already does this).

### 4.3 Cerebro Plan

Build a "hardware-aware model recommender" in the Model Hub: for each GGUF, precompute (file size, KV/token, expert bytes, mmproj size) and, against 8 GB VRAM + 32 GB memory, produce a recommended configuration (ngl allocation, expert offloading strategy, context size, KV quantization), with Recommended/Tight/Insufficient badges, plus a "For Training" button that recommends a fitting QLoRA size in that family with preset parameters. This surpasses Unsloth *Desktop* (which shows badges per quantization), because it also recommends the *training* target and configures offloading — a unified advantage for Cerebro because it controls its own runtime.

---

## 5. Day-One Support and Multimodality (Audio, Vision, Video, Image)

### 5.1 How Unsloth Achieves Fast Releases

- **Partnerships + pre-release access.** "We work directly with the teams behind gpt-oss, Qwen3, Llama 4, Mistral, Gemma 1-3, and Phi-4, fixing serious bugs that severely improve model accuracy." Qwen3.8: "Thanks to Qwen for day-one access." Homepage: "Day-one support for Qwen3.8, NVIDIA, GLM, Gemma, and more." ([docs index](https://unsloth.ai/docs), [Qwen3.8](https://unsloth.ai/docs/models/qwen3.8), [homepage](https://unsloth.ai/))
- **Starting work before the weights are public.** Qwen3.8-Flash-Next: the team published quantization/kernel work *the day before release* ("working hard toward day-one support ... the architecture is extremely new, there may be delays"), with final support landing the next day via an update that drove llama.cpp PR #27742; user logs showed `GGML_ASSERT(inp->self_k_rot == nullptr ...)` crashing before the patch was complete. ([LinkedIn post](https://www.linkedin.com/posts/unsloth_qwen-announces-qwen38-flash-next-a-new-activity-7497985396354633729-bwxg) — **UNVERIFIED** social post; [HF discussion](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/8))
- **Cross-repository kernels and bug fixes.** Triton kernels (backed by hand-written backward passes, "2× faster, 70% less VRAM"). Published fixes: Llama 4 RoPE scaling and QK-Norm epsilon 1e-5 (llama.cpp PR #12889, transformers PR #37418); a bug fix for shared QK-Norm in vLLM raised MMLU-Pro from 68.58 to 71.53; Qwen3 128K chat template fix; Gemma 1-3 and Phi-3/4 fixes. ([dynamic GGUFs page](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs))
- **The quantization pipeline runs before public release** — calibration/importance-matrix work, per-tensor layer selection, and KLD/Divergence testing are performed before or on release day (see §7).
- A team member on Reddit said: "We usually manage day one." The community notes that delays stem from importance-matrix generation and quantization-sensitivity A/B testing. ([r/unsloth discussion](https://www.reddit.com/r/unsloth/comments/1mm3lol/why_is_there_lag_between_an_open_llm_release_and/) — **UNVERIFIED**, user-generated discussion)

### 5.2 Model Family Coverage

It claims to run and train 500+ models, covering text, vision, TTS, audio, embeddings, and RL. The day-one list at the August 2026 Desktop release: Qwen3.8, Muse Glimmer 30B, MiniMax H3, Kimi K3, DeepSeek-V4 Flash, Gemma 4. ([substack](https://unslothai.substack.com/p/introducing-unsloth-desktop) — **UNVERIFIED** as a marketing newsletter, but consistent with the model pages in the docs.)

### 5.3 Multimodality: The GGUF / llama.cpp / mtmd Path

- llama.cpp supports multimodal input via **libmtmd**. "Currently, we support **image**, **audio**, and **video** input." Two loading modes: `-hf repo:QUANT` (auto-matches mmproj) or `-m model.gguf --mmproj mmproj-F16.gguf`; `--no-mmproj` / `--no-mmproj-offload` switches; an OpenAI-compatible multimodal API for `/chat/completions`; `llama-mtmd-cli` for testing. ([llama.cpp docs/multimodal.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md))
- Ready-made multimodal GGUFs (the ggml-org collection, with Unsloth publishing the corresponding paired files): vision — the Qwen3-VL series, etc.; audio — **Ultravox 0.5** (Llama-3.2-1B/3.1-8B), **Voxtral-Mini-3B**, **Qwen3-ASR-0.6B/1.7B**; hybrid — Qwen2.5-Omni 3B/7B, Qwen3-Omni-30B-A3B, Gemma 4 E2B/E4B.
- Unsloth GGUF repositories ship with `mmproj-F16.gguf` paired files (e.g., vision for [Kimi K2.6](https://unsloth.ai/docs/models/kimi-k2.6), loaded as `--mmproj unsloth/Kimi-K2.6-GGUF/mmproj-F16.gguf`).
- Gemma 4 12B Unified uses a single `gemma4uv` projector to unify vision+audio in one 117 MB mmproj; its release story is a perfect case study: on day one the mmproj crashed with SIGFPE due to `clip.vision.attention.head_count = 0` (llama.cpp issue #24085), was fixed on June 4 via converter PR #24118, ggml-org and unsloth re-uploaded within hours, decoding speed went from about 19 t/s on HF to 64–131 t/s on llama.cpp, and VRAM dropped from 24 GB to 14 GB at Q8_0. ([note.com article](https://note.com/unco3/n/n871e994d27b2?hl=en) — a detailed third-party record reconstructed from first-hand timestamps; **UNVERIFIED** as a single source, but internally consistent and consistent with the issue and PR numbers.)
- **Audio front end**: 16 kHz mono WAV; for the llama.cpp audio path, 30-second clips are the safe range; the server can accept `input_audio` (base64 WAV) in an OpenAI-style payload. ([note.com](https://note.com/unco3/n/n871e994d27b2?hl=en), [LocalAI docs](https://localai.io/docs/features/audio-to-text/index.print.html))
- **Image/video generation**: Unsloth Desktop runs and trains diffusion models — MiniMax-H3, Qwen-Image-2.1, Z-Image, FLUX 1/2 klein, SDXL, LTX — with LoRA/ControlNet and reference images, Gallery, recipe replay, low-VRAM options, and GGUF/FP8 paths; there are also ComfyUI and stable-diffusion.cpp tutorials. ([diffusion-image](https://unsloth.ai/docs/basics/diffusion-image), [ComfyUI blog](https://unsloth.ai/docs/blog/comfyui))
- **TTS/STT**: Fine-tune any `transformers`-compatible TTS model via Unsloth (Orpheus-3B, Sesame-CSM-1B, Spark-TTS, Llasa, Oute-TTS) as well as STT (Whisper large v3); Orpheus models can be exported to llama.cpp and are compatible with a large number of inference engines; "models that aren't general-purpose can only save LoRA adapter safetensors"; TTS/STT fine-tuning recommends 16-bit LoRA rather than QLoRA (`load_in_4bit=False`); Orpheus's tokens generate audio tokens directly, with no vocoder needed; `max_new_tokens=125 ≈ 10 s` of audio. ([TTS guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning))

### 5.4 Cerebro Counterpart/Surpass Plan

- **Architecture registry + conversion CI**: a data-driven registry (attention type, RoPE variant, QK-norm epsilon source, MoE routing, MTP heads, mmproj requirements) to drive the Cerebro engine loader; a CI job triggered on each new Hugging Face architecture release, running a convert-load-smoke test when upstream metadata matches; publish a "supported/unsupported and why" page. This is exactly
