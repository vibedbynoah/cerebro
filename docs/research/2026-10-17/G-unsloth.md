# Unsloth (核心 + Studio + 桌面版) — 训练、内存、量化、多模态、权限与桌面端实战手册，以及 Cerebro 对标建设方案

*研究快照：2026年10月17日。主要来源为 Unsloth 官方文档（unsloth.ai/docs）、GitHub 仓库 `unslothai/unsloth`、llama.cpp 文档及 Hugging Face。任何无法从一手资料确认的信息均标记为 **UNVERIFIED**。本文最后附有全部来源与一个包含25项的构建清单。*

---

## 0. 执行摘要

Unsloth 已从“更快的 LoRA 微调库”演变为一个完整的本地 AI 工作空间。如今它提供三种形态：**Unsloth Core**（Python 包）、**Unsloth Studio**（本地浏览器 Web 界面，`unsloth studio`，FastAPI 后端 + 前端）、以及 **Unsloth Desktop**（官方文档明确描述为“基于 Tauri 的应用”，支持 macOS/Windows/Linux/WSL）。官方宣称解锁 500 多个模型，覆盖文本、视觉、TTS/音频、嵌入、扩散图像/视频以及 RL；训练速度提升 2 倍、显存节省 70%、精度无损、支持超长上下文（单张 H100 上训练 gpt-oss-20b 的 50 万 token 上下文），并支持 QLoRA/LoRA/FFT/预训练/GRPO/DPO/FP8 以及导出 GGUF/NVFP4/FP8 等。

对于 Cerebro（Tauri 2 + SolidJS，自研 Rust 推理引擎，CUDA + AVX2/VNNI CPU，GGUF，MoE 卸载——目标硬件为 i9-12900H / 32 GB RAM / RTX 3070 Ti Laptop 8 GB），有三大启示：

1. **训练模式完全可在桌面端实现**，但 Unsloth 本身是 Python/Triton 技术栈。短期内最务实的做法是：Cerebro 管理一个由 `uv` 维护的桩进程（Python 3.11–3.13，`unsloth` 并锁定 `transformers` v5，`trl` 等），通过其自身的任务运行器进行编排。而其模型加载、量化、导出以及大部分推理，Cerebro 已拥有 Rust 原生能力。
2. **Unsloth 的显存估算公式是公开的**，来自 Studio 后端 `VRAM_ESTIMATION.md` 与前端 `vram.ts`。Cerebro 应将它们移植为一个共享的 Rust crate（`cerebro-mem`），以便统一驱动模型中心的适配徽章（推荐 / 紧张 / 显存不足）、训练页以及聊天功能。
3. **Unsloth 最强的一环其实是其发布工程——"Day Zero" 支持**——预发布的量化工作、Triton 内核、上游 llama.cpp/transformers/vLLM 的错误修复，以及覆盖量化质量（KL 散度、Divergence-300@32、前1准确率）的评测。Cerebro 无法复制它们的合作规模，但可以构建一个架构注册表 + 转换 CI + 公开度量，使其自身的 GGUF 和量化方案达到"首日就绪"。

---

## 1. Unsloth 的当前形态

- 文档称 Unsloth 是"一个开源框架，用于运行和训练 LLM" —— 支持 macOS、Linux、Windows、NVIDIA、AMD、Intel 以及纯 CPU 设置。 ([docs index](https://unsloth.ai/docs))
- 三种形态：**Unsloth Desktop**（原生应用）、**Unsloth Studio**（浏览器界面，`unsloth studio [-p 8888] [-H 0.0.0.0] [--secure]`）、**Unsloth Core**（Python 包）。 ([install](https://unsloth.ai/docs/get-started/install), [requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements))
- Desktop 基于 **Tauri**："Unsloth 是一个基于 Tauri 的应用。安装它，下载模型即可开始聊天！" 支持 GGUF、MLX、扩散图像/视频及音频模型；原生支持多 GPU 推理。 ([desktop](https://unsloth.ai/docs/desktop))
- Studio 内部：`studio/backend`（FastAPI、`auth/` JWT、`routes/{training,models,inference,datasets,auth}`、`core/` 训练引擎、`utils/hardware`）以及 `studio/frontend`，具有功能模块（auth、training、studio、chat、export、onboarding）—— 前端使用 React/shadcn/Zustand。 ([Studio start](https://unsloth.ai/docs/new/studio/start))
- 核心数据：LoRA/QLoRA 训练"快 2 倍，显存节省 70%，精度无损"；MoE 训练提速 12 倍；30% 更少的显存来自 `"unsloth"` 梯度检查点；gpt-oss-20b QLoRA 显存 14 GB；Qwen3.8-27B QLoRA 可在 24 GB 显存下运行。 ([requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements), [MoE](https://unsloth.ai/docs/basics/faster-moe), [Qwen3.8 train](https://unsloth.ai/docs/models/qwen3.8/train))

---

## 2. 训练模式

### 2.1 LoRA / QLoRA 完整流程与默认值

标准调用（Unsloth Core）：

```python
from unsloth import FastModel            # 或 FastLanguageModel / FastVisionModel
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit",  # QLoRA 用 -unsloth-bnb-4bit 权重
    max_seq_length = 2048,
    load_in_4bit = True,      # QLoRA；False = bf16
    load_in_16bit = True,     # 16-bit LoRA（稠密模型）
    full_finetuning = False,
    offload_embedding = True,
)
model = FastModel.get_peft_model(
    model,
    r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
    use_gradient_checkpointing = "unsloth",   # 比 True 额外省 30% 显存
    random_state = 3407, use_rslora = False, loftq_config = None,
)
```
([Qwen3.8 train](https://unsloth.ai/docs/models/qwen3.8/train), [gpt-oss fine-tune](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss))

已验证的超参数指导（来自超参数指南）：

- 秩 `r` ∈ {8, 16, 32, 64, 128}；推荐 16 或 32。`lora_alpha = r` 或 `2r`；RS-LoRA 将缩放因子改为 α/√r。
- `lora_dropout = 0` 已优化；Studio UI 默认值为 0.05。建议将 LoRA 应用于全部七个线性层（`q,k,v,o,gate,up,down`）；若跳过 MLP，会降低大模型上逼近 FFT 的能力。
- 有效批大小 = 批大小 × 梯度累积步数。
- 推荐训练步数/轮数：LoRA 学习率 2e-4（区间 1e-4 … 2e-5），典型值为 `max_steps=60` 用于测试，`num_epochs=1–3`；FFT 使用更低的学习率。
- 仅对补全进行训练（通过 `train_on_responses_only` 进行输入掩码）可提升准确率约 1%，尤其在多轮任务中。
  ([LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide))

- QLoRA 与 LoRA 的成本：QLoRA 显存降低 4 倍；对于 70B 的稠密模型，Unsloth 文档称 QLoRA 可在 48 GB 以下运行。
- 一体化示例：**[gpt-oss-20b](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss)** 的 QLoRA 需 14 GB，bf16 LoRA 需 44 GB；**gpt-oss-120b** QLoRA 为 65 GB，LoRA 需 210 GB；**Qwen3.5** 的 bf16 LoRA 数据：0.8B→3 GB，2B→5 GB，4B→10 GB，9B→22 GB，27B→56 GB（MoE 35B-A3B 需 74 GB）；**Qwen3.8-27B** 的 QLoRA 可适配 24 GB，LoRA 需 >36 GB。

### 2.2 数据集与数据配方

- 格式表：原始语料（继续预训练）、指令（SFT）、对话（SFT）、RLHF（RL）。Alpaca/ShareGPT/ChatML 均受支持。实用规则：≥1000 行 → 微调基础模型；300–1000 行 → 二者均可；<300 行 → 微调指令模型。鼓励合成数据；Unsloth 主张“用更通用的数据集进行混合”来增强多样性。 ([datasets guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide))
- Studio 接受来自 Hub 或本地的数据集（PDF/DOCX/JSONL/JSON/CSV/Parquet），支持格式自动检测（`auto`/`alpaca`/`chatml`/`sharegpt`）、子集/划分/切片选择，以及当自动映射失败时的人工列映射对话框/预览。 ([Studio start](https://unsloth.ai/docs/new/studio/start))
- **数据配方** 是一个基于图形节点的合成数据构建器（种子 → LLM 文本/结构化/代码/法官块 → 表达式（Jinja2）→ 验证器（Python/SQL/TS 代码检查）→ 采样器 → 工具配置），具备验证/预览/完整运行流程，结果会落在 Studio 的数据集选择器中。底层由 NVIDIA NeMo DataDesigner 驱动；配方在浏览器中本地存储，可导出/导入。 ([Data Recipes](https://unsloth.ai/docs/new/studio/data-recipe))

### 2.3 聊天模板

- 内置注册表 `unsloth.chat_templates.CHAT_TEMPLATES`，包含 llama 3.x、qwen2.5、mistral、gemma 1/2/3、phi-3.5/4、chatml、zephyr、vicuna、alpaca 等更多模板。Unsloth 对所有上传的仓库都故意重新发布经过检查的 `chat_template` 修复版本：“原始模型拥有者上传的模板有时包含错误”。
- 关键 API：`get_chat_template(tokenizer, chat_template=..., mapping=..., map_eos_token=True)`；在需要时先使用 `standardize_sharegpt()`；`conversation_extension` 用于将单轮数据合并为多轮；`add_new_tokens(model, tokenizer, [...])` **必须**在 `get_peft_model` 之前调用；`train_on_responses_only(trainer, instruction_part=..., response_part=...)`。
- 视觉模型：`UnslothVisionDataCollator(model, processor, ...)` 支持 `train_on_responses_only`、`completion_only_loss=True`、`resize` 策略以及 `snap_to_patch_size`。 ([chat templates](https://unsloth.ai/docs/basics/chat-templates), [vision fine-tuning](https://unsloth.ai/docs/basics/vision-fine-tuning))

### 2.4 GRPO / RL 及偏好对齐

- GRPO 移除了价值/奖励模型；每个提示与 N 次生成结果，对每个完成结果根据奖励函数/验证器进行评分，按组内标准化优势进行更新。提供了 GSM8K 奖励函数（正确性、整数、严格格式、软格式、XML 计数）和基于接近度的奖励；必须注意避免奖励破解，并通过沙盒化代码奖励生成来防止（例如阻断非标准库导入，将 `locals/globals` 限制为 `{}`）。 ([RL guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), [gpt-oss RL tutorial](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/gpt-oss-reinforcement-learning/tutorial-how-to-train-gpt-oss-with-rl))
- 通过 `GRPOConfig` 支持的算法：`loss_type` ∈ `grpo | dr_grpo | dapo | bnpo`（默认 `dapo`），GSPO 使用 `importance_sampling_level="sequence"`，以及 `epsilon`/`epsilon_high`、`mask_truncated_completions`、`scale_rewards` ∈ {group, batch, none}。 ([advanced RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation), [GSPO](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/gspo-reinforcement-learning))
- 吞吐量：vLLM 生成速度最高可提升 11 倍，且在 Unsloth 的 RL 运行中，“训练时间 <4%，纯 vLLM 推理占 96%”。文档给出的典型任务 2048 环境 GRPO 示例：学习率 5e-5，温度 1.0，`num_generations=2`，`optim="adamw_8bit"`，为稳定性可将梯度累积步数提高至 4。
- **FP8 RL 和 Unsloth Standby** 是 RL 的显存技巧：
  - `load_in_fp8=True` → FP8 冻结权重 + 共享 FP8 缓冲区作为 vLLM 权重（单一副本），BF16 LoRA 适配器，反向传播时反量化；宣称提速约 1.4 倍，显存减少 60%，上下文长度是其他 FP8 实现的 10 倍；Qwen3-1.7B FP8 GRPO 适配 5 GB 显存；适用于 4090 之后的消费级显卡。
  - `UNSLOTH_VLLM_STANDBY=1` → 在推理和训练之间共享模型权重；KV 缓存与训练激活/优化器/梯度可以复用同一空间；将 `gpu_memory_utilization` 安全提升至 0.95；引用 H100 实验：Qwen3-32B 16-bit LoRA 上下文长度从 3,600 → 6,144（1.7 倍），并且不再出现 OOM 情况，此前在梯度累积 2 的条件下 TRL+LoRA 在 4B 模型上超过约 1–2K 就会崩溃。
  ([FP8 RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning), [Memory efficient RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl))
- 偏好优化：DPO/ORPO/KTO/SimPO/PPO 在 Unsloth 下均可运行。 ([DPO/ORPO/KTO](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto))

### 2.5 显存技巧（SFT 与 RL 通用）Unsloth 的显存技巧叠加如下：

1. `use_gradient_checkpointing="unsloth"`：将激活值卸载到 CPU 内存（2024 年 4 月引入，现已成为行业标准）；新版本使用 CUDA Streams 降低成本，开销 ≤0.1%。
2. **融合并分块的交叉熵**损失：对 `(batch × seq)` 扁平化后的 logits 分块计算，自动根据空闲显存调整块大小；声称减少 60% 的显存并支持 3.2 倍的上下文长度。
3. **Tiled MLP**（源自 Snowflake Arctic 长序列训练）：`unsloth_tiled_mlp=True`；按 `num_shards = ceil(seq_len/hidden_size)` 分片，代价是约 1.3 倍的步进时间；可自动补丁化任何 MLP 模块；单张 H100 支持 gpt-oss-20b QLoRA 500K 上下文（B200 192 GB 上可达 750K）。
4. **MoE Split LoRA**：避免为所有专家实体化 `B·A`（重排矩阵乘法以利用结合律）；相比 Transformers v5 + `torch._grouped_mm` 号称省约 35% 显存、快 2 倍，相比 v4 曲线快 12–30 倍；后端可通过 `UNSLOTH_MOE_BACKEND` 切换（默认 `grouped_mm`，或 `unsloth_triton` / `native_torch`）；路由器层微调因稳定性考虑默认禁用；**MoE 上的 QLoRA 4-bit 当前不推荐使用**（BitsAndBytes 限制）。gpt-oss-20b 在 12.8 GB 显存中微调；Qwen3-30B-A3B 16 位 LoRA 使用 63 GB。
   （[Long context blog](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning), [MoE](https://unsloth.ai/docs/basics/faster-moe))

### 2.6 本地训练器界面（Unsloth Studio）——确切行为

Studio 主页面包含四个卡片**模型 / 数据集 / 参数 / 训练与配置**：

- **模型类型**选择器：文本 / 视觉 / 音频 / 嵌入；**方法**胶囊选择器：QLoRA（显存最低）/ LoRA（中等）/ 完全微调（最高）。可搜索 HF 中心或本地 `~/.unsloth/studio/models` 目录。**GGUF 模型会从训练中排除——仅支持推理。** 模型选择后会自动填充默认超参数；如需访问受限模型，系统会实时校验 HF 令牌。
- **数据集**选项卡：HF 中心或本地（拖放）；支持自动/alpaca/chatml/sharegpt 格式选择；子集/训练划分/评估划分/切片；列映射预览对话框（`instruction`、`input`、`output`、`image` 等）。
- **默认超参数**（表）：上下文长度 2048（512→32768）；学习率 2e-4；最大步数 0（=按轮次计算）；秩 16（滑块 4–128）；α 32（滑块 4–256）；dropout 0.05；LoRA 变体 LoRA / RS-LoRA / LoftQ；七个目标模块全部开启；视觉模型添加四个复选框（视觉层 / 语言层 / 注意力模块 / MLP 模块）。训练超参数：3 轮，批量大小 4，梯度累积 8，权重衰减 0.01，优化器 AdamW 8-bit；调度器线性，预热 5 步，梯度检查点 `unsloth`，随机种子 3407，保存/评估步数 0，**打包 false**，**仅对补全进行训练 false**；WandB/TensorBoard 关闭，日志频率 10。
- **配置管理**：上传 / 保存 / 重置 YAML；自动命名 `{model}_{method}_{dataset}_{timestamp}.yaml`；结构分为 `training:` / `lora:` / `logging:`。
- **训练界面**：开始训练按钮在模型+数据集均存在前禁用；行内验证（没有评估划分时设置评估步数、视觉数据集搭配纯文本模型等）；全页过渡显示终端阶段蓝色（下载）/琥珀色（加载）/蓝色（配置）/绿色（训练），可通过 × 取消；实时视图显示轮次 (1.23)、以步计量的进度条、4 位小数的损失、科学记数法的学习率、梯度范数、模型/方法、已用时间 / 预计剩余时间 / 每秒步数 / 总令牌数；GPU 监控（利用率百分比、温度摄氏度、显存占用/总量 GB、功耗瓦特/上限），每几秒轮询一次；四张图表（损失 + EMA 平滑 + 运行均值，学习率，梯度范数，评估损失可选）均配有齿轮设置（窗口、EMA 0.6、原始/平滑/均值切换、对数刻度、离群值裁剪 p95/p99）；**停止并保存**与**取消（不保存检查点）**对话框。
  （[Studio start](https://unsloth.ai/docs/new/studio/start)）

### 2.7 导出为 GGUF 及其他格式

- Core API：`model.save_pretrained_gguf("dir", tokenizer, quantization_method="q4_k_m" | "q8_0" | "f16" ...)`；`save_pretrained_merged(..., save_method="merged_16bit")`；对 gpt-oss 使用 `save_method="mxfp4"` 原生格式；LoRA 适配器可通过 `save_pretrained` / `push_to_hub` 推送。Qwen3.8 指南还提到了 NVFP4 / FP8 导出选项。([saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf), [Qwen3.8 train](https://unsloth.ai/docs/models/qwen3.8/train))
- Studio 导出界面：选择训练运行 → 检查点 → 导出方法（**合并模型** 16 位 / **仅 LoRA** / **GGUF 供 llama.cpp/Ollama/LM Studio/Unsloth 使用**） → 本地保存或带 HF 写入令牌推送至中心；详细教程位于 `/docs/new/studio/export`。
- 部署叙事：GGUF、vLLM、SGLang、Ollama、LM Studio、llama-server 和智能手机（ExecuTorch）均提供专属指南；Chat 中提供**对比模式**，在原始模型与 LoRA 微调模型并排运行同一提示。

### 2.8 原生 Rust/桌面端 vs. 对接 Python

Unsloth 完全基于 Python（PyTorch + Triton 内核 + Transformers/TRL/PEFT + bitsandbytes + vLLM、llama.cpp 参与 GGUF 转换/量化）。不存在官方的 Rust API。对于 Cerebro：

- **短期内**，对接一个 `uv` 管理的侧车进程，锁定 Python 3.11–3.13（Studio 要求)，预置 unsloth + trl + transformers v5（Qwen3.5 系列要求为“请使用 `transformers v5`”），在 Windows 上还额外需要 CUDA 工具包/CMake/VS Build Tools（Studio 会通过 `winget` 安装这些组件）以及从 HF 缓存的模型权重。
- **中期内**，将非训练大块环节迁回 Rust 侧：数据集预处理（分词器可移植到 Rust 端：直接在 GGUF 元数据中直接使用 `tokenizer.json` 进行聊天模板渲染）、GGUF 量化/转换（Cerebro 已解析 GGUF；量化可执行与 `llama.cpp` 相同类型的 k-量化/i-量化数学运算，或直接调用自带的 `llama-quantize`），以及管理器/监控/UI（NVML、CUDA 可见性设备）。
- **长期**，将热核函数用 CUDA/cudarc 重写：分块融合交叉熵（Unsloth 的 `chunked_hidden_states_selective_log_softmax` 正是可移植的部分）、激活值卸载检查点、分片 MLP、为 MoE 专家路由融合的 8 位 AdamW。这些是专有技术，可使 Cerebro 的引擎与 Unsloth 拉平，而不必依赖 Python。
- 最小可行参数：训练器任务 = 子进程 + 进程组：设置 `CUDA_VISIBLE_DEVICES`（Studio 的实现将其视为前置条件）、优先选择空闲显存最多的 GPU（`free[0] + 0.85×free[others]` 当作可用容量）、流式输出步骤指标（Studio 使用 SSE `/api/train/stream`）。在 Rust 中，使用 `tokio::process` 并配置 `kill_on_drop`，在侧车进程和 Cerebro 之间放置一个 JSONL 协议。

### 2.9 Cerebro 对标/超越方案

- 附带一个首次运行时在应用数据目录下创建的**训练 tab**，安装固定的 Python 环境（`uv venv --python 3.12`、`uv pip install unsloth` 等）并将其视为实现细节；显示版本，并提供“重装/修复”操作。
- 将训练器设计为任务：排队、快照/YAML 配置、恢复、取消、检查点。所有数字都在 UI 中实时显示（见 §2.6 的字段列表）。
- 因为 Cerebro 掌控其引擎，它有 Studio 不具备的机会：(a) 在训练同时训练相同的 GGUF 可在训练结束后立即运行，无需导出（“训练，然后聊天”回环）；(b) 将相同的适配器加载到 Rust 端进行推理（LoRA 热拔插）；(c) 呈现真实数据而非预测：将内存估算器的预测与 NVML 的实测值对比，并按模型微调校准因子（Unsloth 自身也是通过这种方式校准其估算器的）。

---

## 3. 内存使用估算器——精确公式

### 3.1 训练显存模型（Unsloth Studio 后端，首要来源）

`studio/backend/utils/hardware/VRAM_ESTIMATION.md`（来自 `unslothai/unsloth` 仓库）：总计显存 = 权重 + LoRA 适配器 + 优化器 + 梯度 + 激活值 + CUDA 开销。

符号：`H` 隐藏大小，`L` 层数，`V` 词汇表，`K = (H/注意力头数)×KV头数`，`M` 中间/专家中间大小，`E` 专家数（稠密模型为 1），`r` LoRA 秩，`B` 每设备批次大小，`S` 最大序列长度。

**1. 模型权重**
```
QKVO = (H + K + K + H) * H
MLP  = H * M * 3 * E  +  (E * H if E > 1 else 0)
Quantizable     = (QKVO + MLP) * L
Non-quantizable = 2*H*L + V*H + (V*H if not tie_embeddings else 0)

QLoRA 4-bit:  Quantizable * 2 / 3.2 + Non_quantizable * 2   # 3.2 ≈ 16/5，用于 NF4 块状尺度
              （启用 double-quant 时使用 3.6）
LoRA/FFT fp16: (Quantizable + Non_quantizable) * 2
```
嵌入层、LM 头（除非绑定的）和归一化参数保持 fp16（`Non-quantizable`）。带有 `llm_int8_skip_modules` 指向语言层的 4 位配置，会将这些部分按 fp16 计费。该估算器会尊重按层配置字段（`layer_types`、`head_dim`、`num_kv_shared_layers`、`use_double_wide_mlp` 等）——当此类信息存在时，会根据每层形状推导权重。

**2. LoRA 适配器**——对每个选定的模块，A/B 形状分别为：`q: H×r, r×H; k: H×r, r×K; v: H×r, r×K; o: H×r, r×H; gate: H×r, r×M; up: H×r, r×M; down: M×r, r×H`；MLP 模块乘以 `E` 以适配 MoE。
```
LoRA_bytes = sum(A + B per selected module) * L * 2
```
`"all-linear"` = 上表列出的所有文本线性层。视觉塔式模块有意不纳入估算（模型间差异太大）。

**3. 优化器（经校准）**——`adamw_8bit` 4 字节/参数（BNB 更新时上转 fp32）；`adamw_torch` 6；`paged_adamw_32bit` 8；`sgd` 4。可训练参数 = 仅 LoRA，或完整微调时为全部参数。

**4. 梯度**——`Gradient_bytes = trainable_params * 2`。

**5. 激活值**——每层：
```
Per_layer = (S*B*(H+K+K) + S*B*2 + S*B*(M+M)) * 2 * 1.25
```
如果注意力实现既不是 `flash_attention_2` 也不是 `sdpa`/`flex_attention`，则用二次项作为下限：
```
Non_flash = B * num_attention_heads * S^2 * 2 * 12.0 * effective_layers
Activations = max(Per_layer_with_gc, Non_flash)
```
梯度检查点的有效层乘数：无检查点 `L`；HF `True` → 完整微调 2.0 / LoRA 1.0；`"unsloth"` → 1.5 / 1.0。

**6. 下限**——完整微调：`gradient_bytes = max(computed, weights * 0.15)`（自动求导/NCCL/分片开销）。LoRA/QLoRA：
```
raw = trainable_params * 2
floor = min(weights * 0.15, max(activation_bytes, optimizer_bytes))
gradient_bytes = max(raw, floor)
```

**7. CUDA 开销**——固定 **1.4 GB**（CUDA 驱动 + PyTorch 运行时，在 RTX 5070 Ti 上校准）。

**8. 多 GPU**——`usable = free[gpu0] + Σ free[gpui] × 0.85`（每增加一块 GPU 打 15% 的折扣，用于 NCCL/PCIe/碎片化；在 2–8 GPU 配置上校准）。

能触发重新估算的参数：`batch_size`、`max_seq_length`、`lora_r`、`target_modules`、`gradient_checkpointing`、`optim`。
（[VRAM_ESTIMATION.md](https://github.com/unslothai/unsloth/blob/744b59f0/studio/backend/utils/hardware/VRAM_ESTIMATION.md)）

### 3.2 推理/加载估算器（模型中心适配徽章）

`studio/frontend/src/lib/vram.ts` 计算加载 4 位模型的显存需求：
```
estimateLoadingVram(params, method) = params * bytesPerParam + 1.4 GB
   QLoRA: 0.90 字节/参数（从原始 bnb 0.5 校准，嵌入/lm_head 为 fp16 + 块元数据）
   LoRA/FFT: 2.0 字节/参数（理论值）
适配 = 需求/可用 ≤ 0.75；紧张 = ≤1.0；超出 >1.0。
校准示例：Qwen2.5-0.5B 预估值 1.8 对比实际 1.86 GB；Llama-3.2-1B 预估 2.5 实际 2.54；3B 预估 4.3 实际 4.40；Llama-3.1-8B 预估 8.6 实际 8.14（±6%）。
```
（[vram.ts](https://github.com/unslothai/unsloth/blob/9a261aec/studio/frontend/src/lib/vram.ts)）

对于 GGUF 推理（Cerebro 的场景），业界规则是简单但有效的：**将文件大小 + KV 缓存 + 计算缓冲区与 RAM+VRAM 比较**；Unsloth 对每份 GGUF 的指导是“RAM+VRAM ≈ 量化文件大小，否则仍可运行但会因卸载而变慢” ([Kimi K2.6 guide](https://unsloth.ai/docs/models/kimi-k2.6))。`llama-server` 在运行时报告精确的分段情况（模型缓冲区、KV 缓冲区、计算缓冲区，以及专家的 `-ot ".ffn_.*_exps.=CPU"` 卸载），Cerebro 可在加载时模拟。

### 3.3 KV 缓存

适用于 GQA/MHA 模型的标准推理公式：
```
kv_bytes/token = 2 (K 和 V) × n_layers × n_kv_heads × head_dim × bytes_per_element
kv_total = kv_bytes/token × seq_len × batch
```
fp16 元素 = 2 字节；q8_0 KV 缓存大约减半（llama.cpp: `--cache-type-k q8_0 --cache-type-v q8_0`）。臭名昭著的特殊情况：**MLA**（DeepSeek 风格）压缩为 `(kv_lora_rank + qk_rope_head_dim) × layers × 2`；**混合注意/线性**模型（Mamba/SSM 类）仅统计全注意力层 —— SSM 状态有自己固定大小的递归缓存（例如，llama.cpp 日志“llama_memory_recurrent: RS buffer size ... R (f32)/S (f32)”）。已量化的例子：Qwen3.8-Flash-Next 以 262,144 个单元格，12 个 KV 层，q8_0 → llama.cpp 报告 3,264 MiB（1,632 K + 1,632 V）加上一个 1,224 MiB 的第二个缓存（文档中显示的索引器/虚拟 KV）。([llama.cpp multimodal](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md), HF 讨论日志来自 [Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/8)；第三方 KV 公式交叉验证： [llm-vram-calculator](https://github.com/collindjohnson/llm-vram-calculator) — 第三方公式，官方 Unsloth 估算器使用上述 §3.1 中的按层激活值，而非推理 KV 模型。）

### 3.4 MoE：激活参数 vs 总参数

- **权重/显存**：所有专家都必须驻留；使用 §3.1 公式中的总参数，其中 `E` 乘以 MLP。对于量化基准，同样规则：MoE 的量化大小由*总参数*驱动，而速度由*激活参数*驱动。
- **训练**：Unsloth 估算器的激活值公式没有乘以 `E`（只有激活专家产生层激活值），并且路由器层默认被冻结。MoE 的“Split LoRA”将 `r ≪ n` 的典型 LoRA 参数与专家矩阵路由参数 `k(m+n)` 分开处理；文档推导出在专家数量大时将 LoRA 应用于专家矩阵更优 —— 但印刷的阈值不等式在获取时被截断，因此请将确切阈值视为 **UNVERIFIED**，不过“将 LoRA 应用于 MoE 层（`gate_up_proj`、`down_proj`）”，并且“除非你想要，否则绝不要微调路由器”这些指导方向是明确的。
- **推理**：MoE 架构支持将专家卸载到 CPU，同时注意力/KV 保留在 GPU 中（Cerebro 正是这样做的；llama.cpp 的等价方法是 `-ot ".ffn_.*_exps.=CPU"`）。当 RAM+VRAM ≥ 文件大小时，可获得最佳 token 速率。

### 3.5 针对目标硬件的应用示例：i9-12900H，32 GB，RTX 3070 Ti 8 GB

- 在 8B 语言模型上以 r=32、批次 1 的 QLoRA：Unsloth 表格最小显存为 6 GB，但**实测的最大上下文长度在 8 GB 卡上约为 2,972 token**（HF+FA2：显存不足）；12 GB → 21,848 token；16 GB → 40,724。因此，在 8 GB 笔记本显卡上，8B QLoRA 训练功能上可行，但仅适用于短上下文，并且不允许容纳第二块模型的显存余量。
- 在 8B 上使用 bf16 LoRA（不能用 4 位）：约 22 GB → 绝不适用于 8 GB。
- `adamw_8bit` 下 8B 模型 r=32 的 LoRA 参数：约每适配 `2·r·(in+out)` 个参数 × 7 个模块 × 32 层量级 ⇒ 数千万参数 → 优化器约 4 字节/参数仅为数百 MB；权重占主导地位（4 位下约 4.5 GB + fp16 下的嵌入/头）。
- MoE 训练（Qwen3-30B-A3B bf16 LoRA 63 GB；gpt-oss-20b 12.8 GB）—— 在笔记本 8 GB 上不可行；MoE QLoRA 反正也不推荐。
- 推理：UD-Q4_K_XL 级别 ≤8B 的稠密模型可轻松容纳；Gemma 4 12B 4 位“在 8 GB 显存上运行”和 Qwen3.5 9B“在 12 GB RAM/显存上运行”是公开的参考点；MoE 类模型（Qwen3.5-35B-A3B 约 22 GB）需要利用 32 GB RAM 进行专家卸载。 (数据来源：Requirements 和 Pablo Caro 的注释，第三方汇总了 Unsloth 的按模型页面 — **UNVERIFIED** 作为独立来源，但与 Unsloth 自己的表格一致。)

### 3.6 Cerebro 方案

- 将 §3.1 和 §3.2 移植到一个 Rust crate 中（`cerebro-mem`），使用从 `config.json`/GGUF 元数据填充的相同符号类型结构体（`h, l, v, kv_heads, heads, m, e, tie, per-layer overrides`）。纯函数：权重/适配器/优化器/梯度/激活值/开销，外加 `fits/tight/exceeds` 阈值 0.75/1.0。
- 模型中心徽章对每个量化都从同一估算器驱动，使用测量到的 GGUF 字节大小进行推理骨架计算，当是 MoE 且卸载专家时使用 KV 公式。
- 添加一个 Rust 版本的 MoE 多 GPU 折扣（0.85），为未来捆绑/多 GPU 模式做好准备。
- 收集真实数据：在每个训练步骤记录 NVML 实测值，按模型存储校准比率（Cerebro 的估算器误差表，如同 Unsloth 针对其 0.90 字节/参数的校准所做），随时间推移缩小误差。

---

## 4. 按硬件划分的模型推荐

### 4.1 Unsloth 官方数据

微调显存表（绝对最小值）：3B 3.5/8 GB（QLoRA/LoRA），7B 5/19，8B 6/22，9B 6.5/24，11B 7.5/29，14B 8.5/33，27B 22/64，32B 26/76，40B 30/96，70B 41/164，81B 48/192，90B 53/212，405B 237/950 GB。([requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements))

上下文长度（Llama 3.1 8B QLoRA r32，批次 1；以及 70B）：见 §3.5 中的表格。同一页面还推荐“保持批次大小为 1、2 或 3”。 ([benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks))

按系列划分的运行指南（摘自各模型页面）：

- **Qwen3.8**：27B 在 17 GB RAM/显存下运行（4 位）；在 24 GB 显存上 NVFP4 比 BF16 快约 1.5 倍；使用 8 位 KV 缓存校准可实现 2 倍上下文。([Qwen3.8](https://unsloth.ai/docs/models/qwen3.8))
- **Qwen3.5**：小模型 ≤9B 在 12 GB 下运行；35B-A3B 约 22 GB；122B-A10B 需要 256 GB 显存进行 LoRA。
- **Gemma 4**：12B 在 8 GB 上（4 位），26B-A4B 约 18 GB，31B 约 20 GB。([Pablo Caro 的 Unsloth 文档注释](https://pablocaro.es/en/unsloth-docs) — **UNVERIFIED** 摘要，但其内容与 Unsloth 页面一致；编写时请使用各模型页面核实。）
- **gpt-oss**：20b 推理约 14 GB 4 位；GRPO 在 12.8–14 GB 范围；20b QLoRA 训练 14 GB / bf16 LoRA 44 GB。
- **Kimi K2.6**：推荐 UD-Q2_K_XL 约 350 GB；无损 UD-Q8_K_XL 比 Q4 大 10 GB；需要 ">40 tokens/s" 于 B200 上。
- **DeepSeek-V4-Flash**：UD-Q8_K_XL 162 GB 无损（MXFP4 专家重打包）；UD-IQ3_XXS 适配 128 GB RAM 机器。

### 4.2 目标硬件（8 GB 英伟达笔记本显卡 + 32 GB 系统内存）

**训练（QLoRA/LoRA/FFT）：**
- 舒适区：1.5B–4B QLoRA（表格 3B≈3.5 GB），最长上下文可达数 K token；模型上限约 14B QLoRA（8.5 GB），仅在短上下文和严格省显存的设置下可行。
- 不建议：8B QLoRA 超过 ~3K 上下文；任何 bf16 LoRA ≥7B；任何 MoE LoRA/QLoRA；FFT 超过 ~1B。
- 目标技术：`gradient_checkpointing="unsloth"`，批次 1–2，梯度累积弥补有效批次大小，`adamw_8bit`，仅训练补全，尽可能用嵌入层卸载（`offload_embedding`）。
**推理：**
- 舒适区：≤8B 的 UD-Q4_K_XL（约 4–5 GB 文件 + 1–3 GB KV，取决于上下文）；Gemma 4 12B 类 4 位“适配 8 GB”的宣称；14B 4 位在紧凑上下文下；30B 类 MoE（A3B）通过将专家卸载到 32 GB 内存中（Cerebro 的 MoE 卸载已经做到这一点）。

### 4.3 Cerebro 方案

在模型中心中构建“硬件感知模型推荐器”：每个 GGUF 都预先计算（文件大小、KV/token、专家字节数、mmproj 大小），并针对 8 GB 显存 + 32 GB 内存给出一个推荐配置（ngl 分配、专家卸载策略、上下文大小、KV 量化），带有 推荐/紧张/不足 徽章，以及“用于训练”按钮，推荐该系列中适配的 QLoRA 尺寸并带预设参数。这超越了 Unsloth *Desktop*（它按量化显示徽章），因为它还会推荐*训练*目标并配置卸载——这是 Cerebro 的统一优势，因为它掌控自己的运行时。

---

## 5. 首日支持与多模态（音频、视觉、视频、图像）

### 5.1 Unsloth 如何实现快速发布

- **合作 + 预发布访问。**“我们直接与 gpt-oss、Qwen3、Llama 4、Mistral、Gemma 1-3 和 Phi-4 背后的团队合作，修复了严重提升模型准确率的严重 bug。” Qwen3.8：“感谢 Qwen 提供首日访问。” 官网：“发布首日支持 Qwen3.8、NVIDIA、GLM、Gemma 等。” ([docs index](https://unsloth.ai/docs), [Qwen3.8](https://unsloth.ai/docs/models/qwen3.8), [homepage](https://unsloth.ai/))
- **在权重公开之前就着手工作。** Qwen3.8-Flash-Next：团队在*发布前一天*公开了量化/内核工作（“正在努力实现首日支持……架构极其新，可能会有延迟”），最终支持在次日通过更新落地，推动了 llama.cpp PR #27742；用户日志显示 `GGML_ASSERT(inp->self_k_rot == nullptr ...)` 在补丁完成前崩溃。([LinkedIn 帖子](https://www.linkedin.com/posts/unsloth_qwen-announces-qwen38-flash-next-a-new-activity-7497985396354633729-bwxg) — **UNVERIFIED** 社交帖子；[HF 讨论](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/8))
- **跨仓库内核和 bug 修复。** Triton 内核（由手写反向传播支持，“速度快 2 倍，显存减少 70%”）。发布的修复：Llama 4 RoPE 缩放和 QK-Norm epsilon 1e-5（llama.cpp PR #12889、transformers PR #37418）；在 vLLM 中共享 QK-Norm 的错误修复使 MMLU-Pro 从 68.58 提升到 71.53；Qwen3 128K 聊天模板修复；Gemma 1-3 和 Phi-3/4 修复。([dynamic GGUFs page](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs))
- **量化流水线在公开发布前运行**——校准/重要性矩阵工作、逐张量层选择以及 KLD/Divergence 测试在发布前或发布当日执行（见 §7）。
- Reddit 上的一名团队成员表示：“我们通常能做到首日。” 社区指出，延迟原因包括重要性矩阵生成和量化敏感性 A/B 测试。([r/unsloth 讨论](https://www.reddit.com/r/unsloth/comments/1mm3lol/why_is_there_lag_between_an_open_llm_release_and/) — **UNVERIFIED**，用户生成的讨论)

### 5.2 模型族覆盖

声称运行和训练覆盖 500+ 个模型，涵盖文本、视觉、TTS、音频、嵌入、RL。2026 年 8 月 Desktop 发布时的首日列表：Qwen3.8、Muse Glimmer 30B、MiniMax H3、Kimi K3、DeepSeek-V4 Flash、Gemma 4。([substack](https://unslothai.substack.com/p/introducing-unsloth-desktop) — **UNVERIFIED** 作为营销通讯，但与文档中的模型页面一致。)

### 5.3 多模态：GGUF / llama.cpp / mtmd 路径

- llama.cpp 通过 **libmtmd** 支持多模态输入。“目前，我们支持**图像**、**音频**和**视频**输入。” 两种加载模式：`-hf repo:QUANT`（自动匹配 mmproj）或 `-m model.gguf --mmproj mmproj-F16.gguf`；`--no-mmproj` / `--no-mmproj-offload` 开关；供 `/chat/completions` 使用的 OpenAI 兼容多模态 API；`llama-mtmd-cli` 用于测试。([llama.cpp docs/multimodal.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md))
- 就绪多模态 GGUF（ggml-org 集合，Unsloth 发布对应的配对文件）：视觉——Qwen3-VL 系列等；音频——**Ultravox 0.5**（Llama-3.2-1B/3.1-8B）、**Voxtral-Mini-3B**、**Qwen3-ASR-0.6B/1.7B**；混合——Qwen2.5-Omni 3B/7B、Qwen3-Omni-30B-A3B、Gemma 4 E2B/E4B。
- Unsloth GGUF 仓库附带 `mmproj-F16.gguf` 配对文件（例如 [Kimi K2.6](https://unsloth.ai/docs/models/kimi-k2.6) 的视觉，以 `--mmproj unsloth/Kimi-K2.6-GGUF/mmproj-F16.gguf` 方式加载）。
- Gemma 4 12B Unified 使用单一 `gemma4uv` 投影器将视觉+音频统一在一个 117 MB 的 mmproj 中；其发布故事堪称完美案例：首日 mmproj 因 `clip.vision.attention.head_count = 0` 出现 SIGFPE 崩溃（llama.cpp 问题 #24085），修复后于 6 月 4 日通过转换器 PR #24118 解决，ggml-org 和 unsloth 在数小时内重新上传，解码速度从 HF 约 19 t/s 提升到 llama.cpp 64–131 t/s，显存从 24 GB 降至 Q8_0 的 14 GB。([note.com 文章](https://note.com/unco3/n/n871e994d27b2?hl=en) — 从一手来源时间戳重建的详细第三方记录；**UNVERIFIED** 作为单一来源，但内部一致，并与有问题编号和 PR 编号一致。)
- **音频前端**：16 kHz 单声道 WAV；对于 llama.cpp 音频路径，30 秒片段是安全范围；服务端可以接受 OpenAI 风格负载中的 `input_audio`（base64 WAV）。([note.com](https://note.com/unco3/n/n871e994d27b2?hl=en), [LocalAI 文档](https://localai.io/docs/features/audio-to-text/index.print.html))
- **图像/视频生成**：Unsloth Desktop 运行并训练扩散模型——MiniMax-H3、Qwen-Image-2.1、Z-Image、FLUX 1/2 klein、SDXL、LTX——支持 LoRA/ControlNet 和参考图、Gallery、Recipe 重放、低显存选项，以及 GGUF/FP8 路径；也有 ComfyUI 和 stable-diffusion.cpp 教程。([diffusion-image](https://unsloth.ai/docs/basics/diffusion-image), [ComfyUI blog](https://unsloth.ai/docs/blog/comfyui))
- **TTS/STT**：通过 Unsloth 微调任意 `transformers` 兼容的 TTS 模型（Orpheus-3B、Sesame-CSM-1B、Spark-TTS、Llasa、Oute-TTS）以及 STT（Whisper 大模型 v3）；Orpheus 模型可导出至 llama.cpp，兼容大量推理引擎；“不通用的模型仅能保存 LoRA 适配器 safetensors”；TTS/STT 微调推荐使用 16 位 LoRA，而非 QLoRA（`load_in_4bit=False`）；Orpheus 的 token 直接生成音频 token，无需声码器；`max_new_tokens=125 ≈ 10 s` 音频。([TTS guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning))

### 5.4 Cerebro 对标/超越方案

- **架构注册表 + 转换 CI**：一个数据驱动注册表（注意力类型、RoPE 变体、QK-norm epsilon 来源、MoE 路由、MTP 头、mmproj 要求），用于驱动 Cerebro 引擎加载器；CI 任务在每次发布新 Hugging Face 架构时触发，若上游元数据匹配则运行转换-加载-冒烟测试；发布“支持/不支持及原因”页面。这正是 Unsloth 的手动过程的自动化版本，也是 Cerebro 在不具备合作关系时能够**超越**的地方（速度上）。
- **原生 Rust 版的 mtmd**：Cerebro 已经在解析 GGUF，因此添加 mmproj 加载器（图像/音频编码器与语言模型权重分离），以配对文件形式支持音频/图像输入，匹配 llama.cpp 的 `clip.*` 元数据；在聊天中暴露多模态加载状态（“模式：文本、视觉、音频”），与 llama.cpp 的 `modalities` 行一致。
- **音频流水线**：语音转文字路线：捆绑一个专用 ASR（Whisper 或通过 mtmd 的 Qwen3-ASR GGUF），然后送入任意聊天模型；TTS 路线：Orpheus GGUF 通过 Cerebro 引擎处理——这是 Cerebro“手机访问”功能的绝佳搭配。
- **训练侧**：为文本（视觉 LoRA 通过 Python 侧车）和嵌入模型提供视觉微调 UI，复用 `UnslothVisionDataCollator` 的语义（completion-only 损失、调整大小策略）；添加图像 LoRA 训练标签页，与图像生成并列。
- **超越之处**：统一的多模态模型中心——同一排序器为同一个模型推荐文本 GGUF、mmproj 对、LORA 适配器和训练配置，并在文件提取时自动检测缺失的 mmproj（Unsloth Desktop 的 `TIGHT`/`OOM` 标签展示了基线）。

---

## 6. 权限与设置模型

### 6.1 认证与账户

- Studio 首次运行管理员密码：随机生成、加盐 **PBKDF2-HMAC-SHA256**，首次登录必须修改。通过 JWT 访问/刷新进行令牌认证（除 `/api/auth/*` 和 `/api/health` 外，所有 API 端点均需有效的 `Authorization: Bearer` 头）。
- API 密钥：`sk-unsloth-…` 前缀，**以哈希形式存储（仅显示一次）**，可设置过期时间，可撤销（撤销后返回 401）。密钥位于 设置 → API。
- 登录失败次数限制（示例显示冷却倒计时）。
- 多账户：受管理账户只能访问自己的文件夹，无法接触安装所有者的 HF 令牌，模型使用需要所有者授权，且禁止运行仓库代码。
  ([Security in Unsloth](https://unsloth.ai/docs/new/studio/security), [Studio start](https://unsloth.ai/docs/new/studio/start))

### 6.2 沙盒（两级 + 完全访问）

- **低级（软件）沙盒**：正则/字符串/AST 检查，每次调用约 1–2 毫秒。阻止：删除/磁盘命令（`rm, dd, mkfs, fdisk, mount, umount`），权限提升（`chmod, chown, sudo, su, doas, pkexec, passwd`），网络工具（`curl, wget, nc, socat, ssh, scp, sftp, rsync`），进程/电源操作（`kill, killall, pkill, shutdown, reboot`），隐藏执行（`eval, source`）以及 Windows 特有命令（`rmdir, takeown, icacls, runas, powershell/pwsh`）。在 Python 代码中：阻止 `os.system`、通过 `su` 的非字面 `subprocess` 命令、`requests/urllib`/原始套接字，读取 `/etc/passwd` 类文件，篡改信号/超时。所有调用均会：从环境变量中剥离密钥，拒绝访问 Studio 自身的凭据文件，限制进程数、文件大小（100 MB）、内存（8 GB）、CPU 时间（600 秒）。
- **高级（操作系统级）沙盒**：Windows 上的 **MXC**（Windows 11 24H2+ 预装；与 Windows 团队合作），**bubblewrap**（Linux；选中“High”时安装），**Seatbelt**（macOS）。延迟表——低级/高级：Windows 8/164 毫秒，Linux 3/50 毫秒，macOS 10/110 毫秒。
- **完全访问** 按钮禁用这两种模式。权限按钮显示当前沙盒状态；沙盒设置页面解释安装步骤。
  ([Sandboxing](https://unsloth.ai/docs/new/studio/sandboxing-in-unsloth))

### 6.3 工具审批与远程策略

- 三种审批模式：`ask`、`auto`、`full`。`auto` 标记网络/文件系统导入以进行审批，文件路径需要审批，危险的 shell 命令被直接阻止。工具调用显示 **允许 / 始终允许 / 拒绝**。([Security](https://unsloth.ai/docs/new/studio/security))
- 服务端工具策略按绑定地址确定：localhost → 工具默认开启；`0.0.0.0`/非环回 → 工具默认**关闭**（“网络暴露的服务器上泄漏 API 密钥意味着在主机上任意代码执行”）；`--enable-tools`/`--disable-tools` 强制切换，在 `0.0.0.0` 上 `--enable-tools` 会提示 y/N（`--yes` 跳过）；解析后的策略是进程级硬覆盖，请求无法绕过。([API page](https://unsloth.ai/docs/basics/api))
- Remote 使用 `--secure`（Cloudflare 隧道，HTTPS）并推荐 `--disable-tools`；服务器端工具以用户身份运行，持有 API 密钥和网络访问权限的人可以执行代码。Cloudflare 快速隧道不保留 SSE 流，因此需设置 `stream:false`。([remote access guide](https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth))

### 6.4 模型资源安全

- 检查 Hugging Face 每个仓库的安全扫描状态；恶意文件会被阻止加载并附说明（示例 `mcpotato/42-eicar-street`）。对于需要自定义代码的仓库（`trust_remote_code`）进行审批时，反向外壳、云元数据访问和凭据窃取被直接拦截（示例：上游 `deepseek-ai/deepseek-ocr` 和 `moonshotai/Kimi-VL-A3B-Instruct` 需要对话审批；Unsloth 提供清理过的变体 `unsloth/DeepSeek-OCR(-2)`）。
- 要求 PyTorch ≥2.6，因此 `.bin` 文件以 `weights_only=True` 加载。
- 供应链卫生：npm `min-release-age=7`、安装脚本的 `allowScripts` 允许列表、`npm ci` 结合锁定文件、对 npm/PyPI 包的脚本扫描、`pip-audit`/`npm audit`/`cargo audit`、CodeQL/Semgrep、Dependabot 冷却期、固定版本的 GitHub Actions、对预构建 llama.cpp 二进制进行 SHA-256 验证、发布版 VirusTotal 扫描。

### 6.5 Cerebro 对标/超越方案

- Cerebro 已具备 MCP 的按工具允许/询问/拒绝功能；需添加以下层级：(1) 沙盒等级（低/高/完全访问），映射到 Windows 上的作业对象/AppContainer *或* MXC（若可用，通过文档化的 Windows API 使用，镜像 Unsloth 与 MXC 的合作），(2) 与上述两种模式匹配的命令策略列表，(3) 与绑定地址绑定的默认策略（手机访问 = 非环回 ⇒ 工具默认关闭，需明确提示），(4) 每次调用的资源上限（文件 100 MB / 内存 8 GB / CPU 600 秒），(5) 密钥剥离和凭据文件屏蔽。
- API 密钥：采用相同模式（哈希、一次显示、过期时间、撤销）。添加**按密钥工具范围**和请求审计，使其 API 监视器与 Studio 的监视器一样丰富（提示、令牌、首令牌时间、吞吐量、错误）。
- 模型文件：对 HF 扫描状态和自定义代码许可进行镜像检查；在更新中锁定 llama.cpp 或引擎构建的二进制哈希；发布时进行校验和验证（桌面应用 VirusTotal 等效）——对“手机访问”叙事尤为重要，因为手机用户信任的是桌面二进制。
- 超越之处：权限是 **Cerebro 定义并由 Rust 引擎强制执行**，而非基于 Python 沙盒的尽力而为；操作系统级隔离可以成为默认行为，并在聊天头部显示沙盒徽章（与 Studio 显示位置相同，但引擎级生效）。

---

## 7. 动态量化（UD-*）及其选型方式

### 7.1 演进历程

- **动态 4 位（2024 年 12 月）**：对 BnB 4 位进行“动态选择不量化的参数”；相比普通 BnB，显存增加不到 10%，精度显著提升；对 Llama 3.2 Vision 的例子显示出不同张量的敏感性（“交叉注意力的输出投影在所有层中除第一层外均不应量化”）；甚至有“4 位量化破坏了 Qwen2-VL 2B”的轶事。([dynamic-4bit](https://unsloth.ai/blog/dynamic-4bit))
- **动态 2.0（2025 年 4 月）**：逐张量动态量化类型选择（“每个可能的层……各模型的组合不同”）；>150 万 token 的精选校准数据集；适用于所有模型（不仅是 MoE）；加入 Q4_NL/Q5.1/Q5.0/Q4.1/Q4.0 格式以兼容 Apple Silicon/ARM；用于评估的 KLD 表（例如 Llama 家族 Q4_K_XL 15.64 GB 对应 0.0237 KLD，而基线 15.41 GB 对应 0.0249）；Gemma 3 27B 的 Q4_K_XL 在大小上胜过 Google 的 QAT（71.47 对 70.64 MMLU 5-shot，15.64 对 17.2 GB）。([dynamic-3.0 page](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs), [dynamic-v2 blog](https://unsloth.ai/blog/dynamic-v2))
- **动态 3.0（2026 年 8–9 月）**：更好的 imatrix（“校准数据集已精炼用于代理编码、聊天和多语言”），改进的层选择和更多量化技术；在相同大小下 Top-1 准确率提升 >10%；新的 **Divergence-300 @32** 指标（从 Terminal-Bench 2.1 + DeepSWE + Harbor + MathArena 2025-26 + 非拉丁/长文档中抽取的 300 个样本，使用贪心 32-token 解码与 BF16 对比）；低于 `UD-Q2_K_XL` 的量化版本移除 MTP 模块以节省约 500 MB；对 1 位量化的明确警告（“代理场景不应使用 1 位”——丢工具调用，需要 `presence_penalty=1.5`，开启思考模式）。无 QAT/QAD——“全部通过训练后量化完成”；imatrix 文件对外发布。([dynamic-3.0 page](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs))
- 命名阶梯：`UD-IQ1_S / UD-IQ1_M / UD-IQ2_XXS / UD-IQ2_M / UD-Q2_K_XL / UD-Q3_K_XL / UD-Q4_K_XL / UD-Q8_K_XL`（在有 MTP 时可能还有单独的 MTP 模块）。"UD" = Unsloth Dynamic。

### 7.2 选择方法（公开信息）

- 逐张量敏感度：将高重要性张量向上量化（例如，Q4_K_XL 在安全时使用 Q5_K，而 Q4_K_M 使用 Q6_K）。([HF 讨论](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/discussions/6) —— 维护者回答，基本可视为一手来源)
- 带有混合量化的 imatrix 作为校准；“imatrix 确实有助于降低 KLD 和困惑度，代价是速度降低 5–10%”；Qwen3.5 的注意事项：不要量化 `ssm_out`（Mamba 层）或 `ffn_down_exps`；`attn_gate/attn_q/ssm_beta/ssm_alpha` 上 MXFP4 不如 Q4_K（MXFP4 = 每权重 4.25 位，Q4_K = 4.5 位）；“为所有 GGUF 量化版本中退役 MXFP4……除纯 MXFP4_MOE 外”。([Qwen3.5 GGUF benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks))
- 无损重打包路径：DeepSeek-V4-Flash 专家原生为 MXFP4，因此 Unsloth 逐位重打包（“全部 1,328 个张量逐位相同”，KLD~0，100% 前 token 一致）；Kimi K2.6 在 `UD-Q8_K_XL` 下同理。([DeepSeek-V4](https://unsloth.ai/docs/models/deepseek-v4), [Kimi K2.6](https://unsloth.ai/docs/models/kimi-k2.6))
- 评分体系：(MMLU−25)/GB 效率指标，KLD 均值 + 最大 KLD，Divergence-300@32，Aider Polyglot。([dynamic-3.0](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs), [Aider 页面](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot))
- 文档公布的部分量化规模：Qwen3.8-27B UD-Q2_K_XL 9.83 GB，UD-IQ1_S 6.2 GB（89% 更小，约 72% Top-1），UD-Q4_K_XL 约 15.6 GB 级别；DeepSeek-V4-Flash Q8 162 GB / IQ3 103 GB / 409 等。（数字来自引用的各模型页面。）

### 7.3 Cerebro 对标/超越方案

- 在 Rust 中实现自己的量化阶梯（`CD-*` 或保留 `UD` 兼容命名），使用 k-quants/quants 类型，并实现逐张量的 imatrix 选择。对于 imatrix 计算，可以通过子进程调用 bundled `llama-imatrix`；逐张量比特分配逻辑则可以用 Rust 编写（高效地从 imatrix 中将敏感度与张量形状关联）。
- 重放 Unsloth 的公开校准配方（他们公开了 imatrix 文件，并鼓励社区测试——请引用他们的 imatrix ），然后发布 Cerebro 的 KLD/Divergence-300 数据，让用户将徽章视为真实能力而非小众概念。
- 模型中心徽章：展示 KLD 均值、Top-1 和推荐量化版本（与 PPL 一起显示，或替代 PPL——引用“准确率并非全部”的论点，即 PPL 会相互抵消）。
- MTP 处理：为支持 MTP 的模型分离 MTP 模块，并让引擎在选择较小量化版本时选择退出（与 Unsloth“低于 Q2_K_XL 的量化移除 MTP”一致）。
- 无损路径：当基础模型原生以 MXFP4/FP8 存储时，实现逐位重打包路径（使用 Cerebro 的 GGUF 写入器将 MXFP4 专家重打包为 GGUF MXFP4），以便“无损” GGUF 确实是逐位相同；测试按张量哈希进行（Unsloth 的 1,328 张量检查是一个很好的测试模板）。
- 超越点：量化在 **同一引擎** 中消费，该引擎运行推理（无转换器/运行时偏差），并且 Cerebro 可以在量化时以及运行时测量 KLD（对加载的 GGUF 进行实时逐 token 散度探测），这是 Unsloth 离线报告方式无法做到的。

---
## 8. 整合：Unsloth 的优势所在，以及 Cerebro 应如何应对

| 维度 | Unsloth（2026 年 10 月） | Cerebro 应对策略 |
|---|---|---|
| 训练 | Python/Triton，极为成熟；2 倍速，节省 70% 显存；QLoRA→FFT、GRPO、FP8、MoE 分割 LoRA | 分阶段实施：`uv` 侧车 SFT→RL；中期通过分块融合交叉熵、卸载检查点、分块 MLP、8 位 AdamW 内核将内核迁移到原生 CUDA |
| 内存估算 | 已发布精确公式（VRAM_ESTIMATION.md、vram.ts）；校准基准 | 移植到 Rust 的 `cerebro-mem` crate；根据 NVML 实测值自动校准；驱动模型库徽章、训练器、聊天 |
| 模型推荐 | 各模型页面 + 模型库标签（推荐/紧张/显存不足） | 为每种量化计算 RAM+VRAM 适配并加上训练配置建议；为 8GB/32GB 目标硬件提供“一键配置” |
| 首日支持 | 合作关系、预发布量化、内核、上游修复；媒体日期确认当日上架 | 架构注册表 + 转换 CI + 观察上游 PR；公开支持矩阵；自动回退到 `trust_remote_code` 并附带扫描器 |
| 多模态 | mtmd 视觉/音频/视频 GGUF、mmproj 配套文件、统一的 Gemma 4 投影器、TTS/STT 微调、扩散图像/视频 | 在 Rust 中实现 mtmd 兼容的 mmproj 加载器；在模型库中实现模型+mmproj 配套文件；TTS/STT 模式；扩散标签页 |
| 权限 | 低/高沙箱（MXC 164 毫秒）、ask/auto/full 审批、基于绑定地址的工具策略、哈希 API 密钥、多层供应链控制 | 将已有的按工具允许/询问/拒绝扩展为基于 OS 级别的沙箱层级；绑定地址默认策略；为 API 密钥和审计提供同等控制 |
| 动态量化 | UD-1 到 UD-3 阶梯，imatrix 驱动的逐张量选择，公开 KLD/Divergence 数据，无损 MXFP4 重打包 | 实现相同的阶梯和指标；在推理引擎内加入实时散度探测 |

---

## 9. 排名前 25 位的构建清单

按实施顺序/影响力排序；括号内为依赖/备注。

1. **`cerebro-mem` 估算核心（Rust）** — 根据 Unsloth 的 VRAM_ESTIMATION.md 和 vram.ts 移植权重/KV/优化器/梯度/激活/MoE/CUDA 开销公式；提供 `fits/tight/exceeds` 阈值（0.75/1.0）。[基础功能]
2. **训练页 v1：SFT（QLoRA/LoRA/FFT）** 通过托管的 `uv` Python 侧车进程（固定 unsloth/trl/transformers v5，Python 3.11–3.13），带作业运行器、日志流、取消和检查点。[依赖 1]
3. **实时训练遥测 UI** — 损失/学习率/梯度范数/评估损失图表及 EMA/对数/异常值控制，GPU 利用率/温度/显存/功耗监控，阻止状态覆盖层，停止并保存 vs 停止。[依赖 2]
4. **GGUF 导出流水线** — LoRA 合并 → 转换为 GGUF → 量化（q4_k_m/q8_0/f16 作为基线），多模态模型的 mmproj 配套文件，保存到本地或推送到 HF；然后“在 Cerebro 中运行”。[依赖 2]
5. **数据集导入器和列映射器** — HF/本地（PDF/DOCX/CSV/JSONL/JSON/Parquet），自动/alpaca/chatml/sharegpt 检测，划分/切片，预览对话框。 [依赖 2]
6. **聊天模板注册表 + 模板修复器** — 各系列模板、ShareGPT 标准化、train-on-completions 开关、新增标记定义；像 Unsloth 一样发布修复后的模板。
7. **OOM 守护与自动调优** — 根据 `cerebro-mem` 自动选择批次/梯度累积/上下文；在引擎中提供激活卸载检查点、分块交叉熵、分块 MLP 内核等手段以支持长上下文。
8. **各模型预设** — 精选的秩/α/学习率/优化器/上下文默认值，匹配 Unsloth Studio 默认值并加以改进（每个模型的支持上下文、推荐采样参数如温度/top-p/top-k/重复惩罚）。
9. **Cerebro 动态量化阶梯（`CD-*`）** — 使用 imatrix 的 k-quant/i-quant 逐张量分配，MTP 处理，MXFP4/FP8 的无损重打包路径。 [依赖 4]
10. **量化指标测试框架** — KLD（均值/最大值）、Divergence-300@32 风格指标、Top-1、Aider 风格任务；在模型库中发布各量化的徽章。 [依赖 9]
11. **GRPO/RL 工作区** — 奖励函数编辑器（Rust 或 JS），沙箱化奖励执行（无导入，限制全局变量），多生成批量，支持 DAPO/Dr.GRPO/GSPO 选项。 [依赖 2]
12. **RL 内存共享（Standby 等价物）** — 在推理/生成和训练之间共享权重/KV/激活缓冲区，在引擎中实现；加上量化 KV（q8_0）和 FP8 KV 校准。 [依赖 11]
13. **首日架构流水线** — 注册表 + 自动转换 CI + HF 发布监视器 + 上游 llama.cpp PR 跟踪。 [依赖 4]
14. **多模态引擎：mmproj 加载器（mtmd 兼容）** — 图像/音频/视频输入，在对话中显示多模态状态，自动配套模型和投影器。 [依赖 4]
15. **TTS/STT 训练 + 运行** — Orpheus/CSM/Spark 类 TTS 和 Whisper/Qwen3-ASR/Voxtral 路径；TTS 默认使用 16 位 LoRA；GGUF 导出用于 TTS 模型。 [依赖 2, 14]
16. **权限：沙箱层级** — 低（命令/路径/AST 策略及资源限制）和高（Windows 上作业对象/AppContainer 或 MXC），绑定地址默认策略，状态徽章。 [与 2, 14 并行]
17. **API 密钥与认证加固** — 哈希密钥（显示一次/撤销/过期），按密钥限定工具范围，JWT，失败登录限制，所有密钥操作审计。 [依赖 16]
18. **本地 API 双协议服务** — OpenAI `/v1/chat/completions` + `/v1/responses` 和 Anthropic `/v1/messages` 共用同一端口，可与手机访问功能共用；API 监控 UI（提示/令牌/TTFT/吞吐量）。
19. **数据配方工作室** — 图节点合成数据生成器（Seed/LLM/表达式/验证器/采样器/工具配置），验证–预览–运行；可复用本地模型或外部提供者。 [依赖 2]
20. **模型库集成** — 基于 `cerebro-mem` 的适配徽章，大小/质量前沿视图，KLD 标签，mmproj 配对，训练可行性徽章。 [依赖 1, 10]
21. **MoE 训练器支持** — 等价于 Split-LoRA 的专家适配器路径，路由器冻结，专家卸载感知估算；阻止不支持的组合。 [依赖 2, 7]
22. **扩散图像/视频标签页** — 通过 stable-diffusion.cpp 风格的 Rust/C++ 路径或侧车进程运行 FLUX/Qwen-Image/Z-Image/SDXL，并提供图像 LoRA 训练选项。 [独立]
23. **投机解码 + MTP** — llama.cpp 风格的草稿模型和 MTP 模块处理（也用于量化阶梯决策）。 [依赖 4, 14]
24. **安全扫描器与依赖策略** — 检查 HF 每个仓库的扫描状态，对 `trust_remote_code` 进行审批对话，仅加载 `weights_only` 模型，依赖冷却/验证，发布校验和。 [与 16 并行]
25. **零遥测、离线模式、远程访问加固** — 默认无遥测，完全离线运行，`--secure` 等价方案（隧道），默认关闭工具，云模型可选退出。 [横向]

---
## 附录 A：主要来源（出版物中引用的所有内容）

**Unsloth 官方文档与博客**
- Docs 索引 / llms.txt: https://unsloth.ai/docs ，https://unsloth.ai/docs/llms.txt
- 获取并运行 Studio: https://unsloth.ai/docs/new/studio/start
- Studio 安装: https://unsloth.ai/docs/new/studio/install
- 模型导出: https://unsloth.ai/docs/new/studio/export
- 数据配方: https://unsloth.ai/docs/new/studio/data-recipe
- 安全性: https://unsloth.ai/docs/new/studio/security
- 沙盒: https://unsloth.ai/docs/new/studio/sandboxing-in-unsloth
- 桌面应用: https://unsloth.ai/docs/desktop ；主页: https://unsloth.ai/
- 环境要求（显存表）: https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- 微调指南 / 超参数: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide ，…/lora-hyperparameters-guide
- 数据集: …/datasets-guide ；模型选择: …/what-model-should-i-use
- 聊天模板: https://unsloth.ai/docs/basics/chat-templates
- 视觉微调: https://unsloth.ai/docs/basics/vision-fine-tuning
- TTS/STT: https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning
- 扩散图像: https://unsloth.ai/docs/basics/diffusion-image ；ComfyUI: https://unsloth.ai/docs/blog/comfyui
- MCP: https://unsloth.ai/docs/basics/mcp
- API: https://unsloth.ai/docs/basics/api
- 基准测试: https://unsloth.ai/docs/basics/unsloth-benchmarks
- 动态 GGUF（v2/v3 页面）: https://unsloth.ai/docs/basics/dynamic-3.0-ggufs ；Aider: …/unsloth-dynamic-ggufs-on-aider-polyglot
- 动态 4 位博客: https://unsloth.ai/blog/dynamic-4bit ；Dynamic v2 博客: https://unsloth.ai/blog/dynamic-v2
- 500K 上下文博客: https://unsloth.ai/docs/blog/500k-context-length-fine-tuning
- RL 指南: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide （+ 高级、GSPO、内存高效 RL、FP8 RL、DPO/ORPO/KTO 子页面）
- MoE 训练: https://unsloth.ai/docs/basics/faster-moe
- gpt-oss 微调: https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss
- Qwen3.5 微调: https://unsloth.ai/docs/models/qwen3.5/fine-tune ；Qwen3.8 训练: https://unsloth.ai/docs/models/qwen3.8/train ；Qwen3.8 运行: https://unsloth.ai/docs/models/qwen3.8
- Kimi K2.6: https://unsloth.ai/docs/models/kimi-k2.6 ；DeepSeek-V4: https://unsloth.ai/docs/models/deepseek-v4
- Qwen3.5 GGUF 基准测试: https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks

**仓库（主要来源）**
- Unsloth: https://github.com/unslothai/unsloth
- 显存估算: https://github.com/unslothai/unsloth/blob/744b59f0/studio/backend/utils/hardware/VRAM_ESTIMATION.md
- 前端 vram.ts: https://github.com/unslothai/unsloth/blob/9a261aec/studio/frontend/src/lib/vram.ts
- 聊天模板源: https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py
- llama.cpp 多模态: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md ；音频讨论: https://github.com/ggml-org/llama.cpp/discussions/13759
- llama.cpp PR: #12889（Llama 4），#24118（Gemma 4 统一转换），#27742（Qwen3.8-Flash-Next，进行中）
- transformers PR #37418（Llama 4 QK-Norm 修复）；vLLM PR #16311（QK-Norm 头部共享）

**其他主要来源**
- Q
