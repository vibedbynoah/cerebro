# Self-verification and reasoning effort — fresh research (late Sept 2026)

Focus: what changed since 2026-09-01, plus the 2025–2026 baseline that matters for a
Rust harness spawning llama.cpp on the reference laptop (i9-12900H, RTX 3070 Ti 8 GB,
32 GB RAM, batch 1, fully offline). All numbers carry their own conditions; datacenter or
large-batch figures are flagged as non-transferable.

## 1. Reasoning effort levels in open models

**gpt-oss (OpenAI, Apache-2.0, Aug 2025): low / medium / high.** Trained variable-effort
reasoning; the level is set by a `Reasoning: low|medium|high` line in the Harmony system
message, default medium (https://openai.com/index/introducing-gpt-oss/,
https://arxiv.org/html/2508.10925v1, https://github.com/openai/gpt-oss).
Model-card test-time scaling is smooth but log-linear: gpt-oss-120b AIME-2024 no-tools
goes 56.3 → 80.4 → 95.8 across low/med/high; 20b goes 42.1 → 80.0 → 92.1; SWE-Bench
Verified 120b goes 47.9 → 52.6 → 62.4 (https://openai.com/index/gpt-oss-model-card/).
Cost implication at batch 1: longer CoT = linearly more decode tokens at ~same tok/s, so
high effort costs roughly low's tokens ×3–5 on hard tasks. The Harmony format is mandatory;
without it the model "will not work correctly"
(https://github.com/openai/openai-cookbook/blob/main/articles/openai-harmony.md,
https://github.com/openai/gpt-oss/blob/main/README.md). The reference `gpt_oss.chat`
CLI exposes `-r/--reasoning-effort` defaulting to low (https://github.com/openai/gpt-oss).

**Qwen3 hybrid thinking (Apr 2025): hard switch + soft switch.** `enable_thinking=True/False`
in `apply_chat_template` selects thinking vs non-thinking mode; with thinking enabled,
`/think` and `/no_think` suffixes flip per-turn, last instruction wins
(https://qwenlm.github.io/blog/qwen3/, https://huggingface.co/Qwen/Qwen3-8B).
Local note: llama.cpp did not originally expose the hard switch; the workaround was a custom
`--chat-template-file` forcing `enable_thinking=False`
(https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html). Qwen3 support in
llama.cpp dates to build b5092
(https://qwen.readthedocs.io/en/stable/run_locally/llama.cpp.html).

**Qwen3.8-Max (Aug 2026): tiered `reasoning_effort`.** API levels `low` (fast), `medium`
(balanced), `xhigh` (default, deep); `reasoning_effort` and `thinking_budget` cannot be set
together; `preserve_thinking` on by default (https://qwen.ai/blog?id=qwen3.8,
https://docs.qwencloud.com/developer-guides/text-generation/thinking). The qwen-code CLI
grew a 5-tier `/effort` ladder (low/medium/high/xhigh/max) with per-provider
translate-and-clamp, e.g. OpenAI `max→xhigh`, DeepSeek `low/medium→high, xhigh→max`
(https://qwenlm.github.io/qwen-code-docs/en/design/2026-06-30-unified-reasoning-effort-cli/,
https://github.com/QwenLM/qwen-code/pull/8472). Only `qwen3.8-max*` reads the tiered field;
older Qwen hybrids read only the `enable_thinking` boolean
(https://github.com/QwenLM/qwen-code/pull/8488).

**DeepSeek R1: no clean effort knob.** The R1 API exposes reasoning as `reasoning_content`
but no dedicated budget parameter; `max_tokens` caps reasoning+answer jointly, so a long
trace can starve the visible answer (observed failure mode, mid-2026 API guide:
https://pastagi.com/guides/control-llm-reasoning-effort-api-guide/). R1 training used
16 samples/query, up to 32K tokens pre-checkpoint and 65K after
(https://arxiv.org/pdf/2501.12948v2); evals recommend temperature 0.6, top-p 0.95
(https://github.com/deepseek-ai/DeepSeek-R1/). A 2025 feature request proposes s1-style
budget forcing (`max_thinking_tokens`, append-"Wait") for R1/GRPO but it is unverified
discussion, not shipped code (https://github.com/deepseek-ai/DeepSeek-R1/issues/252).

## 2. What llama.cpp actually exposes ( Sept 2026 state)

- `--reasoning on|off|auto` sets `enable_thinking` via template kwargs; `--reasoning-budget N`
enforces a real token budget via a sampler grammar with delayed launch, plus
`--reasoning-budget-message` (e.g. "...budget exceeded, need to answer") to land the
forced stop coherently (https://github.com/ggml-org/llama.cpp/pull/20297,
https://aihaven.com/news/llama-cpp-reasoning-budget-update/).
- Per-request override: `thinking_budget_tokens` in the OpenAI-compatible request body
(applies only if no CLI budget set), and `chat_template_kwargs: {enable_thinking: false}`
for Qwen-family per-request control
(https://github.com/ggml-org/llama.cpp/discussions/21445,
https://github.com/ggml-org/llama.cpp/pull/13196).
- gpt-oss support: native MXFP4 + Harmony parsing landed Aug 2025
(https://github.com/ggml-org/llama.cpp/pull/15091,
https://github.com/ggml-org/llama.cpp/pull/15181); effort via
`--chat-template-kwargs '{"reasoning_effort":"low"}'` per request
(https://huggingface.co/openai/gpt-oss-20b/discussions/47).
- Sharp edges (all primary-source issues): `--reasoning-budget 0` needed `--jinja` to take
effect and behaved differently in CLI vs server
(https://github.com/ggml-org/llama.cpp/pull/13771); `--reasoning-budget 0` did nothing for
gpt-oss early on (https://github.com/ggml-org/llama.cpp/issues/15266); as of Mar 2026
`--reasoning off` still mapped to `Reasoning: medium` instead of low for gpt-oss, and
`--reasoning-budget 0` likewise (https://github.com/ggml-org/llama.cpp/issues/20458);
Qwen3.5 `enable_thinking:false` repeatedly ignored without the exact flag combination
(https://github.com/ggml-org/llama.cpp/issues/20182). Budget without the transition message
cratered Qwen3.5-9B HumanEval 94% → 79% in maintainer tests; with it, ~89%
(https://github.com/ggml-org/llama.cpp/pull/20297). For Cerebro's allowlist this means:
template-kwarg path and budget path must both be allowlisted and soak-tested per model.

## 3. Self-refine / intrinsic self-correction: the negative result stands

- Huang et al. (ICLR 2024): without oracle labels, self-correction *degrades* reasoning on
GSM8K/CommonSenseQA/HotpotQA; GPT-3.5 flips correct→incorrect (8.8%) more than
incorrect→correct (7.6%); multi-agent debate ≡ self-consistency at equal sample count
(https://arxiv.org/abs/2310.01798).
- Kamoi et al. (TACL): no prior work shows successful prompt-only self-correction on general
tasks under fair settings; it works with reliable external feedback or large-scale
fine-tuning (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713/125177/When-Can-LLMs-Actually-Correct-Their-Own-Mistakes).
- Kambhampati planning study: GPT-4 self-critique collapses Game-of-24 5%→3% and graph
coloring 16%→2% (false-negative rates 80–97%); sound external verifier restores gains
(https://www.alphaxiv.org/abs/2402.08115).
- ACL 2025 dark-side analysis: failure mechanisms are answer wavering, prompt bias (failed
cases have 4.4–6.1× longer prompts), and human-like biases (overthinking, cognitive
overload, perfectionism with 1.7–3.1× longer outputs); mitigations: repeat-the-question
and <10-sample SFT (https://p.rst.im/q/aclanthology.org/2025.acl-long.1314.pdf).
- Small models are worse, not better, at this: LLaMA-1B overthinks 45.4% more than LLaMA-70B
(https://doi.org/10.1609/aaai.v40i40.40723); ≤3B models lose up to 75% accuracy after 8K
long-CoT SFT examples via error accumulation (https://doi.org/10.18653/v1/2025.emnlp-main.251);
SCORE: small-model self-refine needs a *strong* (GPT-4-class) verifier; weak self-verifiers
mislead (https://doi.org/10.18653/v1/2024.findings-acl.924).

## 4. What does work: train it in, or verify from outside

- **SCoRe** (online multi-turn RL on self-generated traces, two-stage + correctness-flip
bonus): +15.6% MATH / +9.1% HumanEval self-correction on Gemini 1.0 Pro / 1.5 Flash;
plain SFT on correction traces collapses to trivial edits (https://arxiv.org/html/2409.12917v1).
- **ReVISE** (ICML 2025): curriculum preference learning for verify-then-rethink plus
confidence-aware decoding; efficient test-time scaling without an external verifier
(https://proceedings.mlr.press/v267/lee25ab.html).
- **Self-verification RL** (NeurIPS 2025, Qwen2.5-Math-7B / R1-Distill-1.5B): unifies
generation+verification in one RL run with an online policy-aligned buffer; weighted
aggregation over N samples beats external-RM best-of-N/beam search at inference
(https://proceedings.neurips.cc/paper_files/paper/2025/file/07320f63b9d44f0f128f68ab56721729-Paper-Conference.pdf).
Complementary 2026 result: training *only* to self-verify matches generation-training on
math and shortens traces (https://arxiv.org/html/2602.07594v1).
- **SSR** (Nov 2025): Socratic (sub-question, sub-answer) decomposition + step-level
re-solve confidence + targeted refine; beats holistic self-refine on 5 benchmarks but at
"substantially higher" compute (https://arxiv.org/abs/2511.10621).
- **External grounding wins**: code-execution feedback (self-debug with unit tests) is the
consistently positive case in every survey above; treat the compiler/test-runner as the
verifier, not the model's own opinion.
- **Generative verifiers**: GenRM (next-token `Yes/No` + CoT rationale, Best-of-N):
5%→45.3% algorithmic, 73%→93.4% GSM8K (Gemma2-9B verifier over Gemini 1.0 Pro solutions)
(https://arxiv.org/abs/2408.15240); GenPRM adds step-level CoT+code verification, 1.5B
beats GPT-4o on ProcessBench (https://ojs.aaai.org/index.php/AAAI/article/view/40797).
Caveat for batch 1: at fixed FLOPs, plain self-consistency beats GenRM until ~8× compute;
GenRM needs ~128× more for +3.8% (https://doi.org/10.48550/arxiv.2504.01005). HSRM (Aug 2026)
cuts verifier cost by reading frozen-generator hidden states with a ~2M-param head instead
of re-reading text (https://arxiv.org/abs/2608.30841). None of these ship in llama.cpp;
all are research-only for Cerebro.

## 5. Multi-path self-consistency: effective, linear-cost, now with early exits

Classic SC (sample N, majority vote) still helps weak-model/hard-task combos, but costs N×
tokens at batch 1. Proven economizers: ESC early-stops on window agreement (GSM8K −80.1%,
MATH −33.8% samples at parity: https://proceedings.iclr.cc/paper_files/paper/2024/file/3fe2a777282299ecb4f9e7ebb531f0ab-Paper-Conference.pdf);
DSC routes easy questions to single-sample via LLM difficulty ranking (−65% cost on GPT-4
vs SC: https://doi.org/10.18653/v1/2025.findings-naacl.383); CISC weights votes by P(True)
confidence (−41–46% paths: https://aclanthology.org/2025.findings-acl.1030.pdf); Slim-SC
prunes similar chains step-wise (−45% latency, −26% KV usage on R1-Distill:
https://aclanthology.org/2025.emnlp-main.1750.pdf); ranked voting (Borda/IRV over ranked
answers) adds +2.7–5% over majority vote on small models
(https://aclanthology.org/2025.findings-acl.744.pdf). Counterpoint (Nov 2025): on strong
models near ceiling (Gemini 2.5 Pro, MATH-500 98% single-pass), 15–20 samples buy only
+0.4–1.6% at ~15–20× tokens, sometimes *hurting* (https://arxiv.org/html/2511.00751).
And under a fixed token budget, parallel independent paths beat one ever-longer "Wait..."
trace by up to ~20% (https://proceedings.neurips.cc/paper_files/paper/2025/file/fc067ac218430c409d6f65403328f740-Paper-Conference.pdf).
Cerebro rule of thumb: single pass by default; spend parallel samples only when single-pass
reliability is demonstrably low (hard coding tasks), with ESC-style early stop.

## 6. Concise reasoning: the cheapest verified win

- CCoT ("think step-by-step + be concise"): −48.7% response length at parity accuracy on GPT-4;
GPT-3.5 loses −27.7% on math only; −22.7% per-token cost
(https://arxiv.org/pdf/2401.05618v3.pdf).
- Chain-of-Draft (≤5 words per step): matches CoT at ~7.6–20% of tokens; GSM8K 91% vs 95%
at 44 vs 205 tokens; sports-understanding −92% tokens
(https://www.alphaxiv.org/abs/2502.18600).
- Token-complexity study (31 prompts × 6 LLMs): all brevity prompts sit on one universal
length–accuracy curve — length matters more than wording; optimal per-question compression
would be 3–11× but prompts only achieve ~1.3×; a verifier router gets closest
(https://arxiv.org/html/2503.01141).
- Overthinking is real and small-model-skewed: "2+3?" can trigger thousand-token traces;
verbosity can flip right answers wrong (https://arxiv.org/html/2503.16419,
https://arxiv.org/pdf/2505.00127). For Cerebro this is pure prompt engineering —
no runtime change, no allowlist impact — and it multiplies every other saving.

## 7. Budget forcing ("Wait" / forced stop)

s1 (Qwen2.5-32B + 1K traces): cap thinking by appending end-of-thinking delimiter; extend by
suppressing it and appending "Wait" → AIME24 50%→57%, beats o1-preview by up to 27% on
MATH/AIME24; flattens after ~6 extensions, loops beyond that
(https://arxiv.org/abs/2501.19393, https://simplescaling.github.io/). Prompt-only
length control fails (models can't count tokens); "Wait" beats "Hmm"/"Alternatively".
Cerebro can approximate the *cap* side today with `--reasoning-budget N` + transition
message; the *extend* side has no llama.cpp equivalent (would need client-side re-prompt).

## 8. Batch-1 laptop cost model (applies to every item)

At batch 1 there is no throughput amortization: wall time ≈ prefill + tokens/tok/s, and the
reference numbers (29 tok/s MoE offloaded, 2–12 tok/s dense) dominate everything. So: halving
thinking tokens halves latency; N parallel samples cost N× latency unless run concurrently
against VRAM headroom the 8 GB card does not have for 20–30B models; reasoning tokens bill
against the same context window as answers (DeepSeek R1 shares `max_tokens`;
Qwen3.5 agents starved at max_tokens 600–700: https://github.com/ggml-org/llama.cpp/issues/20182).
Short benchmarks overstate MoE tok/s ~3× vs sustained soak (existing Cerebro measurement) —
apply the same discount to any paper tok/s claim. Default Cerebro policy implied by the
evidence: effort Low/off for chat and easy code, Medium for standard coding, High only for
verifiably hard tasks; concise-style system prompts everywhere; self-repair loops gated on
compiler/test output, never on the model's own "looks good".
