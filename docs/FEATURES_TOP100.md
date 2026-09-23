# Cerebro: the top 100 features, and more

Distilled on 2026-09-22 from the 534 Opus-reviewed core and experimental decisions in the first
research round (marrow `docs/research/2026-09/FEATURE_CATALOG.md`), the owner's requirements in
`docs/RESTART.md`, and measurements on the reference laptop. Ordered by expected value to users.
"Measured" means measured here; everything else is an expectation to be proven by the benchmark
campaign before it appears in marketing (see `docs/CLAIMS.md`).

## Speed (the core value)
1. Measured auto-tuning of GPU/CPU placement, MoE expert offload, dense FFN offload, KV type, threads and batch sizes. Measured: 30B MoE 3.1 -> 17.9 tok/s (provisional).
2. Never-slower guarantee: llama.cpp's own automatic settings are always measured as the baseline and win ties.
3. Speculative decoding tuning: n-gram self-speculation (no draft model), MTP heads, EAGLE-3 and draft models, on only when it measurably wins.
4. Sustained-soak verification: short-burst winners that sag after minutes are rejected.
5. Spill detection: configs that silently overflow VRAM into shared memory (the 10x cliff) are caught and rejected.
6. Exact-prefix prompt cache with "reused N tokens" shown per turn.
7. Context checkpoints and KV shifting so editing or regenerating a message never reprocesses the whole chat.
8. Slot KV saved to disk: pinned long contexts (a codebase, a handbook) load instantly after restart.
9. Hardware-aware backend choice: CUDA 12/13, Vulkan, ROCm, SYCL, CPU builds of llama.cpp picked per machine.
10. Datacenter backends when present: vLLM/SGLang with FP8/NVFP4 and FlashAttention 3 on Hopper/Ada/Blackwell.
11. Multi-GPU tensor split tuned by measurement.
12. NPU offload where a real runtime exists (Intel, AMD XDNA, Qualcomm).
13. Apple Silicon via MLX (macOS build).
14. Per-power-source profiles: separate tuned configs for AC and battery on laptops.
15. Thermal-aware tie-break: same speed at fewer threads and lower temperature (measured 94 C vs 58 C at equal tok/s).
16. Idle sleep that frees VRAM when you stop chatting, instant wake.
17. Brief-first answers by default: fewer output tokens, faster perceived speed.
18. Live HUD: prompt and decode tok/s, time to first token, VRAM, context fill.
19. Per-model sampler presets (min-p default) that avoid repetition collapse (measured fix on this machine).
20. Energy per token and tokens-per-watt shown for every tuned config.

## Intelligence
21. Effort dial Low to Max: thinking budget + verification rigor + self-check passes, cost shown.
22. Never-empty answers: answer budget reserved separately from thinking (guards a measured trap).
23. Multi-path self-consistency at Max for checkable questions.
24. Fast-model-first cascade: small model answers, escalates to a stronger local model when a check fails.
25. Tool-integrated reasoning: the model runs code to check its maths and logic.
26. Grammar-constrained JSON and tool calls, so small models stop producing malformed output.
27. Validate, repair and retry for structured output with an audit trail.
28. Per-model tool-protocol probe: native tool calls where supported, grammar fallback elsewhere.
29. Chat templates pinned and verified (a wrong template silently degrades any model).
30. Quality gate on every config change (KL divergence vs reference logits), so speed never costs accuracy silently.

## Coding agent
31. Tree-sitter repo map ranked by relevance, fitted to the context budget.
32. Per-model edit format: whole-file for small models (which fail diff formats), diffs for strong ones.
33. Auto lint and test repair loop with a retry cap.
34. Reproduce-then-fix with execution-verified completion.
35. Plan mode: read-only exploration, then a plan the user approves.
36. Shadow-git checkpoints with one-click restore of code and conversation.
37. Diff-before-apply review: accept, edit or reject each change.
38. Todo list and persistent task state across sessions.
39. Loop, no-progress and step-budget guards (small models loop; this stops it).
40. Context compaction that preserves task state.
41. Escalate a stuck run to a slower, stronger local model for one planning step.
42. Best-of-N attempts filtered by tests.
43. Lifecycle hooks (before edit, after test) the user can script.
44. AGENTS.md project memory, layered per folder.

## Tools, plugins and MCP
45. Claude Code / Codex style permissions: ask, session, always, deny, per tool, all editable.
46. Approval dialog that previews the exact command, diff or URL.
47. Plugin and MCP marketplace with one-click install and a clean icon set.
48. Offline plugin install from a file.
49. First-party MCP servers: scoped filesystem, SQLite, time, git, browser (network opt-in).
50. Tool definitions hash-pinned against tool poisoning and rug-pulls.
51. Skills (instruction packs loaded on demand).
52. MCP elicitation (tools can ask the user for input) and long-running tasks.
53. WASI-sandboxed third-party plugins (Wasmtime).
54. Full tool-call audit log.

## Every modality, natively
55. Built-in image generation: text-to-image, img2img, inpainting, upscaling (FLUX, SDXL, SD 3.5, Qwen-Image, GGUF).
56. Vision chat: drop in screenshots, receipts, charts.
57. Voice in: push-to-talk with local Whisper/Parakeet/Moonshine.
58. Voice out: Kokoro, Piper, expressive voices.
59. Full offline voice loop with barge-in.
60. Meeting transcription with speaker labels.
61. Document OCR with specialist vision models.
62. Embeddings and reranking models, managed like any other model.

## Documents and memory
63. Local document search: hybrid keyword + semantic retrieval with reranking.
64. Answers cite the exact document and passage; uncited claims are flagged.
65. Office, PDF and email parsing without cloud services, tables kept intact.
66. Watched folders with incremental indexing.
67. Persistent memory across chats, every memory visible, editable, deletable.
68. Per-chat "don't remember this" and a global memory off switch.
69. Memory provenance and trust levels (what the user said vs what a web page said).
70. Pinned contexts: a handbook loaded once, reused instantly.

## Models
71. One-click model hub: Qwen, Gemma, Llama, Mistral, DeepSeek, Phi, FLUX and more, every modality.
72. Hardware-fit badges on every model and quantization.
73. Resumable, checksum-verified downloads with stall recovery.
74. Licence shown before download; gated models handled with the user's own token, never bypassed.
75. Local library scanner that reads LM Studio's folder layout and any folder you add.
76. Vision models auto-paired with their projector files.
77. LoRA adapters loaded per request; merge and re-quantize tools.
78. Control vectors (activation steering) for tone and behaviour.
79. Model aliases and per-model presets.

## Security (sealed, and user-controlled)
80. Backend binaries hash-verified before every launch.
81. Folder-permission audit against DLL planting.
82. Backend sandboxed in a Job Object with a clean environment.
83. Loopback-only API with a per-launch random key, Host and Origin checks against DNS rebinding.
84. Model files validated before loading; no pickle or remote code, GGUF only.
85. Chat templates treated as code: hashed, shown, diffable.
86. Prompt-injection defences: untrusted content labelled, information-flow rules, human approval for irreversible tools.
87. Signed updates (Ed25519), signed installers, SBOM per release.
88. Zero telemetry with an in-app network panel that proves nothing leaves.
89. Every protection switchable by the user, with one neutral note.

## Experience
90. Guided first run: hardware detection, a recommended model, one click to chat.
91. Ctrl+K command palette for everything.
92. Conversation branching: edit any message, compare branches.
93. Blind A/B compare mode between models or configs.
94. Artifacts side panel (code, documents, diagrams) in a sandboxed frame.
95. "Did you know?" daily spotlight, What's New feed, beginner checklist.
96. Everything editable: load parameters, prompts, effort levels, keybindings, themes.
97. WCAG 2.2 AA accessibility and full keyboard control.
98. Local accounts, roles and quotas for a small office.
99. Reproducible benchmarks: every number carries machine, model, quant, build, depth and sampler.
100. Works everywhere: Windows, macOS and Linux on x64 and ARM, every GPU vendor, CPU-only.

## And more (101+)
101. Verified semantic cache for repeated business questions. 102. Watched-folder scheduled jobs. 103. Business template library (emails, proposals, extraction). 104. Evidence-first data extraction with a review queue. 105. PII detection and redaction in prompts, logs and indexes. 106. Hash-chained audit log for business mode. 107. Encrypted backups with restore drills. 108. Endless mode for long creative sessions. 109. Chat export/import from ChatGPT and Open WebUI. 110. Ollama- and Anthropic-compatible API endpoints so existing tools plug in. 111. OpenAI Responses API. 112. Priority lanes: interactive chat preempts background jobs. 113. Mixture-of-Agents across two local models. 114. Difficulty router that picks the model per question. 115. Idle-time memory consolidation. 116. Local fine-tuning orchestration (QLoRA) with a regression gate. 117. Prompt optimizer that tunes your prompts on your own examples. 118. Quality-per-joule model leaderboard. 119. Remappable keybindings with conflict detection. 120. Windows 11 native look (Mica, accent colour, snap layouts). 121. Daily auto-research feed of new local-AI techniques.
