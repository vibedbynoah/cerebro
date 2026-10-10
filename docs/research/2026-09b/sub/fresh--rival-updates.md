# Rival updates — Aug-Sep 2026 (checked 2026-09-22/23)

Method: websearch (11 queries) + webfetch opens of primary release pages. Every bullet cites a page actually opened in this run. Anything else is marked `unverified`. Batch-1 laptop numbers do not transfer from server/GPU-cluster claims.

## Cross-cutting backend: llama.cpp speculative decoding

- `llama-server --spec-type` now supports `draft-eagle3 | draft-dflash | draft-dspark | draft-mtp | ngram-mod | ngram-simple | ngram-map-k | ngram-map-k4v | ngram-cache | none`; `--spec-default` enables `ngram-mod`. Draft + draftless can be mixed; draftless wins on conflict. (https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- EAGLE-3: 1-layer draft reading target hidden states, needs `--target-model-dir` convert; listed drafts include Qwen3 1.7–32B + 30B-MoE, Gemma-4 31B/26B-A4B, gpt-oss 20B/120B. (https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- DFlash: block-diffusion draft emits whole block per forward pass (`--spec-draft-n-max` clamped to trained block); DSpark adds semi-autoregressive Markov head + `--spec-draft-conf-min P`; currently Qwen3 backbone only, Gemma4 planned. (https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- Upstream EAGLE-3 PR claims 2–3x (e.g. Llama-3.1-8B Q4_K_M 121→274 tok/s) on contributor hardware — datacenter/GPU numbers, not batch-1 laptop; treat as upper bound, verify with soak not short bench. (https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- Cerebro takeaway: LM Studio 0.4.22 + KoboldCpp 1.119 both productized DFlash/DSpark/MTP in Aug 2026 (see below). Cerebro's allowlist should add `draft-dflash/dspark/mtp + ngram-mod` with argv-only emission before any autotuner.

## LM Studio — 0.4.25 (Sep 19), 0.4.24 (Sep 9), 0.4.23/0.4.22 (Aug 28), 0.4.21 (Aug 12)

- 0.4.22: DFlash/DSpark/MTP assistant drafters (needs llama.cpp engine 2.29.1+); tool-image results in OpenAI `/v1/responses|/chat/completions` + Anthropic `/v1/messages`; Linux AppImage/`.deb` fixes + in-app updates. (https://lmstudio.ai/changelog/lmstudio)
- 0.4.24: advanced llama.cpp argv overrides for GGUF load; DSpark/DFlash compat heuristics; Loaded-Instances ctx-length fix. (https://lmstudio.ai/changelog/lmstudio)
- 0.4.25: Mac Splash engine (M3+, macOS 26.4+). 0.4.23: Qwen 3.8 Flash Next reliability. (https://lmstudio.ai/changelog/lmstudio)
- Cerebro takeaway: direct rival already exposes argv overrides + new drafters; parity item.

## Ollama — v0.34.3 (Sep 19, latest stable), v0.34.4-rc0 (Sep 23), v0.34.2 (Sep 15), v0.34.1 (Sep 14), v0.34.0 (Sep 5), v0.33.3 (Sep 2), v0.33.2/0.33.1/0.33.0 (Aug)

- v0.34.0: ChatGPT Desktop reuse of Ollama models; OpenAI tool-search + response compaction; faster structured output on Apple Silicon. (https://github.com/ollama/ollama/releases)
- v0.34.1: MLX safetensors `ollama create` GA; GGUF creation now requires llama.cpp tooling; `typical_p` deprecated for new models; `/api/tags` 3.1s→294ms cold; MLX memory fixes. (https://github.com/ollama/ollama/releases)
- v0.33.3: Gemma4 images+audio on MLX; cached prompt tokens; honor GGUF default params. (https://github.com/ollama/ollama/releases)
- v0.32.15 (Aug 19): metadata cache halves TTFT ~995→524ms in their bench (hardware unverified — do not quote as laptop figure). (https://github.com/ollama/ollama/releases)
- Assets include `Ollama-darwin.zip`, `ollama-linux-*.tar.zst`, `install.ps1/.sh` (Windows via ps1). Backend: bundled llama.cpp + MLX/MLX-C. (https://github.com/ollama/ollama/releases)

## KoboldCpp — 1.121 (Sep 15, latest), 1.120 (Aug 29), 1.119 (Aug 16)

- 1.121: Minimax H3 media refs (audio + multi-image); `--ffncpu` (= `--n-cpu-ffn` partial offload); Adaptive-P sampler opt; MusicUI rework; SDUI runtime LoRA selector. (https://github.com/LostRuins/koboldcpp/releases)
- 1.120: `--usedirectio` (+ mlock/mmap combo); Qwen3.8-Flash-Next + Ling-3.0-flash support (warns bad quants exist). (https://github.com/LostRuins/koboldcpp/releases)
- 1.119: Minimax H3 video/I2V; Muse Glimmer + Qwen3.8 + jinja tool-calling; DSpark/DFlash support; 64 image/audio attachments. (https://github.com/LostRuins/koboldcpp/releases)
- Windows: `koboldcpp.exe` (CUDA pyinstaller), `oldpc` (CUDA11+AVX1), `nocuda` (Vulkan), mac-arm64. Default `http://localhost:5001`. (https://github.com/LostRuins/koboldcpp/releases)

## Open WebUI — v0.11.4 (Sep 21, latest), v0.11.3/v0.11.2 (Aug 31), v0.11.1 (Aug 25)

- v0.11.1: `Tool Permissions` Full/Approval cards (Allow/Deny, per-conversation default); `ask_user` builtin (2–3 options + free text, survives reload); streaming deltas (250k-char reply ~1.46GiB→1.90MiB, 787x in their 20-char-chunk test); Manage Models (load/unload/remove) for llama.cpp + LM Studio; release notes state security fixes withheld for embargo. (https://openwebui.com/blog/v0-11-1-the-model-learns-to-stop-and-ask) (https://github.com/open-webui/open-webui/releases)
- v0.11.4: slim image ~175MB (~89% smaller); standard image −170MB; terminal skills + AGENTS.md per-turn; auto skill discovery; `/skills:create`; terminal tabs; Staan EU search. (https://github.com/open-webui/open-webui/releases)
- Security: upgrade promptly; 16-CVE Sep-9 batch details `unverified` (not opened this run).
- Cerebro takeaway: copy approval-gate + ask_user + delta-streaming patterns; note Docker-only distribution (no portable zip).

## Jan — v0.8.4 (Jul 23, latest): nothing in Aug–Sep window

- v0.8.4: settings/creds → backend store + OS keyring (localStorage kept as downgrade snapshot); native `web_search/web_fetch`; per-model chat-template kwargs (llama.cpp); token counter (remote/MLX); handlebars 4.7.9 RCE bump; MCP approval-hang fix. (https://github.com/janhq/jan/releases)
- Windows: `Jan_0.8.4_x64-setup.exe` (55.1MB) + AppImage/deb/dmg listed. (https://github.com/janhq/jan/releases)
- Cerebro takeaway: keyring-over-localStorage + per-model template kwargs are the two ideas to steal.

## AnythingLLM — v1.16.2 (Sep 22, latest), v1.16.1 (Aug 27), v1.16.0 (Aug 13)

- v1.16.2: `LLMman-OSS` replaces Docker Model Runner; Google Vertex AI; `generate-image` agent skill; You.com default (ZDR); Ollama runtime 0.33.3 bump; KoboldCPP provider overhaul. (https://github.com/Mintplex-Labs/anything-llm/releases)
- v1.16.1: Foundry Local embedded (WinML CPU/GPU/NPU, Win 11 24H2+); GenieX Snapdragon (replaces old NPU engine, re-download needed); CoT/activity rollup; abort warns before killing background inference. (https://github.com/Mintplex-Labs/anything-llm/releases)
- v1.16.0: `/img` generation; folder drag hierarchy; mid-session tool toggle; abort now kills inference (no ghost inference). (https://github.com/Mintplex-Labs/anything-llm/releases)

## Msty Studio — no Aug–Sep Studio release (latest opened: 2.9.6/2.9.5 Jul 24, 2.9.4 Jul 17, 2.9.1 Jul 10)

- 2.9.1: GGUF ready-without-restart; Llama.cpp preset refresh fix. 2.9.0: Context Studio + Dynamic pull mode. 2.8.0: Discover Hub + One Model Hub + provider tool presets. (https://msty.ai/resources/changelog/studio/)
- Nexus 0.5.0 (Sep 8 Console+Pro) and Go 0.15.0 (Aug 6 local STT) reported in vendor blog/newsletter — `unverified` (pages not opened this run).
- Cerebro takeaway: One Model Hub + pull-mode context are UX to beat; local-GGUF-import without restart is a parity bug to avoid.

## goose — v1.51.0 (Sep 17, latest), v1.50.0 (Sep 8), v1.49.0 (Sep 3), v1.48.0 (Aug 27), v1.47.0 (Aug 21)

- v1.51.0: Desktop→state-machine loop via ACP; EUrouter declarative provider; GPT-live API; gateway pairing allowlist; opt-in terminal bell; streamed-thinking ordering. (https://github.com/block/goose/releases)
- v1.49.0/1.48.0: auto-updater (desktop); background extension load; web-search/browser-use built-ins; PreToolUse `on_failure`; ~30-entry hardening batch (tool allowlists, secret-suffix redaction, symlink confinement, permission-denies-precedence). (https://github.com/block/goose/releases)
- Install: 39 assets incl. `download_cli.sh`, rpm/deb/tarbz2 (Windows via portable extract + `Goose.exe` per docs — `unverified` installer name).

## Cline — SDK harness migration at 100% (Aug 23)

- Dual-bundle loader (`legacy/`+`next/`+46KB loader, PostHog flag, auto-fallback, shared state, union `package.json`) worked around no-gradual-rollout Marketplace. (https://cline.bot/blog/how-we-migrated-11-million-users-to-clines-biggest-harness-upgrade)
- `task.mistake_limit_reached`: 6.34%→0.62% (10x); per-model 6–11x (Sonnet-5 11x, DeepSeek V4 Flash 10x). Open-weights tbench.ai: Kimi K3 82.02% vs 71.9/76.4; GLM 5.3 Flash 64.0%; DS V4 Flash 60.67% — vendor-run, verify independently. (https://cline.bot/blog/how-we-migrated-11-million-users-to-clines-biggest-harness-upgrade)
- Thesis: 2024 XML-tag harness taxes 2026 RL-native-tool models — validates Cerebro's harness-first bet. Extension/Desktop/CLI/SDK version numbers `unverified` (release pages not opened).

## DSH Desktop (anywhere-labs) — community DeepSeek-Harness desktop

- MIT; Windows x64 NSIS (`https://www.dshdesktop.cn/api/downloads/windows`) + macOS Universal DMG; no extra env; Electron supervisor spawns Harness child with backoff recovery; everything-a-plugin incl. desktop shell via `dsh-community-fabric` typed IPC. (https://github.com/anywhere-labs/dsh-desktop)
- 28.6k stars / 1.4k forks at open; mobile (iOS/Android) via encrypted WebSocket + short-lived tokens (v2.0.9+); LAN access explicitly unauthenticated — trusted-network-only warning. (https://github.com/anywhere-labs/dsh-desktop)
- Update check sends `X-DSH-Desktop-Version/Channel/Installation-Id` (random UUID, not HW-derived). Latest version number `unverified` (releases page not opened).
