# fresh--security — New security issues in local AI tools (Aug–Sep 2026)

Date: 2026-09-22. Focus: what changed since 2026-09-01, plus the Aug 2026 backlog that still matters.
All URLs below were opened and read. Anything not verifiable is marked `unverified`.
Scope note: Cerebro spawns llama.cpp as a child process with argv only, Windows-first, fully offline.
So the issues that matter most are: (a) llama.cpp/Ollama memory-safety bugs, (b) malicious model
files, (c) DNS rebinding against localhost APIs, (d) MCP/tool poisoning, (e) Tauri/WebView2 IPC,
(f) coding-agent supply chain.

## 1. llama.cpp

**RPC GRAPH_RECOMPUTE use-after-free → unauthenticated RCE (CVE-2026-39909, CVSS 4.0 9.2).**
llama.cpp before b8585: storing a computation graph, freeing referenced buffers, then triggering
re-execution leaves dangling pointers → arbitrary read/write, full RCE, no auth, no user
interaction. Fixed in b8585 (commit `389c7d4`, PR #21175). Only exposes hosts running the RPC
backend (rpc-server / `--rpc`), not standalone llama-server.
Source: https://www.vulncheck.com/advisories/llama-cpp-use-after-free-in-rpc-graph-recompute-handler
Cerebro relevance: HIGH — Cerebro plans CUDA+RPC+CPU backends; never bind RPC to LAN, pin ≥ b8585
(local build here is b10751, i.e. past the fix — verify at runtime anyway).

**GGUF parser heap overflows: two patch floors, not one.** CVE-2026-27940 (CVSS 7.8, integer
overflow → undersized heap alloc in `gguf_init_from_file_impl()`, bypass of the 2025 CVE-2025-53630
fix, fixed in b8146) and CVE-2026-33298 (CVSS 7.8, overflow in `ggml_nbytes()`, fixed in b7824).
Ollama separately ships its own Go GGUF loader with CVE-2026-7482 ("Bleeding Llama", CVSS 9.1,
unauthenticated OOB read over `/api/create`, fixed only in Ollama 0.17.1 — no llama.cpp build
fixes it). LM Studio and Jan vendor the same ggml parser. Rule: llama.cpp ≥ b8146 AND Ollama ≥
0.17.1; treat every untrusted GGUF like an untrusted binary; sandbox model loading.
Source: https://www.stingrai.io/blog/malicious-gguf-file-llama-cpp-parser-rce

**JSON-schema-to-grammar recursion DoS (CVE-2026-52130, CVSS 7.5, published 2026-09-01)** affecting
b5693 and earlier — `unverified` (seen only in search-result snippets, advisory page not opened).
Plausible but do not cite as verified.

**Cyera batch (CVE-2026-43627…43632, CVE-2026-70638…70640: batch init overflow, DRY sampler
underflow, KV-cache restore overflow, server UAFs) — `unverified`** (seen only in snippets; public
PR reportedly closed unmerged). Do not cite without opening the source.

## 2. Ollama

**Tensor-blob redirect SSRF (CVE-2026-85180, CVSS 8.7, published 2026-09-03).** Ollama 0.30.0–0.33.2
fails to validate redirect destinations for tensor-layer blob downloads (`x/transfer/download.go`
`resolve()` returns cross-host `Location` unchecked), so a malicious registry redirects pulls to
internal hosts incl. cloud metadata. No patched release at disclosure; fixes proposed in PR #17082
and PR #18021 (both open/unmerged). Separate path from the earlier CVE-2026-5530 `server/download.go` fix.
Source: https://www.vulncheck.com/advisories/ollama-0.30.0-through-0.33.2-ssrf-via-cross-host-tensor-blob-redirect

**Go GGUF decoder integer overflow (CVE-2026-86289, ≤ 0.31.1, fixed in 0.31.2-rc1).**
`readGGUFV1String` in `fs/ggml/gguf.go`: unbounded length arithmetic wraps, corrupting allocation
sizes; crafted GGUF triggers it (user must load the file). Patch adds overflow-safe element-count
math plus `MaxStringLength` (16 MB), `MaxArraySize` (64 MB), `MaxTensorDims` (4).
Source: https://www.sentinelone.com/vulnerability-database/cve-2026-86289/

**DNS-rebinding → persistent chat-template poisoning (NemoClaw + Ollama, CVE-2026-65105, CVSS 8.1,
disclosed 2026-08-25, Oasis Security via NVIDIA PSIRT).** NemoClaw starts Ollama with
`OLLAMA_HOST=0.0.0.0:11434`; non-loopback bind skips Ollama's Host-header check (the CVE-2024-28224
fix), so a single malicious webpage rebinds DNS to 127.0.0.1 and uses unauthenticated
`/api/show` + `/api/create` to rewrite the model's Go chat template — persistent, invisible,
survives system prompts. Fixed on macOS/Linux (v0.0.35, loopback + token proxy); Windows/WSL path
still exposed at disclosure. Cerebro lesson: bind inference loopback-only, add template-hash
integrity check (`ollama show --template` vs baseline).
Source: https://thehackernews.com/2026/08/a-malicious-webpage-could-poison-your.html

## 3. Open WebUI (Sept batch: upgrade to v0.11.1)

v0.11.1 (2026-08-25, `d3e8bf3`) is a security rollup: channel message-overwrite via chat-completion
API (CVE-2026-87994), same-origin XSS→account-takeover via terminal port-preview iframe
(`PortPreview.svelte`, allow-scripts + allow-same-origin; fix commit `54d7a22`), DNS-rebinding SSRF
in `SafePlaywrightURLLoader` (CVE-2026-87996 — Python validates hostname, Playwright re-resolves it;
fix `27402ff`), knowledge-base metadata disclosure across 11 vector backends (CVE-2026-87017, fix
`1d6d4e6`), plus opt-in human-in-the-loop tool approval.
Source: https://github.com/open-webui/open-webui/releases/tag/v0.11.1
Also: Vega/Vega-Lite chart rendering client-side SSRF (CVE-2026-70480, 0.6.34–0.11.0, fixed 0.11.0 —
unrestricted resource loader lets a chat code-block make the *viewer's* browser issue
attacker-chosen GETs).
Source: https://osv.dev/vulnerability/CVE-2026-70480

## 4. ComfyUI — unauthenticated pickle RCE (CVE-2026-68771, CVSS 9.3/9.8)

`LoadTrainingDataset` passes attacker-uploaded `shard_*.pkl` to `torch.load()` without
`weights_only=True`; chain unauthenticated `POST /upload/image` + `POST /prompt`. Fix: one-arg patch
(commit `94ee49b`, PR #14543). Public exploit exists (堡 2026-08-03). Cerebro rule: never
`torch.load` untrusted files without `weights_only=True`; never expose workflow APIs unauthenticated.
Source: https://www.vulncheck.com/advisories/comfyui-unauthenticated-rce-via-loadtrainingdataset-pickle-deserialization

## 5. MCP ecosystem — the dominant trending technique

**Tool poisoning (OWASP, canonical).** Malicious server embeds instructions in tool descriptions /
responses; agent treats them as trusted context → file reads, exfiltration, prompt bypass. Root
cause: descriptions reviewed once at connect time, responses never. Defenses: structured-output
schemas, privileged-tool isolation, server-side enforcement, allowlists, out-of-band confirmation.
Source: https://owasp.org/www-community/attacks/MCP_Tool_Poisoning

**Deadbugz (active Aug–Sep 2026, Pillar Security via CSA 2026-09-02).** Malicious MCP server pushed
via 23 unsolicited GitHub PRs (2026-08-10); benign for first 3 `tools/call`s, then rewrites its own
tool metadata to harvest SSH keys / AWS creds / kubeconfigs / shell history. No CVE. Defeats
one-time install review → requires continuous per-call schema fingerprinting + hash pinning +
allowlisting.
Source: https://labs.cloudsecurityalliance.org/research/csa-research-note-deadbugz-mcp-metadata-poisoning-20260902-c/

**MCP cache + instructions chaining (MCP-2026-008 / MCP-2026-015, both open/unpatched at
disclosure).** July-28 spec added `CacheableResult` with attacker-controlled `cacheScope: public`,
so a shared gateway re-serves one poisoned `tools/list` to every user; the unsanitized
`instructions` field is concatenated into victims' system prompts. Mitigation: never cache across
identities; treat server metadata as untrusted input; pin servers; log every tool list.
Source: https://datapace.ai/blog/mcp-cache-poisoning-prompt-injection

**LM Studio: MCP commands execute before handshake validation (issue #1819, Apr 2026, still open).**
Enabling (or auto-starting on launch) an `mcp.json` STDIO entry spawns the command first; a failed
handshake cannot un-run it, and forked children persist. `mcp.json` writability = persistence.
Wishlist: first-enable consent dialog with resolved path+hash, kill process group, strip env
(`LD_PRELOAD`, `PYTHONSTARTUP`, …), block shell wrappers, per-platform sandboxing.
Source: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1819

## 6. Tauri / WebView2 — origin confusion (CVE-2026-42184, fixed 2.11.1)

`is_local_url()` on Windows/Android maps `app://` to `http://<scheme>.localhost/` but checks only
the first DNS label (`split_once('.')`), so `http://app.evil.com/` is classified `Origin::Local`
and remote pages invoke local-only IPC commands. Windows is Cerebro's primary target → pin
Tauri ≥ 2.11.1, keep capabilities least-privilege, never render remote content in the privileged
WebView.
Source: https://github.com/tauri-apps/tauri/security/advisories/GHSA-7gmj-67g7-phm9

## 7. AI coding agents — Sept 2026 cluster

**GitSpawn (Manifold Security, 2026-09-01/02): 8 flaws, 7 agents.** Malicious `.git/config`
`core.fsmonitor` value executes when the agent runs background `git status` — outside the sandbox,
no approval prompt. Affects Claude Code (2.1.193→fixed 2.1.196; second `ultrareview` path still open
at 2.1.252), Codex CLI 0.102–0.130 (CVE-2026-19592, CVSS 7.3, fixed 0.131.0), Cursor (patched),
Goose <1.44.0 (CVE-2026-72718, fixed 1.44.0), Hermes/Qwen Code/Grok Build (unpatched at disclosure).
Clone-vector only via intact `.git/` file copies (zips, USB, sync folders), not `git clone`.
Defense: `git config --global core.fsmonitor false`, fresh clones only.
Source: https://shattered.io/gitspawn-ai-coding-agent-vulnerability-2026/

**Plugin4Shell (AIR, disclosed 2026-09-18).** Same missing check in 4 products: SHA pin verified for
the *requested* commit but checkout landing never verified → zero-click RCE at plugin load.
Claude Code 2.1.179 and Codex 0.146.0 patched silently (no CVE); Copilot unpatched; Gemini CLI
deprecated instead of patched.
Source: https://easternherald.com/2026/09/20/plugin4shell-ai-coding-agents-supply-chain-vulnerability/

**PhantomFix / Sentry Seer (CERT/CC VU#212479, CVE-2026-90999, CVSS 9.8, 2026-09-16, unpatched,
vendor status Unknown).** Public Sentry DSN accepts attacker-crafted error events; Seer Autofix
embeds the fabricated root-cause analysis into the coding-agent prompt; agent installs attacker's
package with repo access. Predecessor: Tenet "agentjacking" (June 2026, 85% success across Claude
Code/Cursor/Codex, 2,388 exposed DSNs). Mitigations: disable Seer auto-remediation/handoff, deny
unprompted installs, sandbox + egress-deny, proxy the DSN server-side.
Source: https://dev.to/jamilxt/a-fake-bug-report-can-make-your-ai-coding-agent-run-attacker-code-55pg

**Explosive prompts (arXiv 2609.22510, 2026-09-18, research-only).** Conditional dormant payloads
("when X, do Y") beat imperative injections 43–83% vs ≤3% across 9 production agents (Codex, Gemini
CLI, Claude Code, Cursor, Copilot, Devin, Kiro, Qwen Code, Assistant; n=30 each) and evade Prompt
Guard 2 / PIGuard / PIShield; proposed DeFuse detector (AUC 0.9994, 1.16 ms/doc) cuts live
execution 34.3%→3.0%. Batch-1 laptop irrelevance note: numbers are attack-success rates, not
throughput; no perf claim transfers.
Source: https://arxiv.org/abs/2609.22510

## 8. Malicious model files (cross-cutting)

**GGUF Jinja2 SSTI class.** `tokenizer.chat_template` is executable Jinja2; unsandboxed
`jinja2.Environment()` rendering = RCE. Known cases: llama-cpp-python < 0.72 ("Llama Drama",
CVE-2024-34359 — patched); SGLang `/v1/rerank` (CVE-2026-5760, CVSS 9.8, unpatched at 2026-04-21,
maintainers unresponsive; ZMQ pickle RCEs CVE-2026-3059/3060 fixed in 0.5.10). Scan templates for
`__class__`/`os`/`subprocess`/`eval`/`exec`; render only in `ImmutableSandboxedEnvironment`.
Sources: https://research.jfrog.com/model-threats/gguf-ssti/ ;
https://labs.cloudsecurityalliance.org/research/csa-research-note-sglang-cve-2026-5760-llm-serving-rce-20260/

## 9. Supply chain (AI toolchain)

**LiteLLM/Trivy (TeamPCP/Shai-Hulud, Mar 2026; fallout quantified Aug 2026).** Poisoned
`litellm==1.82.7/1.82.8` on PyPI (~40 min window, 10:39 UTC 2026-03-24); 1.82.8's `litellm_init.pth`
runs at every interpreter startup, harvesting SSH/cloud/k8s/LLM keys to `models.litellm.cloud`.
Aug 2026 analyses: 153 GB archive, 433,909 files, 118,829 runner dumps, 2,488 orgs (Hudson Rock);
2,500+ companies / 434k pipelines (CloudSEK); FBI FLASH-20260702-01 warns harvested creds will be
reused. Vendor: last clean 1.82.6, safe ≥ 1.83.0 via CI/CD v2 + cosign-signed images; Docker-proxy
users unaffected. Cerebro lesson (offline harness): pin + hash every dependency, verify signatures.
Source: https://docs.litellm.ai/blog/security-update-march-2026

## 10. Trending techniques & recommended mitigations (summary)

1. Prompt/tool injection through tool metadata (poisoning, Deadbugz runtime-gating, MCP cache →
   cross-user) — pin + continuously fingerprint schemas, allowlist servers, confirm sensitive ops
   out-of-band, key caches by identity.
2. Malicious model files (GGUF parser overflows, pickle `torch.load`, Jinja2 chat-template SSTI) —
   patched builds, `weights_only=True`, sandboxed rendering, provenance + checksums.
3. DNS rebinding vs localhost APIs (Ollama 11434, Open WebUI Playwright loader) — loopback-only
   bind, Host-header validation, auth reverse proxy, firewall the port, template integrity hashes.
4. Supply chain (poisoned PyPI, SHA-pin bypass, malicious `.git/config`, rogue PRs adding MCP
   servers) — hash-pinned deps, verify checkout landing, fresh clones, review MCP config as code.
5. Over-privileged local surfaces (Tauri IPC origin confusion, MCP pre-handshake exec, agent
   background commands) — least-privilege capabilities, consent gates, sandbox/sanitize subprocesses.
