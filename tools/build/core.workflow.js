export const meta = {
  name: 'cerebro-core',
  description: 'Cerebro engine wave A: harden kept crates, config, bench runner, model scanner, autotuner, loopback server, Hugging Face downloader, CLI. Opus 5.5 builder + Opus 5.5 mentor-critic to 10/10, one pair at a time.',
  phases: [{ title: 'Core', detail: 'one component at a time in the cerebro-core worktree' }],
}

// args: { skip: [ids already committed] }
const SKIP = (args && args.skip) || []
const BUILDER = 'claude-opus-5-5'
// Owner rule 2026-09-23: Opus 5.5 builds and reviews; only the mechanical commit/merge steps use Sonnet 5.
const CLERK = 'claude-sonnet-5'
const CRITIC = 'claude-opus-5-5'
const MAX_ROUNDS = 8
const DONE = (args && args.done) || {}
const REPO = (args && args.repo) || 'C:/Users/noahm/Projects/cerebro-core'
const BRANCH = (args && args.branch) || 'core'
// args.only runs a subset (independent components in a second worktree, owner asked for parallel systems).
const ONLY = (args && args.only) || null
const TOOLS = 'C:/Users/noahm/Projects/cerebro/tools/local'
const LLAMA = 'C:/bench/llamacpp-official'
const TINY = 'C:/Users/noahm/.lmstudio/models/skiesoffire/bastion-control/bastion-control-0.5b-f16.gguf'

const CONTEXT = `REPOSITORY: ${REPO} (git branch ${BRANCH}). Cerebro Engine: open-source (Apache-2.0), Windows-first, fully offline local LLM harness. Read docs/RESTART.md first (what exists, owner requirements, plan), docs/CLAIMS.md (binding), docs/specs/effort-levels.md, and the crates you touch. Kept crates: cerebro-flags (closed llama.cpp flag allowlist, argv emission), cerebro-probe (CPU/GPU/backend identity, SHA-256 manifest, DACL audit), cerebro-supervisor (preflight gate, job-contained spawn with env allowlist, lifecycle), cerebro-bench (llama-bench JSON parse + stats), cerebro-store (hash-chained result records); crates/cerebro-config is parked (not a workspace member).
OWNER RULES: speed and performance are core values; everything works with no network; the user can switch off ANY protection (sandbox, integrity check, folder audit, permission prompts) through settings: safe defaults, one neutral recorded warning, never unrepresentable, never nagging. Measured on this machine: tuned flags beat llama.cpp defaults ~6x for a 30B MoE that does not fit VRAM, but defaults won for a small model that fits: never assume, always measure, and never recommend a configuration slower than llama.cpp's own automatic settings.
STYLE: match the kept crates: typed errors (thiserror), no unwrap/expect/panic outside tests, docs on every pub item, unsafe only in small Win32 FFI functions with SAFETY comments, tests that would fail if the code were wrong. Use vetted crates for anything cryptographic (rand/getrandom OsRng, subtle for constant-time compare, sha2, ed25519-dalek); never hand-roll crypto.
BUILD (PowerShell; msys2 link.exe shadows MSVC, so this preamble is required): $vc = "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Tools\\MSVC\\14.51.36231\\bin\\Hostx64\\x64"; $sdk = "\${env:ProgramFiles(x86)}\\Windows Kits\\10\\bin\\10.0.26100.0\\x64"; $env:PATH = "$vc;$sdk;$env:PATH"; then cargo test --workspace and cargo clippy --workspace --all-targets -- -D warnings.
FIXTURES: official llama.cpp build ${LLAMA} (llama-server.exe, llama-bench.exe; CUDA). Tiny model for live tests: ${TINY}. Machine-dependent tests are #[ignore] with a reason and must pass when run explicitly.
MACHINE RULES: load no model except the tiny one (and only in tests); open no visible windows; kill every process you start; no network except where the task needs it (Hugging Face client tests use recorded fixtures by default and one #[ignore] live test); never run git commit, reset, checkout, stash or rebase.
LOCAL FIRST PASS (free, before finishing each round; verify every point): python ${TOOLS}/local_review.py --repo ${REPO} --focus "<component>". Boilerplate such as test fixture data may be drafted with python ${TOOLS}/qwen_task.py --task "..." --out <file>, then checked line by line.`

const TASKS = [
  { id: 'core-hardening', crate: 'crates/cerebro-probe, crates/cerebro-supervisor', brief: `Harden the kept crates.
1. cerebro-probe DACL audit: a junction or symlink in any ancestor of the backend directory must make the audit fail closed (Unknown with a clear message) or be resolved and re-audited on the canonical path, and probe() must hash, audit and spawn from that same canonical path. Add a test that creates an ancestor junction (mklink /J needs no admin) and asserts the result.
2. cerebro-supervisor: find and fix the timing-sensitive lifecycle test that fails under full-workspace parallel load (run cargo test --workspace 5 times; it must pass every time). Review lifecycle.rs end to end against its module docs and fix any gap.
3. User-switchable protections (owner rule): add a Protections settings struct consumed by preflight and spawn with explicit booleans for integrity verification, folder-permission audit, environment allowlist, job-object containment and elevation refusal. Defaults all on. Turning one off is honoured and recorded as a structured warning in the launch record, never refused. Tests for each switch in both states.` },
  { id: 'config-rework', crate: 'crates/cerebro-config', brief: `Rejoin crates/cerebro-config to the workspace after removing its duplication: delete its private owner-only DACL walk and use cerebro-probe's audit instead (or a small shared function exported from cerebro-probe); delete its separate profile::Preflighter spawn gate and route decisions through cerebro-supervisor's preflight with the Protections struct from core-hardening; replace the "unrepresentable unsafe DACL override" design with ordinary protection settings (owner rule). Keep schema validation, atomic save, file watching and DPAPI secret storage. Extend the schema with: default effort level and editable per-level instructions/budgets (docs/specs/effort-levels.md), default verbosity (Brief) with editable instructions, model folders, default load parameters and named load presets, protections, network/offline mode, and plugin permissions defaults. All tests green; the JSON Schema is regenerated.` },
  { id: 'bench-run', crate: 'crates/cerebro-bench', brief: `Add module run to crates/cerebro-bench: drive llama-bench safely and return parsed, provenance-keyed results. Build argv with cerebro-flags in the Bench dialect only; always -o json, -r (desired reps + 1, the first is discarded by stats), optional --delay, the depth list. Spawn through cerebro-supervisor (preflight with the user's Protections, job object, env allowlist, captured logs, timeout that kills the tree). Record free VRAM before and after (cerebro-probe gpu) and CPU topology; build the provenance key (backend build, commit, backend manifest digest, model SHA-256 cached by path+size+mtime, free VRAM at start, power state if cheaply available). A non-zero exit or missing JSON is an infrastructure error with stderr tail, never an empty result. Tests: exact argv for representative configs, failure classification with a fake spawner, and an #[ignore] live test on the tiny model (CPU, -dev none, 3 kept reps) asserting a complete pp/tg/tg-at-depth triple and that no llama-bench process remains.` },
  { id: 'models', crate: 'crates/cerebro-models', brief: `Create crates/cerebro-models: the local model library. Scan configured folders (defaults: the LM Studio models folder %USERPROFILE%/.lmstudio/models with its publisher/model layout, C:/models, and user-added folders) for GGUF files, reading only the header key-value metadata with a bounded, fuzz-safe parser (limits on string lengths, array counts and total header size; a hostile file cannot cause huge allocation or panic): architecture, name, parameter count, file type/quantization, context length, layer count, expert count and experts used, KV heads, embedding length, chat template presence and whether it supports tools, and modality (text, vision projector mmproj, audio, image-generation GGUF such as FLUX/SD/Qwen-Image). Pair each vision model with its mmproj in the same folder. Group multi-shard files. Estimate fit for this machine: weights + KV at a given context and KV type vs free VRAM and RAM, classifying fits-in-VRAM / MoE-experts-on-CPU / partial offload / too large. Add a cargo-fuzz target for the parser (document how to run it) plus property tests; unit tests with small synthetic GGUF headers; an #[ignore] live test that scans the real folders and prints a table.` },
  { id: 'hardware-backends', crate: 'crates/cerebro-probe, crates/cerebro-backends', brief: `Every feature for every hardware setup (owner rule: this ships to people worldwide, not only to the development laptop). 1) Extend cerebro-probe to detect the full hardware picture: NVIDIA GPUs with compute capability and architecture (Ampere, Ada, Hopper, Blackwell) and multi-GPU topology, AMD GPUs (ROCm/HIP or Vulkan capable), Intel Arc and integrated GPUs (SYCL/Vulkan), NPUs (Intel, AMD XDNA, Qualcomm Hexagon), CPU instruction sets (AVX2, AVX-512, AMX, NEON), RAM, and OS; each detector fails soft to "unknown", never guesses. 2) Create crates/cerebro-backends: a registry of inference backends behind one trait (install location, integrity manifest, capabilities, how to build argv, health check): llama.cpp in its CUDA 12, CUDA 13, Vulkan, ROCm/HIP, SYCL and CPU builds as the default for GGUF, plus optional backends enabled only when the hardware and OS support them: vLLM and SGLang (Linux or WSL, datacenter and multi-GPU, FP8/NVFP4 on Ada/Blackwell/Hopper, FlashAttention 3 on Hopper), ExLlamaV3 (NVIDIA), MLX (Apple Silicon, future macOS build), OpenVINO or vendor NPU runtimes. 3) A capability matrix that maps detected hardware to available features (quant formats, attention kernels, tensor split across GPUs, speculative decoding types, NPU offload) with the reason each unavailable feature is unavailable, so the app can show "available on your hardware", "needs a Hopper GPU", etc. Features unsupported on this laptop are still modelled, tested with fixture hardware profiles, and switchable by users who have the hardware. Tests: detector parsing on recorded outputs from many hardware profiles (fixtures for RTX 3070 Ti laptop, RTX 4090, H100, 2x RTX 3090, RX 7900 XTX, Arc A770, CPU-only, Snapdragon X), capability matrix correctness, and backend selection.` },
  { id: 'tune', crate: 'crates/cerebro-tune', brief: `Create crates/cerebro-tune: autotuner v1. For a model and workload profile (chat, coding, long-documents -> target context depth), generate a small candidate set: (a) llama.cpp's own automatic settings (the --fit baseline, measured, always included), (b) full GPU offload when the fit estimate allows, (c) MoE expert offload levels (-ncmoe steps) when the model is MoE and does not fit, (d) dense FFN offload steps (-ncffn) when dense and does not fit, (e) KV cache type q8_0 vs f16 with flash attention, (f) thread and P-core mask variants from cerebro-probe. Screen every candidate with cerebro-bench run (pp, tg, tg at the profile depth), pick the winner by tg at depth with a bootstrap-CI tie rule preferring fewer threads, and NEVER select a candidate whose confidence interval is below the baseline's (if nothing beats it, the baseline wins and the result says so). Persist every measured candidate to cerebro-store with the sampler profile. Stream progress events (candidate started, sample, finished, winner) through a callback so the CLI and the app can show live progress. Unit tests with a fake bench runner covering winner selection, tie rule, the never-slower rule and cancellation; an #[ignore] live test on the tiny model.` },
  { id: 'telemetry', crate: 'crates/cerebro-telemetry', brief: `Create crates/cerebro-telemetry, the hardware and inference sampler behind Cerebro's Monitor (read C:/Users/noahm/Projects/cerebro/docs/research/strata-analysis-2026-10-07.md and the Strata source at C:/Users/noahm/strata first; Strata's serve/telemetry.py is the reference idea, MIT). A background thread samples once a second and never fails or blocks its host: anything unreadable is None. GPU: NVIDIA through NVML loaded dynamically at runtime from the driver (nvml.dll / libnvidia-ml.so.1, no CUDA or crate that links it at build time): name, load, VRAM used/total, temperature, power and power limit, clocks, PCIe gen/width and RX/TX throughput, per GPU for multi-GPU. AMD and Intel on Windows through DXGI/D3DKMT adapter memory and engine usage where available, AMD on Linux through amdgpu sysfs; every backend behind one trait so new vendors plug in. CPU total and per-core load, RAM used/total, process RSS, disk read/write rates, CPU temperature where the OS exposes it. Inference counters fed by the server (state idle/reading/generating, prompt read/total, generated/max, decode and prefill tok/s, queued, cache reuse). Keep a 60-sample ring per series for sparklines plus a 24 h history downsampled to 1 sample per 10 s, bounded memory. Expose a serde snapshot struct shaped like {live, hardware, hardware_static, engine, history, totals} and a Prometheus text renderer. Unit tests with fake backends (including a backend that panics or hangs: the sampler must survive and time out), a test proving bounded memory, and an ignored live test that reads this machine's RTX 3070 Ti through NVML. Sampling overhead must stay under 1% of one core: measure and report it.` },
  { id: 'turbo', crate: 'crates/cerebro-turbo', brief: `Create crates/cerebro-turbo: Cerebro's experimental performance layer and release channels (read C:/Users/noahm/Projects/cerebro/docs/specs/positioning.md and C:/Users/noahm/Projects/cerebro/docs/research/strata-analysis-2026-10-07.md first). 1) Channels: every optimisation is a typed feature with a channel (Stable/Beta/Experimental), a hardware gate (GPU vendor/arch, VRAM, RAM, CPU ISA, OS), a measured-benefit record and a user override; nothing Experimental is on unless the user opted into that channel. 2) Strata backend adapter: detect a Strata install (run config JSON, strata.exe, packs, version), launch it through cerebro-supervisor's preflight and Job Object path with argv only, health-check its OpenAI endpoint, read its /metrics (expert cache hit rate, MoE routing, draft acceptance, conversation cache) and map them into cerebro-telemetry's snapshot shape; credit its MIT licence. 3) Speculation policy for llama.cpp backends: enable draft/MTP/prompt-lookup only when a quick measured probe shows a win on this machine, otherwise off. 4) Conversation-state reuse policy for agent sessions (llama.cpp slot save/restore keyed by session, message-boundary checkpoints). Unit tests with fakes for every gate and policy, an ignored live test that starts the owner's Strata at C:/Users/noahm/strata on the tiny path only if a run config exists (never load a large model in tests; if only large models exist, test detection and argv building without launching).` },
  { id: 'server', crate: 'crates/cerebro-server', brief: `Create crates/cerebro-server: Cerebro's loopback gateway (axum + tokio). It supervises one llama-server through cerebro-supervisor (tuned argv from cerebro-store or defaults, the user's Protections) and exposes an OpenAI-compatible API on 127.0.0.1 only: /v1/models, /v1/chat/completions (streaming SSE and non-streaming), /v1/completions, /v1/embeddings passthrough, plus /cerebro/health and /cerebro/status (model, tokens/sec, VRAM). Security: bearer token required on every route including reads, 256-bit token from the OS CSPRNG stored via DPAPI (cerebro-config secrets) and compared in constant time; Host header must be 127.0.0.1:<port> or localhost:<port> (DNS-rebinding defence); Origin allowlist (none by default, the desktop app's origin when configured) and CORS closed by default; request body size limit; per-request timeouts; never logs prompts or tokens. Effort and verbosity: a request may carry cerebro_effort (low..max) and cerebro_verbosity (brief, normal, detailed) (defaults from settings, Brief and the settings default effort); the gateway appends the configured instructions from docs/specs/effort-levels.md to the system prompt, maps effort to the backend's reasoning budget when supported while reserving the answer budget separately (guarding the reasoning trap), and for XHigh/Max runs the self-review pass (and Max's independent check) before returning, reporting timing in a cerebro field. Tests: auth required, constant-time compare used, Host/Origin rejection, body limit, SSE passthrough with a fake upstream, effort/verbosity injection exact text, the self-review pass ordering, and an #[ignore] live end-to-end test with the tiny model.` },
  { id: 'hub', crate: 'crates/cerebro-hub', brief: `Create crates/cerebro-hub: the Hugging Face client behind one-click downloads. Search models (by family, task, text), list repository files with sizes and LFS sha256, read licence and gating status, and download files with resumable HTTP range requests, bounded concurrency, progress events, pause/resume/cancel, atomic finish (download to .part then rename), and SHA-256 verification against the LFS hash (mismatch deletes the file and reports). Gated models use the user's own access token stored via DPAPI; never bypass gating, surface the "accept the licence on Hugging Face" step clearly. Handle the case observed on this machine where some downloads stall at 0 bytes (detect no-progress, retry with backoff, fall back to the plain resolve URL). Offline: a cached catalog snapshot is returned with an offline flag and downloads fail fast with a clear error. TLS via rustls with platform roots; no telemetry; a descriptive User-Agent. Tests with recorded HTTP fixtures (a local mock server) for search, listing, resume after interruption, checksum mismatch, gated error and stall recovery; one #[ignore] live test downloading a tiny public file.` },
  { id: 'cli', crate: 'crates/cerebro-cli', brief: `Create crates/cerebro-cli, the cerebro.exe command line and slice 1's deliverable: cerebro models (table of local models with modality badges and fit), cerebro pull <repo-or-search> (hub download with a progress bar and licence display, gated prompts), cerebro tune <model> [--profile chat|coding|long] (live progress, final table comparing the winner with llama.cpp's automatic settings, never slower), cerebro serve <model> (start the loopback gateway, print the endpoint and where the key is, Ctrl+C stops cleanly and kills the child), cerebro chat [<model>] (terminal chat against the gateway with streaming, /effort and /verbosity commands per docs/specs/effort-levels.md, Brief by default), cerebro bench <model>, cerebro doctor (probe summary, protections status, backend integrity). Clean, colourful but accessible output (respects NO_COLOR), helpful errors with next steps, --json output for every command. Tests: argument parsing, JSON output schemas, doctor on a fake probe, and an #[ignore] live smoke test running models, tune and a one-message chat on the tiny model.` },
]

const BUILD_RESULT = {
  type: 'object',
  properties: {
    summary: { type: 'string' }, files_changed: { type: 'array', items: { type: 'string' } },
    tests: { type: 'string' }, clippy: { type: 'string' }, live_tests: { type: 'string' },
    local_review: { type: 'string' }, findings_addressed: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'files_changed', 'tests', 'clippy', 'live_tests', 'local_review', 'findings_addressed'],
}
const FINDING = {
  type: 'object',
  properties: {
    severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
    area: { type: 'string', enum: ['functional', 'security', 'performance', 'code', 'tests', 'docs'] },
    file: { type: 'string' }, line: { type: 'number' }, problem: { type: 'string' },
    fix: { type: 'string' }, suggested_code: { type: 'string' },
  },
  required: ['severity', 'area', 'file', 'line', 'problem', 'fix', 'suggested_code'],
}
const CRITIC_RESULT = {
  type: 'object',
  properties: {
    score: { type: 'number' }, findings: { type: 'array', items: FINDING },
    guidance: {
      type: 'object',
      properties: {
        root_causes: { type: 'array', items: { type: 'string' } }, plan: { type: 'array', items: { type: 'string' } },
        design_advice: { type: 'string' }, pitfalls: { type: 'array', items: { type: 'string' } },
        tests_to_add: { type: 'array', items: { type: 'string' } },
      },
      required: ['root_causes', 'plan', 'design_advice', 'pitfalls', 'tests_to_add'],
    },
    tests: { type: 'string' }, clippy: { type: 'string' }, live_tests: { type: 'string' }, verdict: { type: 'string' },
  },
  required: ['score', 'findings', 'guidance', 'tests', 'clippy', 'live_tests', 'verdict'],
}

const POSITIONING = `\nPOSITIONING (owner, 2026-10-07, binding; full text in C:/Users/noahm/Projects/cerebro/docs/specs/positioning.md, read it): Cerebro WINS, it does not wrap. Never frame it as "llama.cpp with good settings" or "the best of others": it is the CachyOS of local AI, a performance-first harness shipping experimental and beta optimisations (own MoE expert cache, measured speculation, conversation-state reuse, KV streaming, Strata turbo backend) behind Stable/Beta/Experimental channels. Live model + system metrics with percentages (MoE active experts %, expert cache hit %, draft acceptance %, VRAM split, KV fill, decode/prefill, GPU/CPU/RAM/PCIe) are a signature feature. Audience: pros by default (no buzz names for old models, no pre-emptive downloads; show current best quants such as GSQ-RCO, Unsloth UD, imatrix IQ, EXL3, AWQ, MLX with measured quality per GB), with a Beginner experience level that auto-configures and an Advanced escape hatch. Accent colour is now blue on cool slate neutrals (design/tokens.css); never reintroduce amber/orange. Copy: never describe a feature by pointing at another product ("in the style of X", "like X"); say what the user gets. Other products appear only on the comparison page and for interoperability facts.`
const POS_EXEMPT = ['server', 'telemetry']
const ctx = (task) => CONTEXT + (POS_EXEMPT.includes(task.id) ? '' : POSITIONING)

function builderPrompt(task, round, findings, guidance) {
  const fix = findings && findings.length
    ? `\nROUND ${round}. A senior Opus reviewer found the problems below and wrote a plan. Follow it in order; adapt each suggested_code sketch so it compiles. Never weaken a test or suppress a lint to pass. If a suggestion is wrong, do the right thing and explain with evidence in findings_addressed.\nFINDINGS\n${JSON.stringify(findings, null, 1)}\nGUIDANCE\n${JSON.stringify(guidance, null, 1)}\n`
    : `\nROUND 1. Build it. If partial work exists on disk (git status), keep what is right and finish the rest.\n`
  return `You are the builder for one Cerebro engine component: ${task.id} (${task.crate}).
${ctx(task)}

BRIEF
${task.brief}
${fix}
DONE FOR THIS ROUND: cargo test --workspace and cargo clippy -- -D warnings pass (always run both), the component's #[ignore] live tests pass when run explicitly, and you ran the local first-pass review and addressed every valid point. Report exact result lines.`
}

function criticPrompt(task) {
  return `You are a harsh senior Rust and security reviewer AND mentor for Cerebro. Another builder wrote this and will revise it from your review alone. Your bar: production infrastructure that a senior reviewer at a security-focused company approves with no comments, fast, correct and sealed.
${ctx(task)}

COMPONENT: ${task.id} (${task.crate})
BRIEF IT MUST SATISFY
${task.brief}

PROCEDURE: git status and git diff; read every changed file in full; run cargo test --workspace, clippy with -D warnings, and the component's ignored live tests; confirm no llama-server/llama-bench process remains. Check every brief requirement (missing = major), correctness and edge cases, security (injection, TOCTOU, reparse points, auth, constant-time, unbounded input, panics on hostile input = critical), performance, error handling, tests that can actually fail, docs, owner rules (protections switchable, never slower than defaults, offline-safe). For EVERY finding give suggested_code (<= ~40 lines) or "" for a trivial fix, then guidance (root_causes, ordered plan, design_advice, pitfalls, tests_to_add). Do NOT edit files.
Score 10 only with zero findings. Any critical caps at 4, any major at 7, any minor at 9.`
}

const results = []
for (const task of TASKS) {
  if (SKIP.includes(task.id) || (ONLY && !ONLY.includes(task.id))) continue
  // Owner rule 2026-09-29: one pass. Build, review once, then one fix round repairs every critical
  // and major finding and verifies each minor (fix the real ones). args.done replays rounds already
  // run under the old loop so no finished work is repeated.
  let last = null, halted = null
  const rounds = Math.max(1, DONE[task.id] || 0)
  for (let round = 1; round <= rounds; round++) {
    const b = await agent(builderPrompt(task, round, last && last.findings, last && last.guidance), { label: `build:${task.id}:r${round}`, phase: 'Core', schema: BUILD_RESULT, model: BUILDER, effort: 'high' })
    if (!b) { halted = `builder r${round}`; break }
    const c = await agent(criticPrompt(task), { label: `critic:${task.id}:r${round}`, phase: 'Core', schema: CRITIC_RESULT, model: CRITIC, effort: 'high' })
    if (!c) { halted = `critic r${round}`; break }
    last = c
    log(`${task.id} r${round}: ${c.score}/10, ${c.findings.length} findings`)
  }
  if (halted) { results.push({ id: task.id, status: 'halted', where: halted }); log(`halted at ${task.id} (${halted}); resume with skip for committed tasks`); break }
  const major = last.findings.filter(f => f.severity !== 'minor').length
  if (last.findings.length) {
    const p = await agent(`ONE FIX ROUND for Cerebro engine component ${task.id} (${task.crate}). No reviewer follows you, so this round decides the result.
1. Fix every critical and major finding below properly (never by weakening a test or suppressing a lint).
2. Verify each minor finding: decide whether it is real by reading the code or writing a test. Fix every real one; for any not real, say why in one line.
3. Keep it maximally fast: never slower than llama.cpp defaults, no needless allocation or blocking on hot paths.
4. Run cargo test --workspace, cargo clippy --workspace --all-targets -- -D warnings, and the component's ignored live tests; confirm all pass and no llama-server/llama-bench process remains.
${CONTEXT}
FINDINGS
${JSON.stringify(last.findings, null, 1)}
MENTOR GUIDANCE
${JSON.stringify(last.guidance || {}, null, 1)}
Report exact result lines and what changed for each finding.`, { label: `fix:${task.id}`, phase: 'Core', schema: BUILD_RESULT, model: BUILDER, effort: 'high' })
    if (!p) { results.push({ id: task.id, status: 'halted', where: 'fix' }); break }
  }
  const passed = true, open = []
  const commit = await agent(`Commit one finished Cerebro engine component in ${REPO} (branch ${BRANCH}).
Component: ${task.id} (${task.crate}). Review: one review pass found ${last.findings.length} findings (${major} critical/major); the fix round repaired the critical/major ones and verified every minor.
${open.length ? `First append these open findings to docs/build/open-findings.md under "${task.id}":\n${JSON.stringify(open, null, 1)}\n` : ''}
Run git status; stage only this component's crate directories, the root Cargo.toml and Cargo.lock, and docs/build/open-findings.md if changed. Never stage target/ or secrets. Commit with an imperative first line, a short body, and these final lines exactly:
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q4T16r2eHnoWzCJrCw94mq
Never use --no-verify. Return the commit hash and first line.`, { label: `commit:${task.id}`, phase: 'Core', model: CLERK, effort: 'low' })
  if (!commit) { results.push({ id: task.id, status: 'halted', where: 'commit' }); break }
  results.push({ id: task.id, status: passed ? 'passed' : 'open-findings', open: open.length, commit })
}
return { results }
