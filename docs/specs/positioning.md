# Cerebro positioning and audience (owner, 2026-10-07)

Binding for every site page, app screen and engine task from now on. docs/CLAIMS.md still applies:
bold words, but every number is measured or labelled illustrative.

## We win, we do not wrap

- Cerebro is not "llama.cpp with good settings" and not "we take the best of others". Never write copy
  that frames us as a wrapper, a curator or a launcher. We are the fastest, most capable local
  harness, and we say so.
- Think CachyOS for local AI: a performance-first distribution that ships experimental and beta
  optimisations other harnesses do not have, measured on the user's machine, switchable per feature
  (Stable / Beta / Experimental channels). Examples: our own MoE expert cache and CPU+GPU expert
  overlap, MTP and prompt-lookup speculation gated on measured acceptance, conversation-state reuse and
  message-boundary checkpoints for agents, KV streaming past 64K on small cards, Strata as a turbo
  backend for the models it supports, autotuned prefill chunking. llama.cpp is one backend among
  several, never the identity.
- Lead with results: tokens/sec, prefill, time to first token, cache reuse, coding and reasoning scores,
  side by side against named rivals, with the method one click away.

## Show the machine thinking

Live metrics are a signature feature, not a debug page. While a model runs, show the model and the
system together with percentages:

- MoE: active experts per token (e.g. 8 of 512 = 1.6%), expert cache hit rate, experts resident in
  VRAM vs RAM, per-layer routing heat map, hottest experts this session.
- Speculation: draft acceptance %, tokens per verify pass.
- Memory: VRAM split weights / KV / expert cache / scratch, KV fill %, RAM pinned, disk streaming rate.
- Speed: decode and prefill now and over time, time to first token, reuse % of the prompt.
- System: GPU load, clocks, temperature, power vs limit, PCIe throughput, CPU per core, RAM, disk.

## Audience: pros by default, beginners welcome

- Default copy and defaults assume people who know AI. No marketing buzz names for old models, no
  pre-emptive downloads of yesterday's popular models. Show current, best-in-class quantisations and
  formats by measured quality per GB (e.g. GSQ-RCO, Unsloth Dynamic UD-*, imatrix IQ-quants, EXL3,
  AWQ/GPTQ for vLLM, MLX on Apple), with KL/perplexity and speed on the user's hardware.
- Experience level, chosen during install and changeable any time in Settings:
  - **Beginner:** Cerebro edits the configs for them: picks the model, quant, context, offload and
    sampler from the hardware probe, hides raw flags, explains terms inline.
  - **Advanced:** full control: raw llama.cpp/Strata flags, quant and file choice, load parameters,
    channels (Beta/Experimental), metrics density.
  - An "Advanced" escape hatch is visible on every install step, so an experienced user who picked
    Beginner by mistake can switch while the app is still installing, without restarting setup;
    nothing Beginner chose is lost (it becomes the starting values).

## Copy rule: users, not references

- Never describe a Cerebro feature by pointing at another product ("in the style of X", "like X",
  "inspired by X"). Owner requests that name other tools are internal inspiration, not copy. Say what
  the user gets: the benefit, the control, the number.
- Other products appear in public copy only where the user needs the fact: the comparison page (named,
  measured, with method) and interoperability (e.g. "reads the models you already downloaded with
  LM Studio or Ollama").
- Every line is written for the person using it: what it does for them, in plain words, with a number
  where one exists. No filler, no self-congratulation, no hedging.

## Options everywhere, protections optional (owner, 2026-10-08)

- Turbo layer (docs/specs/turbo-system.md): optimised builds, schedulers, CPU/GPU profiles and a
  profiler, CachyOS-style. This, not settings presets, is what separates Cerebro.
- Every safety setting is enabled by default and optional, sandbox included. No feature is blocked
  behind a protection the user turned off.
- Reasoning: native model effort without any added prompt, Cerebro verification prompts, both, or
  neither (docs/specs/effort-levels.md). Options, options, options: anything a pro would want to tune
  is tunable, and Beginner mode hides it.
