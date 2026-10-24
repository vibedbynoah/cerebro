# Demo fixes (owner report, 2026-10-23)

The owner could not present Cerebro to a tech audience because the app did not hold up. Each item is
checked in the real exe before it is ticked.

## Must work
- [x] Chat on Qwen3 0.6B: replied to "Hello" by reasoning about an earlier test chat ("capital of
      France"), then stopped after the reasoning with no answer.
- [x] Chat speed: 2.8 tok/s on the 0.6B ("Both" placement); the whole model fits the graphics card
      (about 300 tok/s there). Every screen must use the fastest placement that fits.
- [x] Crash while chatting; startup crash "failed to create webview ... resource is in use" when a
      previous window is still closing (one instance only; the second launch focuses the first).
- [x] Every GGUF on the computer found automatically (LM Studio, Ollama, Hugging Face cache, common
      folders), no setup.
- [x] `/imagine`: the owner confirmed image generation works (2026-10-23).
- [x] Train and Tune are two screens (Tune off the sidebar; Train has a file picker for datasets). for similar-sounding things: one goes. Train (fine-tuning)
      stays, with: the user's own datasets; every setting editable (method such as LoRA/QLoRA,
      context, ...), each with a recommended value and a plain explanation, the user decides.
      Speed tuning for this machine moves into Turbo / the model's page.
- [x] Tune: the Start tune button is too large and sits oddly below the play button.

- [x] Every model in the library loads and answers in Chat (scripts/e2e-models.mjs over all of them).
- [x] A split model handed to llama.cpp keeps a 16-token stand-in session: Tune failed with "prompt
      of 18 tokens exceeds the 16-token context". Tune, Benchmark and chat must work on any placement.
- [x] Plugins screen is a "planned for a later build" placeholder: make it a real one-click catalogue
      (MCP servers from the official registry) or remove it.
- [x] Model names: "Qwen_Qwen3.6 35B A3B", "Qwen3.8-27B-IQ1_M", "Qwen3 8B Awq Compatible Instruct",
      quant "other"; image models in the Tune list ("sd-turbo ... at 0 (its most)").

- [x] Image editing: a follow-up bar under a generated image ("make the sky darker") and an upload
      to edit the user's own image (img2img on the image model).
- [x] Image gallery ("This session"): thumbnails keep the image's own proportions (not stretched);
      each image can be downloaded, edited, deleted; the user can add images there.
- [ ] Cluster refuses models with recurrent layers ("not all architectures are supported"): carry
      the recurrent state across computers so every model the engine runs can be shared.
- [ ] Video generation (needs a local video model and its pipeline in the engine: after the demo).
- [x] permissions.ts lists tools from plugins that do not exist (web search, git commit, send email).

- [ ] Topic-aware MoE expert cache (owner idea): the router's expert choices while reading the prompt
      are the topic's fingerprint (code, French, math ...); each topic keeps its own usage profile;
      on a recognised topic its hot experts are moved into VRAM/RAM before writing and its cold
      ones left on the SSD. Monitor shows the topic and the hit rate. Measure on Qwen3.6 35B A3B
      and Flash-Next before claiming a gain.
- [x] Big models (27B, 35B A3B, Flash-Next) load without running the computer out of memory:
      prefetch and page-locking keep a reserve (engine), the rest pages from the SSD.

- [x] 27B Qwen3.8 (hybrid) split: the answer after thinking comes out empty (Q4_K_M, IQ1_M); the
      LM Studio copy (with its vision projector) answered "No." to the capital question.
- [ ] Qwen3.6 35B A3B: 66 s to the first token on the first question (cold experts).
- [ ] Flash-Next: 2.4 vs llama.cpp 7.0 tok/s writing (assisted mode; whole-GPU path lacks
      hyper-connections, shared experts, n-gram embeddings).
- [ ] Low-bit 27B writing: 17 vs 28.6 tok/s (IQ1_S), decode kernels to profile.

## Model Hub
- [x] Publisher logos, not two-letter tiles: Hugging Face's logo for Hugging Face, each company's
      for its models (Qwen, Google, Meta, ...). The whole Hub as polished as LM Studio's.
- [ ] Image and video models downloadable and featured on the front page ("Featured", never "staff
      picks"). Image: needs a diffusers-folder downloader (the engine runs SD 2.x-family, e.g.
      SD Turbo). Video: only once the engine runs a video model (no buttons that lead nowhere).
- [x] Recommended file: best quality that runs fast here (largest up to 8-bit that fits the
      graphics card, else a 4-5 bit split), never a 30 GB f32 for an 8B.
- [x] The fit label describes the recommended file.

## Models screen
- [x] Loaded models always at the top.
- [ ] The user can reorder and organize models (drag to reorder, groups).

- [x] Fit label cut off ("Fits in graphics memor") and overlapping the "Not tuned yet" chip.
- [x] "Illustrative data — not a measurement" badge shows in the real app.

## Chat controls
- [x] Thinking: the model's own reasoning level, shown only for models that have one. Off = no
      thinking; Low / Medium / High = the model thinks, with a shorter or longer reasoning budget.
      Low never means "no thinking".
- [x] Effort: how much Cerebro helps (system prompt, review passes, GVS5H agents). Separate control.

- [x] The ledger (GVS5H orchestrate) is only reachable at Max effort: make it available on its own.
- [x] Effort "Custom": tick boxes for each kind of help (verification instruction, self-review,
      independent check, ledger, tools, memory) so the user picks exactly what they want.

## Copy
- [x] No "only for testing", no placeholder or dummy data anywhere.
- [ ] Core components explained in plain, formal language, not internal names.

## Done earlier today (to verify in the build)
- [x] Integrated GPU sized to free RAM (engine 1dabff3).
- [x] llama.cpp download progress on the Load button and the status pill.
- [x] e2e script: own port, orphan cleanup, process-tree kill, watchdog.

## Added 2026-10-24
- [x] Hub front page: the newest Qwen dense and MoE models that run here, found live on Hugging
      Face (so new releases appear the day they ship), with estimated tok/s on this hardware and a
      heavy-swapping warning; Flash-Next appears on machines that can run it. 1-2 bit files are no
      longer picked just because they fit the card.
- [x] Share over the internet: Cloudflare quick tunnel to phone access (pinned, checksummed
      cloudflared), one-time sign-in link plus password, link ends when sharing stops.
- [x] Short videos: animated clips from the image model (frames chained in latent space, optional
      end prompt), GIF with Download.
- [x] Native i-quants on the GPU and AVX2 i-quant kernels on the processor (27B IQ1_S: 0.6 -> 17 tok/s).
- [x] Empty answer after thinking: a reply that thought but wrote nothing answers again with
      thinking off (fallback; the owner's case is not reproduced yet, the log keeps the details).

## Still open
- [ ] Flash-Next: whole-GPU path (hyper-connections, shared experts, n-gram embeddings); 2.4 vs
      llama.cpp 7.0 tok/s, prompts read one token at a time (about a minute to the first word).
- [ ] Topic-aware expert cache: bookkeeping in engine topic.rs (tested); wiring into the VRAM/RAM
      caches and the Monitor, then measuring on 35B A3B and Flash-Next.
- [ ] Hybrid split copy path (no page-locked memory) gives wrong output: fixed by falling back;
      the copy path itself still needs the fix to get the faster prompt reading back.
- [ ] A real text-to-video model (Wan-class) in the engine.
- [ ] Models: drag to reorder and groups (loaded-first and pins are in).
- [x] Train: a recommended value and plain explanation next to every setting (with a Use button).
- [ ] Remaining speed losses: iGPU prompt attention, 8B whole-card prompt reading, low-bit decode.
