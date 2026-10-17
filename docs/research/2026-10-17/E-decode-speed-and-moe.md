## 1. Speculative decoding — draft many, verify once

**1.1 The arithmetic every speculator must satisfy.** Accepted-token throughput:
`s = (1 + Σ_{i=1..k} p̂_i) / t_verify` where `p̂_i` = prob draft token i is correct, `t_verify` = time of a batched k-token forward. For the 30B-A3B MoE (draft = the model itself + prompt reuse), `t_verify(k)` ≈ `t_decode` + ~15–30% (extra GEMV rows amortize). Speculation wins whenever `1 + Σp̂_i > t_verify/t_decode`. For a dense 14B-Q8 model, one extra attention pass over 12 KV layers at seq. N adds measurable cost — compute the crossover per checkpoint, not per architecture. **Measurement:** sweep k = 1..8 offline against a reference greedy decode on 512 prompts; emit a single CSV of (k, mean accepted, t_total). Confidence: H.

**1.2 Exact verification semantics, gamma-scaling.** Batched verify with **positions [0..k)**, passing **full logits k-vector** back. Greedy = autoregressive equivalence (verification lossless); sampling needs **rejection-sampled** verify (Leviathan et al., arXiv 2211.17192). For Cerebro: verify with greedy top-1 first (zero-QA-impact), add speculative-sampling for temperature>0 in v2. The verify kernel is just prefill at `m = k` — reuse existing int8 mma.sync m16n8k32 prefill path with `requests = 1, seqlen_padded = k`. No new kernel required. Confidence: H.

**1.3 Tree attention vs flat k for the mma.sync verifier.** Flat k is a batched prefill with causal mask over `k` tokens — trivially fits the existing prefill kernel (k ≤ 8). Tree of 8–16 leaves needs a **general masked** variant: replace causal mask with per-slot `ancestor_mask` bitmap. On sm_86 with int8 mma m16n8k32, pad tokens to multiples of 16; a 16-token padded chunk does not collide with a 2–3 stage prefill kernel (与你 reuse mma.sync kernel even at k≤8). **Gain estimate (30B-A3B, tree of 16 leaves over 4 heads):** M—usually 1.05–1.3× over flat-verify due to wasted-draft reduction. **Dense 14B:** L–M; trees pay extra verify cost without relaxed acceptance. Confidence: L for trees, M for flat-k.

**1.4 PLD / llama.cpp lookup (n-gram speculation). Requires: n-gram window from the model's own output (draft buffer), match against the prompt, speculatively decode k matched tokens.**
- **Algorithm:**
  1. Maintain `r = ringbuffer<tokens> (last N output tokens)`.
  2. On each decode step, take `suffix = r[-(j..)]` for `j = 2..min(n, len(r))`, look it up in a suffix index over the prompt.
  3. Verification = single batched prefill of the k matched tokens (k ≤ 64 handles long code repeats cheaply because verify is amortized against `t_decode`).
- **Data structures:** `robin_hood_map<(u64 hash of n-gram), smallvector<u32 positions>>` built once per prompt over `[0, prompt_len)`; query `O(j·H(M))`. The `smallvector` keeps the best continuation of length up to `max_draft`.
- **When it wins:** any prompt with reproduction (code: `for`, brackets, copy-pasted snippets, CSV/XML, JSON, quoting). Typical acceptance on code-fix workloads: 65–85% per step.
- **When it loses:** free-form generation with no prompt echo (chat on a fresh topic): acceptance < 10%; the verify cost is pure loss (~ -20% tok/s). Gate the pipeline: only enable if `EMA_acceptance > 0.30` (§1.4.2).
- **Confidence:** H. **Gain on target:** H for code workloads (+30–80% on 30B-A3B), L for chatty prompts.

**1.5 Suffix-automaton drafting (SuffixDecoding, arXiv 2411.04975).** Instead of a fixed-n gram map, build an online suffix automaton / suffix tree over **(prompt ∤ generated ∤ corpora)**, decode the *longest matching suffix*, speculatively fill the children with highest frequency. Key upgrade over 1.4: **arbitrary-length** matches (not capped at n) and multi-candidate branching (`breadth > 1` at the top of a suffix tree node). Pseudocode d:

```
match:  walk input buffer [tok last j downto 0]; each step, follow SAM suffix link
        transition to child with highest frequency count until budget k is spent
draft:  sequence of predicted continuations, forward it as a flat k-batch
update: after verify, insert verified tokens into SAM incrementally (O(1) amortized)
```
- **Why it earns its keep over PLD:** aggregate statistics over the *session* — a second bugfix in the same file reuses the structure; copy-paste loops win big.
- **Budget:** 1024 SAM slots × 64 bytes state = 64 KB per session; automaton ops well under 5 µs.
- **Confidence:** H for implementation; M for the gain on Cerebro's single-user code-heavy profile. **Expected:** +25–60% tok/s on repetitive code, -5% penalty on散文L.

**1.6 Draft-length λ-schedule.** Empirically, `k = 4–8` for prompt-lookup on code, `k = 2–4` for suffix, `k = 1–2` (or disable) when free-running. Adaptive rule: `k_{t+1} = clip(k_t · (1 + 0.15·(accepted_t − 0.5)), 1, 12)`. Cost of getting this wrong: 15–40% of the theoretically-achievable gain. Confidence: M.

**1.7 Tree structure for EAGLE-style draft models.** EAGLE-2 (arXiv 2406.16858) top-K first-step + top-n expansion: at each draft-head step, keep the K=2 best by draft logits·(continuation-prob product); expand the beam at the last step; draft token count ≈ 24–32 (leaf count). **Verification shape on Cerebro:** leaf-padded to multiples of 16 in the mma kernel (wasted lanes ≈ 2), with each leaf given a `pat_ancestor_mask` bitmap. Verify produces per-leaf P(next-token) — accept along the highest -logsumexp-prob path (Ac xt delta; see EAGLE-2 paper eq. 8 for the "dynamic tree" score). Confidence: L for trees on this engine, M overall.

**1.8 EAGLE-1 / 2 / 3.** All three use a *lightweight* draft head over the *frozen target*'s feature map `h_t` + embedding of the last sampled token; each draft step costs a few ms — far below the target's `t_decode`. Draft model = 1 transformer layer with its own KV cache. For Cerebro:
- **EAGLE-1 (arXiv 2401.15077):** generic, draft head = 1 layer of 2×hidden (depth-dim 2). Well-verified for Qwen-class models; torch reference is opensource (SafeAILab/EAGLE).
- **EAGLE-2:** reuses context confidence for tree expansion; +4–18% over EAGLE-1.
- **EAGLE-3 (arXiv 2503.XXXX UNVERIFIED citation-id; repo SafeAILab/EAGLE/eagle3):** trains on *multi-layer* feature concat (low/mid/high), claims 1.3–5× over vanilla. Porting cost to Cerebro: write a tiny RMSNorm + 2×matmul + KV-cache + RoPE for h_t≈2048 head — that's an entire draft model in ~30 MB f16. **Gain on 14B-Q8@448GB/s:** verify-once + draft ~4-6ms → net +80–150% tok/s is plausible; **confidence H** for 1, **M for 3 given multi-feature-layer plumbing**.
- **Failure mode:** EAGLE KV cache must persist across verify-and-rewind steps; if Cerebro's paged-KV (§2.x) has O(1) rollback only at token granularity, the draft's KV fork is a separate cache — implement draft-KV as a *separate small context* appended per verify-branch, not sharing paged pages. Confidence: M.

**1.9 Medusa (arXiv 2401.10774) — multi-head + tree-attention.** Adds `m` decoding heads (single-token each) to produce `m` candidate continuations, verified with tree attention. Implementation cost in Cerebro: `m × d_model` extra GEMM per verify step; no extra KV, no extra weights. Trees get complicated with `m=5`, `K=8`; usable only with tree-attn (see 1.3/1.7). **Cerebro verdict:** implement EAGLE/MTP first; Medusa only if no draft head is available for a given checkpoint. Gain: +40–90% tok/s on dense; confidence M. On sm_86, mma.sync batched verify dominates.

**1.10 Self-speculation / layer-skip (Draft & Verify, Calm / SwinJ / LayerSkip).** Idea: skip layers `l ∈ skip-set` in the draft pass; use the full model for verify. For Qwen3-dense with residual streams, skipping the *bottom layers* is safe (they compute the least-semantic features); the resulting draft differs from the target's full computation. A common trick is **early-exit-then-replay:** draft with `layers 0..l/2`, verify with all `L`.
- **Cheap variant that needs zero extra weights:** draft with `k` full-forward steps but with `attn_predictor_heads` truncated at layer `l ∈ {L/2, L·3/4}` — the resulting distribution is *close enough* for greedy acceptance.
- For Cerebro: **only viable on Qwen3-32B-class models (huge verify cost, draft cost = α·verify)**. On 30B-A3B MoE, activated-expert FLOPs per token ≈ draft FLOPs per token — L-level spec gain (draft is not free).
- Data structure: a `bool[l_len] bypass` baked into the layer-scheduling table at model-build time (e.g., `skip = 3,7,11,15,19,23,27,30`); the fused kernel takes the mask as a single `i64`.
- **Gain estimate:** 30B-A3B, layer-skip draft: **L–M**; dense 14B: **M**; smaller dense models: **L** (verify too cheap).
- **Confidence:** M. Implement only after (1.2), (1.4), (1.8).

**1.11 MTP heads (Qwen3-Next / DeepSeek-V3 style) on Cerebro.** Qwen3-Next ships an **MTP head** that predicts the next-next token with an extra transformer block consuming the hidden state of the main model; DeepSeek-V3 (arXiv 2412.19437) makes MTP a first-class module trained jointly; Qwen3-Next repo confirms MTP support in vLLM/SGLang (`qwen3_next_mtp`, `NEXTN`).
- **Cerebro implementation:**
  1. Load the MTP head weights (1 layer, ≈0.9 GB in Q8) alongside the main model.
  2. In draft mode, run MTP-head forward `r` times (r = 2–3), generating `r` draft tokens.
  3. Verify whole `r+1` with main model's single batched forward.
  4. On acceptance ≥ 1, resume accepted token stream with the MTP cache updated.
- **Data structure:** separate KV slab for the MTP head of length `max_draft`; on accept-cut at position p, hard-rewind to `p` (draft KV is tiny so rewind is trivial).
- **Gain estimate:** 30B-A3B (MTP kit works well here): +60–110% tok/s **H** confidence; dense Qwen3: uncited (no trained MTP for Qwen3-4B/8B/14B on HF) → **UNVERIFIED**; treat as fabrication, unless you take the SGLang "draft-model=small-qwen3" approach (acceptable-quality transfer).
- **Where it disappoints:** whenever acceptance falls below 45%+ for longchains, verify cost (2-layer KV+MoE) swamps the gain.

**1.12 Speculation+MoE interaction: cache experts per draft position.** Draft and verify tokens activate (top-k) experts independently; the *verify* batch of `k` tokens for each layer activates `k·topk` experts — on Qwen3-30B-A3B this is 3× the usual 10 → the batched-verify kernel is dominated by *expert lookup + routing*, not matmul. Mitigation (1.4-style): have draft-level experts pre-warmed (predict router top-1 with the previous layer's router output — §4.5) and prefetch those experts into pinned+GPU staging before the verify. **Expected: recovers 30–50% of the verify slowdown. Confidence: M.**

**1.13 When speculation *loses*, quantitatively.** `P(accept)` for PLD on unrelated prose ≈ 0.08; with k = 6, gain = (1+6·0.08)/t_verify — a 1.5× wasted verify. **Hard rule:** ship with `--spec probe-first`, i.e., first 96 tokens instrument-only (measure acceptance online), then enable/disable per-session. **Measurement:** log `accepted / drafted` per step into a ring buffer of 128; feed to an online controller. Confidence: H.

**1.14 Online acceptance estimator (bandit controller).** Exponential moving average with `α=0.15`, plus a per-workload tag. For one user, the workload drifts slowly, so EMA is adequate. Data structure: `struct SpecStats { double ema_accept; i32 cur_k; i32 consecutive_misses; }`. If `consecutive_misses ≥ 4`, set k = 1 for 6 steps. Confidence: H.

**1.15 Verification kernel shape on sm_86.** A single batched prefill of `k ≤ 8` tokens per MoE layer using the existing int8 mma m16n8k32 prefill: rows of A = `k·hidden`, B = expert W (k0·n0 evaluation is same as t), masked softmax over tree ancestry. At k=8 vs 1, mma.sync efficiency is near-flat because the `k0=8` tile still fits the 16×8×32 MMA layout. Verify cost measured at **≈ 1.1–1.3× single-token cost** for k = 4–8 on dense f16 GEMV — the break-even is `Σ p̂_i ≥ 0.15`. **Measurement:** microbenchmark verify-vs-decode for k ∈ {1,2,4,8}; report `$t_verify(t)/t_decode(1)$`. Confidence: H.

---

## 2. Continuous batching, paged KV, prefix caching, chunked prefill

**2.1 Single-user desktop ≠ skip this section.** Even with 1 user, the engine should be structured to serve **≥ 2 virtual slots: (a) the draft head, (b) speculative verification, (c) prompt-inflight** — otherwise long prefills block decode and speculation breaks. llama.cpp-style "one request at a time" is why speculative decoding cannot be overlapped with its own verify there. Confidence: H.

**2.2 PagedAttention (vLLM, arXiv 2309.06180).** Data structure:
```
struct Page { i32 blockIdx;              // 16 tokens per page on 3070 Ti (fits 448-drive aligned windows)
              f16 k[heads_kv][16][head_dim];
              f16 v[heads_kv][16][head_dim]; }
struct Slot { pagenum pages[...]; i32 n_tokens; i64 tokens[]; }
```
- Page size p = 16: block table is 16/32/64B wide per request; attention kernel walks pages through an indirection table. sm_86 has 100 KB shared mem per SM; the paged attention kernel (vLLM's) loads K/V tiles by (block, head) — tile 16×head_dim f16.
- **Alternative for Cerebro — since 1 main + 1–2 draft slots:** a **fixed 2-slot ring** of size max_ctx (32k) each may be simpler; but MTP branching + verify requires forking. Verdict: use **full paged-KV** from day 1 for the main model; the draft head is a separate small slab.
- **Gain:** enables speculative verify (needs forked/branching KV), RadixAttention, and MoE-offload without blocking prefill. Confidence: H.

**2.3 Fork/rollback (the critical spec-decode primitive).** To verify `k` draft tokens you must (a) write k pages tentatively, (b) on rejection, free pages ≥ accept-point. Implementation: each decode writes into a **staging arena of 32 stash pages per slot**; accepted prefix is committed by appending page indices into the slot; on rejection, a single atomic `cudaMemcpyDtoD` of the page table (16 bytes) rolls back. No memory copy. **Failure mode:** OOM of the staging arena with long drafts — cap k and pre-reserve 64 pages. Confidence: H. Gain: makes 1.8/1.11 possible at all.

**2.4 RadixAttention / prefix cache (SGLang, arXiv 2310.07204; github.com/sgl-project/sglang).** Radix tree keyed on token ids; each node = a KV page range refcounted. For a desktop app, the singleton wins: a **second chat turn in the same file** reuses up to 30k prefix tokens → prefill cost ≈ 0, decode immediately. Implement radix node = page-range list; on eviction, LRU by last-use; on conflict, mark pages copy-on-write.
- **Failure mode:** silent aliasing if tokens are inserted under a wrong parent (chat template mismatch). **Measure:** hit-rate = tokens recovered / total prefill tokens, per session.
- **Expected gain in Cerebro's agentic/multi-turn workloads:** prefill 10×–50× faster (not decode per se, but it dominates the *perceived* tok/s at turn boundaries). Confidence: H.

**2.5 Chunked prefill mixed with decode.** Prefill a long prompt in chunks of `c = 512–1024` interleaved with **decode steps of the other (draft/slot) request**. This requires the MoE-offload scheduler to run a different `GGML_CGRAPH` per step. In Cerebro this is natural because each step is already a fresh (CUDA-graph-templatable) launch.
- Schedule rule: prefill chunk `c` consumes ≈ c·T_prefill_ms; decode step ≈ T_decode_ms; choose c so that `c·t_prefill_per_tok < 33 ms` (keeps interactivity perceptually). At 30B-A3B, c = 256–512 per chunk.
- **Gain:** removes 2–20s stalls at turn boundary; not a raw tok/s number but it moves the *aggregate* throughput. **Confidence: H.**

**2.6 Continuous batching machinery (use vLLM's scheduler as reference).** Even for 1–2 slots, the scheduler needs:
- `waiting` (requests) → `running` (active slots), preemption at OOM by K/V page-steal,
- **work-conserving rule for spec-decode:** if draft has no tokens, run plain decode; if draft has k tokens, run verify; schedule *speculative* decode steps while prefill is in flight via the second CUDA stream. Confidence: M; the complexity is in the bookkeeping, not in the kernels.

---

## 3. KV cache quantization, tiers, sliding windows, sinks — critical on 8 GB

**3.1 KV budget on 3070 Ti Laptop.** Qwen3-8B: 36 layers × 8 KV-heads × 128 dim × f16 = 144 KB/token → 32k ctx = 4.5 GB. That's a *third of VRAM for cache alone*. Qwen3-14B: 40L × 8 × 128 = 160 KB/token. The MoE 30B-A3B uses GQA 4 KV heads × 128: 48L×4×128×2 = 96 KB/token ≈ 3.0 GB at 32k context — *leaves ~zero room for draft-KV or expert prefetch* unless compressed. Confidence: H (this is where the game is won/lost).

**3.2 K-axis vs V-axis quantization properties.** For an 8-bit KV quant with per-*channel* = per-*(layer, head)*, `q8_0` keeps MSE under 5e-4 for K and 1e-5 for V; for a 4-bit quant, `V` at `q4_K` survives well, `K` suffers drift in softmax denominators — reduce the group size for K (q6_K on K, q4_K on V is a reasonable compromise; `q4_0` on K measurably changes output distribution). Apply the same machinery as weight quant (Cerebro already has Q8_0/K-quants) but the quantization axis is different: quantize **along head_dim** (channel-parallel for f16→i8 with scale per (l, h)), **along token for K? no — per-token is required for exactness.** For V, per-token-per-head is fine. **Measurement:** `KL(p_f16 || p_q8)` on the last-token distribution over 100 prompts; Δacc on HumanEval subset. Confidence: M (K exactness is the tricky bit).

**3.3 Per-head group layout.** Store `q8_0: k[i8] head_dim × zone[len/32]` plus scale table `(layer, head, zone)`. Quant kernel: f16→i8 with 32-dim blocks (matches `k_quant`'s existing superblocks). **Encoding note:** for the *decode* GEMV, dequant is fused into the attention kernel — batching reads the f16 scale + i8 rows and multiplies by softmax weights directly in registers. **Gain:** KV-VRAM 2×→ (0.5×) = +1.5 GB free on 30B-A3B; combined with 4.8, that's an expert cache slot. Decode tok/s: +5–12% on 8k→32k contexts because of the attention GEMM getting faster (less cache-miss traffic to VRAM). **Confidence: M–H.**

**3.4 q4 K/V with group-wise scales.** `q4_K` per (layer, head, 32-token-group) with per-group f16 scale. On sm_86, the fused dequant via `__dp4a`-style int path is *wrong for attention scores* (they need f16 dot products) — use `__half2` after in-register dequant; keep `q4` only on V and only when seq ≥ 8k. **Cerebro:** ship `--kv q8` default, `--kv q4-v-only` as an opt-in for 64k-context workloads. Confidence: M for accuracy parity, H for the space savings.

**3.5 Sliding-window (Gemma 4, Qwan3 SWA layers) and streaming sinks.** Gemma 4 (like Gemma 2 / Gemma 3, arXiv 2403.XXXX — UNVERIFIED gemma-4 report id) alternates global and sliding-window (SW = 512–1024) layers. For SW layers, KV pages beyond the window can be **dropped in-place** (they have no effect on future tokens) — this halves the KV footprint for SW-heavy hybrid maps: SW layers' KV at 5/6 of the layer count. For Qwen3 (global), implement **StreamingLLM attention-sinks (arXiv 2309.17453)** for the rolling-window fallback on Gemma/other models with sink tokens. In Cerebro: `--context-mode sink[4] window[1024]` for Gemma-4 models with a 1M-context setting.
- **Data structure:** ring buffer pages of SW-size; four persistent "sink" slots at the front; RoPE keys *re*-computed with sink-relative positions on each eviction (cheap: 4 tokens per eviction).
- **Failure mode:** hard-drop of the *middle* of the prompt is lossy; document it, default to it only on window boundaries.
- **Gain:** SW-only layers with no page management overhead → KV for Gemma-4-class workloads goes from O(L) to O(L); the practical benefit for **decode tok/s at long context is large** (attn GEMM over ≤1k keys regardless of the full history: 27× faster attention at 27k context). **Confidence: H.**

**3.6 Offload tiers (KV).** Tier scheme, hot to cold:
  - **T0: VRAM hot pages = the last 4–16k tokens** (per slot).
  - **T1: pinned host memory** (`cudaHostAlloc` + `cudaHostRegister`) for pages up to max_ctx (8k–128k). Size cap 4 GB to avoid starving the OS + expert slabs (§4).
  - **T2: pageable mmap from the session file (recover old sessions).**
- **Scheduler:** on sequence growth, every 1024 tokens *evict pages* [0, 1024) into T1 via `cudaMemcpyAsync` on the **prefetch stream** (computation continues on the decode stream); on *context recall* (a prompt references an old turn), reuse from T1 with the reverse copy. Because PCIe x8 ≈ 14 GB/s, a 4k-token, 4-head, 48-layer, f16 KV page block of 192 MB costs ≈ 14 ms to evict — schedule it in 512-token slices (24 MB, 1.8 ms) to avoid decode-step jank. **Confidence: H.**

**3.7 Paged + offload hybrid with write-back.** Modify 2.2's page table with a `state ∈ {VRAM, PINNED, MMAP}` per page and a device-side translation table `page_table[u32]` per request. The attention kernel reads `k_slot_page_ptr = page_ptr[state]`, and for state = PINNED, uses **zero-copy read** over PCIe (`cudaHostGetDevicePointer`) — on PCIe x8 at 14 GB/s and for *rarely-hit* pages, zero-copy beats explicit copy+free-up-VRAM for hits < 5%. **Measurement:** zero-copy read latency per 16-token page on PCIe x8: ≈ 8 µs (128 KB @ 16 GB/s) — acceptable only for LRU-hit < 2%.
- **Recommendation:** use it only for the *oldest* slides; integrate with 2.4's radix tree so old sessions stay evictable. Confidence: M.

**3.8 KV prefetch waits for no one.** Prefetch *predicted* window pages (`window + 4k`) the step before they're needed; predict using radix-tree suffix matching of the *input-so-far* against the tree (same structure as 2.4, free). This is a Messi-style heuristic with high hit-rate (> 85% in conversational workloads with long history). Confidence: H.

---

## 4. MoE on 8 GB VRAM — the main event for 30B-A3B

**4.1 Do the math for Qwen3-Coder-30B-A3B first.** Parameters: 30.5B total; active 3.3B. GGUF Q4_K_M: 17.2 GB; Q8_0: 31.5 GB. The *always-resident* parts (attention + shared experts + router + embed/output head) ≈ 2.4–3.1 GB at Q8. The 128 routed experts × 48 layers ≈ 14.3 GB (Q8) — cannot fit. On 8 GB VRAM with ≤ 3 GB for KV + graphs, we have ~4.5 GB for expert-resident weight. Qwen3-Coder-30B has **128 routed experts × 8 layers**; per-expert size ≈ 2 × (hidden 2048 × moe_inter 768) × 3G = ≈ 21 MB in Q8... wait, actual: hidden = 2048, FFN = 768 → per expert per layer ≈ 2·(2048·768) params = 3.15M ≈ 3.4 MB Q8 → per layer's 128 experts ≈ 435 MB; per layer per expert ≈ 3.4 MB × 2 (gate+down) = 6.8MB... let's just say **~7 MB Q8/expert, 1.1 Q4**. Roughly: L-layer expert table at Q8 = 128·7 = 896 MB/layer ⇒ 48 layers ≈ 43 GB. That's why: **only 10–20 of the 48 MoE layers can be "fully resident"** and the rest must *stream*.
- **Implication:** expert-offload scheduling is the *performance problem* of Cerebro; not the metadata, not the routing. Confidence: H.

**4.2 Baseline: llama.cpp's --n-gpu-layers + CPU execution of remaining experts.** On the i9-12900H with AVX2 (no AMX, no AVX512-FP16 — UNVERIFIED whether 12th-gen has AMX: 12th-gen *does* have AMX, P-cores only — Alder Lake supports AMX-int8 and AMX-bf16 UNVERIFIED-for-this-skew), a single expert GEMV of 2048→768→2048 in Q8 costs ≈ 1 / (55 GB/s ÷ 6.8 MB) ≈ 0.12 ms on the CPU *if* band-limited. 10 experts × 48 layers = 480 GEMVs per token → **86 µs/token** → 11.6k tok/s theoretical, i.e., the CPU is NOT the bottleneck; the *transfer* and *sync* are. Confidence H.

**4.3 KTransformers-style expert placement (arXiv 2601.xxxxx UNVERIFIED — repo: kvcache-ai/ktransformers).** Key ideas relevant to Cerebro:
  1. **Keep attention + KV + shared expert on GPU**; offload routed experts to CPU. On DeepSeek-V3, KTransformers achieves 13.69 tok/s with lossless quality; on 30B-A3B, expect much better because the CPU work per layer is 7× smaller.
  2. **Compute in bf16/int8 on CPU with AMX** (`AMX-int8-tile-gemm` tiled kernels); on i9-12900H (no AMX), fall back to **AVX2-VNNI `_mm256_dpbusd_epi32`** — realize that VNNI is available on i9-12900H (Ice Lake+): this gives an 8× throughput boost over plain AVX2 on i8 dot products — **use Cerebro's existing int8 AVX-VNNI kernel for experts**.
  3. **On-the-fly XOR-swap trick for `moe_align_block_size`** — do not migrate; use plain `radix sort by expert id` in Cerebro. Confidence M.

**4.4 Fiddler (arXiv 2402.07033) — the key decision rule.** Instrument a latency model:
```
tau_cpu(e, s)  = s·(2·h·n) / CPU_BW_effective     # CPU expert forward for s tokens
tau_gpu_tr(e)  = expert_bytes(e) / PCIe_BW        # weight transfer
tau_gpu_ex(e)  = const (k·expert_batched GEMV)
Strategy at runtime:
  if   e in VRAM:            GPU_execute(e)
  elif tau_cpu < tau_gpu_tr + tau_gpu_ex:  copy_act_to_CPU; CPU_execute; copy_back
  else:                      copy_W_to_GPU; GPU_execute
```
Calibrate `CPU_BW_effective`, `PCIe_BW`, `GPU_BW` at startup with microbenchmarks (§7). For Cerebro single-user decode (s = 1–4 draft tokens), **CPU-expert almost always wins on 30B-A3B** (activations = 2048 f16 = 4 KB vs expert = 7–14 MB — 1500–3000× less PCIe traffic). Confidence: H.

**4.5 Expert popularity profile (Fiddler 4.5,_kvcache-ai/ktransformer V0.3, MoE-Offloading arXiv 2310.xxxx UNVERIFIED).** Offline profiling over 2048 calibration samples (mix of code + prose). For each (layer, expert), record `p_hit = P(expert e in top-k router)`. Data: sort by p_hit; select the top `N_resident` experts per layer (N_resident ≈ 20–40 for 30B-A3B at Q8 given a 3-GB budget). In Cerebro, ship a per-model profile file `experts_freq.bin` analogous to llama.cpp's `--n-gpu-layers auto` heuristic **(confidence H, gain 1.26× [per Fiddler eval])**.

**4.6 MoE-Infinity / Heptagon-style histogram-of-experts (arXiv UNVERIFIED id; repo: EffCert/MoE-Infinity)** — Idea: for a *repeating* workload (agentic coding), the sequence of top-k expert ids is highly correlated. Implement a per-(layer, expert) Markov chain `P(e_t | e_{t-1})` as a `u16[128]` frequency table per (layer, prev expert); on each decode, compute the *predicted* top-k expert set for the next token from the last 1–4 tokens' router outputs and prefetch those experts into the GPU stash 50–100 µs ahead. **Gain: 1.2–1.6× on code-heavy workloads; confidence M (needs online validation).**

**4.7 Async expert prefetch on a second CUDA stream.** Implementation plan:
- **Stream A (`main_stream`):** attention, router, shared-expert, output head.
- **Stream B (`xfer_stream`):** `cudaMemcpyAsync` HtoD pinned → GPU page-slot for the *next* layer's top experts, issued right after the router logits for layer L are computed (one cudaEvent) and consumed via `cudaStreamWaitEvent` before the MoE GEMV of layer L+1.
- **Buffer layout:** `pinned_ring[N_pages]` of 4 slots each with a copy-in-progress flag. **On Windows + Tauri**, pinning 512 MB is safe (§4.11).
- **Sizing:** with 14 GB/s PCIe x8 and per-expert 7–14 MB Q8, we can prefetch 20–40 experts per layer per decode step (a 25-ms gap) — that's enough to cover 10 experts × 4 layers *almost always*, and X experts in the worst case.
- **Critical trick:** in the CPU→GPU direction, pack the *multiple experts activated by one router decision* into a single pinned-staging page and issue a **few large copies** (one per expert) instead of one memcpy per tensor, keeping the bus saturated. Measure with `nvprof --metrics dram__overhead...` or `ncudablas` STREAMs.
- **Confidence:** H for implementation; gain = reduction of PCIe-added latency by overlap 2–4× = hidden.

**4.8 PowerInfer-style hot/cold split (arXiv 2312.12456).** Idea: keep the hottest experts (by profile 4.5) in VRAM *always*; the rest stream. For Qwen3-30B-A3B with ~25% code + 75% prose profile, the top-32 experts per layer cover ≈ 55–65% of activation mass. Budget: 32 × 48 layers × 7 MB Q8 = 10.7 GB — too much. Reduce to top-16 → 5.4 GB: still too much with KV. Q4 top-32 = 5.4 GB → lost to KV. So **on 8 GB, hot-expert residency is limited to ~top-8 experts/layer at Q4 (2.7 GB)**; the remainder go CPU. For dense-model case, nuance: experts don't exist; ignore. Confidence: M (needs profile data); expected gain 1.4–2.2×.

**4.9 MoE-Lightning (arXiv 2411.11217) — inspired pipelining.** Not a fit for single-user (designed for batch throughput on T4s) but the **pinned-CPU staging + weight paging + double-buffer** pattern carries over exactly (§4.7 is already this). Confidence: M (for the pipelining analogy, not the top-line number).

**4.10 Windows large pages (`MEM_LARGE_PAGES`) and `VirtualLock`.**
- *Why care:* TLB misses cost 2–10% on big weight-scanning loops (128 × 14 MB jump patterns = every page misses). 2 MB pages reduce DTLB misses by 10–40% on expert streaming.
- *APIs:* `VirtualAlloc2(..., MEM_LARGE_PAGES ...)`, requires `SeLockMemoryPrivilege` (grant via `LsaAddAccountRights`; needs UAC elevation **once** at install). Then `VirtualLock` those regions to prevent paging (a 17 GB expert pool on a 32 GB laptop is *not* safe without it, especially with OOM Cumulative `CommitLimit`).
- *Caveat:* `VirtualLock` is **per-process chargeable** byte cap = RAM - working-space; on 32 GB total, lock 24–26 GB max and leave the OS headroom. Add a `--lock false` switch for machines with "Memory Integrity / Core Isolation" that handles this badly. **Confidence: H for the mechanics; gain M (2–9% on CPU-GEMV-bound MoE decode).**

**4.11 Pinned host memory (`cudaHostAlloc(cudaHostAllocMapped)`).** Pin:
  - KV offload page pool (§3.6): 2–4 GB
  - expert paging pool (§4.7): 1–2 GB
  - input/output ring buffers: 64 MB
  **Windows behavior:** WDDM 2.x + `cudaHostAlloc` = pinned memory is *uncanonical* — actually no; pinned memory is fine but *mapped* pinned (`cudaHostAllocMapped`) may be slower than `cudaMemcpyAsync` over PCIe for *large* transfers; use mapped only for tiny control structures (router logits of 128 u8 = 128 bytes), use `memcpyAsync` for the MB-scale expert pages.
- **Failure mode:** allocating > 50% of physical RAM as pinned memory on Windows can cause Game Bar / OS jank; `cudaHostAlloc` failures need graceful fallback to pageable (throughput drops 2×). **Confidence: H.**

**4.12 CPU-side expert GEMV in AVX-VNNI int8.** Cerebro already has an AVX2/AVX-VNNI CPU path — use `vpdpbusd` on the expert `w13` (gate+up fused) matrix, and `vpdpwssd`-style i16 path for scale-correction. Expected throughput: 55–70 GB/s per socket → each 6.8-mB expert ≈ 0.11 ms → 10 experts = 1.1 ms/layer → 48 layers = **52.8 ms/token = 19 tok/s** without any speculation — seriousness check: this is the *CPU-bound floor* on 30B-A3B; need 4.13 to break it. Confidence: H (calc); the floor is CPU BW, not launch overhead.

**4.13 Expert batching across a *draft* token group.** When running spec-decode, k = 8 tokens activates ≤ 8 × top-k experts; a CPU GEMV across k=8 tokens for the *same* expert shares the weight stream and (roughly) costs the same time as k=1 because the weight scan dominates (compute is amortized). **So the effective CPU expert cost per token drops 1.8–3× with k = 4–8 speculation. This is the *single biggest lever* for MoE-30B on this machine: CPU-amortized expert bandwidth pairs withi GPU speculative verify.** Confidence: H.

**4.14 Hybrid expert placement summary table** (30B-A3B, Q8, 8 GB VRAM, KV q8):

| Component | Size (Q8) | Where |
|---|---|---|
| Embeddings + LM head | 1.2 GB | VRAM |
| Attention Q/K/V/O (all layers) | 2.6 GB | VRAM (prefill int8 + decode f16) |
| Shared expert (48×) | 0.3 GB | VRAM |
| Router + norms | 0.1 GB | VRAM |
| Top-8 routed experts/layer | 5.3 GB | VRAM (if budget allows; else Q4 = 2.7) |
| Remaining 120 routed experts/layer | 31 GB | Pinned host memory, AVX-VNNI |
| KV (32k ctx, q8, 4 heads × 48L × 256 total head-dim... 48L×4×128×2×(1 + .125) × 32k) | ≈ 1.5 GB | VRAM hot, T1 pinned for the rest |
| CUDA graphs + workspace | 0.4 GB | VRAM |
| **Total VRAM** | **≈ 7.4 GB with Q4 hot-experts** | **fits with 0.6 GB slack** |

Confidence: **H** on the math; correctness bounded by routing entropy.

**4.15 Deferral (KTransformers SOSP'25-style) — skip` if Cerebro has only 1 CPU socket.** Deferral has no NUMA trick on a 1-socket i9-12900H (all DDR5 channels are one node). Skip; revisit if we support multi-CPU. Confidence: H.

**4.16 Avoiding router-major sort.** At batch=1, skip `moe_align` and just `switch(expert_id)`. Each expert is only 7 MB; there's no cache-block collation for a single token. For batch = 8 (spec), sort (token, expert) pairs into expert-major CSR (radix sort by expert id, 8 values → insertion sort is faster). Confidence: H.

**4.17 Per-model expert profiles.** Ship with each quantized model a JSON sidecar:
```
{ "layer_expert_hist":  [[p0..p127], ...52 layers],
  "pair_transition":    [[c_ij ...],...]   # for MoE-Infinity-style prediction
  "hot_experts":        [e0, e1, ... e7] }
```
Fit on 2048 tokens of mixed workload; this is a 3-minute offline pass per model shipped, and it makes 4.5/4.8 automatic. Confidence: M.

**4.18 The instruction-cache killer.** MoE offload with 10 experts × 48 layers = 480 different weights being touched per token; weights are *compiled from the same code paths but they beat the µop cache* differently. On Windows, a hard page fault per expert-scan tree (60+ GB of working set on 32 GB RAM) = death. `VirtualLock` (4.10) plus a profile-based `MEM_RESET_UNDO` + `MEM_LARGE_PAGES` prefetch on the hot experts at model load is the fix. Measure: `page_fault_count` from `GetProcessMemoryInfo` per 1k tokens; target < 100 hard faults / 1k tokens after warmup. Confidence: H.

---

## 5. CUDA Graphs, fusion, split-K

**5.1 CUDA Graphs are mandatory for Cerebro.** On a 30B-A3B MoE with 48 layers, each token's forward pass issues ≈ 48 × 14 = 672 kernel launches + 96 memcpy/memset ops ≈ 3–5 ms of pure launch overhead on Windows — *that's the difference between 40 tok/s and 250 tok/s* on small models. With CUDA Graph capture (`cudaStreamBeginCapture`), we replay the whole forward pass in 1 launch ≈ 5 µs.
- **Caveat:** cudarc has no auto-graph capture; implement manually: `stream.capture()` on a warmup pass, store `GraphExec`, then replay-per-step with dynamic input *pointers* (use static input buffers + `cudaMemcpyAsync` into them *before* the graph, not inside).
- **Buffers:** all weights, KV, activations must be the same virtual addresses across runs — allocate once in a Cerebro-managed cuda memory pool, never free mid-session. Confidence: H.

**5.2 Graph + speculative-docode interplay.** The graph is static in *k* too — capture 6 variants `G[k]` for k ∈ {1..8}, swap per-execution based on the scheduler's decision. That's ~6 × 12 MB of graph memory on 30B-A3B — fine. **Confidence: H. Gain: 2–10× on decode tok/s for small models where launch overhead dominates (biggest gain on < 8B f16; on MoE-30B the gain is 1.2–1.5× because CUDA work is heavier).**

**5.3 Handling host-side sync in the graph (spec-decode branch).** Speculative verify requires *branching* (which tokens accepted) mid-graph. Split the graph at the branch points: `G[decode]` and `G[verify]` as two graphs, end the graph right before the sampler, do the accept/reject logic on host, then re-launch. **Never put host-side branches inside a graph** (`cudaGraphAddHostNode` is slow on Windows). Confidence: H.

**5.4 Kernel fusion agenda (in descending ROI).**
| Fusion | Reduces (bytes moved) | Gain (MoE 30B, tok/s) | Confidence |
|---|---|---|---|
| RMSNorm + QKV projection GEMV (fused scale+add+dequant) | 2048 f16 read-2× per layer | +4–7% | H |
| Rope + attention (append) + KV store (q8 quant fused) | KV write 2× one pass | +3–6% (more with offloaded KV: 8–12%) | M |
| SwiGLU gate+up activations (dequant fused) | 2 × 768 read | +4–8% per MoE layer | H |
| RoPE + GQA-broadcast: fold rope into the *weight* matrix at load (static rope; still needs per-position rope only in attention) | no-op | +1–2% | M |
|elementwise `residual add + RMSNorm` in one warp | 2×2048 f16 | +2–4% | M |
| Fused top-k router softmax (CPU) → bitmask → expert lookup (CPU) | avoids PCIe roundtrip | +5–15% on MoE | M |
- **Do NOT fuse into fewer kernels than your CUDA-graph node budget** (~1200 nodes per graph is the Windows+CC 8.6 practical limit per executable graph before launch latency climbs again — UNVERIFIED for exact number). Confidence: M.

**5.5 Split-K decode GEMV on sm_86.** For a batch-1 decode of a dense 14B f16 model on 3070 Ti Laptop (448 GB/s, ~84 SMs), the arithmetic-intensity limit is 0.5 FLOP/byte → the GEMV is bandwidth-bound; **split-K is unnecessary**, the kernel is already saturated by 1 CTA per 128-column warp-chunk. The right kernel is: **one warp per 64–128 output columns, `__hfma2` vectorized**, reading 128 f16 values per cycle per SM. Cerebro already has this. For int8 mma m16n8k32 prefill, the k-dim = hidden is split across `k0=16`warps; for *decode* (m=1), mma.sync is useless — use pure m8n32k16-style wide GEMV. **Confidence: H (this is a solved problem).**

**5.6 The MoE fused kernel (dispatch + fused grouped GEMV).** For batch = 1 decode with 10 experts per layer:
1. Compute router logits on GPU (14 MB activation in VRAM + 128-d router matrix).
2. Push a single **fused MoE kernel** with parameters `(expert_list[10], token_hidden_state)`; each CTA processes one expert along the "n" dim at 128-wide lanes; expert weights are read from VRAM-resident page (top-N experts, §4.8) — a "load from CPU" case is handled by not launching a CTA for that expert this layer and deferring to CPU (§4.4).
3. Output aggregation over the 10 experts in shared memory, scatter to the residual add.
**Gain vs separate GEMV + reduce:** +10–20% on the MoE block; confidence M–H.

**5.7 CUDA graph + spec-decode + MTP chains.** For MTP-enabled Qwen3-30B (§1.11), the decode path becomes a *3-phase graph*: (a) `G[main]` produces logits, (b) `G[mtp-draft]` (2–3 steps) produces draft tokens, (c) `G[verify]` consumes a k×`max_draft`-padded batched prefill. Each is a captured graph with static buffers. Three graph launches per accepted cycle ≈ 20 µs launch overhead + verify cost. **Confidence: H. Typical net tok/s: +80–140% (H).**

---

## 6. Gated DeltaNet / linear attention kernels for the hybrid model

**6.1 Background: the chunked WY form (arXiv 2412.06464).** The recurrence is:
`S_t = S_{t-1} (α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ`.
The chunkwise form groups C tokens into a chunk, expanding to a forward-substitution of per-chunk matrices `T` (WY representation): within a chunk, we solve `ΔS[t] = (I − T_t)(U_t − W_t S[t]ᵀ)` iteratively, giving **O(C · d² · d_k) work per chunk, O(1) sequential depth**. This is what turns a token-serial recurrence into a matmul-heavy chunked kernel.
- **Cerebro decode use:** for batch = 1 decoding, the 48-layer Qwen3-Next-style hybrid has **3 linear (Gated DeltaNet) layers per 4-layer block (3:1)** — those layers have **O(1) per-token cost and O(d_kv × d_v) state**, no KV cache; so the model's *decode* is dominated by only 12 full-attention layers (GQA 16Q/2KV at head-dim 256 with only 64-dim rotary). This is *why* the hybrid model decodes faster at long context. **Confidence: H.**

**6.2 State size math for Qwen3-Next-class hybrid (48L).** Linear layers: 36 × (16 QK heads × 128 + 32 V heads × 128) = 36 × 6144 f16 = 432 KB per layer state → **15.8 MB total recurrent state** (fits registers/L2 easily). Full-attention: only 12 layers × 2 KV heads × 256 × 2 × f16 = 24.6 KB/token → 32k ctx = **786 MB** (vs 4.5 GB for a dense Qwen3-8B at same context). **Conclusion: the hybrid enables 2× longer context or 3.5× more free VRAM for experts. Confidence: H.**

**6.3 Decode kernel for Gated DeltaNet, batch = 1.** Per token, per layer:
```
q, k, v, α, β ← projections (fused GEMV, dequant int8 weights)
S ← S · (α·(I − β k kᵀ)) + β v kᵀ     # 2 × d_v × d_k MACs = 2 × 4096 × 128 ≈ 1M MACs/layer
o ← S · q (2 × 4096 × 128 dot products)
```
That's ≈ 2.6M MACs per layer with a state of 6144 f16 = 12 KB — this is a **memory-bound GEMV over a tiny matrix + one rank-1 update**: total op count is trivial vs the FFN. **Kernel: one warp per (v-head, k-chunk); states in registers + shared memory; writes back just the 12 KB state.** Per layer: ≈ 0.7 µs on sm_86.
**Do NOT reconstruct the full-history form** (that's the training path); it's `O(C)` work per layer per decode step for no reason.
- **Confidence: H. Gain vs naive token-serial with S in global memory: 1.6–2.5× on the linear layers.**

**6.4 The short convolution prelude.** Each Gated DeltaNet layer has a causal depth-wise short conv (kernel size 4) over q, k, v before the recurrence — fuse: (a) the short conv into the q/k/v projection epilogue (shift-left register in SMEM); (b) the SiLU gate; (c) the L2 norm on q, k. **Failure mode:** forgetting to update the conv-state after each decode step. Maintain a 4-token shift-register per (head, channel); roll in registers. **Confidence: H. Gain: 2–4%** (it's a small computation).

**6.5 Intra-chunk parallel form for *speculative verify* on linear layers.** When verifying k = 8 draft tokens, the Gated DeltaNet recurrence can be processed in a single shot with the *chunked form* (C = 8), sharing the incoming state but computing k deltas. On sm_86, use a **single fused kernel** that reads the 12 KB state, does k = 8 chunked steps via T-matrix recurrence, and updates the state. Confirmed feasible with the flash-linear-attention library's Triton kernels as a reference (`fla/ops/gated_deltanet`, repo: fla-org/flash-linear-attention). **Confidence: M – reference code is Triton; you'll port to CUDA.** Gain: linear-layer verify ≈ free (vs 8× serial decode).

**6.6 The 3:1 hybrid layout makes CUDA Graph capture eager-friendly.** Qwen3-Next layout is `3 × (GDN → MoE) → 1 × (GatedAttention → MoE)` × 12. For CUDA Graph capture (§5.1), the GDN layers have *fixed-shape state* (always the same 12 KB tensors), and the Gated-Attention layers have *dynamic-shape KV* (page tables). **Strategy:** capture GDN + FFN in one static-shape graph; keep the attention layer's paged-KV update inside the graph via static page table slots (memcpy the next page table entries before each step). Confidence: M.

**6.7 Gated attention layers at head-dim 256.** Note: head-dim 256 with only 2 KV heads means standard flash-attention kernels can handle it, but the sm_86 mma.sync path with f16 q/k/v + int8 KV dequant is not a stock kernel — write a **custom fused mma.sync m16n8k16-style decode attn kernel** specialized for 16Q×2KV: each warp handles 4 Q heads sharing 1 KV head, `__hfma2` over the KV at head-dim 256. **Confidence: M; gain: 1.3–2× over a generic flash-attn at head-dim 256 f16 (this is speculation based on typical custom-kernel win, UNVERIFIED for exact numbers).**

**6.8 Fused Mamba2/delta rule state update — the y-flash trick.** The flash-linear-attention library describes a "state-passing" variant:
- Compute the per-chunk weight matrix `T` (SC = 64) with a single `mma.sync` per (head, chunk) pair,
- store the *entire* T in shared memory (64 × d_v × d_k f16 is too big) — keep it in **register fragments of 8×8 tiles** as mma.sync requires, avoiding spills.
For Cerebro: compile one `.ptx` kernel for the rank-64 update, another for the final output; **verify numerical equivalence on CPU before trusting the mma path.** Confidence: L–M.

**6.9 Failure mode: `alpha`, `beta` are in [0, 1], so `I − βkkᵀ` is fine, but drift accumulates** over 10k+ tokens in f16 state. Keep the state in **f32** (48 layers × 12 KB f32 = 31.5 MB, still trivial) — the cost of the state write-back doubles but precision issues vanish. **Confidence: M.** (Known failure mode for linear attentions in f16; worth engineering before shipping.)

**6.10 Hybrid-model speculative decode.** The linear layers have constant-size state (6.2) — forking them for a draft branch costs only an **O(d_v × d_k) memcpy** = 12 KB × 48 layers / 2 = 300 KB total state fork. That's essentially *free* per speculative iteration. **The hybrid architecture is the best-case for spec decode; dense models pay for the KV fork (§2.3); the hybrid pays ~5 µs.** **Confidence: H. This is a big deal for MTP (§1.11) on Qwen3.8/Next.**

---

## 7. Measurement methodology (how to validate every number above)

**7.1 Instrument Cerebro with a hardware performance-counter overlay.**
- GPU: `nvmlDeviceGetUtilizationRates` (sm/mem), `nvmlDeviceGetFieldValues(DRAM_ACTIVE)` requires driver support for GPUTOP counters — for 3070 Ti Laptop, use **NVIDIA GPU Viewer / Nsight Systems**. Profile with `nsys profile --trace=cuda,nvtx --gpu-metrics-device=all` to get `dram__bytes.sum.per_second`, `sm__cycles_elapsed.avg.per_second`, tensor-pipe utilization.
- CPU: ETW with `Windows Performance Recorder` + `HardwareCounter` profile — capture `CacheMisses`, `BranchMispredictions`, `Cycles`, `INST_RETIRED`.
- Measure tok/s as `(tokens_accepted)/(elapsed_time)` and separately `(kernel_time_total)/(elapsed_time)` at each phase.

**7.2 Per-item microbenchmarks to run on the target box.**
| Item | Harness |
|---|---|
| 1.4 PLD | 512 prompts of CTF/code-fix/refactor; measure mean accept-len, tok/s with and without |
| 1.8 EAGLE | same prompts, ablate `--draft-model yes/no`, tree depth 1/2/3, K=2/4 |
| 2.2 paging overhead | 1024-slot paging at 4/8/16 page size; measure kernel time of attention vs non-paged |
| 2.4 Radix hit-rate | same 8-turn chat replay × 100; hit rate / wall-clock |
| 3.2/3.3 KV q8 vs f16 | 100 prompts × 8k ctx; measure tok/s, KL(p_f16 ‖ p_q8) |
| 4.3 KTransformers-style hybrid CPU/GPU | 30B-A3B: 100-token generation with 0/2/4/6 experts deferred; measure tok/s + GPU util |
| 5.1 CUDA graph | a full forward pass with/without graph; measure µs per token |
| 6.3 GDN decode kernel | single-token forward on a linear layer; state in SMEM vs gmem; measure γ |
| (all) | verify `tok/s` *improves* when the metric is `tokens/second`, not `tokens/prefill-round` |

**7.3 Statistical hygiene.** Every comparison: ≥ 128 independent generations, seed-fixed prompts, after `nvidia-smi -lgc` for a locked clock; report mean ± 95% CI; drop results where the GPU thermal throughput fell > 10% from max (3070 Ti Laptop throttles hard in a laptop chassis at > 78°C).

---

## Top-20 ranked items (expected tok/s gain on the 3070 Ti-L / i9-12900H box, and confidence)

| # | Item | Gain (30B-A3B MoE unless noted) | Confidence |
|---|---|---|---|
| 1 | **5.1/5.2 CUDA Graph capture for whole decode (6 variants for k=1..8)** | 2–5× on dense ≤ 8B; 1.3–2× on 30B-A3B | H |
| 2 | **1.4 PLD n-gram spec for code workloads, no extra weights** | +30–80% on code, ~0 on prose | H |
| 3 | **4.5 offline expert-frequency profile; hot-experts-on-GPU** | 1.3–2.2× on 30B-A3B | M–H |
| 4 | **4.4 Fiddler CPU-vs-GPU-expert decision rule from calibrated latencies** | 1.2–1.6× | H |
| 5 | **4.13/1.11 MTP + speculative CPU-amortized expert execution (k=8 drafis share expert weights)** | +1.8–3× on MoE | H |
| 6 | **2.2/2.3 Paged KV + fork/rollback staging for spec-decode** | enables 2/5; +5% alone | H |
| 7 | **6.3/6.5 GDN decode kernel with state-in-registers + chunked verify on linear layers** | 1.6–2.5× on GDN-layer time; hybrid-model critical | M–H |
| 8 | **3.3 KV q8 fused into attention kernel (q8 attention-dequant)** | +5–12% at 8k–32k ctx; frees 1.5 GB VRAM | M–H |
| 9 | **4.7 async expert prefetch from pinned host memory on stream B** | 1.2–2× on CPU-expert MoE | H |
| 10 | **5.4 RMSNorm+QKV and SwiGLU epilogue fusion (existing int8 prefill kernel)** | +5–12% | H |
| 11 | **2.4 RadixAttention prefix cache across turns; prompt-cache on tokens not on hash-of-text** | 10–50× turn-boundary latency (not decode); *do it regardless* | H |
| 12 | **1.8 EAGLE-3-style draft head at 2048-hidden for Qwen-class checkpoints** | +40–90% on dense; +80–120% w/ MTP on hybrid | M–H |
| 13 | **4.12 AVX-VNNI `vpdpbusd` expert GEMV + `VirtualLock` + large pages** | 1.4–2.2× on CPU-resident experts | H |
| 14 | **2.5 chunked prefill interleave with decode (33ms slices)** | un-blocks the 2–20 s stall at turn start | H |
| 15 | **3.5 SWA + sinks for Gemma-4: drop SW pages in-place, recompute RoPE on the 4 sinks** | 27× attention speedup at 27k ctx; 10–30× on SW layers | H |
| 16 | **3.6/3.7 KV tiering: hot 4–16k in VRAM, the rest in pinned pages with async eviction on stream B** | enables 100k-token contexts at < 5% decode cost | H |
| 17 | **4.16 MoE-Infinity / Heptagon-style per-(layer, expert) transition prediction for prefetch** | 1.15–1.4× | M |
| 18 | **5.5 split-k GEMV revisited: for dense f16, use `__hfma2` warp-per-128-cols (no mma.sync at m=1)** | 1.15–1.3× over naive dot-product GEMV | H |
| 19 | **1.13 online acceptance-rate controller, enable/disable per workload tag (first 96 tokens as probe)** | protects against −20% spec-decode-loses cases | H |
| 20 | **4.18 hard-page-fault elimination: `VirtualLock` expert pool, warmup touch; target < 100 faults/1k tokens** | 1.05–1.5× on 30B-A3B cold starts | H |

**Not on the list, by design:** EAGLE-1 (superseded by 1.8's EAGLE-3 with the same plumbing), Medusa (only if no draft head is available), layer-skip self-speculation (pays the full verify price for a weak draft), Fiddler's AVX512-BF16 and KTransformers' AMX paths (12th-gen has AMX but the i9-12900H laptop's datacenter-oriented AMX tile path is UNVERIFIED; Cerebro's AVX2/VNNI path is sufficient), MoE-Lightning (batch-throughput-oriented; the pipelining insight is already in 4.7), PowerInfer-style hot/cold neuron split for dense models (this machine doesn't need dense CPU offload for ≤ 14B).
