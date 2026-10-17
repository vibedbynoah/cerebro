# Strata mechanics, reimplemented in Cerebro (Task B, 2026-10-16)

Strata: C++/CUDA/HIP engine + Python server for Qwen3.8-Flash-Next (48 layers = 36 GDN linear-attention +
12 QSA full-attention layers; every layer MoE with 512 routed experts, top-10 active, plus shared expert;
hybrid gated-residual streams), MIT, https://github.com/Niko1221/Strata.

Sources read for this paper:

- Local install `C:\Users\noahm\strata`, source read directly. `file:line` references are real from this
  checkout. Full reads: `include/strata/core/{expert_source,session,verify,mtp,conversation_cache,layout}.hpp`,
  `include/strata/kernels/{cpu/pool,ngram,kv_stream}.hpp`, `include/strata/prefill/prefill.hpp`,
  `src/core/expert_cache.cpp`, `src/core/concurrent_main.cpp`, `src/spec/{controller,draft_policy,suffix_drafter}.cpp`,
  `serve/runconfig.py`; targeted greps in `serve/server.py`; big files (`expert_source.cpp` 2,985 lines,
  `prefill.cpp` 3,084 lines, `layer.cpp` 1,349, `server.py` 4,145, `session.cpp` 970, `weights.cpp` 479)
  read selectively or via headers that carry the design comments. Where a claim comes only from shipped
  documentation (`docs/DETAILS.md`, `docs/HOW_IT_WORKS.md`, `docs/MULTI_GPU.md`, README) I cite it as such;
  anything I could not pin to either is UNVERIFIED.
- `docs/research/strata-analysis-2026-10-07.md` (Cerebro's own prior analysis, measured numbers for this PC).
- Cerebro context: `docs/specs/engine.md`, `docs/specs/turbo-system.md`,
  `docs/measurements/engine/2026-10-10-engine-report.md`.

Hardware anchor for "expected gain" fields: i9-12900H (6P+8E, AVX2/VNNI, no AVX-512), 32 GB DDR5-4800,
RTX 3070 Ti Laptop 8 GB sm_86 power-limited, Gen4 x8 laptop link, NVMe. Strata's own benches are on
12–24 GB cards; numbers are quoted as published and adjusted estimates are marked [H] with confidence.
Confidence: high = measured on comparable arithmetic; medium = same mechanism, size-dependent; low = plausible.

Web findings used: GVS5H paper arXiv 2608.26480 + https://github.com/slee-persis/GVS5H; Strata repo README/
HOW_IT_WORKS/DETAILS/HIGHLIGHTS via GitHub search snippets (2026-10-15/16 snapshots); quotes for
MTP/draft-vocab and YARN from Strata's own bench folders cited in DETAILS.md. Related-work mapping
(LLC-less decode, speculation, KV quant) uses widely known references marked as such at first use.

Cerebro file targets cited as `ce/...` = `crates/cerebro-engine/src/...` (engine worktree). MIT allows reuse of
Strata with attribution; every reimplementation sketch below is a clean-room description of behavior
(deep-derivative concerns noted nowhere: mechanisms are algorithms, not code). Where we model a data structure
closely on Strata's (STRP profile format, kv-stream residency map), add an attribution line in the module docs.

## 1. Architecture: the spine a token hangs on

- **Verify window** = decode unit. T ≤ 8 tokens (last accepted + drafts) through all 48 layers in ONE captured
  CUDA graph; per-row head argmax decides acceptance bit-exactly because every kernel is "the single-token kernel
  per token" (`verify.hpp:4-11`). Commit (`verify.hpp:129`): GDN conv history advanced from stored inputs,
  QSA indexer tail restored from snapshot then accepted keys re-appended, speculative K/V cells overwritten on
  replay before any query reads them (`verify.hpp:13-18`).
- **One graph per layer**, host decides order (`session.hpp:127-137`); per-token data arrives via fixed-address
  host-mapped staging (position as data, `session.hpp:152-154, 229-234`). Layer = `pre[l]` (gr_read, mixer,
  router, shared expert, **doorbell ring**) + `post[l]` (combine + gr_write, launched after the CPU pool returns)
  (`session.hpp:140-150`).
- **Doorbell / device wait**: host polls a mapped counter with `cudaEventQuery`, never `cudaStreamSynchronize`
  (`session.hpp:330-343`; the sync variant measured 5,907,703 futile spins). The whole token can instead be ONE
  graph (all 48 layers + head) with embedded device-wait kernels parked on mapped flags; the host thread then
  makes no driver calls per layer; handoff measured 5.2 µs (`session.hpp:386-409`). Replay of graphs vs direct
  launches: 1.585 vs 2.393 ms per layer — launch overhead under WDDM is ~0.3–0.4 ms/launch (`session.hpp:130-136,
  386-392`).
- **CPU/GPU split per layer** (`expert_source.hpp:269-303`): routed experts split BY ROUTER INDEX — residents run
  on the GPU through grouped-hit kernels reading the slot arena; misses are computed on the CPU pool in-place
  from pinned RAM; both land in `parts` at the router-given indices; `moe_combine` is unaware which produced what
  (`layer.cpp` cited at `expert_source.hpp:14-17`).
- **Host-side process**: `serve/server.py` drives the engine over stdin/stdout (commands like `QUIT`), watches
  its log, restarts on death, owns the HTTP API + telemetry (`server.py:87,87-149` per-read lines).

## 2. Mechanism catalogue

### A. Expert placement and the tiered memory system

**A-01 VRAM expert slot cache (compulsory-miss, no-eviction fill).**
What/How: one `cudaMalloc` (or segmented) arena of n_slots × blob_bytes; residency table
`n_layers × n_expert -> slot|-1` (`expert_cache.cpp:456-478`). `admit()` fills the first time (layer,expert) is
routed and never evicts (`expert_cache.cpp:551-574`); fills are `cudaMemcpyAsync` H2D per slot with a blocking
variant for HIP staging (`expert_cache.cpp:588-639`). The allocation is checked against ACTUAL free VRAM before
the call, with a named-numbers refusal; since 0.1.19 it retries smaller instead of failing
(`expert_cache.cpp:417-447`; `DETAILS.md:1001`). Slots zeroed so a not-yet-filled slot is a deterministic
wrong answer, not garbage (`expert_cache.cpp:448-454`); a startup `verify_slot` re-reads a filled slot and
compares bytes (`expert_cache.cpp:667-694`).
Flag: `--expert-cache N` (auto sizing uses NVMe/card/RAM model; `open_sized` supports variable slot sizes
256-byte aligned, `expert_cache.cpp:482-513`).
Gain: ~700 experts per extra GB on the reference card (`DETAILS.md:233`); [H] on 8 GB ≈ 3.5-4.5k resident of
24,576 — matches our measured 31-48% first-turn hit rate scaling. Confidence: high.
Cerebro: exists (`ce/moe/cache.rs`). Adopt: the pre-allocation check + smaller-retry + slot-verify at startup.
Risk: none. Channel: Stable.
Toggle: "Caches the most-used MoE experts in leftover VRAM; misses compute on the CPU in parallel."

**A-02 Shipped expert profile ("STRP") + day-one warm cache.**
What/How: binary format: magic `STRP`, header {version, n_layers, n_expert, slots, n_ranked}, ranked
(layer,expert) pairs as u16-pairs (most-used first), then a full rank-index table; validated against the model's
dims, reader refuses mismatches and out-of-range pairs (`expert_cache.cpp:23-79`); written atomically tmp+rename
(`expert_cache.cpp:110-150`). Startup fills VRAM slots in rank order; `rank_learned_profile` merges live
residency first, then heat, then the shipped prior, stable-sorted (`expert_cache.cpp:81-108`). Shipped profiles
rank all 24,576 pairs (all layers × experts = 48×512); before 0.1.14 only 8,000 were ranked, which wasted ~10 GB
of cache orders (`DETAILS.md:233-238`). `tools/make_profile.py` builds one from your own routing trace
(`--dump-routing trace.bin`).
Flag: `--expert-profile path`; `--reorder` puts your trace's pairs ahead of the shipped ones.
Gain: direct fix for Cerebro's measured 19-23 tok/s first-turn MoE decode; [H] 19-23 → 29-34 tok/s on the
512-token Coder-30B prompt (hit ~70% from cold instead of 31-48%). Confidence: medium (content-domain transfer).
Cerebro: `ce/moe/profile.rs` — own format (u32 magic "CBRP", version, dims, ranked u16 pairs + full rank table),
per-model profiles built by a `dump-routing` harness over a mixed corpus, validated at load. MIT/attribution
noted. Channel: Beta (immediate; the engine report names this next-step 1).
Toggle: "Starts the expert cache pre-loaded with this model's hottest experts so the first answer is fast too."

**A-03 Adaptive tier: decayed routing counts + cross-request swap.**
What/How: during decode the driver decays per-(layer,expert) routing counts (`expert_source.hpp:359-363`); between
windows the most-routed missing experts are swapped into VRAM. The resident-copy exchange mechanism keeps the RAM
mirror consistent with swaps without re-reading files: the evicted expert's slot bytes are copied into an exchange
buffer before overwrite; the RAM complement later swaps `out` into `in`'s place (`expert_source.hpp:473-498`).
`--adapt-async 1` moves tier updates to a helper thread so the answer never pauses: measured +4-5% decode,
+7-10% on code/long docs (README #1904/#1905).
Flags: `--adapt-every N`, `--adapt-async 1`.
Gain: our 38.6-55.3 tok/s post-adaptation already includes a similar policy; adopting async swaps should
remove the serial tier-update pause: [H] +3-7% on the Coder-30B model. Confidence: medium.
Cerebro: `ce/moe/adaptive.rs`; swap thread + copy-stream async commit; exchange-buffer mirror logic.
Risk: unbounded swaps under thrash — count and amortize per window. Channel: Beta.
Toggle: "Learns which experts your conversation uses and swaps them into VRAM in the background."

**A-04 Profile persistence across restarts.**
What/How: on clean exit and every N minutes between requests (default 10), the engine saves the learned tier as
a STRP profile (atomic write); the next start prefers it if it matches the model (`DETAILS.md:456-465`).
Cerebro: `ce/moe/profile.rs` save-on-idle; gain: first-turn speed carries over sessions — our biggest repeat
gap. Risk: profiles reveal usage; document as per-PC data. Channel: Beta.
Toggle: "Remember which experts your projects use, across restarts."

**A-05 Per-layer slot quotas.**
What/How: layer l owns [l·q, (l+1)·q) slots with its own cursor; last layer takes the remainder
(`expert_cache.cpp:540-549`). A single global counter for all 48 layers measured hit rate capped at 2.97%
(`expert_cache.cpp:561-566`) — per-layer partitioning was worth ~10x on cache growth dynamics.
Cerebro: weight per-layer quotas by profile rank density instead of uniform (improvement). Channel: Stable.

**A-06 PCIe share of misses (`--pcie-frac`).**
What/How: a fraction (n/256) of each layer's distinct missed experts is served TO THE GPU over PCIe instead of
by the CPU pool. Three transports probed/typed: (0) copy-engine DMA into VRAM staging via `fetch` (best when
CPU is compute-bound — i-quant kernels), (1) the grouped kernel reads the mapped pinned arena directly over the
bus (UVA device alias), (2) an in-graph copy kernel stages it (no driver calls on the pool thread; best when the
CPU is RAM-bound — Q2_0) (`expert_source.hpp:207-232`, `verify.hpp:199-215`).
Flags: `--pcie-frac` (0-1 of misses), `--pcie-mode 0|1|2`.
Gain: converts CPU DRAM bandwidth demand ↔ PCIe (Gen4 x8 ≈ 13-16 GB/s effective here). [H] our card is CPU+RAM
bound with a slow link: likely a small frac (10-25%) gains 5-15% decode on MoE; the `--calibrate` rule keeps a
setting only if > 3% faster (`DETAILS.md:361-370`). Confidence: medium. Lesson: a share changes GPU rounding vs
CPU (fp16 `d` vs CPU fp32 activation scales) → nondeterministic tokens; `--pcie-frac 0` is the reproducibility
arm (`DETAILS.md:93-101`).
Cerebro: `ce/moe/pcie.rs`; implement all three modes behind one sink trait, calibrate at install like Strata.
Risk: DDR5 + PCIe reconciliation — measure CPU-alone/PCIe-alone/both interference first (Strata's own
`concurrent_main.cpp:1-45` is the A/B harness design worth copying as `ce/bench/concurrent.rs`).
Channel: Experimental until calibrated on this PC.
Toggle: "Instead of computing missed experts on the CPU, send a few to the GPU over the PCIe link."

**A-07 mmap/low-RAM expert placement (mmap, resident, budget).**
What/How: three disk-tier modes. (a) `--mmap-experts`: map the pack's `experts.bin` read-only; the OS page cache
holds what VRAM does not; Coder committed memory 36 → ~13 GB (`DETAILS.md:120-127`). (b) `--resident-experts`:
copy exactly the non-VRAM experts into page-locked (or just locked) RAM so steady state touches no SSD;
adaptive swaps copy evicted experts back into RAM so the RAM mirror stays the complement of the cache
(`DETAILS.md:129-152`, `expert_source.hpp:436-460`). (c) `--resident-budget-gib N`: rank the non-VRAM experts by
profile and keep only N GiB, the rest read from files; the 0.1.31 lookahead warms predicted pages (A-14)
(`DETAILS.md:170-181`). Sizing check: base bytes + slots (top-down from `lend_from`) must fit
available-RAM − 4 GiB headroom else refuse (`choose_resident_keep_from`, `expert_source.hpp:96-104`).
Gain: enabler for the 125B MoE on 32 GB: [H] on our PC the UD-IQ4_XS pack (≈-30 GB experts) runs mapped with
~8-10 GB resident + 8 GB VRAM cache; this is the config Strata currently serves at ~20 tok/s with a cold-ish
profile. Confidence: high that the mechanism transfers (owner-truth); our engine's absolute number unknown.
Cerebro: `ce/weights/mmap.rs` (our GGUF reader is already mmap — extend with verified expert-span tables),
`ce/moe/resident.rs`, RAM-budget resolver with headroom. Channel: mmap Beta; resident+budget Experimental.
Toggle: "Keeps only the experts your RAM can afford and maps the rest straight off the SSD."

**A-08 GGUF-in-place blobs (no experts.bin) + staged buffers.**
What/How: when the pack has no experts.bin, map the model's GGUF shards and assemble each requested blob from
three role slices (gate/up/down) named by `native_experts.txt`, with a pre-flight span check of every tensor's
name/type/shape/offset/bounds (`expert_source.hpp:424-433, 711-718`). Blobs handed out live in a small pool of
stage buffers reused only after `kStageAge = 3` layer epochs, filled by `fetch_threads` (default 8) via one
batched `PrefetchVirtualMemory` (Windows) / `madvise(WILLNEED)` request (`expert_source.hpp:527-533, 575-597`).
Unbuffered mode (issue #286/#577): when the OS cache cannot keep the working set, switch to overlapped
`FILE_FLAG_NO_BUFFERING` reads per 4 KiB-aligned window into thread-local aligned buffers, with batched
assemblies (≤ 64 pairs) for cache fills (`expert_source.hpp:502-517, 555-557`).
Cerebro: `ce/weights/role_reader.rs` — mmap + epoch-pooled staging + a bounded prefetch thread; unbuffered
direct I/O via std::fs overlapped handles. Risk: highest-complexity item; WDDM mapped-alias forces need
testing. Channel: Experimental.

**A-09 Pinned arena + device aliases (the "file cache is not RAM" lesson).**
What/How: the mapped experts.bin path measured ~19 GB/s in-engine vs 42.8 GB/s for the same bytes on a heap
arena (micro-bench `bench/micro/cpu_s2.cpp`) on the SAME machine, because file-backed standby pages are the
first thing the OS evicts when a second 26.8 GB mapping (the PLE table) competes (`expert_source.hpp:637-652`).
`ArenaExpertSource` reads experts.bin once at startup into `PinnedArena` (tries `cudaHostRegister`,
best-effort, falling back to working-set lock) so the token path reads anonymous resident memory
(`expert_source.hpp:653-709`). Loading is measured in GiB/s (≈1.4-1.5 GB/s from NVMe, `DETAILS.md:379-384` —
Task Scheduler "below normal" priority throttled it 24×; must run normal priority with privileges for
`SeLockMemoryPrivilege`).
Gain: [H] this is almost certainly the mechanism signature behind Cerebro's 26-28 GB/s in-step vs 37-41 GB/s
isolated CPU-kernel gap (our load path touches pages at load time, but thread/host placement and page parity
may still be off — A-09 + A-10 + A-11 together target it). Confidence: medium-high for diagnosis value.
Cerebro: `ce/weights/pinned.rs`: allocate expert region with large pages where possible, VirtualLock ≤ the
WDDM pin budget (~8 GiB under WDDM; more mapped instead), `cudaHostRegister` slices for DMA/alias.
Channel: Stable (allocation policy + one benchmark).

**A-10 CPU expert pool: epoch-batched SPMC with provable completion.**
What/How: jobs are one flat batch per layer (a layer-sum is a barrier — a ring buys nothing); the claim is a CAS
on one 64-bit `epoch|njobs|index` word so a worker that slept through a publish can never claim the next batch
(issue #29's deadlock survived three fixes before the epoch claim, `pool.hpp:14-27, 246-253`); workers spin
20 ms then sleep on a condvar with a `sleepers_` flag so publish costs one uncontended load (`pool.hpp:194-201,
255-260`); thehostthread also drains jobs (`host_works`) so the reserved core is not idle (`pool.hpp:126-138`);
the diagnostic spin counter was itself a locked RMW hammering the publish line and was deleted; every hot atomic
got its own 64-byte line (`pool.hpp:231-243`); a 60 s stall watchdog prints every thread's state
(`pool.hpp:202-204, 139`).
Gain: [H] combined with A-09: the pool is scheduling-shaped, not kernel-shaped; measured phase split shows sync
costs masquerading as bandwidth costs (the `pool phases drain/wait_park/repark` tool, `pool.hpp:181-192`).
Cerebro: hand-rolled pool `ce/cpu/pool.rs` (rayon cannot express the epoch claim + park protocol), the same
three phase counters exported to the Monitor (our bottleneck verdict feature). Channel: Stable once it beats
rayon on this PC.
Toggle: "Runs each layer's missing experts across all CPU cores and reports exactly where the time goes."

**A-11 Hybrid-core topology + affinity (`--pool-affinity auto|p-cores`).**
What/How: detect physical P-cores / SMT siblings / E-cores (`pool.hpp:66-91`); one worker per PHYSICAL core
(putting two workers on SMT siblings halves useful bandwidth — `pool.hpp:82-91`); host loop pinned to its own
core: pool 36.32 GB/s on 5 workers but 26.9 GB/s inside the host loop unpinned = 1.35x from placement
(`pool.hpp:93-106`). Windows uses reversible CPU Sets, not hard affinity, on 11's group semantics
(`pool.hpp:107-118`).
Gain: [H] our 6P+8E: fewer, P-only workers likely beat 16 workers on mixed cores (Strata: "fewer can be
faster" on E-core CPUs, `DETAILS.md:365-370`). Confidence: medium-high.
Cerebro: `ce/cpu/topology.rs` (GetLogicalProcessorInformationEx + CPU Sets), wired into Turbo's scheduler
section (turbo-system.md §2). Channel: Stable.
Toggle: "Pins the expert pools to your fast cores and reserves one core for the GPU driver thread."

**A-12 Row-split expert compute (`run_split`, `--no-split-rows` to disable).**
What/How: when jobs < threads (the common case with a warm cache), each job is split by output rows across ALL
threads: gate/up row parts → intermediate quantization → down row parts, three phases, each expert's row range
consumed exactly once (`pool.hpp:160-173`),
split scratch is per-thread `SplitBuf` (alignas(64)) so no false sharing (`pool.hpp:266-269`). At most
kMaxSplit=16 experts; up to kMaxSplitMulti=96 multi-token ones with rows read once for all tokens.
Gain: [H] closes the "high hit rate leaves cores idle" case — direct target of our 26-28 GB/s in-step gap.
Cerebro: `ce/moe/cpu_expert.rs` row-split scheduler with bitwise parity vs whole-expert compute. Channel: Beta.

**A-13 Multi-token expert jobs for verify windows.**
What/How: each distinct missed expert of a T-token window is computed ONCE for all tokens routed to it
(`ExpertJobMulti`, `pool.hpp:55-64`); the union of misses grows sub-linearly: 1.75x one token's misses at T=2,
2.4x at T=3, 3.05x at T=4 (`verify.hpp:10-11`) — the arithmetic reason speculation pays on the CPU-bound half.
Shared quantized activation saves 480 redundant conversions per token (`pool.hpp:44-49`).
`STRATA_IQ_MT_MIN=1` forces the multi-token kernel everywhere for bit-reproducible answers (`DETAILS.md:87-99`).
Cerebro: build `ce/spec/verify.rs` + multi-jobs together with M5/M6. Channel: Beta.

**A-14 Router lookahead prefetch (`STRATA_LOOKAHEAD`, `--no-...` off-switch culture).**
What/How: single background thread; on submit for layer l: copy this layer's MoE input (the residual changes
little across layers), apply layer l+1's router (BF16, host), top-k per token, drop experts the GPU/RAM copies
hold, `warm` the rest; a late pending prediction makes the new one skip, never wait (`expert_source.hpp:169-205`).
Only pages: never changes computed experts. Counters `predicted/skipped/busy_ms` exposed for A/B
(`expert_source.hpp:186-188`).
Gain: with resident-budget (A-07), decoded misses come from SSD; warm pages cut latency-bound faults: [H]
10-20% first-turn decode on the 125B model. Confidence: medium.
Cerebro: `ce/moe/lookahead.rs`; the next-layer router topk is trivially parallel on AVX2. Channel: Experimental.

**A-15 Peer/remote GPU tiers + layer split (`--layer-split auto`, `--peer-device`, pipeline).**
What/How: each card runs a contiguous layer range, caches ITS layers' experts, verify windows hand off the
final residual per stage (bit-exact: handoff = R + pending bo + inject floats, `verify.hpp:112-127`);
`--pipeline-windows 2` starts the next (guessed) window on front stages while the last verifies
(`MULTI_GPU.md`); remote expert caches (`--expert-cache-device1..3`, `--remote-expert-opt`) put helper GPUs
beside the CPU pool for misses (`expert_source.hpp:238-244`).
Gain: on a second NVIDIA box. [H] not applicable to the 780M box (no CUDA). Confidence: n/a.
Cerebro: defer; record the split-handoff design in `ce/session/` notes. Channel: Deferred.

**A-16 VRAM elastic segments (`--vram-elastic`, `POST /v1/vram`).**
What/How: the expert cache is one virtual range backed by physical 512 MiB segments via the driver's VMM API
(looked up through the runtime entry points; HIP excluded, `expert_cache.cpp:154-305`); `shrink` unmaps tail
segments after a device sync (live_slots recomputed from the mapped prefix `slots_within`,
`expert_cache.cpp:314-354`), `grow` remaps up to the started size; nothing resizes by itself; the server route
runs between requests (`DETAILS.md:444-454`). Measured: shrink 78 ms free 4.3 GiB, decode 44→33 tok/s, answers
token-identical; grow 92 ms, same experts in same slots.
Gain: Cerebro's "share GPU" promise without unload. [H] driver+sm_86 supports VMM; confidence high.
Risk: reallocated pages are zeroed→first use must refill (same-expert-same-slot growth already handles it).
Cerebro: `ce/gpu/vmm.rs` via the driver API we already use for launches. Channel: Beta.
Toggle: "Lets other apps borrow the GPU's memory while Cerebro keeps the model loaded and runs slower."

**A-17 VRAM reserve parts (`--vram-reserve-mib`, `--vram-segment-mib`).**
What/How: fixed reservation (head ~0.3-0.8 GB, MTP layer ~0.9 GB, draft head 0.18 GB with subsetting, vision
1.4 GB when on) taken BEFORE the cache is sized so the cache never over-commits; on start failure named-VRAM
refusal names the free/total (`DETAILS.md:105-118`, `mtp.hpp:55-59`, `expert_cache.cpp:417-447`).
Cerebro: engine-side reserve accounting in `ce/moe/cache.rs` sizing; default demonstrated on install.
Channel: Stable. Toggle: "Keeps VRAM aside for the model's support parts so allocation never half-fails."

### B. Concurrency: graphs, streams, batching

**B-01 Per-layer graph capture + fixed-address staging.**
What/How: graphs bake pointers, so all per-token quantities are written into mapped host buffers whose DEVICE
aliases are baked; replay = 48 graph launches instead of ~2,000 kernel launches; measured launch-bound: 1.585 vs
2.393 ms/layer replay-vs-direct (`session.hpp:127-236`).
Flag: engine-internal; `--no-capture` debug arm.
Gain: [H] directly applicable; our engine already uses CUDA graph replay (engine report), extend to the
staging pattern for positions/counts. Channel: Stable.

**B-02 Host loop with event-query poll (no stream sync in the fast path).**
What/How: `session_loop` waits only on `cudaEventQuery` of an event recorded after the launch (`session.hpp:330-343`);
per-token allocations (host staging, events, affinity changes) moved to a reusable `SessionLoopScratch` because
`cudaFreeHost` implicitly SYNCHRONIZES the device (`session.hpp:354-380`).
Gain: [H] any per-token host allocation in our loop has the same hazard — audit `ce/session/`. Channel: Stable.

**B-03 Whole-token graph with device-wait handoff (zero driver calls per layer).**
What/How: one graph holds all 48 layers; after each `pre[l]` a one-thread kernel spins on mapped flags
(`db.h_flag`) the host writes; the host only polls rings, runs the pool, writes flags (`session.hpp:399-409`).
Measured 5.2 µs/handoff vs 0.258 ms/layer of host round trip in the split-launch version (`session.hpp:207-225`).
`all_resident_` fast path: 100% resident layers skip doorbells entirely (zero-doorbell graphs)
(`verify.hpp:258-261`); `device_plan_` lets the GPU build the hit plan when residency is device-static
(E-6, `verify.hpp:259-261`).
Gain: [H] high — on a WDDM laptop this is the difference our engine report's "10 fused kernels + graph replay"
shows partially; going full-token-graph should lift both dense 0.6B decode (344 tok/s) and MoE overhead.
Confidence: medium-high. Risk: host-side latency spikes stall the GPU (no room to slip in more work);
Strata keeps a flush fallback (`flushes` counter `session.hpp:408`).
Cerebro: `ce/gpu/tokengraph.rs`. Channel: Beta.

**B-04 Overlap accounting: rings_mid_graph, ms_to_ring, ms_host.**
What/How: the loop COUNTS where the one-layer-delay goes: rings observed mid-graph (real overlap) vs after
(GPU idle during pool); ms ring→next-launch is host-serial time (`session.hpp:156-226`). `--no-pool` measured
38.73 ms/token vs 26.32 ms pure-GPU floor → a quarter of the token was host round trip (`session.hpp:207-212`).
Cerebro: same three counters + the Monitor's bottleneck verdict ("CPU-bound on experts: +X experts would fix",
turbo-system.md §4). Channel: Stable (telemetry), first.

**B-05 Batch slots: S sequences per verify window.**
What/How: `init_slots` carves S independent sessions; a batch window runs S slots × 1 token in one graph —
weights read once for all rows; a slot's greedy tokens are its solo tokens (`verify.hpp:131-153`);
`run_slot_rows` runs only the listed slots so idle ones keep state; per-slot sampling with Philox(seed, position).
Server opt-in `"parallel": N` (`DETAILS.md:686-688`, `BATCHING.md`). 409 after-request pipelining etc.
Gain: multi-conversation throughput on one user's machine (agent + chat in parallel). [H] MoE: expert union
grows across slots → CPU miss cost rises sub-linearly but real; dense parts amortize beautifully. Confidence:
medium. Cerebro: `ce/session/slots.rs` — big design work; continuous batching already spec'd
(engine.md §6). Channel: Beta.

**B-06 Async commit (`set_commit_async`).**
What/How: commit() returns without waiting for its commit graph; the next window follows on the same stream and
the drafter reads nothing it writes; anything else touching the session must `wait_commit()` first
(`verify.hpp:174-181`). Channel: Beta (with B-03/B-05). Toggle-sentence candidate: "Lets the next step start
while the state save finishes."

**B-07 Split verify window in two token-groups pipelining CPU vs GPU halves.**
What/How: `set_split` (default ON): group 1's CPU experts compute while group 2's GPU work runs and vice versa
(`verify.hpp:199-201`); measured in Strata's decode-timing lines (`DETAILS.md:193-200`: wait-GPU / pool plan /
act-quant / jobs / stage / commit / draft per window).
Cerebro: adopt as `ce/spec/pipeline.rs`; the decode-timing env switch (`STRATA_DECODE_TIMING`) is the same
"toggle with an explanation" we want. Channel: Beta.

### C. Speculation

**C-01 MTP drafter (model-native multi-token head).**
What/How: one MTP layer: cell i pairs the final multi-stream residual at position i with token i+1; predicts
i+2; own K/V; all 512 of its routed experts live in VRAM (~708 MB) and run through the grouped hit kernels; the
large projections run Q8_0 via multi-column MMVQ; attention is DENSE over every cell ever seen (identical to the
model's sparse selection under 2,051 cells) so speculative cells never touch indexer state (`mtp.hpp:1-24`).
Captured step graphs (9 sizes), round graph = catch-up over the window's T residuals then draft chain from the
last accepted row (`mtp.hpp:99-109, 60-75`).
Flags: `--mtp`, `--draft-vocab en|cjk|cyrillic|fr|...`.
Gain: 0.89/0.86/0.85 acceptance at draft steps 1-3 measured via `tools/mtp_probe.py` (`mtp.hpp:4-7`);
2.4-3.2 tokens per pass average (`DETAILS.md:1032-1035`; owner-logs 59-66% on this PC at IQ3_XXS via
strata-analysis). [H] large eye opener: ~1.6-1.8x decode. Confidence: high (validated per mechanism).
Cerebro: `ce/spec/mtp.rs` — needs Qwen3.8 MTP support (M6). Channel: Beta.
Toggle: "The model's own draft layer guesses several tokens at once; verifying them costs one extra pass."

**C-02 Draft-vocab subsets.**
What/How: the draft head only proposes from a token subset (file `draft_vocab.bin`); more subset languages ⇒
more drafts land CJK: acceptance 0.51→0.60, +12% in French, etc. (`DETAILS.md:102-118`); the head costs ~110-180
MiB of VRAM which the expert cache reserves. Build tool: `tools/draft_vocab.py --corpus` computes a
99%-coverage subset for a language; startup refuses overshoot by naming the largest subset that fits
(`DETAILS.md:115-118`).
Gain: [H] for our users' languages; English/code default is `en` (40,525 ids). Confidence: high (per mechanism).
Cerebro: `ce/spec/draft_vocab.rs` — trivial once MTP exists. Channel: Beta.

**C-03 Coupled draft sampling (STRATA_SPEC_COUPLED).**
What/How: for temperature>0 requests, the drafter SAMPLES using the target's sampler chain and the verifying
row's Philox draw; `probs[i]` is the draft's probability under that chain; greedy requests keep argmax graphs
(`mtp.hpp:77-90`). Penalty history rides through `set_draft_history` (`mtp.hpp:82-90`).
Cerebro: adopt for sampled requests; without it sampled acceptance collapses. Channel: Beta (with C-01).

**C-04 Suffix / prompt-lookup drafter.**
What/How: a two-level structure over the request history: hash table of 3-token (bigram/trigram mixing) keys,
open addressing at load factor ≤ 0.5, each slot keeps the last WAYS (4) positions; propose(): hash the current
3-gram, take the most recent match, extend the match back to ≤ max_match tokens, emit the continuation up to
max_k, min_match ≥ 3 to fire (`suffix_drafter.cpp:1-88`). Verified like MTP drafts ⇒ output unchanged
(`DETAILS.md:1034-1035`; gains: code edits/quotes +6-11%).
Cerebro: `ce/spec/suffix.rs` (~200 lines). Channel: Beta (gate enablement by C-05).
Toggle: "When your reply repeats the prompt (code edits, quotes), the earlier text drafts the next tokens."

**C-05 Speculation controller: expected-tokens-per-ms with learned acceptance.**
What/How: `CostModel::step_ms(k, mtp)` = dense part (measured per-window cost curve) + CPU-miss part modeled
via a distinct-expert growth ratio U(n) × hit_rate + sync + drafting cost (`controller.cpp:10-22`); a `choose`
compares sources (none/MTP/lookup, k = 1..K_MAX) by tokens/ms and requires ≥ min_gain margin over the
1-token baseline; per-position acceptance rates are learned online by EMA (`observe`, `controller.cpp:63-80`),
with priors 0.86 (MTP step 1) and {0.35, 0.6, 0.8, 0.92} lookup by match length.
Flag: `--spec-min-p <p>` (the user-side minimum certainty per added draft; the policy-side equivalent is the
margin); priors overridden by measured runs.
Gain: keeps speculation from LOSING — our spec must be gated the same way. [H] with slow CPU misses, the
controller's miss-union term is larger ⇒ only small k survive. Confidence: high on design value.
Cerebro: `ce/spec/controller.rs`; wire into `ce/spec/verify.rs`; expose acceptance + tokens/pass in the Monitor.
Channel: Beta.
Toggle: "Only speculates extra tokens while statistics show it is faster than plain decoding."

**C-06 DraftPolicy: window-size chooser with cost interpolation + probes.**
What/How: measured per-window cost times averaged EMA (alpha 0.1); un-measured sizes interpolated by a fixed
relative-cost shape (1.0→3.6 over kMaxT=8, measured on the reference card `draft_policy.cpp:7-10`); lookup
acceptance learned per match-length bucket {<6, <12, <24, ≥24} with beta-style priors worth 4 observations
(kPriorN), counts decayed 0.97; a confident-but-unmeasured size is TRIED 3 times (kProbes) so the cost model can
never veto a size forever (`draft_policy.cpp:27-32, 91-98`).
Cerebro: same structure in the controller crate; this is exactly our "every optimisation measured + auto-pick"
philosophy in code. Channel: Beta.

**C-07 Verification correctness invariants.**
What/How: multi-token kernels are bitwise-equal per token (`verify.hpp:5-11`); penalties apply per-window-row
exactly as if decoded one token at a time (cost fix 0.1.19: pre-0.1.19 only row 0 got penalties — 1-11% slower
with penalties now, correctness worth it) (`DETAILS.md:696-699`; `verify.hpp:95-110` row-major penalty history).
Cerebro: adopt the invariant "every verify kernel has a per-token parity test" as an engine rule (engine.md
Rules). Channel: Stable (policy).

### D. Prefill

**D-01 Chunked prompt path (`--prefill auto`).**
What/How: prompt read in chunks of up to 8,192 tokens (32,768 opt-in); 0.1.39b's auto rule picks the largest
chunk whose buffers (incl. the streamed-expert ring) fit the expert-cache slots the request borrows — a request
borrows only what its prompt needs, refunded after (`prefill.hpp:64-99, 109-111`). Measured: 0.1.13 doubled
long-prompt speed; 32K MMQ Q2_0 1,290 tok/s (`DETAILS.md:205-216`).
Flags: `--prefill auto|N`.
Gain: [H] our MoE prefill is 200 tok/s vs llama.cpp 295; chunked grouping + ring stream is the biggest lift.
Confidence: medium-high. Cerebro: `ce/prefill/chunk.rs`. Channel: Beta.
Toggle: "Reads long prompts in large blocks instead of token-by-token."

**D-02 Prompt path borrows cache slots (loan protocol).**
What/How: prompt buffers carve from the top of the expert cache for the duration of the request, then refill
the slots (`prefill.hpp:64-72, 98-101`); #583 refinement: ring as a BYTES budget sized to the pack's blob size,
keep ring full for >6,144-token chunks but never shrink the ring (measured −12% if done) (`prefill.hpp:99-111`).
Cerebro: `ce/prefill/borrow.rs`; Strata's numbers (IQ3_XXS +18.5% at 1,500-slot ring) are exactly the right
A/Bs to reproduce. Channel: Beta.

**D-03 Expert streaming over PCIe during attention (the ring).**
What/How: per layer per chunk: resident groups from the VRAM tier; the rest grouped by expert, streamed from
the pinned arena through a byte-sized ring on the copy stream while the current layer's attention runs;
double-buffered ring slots, event-ordered (`prefill.hpp:5-11`; details in `prefill/kernels.cu`, `moe_fused*.cu`,
not fully read here: UNVERIFIED for kernel shapes). `set_pinned_share` sizes the ring by how much of the
streamed bytes is DMA-able directly (`prefill.hpp:75-81`).
Gain: contributes to the same D-01 number. Cerebro: `ce/prefill/ring.rs`. Channel: Beta.

**D-04 MMQ prompt kernels (quantized × Q8 act with dequant-on-the-fly, tensor-core option).**
What/How: experts multiplied by quantized MMQ kernels instead of expanding to FP16 first (0.1.13's biggest
prefill term); on RTX 30+ the Q2_0 path uses fused int8 tensor-core kernels: +16-22% prompt throughput on the
reference card, ~bit-equivalent accuracy (teacher-forced KL 0.009 vs 0.012) (`DETAILS.md:22-28`); IQ fused
kernels opt-in via `STRATA_PF_FUSED=1` (+12% IQ2 at 4K).
Gain: [H] directly targets Cerebro's dp4a-only prefill (10× llama.cpp gap): VNNI dp4a vs int8-mma is exactly
the mma step in our engine report next-step 2. Confidence: high direction; magnitude unknown.
Cerebro: `ce/gpu/mmq.rs` — mma.m16n8k32.s8 pipelined GEMM (no tensor cores on Ampere for BF16x9 advantage
here; int8 mma is the play). Channel: Beta.

**D-05 Prompt-path MTP K/V in batches (`Prefill::draft_kv`).**
What/How: the draft layer's prompt K/V built inside the main prefill pass (borrowing its GEMMs/scratch), so
first-token drafting doesn't re-run the draft layer over the whole prompt on its own stream (`mtp.hpp:91-97`,
`prefill.hpp:45-52`); not bit-identical to the standalone pass but never changes target tokens' logits.
Cerebro: fold into `ce/prefill/` when MTP arrives. Channel: Beta.

**D-06 should_stop / keepalives / cancel.**
What/How: `should_stop` checked before each chunk for cancellation (`prefill.hpp:134-136`); the server streams
keep-alives during a long prompt read, reports progress every 15 s, and stopping the HTTP request truly stops
the engine (`DETAILS.md:532-536`).
Cerebro: cancelation token → engine; channel: Stable. Toggle: "Pressing stop during a long prompt read works
instantly."

### E. KV cache: quantization, streaming tiering, conversation cache

**E-01 KV formats: fp16 / int8+scale / q4_0+Hadamard / k8v4.**
What/How: int8 with fp16 scale per 64 values is the default above 4K; q4_0 Halves KV memory with a Hadamard
rotation before 4-bit rounding: ~4% faster at 128K but +8-12% perplexity on long docs (`DETAILS.md:76-79`);
k8v4 keeps K at 8 bits and V as rotated 4-bit: 23% less KV memory ⇒ more experts fit in VRAM; RTX 3090 Coder at
198K: 85→99 tok/s, prompts 2-5% slower; does not stream (`DETAILS.md:81-85`). Kernels in
`src/kernels/cuda/kv_q4.cu`, `kv_q8.cu` (parity: `kernels/kv_q4_parity.cpp` etc.).
Flags: `--kv fp16|int8|q4_0|k8v4` (setup proposes based on context).
Gain: [H] on 8 GB with 128K context the KV cache is a big share; q8v + resident tiering (E-02) is the
combination. Confidence: medium. Cerebro: `ce/kv/quant.rs` (Had rotation is an online multiply; keep the
rotation's quality A/B harness). Channel: q8 Beta; q4_0/k8v4 Experimental.
Toggle: "Stores the model's memory notes in fewer bits so more of it fits on the graphics card."

**E-02 KV streaming / resident tiering (`--kv-resident N`).**
What/How: past a context threshold (default 64K, N = resident cells), each streamed QSA layer keeps its
AUTHORITATIVE K/V in device-mapped pinned host memory, identity layout; VRAM holds n_slots pages (page =
4 cells × all heads × K+V); the identity page table becomes a RESIDENCY MAP and three attached kernels resolve
it inside the captured graph: (1) walk the per-query selected block ids marking resident blocks with an epoch +
clock reference bit and claiming misses CAS-style into a miss list; (2) pick victims by CLOCK second-chance,
never taking a slot this call uses; (3) copy the missed blocks from the host copy (zero-copy over PCIe) into
their slots. Writers always write the host copy, and slots only if resident — so nothing goes stale after a
rejected draft or conversation resume (`kv_stream.hpp:1-24, 48-81`). Ring special case: an MTP drafter window
uses `block % n_slots` with no resolve and refills from the host copy on resume (`kv_stream.hpp:19-21`).
Flags: `--kv-resident N`, `--kv-streaming on|off`; off on WSL; needs pinned RAM 13.7 KB/context-token.
Gain: measured on the reference card 262K Q2_0: 50.9 → 62.6 tok/s (experts in VRAM 1,589 → 3,872) because VRAM
freed by KV goes to experts (`DETAILS.md:69-74`). [H] on our 8 GB at ≤128K: KV ≈ 1-2 GB at q8 — the same
trade:;
confidence: medium. Cerebro: `ce/kv/stream.rs` — table + 3 kernels per layer; parity = the same values, only
placement moves. Channel: Experimental.

**E-03 Conversation cache: checkpoints + prompt reuse.**
What/How: the engine keeps the live conversation state and up to 6 checkpoints (~118 MB each) in RAM, taken at
the start of each new assistant turn and every 16K prompt tokens; a request reuses the checkpoint whose token
prefix it EXACTLY matches (plus image hashes and steering mode); the oldest checkpoint (≈ system-prompt end)
is kept forever so new conversations sharing the prefix start there (`DETAILS.md:629-637`; matching:
`conversation_cache.hpp:144-157`, exact-token + image-identity equality `conversation_checked/1-3` header).
`--prompt-cache-root` additionally checkpoints the end of a ≥ 2,048-token system prompt on a fresh read
(#62/#65). Engine flags: `--prompt-cache N` (0=off also disables parking), `--prompt-cache-every N`,
`--prompt-cache-root N`, `--turn-token ID`.
Gain: THE agent-amortization win: Codex CLI on Strata reused ~96-97% of a 9,443-token prompt per tool-loop
turn (`DETAILS.md:757-760`); our analysis doc shows 97% reuse at 100K context on this PC. [H] identical order.
Cerebro: already spec'd (engine.md §1); adopt the "start-of-turn + every-16K + system-root" checkpoint policy
verbatim — those three points are cheap and cover agents. Channel: Stable.
Toggle: "Follow-up turns only read the new text instead of the whole history."

**E-04 Conversation parking (multiple parked chats in RAM).**
What/How: `--conversation-cache-mib B --conversation-cache-slots S` parks whole SavedConversation images
(running state, checkpoints, used K/V pages, draft K/V) in a bounded RAM budget; disabled by default; a swap
retains unchanged K/V pages (no re-copy), growth appends storage; rewinds refresh affected pages; a byte
budget counts the incoming snapshot during the switch; oldest-first eviction (`DETAILS.md:639-684`;
`conversation_cache.hpp:122-281` — budget accounting `can_fit/make_room`, parked-LRU deque).
Memory guard: `--conversation-cache-min-free-mib 2560` requires free physical RAM (Windows
`GlobalMemoryStatusEx`, Linux `MemAvailable`) before and after capture.
Cerebro: same shape; our radix-tree prefix cache (engine.md) subsumes checkpoint matching — the parking
budget + min-free guard is what to add (`ce/session/park.rs`). Channel: Beta.
Toggle: "Keeps several conversations warm in RAM so switching chats doesn't re-read everything."

**E-05 drop_superseded eviction.**
What/How: when parking an updated copy of a conversation the older superseded copy of the same chat is dropped if its deepest
checkpoint is contained in the outgoing chain (agent loops otherwise evict the parent after N turns);
sub-chats sharing only the system-prompt root are kept (`conversation_cache.hpp:233-264` — #342).
Cerebro: same rule in the prefix cache. Channel: Beta.

**E-06 Layer-split parking (multi-GPU) — skip (deferred, A-15).**

**E-07 Effort-position trick (`effort_position: end`).**
What/How: the reasoning-effort instruction sits first in the prompt ⇒ changing only the effort re-reads the
whole conversation; putting a non-default effort in a trailing system turn and keeping the engine's checkpoint
in front of it keeps the cache across effort switches (`DETAILS.md:520-527`, `server.py:1394-1417`).
Cerebro: app-level; matches our "message-boundary checkpoints for agents" story. Channel: Beta.

**E-08 Repeat-stop (`repeat_stop_tokens`).**
What/How: 256 identical tokens in a row ends the reply `finish_reason: length` + user-visible note; config key
sets the run length (`DETAILS.md:516-519`). Cerebro: server-side; Stable. Toggle: "Ends a model that repeats
one token forever, with an explanation."

### F. Resilience, lifecycle, serving, monitoring

**F-01 Engine watchdog, stall reports, crash recovery.**
What/How: pool-side 60 s stall watchdog prints per-thread states + memory + page faults + Windows dump
(`strata-stall-<pid>.dmp`) on stall (`pool.hpp:202-204`; `DETAILS.md:1013`); the SERVER ends the engine after
300 s of silence during a request (`engine_silence_s`), errors the request, and the next request starts the
engine again (issue #481, `server.py:87, 1111-1215, 1417-1424`); on death the server restarts per request with
the reason guessed from the log tail (`server.py:399, 535`).
Flags/config: `engine_silence_s`, `STRATA_WATCHDOG_S`.
Cerebro: `cerebro-supervisor` already has Job Objects; add per-thread pool diag + a silence timer with
progress-aware chunk allowance (`server.py:1128-1130` scales the silence budget by the prompt chunk time —
worth copying verbatim as a spec). Channel: Stable.

**F-02 QUIT / unload protocol.**
What/How: `QUIT` over stdin; the engine frees its own memory (unpinning tens of GB takes time), the server
ends the stdin reader after EOF, waits for exit before releasing handles (`DETAILS.md:1089-1091`,
`server.py:1227-1264`). `POST /unload` 409 while a request runs.
Cerebro: same discipline in the supervisor. Channel: Stable.

**F-03 Idle unload / min-free-VRAM / before-load / lazy load.**
What/How: `--idle-unload N` unloads after N idle s and reloads on the next request (files stay in the OS cache
so reload ≈ seconds; measured 4.6 s text after unload); `--min-free-vram-mib` refuses to load (503 naming the
cause) when a game took VRAM; `--before-load cmd` hooks a pre-load command; `"lazy_load": true` starts the API
without the engine and loads on first request (text-only) (`DETAILS.md:424-442, 1076-1087`).
Cerebro: maps 1:1 to our "share GPU" product line; the 503-with-explanation and lazy text-only guard are
nice UX rules to copy. Channel: Stable. Toggle (bundle): "Loads the model only when needed and waits if
another app needs the GPU first."

**F-04 VRAM elastic API (`POST /v1/vram`)** — see A-16.

**F-05 Telemetry: 1 Hz sampler with no hard dependencies.**
What/How: `serve/telemetry.py` (363 lines): NVML loaded via ctypes from the driver's `nvml.dll` (name, VRAM,
temp, power+limit, PCIe gen/width/throughput), AMD via sysfs/ADLX where present, CPU/RAM/disk via psutil or OS
fallbacks; anything unreadable → None and never stops the server (strata-analysis). Sampled values ride
`GET /metrics` once per second with `live`, `hardware`, `hardware_static`, `engine`, `history` (60 samples),
`requests`, `totals`, `conversation_cache`.
Cerebro: our telemetry crate already spec'd (analysis task 1) — copy the "never fails the server" rule and
the metrics JSON shape. Channel: Stable.

**F-06 Request capture / API monitor.**
What/How: opt-in (`api_monitor: true`); retains newest 100 requests in memory with 256K-char per-field caps,
load/queue/first-token timings, tokens, response format; `/api/requests` list + detail; headers NOT recorded;
documented as sensitive (`DETAILS.md:1112-1127`). Cerebro: `ce-server` request store + redaction toggle —
matches our monitor story; ADD what Strata lacks: history > 60 s (we keep 24 h downsampled), alerts, per-
session breakdown (analysis doc tasks). Channel: Beta (off default).

**F-07 API surface + security posture.**
What/How: endpoints `/v1/chat/completions`, `/v1/messages`, `/v1/responses` (stateless, event stream incl.
reasoning_text deltas), `/v1/models`, `/props`, `/slots`, `/health`, `/metrics`, `/mcp`, `/config`,
`/api/requests`, `/unload`/`/load`, `/v1/vram`. Security: loopback-only unless configured; API key; Host
allow-list (DNS-rebinding guard: only known hostnames/localhost/IPs when no key configured, 403 naming the
setting); Origin checks for browser POSTs without key (chrome-extension://` etc. allowed, `Origin: null`
refused); CORS opt-in `["*"]` only sensible with key; TLS/keep-alives for streamed responses
(`DETAILS.md:469-627`).
Cerebro: server parity is already planned (analysis tasks 5-6); adopt the Host/Origin matrix as the reference
spec of "loopback product server". Channel: Stable.

**F-08 Structured output = prompt-and-validate, not constrained decoding.**
What/How: `json_object`/`json_schema` responses are generated then validated with jsonschema (if installed);
strict root-object rule, 502 `structured_output_failed` with buffered SSE (content only after validation),
explicit refusal when combined with tools/MCP (`DETAILS.md:1093-1110`).
Cerebro: same scope decision initially (cheap, safe); a real GBNF grammar goes in later behind the sampler.
Channel: Stable for v1 semantics. Toggle: "Checks the answer actually parses as JSON instead of pretending."

**F-09 MCP both directions.**
What/How: (a) the chat page can give the model tools from MCP servers (`mcp_servers` config, stdio + streamable
HTTP, tool-call round cap 8, 60 s timeout, 20K-char truncation, security warnings); (b) `tools/strata_mcp.py`
exposes Strata's own lifecycle (status/list/install/start/stop/logs/speed/connect) to outside assistants
(`DETAILS.md:762-860`).
Cerebro: both directions already on our product plan (engine.md positioning; analysis doc task 6). Copy the
per-tool caps and the "only the chat page" restriction as defaults. Channel: Stable.

**F-10 Windows Job Object (`serve/winjob.py`).**
What/How: engine + vision helper live in one Job Object so child processes die together on crash (test:
`serve/test_winjob.py`). UNVERIFIED beyond tests read-by-name; mechanism standard. Cerebro: supervisor
already does this. Channel: Stable.

**F-11 Web-page config view (`serve/runconfig.py`).**
What/How: a small whitelist of documented run-config keys (sampling defaults, budgets, aliases, idle unload,
vram reserve...) readable/editable in-page with per-type validation, deep-copy atomic save, .bak
(`runconfig.py:19-42, 121-179`). Other keys never touched, so a save can't grow the blast radius.
Cerebro: same pattern for our settings.toml UI (editability scope fixed by an explicit whitelist).
Channel: Stable.

**F-12 The `--calibrate` machine-tuning loop.**
What/How: measures output speed per candidate for three settings that depend on the PC more than the model
(`pcie-frac`, `spec-min-p`, `pool-workers`), keeps a setting only when > 3% faster, stores per PC+model
(`DETAILS.md:361-370`). Direct template for Cerebro Turbo's tuner (turbo-system.md). Channel: Beta.
Toggle: "Runs a short benchmark and keeps each speed setting only if it is faster on your PC."

### G. Native packs, arch-specfic compute, PLE n-gram table

**G-01 Mixed-quant "native packs" (`--native-dense-gguf`, `--native-head-gguf`, `--expert-profile`).**
What/How: a release pack ships different quant families per role: dense attention weights, output head, and
per-expert blobs (e.g. IQ3_S: IQ2_S-IQ4_XS gate/up + IQ4_NL/Q2_0 down, Coder: 1.89 bpw of ORIGINAL params with
kept-expert mapping file `rco-allocation.txt`), with `native_experts.txt` naming every shard/offset
(`DETAILS.md:255-264`; `strata-ud-iq4_xs.json` shows the local config wiring the flags — mixed quant packs
built by `tools/iq_pack.py`). Layout checks per tensor (name/type/shape/offset) happen at load (`layout.hpp:11-33,
85-96`).
Cerebro: our "signature quantisation: UD IQ quants natively" (engine.md product rules) is the same strategy for
the GGUF spine: keep per-role quant choices free, verify at load, never fail late. Channel: Beta (depends on
pack tooling; a pack-builder `ce-quant`).
Toggle: "Uses smaller numbers for the parts of the model where it costs almost nothing."

**G-02 CPU i-quant AVX2 kernels + VNNI.**
What/How: expert GEMV runs IQ quants directly on AVX2 with ggml-derived dequant tables (transcribed WITH
attribution — llama.cpp MIT, `DETAILS.md:1065-1069`); the activation is quantized once (Q8_0 with fp32 scales)
and shared across the batch (`pool.hpp:44-49`). Parity suites per format (`iq_avx2.cpp` + `kernels/*parity*`).
Cerebro: our AVX2 expert kernels exist; the multi-token shared-activation trick
(pool.hpp:44-49) is the missing piece (A-13). Channel: Stable.

**G-03 QSA + GDN hybrid mixers (decoder-side kernel zoo).**
What/How: 36 GDN layers (gated DeltaNet linear attention) and 12 QSA layers (full attention with a native
indexer: small-key scoring selects which cells to attend per query) — decode attention cost amortized by
selection (`include/strata/kernels/qsa.hpp` 369 lines; `native_qsa_indexer.cu` etc.). Relevant only when
Cerebro supports Qwen3.8 (M6) — adopt the indexer dataflow then. Channel: M6.

**G-04 PLE n-gram table on SSD (decoder component, per-token rows).**
What/How: a 28.8 GB per-layer token-embedding table (IQ4_NL, 320,001,536 rows of 160 = 90 B) hashed over the
last 3 token ids into 16 rows/token: XOR-hash with per-position multipliers and non-power-of-two vocab sizes,
EOS cut propagates forward, missing predecessor = −1 (`ngram.hpp:30-82`); the reader does UNBUFFERED 4 KiB
direct reads (Windows overlapped FILE_FLAG_NO_BUFFERING) with a bounded 1M-row (~95 MB) cache, 256 in-flight,
its own I/O thread, issue/collect split so the reads overlap the first layers; a live mapping on the same file
SERIALIZES direct reads (open drops its map — 311 µs → 1,575 µs lesson, `ngram.hpp:105-128, 157-183`;
implementation `src/ngram/ple_reader.cpp` 547 lines). One batched gather for whole prompt chunks
(`ngram.hpp:148-152`). On a rotational disk warns and moves to `--ple-io ram`.`madvise(MADV_RANDOM)` honesty
note: Windows has no equivalent so the engine documents that instead of pretending (`ngram.hpp:202-214`).
Gain: this is model-specific (Qwen3.8-Flash-Next PLE); the TRANSFERABLE design is the direct-I/O row reader
with a bounded cache: [H] small model-level win when Cerebro supports the model; the I/O pattern helps any
cold weight tier. Cerebro: `ce/io/direct_rows.rs`. Channel: G (M6).

**G-05 Fused prompt kernels + thread-block clusters + GPU stage profiling.**
What/How: prompt kernels fused per role-group; decode block-selection + argmax on thread-block clusters
(sm_90; our sm_86 keeps prior kernels); `STRATA_VERIFY_PROFILE` per-stage GPU timing lines; `--gpu-stages`
one-token per-layer replay (`DETAILS.md:22-30, 193-200`). Cerebro: adopt the stage-timing output as engine
debug + Monitor source. Channel: Beta.

### H. Web insights (2025-2026) that map onto this stack

- Model-native MTP as in-region prefill (cite uses `tools/mtp_probe.py` above). Related art: DeepSeek-V3 MTP
  (arXiv 2412.19437, widely known, UNVERIFIED here), Qwen MTP layers.
- Prompt-lookup/suffix drafting: ApexAlister/prompt-lookup-decoding (GitHub, no arXiv paper — UNVERIFIED id).
- KV quant + Hadamard rotation: QServe W4A8KV4 (arXiv 2405.04532); YARN: arXiv 2309.00071.
- Mooncake KV-cache disaggregation irrelevant locally, but prefix-cache-first scheduling matches E-03
  (arXiv 2407.00079; UNVERIFIED at this writing).
- Expert-offload 2×2 dial: Fiddler (arXiv 2402.05971), Mixtral-offloading layer-wise LRU+speculative expert
  prefetch (arXiv 2401.04088) — Strata's tier policy supersedes both on the desktop: occupancy-first hot expert
  cache + CPU-in-place compute + PCIe share, adjusted to the machine.

## 3. Where Cerebro can beat Strata on THIS PC

Strata's gaps on this exact hardware (32 GB RAM, 8 GB VRAM, 6P+8E, NVMe, auto-config), with the mechanism
and Cerebro map:

1. **First turns are slow without a profile** (`A-04`): `--expert-profile` only ships for Strata's own
   packs; for other packs users get no hot-expert ranking until the adaptive tier has trained (measured
   hit-rate 31.3-48.4% on Q3-Coder-30B prompts). Cerebro: `ce-quant` pack tool writes a profile at pack
   build time (S-01); Turbo calibrates post-install (`F-12`), so day-one decode matches steady-state.
2. **Adaptation pauses the answer** (`A-03`): Strata's tier updates are synchronous by default (async is a
   flag most users won't set). Cerebro: make the async swap the DEFAULT with no flag, since our scheduler
   (Turbo) owns the threads anyway.
3. **Slow cold loads** (`A-09`): pinned arena copy takes ~1.4-1.5 GB/s from NVMe measured in Strata (below
   normal Task Scheduler priority made it 24x slower — an avoidable user footgun). Cerebro: load files at
   normal priority, mmap-first, and pin only the hot subset (profile ranked) with the rest following the
   resident budget; plus `_DIRECT` I/O for the SSD tier.
4. **Slow model switching**: Strata forces full unload + reload; idle-unload measured 4.6 s reload but full
   first-time load is minutes at 1.5 GB/s. Cerebro: keep quantized packs READ-ONLY and mmap SHARED across
   engines so a model switch is page-flip rather than recopy (our multi-model slots, engine.md); Strata can't
   do this because its ArenaExpertSource needs a private pinned copy.
5. **The parallel-session hole** (`B-05`): Strata's batch slots are opt-in (`"parallel": N`) and hidden from
   the API surface's defaults. Cerebro product bet (2-4 parallel local models, engine.md) subsumes it: slot
   batching for ONE model + N model slots, both first-class with per-slot telemetry.
6. **Cerebro already has**: radix-tree prefix cache for agent loops (engine.md §1 vs `E-03`), a monitor with
   bottleneck verdicts (turbo-system.md vs `B-04`), a settings UI with per-PC calibration (turbo-system.md
   vs `F-12`), CUDA-graph-per-token instead of per-layer (analysis doc next-step 3 vs `B-03` — Strata's
   graph-per-layer is a portability choice; graph-per-token on sm_86 cuts the remaining 5.2 µs × 48 handoffs
   too), native YARN in-engine RoPE scaling (our euler rope, no server-side hack).
7. **Cerebro gaps Strata has already solved (don't reinvent)**: the slot exchange trick for consistent RAM
   mirrors (`A-03`, `expert_source.hpp:473-498`), the three transport modes under one `pcie-frac` dial
   (`A-06`), the conversation-parking budget with min-free-RAM guard (`E-04`), the decode-timing
   breakdown as an env flag (`DETAILS.md:193-200`) — this diagnostic is what the Monitor should consume
   before we re-derive it.

Priority order for Cerebro engineering (from the engine report's identified gaps + above): (1) expert
profile at pack time (S-01, gain ~1.5x first-turn MTP decode), (2) int8-mma prefill GEMM (D-04, biggest
prefill term), (3) spec pipeline: MTP + controller + parallel verify + graph-per-token (C-01/C-05/B-03,
decode 13.8 → ~19-22 CPU and 33-36 → ~45-55 GPU), (4) pinned arena + topology (A-09/A-11, closes CPU
37-41→26-28 gap), (5) conversation parking + KV tiers (E-04/E-01/E-02), (6) pcie-frac calibration (A-06).

## Appendix: file/line map

| Area | File (in `C:\Users\noahm\strata\`) | Read level |
|---|---|---|
| Session loop, capture, doorbell, scratch, rings | `include/strata/core/session.hpp:127-411` | full |
| Verify window, commit, chain, slots, split | `include/strata/core/verify.hpp:1-262` | full |
| MTP drafter, graphs, coupled draft | `include/strata/core/mtp.hpp:1-109` | full |
| Expert source contract, split, streams, PLE, arena, lookahead | `include/strata/core/expert_source.hpp:1-303 (selected)` | full |
| Expert cache STRP + slots + VMM | `src/core/expert_cache.cpp:23-694` (selected) + header comments | indexed |
| CPU pool epoch claim, rings, split, topology | `include/strata/kernels/cpu/pool.hpp:1-270` | full |
| Speculation controller, draft policy, suffix drafter | `src/spec/controller.cpp`, `draft_policy.cpp`, `suffix_drafter.cpp` | full |
| Chunked prefill + ring + stats | `include/strata/prefill/prefill.hpp:1-137` | full |
| KV streaming / residency map / clock | `include/strata/kernels/kv_stream.hpp:1-81` | full |
| Conversation cache / parking | `include/strata/core/conversation_cache.hpp:1-281` (selected) | indexed |
| PLE ngram row reader | `include/strata/kernels/ngram.hpp:1-214` | full |
| not read here | `weights.cpp`(479L), `layer.cpp`(1349L), `session.cpp`(970L), `expert_source.cpp`(2985L), `prefill.cpp`(3084L), `ngram/ple_reader.cpp`(547L), `kernels/qsa.hpp`(369L), CUDA kernel files, `server.py` beyond grep, `telemetry.py`(363L), `mcp.py`, `responses.py`, `ledger.py`, `winjob.py`, `structured.py`, test files, `tools/` | — |

Source-score note: claims with `file:line` = source code; `DETAILS.md:NNN` = shipped doc (measured numbers,
kept because the doc names its measurement rigs); README/issue numbers = GitHub links as stated at
`https://github.com/Niko1221/Strata`.




