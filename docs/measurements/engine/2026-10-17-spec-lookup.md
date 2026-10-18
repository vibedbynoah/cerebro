# Prompt-lookup speculation (switch `spec_lookup`, default 8 drafts, Beta)

Qwen3-0.6B Q8_0, whole GPU (3070 Ti Laptop), `examples/spec.rs` (chat template, 300-400 tokens).
Drafts: tokens that followed the latest earlier match of the last 2-3 tokens; verified in one batched
pass with speculative sampling (accept with the sampler's probability, else resample without it).

| prompt | temperature | off tok/s | on tok/s | drafts accepted |
|---|---|---|---|---|
| rewrite 2.5 KB of Rust code exactly | 0 | 322.9 | 508.3 | 231 / 325 |
| same | 0.7 | 283.1 | 421.7 | 231 / 325 |
| short story (nothing to copy) | 0 | 327.8 | 312.4 | 6 / 44 |
| same | 0.7 | 285.5 | 271.3 | 3 / 26 |

Copy text: identical output on and off. Free text: batched verification flips near-ties, so greedy text
can differ (CLAIMS.md: distribution-preserving, never "identical"). The -5% on free text at 300 tok/s
is the per-token lookup and the few rejected verifies; drafts are sized by recent acceptance and pause
(8 -> 64 tokens, doubling) after a full rejection. Hybrid (DeltaNet), MoE and split models do not
speculate yet: a rejected draft would need the recurrent state rolled back.
