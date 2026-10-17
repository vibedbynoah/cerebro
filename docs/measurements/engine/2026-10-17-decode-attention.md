# Decode attention for grouped-query models (2026-10-17)

Machine A (i9-12900H, RTX 3070 Ti Laptop 8 GB, 46 SMs, power-limited). Qwen3-0.6B Q8_0 (16 query
heads, 8 KV heads, head size 128), `examples/bench.rs --reps 3 --decode 256`, greedy decode tok/s
(median). llama.cpp is the official release b10751, `llama-bench -ngl 99 -fa 1 -n 128 -d ...`.

| Context | Before (attn_partial) | After (attn_decode_gqa_128_2) | llama.cpp b10751 |
|---:|---:|---:|---:|
| 32 | 271 | **331** | 307 |
| 512 | 227 | **331** | 293 |
| 2048 | 221 | **297** | 257 |
| 4096 | 201 | **250** | 220 |

What changed (kernels.cu `attn_decode_gqa_t<HD, G>`): one block per KV head and split serves the
whole query group, so K/V are read once per group; 8 lanes share a K row (one full 32-byte sector
each, 3-shuffle reduction); each lane owns contiguous V dims (one load per row); splits sized to the
GPU (8 blocks per SM) and derived from the live position, so the launch stays graph-captured.
Nsight Compute on the first attempt (lane-per-row K reads) showed the reason it lost at long
context: 105 registers (33% occupancy), half of every sector wasted (L2 hit 49%), 124 us per layer
at 4096 positions. Switch: `gpu_decode_gqa` (Beta, on), `CEREBRO_NO_DECODE_GQA=1` turns it off.

Accuracy: `gpu::tests::grouped_decode_attention_matches_the_reference` compares both kernels with
an f64 reference on random queries and an f16 cache at 1-4100 positions: max error 1e-7 or less for
both, within 2x of each other. Greedy tokens vs the CPU reference over 64 tokens at 10 prompt
lengths: mean 0.68 (after) vs 0.73 (before); both diverge on the same prompts (f16 cache and int8
activations against an f32 CPU path at near-tied logits), and match 1.00 at 300/700/2048.
