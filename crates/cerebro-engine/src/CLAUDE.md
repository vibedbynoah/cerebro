# Engine source: reviewer checklist

Read `/REVIEW.md` for the speed table and priorities. Hardware: RTX 3070 Ti Laptop 8 GB (sm_86, ~384 GB/s
real memory peak, GeForce: half-rate FP16 with FP32 accumulate), i9-12900H (AVX2 + VNNI, no AVX-512),
32 GB RAM, Windows WDDM (pageable copies are synchronous; shared-memory fallback when VRAM over-commits).

- **Kernels** (`kernels.cu`, `kernels_mma.cu`): register spills (`ptxas -v`), uncoalesced loads, launch
  counts per layer, barriers per token. Any kernel change must give identical greedy output
  (`examples/greedy.rs`) on Qwen2.5-14B, a Qwen3.8 DeltaNet model and Gemma 4, and a measured speed-up
  (`examples/bench.rs`, nsys kernel table). Keep the old kernel behind a switch until proven.
- **Placement** (`placement.rs`, `gpu.rs` `load_with`): VRAM reserve (1/16), spill allowance, evictable
  head. Over-committing thrashes prompt reading; check both read and write numbers for every change.
- **Processor path** (`model.rs`, `quant.rs`): every DType must have a vectorised dot (BF16 once ran one
  value per call and cost 5x); `dot_any` falling through to `dot_row` on a big matrix is a finding.
- **DeepSeek-V4.1** (`ds41.rs`): processor only; check against llama.cpp b11982 `deepseek41.cpp`
  (attention sinks, Sinkhorn mixes, compressor visibility `(pos+1)/ratio`, YaRN on compressed layers,
  Engram hashing with wrapping u64 maths). Speed: no GPU path yet, prompts read token by token.
- **Stop** (`lib.rs` `STOP`): every long loop (load, prompt read) checks it at layer/chunk boundaries.
- **Memory tiers**: RAM keeps 1/32 free (`gguf::prefetch_room`), VRAM 1/16; experts page from the SSD.
