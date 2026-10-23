# Placement sweep: Cerebro vs llama.cpp

Same machine, models and prompt (512 tokens read, then greedy writing). Speeds in tokens per second;
the ratio is Cerebro / llama.cpp, above 1 = Cerebro faster. llama.cpp has no drive tier, so the
drive rows have no comparison. `-` = not run or failed (see the raw files).

## qwen3-14b-q4_k_m.gguf

| Placement | GPU layers | Drive layers | Read: Cerebro | Read: llama.cpp | Read ratio | Write: Cerebro | Write: llama.cpp | Write ratio |
|---|---:|---:|---:|---:|---|---:|---:|---|
| Processor (RAM) | 0 | 0 | 17.7 | 28.0 | 0.63x (slower) | 4.43 | 5.30 | 0.84x (slower) |
| Graphics card | 33 | 0 | 59.0 | 109.8 | 0.54x (slower) | 8.54 | 7.12 | **1.20x** |
| Drive (SSD) | 0 | 40 | 7.4 | - |  | 0.33 | - |  |
| Processor + graphics card | 20 | 0 | 29.1 | 628.5 | 0.05x (slower) | 4.54 | 6.82 | 0.67x (slower) |
| Processor + drive | 0 | 20 | 12.4 | - |  | 0.70 | - |  |
| Graphics card + drive | 20 | 20 | 15.4 | - |  | 0.70 | - |  |
| All three | 13 | 13 | 12.9 | - |  | 0.88 | - |  |
| Integrated GPU | | | failed: o wait using a submission index (41) that has not been returned by a successful  | | | | | |
| Integrated GPU + processor | 20 | 5 | 4.8 | 50.9 | 0.09x (slower) | 1.67 | 4.26 | 0.39x (slower) |

## qwen3.8-27b-q4_k_m.gguf

| Placement | GPU layers | Drive layers | Read: Cerebro | Read: llama.cpp | Read ratio | Write: Cerebro | Write: llama.cpp | Write ratio |
|---|---:|---:|---:|---:|---|---:|---:|---|
| Processor (RAM) | | | failed: ktracedrive tier: 5 of 20 layers (1.0 GiB) read from the drive on every pass; 3. | | | | | |
| Graphics card | 28 | 0 | 4.6 | 20.9 | 0.22x (slower) | 3.38 | 0.59 | **5.69x** |
| Drive (SSD) | | | failed: ktracedrive tier: 5 of 20 layers (1.0 GiB) read from the drive on every pass; 3. | | | | | |
| Processor + graphics card | 28 | 0 | 4.4 | 64.9 | 0.07x (slower) | 3.31 | 3.85 | 0.86x (slower) |
| Processor + drive | | | failed: ktracedrive tier: 5 of 20 layers (1.0 GiB) read from the drive on every pass; 3. | | | | | |
| Graphics card + drive | | | failed: ktracedrive tier: 5 of 20 layers (1.0 GiB) read from the drive on every pass; 3. | | | | | |
| All three | | | failed: ktracedrive tier: 5 of 20 layers (1.0 GiB) read from the drive on every pass; 3. | | | | | |

## qwen3-0.6b-q8_0.gguf

| Placement | GPU layers | Drive layers | Read: Cerebro | Read: llama.cpp | Read ratio | Write: Cerebro | Write: llama.cpp | Write ratio |
|---|---:|---:|---:|---:|---|---:|---:|---|
| Processor (RAM) | 0 | 0 | 366.3 | 392.7 | 0.93x (slower) | 53.92 | 72.40 | 0.74x (slower) |
| Graphics card | 28 | 0 | 22,979.2 | 13,376.7 | **1.72x** | 327.61 | 314.47 | **1.04x** |
| Drive (SSD) | 0 | 28 | 131.3 | - |  | 4.23 | - |  |
| Processor + graphics card | 14 | 0 | 743.1 | 6,288.5 | 0.12x (slower) | 78.31 | 123.51 | 0.63x (slower) |
| Processor + drive | 0 | 14 | 276.4 | - |  | 11.15 | - |  |
| Graphics card + drive | 14 | 14 | 465.0 | - |  | 10.26 | - |  |
| All three | 9 | 9 | 425.9 | - |  | 15.23 | - |  |
| Integrated GPU | 28 | 0 | 142.8 | 1,000.9 | 0.14x (slower) | 43.40 | 50.83 | 0.85x (slower) |
| Integrated GPU + processor | 14 | 0 | 201.1 | 668.1 | 0.30x (slower) | 38.10 | 50.62 | 0.75x (slower) |

## qwen3-8b-q4_k_m.gguf

| Placement | GPU layers | Drive layers | Read: Cerebro | Read: llama.cpp | Read ratio | Write: Cerebro | Write: llama.cpp | Write ratio |
|---|---:|---:|---:|---:|---|---:|---:|---|
| Processor (RAM) | 0 | 0 | 38.0 | 47.2 | 0.80x (slower) | 8.55 | 8.37 | **1.02x** |
| Graphics card | 36 | 0 | 147.3 | 2,321.9 | 0.06x (slower) | 64.13 | 62.72 | **1.02x** |
| Drive (SSD) | 0 | 36 | 11.5 | - |  | 0.38 | - |  |
| Processor + graphics card | 18 | 0 | 55.3 | 1,145.9 | 0.05x (slower) | 10.92 | 12.31 | 0.89x (slower) |
| Processor + drive | 0 | 18 | 20.2 | - |  | 0.87 | - |  |
| Graphics card + drive | 18 | 18 | 26.5 | - |  | 0.98 | - |  |
| All three | 12 | 12 | 30.0 | - |  | 1.30 | - |  |
| Integrated GPU | 36 | 0 | 4.4 | 82.4 | 0.05x (slower) | 3.93 | 5.93 | 0.66x (slower) |
| Integrated GPU + processor | 18 | 18 | 7.3 | 75.5 | 0.10x (slower) | 0.84 | 4.70 | 0.18x (slower) |

**Head-to-head: Cerebro faster in 6, slower in 26 of 32 comparisons.**
