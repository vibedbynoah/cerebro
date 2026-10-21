# Images in, speech in, speech out, and the Tune screen on bigger models (2026-10-18 to 19)

Machine: Zenbook, i9-12900H, RTX 3070 Ti Laptop (8 GiB), lid open. Every number below was
measured on this machine with Cerebro Engine; nothing is estimated.

## Vision (Qwen3.5-0.8B + its F16 mmproj, `qwen3vl_merger`)

Checked against the reference implementation's tensor dumps (`visioncheck`): rainbow 64,
gray 96 and checkerboard 128 images, every stage within 2.6% (the reference's own GPU and CPU
runs differ by 1.2% on the checkerboard).

| Encoder variant | 768x768 encode | Matrix products | Agreement with f32 |
|---|---|---|---|
| 8-bit weights and activations (VNNI) | 2.22 s | 0.80 s | cosine 0.992 |
| f32, 3x4 register-blocked | 1.95 s | 0.66 s | (reference) |

f32 is both more exact and faster, so the encoder runs in f32.

End to end (`visionchat`, test photo of a newspaper front page): described correctly on the
processor and on CUDA; CUDA reads the 321-token prompt at 4,272 tok/s.

## Speech to text (Qwen3-ASR-0.6B, `qwen3a`)

Two recordings synthesized with Windows' voice (6.9 s, 8.5 s) transcribe exactly as the
reference does, with the same audio token counts (91 and 117), on the processor and on CUDA.
8.5 s of audio encodes in 0.58 s. Through the app path: 3.5 s including loading the model.

## Text to speech (Qwen3-TTS-12Hz-1.7B-Base, Q8_0)

Objective check: speech made by the engine is transcribed back by the engine's speech recogniser.

| Sentence | Audio | Heard back |
|---|---|---|
| The quick brown fox jumps over the lazy dog. | 2.96 s | word for word |
| Please remind me to buy milk, eggs and bread tomorrow ... trip to Lisbon. (26 words) | 7.1 s | word for word |

Speed for the 26-word sentence (all on the processor): 14.1 s in total. Talker 3.8 s, code
predictor 4.7 s (13.3 s when its weights were read as f32), audio decoder 4.1 s (6.5 s before
the patch building and Snake activations were parallelised). That is 2x slower than real time;
moving the talker and decoder to the GPU is the next step.

## Tune screen on bigger models

| Model | Placement | Outcome |
|---|---|---|
| Qwen3-0.6B Q8_0 | all on the GPU | speculation off wins: 242.2 vs 206.7 tok/s at 4K depth, KL 0.0 |
| Qwen3.5-4B Q8_0 | all 32 layers on the GPU | automatic kept: speculation off +0%; 68 tok/s, prefill 2,086 tok/s |
| Qwen3-14B Q4_K_M | 30 of 40 layers on the GPU | first run exposed two bugs (below); see the rerun |

Bugs the 14B run found and their fixes:
- writing at depth read 1,158 tok/s: the 3 s budget included reading a 4K prompt, so one token
  was timed. Writing is now timed from the first written token, at least 64 tokens.
- "one more layer on the GPU" asked for 31 layers, got 30 (the planner's limit), and "won" by
  8% against the identical baseline on noise. A forced placement the planner cannot honour now
  fails that candidate.

Speculation: 0.6B +17% with it off, 4B and 14B no measurable difference. The default stays
(no evidence it hurts the models people run at size).

## Tool calling

The real tool loop (weather tool, approval, result used in the answer) passes on Qwen3.5-4B and
on Qwen3-14B (30 of 40 layers on the GPU).

## The GPU tensor backend (gtensor): images, vision, speech in and out

f16 tensor-core kernels for everything that is not a language model: implicit-GEMM convolution
(square, and 1-D causal/dilated), flash attention (head size 64, causal, windowed, grouped),
one-query attention for any head size, shifted-sum GroupNorm, fused one-token kernels, sampling
on the card. Same machine; the alternatives measured on the same GPU, warm, same inputs.

| Task | Cerebro, GPU | Cerebro, processor | Alternative on this GPU | Agreement |
|---|---|---|---|---|
| SD-Turbo 512x512, 1 step | 0.195 s | 15.3 s | diffusers fp16 CUDA: 0.224 s | cosine 1.000000 per stage vs diffusers |
| Vision encode, 640x480 photo | 26.6 ms | 754 ms | llama.cpp mtmd CUDA: 528 ms (one cold run) | cosine 0.999985 vs processor |
| Speech encode, 8.5 s | 52 ms | 557 ms | llama.cpp mtmd CUDA: 468 ms (one cold run) | cosine 1.000000; transcripts exact |
| Speech synthesis, 26 words | 2.0 s for 6.72 s of audio | 14.1 s | llama-tts CUDA: 3.12 s for 7.1 s | heard back word for word |

Diffusion stages, warm (Cerebro / diffusers): text encoder 10 / 20 ms, UNet 58 / 63 ms, VAE
127 / 123 ms. f16 accumulation over each 32-value slice (twice the rate on GeForce cards) gave
the last 7% with no measurable change to the image.

Speech synthesis by part, 84 frames: talker 0.68 s (whole-model CUDA session, kept between
sentences), code predictor 0.94 s (fused: 6 launches per layer, sampling on the card, one wait
per frame; was 2.2 s unfused, 4.8 s on the processor), audio decoder 0.03 s (was 4.29 s).
