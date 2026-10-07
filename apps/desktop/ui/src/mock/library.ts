import type {
  DuplicateGroup,
  GgufEntry,
  LibraryModel,
  LoadParams,
  ModelArch,
  ModelDetail,
  ModelKind,
  ModelSource,
  Modality,
  QuirkFix,
  Sampler,
  TuneResult,
  Tweaks,
} from "../service/types";
import { estimateFit } from "../lib/fit";
import { MOCK_GPU, MOCK_HARDWARE, MOCK_LIVE_BASE, MOCK_MODELS } from "./fixtures";

/**
 * ILLUSTRATIVE DATA, NOT MEASUREMENTS. A generated library of 500+ model files for the Models
 * screen (layout, virtualized scrolling, search and filters). Model and publisher names are real
 * published ones so the list reads naturally; architectures are approximately as published; file
 * sizes are computed from parameter counts and typical bits per weight; every speed, tune result,
 * path, hash and quirk entry is invented and carries `provenance: "illustrative"`.
 */

interface Base {
  name: string;
  family: string;
  /** GGUF `general.architecture`. */
  ggufArch: string;
  params: string;
  /** Total parameters, billions. */
  billions: number;
  kind: ModelKind;
  arch: ModelArch;
  contextMax: number;
  /** Ollama-style tag stem, for blob paths and slugs. */
  slug: string;
  extraPublisher?: boolean;
}

const dense = (layers: number, kvHeads: number, headDim: number, embedding: number): ModelArch => ({ layers, kvHeads, headDim, embedding, experts: null, activeExperts: null });
const moe = (layers: number, kvHeads: number, headDim: number, embedding: number, experts: number, active: number): ModelArch => ({ layers, kvHeads, headDim, embedding, experts, activeExperts: active });

const K = 1024;
const BASES: readonly Base[] = [
  { name: "Qwen3-0.6B", family: "Qwen3", ggufArch: "qwen3", params: "0.6B", billions: 0.6, kind: "text", arch: dense(28, 8, 128, 1024), contextMax: 32 * K, slug: "qwen3-0.6b", extraPublisher: true },
  { name: "Qwen3-1.7B", family: "Qwen3", ggufArch: "qwen3", params: "1.7B", billions: 1.7, kind: "text", arch: dense(28, 8, 128, 2048), contextMax: 32 * K, slug: "qwen3-1.7b", extraPublisher: true },
  { name: "Qwen3-4B", family: "Qwen3", ggufArch: "qwen3", params: "4B", billions: 4, kind: "text", arch: dense(36, 8, 128, 2560), contextMax: 32 * K, slug: "qwen3-4b", extraPublisher: true },
  { name: "Qwen3-8B", family: "Qwen3", ggufArch: "qwen3", params: "8B", billions: 8.2, kind: "text", arch: dense(36, 8, 128, 4096), contextMax: 32 * K, slug: "qwen3-8b", extraPublisher: true },
  { name: "Qwen3-14B", family: "Qwen3", ggufArch: "qwen3", params: "14B", billions: 14.8, kind: "text", arch: dense(40, 8, 128, 5120), contextMax: 32 * K, slug: "qwen3-14b", extraPublisher: true },
  { name: "Qwen3-32B", family: "Qwen3", ggufArch: "qwen3", params: "32B", billions: 32.8, kind: "text", arch: dense(64, 8, 128, 5120), contextMax: 32 * K, slug: "qwen3-32b" },
  { name: "Qwen3-30B-A3B", family: "Qwen3", ggufArch: "qwen3moe", params: "30B-A3B", billions: 30.5, kind: "text", arch: moe(48, 4, 128, 2048, 128, 8), contextMax: 32 * K, slug: "qwen3-30b-a3b", extraPublisher: true },
  { name: "Qwen3-Coder-30B-A3B-Instruct", family: "Qwen3", ggufArch: "qwen3moe", params: "30B-A3B", billions: 30.5, kind: "text", arch: moe(48, 4, 128, 2048, 128, 8), contextMax: 256 * K, slug: "qwen3-coder-30b-a3b" },
  { name: "Qwen3-235B-A22B", family: "Qwen3", ggufArch: "qwen3moe", params: "235B-A22B", billions: 235, kind: "text", arch: moe(94, 4, 128, 4096, 128, 8), contextMax: 32 * K, slug: "qwen3-235b-a22b" },
  { name: "Gemma-3-1B-it", family: "Gemma", ggufArch: "gemma3", params: "1B", billions: 1, kind: "text", arch: dense(26, 1, 256, 1152), contextMax: 32 * K, slug: "gemma3-1b", extraPublisher: true },
  { name: "Gemma-3-4B-it", family: "Gemma", ggufArch: "gemma3", params: "4B", billions: 3.9, kind: "vision", arch: dense(34, 4, 256, 2560), contextMax: 128 * K, slug: "gemma3-4b", extraPublisher: true },
  { name: "Gemma-3-12B-it", family: "Gemma", ggufArch: "gemma3", params: "12B", billions: 11.8, kind: "vision", arch: dense(48, 8, 256, 3840), contextMax: 128 * K, slug: "gemma3-12b", extraPublisher: true },
  { name: "Gemma-3-27B-it", family: "Gemma", ggufArch: "gemma3", params: "27B", billions: 27, kind: "vision", arch: dense(62, 16, 128, 5376), contextMax: 128 * K, slug: "gemma3-27b" },
  { name: "Llama-3.2-1B-Instruct", family: "Llama", ggufArch: "llama", params: "1B", billions: 1.2, kind: "text", arch: dense(16, 8, 64, 2048), contextMax: 128 * K, slug: "llama3.2-1b" },
  { name: "Llama-3.2-3B-Instruct", family: "Llama", ggufArch: "llama", params: "3B", billions: 3.2, kind: "text", arch: dense(28, 8, 128, 3072), contextMax: 128 * K, slug: "llama3.2-3b", extraPublisher: true },
  { name: "Llama-3.1-8B-Instruct", family: "Llama", ggufArch: "llama", params: "8B", billions: 8, kind: "text", arch: dense(32, 8, 128, 4096), contextMax: 128 * K, slug: "llama3.1-8b", extraPublisher: true },
  { name: "Llama-3.3-70B-Instruct", family: "Llama", ggufArch: "llama", params: "70B", billions: 70.6, kind: "text", arch: dense(80, 8, 128, 8192), contextMax: 128 * K, slug: "llama3.3-70b" },
  { name: "Mistral-Nemo-Instruct-2407", family: "Mistral", ggufArch: "llama", params: "12B", billions: 12.2, kind: "text", arch: dense(40, 8, 128, 5120), contextMax: 128 * K, slug: "mistral-nemo-12b" },
  { name: "Phi-4", family: "Phi", ggufArch: "phi3", params: "14B", billions: 14.7, kind: "text", arch: dense(40, 10, 128, 5120), contextMax: 16 * K, slug: "phi4-14b" },
  { name: "Phi-4-mini-instruct", family: "Phi", ggufArch: "phi3", params: "3.8B", billions: 3.8, kind: "text", arch: dense(32, 8, 128, 3072), contextMax: 128 * K, slug: "phi4-mini" },
  { name: "DeepSeek-R1-Distill-Qwen-7B", family: "DeepSeek", ggufArch: "qwen2", params: "7B", billions: 7.6, kind: "text", arch: dense(28, 4, 128, 3584), contextMax: 128 * K, slug: "r1-distill-qwen-7b" },
  { name: "DeepSeek-R1-Distill-Qwen-14B", family: "DeepSeek", ggufArch: "qwen2", params: "14B", billions: 14.8, kind: "text", arch: dense(48, 8, 128, 5120), contextMax: 128 * K, slug: "r1-distill-qwen-14b" },
  { name: "DeepSeek-R1-Distill-Qwen-32B", family: "DeepSeek", ggufArch: "qwen2", params: "32B", billions: 32.8, kind: "text", arch: dense(64, 8, 128, 5120), contextMax: 128 * K, slug: "r1-distill-qwen-32b" },
  { name: "gpt-oss-20b", family: "gpt-oss", ggufArch: "gpt-oss", params: "21B-A3.6B", billions: 21, kind: "text", arch: moe(24, 8, 64, 2880, 32, 4), contextMax: 128 * K, slug: "gpt-oss-20b", extraPublisher: true },
  { name: "gpt-oss-120b", family: "gpt-oss", ggufArch: "gpt-oss", params: "117B-A5.1B", billions: 117, kind: "text", arch: moe(36, 8, 64, 2880, 128, 4), contextMax: 128 * K, slug: "gpt-oss-120b" },
  { name: "GLM-4.5-Air", family: "GLM", ggufArch: "glm4moe", params: "106B-A12B", billions: 106, kind: "text", arch: moe(46, 8, 128, 4096, 128, 8), contextMax: 128 * K, slug: "glm-4.5-air" },
  { name: "Qwen2.5-VL-3B-Instruct", family: "Qwen2.5-VL", ggufArch: "qwen2vl", params: "3B", billions: 3.8, kind: "vision", arch: dense(36, 2, 128, 2048), contextMax: 128 * K, slug: "qwen2.5vl-3b" },
  { name: "Qwen2.5-VL-7B-Instruct", family: "Qwen2.5-VL", ggufArch: "qwen2vl", params: "7B", billions: 8.3, kind: "vision", arch: dense(28, 4, 128, 3584), contextMax: 128 * K, slug: "qwen2.5vl-7b", extraPublisher: true },
  { name: "Qwen2.5-VL-32B-Instruct", family: "Qwen2.5-VL", ggufArch: "qwen2vl", params: "32B", billions: 33.5, kind: "vision", arch: dense(64, 8, 128, 5120), contextMax: 128 * K, slug: "qwen2.5vl-32b" },
];

/** Speech and image models: each loads into its own engine beside the chat model. */
const ENGINE_BASES: readonly (Base & { quants: readonly string[]; ext: string })[] = [
  { name: "whisper-large-v3-turbo", family: "Whisper", ggufArch: "whisper", params: "809M", billions: 0.81, kind: "stt", arch: dense(4, 20, 64, 1280), contextMax: 448, slug: "whisper-large-v3-turbo", quants: ["Q5_0", "Q8_0", "F16"], ext: "bin" },
  { name: "whisper-small", family: "Whisper", ggufArch: "whisper", params: "244M", billions: 0.24, kind: "stt", arch: dense(12, 12, 64, 768), contextMax: 448, slug: "whisper-small", quants: ["Q5_1", "Q8_0", "F16"], ext: "bin" },
  { name: "whisper-base.en", family: "Whisper", ggufArch: "whisper", params: "74M", billions: 0.074, kind: "stt", arch: dense(6, 8, 64, 512), contextMax: 448, slug: "whisper-base-en", quants: ["Q5_1", "Q8_0", "F16"], ext: "bin" },
  { name: "OuteTTS-1.0-0.6B", family: "OuteTTS", ggufArch: "qwen3", params: "0.6B", billions: 0.6, kind: "tts", arch: dense(28, 8, 128, 1024), contextMax: 8 * K, slug: "outetts-1.0-0.6b", quants: ["Q4_K_M", "Q8_0", "F16"], ext: "gguf" },
  { name: "Orpheus-3B-0.1-ft", family: "Orpheus", ggufArch: "llama", params: "3B", billions: 3.3, kind: "tts", arch: dense(28, 8, 128, 3072), contextMax: 8 * K, slug: "orpheus-3b", quants: ["Q4_K_M", "Q8_0", "F16"], ext: "gguf" },
  { name: "FLUX.1-schnell", family: "FLUX", ggufArch: "flux", params: "12B", billions: 11.9, kind: "image", arch: dense(57, 24, 128, 3072), contextMax: 512, slug: "flux1-schnell", quants: ["Q4_K_S", "Q5_K_S", "Q8_0"], ext: "gguf" },
  { name: "stable-diffusion-3.5-medium", family: "SD3.5", ggufArch: "sd3", params: "2.5B", billions: 2.5, kind: "image", arch: dense(24, 24, 64, 1536), contextMax: 256, slug: "sd3.5-medium", quants: ["Q4_0", "Q8_0", "F16"], ext: "gguf" },
];

/** Typical bits per weight of each quantization, for file sizes. */
const BPW: Readonly<Record<string, number>> = {
  Q3_K_M: 3.91, IQ4_XS: 4.25, Q4_0: 4.55, Q4_K_S: 4.58, Q4_K_M: 4.85, Q5_0: 5.54, Q5_1: 6.04, Q5_K_S: 5.54, Q5_K_M: 5.69, Q6_K: 6.56, Q8_0: 8.5, F16: 16,
};
const TEXT_QUANTS = ["Q3_K_M", "IQ4_XS", "Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"] as const;
const PUBLISHERS = ["bartowski", "unsloth"] as const;
const EXTRA_PUBLISHER = "ggml-org";
const SOURCES: readonly ModelSource[] = ["cerebro", "lmstudio", "lmstudio", "ollama", "cerebro", "other"];

/** FNV-1a, 32-bit: deterministic pseudo-random values from an id. */
function hash32(text: string, seed = 0x811c9dc5): number {
  let h = seed >>> 0;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** A stable, fake 64-hex "hash" for an id. Not a real SHA-256 of anything. */
export function fakeSha(id: string): string {
  let out = "";
  for (let i = 0; out.length < 64; i++) out += hash32(id, 0x9e3779b9 + i * 0x1000193).toString(16).padStart(8, "0");
  return out.slice(0, 64);
}

const round1 = (n: number): number => Math.round(n * 10) / 10;
const sizeOf = (billions: number, quant: string): number => Math.max(0.1, round1((billions * 1e9 * (BPW[quant] ?? 8)) / 8 / 1024 ** 3));

function modalitiesOf(kind: ModelKind): Modality[] {
  switch (kind) {
    case "text":
      return ["text"];
    case "vision":
      return ["text", "vision"];
    case "stt":
      return ["audio-in", "text"];
    case "tts":
      return ["text", "audio-out"];
    case "image":
      return ["text", "image-gen"];
  }
}

function pathFor(source: ModelSource, publisher: string, base: Base, file: string, sha: string): string {
  switch (source) {
    case "cerebro":
      return `C:\\Users\\you\\.cerebro\\models\\${publisher}\\${file}`;
    case "lmstudio":
      return `C:\\Users\\you\\.lmstudio\\models\\${publisher}\\${base.name}-GGUF\\${file}`;
    case "ollama":
      return `C:\\Users\\you\\.ollama\\models\\blobs\\sha256-${sha}`;
    case "other":
      return `D:\\Models\\${file}`;
  }
}

/**
 * Illustrative tuned settings: what a finished tune could have kept for this file. A tune only keeps
 * settings that ran, so the result is fitted to the example machine with the same memory estimate
 * the load panel shows: for a MoE model, the fewest expert layers on the CPU that fit in VRAM; for a
 * dense one, the most GPU layers that do.
 */
function tunedFor(base: Base, sizeGiB: number, r: number, projector: LibraryModel["projector"]): LoadParams {
  const big = sizeGiB > 7;
  const p: LoadParams = {
    contextLength: Math.min(base.contextMax, big ? 16 * K : 32 * K),
    gpuLayers: base.arch.layers,
    moeCpuLayers: 0,
    kvCacheK: big ? "q8_0" : "f16",
    kvCacheV: big ? "q8_0" : "f16",
    flashAttention: "on",
    threads: r > 0.5 ? 6 : 8,
    cpuAffinity: r > 0.5 ? "performance" : "all",
    cpuCores: "",
    batchSize: 2048,
    ubatchSize: big ? 512 : 1024,
    mmap: true,
    mlock: false,
    speculative: r > 0.6 ? "ngram" : "off",
    ngramStrategy: r > 0.8 ? "ngram-map-k" : "ngram-simple",
    draftModelId: null,
    draftMax: 16,
  };
  const model = { sizeGiB, architecture: base.arch.experts ? ("moe" as const) : ("dense" as const), arch: base.arch, projector };
  const fits = (q: LoadParams) => estimateFit(model, q, MOCK_HARDWARE).vram.status === "fits";
  if (fits(p)) return p;
  if (base.arch.experts) {
    for (let n = 1; n <= base.arch.layers; n++) if (fits({ ...p, moeCpuLayers: n })) return { ...p, moeCpuLayers: n };
    p.moeCpuLayers = base.arch.layers;
  }
  for (let n = p.gpuLayers - 1; n > 0; n--) if (fits({ ...p, gpuLayers: n })) return { ...p, gpuLayers: n, moeCpuLayers: Math.min(p.moeCpuLayers, n) };
  return { ...p, gpuLayers: 0, moeCpuLayers: 0 };
}

/** The working depth every illustrative tune result is "measured" at. */
export const TUNE_DEPTH = 8 * K;

/**
 * An illustrative tune result in the shape docs/CLAIMS.md requires: prefill, decode at depth 0 and
 * decode at working depth, plus every condition. `decode0` is the depth-0 writing speed; the other
 * two are derived from it with plausible ratios. Invented numbers, labelled illustrative.
 */
function tuneResultFor(decode0: number, fitsInVram: boolean, r: number, measuredAt: string): TuneResult {
  return {
    prefillTps: Math.round(decode0 * (fitsInVram ? 22 : 9) * (0.9 + r * 0.2)),
    decodeTps0: decode0,
    decodeTpsDepth: round1(decode0 * (0.74 + r * 0.12)),
    depth: TUNE_DEPTH,
    build: MOCK_LIVE_BASE.backend.build,
    gpu: MOCK_GPU.name,
    sampler: "greedy",
    measuredAt,
    provenance: "illustrative",
  };
}

function build(): LibraryModel[] {
  const out: LibraryModel[] = [];
  // The four models the rest of the app shows (Home, Tune) come first, with the same ids and numbers.
  const featured = new Map(MOCK_MODELS.map((m) => [m.id, m]));
  const featuredBase: Record<string, Base> = {
    "qwen3-coder-30b-a3b-q4km": BASES.find((b) => b.slug === "qwen3-coder-30b-a3b")!,
    "qwen3-8b-q5km": BASES.find((b) => b.slug === "qwen3-8b")!,
    "gemma-4-e4b-q4km": { name: "Gemma-4-E4B-it", family: "Gemma", ggufArch: "gemma4", params: "E4B", billions: 8, kind: "text", arch: dense(42, 2, 256, 2560), contextMax: 32 * K, slug: "gemma4-e4b" },
    "mistral-small-24b-q4km": { name: "Mistral-Small-3.2-24B-Instruct", family: "Mistral", ggufArch: "llama", params: "24B", billions: 23.6, kind: "text", arch: dense(40, 8, 128, 5120), contextMax: 16 * K, slug: "mistral-small-3.2-24b" },
  };
  for (const [id, summary] of featured) {
    const base = featuredBase[id]!;
    const sha = fakeSha(id);
    const file = `${base.name}-${summary.quant}.gguf`;
    const r = (hash32(id) % 1000) / 1000;
    const tuned = summary.tune === "tuned" || summary.tune === "defaults-fastest";
    const { defaultsTps: _d, tunedTps, ...rest } = summary;
    out.push({
      ...rest,
      kind: base.kind,
      modalities: modalitiesOf(base.kind),
      family: base.family,
      file,
      path: pathFor("cerebro", "lmstudio-community", base, file, sha),
      source: "cerebro",
      sha256: sha,
      arch: base.arch,
      tunedParams: tuned ? tunedFor(base, summary.sizeGiB, r, null) : null,
      tunedAt: tuned ? "2026-09-28" : null,
      tuneResult: tuned && tunedTps !== null ? tuneResultFor(tunedTps, summary.sizeGiB < 7.2, r, "2026-09-28") : null,
      projector: null,
    });
  }

  const add = (base: Base, quant: string, publisher: string, ext: string) => {
    const id = `${base.slug}-${quant.toLowerCase().replaceAll("_", "")}-${publisher}`;
    if (featured.has(id)) return;
    const h = hash32(id);
    const r = (h % 1000) / 1000;
    const sizeGiB = sizeOf(base.billions, quant);
    const source = SOURCES[h % SOURCES.length] ?? "cerebro";
    const sha = fakeSha(id);
    const file = `${base.name}-${quant}.${ext}`;
    const llm = base.kind === "text" || base.kind === "vision";
    const tune = !llm ? "untuned" : r < 0.18 ? "tuned" : r < 0.24 ? "defaults-fastest" : r < 0.25 ? "tuning" : "untuned";
    const fits = sizeGiB < 7.2;
    const tuned = tune === "tuned" || tune === "defaults-fastest";
    const defaultsTps = round1(Math.max(1.2, (fits ? 260 : 60) / Math.max(1, sizeGiB) - r * 3));
    const decode0 = tune === "defaults-fastest" ? defaultsTps : round1(defaultsTps * (fits ? 1.05 : 2.4 + r));
    const tunedAt = tuned ? `2026-09-${String(10 + (h % 19)).padStart(2, "0")}` : null;
    // One in seven vision files was found without its projector, to show that path.
    const projector = base.kind === "vision" && h % 7 !== 0 ? { file: `mmproj-${base.name}-F16.gguf`, sizeGiB: round1(0.4 + base.billions / 40) } : null;
    out.push({
      id,
      name: base.name,
      quant,
      params: base.params,
      architecture: base.arch.experts ? "moe" : "dense",
      sizeGiB,
      contextMax: base.contextMax,
      tune,
      provenance: "illustrative",
      kind: base.kind,
      modalities: modalitiesOf(base.kind),
      family: base.family,
      file,
      path: pathFor(source, publisher, base, file, sha),
      source,
      sha256: sha,
      arch: base.arch,
      tunedParams: tuned ? tunedFor(base, sizeGiB, r, projector) : null,
      tunedAt,
      tuneResult: tuned && tunedAt ? tuneResultFor(decode0, fits, r, tunedAt) : null,
      projector,
    });
  };

  for (const base of BASES) {
    for (const quant of TEXT_QUANTS) {
      for (const pub of PUBLISHERS) add(base, quant, pub, "gguf");
      if (base.extraPublisher) add(base, quant, EXTRA_PUBLISHER, "gguf");
    }
  }
  for (const base of ENGINE_BASES) for (const quant of base.quants) add(base, quant, "ggml-org", base.ext);
  return out;
}

/** Built once per page: 500+ entries, deterministic across reloads. */
export const MOCK_LIBRARY: readonly LibraryModel[] = build();

// ---- Per-model detail: GGUF metadata, chat templates, quirks, author recommendations ----

const CHATML = `{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n' }}
{%- endif %}`;

const GEMMA = `{{ bos_token }}
{%- for message in messages %}
{%- set role = 'model' if message.role == 'assistant' else message.role %}
{{- '<start_of_turn>' + role + '\\n' + message.content | trim + '<end_of_turn>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<start_of_turn>model\\n' }}
{%- endif %}`;

const LLAMA3 = `{{- bos_token }}
{%- for message in messages %}
{{- '<|start_header_id|>' + message.role + '<|end_header_id|>\\n\\n' + message.content | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
{%- endif %}`;

const TEMPLATES: Readonly<Record<string, string>> = { Qwen3: CHATML, "Qwen2.5-VL": CHATML, DeepSeek: CHATML, OuteTTS: CHATML, Gemma: GEMMA, Llama: LLAMA3, Orpheus: LLAMA3 };

const fix = (id: string, title: string, problem: string, change: string): QuirkFix => ({ id, title, problem, change, provenance: "illustrative" });

const QUIRKS: Readonly<Record<string, readonly QuirkFix[]>> = {
  Qwen3: [
    fix("qwen3-yarn", "Long context with YaRN", "Past 32,768 tokens the model loses track of earlier text.", "Turns on YaRN RoPE scaling (factor 4 from 32,768 tokens), the method the model card describes for longer input."),
    fix("qwen3-im-end", "Stop at <|im_end|>", "Some files do not mark <|im_end|> as an end token, so replies run on.", "Adds <|im_end|> as a stop token."),
  ],
  Gemma: [
    fix("gemma-end-of-turn", "Stop at <end_of_turn>", "Replies can continue into a made-up next turn.", "Adds <end_of_turn> as a stop token."),
    fix("gemma-double-bos", "No second start token", "The template and the tokenizer both add a start-of-text token.", "Sets tokenizer.ggml.add_bos_token to false at load time."),
  ],
  Llama: [fix("llama-eot", "Stop at <|eot_id|>", "Older conversions end turns with a token the file does not list as an end.", "Adds <|eot_id|> as a stop token.")],
  Mistral: [fix("mistral-low-temp", "Low temperature", "Answers wander at a general-purpose temperature.", "Sets temperature 0.15, the value the model card suggests.")],
  DeepSeek: [fix("r1-sampler", "Reasoning sampler", "At very low temperature the model can repeat itself.", "Sets temperature 0.6 and top-p 0.95, the values the model card suggests.")],
  "gpt-oss": [fix("harmony-template", "Harmony template", "An early conversion shipped a template without the Harmony message format.", "Replaces the chat template with the Harmony one from the model's repository.")],
  Phi: [fix("phi-end", "Stop at <|end|>", "Replies can run past the end of the answer.", "Adds <|end|> as a stop token.")],
};

const AUTHOR_SAMPLERS: Readonly<Record<string, Sampler>> = {
  Qwen3: { temperature: 0.6, topK: 20, topP: 0.95, minP: 0, repeatPenalty: 1 },
  Gemma: { temperature: 1, topK: 64, topP: 0.95, minP: 0, repeatPenalty: 1 },
  Mistral: { temperature: 0.15, topK: 40, topP: 0.95, minP: 0.05, repeatPenalty: 1 },
  DeepSeek: { temperature: 0.6, topK: 40, topP: 0.95, minP: 0.05, repeatPenalty: 1 },
};

const FILE_TYPES: Readonly<Record<string, number>> = { F16: 1, Q4_0: 2, Q8_0: 7, Q4_K_M: 15, Q4_K_S: 14, Q5_K_M: 17, Q5_K_S: 16, Q6_K: 18, Q3_K_M: 12, IQ4_XS: 30, Q5_0: 8, Q5_1: 9 };

function ggufFor(m: LibraryModel, archName: string, template: string | null): GgufEntry[] {
  const a = m.arch;
  const entries: GgufEntry[] = [
    { key: "general.architecture", type: "string", value: archName, editable: false },
    { key: "general.name", type: "string", value: m.name, editable: true },
    { key: "general.license", type: "string", value: "see model card", editable: true },
    { key: "general.file_type", type: "uint32", value: FILE_TYPES[m.quant] ?? 0, editable: false },
    { key: "general.quantization_version", type: "uint32", value: 2, editable: false },
    { key: `${archName}.context_length`, type: "uint32", value: m.contextMax, editable: true },
    { key: `${archName}.block_count`, type: "uint32", value: a.layers, editable: false },
    { key: `${archName}.embedding_length`, type: "uint32", value: a.embedding, editable: false },
    { key: `${archName}.attention.head_count_kv`, type: "uint32", value: a.kvHeads, editable: false },
    { key: `${archName}.attention.key_length`, type: "uint32", value: a.headDim, editable: false },
    { key: `${archName}.rope.freq_base`, type: "float32", value: archName.startsWith("qwen") ? 1_000_000 : 500_000, editable: true },
    { key: `${archName}.attention.layer_norm_rms_epsilon`, type: "float32", value: 0.000001, editable: true },
  ];
  if (a.experts) {
    entries.push({ key: `${archName}.expert_count`, type: "uint32", value: a.experts, editable: false });
    entries.push({ key: `${archName}.expert_used_count`, type: "uint32", value: a.activeExperts ?? 0, editable: false });
  }
  entries.push(
    { key: "tokenizer.ggml.model", type: "string", value: m.family === "Llama" ? "gpt2" : m.family === "Gemma" ? "llama" : "gpt2", editable: false },
    { key: "tokenizer.ggml.tokens", type: "array", value: `${m.family === "Gemma" ? "262,144" : m.family === "Llama" ? "128,256" : "151,936"} strings`, editable: false },
    { key: "tokenizer.ggml.bos_token_id", type: "uint32", value: m.family === "Llama" ? 128000 : m.family === "Gemma" ? 2 : 151643, editable: true },
    { key: "tokenizer.ggml.eos_token_id", type: "uint32", value: m.family === "Llama" ? 128009 : m.family === "Gemma" ? 1 : 151645, editable: true },
    { key: "tokenizer.ggml.add_bos_token", type: "bool", value: m.family !== "Qwen3", editable: true },
  );
  if (template !== null) entries.push({ key: "tokenizer.chat_template", type: "string", value: template, editable: true });
  return entries;
}

const ARCH_NAMES: Readonly<Record<string, string>> = {
  Qwen3: "qwen3", "Qwen2.5-VL": "qwen2vl", Gemma: "gemma3", Llama: "llama", Mistral: "llama", Phi: "phi3", DeepSeek: "qwen2", "gpt-oss": "gpt-oss", GLM: "glm4moe", OuteTTS: "qwen3", Orpheus: "llama", FLUX: "flux", "SD3.5": "sd3",
};

export function detailFor(m: LibraryModel): Omit<ModelDetail, "copy"> {
  const archName = m.architecture === "moe" && m.family === "Qwen3" ? "qwen3moe" : (ARCH_NAMES[m.family] ?? "llama");
  const isGguf = m.file.endsWith(".gguf");
  const llm = m.kind === "text" || m.kind === "vision";
  const template = llm ? (TEMPLATES[m.family] ?? CHATML) : null;
  const rope: Tweaks["rope"] = m.family === "Qwen3" && m.contextMax <= 32 * K ? { scaling: "yarn", factor: 4, originalContext: 32 * K } : null;
  return {
    id: m.id,
    gguf: isGguf ? ggufFor(m, archName, template) : null,
    chatTemplate: template,
    quirks: llm ? [...(QUIRKS[m.family] ?? [])] : [],
    author: { rope, sampler: AUTHOR_SAMPLERS[m.family] ? { ...AUTHOR_SAMPLERS[m.family]! } : null },
  };
}

// ---- Disk space: duplicate files across apps' folders ----

export function mockDuplicates(): DuplicateGroup[] {
  const g = (file: string, sizeGiB: number, copies: DuplicateGroup["copies"]): DuplicateGroup => ({
    id: `dup-${hash32(file).toString(16)}`,
    file,
    sizeGiB,
    sha256: fakeSha(file),
    copies,
    state: "duplicate",
    kept: null,
  });
  return [
    g("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", 17.3, [
      { path: "C:\\Users\\you\\.cerebro\\models\\lmstudio-community\\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", source: "cerebro", volume: "C:" },
      { path: "C:\\Users\\you\\.lmstudio\\models\\lmstudio-community\\Qwen3-Coder-30B-A3B-Instruct-GGUF\\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", source: "lmstudio", volume: "C:" },
      { path: `C:\\Users\\you\\.ollama\\models\\blobs\\sha256-${fakeSha("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")}`, source: "ollama", volume: "C:" },
      { path: "D:\\Models\\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", source: "other", volume: "D:" },
    ]),
    g("gemma-3-12b-it-Q4_K_M.gguf", 6.8, [
      { path: "C:\\Users\\you\\.lmstudio\\models\\unsloth\\gemma-3-12b-it-GGUF\\gemma-3-12b-it-Q4_K_M.gguf", source: "lmstudio", volume: "C:" },
      { path: `C:\\Users\\you\\.ollama\\models\\blobs\\sha256-${fakeSha("gemma-3-12b-it-Q4_K_M.gguf")}`, source: "ollama", volume: "C:" },
    ]),
    g("Llama-3.1-8B-Instruct-Q8_0.gguf", 7.9, [
      { path: "C:\\Users\\you\\.lmstudio\\models\\bartowski\\Llama-3.1-8B-Instruct-GGUF\\Llama-3.1-8B-Instruct-Q8_0.gguf", source: "lmstudio", volume: "C:" },
      { path: "D:\\Models\\Llama-3.1-8B-Instruct-Q8_0.gguf", source: "other", volume: "D:" },
    ]),
    g("Qwen3-8B-Q5_K_M.gguf", 5.4, [
      { path: "C:\\Users\\you\\.cerebro\\models\\lmstudio-community\\Qwen3-8B-Q5_K_M.gguf", source: "cerebro", volume: "C:" },
      { path: "C:\\Users\\you\\.lmstudio\\models\\Qwen\\Qwen3-8B-GGUF\\Qwen3-8B-Q5_K_M.gguf", source: "lmstudio", volume: "C:" },
    ]),
  ];
}
