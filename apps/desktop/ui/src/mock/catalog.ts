import type { CatalogModel, CatalogQuant, QuantBackend, QuantFormat } from "../service/types";

/**
 * ILLUSTRATIVE DATA, NOT MEASUREMENTS. The catalogue setup and Model Hub list before anything is
 * downloaded. Model names, parameter counts, layer counts and context lengths are as published;
 * file sizes are close to the published files; every KL divergence, perplexity change and speed is
 * a placeholder carrying `provenance: "illustrative"` (docs/CLAIMS.md).
 *
 * Current models only, no catalogue padding with older releases, and nothing downloads until the
 * user asks for one file. Each model lists today's strongest quantisation families side by side so
 * a pro can pick by quality per GiB on this machine.
 */

const NO_EXL3 = "Needs the ExLlamaV3 engine, which this build does not include.";
const NO_VLLM = "Needs vLLM on Linux; this build does not include it.";
const NO_MLX = "Apple silicon only.";

function q(
  model: string,
  label: string,
  format: QuantFormat,
  backend: QuantBackend,
  bitsPerWeight: number,
  sizeGiB: number,
  klDiv: number,
  pplDeltaPct: number,
  decodeTps: number | null,
): CatalogQuant {
  const unavailable = backend === "exllamav3" ? NO_EXL3 : backend === "vllm" ? NO_VLLM : backend === "mlx" ? NO_MLX : null;
  return {
    id: `${model}--${label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}`,
    label,
    format,
    backend,
    bitsPerWeight,
    sizeGiB,
    klDiv,
    pplDeltaPct,
    decodeTps: unavailable ? null : decodeTps,
    unavailable,
    provenance: "illustrative",
  };
}

/** The three formats other engines run, listed for every model so the comparison is complete. */
function elsewhere(model: string, billions: number): CatalogQuant[] {
  const size = (bpw: number) => Math.round(((billions * 1e9 * bpw) / 8 / 1024 ** 3) * 10) / 10;
  return [
    q(model, "EXL3 4.0bpw", "exl3", "exllamav3", 4.0, size(4.0), 0.019, 1.0, null),
    q(model, "AWQ INT4", "awq", "vllm", 4.25, size(4.25), 0.041, 2.2, null),
    q(model, "MLX 4-bit", "mlx", "mlx", 4.5, size(4.5), 0.036, 1.9, null),
  ];
}

export const MOCK_CATALOG: readonly CatalogModel[] = [
  {
    id: "gemma-4-12b-it",
    name: "Gemma-4-12B-it",
    params: "12B",
    activeParams: null,
    architecture: "dense",
    quant: "UD-Q4_K_XL",
    sizeGiB: 6.9,
    goodFor: "Vision in, 128K context",
    provenance: "illustrative",
    layers: 48,
    kvHeads: 8,
    headDim: 256,
    embedding: 3840,
    contextMax: 131072,
    sampler: { temperature: 1.0, topP: 0.95, topK: 64, minP: 0 },
    quants: [
      q("gemma-4-12b-it", "UD-Q4_K_XL", "unsloth-ud", "llama.cpp", 4.9, 6.9, 0.018, 0.9, 31.4),
      q("gemma-4-12b-it", "GSQ-RCO 3.5", "gsq-rco", "strata", 3.5, 4.9, 0.024, 1.3, 38.2),
      q("gemma-4-12b-it", "IQ4_XS", "imatrix-iq", "llama.cpp", 4.3, 6.1, 0.027, 1.4, 33.0),
      q("gemma-4-12b-it", "Q6_K", "k-quant", "llama.cpp", 6.6, 9.1, 0.006, 0.3, 14.8),
      ...elsewhere("gemma-4-12b-it", 11.95),
    ],
  },
  {
    id: "gemma-4-e4b-it",
    name: "Gemma-4-E4B-it",
    params: "E4B",
    activeParams: null,
    architecture: "dense",
    quant: "UD-Q4_K_XL",
    sizeGiB: 4.6,
    goodFor: "Vision and audio in, low memory",
    provenance: "illustrative",
    layers: 42,
    kvHeads: 4,
    headDim: 256,
    embedding: 2560,
    contextMax: 131072,
    sampler: { temperature: 1.0, topP: 0.95, topK: 64, minP: 0 },
    quants: [
      q("gemma-4-e4b-it", "UD-Q4_K_XL", "unsloth-ud", "llama.cpp", 4.9, 4.6, 0.021, 1.1, 58.3),
      q("gemma-4-e4b-it", "GSQ-RCO 3.5", "gsq-rco", "strata", 3.5, 3.4, 0.029, 1.5, 63.0),
      q("gemma-4-e4b-it", "IQ4_XS", "imatrix-iq", "llama.cpp", 4.3, 4.0, 0.034, 1.8, 60.7),
      q("gemma-4-e4b-it", "Q8_0", "k-quant", "llama.cpp", 8.5, 7.4, 0.002, 0.1, 22.5),
      ...elsewhere("gemma-4-e4b-it", 7.5),
    ],
  },
  {
    id: "gpt-oss-20b",
    name: "gpt-oss-20b",
    params: "21B-A3.6B",
    activeParams: "3.6B",
    architecture: "moe",
    quant: "UD-Q4_K_XL",
    sizeGiB: 11.1,
    goodFor: "Reasoning with adjustable effort, tool calls",
    provenance: "illustrative",
    layers: 24,
    kvHeads: 8,
    headDim: 64,
    embedding: 2880,
    contextMax: 131072,
    sampler: { temperature: 1.0, topP: 1.0, topK: 0, minP: 0 },
    quants: [
      q("gpt-oss-20b", "UD-Q4_K_XL", "unsloth-ud", "llama.cpp", 4.5, 11.1, 0.012, 0.6, 24.6),
      q("gpt-oss-20b", "GSQ-RCO 3.6", "gsq-rco", "strata", 3.6, 9.0, 0.017, 0.9, 29.1),
      q("gpt-oss-20b", "IQ4_XS", "imatrix-iq", "llama.cpp", 4.3, 10.7, 0.019, 1.0, 25.2),
      ...elsewhere("gpt-oss-20b", 21),
    ],
  },
  {
    id: "gemma-4-26b-a4b-it",
    name: "Gemma-4-26B-A4B-it",
    params: "26B-A4B",
    activeParams: "3.8B",
    architecture: "moe",
    quant: "UD-Q4_K_XL",
    sizeGiB: 15.6,
    goodFor: "Vision in, 256K context",
    provenance: "illustrative",
    layers: 30,
    kvHeads: 8,
    headDim: 256,
    embedding: 4096,
    contextMax: 262144,
    sampler: { temperature: 1.0, topP: 0.95, topK: 64, minP: 0 },
    quants: [
      q("gemma-4-26b-a4b-it", "UD-Q4_K_XL", "unsloth-ud", "llama.cpp", 5.1, 15.6, 0.015, 0.8, 19.7),
      q("gemma-4-26b-a4b-it", "GSQ-RCO 3.5", "gsq-rco", "strata", 3.5, 10.9, 0.022, 1.2, 25.4),
      q("gemma-4-26b-a4b-it", "IQ4_XS", "imatrix-iq", "llama.cpp", 4.3, 13.8, 0.025, 1.3, 21.0),
      q("gemma-4-26b-a4b-it", "UD-IQ2_M", "unsloth-ud", "llama.cpp", 2.9, 9.2, 0.094, 4.6, 27.8),
      ...elsewhere("gemma-4-26b-a4b-it", 25.2),
    ],
  },
  {
    id: "laguna-xs-2.1",
    name: "Laguna-XS-2.1",
    params: "33B-A3B",
    activeParams: "3B",
    architecture: "moe",
    quant: "Q4_K_M",
    sizeGiB: 18.6,
    goodFor: "Agentic coding, 256K context",
    provenance: "illustrative",
    layers: 40,
    kvHeads: 4,
    headDim: 256,
    embedding: 2048,
    contextMax: 262144,
    sampler: { temperature: 0.7, topP: 0.95, topK: 40, minP: 0 },
    quants: [
      q("laguna-xs-2.1", "Q4_K_M", "k-quant", "llama.cpp", 4.85, 18.6, 0.024, 1.2, 17.2),
      q("laguna-xs-2.1", "IQ4_XS", "imatrix-iq", "llama.cpp", 4.3, 16.5, 0.028, 1.5, 18.4),
      q("laguna-xs-2.1", "UD-Q3_K_XL", "unsloth-ud", "llama.cpp", 3.7, 14.2, 0.047, 2.4, 20.1),
      ...elsewhere("laguna-xs-2.1", 33),
    ],
  },
  {
    id: "qwen3.8-27b",
    name: "Qwen3.8-27B",
    params: "27B",
    activeParams: null,
    architecture: "dense",
    quant: "UD-Q4_K_M",
    sizeGiB: 15.4,
    goodFor: "Coding agent, vision in, 256K context",
    provenance: "illustrative",
    layers: 64,
    kvHeads: 8,
    headDim: 128,
    embedding: 5120,
    contextMax: 262144,
    sampler: { temperature: 1.0, topP: 0.95, topK: 20, minP: 0 },
    quants: [
      q("qwen3.8-27b", "UD-Q4_K_M", "unsloth-ud", "llama.cpp", 4.9, 15.4, 0.028, 1.4, 4.1),
      q("qwen3.8-27b", "GSQ-RCO IQ3_XXS", "gsq-rco", "strata", 3.2, 9.4, 0.045, 2.1, 6.9),
      q("qwen3.8-27b", "Q4_K_M imatrix", "imatrix-iq", "llama.cpp", 5.3, 16.5, 0.031, 1.6, 3.8),
      q("qwen3.8-27b", "UD-IQ2_XXS", "unsloth-ud", "llama.cpp", 2.3, 6.8, 0.187, 8.9, 11.6),
      q("qwen3.8-27b", "Q8_0", "k-quant", "llama.cpp", 8.5, 27.0, 0.003, 0.1, 2.2),
      ...elsewhere("qwen3.8-27b", 27),
    ],
  },
  {
    id: "mistral-medium-3.5-128b",
    name: "Mistral-Medium-3.5-128B",
    params: "128B",
    activeParams: null,
    architecture: "dense",
    quant: "Q4_K_M",
    sizeGiB: 72.5,
    goodFor: "Coding and reasoning, 256K context",
    provenance: "illustrative",
    layers: 88,
    kvHeads: 8,
    headDim: 128,
    embedding: 8192,
    contextMax: 262144,
    sampler: { temperature: 0.7, topP: 0.95, topK: 40, minP: 0 },
    quants: [
      q("mistral-medium-3.5-128b", "Q4_K_M", "k-quant", "llama.cpp", 4.85, 72.5, 0.022, 1.1, 0.6),
      q("mistral-medium-3.5-128b", "UD-IQ2_XXS", "unsloth-ud", "llama.cpp", 2.3, 34.8, 0.151, 7.2, 1.4),
      ...elsewhere("mistral-medium-3.5-128b", 128),
    ],
  },
];
