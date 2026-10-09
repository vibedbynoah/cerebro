import { readPref, writePref } from "./storage";

/**
 * Image Studio: built into Cerebro, no plugin needed. Modes, local image models, generation
 * parameters with validation, and the gallery (metadata only, kept on this machine).
 */

export type ImageMode = "txt2img" | "img2img" | "inpaint" | "upscale" | "variations";
export const MODES: readonly ImageMode[] = ["txt2img", "img2img", "inpaint", "upscale", "variations"];
export const MODE_LABELS: Readonly<Record<ImageMode, string>> = {
  txt2img: "Text to image",
  img2img: "Image to image",
  inpaint: "Inpaint",
  upscale: "Upscale",
  variations: "Variations",
};

export interface ImageModel {
  id: string;
  name: string;
  family: "FLUX" | "SDXL" | "SD 3.5" | "Qwen-Image" | "HiDream";
  format: "GGUF" | "safetensors";
  quant: string;
  sizeGiB: number;
  /** Illustrative seconds per 1024x1024 image at default steps on this machine. */
  secondsPer1024: number;
  defaultSteps: number;
  defaultGuidance: number;
}

export const IMAGE_MODELS: readonly ImageModel[] = [
  { id: "flux1-dev-q4", name: "FLUX.1 dev", family: "FLUX", format: "GGUF", quant: "Q4_K_S", sizeGiB: 6.8, secondsPer1024: 38, defaultSteps: 28, defaultGuidance: 3.5 },
  { id: "flux1-schnell-q5", name: "FLUX.1 schnell", family: "FLUX", format: "GGUF", quant: "Q5_K_S", sizeGiB: 8.3, secondsPer1024: 6, defaultSteps: 4, defaultGuidance: 0 },
  { id: "qwen-image-q4", name: "Qwen-Image", family: "Qwen-Image", format: "GGUF", quant: "Q4_K_M", sizeGiB: 12.1, secondsPer1024: 52, defaultSteps: 30, defaultGuidance: 4 },
  { id: "sd35-medium", name: "Stable Diffusion 3.5 Medium", family: "SD 3.5", format: "safetensors", quant: "FP16", sizeGiB: 5.1, secondsPer1024: 21, defaultSteps: 28, defaultGuidance: 4.5 },
  { id: "sdxl-base", name: "SDXL 1.0", family: "SDXL", format: "safetensors", quant: "FP16", sizeGiB: 6.9, secondsPer1024: 14, defaultSteps: 30, defaultGuidance: 6 },
];

export const ASPECTS = [
  { id: "1:1", w: 1024, h: 1024 },
  { id: "4:3", w: 1152, h: 864 },
  { id: "3:2", w: 1216, h: 832 },
  { id: "16:9", w: 1344, h: 768 },
  { id: "2:3", w: 832, h: 1216 },
  { id: "9:16", w: 768, h: 1344 },
] as const;
export type AspectId = (typeof ASPECTS)[number]["id"];

export const SAMPLERS = ["Euler", "Euler a", "DPM++ 2M", "DPM++ 2M Karras", "DPM++ SDE", "UniPC", "LCM"] as const;

export interface GenParams {
  mode: ImageMode;
  modelId: string;
  prompt: string;
  negative: string;
  aspect: AspectId;
  seed: number | null;
  steps: number;
  guidance: number;
  sampler: (typeof SAMPLERS)[number];
  /** img2img / inpaint: how far to move from the source, 0-1. */
  strength: number;
  /** upscale factor. */
  scale: 2 | 4;
}

export type ParamErrors = Partial<Record<keyof GenParams, string>>;

export function validate(p: GenParams, hasSource: boolean, hasMask: boolean): ParamErrors {
  const e: ParamErrors = {};
  if ((p.mode === "txt2img" || p.mode === "img2img" || p.mode === "inpaint") && !p.prompt.trim()) e.prompt = "Describe the image.";
  if (p.prompt.length > 2000) e.prompt = "Keep the prompt under 2,000 characters.";
  if (p.negative.length > 1000) e.negative = "Keep the negative prompt under 1,000 characters.";
  if (!Number.isInteger(p.steps) || p.steps < 1 || p.steps > 150) e.steps = "Steps must be 1 to 150.";
  if (!(p.guidance >= 0 && p.guidance <= 30)) e.guidance = "Guidance must be 0 to 30.";
  if (p.seed !== null && (!Number.isInteger(p.seed) || p.seed < 0 || p.seed > 4294967295)) e.seed = "Seed must be a whole number from 0 to 4,294,967,295.";
  if (!(p.strength >= 0 && p.strength <= 1)) e.strength = "Strength must be 0 to 1.";
  if (p.mode !== "txt2img" && !hasSource) e.mode = "Add a source image first.";
  else if (p.mode === "inpaint" && !hasMask) e.mode = "Paint over the area to change first.";
  if (!IMAGE_MODELS.some((m) => m.id === p.modelId)) e.modelId = "Choose a model.";
  return e;
}

export function sizeFor(p: GenParams): { w: number; h: number } {
  const a = ASPECTS.find((x) => x.id === p.aspect) ?? ASPECTS[0];
  return p.mode === "upscale" ? { w: a.w * p.scale, h: a.h * p.scale } : { w: a.w, h: a.h };
}

/** Illustrative time estimate, scaled by pixels and steps from the model's 1024x1024 figure. */
export function estimateSeconds(p: GenParams): number {
  const m = IMAGE_MODELS.find((x) => x.id === p.modelId);
  if (!m) return 0;
  const { w, h } = sizeFor(p);
  const px = (w * h) / (1024 * 1024);
  const steps = p.mode === "upscale" ? 10 : p.steps;
  return Math.max(1, Math.round(m.secondsPer1024 * px * (steps / m.defaultSteps) * (p.mode === "img2img" || p.mode === "inpaint" ? p.strength : 1)));
}

export function randomSeed(): number {
  const b = new Uint32Array(1);
  crypto.getRandomValues(b);
  return b[0]!;
}

// ---- Gallery ------------------------------------------------------------------------------------

export interface GalleryItem {
  id: string;
  prompt: string;
  negative: string;
  modelId: string;
  mode: ImageMode;
  seed: number;
  w: number;
  h: number;
  steps: number;
  guidance: number;
  favourite: boolean;
  at: number;
}

export const GALLERY_KEY = "cerebro-image-gallery";

export function parseGallery(raw: string | null): GalleryItem[] {
  if (!raw) return [];
  try {
    const v: unknown = JSON.parse(raw);
    if (!Array.isArray(v)) return [];
    return v
      .flatMap((x): GalleryItem[] => {
        if (!x || typeof x !== "object") return [];
        const g = x as Record<string, unknown>;
        const ok =
          typeof g.id === "string" &&
          g.id.length <= 64 &&
          typeof g.prompt === "string" &&
          g.prompt.length <= 2000 &&
          typeof g.negative === "string" &&
          g.negative.length <= 1000 &&
          typeof g.modelId === "string" &&
          (MODES as readonly unknown[]).includes(g.mode) &&
          [g.seed, g.w, g.h, g.steps, g.guidance, g.at].every((n) => typeof n === "number" && Number.isFinite(n) && n >= 0);
        return ok ? [{ ...(g as unknown as GalleryItem), favourite: g.favourite === true }] : [];
      })
      .slice(0, 500);
  } catch {
    return [];
  }
}

export function loadGallery(): GalleryItem[] {
  return parseGallery(readPref(GALLERY_KEY));
}

export function saveGallery(items: readonly GalleryItem[]): void {
  writePref(GALLERY_KEY, JSON.stringify(items.slice(0, 500)));
}

/** Deterministic pseudo-random sequence from a seed (mulberry32), for the illustrative artwork. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hashPrompt(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
