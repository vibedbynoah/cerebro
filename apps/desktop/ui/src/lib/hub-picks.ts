/**
 * The Model Hub's front page: model families worth starting with, by task, and how a repository's
 * family and abilities are read from its name and tags. A pick is a search (never a hard-coded
 * repository, which could be renamed or removed): choosing it searches Hugging Face for current
 * GGUF files of that model. Sizes are typical 4-bit file sizes, for the fit badge before the real
 * file list is read.
 */

export type Ability = "chat" | "code" | "vision" | "reasoning" | "tools" | "speech" | "image" | "small";

export interface HubPick {
  title: string;
  family: string;
  query: string;
  /** Typical size of a good 4-bit file, GiB. */
  sizeGiB: number;
  params: string;
  abilities: readonly Ability[];
  blurb: string;
  /** The account on Hugging Face that publishes the original weights of this model family. */
  publisher: string;
}

export interface HubCategory {
  id: string;
  label: string;
  query: string;
}

export const HUB_CATEGORIES: readonly HubCategory[] = [
  { id: "chat", label: "Chat", query: "Instruct GGUF" },
  { id: "code", label: "Coding", query: "Coder GGUF" },
  { id: "vision", label: "Vision", query: "VL GGUF" },
  { id: "reasoning", label: "Reasoning", query: "Thinking GGUF" },
  { id: "small", label: "Small and fast", query: "0.6B GGUF" },
  { id: "speech", label: "Speech", query: "Qwen3 ASR GGUF" },
];

export const HUB_PICKS: readonly HubPick[] = [
  { title: "Qwen3.5 4B", family: "qwen", query: "Qwen3.5 4B GGUF", sizeGiB: 2.6, params: "4B", abilities: ["chat", "reasoning", "tools", "vision"], blurb: "A strong all-rounder that fits almost any graphics card.", publisher: "Qwen" },
  { title: "Qwen3 Coder 30B A3B", family: "qwen", query: "Qwen3 Coder 30B A3B GGUF", sizeGiB: 17.3, params: "30B (3B active)", abilities: ["code", "tools"], blurb: "Fast for its size: only 3B parameters work on each word.", publisher: "Qwen" },
  { title: "Gemma 4 E4B", family: "gemma", query: "gemma-4-E4B-it GGUF", sizeGiB: 4.6, params: "8B (4B effective)", abilities: ["chat", "vision"], blurb: "Quick, friendly answers and image understanding.", publisher: "google" },
  { title: "Qwen3.8 27B", family: "qwen", query: "Qwen3.8 27B GGUF", sizeGiB: 15.4, params: "27B", abilities: ["chat", "reasoning", "code", "tools"], blurb: "For bigger cards, or split between the card and the processor.", publisher: "Qwen" },
  { title: "Llama 3.1 8B", family: "llama", query: "Llama-3.1-8B-Instruct GGUF", sizeGiB: 4.6, params: "8B", abilities: ["chat", "tools"], blurb: "A widely used general model with a large community.", publisher: "meta-llama" },
  { title: "Qwen3 0.6B", family: "qwen", query: "Qwen3-0.6B GGUF", sizeGiB: 0.5, params: "0.6B", abilities: ["chat", "small"], blurb: "Tiny and very fast: good for quick tasks and older machines.", publisher: "Qwen" },
  { title: "Qwen3 ASR 0.6B", family: "qwen", query: "Qwen3-ASR GGUF", sizeGiB: 0.6, params: "0.6B", abilities: ["speech"], blurb: "Speech to text for push-to-talk in Chat.", publisher: "Qwen" },
  { title: "Qwen3 TTS 1.7B", family: "qwen", query: "Qwen3-TTS GGUF", sizeGiB: 1.9, params: "1.7B", abilities: ["speech"], blurb: "Reads replies aloud in a natural voice.", publisher: "Qwen" },
];

/** Image models featured on the front page: the ones the engine runs (Stable Diffusion 2.x family,
 *  diffusers folders), downloaded whole from the publisher's own repository. */
export interface FeaturedImage {
  title: string;
  repo: string;
  publisher: string;
  params: string;
  blurb: string;
}
export const FEATURED_IMAGES: readonly FeaturedImage[] = [
  { title: "SD Turbo", repo: "stabilityai/sd-turbo", publisher: "stabilityai", params: "1.3B", blurb: "A 512 x 512 picture in a single step: makes the images for /imagine and Image Studio." },
];

export const ABILITY_LABELS: Readonly<Record<Ability, string>> = {
  chat: "Chat",
  code: "Code",
  vision: "Vision",
  reasoning: "Reasoning",
  tools: "Tools",
  speech: "Speech",
  image: "Images",
  small: "Small",
};

/** Families the tile recognises: a short mark and a hue for its colour. */
const FAMILIES: ReadonlyArray<{ id: string; test: RegExp; mark: string; hue: number }> = [
  { id: "qwen", test: /qwen|qwq/i, mark: "Qw", hue: 285 },
  { id: "gemma", test: /gemma/i, mark: "Ge", hue: 230 },
  { id: "llama", test: /llama/i, mark: "Ll", hue: 255 },
  { id: "mistral", test: /mistral|mixtral|ministral|devstral|magistral/i, mark: "Mi", hue: 45 },
  { id: "phi", test: /\bphi/i, mark: "Ph", hue: 200 },
  { id: "deepseek", test: /deepseek/i, mark: "DS", hue: 240 },
  { id: "glm", test: /glm/i, mark: "GL", hue: 160 },
  { id: "gpt-oss", test: /gpt-?oss/i, mark: "OS", hue: 150 },
  { id: "kimi", test: /kimi/i, mark: "Ki", hue: 0 },
  { id: "minimax", test: /minimax/i, mark: "MM", hue: 330 },
  { id: "granite", test: /granite/i, mark: "Gr", hue: 210 },
  { id: "smol", test: /smol/i, mark: "Sm", hue: 60 },
  { id: "olmo", test: /olmo/i, mark: "Ol", hue: 120 },
  { id: "nemotron", test: /nemotron/i, mark: "Ne", hue: 100 },
  { id: "stable-diffusion", test: /sd[-_]?turbo|stable[-_ ]?diffusion|sdxl/i, mark: "SD", hue: 20 },
];

/** A repository's family mark and hue (a neutral mark from its name when the family is unknown). */
export function familyOf(name: string): { id: string; mark: string; hue: number | null } {
  const f = FAMILIES.find((x) => x.test.test(name));
  if (f) return { id: f.id, mark: f.mark, hue: f.hue };
  const base = name.split("/").pop() ?? name;
  const letters = base.replace(/[^A-Za-z]/g, "");
  return { id: "other", mark: (letters.slice(0, 2) || "?").replace(/^./, (c) => c.toUpperCase()), hue: null };
}

/** What a repository can do, from its name and Hugging Face tags. */
export function abilitiesOf(id: string, tags: readonly string[]): Ability[] {
  const t = `${id} ${tags.join(" ")}`.toLowerCase();
  const out: Ability[] = [];
  if (/coder|code|devstral|codestral/.test(t)) out.push("code");
  if (/image-text-to-text|vision|\bvl\b|-vl-|vlm|multimodal/.test(t)) out.push("vision");
  if (/thinking|reasoning|\br1\b|qwq|magistral/.test(t)) out.push("reasoning");
  if (/asr|tts|speech|whisper|automatic-speech|text-to-speech/.test(t)) out.push("speech");
  if (/text-to-image|diffusers|stable-diffusion|sd-turbo/.test(t)) out.push("image");
  if (/tool|function/.test(t)) out.push("tools");
  if (/\b0\.[0-9]b\b|\b1(\.[0-9])?b\b|-0\.[0-9]b|-1b/.test(t)) out.push("small");
  if (!out.some((a) => a === "speech" || a === "image")) out.unshift("chat");
  return out;
}

/** Fit of a file of `sizeGiB` on this machine (graphics memory and system memory, GiB). */
export function fitBadge(sizeGiB: number, vramGiB: number, ramGiB: number): { label: string; tone: "good" | "ok" | "slow" | "no" } {
  // About 1.2 GiB above the file for the cache and buffers at a normal context.
  const need = sizeGiB + 1.2;
  if (vramGiB > 0 && need <= vramGiB) return { label: "Fits your graphics card", tone: "good" };
  if (need <= vramGiB + ramGiB * 0.7) return vramGiB > 0 ? { label: "Fits with part on the processor", tone: "ok" } : { label: "Runs on the processor", tone: "slow" };
  return { label: "Too big for this computer", tone: "no" };
}
