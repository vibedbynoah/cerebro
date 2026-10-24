/**
 * The Hub's front page: the newest Qwen model of each kind that this computer can run, found live
 * on Hugging Face (so a model released today is offered today), with an estimate of how fast it
 * writes here and a warning when part of it would be read from the drive on every word.
 *
 * Speeds come from what this engine measured on an 8 GB laptop card with 32 GB of RAM (Qwen3 8B:
 * 56 tok/s from graphics memory, 9.8 tok/s from RAM; Flash-Next from the drive: about 2 GB/s):
 * writing a token reads the active weights once, so time per token is the bytes read from each
 * place divided by how fast that place is read.
 */

import type { HardwareProfile, HubRepo } from "../service/types";
import { recommendQuant, type QuantFile, type QuantPick } from "./hub-quant";

export type Kind = "dense" | "moe";

export interface Parsed {
  /** Total parameters, billions. */
  totalB: number;
  /** Parameters working on each word, billions (the total for a dense model). */
  activeB: number;
  kind: Kind;
}

/** Repositories that are not chat models (speech, embeddings, rerankers, vision-only parts). */
const NOT_CHAT = /embed|rerank|asr|tts|omni|guard|vl-|-vl\b|vision|mmproj|audio|ocr|image|diffusion|draft|eagle|mtp|agentworld|world-?model/i;

/** Size and kind from a repository name: "Qwen3.6-35B-A3B-GGUF" is a 35B MoE with 3B active. */
export function parseRepo(id: string): Parsed | null {
  const name = id.split("/").pop() ?? id;
  if (NOT_CHAT.test(name)) return null;
  const a = /(?:^|[-_ ])A(\d+(?:\.\d+)?)B(?:[-_ ]|$)/i.exec(name);
  const t = /(?:^|[-_ ])(\d+(?:\.\d+)?)B(?:[-_ ]|$)/i.exec(name.replace(/A\d+(?:\.\d+)?B/gi, ""));
  // Flash-Next names carry no size: 177B total, 3B active (its GGUF header).
  if (/flash[-_ ]?next/i.test(name)) return { totalB: 177, activeB: 3, kind: "moe" };
  if (!t) return null;
  const totalB = Number(t[1]);
  if (a) return { totalB, activeB: Number(a[1]), kind: "moe" };
  return { totalB, activeB: totalB, kind: "dense" };
}

/** Bytes per second from each tier, measured on this engine (see the module note; the RAM figure
 *  is what the processor's experts reach beside the card: Qwen3.6-35B-A3B Q4_K_M writes 44.5 tok/s). */
const GPU_GBS = 260;
const RAM_GBS = 56;
const SSD_GBS = 2;

export interface SpeedEstimate {
  tps: number;
  /** True when part of the weights would be read from the drive for every word. */
  swapping: boolean;
  /** Plain sentence for the card. */
  note: string;
}

/** Writing speed of a file of `sizeGiB` (`activeShare` of it read per word) on this machine. */
export function estimateSpeed(sizeGiB: number, activeShare: number, hw: HardwareProfile): SpeedEstimate {
  const vram = Math.max(0, (hw.gpu?.vramTotalGiB ?? 0) - 1.5);
  const ram = Math.max(0, (hw.ramTotalGiB ?? 0) * 0.75 - 2);
  const onGpu = Math.min(sizeGiB, vram);
  const inRam = Math.min(sizeGiB - onGpu, ram);
  const onDrive = Math.max(0, sizeGiB - onGpu - inRam);
  // Each word reads the active share of every tier.
  const secs = (activeShare * onGpu) / GPU_GBS + (activeShare * inRam) / RAM_GBS + (activeShare * onDrive) / SSD_GBS;
  const tps = secs > 0 ? 1 / secs : 0;
  const swapping = onDrive > 0.05 * sizeGiB;
  const rate = `About ${tps < 10 ? tps.toFixed(1) : Math.round(tps)} tokens per second on this computer.`;
  const note = swapping ? `Heavy swapping: ${onDrive.toFixed(0)} GiB of it is read from the drive for every word. ${rate}` : rate;
  return { tps, swapping, note };
}

export interface LatestPick {
  kind: Kind;
  repo: HubRepo;
  parsed: Parsed;
  file: QuantPick;
  speed: SpeedEstimate;
}

/**
 * The newest repository of `kind` (repos newest first) whose recommended file runs here, reading
 * file lists through `files` (one request per repository tried). A model that only runs with heavy
 * swapping is taken only when it is at least 2 tok/s, so a giant model is offered on machines that
 * can run it and skipped on ones that cannot.
 */
export async function pickLatest(kind: Kind, repos: readonly HubRepo[], hw: HardwareProfile, files: (repo: string) => Promise<QuantFile[]>, tries = 6): Promise<LatestPick | null> {
  const seen = new Set<string>();
  let n = 0;
  for (const repo of repos) {
    const parsed = parseRepo(repo.id);
    if (!parsed || parsed.kind !== kind) continue;
    // One repository per model (several publishers upload the same one).
    const key = (repo.id.split("/").pop() ?? "").toLowerCase().replace(/-gguf$/, "");
    if (seen.has(key)) continue;
    seen.add(key);
    if (n++ >= tries) break;
    // Only files whose size fits the model (1.4 to 17 bits per weight): repositories also hold
    // small side files (draft models, projectors) that would otherwise look like a tiny quant.
    const list = (await files(repo.id).catch(() => [] as QuantFile[])).filter((f) => {
      const bpw = (f.sizeGiB * 1024 ** 3 * 8) / (parsed.totalB * 1e9);
      return bpw >= 1.4 && bpw <= 17;
    });
    const file = recommendQuant(list, hw);
    if (!file || file.slow) continue;
    const speed = estimateSpeed(file.file.sizeGiB, parsed.activeB / parsed.totalB, hw);
    if (speed.swapping && speed.tps < 2) continue;
    return { kind, repo, parsed, file, speed };
  }
  return null;
}
