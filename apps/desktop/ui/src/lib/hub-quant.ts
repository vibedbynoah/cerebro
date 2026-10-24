/**
 * The file Cerebro would use on this computer, chosen from a repository's own file list.
 *
 * Nothing here is a guess about a model: the candidates are the files the repository really has,
 * with their real sizes, and the machine is the one the app detected. Unsloth Dynamic quants
 * ("UD-") come first because they keep the most quality per byte; among those, the largest file
 * that still fits wins. When not even the smallest file fits, the smallest is recommended and the
 * reason says it will run slowly from the drive.
 */

import type { HardwareProfile } from "../service/types";

const GIB = 1024 ** 3;

/** One file of a repository, as the chooser sees it. A split model is one entry with all its parts. */
export interface QuantFile {
  /** The file's path in the repository; for a split model, its first part. */
  path: string;
  /** The quant label, e.g. "UD-Q4_K_XL". */
  quant: string;
  /** Size on disk, GiB, all parts of a split model together. */
  sizeGiB: number;
  /** How many parts the file is split into (1 for an ordinary file). */
  parts: number;
}

/** The file Cerebro would use here, and why. */
export interface QuantPick {
  file: QuantFile;
  /** One plain sentence: why this file is the one. */
  why: string;
  /** True when not even this file fits, so it runs slowly from the drive. */
  slow: boolean;
}

/**
 * How a file fits this machine: all in graphics memory, graphics memory with system memory, system
 * memory only, or not at all. The file needs some room over its size for the cache and the driver,
 * and only the memory the machine can spare is counted.
 */
export function fitOf(sizeGiB: number, vramGiB: number, ramGiB: number): { label: string; tone: "good" | "ok" | "slow" | "no" } {
  const need = sizeGiB * 1.1;
  if (vramGiB > 0 && need <= vramGiB * 0.92) return { label: "In graphics memory", tone: "good" };
  if (need <= vramGiB * 0.92 + ramGiB * 0.7) return { label: vramGiB > 0 ? "Graphics + system memory" : "In system memory", tone: vramGiB > 0 ? "ok" : "slow" };
  if (need <= ramGiB * 0.9 + vramGiB) return { label: "Tight: most of system memory", tone: "slow" };
  return { label: "Too large for this computer", tone: "no" };
}

/** True for an Unsloth Dynamic quant: "UD-" in the quant label or in the file name. */
export function isDynamicQuant(file: Pick<QuantFile, "quant" | "path">): boolean {
  return /(^|[-_])ud[-_]/i.test(file.quant) || /[-_]ud[-_]/i.test(file.path);
}

/** The memory of this machine, in GiB, as the Hub's fit test uses it. */
function memoryOf(hw: HardwareProfile): { vramGiB: number; ramGiB: number } {
  return { vramGiB: Math.max(0, hw.gpu?.vramTotalGiB ?? 0), ramGiB: Math.max(0, hw.ramTotalGiB ?? 0) };
}

/** Why a file that does not fit is still recommended: it is the smallest one there is. */
function whySlow(file: QuantFile): string {
  return `${file.quant}: the smallest file here, so it will run slowly from the drive.`;
}

/** The largest of a list; the smallest path wins a tie, so the answer never changes between runs. */
function largest(files: readonly QuantFile[]): QuantFile | undefined {
  return files.reduce<QuantFile | undefined>((best, f) => {
    if (!best) return f;
    if (f.sizeGiB > best.sizeGiB) return f;
    if (f.sizeGiB === best.sizeGiB && f.path < best.path) return f;
    return best;
  }, undefined);
}

/** Bits per weight a quant label stands for, roughly: what decides quality and speed. */
export function bitsOf(quant: string): number {
  const q = quant.toUpperCase();
  if (/F32|FP32/.test(q)) return 32;
  if (/BF16|F16|FP16/.test(q)) return 16;
  const m = /(?:^|[^A-Z])I?Q([1-8])/.exec(q) ?? /^I?Q([1-8])/.exec(q.replace(/^UD-/, ""));
  return m ? Number(m[1]) : 8;
}

/**
 * The file Cerebro would use on this computer, from the repository's own file list.
 *
 * Speed comes first, then quality: a model that runs entirely on the graphics card is several
 * times faster than one split with the processor, and above 8 bits a file gains no quality anyone
 * can tell. So, Unsloth Dynamic quants first when the repository has them:
 *  1. the largest file of at most 8 bits that fits the graphics card (16-bit only when the
 *     repository has nothing smaller, as some speech models do);
 *  2. else the largest 4- or 5-bit file that fits the card and system memory together;
 *  3. else the largest file of at most 4 bits that fits at all;
 *  4. else the smallest file, which runs slowly from the drive.
 *
 * Null when the repository has no files with a size, or when the machine could not be read: then
 * there is nothing to judge a file against.
 */
export function recommendQuant(files: readonly QuantFile[], hw: HardwareProfile | undefined): QuantPick | null {
  if (!hw) return null;
  const list = files.filter((f) => Number.isFinite(f.sizeGiB) && f.sizeGiB > 0);
  if (!list.length) return null;
  const { vramGiB, ramGiB } = memoryOf(hw);
  const tone = (f: QuantFile) => fitOf(f.sizeGiB, vramGiB, ramGiB).tone;
  const dynamic = list.filter(isDynamicQuant);
  // The dynamic quants when they include a match, else every file.
  const best = (ok: (f: QuantFile) => boolean) => largest(dynamic.filter(ok)) ?? largest(list.filter(ok));
  const gpu = Math.round(hw.gpu?.vramTotalGiB ?? 0);
  const ram = Math.round(hw.ramTotalGiB ?? 0);
  const lowest = Math.min(...list.map((f) => bitsOf(f.quant)));
  const cap = Math.max(8, lowest);
  const onCard = best((f) => tone(f) === "good" && bitsOf(f.quant) <= cap);
  if (onCard) return { file: onCard, why: `${onCard.quant}: the best quality that runs entirely on your ${gpu} GB graphics card.`, slow: false };
  const split = best((f) => tone(f) === "ok" && bitsOf(f.quant) >= 4 && bitsOf(f.quant) <= 5);
  if (split) {
    const where = gpu > 0 ? `your ${gpu} GB graphics card and ${ram} GB of memory together` : `your ${ram} GB of memory`;
    return { file: split, why: `${split.quant}: fits ${where}; larger files would run much slower here.`, slow: false };
  }
  const small = best((f) => tone(f) !== "no" && bitsOf(f.quant) <= 4);
  if (small) return { file: small, why: `${small.quant}: the most quality that still fits this computer's memory.`, slow: false };
  const smallest = list.reduce((a, b) => (b.sizeGiB < a.sizeGiB ? b : a));
  return { file: smallest, why: whySlow(smallest), slow: true };
}

/** Bytes to GiB, for the sizes the Hub reads from Hugging Face. */
export const toGiB = (bytes: number): number => bytes / GIB;
