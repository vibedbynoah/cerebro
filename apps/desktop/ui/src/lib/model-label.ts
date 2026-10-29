import type { LibraryModel } from "../service/types";

/** The publisher a model file came from, read from its folder: "publisher__name" (Cerebro's own
 *  downloads) or publisher/name (other apps' model folders). Null outside a models folder. */
export function publisherOf(m: Pick<LibraryModel, "path">): string | null {
  const parts = m.path.split(/[\/]/);
  const i = parts.lastIndexOf("models");
  if (i < 0 || i + 2 >= parts.length) return null;
  const dir = parts[i + 1]!;
  return dir.includes("__") ? dir.split("__")[0]! : dir;
}

/** "Qwen3.8 27B · Q4_K_M · unsloth": enough to tell variants of one model apart. */
export function modelLabel(m: Pick<LibraryModel, "name" | "quant" | "path">): string {
  return [m.name, m.quant, publisherOf(m)].filter((x) => x && x !== "other").join(" · ");
}
