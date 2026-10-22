/**
 * Publishers people trust for GGUF files and original weights: Cerebro's own checked list.
 *
 * The badge it earns says one thing and no more — the publisher is on this list. It is not a
 * guarantee about a file, a licence, or a model. Anyone can be added; nothing is claimed that has
 * not been checked by hand.
 */

import type { HubRepo } from "../service/types";

export interface VerifiedPublisher {
  /** The publisher's account name on Hugging Face, e.g. "unsloth". */
  id: string;
  /** The name people know them by, e.g. "Unsloth". */
  name: string;
  /** What they publish, in a few words. */
  publishes: string;
}

/**
 * The list. Model makers publish the original weights and, some of them, their own conversions;
 * the second group publishes GGUF files of other people's models. Account names are matched
 * without regard to case, the way Hugging Face itself treats them.
 */
export const VERIFIED_PUBLISHERS: readonly VerifiedPublisher[] = [
  { id: "Qwen", name: "Qwen", publishes: "Original Qwen weights and official conversions." },
  { id: "google", name: "Google (Gemma)", publishes: "Original Gemma weights and official conversions." },
  { id: "meta-llama", name: "Meta (Llama)", publishes: "Original Llama weights and official conversions." },
  { id: "mistralai", name: "Mistral AI", publishes: "Original Mistral weights and official conversions." },
  { id: "microsoft", name: "Microsoft", publishes: "Original Phi weights and official conversions." },
  { id: "deepseek-ai", name: "DeepSeek", publishes: "Original DeepSeek weights." },
  { id: "openai", name: "OpenAI", publishes: "Original open-weight models, such as gpt-oss." },
  { id: "nvidia", name: "NVIDIA", publishes: "Original Nemotron weights and official conversions." },
  { id: "ibm-granite", name: "IBM Granite", publishes: "Original Granite weights." },
  { id: "HuggingFaceTB", name: "Hugging Face", publishes: "Original SmolLM weights and official conversions." },
  { id: "unsloth", name: "Unsloth", publishes: "Dynamic GGUF quants and fine-tunes." },
  { id: "ggml-org", name: "llama.cpp (ggml)", publishes: "Official GGUF conversions made with llama.cpp." },
  { id: "lmstudio-community", name: "lmstudio-community", publishes: "Community GGUF conversions." },
  { id: "bartowski", name: "Bartowski", publishes: "Community GGUF quantisations." },
];

const BY_ID: ReadonlyMap<string, VerifiedPublisher> = new Map(VERIFIED_PUBLISHERS.map((p) => [p.id.toLowerCase(), p]));

/** What the badge says it means. Shown as its tooltip and in the note beside the filter. */
export const VERIFIED_TOOLTIP = "On Cerebro's list of known publishers. It is not a guarantee.";

/** The same thing in words on the page, so it does not depend on hovering the badge. */
export const VERIFIED_NOTE = "The Verified badge marks publishers on Cerebro's list of known publishers. It is not a guarantee.";

/** The publisher of a repository, from its Hugging Face id ("unsloth/Qwen3-8B-GGUF" -> "unsloth"). */
export function publisherOf(repoId: string): string {
  const id = repoId.trim();
  const slash = id.indexOf("/");
  return (slash === -1 ? id : id.slice(0, slash)).trim();
}

/** The list's entry for a repository's publisher, or null when the publisher is not on it. */
export function verifiedPublisher(repoId: string): VerifiedPublisher | null {
  return BY_ID.get(publisherOf(repoId).toLowerCase()) ?? null;
}

/** True when the repository's publisher is on Cerebro's list. */
export function isVerified(repoId: string): boolean {
  return verifiedPublisher(repoId) !== null;
}

/**
 * How well a repository matches the search, 3 down to 0:
 * 3 the whole query is in the repository's name, 2 every word of it is, 1 some are, 0 the name
 * does not match at all (Hugging Face matched it another way).
 */
export function relevance(repo: HubRepo, query: string): number {
  const name = (repo.id.split("/")[1] ?? repo.id).toLowerCase();
  const author = repo.author.toLowerCase();
  const q = query.toLowerCase().replace(/\bgguf\b/g, " ").replace(/\s+/g, " ").trim();
  if (!q) return 1;
  if (name.includes(q)) return 3;
  const words = q.split(" ").filter(Boolean);
  if (!words.length) return 1;
  const hits = words.filter((w) => name.includes(w) || author.includes(w)).length;
  if (hits === words.length) return 2;
  return hits > 0 ? 1 : 0;
}

/**
 * The search results in the order they are shown: the best match first, and at equal relevance the
 * publishers on Cerebro's list come before the rest. Downloads settle what is left, so a quiet
 * repository never displaces a busy one of the same standing.
 */
export function sortRepos(repos: readonly HubRepo[], query: string): HubRepo[] {
  return [...repos].sort((a, b) => {
    const rel = relevance(b, query) - relevance(a, query);
    if (rel !== 0) return rel;
    const verified = Number(isVerified(b.id)) - Number(isVerified(a.id));
    if (verified !== 0) return verified;
    return b.downloads - a.downloads;
  });
}
