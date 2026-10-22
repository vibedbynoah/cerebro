import { matches, passages, terms, words, type Doc } from "./documents";

/**
 * Retrieval for Chat: the passages of the indexed documents that best answer a message, and the
 * prompt block they are added to. The ranking is BM25 over every passage (lib/documents.ts ranks
 * one passage per file, which is what the Documents screen shows). Nothing is uploaded: the
 * passages go into the request the app already sends to the model.
 */

export interface Passage {
  docId: string;
  docName: string;
  /** The passage text. */
  text: string;
  /** Which passage of the file it is (0-based), so a chip can open it again. */
  index: number;
  score: number;
}

/** A passage as the model and the user see it: numbered, with the file it came from. */
export interface Source {
  /** The number the model quotes and the chip shows (1-based). */
  n: number;
  docId: string;
  docName: string;
  text: string;
  /** Which passage of the file it is (0-based). */
  index: number;
}

/** Rough token count: four characters a token is close enough for a budget. */
export const approxTokens = (text: string): number => Math.ceil(text.length / 4);

/** How many tokens the passages may take by default. */
export const DEFAULT_CONTEXT_TOKENS = 1500;

export interface RetrieveOptions {
  /** How many passages at most. */
  limit?: number;
  /** How many passages one file may contribute (keeps one long file from filling the budget). */
  perDoc?: number;
}

/**
 * The passages that best answer `query`, best first. A file contributes at most `perDoc` passages,
 * so the answer is not taken from one document alone.
 */
export function retrieve(docs: readonly Doc[], query: string, options: RetrieveOptions = {}): Passage[] {
  const limit = options.limit ?? 6;
  const perDoc = options.perDoc ?? 2;
  const ts = terms(query);
  if (!ts.length) return [];
  const all: Array<{ doc: Doc; text: string; index: number; ws: string[] }> = [];
  for (const doc of docs) {
    if (doc.progress < 1) continue;
    passages(doc.text).forEach((text, index) => all.push({ doc, text, index, ws: words(text) }));
  }
  if (!all.length) return [];
  const avg = all.reduce((n, p) => n + p.ws.length, 0) / all.length;
  const df = ts.map((t) => all.filter((p) => p.ws.some((w) => matches(w, t))).length);
  const hits: Passage[] = [];
  for (const p of all) {
    let score = 0;
    let matched = 0;
    ts.forEach((t, i) => {
      const tf = p.ws.filter((w) => matches(w, t)).length;
      if (!tf) return;
      matched++;
      const idf = Math.log(1 + (all.length - (df[i] ?? 0) + 0.5) / ((df[i] ?? 0) + 0.5));
      score += (idf * tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (p.ws.length / avg)));
    });
    if (!matched) continue;
    hits.push({ docId: p.doc.id, docName: p.doc.name, text: p.text, index: p.index, score });
  }
  hits.sort((a, b) => b.score - a.score);
  const perFile = new Map<string, number>();
  const out: Passage[] = [];
  for (const h of hits) {
    const n = perFile.get(h.docId) ?? 0;
    if (n >= perDoc) continue;
    perFile.set(h.docId, n + 1);
    out.push(h);
    if (out.length >= limit) break;
  }
  return out;
}

export interface ContextBlock {
  /** The passages, numbered, ready to put in a prompt. */
  text: string;
  /** The numbered sources, in the same order as the text. */
  sources: Source[];
  /** Rough tokens the block uses. */
  tokens: number;
}

const normalize = (s: string): string => s.toLowerCase().replace(/\s+/g, " ").trim();

/**
 * Numbers the passages, drops the ones already in it, and keeps the block inside the token
 * budget. The first passage is always kept: an answer with something in it beats an empty one.
 */
export function assembleContext(passages: readonly Passage[], budgetTokens = DEFAULT_CONTEXT_TOKENS): ContextBlock {
  const sources: Source[] = [];
  const blocks: string[] = [];
  const seen = new Set<string>();
  let tokens = 0;
  for (const p of passages) {
    const text = p.text.trim();
    if (!text) continue;
    const key = `${p.docId}::${normalize(text)}`;
    if (seen.has(key)) continue;
    const cost = approxTokens(text) + approxTokens(p.docName) + 8;
    if (sources.length && tokens + cost > budgetTokens) continue;
    seen.add(key);
    const n = sources.length + 1;
    sources.push({ n, docId: p.docId, docName: p.docName, text, index: p.index });
    blocks.push(`[${n}] ${p.docName}\n${text}`);
    tokens += cost;
  }
  return { text: blocks.join("\n\n"), sources, tokens };
}

/** The block as it goes into the request, with the rule for quoting the passages. */
export function ragPrompt(context: ContextBlock): string {
  if (!context.sources.length) return "";
  return [
    "The user has documents on this computer. Use the passages below to answer, and quote the passage number when you use one, like [1].",
    "If the passages do not answer the question, say so plainly.",
    "",
    context.text,
  ].join("\n");
}
