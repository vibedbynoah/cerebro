import { readPref, writePref } from "./storage";

/**
 * Documents: a local knowledge base, indexed on this machine. Files are read as text, split into
 * passages of about PASSAGE_CHARS characters on sentence boundaries, and searched with BM25 keyword
 * ranking over those passages. Nothing is uploaded. Meaning-based search arrives with the engine's
 * embedding models; until then the screen says keyword search, because that is what this is.
 */

export interface Doc {
  id: string;
  name: string;
  type: string;
  bytes: number;
  /** Passages the text was really split into. */
  chunks: number;
  /** 1 once indexed. Indexing is synchronous, so a stored doc is always 1. */
  progress: number;
  text: string;
  addedAt: number;
  /** The file was longer than MAX_CHARS; only the start is searchable. */
  truncated?: boolean;
  /** One optional folder, null for none. */
  folder: string | null;
  /** Tags, like a chat's: what the file is about. */
  tags: string[];
  /** The collections this file is in (collection ids, lib/collections.ts). */
  collections: string[];
}

export const DOCS_KEY = "cerebro-documents";
/** Formats read as plain text. */
const TEXT_FILES = /\.(txt|text|md|markdown|csv|tsv|json|jsonl|xml|ya?ml|toml|ini|cfg|log|html?|rs|ts|tsx|js|jsx|mjs|py|c|cc|cpp|h|hpp|java|kt|go|cs|rb|php|sh|ps1|bat|sql|css|scss)$/i;
/** Formats the converters turn into clean text (lib/convert). */
const CONVERTED_FILES = /\.(pdf|docx)$/i;
export const MAX_CHARS = 200_000;
/** Everything stored together stays well under the browser's ~5 MB local-storage budget. */
export const MAX_TOTAL_CHARS = 2_000_000;
export const PASSAGE_CHARS = 800;

/** True when Cerebro can turn this file into clean text (plain text, or a converted format). */
export function canIndex(name: string): boolean {
  return TEXT_FILES.test(name) || CONVERTED_FILES.test(name);
}

export function checkFile(name: string, bytes: number): string | null {
  if (!canIndex(name)) return "Supported: text, Markdown, HTML, CSV, JSON, PDF, Word (.docx) and code files.";
  if (!(bytes > 0) || bytes > 20 * 1024 * 1024) return "Files must be under 20 MB.";
  return null;
}

/** Splits text into passages of about PASSAGE_CHARS, on paragraph then sentence boundaries. */
export function passages(text: string): string[] {
  const out: string[] = [];
  let cur = "";
  const push = () => {
    const t = cur.trim();
    if (t) out.push(t);
    cur = "";
  };
  for (const para of text.split(/\n\s*\n/)) {
    for (const sentence of para.split(/(?<=[.!?])\s+/)) {
      if (cur && cur.length + sentence.length + 1 > PASSAGE_CHARS) push();
      // A single run-on "sentence" longer than a passage is cut hard.
      for (let s = sentence; s; s = s.slice(PASSAGE_CHARS)) {
        if (s.length > PASSAGE_CHARS) {
          push();
          cur = s.slice(0, PASSAGE_CHARS);
          push();
        } else {
          cur = cur ? `${cur} ${s}` : s;
          break;
        }
      }
    }
    if (cur.length > PASSAGE_CHARS / 2) push();
  }
  push();
  return out;
}

/** Builds a stored document from a file's real text. */
export function makeDoc(
  id: string,
  name: string,
  type: string,
  bytes: number,
  raw: string,
  addedAt: number,
  extra?: { folder?: string | null; tags?: readonly string[]; collections?: readonly string[] },
): Doc {
  const truncated = raw.length > MAX_CHARS;
  const text = truncated ? raw.slice(0, MAX_CHARS) : raw;
  return {
    id,
    name: name.slice(0, 260),
    type: type.slice(0, 100),
    bytes,
    chunks: passages(text).length,
    progress: 1,
    text,
    addedAt,
    truncated,
    folder: extra?.folder ? extra.folder.slice(0, 60) : null,
    tags: [...new Set((extra?.tags ?? []).map((t) => t.trim().slice(0, 30)).filter(Boolean))].slice(0, 8),
    collections: [...new Set((extra?.collections ?? []).map((c) => c.slice(0, 64)).filter(Boolean))].slice(0, 12),
  };
}

/** Example files for the browser preview only; the desktop app starts with the user's own. */
export const SAMPLE_DOCS: readonly Doc[] = [
  {
    id: "d-handbook",
    name: "Employee handbook 2026.txt",
    type: "text/plain",
    bytes: 214,
    chunks: 1,
    progress: 1,
    addedAt: 0,
    text: "Remote work is allowed up to three days a week with manager approval. Expense claims must be filed within 30 days with a receipt. Annual leave is 25 days plus public holidays. Laptops are replaced every three years.",
    folder: null,
    tags: [],
    collections: [],
  },
  {
    id: "d-pricing",
    name: "Pricing notes.md",
    type: "text/markdown",
    bytes: 197,
    chunks: 1,
    progress: 1,
    addedAt: 0,
    text: "The starter plan is 19 euros a month for up to three seats. Annual billing gives two months free. Enterprise pricing is per seat with a minimum of 50 seats and includes on-premise deployment.",
    folder: null,
    tags: [],
    collections: [],
  },
  {
    id: "d-runbook",
    name: "Server runbook.txt",
    type: "text/plain",
    bytes: 206,
    chunks: 1,
    progress: 1,
    addedAt: 0,
    text: "To restart the inference server, stop the service, clear the cache folder and start it again. If the GPU runs out of memory, lower the context length or offload more expert layers to the processor.",
    folder: null,
    tags: [],
    collections: [],
  },
];

export function parseDocs(raw: string | null, samples = !__DESKTOP__): Doc[] {
  if (!raw) return samples ? SAMPLE_DOCS.map((d) => ({ ...d })) : [];
  try {
    const v: unknown = JSON.parse(raw);
    if (!Array.isArray(v)) return [];
    return v
      .flatMap((x): Doc[] => {
        if (!x || typeof x !== "object") return [];
        const d = x as Record<string, unknown>;
        if (typeof d.id !== "string" || typeof d.name !== "string" || d.name.length > 260 || typeof d.text !== "string" || d.text.length > MAX_CHARS) return [];
        const n = (k: string) => (typeof d[k] === "number" && Number.isFinite(d[k]) && (d[k] as number) >= 0 ? (d[k] as number) : 0);
        const folder = typeof d.folder === "string" && d.folder.trim() ? d.folder.trim().slice(0, 60) : null;
        const tags = Array.isArray(d.tags) ? d.tags.flatMap((t) => (typeof t === "string" && t.trim() ? [t.trim().slice(0, 30)] : [])).slice(0, 8) : [];
        const collections = Array.isArray(d.collections) ? d.collections.flatMap((c) => (typeof c === "string" && c.trim() ? [c.trim().slice(0, 64)] : [])).slice(0, 12) : [];
        return [{ id: d.id, name: d.name, type: typeof d.type === "string" ? d.type.slice(0, 100) : "", bytes: n("bytes"), chunks: passages(d.text).length, progress: 1, text: d.text, addedAt: n("addedAt"), truncated: d.truncated === true, folder, tags, collections }];
      })
      .slice(0, 1000);
  } catch {
    return [];
  }
}

export function loadDocs(): Doc[] {
  return parseDocs(readPref(DOCS_KEY));
}

export function saveDocs(docs: readonly Doc[]): void {
  writePref(DOCS_KEY, JSON.stringify(docs));
}

export const totalChars = (docs: readonly Doc[]): number => docs.reduce((n, d) => n + d.text.length, 0);

export interface Hit {
  doc: Doc;
  /** Share of the query's words found in the passage, 0-1. */
  score: number;
  matched: number;
  of: number;
  /** The best sentence of the passage, split into plain and matched parts (rendered as text). */
  parts: Array<{ v: string; hit: boolean }>;
}

const STOP = new Set(["the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for", "how", "what", "do", "i", "my", "with", "be", "it", "when", "does", "can"]);

export function terms(q: string): string[] {
  return [...new Set(q.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter((t) => t.length > 1 && !STOP.has(t)))].slice(0, 12);
}

export const words = (s: string): string[] => s.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean);

/** A word matches a query term when it starts with it, so "backups" finds "backup" and "run" finds "runs". */
export const matches = (word: string, term: string): boolean => word.startsWith(term) || (term.length > 3 && term.startsWith(word) && word.length >= term.length - 1);

/**
 * BM25 (k1 1.2, b 0.75) over every passage of every document; one hit per document, its best passage,
 * best first.
 */
export function search(docs: readonly Doc[], q: string, limit = 5): Hit[] {
  const ts = terms(q);
  if (!ts.length) return [];
  const all: Array<{ doc: Doc; text: string; ws: string[] }> = [];
  for (const doc of docs) {
    if (doc.progress < 1) continue;
    for (const p of passages(doc.text)) all.push({ doc, text: p, ws: words(p) });
  }
  if (!all.length) return [];
  const avg = all.reduce((n, p) => n + p.ws.length, 0) / all.length;
  const df = ts.map((t) => all.filter((p) => p.ws.some((w) => matches(w, t))).length);
  const best = new Map<string, { score: number; matched: number; text: string; doc: Doc }>();
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
    const cur = best.get(p.doc.id);
    if (!cur || score > cur.score) best.set(p.doc.id, { score, matched, text: p.text, doc: p.doc });
  }
  return [...best.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((h) => ({ doc: h.doc, score: h.matched / ts.length, matched: h.matched, of: ts.length, parts: mark(bestSentence(h.text, ts), ts) }));
}

function bestSentence(passage: string, ts: readonly string[]): string {
  const sentences = passage.split(/(?<=[.!?])\s+/);
  let top = sentences[0] ?? passage;
  let topN = -1;
  for (const s of sentences) {
    const ws = words(s);
    const n = ts.filter((t) => ws.some((w) => matches(w, t))).length;
    if (n > topN) [top, topN] = [s, n];
  }
  return top;
}

export function mark(s: string, ts: readonly string[]): Array<{ v: string; hit: boolean }> {
  if (!ts.length) return [{ v: s, hit: false }];
  const esc = ts.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${esc.map((e) => `${e}[\\p{L}\\p{N}]*`).join("|")})`, "giu");
  return s
    .split(re)
    .filter(Boolean)
    .map((v) => ({ v, hit: ts.some((t) => matches(v.toLowerCase(), t)) }));
}
