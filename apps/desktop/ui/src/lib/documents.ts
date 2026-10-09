import { readPref, writePref } from "./storage";

/**
 * Documents: a local knowledge base. Files are indexed on this machine (chunked and embedded by the
 * engine); search ranks chunks by meaning. In this build the index and the ranking are illustrative:
 * a keyword score over each document's text stands in for embeddings.
 */

export interface Doc {
  id: string;
  name: string;
  type: string;
  bytes: number;
  chunks: number;
  /** 0-1 while indexing, 1 when ready. */
  progress: number;
  text: string;
  addedAt: number;
}

export const DOCS_KEY = "cerebro-documents";
const ACCEPT = /\.(pdf|txt|md|docx|html?|csv|json|rs|ts|tsx|py|js)$/i;

export function checkFile(name: string, bytes: number): string | null {
  if (!ACCEPT.test(name)) return "Supported: PDF, Word, text, Markdown, HTML, CSV, JSON and code files.";
  if (!(bytes > 0) || bytes > 100 * 1024 * 1024) return "Files must be under 100 MB.";
  return null;
}

export const SAMPLE_DOCS: readonly Doc[] = [
  {
    id: "d-handbook",
    name: "Employee handbook 2026.pdf",
    type: "application/pdf",
    bytes: 1_840_000,
    chunks: 212,
    progress: 1,
    addedAt: 0,
    text: "Remote work is allowed up to three days a week with manager approval. Expense claims must be filed within 30 days with a receipt. Annual leave is 25 days plus public holidays. Laptops are replaced every three years.",
  },
  {
    id: "d-pricing",
    name: "Pricing notes.md",
    type: "text/markdown",
    bytes: 18_400,
    chunks: 9,
    progress: 1,
    addedAt: 0,
    text: "The starter plan is 19 euros a month for up to three seats. Annual billing gives two months free. Enterprise pricing is per seat with a minimum of 50 seats and includes on-premise deployment.",
  },
  {
    id: "d-runbook",
    name: "Server runbook.txt",
    type: "text/plain",
    bytes: 42_000,
    chunks: 21,
    progress: 1,
    addedAt: 0,
    text: "To restart the inference server, stop the service, clear the cache folder and start it again. If the GPU runs out of memory, lower the context length or offload more expert layers to the processor.",
  },
];

export function parseDocs(raw: string | null): Doc[] {
  if (!raw) return SAMPLE_DOCS.map((d) => ({ ...d }));
  try {
    const v: unknown = JSON.parse(raw);
    if (!Array.isArray(v)) return [];
    return v
      .flatMap((x): Doc[] => {
        if (!x || typeof x !== "object") return [];
        const d = x as Record<string, unknown>;
        if (typeof d.id !== "string" || typeof d.name !== "string" || d.name.length > 260 || typeof d.text !== "string" || d.text.length > 20000) return [];
        const n = (k: string) => (typeof d[k] === "number" && Number.isFinite(d[k]) && (d[k] as number) >= 0 ? (d[k] as number) : 0);
        return [{ id: d.id, name: d.name, type: typeof d.type === "string" ? d.type.slice(0, 100) : "", bytes: n("bytes"), chunks: n("chunks"), progress: Math.min(1, n("progress")), text: d.text, addedAt: n("addedAt") }];
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

export interface Hit {
  doc: Doc;
  score: number;
  /** The sentence that matched, split into plain and matched parts (rendered as text). */
  parts: Array<{ v: string; hit: boolean }>;
}

const STOP = new Set(["the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for", "how", "what", "do", "i", "my", "with", "be", "it"]);

export function terms(q: string): string[] {
  return q
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length > 1 && !STOP.has(t))
    .slice(0, 12);
}

export function search(docs: readonly Doc[], q: string, limit = 5): Hit[] {
  const ts = terms(q);
  if (!ts.length) return [];
  const hits: Hit[] = [];
  for (const doc of docs) {
    if (doc.progress < 1) continue;
    const sentences = doc.text.split(/(?<=[.!?])\s+/);
    let best: { s: string; score: number } | null = null;
    for (const s of sentences) {
      const low = s.toLowerCase();
      const score = ts.reduce((n, t) => n + (low.includes(t) ? 1 : 0), 0);
      if (score > 0 && (!best || score > best.score)) best = { s, score };
    }
    if (best) hits.push({ doc, score: best.score / ts.length, parts: mark(best.s, ts) });
  }
  return hits.sort((a, b) => b.score - a.score).slice(0, limit);
}

export function mark(s: string, ts: readonly string[]): Array<{ v: string; hit: boolean }> {
  if (!ts.length) return [{ v: s, hit: false }];
  const esc = ts.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${esc.join("|")})`, "gi");
  return s.split(re).filter(Boolean).map((v) => ({ v, hit: ts.includes(v.toLowerCase()) }));
}
