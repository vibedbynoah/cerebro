/**
 * Small fuzzy matcher for the command palette: every query character must appear in order
 * (case-insensitive). Contiguous runs, word starts and an early first match score higher.
 * Returns the matched indices so the UI can highlight them as text spans (never as HTML).
 */
export interface FuzzyMatch {
  score: number;
  indices: number[];
}

const isBoundary = (text: string, i: number): boolean => {
  if (i === 0) return true;
  const prev = text[i - 1] ?? "";
  const cur = text[i] ?? "";
  return /[\s\-_/.]/.test(prev) || (prev === prev.toLowerCase() && cur !== cur.toLowerCase());
};

/** Best match of `query` in `text`, or null when `query` is not a subsequence of it. */
export function fuzzyMatch(query: string, text: string): FuzzyMatch | null {
  const q = query.trim().toLowerCase();
  if (q === "") return { score: 0, indices: [] };
  const t = text.toLowerCase();

  // Greedy forward pass finds whether it matches at all; then try each start of the first char and
  // keep the best-scoring alignment (texts are short: labels and keywords).
  let best: FuzzyMatch | null = null;
  for (let start = t.indexOf(q[0] ?? ""); start !== -1; start = t.indexOf(q[0] ?? "", start + 1)) {
    const indices: number[] = [];
    let ti = start;
    for (const ch of q) {
      while (ti < t.length && t[ti] !== ch) ti++;
      if (ti >= t.length) break;
      indices.push(ti);
      ti++;
    }
    if (indices.length !== q.length) break; // later starts can only match less
    // Weights: 1 per matched character, +3 when it starts a word ("MH" in "Model Hub"), +4 when it
    // follows the previous match directly (runs beat scattered hits), a small penalty for a late
    // first match, and +6 when the label starts with the whole query.
    let score = 0;
    indices.forEach((idx, n) => {
      score += 1;
      if (isBoundary(text, idx)) score += 3;
      if (n > 0 && idx === (indices[n - 1] ?? -2) + 1) score += 4;
    });
    score -= Math.min(start, 10) * 0.2;
    if (t.startsWith(q)) score += 6;
    if (!best || score > best.score) best = { score, indices };
  }
  return best;
}

export interface Ranked<T> {
  item: T;
  match: FuzzyMatch;
}

/** Filters and sorts `items` by the best match across the fields `fields(item)` returns; the first
 *  field is the label whose indices are kept for highlighting. Stable for equal scores. */
export function fuzzyFilter<T>(query: string, items: readonly T[], fields: (item: T) => readonly string[]): Ranked<T>[] {
  const ranked: Array<Ranked<T> & { order: number }> = [];
  items.forEach((item, order) => {
    const [label = "", ...rest] = fields(item);
    const onLabel = fuzzyMatch(query, label);
    let match = onLabel;
    for (const extra of rest) {
      const m = fuzzyMatch(query, extra);
      // A keyword hit ranks below an equally good label hit and highlights nothing in the label.
      if (m && (!match || m.score - 2 > match.score)) match = { score: m.score - 2, indices: onLabel?.indices ?? [] };
    }
    if (match) ranked.push({ item, match, order });
  });
  ranked.sort((a, b) => b.match.score - a.match.score || a.order - b.order);
  return ranked.map(({ item, match }) => ({ item, match }));
}

/** Splits `text` into runs for highlighting: `{ text, hit }` pieces in order. */
export function highlightRuns(text: string, indices: readonly number[]): Array<{ text: string; hit: boolean }> {
  const hits = new Set(indices);
  const runs: Array<{ text: string; hit: boolean }> = [];
  for (let i = 0; i < text.length; i++) {
    const hit = hits.has(i);
    const last = runs[runs.length - 1];
    if (last && last.hit === hit) last.text += text[i];
    else runs.push({ text: text[i] ?? "", hit });
  }
  return runs;
}
