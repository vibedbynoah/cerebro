const one = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const whole = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/** "16.8" — one decimal, fixed locale, so readouts never change width between locales. */
export function fmt1(n: number): string {
  return one.format(n);
}

/** "32,768" */
export function fmtInt(n: number): string {
  return whole.format(n);
}

/** Context length in tokens as "32k" (powers of two of 1024), falling back to the full number. */
export function fmtContext(tokens: number): string {
  return tokens % 1024 === 0 ? `${tokens / 1024}k` : fmtInt(tokens);
}

/** Share of the whole, clamped to 0..1; 0 when the total is not positive. */
export function ratio(part: number, total: number): number {
  if (!(total > 0) || !Number.isFinite(part)) return 0;
  return Math.min(1, Math.max(0, part / total));
}

const DAY = 86_400_000;

/**
 * How old an ISO date is, in plain words: "today", "yesterday", "3 days ago", "2 weeks ago",
 * "5 months ago", "a year ago". Used for a repository's age, which is what tells a reader how
 * settled a model is.
 */
export function fmtAge(iso: string, now: number = Date.now()): string {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "an unknown date";
  const days = Math.floor((now - t) / DAY);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 14) return `${days} days ago`;
  if (days < 60) return `${Math.floor(days / 7)} weeks ago`;
  if (days < 365) return `${Math.floor(days / 30)} months ago`;
  const years = Math.floor(days / 365);
  return years === 1 ? "a year ago" : `${years} years ago`;
}
