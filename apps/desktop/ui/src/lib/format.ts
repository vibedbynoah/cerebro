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
