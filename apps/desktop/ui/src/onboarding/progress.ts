import { readPref, writePref } from "../lib/storage";

/**
 * First-run setup: which steps exist, and what is remembered once it is done.
 *
 * The record is a per-machine convenience in local storage, like the theme. It holds no secret and
 * grants nothing: a missing, blocked or tampered record only means setup shows again. Everything
 * read back is validated field by field before use.
 */

export const ONBOARDING_KEY = "cerebro-onboarding";

/** Bump when the plain-language terms change: setup then reopens at the terms step only. */
export const TERMS_VERSION = "2026-09-30";

export const STEPS = [
  { id: "welcome", label: "Welcome" },
  { id: "level", label: "Experience" },
  { id: "hardware", label: "This machine" },
  { id: "model", label: "A first model" },
  { id: "configure", label: "Load settings" },
  { id: "privacy", label: "Privacy and protection" },
  { id: "terms", label: "Terms" },
  { id: "finish", label: "Ready" },
] as const;

export type StepId = (typeof STEPS)[number]["id"];

export const TERMS_STEP = STEPS.findIndex((s) => s.id === "terms");

export interface OnboardingRecord {
  v: 1;
  completed: boolean;
  /** The TERMS_VERSION the user accepted, or null before they have. */
  termsVersion: string | null;
  /** The catalog id picked during setup, or null for "decide later". */
  model: string | null;
}

/** Catalog ids are short lowercase slugs; anything else in storage is ignored. */
const MODEL_ID = /^[a-z0-9][a-z0-9._-]{0,63}$/;
const VERSION = /^\d{4}-\d{2}-\d{2}$/;

export function parseRecord(raw: string | null): OnboardingRecord | null {
  if (raw === null || raw.length > 512) return null;
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof data !== "object" || data === null || Array.isArray(data)) return null;
  const r = data as Record<string, unknown>;
  if (r["v"] !== 1 || typeof r["completed"] !== "boolean") return null;
  const terms = r["termsVersion"];
  const model = r["model"];
  if (terms !== null && (typeof terms !== "string" || !VERSION.test(terms))) return null;
  if (model !== null && (typeof model !== "string" || !MODEL_ID.test(model))) return null;
  return { v: 1, completed: r["completed"], termsVersion: terms, model };
}

export function readRecord(): OnboardingRecord | null {
  return parseRecord(readPref(ONBOARDING_KEY));
}

export function writeRecord(record: OnboardingRecord): void {
  writePref(ONBOARDING_KEY, JSON.stringify(record));
}

export function termsAccepted(record: OnboardingRecord | null): boolean {
  return record?.termsVersion === TERMS_VERSION;
}

/** Setup shows until it has been finished once with the current terms accepted. */
export function needsOnboarding(record: OnboardingRecord | null): boolean {
  return !record || !record.completed || !termsAccepted(record);
}

/** Someone who finished setup before the terms changed only sees the terms step again. */
export function firstStep(record: OnboardingRecord | null): number {
  return record?.completed === true && !termsAccepted(record) ? TERMS_STEP : 0;
}
