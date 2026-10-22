import { readPref, writePref } from "./storage";

/**
 * When the feature guide opens on its own: never seen, or the feature list changed since. Kept
 * apart from the guide and its list (lib/features.ts) so the first screen loads without them;
 * the guide itself loads when it opens. Bump GUIDE_VERSION when the list changes.
 */
export const GUIDE_VERSION = "2026-10-22";
export const GUIDE_SEEN_KEY = "cerebro-guide-seen";

export function guideDue(): boolean {
  return readPref(GUIDE_SEEN_KEY) !== GUIDE_VERSION;
}

export function markGuideSeen(): void {
  writePref(GUIDE_SEEN_KEY, GUIDE_VERSION);
}
