/**
 * Per-viewer conveniences only (theme, sidebar state). Storage can be blocked or throw (private
 * windows, cleared site data), so every access is guarded and the UI works without it.
 */
export function readPref(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writePref(key: string, value: string | null): void {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    // Storage blocked: the choice lasts for this session only.
  }
}
