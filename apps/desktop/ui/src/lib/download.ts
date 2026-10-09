/**
 * Hands the user a text file to save. The bytes go from memory to a `blob:` URL on this origin and
 * straight to the browser's (or WebView2's) download, never to any server; the URL is revoked
 * once the download has started. `name` must already be a safe file name (lib/monitor.ts
 * exportFilename builds one).
 */
export function saveText(name: string, mime: string, text: string): boolean {
  try {
    const url = URL.createObjectURL(new Blob([text], { type: `${mime};charset=utf-8` }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.rel = "noopener";
    a.hidden = true;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
    return true;
  } catch {
    return false;
  }
}

/** Copies text; falls back to a hidden selection where the async clipboard is unavailable. */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.className = "sr-only";
      document.body.append(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}
