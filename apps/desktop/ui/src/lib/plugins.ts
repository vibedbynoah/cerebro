import { readPref, writePref } from "./storage";

/**
 * The plugin and MCP catalogue. Each plugin lists what it can do, the permissions it asks for and
 * whether it needs a network once installed. Installs from the catalogue need a connection;
 * installing from a local file works fully offline. Installed state is kept on this machine.
 */

export type Category = "images" | "search" | "files" | "git" | "email" | "calendar" | "databases" | "browser" | "voice";
export const CATEGORIES: readonly Category[] = ["images", "search", "files", "git", "email", "calendar", "databases", "browser", "voice"];
export const CATEGORY_LABELS: Readonly<Record<Category, string>> = {
  images: "Image generation",
  search: "Web search",
  files: "Files",
  git: "Git",
  email: "Email",
  calendar: "Calendar",
  databases: "Databases",
  browser: "Browser automation",
  voice: "Voice",
};

export interface Plugin {
  id: string;
  name: string;
  category: Category;
  kind: "MCP server" | "Cerebro plugin";
  author: string;
  version: string;
  latest: string;
  licence: string;
  offline: boolean;
  summary: string;
  abilities: string[];
  permissions: string[];
  featured?: string;
}

export const PLUGINS: readonly Plugin[] = [
  { id: "comfy-bridge", name: "Node Graph", category: "images", kind: "Cerebro plugin", author: "Cerebro", version: "1.2.0", latest: "1.2.0", licence: "Apache-2.0", offline: true, summary: "Build image pipelines as node graphs and run them on the built-in image engine.", abilities: ["Chain models, upscalers and control nets", "Save graphs as presets for Image Studio"], permissions: ["Read and write the images folder"], featured: "Image pipelines without leaving Cerebro" },
  { id: "controlnet", name: "Pose & Depth Control", category: "images", kind: "Cerebro plugin", author: "Community", version: "0.9.1", latest: "0.9.3", licence: "MIT", offline: true, summary: "Guide image generation with a pose, depth map or edge sketch.", abilities: ["Pose, depth, canny and scribble guides"], permissions: ["Read images you choose"] },
  { id: "searxng", name: "Private Search", category: "search", kind: "MCP server", author: "Community", version: "2.1.0", latest: "2.1.0", licence: "AGPL-3.0", offline: false, summary: "Web search through a SearXNG instance you control. No tracking.", abilities: ["Search the web", "Fetch and summarise result pages"], permissions: ["Network: your SearXNG address only"], featured: "Search without a search company" },
  { id: "filesystem", name: "Project Files", category: "files", kind: "MCP server", author: "Cerebro", version: "1.0.4", latest: "1.0.4", licence: "Apache-2.0", offline: true, summary: "Read, search and edit files inside folders you approve.", abilities: ["List, read, search and edit files", "Every edit shown as a diff first"], permissions: ["Read and write approved folders only"] },
  { id: "pdf-tools", name: "PDF Tools", category: "files", kind: "Cerebro plugin", author: "Community", version: "0.6.0", latest: "0.6.0", licence: "MIT", offline: true, summary: "Extract text, tables and images from PDFs for chat and search.", abilities: ["Text and table extraction", "OCR for scanned pages"], permissions: ["Read files you choose"] },
  { id: "git", name: "Git", category: "git", kind: "MCP server", author: "Cerebro", version: "1.1.0", latest: "1.1.2", licence: "Apache-2.0", offline: true, summary: "Status, diff, log, branch and commit in repositories you approve.", abilities: ["Read history and diffs", "Stage and commit with your approval"], permissions: ["Read and write approved repositories", "Run the git command"] },
  { id: "github", name: "GitHub", category: "git", kind: "MCP server", author: "Community", version: "3.0.2", latest: "3.0.2", licence: "MIT", offline: false, summary: "Issues, pull requests and reviews on GitHub.", abilities: ["Read and comment on issues and pull requests"], permissions: ["Network: api.github.com", "A token you provide"] },
  { id: "mail", name: "Mail", category: "email", kind: "MCP server", author: "Community", version: "0.8.0", latest: "0.8.0", licence: "MIT", offline: false, summary: "Read and draft email over IMAP and SMTP. Sending always asks first.", abilities: ["Search and read mail", "Draft and send replies"], permissions: ["Network: your mail server", "Your mail password, stored in the OS keychain"] },
  { id: "calendar", name: "Calendar", category: "calendar", kind: "MCP server", author: "Community", version: "0.5.2", latest: "0.5.2", licence: "MIT", offline: true, summary: "Your local calendar files (ICS), or a CalDAV server.", abilities: ["Read events", "Create events with your approval"], permissions: ["Read and write your calendar files"] },
  { id: "sqlite", name: "SQLite", category: "databases", kind: "MCP server", author: "Cerebro", version: "1.0.0", latest: "1.0.0", licence: "Apache-2.0", offline: true, summary: "Query SQLite databases. Read-only unless you allow writes.", abilities: ["Explain schemas", "Run queries"], permissions: ["Read approved database files"], featured: "Ask your data questions in plain words" },
  { id: "postgres", name: "PostgreSQL", category: "databases", kind: "MCP server", author: "Community", version: "1.4.1", latest: "1.4.1", licence: "MIT", offline: false, summary: "Query a PostgreSQL database with a read-only role by default.", abilities: ["Schema browsing", "Read-only queries"], permissions: ["Network: your database address"] },
  { id: "browser", name: "Browser Automation", category: "browser", kind: "MCP server", author: "Community", version: "0.12.0", latest: "0.12.0", licence: "Apache-2.0", offline: false, summary: "Drive a headless browser: open pages, fill forms, take screenshots.", abilities: ["Navigate and click", "Screenshots and page text"], permissions: ["Network: sites you allow", "Run a headless browser"] },
  { id: "whisper", name: "Dictation", category: "voice", kind: "Cerebro plugin", author: "Cerebro", version: "1.0.1", latest: "1.0.1", licence: "Apache-2.0", offline: true, summary: "Push-to-talk anywhere in Windows, transcribed on this machine.", abilities: ["System-wide dictation", "Voice commands for Cerebro"], permissions: ["Microphone while you hold the key"] },
  { id: "kokoro", name: "Natural Voice", category: "voice", kind: "Cerebro plugin", author: "Community", version: "0.4.0", latest: "0.4.0", licence: "Apache-2.0", offline: true, summary: "Read replies aloud with a natural local voice.", abilities: ["Text to speech in 9 languages"], permissions: ["Play audio"] },
];

export const INSTALLED_KEY = "cerebro-plugins";

export interface InstalledState {
  /** id -> enabled */
  installed: Record<string, boolean>;
  /** Plugins installed from a local file, by file name (illustrative). */
  local: string[];
}

export function parseInstalled(raw: string | null): InstalledState {
  const empty: InstalledState = { installed: { filesystem: true, git: true }, local: [] };
  if (!raw) return empty;
  try {
    const v = JSON.parse(raw) as Record<string, unknown>;
    const installed: Record<string, boolean> = {};
    if (v && typeof v.installed === "object" && v.installed) {
      for (const [k, on] of Object.entries(v.installed as Record<string, unknown>)) if (PLUGINS.some((p) => p.id === k) && typeof on === "boolean") installed[k] = on;
    }
    const local = Array.isArray(v?.local) ? (v.local as unknown[]).filter((x): x is string => typeof x === "string" && /^[\w .-]{1,80}\.(cbplugin|zip)$/i.test(x)).slice(0, 50) : [];
    return { installed, local };
  } catch {
    return empty;
  }
}

export function loadInstalled(): InstalledState {
  return parseInstalled(readPref(INSTALLED_KEY));
}

export function saveInstalled(s: InstalledState): void {
  writePref(INSTALLED_KEY, JSON.stringify(s));
}

export function hasUpdate(p: Plugin): boolean {
  return p.latest !== p.version;
}

/** A local plugin file must be a .cbplugin or .zip with a plain name and a sane size. */
export function checkLocalFile(name: string, bytes: number): string | null {
  if (!/^[\w .-]{1,80}\.(cbplugin|zip)$/i.test(name)) return "Choose a .cbplugin or .zip file with a plain name.";
  if (!(bytes > 0) || bytes > 200 * 1024 * 1024) return "The file must be under 200 MB.";
  return null;
}
