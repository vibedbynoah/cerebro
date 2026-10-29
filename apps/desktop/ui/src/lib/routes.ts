import type { IconName } from "../components/icons";

export type RouteId =
  | "home"
  | "hub"
  | "models"
  | "cluster"
  | "tune"
  | "monitor"
  | "turbo"
  | "chat"
  | "images"
  | "documents"
  | "train"
  | "plugins"
  | "tools"
  | "settings"
  | "about";

export type RouteGroup = "Run" | "Create" | "Extend" | "System";

export interface RouteDef {
  id: RouteId;
  label: string;
  icon: IconName;
  group: RouteGroup;
  /** One plain sentence: what the screen is for. Also the command palette hint. */
  summary: string;
  /** Extra words the command palette matches on. */
  keywords: string;
  /** False: reachable from other screens and the command palette, but not listed in the sidebar. */
  inSidebar?: false;
}

const ALL_ROUTES: readonly RouteDef[] = [
  { id: "home", label: "Home", icon: "home", group: "Run", summary: "What is loaded, how fast it runs and what is protecting it.", keywords: "overview dashboard start" },
  { id: "hub", label: "Model Hub", icon: "hub", group: "Run", summary: "Find models that fit this machine and download them. Searching the catalogue uses the network; everything else works offline.", keywords: "download browse catalog gguf" },
  { id: "chat", label: "Chat", icon: "chat", group: "Run", summary: "Talk to a local model. It runs on this computer.", keywords: "conversation prompt assistant agent coding" },
  { id: "models", label: "Models", icon: "models", group: "Run", summary: "The models on this machine and how each one was tuned.", keywords: "library local gguf list" },
  { id: "cluster", label: "Cluster", icon: "link", group: "Extend", summary: "One model across several computers over any network link, each running part of it.", keywords: "multiple computers network lan wifi ethernet usb distributed split homelab worker pair" },
  { id: "tune", label: "Speed tune", icon: "tune", group: "Extend", summary: "Measure the fastest settings for a model on this machine. Opened from a model's page; training is on Train.", keywords: "benchmark measure speed optimize flags tune", inSidebar: false },
  { id: "monitor", label: "Monitor", icon: "activity", group: "Extend", summary: "The model and the machine, live: speed, memory, heat, and every request.", keywords: "metrics telemetry live gpu vram temperature requests inspector dashboard stats" },
  {
    id: "turbo",
    label: "Turbo",
    icon: "bolt",
    group: "Extend",
    summary: __DESKTOP__
      ? "Engine speed-ups for this machine, each with a plain explanation, on or off."
      : "Builds tuned to this machine, scheduling and power profiles, each measured before it is kept.",
    keywords: "performance kernel build profile power scheduler profiler overclock fan speed cachy",
  },
  {
    id: "images",
    label: "Image Studio",
    icon: "images",
    group: "Create",
    summary: __DESKTOP__
      ? "Images made on this computer, each with its seed kept so it can be made again."
      : "Generate and edit images with local models.",
    keywords: "picture generate diffusion",
  },
  { id: "documents", label: "Documents", icon: "documents", group: "Create", summary: "Local files: search, collections, conversion and quick tools.", keywords: "rag search files knowledge pdf collections convert quick tools" },
  { id: "train", label: "Train", icon: "chip", group: "Extend", summary: "Fine-tunes a model on custom examples and adds it to the library.", keywords: "fine-tune finetune lora qlora dataset training adapter" },
  { id: "tools", label: "Tools", icon: "tools", group: "Create", summary: "Actions a model may take, each behind a permission rule.", keywords: "functions actions agent mcp servers", inSidebar: false },
  {
    id: "plugins",
    label: "Plugins",
    icon: "plugins",
    group: "Create",
    summary: __DESKTOP__
      ? "Add-ons that give models new abilities (files, git, web, databases), as MCP servers. Adding one needs the network; most then run offline."
      : "Add capabilities. Installing needs a network; installed plugins run offline.",
    keywords: "extensions addons install",
  },
  { id: "settings", label: "Settings", icon: "settings", group: "System", summary: "Theme, protections and network features.", keywords: "preferences options config theme" },
  { id: "about", label: "About", icon: "about", group: "System", summary: "Features, version and the documents that ship with Cerebro.", keywords: "version licence license credits mark logo info help" },
];

/** Sidebar and command palette order: the everyday screens first, then making things, then the
 *  measuring and multi-machine screens, then the app itself. */
const ORDER: readonly RouteId[] = ["home", "chat", "models", "hub", "images", "documents", "tools", "plugins", "train", "monitor", "cluster", "turbo", "tune", "settings", "about"];
const ORDERED: readonly RouteDef[] = [...ALL_ROUTES].sort((a, b) => ORDER.indexOf(a.id) - ORDER.indexOf(b.id));

/** The desktop build has no separate Plugins screen: plugins there are MCP servers, which Tools
 *  manages, so the two pages were the same. Its old links open Tools. */
export const ROUTES: readonly RouteDef[] = __DESKTOP__ ? ORDERED.filter((r) => r.id !== "plugins") : ORDERED;

const IDS: ReadonlySet<string> = new Set(ROUTES.map((r) => r.id));

export function isRouteId(value: string): value is RouteId {
  return IDS.has(value);
}

export function routeById(id: RouteId): RouteDef {
  const found = ROUTES.find((r) => r.id === id);
  if (!found) throw new Error(`Unknown route ${id}`);
  return found;
}

/** `#/models` -> "models"; a query after the route (`#/tune?model=x`) is allowed and ignored here.
 *  Anything unknown, malformed or empty is Home. */
export function parseHash(hash: string): RouteId {
  const match = /^#\/([a-z]+)\/?(?:\?[^#?]*)?$/.exec(hash);
  const id = match?.[1];
  if (__DESKTOP__ && id === "plugins") return "tools";
  return id !== undefined && isRouteId(id) ? id : "home";
}

export function hrefFor(id: RouteId): string {
  return `#/${id}`;
}

/** Library ids are lower-case slugs; anything else in a link is ignored. */
const MODEL_ID = /^[a-z0-9][a-z0-9._-]{0,127}$/;

/** `#/tune?model=<id>`: a screen opened for one model. */
export function hrefWithModel(id: RouteId, modelId: string): string {
  return `${hrefFor(id)}?model=${encodeURIComponent(modelId)}`;
}

/** The Monitor screen's two views: the live readings and the Benchmark lab. */
export type MonitorTab = "live" | "benchmark";

export function isMonitorTab(value: unknown): value is MonitorTab {
  return value === "live" || value === "benchmark";
}

/** `#/monitor` for the live view, `#/monitor?tab=benchmark` for the Benchmark lab. */
export function hrefForMonitorTab(tab: MonitorTab): string {
  return tab === "live" ? hrefFor("monitor") : `${hrefFor("monitor")}?tab=benchmark`;
}

/** The Monitor view a link asks for ("?tab="). The live view when it names none. */
export function monitorTabFromHash(hash: string): MonitorTab {
  const q = hash.indexOf("?");
  if (q < 0) return "live";
  const tab = new URLSearchParams(hash.slice(q + 1)).get("tab");
  return isMonitorTab(tab) ? tab : "live";
}

/** The model a link names (`?model=`), when it is a well-formed id; callers still check that the
 *  library has it before using it. */
export function modelFromHash(hash: string): string | null {
  const q = hash.indexOf("?");
  if (q < 0) return null;
  const id = new URLSearchParams(hash.slice(q + 1)).get("model");
  return id !== null && MODEL_ID.test(id) ? id : null;
}
