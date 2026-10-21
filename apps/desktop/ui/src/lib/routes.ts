import type { IconName } from "../components/icons";

export type RouteId =
  | "home"
  | "hub"
  | "models"
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
}

export const ROUTES: readonly RouteDef[] = [
  { id: "home", label: "Home", icon: "home", group: "Run", summary: "What is loaded, how fast it runs and what is protecting it.", keywords: "overview dashboard start" },
  { id: "hub", label: "Model Hub", icon: "hub", group: "Run", summary: "Find models that fit this machine and download them. Searching the catalogue uses the network; everything else works offline.", keywords: "download browse catalog gguf" },
  { id: "chat", label: "Chat", icon: "chat", group: "Run", summary: "Talk to a local model. Nothing leaves this computer.", keywords: "conversation prompt assistant agent coding" },
  { id: "models", label: "Models", icon: "models", group: "Run", summary: "The models on this machine and how each one was tuned.", keywords: "library local gguf list" },
  { id: "tune", label: "Tune", icon: "tune", group: "Run", summary: "Measure the fastest safe settings for a model on this machine.", keywords: "benchmark measure speed optimize flags" },
  { id: "monitor", label: "Monitor", icon: "activity", group: "Run", summary: "The model and the machine, live: speed, memory, heat, and every request.", keywords: "metrics telemetry live gpu vram temperature requests inspector dashboard stats" },
  {
    id: "turbo",
    label: "Turbo",
    icon: "bolt",
    group: "Run",
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
      ? "Make images on this computer, with the seed kept so you can make one again."
      : "Generate and edit images with local models.",
    keywords: "picture generate diffusion",
  },
  { id: "documents", label: "Documents", icon: "documents", group: "Create", summary: "Search your own files on this machine; matching passages come back with their file.", keywords: "rag search files knowledge pdf" },
  { id: "train", label: "Train", icon: "chip", group: "Create", summary: "Fine-tune a model on your own examples and add it to your library.", keywords: "fine-tune finetune lora qlora dataset training adapter" },
  { id: "tools", label: "Tools", icon: "tools", group: "Extend", summary: "Actions a model may take, each behind your permission.", keywords: "functions actions agent" },
  {
    id: "plugins",
    label: "Plugins",
    icon: "plugins",
    group: "Extend",
    summary: __DESKTOP__
      ? "Planned add-ons. Today, MCP servers on Tools give models new abilities."
      : "Add capabilities. Installing needs a network; installed plugins run offline.",
    keywords: "extensions addons install",
  },
  { id: "settings", label: "Settings", icon: "settings", group: "System", summary: "Theme, protections and network features. You can change all of it.", keywords: "preferences options config theme" },
  { id: "about", label: "About", icon: "about", group: "System", summary: "What Cerebro does for you, the version on this machine and the documents that come with it.", keywords: "version licence license credits mark logo info help" },
];

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

/** The model a link names (`?model=`), when it is a well-formed id; callers still check that the
 *  library has it before using it. */
export function modelFromHash(hash: string): string | null {
  const q = hash.indexOf("?");
  if (q < 0) return null;
  const id = new URLSearchParams(hash.slice(q + 1)).get("model");
  return id !== null && MODEL_ID.test(id) ? id : null;
}
