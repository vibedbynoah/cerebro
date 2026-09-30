import type { IconName } from "../components/icons";

export type RouteId =
  | "home"
  | "hub"
  | "models"
  | "tune"
  | "chat"
  | "images"
  | "documents"
  | "plugins"
  | "tools"
  | "security"
  | "settings";

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
  { id: "hub", label: "Model Hub", icon: "hub", group: "Run", summary: "Find models that fit this machine and download them. The one optional network feature here.", keywords: "download browse catalog gguf" },
  { id: "models", label: "Models", icon: "models", group: "Run", summary: "The models on this machine and how each one was tuned.", keywords: "library local gguf list" },
  { id: "tune", label: "Tune", icon: "tune", group: "Run", summary: "Measure the fastest safe settings for a model on this machine.", keywords: "benchmark measure speed optimize flags" },
  { id: "chat", label: "Chat", icon: "chat", group: "Run", summary: "Talk to a local model. Nothing leaves this computer.", keywords: "conversation prompt assistant agent coding" },
  { id: "images", label: "Image Studio", icon: "images", group: "Create", summary: "Generate and edit images with local models.", keywords: "picture generate diffusion" },
  { id: "documents", label: "Documents", icon: "documents", group: "Create", summary: "Search your own files by meaning, on this machine.", keywords: "rag search files knowledge pdf" },
  { id: "plugins", label: "Plugins", icon: "plugins", group: "Extend", summary: "Add capabilities. Installing needs a network; installed plugins run offline.", keywords: "extensions addons install" },
  { id: "tools", label: "Tools", icon: "tools", group: "Extend", summary: "Actions a model may take, each behind your permission.", keywords: "functions actions agent" },
  { id: "security", label: "Security", icon: "security", group: "System", summary: "What protects the model process, and what each protection does not do.", keywords: "sandbox integrity protection containment" },
  { id: "settings", label: "Settings", icon: "settings", group: "System", summary: "Theme, protections and network features. You can change all of it.", keywords: "preferences options config theme" },
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

/** `#/models` -> "models". Anything unknown, malformed or empty is Home. */
export function parseHash(hash: string): RouteId {
  const match = /^#\/([a-z]+)\/?$/.exec(hash);
  const id = match?.[1];
  return id !== undefined && isRouteId(id) ? id : "home";
}

export function hrefFor(id: RouteId): string {
  return `#/${id}`;
}
