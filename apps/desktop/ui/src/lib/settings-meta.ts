import type { SettingKey } from "../service/types";
import type { GlossaryKey } from "./glossary";

export interface SettingMeta {
  key: SettingKey;
  group: "protections" | "network";
  label: string;
  /** One short, neutral sentence on what the switch does. No warnings, no nagging. */
  note: string;
  term?: GlossaryKey;
  /** For protections: what the mechanism does NOT do (docs/CLAIMS.md), shown under the switch. */
  limits?: string;
  /** False when it guards something the desktop build does not have (a separate model process, a
   *  local API, plugins): there the switch would change nothing, so it is not shown. */
  desktop?: boolean;
}

export const SETTINGS_META: readonly SettingMeta[] = [
  {
    key: "processContainment",
    group: "protections",
    label: "Process containment",
    note: "Runs the model in a Windows Job Object, so it and anything it starts end when Cerebro stops.",
    term: "jobObject",
    limits: "Does not limit which files or network addresses the model process can open.",
    desktop: false,
  },
  {
    key: "cleanEnvironment",
    group: "protections",
    label: "Clean environment",
    note: "Gives the model process a short, fixed set of environment variables instead of a copy of Cerebro's.",
    term: "cleanEnv",
    limits: "Does not restrict what the process can read or reach once it runs.",
    desktop: false,
  },
  {
    key: "integrityCheck",
    group: "protections",
    label: "Backend integrity check",
    note: "Fingerprints the backend's program files before every start and stops if any changed.",
    term: "integrity",
    desktop: false,
  },
  {
    key: "permissionPrompts",
    group: "protections",
    label: "Permission prompts",
    note: "Asks before a tool or plugin takes an action. Off lets them act without asking.",
  },
  {
    key: "blockLocalFetch",
    group: "protections",
    label: "Keep web fetches off this network",
    note: "A model fetching a web page may not reach this computer, the router or other devices on the network.",
    limits: "Does not filter what public websites a model can read.",
  },
  {
    key: "hideSecretsFromTools",
    group: "protections",
    label: "Hide secrets from tools",
    note: "Commands a model runs do not see environment variables that look like keys, tokens or passwords.",
    limits: "Files on disk that hold secrets are covered by the tool access level, not by this.",
  },
  {
    key: "readBeforeWrite",
    group: "protections",
    label: "Read before write",
    note: "A model must read a file before changing it, and again if it changed, so it never overwrites unseen work.",
    limits: "New files can be created without reading.",
  },
  {
    key: "keepChatCaches",
    group: "protections",
    label: "Keep long chats ready",
    note: "Saves a long chat's reading of its history to disk, so returning to it continues instantly.",
    limits: "Uses up to 8 GB in the app data folder; the oldest are removed first.",
  },
  {
    key: "loopbackOnly",
    group: "protections",
    label: "API on loopback only",
    note: "Serves the API on 127.0.0.1. Off lets other devices on the network connect to it.",
    term: "loopback",
    limits: "Any program on this computer can still reach a loopback address.",
    desktop: false,
  },
  {
    key: "offlineMode",
    group: "network",
    label: "Offline mode",
    note: "Stops Model Hub, web fetches and plugin installs. Trainer, MCP servers, the API and phone access keep working.",
  },
  {
    key: "allowModelDownloads",
    group: "network",
    label: "Model downloads",
    note: "Lets Model Hub search for and download models when a network is available.",
  },
  {
    key: "allowPluginInstalls",
    group: "network",
    label: "Plugin installs",
    note: "Allows plugin installs when a network is available.",
    desktop: false,
  },
];
