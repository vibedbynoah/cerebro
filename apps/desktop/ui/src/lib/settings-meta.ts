import type { SettingKey } from "../service/types";
import type { GlossaryKey } from "./glossary";

export interface SettingMeta {
  key: SettingKey;
  group: "protections" | "network";
  label: string;
  /** One short, neutral sentence on what the switch does. No warnings, no nagging. */
  note: string;
  term?: GlossaryKey;
  /** For protections: what the mechanism does NOT do (docs/CLAIMS.md), shown on Security. */
  limits?: string;
}

export const SETTINGS_META: readonly SettingMeta[] = [
  {
    key: "processContainment",
    group: "protections",
    label: "Process containment",
    note: "Runs the model in a Windows Job Object, so it and anything it starts end when Cerebro stops.",
    term: "jobObject",
    limits: "Does not limit which files or network addresses the model process can open.",
  },
  {
    key: "cleanEnvironment",
    group: "protections",
    label: "Clean environment",
    note: "Gives the model process a short, fixed set of environment variables instead of a copy of Cerebro's.",
    term: "cleanEnv",
    limits: "Does not restrict what the process can read or reach once it runs.",
  },
  {
    key: "integrityCheck",
    group: "protections",
    label: "Backend integrity check",
    note: "Fingerprints the backend's program files before every start and stops if any changed.",
    term: "integrity",
  },
  {
    key: "permissionPrompts",
    group: "protections",
    label: "Permission prompts",
    note: "Asks before a tool or plugin takes an action. Off lets them act without asking.",
  },
  {
    key: "loopbackOnly",
    group: "protections",
    label: "API on loopback only",
    note: "Serves the API on 127.0.0.1. Off lets other devices on your network connect to it.",
    term: "loopback",
    limits: "Any program on this computer can still reach a loopback address.",
  },
  {
    key: "offlineMode",
    group: "network",
    label: "Offline mode",
    note: "Turns off every network feature. Everything else keeps working.",
  },
  {
    key: "allowModelDownloads",
    group: "network",
    label: "Model downloads",
    note: "Lets Model Hub download models when a network is available.",
  },
  {
    key: "allowPluginInstalls",
    group: "network",
    label: "Plugin installs",
    note: "Lets you install plugins when a network is available.",
  },
];
