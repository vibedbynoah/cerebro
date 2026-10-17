import { loadLevelEdits, loadPrefs, parseLevelEdits, parsePrefs, saveLevelEdits, savePrefs, type ChatPrefs, type LevelEdits } from "./chat/prompt";
import { parseChannel, parseLevel, type Channel, type ExperienceLevel } from "./experience";
import { loadPolicy, parsePolicy, savePolicy, type Policy } from "./permissions";
import type { Settings } from "../service/types";

/**
 * The whole configuration as one editable JSON file. Import re-checks every field with the same
 * parsers the app uses on startup; anything unknown, mistyped or out of range is ignored and
 * listed, never applied. Conversations, memories and keys are not part of it.
 */

export const CONFIG_VERSION = 1;

export interface ConfigFile {
  cerebroConfig: number;
  settings: Partial<Settings>;
  level: ExperienceLevel;
  channel: Channel;
  chat: ChatPrefs;
  effortLevels: LevelEdits;
  toolPolicy: Policy;
}

export function exportConfig(settings: Settings | null, level: ExperienceLevel, channel: Channel): string {
  const file: ConfigFile = {
    cerebroConfig: CONFIG_VERSION,
    settings: settings ?? {},
    level,
    channel,
    chat: loadPrefs(),
    effortLevels: loadLevelEdits(),
    toolPolicy: loadPolicy(),
  };
  return `${JSON.stringify(file, null, 2)}\n`;
}

export interface ImportResult {
  settings: Partial<Settings>;
  level: ExperienceLevel | null;
  channel: Channel | null;
  ignored: string[];
}

const SETTING_KEYS: ReadonlyArray<keyof Settings> = [
  "processContainment",
  "cleanEnvironment",
  "integrityCheck",
  "permissionPrompts",
  "blockLocalFetch",
  "hideSecretsFromTools",
  "readBeforeWrite",
  "loopbackOnly",
  "offlineMode",
  "allowModelDownloads",
  "allowPluginInstalls",
];

/** Parses and applies the parts stored in this webview; returns settings, level and channel for
 *  the caller to apply through the app state (which validates them again). Throws on non-JSON. */
export function importConfig(text: string): ImportResult {
  const v = JSON.parse(text) as Record<string, unknown>;
  if (!v || typeof v !== "object" || Array.isArray(v)) throw new TypeError("Not a Cerebro configuration file");
  const ignored: string[] = [];
  for (const k of Object.keys(v)) if (!["cerebroConfig", "settings", "level", "channel", "chat", "effortLevels", "toolPolicy"].includes(k)) ignored.push(k);

  const settings: Partial<Settings> = {};
  if (v.settings && typeof v.settings === "object") {
    for (const [k, val] of Object.entries(v.settings as Record<string, unknown>)) {
      if ((SETTING_KEYS as readonly string[]).includes(k) && typeof val === "boolean") settings[k as keyof Settings] = val;
      else ignored.push(`settings.${k}`);
    }
  }
  if (v.chat !== undefined) savePrefs(parsePrefs(JSON.stringify(v.chat)));
  if (v.effortLevels !== undefined) saveLevelEdits(parseLevelEdits(JSON.stringify(v.effortLevels)));
  if (v.toolPolicy !== undefined) savePolicy(parsePolicy(JSON.stringify(v.toolPolicy)));
  const level = typeof v.level === "string" ? parseLevel(v.level) : null;
  const channel = typeof v.channel === "string" ? parseChannel(v.channel) : null;
  if (v.level !== undefined && !level) ignored.push("level");
  if (v.channel !== undefined && !channel) ignored.push("channel");
  return { settings, level, channel, ignored };
}
