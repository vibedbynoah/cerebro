/**
 * Chat's local commands. A message that starts with a slash and names one of these runs here, on
 * this computer, and is never sent to the model. A slash word nobody here knows is left alone, so
 * the model's own commands (such as /imagine) still reach it.
 */

export interface CommandDef {
  /** The word after the slash, in lower case. */
  name: string;
  /** One plain sentence: what the command does. */
  description: string;
}

export const COMMANDS: readonly CommandDef[] = [
  { name: "bench", description: "Open the Benchmark lab and measure a model on this computer." },
  { name: "arena", description: "Open the Arena and compare answers from two models." },
];

export interface ParsedCommand {
  /** The command's name, in lower case. */
  name: string;
  /** Everything after the name, trimmed. Empty when the command takes no argument. */
  arg: string;
}

const byName = (name: string): CommandDef | undefined => COMMANDS.find((c) => c.name === name);

/** "/bench" or "/bench now" as a known command; null when the text is not one of ours. */
export function parseCommand(text: string): ParsedCommand | null {
  const m = /^\/([a-z][a-z-]*)(?:\s+([\s\S]*))?$/i.exec(text.trim());
  if (!m) return null;
  const name = m[1]!.toLowerCase();
  if (!byName(name)) return null;
  return { name, arg: (m[2] ?? "").trim() };
}

/** The command a partly typed word names, for the composer's hint. Null when it names none. */
export function commandFor(text: string): CommandDef | null {
  const m = /^\/([a-z][a-z-]*)\b/i.exec(text.trim());
  if (!m) return null;
  return byName(m[1]!.toLowerCase()) ?? null;
}
