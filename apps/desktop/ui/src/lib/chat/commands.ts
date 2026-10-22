/**
 * Chat's / commands: typed at the start of a message, run here and never sent to the model. The
 * table is the single source for the composer's menu, the help text and the parser, so every
 * command is tested in one place (tests/unit/commands.test.ts). Chat.tsx carries them out.
 */

export interface CommandDef {
  name: string;
  /** Argument hint shown after the name, e.g. "<prompt>" or "1-5"; empty when it takes none. */
  args: string;
  description: string;
  /** Other names that run it. */
  aliases?: readonly string[];
}

export const COMMANDS: readonly CommandDef[] = [
  { name: "new", args: "", description: "Start a new chat." },
  { name: "clear", args: "", description: "Remove every message in this chat (undo for a few seconds)." },
  { name: "retry", args: "", description: "Write the last reply again.", aliases: ["regenerate"] },
  { name: "model", args: "<name>", description: "Switch to another model from your library." },
  { name: "speed", args: "1-5 or fastest | fast | balanced | accurate | maximum", description: "Move the Speed <-> Accuracy slider." },
  { name: "think", args: "on | off", description: "Let the model think before answering, or answer straight away." },
  { name: "system", args: "<text>", description: "Set the system prompt (empty to reset it)." },
  { name: "temp", args: "0-2", description: "Set the temperature: lower is steadier, higher is more varied.", aliases: ["temperature"] },
  { name: "max", args: "<tokens>", description: "Set the longest reply, in tokens." },
  { name: "image", args: "<prompt>", description: "Make an image with your image model and put it in the chat.", aliases: ["imagine"] },
  { name: "say", args: "", description: "Read the last reply aloud." },
  { name: "voice", args: "", description: "Talk hands-free: it listens, answers aloud, and listens again." },
  { name: "export", args: "md | json", description: "Save this chat as Markdown or JSON." },
  { name: "copy", args: "", description: "Copy the last reply." },
  { name: "plan", args: "", description: "Plan first: the model looks around, then shows a plan to approve." },
  { name: "orchestrate", args: "[workers] [rounds] or off", description: "Turn Orchestrate on (several fresh instances through a shared ledger) or off." },
  { name: "remember", args: "<fact>", description: "Remember a fact about you for later chats.", aliases: ["memory"] },
  { name: "memories", args: "", description: "Open what Cerebro remembers about you.", aliases: ["forget"] },
  { name: "stats", args: "", description: "This chat's speed and token totals." },
  { name: "prompts", args: "", description: "Open your saved prompts.", aliases: ["prompt"] },
  { name: "docs", args: "[off | collection | folder | tag]", description: "Answer from your documents, or search one collection, folder or tag." },
  { name: "summarise", args: "[text]", description: "Summarise text, or the last message." },
  { name: "translate", args: "[language]", description: "Translate the last message into another language." },
  { name: "rewrite", args: "[shorter | clearer | formal | friendly]", description: "Rewrite the last message." },
  { name: "extract", args: "<fields>", description: "Pull the fields you name out of the last message, as JSON." },
  { name: "explain", args: "[code]", description: "Explain a piece of code." },
  { name: "compare", args: "", description: "Compare the last two messages: what agrees, what changed." },
  { name: "notes", args: "[text]", description: "Meeting notes from a pasted transcript." },
  { name: "read", args: "[text]", description: "Read the text aloud with the speech model." },
  { name: "describe", args: "", description: "Describe the image attached to the last message." },
  { name: "help", args: "", description: "List every command." },
];

export interface ParsedCommand {
  def: CommandDef;
  arg: string;
}

const byName = (name: string): CommandDef | undefined => COMMANDS.find((c) => c.name === name || c.aliases?.includes(name));

/** "/name arg..." as a known command; null when the text is not a command at all. A "/" followed by
 *  an unknown word is reported by `unknownCommand`. A message starting "//" is sent as text. */
export function parseCommand(text: string): ParsedCommand | null {
  const m = /^\/([a-z][a-z-]*)(?:\s+([\s\S]*))?$/i.exec(text.trim());
  if (!m) return null;
  const def = byName(m[1]!.toLowerCase());
  return def ? { def, arg: (m[2] ?? "").trim() } : null;
}

/** The unknown command's name when `text` looks like a command nobody knows. */
export function unknownCommand(text: string): string | null {
  const m = /^\/([a-z][a-z-]*)\b/i.exec(text.trim());
  return m && !byName(m[1]!.toLowerCase()) ? m[1]!.toLowerCase() : null;
}

/** Commands for the menu while the user types "/pre": prefix matches first, then names containing
 *  it, then matching descriptions. */
export function suggest(typed: string): CommandDef[] {
  const q = typed.replace(/^\//, "").toLowerCase().split(/\s/)[0] ?? "";
  if (!q) return [...COMMANDS];
  const names = (c: CommandDef) => [c.name, ...(c.aliases ?? [])];
  const starts = COMMANDS.filter((c) => names(c).some((n) => n.startsWith(q)));
  const contains = COMMANDS.filter((c) => !starts.includes(c) && names(c).some((n) => n.includes(q)));
  const described = COMMANDS.filter((c) => !starts.includes(c) && !contains.includes(c) && c.description.toLowerCase().includes(q));
  return [...starts, ...contains, ...described];
}

/** The nearest command name by edit distance (for "did you mean"). */
export function closest(name: string): string {
  const dist = (a: string, b: string): number => {
    const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array<number>(b.length).fill(0)]);
    for (let j = 1; j <= b.length; j++) d[0]![j] = j;
    for (let i = 1; i <= a.length; i++)
      for (let j = 1; j <= b.length; j++) d[i]![j] = Math.min(d[i - 1]![j]! + 1, d[i]![j - 1]! + 1, d[i - 1]![j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1));
    return d[a.length]![b.length]!;
  };
  let best = COMMANDS[0]!.name;
  let bestD = Infinity;
  for (const c of COMMANDS)
    for (const n of [c.name, ...(c.aliases ?? [])]) {
      const dd = dist(name, n);
      if (dd < bestD) {
        bestD = dd;
        best = c.name;
      }
    }
  return best;
}

const SPEED_WORDS: Readonly<Record<string, 1 | 2 | 3 | 4 | 5>> = { fastest: 1, fast: 2, balanced: 3, accurate: 4, maximum: 5, max: 5 };

/** /speed's argument as a slider level, or null. */
export function speedArg(arg: string): 1 | 2 | 3 | 4 | 5 | null {
  const a = arg.trim().toLowerCase();
  if (/^[1-5]$/.test(a)) return Number(a) as 1 | 2 | 3 | 4 | 5;
  return SPEED_WORDS[a] ?? null;
}

/** A number argument within [min, max], or null. */
export function numberArg(arg: string, min: number, max: number, integer = false): number | null {
  const v = Number(arg.trim());
  if (!arg.trim() || !Number.isFinite(v) || v < min || v > max || (integer && !Number.isInteger(v))) return null;
  return v;
}

/** on/off as a boolean, or null. */
export function switchArg(arg: string): boolean | null {
  const a = arg.trim().toLowerCase();
  if (a === "on" || a === "yes" || a === "true") return true;
  if (a === "off" || a === "no" || a === "false") return false;
  return null;
}

/** /orchestrate's arguments: off, or [workers 1-6] [rounds 1-8] (defaults kept when omitted). */
export function orchestrateArg(arg: string): { on: false } | { on: true; workers?: number; rounds?: number } | null {
  const a = arg.trim().toLowerCase();
  if (a === "off") return { on: false };
  if (!a || a === "on") return { on: true };
  const [w, r, ...rest] = a.split(/\s+/);
  if (rest.length) return null;
  const workers = numberArg(w ?? "", 1, 6, true);
  const rounds = r === undefined ? undefined : numberArg(r, 1, 8, true);
  if (workers === null || rounds === null) return null;
  return { on: true, workers, ...(rounds !== undefined ? { rounds } : {}) };
}

/** The help text: one line per command. */
export function helpText(): string {
  return COMMANDS.map((c) => `/${c.name}${c.args ? ` ${c.args}` : ""}: ${c.description}`).join("\n");
}
