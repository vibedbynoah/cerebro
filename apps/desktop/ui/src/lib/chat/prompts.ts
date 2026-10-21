import { readPref, writePref } from "../storage";
import { newId } from "./store";

/**
 * The prompt library: reusable prompts kept on this machine. A prompt may contain {{variables}};
 * the library asks for each one in a small form before putting the filled text in the composer.
 * Imported JSON is checked field by field, exactly like stored chats.
 */

export interface SavedPrompt {
  id: string;
  title: string;
  text: string;
  category: string;
}

export const PROMPTS_KEY = "cerebro-prompts";
export const PROMPT_CATEGORIES = ["Writing", "Coding", "Study", "Work", "Everyday"] as const;
const MAX_PROMPTS = 200;
const MAX_TITLE = 120;
const MAX_TEXT = 20_000;
const MAX_CATEGORY = 40;

const starter = (id: string, title: string, category: string, text: string): SavedPrompt => ({ id: `p-${id}`, title, category, text });

export const STARTER_PROMPTS: readonly SavedPrompt[] = [
  starter(
    "summarise",
    "Summarise a long text",
    "Writing",
    "Summarise the text below in {{length}}. Keep the main points, the numbers that matter and anything the reader must act on. Do not add facts that are not in the text.\n\nText:\n{{text}}",
  ),
  starter(
    "explain-code",
    "Explain code",
    "Coding",
    "Explain what this {{language}} code does, step by step. Say what it reads and changes, what it returns, and anything surprising or risky. Then give one small example of input and output.\n\nCode:\n{{code}}",
  ),
  starter(
    "translate",
    "Translate text",
    "Writing",
    "Translate the text below into {{language}}. Keep the meaning, tone and formatting. If a word has no exact match, keep it and explain it briefly in brackets.\n\nText:\n{{text}}",
  ),
  starter(
    "email-reply",
    "Write an email reply",
    "Work",
    "Write a reply to the email below. Tone: {{tone}}. Answer every question, say clearly what happens next, and keep it short enough to read once.\n\nEmail:\n{{email}}",
  ),
  starter(
    "study-notes",
    "Make study notes",
    "Study",
    "Turn this material into study notes about {{topic}}. Use short headings, bullet points and one worked example. End with five questions I can use to test myself.\n\nMaterial:\n{{material}}",
  ),
  starter(
    "explain-simply",
    "Explain it simply",
    "Study",
    "Explain {{topic}} to a {{level}}. Use plain words, one everyday analogy and a short example. Then say which part people most often get wrong.",
  ),
  starter(
    "find-bugs",
    "Find bugs in code",
    "Coding",
    "Review this code for bugs. List each problem with the line or function, why it is wrong, and the smallest fix. Check edge cases, empty input and errors. Do not rewrite the whole file unless asked.\n\nCode:\n{{code}}",
  ),
  starter(
    "write-tests",
    "Write tests",
    "Coding",
    "Write tests for this {{language}} code. Cover the normal path, the boundaries and one failure case. Use the test framework already in the file if there is one. Explain what each test proves.\n\nCode:\n{{code}}",
  ),
  starter(
    "meeting-notes",
    "Turn notes into decisions",
    "Work",
    "Turn these meeting notes into a short record: decisions made, who owns each next step, and the date it is due. Put anything unresolved under Open questions.\n\nNotes:\n{{notes}}",
  ),
  starter(
    "rewrite-clearly",
    "Rewrite clearly",
    "Writing",
    "Rewrite the text below so it is clear and direct. Keep the meaning and the facts. Use short sentences, cut filler, and keep the same tone: {{tone}}.\n\nText:\n{{text}}",
  ),
  starter(
    "plan-task",
    "Plan a task",
    "Work",
    "Make a step-by-step plan for {{goal}}. Put the steps in the order they must happen, name what could block each one, and say how I will know it is finished. Keep it to the smallest useful number of steps.",
  ),
  starter(
    "compare-options",
    "Compare two options",
    "Everyday",
    "Compare {{option_a}} and {{option_b}} for {{purpose}}. Use a short table with the same criteria for both, then give a recommendation and the one reason that matters most.",
  ),
  starter(
    "brainstorm",
    "Brainstorm ideas",
    "Everyday",
    "Give me {{count}} different ideas for {{topic}}. Make them concrete, not variations of one idea. After the list, mark the two you would try first and say why.",
  ),
  starter(
    "practice-questions",
    "Make practice questions",
    "Study",
    "Write {{count}} practice questions about {{topic}}, from easy to hard. Put the answers and a one-line explanation after all the questions, so I can try them first.",
  ),
  starter(
    "recipe",
    "Cook with what I have",
    "Everyday",
    "Suggest a simple recipe using these ingredients: {{ingredients}}. Say how long it takes, the steps in order, and one substitution for anything that might be missing. Keep it to ingredients I listed plus basics.",
  ),
];

const str = (v: unknown, max: number): string | null => (typeof v === "string" && v.length <= max ? v : null);

function parsePrompt(v: unknown): SavedPrompt | null {
  if (!v || typeof v !== "object") return null;
  const x = v as Record<string, unknown>;
  const id = str(x.id, 64);
  const title = str(x.title, MAX_TITLE)?.trim();
  const text = str(x.text, MAX_TEXT)?.trim();
  const category = str(x.category, MAX_CATEGORY)?.trim();
  if (!id || !title || !text) return null;
  return { id, title: title.slice(0, MAX_TITLE), text, category: (category || "Other").slice(0, MAX_CATEGORY) };
}

/** Reads a list of prompts (also the object form this app exports). Null means malformed JSON. */
export function parsePromptJson(raw: string): SavedPrompt[] | null {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  const list = Array.isArray(value)
    ? value
    : value && typeof value === "object" && Array.isArray((value as Record<string, unknown>).prompts)
      ? ((value as Record<string, unknown>).prompts as unknown[])
      : null;
  if (!list) return null;
  const seen = new Set<string>();
  const out: SavedPrompt[] = [];
  for (const item of list.slice(0, MAX_PROMPTS)) {
    const prompt = parsePrompt(item);
    if (!prompt) continue;
    if (seen.has(prompt.id)) prompt.id = newId();
    seen.add(prompt.id);
    out.push(prompt);
  }
  return out;
}

export function loadPrompts(): SavedPrompt[] {
  const raw = readPref(PROMPTS_KEY);
  if (raw === null) return STARTER_PROMPTS.map((p) => ({ ...p }));
  return parsePromptJson(raw) ?? [];
}

export function savePrompts(list: readonly SavedPrompt[]): void {
  writePref(PROMPTS_KEY, JSON.stringify(list.slice(0, MAX_PROMPTS)));
}

export function exportPrompts(list: readonly SavedPrompt[]): string {
  return JSON.stringify({ version: 1, prompts: list }, null, 2);
}

/** Imported prompts replace ones with the same id and are added otherwise. */
export function mergePrompts(current: readonly SavedPrompt[], incoming: readonly SavedPrompt[]): SavedPrompt[] {
  const byId = new Map(current.map((p) => [p.id, p]));
  for (const prompt of incoming) byId.set(prompt.id, prompt);
  return [...byId.values()].slice(0, MAX_PROMPTS);
}

/** Every distinct {{variable}} in a prompt, in the order it first appears. */
export function promptVariables(text: string): string[] {
  const out: string[] = [];
  for (const match of text.matchAll(/\{\{\s*([^{}]{1,40}?)\s*\}\}/g)) {
    const name = match[1]?.trim();
    if (name && !out.includes(name)) out.push(name);
  }
  return out;
}

export function fillPromptTemplate(text: string, values: Readonly<Record<string, string>>): string {
  return text.replace(/\{\{\s*([^{}]{1,40}?)\s*\}\}/g, (whole, raw: string) => {
    const name = raw.trim();
    const value = values[name];
    return value === undefined ? whole : value;
  });
}

export function promptCategories(list: readonly SavedPrompt[]): string[] {
  const out: string[] = [...PROMPT_CATEGORIES];
  for (const prompt of list) if (prompt.category && !out.includes(prompt.category)) out.push(prompt.category);
  return out;
}
