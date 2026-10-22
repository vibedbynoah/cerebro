/**
 * Quick tools: one-step AI jobs that run on the model the user has loaded (or on a cloud model
 * they picked). The instructions live in this file so they are easy to tune, and every place that
 * offers a tool uses the same words: the cards in Documents > Quick tools, and the / commands in
 * Chat.
 *
 * A tool never changes the user's files and never calls anything else: it sends the text the user
 * gave it, and shows what came back.
 */

export type QuickToolId = "summarise" | "translate" | "rewrite" | "extract" | "explain" | "compare" | "notes" | "read" | "describe";

/** What the tool needs from the user. */
export type QuickToolInputKind = "text" | "text-and-language" | "text-and-style" | "text-and-fields" | "two-texts" | "audio" | "image";

/** The kind of model the tool needs. */
export type QuickToolModel = "text" | "vision" | "speech-in" | "speech-out";

export interface QuickToolDef {
  id: QuickToolId;
  /** The name on the card. */
  name: string;
  /** The / command that runs it in Chat. */
  command: string;
  /** One line: what the tool does. */
  blurb: string;
  /** What the user gives it. */
  input: QuickToolInputKind;
  /** The label of the input. */
  inputLabel: string;
  /** The kind of model it needs. */
  needs: QuickToolModel;
  /** True when the tool needs no model at all (the speech model speaks the text). */
  local?: boolean;
}

export const QUICK_TOOLS: readonly QuickToolDef[] = [
  {
    id: "summarise",
    name: "Summarise",
    command: "summarise",
    blurb: "The main points of a text, a file or this chat.",
    input: "text",
    inputLabel: "Text to summarise",
    needs: "text",
  },
  {
    id: "translate",
    name: "Translate",
    command: "translate",
    blurb: "Text in another language.",
    input: "text-and-language",
    inputLabel: "Text to translate",
    needs: "text",
  },
  {
    id: "rewrite",
    name: "Rewrite",
    command: "rewrite",
    blurb: "The same text, shorter, clearer, formal or friendly.",
    input: "text-and-style",
    inputLabel: "Text to rewrite",
    needs: "text",
  },
  {
    id: "extract",
    name: "Extract",
    command: "extract",
    blurb: "The fields you name, as a table or as JSON.",
    input: "text-and-fields",
    inputLabel: "Text to read",
    needs: "text",
  },
  {
    id: "explain",
    name: "Explain code",
    command: "explain",
    blurb: "What a piece of code does, in plain words.",
    input: "text",
    inputLabel: "Code to explain",
    needs: "text",
  },
  {
    id: "compare",
    name: "Compare two documents",
    command: "compare",
    blurb: "What two texts agree on, and what changed.",
    input: "two-texts",
    inputLabel: "First text",
    needs: "text",
  },
  {
    id: "notes",
    name: "Meeting notes",
    command: "notes",
    blurb: "An audio file: a summary, the decisions and the actions.",
    input: "audio",
    inputLabel: "Audio file",
    needs: "speech-in",
  },
  {
    id: "read",
    name: "Read aloud",
    command: "read",
    blurb: "The speech model reads the text out loud.",
    input: "text",
    inputLabel: "Text to read",
    needs: "speech-out",
  },
  {
    id: "describe",
    name: "Describe an image",
    command: "describe",
    blurb: "What is in a picture, in words.",
    input: "image",
    inputLabel: "Image",
    needs: "vision",
  },
];

export function quickTool(id: string): QuickToolDef | undefined {
  return QUICK_TOOLS.find((t) => t.id === id);
}

export interface RewriteStyle {
  id: string;
  label: string;
  /** What the style means, in one line. */
  note: string;
}

export const REWRITE_STYLES: readonly RewriteStyle[] = [
  { id: "shorter", label: "Shorter", note: "keep every fact, cut the words" },
  { id: "clearer", label: "Clearer", note: "plain words, one idea a sentence" },
  { id: "formal", label: "Formal", note: "for a customer, a manager or a record" },
  { id: "friendly", label: "Friendly", note: "warm, but still short" },
];

/** The languages Translate offers. Any language name works, but these are one click away. */
export const TRANSLATE_LANGUAGES: readonly string[] = [
  "English",
  "Spanish",
  "French",
  "German",
  "Italian",
  "Portuguese",
  "Dutch",
  "Swedish",
  "Polish",
  "Russian",
  "Ukrainian",
  "Turkish",
  "Arabic",
  "Hebrew",
  "Hindi",
  "Chinese (Simplified)",
  "Japanese",
  "Korean",
  "Indonesian",
  "Vietnamese",
  "Thai",
];

/** The marker every instruction starts with, so the app (and the preview) can tell the tools apart. */
export const QUICK_TOOL_MARKER = "Cerebro quick tool:";

export interface QuickToolInputValue {
  /** The text the tool works on. */
  text?: string;
  /** The second text, for Compare. */
  second?: string;
  /** The language to translate into. */
  language?: string;
  /** The style to rewrite in. */
  style?: string;
  /** The fields to extract, separated by commas. */
  fields?: string;
  /** The picture to describe, for Describe an image. */
  attachment?: { name: string; type: string; bytes: number; data: string };
}

export interface QuickToolPrompt {
  /** The instruction, sent as the system prompt. */
  system: string;
  /** The user's text, sent as the one turn. */
  user: string;
}

/** The field names as a list. */
export function fieldList(fields: string | undefined): string[] {
  return [...new Set((fields ?? "").split(",").map((f) => f.trim()).filter(Boolean))].slice(0, 20);
}

/**
 * The instruction and the turn for one tool. Everything the model is told about the job is here,
 * so a tool can be tuned by editing this file alone.
 */
export function buildQuickToolPrompt(id: QuickToolId, input: QuickToolInputValue): QuickToolPrompt {
  const text = (input.text ?? "").trim();
  switch (id) {
    case "summarise":
      return {
        system: [
          `${QUICK_TOOL_MARKER} summarise.`,
          "Write a summary of the text the user gives you.",
          "Keep the main points, the numbers and the names. Use short sentences.",
          "Use a list only when the text is a list of separate items.",
          "Answer in the language of the text.",
        ].join("\n"),
        user: text,
      };
    case "translate":
      return {
        system: [
          `${QUICK_TOOL_MARKER} translate.`,
          `Translate the text the user gives you into ${(input.language ?? "English").trim() || "English"}.`,
          "Keep the formatting, the numbers and the names as they are.",
          "Answer with the translation only, nothing else.",
        ].join("\n"),
        user: text,
      };
    case "rewrite":
      return {
        system: [
          `${QUICK_TOOL_MARKER} rewrite.`,
          `Rewrite the text the user gives you so it is ${(input.style ?? "clearer").trim() || "clearer"}.`,
          "Keep every fact, number and name. Answer with the rewritten text only.",
        ].join("\n"),
        user: text,
      };
    case "extract":
      return {
        system: [
          `${QUICK_TOOL_MARKER} extract.`,
          "Pull the information the user asks for out of the text.",
          "Answer with JSON only: an array of objects, one for each item in the text, with the field names the user gives.",
          "Use an empty string when a field is not in the text.",
          "Write no text before or after the JSON.",
        ].join("\n"),
        user: `Fields: ${fieldList(input.fields).join(", ")}\n\n${text}`,
      };
    case "explain":
      return {
        system: [
          `${QUICK_TOOL_MARKER} explain.`,
          "Explain the code the user gives you: what it does, what each part is for, and anything that would surprise a reader.",
          "Be plain and short. Answer with Markdown.",
        ].join("\n"),
        user: text,
      };
    case "compare":
      return {
        system: [
          `${QUICK_TOOL_MARKER} compare.`,
          "Compare the two texts the user gives you.",
          "Say what they agree on and what changed, as two short lists.",
          "Quote the wording that matters.",
        ].join("\n"),
        user: `First text:\n\n${text}\n\nSecond text:\n\n${(input.second ?? "").trim()}`,
      };
    case "notes":
      return {
        system: [
          `${QUICK_TOOL_MARKER} notes.`,
          "The user gives you the transcript of a meeting.",
          "Write a summary in three sentences, then the decisions that were made, then the actions with who does what.",
          "Leave out anything that was not decided. Answer with Markdown.",
        ].join("\n"),
        user: text,
      };
    case "describe":
      return {
        system: [
          `${QUICK_TOOL_MARKER} describe.`,
          "Describe the image the user gives you in one short paragraph, then list what is in it.",
          "Do not guess about things you cannot see.",
        ].join("\n"),
        user: "",
      };
    case "read":
      // The speech model reads the text; there is nothing to ask a text model.
      return { system: "", user: text };
  }
}

/** The result of checking a reply that should be JSON. */
export interface JsonCheck {
  ok: boolean;
  /** What is wrong, when it is wrong. */
  error?: string;
  /** The parsed JSON, when it parsed. */
  value?: unknown;
}

/**
 * Checks that a reply is JSON, and reads it out of a code fence when the model put it in one.
 */
export function checkJson(reply: string): JsonCheck {
  const trimmed = reply.trim();
  const fenced = /```(?:json)?\s*([\s\S]*?)```/i.exec(trimmed);
  const candidate = (fenced?.[1] ?? trimmed).trim();
  if (!candidate) return { ok: false, error: "There was no JSON in the answer." };
  let value: unknown;
  try {
    value = JSON.parse(candidate);
  } catch (e) {
    return { ok: false, error: `The answer is not valid JSON: ${e instanceof Error ? e.message : "it could not be parsed"}` };
  }
  if (!Array.isArray(value) && (typeof value !== "object" || value === null)) {
    return { ok: false, error: "The answer is JSON, but not an object or a list." };
  }
  return { ok: true, value };
}
