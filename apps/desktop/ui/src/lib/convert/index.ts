import { docxText } from "./docx";
import { csvToText, htmlToMarkdown, htmlToText, jsonToText, markdownToText } from "./markup";
import { pdfText } from "./pdf";

/**
 * Turning a file into clean text, on this computer, with no network call.
 *
 * Every format Cerebro can index goes through here: plain text and Markdown as they are, HTML,
 * CSV and JSON into plain text and a table, PDF and Word through their own readers. The text is
 * what the index searches; the Markdown is what the Convert tab saves.
 */

export type ConvertKind = "text" | "markdown" | "html" | "csv" | "json" | "pdf" | "docx";

export interface Converted {
  /** Which converter ran. */
  kind: ConvertKind;
  /** The clean text, for the index. */
  text: string;
  /** The same content as Markdown. */
  markdown: string;
  /** How many pages the file had, when the format counts pages. */
  pages: number | null;
  /** Characters of clean text. */
  chars: number;
}

export interface ConvertFormat {
  kind: ConvertKind;
  /** The name shown in the screen. */
  label: string;
  /** The file endings it handles. */
  extensions: readonly string[];
}

/** Every format the Convert tab can turn into text. */
export const CONVERT_FORMATS: readonly ConvertFormat[] = [
  { kind: "pdf", label: "PDF", extensions: [".pdf"] },
  { kind: "docx", label: "Word (.docx)", extensions: [".docx"] },
  { kind: "html", label: "HTML", extensions: [".html", ".htm"] },
  { kind: "markdown", label: "Markdown", extensions: [".md", ".markdown"] },
  { kind: "csv", label: "CSV", extensions: [".csv", ".tsv"] },
  { kind: "json", label: "JSON", extensions: [".json", ".jsonl"] },
  { kind: "text", label: "Plain text", extensions: [".txt", ".text", ".log", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg"] },
];

const byExtension = (name: string): ConvertKind | null => {
  const lower = name.toLowerCase();
  for (const format of CONVERT_FORMATS) if (format.extensions.some((ext) => lower.endsWith(ext))) return format.kind;
  return null;
};

/** True when this build can turn the file into text. */
export function canConvert(name: string): boolean {
  return byExtension(name) !== null;
}

/** The converter a file name names, or null. */
export function convertKindOf(name: string): ConvertKind | null {
  return byExtension(name);
}

/** Turns text that is already text into clean text and Markdown. */
export function convertText(name: string, raw: string): Converted {
  const kind = byExtension(name) ?? "text";
  switch (kind) {
    case "markdown":
      return done(kind, markdownToText(raw), raw);
    case "html":
      return done(kind, htmlToText(raw), htmlToMarkdown(raw));
    case "csv": {
      const out = csvToText(raw);
      return done(kind, out.text, out.markdown);
    }
    case "json": {
      const out = jsonToText(raw);
      return done(kind, out.text, out.markdown);
    }
    default:
      return done("text", raw, raw);
  }
}

function done(kind: ConvertKind, text: string, markdown: string, pages: number | null = null): Converted {
  return { kind, text, markdown, pages, chars: text.length };
}

/** Turns a file's bytes into clean text and Markdown. */
export function convertBytes(name: string, bytes: Uint8Array): Converted {
  const kind = byExtension(name);
  if (!kind) throw new Error("This file type is not one Cerebro can read.");
  if (kind === "pdf") {
    const out = pdfText(bytes);
    const text = out.text;
    const markdown = out.pages > 1 ? pageMarkdown(out.text, out.pages) : text;
    return done("pdf", text, markdown, out.pages);
  }
  if (kind === "docx") {
    const out = docxText(bytes);
    return done("docx", out.text, out.markdown);
  }
  const raw = new TextDecoder("utf-8", { fatal: false }).decode(bytes);
  if (raw.includes("\u0000")) throw new Error("This file is not text. Save it as UTF-8 and try again.");
  return convertText(name, raw);
}

/** The Markdown of a PDF's pages, one heading per page. */
function pageMarkdown(text: string, pages: number): string {
  const chunks = text.split("\n\n");
  const per = Math.max(1, Math.ceil(chunks.length / pages));
  const parts: string[] = [];
  for (let i = 0; i < chunks.length; i += per) {
    const page = Math.floor(i / per) + 1;
    parts.push(`## Page ${page}\n\n${chunks.slice(i, i + per).join("\n\n")}`);
  }
  return parts.join("\n\n");
}
