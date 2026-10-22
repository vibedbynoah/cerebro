/**
 * Turning the text formats into clean text: HTML, Markdown, CSV and JSON. Everything happens in
 * the app, with the browser's own parser, so no file is uploaded and no library is needed.
 */

const BLOCK = new Set(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "pre", "blockquote", "section", "article", "header", "footer", "ul", "ol", "table", "figure", "figcaption", "dl", "dt", "dd"]);

const parseHtml = (html: string): HTMLElement | null => {
  const doc = new DOMParser().parseFromString(html, "text/html");
  return doc.body;
};

/** The text of an HTML document, without the tags. */
export function htmlToText(html: string): string {
  const body = parseHtml(html);
  if (!body) return "";
  const walk = (node: Node): string => {
    let out = "";
    for (const child of Array.from(node.childNodes)) {
      if (child.nodeType === 3) {
        out += child.nodeValue ?? "";
        continue;
      }
      if (child.nodeType !== 1) continue;
      const e = child as Element;
      const name = e.localName.toLowerCase();
      if (name === "script" || name === "style" || name === "noscript") continue;
      if (name === "br") {
        out += "\n";
        continue;
      }
      out += walk(e);
      if (BLOCK.has(name)) out += "\n";
    }
    return out;
  };
  return walk(body)
    .split("\n")
    .map((line) => line.replace(/[ \t]+/g, " ").trim())
    .filter(Boolean)
    .join("\n");
}

/** The inline content of an element, as Markdown. */
function inline(node: Node): string {
  let out = "";
  for (const child of Array.from(node.childNodes)) {
    if (child.nodeType === 3) {
      out += child.nodeValue ?? "";
      continue;
    }
    if (child.nodeType !== 1) continue;
    const e = child as Element;
    const name = e.localName.toLowerCase();
    if (name === "script" || name === "style" || name === "noscript") continue;
    if (name === "br") {
      out += "\n";
      continue;
    }
    if (name === "strong" || name === "b") {
      out += `**${inline(e).trim()}**`;
      continue;
    }
    if (name === "em" || name === "i") {
      out += `*${inline(e).trim()}*`;
      continue;
    }
    if (name === "code") {
      out += `\`${(e.textContent ?? "").trim()}\``;
      continue;
    }
    if (name === "a") {
      const label = inline(e).trim();
      const href = e.getAttribute("href") ?? "";
      out += href ? `[${label}](${href})` : label;
      continue;
    }
    if (name === "img") {
      const alt = e.getAttribute("alt") ?? "";
      if (alt) out += `[${alt}]`;
      continue;
    }
    out += inline(e);
  }
  return out;
}

function tableMarkdown(table: Element): string {
  const rows: string[][] = [];
  for (const row of Array.from(table.querySelectorAll("tr"))) {
    const cells: string[] = [];
    for (const cell of Array.from(row.children)) {
      if (cell.localName.toLowerCase() !== "td" && cell.localName.toLowerCase() !== "th") continue;
      cells.push(inline(cell).trim().replace(/\|/g, "\\|"));
    }
    if (cells.length) rows.push(cells);
  }
  if (!rows.length) return "";
  const width = Math.max(...rows.map((r) => r.length));
  const line = (r: string[]): string => `| ${Array.from({ length: width }, (_, i) => r[i] ?? "").join(" | ")} |`;
  return [line(rows[0] ?? []), `| ${Array.from({ length: width }, () => "---").join(" | ")} |`, ...rows.slice(1).map(line)].join("\n");
}

/** An HTML document as Markdown. */
export function htmlToMarkdown(html: string): string {
  const body = parseHtml(html);
  if (!body) return "";
  const blocks = (node: Element): string => {
    const parts: string[] = [];
    for (const child of Array.from(node.children)) {
      const name = child.localName.toLowerCase();
      if (name === "script" || name === "style" || name === "noscript" || name === "head") continue;
      if (/^h[1-6]$/.test(name)) {
        const text = inline(child).trim();
        if (text) parts.push(`${"#".repeat(Number(name[1]))} ${text}`);
        continue;
      }
      if (name === "p") {
        const text = inline(child).trim();
        if (text) parts.push(text);
        continue;
      }
      if (name === "pre") {
        parts.push(`\`\`\`\n${(child.textContent ?? "").replace(/\n+$/, "")}\n\`\`\``);
        continue;
      }
      if (name === "ul" || name === "ol") {
        const items = Array.from(child.children).filter((c) => c.localName.toLowerCase() === "li");
        const lines = items.map((li) => `- ${inline(li).trim()}`);
        if (lines.length) parts.push(lines.join("\n"));
        continue;
      }
      if (name === "table") {
        const table = tableMarkdown(child);
        if (table) parts.push(table);
        continue;
      }
      if (name === "blockquote") {
        const inner = blocks(child);
        if (inner) parts.push(inner.split("\n").map((l) => `> ${l}`).join("\n"));
        continue;
      }
      if (name === "hr") {
        parts.push("---");
        continue;
      }
      if (child.children.length) {
        const inner = blocks(child);
        if (inner) parts.push(inner);
        continue;
      }
      const text = inline(child).trim();
      if (text) parts.push(text);
    }
    return parts.join("\n\n");
  };
  return blocks(body);
}

/** Markdown without the markup: the words, ready to search. */
export function markdownToText(markdown: string): string {
  return markdown
    .replace(/```[a-zA-Z0-9]*\n?/g, "")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/^\s{0,3}>\s?/gm, "")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/^\s*\|/gm, "")
    .replace(/\|/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** A CSV file as rows of cells: quotes, escaped quotes and line breaks inside cells all count. */
export function parseCsv(raw: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  let i = 0;
  const text = raw.replace(/^﻿/, "");
  while (i < text.length) {
    const c = text[i] ?? "";
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          cell += '"';
          i += 2;
          continue;
        }
        quoted = false;
        i++;
        continue;
      }
      cell += c;
      i++;
      continue;
    }
    if (c === '"') {
      quoted = true;
      i++;
      continue;
    }
    if (c === ",") {
      row.push(cell);
      cell = "";
      i++;
      continue;
    }
    if (c === "\r") {
      i++;
      continue;
    }
    if (c === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
      i++;
      continue;
    }
    cell += c;
    i++;
  }
  if (cell.length || row.length) {
    row.push(cell);
    rows.push(row);
  }
  return rows.filter((r) => r.length > 1 || (r[0] ?? "").trim() !== "");
}

/** A CSV file as plain text and as a Markdown table. */
export function csvToText(raw: string): { text: string; markdown: string } {
  const rows = parseCsv(raw);
  if (!rows.length) return { text: "", markdown: "" };
  const width = Math.max(...rows.map((r) => r.length));
  const line = (r: string[]): string => Array.from({ length: width }, (_, i) => r[i] ?? "").join(" | ");
  const text = rows.map(line).join("\n");
  const escape = (s: string): string => s.replace(/\|/g, "\\|");
  const markdown = [
    `| ${Array.from({ length: width }, (_, i) => escape(rows[0]?.[i] ?? "")).join(" | ")} |`,
    `| ${Array.from({ length: width }, () => "---").join(" | ")} |`,
    ...rows.slice(1).map((r) => `| ${Array.from({ length: width }, (_, i) => escape(r[i] ?? "")).join(" | ")} |`),
  ].join("\n");
  return { text, markdown };
}

/** A JSON file as plain text (one value a line) and as Markdown (the JSON itself). */
export function jsonToText(raw: string): { text: string; markdown: string } {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new Error("This file is not valid JSON.");
  }
  const lines: string[] = [];
  const walk = (v: unknown, path: string): void => {
    if (Array.isArray(v)) {
      v.forEach((item, i) => walk(item, `${path}[${i}]`));
      return;
    }
    if (v && typeof v === "object") {
      for (const [key, item] of Object.entries(v as Record<string, unknown>)) walk(item, path ? `${path}.${key}` : key);
      return;
    }
    lines.push(`${path}: ${v === null ? "null" : String(v)}`);
  };
  walk(value, "");
  return { text: lines.join("\n"), markdown: `\`\`\`json\n${JSON.stringify(value, null, 2)}\n\`\`\`` };
}
