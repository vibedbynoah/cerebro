import { readZip } from "./zip";

/**
 * Reading a Word file (.docx). A Word file is a ZIP archive whose `word/document.xml` holds the
 * paragraphs, headings, lists and tables. This reader takes the text out of that XML with the
 * browser's own XML parser, so no library is needed and nothing leaves the computer.
 */

export interface DocxText {
  /** The text, paragraph by paragraph. */
  text: string;
  /** The same content as Markdown, with headings, lists and tables. */
  markdown: string;
}

const el = (node: Element, name: string): Element | null => {
  const found = node.getElementsByTagNameNS("*", name);
  return found.length ? (found[0] as Element) : null;
};

/** The style of a paragraph ("Heading1", "Title", ...), when it has one. */
function styleOf(p: Element): string {
  const pPr = el(p, "pPr");
  if (!pPr) return "";
  const pStyle = el(pPr, "pStyle");
  if (!pStyle) return "";
  return pStyle.getAttribute("w:val") ?? pStyle.getAttribute("val") ?? "";
}

/** True when the paragraph is part of a list. */
function isListItem(p: Element): boolean {
  const pPr = el(p, "pPr");
  if (!pPr) return false;
  return !!el(pPr, "numPr");
}

/** The text of a paragraph: runs, tabs and line breaks. */
function paragraphText(p: Element): string {
  let out = "";
  const walk = (node: Node): void => {
    for (const child of Array.from(node.childNodes)) {
      if (child.nodeType === 3) {
        out += child.nodeValue ?? "";
        continue;
      }
      if (child.nodeType !== 1) continue;
      const e = child as Element;
      const name = e.localName;
      if (name === "t") out += e.textContent ?? "";
      else if (name === "tab") out += "\t";
      else if (name === "br" || name === "cr") out += "\n";
      else if (name === "delText" || name === "instrText") continue;
      else walk(e);
    }
  };
  walk(p);
  return out.replace(/[ \t]+/g, " ").trim();
}

const cellsOf = (row: Element): string[] => {
  const out: string[] = [];
  for (const cell of Array.from(row.children)) {
    if (cell.localName !== "tc") continue;
    const parts: string[] = [];
    for (const p of Array.from(cell.children)) {
      if (p.localName !== "p") continue;
      const t = paragraphText(p);
      if (t) parts.push(t);
    }
    out.push(parts.join(" "));
  }
  return out;
};

const rowsOf = (table: Element): string[][] => {
  const out: string[][] = [];
  for (const row of Array.from(table.children)) {
    if (row.localName !== "tr") continue;
    out.push(cellsOf(row));
  }
  return out;
};

/** The text and the Markdown of a Word file. */
export function docxText(bytes: Uint8Array): DocxText {
  const zip = readZip(bytes);
  const entry = zip.get("word/document.xml");
  if (!entry) throw new Error("This file has no Word document in it.");
  const xml = new TextDecoder().decode(entry.data);
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  if (doc.getElementsByTagName("parsererror").length) throw new Error("The Word document could not be read.");
  const body = doc.getElementsByTagNameNS("*", "body")[0];
  if (!body) throw new Error("This file has no Word document in it.");
  const textParts: string[] = [];
  const mdParts: string[] = [];

  const handle = (node: Element): void => {
    for (const child of Array.from(node.children)) {
      const name = child.localName;
      if (name === "p") {
        const text = paragraphText(child);
        if (!text) continue;
        textParts.push(text);
        const style = styleOf(child);
        const heading = /^Heading\s*(\d)$/i.exec(style) ?? (/^Heading(\d)$/i.exec(style) ?? null);
        if (style.toLowerCase() === "title") mdParts.push(`# ${text}`);
        else if (heading) mdParts.push(`${"#".repeat(Math.min(6, Number(heading[1] ?? 1)))} ${text}`);
        else if (isListItem(child)) mdParts.push(`- ${text}`);
        else mdParts.push(text);
        continue;
      }
      if (name === "tbl") {
        const rows = rowsOf(child);
        if (!rows.length) continue;
        const width = Math.max(...rows.map((r) => r.length));
        const line = (r: string[]): string => `| ${Array.from({ length: width }, (_, i) => r[i] ?? "").join(" | ")} |`;
        textParts.push(rows.map(line).join("\n"));
        mdParts.push(line(rows[0] ?? []));
        mdParts.push(`| ${Array.from({ length: width }, () => "---").join(" | ")} |`);
        for (const row of rows.slice(1)) mdParts.push(line(row));
        continue;
      }
      if (name === "sdt" || name === "sdtContent" || name === "ins") {
        handle(child);
        continue;
      }
    }
  };
  handle(body);
  return { text: textParts.join("\n\n"), markdown: mdParts.join("\n\n") };
}
