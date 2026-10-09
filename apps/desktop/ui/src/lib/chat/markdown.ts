/**
 * A small Markdown parser for model replies. It produces a plain tree that the UI renders with
 * text nodes and a closed set of elements, so it cannot inject markup: raw HTML in a reply is shown
 * as text, link targets are limited to http(s) and mailto, and nothing is ever assigned to
 * innerHTML. The XSS corpus in tests/unit/chat-markdown.test.ts proves it.
 *
 * Supported: paragraphs, ATX headings, bullet and numbered lists, block quotes, fenced code with a
 * language, horizontal rules, and inline code, bold, italic, strikethrough and links. Streaming
 * text is parsed as-is: an unclosed fence is treated as open code until the end.
 */

export type Inline =
  | { t: "text"; v: string }
  | { t: "code"; v: string }
  | { t: "strong"; c: Inline[] }
  | { t: "em"; c: Inline[] }
  | { t: "del"; c: Inline[] }
  | { t: "link"; href: string; c: Inline[] };

export type Block =
  | { t: "p"; c: Inline[] }
  | { t: "h"; level: 1 | 2 | 3 | 4; c: Inline[] }
  | { t: "ul"; items: Inline[][] }
  | { t: "ol"; start: number; items: Inline[][] }
  | { t: "quote"; c: Block[] }
  | { t: "code"; lang: string; v: string; open: boolean }
  | { t: "hr" };

const SAFE_URL = /^(https?:\/\/[^\s<>"'`]+|mailto:[^\s<>"'`]+)$/i;

/** Only http(s) and mailto survive; everything else (javascript:, data:, vbscript:, relative) is text. */
export function safeHref(raw: string): string | null {
  const href = raw.trim();
  if (!SAFE_URL.test(href)) return null;
  // Reject control characters and whitespace tricks browsers strip before the scheme.
  if (/[\u0000-\u001f\u007f]/.test(href)) return null;
  return href;
}

export function parseMarkdown(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const out: Block[] = [];
  let i = 0;
  let para: string[] = [];
  const flush = () => {
    if (para.length) out.push({ t: "p", c: parseInline(para.join(" ")) });
    para = [];
  };

  while (i < lines.length) {
    const line = lines[i]!;
    const fence = /^\s*(```+|~~~+)\s*([\w+#.-]*)\s*$/.exec(line);
    if (fence) {
      flush();
      const marker = fence[1]!;
      const body: string[] = [];
      i++;
      let closed = false;
      while (i < lines.length) {
        if (lines[i]!.trim().startsWith(marker[0]!.repeat(marker.length)) && lines[i]!.trim().replace(/[`~]/g, "") === "") {
          closed = true;
          i++;
          break;
        }
        body.push(lines[i]!);
        i++;
      }
      out.push({ t: "code", lang: (fence[2] ?? "").toLowerCase().slice(0, 20), v: body.join("\n"), open: !closed });
      continue;
    }
    if (/^\s*$/.test(line)) {
      flush();
      i++;
      continue;
    }
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) {
      flush();
      out.push({ t: "h", level: h[1]!.length as 1 | 2 | 3 | 4, c: parseInline(h[2]!.replace(/\s+#+\s*$/, "")) });
      i++;
      continue;
    }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      flush();
      out.push({ t: "hr" });
      i++;
      continue;
    }
    if (/^\s*>/.test(line)) {
      flush();
      const body: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i]!)) body.push(lines[i]!.replace(/^\s*>\s?/, "")), i++;
      out.push({ t: "quote", c: parseMarkdown(body.join("\n")) });
      continue;
    }
    if (/^\s*[-*+]\s+/.test(line)) {
      flush();
      const items: Inline[][] = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i]!)) items.push(parseInline(lines[i]!.replace(/^\s*[-*+]\s+/, ""))), i++;
      out.push({ t: "ul", items });
      continue;
    }
    const ol = /^\s*(\d{1,9})[.)]\s+/.exec(line);
    if (ol) {
      flush();
      const items: Inline[][] = [];
      while (i < lines.length && /^\s*\d{1,9}[.)]\s+/.test(lines[i]!)) items.push(parseInline(lines[i]!.replace(/^\s*\d{1,9}[.)]\s+/, ""))), i++;
      out.push({ t: "ol", start: Number(ol[1]), items });
      continue;
    }
    para.push(line.trim());
    i++;
  }
  flush();
  return out;
}

export function parseInline(src: string): Inline[] {
  const out: Inline[] = [];
  let buf = "";
  const text = () => {
    if (buf) out.push({ t: "text", v: buf });
    buf = "";
  };
  let i = 0;
  while (i < src.length) {
    const ch = src[i]!;
    if (ch === "\\" && i + 1 < src.length && /[\\`*_~[\]()#>-]/.test(src[i + 1]!)) {
      buf += src[i + 1];
      i += 2;
      continue;
    }
    if (ch === "`") {
      const run = /^`+/.exec(src.slice(i))![0];
      const end = src.indexOf(run, i + run.length);
      if (end > 0) {
        text();
        out.push({ t: "code", v: src.slice(i + run.length, end).replace(/^ (.*) $/, "$1") });
        i = end + run.length;
        continue;
      }
    }
    const delim = /^(\*\*|__|~~|\*|_)/.exec(src.slice(i))?.[0];
    if (delim) {
      const end = src.indexOf(delim, i + delim.length);
      const inner = end > 0 ? src.slice(i + delim.length, end) : "";
      // Intraword underscores (snake_case) are text, as in CommonMark.
      const intraword = delim[0] === "_" && /\w/.test(src[i - 1] ?? "");
      if (end > 0 && inner.trim() === inner && inner.length > 0 && !intraword) {
        text();
        const c = parseInline(inner);
        out.push(delim === "~~" ? { t: "del", c } : delim.length === 2 ? { t: "strong", c } : { t: "em", c });
        i = end + delim.length;
        continue;
      }
    }
    if (ch === "[") {
      const m = /^\[([^\]\n]{1,500})\]\(([^)\s]{1,2000})\)/.exec(src.slice(i));
      if (m) {
        const href = safeHref(m[2]!);
        text();
        if (href) out.push({ t: "link", href, c: parseInline(m[1]!) });
        else out.push({ t: "text", v: m[0] });
        i += m[0].length;
        continue;
      }
    }
    const auto = /^https?:\/\/[^\s<>"'`)\]]+/.exec(src.slice(i));
    if (auto && (i === 0 || /[\s(]/.test(src[i - 1]!))) {
      const href = safeHref(auto[0].replace(/[.,;:!?]+$/, ""));
      if (href) {
        text();
        out.push({ t: "link", href, c: [{ t: "text", v: href }] });
        i += href.length;
        continue;
      }
    }
    buf += ch;
    i++;
  }
  text();
  return out;
}

/** Plain text of a reply, for copy and read-aloud. */
export function plainText(blocks: readonly Block[]): string {
  const inl = (c: readonly Inline[]): string => c.map((x) => (x.t === "text" || x.t === "code" ? x.v : inl(x.c))).join("");
  return blocks
    .map((b) => {
      switch (b.t) {
        case "p":
        case "h":
          return inl(b.c);
        case "ul":
        case "ol":
          return b.items.map((it) => `- ${inl(it)}`).join("\n");
        case "quote":
          return plainText(b.c);
        case "code":
          return b.v;
        case "hr":
          return "";
      }
    })
    .filter(Boolean)
    .join("\n\n");
}

/**
 * Splits a reply into top-level chunks at blank lines outside code fences. While a reply streams,
 * only the last chunk changes, so the UI re-parses and re-renders just that one (chunks are keyed
 * by position).
 */
export function splitChunks(src: string): string[] {
  const out: string[] = [];
  let cur: string[] = [];
  let fence: string | null = null;
  for (const line of src.replace(/\r\n?/g, "\n").split("\n")) {
    const f = /^\s*(```+|~~~+)/.exec(line);
    if (f) {
      const mark = f[1]![0]!;
      if (fence === null) fence = mark;
      else if (mark === fence && line.trim().replace(/[`~]/g, "") === "") fence = null;
    }
    if (fence === null && line.trim() === "") {
      if (cur.length) out.push(cur.join("\n"));
      cur = [];
    } else cur.push(line);
  }
  if (cur.length) out.push(cur.join("\n"));
  return out;
}

