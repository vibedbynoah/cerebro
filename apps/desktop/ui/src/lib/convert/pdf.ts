import { inflate } from "./inflate";

/**
 * Reading the text out of a PDF, on this computer, with no network call.
 *
 * A PDF is a set of numbered objects. The pages point at content streams, and a content stream is a
 * list of instructions: "draw this string in this font". This reader finds the pages, decodes
 * their streams (the usual ones are compressed) and takes the strings out, in order.
 *
 * What it does not do: keep the layout, read a scanned page (a picture of text has no strings to
 * read), or follow a font's own encoding beyond the map the file carries (ToUnicode). A PDF whose
 * text cannot be read gives an empty result, and the screen says so.
 */

export interface PdfText {
  /** The text, page by page. */
  text: string;
  /** How many pages the file has. */
  pages: number;
}

interface Obj {
  num: number;
  /** The object's dictionary, as written in the file. */
  dict: string;
  /** The decoded stream, when the object has one. */
  stream: Uint8Array | null;
}

interface Font {
  /** True for composite fonts, whose codes are two bytes. */
  twoByte: boolean;
  /** The map the file carries from code to character, when it has one. */
  toUnicode: Map<number, string> | null;
}

/** Bytes as a string where every character is one byte, so string indexes are byte indexes. */
function latin1(bytes: Uint8Array): string {
  let out = "";
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) out += String.fromCharCode(...bytes.subarray(i, i + step));
  return out;
}

function bytesOf(text: string): Uint8Array {
  const out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i) & 0xff;
  return out;
}

/** Where a bracketed run ends (the matching ">" or "]"). */
function matchEnd(s: string, start: number, open: string, close: string): number {
  let depth = 0;
  for (let i = start; i < s.length; i++) {
    const c = s[i];
    if (c === open) depth++;
    else if (c === close) {
      depth--;
      if (depth === 0) return i + 1;
    }
  }
  return s.length;
}

/** The value of a key in a dictionary, as written (a dictionary, an array, a name or a number). */
function dictValue(dict: string, key: string): string | null {
  const m = new RegExp(`(?:^|[\\s/])${key}(?![A-Za-z0-9])`).exec(dict);
  if (!m) return null;
  let i = m.index + m[0].length;
  while (i < dict.length && /\s/.test(dict[i] ?? "")) i++;
  if (dict.startsWith("<<", i)) return dict.slice(i, matchEnd(dict, i, "<", ">"));
  if (dict[i] === "[") return dict.slice(i, matchEnd(dict, i, "[", "]"));
  // An indirect reference ("12 0 R") is one value, spaces and all.
  const ref = /^\d+\s+\d+\s+R/.exec(dict.slice(i));
  if (ref) return ref[0];
  let end = i;
  while (end < dict.length && !/[\s\]>]/.test(dict[end] ?? "")) end++;
  return dict.slice(i, end);
}

/** The object number a value refers to ("12 0 R" -> 12). */
function refNumber(value: string | null): number | null {
  if (!value) return null;
  const m = /^\s*(\d+)\s+\d+\s+R/.exec(value);
  return m ? Number(m[1]) : null;
}

/** Every object number a value lists ("[12 0 R 13 0 R]" -> [12, 13]). */
function refList(value: string | null): number[] {
  if (!value) return [];
  const out: number[] = [];
  for (const m of value.matchAll(/(\d+)\s+\d+\s+R/g)) out.push(Number(m[1]));
  return out;
}

/** The filters a stream carries, in order ("/FlateDecode" or "[/FlateDecode]"). */
function filtersOf(dict: string): string[] {
  const m = /\/Filter\s*(\[[^\]]*\]|\/[A-Za-z0-9]+)/.exec(dict);
  if (!m) return [];
  const value = m[1] ?? "";
  if (value.startsWith("[")) return [...value.matchAll(/\/([A-Za-z0-9]+)/g)].map((x) => x[1] ?? "");
  return [value.replace(/^\//, "")];
}

function asciiHex(data: Uint8Array): Uint8Array {
  const hex = [...data].map((b) => String.fromCharCode(b)).join("").replace(/[^0-9A-Fa-f]/g, "");
  const out = new Uint8Array(Math.floor(hex.length / 2));
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return out;
}

function ascii85(data: Uint8Array): Uint8Array {
  const text = [...data].map((b) => String.fromCharCode(b)).join("");
  const out: number[] = [];
  let tuple: number[] = [];
  for (let i = 0; i < text.length; i++) {
    const c = text[i] ?? "";
    if (c === "~") break;
    if (c === "z" && !tuple.length) {
      out.push(0, 0, 0, 0);
      continue;
    }
    if (c < "!" || c > "u") continue;
    tuple.push(c.charCodeAt(0) - 33);
    if (tuple.length === 5) {
      let value = 0;
      for (const t of tuple) value = value * 85 + t;
      out.push((value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff);
      tuple = [];
    }
  }
  if (tuple.length) {
    const n = tuple.length;
    for (let i = n; i < 5; i++) tuple.push(84);
    let value = 0;
    for (const t of tuple) value = value * 85 + t;
    const bytes = [(value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff];
    for (let i = 0; i < n - 1; i++) out.push(bytes[i] ?? 0);
  }
  return Uint8Array.from(out);
}

/** Undoes the PNG and TIFF predictors some streams carry. */
function undoPredictor(data: Uint8Array, params: string): Uint8Array | null {
  const predictor = Number(/\/Predictor\s+(\d+)/.exec(params)?.[1] ?? 1);
  if (predictor <= 1) return data;
  const columns = Number(/\/Columns\s+(\d+)/.exec(params)?.[1] ?? 1);
  const colors = Number(/\/Colors\s+(\d+)/.exec(params)?.[1] ?? 1);
  const bpc = Number(/\/BitsPerComponent\s+(\d+)/.exec(params)?.[1] ?? 8);
  const bpp = Math.max(1, Math.ceil((colors * bpc) / 8));
  if (predictor === 2) {
    const rows = Math.floor(data.length / columns);
    for (let r = 0; r < rows; r++) {
      const base = r * columns;
      for (let i = bpp; i < columns; i++) data[base + i] = ((data[base + i] ?? 0) + (data[base + i - bpp] ?? 0)) & 0xff;
    }
    return data;
  }
  // PNG predictors: every row starts with the kind of filter used for it.
  const rowLength = Math.ceil((columns * colors * bpc) / 8);
  const out = new Uint8Array(Math.max(0, Math.floor(data.length / (rowLength + 1)) * rowLength));
  let prev = new Uint8Array(rowLength);
  let at = 0;
  for (let r = 0; r * (rowLength + 1) < data.length; r++) {
    const filter = data[r * (rowLength + 1)] ?? 0;
    const row = data.subarray(r * (rowLength + 1) + 1, r * (rowLength + 1) + 1 + rowLength);
    const cur = new Uint8Array(rowLength);
    for (let i = 0; i < rowLength; i++) {
      const raw = row[i] ?? 0;
      const left = i >= bpp ? (cur[i - bpp] ?? 0) : 0;
      const up = prev[i] ?? 0;
      const upLeft = i >= bpp ? (prev[i - bpp] ?? 0) : 0;
      let value = raw;
      if (filter === 1) value = raw + left;
      else if (filter === 2) value = raw + up;
      else if (filter === 3) value = raw + ((left + up) >> 1);
      else if (filter === 4) {
        const p = left + up - upLeft;
        const pa = Math.abs(p - left);
        const pb = Math.abs(p - up);
        const pc = Math.abs(p - upLeft);
        value = raw + (pa <= pb && pa <= pc ? left : pb <= pc ? up : upLeft);
      }
      cur[i] = value & 0xff;
    }
    out.set(cur, at);
    at += rowLength;
    prev = cur;
  }
  return out.slice(0, at);
}

/** Decodes a stream through the filters its dictionary names. */
function decodeStream(dict: string, data: Uint8Array): Uint8Array | null {
  let out = data;
  for (const filter of filtersOf(dict)) {
    if (filter === "FlateDecode") {
      try {
        out = inflate(out);
      } catch {
        return null;
      }
    } else if (filter === "ASCIIHexDecode") out = asciiHex(out);
    else if (filter === "ASCII85Decode") out = ascii85(out);
    else return null;
  }
  const parms = dictValue(dict, "DecodeParms");
  if (parms) {
    const undone = undoPredictor(out, parms);
    if (!undone) return null;
    out = undone;
  }
  return out;
}

/** Every object in the file, by number. */
function readObjects(raw: string): Map<number, Obj> {
  const objects = new Map<number, Obj>();
  const re = /(\d+)\s+(\d+)\s+obj\b/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(raw))) {
    const num = Number(m[1]);
    const bodyStart = m.index + m[0].length;
    const endObj = raw.indexOf("endobj", bodyStart);
    const bodyEnd = endObj < 0 ? raw.length : endObj;
    const streamAt = raw.indexOf("stream", bodyStart);
    let dict = "";
    let stream: Uint8Array | null = null;
    if (streamAt >= 0 && streamAt < bodyEnd && /^stream[\r\n]/.test(raw.slice(streamAt, streamAt + 8))) {
      dict = raw.slice(bodyStart, streamAt);
      let s = streamAt + "stream".length;
      if (raw[s] === "\r") s++;
      if (raw[s] === "\n") s++;
      // The length the dictionary gives is the truth: compressed bytes can spell "endstream".
      const lengthValue = dictValue(dict, "Length");
      const length = lengthValue !== null && /^\d+$/.test(lengthValue.trim()) ? Number(lengthValue.trim()) : null;
      const endStream = raw.indexOf("endstream", s);
      const end = length !== null && s + length <= bodyEnd ? s + length : endStream < 0 ? bodyEnd : endStream;
      const data = raw.slice(s, end);
      stream = decodeStream(dict, bytesOf(data));
    } else {
      dict = raw.slice(bodyStart, bodyEnd);
    }
    if (!objects.has(num)) objects.set(num, { num, dict, stream });
  }
  return objects;
}

/** The objects packed inside an object stream (PDF 1.5 files keep them compressed). */
function expandObjectStreams(objects: Map<number, Obj>): void {
  for (const obj of [...objects.values()]) {
    if (!obj.stream || !/\/Type\s*\/ObjStm/.test(obj.dict)) continue;
    const data = latin1(obj.stream);
    const head = readInts(data, 1);
    if (!head.values.length) continue;
    const count = head.values[0] ?? 0;
    if (count <= 0 || count > 5000) continue;
    const pairs = readInts(data, 1 + count * 2);
    const numbers = pairs.values;
    const dataStart = skipSpace(data, pairs.end);
    for (let i = 0; i < count; i++) {
      const num = numbers[1 + i * 2];
      const offset = numbers[2 + i * 2];
      if (num === undefined || offset === undefined) continue;
      const nextOffset = numbers[3 + i * 2] ?? data.length - dataStart;
      const text = data.slice(dataStart + offset, dataStart + nextOffset);
      if (!text.trim()) continue;
      if (objects.has(num)) continue;
      objects.set(num, { num, dict: text, stream: null });
    }
  }
}

function readInts(s: string, count: number): { values: number[]; end: number } {
  const values: number[] = [];
  let i = 0;
  while (values.length < count && i < s.length) {
    while (i < s.length && /\s/.test(s[i] ?? "")) i++;
    const m = /^\d+/.exec(s.slice(i));
    if (!m) break;
    values.push(Number(m[0]));
    i += m[0].length;
  }
  return { values, end: i };
}

function skipSpace(s: string, at: number): number {
  let i = at;
  while (i < s.length && (s[i] === " " || s[i] === "\r" || s[i] === "\n")) i++;
  return i;
}

/** The pages of the file, in reading order. */
function pageObjects(objects: Map<number, Obj>): Obj[] {
  const isPage = (o: Obj): boolean => /\/Type\s*\/Page(?![sA-Za-z])/.test(o.dict);
  const isPages = (o: Obj): boolean => /\/Type\s*\/Pages(?![A-Za-z])/.test(o.dict);
  const out: Obj[] = [];
  const seen = new Set<number>();
  const walk = (obj: Obj | undefined): void => {
    if (!obj || seen.has(obj.num)) return;
    seen.add(obj.num);
    if (isPage(obj)) {
      out.push(obj);
      return;
    }
    for (const kid of refList(dictValue(obj.dict, "Kids"))) walk(objects.get(kid));
  };
  const roots = [...objects.values()].filter((o) => isPages(o) && !/\/Parent/.test(o.dict));
  for (const root of roots) walk(root);
  if (out.length) return out;
  return [...objects.values()].filter(isPage).sort((a, b) => a.num - b.num);
}

/** The fonts a page may use, by the name the content stream calls them. */
function fontsOf(page: Obj, objects: Map<number, Obj>, inherited: string | null): Map<string, Font> {
  const out = new Map<string, Font>();
  let resources = dictValue(page.dict, "Resources");
  if (!resources && inherited) resources = inherited;
  const ref = refNumber(resources);
  const resourcesDict = ref !== null ? objects.get(ref)?.dict ?? null : resources;
  if (!resourcesDict) return out;
  const fontValue = dictValue(resourcesDict, "Font");
  const fontRef = refNumber(fontValue);
  const fontDict = fontRef !== null ? objects.get(fontRef)?.dict ?? null : fontValue;
  if (!fontDict) return out;
  for (const m of fontDict.matchAll(/\/([A-Za-z0-9#+._-]+)\s+(\d+)\s+\d+\s+R/g)) {
    const name = m[1] ?? "";
    const obj = objects.get(Number(m[2]));
    if (!obj) continue;
    const subtype = dictValue(obj.dict, "Subtype") ?? "";
    const toUnicodeRef = refNumber(dictValue(obj.dict, "ToUnicode"));
    const toUnicodeObj = toUnicodeRef !== null ? objects.get(toUnicodeRef) : undefined;
    let toUnicode: Map<number, string> | null = null;
    if (toUnicodeObj?.stream) toUnicode = parseCMap(latin1(toUnicodeObj.stream));
    out.set(name, { twoByte: /Type0/.test(subtype), toUnicode });
  }
  return out;
}

/** The map from code to character a font's ToUnicode stream carries. */
function parseCMap(text: string): Map<number, string> {
  const map = new Map<number, string>();
  for (const m of text.matchAll(/beginbfchar([\s\S]*?)endbfchar/g)) {
    for (const p of (m[1] ?? "").matchAll(/<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>/g)) {
      map.set(parseInt(p[1] ?? "0", 16), utf16be(p[2] ?? ""));
    }
  }
  for (const m of text.matchAll(/beginbfrange([\s\S]*?)endbfrange/g)) {
    for (const r of (m[1] ?? "").matchAll(/<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(?:<([0-9A-Fa-f]*)>|\[([\s\S]*?)\])/g)) {
      const lo = parseInt(r[1] ?? "0", 16);
      const hi = parseInt(r[2] ?? "0", 16);
      if (hi < lo || hi - lo > 65535) continue;
      if (r[3] !== undefined) {
        const base = r[3];
        for (let c = lo; c <= hi; c++) map.set(c, utf16be(addHex(base, c - lo)));
      } else if (r[4] !== undefined) {
        const items = (r[4].match(/<([0-9A-Fa-f]*)>/g) ?? []).map((it) => it.slice(1, -1));
        items.forEach((hex, i) => map.set(lo + i, utf16be(hex)));
      }
    }
  }
  return map;
}

function utf16be(hex: string): string {
  let out = "";
  for (let i = 0; i + 3 < hex.length + 1; i += 4) {
    const code = parseInt(hex.slice(i, i + 4) || "0", 16);
    if (!Number.isFinite(code)) break;
    out += String.fromCharCode(code);
  }
  return out;
}

function addHex(hex: string, add: number): string {
  let digits = hex;
  let carry = add;
  for (let i = digits.length - 1; i >= 0 && carry > 0; i--) {
    const value = parseInt(digits[i] ?? "0", 16) + (carry & 0xf);
    digits = digits.slice(0, i) + (value % 16).toString(16) + digits.slice(i + 1);
    carry = (carry - (carry & 0xf)) / 16 + Math.floor(value / 16);
  }
  return digits;
}

/** The characters a simple font uses for the bytes 128 to 159 (Windows Latin 1). */
const WIN_ANSI: Readonly<Record<number, string>> = {
  128: "€", 130: "‚", 131: "ƒ", 132: "„", 133: "…", 134: "†", 135: "‡", 136: "ˆ", 137: "‰", 138: "Š", 139: "‹", 140: "Œ", 142: "Ž",
  145: "‘", 146: "’", 147: "“", 148: "”", 149: "•", 150: "–", 151: "—", 152: "˜", 153: "™", 154: "š", 155: "›", 156: "œ", 158: "ž", 159: "Ÿ",
};

type Operand =
  | { kind: "string"; bytes: number[] }
  | { kind: "number"; value: number }
  | { kind: "name"; name: string }
  | { kind: "array"; items: Operand[] };

/** The operators of a content stream, each with the operands it came with. */
interface Item {  operands: Operand[];
  op: string;
}

function readLiteral(s: string, start: number): { bytes: number[]; next: number } {
  const bytes: number[] = [];
  let i = start + 1;
  let depth = 1;
  while (i < s.length) {
    const c = s[i] ?? "";
    if (c === "\\") {
      const next = s[i + 1] ?? "";
      const simple: Record<string, number> = { n: 10, r: 13, t: 9, b: 8, f: 12 };
      if (next in simple) {
        bytes.push(simple[next] ?? 0);
        i += 2;
        continue;
      }
      if (/[0-7]/.test(next)) {
        const oct = /^[0-7]{1,3}/.exec(s.slice(i + 1))?.[0] ?? "0";
        bytes.push(parseInt(oct, 8) & 0xff);
        i += 1 + oct.length;
        continue;
      }
      if (next === "\n") {
        i += 2;
        continue;
      }
      bytes.push(next.charCodeAt(0) & 0xff);
      i += 2;
      continue;
    }
    if (c === "(") {
      depth++;
      bytes.push(40);
      i++;
      continue;
    }
    if (c === ")") {
      depth--;
      if (!depth) return { bytes, next: i + 1 };
      bytes.push(41);
      i++;
      continue;
    }
    bytes.push(c.charCodeAt(0) & 0xff);
    i++;
  }
  return { bytes, next: i };
}

function hexToBytes(hex: string): number[] {
  const clean = hex.replace(/[^0-9A-Fa-f]/g, "");
  const out: number[] = [];
  for (let i = 0; i + 1 < clean.length; i += 2) out.push(parseInt(clean.slice(i, i + 2), 16));
  if (clean.length % 2) out.push(parseInt(clean.slice(-1), 16) * 16);
  return out;
}

/** The operators of a content stream, each with the operands it came with. */
function tokenize(content: string): Item[] {
  const items: Item[] = [];
  const lists: Operand[][] = [[]];
  const isDelim = (c: string): boolean => c === "" || /[\s()<>[\]{}/%]/.test(c);
  let i = 0;
  const push = (operand: Operand): void => {
    const list = lists[lists.length - 1];
    if (list) list.push(operand);
  };
  while (i < content.length) {
    const c = content[i] ?? "";
    if (/\s/.test(c)) {
      i++;
      continue;
    }
    if (c === "%") {
      while (i < content.length && content[i] !== "\n") i++;
      continue;
    }
    if (c === "(") {
      const lit = readLiteral(content, i);
      push({ kind: "string", bytes: lit.bytes });
      i = lit.next;
      continue;
    }
    if (c === "<") {
      if (content[i + 1] === "<") {
        i = matchEnd(content, i, "<", ">");
        continue;
      }
      const end = content.indexOf(">", i);
      push({ kind: "string", bytes: hexToBytes(content.slice(i + 1, end < 0 ? content.length : end)) });
      i = end < 0 ? content.length : end + 1;
      continue;
    }
    if (c === ">") {
      i = content.indexOf(">", i + 1);
      i = i < 0 ? content.length : i + 1;
      continue;
    }
    if (c === "[") {
      lists.push([]);
      i++;
      continue;
    }
    if (c === "]") {
      const done = lists.pop() ?? [];
      push({ kind: "array", items: done });
      i++;
      continue;
    }
    if (c === "{") {
      i = matchEnd(content, i, "{", "}");
      continue;
    }
    if (c === ")") {
      i++;
      continue;
    }
    if (c === "/") {
      i++;
      let name = "";
      while (i < content.length && !isDelim(content[i] ?? "")) {
        name += content[i];
        i++;
      }
      push({ kind: "name", name });
      continue;
    }
    const num = /^[+-]?(\d+\.?\d*|\.\d+)/.exec(content.slice(i));
    if (num) {
      push({ kind: "number", value: Number(num[0]) });
      i += num[0].length;
      continue;
    }
    let op = "";
    while (i < content.length && !isDelim(content[i] ?? "")) {
      op += content[i];
      i++;
    }
    if (op) {
      const list = lists[lists.length - 1] ?? [];
      items.push({ operands: [...list], op });
      list.length = 0;
    }
  }
  return items;
}

/** The characters in a shown string, for the font that shows it. */
function decodeText(bytes: readonly number[], font: Font | undefined): string {
  if (font?.twoByte) {
    let out = "";
    for (let i = 0; i + 1 < bytes.length; i += 2) {
      const code = ((bytes[i] ?? 0) << 8) | (bytes[i + 1] ?? 0);
      const mapped = font.toUnicode?.get(code);
      if (mapped !== undefined) out += mapped;
    }
    return out;
  }
  let out = "";
  for (const b of bytes) {
    const mapped = font?.toUnicode?.get(b);
    if (mapped !== undefined) {
      out += mapped;
      continue;
    }
    out += WIN_ANSI[b] ?? String.fromCharCode(b);
  }
  return out;
}

const wordChar = (c: string): boolean => !!c && /[\p{L}\p{N}]/u.test(c);

/** The text of one content stream. */
function textOf(content: string, fonts: Map<string, Font>): string {
  const items = tokenize(content);
  const lines: string[] = [];
  let line = "";
  let font: Font | undefined;
  let lastY: number | null = null;

  const flush = () => {
    const trimmed = line.replace(/[ \t]+/g, " ").trimEnd();
    if (trimmed.trim()) lines.push(trimmed);
    line = "";
  };
  const space = () => {
    if (line && !line.endsWith(" ")) line += " ";
  };
  const show = (bytes: readonly number[]) => {
    const text = decodeText(bytes, font);
    if (!text) return;
    if (wordChar(line.slice(-1)) && wordChar(text.slice(0, 1))) space();
    line += text;
  };

  for (const item of items) {
    const stack = item.operands;
    switch (item.op) {
      case "Tf": {
        const name = stack[stack.length - 2];
        font = name && name.kind === "name" ? fonts.get(name.name) : undefined;
        break;
      }
      case "Td":
      case "TD": {
        const ty = stack[stack.length - 1];
        const tx = stack[stack.length - 2];
        if (ty && ty.kind === "number" && ty.value !== 0) flush();
        else if (tx && tx.kind === "number" && tx.value !== 0) space();
        break;
      }
      case "Tm": {
        const f = stack[stack.length - 1];
        const e = stack[stack.length - 2];
        if (f && f.kind === "number") {
          if (lastY === null || Math.abs(f.value - lastY) > 0.01) flush();
          else if (e && e.kind === "number" && e.value !== 0) space();
          lastY = f.value;
        }
        break;
      }
      case "T*": {
        flush();
        break;
      }
      case "Tj": {
        const s = stack[stack.length - 1];
        if (s && s.kind === "string") show(s.bytes);
        break;
      }
      case "'":
      case '"': {
        flush();
        const s = stack[stack.length - 1];
        if (s && s.kind === "string") show(s.bytes);
        break;
      }
      case "TJ": {
        const arr = stack[stack.length - 1];
        if (arr && arr.kind === "array") {
          for (const part of arr.items) {
            if (part.kind === "string") show(part.bytes);
            else if (part.kind === "number" && part.value <= -100) space();
          }
        }
        break;
      }
      case "BT": {
        flush();
        break;
      }
      case "ET": {
        flush();
        break;
      }
      default:
        break;
    }
  }
  flush();
  return lines.join("\n");
}

/** The text of a PDF file, page by page. */
export function pdfText(bytes: Uint8Array): PdfText {
  const raw = latin1(bytes);
  if (!raw.startsWith("%PDF")) throw new Error("This file is not a PDF.");
  const objects = readObjects(raw);
  expandObjectStreams(objects);
  const pages = pageObjects(objects);
  if (!pages.length) throw new Error("No pages could be read from this PDF.");
  const inherited = (() => {
    const root = [...objects.values()].find((o) => /\/Type\s*\/Catalog/.test(o.dict));
    const pagesObj = root ? objects.get(refNumber(dictValue(root.dict, "Pages")) ?? -1) : undefined;
    const resources = pagesObj ? dictValue(pagesObj.dict, "Resources") : null;
    if (!resources) return null;
    const ref = refNumber(resources);
    return ref !== null ? objects.get(ref)?.dict ?? null : resources;
  })();
  const texts: string[] = [];
  for (const page of pages) {
    const contents = refList(dictValue(page.dict, "Contents"));
    let body = "";
    for (const c of contents) {
      const obj = objects.get(c);
      if (obj?.stream) body += `${latin1(obj.stream)}\n`;
    }
    if (!body) continue;
    const fonts = fontsOf(page, objects, inherited);
    const text = textOf(body, fonts);
    if (text.trim()) texts.push(text);
  }
  return { text: texts.join("\n\n"), pages: pages.length };
}

