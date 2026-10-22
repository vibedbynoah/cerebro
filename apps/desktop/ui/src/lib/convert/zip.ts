import { inflateRaw } from "./inflate";

/**
 * A small ZIP reader: enough to open a Word file (.docx is a ZIP archive). It reads the central
 * directory, then each entry's data: stored entries as they are, deflated entries through
 * lib/convert/inflate.ts. Nothing is written; the entries come back as bytes in memory.
 */

export interface ZipEntry {
  name: string;
  data: Uint8Array;
}

const EOCD = 0x06054b50;
const CENTRAL = 0x02014b50;
const LOCAL = 0x04034b50;

const u16 = (b: Uint8Array, at: number): number => (b[at] ?? 0) | ((b[at + 1] ?? 0) << 8);
const u32 = (b: Uint8Array, at: number): number => ((b[at] ?? 0) | ((b[at + 1] ?? 0) << 8) | ((b[at + 2] ?? 0) << 16) | ((b[at + 3] ?? 0) << 24)) >>> 0;

/** The end-of-central-directory record, found by looking back from the end of the file. */
function findEnd(bytes: Uint8Array): number {
  const from = Math.max(0, bytes.length - 66_000);
  for (let i = bytes.length - 22; i >= from; i--) {
    if (u32(bytes, i) === EOCD) return i;
  }
  return -1;
}

/** Every entry in the archive, by name. Names that appear twice keep the last one. */
export function readZip(bytes: Uint8Array): Map<string, ZipEntry> {
  const out = new Map<string, ZipEntry>();
  const eocd = findEnd(bytes);
  if (eocd < 0) throw new Error("This is not a ZIP archive.");
  const count = u16(bytes, eocd + 10);
  let at = u32(bytes, eocd + 16);
  for (let i = 0; i < count && at + 46 <= bytes.length; i++) {
    if (u32(bytes, at) !== CENTRAL) break;
    const method = u16(bytes, at + 10);
    const compressed = u32(bytes, at + 20);
    const nameLen = u16(bytes, at + 28);
    const extraLen = u16(bytes, at + 30);
    const commentLen = u16(bytes, at + 32);
    const localAt = u32(bytes, at + 42);
    const name = new TextDecoder().decode(bytes.subarray(at + 46, at + 46 + nameLen));
    at += 46 + nameLen + extraLen + commentLen;
    if (u32(bytes, localAt) !== LOCAL) continue;
    const localNameLen = u16(bytes, localAt + 26);
    const localExtraLen = u16(bytes, localAt + 28);
    const start = localAt + 30 + localNameLen + localExtraLen;
    const raw = bytes.subarray(start, start + compressed);
    let data: Uint8Array;
    try {
      if (method === 0) data = raw.slice();
      else if (method === 8) data = inflateRaw(raw);
      else continue; // a method this build does not read
    } catch {
      continue; // one damaged entry does not lose the others
    }
    out.set(name, { name, data });
  }
  return out;
}
