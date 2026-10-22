/**
 * DEFLATE decompression (RFC 1951) and the zlib wrapper (RFC 1950), in TypeScript.
 *
 * Cerebro reads PDF and Word files on this computer, with no network call and no large library in
 * the bundle, so it needs its own decompressor. This is the same algorithm every ZIP archive and
 * every compressed PDF stream uses: a bit reader, canonical Huffman tables and the length and
 * distance tables from the specification. It reads bytes; it never writes a file.
 */

/** A byte sink that grows as it is filled. */
class Out {
  private buf = new Uint8Array(1 << 16);
  private len = 0;

  push(b: number): void {
    if (this.len === this.buf.length) {
      const next = new Uint8Array(this.buf.length * 2);
      next.set(this.buf);
      this.buf = next;
    }
    this.buf[this.len] = b;
    this.len++;
  }

  get(i: number): number {
    return this.buf[i] ?? 0;
  }

  get length(): number {
    return this.len;
  }

  bytes(): Uint8Array {
    return this.buf.slice(0, this.len);
  }
}

/** Reads bits least-significant first, as DEFLATE stores them. */
class Bits {
  private pos = 0;
  private buf = 0;
  private cnt = 0;

  constructor(private readonly data: Uint8Array) {}

  bits(need: number): number {
    let val = this.buf;
    while (this.cnt < need) {
      if (this.pos >= this.data.length) throw new Error("The compressed data ends early.");
      val |= (this.data[this.pos] ?? 0) << this.cnt;
      this.pos++;
      this.cnt += 8;
    }
    this.buf = val >>> need;
    this.cnt -= need;
    return val & ((1 << need) - 1);
  }

  /** Drops the rest of the current byte, so the next read starts at a byte boundary. */
  align(): void {
    this.buf = 0;
    this.cnt = 0;
  }
}

const MAXBITS = 15;

interface Huff {
  count: number[];
  symbol: number[];
}

/** The canonical Huffman table for a list of code lengths (the specification's "puff" method). */
function build(lengths: readonly number[]): Huff {
  const count = new Array<number>(MAXBITS + 1).fill(0);
  for (const l of lengths) count[l] = (count[l] ?? 0) + 1;
  count[0] = 0;
  const offs = new Array<number>(MAXBITS + 2).fill(0);
  for (let len = 1; len <= MAXBITS; len++) offs[len + 1] = (offs[len] ?? 0) + (count[len] ?? 0);
  const symbol = new Array<number>(lengths.length).fill(0);
  for (let sym = 0; sym < lengths.length; sym++) {
    const l = lengths[sym] ?? 0;
    if (!l) continue;
    symbol[offs[l] ?? 0] = sym;
    offs[l] = (offs[l] ?? 0) + 1;
  }
  return { count, symbol };
}

function decode(bits: Bits, h: Huff): number {
  let code = 0;
  let first = 0;
  let index = 0;
  for (let len = 1; len <= MAXBITS; len++) {
    code |= bits.bits(1);
    const count = h.count[len] ?? 0;
    if (code - first < count) return h.symbol[index + (code - first)] ?? 0;
    index += count;
    first = (first + count) << 1;
    code <<= 1;
  }
  throw new Error("The compressed data is damaged.");
}

const LENS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEXT = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DISTS = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DEXT = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

function fixedTables(): { lit: Huff; dist: Huff } {
  const lit: number[] = new Array<number>(288).fill(8);
  for (let i = 144; i < 256; i++) lit[i] = 9;
  for (let i = 256; i < 280; i++) lit[i] = 7;
  const dist: number[] = new Array<number>(30).fill(5);
  return { lit: build(lit), dist: build(dist) };
}

function dynamicTables(bits: Bits): { lit: Huff; dist: Huff } {
  const nlen = bits.bits(5) + 257;
  const ndist = bits.bits(5) + 1;
  const ncode = bits.bits(4) + 4;
  const lengths = new Array<number>(19).fill(0);
  for (let i = 0; i < ncode; i++) lengths[ORDER[i] ?? 0] = bits.bits(3);
  const lenHuff = build(lengths);
  const all: number[] = [];
  while (all.length < nlen + ndist) {
    const sym = decode(bits, lenHuff);
    if (sym < 16) {
      all.push(sym);
      continue;
    }
    if (sym === 16) {
      const prev = all[all.length - 1];
      if (prev === undefined) throw new Error("The compressed data is damaged.");
      const rep = 3 + bits.bits(2);
      for (let i = 0; i < rep; i++) all.push(prev);
      continue;
    }
    const rep = sym === 17 ? 3 + bits.bits(3) : 11 + bits.bits(7);
    for (let i = 0; i < rep; i++) all.push(0);
  }
  return { lit: build(all.slice(0, nlen)), dist: build(all.slice(nlen, nlen + ndist)) };
}

/** Decompresses a raw DEFLATE stream (no zlib header). */
export function inflateRaw(data: Uint8Array): Uint8Array {
  const bits = new Bits(data);
  const out = new Out();
  for (;;) {
    const last = bits.bits(1);
    const type = bits.bits(2);
    if (type === 0) {
      bits.align();
      const len = bits.bits(16);
      const nlen = bits.bits(16);
      if ((len ^ 0xffff) !== nlen) throw new Error("The compressed data is damaged.");
      for (let i = 0; i < len; i++) out.push(bits.bits(8));
    } else if (type === 1 || type === 2) {
      const tables = type === 1 ? fixedTables() : dynamicTables(bits);
      for (;;) {
        const sym = decode(bits, tables.lit);
        if (sym < 256) {
          out.push(sym);
          continue;
        }
        if (sym === 256) break;
        const li = sym - 257;
        if (li >= LENS.length) throw new Error("The compressed data is damaged.");
        const length = (LENS[li] ?? 0) + bits.bits(LEXT[li] ?? 0);
        const dsym = decode(bits, tables.dist);
        if (dsym >= DISTS.length) throw new Error("The compressed data is damaged.");
        const distance = (DISTS[dsym] ?? 0) + bits.bits(DEXT[dsym] ?? 0);
        if (distance > out.length) throw new Error("The compressed data is damaged.");
        const from = out.length - distance;
        for (let i = 0; i < length; i++) out.push(out.get(from + i));
      }
    } else {
      throw new Error("The compressed data uses a block type this build does not read.");
    }
    if (last) break;
  }
  return out.bytes();
}

/** Decompresses a zlib stream (the two-byte header DEFLATE files usually carry). */
export function inflate(data: Uint8Array): Uint8Array {
  if (data.length < 2) throw new Error("The compressed data is too short.");
  const cmf = data[0] ?? 0;
  const flg = data[1] ?? 0;
  if ((cmf & 0x0f) !== 8) throw new Error("This is not a zlib stream.");
  if (((cmf << 8) + flg) % 31 !== 0) throw new Error("This is not a zlib stream.");
  const start = flg & 0x20 ? 6 : 2;
  return inflateRaw(data.subarray(start));
}
