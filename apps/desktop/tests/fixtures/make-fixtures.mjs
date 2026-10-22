// @ts-check
// Makes the binary fixtures the converter tests read: a small PDF and a small Word file.
// Run it with `node tests/fixtures/make-fixtures.mjs` after changing what it writes.
import { deflateSync, deflateRawSync } from "node:zlib";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));

// ---- A PDF: two pages, Helvetica text, one compressed content stream --------------------------

/** @param {[number, string][]} lines */
const pageContent = (lines) => {
  const parts = [];
  for (const [size, text] of lines) {
    parts.push(`BT /F1 ${size} Tf 72 720 Td (${text}) Tj ET`);
  }
  return parts.join("\n");
};

const content1 = [
  pageContent([
    [24, "Cerebro test document"],
    [12, "Annual leave is 25 days plus public holidays."],
  ]),
  "BT /F1 12 Tf 72 680 Td [(Backups) -300 (run every night at 2am.)] TJ ET",
  "BT /F1 12 Tf 72 660 Td (The router is in the hall.) Tj ET",
].join("\n");

const content2 = pageContent([
  [18, "Second page"],
  [12, "Backups are checked every Monday morning."],
]);

function buildPdf() {
  const streams = [content1, content2].map((text) => deflateSync(Buffer.from(text, "latin1")));
  const dicts = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    `<< /Length ${streams[0]?.length ?? 0} /Filter /FlateDecode >>`,
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 7 0 R >>",
    `<< /Length ${streams[1]?.length ?? 0} /Filter /FlateDecode >>`,
  ];

  // The file is built as bytes: compressed data must not be turned into text on the way.
  /** @type {Buffer<ArrayBufferLike>[]} */
  const parts = [Buffer.from("%PDF-1.4\n", "latin1")];
  /** @type {number[]} */
  const offsets = [];
  let size = parts[0]?.length ?? 0;
  dicts.forEach((dict, i) => {
    offsets.push(size);
    const head = Buffer.from(`${i + 1} 0 obj\n${dict}\n`, "latin1");
    size += head.length;
    if (i === 4 || i === 6) {
      const stream = streams[i === 4 ? 0 : 1] ?? Buffer.alloc(0);
      parts.push(head, Buffer.from("stream\n", "latin1"), stream, Buffer.from("\nendstream\nendobj\n", "latin1"));
      size += 7 + stream.length + 17;
    } else {
      parts.push(head, Buffer.from("endobj\n", "latin1"));
      size += 7;
    }
  });
  const xref = size;
  let table = `xref\n0 ${dicts.length + 1}\n0000000000 65535 f \n`;
  for (const offset of offsets) table += `${String(offset).padStart(10, "0")} 00000 n \n`;
  table += `trailer\n<< /Size ${dicts.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  parts.push(Buffer.from(table, "latin1"));
  return Buffer.concat(parts);
}

// ---- A Word file: a ZIP with the document XML ---------------------------------------------------

const CONTENT_TYPES = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>`;

const RELS = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>`;

const DOCUMENT = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Quarterly plan</w:t></w:r></w:p>
<w:p><w:r><w:t>The team ships the new search on Friday.</w:t></w:r></w:p>
<w:p><w:pPr><w:numPr/></w:pPr><w:r><w:t>Write the release notes</w:t></w:r></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Task</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Owner</w:t></w:r></w:p></w:tc></w:tr><w:tr><w:tc><w:p><w:r><w:t>Search</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Noah</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
<w:p><w:r><w:t>End of the plan.</w:t></w:r></w:p>
</w:body></w:document>`;

/**
 * A ZIP archive with deflated entries.
 * @param {[string, string][]} files
 */
function buildZip(files) {
  /** @type {Buffer[]} */
  const locals = [];
  /** @type {Buffer[]} */
  const central = [];
  let offset = 0;
  for (const [name, text] of files) {
    const nameBytes = Buffer.from(name, "utf8");
    const data = Buffer.from(text, "utf8");
    const packed = deflateRawSync(data);
    const crc = crc32(data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0, 6);
    local.writeUInt16LE(8, 8);
    local.writeUInt16LE(0, 10);
    local.writeUInt16LE(0, 12);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(packed.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(nameBytes.length, 26);
    local.writeUInt16LE(0, 28);
    locals.push(local, nameBytes, packed);
    const entry = Buffer.alloc(46);
    entry.writeUInt32LE(0x02014b50, 0);
    entry.writeUInt16LE(20, 4);
    entry.writeUInt16LE(20, 6);
    entry.writeUInt16LE(0, 8);
    entry.writeUInt16LE(8, 10);
    entry.writeUInt16LE(0, 12);
    entry.writeUInt16LE(0, 14);
    entry.writeUInt32LE(crc, 16);
    entry.writeUInt32LE(packed.length, 20);
    entry.writeUInt32LE(data.length, 24);
    entry.writeUInt16LE(nameBytes.length, 28);
    entry.writeUInt32LE(offset, 42);
    central.push(entry, nameBytes);
    offset += local.length + nameBytes.length + packed.length;
  }
  const centralBytes = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(files.length, 8);
  end.writeUInt16LE(files.length, 10);
  end.writeUInt32LE(centralBytes.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, centralBytes, end]);
}

/** @type {Int32Array | null} */
let CRC_TABLE = null;
/** @param {Buffer} buffer */
function crc32(buffer) {
  if (!CRC_TABLE) {
    const table = new Int32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      table[n] = c;
    }
    CRC_TABLE = table;
  }
  const table = CRC_TABLE;
  let crc = -1;
  for (const byte of buffer) crc = (crc >>> 8) ^ (table[(crc ^ byte) & 0xff] ?? 0);
  return (crc ^ -1) >>> 0;
}

writeFileSync(path.join(here, "sample.pdf"), buildPdf());
writeFileSync(path.join(here, "sample.docx"), buildZip([
  ["[Content_Types].xml", CONTENT_TYPES],
  ["_rels/.rels", RELS],
  ["word/document.xml", DOCUMENT],
]));
console.log("wrote sample.pdf and sample.docx");
