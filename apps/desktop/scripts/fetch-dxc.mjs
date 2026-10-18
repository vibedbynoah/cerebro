// Downloads Microsoft's DXC shader compiler (dxcompiler.dll, LLVM licence) for the installer.
// The engine's portable GPU backend compiles its DX12 shaders with it: the older FXC compiler that
// Windows has built in produced code that hung an Intel Iris Xe (cerebro-engine portable.rs).
// Pinned release and SHA-256; the files land in src-tauri/resources/dxc (not committed).
// Usage: node scripts/fetch-dxc.mjs
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const VERSION = "v1.9.2609";
const URL = `https://github.com/microsoft/DirectXShaderCompiler/releases/download/${VERSION}/dxc_2026_09_29.zip`;
const SHA256 = "ad31b1fc8443175d204f77a611fdb3ef2ec42759bdc2f1167368de24a4a7e7f1";
/** @type {Array<[string, string]>} Path in the zip, name in the output folder. */
const FILES = [
  ["bin/x64/dxcompiler.dll", "dxcompiler.dll"],
  ["LICENSE-LLVM.txt", "LICENSE-LLVM.txt"],
];

const out = join(dirname(fileURLToPath(import.meta.url)), "..", "src-tauri", "resources", "dxc");
const stamp = join(out, "VERSION");
if (existsSync(stamp) && readFileSync(stamp, "utf8").trim() === VERSION && FILES.every(([, name]) => existsSync(join(out, name)))) {
  console.log(`DXC ${VERSION} already in ${out}`);
  process.exit(0);
}

const res = await fetch(URL);
if (!res.ok) throw new Error(`DXC download failed: HTTP ${res.status}`);
const zip = Buffer.from(await res.arrayBuffer());
const got = createHash("sha256").update(zip).digest("hex");
if (got !== SHA256) throw new Error(`DXC download has the wrong SHA-256: ${got}, expected ${SHA256}`);

// Extract into an empty folder.
const work = mkdtempSync(join(tmpdir(), "cerebro-dxc-"));
try {
  const zipPath = join(work, "dxc.zip");
  writeFileSync(zipPath, zip);
  // Windows' own tar (bsdtar): a GNU tar earlier on PATH (Git Bash) cannot read zip files.
  const tar = process.platform === "win32" ? join(process.env.SystemRoot ?? "C:\\Windows", "System32", "tar.exe") : "tar";
  execFileSync(tar, ["-xf", "dxc.zip", ...FILES.map(([from]) => from)], { cwd: work, stdio: "inherit" });
  mkdirSync(out, { recursive: true });
  for (const [from, name] of FILES) renameSync(join(work, from), join(out, name));
  writeFileSync(stamp, `${VERSION}\n`);
  console.log(`DXC ${VERSION} -> ${out}`);
} finally {
  rmSync(work, { recursive: true, force: true });
}
