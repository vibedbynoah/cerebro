/**
 * Illustrative security data shaped like the engine's cerebro-probe and cerebro-supervisor reports
 * (manifest check, DACL audit, Job Object and environment allowlist). Labelled illustrative on screen.
 */

export interface FileHash {
  path: string;
  sha256: string;
  ok: boolean;
}

export interface WriteGrant {
  principal: string;
  rights: string;
  /** Expected owner of the folder (the user, SYSTEM, Administrators) vs anyone else. */
  expected: boolean;
}

export interface ActivityEntry {
  at: string;
  kind: "start" | "integrity" | "audit" | "key" | "setting" | "update";
  text: string;
}

export const MANIFEST = {
  backend: "Cerebro Engine 0.1.0 (CUDA 12)",
  signedBy: "Cerebro release key 7F3A…C21E",
  verifiedAt: "2026-10-09 09:14",
  files: [
    { path: "cerebro.exe", sha256: "9b1e4c0f6a2d7e8b3c5a1f0e9d8c7b6a5f4e3d2c1b0a9f8e7d6c5b4a3f2e1d0c", ok: true },
    { path: "cerebro-engine.dll", sha256: "3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d", ok: true },
    { path: "cerebro-kernels.dll", sha256: "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90", ok: true },
    { path: "cerebro-core.dll", sha256: "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0", ok: true },
    { path: "cudart64_12.dll", sha256: "5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f", ok: true },
  ] satisfies FileHash[],
};

export const FOLDER_AUDIT = {
  folder: "C:\\Users\\you\\AppData\\Local\\Cerebro\\engines",
  grants: [
    { principal: "you (this account)", rights: "Full control", expected: true },
    { principal: "SYSTEM", rights: "Full control", expected: true },
    { principal: "Administrators", rights: "Full control", expected: true },
  ] satisfies WriteGrant[],
};

export const SANDBOX = {
  jobObject: { killOnClose: true, breakawayBlocked: true, activeProcessLimit: 4 },
  envAllowlist: ["SystemRoot", "TEMP", "TMP", "PATH (engine folder only)", "CUDA_VISIBLE_DEVICES", "CEREBRO_ENGINE_ROOT"],
  envDropped: 46,
};

export const UPDATE = { current: "0.4.0", channel: "Stable", signature: "Valid", signedBy: "Cerebro release key 7F3A…C21E", checkedAt: "2026-10-09 08:00" };

export const ACTIVITY: readonly ActivityEntry[] = [
  { at: "09:14", kind: "integrity", text: "Backend manifest verified: 5 of 5 files match." },
  { at: "09:14", kind: "audit", text: "Engine folder: only you, SYSTEM and Administrators can write." },
  { at: "09:14", kind: "start", text: "Model process started in a Job Object with 6 environment variables." },
  { at: "08:00", kind: "update", text: "Update check: 0.4.0 is current. Signature valid." },
  { at: "Yesterday", kind: "key", text: "API key rotated. The old key stopped working at once." },
  { at: "Yesterday", kind: "setting", text: "Permission prompts switched on." },
];

/** A new API key: 256 bits from the OS CSPRNG, shown once. The engine stores only its hash. */
export function generateKey(): string {
  const b = new Uint8Array(32);
  crypto.getRandomValues(b);
  return `cbr_${Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("")}`;
}

/** First and last four characters only, e.g. "cbr_9f…c21e". */
export function maskKey(key: string): string {
  return key.length > 12 ? `${key.slice(0, 6)}…${key.slice(-4)}` : "••••";
}
