/// <reference types="vite/client" />

/** True in the desktop (Tauri) build, false in web mode. Replaced at build time (vite.config.ts). */
declare const __DESKTOP__: boolean;

/** One table of third-party notices (scripts/third-party.mjs): each licence text is a list of
 *  paragraph indices, each distinct paragraph stored once. */
interface CerebroNotices {
  readonly paragraphs: readonly string[];
  readonly docs: ReadonlyArray<readonly number[]>;
  readonly packages: ReadonlyArray<{ readonly name: string; readonly version: string; readonly license: string; readonly texts: readonly number[]; readonly note?: string }>;
}

/** Everything the bundled Licenses page shows (vite.config.ts, cerebroLicenses). */
declare module "virtual:cerebro-licenses" {
  export const apache: string;
  /** Crates compiled into the desktop binary. */
  export const rust: CerebroNotices;
  /** node_modules packages in the built JavaScript. */
  export const bundled: CerebroNotices;
  /** False under the dev server and Vitest, where nothing is bundled and `bundled` is empty. */
  export const bundledComplete: boolean;
}
