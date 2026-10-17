import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

/** Every .tsx file under `dir`. */
function tsx(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? tsx(p) : p.endsWith(".tsx") ? [p] : [];
  });
}

/**
 * Screen stylesheets load lazily with their screen and then stay, so a shared class defined in one
 * of them works only after that screen has been opened (a button that is small on the second visit
 * and large on the first). Shared button modifiers therefore live in the design system.
 */
describe("shared classes", () => {
  it("every btn-- modifier used by a screen is defined in design/components.css", () => {
    const design = readFileSync("../../design/components.css", "utf8");
    const used = new Set<string>();
    for (const f of tsx("ui/src")) for (const m of readFileSync(f, "utf8").matchAll(/\bbtn--[a-z]+\b/g)) used.add(m[0]);
    const missing = [...used].filter((c) => !new RegExp(`\\.${c}\\b`).test(design));
    expect(missing).toEqual([]);
  });
});
