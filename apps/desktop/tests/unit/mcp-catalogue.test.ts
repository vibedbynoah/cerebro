import { describe, expect, it } from "vitest";
import { MCP_CATALOGUE } from "../../ui/src/lib/mcp-catalogue";

const RISKS = ["read", "write", "run", "network", "account"];

describe("MCP catalogue data", () => {
  it("ships at least 40 servers", () => {
    expect(MCP_CATALOGUE.length).toBeGreaterThanOrEqual(40);
  });

  it("gives every server the fields the browse panel renders", () => {
    for (const e of MCP_CATALOGUE) {
      expect(e.id, e.id).toMatch(/^[a-z0-9]+(?:-[a-z0-9]+)*$/);
      expect(e.name.trim().length, e.id).toBeGreaterThan(0);
      expect(e.category.trim().length, e.id).toBeGreaterThan(0);
      // One plain sentence: no markup, no newline, a full stop at the end.
      expect(e.description, e.id).toMatch(/^[^\n]+\.$/);
      expect(RISKS, e.id).toContain(e.risk);
      expect(typeof e.offline, e.id).toBe("boolean");
      expect(e.homepage, e.id).toMatch(/^https:\/\/\S+$/);
      if ("url" in e.install) {
        expect(e.install.url, e.id).toMatch(/^https:\/\/\S+$/);
      } else {
        expect(e.install.command.trim().length, e.id).toBeGreaterThan(0);
        for (const arg of e.install.args) {
          expect(typeof arg, e.id).toBe("string");
          expect(arg.trim().length, e.id).toBeGreaterThan(0);
        }
      }
    }
  });

  it("keeps each entry's env keys unique, named as variables, with a label, a secret flag and a source url", () => {
    for (const e of MCP_CATALOGUE) {
      const seen = new Set<string>();
      for (const key of e.env) {
        expect(key.name, e.id).toMatch(/^[A-Z][A-Z0-9_]*$/);
        expect(seen.has(key.name), `${e.id} repeats ${key.name}`).toBe(false);
        seen.add(key.name);
        expect(key.label.trim().length, `${e.id} ${key.name}`).toBeGreaterThan(0);
        expect(typeof key.secret, `${e.id} ${key.name}`).toBe("boolean");
        expect(key.url, `${e.id} ${key.name}`).toMatch(/^https:\/\/\S+$/);
      }
    }
  });

  it("has unique ids and unique names", () => {
    const ids = MCP_CATALOGUE.map((e) => e.id);
    expect(new Set(ids).size).toBe(ids.length);
    const names = MCP_CATALOGUE.map((e) => e.name.toLowerCase());
    expect(new Set(names).size).toBe(names.length);
  });

  it("starts local servers with npx, uvx or docker only", () => {
    for (const e of MCP_CATALOGUE) {
      if ("url" in e.install) continue;
      expect(["npx", "uvx", "docker"], e.id).toContain(e.install.command);
    }
  });
});
