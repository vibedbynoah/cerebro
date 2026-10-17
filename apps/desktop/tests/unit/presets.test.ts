import { describe, expect, it } from "vitest";
import { applyPreset, chatToolRules, DEFAULT_POLICY, parsePolicy, PRESETS, presetOf, type Policy } from "../../ui/src/lib/permissions";

describe("tool presets", () => {
  it("the default policy is the 'Ask before changes' preset", () => {
    expect(presetOf(DEFAULT_POLICY)).toBe("ask");
  });

  it("each preset round-trips, keeps the user's MCP rules, and reaches the chat request", () => {
    const start: Policy = { ...DEFAULT_POLICY, rules: { ...DEFAULT_POLICY.rules, "github/create_issue": "deny" } };
    for (const x of PRESETS) {
      const p = applyPreset(start, x.id);
      expect(presetOf(p)).toBe(x.id);
      expect(p.rules["github/create_issue"]).toBe("deny");
      expect(p.access).toBe(x.access);
      expect(chatToolRules(p, ["cerebro/write_file"], true).access).toBe(x.access);
    }
    expect(chatToolRules(applyPreset(start, "read-only"), ["cerebro/write_file"], true).rules["cerebro/write_file"]).toBe("deny");
    expect(chatToolRules(applyPreset(start, "full-auto"), ["cerebro/run_command", "other/tool"], true).rules).toEqual({ "cerebro/run_command": "allow", "other/tool": "allow" });
  });

  it("changing one rule makes it custom", () => {
    const p = applyPreset(DEFAULT_POLICY, "auto-edit");
    expect(presetOf({ ...p, rules: { ...p.rules, "cerebro/run_command": "always" } })).toBeNull();
    expect(presetOf({ ...p, access: "full" })).toBeNull();
  });

  it("access survives storage and bad values fall back to workspace", () => {
    expect(parsePolicy(JSON.stringify({ defaultRule: "ask", rules: {}, access: "full" })).access).toBe("full");
    expect(parsePolicy(JSON.stringify({ defaultRule: "ask", rules: {}, access: "root" })).access).toBe("workspace");
    expect(parsePolicy(JSON.stringify({ defaultRule: "ask", rules: {} })).access).toBe("workspace");
  });
});
