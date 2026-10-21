import { describe, expect, it } from "vitest";
import { closest, COMMANDS, helpText, numberArg, orchestrateArg, parseCommand, speedArg, suggest, switchArg, unknownCommand } from "../../ui/src/lib/chat/commands";

describe("chat / commands", () => {
  it("parses a command and its argument, aliases included", () => {
    expect(parseCommand("/speed 4")).toMatchObject({ def: { name: "speed" }, arg: "4" });
    expect(parseCommand("  /image a red fox in snow ")).toMatchObject({ def: { name: "image" }, arg: "a red fox in snow" });
    expect(parseCommand("/imagine a cat")?.def.name).toBe("image");
    expect(parseCommand("/regenerate")?.def.name).toBe("retry");
    expect(parseCommand("/system line one\nline two")?.arg).toBe("line one\nline two");
  });

  it("is not a command when it is plain text, a path or an escaped slash", () => {
    expect(parseCommand("hello /speed 4")).toBeNull();
    expect(parseCommand("//speed")).toBeNull();
    expect(parseCommand("/usr/bin/env")).toBeNull();
    expect(unknownCommand("/usr/bin/env")).toBe("usr");
    expect(unknownCommand("/sped 3")).toBe("sped");
    expect(unknownCommand("/speed 3")).toBeNull();
  });

  it("suggests by prefix first, then by name and description", () => {
    expect(suggest("/").length).toBe(COMMANDS.length);
    expect(suggest("/s").map((c) => c.name).slice(0, 4)).toEqual(["speed", "system", "say", "stats"]);
    expect(suggest("/mem")[0]!.name).toBe("remember");
    expect(suggest("/zzz")).toEqual([]);
  });

  it("finds the closest command for a typo", () => {
    expect(closest("sped")).toBe("speed");
    expect(closest("expor")).toBe("export");
    expect(closest("halp")).toBe("help");
  });

  it("reads arguments strictly", () => {
    expect(speedArg("3")).toBe(3);
    expect(speedArg("Accurate")).toBe(4);
    expect(speedArg("6")).toBeNull();
    expect(numberArg("0.7", 0, 2)).toBe(0.7);
    expect(numberArg("2.5", 0, 2)).toBeNull();
    expect(numberArg("100.5", 16, 1000, true)).toBeNull();
    expect(numberArg("", 0, 2)).toBeNull();
    expect(switchArg("ON")).toBe(true);
    expect(switchArg("maybe")).toBeNull();
    expect(orchestrateArg("off")).toEqual({ on: false });
    expect(orchestrateArg("")).toEqual({ on: true });
    expect(orchestrateArg("5 3")).toEqual({ on: true, workers: 5, rounds: 3 });
    expect(orchestrateArg("7")).toBeNull();
    expect(orchestrateArg("2 9")).toBeNull();
  });

  it("every command has help, and names are unique", () => {
    const names = COMMANDS.flatMap((c) => [c.name, ...(c.aliases ?? [])]);
    expect(new Set(names).size).toBe(names.length);
    for (const c of COMMANDS) expect(helpText()).toContain(`/${c.name}`);
  });
});
