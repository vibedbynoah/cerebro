/**
 * Chat's local commands: a message that names one runs on this computer and never reaches the
 * model; anything else is an ordinary message.
 */
import { describe, expect, it } from "vitest";
import { COMMANDS, commandFor, parseCommand } from "../../ui/src/lib/chat/commands";
import { hrefForMonitorTab, monitorTabFromHash, parseHash } from "../../ui/src/lib/routes";

describe("chat commands", () => {
  it("parses a known command and what follows it", () => {
    expect(parseCommand("/bench")).toEqual({ name: "bench", arg: "" });
    expect(parseCommand("  /bench  ")?.name).toBe("bench");
    expect(parseCommand("/BENCH")?.name).toBe("bench");
    expect(parseCommand("/bench now")?.arg).toBe("now");
    expect(parseCommand("/arena")?.name).toBe("arena");
  });

  it("leaves every other message to the model", () => {
    expect(parseCommand("/imagine a red fox")).toBeNull();
    expect(parseCommand("hello")).toBeNull();
    expect(parseCommand("//bench")).toBeNull();
    expect(parseCommand("/")).toBeNull();
    expect(parseCommand("")).toBeNull();
    expect(parseCommand(" / bench")).toBeNull();
  });

  it("names a command only when the whole word is typed", () => {
    expect(commandFor("/bench")?.name).toBe("bench");
    expect(commandFor("/bench now")?.name).toBe("bench");
    expect(commandFor("/ben")).toBeNull();
    expect(commandFor("/imagine")).toBeNull();
    expect(commandFor("hello")).toBeNull();
  });

  it("describes every command in one plain line", () => {
    expect(COMMANDS.length).toBeGreaterThan(0);
    for (const c of COMMANDS) {
      expect(c.name).toMatch(/^[a-z][a-z-]*$/);
      expect(c.description.length).toBeGreaterThan(10);
      expect(c.description.endsWith(".")).toBe(true);
    }
  });
});

describe("the Monitor's two views", () => {
  it("keeps the view in the address, so a link or a command can open the lab", () => {
    expect(hrefForMonitorTab("benchmark")).toBe("#/monitor?tab=benchmark");
    expect(hrefForMonitorTab("live")).toBe("#/monitor");
    expect(monitorTabFromHash("#/monitor?tab=benchmark")).toBe("benchmark");
    expect(monitorTabFromHash("#/monitor")).toBe("live");
    expect(monitorTabFromHash("#/monitor?tab=nonsense")).toBe("live");
    // The route itself is unchanged by the query, so the app still opens Monitor.
    expect(parseHash("#/monitor?tab=benchmark")).toBe("monitor");
    expect(parseHash(hrefForMonitorTab("benchmark"))).toBe("monitor");
  });
});
