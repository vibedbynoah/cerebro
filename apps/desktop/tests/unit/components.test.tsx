import { fireEvent, render, screen, waitFor } from "@solidjs/testing-library";
import { createSignal } from "solid-js";
import { describe, expect, it, vi } from "vitest";
import { CommandPalette, type PaletteCommand } from "../../ui/src/components/CommandPalette";
import { shares, StatTile } from "../../ui/src/components/Figures";
import { Icon, ICON_NAMES } from "../../ui/src/components/icons";
import { Preview } from "../../ui/src/components/Preview";
import { SettingSwitch } from "../../ui/src/components/SettingSwitch";
import { StatusBar } from "../../ui/src/components/StatusBar";
import { createMockService } from "../../ui/src/mock/mock-service";
import { ServiceProvider } from "../../ui/src/service/context";

function commands(run: (id: string) => void): PaletteCommand[] {
  return [
    { id: "go-home", label: "Home", group: "Screens", icon: "home", run: () => run("home") },
    { id: "go-tune", label: "Tune", group: "Screens", icon: "tune", keywords: "benchmark", run: () => run("tune") },
    { id: "go-settings", label: "Settings", group: "Screens", icon: "settings", run: () => run("settings") },
    { id: "theme-light", label: "Theme: Light", group: "Appearance", icon: "sun", run: () => run("light") },
  ];
}

describe("CommandPalette", () => {
  function setup() {
    const ran: string[] = [];
    const [open, setOpen] = createSignal(true);
    render(() => <CommandPalette open={open()} onOpenChange={setOpen} commands={commands((id) => ran.push(id))} returnFocus={() => null} />);
    return { ran, open };
  }

  it("lists every command, grouped, with the first one selected", async () => {
    setup();
    const options = await screen.findAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual(["Home", "Tune", "Settings", "Theme: Light"]);
    expect(options[0]?.getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("group", { name: "Appearance" })).toBeTruthy();
  });

  it("filters fuzzily, highlights matches as text, and runs the choice on Enter", async () => {
    const { ran, open } = setup();
    const input = await screen.findByRole("combobox");
    fireEvent.input(input, { target: { value: "stg" } });
    const options = screen.getAllByRole("option");
    expect(options[0]?.dataset["command"]).toBe("go-settings");
    expect(options[0]?.querySelectorAll("mark")).toHaveLength(3);
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(ran).toEqual(["settings"]));
    expect(open()).toBe(false);
  });

  it("matches keywords, wraps with the arrow keys, and says when nothing matches", async () => {
    const { ran } = setup();
    const input = await screen.findByRole("combobox");
    fireEvent.input(input, { target: { value: "benchmark" } });
    expect(screen.getAllByRole("option").map((o) => o.dataset["command"])).toEqual(["go-tune"]);

    fireEvent.input(input, { target: { value: "" } });
    fireEvent.keyDown(input, { key: "ArrowUp" });
    const last = screen.getAllByRole("option").at(-1);
    expect(last?.getAttribute("aria-selected")).toBe("true");
    expect(input.getAttribute("aria-activedescendant")).toBe(last?.id);

    fireEvent.input(input, { target: { value: "<img src=x onerror=alert(1)>" } });
    expect(screen.queryAllByRole("option")).toHaveLength(0);
    // The query is echoed as text, never parsed as markup.
    expect(document.querySelector("img")).toBeNull();
    expect(screen.getByText(/No screens or commands match/).textContent).toContain("<img src=x onerror=alert(1)>");
    fireEvent.keyDown(input, { key: "Enter" });
    expect(ran).toEqual([]);
  });
});

describe("SettingSwitch", () => {
  it("is a labelled switch with its note as the description, and reports changes", async () => {
    const onChange = vi.fn();
    render(() => <SettingSwitch id="s-integrity" label="Backend integrity check" note="Fingerprints the backend." checked onChange={onChange} />);
    const sw = screen.getByRole("switch", { name: "Backend integrity check" }) as HTMLInputElement;
    expect(sw.checked).toBe(true);
    expect(document.getElementById(sw.getAttribute("aria-describedby") ?? "")?.textContent).toBe("Fingerprints the backend.");
    fireEvent.click(sw);
    await waitFor(() => expect(onChange).toHaveBeenCalledWith(false));
  });
});

describe("StatusBar", () => {
  it("shows the mock HUD with its illustrative disclosure", async () => {
    const service = createMockService({ tickMs: 60_000 });
    render(() => (
      <ServiceProvider service={service}>
        <StatusBar network="offline" offlineMode={false} />
      </ServiceProvider>
    ));
    const meter = await screen.findByRole("meter", { name: "VRAM in use" });
    expect(meter.getAttribute("aria-valuemax")).toBe("8");
    expect(screen.getByText("Illustrative data")).toBeTruthy();
    expect(screen.getByText("Running")).toBeTruthy();
    expect(screen.getByText("127.0.0.1:8181")).toBeTruthy();
    expect(screen.getByText("No network")).toBeTruthy();
  });

  it("says offline mode when the user turned it on", async () => {
    render(() => (
      <ServiceProvider service={createMockService({ tickMs: 60_000 })}>
        <StatusBar network="online" offlineMode />
      </ServiceProvider>
    ));
    expect(await screen.findByText("Offline mode")).toBeTruthy();
  });
});

describe("icons", () => {
  it("draws every glyph on the same 24-unit grid, decorative by default", () => {
    for (const name of ICON_NAMES) {
      const { container, unmount } = render(() => <Icon name={name} />);
      const svg = container.querySelector("svg");
      expect(svg?.getAttribute("viewBox")).toBe("0 0 24 24");
      expect(svg?.getAttribute("stroke-width")).toBe("1.6");
      expect(svg?.getAttribute("aria-hidden")).toBe("true");
      expect(svg?.children.length).toBeGreaterThan(0);
      unmount();
    }
  });
});

describe("StatTile", () => {
  it("a live tile follows its value in place and is never handed to the count-up", () => {
    const [value, setValue] = createSignal(16.8);
    const { container } = render(() => <StatTile value={value()} decimals={1} unit="tok/s" live accent label="decode speed" provenance="illustrative" />);
    const digits = () => container.querySelector(".live-value__digits")?.textContent;
    expect(digits()).toBe("16.8");
    expect(container.querySelector(".sr-only")?.textContent).toBe("16.8");
    setValue(17.25);
    expect(digits()).toBe("17.3");
    setValue(9);
    expect(digits()).toBe("9.0");
    expect(container.querySelector("[data-count-to]")).toBeNull();
    expect(container.querySelector(".stat-tile")?.classList.contains("stat-tile--live")).toBe(true);
    expect(screen.getByText("Illustrative data — not a measurement")).toBeTruthy();
  });

  it("a static tile renders its final value for the count-up, and says where a measured value came from", () => {
    const { container } = render(() => <StatTile value={3} label="models on this machine" provenance="measured" source="counted" />);
    const count = container.querySelector<HTMLElement>("[data-count-to]");
    expect(count?.dataset["countTo"]).toBe("3");
    expect(count?.textContent).toBe("3");
    expect(container.querySelector(".live-value")).toBeNull();
    expect(screen.getByText("counted")).toBeTruthy();
    expect(screen.queryByText("Illustrative data — not a measurement")).toBeNull();
  });
});

describe("comparison shares", () => {
  it("scales every bar to the largest value, whichever row it is", () => {
    const rows = shares([
      { label: "Defaults", value: 52 },
      { label: "Best tuned", value: 39 },
    ]);
    expect(rows.map((r) => r.share)).toEqual([1, 0.75]);
    expect(shares([{ label: "a", value: 0 }, { label: "b", value: Number.NaN }]).map((r) => r.share)).toEqual([0, 0]);
  });
});

describe("Preview", () => {
  it("names the planned screen as a heading and hides the sketch from assistive tech", () => {
    const { container } = render(() => (
      <Preview art="chat" title="Conversations" lede="Local chat." features={[{ icon: "chat", title: "Chat", text: "Talk." }]} />
    ));
    expect(screen.getByRole("heading", { level: 2, name: "Conversations" })).toBeTruthy();
    expect(screen.getByText("Planned for a later build")).toBeTruthy();
    expect(container.querySelector(".preview__art")?.getAttribute("aria-hidden")).toBe("true");
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
  });
});
