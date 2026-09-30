import { createSignal, onCleanup, type Accessor } from "solid-js";
import { readPref, writePref } from "./storage";

export type ThemeChoice = "system" | "dark" | "light";
export type ResolvedTheme = "dark" | "light";

/** Same key design/theme-init.ts reads before first paint. */
export const THEME_KEY = "cerebro-theme";

export function storedTheme(): ThemeChoice {
  const v = readPref(THEME_KEY);
  return v === "dark" || v === "light" ? v : "system";
}

/** "system" removes the attribute so tokens.css follows prefers-color-scheme. */
export function applyTheme(choice: ThemeChoice, root: HTMLElement = document.documentElement): void {
  if (choice === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", choice);
  writePref(THEME_KEY, choice === "system" ? null : choice);
}

export interface ThemeController {
  choice: Accessor<ThemeChoice>;
  resolved: Accessor<ResolvedTheme>;
  set(choice: ThemeChoice): void;
}

/** Must be created inside a Solid owner (it cleans up its media-query listener). */
export function createTheme(): ThemeController {
  const mql = window.matchMedia("(prefers-color-scheme: light)");
  const [choice, setChoice] = createSignal<ThemeChoice>(storedTheme());
  const [systemLight, setSystemLight] = createSignal(mql.matches);
  const onChange = () => setSystemLight(mql.matches);
  mql.addEventListener("change", onChange);
  onCleanup(() => mql.removeEventListener("change", onChange));
  return {
    choice,
    resolved: () => {
      const c = choice();
      if (c !== "system") return c;
      return systemLight() ? "light" : "dark";
    },
    set(next) {
      applyTheme(next);
      setChoice(next);
    },
  };
}

export const THEME_LABELS: Record<ThemeChoice, string> = {
  system: "Match system",
  dark: "Dark",
  light: "Light",
};
