import { createEffect, createMemo, createSignal, Match, on, onCleanup, onMount, Show, Switch, type JSX } from "solid-js";
import { CommandPalette, type PaletteCommand } from "./components/CommandPalette";
import { Sidebar } from "./components/Sidebar";
import { StatusBar } from "./components/StatusBar";
import { TitleBar } from "./components/TitleBar";
import { Toasts } from "./components/Toasts";
import { prefersReducedMotion } from "@design/motion";
import { createRouter } from "./lib/router";
import { routeById, ROUTES } from "./lib/routes";
import { readPref, writePref } from "./lib/storage";
import { createTheme, THEME_LABELS, type ThemeChoice } from "./lib/theme";
import { notify } from "./lib/toast";
import { needsOnboarding, readRecord, type OnboardingRecord } from "./onboarding/progress";
import { Setup } from "./onboarding/SetupShell";
import type { Platform } from "./platform/platform";
import { ServiceProvider } from "./service/context";
import type { CerebroService } from "./service/service";
import { AppStateProvider, useAppState } from "./state";
import { Home } from "./screens/Home";
import { Models } from "./screens/Models";
import { Chat, Documents, ImageStudio, ModelHub, Plugins, Tools } from "./screens/Placeholders";
import { Security } from "./screens/Security";
import { Settings } from "./screens/Settings";
import { Tune } from "./screens/Tune";

const SIDEBAR_KEY = "cerebro-sidebar";

export function App(props: { service: CerebroService; platform: Platform }): JSX.Element {
  return (
    <ServiceProvider service={props.service}>
      <AppStateProvider>
        <Root platform={props.platform} />
      </AppStateProvider>
    </ServiceProvider>
  );
}

/**
 * First run shows setup; after that, the app. Setup is remembered once finished (onboarding/
 * progress.ts) and can be run again from Settings or the command palette. Setup loads as its own
 * chunk behind an ErrorBoundary (onboarding/SetupShell.tsx), so it costs later launches nothing and
 * a failure inside it can never keep anyone out of the app.
 */
function Root(props: { platform: Platform }): JSX.Element {
  const [record, setRecord] = createSignal<OnboardingRecord | null>(readRecord());
  const [setup, setSetup] = createSignal(needsOnboarding(record()));
  const done = (next: OnboardingRecord) => {
    setRecord(next);
    setSetup(false);
    // Keyboard and screen-reader users land in the app's content, not on <body>.
    requestAnimationFrame(() => document.getElementById("main")?.focus({ preventScroll: true }));
  };
  return (
    <>
      <Show when={setup()} fallback={<Shell platform={props.platform} onRunSetup={() => setSetup(true)} />}>
        <Setup platform={props.platform} record={record()} onComplete={done} />
      </Show>
      <Toasts />
    </>
  );
}

function Shell(props: { platform: Platform; onRunSetup: () => void }): JSX.Element {
  const theme = createTheme();
  const router = createRouter();
  onCleanup(() => router.dispose());
  const state = useAppState();

  const [collapsed, setCollapsed] = createSignal(readPref(SIDEBAR_KEY) === "collapsed");
  const toggleSidebar = () => {
    const apply = () => {
      const next = !collapsed();
      setCollapsed(next);
      writePref(SIDEBAR_KEY, next ? "collapsed" : null);
    };
    // The column change itself is a one-frame layout change; a View Transition animates it from
    // snapshots (composited transform/opacity, no per-frame layout). Skipped where unsupported
    // (older WebView2) and under reduced motion, where the new layout simply appears.
    if (typeof document.startViewTransition !== "function" || prefersReducedMotion()) {
      apply();
      return;
    }
    document.startViewTransition(apply);
  };

  const [paletteOpen, setPaletteOpen] = createSignal(false);
  let opener: HTMLElement | null = null;
  let focusMainNext = false;
  let main: HTMLElement | undefined;

  const openPalette = () => {
    if (paletteOpen()) return;
    opener = document.activeElement instanceof HTMLElement && document.activeElement !== document.body ? document.activeElement : null;
    focusMainNext = false;
    setPaletteOpen(true);
  };

  const setTheme = (choice: ThemeChoice) => {
    theme.set(choice);
    notify(`Theme: ${THEME_LABELS[choice]}`);
  };

  const commands = createMemo<PaletteCommand[]>(() => [
    ...ROUTES.map((r) => ({
      id: `go-${r.id}`,
      label: r.label,
      group: "Screens",
      icon: r.icon,
      hint: r.summary,
      keywords: r.keywords,
      run: () => {
        focusMainNext = true;
        router.navigate(r.id);
      },
    })),
    ...(Object.keys(THEME_LABELS) as ThemeChoice[]).map((choice) => ({
      id: `theme-${choice}`,
      label: `Theme: ${THEME_LABELS[choice]}`,
      group: "Appearance",
      icon: choice === "dark" ? ("moon" as const) : choice === "light" ? ("sun" as const) : ("monitor" as const),
      current: theme.choice() === choice,
      keywords: "color colour mode appearance dark light",
      run: () => setTheme(choice),
    })),
    {
      id: "sidebar",
      label: collapsed() ? "Expand sidebar" : "Collapse sidebar",
      group: "Appearance",
      icon: "sidebar",
      hint: "Ctrl B",
      keywords: "navigation menu panel",
      run: toggleSidebar,
    },
    {
      id: "setup",
      label: "Run setup again",
      group: "Help",
      icon: "bolt",
      hint: "Hardware check, first model, protections",
      keywords: "onboarding welcome first run tour wizard",
      run: props.onRunSetup,
    },
    {
      id: "offline",
      label: state.settings()?.offlineMode ? "Turn offline mode off" : "Turn offline mode on",
      group: "Network",
      icon: "offline",
      hint: "Network features only",
      keywords: "network internet wifi airplane",
      run: () => {
        const on = !(state.settings()?.offlineMode ?? false);
        void state.setSetting("offlineMode", on, `Offline mode ${on ? "on" : "off"}`);
      },
    },
  ]);

  // Global shortcuts. Ctrl+K works everywhere, including inside text fields, like every palette.
  onMount(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (paletteOpen()) setPaletteOpen(false);
        else openPalette();
      } else if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === "b" && !paletteOpen()) {
        e.preventDefault();
        toggleSidebar();
      }
    };
    window.addEventListener("keydown", onKey);
    onCleanup(() => window.removeEventListener("keydown", onKey));
  });

  // Route change: title, top of the screen, and one polite announcement.
  const [announce, setAnnounce] = createSignal("");
  createEffect(
    on(router.route, (id) => {
      const def = routeById(id);
      document.title = id === "home" ? "Cerebro" : `${def.label} · Cerebro`;
      main?.scrollTo({ top: 0 });
      setAnnounce(`${def.label} screen`);
    }),
  );

  return (
    <div class="app" classList={{ "is-collapsed": collapsed() }}>
      <a class="skip-link" href="#main" onClick={(e) => { e.preventDefault(); main?.focus(); }}>
        Skip to content
      </a>
      <TitleBar platform={props.platform} onOpenPalette={openPalette} paletteOpen={paletteOpen()} />
      <Sidebar route={router.route()} collapsed={collapsed()} onToggle={toggleSidebar} />
      <main id="main" class="main" tabindex="-1" ref={main}>
        <Switch>
          <Match when={router.route() === "home"}>
            <Home />
          </Match>
          <Match when={router.route() === "hub"}>
            <ModelHub />
          </Match>
          <Match when={router.route() === "models"}>
            <Models />
          </Match>
          <Match when={router.route() === "tune"}>
            <Tune />
          </Match>
          <Match when={router.route() === "chat"}>
            <Chat />
          </Match>
          <Match when={router.route() === "images"}>
            <ImageStudio />
          </Match>
          <Match when={router.route() === "documents"}>
            <Documents />
          </Match>
          <Match when={router.route() === "plugins"}>
            <Plugins />
          </Match>
          <Match when={router.route() === "tools"}>
            <Tools />
          </Match>
          <Match when={router.route() === "security"}>
            <Security />
          </Match>
          <Match when={router.route() === "settings"}>
            <Settings theme={theme} platform={props.platform} onRunSetup={props.onRunSetup} />
          </Match>
        </Switch>
      </main>
      <StatusBar network={state.network()} offlineMode={state.settings()?.offlineMode ?? false} />
      <p class="sr-only" role="status" aria-live="polite">
        {announce()}
      </p>
      <CommandPalette
        open={paletteOpen()}
        onOpenChange={setPaletteOpen}
        commands={commands()}
        returnFocus={() => (focusMainNext ? (main ?? null) : opener)}
      />
    </div>
  );
}
