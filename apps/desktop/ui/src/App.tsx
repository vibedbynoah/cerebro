import { BrandLockup } from "./components/Brand";
import { createEffect, createMemo, createSignal, lazy, Match, on, onCleanup, onMount, Show, Suspense, Switch, type Accessor, type JSX } from "solid-js";
import { ChatSearch } from "./components/ChatSearch";
import { CommandPalette, type PaletteCommand } from "./components/CommandPalette";
import { Sidebar } from "./components/Sidebar";
import { StatusPill } from "./components/StatusPill";
import { StatusBar } from "./components/StatusBar";
import { TitleBar } from "./components/TitleBar";
import { Toasts } from "./components/Toasts";
import { guideDue } from "./lib/guide";
import { prefersReducedMotion } from "@design/motion";
import { createRouter } from "./lib/router";
import { chatSearchOpen, setChatJump, setChatSearchOpen } from "./lib/chat/navigation";
import { routeById, ROUTES, type RouteId } from "./lib/routes";
import { readPref, writePref } from "./lib/storage";
import { createTheme, THEME_LABELS, type ThemeChoice } from "./lib/theme";
import { notify } from "./lib/toast";
import { needsOnboarding, readRecord, type OnboardingRecord } from "./onboarding/progress";
import { Setup } from "./onboarding/SetupShell";
import type { Platform } from "./platform/platform";
import { ServiceProvider, useService } from "./service/context";
import { MetricsProvider } from "./service/metrics";
import type { CerebroService } from "./service/service";
import { AppStateProvider, useAppState } from "./state";
import { Home } from "./screens/Home";

// Every screen except Home loads on demand, so first paint carries only the frame and Home; each
// screen is its own chunk under check-bundle's per-chunk cap. No <Suspense> around the screens: it
// would also catch each screen's own data loading and hold back its UI (the Models drawer link).
// The title bar's live logo loads after first paint (the static lockup shows until then).
const LiveLockup = lazy(() => import("./components/LiveMark").then((m) => ({ default: m.LiveLockup })));
const named = <K extends string>(load: () => Promise<Record<K, (p: never) => JSX.Element>>, key: K) =>
  lazy(() => load().then((m) => ({ default: m[key] as (p: any) => JSX.Element })));
const About = named(() => import("./screens/About"), "About");
const FeatureGuide = named(() => import("./components/FeatureGuide"), "FeatureGuide");
const Models = named(() => import("./screens/Models"), "Models");
const Chat = named(() => import("./screens/Chat"), "Chat");
const Documents = named(() => import("./screens/Documents"), "Documents");
const ImageStudio = named(() => import("./screens/ImageStudio"), "ImageStudio");
const Plugins = named(() => import("./screens/Plugins"), "Plugins");
const Tools = named(() => import("./screens/Tools"), "Tools");
const ModelHub = named(() => import("./screens/Hub"), "ModelHub");
const Monitor = named(() => import("./screens/Monitor"), "Monitor");
const Settings = named(() => import("./screens/Settings"), "Settings");
const Train = named(() => import("./screens/Train"), "Train");
const Tune = named(() => import("./screens/Tune"), "Tune");
const Turbo = named(() => import("./screens/Turbo"), "Turbo");
const Cluster = named(() => import("./screens/Cluster"), "Cluster");

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

  // The feature guide opens once on entering the app (and again when the list of features changes).
  const [guideOpen, setGuideOpen] = createSignal(guideDue());
  onMount(() => {
    const open = () => setGuideOpen(true);
    window.addEventListener("cerebro:open-guide", open);
    onCleanup(() => window.removeEventListener("cerebro:open-guide", open));
  });

  const linkService = useService();
  // External links (Hugging Face pages, MCP server homepages) open in the system browser: the
  // desktop window cannot navigate away or open windows of its own, so a plain link did nothing.
  onMount(() => {
    const onClick = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0) return;
      const a = (e.target as Element | null)?.closest?.("a[href]");
      const href = a?.getAttribute("href") ?? "";
      if (!/^https:\/\//i.test(href)) return;
      e.preventDefault();
      linkService.openExternal(href).catch((err: unknown) => notify("Link not opened", err instanceof Error ? err.message : String(err), "warning"));
    };
    document.addEventListener("click", onClick);
    onCleanup(() => document.removeEventListener("click", onClick));
  });

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
      id: "guide",
      label: "What Cerebro does",
      group: "Help",
      icon: "info",
      hint: "Every feature in this version",
      keywords: "features guide tour help whats new",
      run: () => setGuideOpen(true),
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
    {
      id: "search-chats",
      label: "Search chats",
      group: "Chat",
      icon: "search",
      hint: "Ctrl+Shift+F",
      keywords: "find messages conversations text full text",
      run: () => setChatSearchOpen(true),
    },
  ]);

  // Global shortcuts. Ctrl+K works everywhere, including inside text fields, like every palette.
  onMount(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (paletteOpen()) setPaletteOpen(false);
        else openPalette();
      } else if ((e.ctrlKey || e.metaKey) && !e.altKey && e.shiftKey && e.key.toLowerCase() === "f") {
        e.preventDefault();
        setPaletteOpen(false);
        setChatSearchOpen(true);
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
    <MetricsProvider>
      <div class="app" classList={{ "is-collapsed": collapsed() }}>
        <a class="skip-link" href="#main" onClick={(e) => { e.preventDefault(); main?.focus(); }}>
          Skip to content
        </a>
        <TitleBar
          platform={props.platform}
          onOpenPalette={openPalette}
          paletteOpen={paletteOpen()}
          status={<StatusPill />}
          brand={
            <Suspense fallback={<BrandLockup height={24} class="titlebar__lockup" />}>
              <LiveLockup height={24} class="titlebar__lockup" />
            </Suspense>
          }
        />
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
            <Match when={router.route() === "cluster"}>
              <Cluster />
            </Match>
            <Match when={router.route() === "tune"}>
              <Tune />
            </Match>
            <Match when={router.route() === "monitor"}>
              <Monitor />
            </Match>
            <Match when={router.route() === "turbo"}>
              <Turbo />
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
            <Match when={router.route() === "train"}>
              <Train />
            </Match>
            <Match when={router.route() === "plugins"}>
              <Plugins />
            </Match>
            <Match when={router.route() === "tools"}>
              <Tools />
            </Match>
            <Match when={router.route() === "settings"}>
              <Settings theme={theme} platform={props.platform} onRunSetup={props.onRunSetup} />
            </Match>
            <Match when={router.route() === "about"}>
              <About platform={props.platform} />
            </Match>
          </Switch>
        </main>
        <StatusBar network={state.network()} offlineMode={state.settings()?.offlineMode ?? false} />
        <TuneToasts route={router.route} />
        <p class="sr-only" role="status" aria-live="polite">
          {announce()}
        </p>
        {/* Loaded the first time it opens: the first screen does not wait for the feature list. */}
        <Show when={guideOpen()}>
          <FeatureGuide
            open={guideOpen()}
            onClose={() => setGuideOpen(false)}
            onOpen={(r: RouteId) => {
              focusMainNext = true;
              router.navigate(r);
            }}
          />
        </Show>
        <CommandPalette
          open={paletteOpen()}
          onOpenChange={setPaletteOpen}
          commands={commands()}
          returnFocus={() => (focusMainNext ? (main ?? null) : opener)}
        />
        <ChatSearch
          open={chatSearchOpen()}
          onOpenChange={setChatSearchOpen}
          onJump={(conversationId, messageId, query) => {
            setChatJump({ conversationId, messageId, query });
            focusMainNext = false;
            router.navigate("chat");
          }}
        />
      </div>
    </MetricsProvider>
  );
}

/**
 * A tune keeps running while the Tune screen is closed. When it finishes or stops there, one quiet
 * toast says so; on the Tune screen itself the run view already shows it. Replayed events (a run
 * that ended before this subscription) never toast.
 */
function TuneToasts(props: { route: Accessor<RouteId> }): JSX.Element {
  const service = useService();
  onMount(() => {
    let labels = new Map<string, string>();
    let model = "";
    const off = service.subscribeTune((events, replay) => {
      for (const e of events) {
        if (e.kind === "started") {
          labels = new Map();
          model = e.run.modelName;
        } else if (e.kind === "candidate") {
          labels.set(e.candidate.id, e.candidate.label);
        } else if (!replay && props.route() !== "tune") {
          if (e.kind === "winner")
            notify(
              "Tune finished",
              e.candidateId === e.baselineId ? `The engine's automatic settings were fastest for ${model}, so they are kept.` : `${labels.get(e.candidateId) ?? "The fastest settings"} locked in for ${model}.`,
              "success",
            );
          else if (e.kind === "failed") notify("Tune stopped", `${model}: ${e.reason}.`, "error");
        }
      }
    });
    onCleanup(off);
  });
  return null;
}
