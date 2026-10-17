import "../styles/plugins.css";
import { createMemo, createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Icon } from "../components/icons";
import { IllustrativeFlag, ScreenHeader } from "../components/Screen";
import { Preview } from "../components/Preview";
import { SettingSwitch } from "../components/SettingSwitch";
import { CATEGORIES, CATEGORY_LABELS, checkLocalFile, hasUpdate, loadInstalled, PLUGINS, saveInstalled, type Category, type InstalledState, type Plugin } from "../lib/plugins";
import { hrefFor } from "../lib/routes";
import { notify } from "../lib/toast";
import { useAppState } from "../state";

/**
 * Plugins: a catalogue of MCP servers and Cerebro plugins with one-click install, detail pages,
 * an Installed tab with switches and updates, and install from a local file for offline use.
 * Catalogue installs need a network; installed plugins run offline unless marked otherwise.
 */
export function Plugins(): JSX.Element {
  // No plugin runtime ships yet; MCP servers (Tools) are how models get new abilities today.
  if (__DESKTOP__) return <PluginsPlanned />;
  return <PluginsPreview />;
}

function PluginsPlanned(): JSX.Element {
  return (
    <div class="screen">
      <ScreenHeader route="plugins" />
      <Preview
        art="plugins"
        title="Plugins that add to Cerebro"
        lede={
          <>
            A catalogue of add-ons is planned. Today, give your models new tools through MCP servers on the{" "}
            <a class="text-link" href={hrefFor("tools")}>
              Tools
            </a>{" "}
            screen: files, git, web fetch, memory, databases and anything else that speaks MCP.
          </>
        }
        features={[
          { icon: "plugins", title: "One-click add-ons", text: "Install, update and remove from a catalogue, each with the permissions it asks for." },
          { icon: "lock", title: "Your rules", text: "Every plugin runs behind the same per-tool permissions as MCP servers." },
          { icon: "offline", title: "Offline first", text: "Each plugin says whether it ever uses the network." },
        ]}
      />
    </div>
  );
}

/** The browser preview's design study of the catalogue (illustrative; nothing installs). */
function PluginsPreview(): JSX.Element {
  const app = useAppState();
  const [state, setStateS] = createSignal<InstalledState>(loadInstalled());
  const [tab, setTab] = createSignal<"browse" | "installed">("browse");
  const [category, setCategory] = createSignal<Category | "all">("all");
  const [query, setQuery] = createSignal("");
  const [installing, setInstalling] = createSignal<Record<string, number>>({});
  const [detail, setDetail] = createSignal<Plugin | null>(null);
  const timers: ReturnType<typeof setInterval>[] = [];
  onCleanup(() => timers.forEach(clearInterval));

  const setState = (s: InstalledState) => {
    setStateS(s);
    saveInstalled(s);
  };
  const online = () => app.network() === "online" && app.settings()?.offlineMode !== true;
  const installsAllowed = () => app.settings()?.allowPluginInstalls !== false;
  const isInstalled = (id: string) => id in state().installed;

  const install = (p: Plugin) => {
    if (!online() || !installsAllowed()) {
      notify("Not installed", installsAllowed() ? "Installing from the catalogue needs a network. Install from a file works offline." : "Plugin installs are switched off in Settings.", "warning");
      return;
    }
    if (installing()[p.id] !== undefined || isInstalled(p.id)) return;
    setInstalling((m) => ({ ...m, [p.id]: 0 }));
    const fast = new URLSearchParams(location.search).getAll("mock").includes("fast-plugins") ? 10 : 1;
    const t = setInterval(() => {
      const cur = (installing()[p.id] ?? 0) + 0.08 * fast;
      if (cur >= 1) {
        clearInterval(t);
        setInstalling((m) => {
          const n = { ...m };
          delete n[p.id];
          return n;
        });
        setState({ ...state(), installed: { ...state().installed, [p.id]: true } });
        notify(`${p.name} installed`, p.offline ? "Enabled. It runs offline." : "Enabled. It uses the network for what it does.", "success");
      } else setInstalling((m) => ({ ...m, [p.id]: cur }));
    }, 60);
    timers.push(t);
  };
  const uninstall = (p: Plugin) => {
    const installed = { ...state().installed };
    delete installed[p.id];
    setState({ ...state(), installed });
    notify(`${p.name} removed`);
  };
  const toggle = (p: Plugin, on: boolean) => setState({ ...state(), installed: { ...state().installed, [p.id]: on } });

  const fromFile = (f: File | undefined) => {
    if (!f) return;
    const err = checkLocalFile(f.name, f.size);
    if (err) return notify("Not installed", err, "warning");
    if (state().local.includes(f.name)) return notify("Already installed", f.name);
    setState({ ...state(), local: [...state().local, f.name] });
    notify("Installed from file", `${f.name}. Its manifest and signature are checked before it can run.`, "success");
  };

  const browse = createMemo(() => {
    const q = query().toLowerCase().trim();
    return PLUGINS.filter((p) => (category() === "all" || p.category === category()) && (!q || `${p.name} ${p.summary} ${CATEGORY_LABELS[p.category]}`.toLowerCase().includes(q)));
  });
  const installedList = () => PLUGINS.filter((p) => isInstalled(p.id));
  const featured = PLUGINS.filter((p) => p.featured);
  const updates = () => installedList().filter(hasUpdate).length;

  return (
    <div class="screen plugins">
      <ScreenHeader route="plugins" actions={<IllustrativeFlag />} />
      <Show when={!online() || !installsAllowed()}>
        <p class="notice" role="status">
          <Icon name="offline" size={18} />
          <span>
            {installsAllowed() ? "Offline: the catalogue is bundled, installing needs a network. Install from a file works now." : "Plugin installs are switched off."}{" "}
            <a class="text-link" href={hrefFor("settings")}>
              Settings
            </a>
          </span>
        </p>
      </Show>

      <div class="plug-tabs">
        <div class="plug-tabs__list" role="tablist" aria-label="Plugins">
        <button type="button" role="tab" aria-selected={tab() === "browse"} onClick={() => setTab("browse")}>
          Browse
        </button>
        <button type="button" role="tab" aria-selected={tab() === "installed"} onClick={() => setTab("installed")}>
          Installed ({installedList().length + state().local.length}){updates() ? ` · ${updates()} update${updates() > 1 ? "s" : ""}` : ""}
        </button>
        </div>
        <label class="btn btn--secondary plug-file" data-press>
          <Icon name="upload" size={16} />
          Install from file
          <input class="sr-only" type="file" accept=".cbplugin,.zip" onChange={(e) => fromFile(e.currentTarget.files?.[0])} />
        </label>
      </div>

      <Show
        when={tab() === "browse"}
        fallback={
          <ul class="plug-installed" role="list">
            <For each={installedList()} fallback={<li class="hint">Nothing installed yet.</li>}>
              {(p) => (
                <li class="plug-row card" data-plugin={p.id}>
                  <Glyph category={p.category} />
                  <div class="plug-row__text">
                    <button type="button" class="plug-row__name text-link" onClick={() => setDetail(p)}>
                      {p.name}
                    </button>
                    <span class="hint">
                      v{p.version}
                      {hasUpdate(p) ? ` · update to ${p.latest} available` : ""} · {p.offline ? "offline" : "uses network"}
                    </span>
                  </div>
                  <SettingSwitch id={`plugin-${p.id}`} label={`Enable ${p.name}`} note="" checked={state().installed[p.id] ?? false} onChange={(on) => toggle(p, on)} />
                  <button type="button" class="btn btn--tertiary btn--sm" onClick={() => uninstall(p)}>
                    Remove
                  </button>
                </li>
              )}
            </For>
            <For each={state().local}>
              {(f) => (
                <li class="plug-row card">
                  <span class="plug-glyph" data-cat="files" aria-hidden="true">
                    <Icon name="folder" size={20} />
                  </span>
                  <div class="plug-row__text">
                    <span class="plug-row__name">{f}</span>
                    <span class="hint">Installed from a file</span>
                  </div>
                </li>
              )}
            </For>
          </ul>
        }
      >
        <section class="plug-featured" aria-label="Featured this week">
          <h2 class="plug-h">Featured this week</h2>
          <Carousel items={featured} onOpen={setDetail} />
        </section>

        <div class="plug-filters">
          <input type="search" placeholder="Search plugins" aria-label="Search plugins" value={query()} onInput={(e) => setQuery(e.currentTarget.value)} />
          <div class="plug-cats" role="group" aria-label="Category">
            <button type="button" aria-pressed={category() === "all"} onClick={() => setCategory("all")}>
              All
            </button>
            <For each={CATEGORIES}>
              {(c) => (
                <button type="button" aria-pressed={category() === c} onClick={() => setCategory(c)}>
                  {CATEGORY_LABELS[c]}
                </button>
              )}
            </For>
          </div>
        </div>

        <ul class="plug-grid" role="list">
          <For each={browse()} fallback={<li class="hint">No plugin matches.</li>}>
            {(p) => {
              const progress = () => installing()[p.id];
              return (
                <li class="plug-card card" data-plugin={p.id}>
                  <button type="button" class="plug-card__open" onClick={() => setDetail(p)} aria-label={`Details for ${p.name}`}>
                    <Glyph category={p.category} />
                    <span class="plug-card__name">{p.name}</span>
                    <span class="plug-card__meta">
                      {CATEGORY_LABELS[p.category]} · {p.kind}
                    </span>
                    <span class="plug-card__summary">{p.summary}</span>
                  </button>
                  <div class="plug-card__foot">
                    <span class="chip" data-tone="neutral">
                      {p.offline ? "Offline" : "Needs network"}
                    </span>
                    <Show
                      when={!isInstalled(p.id)}
                      fallback={
                        <span class="plug-enabled" role="status">
                          <Icon name="check" size={14} /> Enabled
                        </span>
                      }
                    >
                      <button type="button" class="btn btn--primary btn--sm plug-install" data-progress={progress() !== undefined} onClick={() => install(p)} aria-label={`Install ${p.name}`} data-press>
                        <Show when={progress() !== undefined} fallback="Install">
                          <span class="plug-install__bar" style={{ transform: `scaleX(${progress() ?? 0})` }} />
                          <span class="plug-install__label">Installing</span>
                        </Show>
                      </button>
                    </Show>
                  </div>
                </li>
              );
            }}
          </For>
        </ul>
      </Show>

      <Show when={detail()}>
        {(p) => (
          <aside class="plug-detail card" role="dialog" aria-modal="false" aria-labelledby="plug-detail-title">
            <div class="plug-detail__head">
              <Glyph category={p().category} />
              <div>
                <h2 id="plug-detail-title">{p().name}</h2>
                <span class="hint">
                  {p().kind} · v{p().version} · {p().author} · {p().licence}
                </span>
              </div>
              <button type="button" class="plug-x" aria-label="Close details" onClick={() => setDetail(null)}>
                <Icon name="close" size={16} />
              </button>
            </div>
            <p>{p().summary}</p>
            <h3 class="plug-h3">What it can do</h3>
            <ul>
              <For each={p().abilities}>{(a) => <li>{a}</li>}</For>
            </ul>
            <h3 class="plug-h3">Permissions it asks for</h3>
            <ul>
              <For each={p().permissions}>{(a) => <li>{a}</li>}</For>
            </ul>
            <p class="hint">{p().offline ? "Runs offline once installed." : "Uses the network for what it does."} Each action still goes through your tool rules.</p>
            <p class="hint">Third-party plugins are their authors' responsibility. Cerebro checks the manifest and signature, not what the code does.</p>
            <Show when={!isInstalled(p().id)} fallback={<span class="plug-enabled"><Icon name="check" size={14} /> Installed</span>}>
              <button type="button" class="btn btn--primary" onClick={() => install(p())} data-press>
                Install
              </button>
            </Show>
          </aside>
        )}
      </Show>
    </div>
  );
}

function Carousel(props: { items: Plugin[]; onOpen: (p: Plugin) => void }): JSX.Element {
  let track!: HTMLUListElement;
  const step = (dir: 1 | -1) => track.scrollBy({ left: dir * track.clientWidth * 0.8, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  return (
    <div class="carousel">
      <button type="button" class="carousel__btn" aria-label="Previous" onClick={() => step(-1)}>
        <Icon name="arrowLeft" size={16} />
      </button>
      <ul class="carousel__track" ref={track} role="list">
        <For each={props.items}>
          {(p) => (
            <li class="carousel__item" data-cat={p.category}>
              <button type="button" onClick={() => props.onOpen(p)}>
                <Glyph category={p.category} large />
                <span class="carousel__name">{p.name}</span>
                <span class="carousel__pitch">{p.featured}</span>
              </button>
            </li>
          )}
        </For>
      </ul>
      <button type="button" class="carousel__btn" aria-label="Next" onClick={() => step(1)}>
        <Icon name="arrowRight" size={16} />
      </button>
    </div>
  );
}

/** One icon set, one style: geometric line glyphs on a tile tinted by category. No third-party logos. */
const GLYPHS: Record<Category, string> = {
  images: "M4 17l5-5 4 4 3-3 4 4M4 5h16v14H4zM15.5 8.5h.01",
  search: "M10.5 17a6.5 6.5 0 100-13 6.5 6.5 0 000 13zM15.5 15.5L20 20",
  files: "M6 3h8l4 4v14H6zM14 3v4h4M9 12h6M9 16h6",
  git: "M6 5a2 2 0 100 4 2 2 0 000-4zM6 9v6M6 15a2 2 0 100 4 2 2 0 000-4zM18 7a2 2 0 100 4 2 2 0 000-4zM18 11c0 3-4 3-10 6",
  email: "M3 6h18v12H3zM3 7l9 6 9-6",
  calendar: "M4 6h16v14H4zM4 10h16M8 3v4M16 3v4M8 14h3",
  databases: "M5 6c0-1.7 3.1-3 7-3s7 1.3 7 3-3.1 3-7 3-7-1.3-7-3zM5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3",
  browser: "M3 5h18v14H3zM3 9h18M6 7h.01M9 7h.01M12 14l2 2 4-4",
  voice: "M12 3a3 3 0 00-3 3v6a3 3 0 006 0V6a3 3 0 00-3-3zM6 11a6 6 0 0012 0M12 17v4",
};

function Glyph(props: { category: Category; large?: boolean }): JSX.Element {
  const s = props.large ? 28 : 20;
  return (
    <span class="plug-glyph" data-cat={props.category} data-large={props.large === true} aria-hidden="true">
      <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <path d={GLYPHS[props.category]} />
      </svg>
    </span>
  );
}
