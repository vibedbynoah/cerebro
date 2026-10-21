import { createMemo, createSignal, For, Match, Show, Switch, type JSX } from "solid-js";
import { RouteMotif } from "../../components/Brand";
import { Icon } from "../../components/icons";
import { IllustrativeFlag } from "../../components/Screen";
import { Term } from "../../components/Term";
import { fitClass, type FitClass } from "../../lib/fit";
import { hrefFor } from "../../lib/routes";
import type { LibraryModel, ModelSource } from "../../service/types";
import { FitChip, LastTune, ModalityBadges, publisherOf, Size, SOURCE_LABELS, Spinner } from "./parts";
import { useModels } from "./store";
import { createVirtualWindow } from "./virtual";

const ROW_H = 60;
const HEAD_H = 40;

type ModalityFilter = "all" | "text" | "vision" | "audio" | "image";
type SortKey = "name" | "size" | "speed" | "recent";

const MODALITY_FILTERS: ReadonlyArray<{ id: ModalityFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "text", label: "Text" },
  { id: "vision", label: "Vision" },
  { id: "audio", label: "Audio" },
  { id: "image", label: "Image" },
];

function matchesModality(m: LibraryModel, f: ModalityFilter): boolean {
  switch (f) {
    case "all":
      return true;
    case "text":
      return m.kind === "text";
    case "vision":
      return m.kind === "vision";
    case "audio":
      return m.kind === "stt" || m.kind === "tts";
    case "image":
      return m.kind === "image";
  }
}

/** Every search word must appear in the model's name, quantization, size class, family, publisher
 *  or file name. Plain substring matching on lower-cased text; the query is never parsed as markup
 *  or a pattern. */
export function matchesQuery(m: LibraryModel, query: string): boolean {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return true;
  const hay = `${m.name} ${m.quant} ${m.params} ${m.family} ${m.architecture === "moe" ? "moe" : "dense"} ${publisherOf(m) ?? ""} ${m.file}`.toLowerCase();
  return words.every((w) => hay.includes(w));
}

export function Library(): JSX.Element {
  const store = useModels();
  const [query, setQuery] = createSignal("");
  const [modality, setModality] = createSignal<ModalityFilter>("all");
  const [fit, setFit] = createSignal<FitClass | "all">("all");
  const [source, setSource] = createSignal<ModelSource | "all">("all");
  const [sort, setSort] = createSignal<SortKey>("name");
  const [active, setActive] = createSignal(0);

  const fits = createMemo(() => {
    const hw = store.hw();
    return new Map((store.library.state === "ready" ? store.library() : []).map((m) => [m.id, fitClass(m, hw)]));
  });

  const rows = createMemo<LibraryModel[]>(() => {
    if (store.library.state !== "ready") return [];
    const q = query();
    const out = store.library().filter(
      (m) => matchesModality(m, modality()) && (fit() === "all" || fits().get(m.id) === fit()) && (source() === "all" || m.source === source()) && matchesQuery(m, q),
    );
    const by = sort();
    const cmp: Record<SortKey, (a: LibraryModel, b: LibraryModel) => number> = {
      name: (a, b) => a.name.localeCompare(b.name, "en") || a.sizeGiB - b.sizeGiB,
      size: (a, b) => b.sizeGiB - a.sizeGiB,
      speed: (a, b) => (b.tuneResult?.decodeTpsDepth ?? -1) - (a.tuneResult?.decodeTpsDepth ?? -1),
      recent: (a, b) => (b.tunedAt ?? "").localeCompare(a.tunedAt ?? ""),
    };
    return out.sort(cmp[by]);
  });

  const filtered = () => query() !== "" || modality() !== "all" || fit() !== "all" || source() !== "all";
  const clear = () => {
    setQuery("");
    setModality("all");
    setFit("all");
    setSource("all");
    setActive(0);
  };

  return (
    <div class="library">
      <div class="lib-toolbar" role="search" aria-label="Filter models">
        <label class="lib-search">
          <Icon name="search" size={16} />
          <span class="sr-only">Search models</span>
          <input
            class="lib-search__input"
            type="search"
            placeholder="Search name, quant, family, publisher"
            autocomplete="off"
            spellcheck={false}
            maxLength={120}
            value={query()}
            onInput={(e) => {
              setQuery(e.currentTarget.value);
              setActive(0);
            }}
          />
        </label>
        <fieldset class="segmented segmented--sm">
          <legend class="sr-only">Kind of model</legend>
          <For each={MODALITY_FILTERS}>
            {(f) => (
              <label class="segmented__option" data-press>
                <input type="radio" name="lib-modality" value={f.id} checked={modality() === f.id} onChange={() => (setModality(f.id), setActive(0))} />
                <span>{f.label}</span>
              </label>
            )}
          </For>
        </fieldset>
        <div class="lib-toolbar__filters">
          <span class="select select--sm">
            <select class="input" aria-label="Memory fit" value={fit()} onChange={(e) => (setFit(e.currentTarget.value as FitClass | "all"), setActive(0))}>
              <option value="all">Any fit</option>
              <Show when={store.hw()} fallback={<option value="unknown">Fit unknown</option>}>
                <option value="vram">Fits in graphics memory</option>
                <option value="partial">Partial offload</option>
                <option value="too-large">Too large</option>
              </Show>
            </select>
          </span>
          <span class="select select--sm">
            <select class="input" aria-label="Folder" value={source()} onChange={(e) => (setSource(e.currentTarget.value as ModelSource | "all"), setActive(0))}>
              <option value="all">All folders</option>
              <For each={Object.keys(SOURCE_LABELS) as ModelSource[]}>{(s) => <option value={s}>{SOURCE_LABELS[s]}</option>}</For>
            </select>
          </span>
        </div>
      </div>

      <div class="lib-meta">
        <p class="lib-count" role="status" aria-live="polite">
          <Show when={store.library.state === "ready"} fallback="Reading model folders…">
            <Show when={filtered()} fallback={`${store.library()?.length.toLocaleString("en-US") ?? 0} models`}>
              {`${rows().length.toLocaleString("en-US")} of ${store.library()?.length.toLocaleString("en-US") ?? 0} models`}
            </Show>
          </Show>
        </p>
        <Show when={filtered()}>
          <button type="button" class="btn btn--tertiary btn--xs" data-press onClick={clear}>
            Clear filters
          </button>
        </Show>
        <span class="lib-meta__spacer" />
        <span class="lib-hint" aria-hidden="true">
          <kbd class="kbd">↑</kbd>
          <kbd class="kbd">↓</kbd> move · <kbd class="kbd">Enter</kbd> details · <kbd class="kbd">→</kbd> Load
        </span>
        <label class="lib-sort">
          <span class="lib-sort__label">Sort</span>
          <span class="select select--sm select--quiet">
            <select class="input" aria-label="Sort by" value={sort()} onChange={(e) => setSort(e.currentTarget.value as SortKey)}>
              <option value="name">Name</option>
              <option value="size">Largest first</option>
              <option value="speed">Fastest tuned</option>
              <option value="recent">Recently tuned</option>
            </select>
          </span>
        </label>
        <IllustrativeFlag />
      </div>

      <Switch>
        <Match when={store.library.state === "errored"}>
          <div class="lib-state card" role="alert">
            <Icon name="alert" size={24} />
            <p class="lib-state__title">The model folders could not be read</p>
            <p class="lib-state__text">Nothing was changed. Try again; if it keeps failing, check that the folders still exist.</p>
            <button type="button" class="btn btn--secondary" data-press onClick={() => store.refetchLibrary()}>
              <Icon name="refresh" size={16} />
              Try again
            </button>
          </div>
        </Match>
        <Match when={store.library.state !== "ready"}>
          <Skeleton />
        </Match>
        <Match when={store.library()?.length === 0}>
          <EmptyLibrary />
        </Match>
        <Match when={rows().length === 0}>
          <div class="lib-state card">
            <Icon name="search" size={24} />
            <p class="lib-state__title">No models match</p>
            <p class="lib-state__text">
              Nothing in the library matches <span class="mono">{query() || "these filters"}</span>.
            </p>
            <button type="button" class="btn btn--secondary" data-press onClick={clear}>
              Clear filters
            </button>
          </div>
        </Match>
        <Match when={true}>
          <Grid rows={rows()} fits={fits()} active={active()} setActive={setActive} />
        </Match>
      </Switch>
    </div>
  );
}

function Grid(props: { rows: LibraryModel[]; fits: ReadonlyMap<string, FitClass>; active: number; setActive: (i: number) => void }): JSX.Element {
  const store = useModels();
  const win = createVirtualWindow(
    () => props.rows.length,
    ROW_H,
    () => props.active,
    HEAD_H,
  );
  let grid: HTMLDivElement | undefined;

  const focusRow = (i: number) => {
    const n = props.rows.length;
    if (n === 0) return;
    const next = Math.max(0, Math.min(n - 1, i));
    props.setActive(next);
    win.reveal(next);
    // The grid scrolls itself (reveal); the page may also need to scroll so the grid's edge, and
    // with it the row, is on screen. "nearest" moves nothing that is already visible.
    requestAnimationFrame(() => {
      const row = grid?.querySelector<HTMLElement>(`[data-index="${next}"]`);
      row?.focus({ preventScroll: true });
      row?.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
  };

  const onKey = (e: KeyboardEvent) => {
    const target = e.target as HTMLElement;
    const row = target.closest<HTMLElement>("[data-index]");
    if (!row) return;
    const i = Number(row.dataset["index"]);
    const onRow = target === row;
    const keys: Record<string, () => void> = {
      ArrowDown: () => focusRow(i + 1),
      ArrowUp: () => focusRow(i - 1),
      PageDown: () => focusRow(i + win.pageSize()),
      PageUp: () => focusRow(i - win.pageSize()),
      Home: () => focusRow(0),
      End: () => focusRow(props.rows.length - 1),
      ArrowRight: () => onRow && row.querySelector<HTMLElement>(".lib-load")?.focus(),
      ArrowLeft: () => !onRow && row.focus(),
    };
    if ((e.key === "Enter" || e.key === " ") && onRow) {
      e.preventDefault();
      const m = props.rows[i];
      if (m) store.open(m.id);
      return;
    }
    const run = keys[e.key];
    if (run) {
      e.preventDefault();
      run();
    }
  };

  return (
    <>
    <p id="lib-grid-help" class="sr-only">
      Up and Down arrows move between models; Page Up, Page Down, Home and End jump. Enter opens a model's details. Right arrow moves to its Load button; Left arrow returns to the row.
    </p>
    <div
      class="lib-grid"
      role="grid"
      aria-label="Models on this machine"
      aria-describedby="lib-grid-help"
      aria-rowcount={props.rows.length + 1}
      aria-colcount={6}
      ref={(el) => {
        grid = el;
        win.attach(el);
      }}
      onKeyDown={onKey}
    >
      <div role="rowgroup" class="lib-head">
        <div role="row" aria-rowindex={1} class="lib-row lib-row--head">
          <span role="columnheader">Model</span>
          <span role="columnheader" class="num">
            Size
          </span>
          <span role="columnheader">
            <Term k="vram">Fit</Term>
          </span>
          <span role="columnheader" class="num">
            <Term k="decode">Writing speed</Term>
          </span>
          <span role="columnheader" class="num">
            <Term k="quality">Quality</Term>
          </span>
          <span role="columnheader">
            <span class="sr-only">Load</span>
          </span>
        </div>
      </div>
      <div role="rowgroup" class="lib-body" style={{ height: `${win.height()}px` }}>
        <For each={win.indices()}>
          {(i) => {
            const m = () => props.rows[i];
            return (
              <Show when={m()}>
                {(model) => (
                  <Row model={model()} index={i} fit={props.fits.get(model().id) ?? "unknown"} active={props.active === i} onFocus={() => props.setActive(i)} />
                )}
              </Show>
            );
          }}
        </For>
      </div>
    </div>
    </>
  );
}

function Row(props: { model: LibraryModel; index: number; fit: FitClass; active: boolean; onFocus: () => void }): JSX.Element {
  const store = useModels();
  const loaded = () => store.isLoaded(props.model.id);
  const busy = () => store.isPending(props.model.id);
  const pub = () => publisherOf(props.model);
  return (
    <div
      role="row"
      aria-rowindex={props.index + 2}
      class="lib-row"
      classList={{ "is-loaded": loaded() }}
      data-index={props.index}
      data-model={props.model.id}
      tabindex={props.active ? 0 : -1}
      style={{ transform: `translateY(${props.index * ROW_H}px)` }}
      onFocus={(e) => e.target === e.currentTarget && props.onFocus()}
      onClick={(e) => {
        if ((e.target as HTMLElement).closest("button, a, .term")) return;
        store.open(props.model.id);
      }}
    >
      <span role="rowheader" class="lib-cell lib-cell--model">
        <span class="lib-name">
          <span class="lib-name__text">{props.model.name}</span>
          <ModalityBadges modalities={props.model.modalities} compact />
          <Show when={loaded()}>
            <span class="chip lib-loaded" data-tone="on">
              <span class="status-dot" data-state="running" aria-hidden="true" />
              Loaded
            </span>
          </Show>
        </span>
        <span class="lib-sub">
          <span class="mono">{props.model.quant}</span>
          <span aria-hidden="true">·</span>
          <span>
            {props.model.params} {props.model.architecture === "moe" ? "MoE" : "dense"}
          </span>
          <span aria-hidden="true">·</span>
          <span>{pub() ?? SOURCE_LABELS[props.model.source]}</span>
        </span>
      </span>
      <span role="gridcell" class="lib-cell num mono">
        <Size gib={props.model.sizeGiB} />
      </span>
      <span role="gridcell" class="lib-cell">
        <FitChip fit={props.fit} />
      </span>
      <span role="gridcell" class="lib-cell num">
        <LastTune model={props.model} />
      </span>
      <span role="gridcell" class="lib-cell num">
        {/* No quality run exists yet; speed is never shown as if it were the whole story. */}
        <span class="muted" aria-hidden="true">
          –
        </span>
        <span class="sr-only">Not measured yet</span>
      </span>
      <span role="gridcell" class="lib-cell lib-cell--action">
        <button
          type="button"
          class="btn lib-load"
          classList={{ "btn--secondary": !loaded(), "btn--tertiary": loaded() }}
          data-press
          tabindex={-1}
          aria-busy={busy() ? "true" : undefined}
          aria-label={loaded() ? `Unload ${props.model.name} ${props.model.quant}` : `Load ${props.model.name} ${props.model.quant}`}
          onClick={() => void (loaded() ? store.unload(props.model.id) : store.load(props.model.id, null))}
        >
          <Spinner />
          <span class="btn__label">{loaded() ? "Unload" : "Load"}</span>
        </button>
      </span>
    </div>
  );
}

function Skeleton(): JSX.Element {
  return (
    <div class="lib-grid lib-grid--skeleton" aria-hidden="true">
      <div class="lib-row lib-row--head">
        <span>Model</span>
      </div>
      <For each={Array.from({ length: 8 }, (_, i) => i)}>
        {(i) => (
          <div class="lib-row lib-row--skeleton" data-i={i}>
            <span class="sk-line sk-line--name" />
            <span class="sk-line sk-line--short" />
            <span class="sk-line sk-line--chip" />
            <span class="sk-line sk-line--short" />
            <span class="sk-line sk-line--tiny" />
            <span class="sk-line sk-line--btn" />
          </div>
        )}
      </For>
    </div>
  );
}

function EmptyLibrary(): JSX.Element {
  return (
    <div class="lib-empty card">
      <RouteMotif class="lib-empty__motif" />
      <span class="lib-empty__art" aria-hidden="true">
        <Icon name="models" size={28} />
      </span>
      <h3 class="lib-empty__title">No models on this machine yet</h3>
      <p class="lib-empty__text">
        Download one from Model Hub, or keep using the <Term k="gguf" /> files you already downloaded with another app: Cerebro reads their folders and
        lists every file here on its own.
      </p>
      <div class="lib-empty__actions">
        <a class="btn btn--primary" href={hrefFor("hub")} data-press data-sweep>
          <Icon name="hub" size={16} />
          Open Model Hub
        </a>
      </div>
      <p class="hint">
        <Icon name="offline" size={16} />
        Downloading needs a network; models already on disk never do.
      </p>
    </div>
  );
}
