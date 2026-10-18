import { Dialog } from "@ark-ui/solid/dialog";
import { createEffect, createMemo, createResource, createSignal, createUniqueId, For, Match, on, onMount, Show, Switch, type JSX } from "solid-js";
import { Portal } from "solid-js/web";
import { CodeBlock } from "../../components/CodeBlock";
import { Icon } from "../../components/icons";
import { IllustrativeFlag } from "../../components/Screen";
import { Term } from "../../components/Term";
import { fitClass } from "../../lib/fit";
import { baselineParams, changedFields, draftCandidates, validateLoadParams } from "../../lib/load-params";
import { hrefFor, hrefWithModel } from "../../lib/routes";
import { notify } from "../../lib/toast";
import { activeTweakCount, applyTweaks, contextCeiling, EMPTY_TWEAKS } from "../../lib/tweaks";
import { useService } from "../../service/context";
import type { LibraryModel, LoadParams, TuneResult } from "../../service/types";
import { LoadPanel } from "./LoadPanel";
import { PlacementCard } from "./PlacementCard";
import { Metadata } from "./Metadata";
import { contextLabel, FitChip, ModalityBadges, publisherOf, Size, SOURCE_LABELS, Spinner, TuneChip } from "./parts";
import { useModels } from "./store";
import { SpeechToTextTry, TextToSpeechTry, VisionTry } from "./TryIt";
import { TweakStudio } from "./TweakStudio";

type TabId = "overview" | "load" | "tweak" | "metadata";
const TAB_LABELS: Readonly<Record<TabId, string>> = { overview: "Overview", load: "Load settings", tweak: "Tweak Studio", metadata: "Metadata" };

/** The model detail drawer: metadata, actions, and the load, tweak and metadata tools, in a sheet
 *  that slides in from the right. Escape or the backdrop closes it and focus returns to the row. */
export function ModelDrawer(): JSX.Element {
  const store = useModels();
  const model = createMemo(() => {
    const id = store.selected();
    return id ? store.byId().get(id) : undefined;
  });
  // Keeps the last model on screen while the sheet slides out after closing.
  const [shown, setShown] = createSignal<LibraryModel>();
  createEffect(() => {
    const m = model();
    if (m) setShown(m);
  });
  return (
    <Dialog.Root open={model() !== undefined} onOpenChange={(d) => !d.open && store.open(null)} lazyMount unmountOnExit preventScroll={false}>
      <Portal>
        <Dialog.Backdrop class="drawer-backdrop" />
        <Dialog.Positioner class="drawer-positioner">
          <Dialog.Content class="drawer">
            <Show when={shown()} keyed>
              {(m) => <DrawerBody model={m} />}
            </Show>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
}

function DrawerBody(props: { model: LibraryModel }): JSX.Element {
  const store = useModels();
  const service = useService();
  const m = () => props.model;
  const llm = () => m().kind === "text" || m().kind === "vision";
  const id = createUniqueId();
  const [detail] = createResource(() => m().id, (mid) => service.getModelDetail(mid));
  const tabs = createMemo<TabId[]>(() => {
    const t: TabId[] = ["overview"];
    if (llm()) t.push("load", "tweak");
    if (m().file.endsWith(".gguf")) t.push("metadata");
    return t;
  });
  const [tab, setTab] = createSignal<TabId>("overview");
  const [visited, setVisited] = createSignal<ReadonlySet<TabId>>(new Set<TabId>(["overview"]));
  createEffect(() => {
    const t = tab();
    if (!visited().has(t)) setVisited(new Set([...visited(), t]));
  });
  const tabEls: HTMLButtonElement[] = [];

  // ---- What a load uses: the panel's draft with the active tweaks laid over it (one rule, used by
  // the fit estimate, the validator, the caption and the Load button alike). ----
  const baseline = createMemo(() => baselineParams(m(), store.hw()));
  const [draft, setDraft] = createSignal<LoadParams>(baseline());
  createEffect(on(baseline, (b) => setDraft({ ...b }), { defer: true }));
  onMount(() => {
    if (llm()) void store.ensurePatch(m().sha256);
  });
  const tweaks = () => store.patch(m().sha256)?.tweaks ?? EMPTY_TWEAKS;
  const tweakCount = () => (llm() ? activeTweakCount(tweaks()) + (store.patch(m().sha256)?.fixes.length ?? 0) : 0);
  const effective = createMemo(() => applyTweaks(draft(), tweaks()));
  const ceiling = createMemo(() => contextCeiling(m(), tweaks()));
  const draftIds = createMemo(() => new Set(draftCandidates(m(), store.library() ?? []).map((d) => d.id)));
  const errors = createMemo(() => (llm() ? validateLoadParams(effective(), { ...m(), contextMax: ceiling() }, store.hw(), draftIds()) : {}));
  const changes = () => changedFields(draft(), baseline()).length;
  const errorCount = () => Object.keys(errors()).length;
  const loaded = () => store.isLoaded(m().id);
  const busy = () => store.isPending(m().id);
  // Loaded already: offer Reload once what a load would use differs from what is running.
  const running = () => store.loaded().find((l) => l.id === m().id)?.params ?? null;
  const differsFromRunning = () => {
    const r = running();
    return llm() && (errorCount() > 0 || (r ? changedFields(effective(), r).length > 0 : changes() > 0 || tweakCount() > 0));
  };
  const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
  const caption = () => {
    if (!llm()) return "Own engine, beside the chat model";
    if (errorCount() > 0) return `Fix ${plural(errorCount(), "setting", "settings")} to load`;
    const reload = loaded() && differsFromRunning() ? " · reload to apply" : "";
    const withTweaks = tweakCount() > 0 ? ` + ${plural(tweakCount(), "tweak", "tweaks")}` : "";
    if (changes() > 0) return `Custom settings · ${plural(changes(), "change", "changes")}${withTweaks}${reload}`;
    const base = m().tunedParams ? "Tuned settings" : "Starting settings";
    // A tweak changes the configuration the tune measured, so the caption stops claiming it.
    if (tweakCount() > 0) return `${base}${withTweaks} · not measured with these${reload}`;
    if (!m().tunedParams) return store.hw() ? "Starting settings (not tuned yet)" : "Starting settings, CPU only (machine not detected)";
    return "Tuned settings, measured on this machine";
  };
  const doLoad = () => void store.load(m().id, llm() && (changes() > 0 || tweakCount() > 0) ? effective() : null);

  const onTabKey = (e: KeyboardEvent, i: number) => {
    const list = tabs();
    const to = e.key === "ArrowRight" ? (i + 1) % list.length : e.key === "ArrowLeft" ? (i - 1 + list.length) % list.length : e.key === "Home" ? 0 : e.key === "End" ? list.length - 1 : -1;
    if (to < 0) return;
    e.preventDefault();
    const next = list[to];
    if (next) setTab(next);
    tabEls[to]?.focus();
  };

  const reveal = async () => {
    const opened = await service.revealInFolder(m().id).catch(() => false);
    if (!opened) notify("Reveal in folder works in the desktop app", m().path, "info");
  };

  return (
    <>
      <header class="drawer__head">
        <div class="drawer__title-row">
          <div class="drawer__titles">
            <p class="eyebrow">
              <Icon name="models" size={14} />
              {m().family} · {SOURCE_LABELS[m().source]}
            </p>
            <Dialog.Title class="drawer__title">{m().name}</Dialog.Title>
            <Dialog.Description class="drawer__desc">
              <span class="mono">{m().quant}</span> · {m().params} {m().architecture === "moe" ? "MoE" : "dense"} · <Size gib={m().sizeGiB} />
            </Dialog.Description>
          </div>
          <Dialog.CloseTrigger class="btn btn--icon drawer__close" aria-label="Close details" data-press>
            <Icon name="close" size={18} />
          </Dialog.CloseTrigger>
        </div>
        <div class="drawer__badges">
          <ModalityBadges modalities={m().modalities} />
          <FitChip fit={fitClass(m(), store.hw())} />
          <Show when={llm()}>
            <TuneChip model={m()} />
          </Show>
        </div>
        <div class="drawer__load">
          <Show
            when={!loaded() || differsFromRunning()}
            fallback={
              <button type="button" class="btn btn--secondary drawer__load-btn" data-press onClick={() => void store.unload(m().id)}>
                <span class="status-dot" data-state="running" aria-hidden="true" />
                Loaded · Unload
              </button>
            }
          >
            <button
              type="button"
              class="btn btn--primary drawer__load-btn"
              data-press
              data-sweep
              disabled={errorCount() > 0}
              aria-busy={busy() ? "true" : undefined}
              aria-describedby={`${id}-caption`}
              onClick={doLoad}
            >
              <Spinner />
              <span class="btn__label">
                <Icon name={loaded() ? "refresh" : "bolt"} size={16} />
                {loaded() ? "Reload" : "Load"}
              </span>
            </button>
          </Show>
          <p class="drawer__caption" id={`${id}-caption`} classList={{ "is-error": errorCount() > 0 }}>
            {caption()}
            <Show when={llm() && tab() !== "load"}>
              {" "}
              <button type="button" class="text-link text-link--btn" onClick={() => setTab("load")}>
                {errorCount() > 0 ? "Show" : "Change"}
              </button>
            </Show>
          </p>
        </div>
        <div class="tabs" role="tablist" aria-label="Model details">
          <For each={tabs()}>
            {(t, i) => (
              <button
                ref={(el) => (tabEls[i()] = el)}
                type="button"
                role="tab"
                class="tab"
                id={`${id}-tab-${t}`}
                aria-selected={tab() === t}
                aria-controls={`${id}-panel-${t}`}
                tabindex={tab() === t ? 0 : -1}
                data-tab={t}
                onClick={() => setTab(t)}
                onKeyDown={(e) => onTabKey(e, i())}
              >
                {TAB_LABELS[t]}
                <Show when={t === "load" && changes() > 0}>
                  <span class="tab__dot" aria-hidden="true" />
                  <span class="sr-only">{`, ${changes()} changed`}</span>
                </Show>
              </button>
            )}
          </For>
        </div>
      </header>
      {/* One panel per tab. A panel's content mounts the first time its tab is opened and then stays
          mounted while hidden, so unsaved work (load draft, tweaks, staged metadata edits) survives
          switching tabs. */}
      <For each={tabs()}>
        {(t) => (
          <div class="drawer__body" id={`${id}-panel-${t}`} role="tabpanel" aria-labelledby={`${id}-tab-${t}`} tabindex="0" hidden={tab() !== t}>
            <Show when={visited().has(t)}>
              <Switch>
                <Match when={t === "overview"}>
                  <Overview model={m()} onReveal={() => void reveal()} />
                </Match>
                <Match when={t === "load"}>
                  <Show when={store.loaded().find((l) => l.id === m().id)?.placement}>{(p) => <PlacementCard plan={p()} title="Where it runs now" />}</Show>
                  <LoadPanel model={m()} draft={draft} setDraft={setDraft} baseline={baseline} effective={effective} tweaks={tweaks} tweakCount={tweakCount} contextMax={ceiling} errors={errors} onTweaks={() => setTab("tweak")} />
                </Match>
                <Match when={t === "tweak"}>
                  <Show when={detail()} fallback={<DetailLoading failed={detail.state === "errored"} />}>
                    {(d) => <TweakStudio model={m()} detail={d()} />}
                  </Show>
                </Match>
                <Match when={t === "metadata"}>
                  <Show when={detail()} fallback={<DetailLoading failed={detail.state === "errored"} />}>
                    {(d) => (
                      <Show when={d().gguf} fallback={<p class="hint">This file has no GGUF metadata to show.</p>}>
                        {(g) => <Metadata model={m()} entries={g()} copy={d().copy} />}
                      </Show>
                    )}
                  </Show>
                </Match>
              </Switch>
            </Show>
          </div>
        )}
      </For>
    </>
  );
}

function DetailLoading(props: { failed: boolean }): JSX.Element {
  return (
    <Show when={!props.failed} fallback={<p class="field__error">The file's details could not be read. Close this and try again.</p>}>
      <div class="detail-skeleton" aria-hidden="true">
        <span class="sk-line sk-line--name" />
        <span class="sk-line" />
        <span class="sk-line sk-line--short" />
      </div>
      <p class="sr-only" role="status">
        Reading the file…
      </p>
    </Show>
  );
}

/**
 * A tune result as docs/CLAIMS.md requires it: three numbers (prompt reading, writing with an empty
 * context, writing at working depth) beside quality, and a quiet line with every condition they
 * were measured under. No speedup multiplier: a ratio without its conditions is not a claim.
 */
function TuneMetrics(props: { result: TuneResult; quant: string; defaultsFastest: boolean }): JSX.Element {
  const r = () => props.result;
  return (
    <>
      <dl class="ov-metrics">
        <div>
          <dt>
            <Term k="prefill">Prompt reading</Term>
          </dt>
          <dd class="mono">
            <span class="count" data-count-to={r().prefillTps.toFixed(0)}>
              {r().prefillTps.toFixed(0)}
            </span>
            <span class="unit">tok/s</span>
          </dd>
        </div>
        <div>
          <dt>
            <Term k="decode">Writing</Term>, empty context
          </dt>
          <dd class="mono">
            <span class="count" data-count-to={r().decodeTps0.toFixed(1)}>
              {r().decodeTps0.toFixed(1)}
            </span>
            <span class="unit">tok/s</span>
          </dd>
        </div>
        <div>
          <dt>{`Writing at ${contextLabel(r().depth)} context`}</dt>
          <dd class="mono ov-metrics__win">
            <span class="count" data-count-to={r().decodeTpsDepth.toFixed(1)}>
              {r().decodeTpsDepth.toFixed(1)}
            </span>
            <span class="unit">tok/s</span>
          </dd>
        </div>
        <div>
          <dt>
            <Term k="quality">Quality</Term>
          </dt>
          <dd class="muted">Not measured yet</dd>
        </div>
      </dl>
      <p class="ov-conditions">
        <span class="ov-conditions__label">Measured on</span>
        <span class="mono">{`${r().gpu} · ${r().build} · ${props.quant} · ${r().depth.toLocaleString("en-US")}-token depth · sampler ${r().sampler} · ${r().measuredAt}`}</span>
      </p>
      <Show when={props.defaultsFastest}>
        <p class="studio-card__desc">Defaults were fastest on this machine, so Cerebro kept them.</p>
      </Show>
    </>
  );
}

function Overview(props: { model: LibraryModel; onReveal: () => void }): JSX.Element {
  const store = useModels();
  const m = () => props.model;
  const llm = () => m().kind === "text" || m().kind === "vision";
  const pub = () => publisherOf(m());
  // Chat with this model: load it first (unless it already is), then open Chat. Guarded so a
  // double press never starts two loads.
  const [chatting, setChatting] = createSignal(false);
  const chat = async () => {
    if (chatting()) return;
    setChatting(true);
    try {
      const ready = store.isLoaded(m().id) || (await store.load(m().id, null));
      if (ready) {
        store.open(null);
        location.hash = hrefFor("chat");
      }
    } finally {
      setChatting(false);
    }
  };
  return (
    <div class="overview">
      <Switch>
        <Match when={m().kind === "vision"}>
          <VisionTry model={m()} />
        </Match>
        <Match when={m().kind === "stt"}>
          <SpeechToTextTry model={m()} />
        </Match>
        <Match when={m().kind === "tts"}>
          <TextToSpeechTry model={m()} />
        </Match>
        <Match when={m().kind === "image"}>
          <p class="notice notice--inline">
            <Icon name="images" size={18} />
            <span>Image models load in their own engine and are used from Image Studio.</span>
          </p>
        </Match>
      </Switch>

      <Show when={llm()}>
        <section class="ov-tune card" aria-labelledby="ov-tune-title">
          <div class="studio-card__head">
            <h3 class="studio-card__title" id="ov-tune-title">
              Last tune
            </h3>
            <IllustrativeFlag />
          </div>
          <Show
            when={m().tuneResult}
            fallback={
              <p class="studio-card__desc">
                <Show when={m().tune === "tuning"} fallback="Not tuned yet. A tune measures the fastest safe settings for this file on this machine.">
                  A tune is measuring this file now.
                </Show>
              </p>
            }
          >
            {(r) => <TuneMetrics result={r()} quant={m().quant} defaultsFastest={m().tune === "defaults-fastest"} />}
          </Show>
        </section>
      </Show>

      <section class="ov-facts" aria-labelledby="ov-facts-title">
        <h3 class="studio-card__title" id="ov-facts-title">
          File
        </h3>
        <dl class="facts">
          <dt>Path</dt>
          <dd class="mono facts__path">{m().path}</dd>
          <dt>Folder</dt>
          <dd>
            {SOURCE_LABELS[m().source]}
            {pub() ? ` · ${pub()}` : ""}
          </dd>
          <dt>Size</dt>
          <dd class="mono">
            <Size gib={m().sizeGiB} />
          </dd>
          <dt>
            <Term k="quant">Quantization</Term>
          </dt>
          <dd class="mono">{m().quant}</dd>
          <dt>Architecture</dt>
          <dd>
            {m().params}, {m().arch.layers} layers
            <Show when={m().arch.experts}>
              {" "}
              · {m().arch.experts} experts, {m().arch.activeExperts} active (<Term k="moe" />)
            </Show>
          </dd>
          <Show when={llm()}>
            <dt>Context</dt>
            <dd class="mono">{m().contextMax.toLocaleString("en-US")} tokens</dd>
          </Show>
          <dt>
            <Term k="sha256" />
          </dt>
          <dd class="mono facts__hash">{m().sha256}</dd>
        </dl>
      </section>

      <div class="ov-actions">
        <Show when={llm()}>
          <a class="btn btn--secondary" href={hrefWithModel("tune", m().id)} data-press>
            <Icon name="tune" size={16} />
            Tune
          </a>
          <button type="button" class="btn btn--secondary" data-press aria-busy={chatting() ? "true" : undefined} onClick={() => void chat()}>
            <Spinner />
            <span class="btn__label">
              <Icon name="chat" size={16} />
              Chat
            </span>
          </button>
        </Show>
        <button type="button" class="btn btn--secondary" data-press onClick={() => props.onReveal()}>
          <Icon name="folder" size={16} />
          Reveal in folder
        </button>
      </div>
      <Show when={llm()}>
        <CodeBlock lines={[`cerebro tune ${m().id}`, `cerebro serve ${m().id}`]} label="Copy commands" />
      </Show>
    </div>
  );
}
