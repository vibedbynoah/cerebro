import "../styles/onboarding.css";
import { createEffect, createMemo, createResource, createSignal, For, Match, on, onCleanup, onMount, Show, Switch, untrack, type JSX } from "solid-js";
import { prefersReducedMotion } from "@design/motion";
import { Icon } from "../components/icons";
import { TitleBar } from "../components/TitleBar";
import { LegalDialog } from "../legal/LegalDialog";
import type { LegalDocId } from "../legal/types";
import type { Platform } from "../platform/platform";
import { useService } from "../service/context";
import { firstStep, STEPS, TERMS_STEP, TERMS_VERSION, termsAccepted, writeRecord, type OnboardingRecord, type StepId } from "./progress";
import type { CatalogModel, HardwareProfile } from "../service/types";
import { rankModels, unranked } from "./recommend";
import { FinishStep, HardwareStep, ModelStep, PrivacyStep, TermsStep, WelcomeStep, type ScanState } from "./Steps";
import { ConfigureStep, LevelStep } from "./ExperienceSteps";
import { eyebrowFor } from "./StepHeader";
import { autoConfig, changedFrom, readConfig, runnableQuants, sanitize, writeConfig, type SetupConfig } from "./autoconfig";
import { useAppState } from "../state";
import { LEVEL_LABELS, otherLevel, type Channel, type ExperienceLevel } from "../lib/experience";

/** Rows the hardware step reveals, in order. */
export const SCAN_ROWS = 4;
/** Time each row spends "reading" before its value lands. Presentation only: the mock answers at once. */
const SCAN_STEP_MS = 520;

/**
 * First-run setup: eight short steps in one keyboard-driven flow.
 *
 * Keys (ignored while focus is on a control, which keeps its own keys, or a dialog is open):
 *   Enter or Right: continue.  Left: back.  Escape: skip to the terms (the one step setup keeps).
 * Accepting the terms is only ever a deliberate press of its button, never a stray Enter.
 *
 * The experience level (Beginner or Advanced) is chosen on step 2 and can be switched on every step
 * from the footer, without leaving the step: both views show the same SetupConfig, so Beginner's
 * automatic values become Advanced's starting values and Advanced edits survive a switch back.
 *
 * Nothing here blocks the app for good: Skip goes straight to the terms, and a finished setup is
 * remembered (progress.ts) so it never shows again unless the terms change or the user reruns it.
 */
export function Onboarding(props: { platform: Platform; record: OnboardingRecord | null; onComplete: (record: OnboardingRecord) => void }): JSX.Element {
  const service = useService();
  const app = useAppState();
  const termsChanged = props.record?.completed === true && !termsAccepted(props.record);

  const [step, setStep] = createSignal(firstStep(props.record));
  const [dir, setDir] = createSignal<"forward" | "back">("forward");
  const [reached, setReached] = createSignal(step());
  const [accepted, setAccepted] = createSignal(termsAccepted(props.record));
  const [legal, setLegal] = createSignal<LegalDocId | null>(null);
  const [model, setModel] = createSignal<string | null>(props.record?.model ?? null);
  const [announce, setAnnounce] = createSignal("");

  const stepId = (): StepId => STEPS[step()]?.id ?? "welcome";

  // ---- Hardware scan -------------------------------------------------------------------------
  const [hw, { refetch }] = createResource(() => service.detectHardware());
  const [catalog] = createResource(() => service.listCatalog());
  // Reading an errored resource throws (solid-js), and so does `.latest`. Every read goes through
  // these, which only touch the value in a state that holds one: an error is a state, not a throw.
  const hasValue = (state: string) => state === "ready" || state === "refreshing";
  const hwData = (): HardwareProfile | undefined => (hasValue(hw.state) ? hw() : undefined);
  const catalogData = (): CatalogModel[] | undefined => (hasValue(catalog.state) ? catalog() : undefined);
  const hwUnknown = () => hw.state === "errored";
  const catalogError = () => catalog.state === "errored";
  const recs = createMemo(() => {
    const c = catalogData();
    if (!c) return [];
    const h = hwData();
    // Hardware unknown: the whole catalog, unranked, to pick from by hand (Tune measures it later).
    return h ? rankModels(c, h) : hwUnknown() ? unranked(c) : [];
  });
  /** Ids a user may pick: everything that would load here (or might, when unranked). */
  const selectable = createMemo(() => recs().filter((r) => r.fit !== "too-large").map((r) => r.model.id));
  const modelsLoading = () => hw.loading || catalog.loading;

  const [scanned, setScanned] = createSignal(0);
  const [scanStarted, setScanStarted] = createSignal(false);
  let scanTimer: ReturnType<typeof setTimeout> | undefined;
  const stopScan = () => clearTimeout(scanTimer);
  onCleanup(stopScan);

  const runScan = () => {
    stopScan();
    setScanStarted(true);
    if (prefersReducedMotion()) {
      setScanned(SCAN_ROWS);
      return;
    }
    setScanned(0);
    const tick = () => {
      setScanned((n) => n + 1);
      if (scanned() < SCAN_ROWS) scanTimer = setTimeout(tick, SCAN_STEP_MS);
    };
    scanTimer = setTimeout(tick, SCAN_STEP_MS);
  };

  // The scan plays the first time the hardware step is on screen with data in hand.
  createEffect(() => {
    if (stepId() === "hardware" && hw.state === "ready" && !scanStarted()) runScan();
  });

  const scanState = (): ScanState => {
    if (hw.state === "errored") return "error";
    if (!scanStarted() || hw.state !== "ready") return "idle";
    return scanned() >= SCAN_ROWS ? "done" : "scanning";
  };

  createEffect(on(scanState, (s) => s === "done" && setAnnounce("Hardware check complete. Four items found."), { defer: true }));

  const rescan = () => {
    stopScan();
    setScanStarted(false);
    setScanned(0);
    void refetch();
  };

  // On the model step the best loadable model is preselected the first time (null: nothing chosen
  // yet), and a stored id that no longer loads here (hardware or catalog changed since) is replaced
  // by the best fit instead of leaving no radio checked. "" is an explicit "decide later" and stays.
  // Unranked (hardware unknown) there is no best fit, so nothing is preselected: the user picks.
  createEffect(() => {
    if (stepId() !== "model" || modelsLoading() || catalogError()) return;
    const ids = selectable();
    const cur = model();
    if (cur === "" || (cur !== null && ids.includes(cur))) return;
    setModel(hwUnknown() ? null : (ids[0] ?? ""));
  });

  // ---- Load settings for the chosen model -------------------------------------------------------
  const chosenModel = createMemo(() => {
    const id = model();
    return id ? catalogData()?.find((m) => m.id === id) : undefined;
  });
  /** The automatic plan for the chosen model on this machine (null without a model). */
  const auto = createMemo(() => {
    const m = chosenModel();
    return m ? autoConfig(m, hwData(), app.channel()) : null;
  });
  // A plan saved by an earlier setup run is the starting point when it is for the same model.
  const stored = readConfig();
  const [config, setConfig] = createSignal<SetupConfig | null>(null);
  // The plan follows the model: a new model gets its automatic plan (the stored one when it matches).
  // Edits stay while the model stays.
  createEffect(
    on(chosenModel, (m) => {
      if (!m) {
        setConfig(null);
        return;
      }
      if (untrack(config)?.modelId === m.id) return;
      setConfig(stored?.modelId === m.id ? sanitize(m, stored) : untrack(auto));
    }),
  );
  // The hardware answer can arrive after the model is known (a rescan): an untouched plan follows it.
  createEffect(
    on(
      auto,
      (next, prev) => {
        const cur = untrack(config);
        if (next && prev && cur && cur.modelId === next.modelId && changedFrom(prev, cur).length === 0) setConfig(next);
      },
      { defer: true },
    ),
  );
  // A channel change can take away the chosen file (GSQ-RCO needs Experimental): fall back to the
  // automatic file for the new channel, keeping every other value.
  createEffect(
    on(
      () => app.channel(),
      (ch) => {
        const m = chosenModel();
        const cur = untrack(config);
        if (!m || !cur) return;
        if (runnableQuants(m, ch).some((q) => q.id === cur.quantId)) return;
        const fresh = autoConfig(m, hwData(), ch);
        if (fresh) setConfig({ ...cur, quantId: fresh.quantId });
      },
      { defer: true },
    ),
  );
  const editConfig = (next: SetupConfig) => {
    const m = chosenModel();
    if (m) setConfig(sanitize(m, next));
  };
  const resetConfig = () => {
    setConfig(auto());
    setAnnounce("Load settings back to the automatic plan.");
  };

  // ---- Experience level: the escape hatch ----------------------------------------------------
  const switchLevel = (to: ExperienceLevel) => {
    if (to === app.level()) return;
    app.setLevel(to);
    setAnnounce(`${LEVEL_LABELS[to]} view. Every value is kept.`);
  };
  const setChannel = (c: Channel) => app.setChannel(c);

  // ---- Navigation ----------------------------------------------------------------------------
  const go = (to: number) => {
    if (to < 0 || to >= STEPS.length || to === step()) return;
    // Past the terms only once they are accepted.
    if (to > TERMS_STEP && !accepted()) return;
    setDir(to > step() ? "forward" : "back");
    setStep(to);
    setReached((r) => Math.max(r, to));
  };

  const complete = () => {
    if (!accepted()) return;
    // Only an id that loads here is remembered: "decide later" or an id the catalog shows no longer
    // loads is null. Until the catalog has been read (still loading, or unavailable) a stored choice
    // cannot be checked, so it is kept rather than wiped by a quick Skip.
    const chosen = model();
    const checkable = catalogData() !== undefined && !modelsLoading();
    const keep = chosen !== null && chosen !== "" && (!checkable || selectable().includes(chosen));
    const record: OnboardingRecord = { v: 1, completed: true, termsVersion: TERMS_VERSION, model: keep ? chosen : null };
    writeRecord(record);
    // The load plan is remembered with the model it belongs to; no model, no plan.
    const cfg = config();
    if (keep && cfg && cfg.modelId === chosen) writeConfig(cfg);
    props.onComplete(record);
  };

  const next = () => {
    const id = stepId();
    if (id === "finish") complete();
    else if (id === "terms") {
      if (accepted()) go(step() + 1);
    } else go(step() + 1);
  };

  const accept = () => {
    setAccepted(true);
    go(TERMS_STEP + 1);
  };

  const skip = () => {
    if (accepted()) {
      complete();
      return;
    }
    go(TERMS_STEP);
    setAnnounce("Setup skipped. One step left: the terms.");
  };

  // New step: back to the top and focus its heading. The heading is read on focus with "Step N of 6"
  // as its description (aria-describedby), so nothing announces the same step a second time.
  // Skipping keeps its own announcement: that change of context deserves the words.
  let scroller: HTMLDivElement | undefined;
  createEffect(
    on(
      step,
      (s) => {
        document.title = `${STEPS[s]?.label ?? "Setup"} · Set up Cerebro`;
        scroller?.scrollTo({ top: 0 });
        requestAnimationFrame(() => scroller?.querySelector<HTMLElement>(".onb-title")?.focus({ preventScroll: true }));
      },
      { defer: true },
    ),
  );

  onMount(() => {
    document.title = `${STEPS[step()]?.label ?? "Setup"} · Set up Cerebro`;
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey || legal() !== null) return;
      // An open definition popover owns Escape (it closes itself).
      if (document.querySelector(":popover-open")) return;
      const target = e.target instanceof Element ? e.target : null;
      const onControl = target?.closest("button, a[href], input, select, textarea, summary, [role='tab'], [contenteditable='true']") != null;
      if (e.key === "Escape") {
        if (step() < TERMS_STEP) {
          e.preventDefault();
          skip();
        }
        return;
      }
      // Enter on a model radio continues, as Enter does in a native form; the radio group keeps its
      // own arrows and Space.
      const enterOnRadio = e.key === "Enter" && target instanceof HTMLInputElement && target.type === "radio";
      if (onControl && !enterOnRadio) return;
      if (e.key === "Enter" || e.key === "ArrowRight") {
        e.preventDefault();
        next();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        go(step() - 1);
      }
    };
    window.addEventListener("keydown", onKey);
    onCleanup(() => window.removeEventListener("keydown", onKey));
  });

  const primaryLabel = () => {
    switch (stepId()) {
      case "welcome":
        return "Get started";
      case "terms":
        return accepted() ? "Continue" : "Accept and continue";
      case "finish":
        return "Open Cerebro";
      default:
        return "Continue";
    }
  };

  const onPrimary = () => (stepId() === "terms" && !accepted() ? accept() : next());

  return (
    <div class="onb" data-step={stepId()}>
      <TitleBar platform={props.platform} />
      <div class="onb__body">
        <nav class="onb-rail" aria-label="Setup steps">
          <p class="onb-rail__title">Set up Cerebro</p>
          <ol class="onb-rail__list">
            <For each={STEPS}>
              {(s, i) => {
                const state = () => (i() === step() ? "current" : i() < step() || i() <= reached() ? "done" : "todo");
                // Earlier steps and ones already seen are links back; the terms gate the rest.
                const reachable = () => i() !== step() && i() <= reached() && (i() <= TERMS_STEP || accepted());
                // The check mark is decoration; assistive tech gets the state in words.
                const said = () => (state() === "done" ? ", completed" : state() === "current" ? ", current step" : "");
                return (
                  <li class="onb-rail__item" data-state={state()}>
                    <Show
                      when={reachable()}
                      fallback={
                        <span class="onb-rail__step" aria-current={i() === step() ? "step" : undefined}>
                          <RailMark index={i()} done={state() === "done"} />
                          <span class="onb-rail__label">
                            {s.label}
                            <span class="sr-only">{said()}</span>
                          </span>
                        </span>
                      }
                    >
                      <button type="button" class="onb-rail__step onb-rail__step--link" data-press onClick={() => go(i())}>
                        <RailMark index={i()} done={state() === "done"} />
                        <span class="onb-rail__label">
                          {s.label}
                          <span class="sr-only">{said()}</span>
                        </span>
                      </button>
                    </Show>
                  </li>
                );
              }}
            </For>
          </ol>
          <div class="onb-rail__foot">
            <p class="onb-rail__offline">
              <Icon name="offline" size={14} />
              Works offline. Setup needs no network.
            </p>
            <dl class="onb-keys" aria-label="Keyboard shortcuts">
              <div>
                <dt>
                  <kbd class="kbd">Enter</kbd>
                </dt>
                <dd>Continue</dd>
              </div>
              <div>
                <dt>
                  <kbd class="kbd" aria-label="Left arrow">
                    ←
                  </kbd>
                </dt>
                <dd>Back</dd>
              </div>
              <Show when={step() < TERMS_STEP}>
                <div>
                  <dt>
                    <kbd class="kbd">Esc</kbd>
                  </dt>
                  <dd>Skip setup</dd>
                </div>
              </Show>
            </dl>
          </div>
        </nav>

        <main class="onb-stage" id="main" tabindex="-1" aria-labelledby="onb-step-title">
          <div class="onb-scroll" ref={scroller}>
            <div class="backdrop-grid onb-scroll__grid" aria-hidden="true" />
            <Switch>
              <Match when={stepId() === "welcome"}>
                <Frame dir={dir()}>
                  <WelcomeStep />
                </Frame>
              </Match>
              <Match when={stepId() === "level"}>
                <Frame dir={dir()}>
                  <LevelStep eyebrow={eyebrowFor("level")} level={app.level()} onSelect={switchLevel} />
                </Frame>
              </Match>
              <Match when={stepId() === "hardware"}>
                <Frame dir={dir()}>
                  <HardwareStep hw={hwData()} state={scanState()} scanned={scanned()} onRescan={rescan} />
                </Frame>
              </Match>
              <Match when={stepId() === "model"}>
                <Frame dir={dir()}>
                  <ModelStep
                    recs={recs()}
                    loading={modelsLoading()}
                    hwUnknown={hwUnknown()}
                    catalogError={catalogError()}
                    selected={model()}
                    onSelect={setModel}
                    hw={hwData()}
                  />
                </Frame>
              </Match>
              <Match when={stepId() === "configure"}>
                <Frame dir={dir()}>
                  <ConfigureStep
                    eyebrow={eyebrowFor("configure")}
                    level={app.level()}
                    model={chosenModel()}
                    hw={hwData()}
                    config={config()}
                    auto={auto()}
                    channel={app.channel()}
                    onChange={editConfig}
                    onReset={resetConfig}
                    onChannel={setChannel}
                  />
                </Frame>
              </Match>
              <Match when={stepId() === "privacy"}>
                <Frame dir={dir()}>
                  <PrivacyStep />
                </Frame>
              </Match>
              <Match when={stepId() === "terms"}>
                <Frame dir={dir()}>
                  <TermsStep accepted={accepted()} changed={termsChanged} onOpen={setLegal} />
                </Frame>
              </Match>
              <Match when={stepId() === "finish"}>
                <Frame dir={dir()}>
                  <FinishStep hw={hwData()} recs={recs()} model={model()} level={app.level()} config={config()} onOpen={setLegal} />
                </Frame>
              </Match>
            </Switch>
          </div>

          <footer class="onb-foot">
            <div class="onb-foot__start">
              <Show when={step() < TERMS_STEP}>
                <button type="button" class="btn btn--tertiary onb-foot__skip" data-press aria-keyshortcuts="Escape" onClick={skip}>
                  Skip setup
                </button>
              </Show>
              {/* The escape hatch: on every step, switches the view in place. */}
              <button
                type="button"
                class="btn btn--tertiary onb-level-switch"
                data-press
                data-level={app.level()}
                aria-describedby="onb-level-now"
                onClick={() => switchLevel(otherLevel(app.level()))}
              >
                <Icon name={app.level() === "beginner" ? "sliders" : "bolt"} size={16} />
                Switch to {LEVEL_LABELS[otherLevel(app.level())]}
              </button>
              <span class="sr-only" id="onb-level-now">
                {`Now ${LEVEL_LABELS[app.level()]}. Nothing you chose is lost.`}
              </span>
            </div>
            <span class="onb-progress" data-at={step()} aria-hidden="true">
              <span class="onb-progress__fill" />
            </span>
            <p class="onb-foot__count" id="onb-step-count">
              Step {step() + 1} of {STEPS.length}
            </p>
            <div class="onb-foot__end">
              <Show when={step() > 0}>
                <button type="button" class="btn btn--secondary" data-press aria-keyshortcuts="ArrowLeft" onClick={() => go(step() - 1)}>
                  <Icon name="arrowLeft" size={16} />
                  Back
                </button>
              </Show>
              <button
                type="button"
                class="btn btn--primary onb-foot__next"
                classList={{ "btn--hero": stepId() === "welcome" || stepId() === "finish" }}
                data-press
                data-sweep
                data-primary
                aria-keyshortcuts={stepId() === "terms" && !accepted() ? undefined : "Enter"}
                onClick={onPrimary}
              >
                {primaryLabel()}
                <Show when={stepId() !== "terms" || accepted()} fallback={<Icon name="check" size={16} />}>
                  <span class="btn__arrow" aria-hidden="true">
                    <Icon name="arrowRight" size={16} />
                  </span>
                </Show>
              </button>
            </div>
          </footer>
        </main>
      </div>
      <p class="sr-only" role="status" aria-live="polite">
        {announce()}
      </p>
      <LegalDialog doc={legal()} onSelect={setLegal} onClose={() => setLegal(null)} />
    </div>
  );
}

/** One step's frame: mounts fresh per step, so its entry motion (direction-aware) plays each time. */
function Frame(props: { dir: "forward" | "back"; children: JSX.Element }): JSX.Element {
  return (
    <div class="onb-step" data-dir={props.dir}>
      {props.children}
    </div>
  );
}

function RailMark(props: { index: number; done: boolean }): JSX.Element {
  return (
    <span class="onb-rail__mark" aria-hidden="true">
      <Show when={props.done} fallback={<span class="onb-rail__num">{props.index + 1}</span>}>
        <Icon name="check" size={14} />
      </Show>
    </span>
  );
}
