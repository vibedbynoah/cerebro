import { createMemo, For, Index, Show, type JSX } from "solid-js";
import { CodeBlock } from "../components/CodeBlock";
import { Icon, type IconName } from "../components/icons";
import { IllustrativeFlag } from "../components/Screen";
import { Term } from "../components/Term";
import type { LegalDocId } from "../legal/types";
import type { HardwareProfile, SettingKey } from "../service/types";
import { useAppState } from "../state";
import { BrandHero } from "./BrandHero";
import { TERMS_VERSION } from "./progress";
import { eyebrowFor, StepHeader } from "./StepHeader";
import { contextLabel, type SetupConfig } from "./autoconfig";
import { LEVEL_LABELS, type ExperienceLevel } from "../lib/experience";
import { pct, type Recommendation } from "./recommend";

/**
 * The six setup steps. Each is one screen with one h1 (focused on arrival, id onb-step-title), a
 * lede, and its content. Every number shown comes from the service and carries its provenance flag
 * (docs/CLAIMS.md); every technical term is defined in place with <Term>.
 */

const gib = (n: number): string => (Math.round(n * 10) / 10).toFixed(1);

// ---- 1. Welcome ---------------------------------------------------------------------------------

export function WelcomeStep(): JSX.Element {
  const plan: Array<{ icon: IconName; title: string; text: string }> = [
    { icon: "chip", title: "Check this machine", text: "Processor, graphics card and memory." },
    { icon: "models", title: "Pick a first model", text: "Ranked by how well it fits here." },
    { icon: "security", title: "Review what protects you", text: "All on by default, all yours to change." },
  ];
  return (
    <div class="onb-welcome">
      <div class="onb-welcome__text">
        <p class="eyebrow">
          <Icon name="bolt" size={14} />
          Welcome to Cerebro
        </p>
        <h1 class="onb-title onb-title--display" id="onb-step-title" tabindex="-1" aria-describedby="onb-step-count">
          Your models, measured on this machine.
        </h1>
        <p class="onb-lede">
          Cerebro measures the fastest safe configuration for each model on this computer instead of guessing, runs it with its protections on, and serves it on{" "}
          <Term k="loopback" /> only.
        </p>
        <ol class="onb-plan">
          <For each={plan}>
            {(p, i) => (
              <li class="onb-plan__item">
                <span class="onb-plan__num mono" aria-hidden="true">
                  0{i() + 1}
                </span>
                <span class="onb-plan__icon" aria-hidden="true">
                  <Icon name={p.icon} size={18} />
                </span>
                <span class="onb-plan__text">
                  <span class="onb-plan__title">{p.title}</span>
                  <span class="onb-plan__desc">{p.text}</span>
                </span>
              </li>
            )}
          </For>
        </ol>
      </div>
      <figure class="onb-welcome__art">
        <BrandHero />
        <figcaption class="brand-hero__caption">
          Cerebro tries configurations on this machine and keeps the fastest one that passes its checks. Sometimes that is the defaults, and it says so.
        </figcaption>
      </figure>
    </div>
  );
}

// ---- 2. Hardware --------------------------------------------------------------------------------

export type ScanState = "idle" | "scanning" | "done" | "error";

interface ScanRow {
  id: string;
  icon: IconName;
  label: JSX.Element;
  value: string;
  unit?: string;
  meta: string;
}

export function HardwareStep(props: { hw: HardwareProfile | undefined; state: ScanState; scanned: number; onRescan: () => void }): JSX.Element {
  const rows = createMemo<ScanRow[]>(() => {
    const h = props.hw;
    if (!h) return [];
    return [
      { id: "cpu", icon: "chip", label: <>Processor <Term k="cpu" /></>, value: h.cpu.name, meta: `${h.cpu.cores} cores, ${h.cpu.threads} threads` },
      {
        id: "gpu",
        icon: "gpu",
        label: <>Graphics card <Term k="gpu" /></>,
        value: h.gpu?.name ?? "None found",
        meta: h.gpu ? "Runs the model's heavy maths" : "Models will run on the processor",
      },
      { id: "vram", icon: "bolt", label: <>Video memory <Term k="vram" /></>, value: gib(h.gpu?.vramTotalGiB ?? 0), unit: "GiB", meta: "The fastest place for a model" },
      { id: "ram", icon: "memory", label: <>System memory <Term k="ram" /></>, value: gib(h.ramTotalGiB), unit: "GiB", meta: "Holds what does not fit" },
    ];
  });
  const rowState = (i: number) => (props.state === "done" || i < props.scanned ? "done" : props.state === "scanning" && i === props.scanned ? "scanning" : "waiting");
  const vramPct = () => {
    const h = props.hw;
    const vram = h?.gpu?.vramTotalGiB ?? 0;
    const total = vram + (h?.ramTotalGiB ?? 0);
    return total > 0 ? pct(vram / total) : 0;
  };
  const status = () =>
    props.state === "error"
      ? "Could not read this machine"
      : props.state === "done"
        ? "Found all four"
        : props.state === "scanning"
          ? `Reading ${Math.min(props.scanned + 1, 4)} of 4`
          : "Starting";

  return (
    <div class="onb-page">
      <StepHeader
        eyebrow={eyebrowFor("hardware")}
        icon="chip"
        title="First, what this computer has."
        lede="Cerebro sizes models from four readings. They stay on this computer."
      />
      <section class="scan-panel card" aria-labelledby="scan-title" data-state={props.state}>
        <div class="scan-panel__head">
          <h2 class="scan-panel__title" id="scan-title">
            Hardware
          </h2>
          <span class="scan-panel__status mono" data-state={props.state}>
            <span class="scan-panel__pulse" aria-hidden="true" />
            {status()}
          </span>
        </div>
        <Show
          when={props.state !== "error"}
          fallback={
            <p class="scan-panel__error" role="alert">
              <Icon name="alert" size={16} />
              The hardware check did not finish. Scan again, or continue and pick a model by hand: Tune measures it on this machine after download.
            </p>
          }
        >
          <ol class="scan">
            <Index each={rows()}>
              {(row, i) => (
                <li class="scan__row" data-state={rowState(i)} data-row={row().id}>
                  <span class="scan__icon" aria-hidden="true">
                    <Icon name={row().icon} size={18} />
                  </span>
                  <span class="scan__label">{row().label}</span>
                  <span class="scan__value">
                    <Show
                      when={rowState(i) === "done"}
                      fallback={
                        <span class="scan__pending" aria-hidden={rowState(i) === "waiting"}>
                          {rowState(i) === "scanning" ? "Reading…" : "Waiting"}
                        </span>
                      }
                    >
                      <span class="scan__reading">
                        <span class="scan__number mono">
                          {/* Numbers count up as they land (design-system count-up: one shot, the
                              final value is in the markup and read by assistive tech). */}
                          <Show when={row().unit ? row().value : undefined} fallback={row().value} keyed>
                            {(v) => (
                              <span data-count-to={v} data-count-duration="600">
                                {v}
                              </span>
                            )}
                          </Show>
                          <Show when={row().unit}>
                            <span class="unit">{row().unit}</span>
                          </Show>
                        </span>
                        <span class="scan__meta">{row().meta}</span>
                      </span>
                    </Show>
                  </span>
                  <span class="scan__check" aria-hidden="true">
                    <Icon name="check" size={14} />
                  </span>
                  <span class="scan__beam" aria-hidden="true" />
                </li>
              )}
            </Index>
          </ol>
        </Show>
        <div class="scan-panel__foot">
          <Show when={props.hw?.provenance === "illustrative"} fallback={<span />}>
            <IllustrativeFlag />
          </Show>
          {/* After a failed check, Scan again is the way forward, so it carries more weight. */}
          <button
            type="button"
            class="btn scan-panel__rescan"
            classList={{ "btn--secondary": props.state === "error", "btn--tertiary": props.state !== "error" }}
            data-press
            disabled={props.state === "scanning"}
            onClick={() => props.onRescan()}
          >
            <Icon name="refresh" size={14} />
            Scan again
          </button>
        </div>
      </section>

      <Show when={props.state === "done"}>
        <figure class="memmap">
          <figcaption class="memmap__caption">Where a model can live on this machine</figcaption>
          <div class="memmap__track" aria-hidden="true">
            <span class={`memmap__vram bar--p${vramPct()}`} />
            <span class="memmap__ram" />
          </div>
          <div class="memmap__legend">
            <span class="memmap__key">
              <span class="memmap__swatch memmap__swatch--vram" aria-hidden="true" />
              Video memory, <span class="mono">{gib(props.hw?.gpu?.vramTotalGiB ?? 0)}</span>
              <span class="unit">GiB</span>: fastest
            </span>
            <span class="memmap__key">
              <span class="memmap__swatch memmap__swatch--ram" aria-hidden="true" />
              System memory, <span class="mono">{gib(props.hw?.ramTotalGiB ?? 0)}</span>
              <span class="unit">GiB</span>: larger, slower
            </span>
          </div>
          <p class="memmap__note">A model that fits in video memory runs fastest. A larger one can spill into system memory, and Cerebro measures what that costs.</p>
        </figure>
      </Show>
    </div>
  );
}

// ---- 3. Model -----------------------------------------------------------------------------------

export function ModelStep(props: {
  recs: readonly Recommendation[];
  loading: boolean;
  /** The hardware check failed: the catalog is listed unranked, to pick from by hand. */
  hwUnknown: boolean;
  /** The catalog itself could not be read. */
  catalogError: boolean;
  /** A catalog id, "" for "decide later", or null before anything is chosen. */
  selected: string | null;
  onSelect: (id: string) => void;
  hw: HardwareProfile | undefined;
}): JSX.Element {
  // Models that would not load are not recommendations: they are named once, below the list.
  const loadable = () => props.recs.filter((r) => r.fit !== "too-large");
  const tooLarge = () => props.recs.filter((r) => r.fit === "too-large");
  const ranked = () => !props.hwUnknown;
  return (
    <div class="onb-page onb-page--dense">
      <StepHeader
        eyebrow={eyebrowFor("model")}
        icon="models"
        title={ranked() ? "Models that suit this machine." : "Pick a first model."}
        lede={
          ranked()
            ? "Ranked by how each fits this computer's memory. Nothing downloads now. Models you already have in LM Studio or Ollama appear in Models as they are."
            : "Choose any model below, or decide later. Nothing downloads now. Models you already have in LM Studio or Ollama appear in Models as they are."
        }
      />
      <Show when={!props.loading} fallback={<p class="hint">Reading the catalog…</p>}>
        <Show
          when={!props.catalogError}
          fallback={
            <p class="notice rec-notice" role="note">
              <Icon name="info" size={18} />
              <span>The model list could not be read. Setup can finish without a model: add a GGUF file or pick one later in Model Hub.</span>
            </p>
          }
        >
          <div class="rec-legend">
            <Show
              when={ranked()}
              fallback={
                <span class="rec-legend__item">
                  <Icon name="info" size={14} />
                  This machine could not be read, so these are listed unranked. Tune measures whichever you pick after download.
                </span>
              }
            >
              <span class="rec-legend__item">
                <span class="rec-legend__swatch" aria-hidden="true">
                  <span class="rec-legend__tick" />
                </span>
                Ranked by fit, not speed. Bar: file size against spare memory; the tick marks where video memory ends.
              </span>
            </Show>
            <Show when={props.recs.some((r) => r.model.provenance === "illustrative") || props.hw?.provenance === "illustrative"}>
              <IllustrativeFlag />
            </Show>
          </div>
          <fieldset class="rec-list">
            <legend class="sr-only">Choose a first model</legend>
            <For each={loadable()}>
              {(r) => {
                const inputId = `rec-${r.model.id}`;
                const best = () => r.ranked && r.rank === 1;
                return (
                  <label class="rec" data-fit={r.fit} data-rank={best() ? "1" : undefined} classList={{ "is-selected": props.selected === r.model.id }} for={inputId}>
                    <input
                      id={inputId}
                      class="rec__input"
                      type="radio"
                      name="first-model"
                      value={r.model.id}
                      checked={props.selected === r.model.id}
                      aria-labelledby={best() ? `${inputId}-name ${inputId}-best ${inputId}-fit` : `${inputId}-name ${inputId}-fit`}
                      aria-describedby={`${inputId}-why ${inputId}-spec`}
                      onChange={() => props.onSelect(r.model.id)}
                    />
                    <span class="rec__main">
                      <span class="rec__head">
                        <span class="rec__name" id={`${inputId}-name`}>
                          {r.model.name}
                        </span>
                        <Show when={best()}>
                          <span class="rec__best" id={`${inputId}-best`}>
                            Best fit
                          </span>
                        </Show>
                      </span>
                      <span class="rec__use">{r.model.goodFor}</span>
                      <span class="rec__why" id={`${inputId}-why`}>
                        {r.why}
                      </span>
                      <Show when={r.ranked}>
                        <span class="rec__track" aria-hidden="true">
                          <span class={`bar rec__fill bar--p${pct(r.share)}`} data-bar data-bar-to={String(r.share)} data-bar-delay={String((r.rank - 1) * 60)} />
                          <span class={`rec__tick bar--p${pct(r.vramShare)}`} />
                        </span>
                      </Show>
                      <span class="rec__spec mono" id={`${inputId}-spec`}>
                        {`${r.model.params} · ${r.model.quant} · ${gib(r.model.sizeGiB)} GiB`}
                      </span>
                    </span>
                    <span class="chip rec__fit" id={`${inputId}-fit`} data-tone={r.fit === "gpu" ? "on" : "neutral"}>
                      {r.fitLabel}
                    </span>
                    <span class="rec__radio" aria-hidden="true" />
                  </label>
                );
              }}
            </For>
            <label class="rec rec--later" classList={{ "is-selected": props.selected === "" }} for="rec-later">
              <input
                id="rec-later"
                class="rec__input"
                type="radio"
                name="first-model"
                value=""
                checked={props.selected === ""}
                aria-labelledby="rec-later-name"
                aria-describedby="rec-later-why"
                onChange={() => props.onSelect("")}
              />
              <span class="rec__main">
                <span class="rec__head">
                  <span class="rec__name" id="rec-later-name">
                    Decide later
                  </span>
                </span>
                <span class="rec__why" id="rec-later-why">
                  {loadable().length > 0 ? "Finish setup without a model and choose one in Model Hub." : "No model in the catalog fits this machine yet. Add your own GGUF file later in Model Hub."}
                </span>
              </span>
              <span class="rec__radio" aria-hidden="true" />
            </label>
          </fieldset>
          <Show when={tooLarge().length > 0}>
            <p class="rec-skip">
              <Icon name="info" size={14} />
              <span>
                Also checked, too large to load here:{" "}
                <For each={tooLarge()}>
                  {(r, i) => (
                    <>
                      {i() > 0 ? ", " : ""}
                      <span class="rec-skip__name">{r.model.name}</span> (<span class="mono">{gib(r.model.sizeGiB)}</span>
                      <span class="unit">GiB</span>)
                    </>
                  )}
                </For>
                .
              </span>
            </p>
          </Show>
        </Show>
      </Show>
    </div>
  );
}

// ---- 4. Privacy and protection ------------------------------------------------------------------

export function PrivacyStep(): JSX.Element {
  const { settings } = useAppState();
  const on = (...keys: SettingKey[]) => {
    const s = settings();
    return s ? keys.every((k) => s[k] === true) : null;
  };
  const cards: Array<{ icon: IconName; title: string; body: () => JSX.Element; keys: SettingKey[] | null }> = [
    {
      icon: "offline",
      title: "Works without a network",
      body: () => <>Chat, tuning, document search and the API run on this computer. Only model downloads and plugin installs use a network, and both are optional.</>,
      keys: null,
    },
    {
      icon: "security",
      title: "A contained model process",
      body: () => (
        <>
          The model runs in a <Term k="jobObject" /> with a <Term k="cleanEnv" />, so it and anything it starts end when Cerebro stops. It does not limit which files or network addresses
          the process can open.
        </>
      ),
      keys: ["processContainment", "cleanEnvironment"],
    },
    {
      icon: "lock",
      title: "Served on loopback only",
      body: () => (
        <>
          The <Term k="api" /> listens on 127.0.0.1, which other devices cannot reach. Programs on this computer still can.
        </>
      ),
      keys: ["loopbackOnly"],
    },
    {
      icon: "check",
      title: "Checked before every start",
      body: () => (
        <>
          An <Term k="integrity" /> fingerprints the backend's program files and stops the start if any changed, went missing or appeared.
        </>
      ),
      keys: ["integrityCheck"],
    },
  ];
  // What is true of the desktop build: the engine runs inside Cerebro (no separate process or API to
  // contain), tools ask first, and phone access stays off until it is set up.
  const desktopCards: typeof cards = [
    {
      icon: "offline",
      title: "Works without a network",
      body: () => <>Chat, the Monitor, document search and every engine switch run on this computer. Only Model Hub downloads use a network, and you can turn them off.</>,
      keys: null,
    },
    {
      icon: "lock",
      title: "Nothing leaves this computer",
      body: () => <>The engine runs inside Cerebro. No account, no telemetry, no cloud model behind it.</>,
      keys: null,
    },
    {
      icon: "check",
      title: "Tools ask first",
      body: () => <>Before a model uses a tool, such as reading a file through an MCP server, Cerebro shows what it will do and waits for you.</>,
      keys: ["permissionPrompts"],
    },
    {
      icon: "monitor",
      title: "Phone access is off",
      body: () => <>Checking on the model from your phone needs a password and a QR code, and only works on your own network. It stays off until you set it up in Settings.</>,
      keys: null,
    },
  ];
  return (
    <div class="onb-page">
      <StepHeader
        eyebrow={eyebrowFor("privacy")}
        icon="security"
        title="What stays on this machine."
        lede="No accounts and no telemetry. These protections are on from the first start, and each one says exactly what it does."
      />
      <ul class="promise-grid">
        <For each={__DESKTOP__ ? desktopCards : cards}>
          {(c) => {
            const state = () => (c.keys ? on(...c.keys) : null);
            return (
              <li class="promise reveal" data-reveal>
                <span class="promise__icon" aria-hidden="true">
                  <Icon name={c.icon} size={20} />
                </span>
                <span class="promise__head">
                  <span class="promise__title">{c.title}</span>
                  <Show when={c.keys} fallback={<span class="chip" data-tone="neutral">Always</span>}>
                    <span class="chip" data-tone={state() === false ? "off" : "on"}>
                      {state() === false ? "Off" : "On"}
                    </span>
                  </Show>
                </span>
                <span class="promise__body">{c.body()}</span>
              </li>
            );
          }}
        </For>
      </ul>
      <p class="hint onb-note">
        <Icon name="settings" size={14} />
        On by default. Each one can be switched off in Settings; nothing else changes when you do.
      </p>
    </div>
  );
}

// ---- 5. Terms -----------------------------------------------------------------------------------

export function TermsStep(props: { accepted: boolean; changed: boolean; onOpen: (doc: LegalDocId) => void }): JSX.Element {
  const points: Array<{ icon: IconName; title: string; text: string }> = [
    { icon: "info", title: "Provided as is.", text: "Cerebro is free, open-source software under the Apache-2.0 licence, with no warranty of any kind." },
    { icon: "tools", title: "You answer for what you run.", text: "You are responsible for the models you use, the plugins you install and everything they generate." },
    { icon: "documents", title: "Model licences apply.", text: "Each model has its own licence from its publisher, and its terms apply whenever you use it." },
    { icon: "lock", title: "Your data stays here.", text: "No accounts, no telemetry. Nothing leaves this computer unless you use a network feature." },
  ];
  const docs: Array<{ id: LegalDocId; label: string }> = [
    { id: "terms", label: "Full terms" },
    { id: "privacy", label: "Privacy" },
    { id: "licenses", label: "Licenses" },
  ];
  return (
    <div class="onb-page">
      <StepHeader eyebrow={eyebrowFor("terms")} icon="documents" title="The terms, in plain words." lede="One screen. The full versions are bundled with the app and open without a network." />
      <Show when={props.changed}>
        <p class="notice onb-changed" role="note">
          <Icon name="info" size={18} />
          <span>The terms changed since you last accepted them. This is the new summary; nothing else about your setup changes.</span>
        </p>
      </Show>
      <ol class="terms">
        <For each={points}>
          {(p) => (
            <li class="terms__item">
              <span class="terms__icon" aria-hidden="true">
                <Icon name={p.icon} size={18} />
              </span>
              <p class="terms__title">{p.title}</p>
              <p class="terms__text">{p.text}</p>
            </li>
          )}
        </For>
      </ol>
      <div class="terms-docs">
        <span class="terms-docs__label">Read in full:</span>
        <For each={docs}>
          {(d) => (
            <button type="button" class="btn btn--tertiary terms-docs__btn" data-press aria-haspopup="dialog" onClick={() => props.onOpen(d.id)}>
              <Icon name="documents" size={16} />
              {d.label}
            </button>
          )}
        </For>
      </div>
      <p class="terms-accept" data-accepted={props.accepted}>
        <Show
          when={props.accepted}
          fallback={
            <>
              <Icon name="documents" size={16} />
              <span>
                <strong>Accept and continue</strong> agrees to the Terms of use, version {TERMS_VERSION}. The choice is remembered on this computer.
              </span>
            </>
          }
        >
          <Icon name="check" size={16} />
          <span>
            <strong>Accepted</strong>, version {TERMS_VERSION}.
          </span>
        </Show>
      </p>
    </div>
  );
}

// ---- 6. Finish ----------------------------------------------------------------------------------

export function FinishStep(props: {
  hw: HardwareProfile | undefined;
  recs: readonly Recommendation[];
  model: string | null;
  level: ExperienceLevel;
  config: SetupConfig | null;
  onOpen: (doc: LegalDocId) => void;
}): JSX.Element {
  const { settings } = useAppState();
  const PROTECTIONS: SettingKey[] = ["processContainment", "cleanEnvironment", "integrityCheck", "permissionPrompts", "loopbackOnly"];
  const onCount = () => {
    const s = settings();
    return s ? PROTECTIONS.filter((k) => s[k]).length : PROTECTIONS.length;
  };
  const picked = () => props.recs.find((r) => r.model.id === props.model);
  const id = () => picked()?.model.id ?? "<model>";
  return (
    <div class="onb-page onb-finish">
      <div class="finish-head">
        <div class="finish-mark" aria-hidden="true">
          <svg viewBox="0 0 64 64" class="finish-mark__svg">
            <circle class="finish-mark__ring" cx="32" cy="32" r="28" pathLength="1" />
            <path class="finish-mark__check" d="M20 33.5 28.5 42 45 24" pathLength="1" />
          </svg>
        </div>
        <StepHeader eyebrow={eyebrowFor("finish")} icon="check" title="Cerebro is ready." lede="Here is what setup found and chose. All of it can be changed later." />
      </div>
      {/* The one line that matters: what to do next with what was chosen. */}
      <p class="finish-payoff">
        <Show when={picked()} fallback={<>No first model yet. Pick one in Model Hub whenever you are ready; Tune measures it on this machine.</>}>
          {(r) => (
            <>
              <strong>{r().model.name}</strong>
              {r().ranked ? `, ${r().fitLabel.toLowerCase()}` : ""}: download it in Model Hub, then let Tune measure it here.
            </>
          )}
        </Show>
      </p>
      <dl class="summary">
        <div class="summary__item">
          <dt>
            <Icon name="chip" size={16} />
            This machine
          </dt>
          <dd>
            <Show when={props.hw} fallback="Not checked">
              {(h) => (
                <>
                  <span class="summary__value">{h().gpu?.name ?? "No graphics card found"}</span>
                  <span class="summary__meta">
                    <span class="mono">{gib(h().gpu?.vramTotalGiB ?? 0)}</span>
                    <span class="unit">GiB</span> video memory, <span class="mono">{gib(h().ramTotalGiB)}</span>
                    <span class="unit">GiB</span> system memory
                  </span>
                  <Show when={h().provenance === "illustrative"}>
                    <IllustrativeFlag />
                  </Show>
                </>
              )}
            </Show>
          </dd>
        </div>
        <div class="summary__item">
          <dt>
            <Icon name="models" size={16} />
            First model
          </dt>
          <dd>
            <Show when={picked()} fallback={<span class="summary__value">None yet: choose one in Model Hub</span>}>
              {(r) => (
                <>
                  <span class="summary__value">{r().model.name}</span>
                  <span class="summary__meta">
                    {r().fitLabel}. Not downloaded yet.
                  </span>
                </>
              )}
            </Show>
          </dd>
        </div>
        <div class="summary__item">
          <dt>
            <Icon name={props.level === "beginner" ? "bolt" : "sliders"} size={16} />
            Experience
          </dt>
          <dd>
            <span class="summary__value">{LEVEL_LABELS[props.level]}</span>
            <span class="summary__meta">
              <Show when={picked() && props.config ? { r: picked() as Recommendation, c: props.config } : undefined} fallback="Change it any time in Settings.">
                {(v) => {
                  const q = () => v().r.model.quants.find((x) => x.id === v().c.quantId);
                  return (
                    <>
                      <span class="mono">{q()?.label ?? v().r.model.quant}</span>, <span class="mono">{contextLabel(v().c.contextLength)}</span> context. Change it any time in Settings.
                    </>
                  );
                }}
              </Show>
            </span>
          </dd>
        </div>
        <div class="summary__item">
          <dt>
            <Icon name="security" size={16} />
            Protections
          </dt>
          <dd>
            <span class="summary__value">
              {onCount() === PROTECTIONS.length ? `All ${PROTECTIONS.length} on` : `${onCount()} of ${PROTECTIONS.length} on`}
            </span>
            <span class="summary__meta">Change any of them in Settings.</span>
          </dd>
        </div>
        <div class="summary__item">
          <dt>
            <Icon name="documents" size={16} />
            Terms
          </dt>
          <dd>
            <span class="summary__value">Accepted</span>
            <span class="summary__meta">
              Version {TERMS_VERSION}.{" "}
              <button type="button" class="text-link summary__link" aria-haspopup="dialog" onClick={() => props.onOpen("terms")}>
                Read them again
              </button>
            </span>
          </dd>
        </div>
      </dl>
      <details class="finish-cli">
        <summary class="finish-cli__toggle">
          <Icon name="chevronRight" size={14} />
          Prefer a terminal? The same steps as commands
        </summary>
        <CodeBlock title="Terminal" lines={["cerebro models", `cerebro tune ${id()}`, `cerebro serve ${id()}`, "cerebro chat"]} />
      </details>
      <p class="hint onb-note">
        <Icon name="info" size={14} />
        <span>
          A model has to be on this computer before it can be tuned: download it in Model Hub when you are online, or add a <Term k="gguf" /> file you already have.
        </span>
      </p>
    </div>
  );
}
