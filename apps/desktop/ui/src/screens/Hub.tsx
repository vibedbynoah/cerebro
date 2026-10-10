import "../styles/hub.css";
import { createMemo, createResource, createSignal, createUniqueId, For, Show, type JSX } from "solid-js";
import { CodeBlock } from "../components/CodeBlock";
import { Icon } from "../components/icons";
import { QuantTable } from "../components/QuantTable";
import { IllustrativeFlag, ScreenHeader, Section } from "../components/Screen";
import { Term } from "../components/Term";
import { CHANNEL_LABELS, type Channel, type ExperienceLevel } from "../lib/experience";
import { availability, FORMAT_LABELS, FORMAT_NOTES, qualityKept } from "../lib/quants";
import { hrefFor } from "../lib/routes";
import { contextLabel, pickQuant } from "../onboarding/autoconfig";
import { rankModels, unranked, type Recommendation } from "../onboarding/recommend";
import { useService } from "../service/context";
import type { HardwareProfile } from "../service/types";
import { useAppState } from "../state";
import { LiveHub } from "./hub/LiveHub";

/**
 * Model Hub: the catalogue of current models, each with every quantised file side by side by
 * quality per GiB, KL divergence, perplexity change and speed on this machine. Nothing downloads
 * from this screen on its own: listing needs no network and no file is fetched until the user asks
 * for one. Advanced sees every file at once; Beginner sees the file Cerebro would pick, with the
 * full table one click away. Every figure carries its provenance (docs/CLAIMS.md).
 */
export function ModelHub(): JSX.Element {
  // The desktop app searches Hugging Face live; the web preview shows the bundled catalogue.
  if (__DESKTOP__) return <LiveHub />;
  const service = useService();
  const app = useAppState();
  const [catalog] = createResource(() => service.listCatalog());
  const [hw] = createResource(() => service.detectHardware().catch(() => undefined));
  const offline = () => app.settings()?.offlineMode === true || app.network() !== "online";
  const downloadsOff = () => app.settings()?.allowModelDownloads === false;

  const recs = createMemo<Recommendation[]>(() => {
    const list = catalog.error ? undefined : catalog();
    if (!list) return [];
    const h = hw.error ? undefined : hw();
    return h ? rankModels(list, h) : unranked(list);
  });

  return (
    <div class="screen">
      <ScreenHeader route="hub" />
      <Show when={offline() || downloadsOff()}>
        <div class="notice reveal" data-reveal role="status">
          <Icon name="offline" size={18} />
          <p>
            <Show when={downloadsOff()} fallback={<>No network right now. The catalogue below is bundled with the app; downloading a file needs a connection.</>}>
              Model downloads are switched off. <SettingsLink>Change this in Settings</SettingsLink>.
            </Show>
          </p>
        </div>
      </Show>

      <Section
        id="catalog"
        title="Current models, every file compared"
        description={
          <>
            Ranked by how each fits this machine's memory. Each <Term k="quant">quantised</Term> file is compared by how close it stays to the original weights (<Term k="klDiv" />) per GiB, with
            its writing speed here. Nothing downloads until you choose a file.
          </>
        }
      >
        <Show when={!catalog.loading} fallback={<p class="hint">Reading the catalogue…</p>}>
          <Show
            when={!catalog.error}
            fallback={
              <p class="notice" role="alert">
                <Icon name="alert" size={18} />
                <span>The catalogue could not be read. Models already on this machine are in Models.</span>
              </p>
            }
          >
            <div class="hub-meta">
              <span class="hint">
                {app.level() === "beginner" ? "Showing the file Cerebro would pick for each model." : `Showing every file. Channel: ${CHANNEL_LABELS[app.channel()]}.`}
              </span>
              <IllustrativeFlag />
            </div>
            <ol class="hub-list">
              <For each={recs()}>{(r) => <HubCard rec={r} hw={hw.error ? undefined : hw()} level={app.level()} channel={app.channel()} />}</For>
            </ol>
          </Show>
        </Show>
      </Section>

      <Section id="today" title="Works today" description="Lists the models already on this machine, with their file paths.">
        <CodeBlock lines={["cerebro models"]} label="Copy command" />
      </Section>
    </div>
  );
}

function SettingsLink(props: { children: JSX.Element }): JSX.Element {
  return (
    <a class="text-link" href={hrefFor("settings")}>
      {props.children}
    </a>
  );
}

function HubCard(props: { rec: Recommendation; hw: HardwareProfile | undefined; level: ExperienceLevel; channel: Channel }): JSX.Element {
  const id = createUniqueId();
  const [open, setOpen] = createSignal(false);
  const m = () => props.rec.model;
  const pick = createMemo(() => pickQuant(m(), props.hw, props.channel));
  const runnable = () => m().quants.filter((q) => availability(q, props.channel) === "runs").length;
  const showTable = () => props.level === "advanced" || open();
  return (
    <li class="hub-card card" data-model={m().id} data-fit={props.rec.fit} aria-labelledby={`${id}-name`}>
      <div class="hub-card__head">
        <div class="hub-card__title">
          <h3 class="hub-card__name" id={`${id}-name`}>
            {m().name}
          </h3>
          <p class="hub-card__facts">
            <span class="mono">{m().params}</span>
            <span aria-hidden="true">·</span>
            <span>
              {m().architecture === "moe" ? (
                <>
                  <Term k="moe">MoE</Term>, <span class="mono">{m().activeParams ?? "?"}</span> active
                </>
              ) : (
                "dense"
              )}
            </span>
            <span aria-hidden="true">·</span>
            <span>
              <span class="mono">{contextLabel(m().contextMax)}</span> context
            </span>
            <span aria-hidden="true">·</span>
            <span>{m().goodFor}</span>
          </p>
        </div>
        <span class="chip hub-card__fit" data-tone={props.rec.fit === "gpu" ? "on" : props.rec.fit === "too-large" ? "off" : "neutral"}>
          {props.rec.fitLabel}
        </span>
      </div>

      <Show when={props.level === "beginner"}>
        <Show
          when={pick()}
          fallback={<p class="hint">No file of this model runs on this machine with the {CHANNEL_LABELS[props.channel]} channel.</p>}
        >
          {(q) => (
            <div class="hub-pick">
              <span class="hub-pick__label">Cerebro would use</span>
              <span class="hub-pick__file mono">{q().label}</span>
              <span class="hub-pick__nums">
                <span class="mono">{q().sizeGiB.toFixed(1)}</span> GiB · keeps <span class="mono">{qualityKept(q()).toFixed(1)}%</span> of the original · <span class="mono">{(q().decodeTps ?? 0).toFixed(1)}</span>{" "}
                tok/s here
              </span>
              <span class="hub-pick__why">
                {FORMAT_LABELS[q().format]}: {FORMAT_NOTES[q().format]}
              </span>
            </div>
          )}
        </Show>
        <button type="button" class="btn btn--tertiary btn--sm hub-card__more" data-press aria-expanded={open()} aria-controls={`${id}-files`} onClick={() => setOpen(!open())}>
          <Icon name="chevron" size={14} />
          {open() ? "Hide the other files" : `Compare all ${m().quants.length} files (${runnable()} run here)`}
        </button>
      </Show>

      <div id={`${id}-files`} class="hub-card__files" hidden={!showTable()}>
        <Show when={showTable()}>
          <QuantTable quants={m().quants} channel={props.channel} caption={`Files for ${m().name}`} />
        </Show>
      </div>
    </li>
  );
}
