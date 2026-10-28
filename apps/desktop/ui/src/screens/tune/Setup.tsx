import { createEffect, createMemo, createSignal, createUniqueId, For, on, Show, type Accessor, type JSX } from "solid-js";
import { Icon, type IconName } from "../../components/icons";
import { Term } from "../../components/Term";
import { hrefFor } from "../../lib/routes";
import { readPref, writePref } from "../../lib/storage";
import { depthLabel, isTuneProfile, pickFit, PROFILE_IDS, PROFILES } from "../../lib/tune";
import type { HardwareProfile, LibraryModel, TuneProfile, TuneState } from "../../service/types";
import { Spinner, TuneChip } from "../models/parts";
import { TermLine } from "./bits";
import type { TuneSession } from "./session";

/** A model the setup can offer: the fields its card shows. Kept stable while a run streams in (the
 *  live tune state comes from `tuneOf`), so the cards are never rebuilt mid-run. */
export interface PickModel {
  id: string;
  name: string;
  quant: string;
  params: string;
  architecture: "dense" | "moe";
  sizeGiB: number;
}

const PROFILE_ICONS: Readonly<Record<TuneProfile, IconName>> = { chat: "chat", coding: "code", "long-docs": "documents" };

/** Per-viewer convenience: the last model and profile chosen here. Validated on every read. */
const SETUP_KEY = "cerebro-tune-setup";

function readSetup(): { model: string | null; profile: TuneProfile | null } {
  try {
    const raw: unknown = JSON.parse(readPref(SETUP_KEY) ?? "null");
    if (!raw || typeof raw !== "object") return { model: null, profile: null };
    const r = raw as Record<string, unknown>;
    return {
      model: typeof r["model"] === "string" && /^[a-z0-9][a-z0-9._-]{0,127}$/.test(r["model"]) ? r["model"] : null,
      profile: isTuneProfile(r["profile"]) ? r["profile"] : null,
    };
  } catch {
    return { model: null, profile: null };
  }
}

export function TuneSetup(props: {
  session: TuneSession;
  models: Accessor<readonly PickModel[] | undefined>;
  /** The model's tune state right now: the library's, overridden by the run in view. */
  tuneOf: (id: string) => TuneState;
  /** Library entries by id, for the memory estimate (architecture and context limit). */
  library: Accessor<ReadonlyMap<string, LibraryModel>>;
  /** This machine, or null when it was not detected: no estimate is then made. */
  hardware: Accessor<HardwareProfile | null>;
  /** A model chosen from the Models screen, preselected when it is offered. */
  pickedId: Accessor<string | null>;
  /** The model currently chosen, so the screen's command line can follow it. */
  onChoose: (id: string | null) => void;
  onStarted: () => void;
}): JSX.Element {
  const id = createUniqueId();
  const saved = readSetup();
  const [model, setModel] = createSignal<string | null>(saved.model);
  const [profile, setProfile] = createSignal<TuneProfile>(saved.profile ?? "chat");
  const running = () => props.session.view.status === "running";

  // The chosen model must be one on offer: a link's model wins, then the saved one, then the first.
  createEffect(
    on([props.models, props.pickedId], ([list, picked]) => {
      if (!list || list.length === 0) return;
      if (picked && list.some((m) => m.id === picked)) setModel(picked);
      else if (!list.some((m) => m.id === model())) setModel(list[0]!.id);
    }),
  );
  const remember = () => writePref(SETUP_KEY, JSON.stringify({ model: model(), profile: profile() }));
  const chosen = createMemo(() => props.models()?.find((m) => m.id === model()) ?? null);
  createEffect(() => props.onChoose(chosen()?.id ?? null));

  const start = async () => {
    const m = chosen();
    if (!m || running()) return;
    if (await props.session.start(m.id, profile())) props.onStarted();
  };

  // The same estimate as the Models screen (lib/fit.ts), at the chosen workload's depth.
  const fits = (m: PickModel) => {
    const lib = props.library().get(m.id);
    return lib ? pickFit(lib, props.hardware(), profile()) : null;
  };

  return (
    <div class="setup">
      <fieldset class="pick" disabled={running()}>
        <legend class="pick__legend">
          <span class="pick__step mono" aria-hidden="true">
            1
          </span>
          Model
        </legend>
        <Show when={props.models()} fallback={<div class="pick__grid pick__grid--loading" aria-hidden="true" />}>
          {(list) => (
            <div class="pick__grid pick__grid--models">
              <For each={list()}>
                {(m) => {
                  const fit = createMemo(() => fits(m));
                  return (
                    <label class="pick-card" data-press data-model={m.id}>
                      <input
                        type="radio"
                        name={`${id}-model`}
                        value={m.id}
                        checked={model() === m.id}
                        onChange={() => {
                          setModel(m.id);
                          remember();
                        }}
                      />
                      <span class="pick-card__check" aria-hidden="true">
                        <Icon name="check" size={12} />
                      </span>
                      <span class="pick-card__name">{m.name}</span>
                      <span class="pick-card__meta mono">
                        {m.quant} · {m.params} · {m.architecture === "moe" ? "MoE" : "dense"}
                      </span>
                      <span class="pick-card__fit">
                        <span class="num-unit mono">
                          {m.sizeGiB.toFixed(1)}
                          <span class="unit">GiB</span>
                        </span>
                        <span class="pick-card__fit-text" data-fits={fit() ? String(fit()!.fits) : undefined}>
                          {fit()?.text ?? ""}
                        </span>
                      </span>
                      <TuneChip
                        model={{
                          get tune() {
                            return props.tuneOf(m.id);
                          },
                        }}
                      />
                    </label>
                  );
                }}
              </For>
            </div>
          )}
        </Show>
        <p class="pick__terms">
          On the cards: the <Term k="quant" /> (Q4_K_M, Q5_K_M), whether it is an <Term k="moe" /> model or dense (every part works on every token), its size on disk, and
          whether it fits in <Term k="vram">graphics memory</Term> with its <Term k="kvCache" /> at the chosen workload's depth (an estimate; the tune measures the real thing).
        </p>
        <p class="hint">
          <Icon name="info" size={16} />
          <span>
            Another model? Open it in{" "}
            <a class="text-link" href={hrefFor("models")}>
              Models
            </a>{" "}
            and choose Tune.
          </span>
        </p>
      </fieldset>

      <fieldset class="pick" disabled={running()}>
        <legend class="pick__legend">
          <span class="pick__step mono" aria-hidden="true">
            2
          </span>
          Main use
        </legend>
        <div class="pick__grid pick__grid--profiles">
          <For each={PROFILE_IDS}>
            {(p) => (
              <label class="pick-card pick-card--profile" data-press data-profile={p}>
                <input
                  type="radio"
                  name={`${id}-profile`}
                  value={p}
                  checked={profile() === p}
                  aria-describedby={`${id}-objective`}
                  onChange={() => {
                    setProfile(p);
                    remember();
                  }}
                />
                <span class="pick-card__check" aria-hidden="true">
                  <Icon name="check" size={12} />
                </span>
                <span class="pick-card__icon" aria-hidden="true">
                  <Icon name={PROFILE_ICONS[p]} size={18} />
                </span>
                <span class="pick-card__name">{PROFILES[p].label}</span>
                <span class="pick-card__summary">{PROFILES[p].summary}</span>
                <span class="pick-card__meta mono">{`${depthLabel(PROFILES[p].depth)} tokens in view`}</span>
              </label>
            )}
          </For>
        </div>
        <p class="objective" id={`${id}-objective`}>
          <span class="objective__label">What fastest means here</span>
          <span class="objective__text">
            <TermLine parts={PROFILES[profile()].objective} /> Measured at a <Term k="depth" /> of {depthLabel(PROFILES[profile()].depth)} tokens, or the model's own limit if
            that is lower.
          </span>
        </p>
      </fieldset>

      <div class="setup__foot">
        <button
          type="button"
          class="btn btn--primary setup__start"
          data-press
          data-sweep
          disabled={running() || !chosen()}
          aria-busy={props.session.busy() === "start" ? "true" : undefined}
          aria-describedby={`${id}-note`}
          onClick={() => void start()}
        >
          <Spinner />
          <span class="btn__label">
            <Icon name="play" size={16} />
            Start tune
          </span>
        </button>
        <p class="setup__note" id={`${id}-note`}>
          {running()
            ? "One tune at a time, because it uses the graphics card fully. Cancel the current one to start another."
            : "Runs on this machine, fully offline. Keep working in other screens while it measures."}
        </p>
      </div>
    </div>
  );
}
