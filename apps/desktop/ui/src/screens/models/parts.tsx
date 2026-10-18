import { For, Match, Show, Switch, type JSX } from "solid-js";
import { Icon, type IconName } from "../../components/icons";
import { FIT_LABELS, type FitClass } from "../../lib/fit";
import type { LibraryModel, Modality, ModelSource, ModelSummary } from "../../service/types";

export function TuneChip(props: { model: Pick<ModelSummary, "tune"> }): JSX.Element {
  return (
    <Switch>
      <Match when={props.model.tune === "tuned"}>
        <span class="chip" data-tone="on">
          Tuned
        </span>
      </Match>
      <Match when={props.model.tune === "defaults-fastest"}>
        <span class="chip" data-tone="neutral">
          Defaults were fastest
        </span>
      </Match>
      <Match when={props.model.tune === "tuning"}>
        <span class="chip" data-tone="active">
          Measuring
        </span>
      </Match>
      <Match when={props.model.tune === "untuned"}>
        <span class="chip" data-tone="off">
          Not tuned yet
        </span>
      </Match>
    </Switch>
  );
}

export const MODALITY: Readonly<Record<Modality, { label: string; icon: IconName }>> = {
  text: { label: "Text", icon: "text" },
  vision: { label: "Vision", icon: "eye" },
  "audio-in": { label: "Audio in", icon: "mic" },
  "audio-out": { label: "Audio out", icon: "speaker" },
  "image-gen": { label: "Image generation", icon: "images" },
};

/** Modality badges: icon plus text, so colour is never the only signal. `compact` drops the
 *  text badge on rows where every model reads text, but keeps its name for screen readers. */
export function ModalityBadges(props: { modalities: readonly Modality[]; compact?: boolean }): JSX.Element {
  return (
    <span class="badges">
      <For each={props.modalities}>
        {(m) => (
          <span class="badge" data-modality={m} classList={{ "badge--quiet": props.compact === true && m === "text" }}>
            <Icon name={MODALITY[m].icon} size={12} />
            <span class="badge__label">{props.compact && m === "image-gen" ? "Image gen" : MODALITY[m].label}</span>
          </span>
        )}
      </For>
    </span>
  );
}

export function FitChip(props: { fit: FitClass }): JSX.Element {
  return (
    <span class="fit-chip" data-fit={props.fit}>
      <span class="fit-chip__dot" aria-hidden="true" />
      {FIT_LABELS[props.fit]}
    </span>
  );
}

export const SOURCE_LABELS: Readonly<Record<ModelSource, string>> = {
  cerebro: "Cerebro",
  lmstudio: "LM Studio",
  ollama: "Ollama",
  other: "Other folder",
};

/** "17.3 GiB" with the unit as its own span (DESIGN.md: a number and its unit are one unit of type). */
export function Size(props: { gib: number }): JSX.Element {
  return (
    <span class="num-unit">
      {props.gib >= 1000 ? (props.gib / 1024).toFixed(1) : props.gib.toFixed(1)}
      <span class="unit">{props.gib >= 1000 ? "TiB" : "GiB"}</span>
    </span>
  );
}

/** The publisher folder a file came from, when its path shows one. */
export function publisherOf(m: LibraryModel): string | null {
  const parts = m.path.split("\\");
  if (m.source === "cerebro" || m.source === "lmstudio") return parts[5] ?? null;
  return null;
}

/** "8K", "32K", "1M": a context length the way the panel and tables label it. */
export function contextLabel(tokens: number): string {
  if (tokens >= 1024 * 1024 && tokens % (1024 * 1024) === 0) return `${tokens / (1024 * 1024)}M`;
  if (tokens >= 1024 && tokens % 1024 === 0) return `${tokens / 1024}K`;
  return tokens.toLocaleString("en-US");
}

/**
 * Last tune in the library: writing speed at the working depth only, labelled with that depth, and
 * the other two numbers and the conditions in a tooltip and for screen readers. Never a single
 * unlabelled tok/s (docs/CLAIMS.md: three numbers, never one).
 */
export function LastTune(props: { model: LibraryModel }): JSX.Element {
  const llm = () => props.model.kind === "text" || props.model.kind === "vision";
  return (
    <Show when={llm()} fallback={<span class="muted">Own engine</span>}>
      <Show when={props.model.tuneResult} fallback={<TuneChip model={props.model} />}>
        {(r) => {
          const detail = () =>
            `Prompt reading ${r().prefillTps.toFixed(0)} tok/s; writing ${r().decodeTps0.toFixed(1)} tok/s with an empty context and ${r().decodeTpsDepth.toFixed(1)} tok/s at ${contextLabel(r().depth)}. ${r().gpu}, ${r().build}, sampler ${r().sampler}.`;
          return (
            <span class="last-tune" title={detail()}>
              <span class="num-unit mono" aria-hidden="true">
                {r().decodeTpsDepth.toFixed(1)}
                <span class="unit">tok/s</span>
              </span>
              <span class="last-tune__depth mono" aria-hidden="true">{`@${contextLabel(r().depth)}`}</span>
              <span class="sr-only">{detail()}</span>
            </span>
          );
        }}
      </Show>
    </Show>
  );
}

export function Spinner(): JSX.Element {
  return <span class="btn__spinner" aria-hidden="true" />;
}
