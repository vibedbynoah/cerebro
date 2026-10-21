import { createMemo, For, Show, type JSX } from "solid-js";
import { CodeBlock } from "../components/CodeBlock";
import { Icon, type IconName } from "../components/icons";
import { QuantTable } from "../components/QuantTable";
import { IllustrativeFlag } from "../components/Screen";
import { Term } from "../components/Term";
import { CHANNEL_LABELS, CHANNEL_NOTES, CHANNELS, LEVEL_LABELS, type Channel, type ExperienceLevel } from "../lib/experience";
import { FORMAT_NOTES } from "../lib/quants";
import type { CatalogModel, HardwareProfile } from "../service/types";
import {
  changedFrom,
  configFlags,
  CONTEXT_STEPS,
  contextLabel,
  FA_CHOICES,
  KV_CHOICES,
  LIMITS,
  placementFor,
  type ConfigField,
  type SetupConfig,
} from "./autoconfig";
import { StepHeader } from "./StepHeader";

/**
 * Setup steps for the experience level: choosing Beginner or Advanced, and the first model's load
 * settings in the view that level asks for. Both views read and write one SetupConfig, so the
 * escape hatch in the footer can switch views on any step without losing a value.
 */

const LEVEL_POINTS: Readonly<Record<ExperienceLevel, { icon: IconName; lede: string; points: string[] }>> = {
  beginner: {
    icon: "bolt",
    lede: "Cerebro sets everything up from the hardware check and explains each choice in plain words.",
    points: ["Picks the model file, context, offload, KV cache and sampler for you", "No raw flags; every term explained where it appears", "Starts on the Stable optimisation channel"],
  },
  advanced: {
    icon: "sliders",
    lede: "Every setting in view, starting from the same automatic plan.",
    points: ["Raw backend flags, file choice and every load parameter", "Stable, Beta and Experimental optimisation channels", "Dense live metrics: experts, cache hits, draft acceptance, KV fill"],
  },
};

export function LevelStep(props: { eyebrow: string; level: ExperienceLevel; onSelect: (l: ExperienceLevel) => void }): JSX.Element {
  return (
    <div class="onb-page">
      <StepHeader
        eyebrow={props.eyebrow}
        icon="settings"
        title="How much do you want to see?"
        lede="Both get the same measured speed and the same protections. Switch any time: at the bottom of every setup step, and later in Settings."
      />
      <fieldset class="level-pick">
        <legend class="sr-only">Experience level</legend>
        <For each={["beginner", "advanced"] as const}>
          {(l) => (
            <label class="level-card" classList={{ "is-selected": props.level === l }} data-level={l} for={`level-${l}`}>
              <input
                id={`level-${l}`}
                class="level-card__input"
                type="radio"
                name="experience-level"
                value={l}
                checked={props.level === l}
                aria-describedby={`level-${l}-lede`}
                onChange={() => props.onSelect(l)}
              />
              <span class="level-card__icon" aria-hidden="true">
                <Icon name={LEVEL_POINTS[l].icon} size={20} />
              </span>
              <span class="level-card__name">{LEVEL_LABELS[l]}</span>
              <span class="level-card__lede" id={`level-${l}-lede`}>
                {LEVEL_POINTS[l].lede}
              </span>
              <ul class="level-card__points">
                <For each={LEVEL_POINTS[l].points}>
                  {(p) => (
                    <li>
                      <Icon name="check" size={14} />
                      {p}
                    </li>
                  )}
                </For>
              </ul>
              <span class="level-card__radio" aria-hidden="true" />
            </label>
          )}
        </For>
      </fieldset>
    </div>
  );
}

// ---- Load settings ---------------------------------------------------------------------------------

const KV_WORDS: Readonly<Record<string, string>> = {
  f16: "Full precision (f16): the most memory, no quality cost.",
  q8_0: "8-bit (q8_0): half the memory of full precision, at a very small quality cost.",
  q4_0: "4-bit (q4_0): a quarter of the memory, with a noticeable quality cost on long contexts.",
};

/** About 0.75 words per token and 500 words per page. A plain-words scale, not a promise. */
const pages = (tokens: number): number => Math.max(1, Math.round((tokens * 0.75) / 500));

export function ConfigureStep(props: {
  eyebrow: string;
  level: ExperienceLevel;
  model: CatalogModel | undefined;
  hw: HardwareProfile | undefined;
  config: SetupConfig | null;
  auto: SetupConfig | null;
  channel: Channel;
  onChange: (cfg: SetupConfig) => void;
  onReset: () => void;
  onChannel: (c: Channel) => void;
}): JSX.Element {
  const changed = createMemo(() => changedFrom(props.auto, props.config));
  const beginner = () => props.level === "beginner";
  return (
    <div class="onb-page onb-page--dense" data-view={props.level}>
      <StepHeader
        eyebrow={props.eyebrow}
        icon={beginner() ? "bolt" : "sliders"}
        title={beginner() ? "Cerebro set this up for you." : "Every load setting, in view."}
        wide
        lede={
          beginner()
            ? "Chosen from the hardware check for the model you picked. Nothing to do here; each choice is explained below."
            : "Starting from the automatic plan. Change anything; the backend flags below follow every edit."
        }
      />
      <Show
        when={props.model && props.config ? { model: props.model, config: props.config } : undefined}
        fallback={
          <p class="notice rec-notice" role="note">
            <Icon name="info" size={18} />
            <span>No first model chosen, so there is nothing to set yet. Cerebro sets it up the same way when you add a model.</span>
          </p>
        }
      >
        {(v) => (
          <>
            <Show when={changed().length > 0}>
              <p class="notice config-changed" role="note" data-changed={changed().length}>
                <Icon name="sliders" size={18} />
                <span>
                  {changed().length === 1 ? "1 value differs" : `${changed().length} values differ`} from the automatic plan
                  {beginner() ? ", changed in Advanced. They are kept." : "."}
                </span>
                <button type="button" class="btn btn--secondary btn--xs" data-press onClick={() => props.onReset()}>
                  <Icon name="undo" size={14} />
                  Use automatic values
                </button>
              </p>
            </Show>
            <Show when={beginner()} fallback={<AdvancedConfig model={v().model} config={v().config} channel={props.channel} changed={changed()} onChange={props.onChange} onChannel={props.onChannel} />}>
              <BeginnerPlan model={v().model} hw={props.hw} config={v().config} changed={changed()} />
            </Show>
          </>
        )}
      </Show>
    </div>
  );
}

function BeginnerPlan(props: { model: CatalogModel; hw: HardwareProfile | undefined; config: SetupConfig; changed: ConfigField[] }): JSX.Element {
  const quant = () => props.model.quants.find((q) => q.id === props.config.quantId);
  const place = () => {
    const q = quant();
    return q ? placementFor(props.model, q, props.hw) : "cpu";
  };
  const L = () => props.model.layers;
  const where = (): string => {
    const c = props.config;
    if (c.gpuLayers === 0) return "On the processor, from system memory.";
    if (props.model.architecture === "moe" && c.moeCpuLayers > 0)
      return `All ${L()} layers on the graphics card, with the experts of ${c.moeCpuLayers} layers in system memory. Only a few experts work per token, so this costs little.`;
    if (c.gpuLayers >= L()) return `All ${L()} layers on the graphics card: the fastest place.`;
    return `${c.gpuLayers} of ${L()} layers on the graphics card, the rest in system memory.`;
  };
  const why = (): string => {
    switch (place()) {
      case "gpu":
        return "The closest to the original weights that fits in your graphics memory with room for context.";
      case "moe-offload":
        return "The closest to the original that fits across video and system memory with room to spare.";
      case "split":
        return "The closest to the original that still leaves a quarter of your memory free.";
      case "cpu":
        return "Small enough to run on the processor at a usable speed.";
    }
  };
  const isChanged = (f: ConfigField) => props.changed.includes(f);
  const items = (): Array<{ id: ConfigField; icon: IconName; label: JSX.Element; value: string; text: string }> => {
    const c = props.config;
    const q = quant();
    return [
      {
        id: "quantId",
        icon: "models",
        label: (
          <>
            Model file <Term k="quant" />
          </>
        ),
        value: q ? `${q.label}, ${q.sizeGiB.toFixed(1)} GiB` : "–",
        text: `${why()} ${q ? FORMAT_NOTES[q.format] : ""}`,
      },
      {
        id: "contextLength",
        icon: "documents",
        label: (
          <>
            Context <Term k="context" />
          </>
        ),
        value: `${contextLabel(c.contextLength)} tokens`,
        text: `About ${pages(c.contextLength)} pages of text in view at once.`,
      },
      {
        id: "gpuLayers",
        icon: "gpu",
        label: (
          <>
            Where it runs <Term k="offload" />
          </>
        ),
        value: c.gpuLayers === 0 ? "Processor" : c.gpuLayers >= L() ? "Graphics card" : "Card and system memory",
        text: where(),
      },
      {
        id: "kvCache",
        icon: "memory",
        label: (
          <>
            Memory for context <Term k="kvCache" />
          </>
        ),
        value: c.kvCache,
        text: KV_WORDS[c.kvCache] ?? "",
      },
      {
        id: "temperature",
        icon: "chat",
        label: (
          <>
            How it picks words <Term k="sampler" />
          </>
        ),
        value: `temperature ${c.temperature}`,
        text: `The publisher's recommended values: temperature ${c.temperature}, top-p ${c.topP}${c.topK > 0 ? `, top-k ${c.topK}` : ""}.`,
      },
    ];
  };
  return (
    <>
      <ul class="plan">
        <For each={items()}>
          {(it) => (
            <li class="plan__item" data-field={it.id} data-changed={isChanged(it.id) || (it.id === "gpuLayers" && isChanged("moeCpuLayers")) || (it.id === "temperature" && (isChanged("topP") || isChanged("topK") || isChanged("minP")))}>
              <span class="plan__icon" aria-hidden="true">
                <Icon name={it.icon} size={18} />
              </span>
              <span class="plan__label">{it.label}</span>
              <span class="plan__value mono">{it.value}</span>
              <span class="plan__text">{it.text}</span>
            </li>
          )}
        </For>
      </ul>
      <p class="hint onb-note">
        <Icon name="tune" size={14} />
        <span>
          Starting values from file size and memory, not measurements. After download, Tune measures this machine and keeps whatever runs fastest within the quality gate.
        </span>
        <Show when={props.model.provenance === "illustrative"}>
          <IllustrativeFlag />
        </Show>
      </p>
    </>
  );
}

function AdvancedConfig(props: {
  model: CatalogModel;
  config: SetupConfig;
  channel: Channel;
  changed: ConfigField[];
  onChange: (cfg: SetupConfig) => void;
  onChannel: (c: Channel) => void;
}): JSX.Element {
  const set = <K extends ConfigField>(k: K, v: SetupConfig[K]) => props.onChange({ ...props.config, [k]: v });
  const num = (k: "gpuLayers" | "moeCpuLayers" | "temperature" | "topP" | "topK" | "minP", raw: string) => {
    const n = Number(raw);
    if (raw.trim() !== "" && Number.isFinite(n)) set(k, n);
  };
  const isChanged = (f: ConfigField) => props.changed.includes(f);
  const contexts = () => CONTEXT_STEPS.filter((c) => c <= props.model.contextMax);
  return (
    <div class="adv">
      <section class="adv__block" aria-labelledby="adv-file-title">
        <h2 class="adv__title" id="adv-file-title">
          File <Term k="quant" />
        </h2>
        <QuantTable
          quants={props.model.quants}
          channel={props.channel}
          caption={`Files for ${props.model.name}`}
          pick={{ name: "setup-quant", selected: props.config.quantId, onSelect: (id) => set("quantId", id) }}
        />
      </section>

      <section class="adv__block" aria-labelledby="adv-load-title">
        <h2 class="adv__title" id="adv-load-title">
          Load parameters
        </h2>
        <div class="adv-grid">
          <AdvField id="cfg-ctx" label="Context" term="context" changed={isChanged("contextLength")} hint={`Up to ${contextLabel(props.model.contextMax)}.`}>
            <span class="select select--sm">
              <select id="cfg-ctx" class="input" value={String(props.config.contextLength)} onChange={(e) => set("contextLength", Number(e.currentTarget.value))}>
                <For each={contexts()}>{(c) => <option value={String(c)}>{`${contextLabel(c)} tokens`}</option>}</For>
              </select>
            </span>
          </AdvField>
          <AdvField id="cfg-ngl" label="GPU layers" term="offload" changed={isChanged("gpuLayers")} hint={`0 to ${props.model.layers}.`}>
            <input id="cfg-ngl" class="input input--sm mono" type="number" min="0" max={props.model.layers} step="1" value={props.config.gpuLayers} onChange={(e) => num("gpuLayers", e.currentTarget.value)} />
          </AdvField>
          <Show when={props.model.architecture === "moe"}>
            <AdvField id="cfg-ncmoe" label="MoE layers on CPU" term="moeCpu" changed={isChanged("moeCpuLayers")} hint="Expert weights of this many layers stay in system memory.">
              <input id="cfg-ncmoe" class="input input--sm mono" type="number" min="0" max={props.model.layers} step="1" value={props.config.moeCpuLayers} onChange={(e) => num("moeCpuLayers", e.currentTarget.value)} />
            </AdvField>
          </Show>
          <AdvField id="cfg-kv" label="KV cache type" term="kvCache" changed={isChanged("kvCache")}>
            <span class="select select--sm">
              <select id="cfg-kv" class="input mono" value={props.config.kvCache} onChange={(e) => set("kvCache", e.currentTarget.value as SetupConfig["kvCache"])}>
                <For each={KV_CHOICES}>{(k) => <option value={k}>{k}</option>}</For>
              </select>
            </span>
          </AdvField>
          <AdvField id="cfg-fa" label="Flash attention" term="flashAttention" changed={isChanged("flashAttention")}>
            <span class="select select--sm">
              <select id="cfg-fa" class="input" value={props.config.flashAttention} onChange={(e) => set("flashAttention", e.currentTarget.value as SetupConfig["flashAttention"])}>
                <For each={FA_CHOICES}>{(k) => <option value={k}>{k}</option>}</For>
              </select>
            </span>
          </AdvField>
          <AdvField id="cfg-temp" label="Temperature" term="sampler" changed={isChanged("temperature")}>
            <input id="cfg-temp" class="input input--sm mono" type="number" min={LIMITS.temperature.min} max={LIMITS.temperature.max} step={LIMITS.temperature.step} value={props.config.temperature} onChange={(e) => num("temperature", e.currentTarget.value)} />
          </AdvField>
          <AdvField id="cfg-topp" label="Top-p" changed={isChanged("topP")}>
            <input id="cfg-topp" class="input input--sm mono" type="number" min={LIMITS.topP.min} max={LIMITS.topP.max} step={LIMITS.topP.step} value={props.config.topP} onChange={(e) => num("topP", e.currentTarget.value)} />
          </AdvField>
          <AdvField id="cfg-topk" label="Top-k" changed={isChanged("topK")} hint="0 turns it off.">
            <input id="cfg-topk" class="input input--sm mono" type="number" min={LIMITS.topK.min} max={LIMITS.topK.max} step={LIMITS.topK.step} value={props.config.topK} onChange={(e) => num("topK", e.currentTarget.value)} />
          </AdvField>
          <AdvField id="cfg-minp" label="Min-p" changed={isChanged("minP")}>
            <input id="cfg-minp" class="input input--sm mono" type="number" min={LIMITS.minP.min} max={LIMITS.minP.max} step={LIMITS.minP.step} value={props.config.minP} onChange={(e) => num("minP", e.currentTarget.value)} />
          </AdvField>
        </div>
      </section>

      <section class="adv__block" aria-labelledby="adv-channel-title">
        <h2 class="adv__title" id="adv-channel-title">
          Optimisation channel
        </h2>
        <fieldset class="segmented" aria-describedby="adv-channel-note">
          <legend class="sr-only">Optimisation channel</legend>
          <For each={CHANNELS}>
            {(c) => (
              <label class="segmented__option" data-press>
                <input type="radio" name="setup-channel" value={c} checked={props.channel === c} onChange={() => props.onChannel(c)} />
                <span>{CHANNEL_LABELS[c]}</span>
              </label>
            )}
          </For>
        </fieldset>
        <p class="hint" id="adv-channel-note">
          {CHANNEL_NOTES[props.channel]}
        </p>
      </section>

      {/* The desktop engine runs inside the app with no command line, so there are no flags to copy. */}
      <Show when={!__DESKTOP__}>
        <section class="adv__block" aria-labelledby="adv-flags-title">
          <h2 class="adv__title" id="adv-flags-title">
            Backend flags
          </h2>
          <CodeBlock title="Backend command" lines={configFlags(props.model, props.config)} label="Copy flags" oneCommand />
        </section>
      </Show>
    </div>
  );
}

function AdvField(props: { id: string; label: string; term?: "context" | "offload" | "moeCpu" | "kvCache" | "flashAttention" | "sampler"; hint?: string; changed: boolean; children: JSX.Element }): JSX.Element {
  return (
    <div class="field adv-field" data-changed={props.changed}>
      <span class="field__label">
        {/* With a definition, the visible word is the <Term> (its ? button cannot sit inside a
            label); the input keeps its name from a visually hidden label. */}
        <Show when={props.term} fallback={<label for={props.id}>{props.label}</label>}>
          {(t) => (
            <>
              <label for={props.id} class="sr-only">
                {props.label}
              </label>
              <Term k={t()}>{props.label}</Term>
            </>
          )}
        </Show>
        <Show when={props.changed}>
          <span class="adv-field__dot" title="Changed from the automatic plan">
            <span class="sr-only">changed from the automatic plan</span>
          </span>
        </Show>
      </span>
      {props.children}
      <Show when={props.hint}>
        <span class="field__hint">{props.hint}</span>
      </Show>
    </div>
  );
}
