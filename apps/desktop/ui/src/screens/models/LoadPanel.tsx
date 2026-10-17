import { createEffect, createMemo, createResource, createSignal, createUniqueId, For, onCleanup, Show, type Accessor, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { SettingSwitch } from "../../components/SettingSwitch";
import { Term } from "../../components/Term";
import { estimateFit, type FitEstimate, type FitStatus, type MemoryUse } from "../../lib/fit";
import type { GlossaryKey } from "../../lib/glossary";
import { changedFields, draftCandidates, KV_TYPES, LIMITS, maxThreadsFor, NGRAM_STRATEGIES, type LoadErrors, type LoadField } from "../../lib/load-params";
import { notify } from "../../lib/toast";
import { profileNameError, tweakedFields } from "../../lib/tweaks";
import { useService } from "../../service/context";
import type { LibraryModel, LoadParams, LoadPreset, NgramStrategy, Tweaks } from "../../service/types";
import { PlacementCard } from "./PlacementCard";
import { useModels } from "./store";

/** Plain-words names for the draft-free strategies (docs/CLAIMS.md lists the ids). */
export const NGRAM_LABELS: Readonly<Record<NgramStrategy, string>> = {
  "ngram-simple": "Simple: repeat recent text",
  "ngram-map-k": "Map: match the last few tokens",
  "ngram-map-k4v": "Map, longer 4-token matches",
  "ngram-mod": "Hash of recent text",
  "ngram-cache": "Cache of past replies",
};

/**
 * The load panel for text and vision models: every llama.cpp load parameter, a live memory
 * fit estimate, named presets, and "reset to tuned". The Load button in the drawer header uses
 * what this panel holds with Tweak Studio's tweaks laid over it (`effective`); a field a tweak
 * decides shows the tweak's value, read-only, and says so.
 */
export function LoadPanel(props: {
  model: LibraryModel;
  draft: Accessor<LoadParams>;
  setDraft: (p: LoadParams) => void;
  baseline: Accessor<LoadParams>;
  /** The draft with the active tweaks applied: what a load will use. */
  effective: Accessor<LoadParams>;
  tweaks: Accessor<Tweaks>;
  tweakCount: Accessor<number>;
  /** Longest context allowed: the file's maximum, or more with RoPE scaling switched on. */
  contextMax: Accessor<number>;
  errors: Accessor<LoadErrors>;
  onTweaks: () => void;
}): JSX.Element {
  const store = useModels();
  const service = useService();
  const m = () => props.model;
  const hw = () => store.hw();
  const byTweak = createMemo(() => tweakedFields(props.tweaks()));
  const tweaked = (k: LoadField) => byTweak().has(k);
  const set = <K extends LoadField>(key: K, value: LoadParams[K]) => props.setDraft({ ...props.draft(), [key]: value });
  const changed = createMemo(() => new Set(changedFields(props.draft(), props.baseline())));
  const isChanged = (k: LoadField) => changed().has(k);
  const reset = (k: LoadField) => set(k, props.baseline()[k]);
  const drafts = createMemo(() => draftCandidates(m(), store.library() ?? []));
  const draftModel = () => (props.draft().draftModelId ? store.byId().get(props.draft().draftModelId ?? "") : undefined);
  const fit = createMemo<FitEstimate | null>(() => {
    const h = hw();
    return h ? estimateFit(m(), props.effective(), h, draftModel() ?? null) : null;
  });
  const baselineName = () => (m().tunedParams ? "tuned" : "starting");

  // The engine's own placement for these settings (desktop): asked again 300 ms after the context
  // or the GPU layer count stops changing.
  const [placeKey, setPlaceKey] = createSignal<{ id: string; ctx: number; gpu: number } | null>(null);
  let placeTimer: ReturnType<typeof setTimeout> | undefined;
  createEffect(() => {
    const key = { id: m().id, ctx: props.effective().contextLength, gpu: props.effective().gpuLayers };
    clearTimeout(placeTimer);
    placeTimer = setTimeout(() => setPlaceKey(key), 300);
  });
  onCleanup(() => clearTimeout(placeTimer));
  const [placement] = createResource(placeKey, (k) => (service.placementPreview ? service.placementPreview(k.id, k.ctx, k.gpu).catch(() => null) : Promise.resolve(null)));

  // ---- Presets ----
  const [presets, { mutate }] = createResource(
    () => m().id,
    (id) => service.listPresets(id).catch(() => [] as LoadPreset[]),
  );
  const same = (a: LoadParams, b: LoadParams) => changedFields(a, b).length === 0;
  const current = createMemo(() => {
    if (same(props.draft(), props.baseline())) return "baseline";
    return presets()?.find((p) => same(p.params, props.draft()))?.id ?? "custom";
  });
  const [naming, setNaming] = createSignal(false);
  const [name, setName] = createSignal("");
  const nameError = () => (name() === "" ? null : profileNameError(name(), (presets() ?? []).map((p) => p.name)));
  let saving = false;
  const save = async () => {
    const err = profileNameError(name(), (presets() ?? []).map((p) => p.name));
    if (err || saving) return;
    saving = true;
    try {
      mutate(await service.savePreset(m().id, name().trim(), props.draft()));
      notify(`Preset saved: ${name().trim()}`);
      setNaming(false);
      setName("");
    } catch {
      notify("The preset was not saved", "Check the settings for errors and try again.", "error");
    } finally {
      saving = false;
    }
  };
  const remove = async (id: string) => {
    const p = presets()?.find((x) => x.id === id);
    try {
      mutate(await service.deletePreset(m().id, id));
      if (p) notify(`Preset deleted: ${p.name}`);
    } catch {
      notify("The preset was not deleted", undefined, "error");
    }
  };

  const moe = () => m().architecture === "moe";
  const summaryText = () => {
    const base = changed().size > 0 ? "Custom" : m().tunedParams ? "Tuned" : "Starting";
    const n = props.tweakCount();
    return n > 0 ? `${base} + ${n} ${n === 1 ? "tweak" : "tweaks"}` : `${base} settings`;
  };

  return (
    <div class="load-panel">
      <div class="preset-bar">
        <label class="preset-bar__pick">
          <span class="field__label">Settings</span>
          <span class="select select--sm">
            <select
              class="input"
              value={current()}
              onChange={(e) => {
                const v = e.currentTarget.value;
                if (v === "baseline") props.setDraft({ ...props.baseline() });
                const p = presets()?.find((x) => x.id === v);
                if (p) props.setDraft({ ...p.params });
              }}
            >
              <option value="baseline">{m().tunedParams ? "Tuned (measured)" : "Starting settings"}</option>
              <For each={presets() ?? []}>{(p) => <option value={p.id}>{p.name}</option>}</For>
              <Show when={current() === "custom"}>
                <option value="custom" disabled>
                  Custom, unsaved
                </option>
              </Show>
            </select>
          </span>
        </label>
        <div class="preset-bar__actions">
          <Show
            when={naming()}
            fallback={
              <button type="button" class="btn btn--tertiary btn--xs" data-press onClick={() => setNaming(true)}>
                <Icon name="plus" size={14} />
                Save as preset
              </button>
            }
          >
            <div class="inline-form">
              <input
                class="input input--sm"
                aria-label="Preset name"
                placeholder="Preset name"
                maxLength={40}
                value={name()}
                aria-invalid={nameError() ? "true" : undefined}
                ref={(el) => requestAnimationFrame(() => el.focus())}
                onInput={(e) => setName(e.currentTarget.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void save();
                  if (e.key === "Escape") {
                    e.stopPropagation();
                    setNaming(false);
                  }
                }}
              />
              <button type="button" class="btn btn--secondary btn--xs" data-press onClick={() => void save()} disabled={name().trim() === "" || nameError() !== null || Object.keys(props.errors()).length > 0}>
                Save
              </button>
              <button type="button" class="btn btn--tertiary btn--xs" data-press onClick={() => setNaming(false)}>
                Cancel
              </button>
            </div>
          </Show>
          <Show when={current() !== "baseline" && current() !== "custom"}>
            <button type="button" class="btn btn--tertiary btn--xs" data-press onClick={() => void remove(current())}>
              <Icon name="trash" size={14} />
              Delete preset
            </button>
          </Show>
          <button type="button" class="btn btn--tertiary btn--xs" data-press disabled={current() === "baseline"} onClick={() => props.setDraft({ ...props.baseline() })}>
            <Icon name="undo" size={14} />
            Reset to {baselineName()}
          </button>
        </div>
      </div>
      <Show when={nameError()}>
        <p class="field__error">{nameError()}</p>
      </Show>

      <Show
        when={fit()}
        fallback={
          <p class="notice notice--inline fit-unknown" role="status">
            <Icon name="info" size={18} />
            <span>
              {store.hwState() === "detecting"
                ? "Detecting this machine…"
                : "This machine was not detected, so there is no memory estimate. The starting settings run on the processor only."}
            </span>
            <Show when={store.hwState() === "missing"}>
              <button type="button" class="btn btn--secondary btn--xs" data-press onClick={() => store.redetect()}>
                <Icon name="refresh" size={14} />
                Detect again
              </button>
            </Show>
          </p>
        }
      >
        {(f) => <FitBar fit={f()} summary={summaryText()} />}
      </Show>
      <Show when={placement()}>{(p) => <PlacementCard plan={p()} title="Where it would run" />}</Show>

      <Show when={props.tweakCount() > 0}>
        <p class="hint tweak-note">
          <Icon name="sliders" size={16} />
          <span>
            {`${props.tweakCount()} ${props.tweakCount() === 1 ? "tweak" : "tweaks"} from Tweak Studio apply on top of these settings, and the estimate includes them. No tune has measured this combination. `}
            <button type="button" class="text-link text-link--btn" onClick={() => props.onTweaks()}>
              Open Tweak Studio
            </button>
          </span>
        </p>
      </Show>

      <Show when={m().projector}>
        {(p) => (
          <p class="hint">
            <Icon name="link" size={16} />
            Loads with its <Term k="mmproj" /> <span class="mono">{p().file}</span>, paired automatically.
          </p>
        )}
      </Show>

      <div class="load-groups">
        <fieldset class="load-group">
          <legend class="load-group__title">Memory</legend>
          <Field
            label="Context length"
            term="context"
            changed={isChanged("contextLength")}
            onReset={() => reset("contextLength")}
            error={props.errors().contextLength}
            hint={
              tweaked("contextLength")
                ? "Set by Tweak Studio"
                : props.contextMax() > m().contextMax
                  ? `Up to ${props.contextMax().toLocaleString("en-US")} tokens with RoPE scaling`
                  : `Up to ${m().contextMax.toLocaleString("en-US")} tokens`
            }
          >
            {(id, a11y) => (
              <RangeNumber
                id={id}
                a11y={a11y}
                value={props.effective().contextLength}
                min={LIMITS.contextMin}
                max={Math.max(LIMITS.contextMin, props.contextMax())}
                step={256}
                unit="tokens"
                readOnly={tweaked("contextLength")}
                onChange={(v) => set("contextLength", v)}
              />
            )}
          </Field>
          <Field
            label="GPU layers"
            term="offload"
            changed={isChanged("gpuLayers")}
            onReset={() => reset("gpuLayers")}
            error={props.errors().gpuLayers}
            hint={hw()?.gpu ? `of ${m().arch.layers}` : hw() ? "No usable GPU found" : "GPU not detected"}
          >
            {(id, a11y) => <RangeNumber id={id} a11y={a11y} value={props.draft().gpuLayers} min={0} max={m().arch.layers} step={1} unit="layers" onChange={(v) => set("gpuLayers", v)} />}
          </Field>
          <Show when={moe()}>
            <Field label="Experts on CPU" term="moeCpu" changed={isChanged("moeCpuLayers")} onReset={() => reset("moeCpuLayers")} error={props.errors().moeCpuLayers} hint={`${m().arch.experts} experts per layer, ${m().arch.activeExperts} used per token`}>
              {(id, a11y) => <RangeNumber id={id} a11y={a11y} value={props.draft().moeCpuLayers} min={0} max={m().arch.layers} step={1} unit="layers" onChange={(v) => set("moeCpuLayers", v)} />}
            </Field>
          </Show>
          <div class="field-pair">
            <Field label="K cache" term="kvCache" changed={isChanged("kvCacheK")} onReset={() => reset("kvCacheK")} error={props.errors().kvCacheK} hint={tweaked("kvCacheK") ? "Set by Tweak Studio" : undefined}>
              {(id, a11y) => <Choice id={id} a11y={a11y} value={props.effective().kvCacheK} options={KV_TYPES} disabled={tweaked("kvCacheK")} onChange={(v) => set("kvCacheK", v as LoadParams["kvCacheK"])} />}
            </Field>
            <Field label="V cache" changed={isChanged("kvCacheV")} onReset={() => reset("kvCacheV")} error={props.errors().kvCacheV} hint={tweaked("kvCacheV") ? "Set by Tweak Studio" : undefined}>
              {(id, a11y) => <Choice id={id} a11y={a11y} value={props.effective().kvCacheV} options={KV_TYPES} disabled={tweaked("kvCacheV")} onChange={(v) => set("kvCacheV", v as LoadParams["kvCacheV"])} />}
            </Field>
          </div>
          <Field label="Flash attention" group term="flashAttention" changed={isChanged("flashAttention")} onReset={() => reset("flashAttention")} error={props.errors().flashAttention}>
            {(id, a11y) => (
              <Segmented
                id={id}
                a11y={a11y}
                value={props.draft().flashAttention}
                options={[
                  ["auto", "Auto"],
                  ["on", "On"],
                  ["off", "Off"],
                ]}
                onChange={(v) => set("flashAttention", v as LoadParams["flashAttention"])}
              />
            )}
          </Field>
        </fieldset>

        <fieldset class="load-group">
          <legend class="load-group__title">Processor</legend>
          <Field
            label="Threads"
            term="threads"
            changed={isChanged("threads")}
            onReset={() => reset("threads")}
            error={props.errors().threads}
            hint={hw() ? `${hw()?.cpu.cores} cores, ${hw()?.cpu.threads} threads` : "Processor not detected"}
          >
            {(id, a11y) => <NumberInput id={id} a11y={a11y} value={props.draft().threads} min={1} max={maxThreadsFor(hw())} step={1} onChange={(v) => set("threads", v)} />}
          </Field>
          <Field label="CPU cores" changed={isChanged("cpuAffinity") || isChanged("cpuCores")} onReset={() => props.setDraft({ ...props.draft(), cpuAffinity: props.baseline().cpuAffinity, cpuCores: props.baseline().cpuCores })} error={props.errors().cpuAffinity ?? props.errors().cpuCores}>
            {(id, a11y) => (
              <div class="field-inline">
                <Choice
                  id={id}
                  a11y={a11y}
                  value={props.draft().cpuAffinity}
                  options={["all", "performance", "custom"]}
                  labels={{ all: "All cores", performance: "Performance cores only", custom: "Custom list" }}
                  onChange={(v) =>
                    props.setDraft({ ...props.draft(), cpuAffinity: v as LoadParams["cpuAffinity"], cpuCores: v === "custom" ? props.draft().cpuCores || `0-${Math.max(0, (hw()?.cpu.cores ?? 4) - 1)}` : "" })
                  }
                />
                <Show when={props.draft().cpuAffinity === "custom"}>
                  <input
                    class="input input--sm mono"
                    aria-label="Core list"
                    placeholder="0-5,8"
                    maxLength={LIMITS.coresMaxLength}
                    value={props.draft().cpuCores}
                    aria-invalid={props.errors().cpuCores ? "true" : undefined}
                    aria-describedby={a11y.describedBy || undefined}
                    onInput={(e) => set("cpuCores", e.currentTarget.value.replace(/\s+/g, ""))}
                  />
                </Show>
              </div>
            )}
          </Field>
          <div class="load-switches">
            <SettingSwitch id="load-mmap" label="Memory mapping" note="Reads the file from disk as needed instead of copying it into memory first." checked={props.draft().mmap} onChange={(v) => set("mmap", v)} />
            <SettingSwitch id="load-mlock" label="Keep in memory" note="Stops Windows from paging the model out to disk. Uses more RAM." checked={props.draft().mlock} onChange={(v) => set("mlock", v)} />
          </div>
          <p class="hint">
            <Term k="mmap" />: on by default.
          </p>
        </fieldset>

        <fieldset class="load-group">
          <legend class="load-group__title">Throughput</legend>
          <div class="field-pair">
            <Field label="Batch size" term="batch" changed={isChanged("batchSize")} onReset={() => reset("batchSize")} error={props.errors().batchSize}>
              {(id, a11y) => <NumberInput id={id} a11y={a11y} value={props.draft().batchSize} min={LIMITS.batchMin} max={LIMITS.batchMax} step={32} onChange={(v) => set("batchSize", v)} />}
            </Field>
            <Field label="Micro-batch" changed={isChanged("ubatchSize")} onReset={() => reset("ubatchSize")} error={props.errors().ubatchSize} hint="Slice sent at once">
              {(id, a11y) => <NumberInput id={id} a11y={a11y} value={props.draft().ubatchSize} min={LIMITS.batchMin} max={LIMITS.batchMax} step={32} onChange={(v) => set("ubatchSize", v)} />}
            </Field>
          </div>
          <Field label="Speculative decoding" term="speculative" changed={isChanged("speculative") || isChanged("ngramStrategy") || isChanged("draftModelId") || isChanged("draftMax")} onReset={() => props.setDraft({ ...props.draft(), speculative: props.baseline().speculative, ngramStrategy: props.baseline().ngramStrategy, draftModelId: props.baseline().draftModelId, draftMax: props.baseline().draftMax })} error={props.errors().speculative ?? props.errors().draftModelId ?? props.errors().draftMax}>
            {(id, a11y) => (
              <div class="field-stack">
                <Choice
                  id={id}
                  a11y={a11y}
                  value={props.draft().speculative}
                  options={drafts().length > 0 ? ["off", "ngram", "draft"] : ["off", "ngram"]}
                  labels={{ off: "Off", ngram: "N-gram (no draft model)", draft: "Draft model" }}
                  onChange={(v) => props.setDraft({ ...props.draft(), speculative: v as LoadParams["speculative"], draftModelId: v === "draft" ? (drafts()[0]?.id ?? null) : null })}
                />
                <Show when={props.draft().speculative === "ngram"}>
                  <span class="field-inline">
                    <Choice id={`${id}-strategy`} label="N-gram strategy" value={props.draft().ngramStrategy} options={NGRAM_STRATEGIES} labels={NGRAM_LABELS} onChange={(v) => set("ngramStrategy", v as LoadParams["ngramStrategy"])} />
                    <Term k="ngram">{""}</Term>
                  </span>
                </Show>
                <Show when={props.draft().speculative === "draft"}>
                  <span class="select select--sm">
                    <select class="input" aria-label="Draft model" value={props.draft().draftModelId ?? ""} onChange={(e) => set("draftModelId", e.currentTarget.value)}>
                      <For each={drafts()}>{(d) => <option value={d.id}>{`${d.name} ${d.quant} (${d.sizeGiB.toFixed(1)} GiB)`}</option>}</For>
                    </select>
                  </span>
                </Show>
                <Show when={props.draft().speculative !== "off"}>
                  <label class="field-inline">
                    <span class="field__label">Draft up to</span>
                    <NumberInput value={props.draft().draftMax} min={LIMITS.draftMaxMin} max={LIMITS.draftMaxMax} step={1} label="Draft tokens" onChange={(v) => set("draftMax", v)} />
                    <span class="muted">tokens</span>
                  </label>
                </Show>
              </div>
            )}
          </Field>
          <p class="hint">
            <Icon name="info" size={16} />
            The draft only proposes; the model verifies every token. Speculation changes speed, not the output distribution.
          </p>
        </fieldset>
      </div>
    </div>
  );
}

// ---- Fit estimate ----

const STATUS_TEXT: Readonly<Record<FitStatus, string>> = { fits: "Fits", tight: "Tight", over: "Does not fit" };

/** The memory estimate, led by one plain reading ("Tuned + 2 tweaks · Fits, 7.4 of 8.0 GiB video
 *  memory") with the two bars as the detail beneath it. */
export function FitBar(props: { fit: FitEstimate; summary: string }): JSX.Element {
  const tightest = () => (props.fit.vram.capacity > 0 ? props.fit.vram : props.fit.ram);
  return (
    <div class="fit" data-status={props.fit.status}>
      <div class="fit__head">
        <span class="fit__title">{props.summary}</span>
        <span class="fit-status" data-status={props.fit.status} role="status" aria-live="polite">
          <Icon name={props.fit.status === "over" ? "alert" : "check"} size={14} />
          {STATUS_TEXT[props.fit.status]}
          <span class="fit-status__amount mono">{`${tightest().total.toFixed(1)} of ${tightest().capacity.toFixed(1)} GiB ${props.fit.vram.capacity > 0 ? "VRAM" : "RAM"}`}</span>
        </span>
        <span class="fit__note">Memory estimate: arithmetic from the file and these settings; a tune measures the real thing.</span>
      </div>
      <Meter label="VRAM" use={props.fit.vram} />
      <Meter label="RAM" use={props.fit.ram} />
      <ul class="fit-legend" aria-label="Legend">
        <li data-seg="weights">Weights</li>
        <li data-seg="kv">KV cache</li>
        <li data-seg="compute">Compute</li>
        <li data-seg="extra">Draft, projector</li>
      </ul>
    </div>
  );
}

function Meter(props: { label: string; use: MemoryUse }): JSX.Element {
  const pct = (v: number) => (props.use.capacity > 0 ? Math.min(100, Math.max(0, (v / props.use.capacity) * 100)) : v > 0 ? 100 : 0);
  const u = () => props.use;
  const cumulative = () => {
    const a = u().weights;
    const b = a + u().kv;
    const c = b + u().compute;
    const d = c + u().extra;
    return [d, c, b, a].map(pct);
  };
  const over = () => Math.max(0, u().total - u().capacity);
  const text = () =>
    `${u().total.toFixed(1)} of ${u().capacity.toFixed(1)} GiB, ${STATUS_TEXT[u().status].toLowerCase()}. Weights ${u().weights.toFixed(1)}, KV cache ${u().kv.toFixed(1)}, compute ${u().compute.toFixed(1)}, draft and projector ${u().extra.toFixed(1)} GiB.`;
  return (
    <div class="meter-row" data-status={u().status}>
      <span class="meter-row__label">{props.label}</span>
      <div class="meter" role="meter" aria-label={`Estimated ${props.label} use`} aria-valuemin={0} aria-valuemax={u().capacity} aria-valuenow={Math.min(u().total, u().capacity)} aria-valuetext={text()}>
        <For each={["extra", "compute", "kv", "weights"] as const}>
          {(seg, i) => <span class="meter__seg" data-seg={seg} style={{ transform: `translateX(${cumulative()[i()]! - 100}%)` }} />}
        </For>
      </div>
      <span class="meter-row__value mono">
        {u().total.toFixed(1)}
        <span class="unit">/ {u().capacity.toFixed(1)} GiB</span>
        <Show when={over() > 0}>
          <span class="meter-row__over">+{over().toFixed(1)} over</span>
        </Show>
      </span>
    </div>
  );
}

// ---- Controls ----

function Field(props: {
  label: string;
  term?: GlossaryKey;
  changed: boolean;
  onReset: () => void;
  error?: string | undefined;
  hint?: string | undefined;
  /** The control is a group (radios), labelled by the heading rather than a <label for>. */
  group?: boolean;
  children: (id: string, a11y: FieldA11y) => JSX.Element;
}): JSX.Element {
  const id = `f-${createUniqueId()}`;
  // The control is described by the hint and, while there is one, the error; it is marked invalid
  // with it. Getters keep both reactive inside the child.
  const a11y: FieldA11y = {
    get describedBy() {
      return [props.hint ? `${id}-hint` : "", props.error ? `${id}-err` : ""].filter(Boolean).join(" ");
    },
    get invalid() {
      return props.error !== undefined;
    },
  };
  return (
    <div class="lfield" classList={{ "is-changed": props.changed, "is-invalid": props.error !== undefined }}>
      <div class="lfield__head">
        <Show
          when={!props.group}
          fallback={
            <span class="lfield__label" id={`${id}-label`}>
              {props.label}
            </span>
          }
        >
          <label class="lfield__label" for={id}>
            {props.label}
          </label>
        </Show>
        <Show when={props.term}>{(k) => <Term k={k()}>{""}</Term>}</Show>
        <Show when={props.changed}>
          <span class="lfield__changed">Changed</span>
          <button type="button" class="btn btn--icon btn--xs lfield__reset" aria-label={`Reset ${props.label}`} data-press onClick={() => props.onReset()}>
            <Icon name="undo" size={14} />
          </button>
        </Show>
        <Show when={props.hint}>
          <span class="lfield__hint" id={`${id}-hint`}>
            {props.hint}
          </span>
        </Show>
      </div>
      {props.children(id, a11y)}
      {/* A polite live region that is always present, so a new error is announced once, after the
          current speech, instead of interrupting every keystroke. */}
      <div class="lfield__msg" aria-live="polite">
        <Show when={props.error}>
          <p class="field__error" id={`${id}-err`}>
            <Icon name="alert" size={14} />
            {props.error}
          </p>
        </Show>
      </div>
    </div>
  );
}

interface FieldA11y {
  readonly describedBy: string;
  readonly invalid: boolean;
}

/** A number field that keeps what the user typed (so an empty or partial entry is not snapped back)
 *  and reports a number; anything unparsable reports NaN, which validation rejects. */
function NumberInput(props: {
  id?: string;
  a11y?: FieldA11y | undefined;
  value: number;
  min: number;
  max: number;
  step: number;
  label?: string;
  readOnly?: boolean | undefined;
  onChange: (n: number) => void;
}): JSX.Element {
  const [text, setText] = createSignal(String(props.value));
  createEffect(() => {
    const v = props.value;
    if (Number(text()) !== v) setText(Number.isFinite(v) ? String(v) : text());
  });
  return (
    <input
      id={props.id}
      class="input input--sm mono lnum"
      type="number"
      inputmode="numeric"
      min={props.min}
      max={props.max}
      step={props.step}
      aria-label={props.label}
      aria-invalid={props.a11y?.invalid ? "true" : undefined}
      aria-describedby={props.a11y?.describedBy || undefined}
      readOnly={props.readOnly}
      value={text()}
      onInput={(e) => {
        const t = e.currentTarget.value;
        setText(t);
        props.onChange(t.trim() === "" ? Number.NaN : Number(t));
      }}
    />
  );
}

function RangeNumber(props: {
  id: string;
  a11y?: FieldA11y | undefined;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  readOnly?: boolean | undefined;
  onChange: (n: number) => void;
}): JSX.Element {
  const pct = () => (props.max > props.min ? ((Math.min(props.max, Math.max(props.min, props.value || 0)) - props.min) / (props.max - props.min)) * 100 : 0);
  return (
    <div class="range-field" classList={{ "is-readonly": props.readOnly === true }}>
      <span class="range">
        <input
          class="range__input"
          type="range"
          min={props.min}
          max={props.max}
          step={props.step}
          value={Number.isFinite(props.value) ? props.value : props.min}
          aria-hidden="true"
          tabindex={-1}
          disabled={props.readOnly}
          onInput={(e) => props.onChange(Number(e.currentTarget.value))}
        />
        <span class="range__fill" aria-hidden="true" style={{ transform: `translateX(${pct() - 100}%)` }} />
      </span>
      <NumberInput id={props.id} a11y={props.a11y} value={props.value} min={props.min} max={props.max} step={props.step} readOnly={props.readOnly} onChange={props.onChange} />
      <span class="range-field__unit">{props.unit}</span>
    </div>
  );
}

function Choice(props: {
  id: string;
  a11y?: FieldA11y | undefined;
  value: string;
  options: readonly string[];
  labels?: Readonly<Record<string, string>>;
  label?: string;
  disabled?: boolean | undefined;
  onChange: (v: string) => void;
}): JSX.Element {
  return (
    <span class="select select--sm">
      <select
        id={props.id}
        class="input"
        aria-label={props.label}
        aria-invalid={props.a11y?.invalid ? "true" : undefined}
        aria-describedby={props.a11y?.describedBy || undefined}
        disabled={props.disabled}
        value={props.value}
        onChange={(e) => props.onChange(e.currentTarget.value)}
      >
        <For each={props.options}>{(o) => <option value={o}>{props.labels?.[o] ?? o}</option>}</For>
      </select>
    </span>
  );
}

function Segmented(props: { id: string; a11y?: FieldA11y | undefined; value: string; options: ReadonlyArray<readonly [string, string]>; onChange: (v: string) => void }): JSX.Element {
  const name = `seg-${createUniqueId()}`;
  return (
    <div
      class="segmented segmented--sm"
      role="radiogroup"
      id={props.id}
      aria-labelledby={`${props.id}-label`}
      aria-invalid={props.a11y?.invalid ? "true" : undefined}
      aria-describedby={props.a11y?.describedBy || undefined}
    >
      <For each={props.options}>
        {([value, label]) => (
          <label class="segmented__option" data-press>
            <input type="radio" name={name} value={value} checked={props.value === value} onChange={() => props.onChange(value)} />
            <span>{label}</span>
          </label>
        )}
      </For>
    </div>
  );
}
