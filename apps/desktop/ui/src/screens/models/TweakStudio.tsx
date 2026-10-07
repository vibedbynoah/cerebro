import { createMemo, createResource, createSignal, createUniqueId, For, onCleanup, Show, type Accessor, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { IllustrativeFlag } from "../../components/Screen";
import { SettingSwitch } from "../../components/SettingSwitch";
import { Term } from "../../components/Term";
import { KV_TYPES } from "../../lib/load-params";
import { notify } from "../../lib/toast";
import {
  activeTweakCount,
  checkTemplate,
  diffLines,
  EMPTY_TWEAKS,
  exportProfile,
  importProfile,
  profileNameError,
  ROPE_SCALINGS,
  sanitizeTweaks,
  SAMPLER_LIMITS,
  stopTokenError,
  TWEAK_LIMITS,
} from "../../lib/tweaks";
import { useService } from "../../service/context";
import type { KvCacheType, LibraryModel, ModelDetail, PatchProfile, RopeScaling, Sampler, Tweaks } from "../../service/types";
import { useModels } from "./store";

/** Typing in the template box saves at most this often (and once more when the drawer closes). */
const SAVE_DEBOUNCE_MS = 300;

/**
 * Tweak Studio: small, reversible, user-owned tweaks applied when the model loads, without
 * re-downloading or rewriting the file. Saved as named patch profiles keyed by the file's SHA-256,
 * which can be copied as text and imported on another machine.
 */
export function TweakStudio(props: { model: LibraryModel; detail: ModelDetail }): JSX.Element {
  const service = useService();
  const store = useModels();
  const sha = props.model.sha256;
  const [tweaks, setTweaksLocal] = createSignal<Tweaks>({ ...EMPTY_TWEAKS, stopTokens: [] });
  const [fixes, setFixesLocal] = createSignal<ReadonlySet<string>>(new Set());
  const [ready, setReady] = createSignal(false);
  const [factorBad, setFactorBad] = createSignal(false);
  let alive = true;
  onCleanup(() => (alive = false));

  // The tweaks in effect live in the service, keyed by the file's hash, so they are still there when
  // the drawer is closed and opened again, and the next load applies them. The Models store holds
  // the same patch, so the load panel, its estimate and the Load button see each change at once.
  void store.ensurePatch(sha).then((p) => {
    if (!alive) return;
    setTweaksLocal({ ...p.tweaks, stopTokens: [...p.tweaks.stopTokens] });
    setFixesLocal(new Set(p.fixes));
    setReady(true);
  });

  // Saving: debounced (a keystroke in a 32 KB template is not a write), flushed when the drawer
  // closes, and serialized so writes reach the service in order and the last one wins.
  let saveSeq = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let dirty = false;
  let chain: Promise<unknown> = Promise.resolve();
  const flush = () => {
    clearTimeout(timer);
    timer = undefined;
    if (!dirty) return;
    dirty = false;
    // An unfinished edit (an emptied template, say) stays on screen and is saved once it is valid.
    const clean = sanitizeTweaks(tweaks());
    if (!clean) return;
    const patch = { tweaks: clean, fixes: [...fixes()] };
    const seq = ++saveSeq;
    chain = chain.then(() =>
      service.setActivePatch(sha, patch).catch(() => {
        if (seq !== saveSeq) return;
        notify("That tweak was not saved", "It will not apply until it is saved. Try again.", "error");
        store.reloadPatch(sha);
      }),
    );
  };
  const persist = () => {
    const clean = sanitizeTweaks(tweaks());
    if (clean) store.putPatch(sha, { tweaks: clean, fixes: [...fixes()] });
    dirty = true;
    clearTimeout(timer);
    timer = setTimeout(flush, SAVE_DEBOUNCE_MS);
  };
  onCleanup(flush);
  const setTweaks = (t: Tweaks) => {
    setTweaksLocal(t);
    persist();
  };
  const setFixes = (f: ReadonlySet<string>) => {
    setFixesLocal(f);
    persist();
  };
  const [profiles, { mutate }] = createResource(
    () => sha,
    (sha) => service.listPatchProfiles(sha).catch(() => [] as PatchProfile[]),
  );
  const count = () => activeTweakCount(tweaks()) + fixes().size;
  const patch = (p: Partial<Tweaks>) => setTweaks({ ...tweaks(), ...p });
  const revertAll = () => {
    setTweaks({ ...EMPTY_TWEAKS, stopTokens: [] });
    setFixes(new Set<string>());
    notify("All tweaks reverted", "The model loads exactly as its file says.");
  };

  const fileCtx = () => {
    const e = props.detail.gguf?.find((x) => x.key.endsWith(".context_length"));
    return typeof e?.value === "number" ? e.value : props.model.contextMax;
  };

  return (
    <Show when={ready()} fallback={<DetailSkeleton />}>
    <div class="studio">
      <div class="studio__bar">
        <p class="studio__summary">
          <span class="studio__count mono">{count()}</span> {count() === 1 ? "change" : "changes"} on top of the file. Applied when the model loads; the file itself never changes.
        </p>
        <button type="button" class="btn btn--tertiary btn--xs" data-press disabled={count() === 0} onClick={revertAll}>
          <Icon name="undo" size={14} />
          Revert all
        </button>
      </div>

      <Show when={props.detail.quirks.length > 0}>
        <section class="studio-card card" aria-labelledby="fixes-title">
          <div class="studio-card__head">
            <h3 class="studio-card__title" id="fixes-title">
              Known fixes for this model
            </h3>
            <IllustrativeFlag />
          </div>
          <p class="studio-card__desc">From Cerebro's quirks list for the {props.model.family} family. Each one is off until you switch it on.</p>
          <div class="settings-card fix-list">
            <For each={props.detail.quirks}>
              {(q) => (
                <SettingSwitch
                  id={`fix-${q.id}`}
                  label={q.title}
                  note={`${q.problem} ${q.change}`}
                  checked={fixes().has(q.id)}
                  onChange={(on) => {
                    const next = new Set(fixes());
                    if (on) next.add(q.id);
                    else next.delete(q.id);
                    setFixes(next);
                  }}
                />
              )}
            </For>
          </div>
        </section>
      </Show>

      <section class="studio-card card" aria-labelledby="tweaks-title">
        <div class="studio-card__head">
          <h3 class="studio-card__title" id="tweaks-title">
            Tweaks
          </h3>
        </div>
        <div class="tweak-list">
          <Tweak
            title="Quantized KV cache"
            term="kvCache"
            active={tweaks().kvQuant !== null}
            onReset={() => patch({ kvQuant: null })}
            note="Overrides the K and V cache type in Load settings: q8_0 roughly halves the conversation memory, q4_0 roughly quarters it. A quantized V cache needs flash attention on or auto."
          >
            <span class="select select--sm">
              <select class="input" aria-label="KV cache type" value={tweaks().kvQuant ?? "file"} onChange={(e) => patch({ kvQuant: e.currentTarget.value === "file" ? null : (e.currentTarget.value as KvCacheType) })}>
                <option value="file">Use the load settings (f16 unless changed)</option>
                <For each={KV_TYPES.filter((t) => t !== "f16")}>{(t) => <option value={t}>{t}</option>}</For>
              </select>
            </span>
          </Tweak>

          <Tweak title="Context length and RoPE" term="rope" active={tweaks().contextOverride !== null || tweaks().rope !== null} onReset={() => patch({ contextOverride: null, rope: null })} note={`The file says ${fileCtx().toLocaleString("en-US")} tokens. Stretch it only the way the model's authors specify.`}>
            <div class="tweak-rope">
              <label class="field-inline">
                <span class="field__label">Context</span>
                <IntInput
                  label="Context length override"
                  value={tweaks().contextOverride}
                  placeholder={fileCtx().toLocaleString("en-US")}
                  min={TWEAK_LIMITS.contextMin}
                  max={TWEAK_LIMITS.contextMax}
                  onChange={(v) => patch({ contextOverride: v })}
                />
              </label>
              <label class="field-inline">
                <span class="field__label">Scaling</span>
                <span class="select select--sm">
                  <select
                    class="input"
                    aria-label="RoPE scaling"
                    value={tweaks().rope?.scaling ?? "file"}
                    onChange={(e) => {
                      const v = e.currentTarget.value;
                      patch({ rope: v === "file" ? null : { scaling: v as RopeScaling, factor: tweaks().rope?.factor ?? 2, originalContext: tweaks().rope?.originalContext ?? fileCtx() } });
                    }}
                  >
                    <option value="file">As the file says</option>
                    <For each={ROPE_SCALINGS}>{(s) => <option value={s}>{s === "none" ? "None" : s === "yarn" ? "YaRN" : "Linear"}</option>}</For>
                  </select>
                </span>
              </label>
              <Show when={tweaks().rope && tweaks().rope?.scaling !== "none"}>
                <label class="field-inline">
                  <span class="field__label">Factor</span>
                  <input
                    class="input input--sm mono lnum"
                    type="number"
                    min={1}
                    max={TWEAK_LIMITS.ropeFactorMax}
                    step={0.5}
                    aria-label="RoPE scale factor"
                    value={tweaks().rope?.factor ?? 2}
                    aria-invalid={factorBad() ? "true" : undefined}
                    onInput={(e) => {
                      const f = Number(e.currentTarget.value);
                      const r = tweaks().rope;
                      const ok = e.currentTarget.value.trim() !== "" && Number.isFinite(f) && f >= 1 && f <= TWEAK_LIMITS.ropeFactorMax;
                      setFactorBad(!ok);
                      if (r && ok) patch({ rope: { ...r, factor: f } });
                    }}
                  />
                  <Show when={factorBad()}>
                    <span class="field__error">{`1 to ${TWEAK_LIMITS.ropeFactorMax}`}</span>
                  </Show>
                </label>
              </Show>
              <Show when={props.detail.author.rope}>
                {(a) => (
                  <button
                    type="button"
                    class="btn btn--tertiary btn--xs"
                    data-press
                    onClick={() => patch({ rope: { ...a() }, contextOverride: Math.round(a().originalContext * a().factor) })}
                  >
                    Use the authors' values ({a().scaling === "yarn" ? "YaRN" : a().scaling}, ×{a().factor})
                  </button>
                )}
              </Show>
            </div>
          </Tweak>

          <Tweak title="Stop tokens" term="stopToken" active={tweaks().stopTokens.length > 0} onReset={() => patch({ stopTokens: [] })} note="Extra text that ends a reply, added to the ones the file lists.">
            <StopTokens tokens={tweaks().stopTokens} onChange={(stopTokens) => patch({ stopTokens })} />
          </Tweak>

          <Tweak title="Default sampler" term="sampler" active={tweaks().sampler !== null} onReset={() => patch({ sampler: null })} note="Used when a chat does not set its own. Off: chat uses its own explicit settings.">
            <SamplerFields sampler={tweaks().sampler} author={props.detail.author.sampler} onChange={(sampler) => patch({ sampler })} />
          </Tweak>

          <Show when={props.detail.chatTemplate !== null}>
            <Tweak title="Chat template" term="chatTemplate" active={tweaks().chatTemplate !== null} onReset={() => patch({ chatTemplate: null })} note="Fix or replace the template. The diff shows exactly what changes.">
              <TemplateEditor original={props.detail.chatTemplate ?? ""} value={tweaks().chatTemplate} onChange={(chatTemplate) => patch({ chatTemplate })} />
            </Tweak>
          </Show>
        </div>
      </section>

      <Profiles
        model={props.model}
        profiles={() => profiles() ?? []}
        current={() => ({ tweaks: tweaks(), fixes: [...fixes()] })}
        knownFixes={() => new Set(props.detail.quirks.map((q) => q.id))}
        onApply={(p) => {
          setTweaks({ ...p.tweaks, stopTokens: [...p.tweaks.stopTokens] });
          setFixes(new Set(p.fixes.filter((f) => props.detail.quirks.some((q) => q.id === f))));
          notify(`Profile applied: ${p.name}`);
        }}
        onSaved={mutate}
      />
    </div>
    </Show>
  );
}

function DetailSkeleton(): JSX.Element {
  return (
    <>
      <div class="detail-skeleton" aria-hidden="true">
        <span class="sk-line sk-line--name" />
        <span class="sk-line" />
        <span class="sk-line sk-line--short" />
      </div>
      <p class="sr-only" role="status">
        Reading the tweaks for this file…
      </p>
    </>
  );
}

function Tweak(props: { title: string; term: Parameters<typeof Term>[0]["k"]; active: boolean; note: string; onReset: () => void; children: JSX.Element }): JSX.Element {
  return (
    <div class="tweak" classList={{ "is-active": props.active }}>
      <div class="tweak__head">
        <span class="tweak__mark" aria-hidden="true" />
        <span class="tweak__title">{props.title}</span>
        <Term k={props.term}>{""}</Term>
        <span class="tweak__spacer" />
        <Show when={props.active}>
          <span class="chip" data-tone="active">
            On
          </span>
          <button type="button" class="btn btn--icon btn--xs" aria-label={`Undo ${props.title}`} data-press onClick={() => props.onReset()}>
            <Icon name="undo" size={14} />
          </button>
        </Show>
      </div>
      <p class="tweak__note">{props.note}</p>
      <div class="tweak__body">{props.children}</div>
    </div>
  );
}

function IntInput(props: { label: string; value: number | null; placeholder: string; min: number; max: number; onChange: (v: number | null) => void }): JSX.Element {
  const [bad, setBad] = createSignal(false);
  return (
    <>
      <input
        class="input input--sm mono lnum lnum--wide"
        type="number"
        inputmode="numeric"
        min={props.min}
        max={props.max}
        step={256}
        aria-label={props.label}
        placeholder={props.placeholder}
        value={props.value ?? ""}
        aria-invalid={bad() ? "true" : undefined}
        onInput={(e) => {
          const t = e.currentTarget.value.trim();
          if (t === "") {
            setBad(false);
            props.onChange(null);
            return;
          }
          const n = Number(t);
          const ok = Number.isInteger(n) && n >= props.min && n <= props.max;
          setBad(!ok);
          if (ok) props.onChange(n);
        }}
      />
      <Show when={bad()}>
        <span class="field__error">{`${props.min.toLocaleString("en-US")} to ${props.max.toLocaleString("en-US")}`}</span>
      </Show>
    </>
  );
}

function StopTokens(props: { tokens: readonly string[]; onChange: (t: string[]) => void }): JSX.Element {
  const [text, setText] = createSignal("");
  const error = () => (text() === "" ? null : stopTokenError(text(), props.tokens));
  const add = () => {
    if (stopTokenError(text(), props.tokens) !== null) return;
    props.onChange([...props.tokens, text()]);
    setText("");
  };
  return (
    <div class="stops">
      <ul class="stops__list" aria-label="Stop tokens">
        <For each={props.tokens}>
          {(t) => (
            <li class="token">
              <span class="mono">{t}</span>
              <button type="button" class="token__remove" aria-label={`Remove ${t}`} data-press onClick={() => props.onChange(props.tokens.filter((x) => x !== t))}>
                <Icon name="close" size={12} />
              </button>
            </li>
          )}
        </For>
      </ul>
      <div class="inline-form">
        <input
          class="input input--sm mono"
          aria-label="New stop token"
          placeholder="<|im_end|>"
          maxLength={TWEAK_LIMITS.stopTokenMaxChars + 1}
          value={text()}
          aria-invalid={error() ? "true" : undefined}
          onInput={(e) => setText(e.currentTarget.value)}
          onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), add())}
        />
        <button type="button" class="btn btn--secondary btn--xs" data-press disabled={text() === "" || error() !== null} onClick={add}>
          Add
        </button>
      </div>
      <Show when={error()}>
        <p class="field__error">{error()}</p>
      </Show>
    </div>
  );
}

const SAMPLER_LABELS: Readonly<Record<keyof Sampler, string>> = { temperature: "Temperature", topK: "Top-k", topP: "Top-p", minP: "Min-p", repeatPenalty: "Repeat penalty" };
const SAMPLER_START: Sampler = { temperature: 0.8, topK: 40, topP: 0.95, minP: 0.05, repeatPenalty: 1 };

function SamplerFields(props: { sampler: Sampler | null; author: Sampler | null; onChange: (s: Sampler | null) => void }): JSX.Element {
  // A value out of range stays in the box with an error (never silently ignored), and the box goes
  // back to the saved value when focus leaves it.
  const [bad, setBad] = createSignal<ReadonlySet<keyof Sampler>>(new Set());
  const uid = `sampler-${createUniqueId()}`;
  const mark = (k: keyof Sampler, isBad: boolean) => {
    if (bad().has(k) === isBad) return;
    const next = new Set(bad());
    if (isBad) next.add(k);
    else next.delete(k);
    setBad(next);
  };
  return (
    <div class="sampler">
      <Show
        when={props.sampler}
        fallback={
          <div class="sampler__off">
            <button type="button" class="btn btn--secondary btn--xs" data-press onClick={() => props.onChange({ ...(props.author ?? SAMPLER_START) })}>
              Set a default sampler
            </button>
          </div>
        }
      >
        {(s) => (
          <div class="sampler__grid">
            <For each={Object.keys(SAMPLER_LIMITS) as (keyof Sampler)[]}>
              {(k) => {
                const fieldId = `${uid}-${k}`;
                const errId = `${fieldId}-err`;
                return (
                  <div class="sampler__field">
                    <label class="field__label" for={fieldId}>
                      {SAMPLER_LABELS[k]}
                    </label>
                    <input
                      id={fieldId}
                      class="input input--sm mono lnum"
                      type="number"
                      min={SAMPLER_LIMITS[k].min}
                      max={SAMPLER_LIMITS[k].max}
                      step={SAMPLER_LIMITS[k].step}
                      value={s()[k]}
                      aria-invalid={bad().has(k) ? "true" : undefined}
                      aria-describedby={bad().has(k) ? errId : undefined}
                      onInput={(e) => {
                        const raw = e.currentTarget.value;
                        const v = Number(raw);
                        const lim = SAMPLER_LIMITS[k];
                        const ok = raw.trim() !== "" && Number.isFinite(v) && v >= lim.min && v <= lim.max && (!lim.int || Number.isInteger(v));
                        mark(k, !ok);
                        if (ok) props.onChange({ ...s(), [k]: v });
                      }}
                      onBlur={(e) => {
                        if (!bad().has(k)) return;
                        e.currentTarget.value = String(s()[k]);
                        mark(k, false);
                      }}
                    />
                    <span class="sampler__msg" aria-live="polite">
                      <Show when={bad().has(k)}>
                        <span class="field__error" id={errId}>{`${SAMPLER_LIMITS[k].min} to ${SAMPLER_LIMITS[k].max}${SAMPLER_LIMITS[k].int ? ", whole number" : ""}`}</span>
                      </Show>
                    </span>
                  </div>
                );
              }}
            </For>
          </div>
        )}
      </Show>
      <Show when={props.author}>
        {(a) => (
          <p class="hint">
            <Icon name="info" size={16} />
            <span>
              The authors suggest{" "}
              <span class="mono">
                temp {a().temperature}, top-k {a().topK}, top-p {a().topP}, min-p {a().minP}
              </span>
              .{" "}
              <button type="button" class="text-link text-link--btn" onClick={() => props.onChange({ ...a() })}>
                Use these
              </button>
            </span>
          </p>
        )}
      </Show>
    </div>
  );
}

function TemplateEditor(props: { original: string; value: string | null; onChange: (t: string | null) => void }): JSX.Element {
  const [view, setView] = createSignal<"edit" | "diff">("edit");
  const text = () => props.value ?? props.original;
  const issues = createMemo(() => checkTemplate(text()));
  const diff = createMemo(() => (view() === "diff" ? diffLines(props.original, text()) : null));
  const stats = createMemo(() => {
    const d = diffLines(props.original, text());
    return d ? { add: d.filter((l) => l.kind === "add").length, del: d.filter((l) => l.kind === "del").length } : null;
  });
  return (
    <div class="template">
      <div class="template__bar">
        <div class="toggle-group" role="group" aria-label="Template view">
          <button type="button" class="toggle" aria-pressed={view() === "edit"} data-press onClick={() => setView("edit")}>
            Edit
          </button>
          <button type="button" class="toggle" aria-pressed={view() === "diff"} data-press onClick={() => setView("diff")}>
            Diff
            <Show when={stats() && (stats()!.add > 0 || stats()!.del > 0)}>
              <span class="diff-count">
                <span class="diff-count__add">+{stats()!.add}</span>
                <span class="diff-count__del">−{stats()!.del}</span>
              </span>
            </Show>
          </button>
        </div>
        <span class="template__status" role="status">
          <Show when={issues().length === 0} fallback={<span class="template__bad">{issues()[0]}</span>}>
            <Icon name="check" size={14} />
            Structure looks right
          </Show>
        </span>
      </div>
      <Show
        when={view() === "diff"}
        fallback={
          <textarea
            class="input template__text mono"
            aria-label="Chat template"
            spellcheck={false}
            rows={9}
            maxLength={TWEAK_LIMITS.templateMaxChars}
            value={text()}
            onInput={(e) => {
              const v = e.currentTarget.value;
              props.onChange(v === props.original ? null : v);
            }}
          />
        }
      >
        <Show when={diff()} fallback={<p class="hint">The template is too long to diff here.</p>}>
          {(d) => (
            <pre class="diff" tabindex="0" aria-label="Changes to the chat template">
              <For each={d()}>
                {(line) => (
                  <span class="diff__line" data-kind={line.kind}>
                    <span class="diff__sign" aria-hidden="true">
                      {line.kind === "add" ? "+" : line.kind === "del" ? "−" : " "}
                    </span>
                    <span class="sr-only">{line.kind === "add" ? "Added: " : line.kind === "del" ? "Removed: " : ""}</span>
                    {line.text || " "}
                    {"\n"}
                  </span>
                )}
              </For>
            </pre>
          )}
        </Show>
      </Show>
    </div>
  );
}

/** A free name for an imported profile whose own name is taken: "Name (import)", then
 *  "Name (import 2)" and so on, shortened so it always fits the name limit. */
export function importedName(name: string, taken: readonly string[]): string {
  for (let n = 1; n < 1000; n++) {
    const suffix = n === 1 ? " (import)" : ` (import ${n})`;
    const candidate = `${name.slice(0, TWEAK_LIMITS.profileNameMax - suffix.length).trimEnd()}${suffix}`;
    if (profileNameError(candidate, taken) === null) return candidate;
  }
  return name;
}

function Profiles(props: {
  model: LibraryModel;
  profiles: Accessor<PatchProfile[]>;
  current: Accessor<{ tweaks: Tweaks; fixes: string[] }>;
  knownFixes: Accessor<ReadonlySet<string>>;
  onApply: (p: Omit<PatchProfile, "id">) => void;
  onSaved: (list: PatchProfile[]) => void;
}): JSX.Element {
  const service = useService();
  const [name, setName] = createSignal("");
  const [importing, setImporting] = createSignal(false);
  const [importText, setImportText] = createSignal("");
  const [importError, setImportError] = createSignal("");
  const [mismatch, setMismatch] = createSignal<Omit<PatchProfile, "id"> | null>(null);
  const nameError = () => (name() === "" ? null : profileNameError(name(), props.profiles().map((p) => p.name)));

  const save = async (profile: Omit<PatchProfile, "id">) => {
    try {
      props.onSaved(await service.savePatchProfile(profile));
      notify(`Profile saved: ${profile.name}`);
      return true;
    } catch {
      notify("The profile was not saved", "Its name may be taken, or a value is out of range.", "error");
      return false;
    }
  };
  const saveCurrent = async () => {
    if (profileNameError(name(), props.profiles().map((p) => p.name)) !== null) return;
    if (await save({ name: name().trim(), modelSha256: props.model.sha256, tweaks: props.current().tweaks, fixes: props.current().fixes })) setName("");
  };
  const copy = async (p: PatchProfile) => {
    try {
      await navigator.clipboard.writeText(exportProfile(p));
      notify(`Copied ${p.name}`, "Paste it on another machine with Import.");
    } catch {
      notify("Copy is unavailable here", undefined, "warning");
    }
  };
  const remove = async (p: PatchProfile) => {
    try {
      props.onSaved(await service.deletePatchProfile(props.model.sha256, p.id));
      notify(`Profile deleted: ${p.name}`);
    } catch {
      notify("The profile was not deleted", undefined, "error");
    }
  };
  const doImport = async (forceHere: boolean) => {
    setImportError("");
    const parsed = importProfile(importText());
    if (!parsed.ok) {
      setImportError(parsed.error);
      return;
    }
    let p = parsed.profile;
    if (p.modelSha256 !== props.model.sha256 && !forceHere) {
      setMismatch(p);
      return;
    }
    if (forceHere) p = { ...p, modelSha256: props.model.sha256 };
    p = { ...p, fixes: p.fixes.filter((f) => props.knownFixes().has(f)) };
    const taken = props.profiles().map((x) => x.name);
    if (profileNameError(p.name, taken) !== null) p = { ...p, name: importedName(p.name, taken) };
    setMismatch(null);
    if (await save(p)) {
      props.onApply(p);
      setImportText("");
      setImporting(false);
    }
  };

  return (
    <section class="studio-card card" aria-labelledby="profiles-title">
      <div class="studio-card__head">
        <h3 class="studio-card__title" id="profiles-title">
          Patch profiles
        </h3>
        <button type="button" class="btn btn--tertiary btn--xs" data-press aria-expanded={importing()} onClick={() => setImporting(!importing())}>
          <Icon name="upload" size={14} />
          Import
        </button>
      </div>
      <p class="studio-card__desc">
        Named sets of these tweaks for this exact file, matched by its <Term k="sha256" /> <span class="mono">{props.model.sha256.slice(0, 12)}…</span>. Copy one to share it.
      </p>
      <div class="inline-form">
        <input
          class="input input--sm"
          aria-label="Profile name"
          placeholder="Name this set of tweaks"
          maxLength={TWEAK_LIMITS.profileNameMax}
          value={name()}
          aria-invalid={nameError() ? "true" : undefined}
          onInput={(e) => setName(e.currentTarget.value)}
          onKeyDown={(e) => e.key === "Enter" && void saveCurrent()}
        />
        <button type="button" class="btn btn--secondary btn--xs" data-press disabled={name().trim() === "" || nameError() !== null} onClick={() => void saveCurrent()}>
          Save profile
        </button>
      </div>
      <Show when={nameError()}>
        <p class="field__error">{nameError()}</p>
      </Show>
      <Show when={importing()}>
        <div class="import">
          <textarea
            class="input import__text mono"
            rows={4}
            aria-label="Profile text to import"
            placeholder='{"format": "cerebro-patch-profile", ...}'
            value={importText()}
            onInput={(e) => (setImportText(e.currentTarget.value), setImportError(""), setMismatch(null))}
          />
          <div class="inline-form">
            <button type="button" class="btn btn--secondary btn--xs" data-press disabled={importText().trim() === ""} onClick={() => void doImport(false)}>
              Import profile
            </button>
          </div>
          <Show when={importError()}>
            <p class="field__error" role="alert">
              <Icon name="alert" size={14} />
              {importError()}
            </p>
          </Show>
          <Show when={mismatch()}>
            <div class="notice notice--inline" role="status">
              <Icon name="info" size={18} />
              <p>
                This profile was made for a different file (<span class="mono">{mismatch()?.modelSha256.slice(0, 12)}…</span>). Tweaks can behave differently on another file.{" "}
                <button type="button" class="text-link text-link--btn" onClick={() => void doImport(true)}>
                  Use it for this file anyway
                </button>
              </p>
            </div>
          </Show>
        </div>
      </Show>
      <Show when={props.profiles().length > 0} fallback={<p class="studio-empty">No profiles saved for this file yet.</p>}>
        <ul class="profile-list">
          <For each={props.profiles()}>
            {(p) => (
              <li class="profile">
                <span class="profile__name">{p.name}</span>
                <span class="profile__meta">
                  {activeTweakCount(p.tweaks)} tweaks · {p.fixes.length} fixes
                </span>
                <span class="profile__actions">
                  <button type="button" class="btn btn--tertiary btn--xs" data-press onClick={() => props.onApply(p)}>
                    Apply
                  </button>
                  <button type="button" class="btn btn--icon btn--xs" aria-label={`Copy ${p.name} as text`} data-press onClick={() => void copy(p)}>
                    <Icon name="copy" size={14} />
                  </button>
                  <button type="button" class="btn btn--icon btn--xs" aria-label={`Delete ${p.name}`} data-press onClick={() => void remove(p)}>
                    <Icon name="trash" size={14} />
                  </button>
                </span>
              </li>
            )}
          </For>
        </ul>
      </Show>
    </section>
  );
}
