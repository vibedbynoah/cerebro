import { createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { EFFORT_LABELS, EFFORTS, REASONING_LABELS, REASONING_NOTES, REASONING_SOURCES, VERBOSITIES, VERBOSITY_LABELS, type ChatPrefs, type Effort, type EffortLevel } from "../../lib/chat/prompt";
import type { Attachment } from "../../lib/chat/store";

const IMAGE = /^image\/(png|jpeg|gif|webp)$/;
const MAX_IMAGE = 10 * 1024 * 1024;
const MAX_FILES = 10;
const ORCHESTRATE_WORKERS: readonly [number, number] = [1, 6];
const ORCHESTRATE_ROUNDS: readonly [number, number] = [1, 8];

export function Composer(props: {
  levels: Readonly<Record<Effort, EffortLevel>>;
  prefs: ChatPrefs;
  setPrefs: (p: Partial<ChatPrefs>) => void;
  streaming: boolean;
  notes: string[];
  onSend: (text: string, atts: Attachment[]) => void;
  onStop: () => void;
  onTranscribe: (seconds: number) => Promise<string>;
  onOpenPrompt: () => void;
}): JSX.Element {
  const [text, setText] = createSignal("");
  const [atts, setAtts] = createSignal<Attachment[]>([]);
  const [dragging, setDragging] = createSignal(false);
  const [recording, setRecording] = createSignal(false);
  const [level, setLevel] = createSignal<number[]>(Array(16).fill(0.1));
  const [error, setError] = createSignal("");
  let area!: HTMLTextAreaElement;
  let fileInput!: HTMLInputElement;
  let recStart = 0;
  let wave: ReturnType<typeof setInterval> | undefined;
  onCleanup(() => clearInterval(wave));

  const send = () => {
    const t = text().trim();
    if ((!t && !atts().length) || props.streaming) return;
    props.onSend(t, atts());
    setText("");
    area.value = "";
    setAtts([]);
    setError("");
    area.style.height = "";
  };

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    } else if (e.key === "Escape" && props.streaming) {
      e.preventDefault();
      props.onStop();
    }
  };

  // Sized once per frame, so pasting a long text costs one layout, not one per input event.
  let growFrame = 0;
  const grow = () => {
    if (growFrame) return;
    growFrame = requestAnimationFrame(() => {
      growFrame = 0;
      area.style.height = "auto";
      area.style.height = `${Math.min(area.scrollHeight, 240)}px`;
    });
  };
  onCleanup(() => cancelAnimationFrame(growFrame));

  const addFiles = async (files: FileList | File[]) => {
    const list = Array.from(files).slice(0, MAX_FILES - atts().length);
    for (const f of list) {
      if (IMAGE.test(f.type)) {
        if (f.size > MAX_IMAGE) {
          setError(`${f.name} is over 10 MB.`);
          continue;
        }
        const preview = await new Promise<string | null>((resolve) => {
          const r = new FileReader();
          r.onload = () => resolve(typeof r.result === "string" ? r.result : null);
          r.onerror = () => resolve(null);
          r.readAsDataURL(f);
        });
        setAtts((a) => [...a, { name: f.name.slice(0, 260), type: f.type, bytes: f.size, preview }]);
      } else {
        setAtts((a) => [...a, { name: f.name.slice(0, 260), type: f.type.slice(0, 100) || "application/octet-stream", bytes: f.size, preview: null }]);
      }
    }
  };

  const onPaste = (e: ClipboardEvent) => {
    const files = e.clipboardData?.files;
    if (files && files.length) {
      e.preventDefault();
      void addFiles(files);
    }
  };

  // Push-to-talk: hold the button (or press Space on it). The waveform is simulated in this build.
  const startRec = () => {
    if (recording()) return;
    setRecording(true);
    recStart = performance.now();
    wave = setInterval(() => setLevel((l) => [...l.slice(1), 0.15 + Math.random() * 0.85]), 70);
  };
  const stopRec = async () => {
    if (!recording()) return;
    setRecording(false);
    clearInterval(wave);
    setLevel(Array(16).fill(0.1));
    const secs = Math.max(0.5, (performance.now() - recStart) / 1000);
    try {
      const t = await props.onTranscribe(Math.min(secs, 120));
      const next = text() ? `${text()} ${t}` : t;
      setText(next);
      area.value = next;
      grow();
      area.focus();
    } catch {
      setError("Could not transcribe that recording.");
    }
  };

  return (
    <div
      class="composer"
      data-dragging={dragging()}
      onDragOver={(e) => {
        if (e.dataTransfer?.types.includes("Files")) {
          e.preventDefault();
          setDragging(true);
        }
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (e.dataTransfer?.files) void addFiles(e.dataTransfer.files);
      }}
    >
      <Show when={atts().length}>
        <ul class="composer__atts" role="list" aria-label="Attachments">
          <For each={atts()}>
            {(a, i) => (
              <li class="att">
                <Show when={a.preview} fallback={<Icon name="text" size={16} />}>
                  {(src) => <img src={src()} alt="" width="40" height="40" />}
                </Show>
                <span class="att__name">{a.name}</span>
                <button class="icon-btn icon-btn--sm" type="button" aria-label={`Remove ${a.name}`} onClick={() => setAtts((l) => l.filter((_, j) => j !== i()))}>
                  <Icon name="close" size={12} />
                </button>
              </li>
            )}
          </For>
        </ul>
      </Show>
      <div class="composer__row">
        <textarea
          ref={area}
          class="composer__input"
          rows="1"
          placeholder={recording() ? "Listening…" : "Message the model. / for commands: /imagine a red fox in snow"}
          aria-label="Message"
          onInput={(e) => {
            setText(e.currentTarget.value);
            grow();
          }}
          onKeyDown={onKey}
          onPaste={onPaste}
          maxlength="100000"
        />
        <div class="composer__buttons">
          <input ref={fileInput} type="file" multiple hidden onChange={(e) => e.currentTarget.files && void addFiles(e.currentTarget.files)} />
          <button class="icon-btn" type="button" aria-label="Attach files or images" onClick={() => fileInput.click()}>
            <Icon name="upload" size={18} />
          </button>
          <button
            class="icon-btn mic"
            type="button"
            aria-label={recording() ? "Recording: release to transcribe" : "Hold to talk"}
            aria-pressed={recording()}
            onPointerDown={startRec}
            onPointerUp={() => void stopRec()}
            onPointerLeave={() => void stopRec()}
            onKeyDown={(e) => (e.key === " " || e.key === "Enter") && !e.repeat && (e.preventDefault(), startRec())}
            onKeyUp={(e) => (e.key === " " || e.key === "Enter") && (e.preventDefault(), void stopRec())}
          >
            <Show when={recording()} fallback={<Icon name="mic" size={18} />}>
              <span class="wave" aria-hidden="true">
                <For each={level()}>{(v) => <span class="wave__bar" style={{ transform: `scaleY(${v})` }} />}</For>
              </span>
            </Show>
          </button>
          <Show
            when={props.streaming}
            fallback={
              <button class="btn btn--primary composer__send" type="button" onClick={send} disabled={!text().trim() && !atts().length} aria-label="Send" data-press>
                <Icon name="arrowRight" size={18} />
              </button>
            }
          >
            <button class="btn btn--secondary composer__send" type="button" onClick={() => props.onStop()} aria-label="Stop generating" data-press>
              <Icon name="stop" size={18} />
            </button>
          </Show>
        </div>
      </div>
      <Show when={error()}>
        <p class="composer__error" role="alert">
          {error()}
        </p>
      </Show>
      <div class="composer__controls">
        <fieldset class="segmented segmented--sm effort" aria-describedby="effort-note">
          <legend>Effort</legend>
          <For each={EFFORTS}>
            {(e) => (
              <label class="segmented__option" data-press>
                <input type="radio" name="effort" value={e} checked={props.prefs.effort === e} onChange={() => props.setPrefs({ effort: e })} />
                <span>{EFFORT_LABELS[e]}</span>
              </label>
            )}
          </For>
        </fieldset>
        <fieldset class="segmented segmented--sm">
          <legend>Verbosity</legend>
          <For each={VERBOSITIES}>
            {(v) => (
              <label class="segmented__option" data-press>
                <input type="radio" name="verbosity" value={v} checked={props.prefs.verbosity === v} onChange={() => props.setPrefs({ verbosity: v })} />
                <span>{VERBOSITY_LABELS[v]}</span>
              </label>
            )}
          </For>
        </fieldset>
        <label class="composer__select">
          <span>Reasoning</span>
          <select value={props.prefs.source} onChange={(e) => props.setPrefs({ source: e.currentTarget.value as ChatPrefs["source"] })} aria-describedby="source-note">
            <For each={REASONING_SOURCES}>{(s) => <option value={s}>{REASONING_LABELS[s]}</option>}</For>
          </select>
        </label>
        <label class="composer__check">
          <input type="checkbox" checked={!props.prefs.addInstructions} onChange={(e) => props.setPrefs({ addInstructions: !e.currentTarget.checked })} />
          <span>No added prompt</span>
        </label>
        <label class="composer__check" title="The model looks around without changing anything, then shows you a plan to approve.">
          <input type="checkbox" checked={props.prefs.planMode} onChange={(e) => props.setPrefs({ planMode: e.currentTarget.checked })} />
          <span>Plan first</span>
        </label>
        <label class="composer__check">
          <input type="checkbox" checked={props.prefs.orchestrate} onChange={(e) => props.setPrefs({ orchestrate: e.currentTarget.checked })} />
          <span>Orchestrate</span>
        </label>
        <Show when={props.prefs.orchestrate}>
          <span class="stepper">
            Workers
            <button class="icon-btn icon-btn--sm" type="button" aria-label="Fewer workers" disabled={props.prefs.workers <= ORCHESTRATE_WORKERS[0]} onClick={() => props.setPrefs({ workers: props.prefs.workers - 1 })}>
              <Icon name="chevron" size={14} />
            </button>
            <span class="mono stepper__value" data-testid="orchestrate-workers">
              {props.prefs.workers}
            </span>
            <button class="icon-btn icon-btn--sm stepper__up" type="button" aria-label="More workers" disabled={props.prefs.workers >= ORCHESTRATE_WORKERS[1]} onClick={() => props.setPrefs({ workers: props.prefs.workers + 1 })}>
              <Icon name="chevron" size={14} />
            </button>
          </span>
          <span class="stepper">
            Rounds
            <button class="icon-btn icon-btn--sm" type="button" aria-label="Fewer rounds" disabled={props.prefs.rounds <= ORCHESTRATE_ROUNDS[0]} onClick={() => props.setPrefs({ rounds: props.prefs.rounds - 1 })}>
              <Icon name="chevron" size={14} />
            </button>
            <span class="mono stepper__value" data-testid="orchestrate-rounds">
              {props.prefs.rounds}
            </span>
            <button class="icon-btn icon-btn--sm stepper__up" type="button" aria-label="More rounds" disabled={props.prefs.rounds >= ORCHESTRATE_ROUNDS[1]} onClick={() => props.setPrefs({ rounds: props.prefs.rounds + 1 })}>
              <Icon name="chevron" size={14} />
            </button>
          </span>
        </Show>
        <button class="btn btn--tertiary btn--sm" type="button" onClick={() => props.onOpenPrompt()} data-press>
          <Icon name="code" size={14} />
          <span class="btn__label">System prompt</span>
        </button>
      </div>
      <p class="composer__hint" id="effort-note">
        <span id="source-note">{REASONING_NOTES[props.prefs.source]}</span> {EFFORT_LABELS[props.prefs.effort]}: {props.levels[props.prefs.effort].slowdown}
        {props.levels[props.prefs.effort].passes ? `, ${props.levels[props.prefs.effort].passes} review pass${props.levels[props.prefs.effort].passes > 1 ? "es" : ""}` : ""}.
        <For each={props.notes}>{(n) => <> {n}</>}</For> <kbd>Enter</kbd> sends, <kbd>Shift</kbd>+<kbd>Enter</kbd> new line, <kbd>Esc</kbd> stops.
      </p>
      <Show when={props.prefs.orchestrate}>
        <p class="composer__hint composer__hint--orchestrate">Several instances of this model plan, split the work and check each other through a shared ledger. Slower, often better on hard problems.</p>
      </Show>
    </div>
  );
}
