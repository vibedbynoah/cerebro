import { createEffect, createSignal, For, on, onCleanup, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { closest, parseCommand, suggest, unknownCommand, type CommandDef } from "../../lib/chat/commands";
import { MODE_PRESETS, MODES, type Mode } from "../../lib/chat/modes";
import { EFFORT_LABELS, EFFORTS, REASONING_LABELS, REASONING_NOTES, REASONING_SOURCES, VERBOSITIES, VERBOSITY_LABELS, type ChatPrefs, type Effort, type EffortLevel } from "../../lib/chat/prompt";
import type { Attachment } from "../../lib/chat/store";
import { readPref, writePref } from "../../lib/storage";

/** Whether the composer's fine controls were left open; they start folded away. */
const FINE_TUNE_KEY = "cerebro-chat-fine-tune";

const IMAGE = /^image\/(png|jpeg|gif|webp)$/;
const MAX_IMAGE = 10 * 1024 * 1024;
const MAX_FILES = 10;
/** Longest side an attached image keeps: the image is stored with the chat and sent with every
 *  turn, and a vision model reads at most about this much detail anyway. */
const MAX_SIDE = 1344;
/** Data URLs above this are re-encoded smaller (the chat store keeps previews up to 2 MB). */
const MAX_DATA = 1_500_000;

/** Downscales an image data URL to MAX_SIDE (JPEG) when it is larger, else returns it unchanged.
 *  Where the browser cannot decode or draw it, the original is kept. */
async function shrink(url: string): Promise<string> {
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const long = Math.max(img.naturalWidth, img.naturalHeight);
    if (long <= MAX_SIDE && url.length <= MAX_DATA) return url;
    const scale = Math.min(1, MAX_SIDE / long);
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(img.naturalWidth * scale));
    c.height = Math.max(1, Math.round(img.naturalHeight * scale));
    const g = c.getContext("2d");
    if (!g) return url;
    g.fillStyle = "#fff";
    g.fillRect(0, 0, c.width, c.height);
    g.drawImage(img, 0, 0, c.width, c.height);
    for (const q of [0.9, 0.8, 0.65]) {
      const out = c.toDataURL("image/jpeg", q);
      if (out.length <= MAX_DATA) return out;
    }
    return c.toDataURL("image/jpeg", 0.5);
  } catch {
    return url;
  }
}
const ORCHESTRATE_WORKERS: readonly [number, number] = [1, 6];
const ORCHESTRATE_ROUNDS: readonly [number, number] = [1, 8];

export function Composer(props: {
  levels: Readonly<Record<Effort, EffortLevel>>;
  /** The Speed <-> Accuracy level the settings match (null: changed by hand). */
  mode: Mode | null;
  onMode: (m: Mode) => void;
  prefs: ChatPrefs;
  setPrefs: (p: Partial<ChatPrefs>) => void;
  streaming: boolean;
  notes: string[];
  onSend: (text: string, atts: Attachment[]) => void;
  /** A / command typed in the composer (lib/chat/commands.ts); never sent to the model. */
  onCommand?: (name: string, arg: string) => void;
  onStop: () => void;
  onTranscribe: (seconds: number) => Promise<string>;
  /** Starts the microphone when push-to-talk is pressed (desktop records natively). */
  onRecordStart?: () => Promise<void>;
  /** The microphone's level, 0..1, polled for the waveform while recording. */
  onLevel?: () => Promise<number>;
  onOpenPrompt: () => void;
  onOpenPrompts: () => void;
  /** A prompt chosen in the library: replaces the composer text whenever `at` changes. */
  draft?: { text: string; at: number } | null;
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

  const clearInput = () => {
    setText("");
    area.value = "";
    setAtts([]);
    setError("");
    area.style.height = "";
  };

  const send = () => {
    const t = text().trim();
    if (props.onCommand && t.startsWith("/") && !t.startsWith("//")) {
      const cmd = parseCommand(t);
      if (cmd) {
        props.onCommand(cmd.def.name, cmd.arg);
        clearInput();
        return;
      }
      const unknown = unknownCommand(t);
      if (unknown) {
        setError(`There is no /${unknown} command. Did you mean /${closest(unknown)}? Type /help for the list.`);
        return;
      }
    }
    if ((!t && !atts().length) || props.streaming) return;
    // "//text" sends a message that starts with "/".
    props.onSend(t.startsWith("//") ? t.slice(1) : t, atts());
    clearInput();
  };

  // The / command menu: open while the message is a command name being typed.
  const [menuIndex, setMenuIndex] = createSignal(0);
  const [menuClosed, setMenuClosed] = createSignal(false);
  const menuItems = (): CommandDef[] => {
    const t = text();
    if (!props.onCommand || menuClosed() || !/^\/[a-z-]*$/i.test(t)) return [];
    return suggest(t);
  };
  const pick = (c: CommandDef) => {
    setMenuIndex(0);
    if (!c.args) {
      setText(`/${c.name}`);
      area.value = `/${c.name}`;
      send();
      return;
    }
    const v = `/${c.name} `;
    setText(v);
    area.value = v;
    area.focus();
  };

  const onKey = (e: KeyboardEvent) => {
    const items = menuItems();
    if (items.length) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        setMenuIndex((i) => (i + (e.key === "ArrowDown" ? 1 : items.length - 1)) % items.length);
        return;
      }
      if ((e.key === "Enter" && !e.shiftKey) || e.key === "Tab") {
        e.preventDefault();
        pick(items[Math.min(menuIndex(), items.length - 1)]!);
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setMenuClosed(true);
        return;
      }
    }
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

  // A prompt from the library fills the box and puts the cursor at the end, ready to edit or send.
  createEffect(
    on(
      () => props.draft,
      (draft) => {
        if (!draft) return;
        setText(draft.text);
        area.value = draft.text;
        grow();
        area.focus();
        area.setSelectionRange(draft.text.length, draft.text.length);
      },
      { defer: true },
    ),
  );

  const addFiles = async (files: FileList | File[]) => {
    const list = Array.from(files).slice(0, MAX_FILES - atts().length);
    for (const f of list) {
      if (IMAGE.test(f.type)) {
        if (f.size > MAX_IMAGE) {
          setError(`${f.name} is over 10 MB.`);
          continue;
        }
        const raw = await new Promise<string | null>((resolve) => {
          const r = new FileReader();
          r.onload = () => resolve(typeof r.result === "string" ? r.result : null);
          r.onerror = () => resolve(null);
          r.readAsDataURL(f);
        });
        const preview = raw ? await shrink(raw) : null;
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

  // Push-to-talk: hold the button (or press Space on it). The waveform shows the microphone's level.
  const startRec = () => {
    if (recording()) return;
    setRecording(true);
    recStart = performance.now();
    props.onRecordStart?.().catch((e: unknown) => {
      setRecording(false);
      clearInterval(wave);
      setError(e instanceof Error ? e.message : "The microphone could not start.");
    });
    wave = setInterval(() => {
      const read = props.onLevel ? props.onLevel() : Promise.resolve(0.15 + Math.random() * 0.85);
      void read.then((v) => setLevel((l) => [...l.slice(1), Math.max(0.1, Math.min(1, v * 3))])).catch(() => {});
    }, 70);
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
    } catch (e) {
      setError(e instanceof Error && e.message ? e.message : "Could not transcribe that recording.");
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
      <Show when={menuItems().length}>
        <ul class="command-menu card" id="command-menu" role="listbox" aria-label="Commands">
          <For each={menuItems()}>
            {(c, i) => (
              <li
                id={`cmd-${c.name}`}
                role="option"
                class="command-menu__item"
                aria-selected={i() === Math.min(menuIndex(), menuItems().length - 1)}
                onMouseMove={() => i() !== menuIndex() && setMenuIndex(i())}
                onMouseDown={(e) => {
                  e.preventDefault();
                  pick(c);
                }}
              >
                <span class="mono command-menu__name">/{c.name}</span>
                <Show when={c.args}>
                  <span class="mono command-menu__args">{c.args}</span>
                </Show>
                <span class="command-menu__desc">{c.description}</span>
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
          placeholder={recording() ? "Listening…" : "Message the model. /imagine for an image, /prompt for saved prompts"}
          aria-label="Message"
          aria-autocomplete="list"
          aria-controls={menuItems().length ? "command-menu" : undefined}
          aria-activedescendant={menuItems().length ? `cmd-${menuItems()[Math.min(menuIndex(), menuItems().length - 1)]!.name}` : undefined}
          onInput={(e) => {
            setText(e.currentTarget.value);
            setMenuClosed(false);
            setMenuIndex(0);
            setError("");
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
      <div class="composer__mode">
        <label for="speed-mode" class="composer__mode-end">Speed</label>
        <input
          id="speed-mode"
          type="range"
          min="1"
          max="5"
          step="1"
          value={props.mode ?? 3}
          data-custom={props.mode === null ? "true" : undefined}
          aria-valuetext={props.mode ? MODE_PRESETS[props.mode].label : "Custom"}
          aria-describedby="mode-note"
          onInput={(e) => props.onMode(Number(e.currentTarget.value) as Mode)}
        />
        <span class="composer__mode-end" aria-hidden="true">
          Accuracy
        </span>
        <output class="composer__mode-level" for="speed-mode" data-testid="speed-mode">
          {props.mode ? MODE_PRESETS[props.mode].label : "Custom"}
        </output>
        <span class="composer__mode-ticks" aria-hidden="true">
          <For each={MODES}>{(m) => <span data-on={props.mode === m ? "true" : undefined} />}</For>
        </span>
      </div>
      <p class="composer__hint composer__hint--mode" id="mode-note">
        {props.mode ? MODE_PRESETS[props.mode].summary : "Custom: you changed one of the settings in Fine-tune by hand. Move the slider to pick a level again."}
      </p>
      <details class="composer__more" open={readPref(FINE_TUNE_KEY) === "open"} on:toggle={(e) => writePref(FINE_TUNE_KEY, e.currentTarget.open ? "open" : "closed")}>
        <summary class="composer__more-summary">
          <Icon name="sliders" size={15} />
          <span class="composer__more-title">Fine-tune</span>
          <span class="composer__more-meta" aria-hidden="true">Effort, reasoning and prompt</span>
          <span class="composer__more-chevron" aria-hidden="true">
            <Icon name="chevron" size={14} />
          </span>
        </summary>
        <div class="composer__more-body">
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
            <button class="btn btn--tertiary btn--sm" type="button" onClick={() => props.onOpenPrompts()} data-press>
              <Icon name="book" size={14} />
              <span class="btn__label">Prompts</span>
            </button>
          </div>
          <p class="composer__hint" id="effort-note">
            <span id="source-note">{REASONING_NOTES[props.prefs.source]}</span> {EFFORT_LABELS[props.prefs.effort]}: {props.levels[props.prefs.effort].slowdown}
            {props.levels[props.prefs.effort].passes ? `, ${props.levels[props.prefs.effort].passes} review pass${props.levels[props.prefs.effort].passes > 1 ? "es" : ""}` : ""}.
            <For each={props.notes}>{(n) => <> {n}</>}</For>
          </p>
          <Show when={props.prefs.orchestrate}>
            <p class="composer__hint composer__hint--orchestrate">Several instances of this model plan, split the work and check each other through a shared ledger. Slower, often better on hard problems.</p>
          </Show>
        </div>
      </details>
      <p class="composer__hint composer__hint--keys">
        <kbd>Enter</kbd> sends, <kbd>Shift</kbd>+<kbd>Enter</kbd> new line, <kbd>Esc</kbd> stops.
      </p>
    </div>
  );
}
