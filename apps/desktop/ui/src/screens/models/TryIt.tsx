import { createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { IllustrativeFlag } from "../../components/Screen";
import { Term } from "../../components/Term";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import type { LibraryModel, TestOutput } from "../../service/types";
import { Spinner } from "./parts";

/**
 * Vision and audio models, done better than "load and hope": a vision model shows the projector it
 * was paired with and a "try it" drop zone; speech models get a record-and-transcribe or a
 * speak-this-sentence test. Everything stays on this computer.
 */

const IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]);
const IMAGE_MAX_BYTES = 20 * 1024 * 1024;

export function VisionTry(props: { model: LibraryModel }): JSX.Element {
  const service = useService();
  const [preview, setPreview] = createSignal<string | null>(null);
  const [fileName, setFileName] = createSignal("");
  const [error, setError] = createSignal("");
  const [busy, setBusy] = createSignal(false);
  const [output, setOutput] = createSignal<TestOutput | null>(null);
  const [over, setOver] = createSignal(false);
  let input: HTMLInputElement | undefined;
  let alive = true;
  onCleanup(() => (alive = false));
  // Each image gets a ticket; only the newest one's answer (or error) is ever shown, so a slow reply
  // for an earlier image never lands under a later preview.
  let ticket = 0;

  const take = (file: File | undefined) => {
    const mine = ++ticket;
    const current = () => alive && mine === ticket;
    setError("");
    setOutput(null);
    setBusy(false);
    if (!file) return;
    // A new file replaces the old preview at once, so an error or a description is never shown
    // under the previous image.
    setPreview(null);
    setFileName("");
    // The type the OS reports picks the check; the image is only ever shown through <img> from a
    // data: URL, which cannot run script whatever the bytes are.
    if (!IMAGE_TYPES.has(file.type)) {
      setError("Use a PNG, JPEG, WebP or GIF image.");
      return;
    }
    if (file.size > IMAGE_MAX_BYTES) {
      setError("Use an image under 20 MB.");
      return;
    }
    setFileName(file.name.slice(0, 120));
    const reader = new FileReader();
    setBusy(true);
    reader.onload = () => {
      if (!current() || typeof reader.result !== "string" || !reader.result.startsWith("data:image/")) return;
      const data = reader.result;
      setPreview(data);
      service
        .describeImage(props.model.id, { name: file.name.slice(0, 120), type: file.type, bytes: file.size, data })
        .then((o) => current() && setOutput(o))
        .catch((e: unknown) => current() && setError(e instanceof Error && e.message ? e.message : "The model could not read that image. Try another one."))
        .finally(() => current() && setBusy(false));
    };
    reader.onerror = () => {
      if (!current()) return;
      setError("That image could not be read from disk. Try another one.");
      setBusy(false);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div class="try card">
      <div class="try__head">
        <h3 class="try__title">Try it</h3>
        <IllustrativeFlag />
      </div>
      <Show
        when={props.model.projector}
        fallback={
          <div class="projector projector--missing" role="status">
            <Icon name="alert" size={18} />
            <p>
              No <Term k="mmproj" /> was found next to this file. Text chat works; to read images, put the matching mmproj file in the same folder and scan again.
            </p>
          </div>
        }
      >
        {(p) => (
          <div class="projector">
            <Icon name="link" size={18} />
            <p>
              Paired with <span class="mono">{p().file}</span> ({p().sizeGiB.toFixed(1)} GiB), found beside the model and matched by architecture. It loads with the model.
            </p>
          </div>
        )}
      </Show>
      <Show when={props.model.projector}>
        <label
          class="dropzone"
          classList={{ "is-over": over(), "has-image": preview() !== null }}
          onDragEnter={(e) => (e.preventDefault(), setOver(true))}
          onDragOver={(e) => e.preventDefault()}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            take(e.dataTransfer?.files[0]);
          }}
        >
          <input ref={input} class="sr-only" type="file" accept="image/png,image/jpeg,image/webp,image/gif" onChange={(e) => take(e.currentTarget.files?.[0])} />
          <Show
            when={preview()}
            fallback={
              <span class="dropzone__empty">
                <span class="dropzone__icon" aria-hidden="true">
                  <Icon name="upload" size={22} />
                </span>
                <span class="dropzone__title">
                  Drop an image here, or <span class="dropzone__choose">choose a file</span>
                </span>
                <span class="dropzone__text">PNG, JPEG, WebP or GIF up to 20 MB. It never leaves this computer.</span>
              </span>
            }
          >
            {(src) => <img class="dropzone__img" src={src()} alt={`Your image: ${fileName()}`} />}
          </Show>
        </label>
        <Show when={error()}>
          <p class="field__error" role="alert">
            <Icon name="alert" size={14} />
            {error()}
          </p>
        </Show>
        <Output busy={busy()} output={output()} busyText="Reading the image…" />
      </Show>
    </div>
  );
}

const RECORD_SECONDS = 4;

export function SpeechToTextTry(props: { model: LibraryModel }): JSX.Element {
  const service = useService();
  const [phase, setPhase] = createSignal<"idle" | "recording" | "working">("idle");
  const [left, setLeft] = createSignal(RECORD_SECONDS);
  const [output, setOutput] = createSignal<TestOutput | null>(null);
  let timer: ReturnType<typeof setInterval> | undefined;
  let alive = true;
  onCleanup(() => {
    alive = false;
    clearInterval(timer);
  });

  // The timer and the Stop button can both end a recording; only the first one counts.
  const finish = (seconds: number) => {
    if (phase() !== "recording") return;
    clearInterval(timer);
    setPhase("working");
    service
      .transcribeSample(props.model.id, Math.max(0.5, seconds))
      .then((o) => alive && setOutput(o))
      .catch(() => alive && notify("Transcription failed", "Try recording again.", "error"))
      .finally(() => alive && setPhase("idle"));
  };
  const start = () => {
    if (phase() !== "idle") return;
    setOutput(null);
    setLeft(RECORD_SECONDS);
    setPhase("recording");
    service.startRecording().catch((e: unknown) => {
      clearInterval(timer);
      if (alive) {
        setPhase("idle");
        notify("The microphone could not start", e instanceof Error ? e.message : "Check that a microphone is connected.", "error");
      }
    });
    timer = setInterval(() => {
      const next = left() - 1;
      setLeft(next);
      if (next <= 0) finish(RECORD_SECONDS);
    }, 1000);
  };

  return (
    <div class="try card">
      <div class="try__head">
        <h3 class="try__title">Record and transcribe</h3>
        <IllustrativeFlag />
      </div>
      <p class="try__text">Say a sentence; the model writes it down. Runs in its own engine, so your chat model stays loaded.</p>
      <div class="try__row">
        <Show
          when={phase() === "recording"}
          fallback={
            <button type="button" class="btn btn--secondary" data-press aria-busy={phase() === "working" ? "true" : undefined} onClick={start}>
              <Spinner />
              <span class="btn__label">
                <Icon name="mic" size={16} />
                Record
              </span>
            </button>
          }
        >
          <button type="button" class="btn btn--secondary is-recording" data-press onClick={() => finish(RECORD_SECONDS - left())}>
            <Icon name="stop" size={16} />
            Stop ({left()} s)
          </button>
        </Show>
        <span class="levels" classList={{ "is-live": phase() === "recording" }} aria-hidden="true">
          <For each={[0, 1, 2, 3, 4, 5, 6]}>{(i) => <span class="levels__bar" data-i={i} />}</For>
        </span>
        <span class="sr-only" role="status" aria-live="polite">
          {phase() === "recording" ? "Recording" : phase() === "working" ? "Transcribing" : ""}
        </span>
      </div>
      <p class="hint">
        <Icon name="info" size={16} />
        The browser preview simulates the microphone; the desktop app records from your default input.
      </p>
      <Output busy={phase() === "working"} output={output()} busyText="Transcribing…" />
    </div>
  );
}

const DEFAULT_SENTENCE = "Cerebro runs this voice on your own computer.";

export function TextToSpeechTry(props: { model: LibraryModel }): JSX.Element {
  const service = useService();
  const [text, setText] = createSignal(DEFAULT_SENTENCE);
  const [busy, setBusy] = createSignal(false);
  const [seconds, setSeconds] = createSignal<number | null>(null);
  const [playing, setPlaying] = createSignal(false);
  let timer: ReturnType<typeof setTimeout> | undefined;
  let alive = true;
  onCleanup(() => {
    alive = false;
    clearTimeout(timer);
  });
  const valid = () => text().trim().length > 0 && text().length <= 280;

  const speak = () => {
    if (!valid() || busy()) return;
    setBusy(true);
    setSeconds(null);
    service
      .speakSample(props.model.id, text())
      .then((s) => {
        if (!alive) return;
        setSeconds(s);
        setPlaying(true);
        clearTimeout(timer);
        timer = setTimeout(() => alive && setPlaying(false), s * 1000);
      })
      .catch(() => alive && notify("Speech failed", "Try a shorter sentence.", "error"))
      .finally(() => alive && setBusy(false));
  };

  return (
    <div class="try card">
      <div class="try__head">
        <h3 class="try__title">Speak this sentence</h3>
        <IllustrativeFlag />
      </div>
      <label class="field try__field">
        <span class="field__label">Sentence</span>
        <textarea class="input try__textarea" rows={2} maxLength={280} value={text()} onInput={(e) => setText(e.currentTarget.value)} aria-invalid={valid() ? undefined : "true"} />
        <span class="field__hint">{`${text().length} of 280 characters`}</span>
      </label>
      <div class="try__row">
        <button type="button" class="btn btn--secondary" data-press disabled={!valid()} aria-busy={busy() ? "true" : undefined} onClick={speak}>
          <Spinner />
          <span class="btn__label">
            <Icon name="play" size={16} />
            Speak
          </span>
        </button>
        <Show when={seconds() !== null}>
          <span class="playback" classList={{ "is-playing": playing() }}>
            <span class="playback__track" aria-hidden="true">
              <span class="playback__fill" style={{ "animation-duration": `${seconds() ?? 1}s` }} />
            </span>
            <span class="mono">
              {seconds()?.toFixed(1)}
              <span class="unit">s</span>
            </span>
          </span>
        </Show>
        <span class="sr-only" role="status" aria-live="polite">
          {playing() ? "Speaking" : seconds() !== null ? "Done" : ""}
        </span>
      </div>
      <p class="hint">
        <Icon name="info" size={16} />
        The browser preview shows the timing only; the desktop app plays the audio.
      </p>
    </div>
  );
}

function Output(props: { busy: boolean; output: TestOutput | null; busyText: string }): JSX.Element {
  return (
    <Show when={props.busy || props.output}>
      <div class="try-output" aria-live="polite">
        <Show when={!props.busy} fallback={<p class="try-output__busy">{props.busyText}</p>}>
          <p class="try-output__text">{props.output?.text}</p>
        </Show>
      </div>
    </Show>
  );
}
