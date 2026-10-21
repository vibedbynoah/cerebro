import { createSignal, onCleanup, onMount, Show, type JSX } from "solid-js";
import { Portal } from "solid-js/web";
import { Icon } from "../../components/icons";
import { EndOfSpeech, thresholdFor } from "../../lib/chat/vad";
import { parseMarkdown, plainText } from "../../lib/chat/markdown";
import { readPref, writePref } from "../../lib/storage";
import { useService } from "../../service/context";

type Phase = "listening" | "hearing" | "transcribing" | "thinking" | "speaking" | "error";

const LABELS: Readonly<Record<Phase, string>> = {
  listening: "Listening…",
  hearing: "Hearing you…",
  transcribing: "Writing down what you said…",
  thinking: "Thinking…",
  speaking: "Speaking",
  error: "Voice mode stopped",
};

/**
 * Voice mode: a hands-free conversation. Listen until you stop talking, transcribe on this
 * computer, send, read the reply aloud, and listen again, until Esc or End. While it speaks, the
 * microphone is off (so it does not hear itself); "Interrupt" or a click stops the reading and
 * listens again.
 */
export function VoiceMode(props: { onAsk: (text: string) => Promise<string>; onClose: () => void }): JSX.Element {
  const service = useService();
  const [phase, setPhase] = createSignal<Phase>("listening");
  const [heard, setHeard] = createSignal("");
  const [reply, setReply] = createSignal("");
  const [level, setLevel] = createSignal(0);
  const [error, setError] = createSignal("");
  const [sensitivity, setSensitivity] = createSignal(Number(readPref("cerebro-voice-sensitivity") ?? "0.6") || 0.6);
  let alive = true;
  let poll: ReturnType<typeof setInterval> | undefined;

  const stopPolling = () => clearInterval(poll);

  const listen = async (): Promise<void> => {
    if (!alive) return;
    setPhase("listening");
    setLevel(0);
    try {
      await service.startRecording();
    } catch (e) {
      setError(e instanceof Error ? e.message : "The microphone could not start.");
      setPhase("error");
      return;
    }
    const vad = new EndOfSpeech({ threshold: thresholdFor(sensitivity()), silenceMs: 800, minSpeechMs: 250 });
    const t0 = performance.now();
    await new Promise<void>((resolve) => {
      poll = setInterval(() => {
        if (!alive) {
          stopPolling();
          return resolve();
        }
        void service.recordingLevel().then((v) => {
          setLevel(v);
          const st = vad.feed(v, performance.now() - t0);
          if (st === "speaking" && phase() === "listening") setPhase("hearing");
          if (st === "done") {
            stopPolling();
            resolve();
          }
          // Two minutes is the longest a turn records.
          if (performance.now() - t0 > 120_000) {
            stopPolling();
            resolve();
          }
        });
      }, 100);
    });
    if (!alive) return;
    setPhase("transcribing");
    let text = "";
    try {
      text = (await service.transcribe(Math.min(120, (performance.now() - t0) / 1000))).trim();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not transcribe that.");
      setPhase("error");
      return;
    }
    if (!alive) return;
    if (!text) {
      void listen();
      return;
    }
    setHeard(text);
    setPhase("thinking");
    let answer = "";
    try {
      answer = await props.onAsk(text);
    } catch (e) {
      setError(e instanceof Error ? e.message : "The model did not answer.");
      setPhase("error");
      return;
    }
    if (!alive) return;
    setReply(answer);
    setPhase("speaking");
    try {
      await service.readAloud(plainText(parseMarkdown(answer)));
    } catch {
      // Reading aloud needs a speech model; without one the reply is shown and listening goes on.
    }
    void listen();
  };

  const interrupt = () => {
    if (phase() === "speaking") void service.stopSpeaking();
  };

  const close = () => {
    alive = false;
    stopPolling();
    void service.stopSpeaking();
    props.onClose();
  };

  onMount(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        close();
      }
    };
    window.addEventListener("keydown", onKey);
    onCleanup(() => window.removeEventListener("keydown", onKey));
    void listen();
  });
  onCleanup(() => {
    alive = false;
    stopPolling();
  });

  return (
    <Portal>
      <div class="voice" role="dialog" aria-modal="true" aria-label="Voice mode" data-testid="voice-mode" data-phase={phase()}>
        <div class="voice__orb" aria-hidden="true" style={{ "--level": String(Math.min(1, level() * 4)) }} />
        <p class="voice__phase" role="status" aria-live="polite">
          {LABELS[phase()]}
        </p>
        <Show when={heard()}>
          <p class="voice__heard">
            <span class="voice__who">You</span> {heard()}
          </p>
        </Show>
        <Show when={reply() && (phase() === "speaking" || phase() === "listening")}>
          <p class="voice__reply">{reply().length > 400 ? `${reply().slice(0, 400)}…` : reply()}</p>
        </Show>
        <Show when={phase() === "error"}>
          <p class="voice__error" role="alert">
            <Icon name="alert" size={14} /> {error()}
          </p>
        </Show>
        <label class="voice__sens">
          <span>Microphone sensitivity</span>
          <input type="range" min="0" max="1" step="0.05" value={sensitivity()} onInput={(e) => {
            const v = Number(e.currentTarget.value);
            setSensitivity(v);
            writePref("cerebro-voice-sensitivity", String(v));
          }} />
        </label>
        <div class="voice__actions">
          <Show when={phase() === "speaking"}>
            <button type="button" class="btn btn--secondary" onClick={interrupt}>
              Interrupt
            </button>
          </Show>
          <Show when={phase() === "error"}>
            <button type="button" class="btn btn--secondary" onClick={() => void listen()}>
              Try again
            </button>
          </Show>
          <button type="button" class="btn btn--primary" onClick={close}>
            End voice mode
          </button>
        </div>
        <p class="voice__hint">Speak, then pause: the turn ends after a short silence. Esc ends voice mode.</p>
      </div>
    </Portal>
  );
}
