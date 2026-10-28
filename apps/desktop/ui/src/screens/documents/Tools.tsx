import { createMemo, createResource, createSignal, For, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { Section } from "../../components/Screen";
import { convertBytes } from "../../lib/convert/index";
import { copyText, saveText } from "../../lib/download";
import { checkJson, QUICK_TOOLS, REWRITE_STYLES, TRANSLATE_LANGUAGES, type QuickToolDef, type QuickToolInputValue, type QuickToolModel } from "../../lib/quick-tools";
import { runQuickTool } from "../../lib/quick-tools-run";
import { useService } from "../../service/context";
import type { LibraryModel, LoadedModel } from "../../service/types";
import { notify } from "../../lib/toast";

/** The bytes of a file as base64, for the speech model. */
async function toBase64(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) binary += String.fromCharCode(...bytes.subarray(i, i + step));
  return btoa(binary);
}

/** A picture as a data URL, for the vision model. */
async function toDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : "");
    reader.onerror = () => reject(new Error("That picture could not be read."));
    reader.readAsDataURL(file);
  });
}

/**
 * Documents > Quick tools: one-step AI jobs that run on the model on this computer. Each card has
 * one input, one button, and the result with Copy and Save. Nothing leaves the computer unless a
 * cloud model is picked in Chat.
 */
export function DocumentsTools(): JSX.Element {
  const service = useService();
  const [library] = createResource(() => service.listLibrary().catch(() => [] as LibraryModel[]));
  const [loaded] = createResource(() => service.listLoaded().catch(() => [] as LoadedModel[]));

  /** The model a tool needs, and what to say when it is not there. */
  const modelFor = (need: QuickToolModel): { id: string | null; missing: string } => {
    const lib = library() ?? [];
    const loadedIds = new Set((loaded() ?? []).map((l) => l.id));
    const here = lib.filter((m) => loadedIds.has(m.id));
    if (need === "text") {
      return { id: here.find((m) => m.kind === "text")?.id ?? here.find((m) => m.kind === "vision")?.id ?? null, missing: "No text model is loaded. Load one from the Models screen, then run this again." };
    }
    if (need === "vision") {
      return { id: here.find((m) => m.kind === "vision")?.id ?? null, missing: "No vision model is loaded. Load one from the Models screen, then run this again." };
    }
    if (need === "speech-in") {
      return { id: here.find((m) => m.kind === "stt")?.id ?? lib.find((m) => m.kind === "stt")?.id ?? null, missing: "No speech-to-text model is in the library. Add one from the Model Hub." };
    }
    return { id: here.find((m) => m.kind === "tts")?.id ?? lib.find((m) => m.kind === "tts")?.id ?? null, missing: "No speech model is in the library. Add one from the Model Hub." };
  };

  return (
    <Section
      id="doc-tools"
      title="Quick tools"
      description="One step, one answer, on the model on this computer. Each tool is also a / command in Chat."
    >
      <div class="doc-tools">
        <For each={QUICK_TOOLS}>
          {(tool) => (
            <ToolCard tool={tool} service={service} model={modelFor(tool.needs)} />
          )}
        </For>
      </div>
    </Section>
  );
}

function ToolCard(props: { tool: QuickToolDef; service: ReturnType<typeof useService>; model: { id: string | null; missing: string } }): JSX.Element {
  const tool = props.tool;
  const [text, setText] = createSignal("");
  const [second, setSecond] = createSignal("");
  const [language, setLanguage] = createSignal(TRANSLATE_LANGUAGES[0] ?? "English");
  const [style, setStyle] = createSignal(REWRITE_STYLES[0]?.id ?? "clearer");
  const [fields, setFields] = createSignal("");
  const [file, setFile] = createSignal<File | null>(null);
  const [result, setResult] = createSignal("");
  const [error, setError] = createSignal("");
  const [busy, setBusy] = createSignal(false);
  let controller: AbortController | null = null;

  const isText = tool.input === "text" || tool.input === "text-and-language" || tool.input === "text-and-style" || tool.input === "text-and-fields";
  const needsFile = tool.input === "audio" || tool.input === "image";
  const ready = createMemo(() => (tool.needs === "text" || tool.needs === "vision" ? !!props.model.id : true));

  /** A file chosen for the card: a picture, an audio file, or text that fills the box. */
  const pick = async (picked: FileList | File[]) => {
    const f = Array.from(picked)[0];
    if (!f) return;
    setError("");
    if (needsFile) {
      setFile(f);
      return;
    }
    try {
      const bytes = new Uint8Array(await f.arrayBuffer());
      const converted = convertBytes(f.name, bytes);
      setText(converted.text);
      notify("File read", `${f.name}: ${converted.chars.toLocaleString()} characters of clean text.`, "success");
    } catch (e) {
      setError(e instanceof Error ? e.message : "That file could not be read.");
    }
  };

  const run = async () => {
    setError("");
    setResult("");
    const input: QuickToolInputValue = { text: text(), second: second(), language: language(), style: style(), fields: fields() };
    if (tool.id === "read") {
      if (!input.text?.trim()) {
        setError("There is no text to read.");
        return;
      }
      setBusy(true);
      try {
        await props.service.readAloud(input.text ?? "");
        setResult("The speech model read it aloud.");
      } catch (e) {
        setError(e instanceof Error ? e.message : "It could not be read aloud.");
      } finally {
        setBusy(false);
      }
      return;
    }
    if (tool.id === "notes") {
      const audio = file();
      if (!audio) {
        setError("Choose an audio file first.");
        return;
      }
      setBusy(true);
      try {
        input.text = await props.service.transcribeAudio({ name: audio.name, data: await toBase64(audio) });
      } catch (e) {
        setBusy(false);
        setError(e instanceof Error ? e.message : "That recording could not be transcribed.");
        return;
      }
    }
    if (tool.id === "describe") {
      const image = file();
      if (!image) {
        setError("Choose a picture first.");
        return;
      }
      setBusy(true);
      try {
        input.attachment = { name: image.name, type: image.type, bytes: image.size, data: await toDataUrl(image) };
      } catch (e) {
        setBusy(false);
        setError(e instanceof Error ? e.message : "That picture could not be read.");
        return;
      }
    }
    if (!props.model.id) {
      setBusy(false);
      setError(props.model.missing);
      return;
    }
    setBusy(true);
    controller = new AbortController();
    try {
      const out = await runQuickTool(props.service, {
        tool: tool.id,
        input,
        modelId: props.model.id,
        signal: controller.signal,
        onToken: (t) => setResult((r) => r + t),
      });
      if (tool.id === "extract") {
        const check = checkJson(out);
        if (!check.ok) setError(check.error ?? "The answer is not JSON.");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "The tool did not finish.");
    } finally {
      setBusy(false);
      controller = null;
    }
  };

  const copy = async () => {
    const ok = await copyText(result());
    notify(ok ? "Copied" : "Could not copy", ok ? undefined : "Select the text and copy it.", ok ? "success" : "error");
  };

  const save = () => {
    const ok = saveText(`${tool.id}.md`, "text/markdown", result());
    notify(ok ? "Saved" : "Could not save", ok ? `${tool.id}.md` : undefined, ok ? "success" : "error");
  };

  return (
    <article class="tool-card card" data-tool={tool.id}>
      <header class="tool-card__head">
        <h3 class="tool-card__title">{tool.name}</h3>
        <p class="tool-card__blurb">{tool.blurb}</p>
      </header>

      <Show when={isText}>
        <label class="tool-card__field">
          <span class="sr-only">{tool.inputLabel}</span>
          <textarea
            class="input tool-card__input"
            rows="4"
            placeholder={tool.inputLabel}
            aria-label={tool.inputLabel}
            value={text()}
            onInput={(e) => setText(e.currentTarget.value)}
          />
        </label>
        <label class="doc-pick doc-pick--small">
          <Icon name="upload" size={14} />
          <span class="btn__label">Use a file</span>
          <input class="sr-only" type="file" onChange={(e) => e.currentTarget.files && void pick(e.currentTarget.files)} />
        </label>
      </Show>

      <Show when={tool.input === "two-texts"}>
        <label class="tool-card__field">
          <span class="tool-card__label">First text</span>
          <textarea class="input tool-card__input" rows="3" aria-label="First text" value={text()} onInput={(e) => setText(e.currentTarget.value)} />
        </label>
        <label class="tool-card__field">
          <span class="tool-card__label">Second text</span>
          <textarea class="input tool-card__input" rows="3" aria-label="Second text" value={second()} onInput={(e) => setSecond(e.currentTarget.value)} />
        </label>
      </Show>

      <Show when={tool.input === "text-and-language"}>
        <label class="composer__select">
          <span>Language</span>
          <select value={language()} onChange={(e) => setLanguage(e.currentTarget.value)} aria-label="Language">
            <For each={TRANSLATE_LANGUAGES}>{(l) => <option value={l}>{l}</option>}</For>
          </select>
        </label>
      </Show>

      <Show when={tool.input === "text-and-style"}>
        <label class="composer__select">
          <span>Style</span>
          <select value={style()} onChange={(e) => setStyle(e.currentTarget.value)} aria-label="Style">
            <For each={REWRITE_STYLES}>{(s) => <option value={s.id}>{s.label}</option>}</For>
          </select>
        </label>
        <p class="hint">{REWRITE_STYLES.find((s) => s.id === style())?.note}</p>
      </Show>

      <Show when={tool.input === "text-and-fields"}>
        <label class="tool-card__field">
          <span class="tool-card__label">Fields, separated by commas</span>
          <input class="input" value={fields()} maxlength="200" placeholder="name, email" aria-label="Fields to extract" onInput={(e) => setFields(e.currentTarget.value)} />
        </label>
      </Show>

      <Show when={needsFile}>
        <label class="doc-pick doc-pick--small">
          <Icon name="upload" size={14} />
          <span class="btn__label">{file() ? file()!.name : tool.inputLabel}</span>
          <input class="sr-only" type="file" accept={tool.input === "image" ? "image/*" : "audio/*,.wav"} onChange={(e) => e.currentTarget.files && void pick(e.currentTarget.files)} />
        </label>
      </Show>

      <Show when={!ready()}>
        <p class="notice" role="status">
          <Icon name="alert" size={16} />
          <span>{props.model.missing}</span>
        </p>
      </Show>

      <div class="tool-card__actions">
        <button type="button" class="btn btn--primary btn--sm" onClick={() => void run()} disabled={busy()} data-press>
          {busy() ? "Working…" : tool.name}
        </button>
        <Show when={!ready()}>
          <button type="button" class="btn btn--tertiary btn--sm" onClick={() => (window.location.hash = "#/models")} data-press>
            <Icon name="models" size={14} />
            <span class="btn__label">Open Models</span>
          </button>
        </Show>
      </div>

      <Show when={error()}>
        <p class="notice" role="alert">
          <Icon name="alert" size={16} /> <span>{error()}</span>
        </p>
      </Show>

      <Show when={result()}>
        <div class="tool-card__result">
          <pre class="doc-preview">{result()}</pre>
          <div class="tool-card__actions">
            <button type="button" class="icon-btn" aria-label="Copy the result" onClick={() => void copy()}>
              <Icon name="copy" size={16} />
            </button>
            <button type="button" class="icon-btn" aria-label="Save the result" onClick={save}>
              <Icon name="download" size={16} />
            </button>
          </div>
        </div>
      </Show>
    </article>
  );
}
