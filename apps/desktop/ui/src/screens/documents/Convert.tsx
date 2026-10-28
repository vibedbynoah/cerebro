import { createSignal, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { Section } from "../../components/Screen";
import { CONVERT_FORMATS, convertBytes, type Converted } from "../../lib/convert/index";
import { saveText } from "../../lib/download";
import { notify } from "../../lib/toast";

/**
 * Documents > Convert: turn a file into Markdown or plain text on this computer, and save the
 * result where the user picks. The conversion never leaves the app.
 */
export function DocumentsConvert(): JSX.Element {
  const [file, setFile] = createSignal<{ name: string; result: Converted } | null>(null);
  const [format, setFormat] = createSignal<"markdown" | "text">("markdown");
  const [name, setName] = createSignal("");
  const [error, setError] = createSignal("");

  const choose = async (picked: FileList | File[]) => {
    const f = Array.from(picked)[0];
    if (!f) return;
    setError("");
    try {
      const bytes = new Uint8Array(await f.arrayBuffer());
      const result = convertBytes(f.name, bytes);
      setFile({ name: f.name, result });
      setName(`${f.name.replace(/\.[^.]+$/, "")}.${format() === "markdown" ? "md" : "txt"}`);
    } catch (e) {
      setFile(null);
      setError(e instanceof Error ? e.message : "That file could not be converted.");
    }
  };

  const text = () => {
    const f = file();
    if (!f) return "";
    return format() === "markdown" ? f.result.markdown : f.result.text;
  };

  const save = () => {
    const f = file();
    if (!f) return;
    const out = name().trim() || `converted.${format() === "markdown" ? "md" : "txt"}`;
    const ok = saveText(out, format() === "markdown" ? "text/markdown" : "text/plain", text());
    notify(ok ? "Saved" : "Could not save", ok ? out : undefined, ok ? "success" : "error");
  };

  return (
    <>
      <Section
        id="doc-convert"
        title="Convert"
        description="Turn a file into Markdown or plain text, on this computer. The result is saved to the chosen place."
      >
        <ul class="doc-formats" role="list">
          {CONVERT_FORMATS.map((f) => (
            <li class="chip">{f.label}</li>
          ))}
        </ul>
        <label class="doc-pick">
          <Icon name="upload" size={18} />
          <span class="btn__label">Choose a file</span>
          <input class="sr-only" type="file" onChange={(e) => e.currentTarget.files && void choose(e.currentTarget.files)} />
        </label>
        <Show when={error()}>
          <p class="notice" role="alert">
            <Icon name="alert" size={16} /> <span>{error()}</span>
          </p>
        </Show>
      </Section>

      <Show when={file()}>
        {(f) => (
          <Section id="doc-convert-result" title={`Result: ${f().name}`} description={`${f().result.chars.toLocaleString()} characters of clean text${f().result.pages ? `, ${f().result.pages} pages` : ""}.`}>
            <div class="doc-convert__row">
              <label class="composer__select">
                <span>Save as</span>
                <select
                  value={format()}
                  onChange={(e) => {
                    const next = e.currentTarget.value === "text" ? "text" : "markdown";
                    setFormat(next);
                    setName(`${f().name.replace(/\.[^.]+$/, "")}.${next === "markdown" ? "md" : "txt"}`);
                  }}
                >
                  <option value="markdown">Markdown</option>
                  <option value="text">Plain text</option>
                </select>
              </label>
              <label class="doc-field">
                <span>File name</span>
                <input class="input" value={name()} maxlength="120" onInput={(e) => setName(e.currentTarget.value)} />
              </label>
              <button type="button" class="btn btn--primary btn--sm" onClick={save} data-press>
                <Icon name="download" size={14} />
                <span class="btn__label">Save</span>
              </button>
            </div>
            <pre class="doc-preview">{text()}</pre>
          </Section>
        )}
      </Show>
    </>
  );
}
