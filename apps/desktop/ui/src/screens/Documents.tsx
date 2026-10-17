import "../styles/documents.css";
import { createMemo, createSignal, For, Show, type JSX } from "solid-js";
import { Icon } from "../components/icons";
import { IllustrativeFlag, ScreenHeader, Section } from "../components/Screen";
import { Term } from "../components/Term";
import { checkFile, loadDocs, makeDoc, MAX_CHARS, MAX_TOTAL_CHARS, saveDocs, search, totalChars, type Doc } from "../lib/documents";
import { newId } from "../lib/chat/store";
import { notify } from "../lib/toast";

const size = (b: number) => (b >= 1_000_000 ? `${(b / 1_000_000).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1000))} KB`);

/**
 * Documents: search your own files on this machine. Drop text files to index them (read, split into
 * passages, ranked by BM25 keyword search, lib/documents.ts); search shows the passage that answers.
 * Nothing is uploaded.
 */
export function Documents(): JSX.Element {
  const [docs, setDocsS] = createSignal<Doc[]>(loadDocs());
  const [query, setQuery] = createSignal("");
  const [over, setOver] = createSignal(false);
  const setDocs = (d: Doc[]) => {
    setDocsS(d);
    saveDocs(d);
  };

  const add = async (files: FileList | File[]) => {
    for (const f of Array.from(files).slice(0, 50)) {
      const err = checkFile(f.name, f.size);
      if (err) {
        notify(`${f.name} not added`, err, "warning");
        continue;
      }
      let raw: string;
      try {
        raw = await f.text();
      } catch {
        notify(`${f.name} not added`, "The file could not be read.", "error");
        continue;
      }
      if (raw.includes("\u0000")) {
        notify(`${f.name} not added`, "That is not a text file.", "warning");
        continue;
      }
      const doc = makeDoc(newId(), f.name, f.type, f.size, raw, Date.now());
      if (totalChars(docs()) + doc.text.length > MAX_TOTAL_CHARS) {
        notify(`${f.name} not added`, "The index is full. Remove a file to make room.", "warning");
        continue;
      }
      setDocs([doc, ...docs()]);
      notify(
        "Indexed",
        `${doc.name}: ${doc.chunks} passage${doc.chunks === 1 ? "" : "s"}, searchable now.${doc.truncated ? ` Only the first ${MAX_CHARS.toLocaleString()} characters are searchable.` : ""}`,
        doc.truncated ? "warning" : "success",
      );
    }
  };
  const remove = (id: string) => setDocs(docs().filter((d) => d.id !== id));
  const hits = createMemo(() => search(docs(), query()));
  const ready = () => docs().filter((d) => d.progress >= 1);

  return (
    <div class="screen documents">
      <ScreenHeader route="documents" actions={__DESKTOP__ ? undefined : <IllustrativeFlag />} />
      <label
        class="doc-drop"
        data-over={over()}
        onDragOver={(e) => (e.preventDefault(), setOver(true))}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          if (e.dataTransfer?.files) void add(e.dataTransfer.files);
        }}
      >
        <Icon name="upload" size={22} />
        <span class="doc-drop__title">Drop files to make them searchable</span>
        <span class="hint">Text, Markdown, HTML, CSV, JSON, logs and code. Indexed on this computer; nothing is uploaded.</span>
        <input class="sr-only" type="file" multiple onChange={(e) => e.currentTarget.files && void add(e.currentTarget.files)} />
      </label>

      <Section id="doc-search" title="Search" description={<>Ask a question; the passages that share its words come back with their file. Keyword search for now; search by meaning (<Term k="rag" />) comes with the embedding engine.</>}>
        <input class="doc-search" type="search" placeholder="How many days of annual leave?" aria-label="Search documents" value={query()} onInput={(e) => setQuery(e.currentTarget.value)} />
        <Show when={query().trim()}>
          <ol class="doc-hits" role="list" aria-live="polite">
            <For each={hits()} fallback={<li class="hint">Nothing in {ready().length} indexed file{ready().length === 1 ? "" : "s"} matches.</li>}>
              {(h) => (
                <li class="doc-hit card">
                  <div class="doc-hit__head">
                    <Icon name="documents" size={16} />
                    <span class="doc-hit__name">{h.doc.name}</span>
                    <span class="mono hint">
                      {h.matched} of {h.of} words
                    </span>
                  </div>
                  <p class="doc-hit__text">
                    <For each={h.parts}>{(p) => (p.hit ? <mark>{p.v}</mark> : p.v)}</For>
                  </p>
                </li>
              )}
            </For>
          </ol>
        </Show>
      </Section>

      <Section id="doc-list" title={`Indexed files (${ready().length})`}>
        <ul class="doc-list" role="list">
          <For each={docs()} fallback={<li class="hint">No files yet.</li>}>
            {(d) => (
              <li class="doc-row" data-doc={d.id}>
                <Icon name="documents" size={16} />
                <span class="doc-row__name">{d.name}</span>
                <span class="hint mono">
                  {size(d.bytes)} · {d.chunks} passages
                </span>
                <Show
                  when={d.progress >= 1}
                  fallback={
                    <span class="doc-progress" role="progressbar" aria-label={`Indexing ${d.name}`} aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(d.progress * 100)}>
                      <span style={{ transform: `scaleX(${d.progress})` }} />
                    </span>
                  }
                >
                  <span class="chip" data-tone="on">
                    Ready
                  </span>
                </Show>
                <button type="button" class="doc-x" aria-label={`Remove ${d.name}`} onClick={() => remove(d.id)}>
                  <Icon name="trash" size={14} />
                </button>
              </li>
            )}
          </For>
        </ul>
      </Section>
    </div>
  );
}
