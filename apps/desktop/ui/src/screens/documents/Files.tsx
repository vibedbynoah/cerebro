import { createSignal, For, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { Section } from "../../components/Screen";
import { newId } from "../../lib/chat/store";
import { addToCollection, docsInCollection, makeCollection, removeFromCollection } from "../../lib/collections";
import { convertBytes } from "../../lib/convert/index";
import { checkFile, makeDoc, MAX_TOTAL_CHARS, totalChars, type Doc } from "../../lib/documents";
import { notify } from "../../lib/toast";
import type { Collection } from "../../lib/collections";

/** What happened to one file the user added. */
export interface AddResult {
  name: string;
  ok: boolean;
  /** The characters of clean text, when it worked. */
  chars?: number;
  /** The pages the file had, when the format counts pages. */
  pages?: number | null;
  /** Why it failed, when it did. */
  error?: string;
}

const size = (b: number) => (b >= 1_000_000 ? `${(b / 1_000_000).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1000))} KB`);

/**
 * Documents > Files: adding files (they are converted to clean text on this computer and indexed),
 * the collections they are grouped in, and what each file is. Nothing is uploaded.
 */
export function DocumentsFiles(props: {
  docs: readonly Doc[];
  setDocs: (docs: Doc[]) => void;
  collections: readonly Collection[];
  setCollections: (list: Collection[]) => void;
}): JSX.Element {
  const [results, setResults] = createSignal<AddResult[]>([]);
  const [over, setOver] = createSignal(false);
  const [newCollection, setNewCollection] = createSignal("");
  const [editing, setEditing] = createSignal("");

  const add = async (files: FileList | File[]) => {
    const done: AddResult[] = [];
    for (const f of Array.from(files).slice(0, 50)) {
      const err = checkFile(f.name, f.size);
      if (err) {
        done.push({ name: f.name, ok: false, error: err });
        continue;
      }
      let text = "";
      let chars = 0;
      let pages: number | null = null;
      try {
        const bytes = new Uint8Array(await f.arrayBuffer());
        const converted = convertBytes(f.name, bytes);
        text = converted.text;
        chars = converted.chars;
        pages = converted.pages;
      } catch (e) {
        done.push({ name: f.name, ok: false, error: e instanceof Error ? e.message : "The file could not be read." });
        continue;
      }
      if (!text.trim()) {
        done.push({ name: f.name, ok: false, error: "No text could be read from this file." });
        continue;
      }
      const doc = makeDoc(newId(), f.name, f.type, f.size, text, Date.now());
      if (totalChars(props.docs) + doc.text.length > MAX_TOTAL_CHARS) {
        done.push({ name: f.name, ok: false, error: "The index is full. Remove a file to make room." });
        continue;
      }
      props.setDocs([doc, ...props.docs]);
      done.push({ name: f.name, ok: true, chars, pages });
    }
    if (done.length) setResults(done);
  };

  const remove = (id: string) => {
    props.setDocs(props.docs.filter((d) => d.id !== id));
  };

  const createCollection = () => {
    const name = newCollection().trim();
    if (!name) return;
    if (props.collections.some((c) => c.name.toLowerCase() === name.toLowerCase())) {
      notify("That collection already exists", name, "warning");
      return;
    }
    props.setCollections([...props.collections, makeCollection(name)]);
    setNewCollection("");
    notify("Collection added", name, "success");
  };

  const toggleIn = (doc: Doc, collection: Collection) => {
    const inIt = doc.collections.includes(collection.id);
    const next = inIt ? removeFromCollection(props.docs, [doc.id], collection.id) : addToCollection(props.docs, [doc.id], collection.id);
    props.setDocs(next);
  };

  const setFolder = (doc: Doc, folder: string) => {
    const value = folder.trim().slice(0, 60);
    props.setDocs(props.docs.map((d) => (d.id === doc.id ? { ...d, folder: value || null } : d)));
  };

  const setTags = (doc: Doc, tags: string) => {
    const value = [...new Set(tags.split(",").map((t) => t.trim().slice(0, 30)).filter(Boolean))].slice(0, 8);
    props.setDocs(props.docs.map((d) => (d.id === doc.id ? { ...d, tags: value } : d)));
  };

  const collectionsOf = (doc: Doc) => props.collections.filter((c) => doc.collections.includes(c.id)).map((c) => c.name);

  return (
    <>
      <Section id="doc-add" title="Add files" description="Files are converted to clean text on this computer and made searchable. Nothing is uploaded.">
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
          <span class="hint">
            PDF, Word (.docx), HTML, Markdown, CSV, JSON and plain text. Indexed on this computer; nothing is uploaded.
          </span>
          <input class="sr-only" type="file" multiple onChange={(e) => e.currentTarget.files && void add(e.currentTarget.files)} />
        </label>
        <Show when={results().length}>
          <ul class="doc-results" role="list" aria-label="What happened to each file">
            <For each={results()}>
              {(r) => (
                <li class="doc-result" data-ok={r.ok ? "true" : "false"}>
                  <Icon name={r.ok ? "check" : "alert"} size={14} />
                  <span class="doc-result__name">{r.name}</span>
                  <span class="hint mono">
                    {r.ok ? `${r.chars?.toLocaleString()} characters${r.pages ? ` · ${r.pages} pages` : ""}` : r.error}
                  </span>
                </li>
              )}
            </For>
          </ul>
        </Show>
      </Section>

      <Section id="doc-collections" title={`Collections (${props.collections.length})`} description="Named groups of files. Chat can search one collection instead of everything.">
        <div class="doc-new-collection">
          <label class="sr-only" for="doc-collection-name">
            New collection
          </label>
          <input
            id="doc-collection-name"
            class="input"
            value={newCollection()}
            maxlength="60"
            placeholder="Collection name"
            onInput={(e) => setNewCollection(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                createCollection();
              }
            }}
          />
          <button type="button" class="btn btn--tertiary btn--sm" onClick={createCollection} disabled={!newCollection().trim()} data-press>
            <Icon name="plus" size={14} />
            <span class="btn__label">Add collection</span>
          </button>
        </div>
        <Show when={props.collections.length} fallback={<p class="hint">No collections yet.</p>}>
          <ul class="doc-collections" role="list">
            <For each={props.collections}>
              {(c) => (
                <li class="doc-collection">
                  <Icon name="folder" size={14} />
                  <span class="doc-collection__name">{c.name}</span>
                  <span class="hint mono">{docsInCollection(props.docs, c.id).length} files</span>
                  <button
                    type="button"
                    class="doc-x"
                    aria-label={`Delete the collection ${c.name}`}
                    onClick={() => {
                      props.setDocs(props.docs.map((d) => ({ ...d, collections: d.collections.filter((x) => x !== c.id) })));
                      props.setCollections(props.collections.filter((x) => x.id !== c.id));
                    }}
                  >
                    <Icon name="trash" size={14} />
                  </button>
                </li>
              )}
            </For>
          </ul>
        </Show>
      </Section>

      <Section id="doc-list" title={`Indexed files (${props.docs.length})`}>
        <ul class="doc-list" role="list">
          <For each={props.docs} fallback={<li class="hint">No files yet.</li>}>
            {(d) => (
              <li class="doc-row" data-doc={d.id}>
                <Icon name="documents" size={16} />
                <span class="doc-row__name">{d.name}</span>
                <span class="hint mono">
                  {size(d.bytes)} · {d.chunks} passages
                </span>
                <span class="chip" data-tone="on">
                  Ready
                </span>
                <button type="button" class="doc-x" aria-label={`Remove ${d.name}`} onClick={() => remove(d.id)}>
                  <Icon name="trash" size={14} />
                </button>
                <details class="doc-more" open={editing() === d.id} on:toggle={(e) => setEditing(e.currentTarget.open ? d.id : "")}>
                  <summary class="icon-btn icon-btn--sm" aria-label={`Details of ${d.name}`} title="Folder, tags and collections">
                    <Icon name="sliders" size={14} />
                  </summary>
                  <div class="doc-more__body">
                    <label class="doc-field">
                      <span>Folder</span>
                      <input class="input" value={d.folder ?? ""} maxlength="60" placeholder="No folder" onChange={(e) => setFolder(d, e.currentTarget.value)} />
                    </label>
                    <label class="doc-field">
                      <span>Tags, separated by commas</span>
                      <input class="input" value={d.tags.join(", ")} maxlength="260" placeholder="No tags" onChange={(e) => setTags(d, e.currentTarget.value)} />
                    </label>
                    <Show when={props.collections.length}>
                      <fieldset class="doc-field">
                        <legend>Collections</legend>
                        <For each={props.collections}>
                          {(c) => (
                            <label class="composer__check">
                              <input type="checkbox" checked={d.collections.includes(c.id)} onChange={() => toggleIn(d, c)} />
                              <span>{c.name}</span>
                            </label>
                          )}
                        </For>
                      </fieldset>
                    </Show>
                    <Show when={collectionsOf(d).length}>
                      <p class="hint">In: {collectionsOf(d).join(", ")}</p>
                    </Show>
                  </div>
                </details>
              </li>
            )}
          </For>
        </ul>
      </Section>
    </>
  );
}
