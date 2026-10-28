import { createMemo, createSignal, For, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { Section } from "../../components/Screen";
import { Term } from "../../components/Term";
import { passages, search, type Doc } from "../../lib/documents";

/**
 * Documents > Search: the passages of the indexed files that share the words of a question, with
 * the file each came from. Keyword search (BM25) over the passages; the screen says so, because
 * search by meaning comes with the engine's embedding models.
 */
export function DocumentsSearch(props: { docs: readonly Doc[]; opened: { docId: string; index: number } | null }): JSX.Element {
  const [query, setQuery] = createSignal("");
  const hits = createMemo(() => search(props.docs, query()));
  const ready = () => props.docs.filter((d) => d.progress >= 1);
  const openedDoc = createMemo(() => props.docs.find((d) => d.id === props.opened?.docId) ?? null);
  const openedPassage = createMemo(() => {
    const doc = openedDoc();
    if (!doc || !props.opened) return null;
    return passages(doc.text)[props.opened.index] ?? null;
  });

  return (
    <>
      <Show when={openedPassage()}>
        {(text) => (
          <Section id="doc-opened" title="Opened from Chat" description="The passage the last answer used.">
            <article class="doc-hit card">
              <div class="doc-hit__head">
                <Icon name="documents" size={16} />
                <span class="doc-hit__name">{openedDoc()?.name}</span>
              </div>
              <p class="doc-hit__text">{text()}</p>
            </article>
          </Section>
        )}
      </Show>

      <Section
        id="doc-search"
        title="Search"
        description={
          <>
            Ask a question; the passages that share its words come back with their file. Keyword search for now; search by meaning (<Term k="rag" />) comes with the embedding engine.
          </>
        }
      >
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
    </>
  );
}
