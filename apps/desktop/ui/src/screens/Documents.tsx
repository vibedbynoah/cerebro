import "../styles/documents.css";
import { createEffect, createSignal, For, Show, type JSX } from "solid-js";
import { Icon } from "../components/icons";
import { IllustrativeFlag, ScreenHeader } from "../components/Screen";
import { loadCollections, saveCollections, type Collection } from "../lib/collections";
import { loadDocs, saveDocs, type Doc } from "../lib/documents";
import { DocumentsSearch } from "./documents/Search";
import { DocumentsFiles } from "./documents/Files";
import { DocumentsConvert } from "./documents/Convert";
import { DocumentsTools } from "./documents/Tools";

type TabId = "search" | "files" | "convert" | "tools";

const TABS: readonly { id: TabId; label: string }[] = [
  { id: "search", label: "Search" },
  { id: "files", label: "Files" },
  { id: "convert", label: "Convert" },
  { id: "tools", label: "Quick tools" },
];

/**
 * Documents: your own files on this computer. Search them, group them in collections, convert them
 * to Markdown or plain text, and run the quick tools on them. Nothing is uploaded; the screen opens
 * on search, the simplest thing to do with it.
 */
export function Documents(): JSX.Element {
  const [tab, setTab] = createSignal<TabId>("search");
  const [docs, setDocsS] = createSignal<Doc[]>(loadDocs());
  const [collections, setCollectionsS] = createSignal<Collection[]>(loadCollections());
  // The passage a chat answer used, opened from the source chip.
  const [opened, setOpened] = createSignal<{ docId: string; index: number } | null>(null);

  const setDocs = (d: Doc[]) => {
    setDocsS(d);
    saveDocs(d);
  };
  const setCollections = (list: Collection[]) => {
    setCollectionsS(list);
    saveCollections(list);
  };

  createEffect(() => {
    const hash = window.location.hash;
    const at = hash.indexOf("?");
    if (at < 0) return;
    const params = new URLSearchParams(hash.slice(at + 1));
    const docId = params.get("doc");
    if (!docId) return;
    const index = Number(params.get("p") ?? "0");
    setOpened({ docId, index: Number.isFinite(index) && index >= 0 ? index : 0 });
    setTab("search");
  });

  return (
    <div class="screen documents">
      <ScreenHeader route="documents" actions={__DESKTOP__ ? undefined : <IllustrativeFlag />} />

      <div class="doc-tabs" role="tablist" aria-label="Documents">
        <For each={TABS}>
          {(t) => (
            <button
              type="button"
              role="tab"
              id={`doc-tab-${t.id}`}
              class="doc-tab"
              aria-selected={tab() === t.id}
              aria-controls={`doc-panel-${t.id}`}
              tabindex={tab() === t.id ? 0 : -1}
              onClick={() => setTab(t.id)}
              onKeyDown={(e) => {
                if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
                e.preventDefault();
                const at = TABS.findIndex((x) => x.id === tab());
                const next = TABS[(at + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length]!;
                setTab(next.id);
                document.getElementById(`doc-tab-${next.id}`)?.focus();
              }}
            >
              {t.label}
            </button>
          )}
        </For>
      </div>

      <For each={TABS}>
        {(t) => (
          <div class="doc-panel" role="tabpanel" id={`doc-panel-${t.id}`} aria-labelledby={`doc-tab-${t.id}`} hidden={tab() !== t.id}>
            <Show when={t.id === "search"}>
              <DocumentsSearch docs={docs()} opened={opened()} />
            </Show>
            <Show when={t.id === "files"}>
              <DocumentsFiles docs={docs()} setDocs={setDocs} collections={collections()} setCollections={setCollections} />
            </Show>
            <Show when={t.id === "convert"}>
              <DocumentsConvert />
            </Show>
            <Show when={t.id === "tools"}>
              <DocumentsTools />
            </Show>
          </div>
        )}
      </For>

      <p class="hint doc-note">
        <Icon name="lock" size={14} /> Everything on this screen happens on this computer. Nothing is uploaded.
      </p>
    </div>
  );
}
