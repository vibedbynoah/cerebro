import { createEffect, createMemo, createSignal, For, on, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { saveText } from "../../lib/download";
import { exportPrompts, fillPromptTemplate, mergePrompts, parsePromptJson, promptCategories, promptVariables, type SavedPrompt } from "../../lib/chat/prompts";
import { newId } from "../../lib/chat/store";
import { notify } from "../../lib/toast";

/**
 * The prompt library in Chat's side panel: searchable saved prompts, a small form for
 * {{variables}}, editing, and plain JSON import/export. A starter set is included; everything is
 * stored on this computer.
 */
export function PromptLibrary(props: {
  prompts: SavedPrompt[];
  setPrompts: (fn: (list: SavedPrompt[]) => SavedPrompt[]) => void;
  onUse: (text: string) => void;
}): JSX.Element {
  const [query, setQuery] = createSignal("");
  const [category, setCategory] = createSignal("all");
  // Start on the first prompt, but let Close put the panel back to the plain list.
  const [selectedId, setSelectedId] = createSignal(props.prompts[0]?.id ?? "");
  const [values, setValues] = createSignal<Record<string, string>>({});
  const [editing, setEditing] = createSignal<SavedPrompt | null>(null);
  let fileInput!: HTMLInputElement;

  const categories = createMemo(() => promptCategories(props.prompts));
  const selected = createMemo(() => props.prompts.find((p) => p.id === selectedId()));
  const variables = createMemo(() => (selected() ? promptVariables(selected()!.text) : []));
  const filtered = createMemo(() => {
    const q = query().trim().toLowerCase();
    return props.prompts.filter((p) => {
      if (category() !== "all" && p.category !== category()) return false;
      return !q || `${p.title} ${p.text} ${p.category}`.toLowerCase().includes(q);
    });
  });

  createEffect(
    on(selected, (prompt) => {
      const next: Record<string, string> = {};
      for (const name of prompt ? promptVariables(prompt.text) : []) next[name] = "";
      setValues(next);
    }),
  );

  const use = () => {
    const prompt = selected();
    if (!prompt) return;
    const missing = variables().find((v) => !values()[v]?.trim());
    if (missing) {
      notify("Fill in the variable", `“${missing}” needs a value before the prompt can be used.`, "warning");
      return;
    }
    props.onUse(fillPromptTemplate(prompt.text, values()));
  };

  const startNew = () => {
    setEditing({ id: newId(), title: "", text: "", category: categories()[0] ?? "Other" });
  };

  const saveEdit = () => {
    const current = editing();
    if (!current) return;
    const title = current.title.trim();
    const text = current.text.trim();
    if (!title || !text) {
      notify("Add a title and text", "Both are needed to save a prompt.", "warning");
      return;
    }
    const next: SavedPrompt = { id: current.id, title: title.slice(0, 120), text: text.slice(0, 20_000), category: current.category.trim().slice(0, 40) || "Other" };
    props.setPrompts((list) => (list.some((p) => p.id === next.id) ? list.map((p) => (p.id === next.id ? next : p)) : [...list, next]));
    setSelectedId(next.id);
    setEditing(null);
    notify("Prompt saved", next.title, "success");
  };

  const remove = () => {
    const current = editing();
    if (!current) return;
    props.setPrompts((list) => list.filter((p) => p.id !== current.id));
    if (selectedId() === current.id) setSelectedId("");
    setEditing(null);
    notify("Prompt deleted", current.title || "The draft", "info");
  };

  const importFile = async (file: File | null) => {
    if (!file) return;
    const raw = await file.text().catch(() => null);
    const parsed = raw === null ? null : parsePromptJson(raw);
    if (parsed === null) {
      notify("Import failed", "That file is not a prompt library in JSON format.", "error");
      return;
    }
    if (!parsed.length) {
      notify("Nothing to import", "The file has no valid prompts.", "info");
      return;
    }
    props.setPrompts((list) => mergePrompts(list, parsed));
    setSelectedId(parsed[0]!.id);
    notify(`Imported ${parsed.length} prompt${parsed.length === 1 ? "" : "s"}`, "They are in the library now.", "success");
  };

  const exportFile = () => {
    if (saveText("cerebro-prompts.json", "application/json", exportPrompts(props.prompts))) notify("Prompts exported", `${props.prompts.length} prompts in cerebro-prompts.json.`, "success");
    else notify("Could not export", "The browser blocked the download.", "error");
  };

  return (
    <div class="prompt-library">
      <div class="prompt-library__toolbar">
        <button class="btn btn--secondary btn--sm" type="button" onClick={startNew} data-press>
          <Icon name="plus" size={14} />
          New prompt
        </button>
        <button class="btn btn--tertiary btn--sm" type="button" onClick={() => fileInput.click()}>
          <Icon name="upload" size={14} />
          Import
        </button>
        <button class="btn btn--tertiary btn--sm" type="button" onClick={exportFile}>
          <Icon name="download" size={14} />
          Export
        </button>
        <input
          ref={fileInput}
          type="file"
          accept="application/json,.json"
          hidden
          onChange={(e) => {
            // currentTarget is cleared once the handler returns; keep the input for the async reset.
            const input = e.currentTarget;
            void importFile(input.files?.[0] ?? null).finally(() => (input.value = ""));
          }}
        />
      </div>

      <label class="prompt-library__search">
        <span class="sr-only">Search prompts</span>
        <Icon name="search" size={14} />
        <input class="input" type="search" value={query()} placeholder="Search prompts" onInput={(e) => setQuery(e.currentTarget.value)} />
      </label>
      <label class="prompt-library__category">
        <span class="sr-only">Category</span>
        <select class="input" value={category()} onChange={(e) => setCategory(e.currentTarget.value)} aria-label="Prompt category">
          <option value="all">All categories</option>
          <For each={categories()}>{(c) => <option value={c}>{c}</option>}</For>
        </select>
      </label>

      <Show
        when={editing()}
        fallback={
          <Show when={selected()} fallback={<p class="hint">Choose a prompt from the list, or make a new one.</p>}>
            {(prompt) => (
              <section class="prompt-library__use" aria-label={`Use ${prompt().title}`}>
                <div class="prompt-library__use-head">
                  <h3>{prompt().title}</h3>
                  <span class="chip" data-tone="neutral">
                    {prompt().category}
                  </span>
                </div>
                <p class="prompt-library__preview">{prompt().text}</p>
                <form
                  class="prompt-library__vars"
                  onSubmit={(e) => {
                    e.preventDefault();
                    use();
                  }}
                >
                  <Show when={variables().length} fallback={<p class="hint">This prompt has no variables. It goes into the message box as written.</p>}>
                    <p class="hint">Fill these in, then the prompt goes into the message box.</p>
                    <For each={variables()}>
                      {(name) => (
                        <label class="prompt-library__var">
                          <span>{name}</span>
                          <input
                            class="input"
                            value={values()[name] ?? ""}
                            maxlength="2000"
                            onInput={(e) => {
                              const v = e.currentTarget.value;
                              setValues((old) => ({ ...old, [name]: v }));
                            }}
                          />
                        </label>
                      )}
                    </For>
                  </Show>
                  <div class="prompt-library__actions">
                    <button class="btn btn--primary btn--sm" type="submit" data-press>
                      Use prompt
                    </button>
                    <button class="btn btn--tertiary btn--sm" type="button" onClick={() => setEditing({ ...prompt() })}>
                      <Icon name="edit" size={14} />
                      Edit
                    </button>
                    <button class="btn btn--tertiary btn--sm" type="button" onClick={() => setSelectedId("")}>
                      Close
                    </button>
                  </div>
                </form>
              </section>
            )}
          </Show>
        }
      >
        {(prompt) => (
          <form
            class="prompt-library__editor"
            onSubmit={(e) => {
              e.preventDefault();
              saveEdit();
            }}
          >
            <label class="prompt-library__var">
              <span>Title</span>
              <input
                class="input"
                value={prompt().title}
                maxlength="120"
                onInput={(e) => setEditing((old) => (old ? { ...old, title: e.currentTarget.value } : old))}
              />
            </label>
            <label class="prompt-library__var">
              <span>Category</span>
              <input
                class="input"
                list="prompt-category-options"
                value={prompt().category}
                maxlength="40"
                onInput={(e) => setEditing((old) => (old ? { ...old, category: e.currentTarget.value } : old))}
              />
              <datalist id="prompt-category-options">
                <For each={categories()}>{(c) => <option value={c} />}</For>
              </datalist>
            </label>
            <label class="prompt-library__var">
              <span>Prompt text, with variables in double braces</span>
              <textarea
                class="input prompt-library__editor-text"
                rows="7"
                value={prompt().text}
                maxlength="20000"
                onInput={(e) => setEditing((old) => (old ? { ...old, text: e.currentTarget.value } : old))}
              />
            </label>
            <p class="hint">Example: Summarise this in {"{{length}}"} and keep the key numbers.</p>
            <div class="prompt-library__actions">
              <button class="btn btn--primary btn--sm" type="submit" data-press>
                Save prompt
              </button>
              <button class="btn btn--tertiary btn--sm" type="button" onClick={() => setEditing(null)}>
                Cancel
              </button>
              <Show when={props.prompts.some((p) => p.id === prompt().id)}>
                <button class="btn btn--tertiary btn--sm prompt-library__delete" type="button" onClick={remove}>
                  <Icon name="trash" size={14} />
                  Delete
                </button>
              </Show>
            </div>
          </form>
        )}
      </Show>

      <div class="prompt-library__list-head">
        <h3>Saved prompts</h3>
        <span class="hint">{filtered().length}</span>
      </div>
      <ul class="prompt-library__list" role="list">
        <For each={filtered()} fallback={<li class="hint">No prompts match that search.</li>}>
          {(prompt) => (
            <li>
              <button
                class="prompt-library__item"
                type="button"
                aria-pressed={prompt.id === selectedId()}
                onClick={() => {
                  setSelectedId(prompt.id);
                  setEditing(null);
                }}
              >
                <span class="prompt-library__item-title">{prompt.title}</span>
                <span class="chip" data-tone="neutral">
                  {prompt.category}
                </span>
              </button>
            </li>
          )}
        </For>
      </ul>
    </div>
  );
}
