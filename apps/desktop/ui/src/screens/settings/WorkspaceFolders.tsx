import { createSignal, For, onMount, Show, type JSX } from "solid-js";
import { Section } from "../../components/Screen";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";

/**
 * The folders Cerebro's built-in tools (read_file, write_file, run_command, …) may touch. Every
 * path a tool is given is resolved and must land inside one of these; anything else is refused.
 * Adding a folder here is what opens it to the model, and the app saves the list on every change.
 */
export function WorkspaceFolders(): JSX.Element {
  const service = useService();
  const errMessage = (e: unknown): string => (e instanceof Error ? e.message : "Unknown error");

  const [folders, setFolders] = createSignal<string[] | null>(null);
  const [draft, setDraft] = createSignal("");
  const [saving, setSaving] = createSignal(false);

  onMount(async () => {
    try {
      setFolders(await service.builtinFoldersGet());
    } catch (e) {
      notify("Could not load workspace folders", errMessage(e), "error");
      setFolders([]);
    }
  });

  // Every change replaces the whole list: the app validates each path (it must exist and be a
  // folder), saves it, and answers with what it kept.
  const save = async (next: string[]) => {
    setSaving(true);
    try {
      setFolders(await service.builtinFoldersSet(next));
    } catch (e) {
      notify("Could not save the folders", errMessage(e), "error");
    } finally {
      setSaving(false);
    }
  };

  const add = (event: SubmitEvent) => {
    event.preventDefault();
    const path = draft().trim();
    if (!path) return;
    const current = folders() ?? [];
    if (current.length >= 32) {
      notify("At most 32 folders", "Remove one before adding another.", "warning");
      return;
    }
    setDraft("");
    void save([...current, path]);
  };

  const remove = (path: string) => void save((folders() ?? []).filter((f) => f !== path));

  return (
    <Section
      id="workspace-folders"
      title="Workspace folders"
      description="Where Cerebro's built-in tools may read and write: files, search and commands. Each path must exist; a path outside these folders is refused, symlinks included."
    >
      <div class="folders">
        <Show when={folders()} fallback={<p class="hint">Loading folders…</p>}>
          {(list) => (
            <ul class="folders__list">
              <For each={list()} fallback={<li class="hint">No folders yet. The built-in tools refuse every path until one is added.</li>}>
                {(folder) => (
                  <li class="folders__item">
                    <code class="folders__path">{folder}</code>
                    <button class="btn btn--tertiary btn--sm" type="button" disabled={saving()} aria-label={`Remove ${folder}`} onClick={() => remove(folder)}>
                      Remove
                    </button>
                  </li>
                )}
              </For>
            </ul>
          )}
        </Show>
        <form class="folders__form" onSubmit={add}>
          <label class="folders__field">
            <span>Folder path</span>
            <input class="input" aria-label="Workspace folder path" value={draft()} onInput={(e) => setDraft(e.currentTarget.value)} placeholder="C:\Users\name\code" />
          </label>
          <button class="btn btn--primary btn--sm" type="submit" disabled={saving()}>
            Add folder
          </button>
        </form>
      </div>
    </Section>
  );
}
