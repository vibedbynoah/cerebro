import { createResource, createSignal, For, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { Section } from "../../components/Screen";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import type { CloudPreset, CloudProvider } from "../../service/types";

const errText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/**
 * Settings > Cloud models: models the user already pays for elsewhere, through their own API key or
 * plan. The provider's site handles sign-up and billing; Cerebro stores the key in the Windows
 * Credential Manager and sends chats with these models straight to the provider.
 */
export function CloudModels(): JSX.Element {
  const service = useService();
  const [presets] = createResource(() => service.cloudPresets().catch(() => [] as CloudPreset[]));
  const [providers, { mutate }] = createResource(() => service.cloudProviders().catch(() => [] as CloudProvider[]));
  const [editing, setEditing] = createSignal<CloudProvider | null>(null);
  const [key, setKey] = createSignal("");
  const [found, setFound] = createSignal<string[] | null>(null);
  const [filter, setFilter] = createSignal("");
  const [busy, setBusy] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

  const startAdd = (presetId: string) => {
    const p = (presets() ?? []).find((x) => x.id === presetId);
    const taken = new Set((providers() ?? []).map((x) => x.id));
    let id = p?.id ?? "custom";
    for (let i = 2; taken.has(id); i++) id = `${p?.id ?? "custom"}-${i}`;
    setEditing({ id, name: p?.name ?? "My provider", kind: p?.kind ?? "openAi", baseUrl: p?.baseUrl ?? "https://", models: [], hasKey: false });
    setKey("");
    setFound(null);
    setFilter("");
    setError(null);
  };
  const preset = () => (presets() ?? []).find((p) => editing()?.id.startsWith(p.id));

  const test = async () => {
    const p = editing();
    if (!p) return;
    setBusy(true);
    setError(null);
    try {
      const models = await service.listCloudModels(p, key() || null);
      setFound(models);
      if (!p.models.length) setEditing({ ...p, models: models.slice(0, 5) });
      notify("Connected", `${p.name} offers ${models.length} model${models.length === 1 ? "" : "s"} with this key.`, "success");
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    const p = editing();
    if (!p) return;
    if (!p.models.length) {
      setError("Choose at least one model to show in Chat.");
      return;
    }
    setBusy(true);
    try {
      mutate(await service.saveCloudProvider(p, key() || null));
      notify("Saved", `${p.models.length} ${p.name} model${p.models.length === 1 ? "" : "s"} now appear in Chat.`, "success");
      setEditing(null);
      setKey("");
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const toggleModel = (m: string, on: boolean) => {
    const p = editing();
    if (!p) return;
    setEditing({ ...p, models: on ? [...p.models, m] : p.models.filter((x) => x !== m) });
  };

  return (
    <Section id="cloud-models" title="Cloud models" description="Use models you already pay for elsewhere, with your own API key or plan. Sign-up and billing happen on the provider's site. Chats with these models go from this computer straight to the provider over the internet; your local models stay on this computer.">
      <div class="cloud" data-testid="cloud-models">
        <Show when={(providers() ?? []).length}>
          <ul class="cloud__list">
            <For each={providers() ?? []}>
              {(p) => (
                <li class="cloud__provider">
                  <Icon name="online" size={18} />
                  <div class="cloud__info">
                    <strong>{p.name}</strong>
                    <span>
                      {p.models.length} model{p.models.length === 1 ? "" : "s"} in Chat · {p.hasKey ? "key stored in Windows Credential Manager" : "no key stored"}
                    </span>
                  </div>
                  <button type="button" class="btn btn--tertiary btn--sm" onClick={() => {
                    setEditing({ ...p });
                    setKey("");
                    setFound(null);
                    setError(null);
                  }}>
                    Edit
                  </button>
                  <button type="button" class="btn btn--tertiary btn--sm" aria-label={`Remove ${p.name}`} onClick={() => void service.removeCloudProvider(p.id).then((l) => {
                    mutate(l);
                    notify("Removed", `${p.name} and its stored key are gone.`);
                  }, (e: unknown) => notify("Could not remove", errText(e), "error"))}>
                    Remove
                  </button>
                </li>
              )}
            </For>
          </ul>
        </Show>

        <Show
          when={editing()}
          fallback={
            <label class="cloud__add">
              <span>Add a provider</span>
              <select value="" onChange={(e) => e.currentTarget.value && startAdd(e.currentTarget.value)} aria-label="Add a provider">
                <option value="">Choose…</option>
                <For each={presets() ?? []}>{(p) => <option value={p.id}>{p.name}</option>}</For>
                <option value="custom">Another service (OpenAI-compatible address)</option>
              </select>
            </label>
          }
        >
          {(p) => (
            <form class="cloud__form card" onSubmit={(e) => {
              e.preventDefault();
              void save();
            }}>
              <div class="cloud__fields">
                <label>
                  <span>Name</span>
                  <input value={p().name} maxLength={60} onInput={(e) => setEditing({ ...p(), name: e.currentTarget.value })} />
                </label>
                <label>
                  <span>Address</span>
                  <input class="mono" value={p().baseUrl} maxLength={300} onInput={(e) => setEditing({ ...p(), baseUrl: e.currentTarget.value })} disabled={!!preset()} />
                </label>
                <label>
                  <span>API key {p().hasKey ? "(leave empty to keep the stored one)" : ""}</span>
                  <input type="password" class="mono" autocomplete="off" value={key()} maxLength={400} onInput={(e) => setKey(e.currentTarget.value)} placeholder={p().hasKey ? "Stored" : "Paste your key"} />
                </label>
              </div>
              <Show when={preset()}>
                {(pr) => (
                  <p class="cloud__hint">
                    No key yet?{" "}
                    <a href={pr().keysUrl} target="_blank" rel="noreferrer noopener">
                      Create one on {pr().name}'s site
                    </a>
                    . Pricing and plans are set there.
                  </p>
                )}
              </Show>
              <div class="cloud__actions">
                <button type="button" class="btn btn--secondary" onClick={() => void test()} disabled={busy() || (!key() && !p().hasKey)}>
                  {busy() ? "Checking…" : "Test and list models"}
                </button>
              </div>
              <Show when={found()}>
                {(list) => (
                  <fieldset class="cloud__models">
                    <legend>Models to show in Chat ({p().models.length} chosen)</legend>
                    <input class="cloud__filter" type="search" placeholder="Filter models" value={filter()} onInput={(e) => setFilter(e.currentTarget.value)} aria-label="Filter models" />
                    <div class="cloud__model-list">
                      <For each={list().filter((m) => m.toLowerCase().includes(filter().toLowerCase())).slice(0, 300)}>
                        {(m) => (
                          <label class="cloud__model">
                            <input type="checkbox" checked={p().models.includes(m)} onChange={(e) => toggleModel(m, e.currentTarget.checked)} />
                            <span class="mono">{m}</span>
                          </label>
                        )}
                      </For>
                    </div>
                  </fieldset>
                )}
              </Show>
              <Show when={!found() && p().models.length}>
                <p class="cloud__hint">Shown in Chat: {p().models.join(", ")}. Test to change the list.</p>
              </Show>
              <Show when={error()}>
                <p class="cloud__error" role="alert">
                  <Icon name="alert" size={14} /> {error()}
                </p>
              </Show>
              <div class="cloud__actions">
                <button type="submit" class="btn btn--primary" disabled={busy()}>
                  Save
                </button>
                <button type="button" class="btn btn--tertiary" onClick={() => setEditing(null)}>
                  Cancel
                </button>
              </div>
            </form>
          )}
        </Show>
      </div>
    </Section>
  );
}
