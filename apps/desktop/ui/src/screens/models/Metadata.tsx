import { createMemo, createSignal, For, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { Term } from "../../components/Term";
import { ggufText, parseGgufValue } from "../../lib/gguf";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import type { GgufCopy, GgufEdit, GgufEntry, LibraryModel } from "../../service/types";
import { Spinner } from "./parts";

type Value = string | number | boolean;

/** Section titles for the key namespaces a GGUF file uses. */
function groupTitle(ns: string, arch: string): string {
  if (ns === "general") return "General";
  if (ns === "tokenizer") return "Tokenizer";
  if (ns === arch) return `Architecture (${arch})`;
  return ns;
}

/**
 * A validated GGUF metadata viewer and editor. Every value is checked against its stored type as
 * it is typed; structural keys are read-only. Changes are written to a copy beside the original,
 * shown as a diff first, and reverted with one click (the copy is deleted). When a copy exists it
 * is the base: the table shows its values, new edits stage against it, and writing again merges
 * into it, so an earlier change is never silently dropped.
 */
export function Metadata(props: { model: LibraryModel; entries: GgufEntry[]; copy: GgufCopy | null }): JSX.Element {
  const service = useService();
  const [filter, setFilter] = createSignal("");
  /** Staged changes, relative to the copy when there is one, else to the original. */
  const [edits, setEdits] = createSignal<ReadonlyMap<string, Value>>(new Map());
  const [editing, setEditing] = createSignal<string | null>(null);
  const [copy, setCopy] = createSignal<GgufCopy | null>(props.copy);
  const [busy, setBusy] = createSignal(false);

  const originals = createMemo(() => new Map<string, Value>(props.entries.map((e) => [e.key, e.value])));
  const original = (key: string) => originals().get(key);
  /** What the copy holds, by key. */
  const base = createMemo(() => new Map<string, Value>((copy()?.edits ?? []).map((e) => [e.key, e.value])));
  const current = (key: string): Value | undefined => edits().get(key) ?? base().get(key) ?? original(key);

  /** Everything the copy will differ from the original by once written: its edits plus the staged ones. */
  const merged = createMemo<GgufEdit[]>(() => {
    const all = new Map(base());
    for (const [k, v] of edits()) {
      if (v === original(k)) all.delete(k);
      else all.set(k, v);
    }
    return [...all].map(([key, value]) => ({ key, value }));
  });
  /** What is sent: every key the copy will hold, plus keys the copy holds that go back to their
   *  original value (the service drops those from the copy). */
  const writeList = createMemo<GgufEdit[]>(() => {
    const keys = new Set([...base().keys(), ...edits().keys()]);
    return [...keys].map((key) => ({ key, value: (edits().get(key) ?? base().get(key)) as Value }));
  });

  const arch = () => String(original("general.architecture") ?? "");
  const rows = createMemo(() => {
    const f = filter().toLowerCase().trim();
    return f === "" ? props.entries : props.entries.filter((e) => e.key.toLowerCase().includes(f) || (e.type !== "string" && String(current(e.key)).toLowerCase().includes(f)));
  });
  const groups = createMemo(() => {
    const out = new Map<string, GgufEntry[]>();
    for (const e of rows()) {
      const ns = e.key.split(".")[0] ?? "other";
      const list = out.get(ns);
      if (list) list.push(e);
      else out.set(ns, [e]);
    }
    return [...out];
  });

  const stage = (key: string, value: Value) => {
    const next = new Map(edits());
    if (value === (base().get(key) ?? original(key))) next.delete(key);
    else next.set(key, value);
    setEdits(next);
  };

  const volume = () => /^[A-Za-z]:/.exec(props.model.path)?.[0].toUpperCase() ?? "the same drive";

  const write = async () => {
    if (busy()) return;
    setBusy(true);
    try {
      if (merged().length === 0) {
        // Every change in the copy was undone: the copy would equal the original, so delete it.
        await service.revertGgufCopy(props.model.id);
        setCopy(null);
        setEdits(new Map());
        notify("Copy deleted", "Nothing differed from the original any more.");
        return;
      }
      const c = await service.writeGgufCopy(props.model.id, writeList());
      setCopy(c);
      setEdits(new Map());
      notify("Copy written", `${c.edits.length} ${c.edits.length === 1 ? "change" : "changes"} in the copy. The original file is unchanged.`, "success");
    } catch {
      notify("The copy was not written", "Nothing changed on disk.", "error");
    } finally {
      setBusy(false);
    }
  };
  const revert = async () => {
    if (busy()) return;
    setBusy(true);
    try {
      await service.revertGgufCopy(props.model.id);
      setCopy(null);
      // Staged edits now compare with the original; one that equals it is no change at all.
      setEdits(new Map([...edits()].filter(([k, v]) => v !== original(k))));
      notify("Reverted", "The edited copy was deleted; only the original remains.");
    } catch {
      notify("Revert failed", "The copy is still there.", "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div class="meta">
      <p class="meta__intro">
        <Term k="ggufMeta" /> stored in <span class="mono">{props.model.file}</span>. Edits are checked while typing and written to a copy; the original is never opened for writing.
      </p>

      <Show when={copy()}>
        {(c) => (
          <div class="notice notice--inline meta__copy" role="status">
            <Icon name="check" size={18} />
            <p>
              Edited copy: <span class="mono meta__path">{c().path}</span> ({c().edits.length} {c().edits.length === 1 ? "change" : "changes"}). The table shows the copy's values.
            </p>
            <button type="button" class="btn btn--secondary btn--xs" data-press aria-busy={busy() ? "true" : undefined} onClick={() => void revert()}>
              <Icon name="undo" size={14} />
              Revert
            </button>
          </div>
        )}
      </Show>

      <label class="lib-search meta__search">
        <Icon name="search" size={16} />
        <span class="sr-only">Filter keys</span>
        <input class="lib-search__input" type="search" placeholder="Filter keys" maxLength={80} value={filter()} onInput={(e) => setFilter(e.currentTarget.value)} />
      </label>

      <div class="meta-table" role="table" aria-label="GGUF metadata">
        <div role="row" class="meta-row meta-row--head">
          <span role="columnheader">Key</span>
          <span role="columnheader">Type</span>
          <span role="columnheader">Value</span>
          <span role="columnheader">
            <span class="sr-only">Edit</span>
          </span>
        </div>
        <For each={groups()}>
          {([ns, list]) => (
            <div role="rowgroup" class="meta-group" aria-label={groupTitle(ns, arch())}>
              <div role="row" class="meta-row meta-row--group">
                <span role="rowheader" aria-colspan={4} class="meta-group__title">
                  {groupTitle(ns, arch())}
                  <span class="meta-group__count mono">{list.length}</span>
                </span>
              </div>
              <For each={list}>
                {(entry) => (
                  <MetaRow
                    entry={entry}
                    value={current(entry.key) ?? entry.value}
                    staged={edits().has(entry.key)}
                    inCopy={base().has(entry.key) && !edits().has(entry.key)}
                    editing={editing() === entry.key}
                    onEdit={() => setEditing(entry.key)}
                    onDone={() => setEditing(null)}
                    onStage={(v) => stage(entry.key, v)}
                  />
                )}
              </For>
            </div>
          )}
        </For>
      </div>

      <Show when={edits().size > 0}>
        <section class="meta-diff card" aria-labelledby="meta-diff-title">
          <h3 class="studio-card__title" id="meta-diff-title">
            {merged().length === 0 ? "The copy would match the original" : `The copy will hold ${merged().length} ${merged().length === 1 ? "change" : "changes"}`}
          </h3>
          <Show when={merged().length > 0}>
            <ul class="meta-diff__list">
              <For each={merged()}>
                {(e) => (
                  <li class="meta-diff__item" classList={{ "is-new": edits().has(e.key) }}>
                    <span class="mono meta-diff__key">{e.key}</span>
                    <span class="meta-diff__old mono">
                      <span class="sr-only">from </span>
                      {preview(original(e.key))}
                    </span>
                    <Icon name="chevron" size={14} class="meta-diff__arrow" />
                    <span class="meta-diff__new mono">
                      <span class="sr-only">to </span>
                      {preview(e.value)}
                    </span>
                    <span class="meta-diff__tag">{edits().has(e.key) ? "New" : "In copy"}</span>
                    <button type="button" class="btn btn--icon btn--xs" aria-label={`Undo change to ${e.key}`} data-press onClick={() => stage(e.key, original(e.key) ?? e.value)}>
                      <Icon name="close" size={14} />
                    </button>
                  </li>
                )}
              </For>
            </ul>
          </Show>
          <p class="hint">
            <Icon name="disk" size={16} />
            {copy() ? `Rewriting the copy needs ${props.model.sizeGiB.toFixed(1)} GiB free on ${volume()} while it is written.` : `The copy needs ${props.model.sizeGiB.toFixed(1)} GiB free on ${volume()}, beside the original.`}
          </p>
          <div class="meta-diff__actions">
            <button type="button" class="btn btn--secondary" data-press aria-busy={busy() ? "true" : undefined} onClick={() => void write()}>
              <Spinner />
              <span class="btn__label">{merged().length === 0 ? "Delete the copy" : copy() ? "Update the copy" : "Write to a copy"}</span>
            </button>
            <button type="button" class="btn btn--tertiary" data-press onClick={() => setEdits(new Map())}>
              Discard staged changes
            </button>
          </div>
        </section>
      </Show>
    </div>
  );
}

function preview(v: string | number | boolean | undefined): string {
  if (v === undefined) return "";
  const t = ggufText(v);
  const first = t.split("\n")[0] ?? "";
  return first.length > 48 ? `${first.slice(0, 48)}…` : t.includes("\n") ? `${first} …` : first;
}

function MetaRow(props: {
  entry: GgufEntry;
  /** What the row shows: staged, else the copy's, else the original value. */
  value: Value;
  staged: boolean;
  inCopy: boolean;
  editing: boolean;
  onEdit: () => void;
  onDone: () => void;
  onStage: (v: string | number | boolean) => void;
}): JSX.Element {
  const value = () => props.value;
  const [text, setText] = createSignal("");
  const [error, setError] = createSignal("");
  const long = () => props.entry.type === "string" && (ggufText(props.entry.value).includes("\n") || ggufText(props.entry.value).length > 80);

  const begin = () => {
    setText(ggufText(value()));
    setError("");
    props.onEdit();
  };
  const commit = () => {
    const r = parseGgufValue(props.entry, text());
    if (!r.ok) {
      setError(r.error);
      return;
    }
    props.onStage(r.value);
    props.onDone();
  };

  return (
    <div role="row" class="meta-row" classList={{ "is-staged": props.staged, "is-in-copy": props.inCopy, "is-editing": props.editing }}>
      <span role="cell" class="meta-row__key mono">
        {props.entry.key}
        <Show when={props.inCopy}>
          <span class="meta-row__mark">changed in copy</span>
        </Show>
        <Show when={props.staged}>
          <span class="meta-row__mark meta-row__mark--staged">staged</span>
        </Show>
      </span>
      <span role="cell" class="meta-row__type mono">
        {props.entry.type}
      </span>
      <span role="cell" class="meta-row__value">
        <Show
          when={props.editing}
          fallback={
            <span class="mono meta-row__text" classList={{ "meta-row__text--long": long() }}>
              {preview(value())}
            </span>
          }
        >
          <span class="meta-edit">
            <Show
              when={props.entry.type === "bool"}
              fallback={
                <Show
                  when={long()}
                  fallback={
                    <input
                      class="input input--sm mono"
                      aria-label={`New value for ${props.entry.key}`}
                      value={text()}
                      aria-invalid={error() ? "true" : undefined}
                      ref={(el) => requestAnimationFrame(() => el.focus())}
                      onInput={(e) => (setText(e.currentTarget.value), setError(""))}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") commit();
                        if (e.key === "Escape") (e.stopPropagation(), props.onDone());
                      }}
                    />
                  }
                >
                  <textarea
                    class="input input--sm mono meta-edit__area"
                    aria-keyshortcuts="Control+Enter"
                    rows={6}
                    aria-label={`New value for ${props.entry.key}`}
                    value={text()}
                    ref={(el) => requestAnimationFrame(() => el.focus())}
                    onInput={(e) => (setText(e.currentTarget.value), setError(""))}
                    onKeyDown={(e) => {
                      // Enter makes a new line here; Ctrl+Enter stages, like the Stage button.
                      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                        e.preventDefault();
                        commit();
                      }
                      if (e.key === "Escape") (e.stopPropagation(), props.onDone());
                    }}
                  />
                </Show>
              }
            >
              <span class="select select--sm">
                <select class="input" aria-label={`New value for ${props.entry.key}`} value={text()} onChange={(e) => setText(e.currentTarget.value)}>
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              </span>
            </Show>
            <button type="button" class="btn btn--secondary btn--xs" data-press onClick={commit}>
              Stage
            </button>
            <button type="button" class="btn btn--tertiary btn--xs" data-press onClick={() => props.onDone()}>
              Cancel
            </button>
          </span>
          <Show when={error()}>
            <span class="field__error" role="alert">
              {error()}
            </span>
          </Show>
        </Show>
      </span>
      <span role="cell" class="meta-row__action">
        <Show
          when={props.entry.editable}
          fallback={
            <span class="meta-lock" title="Structural: changing it would make the file unreadable">
              <Icon name="lock" size={14} />
              <span class="sr-only">Read-only: structural</span>
            </span>
          }
        >
          <Show when={!props.editing}>
            <button type="button" class="btn btn--tertiary btn--xs" data-press aria-label={`Edit ${props.entry.key}`} onClick={begin}>
              Edit
            </button>
          </Show>
        </Show>
      </span>
    </div>
  );
}
