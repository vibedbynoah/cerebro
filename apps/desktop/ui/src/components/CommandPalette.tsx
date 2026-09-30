import { Dialog } from "@ark-ui/solid/dialog";
import { createEffect, createMemo, createSignal, createUniqueId, For, on, Show, type JSX } from "solid-js";
import { Portal } from "solid-js/web";
import { fuzzyFilter, highlightRuns, type FuzzyMatch } from "../lib/fuzzy";
import { Icon, type IconName } from "./icons";

export interface PaletteCommand {
  id: string;
  label: string;
  group: string;
  icon: IconName;
  /** Shown to the right of the label, e.g. the screen's one-line purpose or a shortcut. */
  hint?: string;
  keywords?: string;
  /** Marks the option that is already in effect (the current theme): shown as a small badge. */
  current?: boolean;
  run: () => void;
}

interface Row {
  command: PaletteCommand;
  match: FuzzyMatch;
}

/**
 * Ctrl+K command palette: an Ark UI dialog (focus trap, Escape, focus restore) holding an ARIA
 * combobox. Typing filters with the fuzzy matcher; Up/Down/Home/End move, Enter runs, and matched
 * characters are highlighted as text spans.
 */
export function CommandPalette(props: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  commands: readonly PaletteCommand[];
  returnFocus: () => HTMLElement | null;
}): JSX.Element {
  const [query, setQuery] = createSignal("");
  const [active, setActive] = createSignal(0);
  const listId = createUniqueId();
  let input: HTMLInputElement | undefined;
  let list: HTMLDivElement | undefined;

  const rows = createMemo<Row[]>(() =>
    fuzzyFilter(query(), props.commands, (c) => [c.label, c.group, c.keywords ?? "", c.hint ?? ""]).map(({ item, match }) => ({
      command: item,
      match,
    })),
  );

  // Empty query: grouped in the authored order. With a query: one ranked list.
  const sections = createMemo(() => {
    const all = rows();
    if (query().trim() !== "") return [{ title: "Best matches", rows: all }];
    const order: string[] = [];
    for (const r of all) if (!order.includes(r.command.group)) order.push(r.command.group);
    return order.map((title) => ({ title, rows: all.filter((r) => r.command.group === title) }));
  });
  const flat = createMemo(() => sections().flatMap((s) => s.rows));
  const optionId = (cmd: PaletteCommand) => `${listId}-${cmd.id}`;

  createEffect(on(query, () => setActive(0)));
  createEffect(
    on(
      () => props.open,
      (open) => {
        if (open) {
          setQuery("");
          setActive(0);
        }
      },
    ),
  );
  createEffect(() => {
    const row = flat()[active()];
    if (!row || !list) return;
    list.querySelector<HTMLElement>(`[id="${optionId(row.command)}"]`)?.scrollIntoView({ block: "nearest" });
  });

  const runRow = (row: Row | undefined) => {
    if (!row) return;
    props.onOpenChange(false);
    // After the dialog has released focus, so the command can move it (navigation focuses <main>).
    queueMicrotask(() => row.command.run());
  };

  const onKeyDown = (e: KeyboardEvent) => {
    const count = flat().length;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (count === 0) return;
      const step = e.key === "ArrowDown" ? 1 : -1;
      setActive((i) => (i + step + count) % count);
    } else if (e.key === "Home" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      setActive(0);
    } else if (e.key === "End" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      setActive(Math.max(0, count - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      runRow(flat()[active()]);
    }
  };

  const indexOf = (row: Row) => flat().indexOf(row);

  return (
    <Dialog.Root
      open={props.open}
      onOpenChange={(d) => props.onOpenChange(d.open)}
      initialFocusEl={() => input ?? null}
      finalFocusEl={props.returnFocus}
      preventScroll={false}
      lazyMount
      unmountOnExit
    >
      <Portal>
        <Dialog.Backdrop class="palette-backdrop" />
        <Dialog.Positioner class="palette-positioner">
          <Dialog.Content class="palette">
            <Dialog.Title class="sr-only">Command palette</Dialog.Title>
            <Dialog.Description class="sr-only">Type to filter. Use the arrow keys to choose and Enter to run.</Dialog.Description>
            <div class="palette__search">
              <Icon name="search" size={18} class="palette__search-icon" />
              <input
                ref={input}
                class="palette__input"
                type="text"
                role="combobox"
                aria-label="Search screens and commands"
                aria-expanded="true"
                aria-controls={listId}
                aria-autocomplete="list"
                aria-activedescendant={flat()[active()] ? optionId(flat()[active()]!.command) : undefined}
                placeholder="Search screens and commands"
                autocomplete="off"
                spellcheck={false}
                maxLength={120}
                value={query()}
                onInput={(e) => setQuery(e.currentTarget.value)}
                onKeyDown={onKeyDown}
              />
              <kbd class="kbd">Esc</kbd>
            </div>
            <div class="palette__list" id={listId} role="listbox" aria-label="Results" ref={list}>
              <For each={sections()}>
                {(section) => (
                  <div role="group" aria-label={section.title} class="palette__group">
                    <p class="palette__group-title" aria-hidden="true">
                      {section.title}
                    </p>
                    <For each={section.rows}>
                      {(row) => (
                        <div
                          id={optionId(row.command)}
                          role="option"
                          class="palette__option"
                          aria-selected={indexOf(row) === active()}
                          data-command={row.command.id}
                          onPointerMove={() => setActive(indexOf(row))}
                          onClick={() => runRow(row)}
                        >
                          <Icon name={row.command.icon} size={18} class="palette__option-icon" />
                          <span class="palette__option-label">
                            <For each={highlightRuns(row.command.label, row.match.indices)}>
                              {(run) => (run.hit ? <mark class="palette__hit">{run.text}</mark> : run.text)}
                            </For>
                          </span>
                          <Show when={row.command.hint}>
                            <span class="palette__option-hint">{row.command.hint}</span>
                          </Show>
                          <Show when={row.command.current}>
                            <span class="palette__current">
                              <Icon name="check" size={12} />
                              Current
                            </span>
                          </Show>
                        </div>
                      )}
                    </For>
                  </div>
                )}
              </For>
            </div>
            <p class="palette__empty" role="status">
              {flat().length === 0 ? `No screens or commands match “${query().trim()}”.` : ""}
            </p>
            <div class="palette__foot" aria-hidden="true">
              <span>
                <kbd class="kbd">↑</kbd>
                <kbd class="kbd">↓</kbd> to move
              </span>
              <span>
                <kbd class="kbd">Enter</kbd> to open
              </span>
              <span>
                <kbd class="kbd">Esc</kbd> to close
              </span>
            </div>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
}
