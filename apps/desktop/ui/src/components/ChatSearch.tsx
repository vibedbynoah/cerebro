import { Dialog } from "@ark-ui/solid/dialog";
import { createEffect, createMemo, createSignal, createUniqueId, For, on, Show, type JSX } from "solid-js";
import { Portal } from "solid-js/web";
import { highlightRuns, searchConversations, type ChatSearchResult } from "../lib/chat/search";
import { liveChats } from "../lib/chat/navigation";
import { loadConversations, type Conversation } from "../lib/chat/store";
import { Icon } from "./icons";
import "../styles/chat-search.css";

/** A run of text with every query hit wrapped in <mark>; input is plain text throughout. */
export function HighlightQuery(props: { text: string; query: string }): JSX.Element {
  return (
    <For each={highlightRuns(props.text, props.query)}>
      {(run) => (run.hit ? <mark class="chat-search__hit">{run.text}</mark> : run.text)}
    </For>
  );
}

/**
 * Search every chat, from anywhere (Ctrl+Shift+F). The dialog lists title and message matches with
 * highlighted snippets; choosing one opens the chat at that message. It uses the same search as the
 * box in the chat list (lib/chat/search.ts).
 */
export function ChatSearch(props: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onJump: (conversationId: string, messageId: string | null, query: string) => void;
}): JSX.Element {
  const [query, setQuery] = createSignal("");
  const [source, setSource] = createSignal<Conversation[]>([]);
  const [active, setActive] = createSignal(0);
  const listId = createUniqueId();
  let input: HTMLInputElement | undefined;
  let list: HTMLDivElement | undefined;

  createEffect(
    on(
      () => props.open,
      (open) => {
        if (!open) return;
        setQuery("");
        setActive(0);
        setSource(liveChats() ?? loadConversations());
      },
    ),
  );
  const results = createMemo(() => searchConversations(source(), query()));
  createEffect(on(query, () => setActive(0)));
  createEffect(() => {
    const row = results()[active()];
    if (!row || !list) return;
    list.querySelector<HTMLElement>(`[data-result="${active()}"]`)?.scrollIntoView({ block: "nearest" });
  });

  const optionId = (index: number) => `${listId}-${index}`;
  const openResult = (result: ChatSearchResult | undefined) => {
    if (!result) return;
    props.onOpenChange(false);
    queueMicrotask(() => props.onJump(result.conversationId, result.messageId, query().trim()));
  };
  const onKeyDown = (e: KeyboardEvent) => {
    const count = results().length;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!count) return;
      const step = e.key === "ArrowDown" ? 1 : -1;
      setActive((i) => (i + step + count) % count);
    } else if (e.key === "Enter") {
      e.preventDefault();
      openResult(results()[active()]);
    }
  };

  return (
    <Dialog.Root open={props.open} onOpenChange={(d) => props.onOpenChange(d.open)} initialFocusEl={() => input ?? null} preventScroll={false} lazyMount unmountOnExit>
      <Portal>
        <Dialog.Backdrop class="palette-backdrop" />
        <Dialog.Positioner class="palette-positioner">
          <Dialog.Content class="palette chat-search">
            <Dialog.Title class="sr-only">Search chats</Dialog.Title>
            <Dialog.Description class="sr-only">Type words to search every chat title and message. Use the arrow keys to choose a result and Enter to open it.</Dialog.Description>
            <div class="palette__search">
              <Icon name="search" size={18} class="palette__search-icon" />
              <input
                ref={input}
                class="palette__input"
                type="text"
                role="combobox"
                aria-label="Search chats"
                aria-expanded="true"
                aria-controls={listId}
                aria-autocomplete="list"
                aria-activedescendant={results()[active()] ? optionId(active()) : undefined}
                placeholder="Search chats and messages"
                autocomplete="off"
                spellcheck={false}
                maxLength={200}
                value={query()}
                onInput={(e) => setQuery(e.currentTarget.value)}
                onKeyDown={onKeyDown}
              />
              <kbd class="kbd">Esc</kbd>
            </div>
            <p class="chat-search__summary" role="status">
              <Show when={query().trim()} fallback="Search every chat on this computer. Titles and messages are included.">
                {results().length} {results().length === 1 ? "result" : "results"}
              </Show>
            </p>
            <div class="palette__list chat-search__list" id={listId} role="listbox" aria-label="Chat results" ref={list}>
              <Show when={query().trim()} fallback={<p class="chat-search__hint">Type a word or two to find a chat.</p>}>
                <For each={results()}>
                  {(result, i) => (
                    <button
                      id={optionId(i())}
                      type="button"
                      role="option"
                      aria-selected={i() === active()}
                      class="chat-search__result"
                      data-result={i()}
                      onPointerMove={() => setActive(i())}
                      onClick={() => openResult(result)}
                    >
                      <span class="chat-search__head">
                        <span class="chat-search__title">
                          <HighlightQuery text={result.conversationTitle} query={query()} />
                        </span>
                        <span class="chat-search__where">{result.messageId === null ? "Chat title" : result.role === "user" ? "You" : "Reply"}</span>
                      </span>
                      <Show when={result.messageId !== null}>
                        <span class="chat-search__snippet">
                          <HighlightQuery text={result.snippet} query={query()} />
                        </span>
                      </Show>
                    </button>
                  )}
                </For>
                <Show when={results().length === 0}>
                  <p class="chat-search__hint" role="status">
                    No chats match “{query().trim()}”.
                  </p>
                </Show>
              </Show>
            </div>
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
