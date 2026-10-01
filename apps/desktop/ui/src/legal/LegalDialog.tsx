import "../styles/legal.css";
import { Dialog } from "@ark-ui/solid/dialog";
import { createEffect, createResource, createSignal, createUniqueId, For, Match, on, Show, Switch, type JSX } from "solid-js";
import { Portal } from "solid-js/web";
import { Icon } from "../components/icons";
import { LEGAL_DOCS, type LegalBlock, type LegalDocId } from "./types";

/**
 * The full Terms, Privacy and Licenses, opened in place over whatever is on screen (setup or
 * Settings) so nothing is lost by reading them. The text is bundled (content.ts, loaded the first
 * time the dialog opens) and rendered as text nodes only. Works fully offline.
 */
export function LegalDialog(props: { doc: LegalDocId | null; onSelect: (doc: LegalDocId) => void; onClose: () => void }): JSX.Element {
  const id = createUniqueId();
  // Fetched once, on first open; later opens reuse the loaded module.
  const [docs] = createResource(
    () => props.doc !== null || undefined,
    () => import("./content").then((m) => m.DOCS),
  );
  // Guarded read: an errored resource throws when read (solid-js); its state never does.
  const current = () => (props.doc !== null && docs.state === "ready" ? docs()?.[props.doc] : undefined);
  const tabs: HTMLButtonElement[] = [];
  // Each document opens at its top, wherever the previous one was scrolled to.
  let body: HTMLDivElement | undefined;
  createEffect(on(() => props.doc, () => body?.scrollTo({ top: 0 }), { defer: true }));

  const onTabKey = (e: KeyboardEvent, index: number) => {
    const last = LEGAL_DOCS.length - 1;
    const to = e.key === "ArrowRight" ? (index === last ? 0 : index + 1) : e.key === "ArrowLeft" ? (index === 0 ? last : index - 1) : e.key === "Home" ? 0 : e.key === "End" ? last : -1;
    if (to < 0) return;
    e.preventDefault();
    const next = LEGAL_DOCS[to];
    if (!next) return;
    props.onSelect(next.id);
    tabs[to]?.focus();
  };

  return (
    <Dialog.Root open={props.doc !== null} onOpenChange={(d) => !d.open && props.onClose()} lazyMount unmountOnExit preventScroll={false}>
      <Portal>
        <Dialog.Backdrop class="legal-backdrop" />
        <Dialog.Positioner class="legal-positioner">
          <Dialog.Content class="legal">
            <header class="legal__head">
              <div class="legal__tabs" role="tablist" aria-label="Documents">
                <For each={LEGAL_DOCS}>
                  {(d, i) => (
                    <button
                      ref={(el) => (tabs[i()] = el)}
                      type="button"
                      role="tab"
                      class="legal__tab"
                      id={`${id}-tab-${d.id}`}
                      aria-selected={props.doc === d.id}
                      aria-controls={`${id}-panel`}
                      tabindex={props.doc === d.id ? 0 : -1}
                      data-press
                      onClick={() => props.onSelect(d.id)}
                      onKeyDown={(e) => onTabKey(e, i())}
                    >
                      {d.label}
                    </button>
                  )}
                </For>
              </div>
              <Dialog.CloseTrigger class="btn btn--icon legal__close" aria-label="Close" data-press>
                <Icon name="close" size={18} />
              </Dialog.CloseTrigger>
            </header>
            <div class="legal__body" ref={body} id={`${id}-panel`} role="tabpanel" aria-labelledby={props.doc ? `${id}-tab-${props.doc}` : undefined} tabindex="0">
              <article class="legal__doc">
                <Dialog.Title class="legal__title">{current()?.title ?? LEGAL_DOCS.find((d) => d.id === props.doc)?.label ?? ""}</Dialog.Title>
                <Dialog.Description class="legal__summary">
                  {current()?.summary ?? (docs.error ? "The document could not be opened. Close this and try again." : "Opening…")}
                </Dialog.Description>
                <Show when={current()}>
                  {(doc) => (
                    <For each={doc().sections}>
                      {(section) => (
                        <section class="legal__section">
                          <h3 class="legal__heading">{section.heading}</h3>
                          <For each={section.blocks}>{(block) => <Block block={block} />}</For>
                        </section>
                      )}
                    </For>
                  )}
                </Show>
              </article>
            </div>
            <p class="legal__foot">
              <Icon name="offline" size={14} />
              Bundled with the app: these pages open without a network.
            </p>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
}

function Block(props: { block: LegalBlock }): JSX.Element {
  const para = () => (props.block.kind === "p" ? props.block : undefined);
  const list = () => (props.block.kind === "list" ? props.block : undefined);
  const license = () => (props.block.kind === "license" ? props.block : undefined);
  const notices = () => (props.block.kind === "notices" ? props.block : undefined);
  return (
    <Switch>
      <Match when={notices()}>{(b) => <NoticeGroup block={b()} />}</Match>
      <Match when={para()}>{(b) => <p class="legal__p">{b().text}</p>}</Match>
      <Match when={list()}>
        {(b) => (
          <ul class="legal__list">
            <For each={b().items}>{(item) => <li>{item}</li>}</For>
          </ul>
        )}
      </Match>
      <Match when={license()}>
        {(b) => (
          <div class="legal__license">
            <div class="legal__license-row">
              <span class="legal__license-name">{b().name}</span>
              <span class="legal__license-id">{b().license}</span>
            </div>
            <p class="legal__license-detail">{b().detail}</p>
            <Show when={b().text}>
              {(text) => (
                <details class="legal__details">
                  <summary class="legal__summary-toggle">Full licence text</summary>
                  <pre class="legal__pre" tabindex="0">
                    {text()}
                  </pre>
                </details>
              )}
            </Show>
          </div>
        )}
      </Match>
    </Switch>
  );
}

type NoticesBlock = Extract<LegalBlock, { kind: "notices" }>;

/** Rebuilds one licence text from the paragraph table (scripts/third-party.mjs). */
function noticeText(n: CerebroNotices, i: number): string {
  return (n.docs[i] ?? []).map((p) => n.paragraphs[p] ?? "").join("\n\n");
}

/**
 * A generated list of third-party packages: one collapsed group, whose rows are only built when it
 * is first opened, and each row's licence texts only when that row is opened (hundreds of crates
 * would otherwise mean megabytes of text nodes on every visit to the Licenses page).
 */
function NoticeGroup(props: { block: NoticesBlock }): JSX.Element {
  const [opened, setOpened] = createSignal(false);
  const count = () => props.block.notices.packages.length;
  return (
    <div class="legal__license legal__notices" data-notices={props.block.id}>
      <div class="legal__license-row">
        <span class="legal__license-name">{props.block.title}</span>
        <span class="legal__license-id">{props.block.complete ? `${count()} packages` : "Generated by a build"}</span>
      </div>
      <p class="legal__license-detail">{props.block.detail}</p>
      <Show
        when={props.block.complete}
        fallback={<p class="legal__license-detail">This list is generated when the app is built from what the build contains. This development copy has none.</p>}
      >
        <details class="legal__details" on:toggle={(e) => e.currentTarget.open && setOpened(true)}>
          <summary class="legal__summary-toggle">Show all {count()}</summary>
          <Show when={opened()}>
            <ul class="legal__pkgs">
              <For each={props.block.notices.packages}>{(p) => <NoticeRow notices={props.block.notices} pkg={p} />}</For>
            </ul>
          </Show>
        </details>
      </Show>
    </div>
  );
}

function NoticeRow(props: { notices: CerebroNotices; pkg: CerebroNotices["packages"][number] }): JSX.Element {
  const [opened, setOpened] = createSignal(false);
  return (
    <li class="legal__pkg">
      <details class="legal__pkg-details" on:toggle={(e) => e.currentTarget.open && setOpened(true)}>
        <summary class="legal__pkg-row">
          <span class="legal__pkg-name">{props.pkg.name}</span>
          <span class="legal__pkg-version mono">{props.pkg.version}</span>
          <span class="legal__license-id">{props.pkg.license}</span>
        </summary>
        <Show when={opened()}>
          <Show when={props.pkg.note}>{(note) => <p class="legal__license-detail">{note()}</p>}</Show>
          <For each={props.pkg.texts}>
            {(t) => (
              <pre class="legal__pre" tabindex="0">
                {noticeText(props.notices, t)}
              </pre>
            )}
          </For>
        </Show>
      </details>
    </li>
  );
}
