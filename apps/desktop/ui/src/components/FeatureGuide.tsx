import "../styles/guide.css";
import { Dialog } from "@ark-ui/solid/dialog";
import { For, Show, type JSX } from "solid-js";
import { Portal } from "solid-js/web";
import { FEATURE_GROUPS, GUIDE_SEEN_KEY, GUIDE_VERSION } from "../lib/features";
import type { RouteId } from "../lib/routes";
import { readPref, writePref } from "../lib/storage";
import { Icon } from "./icons";

/** Whether the guide should open on its own: never seen, or the feature list changed since. */
export function guideDue(): boolean {
  return readPref(GUIDE_SEEN_KEY) !== GUIDE_VERSION;
}

export function markGuideSeen(): void {
  writePref(GUIDE_SEEN_KEY, GUIDE_VERSION);
}

/**
 * "What Cerebro does": every feature in this build, grouped, each with a way to open its screen.
 * Opens once after setup (and again when the list changes); the command palette opens it any time.
 * Closing it in any way (X, Esc, a click outside, "Start using Cerebro") counts as seen.
 */
export function FeatureGuide(props: { open: boolean; onClose: () => void; onOpen: (r: RouteId) => void }): JSX.Element {
  const close = () => {
    markGuideSeen();
    props.onClose();
  };
  return (
    <Dialog.Root open={props.open} onOpenChange={(d) => !d.open && close()} lazyMount unmountOnExit>
      <Portal>
        <Dialog.Backdrop class="guide-backdrop" />
        <Dialog.Positioner class="guide-positioner">
          <Dialog.Content class="guide card" data-testid="feature-guide">
            <header class="guide__head">
              <div>
                <Dialog.Title class="guide__title">What Cerebro does</Dialog.Title>
                <Dialog.Description class="guide__desc">Everything in this version, in one place. Open any of it from here, or later from the command palette (Ctrl K, "What Cerebro does").</Dialog.Description>
              </div>
              <Dialog.CloseTrigger class="icon-btn" aria-label="Close">
                <Icon name="close" size={18} />
              </Dialog.CloseTrigger>
            </header>
            <div class="guide__body">
              <For each={FEATURE_GROUPS}>
                {(g) => (
                  <section class="guide__group" aria-labelledby={`guide-${g.title}`}>
                    <h3 id={`guide-${g.title}`}>{g.title}</h3>
                    <ul class="guide__list">
                      <For each={g.features}>
                        {(f) => (
                          <li class="guide__item">
                            <span class="guide__icon" aria-hidden="true">
                              <Icon name={f.icon} size={18} />
                            </span>
                            <div class="guide__text">
                              <strong>
                                {f.title}
                                <Show when={f.isNew}>
                                  <span class="guide__new">New</span>
                                </Show>
                              </strong>
                              <span>{f.text}</span>
                            </div>
                            <button
                              type="button"
                              class="btn btn--tertiary btn--sm"
                              aria-label={`Open ${f.title}`}
                              onClick={() => {
                                close();
                                props.onOpen(f.route);
                              }}
                            >
                              Open
                            </button>
                          </li>
                        )}
                      </For>
                    </ul>
                  </section>
                )}
              </For>
            </div>
            <footer class="guide__foot">
              <button type="button" class="btn btn--primary" onClick={close}>
                Start using Cerebro
              </button>
            </footer>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
}
