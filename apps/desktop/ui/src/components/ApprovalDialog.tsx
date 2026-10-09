import { Dialog } from "@ark-ui/solid/dialog";
import { For, Match, Show, Switch, type JSX } from "solid-js";
import { Portal } from "solid-js/web";
import { RISK_LABELS, TOOLS, type Decision, type ToolCall } from "../lib/permissions";
import { Icon } from "./icons";

/**
 * Asks before a tool acts, showing exactly what it will do: the command line, the file diff, the
 * URL or the plugin action. Keyboard: Y allow once, S allow for session, A always, N or Esc deny.
 * Everything shown is text; nothing from the call is rendered as markup.
 */
export function ApprovalDialog(props: { call: ToolCall | null; onDecide: (d: Decision) => void }): JSX.Element {
  const tool = () => TOOLS.find((t) => t.id === props.call?.toolId);
  const onKey = (e: KeyboardEvent) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const map: Record<string, Decision> = { y: "allow-once", s: "allow-session", a: "always", n: "deny" };
    const d = map[e.key.toLowerCase()];
    if (d) {
      e.preventDefault();
      props.onDecide(d);
    }
  };
  return (
    <Dialog.Root open={props.call !== null} onOpenChange={(d) => !d.open && props.onDecide("deny")} lazyMount unmountOnExit role="alertdialog">
      <Portal>
        <Dialog.Backdrop class="approve-backdrop" />
        <Dialog.Positioner class="approve-positioner">
          <Dialog.Content class="approve card" onKeyDown={onKey}>
            <Show when={props.call && tool()}>
              {(t) => (
                <>
                  <Dialog.Title class="approve__title">
                    Allow <span class="mono">{t().name}</span>?
                  </Dialog.Title>
                  <Dialog.Description class="approve__desc">
                    {t().source === "plugin" ? `From the ${t().plugin} plugin. ` : ""}
                    {RISK_LABELS[t().risk]}. {t().description}
                  </Dialog.Description>
                  <div class="approve__preview" data-kind={props.call!.preview.kind}>
                    <Switch>
                      <Match when={props.call!.preview.kind === "command" && props.call!.preview}>
                        {(p) => (
                          <>
                            <span class="approve__label">In {p().cwd}</span>
                            <pre class="mono">
                              <span aria-hidden="true">&gt; </span>
                              {p().command}
                            </pre>
                          </>
                        )}
                      </Match>
                      <Match when={props.call!.preview.kind === "diff" && props.call!.preview}>
                        {(p) => (
                          <>
                            <span class="approve__label">{p().path}</span>
                            <pre class="mono approve__diff">
                              <For each={p().lines}>
                                {(l) => (
                                  <span class="diff-line" data-t={l.t === "+" ? "add" : l.t === "-" ? "del" : "ctx"}>
                                    {l.t} {l.v}
                                    {"\n"}
                                  </span>
                                )}
                              </For>
                            </pre>
                          </>
                        )}
                      </Match>
                      <Match when={props.call!.preview.kind === "url" && props.call!.preview}>
                        {(p) => (
                          <pre class="mono">
                            {p().method} {p().url}
                          </pre>
                        )}
                      </Match>
                      <Match when={props.call!.preview.kind === "plugin" && props.call!.preview}>
                        {(p) => (
                          <>
                            <span class="approve__label">
                              {p().plugin}: {p().action}
                            </span>
                            <p>{p().details}</p>
                          </>
                        )}
                      </Match>
                    </Switch>
                  </div>
                  <div class="approve__actions">
                    <button class="btn btn--primary" type="button" onClick={() => props.onDecide("allow-once")} data-press>
                      Allow once <kbd>Y</kbd>
                    </button>
                    <button class="btn btn--secondary" type="button" onClick={() => props.onDecide("allow-session")} data-press>
                      For this session <kbd>S</kbd>
                    </button>
                    <button class="btn btn--secondary" type="button" onClick={() => props.onDecide("always")} data-press>
                      Always <kbd>A</kbd>
                    </button>
                    <button class="btn btn--tertiary" type="button" onClick={() => props.onDecide("deny")} data-press>
                      <Icon name="close" size={14} /> Deny <kbd>N</kbd>
                    </button>
                  </div>
                </>
              )}
            </Show>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
}
