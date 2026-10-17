import { createSignal, Show, type JSX } from "solid-js";
import "../../styles/toolcard.css";
import { Icon } from "../../components/icons";

import type { ToolActivity } from "../../service/chat-types";

export type ToolDecision = "allow-once" | "always" | "deny";
type Decision = ToolDecision;

const TRUNCATE_AT = 2000;

function chipLabel(state: ToolActivity["state"]): string {
  switch (state) {
    case "approval": return "Waiting for you";
    case "running": return "Running";
    case "done": return "Done";
    case "denied": return "Denied";
    case "error": return "Failed";
  }
}

export function ToolCard(props: {
  tool: ToolActivity;
  onDecide?: ((id: string, decision: Decision) => void) | undefined;
}): JSX.Element {
  const tool = () => props.tool;
  const [showAll, setShowAll] = createSignal(false);

  const argsText = () => JSON.stringify(tool().arguments ?? {}, null, 2);
  const resultText = () => tool().result ?? "";
  const isLong = () => resultText().length > TRUNCATE_AT;
  const shownResult = () => (showAll() || !isLong() ? resultText() : resultText().slice(0, TRUNCATE_AT));
  const chipId = () => `toolcard-${tool().id}-state`;
  const descId = () => `toolcard-${tool().id}-desc`;

  const decide = (d: Decision) => () => props.onDecide?.(tool().id, d);

  return (
    <div class="toolcard" data-state={tool().state}>
      <div class="toolcard__head">
        <Icon name="tools" size={16} />
        <span class="toolcard__name mono">{tool().name}</span>
        <span class="toolcard__server">from {tool().server}</span>
        <span
          class={`toolcard__chip toolcard__chip--${tool().state}`}
          id={chipId()}
          aria-live="polite"
        >
          <Show when={tool().state === "running"}>
            <span class="toolcard__spinner" aria-hidden="true" />
          </Show>
          {chipLabel(tool().state)}
        </span>
      </div>

      <details class="toolcard__fold">
        <summary>Arguments</summary>
        <pre class="toolcard__pre">{argsText()}</pre>
      </details>

      <Show when={tool().state === "approval"}>
        <div
          class="toolcard__approval"
          role="group"
          aria-label={`Approve ${tool().name}`}
          aria-describedby={descId()}
        >
          <p id={descId()} class="toolcard__prompt">The model wants to run this tool.</p>
          <div class="toolcard__actions">
            <button class="btn btn--primary btn--sm" type="button" onClick={decide("allow-once")}>Allow once</button>
            <button class="btn btn--secondary btn--sm" type="button" onClick={decide("always")}>Always allow</button>
            <button class="btn btn--tertiary btn--sm" type="button" onClick={decide("deny")}>Deny</button>
          </div>
        </div>
      </Show>

      <Show when={tool().result}>
        <details class="toolcard__fold">
          <summary>Result</summary>
          <pre class="toolcard__pre">{shownResult()}</pre>
          <Show when={isLong()}>
            <button
              class="btn btn--tertiary btn--sm toolcard__showall"
              type="button"
              onClick={() => setShowAll(!showAll())}
              aria-expanded={showAll()}
            >
              {showAll() ? "Show less" : "Show all"}
            </button>
          </Show>
        </details>
      </Show>

      <Show when={tool().error}>
        <p
          class="toolcard__error"
          role={tool().state === "error" ? "alert" : undefined}
        >
          {tool().error}
        </p>
      </Show>
    </div>
  );
}
