import { Show, type JSX } from "solid-js";
import { VERIFIED_TOOLTIP } from "../lib/hub-verified";
import { Icon } from "./icons";

/**
 * The "Verified" badge: an icon and the word, never colour on its own, with a tooltip saying
 * exactly what it means — the publisher is on Cerebro's list of known publishers, and that is all
 * it says. The publisher's name follows for screen readers.
 */
export function VerifiedBadge(props: { publisher?: string; class?: string }): JSX.Element {
  return (
    <span class={`verified ${props.class ?? ""}`.trim()} data-verified="true" title={VERIFIED_TOOLTIP}>
      <Icon name="verified" size={13} />
      <span class="verified__text">Verified</span>
      <Show when={props.publisher}>
        <span class="sr-only"> ({props.publisher})</span>
      </Show>
    </span>
  );
}
