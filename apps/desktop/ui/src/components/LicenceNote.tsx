import { Show, type JSX } from "solid-js";
import { commercialLabel, type ModelLicence } from "../lib/model-licences";

/**
 * One model's licence, with the plain answer to "may I use this commercially?".
 * `compact` keeps the licence name and badge for dense rows; the summary is for details and
 * before a download.
 */
export function LicenceNote(props: { licence: ModelLicence; compact?: boolean; class?: string }): JSX.Element {
  return (
    <span class={`licence-note${props.compact ? " licence-note--compact" : ""}${props.class ? ` ${props.class}` : ""}`} data-licence={props.licence.name} data-commercial={props.licence.commercial}>
      <span class="licence-note__name">{props.licence.name}</span>
      <span class="licence-note__use">{commercialLabel(props.licence.commercial)}</span>
      <Show when={!props.compact}>
        <span class="licence-note__summary">{props.licence.summary}</span>
      </Show>
    </span>
  );
}
