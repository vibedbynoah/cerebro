import { Switch } from "@ark-ui/solid/switch";
import type { JSX } from "solid-js";

/**
 * One user-controlled switch: a label, and one short, neutral note on what it does. Never blocks,
 * warns or asks for confirmation (owner rule: the user may turn anything off).
 */
export function SettingSwitch(props: {
  id: string;
  label: string;
  note: JSX.Element;
  checked: boolean;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}): JSX.Element {
  return (
    <div class="setting">
      <Switch.Root
        class="switch"
        checked={props.checked}
        disabled={props.disabled ?? false}
        onCheckedChange={(d) => props.onChange(d.checked)}
        data-setting={props.id}
      >
        <span class="setting__text">
          <Switch.Label class="setting__label">{props.label}</Switch.Label>
          <span class="setting__note" id={`${props.id}-note`}>
            {props.note}
          </span>
        </span>
        <Switch.Control class="switch__control" data-press>
          <Switch.Thumb class="switch__thumb" />
        </Switch.Control>
        <Switch.HiddenInput role="switch" aria-describedby={`${props.id}-note`} />
      </Switch.Root>
    </div>
  );
}
