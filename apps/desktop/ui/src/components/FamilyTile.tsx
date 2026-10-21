import type { JSX } from "solid-js";
import { familyOf } from "../lib/hub-picks";

/**
 * A model family's tile: a two-letter mark on the family's colour, drawn here (no logos and no
 * images fetched from anywhere). Unknown families get a neutral tile from the name.
 */
export function FamilyTile(props: { name: string; size?: number }): JSX.Element {
  const f = () => familyOf(props.name);
  const s = () => props.size ?? 40;
  return (
    <span
      class="family-tile"
      aria-hidden="true"
      data-family={f().id}
      style={{
        width: `${s()}px`,
        height: `${s()}px`,
        "font-size": `${Math.round(s() * 0.38)}px`,
        ...(f().hue !== null ? { "--tile-hue": String(f().hue) } : {}),
      }}
    >
      {f().mark}
    </span>
  );
}
