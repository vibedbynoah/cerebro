import { createSignal, onMount, Show, type JSX } from "solid-js";
import { familyOf } from "../lib/hub-picks";
import { useService } from "../service/context";

/** Logos already asked for, by account: one request per publisher for the whole app. */
const logos = new Map<string, Promise<string | null>>();

/**
 * A model's tile: its publisher's logo from Hugging Face (Qwen's for Qwen, Google's for Gemma),
 * fetched once and kept on disk by the app. Until it arrives, or offline when it never has, a mark
 * on the family's colour stands in.
 */
export function FamilyTile(props: { name: string; owner?: string; size?: number }): JSX.Element {
  const f = () => familyOf(props.name);
  const s = () => props.size ?? 40;
  const service = useService();
  const [logo, setLogo] = createSignal<string | null>(null);
  onMount(() => {
    const owner = props.owner;
    if (!owner || !service.hubAvatar) return;
    const key = owner.toLowerCase();
    let p = logos.get(key);
    if (!p) {
      p = service.hubAvatar(owner).catch(() => null);
      logos.set(key, p);
    }
    void p.then(setLogo);
  });
  return (
    <span
      class="family-tile"
      classList={{ "family-tile--logo": logo() !== null }}
      aria-hidden="true"
      data-family={f().id}
      style={{
        width: `${s()}px`,
        height: `${s()}px`,
        "font-size": `${Math.round(s() * 0.38)}px`,
        ...(f().hue !== null ? { "--tile-hue": String(f().hue) } : {}),
      }}
    >
      <Show when={logo()} fallback={f().mark}>
        {(src) => <img src={src()} alt="" width={s()} height={s()} />}
      </Show>
    </span>
  );
}
