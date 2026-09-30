import { For, Show, type JSX } from "solid-js";
import { Icon } from "../components/icons";
import { ScreenHeader, Section } from "../components/Screen";
import { Term } from "../components/Term";
import { hrefFor } from "../lib/routes";
import { SETTINGS_META } from "../lib/settings-meta";
import { useAppState } from "../state";

export function Security(): JSX.Element {
  const { settings } = useAppState();
  const protections = SETTINGS_META.filter((m) => m.group === "protections");
  const onCount = () => {
    const s = settings();
    return s ? protections.filter((p) => s[p.key]).length : 0;
  };

  return (
    <div class="screen">
      <ScreenHeader
        route="security"
        actions={
          <a class="btn btn--secondary" href={hrefFor("settings")} data-press>
            <Icon name="settings" size={16} />
            Change in Settings
          </a>
        }
      />
      <Section
        id="protections"
        title={settings() ? `${onCount()} of ${protections.length} protections on` : "Protections"}
        description="Each entry says what it does and, where it matters, what it does not do."
      >
        <ul class="protection-list">
          <For each={protections}>
            {(p) => {
              const on = () => settings()?.[p.key] ?? false;
              return (
                <li class="protection card" data-setting={p.key} data-on={on()}>
                  <span class="protection__icon" aria-hidden="true">
                    <Icon name={on() ? "security" : "lock"} size={20} />
                  </span>
                  <div class="protection__text">
                    <h3 class="protection__title">
                      <Show when={p.term} fallback={p.label}>
                        {(t) => <Term k={t()}>{p.label}</Term>}
                      </Show>
                    </h3>
                    <p class="protection__note">{p.note}</p>
                    <Show when={p.limits}>
                      <p class="protection__limits">
                        <span class="protection__limits-label">Limits:</span> {p.limits}
                      </p>
                    </Show>
                  </div>
                  <span class="chip" data-tone={on() ? "on" : "off"}>
                    {on() ? "On" : "Off"}
                  </span>
                </li>
              );
            }}
          </For>
        </ul>
      </Section>
    </div>
  );
}
