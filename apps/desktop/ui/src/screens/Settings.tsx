import { createResource, createSignal, For, Show, type JSX } from "solid-js";
import { Icon, type IconName } from "../components/icons";
import { ScreenHeader, Section } from "../components/Screen";
import { SettingSwitch } from "../components/SettingSwitch";
import { LegalDialog } from "../legal/LegalDialog";
import { LEGAL_DOCS, type LegalDocId } from "../legal/types";
import type { Platform } from "../platform/platform";
import { SETTINGS_META, type SettingMeta } from "../lib/settings-meta";
import { THEME_LABELS, type ThemeChoice, type ThemeController } from "../lib/theme";
import { useAppState } from "../state";

const THEME_ICONS: Record<ThemeChoice, IconName> = { system: "monitor", dark: "moon", light: "sun" };

export function Settings(props: { theme: ThemeController; platform: Platform; onRunSetup: () => void }): JSX.Element {
  const { settings, setSetting } = useAppState();
  const [legal, setLegal] = createSignal<LegalDocId | null>(null);
  const [info] = createResource(() => props.platform.appInfo());

  const group = (g: SettingMeta["group"]) => (
    <div class="card settings-card">
      <For each={SETTINGS_META.filter((m) => m.group === g)}>
        {(meta) => (
          <SettingSwitch
            id={`setting-${meta.key}`}
            label={meta.label}
            note={meta.note}
            checked={settings()?.[meta.key] ?? false}
            disabled={settings() === null}
            onChange={(on) => void setSetting(meta.key, on, `${meta.label} ${on ? "on" : "off"}`)}
          />
        )}
      </For>
    </div>
  );

  return (
    <div class="screen">
      <ScreenHeader route="settings" />

      <Section id="appearance" title="Appearance">
        <fieldset class="segmented" aria-describedby="theme-note">
          <legend class="sr-only">Theme</legend>
          <For each={Object.keys(THEME_LABELS) as ThemeChoice[]}>
            {(choice) => (
              <label class="segmented__option" data-press>
                <input
                  type="radio"
                  name="theme"
                  value={choice}
                  checked={props.theme.choice() === choice}
                  onChange={() => props.theme.set(choice)}
                />
                <Icon name={THEME_ICONS[choice]} size={16} />
                <span>{THEME_LABELS[choice]}</span>
              </label>
            )}
          </For>
        </fieldset>
        <p class="hint" id="theme-note">
          Also in the command palette (Ctrl K). Match system follows Windows' light or dark setting.
        </p>
      </Section>

      <Section id="protections" title="Protections" description="On by default. You can turn any of them off; nothing else changes when you do.">
        {group("protections")}
      </Section>

      <Section id="network" title="Network" description="Every network feature is optional. Cerebro works fully offline.">
        {group("network")}
      </Section>

      <Section id="about" title="About">
        <dl class="about">
          <dt>Version</dt>
          <dd class="mono">{info()?.version ?? "…"}</dd>
          <dt>Running as</dt>
          <dd>
            <Show when={info()} fallback="…">
              {(i) => (i().mode === "desktop" ? "Desktop app" : "Browser preview with illustrative data")}
            </Show>
          </dd>
          <dt>License</dt>
          <dd>Apache-2.0</dd>
          <dt>Documents</dt>
          <dd class="about__docs">
            <For each={LEGAL_DOCS}>
              {(d) => (
                <button type="button" class="text-link about__doc" aria-haspopup="dialog" onClick={() => setLegal(d.id)}>
                  {d.label}
                </button>
              )}
            </For>
          </dd>
        </dl>
      </Section>

      <Section id="setup" title="First-run setup" description="Walk through the hardware check, a first model, the protections and the terms again. Nothing changes until you choose.">
        <button type="button" class="btn btn--secondary" data-press onClick={() => props.onRunSetup()}>
          Run setup again
        </button>
      </Section>
      <LegalDialog doc={legal()} onSelect={setLegal} onClose={() => setLegal(null)} />
    </div>
  );
}
