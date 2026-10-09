import { createResource, createSignal, For, Show, type JSX } from "solid-js";
import { Icon, type IconName } from "../components/icons";
import { ScreenHeader, Section } from "../components/Screen";
import { SettingSwitch } from "../components/SettingSwitch";
import { LegalDialog } from "../legal/LegalDialog";
import { LEGAL_DOCS, type LegalDocId } from "../legal/types";
import type { Platform } from "../platform/platform";
import { hrefFor } from "../lib/routes";
import { SETTINGS_META, type SettingMeta } from "../lib/settings-meta";
import { THEME_LABELS, type ThemeChoice, type ThemeController } from "../lib/theme";
import { useAppState } from "../state";
import { CHANNEL_LABELS, CHANNEL_NOTES, CHANNELS, channelIncludes, LEVEL_LABELS, OPTIMISATIONS, type ExperienceLevel } from "../lib/experience";
import { notify } from "../lib/toast";

const THEME_ICONS: Record<ThemeChoice, IconName> = { system: "monitor", dark: "moon", light: "sun" };

const LEVEL_ICONS: Record<ExperienceLevel, IconName> = { beginner: "bolt", advanced: "sliders" };
const LEVEL_NOTES: Record<ExperienceLevel, string> = {
  beginner: "Cerebro sets the model file, context, offload, KV cache and sampler from the hardware check, hides raw backend flags and explains each term where it appears.",
  advanced: "Raw backend flags, file choice, every load parameter and the optimisation channels. Starts from the same automatic values, so switching loses nothing.",
};

export function Settings(props: { theme: ThemeController; platform: Platform; onRunSetup: () => void }): JSX.Element {
  const { settings, setSetting, level, setLevel, channel, setChannel } = useAppState();
  const pickLevel = (l: ExperienceLevel) => {
    if (l === level()) return;
    setLevel(l);
    notify(`${LEVEL_LABELS[l]} view`, "Every value you set is kept.", "success");
  };
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

      <Section id="experience" title="Experience level" description="How much of the engine you see. The speed and the protections are the same at both levels.">
        <fieldset class="segmented" aria-describedby="level-note">
          <legend class="sr-only">Experience level</legend>
          <For each={["beginner", "advanced"] as const}>
            {(l) => (
              <label class="segmented__option" data-press>
                <input type="radio" name="experience-level-setting" value={l} checked={level() === l} onChange={() => pickLevel(l)} />
                <Icon name={LEVEL_ICONS[l]} size={16} />
                <span>{LEVEL_LABELS[l]}</span>
              </label>
            )}
          </For>
        </fieldset>
        <p class="hint" id="level-note">
          {LEVEL_NOTES[level()]}
        </p>
        <div class="card settings-card channel-card" data-level={level()}>
          <div class="channel-card__head">
            <h3 class="channel-card__title" id="channel-title">
              Optimisation channel
            </h3>
            <Show
              when={level() === "advanced"}
              fallback={
                <p class="hint channel-card__locked">
                  <Icon name="lock" size={14} />
                  <span>{CHANNEL_LABELS[channel()]}. Switch to Advanced to change it.</span>
                </p>
              }
            >
              <fieldset class="segmented segmented--sm" aria-labelledby="channel-title" aria-describedby="channel-note">
                <For each={CHANNELS}>
                  {(c) => (
                    <label class="segmented__option" data-press>
                      <input
                        type="radio"
                        name="channel-setting"
                        value={c}
                        checked={channel() === c}
                        onChange={() => {
                          setChannel(c);
                          notify(`${CHANNEL_LABELS[c]} channel`, CHANNEL_NOTES[c], "success");
                        }}
                      />
                      <span>{CHANNEL_LABELS[c]}</span>
                    </label>
                  )}
                </For>
              </fieldset>
            </Show>
          </div>
          <p class="hint" id="channel-note">
            {CHANNEL_NOTES[channel()]}
          </p>
          <ul class="opt-list" aria-label={`Optimisations on the ${CHANNEL_LABELS[channel()]} channel`}>
            <For each={OPTIMISATIONS}>
              {(o) => {
                const on = () => channelIncludes(channel(), o.channel);
                return (
                  <li class="opt-list__item" data-on={on()} data-opt={o.id}>
                    <span class="opt-list__mark" aria-hidden="true">
                      <Show when={on()}>
                        <Icon name="check" size={12} />
                      </Show>
                    </span>
                    <span class="opt-list__name">{o.label}</span>
                    <span class="opt-list__channel">{CHANNEL_LABELS[o.channel]}</span>
                    <span class="opt-list__note">{o.note}</span>
                    <span class="sr-only">{on() ? "On." : `Off; needs the ${CHANNEL_LABELS[o.channel]} channel.`}</span>
                  </li>
                );
              }}
            </For>
          </ul>
        </div>
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
        <a class="text-link about__more" href={hrefFor("about")}>
          More about Cerebro
          <Icon name="arrowRight" size={14} />
        </a>
      </Section>

      <Section id="setup" title="First-run setup" description="Walk through the experience level, the hardware check, a first model and its load settings, the protections and the terms again. Nothing changes until you choose.">
        <button type="button" class="btn btn--secondary" data-press onClick={() => props.onRunSetup()}>
          Run setup again
        </button>
      </Section>
      <LegalDialog doc={legal()} onSelect={setLegal} onClose={() => setLegal(null)} />
    </div>
  );
}
