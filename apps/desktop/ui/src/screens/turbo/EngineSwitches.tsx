import { createEffect, createResource, createSignal, For, on, Show, type JSX } from "solid-js";
import { Section } from "../../components/Screen";
import { SettingSwitch } from "../../components/SettingSwitch";
import { CHANNEL_LABELS } from "../../lib/experience";
import { notify } from "../../lib/toast";
import { useAppState } from "../../state";
import { useService } from "../../service/context";
import type { EngineSwitch } from "../../service/types";

/** Engine switches: the engine's optimisation registry, each with an explanation for the current level. */
export function EngineSwitches(props: { version?: number }): JSX.Element {
  const service = useService();
  const app = useAppState();

  const [switches, setSwitches] = createSignal<EngineSwitch[]>([]);
  const [list, { refetch }] = createResource(() => service.listSwitches());
  createEffect(() => {
    const l = list();
    if (l) setSwitches(l);
  });
  // A finished tune for this machine changed the engine's switches; the parent bumps `version` so
  // the list is read again and shows the kept values.
  createEffect(
    on(
      () => props.version,
      () => {
        void refetch();
      },
      { defer: true },
    ),
  );
  const error = (): string | null => (list.error ? (list.error instanceof Error ? list.error.message : String(list.error)) : null);

  const update = async (s: EngineSwitch, value: number) => {
    const previous = s.value;
    setSwitches((prev) => prev.map((sw) => (sw.id === s.id ? { ...sw, value } : sw)));
    try {
      const stored = await service.setSwitch(s.id, value);
      setSwitches((prev) => prev.map((sw) => (sw.id === s.id ? { ...sw, value: stored } : sw)));
    } catch (e) {
      setSwitches((prev) => prev.map((sw) => (sw.id === s.id ? { ...sw, value: previous } : sw)));
      notify(`Could not change ${s.name}: ${e instanceof Error ? e.message : "Unknown error"}`);
    }
  };

  const resetAll = async () => {
    try {
      const list = await service.resetSwitches();
      setSwitches(list);
    } catch (e) {
      notify(`Could not restore defaults: ${e instanceof Error ? e.message : "Unknown error"}`);
    }
  };

  const groups = () => {
    const seen = new Set<string>();
    const groups: string[] = [];
    switches().forEach((s) => {
      if (!seen.has(s.group)) {
        seen.add(s.group);
        groups.push(s.group);
      }
    });
    return groups;
  };

  const renderNote = (s: EngineSwitch) => {
    const level = app.level();
    const text = level === "beginner" ? s.beginner : s.advanced;
    return (
      <>
        {text}
        <span class={`badge badge--${s.channel}`}>{CHANNEL_LABELS[s.channel]}</span>
        <Show when={s.applies !== "live"}>
          <span class="badge">
            Applies {s.applies === "reload" ? "when a model loads" : "after restart"}
          </span>
        </Show>
        <Show when={level === "advanced"}>
          <code class="switches__env">{s.env}</code>
          <Show when={s.value !== s.default}>
            <button class="btn btn--tertiary btn--sm" type="button" onClick={() => update(s, s.default)}>
              Default
            </button>
          </Show>
        </Show>
      </>
    );
  };

  const renderNumberInput = (s: EngineSwitch) => {
    const kind = s.kind as { type: "number"; min: number; max: number; zero_auto: boolean; unit: string };
    const displayValue = kind.zero_auto && s.value === 0 ? "" : s.value;

    return (
      <div class="setting">
        <label for={`sw-${s.id}`} class="setting__text">
          <span class="setting__label">{s.name}</span>
          <span class="setting__note" id={`sw-${s.id}-note`}>{renderNote(s)}</span>
        </label>
        <input
          type="number"
          id={`sw-${s.id}`}
          class="input switches__number"
          min={kind.min}
          max={kind.max}
          value={displayValue}
          placeholder={kind.zero_auto ? "Auto" : undefined}
          onChange={(e) => {
            const parsed = parseInt(e.currentTarget.value, 10);
            const val = isNaN(parsed) ? 0 : parsed;
            update(s, val);
          }}
        />
        <span class="switches__unit">{kind.unit}</span>
      </div>
    );
  };

  return (
    <Section
      id="engine-switches"
      title="Engine switches"
      description="Every optimisation the engine uses, explained. Changes are saved on this machine."
    >
      <Show when={error()}>
        <p class="hint">Engine switches are unavailable: {error()}</p>
      </Show>
      <Show when={!error()}>
        <div data-testid="engine-switches">
          <For each={groups()}>
            {(group) => (
              <>
                <h3 class="switches__group">{group}</h3>
                <For each={switches().filter((s) => s.group === group)}>
                  {(s) => {
                    if (s.kind.type === "toggle") {
                      return (
                        <SettingSwitch
                          id={`sw-${s.id}`}
                          label={s.name}
                          note={renderNote(s)}
                          checked={s.value !== 0}
                          onChange={(c) => update(s, c ? 1 : 0)}
                        />
                      );
                    }
                    return renderNumberInput(s);
                  }}
                </For>
              </>
            )}
          </For>
          <div class="switches__footer">
            <button class="btn btn--tertiary" type="button" onClick={resetAll}>
              Restore all defaults
            </button>
          </div>
        </div>
      </Show>
    </Section>
  );
}