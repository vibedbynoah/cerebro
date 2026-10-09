import { createSignal, For, type JSX } from "solid-js";
import { Section } from "../../components/Screen";
import { SettingSwitch } from "../../components/SettingSwitch";
import {
  DEFAULT_BASE_PROMPT,
  EFFORT_LABELS,
  EFFORT_LEVELS,
  EFFORTS,
  effectiveLevels,
  loadLevelEdits,
  loadPrefs,
  REASONING_LABELS,
  REASONING_NOTES,
  REASONING_SOURCES,
  saveLevelEdits,
  savePrefs,
  VERBOSITIES,
  VERBOSITY_INSTRUCTIONS,
  VERBOSITY_LABELS,
  type ChatPrefs,
  type Effort,
  type LevelEdits,
} from "../../lib/chat/prompt";
import { notify } from "../../lib/toast";

/**
 * Settings for replies (docs/specs/effort-levels.md): default verbosity, default effort and
 * reasoning source, every level's instruction, thinking budget and passes, and the system prompt,
 * including sending no added prompt at all. Chat reads the same stored values.
 */
export function ChatSettings(): JSX.Element {
  const [prefs, setPrefsS] = createSignal<ChatPrefs>(loadPrefs());
  const [edits, setEditsS] = createSignal<LevelEdits>(loadLevelEdits());
  const setPrefs = (p: Partial<ChatPrefs>) => {
    const next = { ...prefs(), ...p };
    setPrefsS(next);
    savePrefs(next);
  };
  const setEdit = (e: Effort, patch: NonNullable<LevelEdits[Effort]>) => {
    const next = { ...edits(), [e]: { ...(edits()[e] ?? {}), ...patch } };
    setEditsS(next);
    saveLevelEdits(next);
  };
  const resetLevel = (e: Effort) => {
    const next = { ...edits() };
    delete next[e];
    setEditsS(next);
    saveLevelEdits(next);
    notify(`${EFFORT_LABELS[e]} reset`, "Back to the built-in instruction, budget and passes.");
  };
  const levels = () => effectiveLevels(edits());

  return (
    <>
      <Section id="responses" title="Responses" description="How long replies are by default. Chat can change it per message.">
        <fieldset class="segmented" aria-describedby="verbosity-note">
          <legend class="sr-only">Default verbosity</legend>
          <For each={VERBOSITIES}>
            {(v) => (
              <label class="segmented__option" data-press>
                <input type="radio" name="verbosity-setting" value={v} checked={prefs().verbosity === v} onChange={() => setPrefs({ verbosity: v })} />
                <span>{VERBOSITY_LABELS[v]}</span>
              </label>
            )}
          </For>
        </fieldset>
        <p class="hint" id="verbosity-note">
          Adds: “{VERBOSITY_INSTRUCTIONS[prefs().verbosity]}”
        </p>
      </Section>

      <Section id="effort" title="Effort and reasoning" description="The default for new messages, and what each effort level does. Edit any level; Reset restores it.">
        <div class="settings-row">
          <label class="settings-field">
            <span>Default effort</span>
            <select value={prefs().effort} onChange={(e) => setPrefs({ effort: e.currentTarget.value as Effort })}>
              <For each={EFFORTS}>{(e) => <option value={e}>{EFFORT_LABELS[e]}</option>}</For>
            </select>
          </label>
          <label class="settings-field">
            <span>Reasoning source</span>
            <select value={prefs().source} onChange={(e) => setPrefs({ source: e.currentTarget.value as ChatPrefs["source"] })} aria-describedby="source-setting-note">
              <For each={REASONING_SOURCES}>{(s) => <option value={s}>{REASONING_LABELS[s]}</option>}</For>
            </select>
          </label>
        </div>
        <p class="hint" id="source-setting-note">
          {REASONING_NOTES[prefs().source]}
        </p>
        <div class="effort-levels">
          <For each={EFFORTS}>
            {(e) => (
              <details class="card effort-level" data-level={e}>
                <summary>
                  <span class="effort-level__name">{EFFORT_LABELS[e]}</span>
                  <span class="hint">
                    {levels()[e].thinkingTokens === 0 ? "no thinking" : Number.isFinite(levels()[e].thinkingTokens) ? `${levels()[e].thinkingTokens.toLocaleString()} thinking tokens` : "model maximum thinking"} ·{" "}
                    {levels()[e].passes} review pass{levels()[e].passes === 1 ? "" : "es"}
                    {edits()[e] ? " · edited" : ""}
                  </span>
                </summary>
                <label class="settings-field settings-field--wide">
                  <span>Instruction</span>
                  <textarea rows="3" maxlength="4000" value={levels()[e].instruction} onChange={(ev) => setEdit(e, { instruction: ev.currentTarget.value })} />
                </label>
                <div class="settings-row">
                  <label class="settings-field">
                    <span>Thinking budget (tokens)</span>
                    <input
                      type="number"
                      min="0"
                      max="262144"
                      step="256"
                      value={Number.isFinite(levels()[e].thinkingTokens) ? levels()[e].thinkingTokens : 262144}
                      onChange={(ev) => {
                        const n = Math.round(Number(ev.currentTarget.value));
                        if (Number.isFinite(n) && n >= 0 && n <= 262144) setEdit(e, { thinkingTokens: n });
                      }}
                    />
                  </label>
                  <label class="settings-field">
                    <span>Review passes</span>
                    <select value={String(levels()[e].passes)} onChange={(ev) => setEdit(e, { passes: Number(ev.currentTarget.value) as 0 | 1 | 2 })}>
                      <option value="0">None</option>
                      <option value="1">Self-review</option>
                      <option value="2">Self-review and independent check</option>
                    </select>
                  </label>
                  <button class="btn btn--tertiary" type="button" onClick={() => resetLevel(e)} disabled={!edits()[e]}>
                    Reset
                  </button>
                </div>
                <p class="hint">Built-in: {EFFORT_LEVELS[e].slowdown}.</p>
              </details>
            )}
          </For>
        </div>
      </Section>

      <Section id="system-prompt" title="System prompt" description="What every chat starts with. You can send it alone, with nothing added.">
        <label class="settings-field settings-field--wide">
          <span>Base system prompt</span>
          <textarea rows="3" maxlength="8000" value={prefs().base} onChange={(e) => setPrefs({ base: e.currentTarget.value })} />
        </label>
        <SettingSwitch
          id="add-instructions"
          label="Add Cerebro's instructions"
          note="Verbosity, effort and memory text are appended to your prompt. Off sends your prompt alone."
          checked={prefs().addInstructions}
          onChange={(v) => setPrefs({ addInstructions: v })}
        />
        <button class="btn btn--tertiary" type="button" onClick={() => setPrefs({ base: DEFAULT_BASE_PROMPT })} disabled={prefs().base === DEFAULT_BASE_PROMPT}>
          Reset prompt
        </button>
      </Section>
    </>
  );
}
