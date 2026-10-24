import { createMemo, For, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { MODE_PRESETS, MODES, type Mode } from "../../lib/chat/modes";
import { DASH, num } from "../../lib/monitor";
import type { Sampler } from "../../service/chat-types";
import { useMetrics } from "../../service/metrics";
import { Sparkline } from "../monitor/Sparkline";
import { ExpertHeatmap } from "./ExpertHeatmap";

/**
 * Chat's live dock: what the model and the machine are doing right now, and the settings people
 * change most, beside the conversation. Metrics come from the app-wide feed (service/metrics.tsx),
 * so the dock adds no polling of its own.
 */
export function LiveDock(props: {
  mode: Mode | null;
  onMode: (m: Mode) => void;
  sampler: Sampler;
  setSampler: (s: Sampler) => void;
  thinking: boolean;
  setThinking: (on: boolean) => void;
  onOpenPrompt: () => void;
  onUnload: () => void;
  /** This model has its own saved settings. */
  hasPreset: boolean;
  onSavePreset: () => void;
  onClearPreset: () => void;
}): JSX.Element {
  const feed = useMetrics();
  const s = () => feed.snapshot();
  const live = () => s()?.live;
  const hw = () => s()?.hardware;
  const model = () => s()?.engine.model ?? null;
  const decode = createMemo(() => s()?.history.series.decodeTps ?? []);
  const lastDecode = () => {
    const d = decode();
    for (let i = d.length - 1; i >= 0; i--) if (d[i] !== null && d[i] !== undefined) return d[i]!;
    return null;
  };
  const bar = (used: number | null | undefined, total: number | null | undefined) => (used !== null && used !== undefined && total ? Math.min(100, Math.max(0, (used / total) * 100)) : 0);
  const where = () => {
    const m = model();
    if (!m) return "";
    if (m.gpuLayers >= m.layers) return "All on the graphics card";
    if (m.gpuLayers === 0) return "On the processor";
    return `${m.gpuLayers} of ${m.layers} layers on the graphics card`;
  };

  return (
    <div class="dock" data-testid="live-dock">
      <section class="dock__section" aria-labelledby="dock-live">
        <h3 id="dock-live">Live</h3>
        <div class="dock__big">
          <span class="mono dock__value" data-testid="dock-tps">
            {num(live()?.decodeTps ?? lastDecode(), 1)}
          </span>
          <span class="dock__unit">tokens per second{live()?.decodeTps ? "" : " (last reply)"}</span>
        </div>
        <Show when={decode().length}>
          <Sparkline
            label="Writing speed, last 60 seconds"
            lines={[{ key: "decode", label: "Writing", values: decode(), tone: "accent", area: true, format: (v) => `${v.toFixed(1)} tok/s` }]}
            stepS={s()?.history.stepS ?? 1}
            end={s()?.history.end ?? Date.now()}
            rangeLabel="last 60 seconds"
          />
        </Show>
        <dl class="dock__stats">
          <dt>First word after</dt>
          <dd class="mono">{live()?.firstTokenS !== null && live()?.firstTokenS !== undefined ? `${live()!.firstTokenS!.toFixed(2)} s` : DASH}</dd>
          <dt>Reading speed</dt>
          <dd class="mono">{live()?.prefillTps ? `${num(live()!.prefillTps, 0)} tok/s` : DASH}</dd>
          <dt>Written this reply</dt>
          <dd class="mono">{live()?.generated ? num(live()!.generated) : DASH}</dd>
        </dl>
        <Show when={model()}>
          {(m) => (
            <div class="dock__meter">
              <span>
                Context <span class="mono">{num(m().contextUsed)} / {num(m().contextMax)}</span>
              </span>
              <div class="dock__bar" role="meter" aria-label="Context used" aria-valuemin={0} aria-valuemax={m().contextMax} aria-valuenow={m().contextUsed}>
                <span style={{ width: `${bar(m().contextUsed, m().contextMax)}%` }} />
              </div>
            </div>
          )}
        </Show>
      </section>

      <Show when={s()?.engine.moe}>
        <section class="dock__section" aria-labelledby="dock-experts">
          <h3 id="dock-experts">Experts firing</h3>
          <ExpertHeatmap />
        </section>
      </Show>

      <section class="dock__section" aria-labelledby="dock-machine">
        <h3 id="dock-machine">Machine</h3>
        <div class="dock__meter">
          <span>
            Graphics memory{" "}
            <span class="mono">
              {hw()?.vramUsedGiB !== null && hw()?.vramUsedGiB !== undefined ? `${hw()!.vramUsedGiB!.toFixed(1)} / ${hw()!.vramTotalGiB.toFixed(1)} GiB` : DASH}
            </span>
          </span>
          <div class="dock__bar" role="meter" aria-label="Graphics memory used" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(bar(hw()?.vramUsedGiB, hw()?.vramTotalGiB))}>
            <span style={{ width: `${bar(hw()?.vramUsedGiB, hw()?.vramTotalGiB)}%` }} />
          </div>
        </div>
        <dl class="dock__stats">
          <dt>Graphics card load</dt>
          <dd class="mono">{hw()?.gpuUtil !== null && hw()?.gpuUtil !== undefined ? `${num(hw()!.gpuUtil)}%` : DASH}</dd>
          <dt>Temperature</dt>
          <dd class="mono">{hw()?.gpuTempC ? `${num(hw()!.gpuTempC)} °C` : DASH}</dd>
          <dt>Power</dt>
          <dd class="mono">{hw()?.gpuPowerW ? `${num(hw()!.gpuPowerW)} W${hw()?.gpuPowerLimitW ? ` of ${num(hw()!.gpuPowerLimitW)}` : ""}` : DASH}</dd>
          <dt>Processor load</dt>
          <dd class="mono">{hw()?.cpu !== null && hw()?.cpu !== undefined ? `${num(hw()!.cpu)}%` : DASH}</dd>
          <dt>Memory</dt>
          <dd class="mono">{hw()?.ramUsedGiB ? `${hw()!.ramUsedGiB!.toFixed(1)} / ${hw()!.ramTotalGiB.toFixed(0)} GiB` : DASH}</dd>
        </dl>
      </section>

      <section class="dock__section" aria-labelledby="dock-settings">
        <h3 id="dock-settings">Quick settings</h3>
        <label class="dock__field">
          <span>
            Speed or accuracy <strong>{props.mode ? MODE_PRESETS[props.mode].label : "Custom"}</strong>
          </span>
          <input type="range" min="1" max="5" step="1" value={props.mode ?? 3} aria-valuetext={props.mode ? MODE_PRESETS[props.mode].label : "Custom"} onInput={(e) => props.onMode(Number(e.currentTarget.value) as Mode)} list="dock-mode-ticks" />
          <datalist id="dock-mode-ticks">
            <For each={MODES}>{(m) => <option value={m} />}</For>
          </datalist>
        </label>
        <Show when={props.mode}>{(m) => <p class="dock__note" data-testid="dock-mode-summary">{MODE_PRESETS[m()].summary}</p>}</Show>
        <label class="dock__field">
          <span>
            Temperature <strong class="mono">{props.sampler.temperature.toFixed(2)}</strong>
          </span>
          <input type="range" min="0" max="2" step="0.05" value={props.sampler.temperature} onInput={(e) => props.setSampler({ ...props.sampler, temperature: Number(e.currentTarget.value) })} />
        </label>
        <label class="dock__field">
          <span>
            Longest reply <strong class="mono">{props.sampler.maxTokens.toLocaleString()}</strong> tokens
          </span>
          <input type="range" min="256" max="16384" step="256" value={Math.min(16384, Math.max(256, props.sampler.maxTokens))} onInput={(e) => props.setSampler({ ...props.sampler, maxTokens: Number(e.currentTarget.value) })} />
        </label>
        <label class="dock__check">
          <input type="checkbox" checked={props.thinking} onChange={(e) => props.setThinking(e.currentTarget.checked)} />
          <span>Think before answering</span>
        </label>
        <button type="button" class="btn btn--tertiary btn--sm" onClick={() => props.onOpenPrompt()}>
          <Icon name="code" size={14} />
          <span class="btn__label">System prompt</span>
        </button>
        <div class="dock__preset">
          <button type="button" class="btn btn--secondary btn--sm" onClick={() => props.onSavePreset()}>
            Save as this model's settings
          </button>
          <Show when={props.hasPreset}>
            <button type="button" class="btn btn--tertiary btn--sm" onClick={() => props.onClearPreset()}>
              Forget them
            </button>
          </Show>
        </div>
        <p class="dock__note">{props.hasPreset ? "This model has its own settings: they come back whenever you pick it." : "Saved settings come back whenever you pick this model."}</p>
      </section>

      <Show when={model()}>
        {(m) => (
          <section class="dock__section" aria-labelledby="dock-model">
            <h3 id="dock-model">Loaded model</h3>
            <p class="dock__model">
              <strong>{m().name}</strong>
              <span class="mono">
                {m().params} · {m().quant}
              </span>
              <span>{where()}</span>
            </p>
            <button type="button" class="btn btn--secondary btn--sm" onClick={() => props.onUnload()}>
              Unload to free memory
            </button>
          </section>
        )}
      </Show>
    </div>
  );
}
