import "../styles/turbo.css";
import { createMemo, createSignal, For, Show, type JSX } from "solid-js";
import { Icon } from "../components/icons";
import { IllustrativeFlag, ScreenHeader, Section } from "../components/Screen";
import { SettingSwitch } from "../components/SettingSwitch";
import { Term } from "../components/Term";
import { CHANNEL_LABELS, CHANNEL_NOTES, CHANNELS, type Channel } from "../lib/experience";
import { notify } from "../lib/toast";
import {
  bottleneck,
  canPromote,
  clampProfile,
  clampSchedule,
  DEFAULT_SCHEDULE,
  diffProfiles,
  formatRatio,
  isControl,
  keepOrRevert,
  type CoreSet,
  type EngineBuild,
  type MachineProfile,
  type Priority,
  type ProfileChange,
  type QueuePolicy,
  type Schedule,
} from "../lib/turbo";
import { ENGINE_BUILDS, GPU_RANGE, LOGICAL_CORES, MEASURED_SWEET_SPOT, PROFILE_MEASUREMENTS, PROFILES, TOKEN_PROFILE } from "../mock/turbo";
import { useAppState } from "../state";

/**
 * Turbo: engine builds, schedulers, machine profiles and the profiler (docs/specs/turbo-system.md).
 * Every change shows what it touched, is measured, and can be reverted in one click. Beginner sees
 * only the profile picker; Advanced sees everything. Data is illustrative until the engine's
 * cerebro-builds and cerebro-profiles crates are wired in.
 */
export function Turbo(): JSX.Element {
  const app = useAppState();
  const advanced = () => app.level() === "advanced";
  return (
    <div class="screen turbo">
      <ScreenHeader
        route="turbo"
        lede="Builds tuned to this machine, smarter scheduling and power profiles. Each change is measured here and kept only if it is faster."
        actions={<IllustrativeFlag />}
      />
      <Profiles />
      <Show
        when={advanced()}
        fallback={
          <p class="hint turbo__beginner" data-testid="turbo-beginner">
            Builds, schedulers and the profiler are in Advanced. <button class="text-link" type="button" onClick={() => app.setLevel("advanced")}>Show everything</button>
          </p>
        }
      >
        <Engines channel={app.channel()} />
        <Schedulers />
        <Profiler />
        <Channels channel={app.channel()} onChange={(c) => app.setChannel(c)} />
      </Show>
    </div>
  );
}

// ---- Profiles -----------------------------------------------------------------------------------

function Profiles(): JSX.Element {
  const [active, setActive] = createSignal<MachineProfile>(PROFILES.find((p) => p.id === "balanced")!);
  const [previous, setPrevious] = createSignal<MachineProfile | null>(null);
  const [changes, setChanges] = createSignal<ProfileChange[]>([]);
  const [result, setResult] = createSignal<string>("");
  const [customW, setCustomW] = createSignal(PROFILES.find((p) => p.id === "custom")!.gpu.powerLimitW);

  const apply = (next: MachineProfile) => {
    const target = clampProfile(next.id === "custom" ? { ...next, gpu: { ...next.gpu, powerLimitW: customW() } } : next, GPU_RANGE);
    const before = PROFILE_MEASUREMENTS[active().id]!;
    const after = next.id === "custom" ? customMeasurement(target.gpu.powerLimitW) : PROFILE_MEASUREMENTS[next.id]!;
    const verdict = keepOrRevert(before, after);
    const diff = diffProfiles(active(), target);
    // Switching to a slower profile on purpose (Quiet, Battery) is the user's choice, not a regression.
    const intentional = next.id === "quiet" || next.id === "battery";
    if (verdict.keep || intentional || next.id === "balanced" || next.id === "max") {
      setPrevious(active());
      setActive(target);
      setChanges(diff);
      setResult(`${target.name}: ${after.tokPerSec.toFixed(1)} tok/s at ${after.gpuTempC} °C.`);
      notify(`${target.name} profile on`, diff.length ? `${diff.length} setting${diff.length === 1 ? "" : "s"} changed. Revert any time.` : "Nothing needed changing.");
    } else {
      setResult(`${target.name} not kept: ${verdict.reason}`);
      notify("Profile reverted", verdict.reason, "warning");
    }
  };

  const revert = () => {
    const prev = previous();
    if (!prev) return;
    setChanges(diffProfiles(active(), prev));
    setActive(prev);
    setPrevious(null);
    setResult(`Back to ${prev.name}.`);
  };

  return (
    <Section id="profiles" title="Machine profile" description={<>Power and fan limits for the processor and graphics card, measured on this machine. Settings return to how they were when Cerebro exits.</>}>
      <fieldset class="turbo-profiles" aria-describedby="profile-result">
        <legend class="sr-only">Machine profile</legend>
        <For each={PROFILES}>
          {(p) => (
            <label class="turbo-profile card" data-active={active().id === p.id} data-press>
              <input type="radio" name="machine-profile" value={p.id} checked={active().id === p.id} onChange={() => apply(p)} />
              <span class="turbo-profile__name">{p.name}</span>
              <span class="turbo-profile__summary">{p.summary}</span>
              <span class="turbo-profile__speed mono">{PROFILE_MEASUREMENTS[p.id]!.tokPerSec.toFixed(1)} tok/s</span>
            </label>
          )}
        </For>
      </fieldset>
      <Show when={active().id === "custom"}>
        <label class="turbo-field">
          <span>GPU power limit</span>
          <input
            type="range"
            min={GPU_RANGE.minPowerW}
            max={GPU_RANGE.maxPowerW}
            step={5}
            value={customW()}
            onInput={(e) => setCustomW(Number(e.currentTarget.value))}
            onChange={() => apply(PROFILES.find((p) => p.id === "custom")!)}
            aria-valuetext={`${customW()} watts`}
          />
          <span class="mono">{customW()} W</span>
          <span class="hint">The driver allows {GPU_RANGE.minPowerW}–{GPU_RANGE.maxPowerW} W on this card. Nothing outside that range is ever sent.</span>
        </label>
      </Show>
      <p class="turbo__result" id="profile-result" role="status" aria-live="polite">
        {result()}
      </p>
      <Show when={changes().length > 0}>
        <div class="turbo-changes card">
          <div class="turbo-changes__head">
            <h3>What changed</h3>
            <Show when={previous()}>
              <button class="btn btn--secondary" type="button" onClick={revert} data-press>
                <Icon name="undo" size={16} />
                <span class="btn__label">Revert</span>
              </button>
            </Show>
          </div>
          <table class="turbo-table">
            <thead>
              <tr>
                <th scope="col">Setting</th>
                <th scope="col">Before</th>
                <th scope="col">After</th>
              </tr>
            </thead>
            <tbody>
              <For each={changes()}>
                {(c) => (
                  <tr>
                    <th scope="row">{c.what}</th>
                    <td>{c.from}</td>
                    <td>{c.to}</td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </div>
      </Show>
    </Section>
  );
}

/** Illustrative: a power-limited laptop GPU gains little above its default and loses below it. */
function customMeasurement(watts: number) {
  const base = 17.9;
  const tok = watts >= 80 ? base + (watts - 80) * 0.02 : base - (80 - watts) * 0.25;
  return { tokPerSec: Math.round(tok * 10) / 10, gpuTempC: Math.round(60 + watts * 0.2), throttled: false };
}

// ---- Engines ------------------------------------------------------------------------------------

function Engines(props: { channel: Channel }): JSX.Element {
  const [active, setActive] = createSignal("llama-native");
  const choose = (b: EngineBuild) => {
    const v = canPromote(b, props.channel);
    if (!v.ok) {
      notify(`${b.name} not switched`, v.reason, "warning");
      return;
    }
    setActive(b.id);
    notify(`${b.name} is now the default`, isControl(b) ? "The official release." : formatRatio(b.speedVsControl));
  };
  return (
    <Section
      id="engines"
      title="Engines and builds"
      description={
        <>
          Builds compiled for this processor and graphics card, forks and other engines. The official <Term k="llamacpp" /> release always stays installed as the control; another build
          becomes the default only when it is faster here and passes the quality check.
        </>
      }
    >
      <ul class="turbo-engines" role="list">
        <For each={ENGINE_BUILDS}>
          {(b) => {
            const verdict = createMemo(() => canPromote(b, props.channel));
            return (
              <li class="turbo-engine card" data-engine={b.id} data-active={active() === b.id}>
                <div class="turbo-engine__head">
                  <h3 class="turbo-engine__name">{b.name}</h3>
                  <span class="chip" data-tone={isControl(b) || b.channel === "stable" ? "neutral" : "active"} data-channel={b.channel}>
                    {isControl(b) ? "Control" : CHANNEL_LABELS[b.channel]}
                  </span>
                </div>
                <p class="turbo-engine__target mono">
                  {b.target} · {b.version}
                </p>
                <p class="turbo-engine__note">{b.note}</p>
                <div class="turbo-engine__foot">
                  <span class="turbo-engine__speed" data-faster={(b.speedVsControl ?? 0) > 1}>
                    {formatRatio(b.speedVsControl)}
                  </span>
                  <Show when={active() !== b.id} fallback={<span class="chip" data-tone="on">Default</span>}>
                    <button class="btn btn--secondary" type="button" onClick={() => choose(b)} aria-describedby={`why-${b.id}`} data-press>
                      <span class="btn__label">{b.installed ? "Make default" : "Install"}</span>
                    </button>
                  </Show>
                </div>
                <Show when={!verdict().ok}>
                  <p class="hint" id={`why-${b.id}`}>
                    {(verdict() as { reason: string }).reason}
                  </p>
                </Show>
              </li>
            );
          }}
        </For>
      </ul>
    </Section>
  );
}

// ---- Schedulers ---------------------------------------------------------------------------------

function Schedulers(): JSX.Element {
  const [s, setS] = createSignal<Schedule>(DEFAULT_SCHEDULE);
  const set = (patch: Partial<Schedule>) => setS((cur) => clampSchedule({ ...cur, ...patch }, LOGICAL_CORES));
  return (
    <Section id="schedulers" title="Schedulers" description={<>How work is spread over the processor's cores and how requests take turns.</>}>
      <div class="turbo-grid">
        <label class="turbo-field">
          <span>
            CPU <Term k="threads" />
          </span>
          <select value={s().threads === null ? "auto" : String(s().threads)} onChange={(e) => set({ threads: e.currentTarget.value === "auto" ? null : Number(e.currentTarget.value) })}>
            <option value="auto">Measured best ({MEASURED_SWEET_SPOT})</option>
            <For each={[2, 4, 6, 8, 12, 16]}>{(n) => <option value={n}>{n}</option>}</For>
          </select>
          <span class="hint">More threads often run hotter at the same speed. {MEASURED_SWEET_SPOT} was fastest here.</span>
        </label>
        <Choice<CoreSet> label="Cores" value={s().cores} onChange={(v) => set({ cores: v })} options={[["performance", "Performance cores"], ["all", "All cores"], ["efficiency", "Efficiency cores"]]} />
        <Choice<Priority> label="Process priority" value={s().priority} onChange={(v) => set({ priority: v })} options={[["normal", "Normal"], ["above-normal", "Above normal"], ["high", "High"]]} />
        <Choice<QueuePolicy> label="Request queue" value={s().queue} onChange={(v) => set({ queue: v })} options={[["interactive-first", "Interactive first"], ["fifo", "First come"], ["shortest-first", "Shortest first"]]} />
        <label class="turbo-field">
          <span>Parallel conversations</span>
          <input type="number" min="1" max="8" value={s().parallelSlots} onChange={(e) => set({ parallelSlots: Number(e.currentTarget.value) })} />
          <span class="hint">Decoded together in one pass. Faster in total, a little slower each.</span>
        </label>
        <label class="turbo-field">
          <span>
            <Term k="prefill">Prefill</Term> chunk
          </span>
          <select value={s().prefillChunk === null ? "auto" : String(s().prefillChunk)} onChange={(e) => set({ prefillChunk: e.currentTarget.value === "auto" ? null : Number(e.currentTarget.value) })}>
            <option value="auto">Automatic</option>
            <For each={[1024, 2048, 4096, 8192, 16384]}>{(n) => <option value={n}>{n.toLocaleString()} tokens</option>}</For>
          </select>
        </label>
      </div>
      <SettingSwitch
        id="park-conversations"
        label="Keep recent conversations ready"
        note={<>Agents that take turns get their state back instead of reading the whole history again (<Term k="conversationCache" />).</>}
        checked={s().parkConversations}
        onChange={(v) => set({ parkConversations: v })}
      />
    </Section>
  );
}

function Choice<T extends string>(props: { label: string; value: T; onChange: (v: T) => void; options: ReadonlyArray<readonly [T, string]> }): JSX.Element {
  return (
    <label class="turbo-field">
      <span>{props.label}</span>
      <select value={props.value} onChange={(e) => props.onChange(e.currentTarget.value as T)}>
        <For each={props.options}>{([v, l]) => <option value={v}>{l}</option>}</For>
      </select>
    </label>
  );
}

// ---- Profiler -----------------------------------------------------------------------------------

function Profiler(): JSX.Element {
  const p = TOKEN_PROFILE;
  const parts = [
    { key: "gpu", label: "GPU kernels", share: p.gpuKernels },
    { key: "cpu", label: "CPU experts", share: p.cpuExperts },
    { key: "pcie", label: "PCIe transfers", share: p.pcie },
    { key: "other", label: "Other", share: p.other },
  ];
  return (
    <Section id="profiler" title="Profiler" description={<>Where the time in one generated token goes, and what would make it faster.</>}>
      <div class="turbo-profiler card">
        <div class="turbo-bar" role="img" aria-label={parts.map((x) => `${x.label} ${Math.round(x.share * 100)}%`).join(", ")}>
          <For each={parts}>{(x) => <span class="turbo-bar__seg" data-part={x.key} style={{ "flex-grow": x.share }} />}</For>
        </div>
        <ul class="turbo-legend" role="list">
          <For each={parts}>
            {(x) => (
              <li data-part={x.key}>
                <span class="turbo-legend__dot" aria-hidden="true" />
                {x.label} <span class="mono">{Math.round(x.share * 100)}%</span>
              </li>
            )}
          </For>
        </ul>
        <dl class="turbo-facts">
          <div>
            <dt>
              <Term k="expertCache">Expert cache</Term> hits
            </dt>
            <dd class="mono">{Math.round(p.expertCacheHit * 100)}%</dd>
          </div>
          <div>
            <dt>
              Draft <Term k="acceptance" />
            </dt>
            <dd class="mono">{p.draftAcceptance === null ? "off" : `${Math.round(p.draftAcceptance * 100)}%`}</dd>
          </div>
          <div>
            <dt>Free VRAM</dt>
            <dd class="mono">{p.vramFreeGiB.toFixed(1)} GiB</dd>
          </div>
        </dl>
        <p class="turbo-verdict">
          <Icon name="info" size={16} />
          <span data-testid="bottleneck">{bottleneck(p)}</span>
        </p>
      </div>
    </Section>
  );
}

// ---- Channels -----------------------------------------------------------------------------------

function Channels(props: { channel: Channel; onChange: (c: Channel) => void }): JSX.Element {
  return (
    <Section id="channels" title="Channels" description={<>Which optimisations Cerebro may try. Every one is still measured here and kept only if it helps.</>}>
      <fieldset class="segmented" aria-describedby="channel-note">
        <legend class="sr-only">Optimisation channel</legend>
        <For each={CHANNELS}>
          {(c) => (
            <label class="segmented__option" data-press>
              <input type="radio" name="turbo-channel" value={c} checked={props.channel === c} onChange={() => props.onChange(c)} />
              <span>{CHANNEL_LABELS[c]}</span>
            </label>
          )}
        </For>
      </fieldset>
      <p class="hint" id="channel-note">
        {CHANNEL_NOTES[props.channel]}
      </p>
    </Section>
  );
}
