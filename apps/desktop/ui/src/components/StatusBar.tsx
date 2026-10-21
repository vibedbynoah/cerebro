import { createEffect, Show, type JSX } from "solid-js";
import { setBarValue } from "@design/motion";
import type { LiveStats, NetworkState } from "../service/types";
import { useService } from "../service/context";
import { createLiveStats } from "../service/live";
import { fmt1, ratio } from "../lib/format";
import { Icon } from "./icons";

const BACKEND_LABEL: Record<LiveStats["backend"]["state"], string> = {
  stopped: "Stopped",
  starting: "Starting",
  running: "Running",
  error: "Error",
};

/**
 * The HUD along the bottom: backend state, loaded model, decode speed and graphics memory. Values come from the
 * service; while they are illustrative (web mode, mock) the bar says so in words. Not a live region:
 * a number changing every second would flood a screen reader.
 */
export function StatusBar(props: { network: NetworkState | null; offlineMode: boolean }): JSX.Element {
  const service = useService();
  const stats = createLiveStats(service);
  let gauge: HTMLSpanElement | undefined;

  const vramShare = () => {
    const s = stats();
    return s ? ratio(s.vramUsedGiB, s.vramTotalGiB) : 0;
  };
  createEffect(() => {
    const share = vramShare();
    if (gauge) setBarValue(gauge, share);
  });

  const networkLabel = () => {
    if (props.offlineMode) return "Offline mode";
    if (props.network === null) return "Checking network";
    return props.network === "online" ? "Online" : "No network";
  };

  return (
    <footer class="statusbar" aria-label="Status">
      <Show when={stats()} fallback={<span class="hud-item hud-item--muted">Connecting to backend…</span>}>
        {(s) => (
          <>
            <span class="hud-item" data-hud="backend">
              <span class="status-dot" data-state={s().backend.state} aria-hidden="true" />
              <span class="hud-item__label">Backend</span>
              <span class="hud-item__value">{BACKEND_LABEL[s().backend.state]}</span>
              <Show when={s().backend.address}>{(addr) => <span class="hud-item__meta mono">{addr()}</span>}</Show>
            </span>
            <span class="hud-sep" aria-hidden="true" />
            <span class="hud-item" data-hud="model">
              <Icon name="models" size={14} />
              <span class="sr-only">Model</span>
              <span class="hud-item__value hud-item__value--truncate" title={s().model ?? undefined}>
                {s().model ?? "No model loaded"}
              </span>
            </span>
            <span class="hud-sep" aria-hidden="true" />
            <span class="hud-item" data-hud="speed">
              <Icon name="bolt" size={14} />
              <span class="sr-only">Decode speed</span>
              <span class="hud-item__value mono readout">
                {s().tokensPerSecond === null ? "–" : fmt1(s().tokensPerSecond ?? 0)}
                <span class="unit">tok/s</span>
              </span>
            </span>
            <span class="hud-sep" aria-hidden="true" />
            <span class="hud-item" data-hud="vram">
              <Icon name="chip" size={14} />
              <span class="hud-item__label">Graphics memory</span>
              <span
                class="hud-gauge"
                role="meter"
                aria-label="Graphics memory in use"
                aria-valuemin={0}
                aria-valuemax={s().vramTotalGiB}
                aria-valuenow={s().vramUsedGiB}
                aria-valuetext={`${fmt1(s().vramUsedGiB)} of ${fmt1(s().vramTotalGiB)} GiB`}
              >
                <span class="bar hud-gauge__fill" classList={{ "is-high": vramShare() > 0.9 }} ref={gauge} />
              </span>
              <span class="hud-item__value mono readout">
                {fmt1(s().vramUsedGiB)}
                <span class="hud-item__of">/{fmt1(s().vramTotalGiB)}</span>
                <span class="unit">GiB</span>
              </span>
            </span>
          </>
        )}
      </Show>
      <span class="statusbar__spacer" />
      <Show when={stats()?.provenance === "illustrative"}>
        <span class="hud-flag">Illustrative data</span>
      </Show>
      <span class="hud-item hud-item--muted" data-hud="network">
        <Icon name={props.offlineMode || props.network !== "online" ? "offline" : "online"} size={14} />
        <span>{networkLabel()}</span>
      </span>
    </footer>
  );
}
