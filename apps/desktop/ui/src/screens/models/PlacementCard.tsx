import { For, Show, type JSX } from "solid-js";
import type { Placement } from "../../service/types";

const GIB = 1024 ** 3;
const gib = (b: number) => `${(b / GIB).toFixed(1)} GiB`;

export const DRIVE_LABELS: Readonly<Record<Placement["drive"], string>> = {
  nvme: "NVMe SSD",
  "sata-ssd": "SATA SSD",
  "hard-disk": "Hard disk",
  removable: "USB drive or memory card",
  unknown: "Drive",
};

const LAYOUT_LABELS: Readonly<Record<Placement["layout"], string>> = {
  "gpu-int8": "All on the graphics card",
  "gpu-native": "All on the graphics card",
  split: "Graphics card and processor",
  cpu: "Processor only",
};

/**
 * Where the engine puts a model (cerebro_engine::placement): layers on the GPU, what the processor
 * reads from RAM, what comes from the drive for every token, and an estimated decode speed. The
 * estimate is computed from each tier's bandwidth, never shown as a measurement.
 */
export function PlacementCard(props: { plan: Placement; title?: string }): JSX.Element {
  const p = () => props.plan;
  const slowDrive = () => p().driveBytesPerToken > 0;
  return (
    <section class="placement card" data-layout={p().layout} data-slow={slowDrive() ? "true" : undefined} aria-label={props.title ?? "Placement"}>
      <header class="placement__head">
        <span class="placement__title">{props.title ?? "Placement"}</span>
        <span class="chip" data-tone={p().layout === "cpu" ? "neutral" : "on"}>
          {LAYOUT_LABELS[p().layout]}
        </span>
      </header>
      <dl class="placement__tiers">
        <Show when={p().totalLayers > 0}>
          <div>
            <dt>Graphics card</dt>
            <dd class="mono">
              {p().gpuLayers}/{p().totalLayers} layers · {gib(p().vramBytes)}
            </dd>
          </div>
        </Show>
        <Show when={p().totalLayers === 0 && p().vramBytes > 0}>
          <div>
            <dt>Graphics card</dt>
            <dd class="mono">{gib(p().vramBytes)}</dd>
          </div>
        </Show>
        <div>
          <dt>RAM</dt>
          <dd class="mono">{gib(p().ramResident)}</dd>
        </div>
        <div>
          <dt>{DRIVE_LABELS[p().drive]}, per token</dt>
          <dd class="mono" data-warn={slowDrive() ? "true" : undefined}>
            {slowDrive() ? gib(p().driveBytesPerToken) : "none"}
          </dd>
        </div>
        <div>
          <dt>Estimated speed</dt>
          <dd class="mono">{p().estDecodeTps >= 10 ? Math.round(p().estDecodeTps) : p().estDecodeTps.toFixed(p().estDecodeTps >= 1 ? 1 : 2)} tok/s</dd>
        </div>
      </dl>
      <ul class="placement__notes">
        <For each={p().notes}>{(n) => <li>{n}</li>}</For>
      </ul>
      <p class="hint">An estimate from each part's speed; Tune measures the real figure.</p>
    </section>
  );
}
