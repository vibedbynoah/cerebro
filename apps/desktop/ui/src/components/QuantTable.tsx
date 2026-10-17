import "../styles/quants.css";
import { createMemo, For, Show, type JSX } from "solid-js";
import { CHANNEL_LABELS, type Channel } from "../lib/experience";
import type { MemoryEstimate } from "../lib/memory-estimate";
import { availability, BACKEND_LABELS, bestAtSize, fmtKl, FORMAT_LABELS, qualityPerGiB } from "../lib/quants";
import { pct } from "../onboarding/recommend";
import type { CatalogQuant } from "../service/types";
import { IllustrativeFlag } from "./Screen";
import { Term } from "./Term";

const f1 = (n: number): string => (Math.round(n * 10) / 10).toFixed(1);

/** "Needs 6.2 GiB VRAM + 1.4 GiB RAM at 8K context": the Hub's per-file memory line. */
export function memoryNeeds(e: MemoryEstimate, ctxLabel: string): string {
  return `Needs ${f1(e.vramGiB)} GiB VRAM + ${f1(e.ramGiB)} GiB RAM at ${ctxLabel} context`;
}

/**
 * Every file of one model side by side: format, engine, bits per weight, size, KL divergence and
 * perplexity change against BF16, quality kept per GiB, and decode speed on this machine. Read-only
 * (Model Hub) or a radio group (Advanced setup). Files that need another engine stay listed, greyed
 * and unselectable, with the reason; GSQ-RCO files say they need the Experimental channel.
 */
export function QuantTable(props: {
  quants: readonly CatalogQuant[];
  channel: Channel;
  /** Accessible name of the table, e.g. "Files for Gemma-4-12B-it". */
  caption: string;
  /** Radio mode: the group's name and the chosen file. */
  pick?: { name: string; selected: string | null; onSelect: (id: string) => void };
  /** Model Hub only: each file's memory at the chosen context, and the label for that context. */
  memory?: { ctxLabel: string; of: (q: CatalogQuant) => MemoryEstimate | null } | undefined;
}): JSX.Element {
  const best = createMemo(() => bestAtSize(props.quants, props.channel));
  const maxQpg = createMemo(() => Math.max(1e-9, ...props.quants.map((q) => qualityPerGiB(q))));
  const illustrative = () => props.quants.some((q) => q.provenance === "illustrative");
  return (
    <div class="qtable-wrap">
      {/* Scrolls sideways on narrow windows; focusable so the keyboard can scroll it too. */}
      <div class="qtable-scroll" role="region" aria-label={props.caption} tabindex="0">
        <table class="qtable" classList={{ "qtable--pick": props.pick !== undefined }}>
          <caption class="sr-only">{props.caption}</caption>
          <thead>
            <tr>
              <Show when={props.pick}>
                <th scope="col">
                  <span class="sr-only">Choose</span>
                </th>
              </Show>
              <th scope="col">File</th>
              <th scope="col">Engine</th>
              <th scope="col" class="num">
                Bits
              </th>
              <th scope="col" class="num">
                Size
              </th>
              <Show when={props.memory}>
                <th scope="col" class="num">
                  Memory
                </th>
              </Show>
              <th scope="col" class="num">
                <abbr title="KL divergence against the unquantised weights. Lower is closer.">KL</abbr>
              </th>
              <th scope="col" class="num">
                <abbr title="Perplexity increase against the unquantised weights. Lower is better.">PPL Δ</abbr>
              </th>
              <th scope="col">Quality per GiB</th>
              <th scope="col" class="num">
                Here
              </th>
            </tr>
          </thead>
          <tbody>
            <For each={props.quants}>
              {(q) => {
                const avail = () => availability(q, props.channel);
                const rowId = () => `${props.pick?.name ?? "q"}-${q.id}`;
                const share = () => qualityPerGiB(q) / maxQpg();
                return (
                  <tr class="qrow" data-avail={avail()} data-quant={q.id} classList={{ "is-selected": props.pick?.selected === q.id }}>
                    <Show when={props.pick}>
                      {(pick) => (
                        <td class="qrow__pick">
                          <input
                            type="radio"
                            class="qrow__radio"
                            name={pick().name}
                            value={q.id}
                            id={rowId()}
                            checked={pick().selected === q.id}
                            disabled={avail() !== "runs"}
                            aria-labelledby={`${rowId()}-label`}
                            aria-describedby={`${rowId()}-status`}
                            onChange={() => pick().onSelect(q.id)}
                          />
                        </td>
                      )}
                    </Show>
                    <th scope="row" class="qrow__file">
                      <label for={props.pick ? rowId() : undefined} id={`${rowId()}-label`} class="qrow__label mono">
                        {q.label}
                      </label>
                      <span class="qrow__format" data-format={q.format}>
                        {FORMAT_LABELS[q.format]}
                      </span>
                      <Show when={best().has(q.id)}>
                        <span class="qrow__best">Best at its size</span>
                      </Show>
                    </th>
                    <td class="qrow__engine">{BACKEND_LABELS[q.backend]}</td>
                    <td class="num mono">{f1(q.bitsPerWeight)}</td>
                    <td class="num mono">
                      {f1(q.sizeGiB)}
                      <span class="unit">GiB</span>
                    </td>
                    <Show when={props.memory}>
                      {(mem) => (
                        <td class="num qrow__mem">
                          <Show when={mem().of(q)} fallback={<span class="qrow__why">Header not read</span>}>
                            {(e) => <span class="mono" data-estimate={q.id}>{memoryNeeds(e(), mem().ctxLabel)}</span>}
                          </Show>
                        </td>
                      )}
                    </Show>
                    <td class="num mono">{fmtKl(q.klDiv)}</td>
                    <td class="num mono">+{f1(q.pplDeltaPct)}%</td>
                    <td class="qrow__qpg">
                      <span class="qbar" aria-hidden="true">
                        <span class={`bar qbar__fill bar--p${pct(share())}`} data-bar data-bar-to={String(share())} />
                      </span>
                      <span class="mono qrow__qpg-num">{f1(qualityPerGiB(q))}</span>
                    </td>
                    <td class="num qrow__here" id={`${rowId()}-status`}>
                      <Show
                        when={avail() === "runs"}
                        fallback={
                          <span class="qrow__why">
                            {avail() === "channel" ? `Needs the ${CHANNEL_LABELS.experimental} channel` : (q.unavailable ?? "Not on this machine")}
                          </span>
                        }
                      >
                        <span class="mono">{f1(q.decodeTps ?? 0)}</span>
                        <span class="unit">tok/s</span>
                      </Show>
                    </td>
                  </tr>
                );
              }}
            </For>
          </tbody>
        </table>
      </div>
      <p class="qtable__foot">
        <span>
          <Term k="klDiv" /> and perplexity are against the unquantised weights; quality per GiB is quality kept (e<sup>−KL</sup>) per GiB of file. Speed is <Term k="decode">writing speed</Term> on
          this machine before Tune.
        </span>
        <Show when={illustrative()}>
          <IllustrativeFlag />
        </Show>
      </p>
    </div>
  );
}
