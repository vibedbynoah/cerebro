import { createMemo, createSignal, createUniqueId, Index, Show, type JSX } from "solid-js";
import { breakdown, compact, num, pct, seconds } from "../../lib/monitor";
import type { RequestSummary } from "../../service/monitor-types";

/**
 * Where the time and tokens went, per client session or per tool the model called, across every
 * request the engine keeps. Rows are keyed by position (Index), so a new sample updates cells in
 * place.
 */
export function Breakdown(props: { requests: readonly RequestSummary[]; illustrative: boolean }): JSX.Element {
  const id = createUniqueId();
  const [by, setBy] = createSignal<"session" | "tool">("session");
  const rows = createMemo(() => breakdown(props.requests, by()));
  return (
    <div class="mon-breakdown">
      <div class="mon-breakdown__bar">
        <fieldset class="segmented">
          <legend class="sr-only">Group requests by</legend>
          <label class="segmented__option">
            <input type="radio" name={`${id}-by`} value="session" checked={by() === "session"} onChange={() => setBy("session")} />
            By session
          </label>
          <label class="segmented__option">
            <input type="radio" name={`${id}-by`} value="tool" checked={by() === "tool"} onChange={() => setBy("tool")} />
            By tool
          </label>
        </fieldset>
        <Show when={props.illustrative}>
          <span class="stat-tile__flag">Illustrative data — not a measurement</span>
        </Show>
      </div>
      <div class="table-card card">
        <div class="table-scroll" tabindex="0" role="region" aria-label={by() === "session" ? "Requests by session" : "Requests by tool"}>
          <table class="table mon-table">
            <thead>
              <tr>
                <th scope="col">{by() === "session" ? "Session" : "Tool the model called"}</th>
                <th scope="col" class="num">
                  Requests
                </th>
                <th scope="col" class="num">
                  Prompt tokens
                </th>
                <th scope="col" class="num">
                  Reused
                </th>
                <th scope="col" class="num">
                  Written
                </th>
                <th scope="col" class="num">
                  Writing speed
                </th>
                <th scope="col" class="num">
                  First token
                </th>
                <th scope="col" class="num">
                  Failed
                </th>
              </tr>
            </thead>
            <tbody>
              <Index each={rows()} fallback={<tr><td colspan="8" class="table__empty">No requests yet.</td></tr>}>
                {(r) => (
                  <tr>
                    <th scope="row" class={by() === "tool" && r().key ? "mono" : undefined}>
                      {r().label}
                    </th>
                    <td class="num">{num(r().requests)}</td>
                    <td class="num">{compact(r().promptTokens)}</td>
                    <td class="num">{num(pct(r().reusedTokens, r().promptTokens))}%</td>
                    <td class="num">{compact(r().outputTokens)}</td>
                    <td class="num">{r().decodeTps === null ? "–" : `${num(r().decodeTps, 1)} tok/s`}</td>
                    <td class="num">{seconds(r().meanFirstTokenS)}</td>
                    <td class="num">{r().failed > 0 ? num(r().failed) : "–"}</td>
                  </tr>
                )}
              </Index>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
