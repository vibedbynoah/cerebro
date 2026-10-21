import { createMemo, For, Show, type JSX } from "solid-js";
import { CodeBlock } from "../../components/CodeBlock";
import { Icon } from "../../components/icons";
import { IllustrativeFlag } from "../../components/Screen";
import { Term } from "../../components/Term";
import { hrefWithModel } from "../../lib/routes";
import { answerSeconds, depthLabel, formatClock, markTerms, objectiveScore, type CandidateView, type TuneView } from "../../lib/tune";
import type { TuneMeasurement } from "../../service/types";
import { fmt, Num, SettingName, TermLine } from "./bits";
import { useAppState } from "../../state";

interface Metric {
  key: keyof TuneMeasurement;
  label: JSX.Element;
  digits: number;
}

/** The headline figure for the profile: what "fastest" means for it, as one number. */
interface Headline {
  value: number;
  digits: number;
  unit: string;
  /** How many times better than the comparison (always >= 1 when tuned settings won). */
  ratio: number | null;
}

/**
 * The result: one large figure (the profile's objective with the kept settings, against llama.cpp's
 * automatic settings, every condition right below it: docs/CLAIMS.md rule zero), then the three
 * numbers side by side, why it won, and how to use it. When llama.cpp's own settings won, the
 * comparison is against the best candidate instead, and says so plainly.
 *
 * `animate` is false when the result was already there as the screen opened (a replay): it then
 * appears at once, with no entrance and no count-up.
 */
export function WinnerCard(props: { view: TuneView; animate: boolean }): JSX.Element {
  const app = useAppState();
  const run = () => props.view.run!;
  const byId = (id: string) => props.view.candidates.find((c) => c.id === id);
  const winner = () => byId(props.view.winner!.candidateId)!;
  const baseline = () => byId(props.view.winner!.baselineId)!;
  const keptDefaults = () => winner().candidate.baseline;
  const longDocs = () => run().profile === "long-docs";
  /** What the kept settings are drawn against: the baseline, or (when the baseline won) the best
   *  measured candidate. */
  const other = createMemo<CandidateView | null>(() => {
    if (!keptDefaults()) return baseline();
    const measured = props.view.candidates.filter((c) => !c.candidate.baseline && c.result);
    measured.sort((a, b) => objectiveScore(run().profile, b.result!, run().depth) - objectiveScore(run().profile, a.result!, run().depth));
    return measured[0] ?? null;
  });
  const headline = createMemo<Headline | null>(() => {
    const w = winner().result;
    if (!w) return null;
    const o = other()?.result ?? null;
    if (longDocs()) {
      const s = answerSeconds(w, run().depth);
      return { value: Math.round(s), digits: 0, unit: "s", ratio: o && !keptDefaults() ? answerSeconds(o, run().depth) / s : null };
    }
    return { value: w.decodeTpsDepth, digits: 1, unit: "tok/s", ratio: o && !keptDefaults() ? w.decodeTpsDepth / o.decodeTpsDepth : null };
  });
  const otherFigure = () => {
    const o = other()?.result;
    if (!o) return "";
    return longDocs() ? `${fmt(Math.round(answerSeconds(o, run().depth)), 0)} s` : `${fmt(o.decodeTpsDepth, 1)} tok/s`;
  };
  // Each term is defined once: the headline defines writing speed and depth (or, for long documents,
  // prompt reading); the bar labels define whichever the headline did not.
  const metrics = (): Metric[] => [
    { key: "prefillTps", label: longDocs() ? <>Prompt reading</> : <Term k="prefill">Prompt reading</Term>, digits: 0 },
    { key: "decodeTps0", label: <>Writing, empty context</>, digits: 1 },
    {
      key: "decodeTpsDepth",
      label: longDocs() ? (
        <>
          <Term k="decode">Writing</Term> at <Term k="depth">{`${depthLabel(run().depth)} depth`}</Term>
        </>
      ) : (
        <>{`Writing at ${depthLabel(run().depth)} depth`}</>
      ),
      digits: 1,
    },
  ];
  const rows = (): Array<{ c: CandidateView; name: string; win: boolean }> => {
    const o = other();
    const list = [{ c: winner(), name: keptDefaults() ? "Automatic settings (kept)" : "Cerebro tuned", win: true }];
    if (o) list.unshift({ c: o, name: keptDefaults() ? "Best candidate" : "Automatic settings", win: false });
    return list;
  };
  const soak = () => props.view.soaks.find((s) => s.candidateId === winner().id);
  const quality = () => props.view.quality.find((q) => q.candidateId === winner().id);
  const conditions = () =>
    [
      `${run().gpu} (${fmt(run().vramTotalGiB, 0)} GiB)`,
      `${run().cpu}, ${fmt(run().ramTotalGiB, 0)} GiB RAM`,
      `${run().modelName} ${run().quant}`,
      `${run().build} (${run().backendCommit})`,
      `depth ${depthLabel(run().depth)}`,
      run().sampler,
      "first repetition discarded",
    ].join(" · ");
  // The subtitle defines "automatic settings" when tuned settings won; the reasons define the rest.
  const reasons = createMemo(() => markTerms(props.view.winner!.reasons, keptDefaults() ? [] : ["autofit"]));

  return (
    <section class="winner" classList={{ "winner--enter": props.animate }} aria-labelledby="winner-title" data-kept={keptDefaults() ? "defaults" : "tuned"}>
      <header class="winner__head">
        <div>
          <p class="winner__eyebrow">
            <Icon name={keptDefaults() ? "check" : "lock"} size={14} />
            {keptDefaults() ? "Kept the automatic settings" : "Locked in"}
          </p>
          <h3 class="winner__title" id="winner-title">
            {keptDefaults() ? "Defaults were fastest" : winner().candidate.label}
          </h3>
        </div>
        <span class={`result-card__status${keptDefaults() ? " result-card__status--neutral" : ""}`}>
          <span class="result-card__status-dot" aria-hidden="true" />
          {keptDefaults() ? "Nothing beat them" : "Passed every check"}
        </span>
      </header>

      <figure class="result-card__figure winner__figure" aria-label={`Three speeds, ${rows().map((r) => r.name).join(" against ")}`}>
        <figcaption class="winner__caption">
          <Show when={headline()}>
            {(h) => (
              <div class="winner__headline">
                <p class="winner__hero">
                  <span class="num-unit mono winner__hero-num">
                    <span data-count-to={props.animate ? fmt(h().value, h().digits).replaceAll(",", "") : undefined}>{fmt(h().value, h().digits)}</span>
                    <span class="unit">{h().unit}</span>
                  </span>
                  <IllustrativeFlag />
                </p>
                <p class="winner__hero-text">
                  <Show
                    when={longDocs()}
                    fallback={
                      <>
                        <Term k="decode">Writing speed</Term> at <Term k="depth">{`${depthLabel(run().depth)} depth`}</Term>
                      </>
                    }
                  >
                    To read a {depthLabel(run().depth)}-token document (<Term k="prefill">prompt reading</Term>) and write a 500-token answer
                  </Show>
                  <Show when={h().ratio !== null && other()}>
                    {". "}
                    <strong class="winner__ratio mono">{`${fmt(h().ratio!, 1)}×`}</strong>
                    {longDocs() ? ` faster than the automatic settings (${otherFigure()}).` : ` the writing speed of the automatic settings (${otherFigure()}).`}
                  </Show>
                  <Show when={keptDefaults() && other()}>{`. The best candidate reached ${otherFigure()}.`}</Show>
                  <Show when={keptDefaults() && !other()}>.</Show>
                </p>
              </div>
            )}
          </Show>
          <p class="winner__conditions">
            <span class="winner__conditions-label">Conditions</span> {conditions()}
          </p>
        </figcaption>
        <div class="versus">
          <For each={metrics()}>
            {(m, mi) => {
              const max = () => Math.max(...rows().map((r) => r.c.result?.[m.key] ?? 0), 1e-9);
              return (
                <div class="versus__metric">
                  <p class="versus__label">{m.label}</p>
                  <For each={rows()}>
                    {(r, ri) => {
                      const value = () => r.c.result?.[m.key] ?? 0;
                      const share = () => Math.min(1, value() / max());
                      return (
                        <div class="versus__row" classList={{ "versus__row--win": r.win }}>
                          <span class="versus__name">{r.name}</span>
                          <span class="versus__track">
                            <span
                              class={`bar bar--p${Math.round(share() * 100)}`}
                              classList={{ "bar--tuned": r.win }}
                              data-bar={props.animate ? "" : undefined}
                              data-bar-to={String(share())}
                              data-bar-delay={String(mi() * 140 + ri() * 90)}
                            />
                          </span>
                          <span class="versus__value">
                            <Num value={value()} digits={m.digits} unit="tok/s" />
                          </span>
                        </div>
                      );
                    }}
                  </For>
                </div>
              );
            }}
          </For>
        </div>
        <Show when={longDocs() && rows().length === 2}>
          <p class="versus__answer">
            Read a {depthLabel(run().depth)}-token document and write a 500-token answer:{" "}
            <For each={rows()}>
              {(r, i) => (
                <>
                  {i() > 0 ? " against " : ""}
                  <strong class="mono">{formatClock(r.c.result ? answerSeconds(r.c.result, run().depth) : 0)}</strong> {r.win ? "kept" : `for ${r.name.toLowerCase()}`}
                </>
              )}
            </For>
            .
          </p>
        </Show>
      </figure>

      <div class="winner__body">
        <div class="winner__why">
          <h4 class="winner__subtitle">
            {keptDefaults() ? "Why the automatic settings stayed" : "Why it won against the automatic "}
            <Show when={!keptDefaults()}>
              <Term k="autofit" />
            </Show>
          </h4>
          <ol class="why">
            <For each={reasons()}>
              {(parts) => (
                <li>
                  <TermLine parts={parts} />
                </li>
              )}
            </For>
          </ol>
          <Show when={soak()?.result || quality()}>
            <ul class="checks" aria-label="Checks it passed">
              <Show when={soak()?.result}>
                {(r) => (
                  <li>
                    <Icon name="check" size={14} />
                    {`Soak: free graphics memory trend ${r().slopeMiBPerMin >= 0 ? "+" : "−"}${fmt(Math.abs(r().slopeMiBPerMin), 1)} MiB/min over the final third of ${formatClock(run().soakSeconds)}`}
                  </li>
                )}
              </Show>
              <Show when={quality()}>
                {(q) => (
                  <li>
                    <Icon name="check" size={14} />
                    {`Quality gate: ${q().metric} ${fmt(q().value, 4)}, limit ${fmt(q().gate, 3)}`}
                  </li>
                )}
              </Show>
            </ul>
          </Show>
        </div>
        <div class="winner__use">
          <h4 class="winner__subtitle">Settings</h4>
          <dl class="winner__settings">
            <For each={winner().candidate.settings}>
              {(s) => (
                <div>
                  <dt>
                    <SettingName name={s.name} />
                  </dt>
                  <dd>
                    {s.value}
                    {/* Raw flags are Advanced only (lib/experience.ts). */}
                    <Show when={app.level() === "advanced" && s.flag}>
                      <code class="mono">{s.flag}</code>
                    </Show>
                  </dd>
                </div>
              )}
            </For>
          </dl>
          <CodeBlock lines={[`cerebro serve ${run().modelId}`, "cerebro chat"]} label="Copy commands" />
          <a class="btn btn--tertiary btn--sm winner__models" href={hrefWithModel("models", run().modelId)} data-press>
            Open in Models<span class="sr-only">{`: ${run().modelName}`}</span>
            <Icon name="arrowRight" size={14} class="btn__arrow" />
          </a>
        </div>
      </div>
    </section>
  );
}
