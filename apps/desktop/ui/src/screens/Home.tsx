import { createResource, For, lazy, Show, type JSX } from "solid-js";
import { CodeBlock } from "../components/CodeBlock";
import { Comparison, shares, StatTile } from "../components/Figures";
import { Icon } from "../components/icons";
import { ScreenHeader, Section } from "../components/Screen";
import { Term } from "../components/Term";
import { fmt1 } from "../lib/format";
import { hrefFor } from "../lib/routes";
import { useService } from "../service/context";
import { createLiveStats } from "../service/live";
import { useAppState } from "../state";

// Quick actions and recent activity load after first paint.
const Discover = lazy(() => import("./home/Discover").then((m) => ({ default: m.Discover })));

export function Home(): JSX.Element {
  const service = useService();
  const { settings } = useAppState();
  // Live tiles read the same subscribed sample as the status bar, so the two always agree.
  const live = createLiveStats(service);
  const [data] = createResource(async () => ({ models: await service.listModels() }));
  /** Counted from the data: a fact about this machine only once every model's data is measured. */
  const modelsProvenance = () => (data()?.models.every((m) => m.provenance === "measured") ? "measured" : "illustrative");
  const tuned = () => data()?.models.find((m) => m.tune === "tuned" && m.defaultsTps !== null && m.tunedTps !== null);
  const fitsDefaults = () => data()?.models.find((m) => m.tune === "defaults-fastest");

  const protections = () => {
    const s = settings();
    if (!s) return [];
    return [
      { label: "Process containment", on: s.processContainment, term: "jobObject" as const },
      { label: "Integrity check before every start", on: s.integrityCheck, term: "integrity" as const },
      { label: "API on loopback only", on: s.loopbackOnly, term: "loopback" as const },
    ];
  };

  return (
    <div class="screen">
      <ScreenHeader route="home" title="Home" />

      <Discover />
      <Show when={data()}>
        {(d) => (
          <>
            <Section id="now" title="Now">
              <div class="tile-grid">
                <Show when={live()} fallback={<TilePlaceholder count={2} />}>
                  {(s) => (
                    <>
                      <StatTile
                        value={s().tokensPerSecond ?? Number.NaN}
                        decimals={1}
                        unit="tok/s"
                        live
                        accent
                        label={
                          <>
                            decode speed, <Term k="tokps">tok/s</Term>
                          </>
                        }
                        provenance={s().provenance}
                      />
                      <StatTile
                        value={s().vramUsedGiB}
                        decimals={1}
                        of={`/${fmt1(s().vramTotalGiB)}`}
                        unit="GiB"
                        live
                        label={
                          <>
                            <Term k="vram" /> in use
                          </>
                        }
                        provenance={s().provenance}
                      />
                    </>
                  )}
                </Show>
                <StatTile value={d().models.length} label="models on this machine" provenance={modelsProvenance()} source="counted" />
              </div>
            </Section>

            <Show when={tuned()}>
              {(m) => (
                <Section
                  id="measured"
                  title="Measured, not assumed"
                  description={
                    <>
                      {m().name} is a <Term k="moe" /> model larger than this card's memory, where settings matter most. A small model that
                      fits can be fastest on defaults; Cerebro measures both and says which won.
                    </>
                  }
                >
                  <Comparison
                    label={`Decode speed of ${m().name}, defaults against tuned`}
                    unit="tok/s"
                    rows={shares([
                      { label: "Automatic settings", value: m().defaultsTps ?? 0 },
                      { label: "Cerebro tuned", value: m().tunedTps ?? 0, winner: true },
                    ])}
                    note={
                      <Show when={fitsDefaults()}>
                        {(f) => (
                          <>
                            {f().name}: <strong>defaults were fastest</strong>, so Cerebro kept them.
                          </>
                        )}
                      </Show>
                    }
                  />
                </Section>
              )}
            </Show>
          </>
        )}
      </Show>

      {/* The desktop app has no command line and runs the engine in its own process, so neither of
          these sections applies there; the browser preview keeps them as a design study. */}
      <Show when={!__DESKTOP__}>
      <Section id="start" title="From the command line" description="Everything here also works from a terminal, and none of it needs a network.">
        <CodeBlock large lines={["cerebro models", "cerebro tune qwen3-coder-30b-a3b-q4km", "cerebro serve qwen3-coder-30b-a3b-q4km", "cerebro chat"]} />
      </Section>

      <Section id="protect" title="What protects the model process" description="Safe by default. Each one can be switched off in Settings.">
        <ul class="check-list">
          <For each={protections()}>
            {(p) => (
              <li class="check-list__item" data-on={p.on}>
                <span class="check-list__mark" aria-hidden="true">
                  <Icon name={p.on ? "check" : "minimize"} size={14} />
                </span>
                <span class="check-list__label">
                  {p.label} <Term k={p.term} />
                </span>
                <span class="chip" data-tone={p.on ? "on" : "off"}>
                  {p.on ? "On" : "Off"}
                </span>
              </li>
            )}
          </For>
        </ul>
        <a class="btn btn--secondary" href={`${hrefFor("settings")}`} data-press>
          Review protections
          <span class="btn__arrow" aria-hidden="true">
            <Icon name="arrowRight" size={16} />
          </span>
        </a>
      </Section>
      </Show>
    </div>
  );
}

/** Holds the tiles' space until the first live sample arrives, so nothing shifts when it does. */
function TilePlaceholder(props: { count: number }): JSX.Element {
  return (
    <For each={Array.from({ length: props.count })}>
      {() => <div class="stat-tile stat-tile--pending" aria-hidden="true" />}
    </For>
  );
}
