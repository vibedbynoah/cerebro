import { createResource, For, Show, type JSX } from "solid-js";
import { CodeBlock } from "../components/CodeBlock";
import { Comparison, shares } from "../components/Figures";
import { Icon } from "../components/icons";
import { ScreenHeader, Section } from "../components/Screen";
import { Term } from "../components/Term";
import { useService } from "../service/context";

/** The stages of a tune, from docs/CLAIMS.md "About measurement". */
const STEPS = [
  { name: "Probe", label: "Read the machine: GPU, VRAM and the verified backend build." },
  { name: "Screen", label: "Try candidate settings quickly, discarding each first repetition." },
  { name: "Measure", label: "Prefill, decode at depth 0 and decode at your working context depth." },
  { name: "Soak", label: "Confirm survivors under a sustained run with a steady VRAM trend." },
  { name: "Decide", label: "Keep the fastest safe result, or keep the defaults if they won." },
];

export function Tune(): JSX.Element {
  const service = useService();
  const [models] = createResource(() => service.listModels());
  const moe = () => models()?.find((m) => m.tune === "tuned");
  const small = () => models()?.find((m) => m.tune === "defaults-fastest");

  return (
    <div class="screen">
      <ScreenHeader
        route="tune"
        lede={
          <>
            <Term k="tuning">Tuning</Term> runs real benchmarks on this machine and keeps the fastest configuration that passes every check. It
            never assumes a setting is faster.
          </>
        }
      />
      <Section id="how" title="How a tune runs" description="Every configuration gets three numbers, never one, and short bursts are confirmed by a longer run.">
        <div class="pipeline card">
          <span class="pipeline__rail" aria-hidden="true">
            <span class="pipeline__rail-fill" />
          </span>
          <ol class="pipeline__steps">
            <For each={STEPS}>
              {(step, i) => (
                <li class="pipeline__step">
                  <span class="pipeline__node" aria-hidden="true">
                    {i() + 1}
                  </span>
                  <span class="pipeline__name">{step.name}</span>
                  <span class="pipeline__desc">{step.label}</span>
                </li>
              )}
            </For>
          </ol>
        </div>
      </Section>
      <Show when={moe() && small()}>
        <Section id="outcomes" title="Two honest outcomes" description="A large model can gain a lot from tuning. A small one that fits in memory may not.">
          <div class="outcome-grid">
            <Comparison
              label={`${moe()?.name ?? ""}: defaults against tuned`}
              unit="tok/s"
              rows={shares([
                { label: "Defaults", value: moe()?.defaultsTps ?? 0 },
                { label: "Tuned", value: moe()?.tunedTps ?? 0, winner: true },
              ])}
              note={
                <>
                  {moe()?.name}: <strong>tuned settings won</strong> and passed the soak.
                </>
              }
            />
            <Comparison
              label={`${small()?.name ?? ""}: defaults against tuned`}
              unit="tok/s"
              rows={shares([
                { label: "Defaults", value: small()?.defaultsTps ?? 0, winner: true },
                { label: "Best tuned", value: small()?.tunedTps ?? 0 },
              ])}
              note={
                <>
                  {small()?.name}: <strong>defaults were fastest</strong>, so Cerebro kept them.
                </>
              }
            />
          </div>
        </Section>
      </Show>
      <Section id="cli" title="Start a tune" description="Tuning a model takes a while and uses the GPU fully; you can keep working in other screens.">
        <CodeBlock lines={["cerebro tune qwen3-8b-q5km"]} label="Copy command" />
        <p class="hint">
          <Icon name="info" size={16} />
          Starting a tune from this screen arrives in a later build.
        </p>
      </Section>
    </div>
  );
}
