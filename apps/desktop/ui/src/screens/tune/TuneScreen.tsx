import "../../styles/tune.css";
import { createMemo, createResource, createSignal, onCleanup, Show, type JSX } from "solid-js";
import { prefersReducedMotion } from "@design/motion";
import { CodeBlock } from "../../components/CodeBlock";
import { ScreenHeader, Section } from "../../components/Screen";
import { Term } from "../../components/Term";
import { modelFromHash } from "../../lib/routes";
import { announcement } from "../../lib/tune";
import { useService } from "../../service/context";
import type { HardwareProfile, LibraryModel, TuneState } from "../../service/types";
import { PhaseStepper, RunPanel } from "./RunPanel";
import { createTuneSession } from "./session";
import { TuneSetup, type PickModel } from "./Setup";

/**
 * The Tune screen: pick a model and a workload, start a tune, and watch it measure. The run lives in
 * the service, so this screen can be left and reopened (the run is replayed), and a cancelled run
 * can be resumed. Loaded as its own chunk (screens/Tune.tsx), with its own stylesheet.
 */
export function TuneScreen(): JSX.Element {
  const service = useService();
  const session = createTuneSession(service);
  const [summaries] = createResource(() => service.listModels().catch(() => []));
  const [libraryList] = createResource(() => service.listLibrary().catch((): LibraryModel[] => []));
  const library = createMemo<ReadonlyMap<string, LibraryModel>>(() => new Map((libraryList() ?? []).map((m) => [m.id, m])));
  // This machine, or null: a failed or malformed detection is never replaced by a stand-in machine.
  const [hardwareRaw] = createResource(() => service.detectHardware().catch(() => null));
  const hardware = createMemo<HardwareProfile | null>(() => {
    const h = hardwareRaw();
    return h && typeof h === "object" && h.cpu && Number.isInteger(h.cpu.threads) && h.cpu.threads > 0 ? h : null;
  });

  // Opened from a model's Tune action (`#/tune?model=<id>`): that model is offered and chosen, once
  // the library confirms it is a model that can be tuned. An unknown or malformed id is ignored.
  const [hash, setHash] = createSignal(window.location.hash);
  const onHash = () => setHash(window.location.hash);
  window.addEventListener("hashchange", onHash);
  onCleanup(() => window.removeEventListener("hashchange", onHash));
  const picked = createMemo(() => {
    const id = modelFromHash(hash());
    const m = id ? library().get(id) : undefined;
    return m && (m.kind === "text" || m.kind === "vision") ? m : null;
  });
  const pickedId = () => picked()?.id ?? null;

  // The cards' models change only when the library does; the tune state streams separately
  // (tuneOf), so starting, finishing or cancelling a run never rebuilds a card.
  const toPick = (m: PickModel): PickModel => ({ id: m.id, name: m.name, quant: m.quant, params: m.params, architecture: m.architecture, sizeGiB: m.sizeGiB });
  const models = createMemo<PickModel[] | undefined>(() => {
    const list = summaries();
    // The cards wait for the library and the hardware as well, so their memory line is there from
    // the first frame instead of arriving late and shifting the layout.
    if (!list || libraryList.loading || hardwareRaw.loading) return undefined;
    const out: PickModel[] = list.map(toPick);
    const p = picked();
    if (p && !out.some((m) => m.id === p.id)) out.unshift(toPick(p));
    return out;
  });
  const savedTune = createMemo(() => {
    const out = new Map<string, TuneState>();
    for (const m of libraryList() ?? []) out.set(m.id, m.tune);
    for (const m of summaries() ?? []) out.set(m.id, m.tune);
    return out;
  });
  // The run in view overrides the saved state for its model: measuring while it runs, and its
  // outcome once it is done.
  const tuneOf = (id: string): TuneState => {
    const run = session.view.run;
    const saved = savedTune().get(id) ?? "untuned";
    if (!run || run.modelId !== id) return saved;
    if (session.view.status === "running") return "tuning";
    if (session.view.status === "done") return session.view.candidates.find((c) => c.winner)?.candidate.baseline ? "defaults-fastest" : "tuned";
    return saved === "tuning" ? "untuned" : saved;
  };

  const [chosenId, setChosenId] = createSignal<string | null>(null);
  const hasRun = () => session.view.run !== null;
  const scrollTo = (id: string, focus?: string) =>
    requestAnimationFrame(() => {
      const el = document.getElementById(id);
      el?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
      if (focus) document.querySelector<HTMLElement>(focus)?.focus({ preventScroll: true });
    });

  const runTitle = () => {
    switch (session.view.status) {
      case "running":
        return "Tune in progress";
      case "cancelled":
        return "Tune cancelled";
      case "failed":
        return "Tune stopped";
      default:
        return "Tune result";
    }
  };
  const command = () => {
    const id = chosenId() ?? session.view.run?.modelId ?? models()?.[0]?.id;
    return id ? `cerebro tune ${id}` : "cerebro tune <model>";
  };

  return (
    <div class="screen screen--tune">
      <ScreenHeader
        route="tune"
        lede={
          <>
            <Term k="tuning">Tuning</Term> runs real benchmarks on this machine and keeps the fastest configuration that passes every check. It never assumes a setting is
            faster: when nothing beats <Term k="autofit">llama.cpp's automatic settings</Term>, it keeps them.
          </>
        }
      />
      <Show when={session.ready()} fallback={<p class="sr-only">Checking for a tune in progress…</p>}>
        <Show when={hasRun()}>
          <Section
            id="run"
            title={runTitle()}
            description={
              session.view.status === "running"
                ? "Live from the tune. You can leave this screen; it keeps measuring and picks up here when you return."
                : session.view.status === "done"
                  ? "Every candidate, what it measured, and what was kept."
                  : "Everything it measured before it stopped."
            }
          >
            <RunPanel session={session} onNewTune={() => scrollTo("setup-title", ".setup .pick-card input:checked")} />
          </Section>
        </Show>
        <Section
          id="setup"
          title={hasRun() ? "Start another tune" : "Start a tune"}
          description="Choose a model and what you will mostly use it for. The workload sets how much text is in view while Cerebro measures, and what counts as fastest."
        >
          <TuneSetup
            session={session}
            models={models}
            tuneOf={tuneOf}
            library={library}
            hardware={hardware}
            pickedId={pickedId}
            onChoose={setChosenId}
            onStarted={() => scrollTo("run-title")}
          />
        </Section>
        <Show when={!hasRun()}>
          <Section id="how" title="How a tune runs" description="Every configuration gets three numbers, never one, and short bursts are confirmed by a longer run before anything is kept.">
            <PhaseStepper view={null} />
          </Section>
        </Show>
        <Section id="cli" title="From the command line" description="The same tune from a terminal. It uses the GPU fully while it measures.">
          <CodeBlock lines={[command()]} label="Copy command" />
        </Section>
      </Show>
      <p class="sr-only" role="status" aria-live="polite">
        {announcement(session.view)}
      </p>
    </div>
  );
}
