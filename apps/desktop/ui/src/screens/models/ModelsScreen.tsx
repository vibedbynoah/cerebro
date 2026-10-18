import "../../styles/models.css";
import { For, Show, type JSX } from "solid-js";
import { CodeBlock } from "../../components/CodeBlock";
import { Icon } from "../../components/icons";
import { ScreenHeader, Section } from "../../components/Screen";
import { Term } from "../../components/Term";
import type { EngineSlot } from "../../service/types";
import { DiskPanel } from "./DiskPanel";
import { ModelDrawer } from "./Drawer";
import { Library } from "./Library";
import { ModalityBadges } from "./parts";
import { ModelsProvider, useModels } from "./store";

/** The Models screen: what is loaded, the library, disk space, and the matching commands. Loaded as
 *  its own chunk (screens/Models.tsx), with its own stylesheet. */
export function ModelsScreen(): JSX.Element {
  return (
    <ModelsProvider>
      <div class="screen screen--models">
        <ScreenHeader
          route="models"
          lede={
            <>
              Every model file on this machine, from Cerebro's, LM Studio's and Ollama's folders. Load one with its tuned settings in one click, or open it to change any{" "}
              <Term k="backend">engine</Term> setting.
            </>
          }
        />
        <LoadedStrip />
        <Section id="library" title="Library" description="Search, filter and open any model. Load uses the settings its last tune measured; never-tuned models start from explicit, conservative settings.">
          <Library />
        </Section>
        <Section id="disk" title="Disk space" description="The same file kept by more than one app, and how much a safe deduplicate would free.">
          <DiskPanel />
        </Section>
        <Section id="cli" title="From the command line" description="The same library and the same tuned settings, from a terminal.">
          <CodeBlock lines={["cerebro models", "cerebro tune qwen3-8b-q5km", "cerebro serve qwen3-8b-q5km", "cerebro chat"]} />
        </Section>
        <ModelDrawer />
      </div>
    </ModelsProvider>
  );
}

const SLOT_LABELS: Readonly<Record<EngineSlot, string>> = { llm: "Chat model", stt: "Speech to text", tts: "Text to speech", image: "Image" };

function LoadedStrip(): JSX.Element {
  const store = useModels();
  const items = () =>
    store
      .loaded()
      .map((l) => ({ l, m: store.byId().get(l.id) }))
      .filter((x) => x.m !== undefined);
  return (
    <section class="loaded" aria-label="Loaded now">
      <span class="loaded__label">Loaded now</span>
      <Show when={items().length > 0} fallback={<span class="loaded__none">{store.library.state === "ready" ? "Nothing loaded" : "Checking…"}</span>}>
        <ul class="loaded__list">
          <For each={items()}>
            {({ l, m }) => (
              <li class="loaded-item" data-slot={l.slot}>
                <span class="status-dot" data-state="running" aria-hidden="true" />
                <button type="button" class="loaded-item__name" data-press onClick={() => m && store.open(m.id)}>
                  {m?.name} <span class="mono loaded-item__quant">{m?.quant}</span>
                </button>
                <span class="loaded-item__slot">
                  {SLOT_LABELS[l.slot]}
                  {l.slot === "llm" ? (l.tuned ? (m?.tunedParams ? " · tuned" : " · starting settings") : " · custom") : ""}
                </span>
                <Show when={m}>{(mm) => <ModalityBadges modalities={mm().modalities} compact />}</Show>
                <button type="button" class="btn btn--icon btn--xs" aria-label={`Unload ${m?.name ?? "model"}`} data-press onClick={() => void store.unload(l.id)}>
                  <Icon name="close" size={14} />
                </button>
              </li>
            )}
          </For>
        </ul>
      </Show>
    </section>
  );
}
