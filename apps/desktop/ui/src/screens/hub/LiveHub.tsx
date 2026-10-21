import { createEffect, createMemo, createResource, createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { FamilyTile } from "../../components/FamilyTile";
import { ABILITY_LABELS, abilitiesOf, fitBadge, HUB_CATEGORIES, HUB_PICKS, type HubPick } from "../../lib/hub-picks";
import { Icon } from "../../components/icons";
import { LicenceNote } from "../../components/LicenceNote";
import { memoryNeeds } from "../../components/QuantTable";
import { ScreenHeader } from "../../components/Screen";
import { licenceForModel } from "../../lib/model-licences";
import { estimateFor, recommend, type MemoryModel } from "../../lib/memory-estimate";
import { contextLabel } from "../../onboarding/autoconfig";
import { useService } from "../../service/context";
import type { HubDownloadEvent, HubFile, HubFileMeta, HubRepo } from "../../service/types";
import { hrefFor } from "../../lib/routes";
import { useAppState } from "../../state";

const GIB = 1024 ** 3;
const gib = (b: number) => (b / GIB).toFixed(b >= 10 * GIB ? 1 : 2);

/** The Hub sizes every file with a full-precision KV cache. */
const KV_TYPE = "f16" as const;

/** A hub file's architecture, for the memory estimate. */
const memoryModel = (meta: HubFileMeta): MemoryModel => ({ architecture: meta.experts ? "moe" : "dense", layers: meta.layers, kvHeads: meta.kvHeads, headDim: meta.headDim, embedding: meta.embedding, contextMax: meta.contextMax, attentionLayers: meta.attentionLayers });

/** How a file of `bytes` fits this machine: all in graphics memory, graphics memory + system
 *  memory (experts or layers on the processor), system memory only, or not at all. */
function fitOf(bytes: number, vram: number, ram: number): { label: string; tone: "good" | "ok" | "slow" | "no" } {
  const need = bytes * 1.1;
  if (vram > 0 && need <= vram * 0.92) return { label: "Fits in graphics memory", tone: "good" };
  if (need <= vram * 0.92 + ram * 0.7) return { label: vram > 0 ? "Graphics memory + system memory" : "Fits in system memory", tone: vram > 0 ? "ok" : "slow" };
  if (need <= ram * 0.9 + vram) return { label: "Tight: needs most of system memory", tone: "slow" };
  return { label: "Too large for this machine", tone: "no" };
}

/**
 * Model Hub on the desktop: live search of Hugging Face for GGUF models, every file of a repository
 * by quant with its real size and how it fits this machine, and downloads with progress into
 * Cerebro's model folder (the Models screen picks them up).
 */
export function LiveHub(): JSX.Element {
  const service = useService();
  const app = useAppState();
  // Nothing is searched until the user asks: the front page is the recommendations below.
  const [query, setQuery] = createSignal("");
  const [submitted, setSubmitted] = createSignal("");
  const [repo, setRepo] = createSignal<string | null>(null);
  const [hw] = createResource(() => service.detectHardware().catch(() => undefined));
  const offline = () => app.settings()?.offlineMode === true || app.network() !== "online";
  const downloadsOff = () => app.settings()?.allowModelDownloads === false;
  const blocked = () => offline() || downloadsOff();
  const search = () => (blocked() || !submitted() ? undefined : submitted());
  const runSearch = (q: string) => {
    if (blocked()) return;
    setQuery(q);
    setRepo(null);
    setSubmitted(q.trim());
  };
  const vramGiB = () => hw()?.gpu?.vramTotalGiB ?? 0;
  const ramGiB = () => hw()?.ramTotalGiB ?? 0;
  const order = { good: 0, ok: 1, slow: 2, no: 3 } as const;
  const picks = createMemo(() =>
    HUB_PICKS.map((p) => ({ p, fit: fitBadge(p.sizeGiB, vramGiB(), ramGiB()) })).sort((a, b) => order[a.fit.tone] - order[b.fit.tone]),
  );
  const filesFor = () => (blocked() ? undefined : repo());
  const [results] = createResource(search, (q) => service.hubSearch!(q));
  const [files, { refetch }] = createResource(filesFor, (r) => service.hubFiles!(r));
  const selected = () => (results() ?? []).find((r) => r.id === repo()) ?? null;
  const [progress, setProgress] = createSignal<Record<string, { done: number; total: number; bps: number; error?: string; path?: string }>>({});
  const vram = createMemo(() => (hw()?.gpu?.vramTotalGiB ?? 0) * GIB);
  const ram = createMemo(() => (hw()?.ramTotalGiB ?? 0) * GIB);
  const [ctx, setCtx] = createSignal(8192);
  const ctxLabel = () => contextLabel(ctx());
  const [metas, setMetas] = createSignal<Record<string, HubFileMeta | null>>({});

  // Read each file's GGUF header (metadata only) once its repository is listed, so the memory line
  // and the recommendation use the real architecture, not a guess.
  createEffect(() => {
    const r = repo();
    const list = files.error ? undefined : files();
    const api = service.hubFileMeta;
    let alive = true;
    onCleanup(() => {
      alive = false;
    });
    if (!r || !list || !api) {
      setMetas({});
      return;
    }
    setMetas({});
    void (async () => {
      for (const f of list) {
        try {
          const meta = await api(r, f.path);
          if (alive) setMetas((p) => ({ ...p, [f.path]: meta }));
        } catch {
          if (alive) setMetas((p) => ({ ...p, [f.path]: null }));
        }
      }
    })();
  });

  const estimateOf = (f: HubFile) => {
    const meta = metas()[f.path];
    const h = hw();
    return meta && h ? estimateFor(memoryModel(meta), { label: f.quant, sizeGiB: f.size / GIB }, ctx(), KV_TYPE, h) : null;
  };

  const best = createMemo(() => {
    const h = hw();
    const list = files.error ? undefined : files();
    if (!h || !list) return [];
    const m = metas();
    const candidates = list.flatMap((f) => {
      const meta = m[f.path];
      return meta ? [{ model: memoryModel(meta), quant: { label: f.quant, sizeGiB: f.size / GIB }, file: f }] : [];
    });
    return recommend(candidates, h, ctx(), KV_TYPE, 3);
  });

  const start = (r: string, f: HubFile) => {
    if (blocked()) return;
    const key = `${r}/${f.path}`;
    setProgress((p) => ({ ...p, [key]: { done: 0, total: f.size, bps: 0 } }));
    service
      .hubDownload!(r, f, (e: HubDownloadEvent) => {
        setProgress((p) => {
          const cur = p[key] ?? { done: 0, total: f.size, bps: 0 };
          if (e.type === "progress") return { ...p, [key]: { ...cur, done: e.done, total: e.total, bps: e.bytesPerSec } };
          if (e.type === "done") return { ...p, [key]: { ...cur, done: cur.total, path: e.path } };
          return { ...p, [key]: { ...cur, error: e.message } };
        });
      })
      .then(() => refetch())
      .catch((err: unknown) => setProgress((p) => ({ ...p, [key]: { ...(p[key] ?? { done: 0, total: f.size, bps: 0 }), error: String((err as Error)?.message ?? err) } })));
  };

  return (
    <div class="screen livehub">
      <ScreenHeader route="hub" />
      <Show when={blocked()}>
        <div class="notice reveal" role="status">
          <Icon name="offline" size={18} />
          <p>
            <Show when={downloadsOff()} fallback={<>No network right now. Model Hub search and downloads need a connection.</>}>
              Model downloads are switched off. <a class="text-link" href={hrefFor("settings")}>Change this in Settings</a>.
            </Show>
          </p>
        </div>
      </Show>
      <form
        class="livehub__search"
        onSubmit={(e) => {
          e.preventDefault();
          if (blocked()) return;
          setRepo(null);
          setSubmitted(query().trim());
        }}
      >
        <Icon name="search" size={18} />
        <input class="input" type="search" value={query()} onInput={(e) => setQuery(e.currentTarget.value)} placeholder="Search Hugging Face: a model, a publisher, a quant…" aria-label="Search Hugging Face" maxLength={120} disabled={blocked()} />
        <button class="btn btn--primary" type="submit" disabled={blocked()}>
          Search
        </button>
      </form>
      <div class="livehub__chips" role="group" aria-label="Browse by task">
        <For each={HUB_CATEGORIES}>
          {(c) => (
            <button type="button" class="chip livehub__chip" aria-pressed={submitted() === c.query} disabled={blocked()} onClick={() => runSearch(c.query)}>
              {c.label}
            </button>
          )}
        </For>
        <Show when={submitted()}>
          <button type="button" class="btn btn--tertiary btn--sm" onClick={() => runSearch("")}>
            Back to recommendations
          </button>
        </Show>
      </div>
      <Show when={!submitted()}>
        <section class="livehub__picks" aria-labelledby="picks-title">
          <h2 id="picks-title">Recommended for this computer</h2>
          <ul class="livehub__pick-grid">
            <For each={picks()}>
              {(x) => <PickCard pick={x.p} fit={x.fit} disabled={blocked()} onOpen={() => runSearch(x.p.query)} />}
            </For>
          </ul>
        </section>
      </Show>
      <p class="livehub__note">
        Live from huggingface.co. Sizes are the files' real sizes; fit is judged against this machine: {hw() ? `${hw()!.gpu ? `${hw()!.gpu!.name}, ${hw()!.gpu!.vramTotalGiB.toFixed(0)} GiB graphics memory · ` : ""}${hw()!.ramTotalGiB.toFixed(0)} GiB system memory` : "reading…"}
      </p>
      <Show when={submitted()}>
      <div class="livehub__panes">
        <section class="livehub__results" aria-label="Repositories">
          <Show when={!results.loading} fallback={<p class="livehub__muted">Searching…</p>}>
            <Show when={!results.error} fallback={<p class="notice">{String(results.error?.message ?? results.error)}</p>}>
              <Show when={(results() ?? []).length > 0} fallback={<p class="livehub__muted">No GGUF repositories match.</p>}>
                <ul>
                  <For each={results()}>
                    {(r: HubRepo) => (
                      <li>
                        <button type="button" class="livehub__repo" aria-pressed={repo() === r.id} onClick={() => setRepo(r.id)}>
                          <FamilyTile name={r.id} size={34} />
                          <span class="livehub__repo-body">
                          <span class="livehub__repo-name">{r.id}</span>
                          <span class="livehub__muted">
                            {compactCount(r.downloads)} downloads · {compactCount(r.likes)} likes{r.updated ? ` · ${r.updated.slice(0, 10)}` : ""}
                          </span>
                          <span class="livehub__badges">
                            <For each={abilitiesOf(r.id, r.tags)}>{(a) => <span class="livehub__badge">{ABILITY_LABELS[a]}</span>}</For>
                          </span>
                          <LicenceNote licence={licenceForModel(r.id, null, r.license)} compact />
                          </span>
                        </button>
                      </li>
                    )}
                  </For>
                </ul>
              </Show>
            </Show>
          </Show>
        </section>
        <section class="livehub__files" aria-label="Files">
          <Show when={repo()} fallback={<p class="livehub__muted">Pick a repository to see its files.</p>}>
            <h2 class="livehub__title">{repo()}</h2>
            <Show when={selected()}>
              {(r) => <LicenceNote licence={licenceForModel(r().id, null, r().license)} class="livehub__licence" />}
            </Show>
            <div class="livehub__controls">
              <label class="livehub__ctx" for="livehub-context">
                <span>Context</span>
                <input
                  id="livehub-context"
                  class="hub-slider"
                  type="range"
                  min={2048}
                  max={131072}
                  step={2048}
                  value={ctx()}
                  data-ctx-slider
                  aria-valuetext={`${ctxLabel()} tokens`}
                  onInput={(e) => setCtx(e.currentTarget.valueAsNumber)}
                />
                <span class="mono">{ctxLabel()}</span>
              </label>
              <Show when={best().length > 0}>
                <ol class="livehub__best" aria-label="Best for this computer">
                  <For each={best()}>
                    {(p) => (
                      <li data-best={p.file.quant}>
                        <span class="mono">{p.file.quant}</span> — {p.why} <span class="mono" data-estimate={p.file.path}>{memoryNeeds(p.estimate, ctxLabel())}</span>
                      </li>
                    )}
                  </For>
                </ol>
              </Show>
            </div>
            <Show when={!files.loading} fallback={<p class="livehub__muted">Reading the file list…</p>}>
              <Show when={!files.error} fallback={<p class="notice">{String(files.error?.message ?? files.error)}</p>}>
                <table class="livehub__table">
                  <thead>
                    <tr>
                      <th>Quant</th>
                      <th>Size</th>
                      <th>Memory</th>
                      <th>Fit here</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    <For each={files()}>
                      {(f: HubFile) => {
                        const key = () => `${repo()}/${f.path}`;
                        const p = () => progress()[key()];
                        const fit = () => fitOf(f.size, vram(), ram());
                        return (
                          <tr>
                            <td>
                              <span class="livehub__quant">{f.quant}</span>
                              <span class="livehub__file">{f.path}{f.parts.length > 1 ? ` (+${f.parts.length - 1} parts)` : ""}</span>
                            </td>
                            <td class="num">{gib(f.size)} GiB</td>
                            <td class="num livehub__mem">
                              <Show when={estimateOf(f)} fallback={<span class="livehub__muted">—</span>}>
                                {(e) => <span class="mono" data-estimate={f.path}>{memoryNeeds(e(), ctxLabel())}</span>}
                              </Show>
                            </td>
                            <td>
                              <span class={`livehub__fit livehub__fit--${fit().tone}`}>{fit().label}</span>
                            </td>
                            <td class="livehub__action">
                              <Show
                                when={!f.downloaded && !p()?.path}
                                fallback={<span class="livehub__ok"><Icon name="check" size={16} /> In your library</span>}
                              >
                                <Show
                                  when={p() && !p()!.error}
                                  fallback={
                                    <>
                                      <Show when={p()?.error}>
                                        <span class="livehub__err">{p()!.error}</span>
                                      </Show>
                                      <button class="btn btn--secondary" type="button" disabled={blocked() || fit().tone === "no"} onClick={() => start(repo()!, f)}>
                                        {p()?.error ? "Resume" : "Download"}
                                      </button>
                                    </>
                                  }
                                >
                                  <div class="livehub__progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round((100 * p()!.done) / Math.max(1, p()!.total))}>
                                    <div style={{ width: `${(100 * p()!.done) / Math.max(1, p()!.total)}%` }} />
                                  </div>
                                  <span class="livehub__muted num">
                                    {gib(p()!.done)} / {gib(p()!.total)} GiB · {(p()!.bps / 1e6).toFixed(0)} MB/s
                                  </span>
                                  <button class="btn btn--tertiary" type="button" onClick={() => service.hubCancel!(repo()!, f.path)}>
                                    Stop
                                  </button>
                                </Show>
                              </Show>
                            </td>
                          </tr>
                        );
                      }}
                    </For>
                  </tbody>
                </table>
              </Show>
            </Show>
          </Show>
        </section>
      </div>
      </Show>
    </div>
  );
}

/** 12,345 as "12K", 1,234,567 as "1.2M". */
function compactCount(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}K`;
  return String(n);
}

function PickCard(props: { pick: HubPick; fit: { label: string; tone: string }; disabled: boolean; onOpen: () => void }): JSX.Element {
  return (
    <li class="livehub__pick card" data-fit={props.fit.tone}>
      <div class="livehub__pick-head">
        <FamilyTile name={props.pick.family} size={44} />
        <div>
          <h3>{props.pick.title}</h3>
          <span class="livehub__muted mono">
            {props.pick.params} · about {props.pick.sizeGiB.toFixed(1)} GiB
          </span>
        </div>
      </div>
      <p>{props.pick.blurb}</p>
      <div class="livehub__badges">
        <For each={props.pick.abilities}>{(a) => <span class="livehub__badge">{ABILITY_LABELS[a]}</span>}</For>
      </div>
      <div class="livehub__pick-foot">
        <span class={`livehub__fit livehub__fit--${props.fit.tone}`}>{props.fit.label}</span>
        <button type="button" class="btn btn--secondary btn--sm" disabled={props.disabled} onClick={() => props.onOpen()} aria-label={`See files for ${props.pick.title}`}>
          See files
        </button>
      </div>
    </li>
  );
}
