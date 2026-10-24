import { createEffect, createMemo, createResource, createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { FamilyTile } from "../../components/FamilyTile";
import { Icon } from "../../components/icons";
import { LicenceNote } from "../../components/LicenceNote";
import { memoryNeeds } from "../../components/QuantTable";
import { ScreenHeader } from "../../components/Screen";
import { VerifiedBadge } from "../../components/VerifiedBadge";
import { ABILITY_LABELS, abilitiesOf, FEATURED_IMAGES, fitBadge, HUB_CATEGORIES, type FeaturedImage } from "../../lib/hub-picks";
import { fitOf, recommendQuant, toGiB, type QuantFile, type QuantPick } from "../../lib/hub-quant";
import { pickLatest, type LatestPick } from "../../lib/hub-latest";
import { isVerified, sortRepos, verifiedPublisher, VERIFIED_NOTE } from "../../lib/hub-verified";
import { fmtAge } from "../../lib/format";
import { licenceForModel } from "../../lib/model-licences";
import { estimateFor, type MemoryModel } from "../../lib/memory-estimate";
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

/** A file of a repository, as the quant chooser sees it. */
const quantFile = (f: HubFile): QuantFile => ({ path: f.path, quant: f.quant, sizeGiB: toGiB(f.size), parts: f.parts.length });

/**
 * Model Hub on the desktop: live search of Hugging Face for GGUF models, every file of a repository
 * by quant with its real size and how it fits this machine, and downloads with progress into
 * Cerebro's model folder (the Models screen picks them up).
 *
 * Two things are picked for the user, both from what the machine really has: which publishers are
 * worth trusting (Cerebro's own checked list) and which file of a repository to download (the
 * largest one that fits, preferring Unsloth's dynamic quants). Everything else stays one click
 * away; nothing is hidden.
 */
export function LiveHub(): JSX.Element {
  const service = useService();
  const app = useAppState();
  // Nothing is searched until the user asks: the front page is the recommendations below.
  const [query, setQuery] = createSignal("");
  const [submitted, setSubmitted] = createSignal("");
  const [repo, setRepo] = createSignal<string | null>(null);
  const [verifiedOnly, setVerifiedOnly] = createSignal(false);
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
  const filesFor = () => (blocked() ? undefined : repo());
  const [results] = createResource(search, (q) => service.hubSearch!(q));
  const [files, { refetch }] = createResource(filesFor, (r) => service.hubFiles!(r));
  const selected = () => (results() ?? []).find((r) => r.id === repo()) ?? null;
  const [progress, setProgress] = createSignal<Record<string, { done: number; total: number; bps: number; error?: string; path?: string }>>({});
  const [ctx, setCtx] = createSignal(8192);
  const ctxLabel = () => contextLabel(ctx());
  const [metas, setMetas] = createSignal<Record<string, HubFileMeta | null>>({});

  /** The results as they are shown: the best match first, and at equal relevance the publishers on
   *  Cerebro's list before the rest. */
  const shown = createMemo(() => {
    const sorted = sortRepos(results() ?? [], submitted());
    return verifiedOnly() ? sorted.filter((r) => isVerified(r.id)) : sorted;
  });

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
    return meta && h ? estimateFor(memoryModel(meta), { label: f.quant, sizeGiB: toGiB(f.size) }, ctx(), KV_TYPE, h) : null;
  };

  /** The file Cerebro would use here, from the repository's own file list and this machine. */
  const rec = createMemo<QuantPick | null>(() => {
    const list = files.error ? undefined : files();
    const h = hw();
    if (!list || !h) return null;
    return recommendQuant(list.map(quantFile), h);
  });

  const fileOf = (path: string): HubFile | undefined => (files() ?? []).find((f) => f.path === path);

  /**
   * The front page: the newest Qwen dense and MoE models that run on this computer, found live on
   * Hugging Face (Unsloth's and Qwen's own GGUF uploads, newest first), so new releases appear the
   * day they are published. File lists are kept for the Download button.
   */
  const latestFiles = new Map<string, HubFile[]>();
  const [latest] = createResource(
    () => {
      if (blocked() || !service.hubLatest) return undefined;
      const h = hw();
      return h ? `${h.gpu?.vramTotalGiB ?? 0}:${h.ramTotalGiB ?? 0}` : undefined;
    },
    async () => {
      const h = hw()!;
      const [a, b] = await Promise.all([service.hubLatest!("unsloth", "Qwen").catch(() => []), service.hubLatest!("Qwen", "Qwen").catch(() => [])]);
      const repos = [...a, ...b];
      const files = async (id: string): Promise<QuantFile[]> => {
        const list = await service.hubFiles!(id);
        latestFiles.set(id, list);
        return list.map(quantFile);
      };
      const [dense, moe] = await Promise.all([pickLatest("dense", repos, h, files), pickLatest("moe", repos, h, files)]);
      return [dense, moe].filter((x): x is LatestPick => x !== null);
    },
  );

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
          <p class="livehub__muted">The newest Qwen models that run on this machine, checked against its graphics card and memory, with how fast each would write here.</p>
          <Show when={!latest.loading} fallback={<p class="livehub__muted">Finding the newest models for this computer…</p>}>
            <Show when={(latest() ?? []).length > 0} fallback={<p class="livehub__muted">{blocked() ? "Searching needs the network (Offline mode is on or downloads are switched off)." : "No model was found right now. Search above for any model."}</p>}>
              <ul class="livehub__pick-grid">
                <For each={latest()}>
                  {(x) => {
                    const key = () => `${x.repo.id}/${x.file.file.path}`;
                    const hubFile = () => latestFiles.get(x.repo.id)?.find((f) => f.path === x.file.file.path);
                    const name = x.repo.id.split("/").pop()!.replace(/-GGUF$/i, "").replace(/-/g, " ");
                    return (
                      <li class="livehub__pick card" data-fit={x.speed.swapping ? "slow" : "good"}>
                        <div class="livehub__pick-head">
                          <FamilyTile name="qwen" owner="Qwen" size={44} />
                          <div>
                            <span class="livehub__muted">{x.kind === "moe" ? "Newest MoE model" : "Newest dense model"}</span>
                            <h3>{name}</h3>
                            <span class="livehub__muted mono">{x.kind === "moe" ? `${x.parsed.totalB}B total, ${x.parsed.activeB}B active` : `${x.parsed.totalB}B`}</span>
                          </div>
                        </div>
                        <p class="livehub__pick-size">
                          <span class="livehub__muted">
                            Recommended <span class="mono">{x.file.file.quant}</span> · {x.file.file.sizeGiB.toFixed(1)} GiB · from {x.repo.author}
                          </span>
                        </p>
                        <p classList={{ "livehub__swap": x.speed.swapping }}>{x.speed.note}</p>
                        <div class="livehub__pick-foot">
                          <Show
                            when={progress()[key()]}
                            fallback={
                              <button type="button" class="btn btn--primary btn--sm" disabled={blocked() || !hubFile()} onClick={() => hubFile() && start(x.repo.id, hubFile()!)}>
                                Download
                              </button>
                            }
                          >
                            {(pr) => <span class="livehub__muted">{pr().error ? pr().error : pr().path ? "Downloaded: it is in Models" : `Downloading ${pr().total > 0 ? Math.floor((100 * pr().done) / pr().total) : 0}%`}</span>}
                          </Show>
                          <button type="button" class="btn btn--secondary btn--sm" disabled={blocked()} onClick={() => runSearch(x.repo.id.split("/").pop()!.replace(/-GGUF$/i, ""))}>
                            See all files
                          </button>
                        </div>
                      </li>
                    );
                  }}
                </For>
              </ul>
            </Show>
          </Show>
        </section>
        <Show when={service.hubPipelineFiles}>
          <section class="livehub__picks" aria-labelledby="featured-title">
            <h2 id="featured-title">Featured image models</h2>
            <ul class="livehub__pick-grid">
              <For each={FEATURED_IMAGES}>
                {(img) => {
                  const [list] = createResource(
                    () => (blocked() ? undefined : img.repo),
                    (r) => service.hubPipelineFiles!(r).catch(() => null),
                  );
                  const key = () => `${img.repo}/model_index.json`;
                  const p = () => progress()[key()];
                  return (
                    <ImageCard
                      img={img}
                      sizeGiB={list() ? list()!.total / GIB : null}
                      fit={list() ? fitBadge(list()!.total / GIB, vramGiB(), ramGiB()) : null}
                      progress={p()}
                      disabled={blocked() || !list()}
                      onDownload={() => {
                        const l = list();
                        if (l) start(img.repo, { path: "model_index.json", size: l.total, parts: l.files, quant: "FP16", downloaded: false });
                      }}
                    />
                  );
                }}
              </For>
            </ul>
          </section>
        </Show>
      </Show>
      <p class="livehub__note">
        Live from huggingface.co. Sizes are the files' real sizes; fit is judged against this machine: {hw() ? `${hw()!.gpu ? `${hw()!.gpu!.name}, ${hw()!.gpu!.vramTotalGiB.toFixed(0)} GiB graphics memory · ` : ""}${hw()!.ramTotalGiB.toFixed(0)} GiB system memory` : "reading…"}
      </p>
      <Show when={submitted()}>
        <div class="livehub__filters">
          <button type="button" class="chip livehub__chip livehub__filter" aria-pressed={verifiedOnly()} disabled={blocked()} onClick={() => setVerifiedOnly(!verifiedOnly())}>
            <Icon name="verified" size={12} />
            Verified publishers only
          </button>
          <span class="livehub__muted">{VERIFIED_NOTE}</span>
        </div>
      <div class="livehub__panes">
        <section class="livehub__results" aria-label="Repositories">
          <Show when={!results.loading} fallback={<p class="livehub__muted">Searching…</p>}>
            <Show when={!results.error} fallback={<p class="notice">{String(results.error?.message ?? results.error)}</p>}>
              <Show when={shown().length > 0} fallback={<p class="livehub__muted">{verifiedOnly() ? "No repository from a verified publisher matches this search." : "No GGUF repositories match."}</p>}>
                <ul>
                  <For each={shown()}>
                    {(r: HubRepo) => (
                      <li>
                        <button type="button" class="livehub__repo" aria-pressed={repo() === r.id} onClick={() => setRepo(r.id)}>
                          <FamilyTile name={r.id} owner={r.author} size={34} />
                          <span class="livehub__repo-body">
                          <span class="livehub__repo-name">
                            <span class="livehub__repo-id">{r.id}</span>
                            <Show when={verifiedPublisher(r.id)}>{(v) => <VerifiedBadge publisher={v().name} />}</Show>
                          </span>
                          <span class="livehub__muted">
                            {compactCount(r.downloads)} downloads · updated {fmtAge(r.updated)} · {compactCount(r.likes)} likes
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
            <div class="livehub__repo-head">
              <h2 class="livehub__title">{repo()}</h2>
              <Show when={selected() ? verifiedPublisher(selected()!.id) : null}>
                {(v) => <VerifiedBadge publisher={v().name} />}
              </Show>
            </div>
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
              <Show when={rec()}>
                {(r) => (
                  <div class="livehub__rec" data-recommended={r().file.path}>
                    <p class="livehub__rec-head">
                      <Icon name="check" size={16} />
                      <strong>Recommended for this computer</strong>
                    </p>
                    <p class="livehub__rec-why" data-rec-why={r().file.quant}>
                      {r().why}
                    </p>
                    <p class="livehub__rec-meta">
                      <span class="mono">{r().file.quant}</span>
                      <span aria-hidden="true">·</span>
                      <span>{gib(r().file.sizeGiB * GIB)} GiB</span>
                      <Show when={r().file.parts > 1}>
                        <span aria-hidden="true">·</span>
                        <span>
                          {r().file.parts} parts, one download
                        </span>
                      </Show>
                    </p>
                    <button
                      type="button"
                      class="btn btn--primary btn--sm livehub__rec-download"
                      disabled={blocked()}
                      aria-label={`Download the recommended file ${r().file.quant}`}
                      onClick={() => {
                        const f = fileOf(r().file.path);
                        if (f) start(repo()!, f);
                      }}
                    >
                      Download
                    </button>
                  </div>
                )}
              </Show>
              <Show when={!rec() && !!hw() && !files.loading}>
                <p class="livehub__muted">This repository has no file for Cerebro to pick.</p>
              </Show>
              <Show when={hw.error}>
                <p class="livehub__muted">Cerebro could not read this computer's memory, so it cannot pick a file here.</p>
              </Show>
            </div>
            <Show when={!files.loading} fallback={<p class="livehub__muted">Reading the file list…</p>}>
              <Show when={!files.error} fallback={<p class="notice">{String(files.error?.message ?? files.error)}</p>}>
                <div class="livehub__table-wrap">
                <table class="livehub__table">
                  <thead>
                    <tr>
                      <th>Quant</th>
                      <th>Size</th>
                      <th>On this computer</th>
                      <th>
                        <span class="sr-only">Download</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <For each={files()}>
                      {(f: HubFile) => {
                        const key = () => `${repo()}/${f.path}`;
                        const p = () => progress()[key()];
                        const fit = () => fitOf(toGiB(f.size), vramGiB(), ramGiB());
                        const isRec = () => rec()?.file.path === f.path;
                        return (
                          <tr classList={{ "is-recommended": isRec() }} data-recommended={isRec() ? "true" : undefined}>
                            <td>
                              <span class="livehub__quant">{f.quant}</span>
                              <span class="livehub__file" title={f.path}>
                                {f.path}
                              </span>
                              <Show when={f.parts.length > 1}>
                                <span class="livehub__file-note">{f.parts.length} parts, one download</span>
                              </Show>
                              <Show when={isRec()}>
                                <span class="livehub__rec-badge">
                                  <Icon name="check" size={12} />
                                  Recommended for this computer
                                </span>
                              </Show>
                            </td>
                            <td class="num">{gib(f.size)} GiB</td>
                            <td class="num livehub__mem">
                              <Show when={estimateOf(f)} fallback={<span class="livehub__muted">—</span>}>
                                {(e) => (
                                  // Short in the table; the whole sentence is the tooltip and what a screen reader hears.
                                  <span class="mono" data-estimate={f.path} title={memoryNeeds(e(), ctxLabel())} aria-label={memoryNeeds(e(), ctxLabel())}>
                                    {e().vramGiB.toFixed(1)} + {e().ramGiB.toFixed(1)} GiB
                                  </span>
                                )}
                              </Show>
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
                                      <button class={`btn ${isRec() ? "btn--primary" : "btn--secondary"}`} type="button" disabled={blocked() || fit().tone === "no"} onClick={() => start(repo()!, f)}>
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
                </div>
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

function ImageCard(props: { img: FeaturedImage; sizeGiB: number | null; fit: { label: string; tone: string } | null; progress: { done: number; total: number; error?: string; path?: string } | undefined; disabled: boolean; onDownload: () => void }): JSX.Element {
  const pct = () => (props.progress && props.progress.total > 0 ? Math.floor((100 * props.progress.done) / props.progress.total) : 0);
  return (
    <li class="livehub__pick card" data-fit={props.fit?.tone ?? "ok"}>
      <div class="livehub__pick-head">
        <FamilyTile name={props.img.title} owner={props.img.publisher} size={44} />
        <div>
          <h3>{props.img.title}</h3>
          <span class="livehub__muted mono">{props.img.params}</span>
        </div>
      </div>
      <p class="livehub__pick-size">
        <span class="livehub__muted">{props.sizeGiB !== null ? `16-bit · ${props.sizeGiB.toFixed(2)} GiB` : "Reading the file list…"}</span>
      </p>
      <p>{props.img.blurb}</p>
      <div class="livehub__badges">
        <span class="livehub__badge">Images</span>
      </div>
      <div class="livehub__pick-foot">
        <Show when={props.fit}>{(f) => <span class={`livehub__fit livehub__fit--${f().tone}`}>{f().label}</span>}</Show>
        <Show
          when={props.progress}
          fallback={
            <button type="button" class="btn btn--secondary btn--sm" disabled={props.disabled} onClick={() => props.onDownload()} aria-label={`Download ${props.img.title}`}>
              Download
            </button>
          }
        >
          {(p) => <span class="livehub__muted">{p().error ? p().error : p().path ? "Downloaded: open Image Studio" : `Downloading ${pct()}%`}</span>}
        </Show>
      </div>
    </li>
  );
}

