import { createResource, createSignal, For, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { IllustrativeFlag } from "../../components/Screen";
import { Term } from "../../components/Term";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import type { DuplicateCopy, DuplicateGroup } from "../../service/types";
import { SOURCE_LABELS, Spinner } from "./parts";

/**
 * Disk space: the same model file stored more than once across Cerebro's folder, other apps'
 * folders and folders you added (matched by SHA-256, not by name), and a safe deduplicate: the user
 * picks the copy to keep, every copy's hash is checked, then the extra copies on the same drive
 * become hard links to it. Every app still finds its file where it expects it, and Undo copies the
 * bytes back.
 */
export function DiskPanel(): JSX.Element {
  const service = useService();
  const [report, { mutate, refetch }] = createResource(() => service.scanDisk());
  // The copy to keep per group: the user's pick, else the default (Cerebro's own copy, else the first).
  const [keep, setKeep] = createSignal<ReadonlyMap<string, string>>(new Map());
  const keepOf = (g: DuplicateGroup) => keep().get(g.id) ?? defaultKeep(g);
  const reclaimable = () => (report()?.groups ?? []).filter((g) => g.state === "duplicate").reduce((sum, g) => sum + g.sizeGiB * linkable(g, keepOf(g)).length, 0);
  const open = () => (report()?.groups ?? []).filter((g) => g.state === "duplicate").length;

  // Space freed per group this session. Kept here, not in the row: replacing a group re-creates its row.
  const [freed, setFreed] = createSignal<ReadonlyMap<string, number>>(new Map());
  const replace = (g: DuplicateGroup, reclaimedGiB: number | null) => {
    const f = new Map(freed());
    if (reclaimedGiB === null) f.delete(g.id);
    else f.set(g.id, reclaimedGiB);
    setFreed(f);
    const r = report();
    if (!r) return;
    // The total follows what is stored: linking frees the extra copies, undoing stores them again.
    const before = r.groups.find((x) => x.id === g.id);
    const delta = before && before.state !== g.state ? (g.state === "linked" ? -1 : 1) * g.sizeGiB * linkable(g, before.kept ?? g.kept ?? defaultKeep(g)).length : 0;
    mutate({ ...r, totalGiB: Math.max(0, Math.round((r.totalGiB + delta) * 10) / 10), groups: r.groups.map((x) => (x.id === g.id ? g : x)) });
  };
  const rescan = () => {
    if (report.loading) return;
    void refetch();
  };

  return (
    <div class="disk card">
      <div class="disk__summary">
        <div class="disk__stat">
          <span class="disk__value mono">
            <Show when={report()} fallback="–">
              {(r) => <CountGiB gib={r().totalGiB} />}
            </Show>
          </span>
          <span class="disk__label">in model files</span>
        </div>
        <div class="disk__stat">
          <span class="disk__value mono">
            <Show when={report()} fallback="–">
              <Count value={open()} decimals={0} />
            </Show>
          </span>
          <span class="disk__label">files stored more than once</span>
        </div>
        <div class="disk__stat disk__stat--accent">
          <span class="disk__value mono">
            <Show when={report()} fallback="–">
              <CountGiB gib={reclaimable()} />
            </Show>
          </span>
          <span class="disk__label">could be freed</span>
        </div>
        <span class="disk__spacer" />
        <IllustrativeFlag />
        <button type="button" class="btn btn--tertiary btn--xs" data-press aria-busy={report.loading ? "true" : undefined} onClick={rescan}>
          <Spinner />
          <span class="btn__label">
            <Icon name="refresh" size={14} />
            Scan again
          </span>
        </button>
      </div>
      <p class="disk__how">
        Files are matched by their <Term k="sha256" /> fingerprint, not their names. Deduplicating keeps the copy you choose and turns the others on the same drive into{" "}
        <Term k="hardLink">hard links</Term>, so every app keeps working. Undo turns them back into separate files.
      </p>
      <Show when={report.state === "errored"}>
        <p class="field__error" role="alert">
          The folders could not be scanned. Nothing was changed.
        </p>
      </Show>
      <Show when={report()}>
        {(r) => (
          <Show when={r().groups.length > 0} fallback={<p class="studio-empty">No file is stored twice.</p>}>
            <ul class="dup-list">
              <For each={r().groups}>
                {(g) => <DuplicateRow group={g} keep={keepOf(g)} onKeep={(path) => setKeep(new Map(keep()).set(g.id, path))} freed={freed().get(g.id) ?? null} onDone={replace} />}
              </For>
            </ul>
          </Show>
        )}
      </Show>
    </div>
  );
}

/** Cerebro's own copy when there is one (it is the folder Cerebro manages), else the first. */
function defaultKeep(g: DuplicateGroup): string {
  return (g.copies.find((c) => c.source === "cerebro") ?? g.copies[0])?.path ?? "";
}

/** The copies a deduplicate keeping `keepPath` would replace: same drive, not the kept one. */
function linkable(g: DuplicateGroup, keepPath: string): DuplicateCopy[] {
  const keep = g.copies.find((c) => c.path === keepPath);
  return keep ? g.copies.filter((c) => c !== keep && c.volume === keep.volume) : [];
}

/** A number that counts from its previous value to the new one when it changes or scrolls into
 *  view (design/motion.ts, instant under reduced motion). Keyed on the value so each change gets a
 *  fresh element for the count-up to own. */
function Count(props: { value: number; decimals: number }): JSX.Element {
  let last = 0;
  return (
    <Show when={props.value.toFixed(props.decimals)} keyed>
      {(text) => {
        const from = last;
        last = Number(text);
        return (
          <span class="count" data-count-to={text} data-count-from={from.toFixed(props.decimals)}>
            {text}
          </span>
        );
      }}
    </Show>
  );
}

/** "17.3 GiB" (or TiB from 1,000 GiB) with the number counting. */
function CountGiB(props: { gib: number }): JSX.Element {
  const tib = () => props.gib >= 1000;
  return (
    <span class="num-unit">
      <Count value={tib() ? props.gib / 1024 : props.gib} decimals={1} />
      <span class="unit">{tib() ? "TiB" : "GiB"}</span>
    </span>
  );
}

/** The copy in a sentence, e.g. "the copy in another app's folder". */
function copyName(source: DuplicateCopy["source"]): string {
  if (source === "cerebro") return "Cerebro's copy";
  if (source === "other") return "the copy in your folder";
  return "the copy in another app's folder";
}

function DuplicateRow(props: {
  group: DuplicateGroup;
  keep: string;
  onKeep: (path: string) => void;
  freed: number | null;
  onDone: (g: DuplicateGroup, reclaimedGiB: number | null) => void;
}): JSX.Element {
  const service = useService();
  const [confirming, setConfirming] = createSignal<"link" | "undo" | null>(null);
  const [busy, setBusy] = createSignal(false);
  const g = () => props.group;
  const kept = () => (g().state === "linked" ? (g().kept ?? props.keep) : props.keep);
  const keptCopy = () => g().copies.find((c) => c.path === kept());
  const n = () => linkable(g(), kept()).length;
  const name = `keep-${g().id}`;

  // Busy buttons stay focusable for screen readers, so every handler refuses a second run itself.
  const run = async () => {
    if (busy()) return;
    setBusy(true);
    try {
      const r = await service.deduplicate(g().id, props.keep);
      props.onDone(r.group, r.reclaimedGiB);
      notify(`Freed ${r.reclaimedGiB.toFixed(1)} GiB`, r.skipped > 0 ? `${r.skipped} copy on another drive stays separate.` : "Every app still finds its file.", "success");
    } catch {
      notify("Nothing was deduplicated", "The files are unchanged. Scan again and retry.", "error");
    } finally {
      setBusy(false);
      setConfirming(null);
    }
  };
  const undo = async () => {
    if (busy()) return;
    setBusy(true);
    try {
      const back = await service.unlink(g().id);
      props.onDone(back, null);
      notify("Copies restored", "Each app has its own file again.", "success");
    } catch {
      notify("Nothing was restored", "The links are unchanged. Check the free space and retry.", "error");
    } finally {
      setBusy(false);
      setConfirming(null);
    }
  };

  return (
    <li class="dup" classList={{ "is-linked": g().state === "linked" }} data-group={g().id}>
      <div class="dup__head">
        <span class="dup__file mono">{g().file}</span>
        <span class="dup__size mono">
          {g().sizeGiB.toFixed(1)}
          <span class="unit">GiB</span>
          <span class="muted"> × {g().copies.length}</span>
        </span>
        <span class="dup__spacer" />
        <Show
          when={confirming()}
          fallback={
            <Show
              when={g().state === "duplicate"}
              fallback={
                <span class="dup__linked">
                  <span class="chip dup__done" data-tone="on">
                    <Icon name="link" size={12} />
                    Linked{props.freed !== null ? ` · ${props.freed.toFixed(1)} GiB freed` : ""}
                  </span>
                  <button type="button" class="btn btn--tertiary btn--xs" data-press onClick={() => setConfirming("undo")}>
                    <Icon name="undo" size={14} />
                    Undo
                  </button>
                </span>
              }
            >
              <button type="button" class="btn btn--secondary btn--xs" data-press disabled={n() === 0} onClick={() => setConfirming("link")}>
                <Icon name="link" size={14} />
                Deduplicate
              </button>
            </Show>
          }
        >
          {(mode) => (
            <span class="dup__confirm" role="group" aria-label={mode() === "link" ? "Confirm deduplicate" : "Confirm undo"}>
              <span class="dup__confirm-text">
                {mode() === "link"
                  ? `Link ${n()} ${n() === 1 ? "copy" : "copies"} to ${copyName(keptCopy()?.source ?? "other")}?`
                  : `Copy the file back? Needs ${(g().sizeGiB * n()).toFixed(1)} GiB free on ${keptCopy()?.volume ?? "the drive"}.`}
              </span>
              <button
                type="button"
                class="btn btn--secondary btn--xs"
                data-press
                aria-busy={busy() ? "true" : undefined}
                onClick={() => void (mode() === "link" ? run() : undo())}
                ref={(el) => requestAnimationFrame(() => el.focus())}
              >
                <Spinner />
                <span class="btn__label">{mode() === "link" ? "Link" : "Restore copies"}</span>
              </button>
              <button type="button" class="btn btn--tertiary btn--xs" data-press onClick={() => setConfirming(null)}>
                Cancel
              </button>
            </span>
          )}
        </Show>
      </div>
      <fieldset class="dup__copies" disabled={g().state === "linked" || busy()}>
        <legend class="sr-only">{`Copy of ${g().file} to keep`}</legend>
        <For each={g().copies}>
          {(c) => {
            const isKept = () => c.path === kept();
            const role = () => {
              if (isKept()) return "Kept";
              if (c.volume !== keptCopy()?.volume) return `Drive ${c.volume} stays a copy`;
              return g().state === "linked" ? "Hard link" : "Becomes a link";
            };
            return (
              <label class="dup-copy" data-source={c.source} classList={{ "is-kept": isKept() }}>
                <input type="radio" class="dup-copy__radio" name={name} value={c.path} checked={isKept()} onChange={() => props.onKeep(c.path)} aria-label={`Keep this copy: ${SOURCE_LABELS[c.source]} — ${c.path}`} />
                <span class="dup-copy__source">{SOURCE_LABELS[c.source]}</span>
                <span class="dup-copy__path mono" title={c.path}>
                  {c.path}
                </span>
                <span class="dup-copy__role">{role()}</span>
              </label>
            );
          }}
        </For>
      </fieldset>
    </li>
  );
}
