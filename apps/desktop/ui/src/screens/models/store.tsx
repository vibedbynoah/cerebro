import { createContext, createMemo, createResource, createSignal, onMount, useContext, type Accessor, type JSX, type Resource } from "solid-js";
import { notify } from "../../lib/toast";
import { EMPTY_TWEAKS } from "../../lib/tweaks";
import { useService } from "../../service/context";
import type { ActivePatch, HardwareProfile, LibraryModel, LoadedModel, LoadParams } from "../../service/types";

/** "detecting" while the first answer is pending, "missing" when detection failed or answered
 *  without a CPU: nothing is then estimated for this machine. */
export type HardwareState = "detecting" | "ready" | "missing";

/**
 * State the Models screen shares between the library, the drawer and the loaded strip: the library
 * itself, this machine's hardware (for fit estimates), what is loaded, what is loading, and the
 * tweaks in effect per file (Tweak Studio writes them, the load panel and the Load button read
 * them, so both always agree on what a load will use).
 */
export interface ModelsStore {
  library: Resource<LibraryModel[]>;
  refetchLibrary: () => void;
  byId: Accessor<ReadonlyMap<string, LibraryModel>>;
  /** This machine, or null when it was not (yet) detected. Never a stand-in machine. */
  hw: Accessor<HardwareProfile | null>;
  hwState: Accessor<HardwareState>;
  redetect: () => void;
  loaded: Accessor<readonly LoadedModel[]>;
  isLoaded: (id: string) => boolean;
  isPending: (id: string) => boolean;
  /** `params` null: the tuned (or starting) settings. Resolves true when it loaded. */
  load: (id: string, params: LoadParams | null) => Promise<boolean>;
  unload: (id: string) => Promise<void>;
  /** The model whose drawer is open, or null. */
  selected: Accessor<string | null>;
  open: (id: string | null) => void;
  /** The valid tweaks in effect for a file, once read; undefined until then. */
  patch: (sha256: string) => ActivePatch | undefined;
  /** Reads a file's tweaks from the service once and caches them. */
  ensurePatch: (sha256: string) => Promise<ActivePatch>;
  /** Tweak Studio's valid state, applied here at once (persisting is Tweak Studio's job). */
  putPatch: (sha256: string, patch: ActivePatch) => void;
  /** Re-reads a file's tweaks from the service, e.g. after a save failed. */
  reloadPatch: (sha256: string) => void;
}

const Ctx = createContext<ModelsStore>();

export function ModelsProvider(props: { children: JSX.Element }): JSX.Element {
  const service = useService();
  const [library, { refetch }] = createResource(() => service.listLibrary());
  // A failed detection is null, never a fixture machine: fit estimates, starting settings and the
  // thread limit must describe this computer or say they cannot.
  const [hardware, { refetch: refetchHardware }] = createResource(() => service.detectHardware().catch(() => null));
  const [loaded, setLoaded] = createSignal<readonly LoadedModel[]>([]);
  const [pending, setPending] = createSignal<ReadonlySet<string>>(new Set());
  const [selected, setSelected] = createSignal<string | null>(null);
  const [patches, setPatches] = createSignal<ReadonlyMap<string, ActivePatch>>(new Map());
  const reading = new Map<string, Promise<ActivePatch>>();

  onMount(() => {
    service
      .listLoaded()
      .then(setLoaded)
      .catch(() => setLoaded([]));
  });

  const byId = createMemo<ReadonlyMap<string, LibraryModel>>(() => new Map((library.state === "ready" ? library() : []).map((m) => [m.id, m])));
  const hw = createMemo<HardwareProfile | null>(() => {
    const h = hardware.state === "ready" ? hardware() : null;
    return h && typeof h === "object" && h.cpu && Number.isInteger(h.cpu.threads) && h.cpu.threads > 0 ? h : null;
  });
  const hwState = (): HardwareState => (hardware.loading ? "detecting" : hw() ? "ready" : "missing");
  const mark = (id: string, on: boolean) =>
    setPending((prev) => {
      const next = new Set(prev);
      if (on) next.add(id);
      else next.delete(id);
      return next;
    });
  const putPatch = (sha: string, p: ActivePatch) => setPatches((prev) => new Map(prev).set(sha, p));

  const store: ModelsStore = {
    library,
    refetchLibrary: () => void refetch(),
    byId,
    hw,
    hwState,
    redetect: () => {
      if (!hardware.loading) void refetchHardware();
    },
    loaded,
    isLoaded: (id) => loaded().some((l) => l.id === id),
    isPending: (id) => pending().has(id),
    async load(id, params) {
      const m = byId().get(id);
      if (!m || pending().has(id)) return false;
      mark(id, true);
      try {
        setLoaded(await service.loadModel(id, params));
        const how = m.kind !== "text" && m.kind !== "vision" ? "in its own engine" : params ? "with your settings" : m.tunedParams ? "with its tuned settings" : "with starting settings";
        notify(`${m.name} loaded`, `${m.quant}, ${how}.`, "success");
        return true;
      } catch {
        notify(`${m.name} did not load`, "Nothing changed. Check the load settings and try again.", "error");
        return false;
      } finally {
        mark(id, false);
      }
    },
    async unload(id) {
      const m = byId().get(id);
      try {
        setLoaded(await service.unloadModel(id));
        notify(`${m?.name ?? "Model"} unloaded`);
      } catch {
        notify("That model could not be unloaded", "It is still running. Try again.", "error");
      }
    },
    selected,
    open: setSelected,
    patch: (sha) => patches().get(sha),
    ensurePatch(sha) {
      const have = patches().get(sha);
      if (have) return Promise.resolve(have);
      let p = reading.get(sha);
      if (!p) {
        p = service
          .getActivePatch(sha)
          .then((got) => {
            // A Tweak Studio edit made while this was in flight is newer; keep it.
            if (!patches().has(sha)) putPatch(sha, got);
            return patches().get(sha) ?? got;
          })
          // Unreadable: show no tweaks, and do not cache that guess.
          .catch(() => ({ tweaks: { ...EMPTY_TWEAKS, stopTokens: [] }, fixes: [] }))
          .finally(() => reading.delete(sha));
        reading.set(sha, p);
      }
      return p;
    },
    putPatch,
    reloadPatch(sha) {
      service
        .getActivePatch(sha)
        .then((got) => putPatch(sha, got))
        .catch(() => undefined);
    },
  };
  return <Ctx.Provider value={store}>{props.children}</Ctx.Provider>;
}

export function useModels(): ModelsStore {
  const s = useContext(Ctx);
  if (!s) throw new Error("useModels() called outside <ModelsProvider>");
  return s;
}
