import "../styles/cluster.css";
import { createMemo, createResource, createSignal, For, Show, type JSX } from "solid-js";
import { Icon } from "../components/icons";
import { ScreenHeader, Section } from "../components/Screen";
import { hrefFor } from "../lib/routes";
import { notify } from "../lib/toast";
import { useService } from "../service/context";
import type { ClusterLink, ClusterWorker, LibraryModel } from "../service/types";

const errText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

interface Chosen {
  addr: string;
  name: string;
  code: string;
  share: number;
}

/**
 * Cluster: one model across several computers on your network (any link: Wi-Fi, Ethernet, a USB
 * network cable). One computer shares itself and shows a pairing code; another loads a model with
 * part of its layers there. Each computer needs the same model file.
 */
export function Cluster(): JSX.Element {
  const service = useService();
  const [library] = createResource(() => service.listLibrary().catch(() => [] as LibraryModel[]));
  const models = createMemo(() => (library() ?? []).filter((m) => m.kind === "text" || m.kind === "vision"));
  const [share, { mutate: setShare }] = createResource(() => service.clusterShareGet());
  const [shareModel, setShareModel] = createSignal("");
  const [useModel, setUseModel] = createSignal("");
  const [found, setFound] = createSignal<ClusterWorker[] | null>(null);
  const [looking, setLooking] = createSignal(false);
  const [chosen, setChosen] = createSignal<Chosen[]>([]);
  const [manual, setManual] = createSignal("");
  const [localShare, setLocalShare] = createSignal(0.5);
  const [busy, setBusy] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [links, setLinks] = createSignal<ClusterLink[] | null>(null);
  const pickShare = () => shareModel() || models()[0]?.id || "";
  const pickUse = () => useModel() || models()[0]?.id || "";

  const startShare = async () => {
    setBusy(true);
    try {
      setShare(await service.clusterShareStart(pickShare()));
      notify("Sharing this computer", "Give the pairing code to the computer that will use it.", "success");
    } catch (e) {
      notify("Could not share this computer", errText(e), "error");
    } finally {
      setBusy(false);
    }
  };

  const look = async () => {
    setLooking(true);
    try {
      setFound(await service.clusterDiscover());
    } catch (e) {
      notify("Could not look for computers", errText(e), "error");
    } finally {
      setLooking(false);
    }
  };

  const add = (addr: string, name: string) => {
    if (chosen().some((c) => c.addr === addr)) return;
    setChosen((l) => [...l, { addr, name, code: "", share: 0.5 }]);
  };
  const update = (addr: string, patch: Partial<Chosen>) => setChosen((l) => l.map((c) => (c.addr === addr ? { ...c, ...patch } : c)));

  const load = async () => {
    setError(null);
    setLinks(null);
    const ws = chosen();
    if (!ws.length) {
      setError("Add at least one other computer.");
      return;
    }
    if (ws.some((w) => w.code.replace(/\D/g, "").length !== 6)) {
      setError("Type each computer's 6-digit pairing code.");
      return;
    }
    setBusy(true);
    try {
      const r = await service.clusterLoad(pickUse(), localShare(), ws.map((w) => ({ addr: w.addr, code: w.code, share: w.share })));
      setLinks(r.links);
      notify("Loaded across computers", `${ws.length + 1} computers share the model. Chat uses it now.`, "success");
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div class="screen cluster">
      <ScreenHeader route="cluster" lede="One model across several computers: each one runs part of its layers. Any network link works: Wi-Fi, Ethernet, or a USB network cable between two computers." />

      <Section id="share" title="Share this computer" description="Another Cerebro on the network can then run part of a model here. Nothing listens until this is start sharing, and only a computer with the pairing code can connect.">
        <Show
          when={share()?.sharing}
          fallback={
            <div class="cluster__row">
              <label class="cluster__field">
                <span>Model to share</span>
                <select value={pickShare()} onChange={(e) => setShareModel(e.currentTarget.value)} disabled={!models().length}>
                  <For each={models()}>{(m) => <option value={m.id}>{m.name}</option>}</For>
                </select>
              </label>
              <button type="button" class="btn btn--primary" onClick={() => void startShare()} disabled={busy() || !models().length}>
                <Icon name="link" size={16} /> Start sharing
              </button>
            </div>
          }
        >
          <div class="cluster__sharing card" data-testid="cluster-sharing">
            <div>
              <span class="cluster__label">Pairing code</span>
              <span class="cluster__code mono" data-testid="pairing-code">
                {share()!.code!.slice(0, 3)} {share()!.code!.slice(3)}
              </span>
            </div>
            <div class="cluster__facts">
              <span>
                <strong>{share()!.computer}</strong> shares <strong>{share()!.model}</strong>
              </span>
              <span class="mono">{share()!.addresses.length ? share()!.addresses.map((a) => `${a}:${share()!.port}`).join(", ") : `port ${share()!.port}`}</span>
              <span class="hint">Windows may ask whether to let Cerebro use private networks: allow it. Other computers find this one by itself, or type the address above.</span>
            </div>
            <button type="button" class="btn btn--secondary" onClick={() => void service.clusterShareStop().then(setShare)}>
              Stop sharing
            </button>
          </div>
        </Show>
      </Section>

      <Section id="use" title="Use other computers" description="Load a model here with part of it on other computers that are sharing. Each one needs the same model file (download it from the Model Hub on each).">
        <div class="cluster__row">
          <button type="button" class="btn btn--secondary" onClick={() => void look()} disabled={looking()}>
            <Icon name="search" size={16} /> {looking() ? "Looking…" : "Look for computers"}
          </button>
          <form class="cluster__manual" onSubmit={(e) => {
            e.preventDefault();
            const a = manual().trim();
            if (!/^[\w.:[\]-]+:\d{2,5}$/.test(a)) return notify("Use an address and port", "For example 192.168.1.34:50080", "warning");
            add(a, a);
            setManual("");
          }}>
            <input class="mono" value={manual()} onInput={(e) => setManual(e.currentTarget.value)} placeholder="Or type an address: 192.168.1.34:50080" aria-label="Computer address" />
            <button type="submit" class="btn btn--tertiary">
              Add
            </button>
          </form>
        </div>
        <Show when={found()}>
          {(list) => (
            <Show when={list().length} fallback={<p class="hint">No sharing computer answered. Check that it shares and is on the same network, or type its address.</p>}>
              <ul class="cluster__found">
                <For each={list()}>
                  {(w) => (
                    <li>
                      <Icon name="monitor" size={18} />
                      <span>
                        <strong>{w.name}</strong> <span class="mono hint">{w.addr}</span>
                      </span>
                      <button type="button" class="btn btn--tertiary btn--sm" disabled={chosen().some((c) => c.addr === w.addr)} onClick={() => add(w.addr, w.name)}>
                        {chosen().some((c) => c.addr === w.addr) ? "Added" : "Use"}
                      </button>
                    </li>
                  )}
                </For>
              </ul>
            </Show>
          )}
        </Show>

        <Show when={chosen().length}>
          <div class="cluster__plan card">
            <label class="cluster__field">
              <span>Model</span>
              <select value={pickUse()} onChange={(e) => setUseModel(e.currentTarget.value)}>
                <For each={models()}>{(m) => <option value={m.id}>{m.name}</option>}</For>
              </select>
            </label>
            <label class="cluster__field">
              <span>
                This computer's share <strong class="mono">{Math.round(localShare() * 100)}%</strong>
              </span>
              <input type="range" min="0.1" max="0.9" step="0.05" value={localShare()} onInput={(e) => setLocalShare(Number(e.currentTarget.value))} />
            </label>
            <ul class="cluster__chosen">
              <For each={chosen()}>
                {(c) => (
                  <li>
                    <span class="cluster__who">
                      <strong>{c.name}</strong>
                      <span class="mono hint">{c.addr}</span>
                    </span>
                    <label class="cluster__field cluster__field--code">
                      <span>Pairing code</span>
                      <input class="mono" inputMode="numeric" maxLength={7} value={c.code} onInput={(e) => update(c.addr, { code: e.currentTarget.value })} placeholder="000 000" />
                    </label>
                    <label class="cluster__field">
                      <span>
                        Share <strong class="mono">{Math.round(c.share * 100)}%</strong>
                      </span>
                      <input type="range" min="0.1" max="0.9" step="0.05" value={c.share} onInput={(e) => update(c.addr, { share: Number(e.currentTarget.value) })} aria-label={`Share for ${c.name}`} />
                    </label>
                    <button type="button" class="icon-btn" aria-label={`Remove ${c.name}`} onClick={() => setChosen((l) => l.filter((x) => x.addr !== c.addr))}>
                      <Icon name="close" size={16} />
                    </button>
                  </li>
                )}
              </For>
            </ul>
            <p class="hint">Shares are how much of the model each computer runs. Give more to the computers with more free memory. Every token crosses each link once, so a wired link is fastest.</p>
            <Show when={error()}>
              <p class="cluster__error" role="alert">
                <Icon name="alert" size={14} /> {error()}
              </p>
            </Show>
            <button type="button" class="btn btn--primary" onClick={() => void load()} disabled={busy() || !models().length}>
              {busy() ? "Connecting…" : `Load across ${chosen().length + 1} computers`}
            </button>
          </div>
        </Show>

        <Show when={links()}>
          {(l) => (
            <div class="cluster__links card" data-testid="cluster-links">
              <h3>Running across computers</h3>
              <table>
                <thead>
                  <tr>
                    <th>Computer</th>
                    <th>Layers</th>
                    <th>Round trip</th>
                    <th>Link speed</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>This computer</td>
                    <td class="mono">0 – {(l()[0]?.lo ?? 1) - 1}</td>
                    <td>–</td>
                    <td>–</td>
                  </tr>
                  <For each={l()}>
                    {(x) => (
                      <tr>
                        <td class="mono">{x.addr}</td>
                        <td class="mono">
                          {x.lo} – {x.hi - 1}
                        </td>
                        <td class="mono">{x.rttMs.toFixed(2)} ms</td>
                        <td class="mono">{x.mbPerS.toFixed(0)} MB/s</td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
              <a class="btn btn--secondary" href={hrefFor("chat")}>
                Open Chat
              </a>
            </div>
          )}
        </Show>
      </Section>
    </div>
  );
}
