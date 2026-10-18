import { createSignal, onMount, Show, type JSX } from "solid-js";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import type { KvStoreUsage } from "../../service/types";

/** Bytes as a short size: "0 MB", "340 MB", "2.4 GB". */
export function sizeLabel(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  return `${Math.round(bytes / 1e6)} MB`;
}

/**
 * Under "Keep long chats ready": how much the saved-chat folder holds, and a way to empty it. A
 * cleared chat is only read again in full the next time it is opened; nothing else is lost.
 */
export function SavedChats(): JSX.Element {
  const service = useService();
  const [usage, setUsage] = createSignal<KvStoreUsage | null>(null);
  const [busy, setBusy] = createSignal(false);
  onMount(async () => {
    try {
      setUsage(await service.kvStoreUsage());
    } catch {
      setUsage(null);
    }
  });
  const clear = async () => {
    setBusy(true);
    try {
      const held = await service.kvStoreClear();
      setUsage({ files: 0, bytes: 0 });
      notify("Saved chats cleared", `${held.files} saved chat${held.files === 1 ? "" : "s"}, ${sizeLabel(held.bytes)} freed. Each is read in full the next time it is opened.`, "success");
    } catch (e) {
      notify("Not cleared", e instanceof Error ? e.message : "The folder could not be emptied.", "error");
    } finally {
      setBusy(false);
    }
  };
  return (
    <Show when={usage()}>
      {(u) => (
        <p class="hint setting__extra" data-testid="saved-chats">
          {u().files === 0 ? "No saved chats yet." : `${u().files} saved chat${u().files === 1 ? "" : "s"}, ${sizeLabel(u().bytes)}.`}{" "}
          <Show when={u().files > 0}>
            <button class="text-link" type="button" disabled={busy()} onClick={() => void clear()}>
              Clear saved chats
            </button>
          </Show>
        </p>
      )}
    </Show>
  );
}
