import "../../styles/mcpserver.css";
import { createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Section } from "../../components/Screen";
import { SettingSwitch } from "../../components/SettingSwitch";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import type { LocalApiView } from "../../service/types";

const CLIENTS = ["OpenAI SDK", "Anthropic SDK", "curl", "Other"] as const;
type Client = (typeof CLIENTS)[number];

const ROTATE_CONFIRM_MS = 4000;

/** What a client needs, for the chosen client. The key appears only when `key` is given. */
export function apiSnippet(client: Client, base: string, key: string): string {
  const root = base.replace(/\/v1$/, "");
  switch (client) {
    case "OpenAI SDK":
      return [
        "from openai import OpenAI",
        `client = OpenAI(base_url="${base}", api_key="${key}")`,
        'r = client.chat.completions.create(model="local", messages=[{"role": "user", "content": "Hello"}])',
        "print(r.choices[0].message.content)",
      ].join("\n");
    case "Anthropic SDK":
      return [
        "import anthropic",
        `client = anthropic.Anthropic(base_url="${root}", api_key="${key}")`,
        'm = client.messages.create(model="local", max_tokens=1024, messages=[{"role": "user", "content": "Hello"}])',
        "print(m.content[0].text)",
      ].join("\n");
    case "curl":
      return `curl ${base}/chat/completions -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" -d "{\\"model\\":\\"local\\",\\"messages\\":[{\\"role\\":\\"user\\",\\"content\\":\\"Hello\\"}]}"`;
    case "Other":
      return `Base URL: ${base}\nKey: ${key}\nFormats: OpenAI chat completions, Anthropic messages`;
  }
}

/**
 * Local API (Settings, id="local-api"): the loaded model behind the OpenAI and Anthropic request
 * formats on one port (src-tauri/src/local_api.rs). Off by default; each protection is a switch.
 */
export function LocalApiAccess(): JSX.Element {
  const service = useService();
  const errMessage = (e: unknown): string => (e instanceof Error ? e.message : String(e));
  const [view, setView] = createSignal<LocalApiView | null>(null);
  const [revealed, setRevealed] = createSignal(false);
  const [confirming, setConfirming] = createSignal(false);
  const [client, setClient] = createSignal<Client>("OpenAI SDK");

  // Polled 3 times a second (never overlapping, skipped while hidden): the request count is live.
  let pollFailed = false;
  let inFlight = false;
  const load = async () => {
    if (inFlight || document.visibilityState === "hidden") return;
    inFlight = true;
    try {
      setView(await service.localApiGet());
      pollFailed = false;
    } catch (e) {
      if (!pollFailed) notify("Could not read the local API", errMessage(e), "error");
      pollFailed = true;
    } finally {
      inFlight = false;
    }
  };
  void load();
  const poll = setInterval(() => void load(), 333);
  let confirmTimer: ReturnType<typeof setTimeout> | undefined;
  onCleanup(() => {
    clearInterval(poll);
    clearTimeout(confirmTimer);
  });

  const state = (): "on" | "off" | "error" => {
    const v = view();
    if (!v) return "off";
    if (v.error) return "error";
    return v.enabled && v.running ? "on" : "off";
  };
  const statusText = (): string => {
    const v = view();
    if (!v) return "Checking…";
    if (v.error) return v.error;
    if (!v.enabled) return "Off";
    if (!v.running || !v.url) return "Starting…";
    return `On at ${v.url}${v.requests ? ` · ${v.requests} request${v.requests === 1 ? "" : "s"}` : ""}`;
  };
  const masked = (): string => {
    const k = view()?.key ?? "";
    return k ? `${k.slice(0, 11)}${"•".repeat(12)}` : "";
  };
  const keyFor = (): string => (view()?.requireKey ? (view()?.key ?? "") : "not-needed");
  const snippet = (): string => {
    const v = view();
    return v?.url ? apiSnippet(client(), v.url, keyFor()) : "";
  };

  const copy = async (value: string, what: string) => {
    try {
      await navigator.clipboard.writeText(value);
      notify("Copied", what, "success");
    } catch {
      notify("Could not copy", "Read it off the screen and type it instead.", "warning");
    }
  };

  const toggle = async (on: boolean) => {
    try {
      setView(await service.localApiEnable(on));
      notify(on ? "Local API is on" : "Local API is off", on ? "Apps can now use the loaded model." : "No app can use it until you turn it back on.", "success");
    } catch (e) {
      notify("Could not change the local API", errMessage(e), "error");
    }
  };

  const setOption = async (patch: Partial<Pick<LocalApiView, "requireKey" | "allowLan" | "allowModelSwitch">>) => {
    const v = view();
    if (!v) return;
    const next = { requireKey: v.requireKey, allowLan: v.allowLan, allowModelSwitch: v.allowModelSwitch, ...patch };
    try {
      setView(await service.localApiOptions(next.requireKey, next.allowLan, next.allowModelSwitch));
    } catch (e) {
      notify("Could not change the setting", errMessage(e), "error");
    }
  };

  const rotate = async () => {
    if (!confirming()) {
      setConfirming(true);
      confirmTimer = setTimeout(() => setConfirming(false), ROTATE_CONFIRM_MS);
      return;
    }
    setConfirming(false);
    try {
      setView(await service.localApiRotateKey());
      setRevealed(false);
      notify("New key generated", "Apps using the old key must be given the new one.", "success");
    } catch (e) {
      notify("Could not replace the key", errMessage(e), "error");
    }
  };

  return (
    <Section
      id="local-api"
      title="Local API"
      description="The loaded model behind the OpenAI and Anthropic request formats, so apps and scripts written for those use it unchanged. Streaming, tool calls and reasoning are passed through."
    >
      <div class="mcpsrv" data-testid="local-api">
        <div class="mcpsrv__status-row">
          <p class="mcpsrv__status" data-state={state()} data-testid="local-api-status">
            <span class="mcpsrv__dot" aria-hidden="true" />
            {statusText()}
          </p>
        </div>

        <SettingSwitch id="local-api-enabled" label="Local API" note="Off until you turn it on. Requests run one at a time on the loaded model." checked={view()?.enabled ?? false} onChange={(on) => void toggle(on)} />

        <Show when={view()?.enabled}>
          <div class="mcpsrv__card">
            <h3 class="mcpsrv__title">Protections</h3>
            <SettingSwitch id="local-api-key" label="Require the key" note="Every request must carry the key below (Authorization: Bearer, or x-api-key)." checked={view()?.requireKey ?? true} onChange={(on) => void setOption({ requireKey: on })} />
            <SettingSwitch
              id="local-api-lan"
              label="Allow the local network"
              note="Other devices on your home or office network can connect too. Never the internet."
              checked={view()?.allowLan ?? false}
              onChange={(on) => void setOption({ allowLan: on })}
            />
            <SettingSwitch
              id="local-api-switch"
              label="Let apps load models"
              note="A request naming another model in your library loads it, replacing the one loaded now."
              checked={view()?.allowModelSwitch ?? true}
              onChange={(on) => void setOption({ allowModelSwitch: on })}
            />
            <Show when={view()?.allowLan && !view()?.requireKey}>
              <p class="hint mcpsrv__warn" role="alert">
                Anyone on your network can use this computer's model without a key.
              </p>
            </Show>
            <Show when={(view()?.lanUrls.length ?? 0) > 0}>
              <p class="hint">
                On the network: <span class="mono">{view()?.lanUrls.join(", ")}</span>
              </p>
            </Show>
          </div>

          <Show when={view()?.requireKey}>
            <div class="mcpsrv__card">
              <h3 class="mcpsrv__title">API key</h3>
              <div class="mcpsrv__token-row">
                <code class="mono mcpsrv__token" data-testid="local-api-key-value">
                  {revealed() ? (view()?.key ?? "") : masked()}
                </code>
                <div class="mcpsrv__actions">
                  <button class="btn btn--tertiary btn--sm" type="button" onClick={() => setRevealed((r) => !r)}>
                    {revealed() ? "Hide" : "Show"}
                  </button>
                  <button class="btn btn--tertiary btn--sm" type="button" onClick={() => void copy(view()?.key ?? "", "Paste it into the app you are connecting.")}>
                    Copy
                  </button>
                  <button class="btn btn--secondary btn--sm" type="button" onClick={() => void rotate()}>
                    {confirming() ? "Click again to replace" : "New key"}
                  </button>
                </div>
              </div>
            </div>
          </Show>

          <div class="mcpsrv__card">
            <h3 class="mcpsrv__title">Connect</h3>
            <div class="mcpsrv__segmented" role="radiogroup" aria-label="Client to connect">
              <For each={CLIENTS}>
                {(name) => (
                  <button class="mcpsrv__segment" type="button" role="radio" aria-checked={client() === name} data-active={client() === name} onClick={() => setClient(name)}>
                    {name}
                  </button>
                )}
              </For>
            </div>
            <Show when={revealed() || !view()?.requireKey}>
              <pre class="mcpsrv__snippet" data-testid="local-api-snippet">
                <code class="mono">{snippet()}</code>
              </pre>
            </Show>
            <div class="mcpsrv__actions">
              <button class="btn btn--tertiary btn--sm" type="button" disabled={!snippet()} onClick={() => void copy(snippet(), "Run it, or paste it into the app's settings.")}>
                Copy
              </button>
              <Show when={!revealed() && view()?.requireKey}>
                <p class="hint">The example holds the key, so it stays hidden until you press Show. Copy works without showing it.</p>
              </Show>
            </div>
          </div>
        </Show>
      </div>
    </Section>
  );
}
