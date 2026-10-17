import "../../styles/mcpserver.css";
import { createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Section } from "../../components/Screen";
import { SettingSwitch } from "../../components/SettingSwitch";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import type { McpServerView } from "../../service/types";

const APPS = ["Claude Code", "VS Code", "Cursor", "DeepSeek harness", "Other"] as const;
type App = (typeof APPS)[number];

const TOOLS: Array<{ name: string; desc: string }> = [
  { name: "generate", desc: "write text with the loaded model" },
  { name: "chat", desc: "reply to a conversation" },
  { name: "list_models", desc: "models in your library" },
  { name: "load_model", desc: "load one" },
  { name: "status", desc: "speed, memory and what the engine is doing" },
];

const ROTATE_CONFIRM_MS = 4000;

/**
 * MCP server access (Settings screen, id="mcp-server"): lets other apps on this computer use local
 * models through MCP. Loopback only — nothing outside this machine can connect.
 */
export function McpServerAccess(): JSX.Element {
  const service = useService();
  const errMessage = (e: unknown): string => (e instanceof Error ? e.message : "Unknown error");

  const [view, setView] = createSignal<McpServerView | null>(null);
  const [now, setNow] = createSignal(Date.now());

  const [revealed, setRevealed] = createSignal(false);
  const [confirming, setConfirming] = createSignal(false);
  const [rotating, setRotating] = createSignal(false);
  const [app, setApp] = createSignal<App>("Claude Code");

  // Polled 3 times a second (never overlapping, skipped while hidden): connected clients and their
  // last-seen times change under the reader's eyes. A failure is reported once.
  let pollFailed = false;
  let inFlight = false;
  const load = async () => {
    setNow(Date.now());
    if (inFlight || document.visibilityState === "hidden") return;
    inFlight = true;
    try {
      setView(await service.mcpServerGet());
      pollFailed = false;
    } catch (e) {
      if (!pollFailed) notify("Could not read MCP server", errMessage(e), "error");
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

  const statusState = (): "on" | "off" | "error" => {
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
    return `On at ${v.url}`;
  };

  const masked = (): string => {
    const t = view()?.token ?? "";
    return t ? `${t.slice(0, 4)}${"•".repeat(Math.max(0, Math.min(12, t.length - 4)))}` : "";
  };

  const snippet = (): string => {
    const v = view();
    if (!v?.url || !v.token) return "";
    const auth = `Authorization: Bearer ${v.token}`;
    switch (app()) {
      case "Claude Code":
        return `claude mcp add --transport http cerebro ${v.url} --header "${auth}"`;
      case "VS Code":
        return JSON.stringify(
          { servers: { cerebro: { type: "http", url: v.url, headers: { Authorization: `Bearer ${v.token}` } } } },
          null,
          2,
        );
      case "Cursor":
        return JSON.stringify(
          { mcpServers: { cerebro: { url: v.url, headers: { Authorization: `Bearer ${v.token}` } } } },
          null,
          2,
        );
      case "DeepSeek harness":
        return [
          `- id: mcp-cerebro`,
          `  name: '@deepseek-ai/dsh-mcp-client'`,
          `  config:`,
          `    serverName: cerebro`,
          `    transport: streamable-http`,
          `    url: ${v.url}`,
          `    headers:`,
          `      Authorization: 'Bearer ${v.token}'`,
        ].join("\n");
      case "Other":
        return `URL: ${v.url}\nHeader: ${auth}\nTransport: Streamable HTTP (MCP 2025-06-18)`;
    }
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
      setView(await service.mcpServerEnable(on));
      notify(
        on ? "MCP server is on" : "MCP server is off",
        on ? "Other apps on this computer can connect with the token below." : "No app can connect until you turn it back on.",
        "success",
      );
    } catch (e) {
      notify(on ? "Could not turn the MCP server on" : "Could not turn the MCP server off", errMessage(e), "error");
    }
  };

  const rotate = async () => {
    if (!confirming()) {
      setConfirming(true);
      confirmTimer = setTimeout(() => setConfirming(false), ROTATE_CONFIRM_MS);
      return;
    }
    setConfirming(false);
    setRotating(true);
    try {
      setView(await service.mcpServerRotateToken());
      setRevealed(false);
      notify("New token generated", "Every app using the old token is signed out.", "success");
    } catch (e) {
      notify("Could not replace the token", errMessage(e), "error");
    } finally {
      setRotating(false);
    }
  };

  const relative = (at: number): string => {
    const seconds = Math.max(0, Math.round((now() - at) / 1000));
    if (seconds < 45) return "just now";
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes} min ago`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `${hours} ${hours === 1 ? "hour" : "hours"} ago`;
    const days = Math.round(hours / 24);
    return `${days} ${days === 1 ? "day" : "days"} ago`;
  };

  const snippetVisible = (): boolean => revealed();

  return (
    <Section
      id="mcp-server"
      title="Use Cerebro from other apps"
      description="Other programs on this computer can use your local models through MCP: Claude Code, VS Code, Cursor, the DeepSeek harness and more. Nothing outside this computer can connect."
    >
      <div class="mcpsrv" data-testid="mcp-server-access">
        <div class="mcpsrv__status-row">
          <p class="mcpsrv__status" data-state={statusState()} data-testid="mcp-server-status">
            <span class="mcpsrv__dot" aria-hidden="true" />
            {statusText()}
          </p>
        </div>

        <SettingSwitch
          id="mcp-server-enabled"
          label="Cerebro MCP server"
          note="Listens on 127.0.0.1 only and needs the access token below."
          checked={view()?.enabled ?? false}
          onChange={(on) => void toggle(on)}
        />

        <Show when={view()?.enabled}>
          <div class="mcpsrv__card">
            <h3 class="mcpsrv__title">Access token</h3>
            <div class="mcpsrv__token-row">
              <code class="mono mcpsrv__token" data-testid="mcp-server-token">
                {revealed() ? view()?.token ?? "" : masked()}
              </code>
              <div class="mcpsrv__actions">
                <button class="btn btn--tertiary btn--sm" type="button" onClick={() => setRevealed((r) => !r)}>
                  {revealed() ? "Hide" : "Show"}
                </button>
                <button
                  class="btn btn--tertiary btn--sm"
                  type="button"
                  onClick={() => void copy(view()?.token ?? "", "Paste it into the app you are connecting.")}
                >
                  Copy
                </button>
                <button
                  class="btn btn--secondary btn--sm"
                  type="button"
                  onClick={() => void rotate()}
                  disabled={rotating()}
                  aria-busy={rotating()}
                >
                  {confirming() ? "Click again to replace" : "New token"}
                </button>
              </div>
            </div>
            <Show when={confirming()}>
              <p class="hint mcpsrv__warn">Replacing the token signs out every app using the old one.</p>
            </Show>
          </div>

          <div class="mcpsrv__card">
            <h3 class="mcpsrv__title">Connect an app</h3>
            <div class="mcpsrv__segmented" role="radiogroup" aria-label="App to connect">
              <For each={APPS}>
                {(name) => (
                  <button
                    class="mcpsrv__segment"
                    type="button"
                    role="radio"
                    aria-checked={app() === name}
                    data-active={app() === name}
                    onClick={() => setApp(name)}
                  >
                    {name}
                  </button>
                )}
              </For>
            </div>
            <Show when={snippetVisible()}>
              <pre class="mcpsrv__snippet" data-testid="mcp-server-snippet">
                <code class="mono">{snippet()}</code>
              </pre>
            </Show>
            <div class="mcpsrv__actions">
              <button
                class="btn btn--tertiary btn--sm"
                type="button"
                onClick={() => void copy(snippet(), "Paste it into the app's MCP config.")}
                disabled={!snippet()}
              >
                Copy
              </button>
              <Show when={!revealed()}>
                <p class="hint">The snippet stays hidden until you press Show — or copy it without showing.</p>
              </Show>
            </div>
          </div>

          <div class="mcpsrv__card">
            <h3 class="mcpsrv__title">Connected apps</h3>
            <Show
              when={(view()?.clients.length ?? 0) > 0}
              fallback={<p class="hint">No app has connected yet.</p>}
            >
              <ul class="mcpsrv__client-list">
                <For each={view()?.clients}>
                  {(client) => (
                    <li class="mcpsrv__client">
                      <div class="mcpsrv__client-text">
                        <span class="mcpsrv__client-name">{client.name}</span>
                        <Show when={client.version}>
                          <span class="mono mcpsrv__client-version">{client.version}</span>
                        </Show>
                        <span class="hint">last call {relative(client.lastSeenAt)}</span>
                      </div>
                      <span class="mcpsrv__client-calls">{client.calls} calls</span>
                    </li>
                  )}
                </For>
              </ul>
            </Show>
          </div>

          <div class="mcpsrv__card">
            <h3 class="mcpsrv__title">Tools it offers</h3>
            <ul class="mcpsrv__tool-list">
              <For each={TOOLS}>
                {(tool) => (
                  <li class="mcpsrv__tool">
                    <span class="mono mcpsrv__tool-name">{tool.name}</span>
                    <span class="hint">{tool.desc}</span>
                  </li>
                )}
              </For>
            </ul>
          </div>
        </Show>
      </div>
    </Section>
  );
}
