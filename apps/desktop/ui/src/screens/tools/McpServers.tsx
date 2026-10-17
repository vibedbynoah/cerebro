import { createEffect, createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Section } from "../../components/Screen";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import type { McpCallResult, McpConfig, McpServerConfig, McpServerStatus, McpTool } from "../../service/types";

/** The two ways a server can be reached (mcp.json entry shape). */
const ADD_KINDS = [
  { value: "command" as const, label: "Local program", note: "Starts a program on this machine and talks to it over standard input and output." },
  { value: "url" as const, label: "Web address", note: "Talks to an MCP server at a web address, with headers for anything it asks for (like a token)." },
];
type AddKind = (typeof ADD_KINDS)[number]["value"];

/** MCP servers: configure, monitor and try the servers that give your models tools (screens/Tools.tsx, id="mcp"). */
export function McpServers(): JSX.Element {
  const service = useService();
  const errMessage = (e: unknown): string => (e instanceof Error ? e.message : "Unknown error");
  const toolKey = (t: McpTool): string => `${t.server}/${t.name}`;

  const [config, setConfig] = createSignal<McpConfig | null>(null);
  const [statuses, setStatuses] = createSignal<McpServerStatus[]>([]);
  const [tools, setTools] = createSignal<McpTool[]>([]);
  const [jsonDraft, setJsonDraft] = createSignal("");

  const [toolSignature, setToolSignature] = createSignal("");

  const [addKind, setAddKind] = createSignal<AddKind>("command");
  const [addName, setAddName] = createSignal("");
  const [addCommand, setAddCommand] = createSignal("");
  const [addUrl, setAddUrl] = createSignal("");
  const [addArgs, setAddArgs] = createSignal("");
  const [addEnv, setAddEnv] = createSignal("");

  const [selectedTool, setSelectedTool] = createSignal("");
  const [callArgs, setCallArgs] = createSignal("{}");
  const [callResult, setCallResult] = createSignal<McpCallResult | null>(null);

  const loadConfig = async () => {
    try {
      const c = await service.mcpConfig();
      setConfig(c);
      setJsonDraft(JSON.stringify(c, null, 2));
    } catch (e) {
      notify("Could not load MCP servers", errMessage(e), "error");
    }
  };

  // Polled 3 times a second (never overlapping, skipped while the window is hidden): report a
  // failure once, not on every poll.
  let statusFailed = false;
  let statusInFlight = false;
  const loadStatuses = async () => {
    if (statusInFlight || document.visibilityState === "hidden") return;
    statusInFlight = true;
    try {
      setStatuses(await service.mcpStatus());
      statusFailed = false;
    } catch (e) {
      if (!statusFailed) notify("Could not load server status", errMessage(e), "error");
      statusFailed = true;
    } finally {
      statusInFlight = false;
    }
  };

  const loadTools = async () => {
    try {
      setTools(await service.mcpTools());
    } catch (e) {
      notify("Could not load tools", errMessage(e), "error");
    }
  };

  void loadConfig();
  void loadStatuses();
  const poll = setInterval(() => void loadStatuses(), 333);
  onCleanup(() => clearInterval(poll));

  // Tool list follows the servers: refresh whenever reported tool counts change.
  createEffect(() => {
    const signature = statuses()
      .map((s) => `${s.name}:${s.tools}`)
      .join("|");
    if (signature === toolSignature()) return;
    setToolSignature(signature);
    void loadTools();
  });
  // Keep a valid selection as the tool list changes.
  createEffect(() => {
    const list = tools();
    setSelectedTool((current) => (current && list.some((t) => toolKey(t) === current) ? current : list[0] ? toolKey(list[0]!) : ""));
  });

  // "quoted parts stay together": arguments split on spaces unless wrapped in quotes.
  const splitArguments = (value: string): string[] => {
    const parts: string[] = [];
    const pattern = /"([^"]*)"|'([^']*)'|(\S+)/g;
    let m: RegExpExecArray | null;
    while ((m = pattern.exec(value)) !== null) parts.push(m[1] ?? m[2] ?? m[3] ?? "");
    return parts;
  };

  const parseEnv = (value: string): Record<string, string> => {
    const env: Record<string, string> = {};
    for (const line of value.split("\n")) {
      const text = line.trim();
      const eq = text.indexOf("=");
      if (eq > 0) env[text.slice(0, eq).trim()] = text.slice(eq + 1).trim();
    }
    return env;
  };

  const save = async (next: McpConfig) => {
    await service.mcpSaveConfig(next);
    setConfig(next);
    setJsonDraft(JSON.stringify(next, null, 2));
    void loadStatuses();
  };

  const restart = async (name: string) => {
    try {
      await service.mcpRestart(name);
      await loadStatuses();
    } catch (e) {
      notify(`Could not restart ${name}`, errMessage(e), "error");
    }
  };

  const toggle = async (name: string, disabled: boolean) => {
    const current = config();
    if (!current) return;
    try {
      const next: McpConfig = {
        mcpServers: { ...current.mcpServers, [name]: { ...current.mcpServers[name]!, disabled } },
      };
      await save(next);
    } catch (e) {
      notify(`Could not ${disabled ? "disable" : "enable"} ${name}`, errMessage(e), "error");
    }
  };

  const remove = async (name: string) => {
    const current = config();
    if (!current) return;
    try {
      const servers = { ...current.mcpServers };
      delete servers[name];
      await save({ mcpServers: servers });
    } catch (e) {
      notify(`Could not remove ${name}`, errMessage(e), "error");
    }
  };

  const addServer = async (event: SubmitEvent) => {
    event.preventDefault();
    const current = config();
    if (!current) return;
    const name = addName().trim();
    if (!name) {
      notify("A server needs a name", "Enter a name before saving.", "warning");
      return;
    }
    if (current.mcpServers[name]) {
      notify(`"${name}" already exists`, "Pick another name, or turn the existing server off.", "warning");
      return;
    }
    let entry: McpServerConfig;
    if (addKind() === "command") {
      const command = addCommand().trim();
      if (!command) {
        notify("A server needs a command", "For example: npx -y @modelcontextprotocol/server-filesystem", "warning");
        return;
      }
      entry = { command, args: splitArguments(addArgs()), env: parseEnv(addEnv()), disabled: false };
    } else {
      const url = addUrl().trim();
      if (!/^https?:\/\/\S+$/.test(url)) {
        notify("A web address must start with http:// or https://", "For example: https://example.com/mcp", "warning");
        return;
      }
      entry = { url, headers: parseEnv(addEnv()), disabled: false };
    }
    try {
      await save({ mcpServers: { ...current.mcpServers, [name]: entry } });
      setAddName("");
      setAddCommand("");
      setAddUrl("");
      setAddArgs("");
      setAddEnv("");
    } catch (e) {
      notify(`Could not add ${name}`, errMessage(e), "error");
    }
  };

  const saveJson = async () => {
    let parsed: McpConfig;
    try {
      parsed = JSON.parse(jsonDraft()) as McpConfig;
    } catch (e) {
      notify("mcp.json is not valid JSON", errMessage(e), "error");
      return;
    }
    try {
      await save(parsed);
    } catch (e) {
      notify("Could not save mcp.json", errMessage(e), "error");
    }
  };

  type Preset = { label: string; command: string; args: string };
  const PRESETS: Preset[] = [
    { label: "Files", command: "npx", args: "-y @modelcontextprotocol/server-filesystem <your folder>" },
    { label: "Git", command: "uvx", args: "mcp-server-git" },
    { label: "Fetch", command: "uvx", args: "mcp-server-fetch" },
    { label: "Memory", command: "npx", args: "-y @modelcontextprotocol/server-memory" },
    { label: "SQLite", command: "uvx", args: "mcp-server-sqlite --db-path <file>" },
  ];

  const applyPreset = (preset: Preset) => {
    setAddKind("command");
    setAddName(preset.label.toLowerCase());
    setAddCommand(preset.command);
    setAddArgs(preset.args);
  };

  const stateFor = (name: string, disabled: boolean): { label: string; pill: string } => {
    if (disabled) return { label: "Off", pill: "pill pill--off" };
    const status = statuses().find((s) => s.name === name);
    const state = status?.state ?? "stopped";
    return { label: state, pill: `pill pill--${state}` };
  };

  /** How a server entry reads in the list: its address, or its program and arguments. */
  const serverLine = (s: McpServerConfig): string => s.url ?? [s.command, ...(s.args ?? [])].filter(Boolean).join(" ");

  const currentTool = () => tools().find((t) => toolKey(t) === selectedTool());

  const runTool = async () => {
    const t = currentTool();
    if (!t) return;
    let args: unknown;
    try {
      args = JSON.parse(callArgs() || "{}");
    } catch (e) {
      notify("Arguments are not valid JSON", errMessage(e), "error");
      return;
    }
    try {
      setCallResult(await service.mcpCall(t.server, t.name, args));
    } catch (e) {
      notify(`Could not run ${t.name}`, errMessage(e), "error");
    }
  };

  const resultText = (): string =>
    (callResult()?.content ?? [])
      .filter((part) => part.type === "text" && part.text !== undefined)
      .map((part) => part.text)
      .join("\n");

  return (
    <Section
      id="mcp"
      title="MCP servers"
      description="Give your models tools: files, git, search, databases and more. Servers run on this machine; each tool call still follows your rules below."
    >
      <div class="mcp" data-testid="mcp-servers">
        <ul class="mcp__servers">
          <For
            each={config() ? Object.entries(config()!.mcpServers) : []}
            fallback={<li class="hint">No servers yet. Add one below to give your models tools.</li>}
          >
            {([name, server]) => {
              const state = () => stateFor(name, server.disabled);
              const status = () => statuses().find((s) => s.name === name);
              return (
                <li class="mcp__server" data-testid={`mcp-server-${name}`}>
                  <div class="mcp__server-head">
                    <span class="mcp__server-name">{name}</span>
                    <span class={state().pill}>{state().label}</span>
                    <Show when={status()}>
                      {(s) => <span class="mcp__server-tools">{s().tools} tools</span>}
                    </Show>
                  </div>
                  <code class="mcp__command">{serverLine(server)}</code>
                  <Show when={status()?.error}>
                    {(error) => <p class="mcp__error">{error()}</p>}
                  </Show>
                  <details class="mcp__log">
                    <summary>Log</summary>
                    <pre class="mcp__log-output">{status()?.stderr_tail.join("\n") ?? ""}</pre>
                  </details>
                  <div class="mcp__server-actions">
                    <button class="btn btn--ghost btn--sm" type="button" onClick={() => void restart(name)}>
                      Restart
                    </button>
                    <button class="btn btn--ghost btn--sm" type="button" onClick={() => void toggle(name, !server.disabled)}>
                      {server.disabled ? "On" : "Off"}
                    </button>
                    <button class="btn btn--ghost btn--sm" type="button" onClick={() => void remove(name)}>
                      Remove
                    </button>
                  </div>
                </li>
              );
            }}
          </For>
        </ul>

        <form class="mcp__form" onSubmit={addServer}>
          <h3 class="mcp__form-title">Add a server</h3>
          <fieldset class="segmented segmented--sm" aria-describedby="mcp-add-note">
            <legend class="sr-only">How the server runs</legend>
            <For each={ADD_KINDS}>
              {(kind) => (
                <label class="segmented__option" data-press>
                  <input type="radio" name="mcp-add-kind" value={kind.value} checked={addKind() === kind.value} onChange={() => setAddKind(kind.value)} />
                  <span>{kind.label}</span>
                </label>
              )}
            </For>
          </fieldset>
          <p class="hint" id="mcp-add-note">
            {ADD_KINDS.find((kind) => kind.value === addKind())!.note}
          </p>
          <Show when={addKind() === "command"}>
            <div class="mcp__presets" role="group" aria-label="Server presets">
              <For each={PRESETS}>
                {(preset) => (
                  <button class="btn btn--ghost btn--sm" type="button" onClick={() => applyPreset(preset)}>
                    {preset.label}
                  </button>
                )}
              </For>
            </div>
          </Show>
          <label class="mcp__field">
            <span>Name</span>
            <input class="input" aria-label="Server name" value={addName()} onInput={(e) => setAddName(e.currentTarget.value)} placeholder="files" />
          </label>
          <Show
            when={addKind() === "command"}
            fallback={
              <>
                <label class="mcp__field">
                  <span>Address</span>
                  <input class="input" aria-label="Server address" value={addUrl()} onInput={(e) => setAddUrl(e.currentTarget.value)} placeholder="https://example.com/mcp" />
                </label>
                <label class="mcp__field">
                  <span>Headers</span>
                  <textarea
                    class="input mcp__env"
                    aria-label="Headers, one KEY=VALUE per line"
                    rows={3}
                    value={addEnv()}
                    onInput={(e) => setAddEnv(e.currentTarget.value)}
                    placeholder={"Authorization=Bearer <token>"}
                  />
                </label>
              </>
            }
          >
            <label class="mcp__field">
              <span>Command</span>
              <input class="input" aria-label="Server command" value={addCommand()} onInput={(e) => setAddCommand(e.currentTarget.value)} placeholder="npx" />
            </label>
            <label class="mcp__field">
              <span>Arguments</span>
              <input class="input" aria-label="Server arguments, quoted parts stay together" value={addArgs()} onInput={(e) => setAddArgs(e.currentTarget.value)} placeholder="C:\my\folder" />
            </label>
            <label class="mcp__field">
              <span>Environment</span>
              <textarea
                class="input mcp__env"
                aria-label="Environment variables, one KEY=VALUE per line"
                rows={3}
                value={addEnv()}
                onInput={(e) => setAddEnv(e.currentTarget.value)}
                placeholder={"KEY=VALUE\nOTHER_KEY=OTHER_VALUE"}
              />
            </label>
          </Show>
          <button class="btn btn--primary btn--sm" type="submit">
            Add server
          </button>
        </form>

        <details class="mcp__advanced">
          <summary>Advanced: edit mcp.json</summary>
          <textarea
            class="input mcp__json"
            aria-label="Edit mcp.json"
            rows={10}
            value={jsonDraft()}
            onInput={(e) => setJsonDraft(e.currentTarget.value)}
          />
          <button class="btn btn--ghost btn--sm" type="button" onClick={() => void saveJson()}>
            Save mcp.json
          </button>
        </details>

        <div class="mcp__try">
          <h3 class="mcp__try-title">Try a tool</h3>
          <label class="mcp__field">
            <span>Tool</span>
            <select
              aria-label="Tool to try"
              value={selectedTool()}
              onChange={(e) => {
                setSelectedTool(e.currentTarget.value);
                setCallResult(null);
              }}
            >
              <For each={tools()} fallback={<option value="">No tools available</option>}>
                {(t) => <option value={toolKey(t)}>{t.server} / {t.name}</option>}
              </For>
            </select>
          </label>
          <Show when={currentTool()}>
            {(t) => <p class="hint">{t().description ?? "No description provided."}</p>}
          </Show>
          <label class="mcp__field">
            <span>Arguments (JSON)</span>
            <textarea
              class="input mcp__args"
              aria-label="Tool arguments as JSON"
              rows={4}
              value={callArgs()}
              onInput={(e) => setCallArgs(e.currentTarget.value)}
            />
          </label>
          <button class="btn btn--primary btn--sm" type="button" onClick={() => void runTool()}>
            Run
          </button>
          <Show when={callResult()}>
            {(r) => (
              <pre class="mcp__output" data-error={r().isError ? "true" : undefined}>
                {resultText() || "(no text content)"}
              </pre>
            )}
          </Show>
        </div>
      </div>
    </Section>
  );
}
