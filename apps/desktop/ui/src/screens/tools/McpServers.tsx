import { createEffect, createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Section } from "../../components/Screen";
import { MCP_CATALOGUE, type McpCatalogueEntry, type McpRisk } from "../../lib/mcp-catalogue";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import type { McpCallResult, McpConfig, McpServerStatus, McpTool } from "../../service/types";

const RISK_LABELS: Record<McpRisk, string> = { read: "Read", write: "Write", run: "Run", network: "Network", account: "Account" };

/** Catalogue categories in first-appearance order, so the chips read like the data. */
const CATEGORIES = [...new Set(MCP_CATALOGUE.map((e) => e.category))];

/** The one line a card shows for its install: the command line or the hosted endpoint. */
const installText = (entry: McpCatalogueEntry): string =>
  "url" in entry.install ? entry.install.url : [entry.install.command, ...entry.install.args].join(" ");

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

  const [addName, setAddName] = createSignal("");
  const [addCommand, setAddCommand] = createSignal("");
  const [addArgs, setAddArgs] = createSignal("");
  const [addEnv, setAddEnv] = createSignal("");

  // Browse servers: text and category filter the catalogue; choosing an entry pre-fills the form.
  const [browseQuery, setBrowseQuery] = createSignal("");
  const [browseCategory, setBrowseCategory] = createSignal("");
  const [chosen, setChosen] = createSignal<McpCatalogueEntry | null>(null);
  const [chosenEnv, setChosenEnv] = createSignal<Record<string, string>>({});
  let formCard: HTMLFormElement | undefined;

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

  const browseResults = () => {
    const category = browseCategory();
    const inCategory = category ? MCP_CATALOGUE.filter((e) => e.category === category) : MCP_CATALOGUE;
    const query = browseQuery().trim().toLowerCase();
    if (!query) return inCategory;
    return inCategory.filter((e) => `${e.name} ${e.category} ${e.description}`.toLowerCase().includes(query));
  };

  // Pre-fill the add form from the catalogue; the user still adjusts paths and fills the keys.
  const chooseFromCatalogue = (entry: McpCatalogueEntry) => {
    if ("url" in entry.install) return; // hosted entries are shown, never added here
    setChosen(entry);
    setChosenEnv({});
    setAddName(entry.id);
    setAddCommand(entry.install.command);
    setAddArgs(entry.install.args.join(" "));
    formCard?.scrollIntoView({ block: "start" });
  };

  const clearChosen = () => {
    setChosen(null);
    setChosenEnv({});
  };

  const addServer = async (event: SubmitEvent) => {
    event.preventDefault();
    const current = config();
    if (!current) return;
    const entry = chosen();
    const name = addName().trim();
    const command = addCommand().trim();
    const args = splitArguments(addArgs());
    if (!name) {
      notify("A server needs a name", "Enter a name before saving.", "warning");
      return;
    }
    if (current.mcpServers[name]) {
      notify(`"${name}" already exists`, "Pick another name, or turn the existing server off.", "warning");
      return;
    }
    if (!command) {
      notify("A server needs a command", "For example: npx -y @modelcontextprotocol/server-filesystem", "warning");
      return;
    }
    // Catalogue entries keep their keys in named fields; a manual add uses the free-form textarea.
    const env = entry
      ? Object.fromEntries(Object.entries(chosenEnv()).filter(([, value]) => value.trim() !== ""))
      : parseEnv(addEnv());
    try {
      await save({
        mcpServers: { ...current.mcpServers, [name]: { command, args, env, disabled: false } },
      });
      setAddName("");
      setAddCommand("");
      setAddArgs("");
      setAddEnv("");
      clearChosen();
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

  const stateFor = (name: string, disabled: boolean): { label: string; pill: string } => {
    if (disabled) return { label: "Off", pill: "pill pill--off" };
    const status = statuses().find((s) => s.name === name);
    const state = status?.state ?? "stopped";
    return { label: state, pill: `pill pill--${state}` };
  };

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
                  <code class="mcp__command">{[server.command, ...server.args].join(" ")}</code>
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

        <div class="mcp__browse" data-testid="mcp-catalogue">
          <h3 class="mcp__form-title">Browse servers</h3>
          <p class="hint">
            Real servers from the official MCP project and their vendors. "Add" fills the form below with the command and the keys that server
            needs.
          </p>
          <div class="mcp__browse-controls">
            <input
              class="tools-search mcp__search"
              type="search"
              placeholder="Search servers"
              aria-label="Search servers"
              value={browseQuery()}
              onInput={(e) => setBrowseQuery(e.currentTarget.value)}
            />
            <div class="mcp__chips" role="group" aria-label="Filter by category">
              <button class="mcp__chip" type="button" aria-pressed={browseCategory() === ""} onClick={() => setBrowseCategory("")}>
                All
              </button>
              <For each={CATEGORIES}>
                {(category) => (
                  <button
                    class="mcp__chip"
                    type="button"
                    aria-pressed={browseCategory() === category}
                    onClick={() => setBrowseCategory(browseCategory() === category ? "" : category)}
                  >
                    {category}
                  </button>
                )}
              </For>
            </div>
          </div>
          <p class="hint" data-testid="mcp-catalogue-count">
            {browseResults().length} of {MCP_CATALOGUE.length} servers
          </p>
          <ul class="mcp__grid">
            <For each={browseResults()} fallback={<li class="hint">Nothing matches. Clear the search or pick another category.</li>}>
              {(entry) => (
                <li class="mcp__card" data-testid={`mcp-catalogue-${entry.id}`}>
                  <div class="mcp__card-head">
                    <span class="mcp__card-name">{entry.name}</span>
                    <span class="mcp__badge" data-risk={entry.risk}>
                      {RISK_LABELS[entry.risk]}
                    </span>
                    <span class="mcp__badge" data-offline={entry.offline ? "yes" : "no"}>
                      {entry.offline ? "Works offline" : "Needs internet"}
                    </span>
                    <Show when={"url" in entry.install}>
                      <span class="mcp__badge" data-remote="yes">
                        Remote
                      </span>
                    </Show>
                  </div>
                  <p class="mcp__card-desc">{entry.description}</p>
                  <code class="mcp__command mcp__card-command">{installText(entry)}</code>
                  <div class="mcp__card-actions">
                    <a class="text-link" href={entry.homepage} target="_blank" rel="noopener noreferrer nofollow" aria-label={`${entry.name} homepage`}>
                      Homepage
                    </a>
                    <Show
                      when={"command" in entry.install}
                      fallback={<span class="hint">Hosted over HTTP; this app starts local servers only.</span>}
                    >
                      <button class="btn btn--ghost btn--sm" type="button" aria-label={`Add ${entry.name}`} onClick={() => chooseFromCatalogue(entry)}>
                        Add
                      </button>
                    </Show>
                  </div>
                </li>
              )}
            </For>
          </ul>
        </div>

        <form class="mcp__form" onSubmit={addServer} ref={formCard}>
          <h3 class="mcp__form-title">Add a server</h3>
          <Show when={chosen()}>
            {(entry) => (
              <div class="mcp__keys" data-testid="mcp-catalogue-keys">
                <div class="mcp__keys-head">
                  <span>
                    From the catalogue: <strong>{entry().name}</strong>
                  </span>
                  <button class="btn btn--ghost btn--sm" type="button" onClick={clearChosen}>
                    Clear
                  </button>
                </div>
                <For
                  each={entry().env}
                  fallback={<p class="hint">This server needs no keys. Adjust the arguments if needed, then add it.</p>}
                >
                  {(key) => {
                    const inputId = `mcp-key-${entry().id}-${key.name}`;
                    return (
                      <div class="mcp__field">
                        <label for={inputId}>{key.label}</label>
                        <input
                          id={inputId}
                          class="input"
                          type={key.secret ? "password" : "text"}
                          autocomplete={key.secret ? "new-password" : "off"}
                          placeholder={key.name}
                          value={chosenEnv()[key.name] ?? ""}
                          onInput={(e) => setChosenEnv((v) => ({ ...v, [key.name]: e.currentTarget.value }))}
                        />
                        <a class="hint mcp__key-link" href={key.url} target="_blank" rel="noopener noreferrer nofollow">
                          {new URL(key.url).hostname}
                        </a>
                      </div>
                    );
                  }}
                </For>
              </div>
            )}
          </Show>
          <label class="mcp__field">
            <span>Name</span>
            <input class="input" aria-label="Server name" value={addName()} onInput={(e) => setAddName(e.currentTarget.value)} placeholder="files" />
          </label>
          <label class="mcp__field">
            <span>Command</span>
            <input class="input" aria-label="Server command" value={addCommand()} onInput={(e) => setAddCommand(e.currentTarget.value)} placeholder="npx" />
          </label>
          <label class="mcp__field">
            <span>Arguments</span>
            <input class="input" aria-label="Server arguments, quoted parts stay together" value={addArgs()} onInput={(e) => setAddArgs(e.currentTarget.value)} placeholder="C:\my\folder" />
          </label>
          <Show when={!chosen()}>
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
