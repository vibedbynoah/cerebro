import "../styles/chat.css";
import { batch, createEffect, createMemo, createResource, createSignal, For, on, onCleanup, onMount, Show, type JSX } from "solid-js";
import { createStore, produce } from "solid-js/store";
import { Icon } from "../components/icons";
import { IllustrativeFlag } from "../components/Screen";
import { SettingSwitch } from "../components/SettingSwitch";
import { applyMode, loadSampler, MODE_PRESETS, modeOf, saveSampler, type Mode } from "../lib/chat/modes";
import { buildRequest, DEFAULT_BASE_PROMPT, effectiveLevels, loadLevelEdits, loadPrefs, savePrefs, supportsNativeReasoning, type ChatPrefs, type Memory } from "../lib/chat/prompt";
import {
  activeMessages,
  branchPath,
  branchPosition,
  emptyConversation,
  focusBranch,
  loadConversations,
  loadMemories,
  memoryCandidate,
  newId,
  saveConversations,
  saveMemories,
  switchBranch,
  titleFrom,
  treeFromMessages,
  type Attachment,
  type ChatMessage,
  type Conversation,
  type LedgerState,
} from "../lib/chat/store";
import { chatJump, publishChats, setChatJump } from "../lib/chat/navigation";
import { loadPrompts, savePrompts, type SavedPrompt } from "../lib/chat/prompts";
import { searchConversations } from "../lib/chat/search";
import { docsInScope, folderNames as docFolderNames, loadCollections, scopeKey, tagNames as docTagNames } from "../lib/collections";
import { loadDocs } from "../lib/documents";
import { assembleContext, ragPrompt, retrieve, type ContextBlock, type Source } from "../lib/rag";
import { buildQuickToolPrompt, quickTool, REWRITE_STYLES, TRANSLATE_LANGUAGES, type QuickToolId, type QuickToolInputValue } from "../lib/quick-tools";
import { loadDocsPrefs, saveDocsPrefs, scopeOf, type DocsPrefs } from "../lib/chat/docs";
import { chatToolRules, loadPolicy, mcpToolKey, savePolicy } from "../lib/permissions";
import { hrefFor } from "../lib/routes";
import { notify, notifyAction } from "../lib/toast";
import { COMMANDS, numberArg, orchestrateArg, speedArg, switchArg } from "../lib/chat/commands";
import { parseMarkdown, plainText } from "../lib/chat/markdown";
import { saveText } from "../lib/download";
import { Dialog } from "@ark-ui/solid/dialog";
import { Portal } from "solid-js/web";
import "../styles/guide.css";
import { useAppState } from "../state";
import { DEFAULT_SAMPLER, LEDGER_FILES, ranOn, type ChatRequest, type ChatTools, type ChatTurn, type LedgerFileName, type Sampler, type ToolActivity } from "../service/chat-types";
import { useService } from "../service/context";
import type { CloudProvider, LibraryModel } from "../service/types";
import { HighlightQuery } from "../components/ChatSearch";
import { hrefForMonitorTab } from "../lib/routes";
import { Arena } from "./chat/Arena";
import { ChatSettings, Composer } from "./chat/Composer";
import { LiveDock } from "./chat/LiveDock";
import { loadPresets, removePreset, savePreset } from "../lib/chat/presets";
import { VoiceMode } from "./chat/VoiceMode";
import { readPref, writePref } from "../lib/storage";
import { MessageView } from "./chat/Message";
import { PromptLibrary } from "./chat/PromptLibrary";
import type { ToolAnswer, ToolDecision } from "./chat/ToolCard";
import { recordToolActivity, summarizeArgs } from "../lib/tool-activity";

const MAX_THINKING = 32768;

/**
 * Chat: conversations on this machine, streaming replies with live speed, effort and reasoning
 * source per message (docs/specs/effort-levels.md), verbosity, memory, attachments, push-to-talk,
 * /imagine, an editable system prompt and sampler controls. Replies are rendered by the safe
 * Markdown renderer only.
 */
export function Chat(): JSX.Element {
  const service = useService();
  const app = useAppState();
  const [library] = createResource(() => service.listLibrary().catch(() => [] as LibraryModel[]));
  // Models the engine can run; ones it cannot are listed (and explained) on the Models screen.
  const models = createMemo(() => (library() ?? []).filter((m) => (m.kind === "text" || m.kind === "vision") && m.supported !== false));
  const [modelId, setModelId] = createSignal<string>("");
  // Start on the model that is already in memory, so the first message needs no load.
  const [loadedNow] = createResource(() => service.listLoaded().catch(() => []));
  createEffect(() => {
    if (modelId() || !models().length || loadedNow.loading) return;
    const inMemory = loadedNow()?.find((l) => l.slot === "llm" && models().some((m) => m.id === l.id));
    setModelId(inMemory?.id ?? models()[0]!.id);
  });
  const model = () => models().find((m) => m.id === modelId());
  // A model's own saved settings come back when it is picked (lib/chat/presets.ts).
  const [presetIds, setPresetIds] = createSignal(new Set(Object.keys(loadPresets())));
  createEffect(
    on(
      modelId,
      (id) => {
        const p = id ? loadPresets()[id] : undefined;
        if (!p) return;
        setPrefs(p.prefs);
        setSampler(p.sampler);
        notify("This model's settings", "Its saved settings are back. Change them in the live panel.", "info");
      },
      { defer: true },
    ),
  );
  // Cloud models the user added in Settings > Cloud models: "cloud:<provider>/<model>".
  const [cloud] = createResource(() => service.cloudProviders().catch(() => [] as CloudProvider[]));
  const cloudModels = createMemo(() => (cloud() ?? []).flatMap((p) => p.models.map((m) => ({ id: `cloud:${p.id}/${m}`, name: m, provider: p.name }))));
  const cloudOf = () => cloudModels().find((m) => m.id === modelId()) ?? null;

  const [prefs, setPrefsSignal] = createSignal<ChatPrefs>(loadPrefs());
  // The user's documents: off until they ask for them, and then only the files they pick.
  const [docsPrefs, setDocsPrefsSignal] = createSignal<DocsPrefs>(loadDocsPrefs());
  // The collections the Documents screen keeps; the screen is remounted when the route changes, so
  // these are the ones the user can see now.
  const collections = loadCollections();
  const setDocsPrefs = (p: Partial<DocsPrefs>) => {
    const next = { ...docsPrefs(), ...p };
    setDocsPrefsSignal(next);
    saveDocsPrefs(next);
  };
  const levels = effectiveLevels(loadLevelEdits());
  const setPrefs = (p: Partial<ChatPrefs>) => {
    const next = { ...prefs(), ...p };
    setPrefsSignal(next);
    savePrefs(next);
  };
  const [sampler, setSampler] = createSignal<Sampler>(loadSampler());
  createEffect(on(sampler, (s) => saveSampler(s), { defer: true }));

  const [state, setState] = createStore<{ list: Conversation[] }>({ list: loadConversations() });
  const [currentId, setCurrentId] = createSignal<string>(state.list[0]?.id ?? "");
  const current = () => state.list.find((c) => c.id === currentId());
  const [memories, setMemories] = createSignal<Memory[]>(loadMemories());
  const [prompts, setPromptsSignal] = createSignal<SavedPrompt[]>(loadPrompts());
  // The side panel: memory, the system prompt, or the live dock (remembered while it is the dock).
  const DOCK_KEY = "cerebro-chat-dock";
  const PANEL_KEY = "cerebro-chat-panel";
  type PanelId = "none" | "settings" | "memory" | "prompt" | "prompts" | "dock";
  // Open on Settings unless the person closed the panel; the live dock is remembered as before.
  const [panel, setPanelSignal] = createSignal<PanelId>(readPref(DOCK_KEY) === "open" ? "dock" : readPref(PANEL_KEY) === "closed" ? "none" : "settings");
  const setPanel = (p: PanelId) => {
    setPanelSignal(p);
    writePref(DOCK_KEY, p === "dock" ? "open" : null);
    writePref(PANEL_KEY, p === "none" ? "closed" : null);
  };
  const [dockWidth, setDockWidth] = createSignal(Math.min(520, Math.max(280, Number(readPref("cerebro-chat-dock-width")) || 340)));
  const startResize = (e: PointerEvent) => {
    e.preventDefault();
    const x0 = e.clientX;
    const w0 = dockWidth();
    const move = (ev: PointerEvent) => setDockWidth(Math.min(520, Math.max(280, w0 + (x0 - ev.clientX))));
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      writePref("cerebro-chat-dock-width", String(dockWidth()));
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };
  onMount(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === ".") {
        e.preventDefault();
        setPanel(panel() === "dock" ? "none" : "dock");
      }
    };
    window.addEventListener("keydown", onKey);
    onCleanup(() => window.removeEventListener("keydown", onKey));
  });
  const [view, setView] = createSignal<"chat" | "arena">("chat");
  const [live, setLive] = createSignal<{ tps: number; ttft: number } | null>(null);
  const [draft, setDraft] = createSignal<{ text: string; at: number } | null>(null);
  const [listQuery, setListQuery] = createSignal("");
  const [folderFilter, setFolderFilter] = createSignal("all");
  const [tagFilter, setTagFilter] = createSignal("all");
  const [showArchived, setShowArchived] = createSignal(false);
  const [foundId, setFoundId] = createSignal("");
  let foundTimer: ReturnType<typeof setTimeout> | undefined;

  const setPrompts = (fn: (list: SavedPrompt[]) => SavedPrompt[]) => setPromptsSignal(fn);

  let saveTimer: ReturnType<typeof setTimeout> | undefined;
  const persist = () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => saveConversations(state.list), 400);
  };
  onCleanup(() => {
    clearTimeout(saveTimer);
    clearTimeout(foundTimer);
    saveConversations(state.list);
    abort?.abort();
  });
  createEffect(on(memories, (m) => saveMemories(m), { defer: true }));
  createEffect(on(prompts, (p) => savePrompts(p), { defer: true }));
  // Search in the shell sees a just-sent message before the debounced save reaches local storage.
  createEffect(
    on(
      () => state.list.map((c) => `${c.id}:${c.updatedAt}:${c.messages.length}:${c.title}`).join("|"),
      () => publishChats(state.list),
    ),
  );

  const built = createMemo(() =>
    buildRequest({
      levels,
      base: prefs().base,
      effort: prefs().effort,
      custom: prefs().effortCustom,
      thinking: prefs().thinking,
      verbosity: prefs().verbosity,
      source: prefs().source,
      addInstructions: prefs().addInstructions,
      memories: memories(),
      memoryOn: prefs().memoryOn && !(current()?.noMemory ?? false),
      nativeSupported: supportsNativeReasoning(`${model()?.family ?? ""} ${model()?.name ?? ""}`),
      modelMaxThinking: MAX_THINKING,
    }),
  );

  /** The Speed <-> Accuracy slider: every setting it covers, then a notice saying what changed. */
  let modeNotice: ReturnType<typeof setTimeout> | undefined;
  /** `announce`: say what the level does in a toast (for /speed; the slider and the live panel say
   * it in place, where it does not cover the reply). */
  const pickMode = (m: Mode, announce = true) => {
    const next = applyMode(m, prefs(), sampler());
    setPrefs(next.prefs);
    setSampler(next.sampler);
    // Dragging across levels shows only the one it stops on.
    clearTimeout(modeNotice);
    if (!announce) return;
    modeNotice = setTimeout(() => {
      const p = MODE_PRESETS[m];
      notify(`${p.label}`, `${p.summary} ${p.changes.join(". ")}.`, "info");
    }, 350);
  };
  onCleanup(() => clearTimeout(modeNotice));

  let abort: AbortController | null = null;
  const streaming = () => current()?.messages.some((m) => m.streaming) ?? false;

  const ensureConversation = (): string => {
    if (current()) return currentId();
    const c = emptyConversation();
    setState("list", (l) => [c, ...l]);
    setCurrentId(c.id);
    return c.id;
  };

  /** Old records and hand-built state always get a tree before a branch operation needs one. */
  const treeOf = (c: Conversation) => {
    if (!c.tree || !Array.isArray(c.tree.nodes)) c.tree = treeFromMessages(c.messages);
    return c.tree;
  };

  /** Adds a message node under `parentId` (null for the first message) and makes it the active child. */
  const addMessage = (cid: string, message: ChatMessage, parentId: string | null) => {
    setState(
      "list",
      (c) => c.id === cid,
      produce((c) => {
        const tree = treeOf(c);
        tree.nodes.push({ id: message.id, parentId, message });
        tree.selected[parentId ?? ""] = message.id;
        c.messages = activeMessages(tree);
        c.updatedAt = Date.now();
      }),
    );
  };

  const updateMsg = (cid: string, mid: string, fn: (m: ChatMessage) => void) =>
    setState(
      "list",
      (c) => c.id === cid,
      produce((c) => {
        const node = c.tree?.nodes.find((n) => n.id === mid);
        const message = c.messages.find((m) => m.id === mid);
        if (node) fn(node.message);
        // A parsed chat shares the node's message object with the active path; only touch it twice
        // when they really are different objects.
        if (message && message !== node?.message) fn(message);
      }),
    );

  /** The tools the model may call this turn: the built-ins plus every MCP tool, each resolved
   *  against the Tools rules now, with Offline mode passed so fetch_url can refuse. */
  const toolsForRequest = async (): Promise<ChatTools | null> => {
    try {
      const [mcp, builtins] = await Promise.all([service.mcpTools(), service.builtinTools()]);
      const all = [...builtins, ...mcp];
      if (!all.length) return null;
      const rules = chatToolRules(loadPolicy(), all.map((t) => mcpToolKey(t.server, t.name)), app.settings()?.permissionPrompts ?? true);
      const s = app.settings();
      return { ...rules, offline: s?.offlineMode === true, allowLocalFetch: s?.blockLocalFetch === false, showSecrets: s?.hideSecretsFromTools === false, blindWrites: s?.readBeforeWrite === false, planMode: prefs().planMode };
    } catch {
      return null;
    }
  };

  /** Allow once / Always allow / Deny on a tool card. "Always" is saved as the tool's rule. */
  const decideTool = (cid: string, mid: string, id: string, decision: ToolDecision) => {
    const msg = state.list.find((c) => c.id === cid)?.messages.find((m) => m.id === mid);
    const tool = msg?.tools.find((t) => t.id === id);
    if (tool && decision === "always") {
      const p = loadPolicy();
      p.rules[mcpToolKey(tool.server, tool.name)] = "always";
      savePolicy(p);
      notify("Always allowed", `${tool.name} from ${tool.server} now runs without asking. Change it on the Tools screen.`, "success");
    }
    void service.chatToolDecide(id, decision !== "deny").then((ok) => {
      if (!ok) notify("That request already ended", "The reply stopped before your answer arrived.", "info");
    });
  };

  /** An answer to a question or a plan review on a tool card. An approved plan ends plan mode. */
  const answerTool = (id: string, answer: ToolAnswer) => {
    if ("approved" in answer && answer.approved && prefs().planMode) {
      setPrefs({ planMode: false });
      notify("Plan approved", "Plan first is off; the model carries the plan out now.", "success");
    }
    void service.chatToolAnswer(id, answer as Record<string, unknown>).then((ok) => {
      if (!ok) notify("That request already ended", "The reply stopped before your answer arrived.", "info");
    });
  };

  /** Loads the picked model when another one (or none) is in memory. False when it cannot load. */
  const ensureLoaded = async (): Promise<boolean> => {
    const id = modelId();
    const now = await service.listLoaded().catch(() => [] as Awaited<ReturnType<typeof service.listLoaded>>);
    if (now.some((l) => l.slot === "llm" && l.id === id)) return true;
    const m = model();
    notify(`Loading ${m?.name ?? "the model"}`, "It answers as soon as it is in memory.", "info");
    try {
      await service.loadModel(id, m?.tunedParams ?? null);
      return true;
    } catch (e) {
      notify("Could not load the model", e instanceof Error ? e.message : String(e), "error");
      return false;
    }
  };

  /** The passages of the user's documents that answer this message, when they are switched on. */
  const docsContext = (question: string): ContextBlock | null => {
    if (!docsPrefs().on || !question.trim()) return null;
    const scope = scopeOf(docsPrefs(), collections);
    const docs = docsInScope(loadDocs(), scope);
    if (!docs.length) return null;
    const found = retrieve(docs, question, { limit: 6 });
    if (!found.length) return null;
    return assembleContext(found);
  };

  /** Streams a reply into a new message of the chat, and keeps the sources it used. */
  const streamReply = async (cid: string, parentId: string | null, request: ChatRequest, sources: Source[] = []) => {
    const p = prefs();
    const mid = newId();
    const reply: ChatMessage = { id: mid, role: "assistant", text: "", thinking: "", checks: [], image: null, tools: [], ledger: null, attachments: [], stats: null, effort: p.effort, verbosity: p.verbosity, source: p.source, streaming: true, error: null, sources };
    addMessage(cid, reply, parentId);
    abort = new AbortController();
    let tokens = 0;
    let firstAt = 0;
    // Tokens are buffered and flushed once per animation frame, so long replies stay at 60 fps.
    let pending = "";
    let frame = 0;
    const flush = () => {
      frame = 0;
      if (!pending) return;
      const add = pending;
      pending = "";
      updateMsg(cid, mid, (m) => void (m.text += add));
    };
    try {
      await service.chat(
        request,
        (e) => {
          switch (e.type) {
            case "start":
              setLive({ tps: 0, ttft: e.ttftMs });
              break;
            case "thinking":
              updateMsg(cid, mid, (m) => void (m.thinking += e.text));
              break;
            case "token":
              tokens++;
              if (!firstAt) firstAt = performance.now();
              pending += e.text;
              if (!frame) frame = requestAnimationFrame(flush);
              setLive((l) => ({ ttft: l?.ttft ?? 0, tps: tokens / Math.max(0.05, (performance.now() - firstAt) / 1000) }));
              break;
            case "check":
              updateMsg(cid, mid, (m) => void m.checks.push(e.text));
              break;
            case "revise":
              pending = "";
              updateMsg(cid, mid, (m) => void (m.text = e.text));
              break;
            case "tool": {
              cancelAnimationFrame(frame);
              flush();
              const t: ToolActivity = { id: e.id, server: e.server, name: e.name, arguments: e.arguments, state: e.state, result: e.result, error: e.error };
              updateMsg(cid, mid, (m) => {
                const i = m.tools.findIndex((x) => x.id === t.id);
                if (i >= 0) m.tools[i] = t;
                else m.tools.push(t);
              });
              if (e.state === "approval") notify("A tool needs your approval", `${e.name} from ${e.server}. Answer in the chat.`, "info");
              if (e.state === "question") notify("The model has a question", "Answer it in the chat.", "info");
              if (e.state === "plan") notify("A plan is ready", "Approve it, or say what to change, in the chat.", "info");
              if (e.state === "done" || e.state === "denied" || e.state === "error")
                recordToolActivity({ at: Date.now(), toolId: mcpToolKey(e.server, e.name), summary: summarizeArgs(e.name, e.arguments), outcome: e.state === "done" ? "ran" : e.state === "denied" ? "deny" : "failed" });
              break;
            }
            case "ledger":
              updateMsg(cid, mid, (m) => {
                const l: LedgerState = m.ledger ?? { dir: null, step: null, files: {} };
                m.ledger = l;
                if (typeof e.dir === "string" && e.dir.length <= 1000) l.dir = e.dir;
                // The engine's roles and files come from a closed set; a foreign value is ignored.
                if (typeof e.step === "string" && /^(plan|ideate|manage|finalize|worker [1-6])$/.test(e.step) && Number.isInteger(e.round) && e.round >= 0) l.step = { role: e.step, round: e.round };
                if (e.file !== null && (LEDGER_FILES as readonly string[]).includes(e.file) && typeof e.content === "string") l.files[e.file as LedgerFileName] = e.content.slice(0, 20000);
              });
              break;
            case "image":
              updateMsg(cid, mid, (m) => void (m.image = { prompt: e.prompt, seed: e.seed }));
              break;
            case "done":
              cancelAnimationFrame(frame);
              flush();
              updateMsg(cid, mid, (m) => {
                m.stats = e.stats;
                m.streaming = false;
              });
              // Where it ran, so the speed under the reply can be checked against the hardware.
              void service
                .listLoaded()
                .then((list) => {
                  const on = ranOn(list.find((l) => l.slot === "llm")?.placement);
                  if (on) updateMsg(cid, mid, (m) => void (m.stats && (m.stats.ranOn = on)));
                })
                .catch(() => undefined);
              break;
            case "error":
              updateMsg(cid, mid, (m) => {
                m.error = e.message;
                m.streaming = false;
              });
          }
        },
        abort.signal,
      );
    } catch (err) {
      updateMsg(cid, mid, (m) => {
        m.error = err instanceof Error ? err.message : "The model stopped answering.";
        m.streaming = false;
      });
    } finally {
      cancelAnimationFrame(frame);
      flush();
      updateMsg(cid, mid, (m) => void (m.streaming = false));
      setLive(null);
      abort = null;
      setState("list", (c) => c.id === cid, "updatedAt", Date.now());
      persist();
    }
  };

  /**
   * A local command ("/bench") runs here, on this computer, and is never sent to the model. The
   * Benchmark lab is a view on the Monitor screen, so the hash says which one to open.
   */
  const runCommand = (name: string) => {
    if (name === "bench") {
      location.hash = hrefForMonitorTab("benchmark");
      notify("Benchmark lab", "Choose a model and run the benchmark on this computer.", "info");
      return;
    }
    if (name === "arena") {
      abort?.abort();
      setView("arena");
      notify("Arena", "Pick two models and ask them the same question.", "info");
    }
  };

  const run = async (cid: string, parentId: string | null) => {
    const conv = state.list.find((c) => c.id === cid);
    if (!conv || !modelId()) return;
    const cm = cloudOf();
    if (cm && app.settings()?.offlineMode) {
      notify("Offline mode is on", `${cm.provider} is a cloud model, so it is blocked. Turn offline mode off in Settings, or pick a model on this computer.`, "warning");
      return;
    }
    if (!cm && !(await ensureLoaded())) return;
    // Tools only when the person turned them on (or plans, which needs them): otherwise the system
    // prompt stays short.
    const tools = cm || !(prefs().toolsOn || prefs().planMode) ? null : await toolsForRequest();
    const req = built();
    const p = prefs();
    const path = branchPath(treeOf(conv));
    const at = parentId === null ? -1 : path.findIndex((n) => n.id === parentId);
    const context = (at >= 0 ? path.slice(0, at + 1) : path).map((n) => n.message);
    // The user's documents: the passages that answer the message go into the request, numbered.
    const question = [...context].reverse().find((m) => m.role === "user")?.text ?? "";
    const rag = docsContext(question);
    const turns: ChatTurn[] = context
      .filter((m) => !m.streaming && !m.error)
      .map((m) => ({ role: m.role, text: m.text, ...(m.attachments.length ? { attachments: m.attachments.map(({ name, type, bytes, preview }) => ({ name, type, bytes, ...(preview ? { data: preview } : {}) })) } : {}) }));
    await streamReply(
      cid,
      parentId,
      {
        modelId: modelId(),
        systemPrompt: rag ? [req.systemPrompt, ragPrompt(rag)].filter(Boolean).join("\n\n") : req.systemPrompt,
        turns,
        sampler: sampler(),
        native: req.native,
        passes: req.passes,
        effort: p.effort,
        verbosity: p.verbosity,
        source: p.source,
        tools,
        orchestrate: p.orchestrate ? { workers: p.workers, rounds: p.rounds } : null,
        keepCache: app.settings()?.keepChatCaches !== false,
      },
      rag?.sources ?? [],
    );
  };

  const send = (text: string, atts: Attachment[]) => {
    const cid = ensureConversation();
    const conv = state.list.find((c) => c.id === cid)!;
    const fact = memoryCandidate(text);
    if (fact && prefs().memoryOn && !conv.noMemory) {
      setMemories((m) => [...m, { id: newId(), text: fact, scope: "you" }]);
      notify("Remembered", fact);
    }
    const parentId = conv.messages[conv.messages.length - 1]?.id ?? null;
    const user: ChatMessage = {
      id: newId(),
      role: "user",
      text,
      thinking: "",
      checks: [],
      image: null,
      tools: [],
      ledger: null,
      attachments: atts,
      stats: null,
      effort: prefs().effort,
      verbosity: prefs().verbosity,
      source: prefs().source,
      streaming: false,
      error: null,
      sources: [],
    };
    batch(() => {
      if (!conv.messages.length) setState("list", (c) => c.id === cid, "title", titleFrom(text || atts[0]?.name || ""));
      addMessage(cid, user, parentId);
    });
    void run(cid, user.id);
  };

  /** Edits a user message into a new sibling branch; the old wording stays reachable. */
  const editMessage = (mid: string, text: string) => {
    const c = current();
    if (!c || streaming() || !text.trim()) return;
    const node = treeOf(c).nodes.find((n) => n.id === mid);
    if (!node || node.message.role !== "user") return;
    const original = node.message;
    const edited: ChatMessage = { ...original, id: newId(), text: text.trim(), thinking: "", checks: [], image: null, tools: [], ledger: null, stats: null, streaming: false, error: null };
    setState(
      "list",
      (x) => x.id === c.id,
      produce((x) => {
        const tree = treeOf(x);
        tree.nodes.push({ id: edited.id, parentId: node.parentId, message: edited });
        tree.selected[node.parentId ?? ""] = edited.id;
        x.messages = activeMessages(tree);
        x.updatedAt = Date.now();
      }),
    );
    persist();
    void run(c.id, edited.id);
  };

  /** A new reply to the same prompt, as a sibling of the reply being regenerated. */
  const regenerate = (mid: string) => {
    const c = current();
    if (!c || streaming()) return;
    const node = treeOf(c).nodes.find((n) => n.id === mid);
    if (!node || node.message.role !== "assistant" || node.parentId === null) return;
    void run(c.id, node.parentId);
  };

  const switchMessageBranch = (mid: string, delta: number) => {
    if (streaming()) return;
    const c = current();
    if (!c) return;
    const parentKey = treeOf(c).nodes.find((n) => n.id === mid)?.parentId ?? "";
    let nextId = "";
    setState(
      "list",
      (x) => x.id === c.id,
      produce((x) => {
        x.tree = switchBranch(treeOf(x), mid, delta);
        x.messages = activeMessages(x.tree);
        nextId = x.tree.selected[parentKey] ?? "";
      }),
    );
    persist();
    // Keep the keyboard where the switch happened: the message that is now shown.
    if (nextId)
      requestAnimationFrame(() => {
        const target = Array.from(thread.querySelectorAll<HTMLElement>("[data-message-id]")).find((el) => el.dataset.messageId === nextId);
        target?.focus({ preventScroll: true });
      });
  };

  const more = () => send("Tell me more.", []);

  /** A quick tool run from a / command: the tool's answer becomes a reply in this chat. */
  const runToolCommand = async (tool: QuickToolId, arg: string) => {
    const def = quickTool(tool);
    if (!def) return;
    const messages = current()?.messages ?? [];
    const lastUser = () => [...messages].reverse().find((m) => m.role === "user")?.text.trim() ?? "";
    const lastReply = () => [...messages].reverse().find((m) => m.role === "assistant" && m.text.trim())?.text.trim() ?? "";
    const input: QuickToolInputValue = {};
    if (tool === "translate") {
      input.language = TRANSLATE_LANGUAGES.find((l) => l.toLowerCase() === arg.trim().toLowerCase()) ?? "English";
      input.text = lastUser() || lastReply();
    } else if (tool === "rewrite") {
      input.style = REWRITE_STYLES.find((s) => s.id === arg.trim().toLowerCase())?.id ?? "clearer";
      input.text = lastUser() || lastReply();
    } else if (tool === "extract") {
      input.fields = arg.trim();
      input.text = lastUser() || lastReply();
    } else if (tool === "compare") {
      const users = messages.filter((m) => m.role === "user" && m.text.trim()).map((m) => m.text.trim());
      if (users.length < 2) return notify("/compare needs two texts", "Send two messages, then run /compare again.", "warning");
      input.text = users[users.length - 2] ?? "";
      input.second = users[users.length - 1] ?? "";
    } else if (tool === "describe") {
      const image = [...messages].reverse().flatMap((m) => m.attachments).find((a) => a.preview && a.type.startsWith("image/"));
      if (!image?.preview) return notify("/describe needs an image", "Attach a picture to a message, then run /describe.", "warning");
      input.attachment = { name: image.name, type: image.type, bytes: image.bytes, data: image.preview };
      input.text = arg.trim();
    } else if (tool === "read") {
      const text = arg.trim() || lastReply() || lastUser();
      if (!text) return notify("/read needs some text", "Type /help to see how each command is used.", "warning");
      return void service.readAloud(text).catch((e: unknown) => notify("Could not read aloud", e instanceof Error ? e.message : undefined, "error"));
    } else {
      input.text = arg.trim() || lastUser() || lastReply();
    }
    if (!input.text?.trim() && tool !== "describe") return notify(`/${def.command} needs some text`, "Type /help to see how each command is used.", "warning");
    const cm = cloudOf();
    if (cm && app.settings()?.offlineMode) {
      notify("Offline mode is on", `${cm.provider} is a cloud model, so it is blocked. Turn offline mode off in Settings, or pick a model on this computer.`, "warning");
      return;
    }
    if (!cm && !(await ensureLoaded())) return;
    const prompt = buildQuickToolPrompt(tool, input);
    const cid = ensureConversation();
    const parentId = state.list.find((c) => c.id === cid)?.messages.slice(-1)[0]?.id ?? null;
    await streamReply(cid, parentId, {
      modelId: modelId(),
      systemPrompt: prompt.system,
      turns: [{ role: "user", text: prompt.user || "Do the task." }],
      sampler: sampler(),
      native: null,
      passes: 0,
      effort: prefs().effort,
      verbosity: prefs().verbosity,
      source: prefs().source,
      tools: null,
      orchestrate: null,
      keepCache: app.settings()?.keepChatCaches !== false,
    });
  };

  /** The files the chat may search: everything, or one collection, folder or tag. */
  const docsScopes = createMemo(() => {
    const docs = loadDocs();
    const out: { key: string; label: string }[] = [{ key: "all", label: "All documents" }];
    for (const c of collections) out.push({ key: scopeKey({ kind: "collection", id: c.id }), label: `Collection: ${c.name}` });
    for (const f of docFolderNames(docs)) out.push({ key: scopeKey({ kind: "folder", name: f }), label: `Folder: ${f}` });
    for (const t of docTagNames(docs)) out.push({ key: scopeKey({ kind: "tag", name: t }), label: `Tag: ${t}` });
    return out;
  });

  /** Where the passages go when a cloud model answers. */
  const docsNote = createMemo(() => {
    const c = cloudOf();
    if (!docsPrefs().on || !c) return "";
    return `Your passages are sent to ${c.provider}, with your own API key.`;
  });

  /** Opens a passage of the user's documents in the Documents screen. */
  const openSource = (s: Source) => {
    window.location.hash = `${hrefFor("documents")}?doc=${encodeURIComponent(s.docId)}&p=${encodeURIComponent(String(s.index))}`;
  };

  /** A / command from the composer (lib/chat/commands.ts). Runs here; nothing reaches the model. */
  const command = (name: string, arg: string) => {
    const lastReply = () => [...(current()?.messages ?? [])].reverse().find((m) => m.role === "assistant" && m.text.trim());
    const need = (what: string) => notify(`/${name} needs ${what}`, `Type /help to see how each command is used.`, "warning");
    switch (name) {
      case "new":
        return newChat();
      case "clear": {
        const c = current();
        if (!c || !c.messages.length) return notify("Nothing to clear");
        if (streaming()) abort?.abort();
        const kept = c.messages;
        setState("list", (x) => x.id === c.id, "messages", []);
        persist();
        return notifyAction("Chat cleared", `${kept.length} message${kept.length === 1 ? "" : "s"} removed.`, "Undo", () => {
          setState("list", (x) => x.id === c.id, "messages", kept);
          persist();
        });
      }
      case "retry":
        if (!lastReply()) return notify("No reply to write again");
        return regenerate(lastReply()!.id);
      case "model": {
        if (!arg) return need("a model name");
        const q = arg.toLowerCase();
        const m = models().find((x) => x.name.toLowerCase() === q) ?? models().find((x) => x.name.toLowerCase().includes(q) || x.id.toLowerCase().includes(q));
        if (!m) return notify(`No model matches "${arg}"`, "Add it from the Model Hub, or check its name on Models.", "warning");
        setModelId(m.id);
        return notify(`Model: ${m.name}`);
      }
      case "speed": {
        const lvl = speedArg(arg);
        if (!lvl) return need("a level: 1 to 5, or fastest, fast, balanced, accurate or maximum");
        return pickMode(lvl);
      }
      case "think": {
        const on = switchArg(arg);
        if (on === null) return need("on or off");
        setPrefs({ thinking: on ? (prefs().thinking === "off" ? "low" : prefs().thinking) : "off" });
        return notify(on ? "Thinking on" : "Thinking off", on ? "The model thinks before it answers." : "The model answers straight away.");
      }
      case "system":
        setPrefs({ base: arg || DEFAULT_BASE_PROMPT });
        return notify(arg ? "System prompt set" : "System prompt reset", arg ? arg.slice(0, 120) : undefined);
      case "temp": {
        const v = numberArg(arg, 0, 2);
        if (v === null) return need("a number from 0 to 2");
        setSampler((s) => ({ ...s, temperature: v }));
        return notify(`Temperature ${v}`);
      }
      case "max": {
        const v = numberArg(arg, 16, 131072, true);
        if (v === null) return need("a whole number of tokens from 16 to 131,072");
        setSampler((s) => ({ ...s, maxTokens: v }));
        return notify(`Longest reply: ${v.toLocaleString()} tokens`);
      }
      case "image":
        if (!arg) return need("a description of the picture");
        return void makeImage(arg);
      case "say": {
        const r = lastReply();
        if (!r) return notify("No reply to read aloud");
        return void service.readAloud(plainText(parseMarkdown(r.text))).catch((e: unknown) => notify("Could not read aloud", e instanceof Error ? e.message : undefined, "error"));
      }
      case "export": {
        const c = current();
        if (!c || !c.messages.length) return notify("Nothing to export yet");
        const kind = arg.toLowerCase() === "json" ? "json" : arg.toLowerCase() === "md" || !arg ? "md" : null;
        if (!kind) return need("md or json");
        const base = (c.title || "chat").replace(/[^\w -]+/g, "").trim().slice(0, 60) || "chat";
        const ok =
          kind === "json"
            ? saveText(`${base}.json`, "application/json", JSON.stringify({ title: c.title, messages: c.messages.map((m) => ({ role: m.role, text: m.text })) }, null, 2))
            : saveText(`${base}.md`, "text/markdown", `# ${c.title}\n\n` + c.messages.map((m) => `**${m.role === "user" ? "You" : "Model"}:**\n\n${m.text}`).join("\n\n---\n\n"));
        return notify(ok ? "Chat exported" : "Could not export", ok ? `${base}.${kind}` : undefined, ok ? "success" : "error");
      }
      case "copy": {
        const r = lastReply();
        if (!r) return notify("No reply to copy");
        return void navigator.clipboard.writeText(r.text).then(
          () => notify("Copied the last reply"),
          () => notify("Could not copy", undefined, "error"),
        );
      }
      case "plan":
        setPrefs({ planMode: !prefs().planMode });
        return notify(prefs().planMode ? "Plan first is on" : "Plan first is off");
      case "orchestrate": {
        const o = orchestrateArg(arg);
        if (!o) return need("off, or workers (1-6) and rounds (1-8)");
        if (!o.on) {
          setPrefs({ orchestrate: false });
          return notify("Orchestrate off");
        }
        setPrefs({ orchestrate: true, ...(o.workers ? { workers: o.workers } : {}), ...(o.rounds ? { rounds: o.rounds } : {}) });
        return notify("Orchestrate on", `${prefs().workers} workers, up to ${prefs().rounds} rounds.`);
      }
      case "remember":
        if (!arg) return need("a fact to remember");
        setMemories((m) => [...m, { id: newId(), text: arg.slice(0, 500), scope: "you" }]);
        return notify("Remembered", arg.slice(0, 120));
      case "memories":
        return setPanel("memory");
      case "stats": {
        const s = (current()?.messages ?? []).map((m) => m.stats).filter((x): x is NonNullable<typeof x> => x !== null);
        if (!s.length) return notify("No replies in this chat yet");
        const out = s.reduce((n, x) => n + x.outputTokens, 0);
        const secs = s.reduce((n, x) => n + x.totalMs, 0) / 1000;
        const tps = s.reduce((n, x) => n + x.tokPerSec, 0) / s.length;
        return notify("This chat", `${s.length} replies, ${out.toLocaleString()} tokens in ${secs.toFixed(1)} s, ${tps.toFixed(1)} tokens per second on average.`);
      }
      case "help":
        return setHelpOpen(true);
      case "voice":
        return setVoice(true);
      case "bench":
      case "arena":
        return runCommand(name);
      case "prompts":
        return setPanel("prompts");
      case "docs": {
        const docs = loadDocs();
        const q = arg.trim().toLowerCase();
        if (!q) {
          const on = !docsPrefs().on;
          setDocsPrefs({ on });
          return notify(on ? "Using your documents" : "Not using your documents", on ? `${docs.length} indexed file${docs.length === 1 ? "" : "s"} can be searched. /docs <name> searches one collection, folder or tag.` : "Chat answers without your documents.");
        }
        if (q === "off" || q === "no") {
          setDocsPrefs({ on: false });
          return notify("Not using your documents");
        }
        const collection = collections.find((c) => c.name.toLowerCase() === q);
        if (collection) {
          setDocsPrefs({ on: true, scope: scopeKey({ kind: "collection", id: collection.id }) });
          return notify(`Using your documents: ${collection.name}`);
        }
        const folder = docFolderNames(docs).find((f) => f.toLowerCase() === q);
        if (folder) {
          setDocsPrefs({ on: true, scope: scopeKey({ kind: "folder", name: folder }) });
          return notify(`Using your documents: the folder ${folder}`);
        }
        const tag = docTagNames(docs).find((t) => t.toLowerCase() === q);
        if (tag) {
          setDocsPrefs({ on: true, scope: scopeKey({ kind: "tag", name: tag }) });
          return notify(`Using your documents: the tag ${tag}`);
        }
        setDocsPrefs({ on: true, scope: "all" });
        return notify("No collection, folder or tag with that name", "Searching all your documents instead. Give a file a folder or a tag on the Documents screen.", "warning");
      }
      case "summarise":
      case "translate":
      case "rewrite":
      case "extract":
      case "explain":
      case "compare":
      case "notes":
      case "read":
      case "describe":
        return void runToolCommand(name as QuickToolId, arg);
    }
  };
  const [helpOpen, setHelpOpen] = createSignal(false);
  const [voice, setVoice] = createSignal(false);

  /** Voice mode's turn: send what was said, and resolve with the reply once it has finished. */
  const askAndWait = (text: string): Promise<string> =>
    new Promise((resolve, reject) => {
      send(text, []);
      const t0 = Date.now();
      const tick = setInterval(() => {
        if (streaming() && Date.now() - t0 < 600_000) return;
        if (Date.now() - t0 < 300) return;
        clearInterval(tick);
        const last = [...(current()?.messages ?? [])].reverse().find((m) => m.role === "assistant");
        if (last?.error) reject(new Error(last.error));
        else resolve(last?.text ?? "");
      }, 150);
    });

  /** /image: on the desktop, the first image model in the library makes it; the browser preview
   *  draws its illustrative image through the mock chat as before. */
  const makeImage = async (prompt: string) => {
    if (!__DESKTOP__) return send(`/imagine ${prompt}`, []);
    const lib = library() ?? [];
    const im = lib.find((m) => m.kind === "image");
    if (!im) return notify("No image model yet", "Add an image model (such as SD-Turbo) from the Model Hub to make images.", "warning");
    const cid = ensureConversation();
    const mid = newId();
    setState("list", (c) => c.id === cid, "messages", (l) => [
      ...l,
      { id: newId(), role: "user", text: `/image ${prompt}`, thinking: "", checks: [], image: null, tools: [], ledger: null, attachments: [], stats: null, effort: prefs().effort, verbosity: prefs().verbosity, source: prefs().source, streaming: false, error: null, sources: [] } satisfies ChatMessage,
      { id: mid, role: "assistant", text: "Making the image…", thinking: "", checks: [], image: null, tools: [], ledger: null, attachments: [], stats: null, effort: prefs().effort, verbosity: prefs().verbosity, source: prefs().source, streaming: true, error: null, sources: [] } satisfies ChatMessage,
    ]);
    try {
      const g = await service.generateImage({ modelId: im.id, prompt, size: 512, steps: 1, seed: null });
      updateMsg(cid, mid, (m) => {
        m.text = `Made with ${im.name} in ${g.seconds.toFixed(1)} s (${g.seconds > 0 ? (1 / g.seconds).toFixed(2) : "–"} steps/s, 512×512), seed ${g.seed}.`;
        m.attachments = [{ name: "image.png", type: "image/png", bytes: Math.round((g.dataUrl.length * 3) / 4), preview: g.dataUrl }];
        m.streaming = false;
      });
    } catch (e) {
      updateMsg(cid, mid, (m) => {
        m.text = "";
        m.error = e instanceof Error ? e.message : "The image could not be made.";
        m.streaming = false;
      });
    }
    persist();
  };

  const newChat = () => {
    abort?.abort();
    const c = emptyConversation();
    setState("list", (l) => [c, ...l]);
    setCurrentId(c.id);
    setListQuery("");
    persist();
  };

  const remove = (id: string) => {
    if (id === currentId()) abort?.abort();
    setState("list", (l) => l.filter((c) => c.id !== id));
    if (id === currentId()) setCurrentId(state.list[0]?.id ?? "");
    persist();
  };

  const setPinned = (id: string, pinned: boolean) => {
    setState("list", (c) => c.id === id, "pinned", pinned);
    persist();
  };

  const setArchived = (id: string, archived: boolean) => {
    setState("list", (c) => c.id === id, "archived", archived);
    if (id === currentId() && archived && !showArchived()) {
      const next = state.list.find((c) => c.id !== id && !c.archived);
      setCurrentId(next?.id ?? "");
    }
    persist();
  };

  const setFolder = (id: string, value: string) => {
    const folder = value.trim().slice(0, 60);
    setState("list", (c) => c.id === id, "folder", folder || null);
    persist();
  };

  const setTags = (id: string, value: string) => {
    const tags = [...new Set(value.split(",").map((t) => t.trim().slice(0, 30)).filter(Boolean))].slice(0, 8);
    setState("list", (c) => c.id === id, "tags", tags);
    persist();
  };

  const folderNames = createMemo(() => [...new Set(state.list.flatMap((c) => (c.folder ? [c.folder] : [])))].sort((a, b) => a.localeCompare(b)));
  const tagNames = createMemo(() => [...new Set(state.list.flatMap((c) => c.tags))].sort((a, b) => a.localeCompare(b)));
  const searchResults = createMemo(() => (listQuery().trim() ? searchConversations(state.list, listQuery()) : []));
  const visibleConversations = createMemo(() =>
    [...state.list]
      .filter((c) => {
        if (c.archived !== showArchived()) return false;
        if (folderFilter() === "__none__" ? c.folder !== null : folderFilter() !== "all" && c.folder !== folderFilter()) return false;
        if (tagFilter() !== "all" && !c.tags.includes(tagFilter())) return false;
        return true;
      })
      .sort((a, b) => Number(b.pinned) - Number(a.pinned) || b.updatedAt - a.updatedAt),
  );

  /** Opens a chat (and a message in it), switching to that message's branch when needed. */
  const jumpTo = (conversationId: string, messageId: string | null) => {
    if (!state.list.some((c) => c.id === conversationId)) {
      const loaded = loadConversations();
      if (loaded.length) setState("list", loaded);
    }
    setCurrentId(conversationId);
    if (messageId) {
      setState(
        "list",
        (c) => c.id === conversationId,
        produce((c) => {
          if (!c.messages.some((m) => m.id === messageId)) {
            c.tree = focusBranch(treeOf(c), messageId);
            c.messages = activeMessages(c.tree);
          }
        }),
      );
      setFoundId(messageId);
      clearTimeout(foundTimer);
      foundTimer = setTimeout(() => setFoundId(""), 2600);
    }
    setListQuery("");
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        if (!thread) return;
        if (!messageId) {
          thread.scrollTop = 0;
          return;
        }
        const target = Array.from(thread.querySelectorAll<HTMLElement>("[data-message-id]")).find((el) => el.dataset.messageId === messageId);
        if (!target) return;
        target.scrollIntoView({ block: "center" });
        target.focus({ preventScroll: true });
      }),
    );
  };

  createEffect(
    on(chatJump, (jump) => {
      if (!jump) return;
      jumpTo(jump.conversationId, jump.messageId);
      setChatJump(null);
    }),
  );

  let thread!: HTMLDivElement;
  createEffect(
    on(
      () => current()?.messages.map((m) => m.text.length).join(),
      () => {
        if (!thread) return;
        const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 160;
        if (nearBottom) thread.scrollTop = thread.scrollHeight;
      },
    ),
  );

  const panelLabel = () => (panel() === "settings" ? "Chat settings" : panel() === "memory" ? "Memory" : panel() === "prompt" ? "System prompt" : "Prompt library");

  return (
    <div class="screen chat" data-panel={view() === "chat" && panel() !== "none" ? "open" : "closed"} data-view={view()}>
      <aside class="chat-list" aria-label="Conversations">
        <button class="btn btn--tertiary chat-list__new" type="button" onClick={newChat} data-press>
          <Icon name="plus" size={16} />
          <span class="btn__label">New chat</span>
        </button>
        <div class="chat-list__search">
          <Icon name="search" size={14} />
          <input
            class="input"
            type="search"
            value={listQuery()}
            placeholder="Search chats"
            aria-label="Search chats and messages"
            maxlength="200"
            onInput={(e) => setListQuery(e.currentTarget.value)}
          />
          <Show when={listQuery()}>
            <button class="icon-btn icon-btn--sm" type="button" aria-label="Clear search" onClick={() => setListQuery("")}>
              <Icon name="close" size={12} />
            </button>
          </Show>
        </div>

        <Show
          when={listQuery().trim()}
          fallback={
            <>
              <details class="chat-list__filtermenu">
                <summary class="chat-list__filtersummary">
                  <Icon name="sliders" size={13} /> Filter
                </summary>
              <div class="chat-list__filters">
                <label class="chat-list__filter">
                  <span class="sr-only">Folder</span>
                  <select class="input" value={folderFilter()} aria-label="Filter by folder" onChange={(e) => setFolderFilter(e.currentTarget.value)}>
                    <option value="all">All folders</option>
                    <option value="__none__">No folder</option>
                    <For each={folderNames()}>{(f) => <option value={f}>{f}</option>}</For>
                  </select>
                </label>
                <label class="chat-list__filter">
                  <span class="sr-only">Tag</span>
                  <select class="input" value={tagFilter()} aria-label="Filter by tag" onChange={(e) => setTagFilter(e.currentTarget.value)}>
                    <option value="all">All tags</option>
                    <For each={tagNames()}>{(t) => <option value={t}>{t}</option>}</For>
                  </select>
                </label>
                <label class="chat-list__archived">
                  <input type="checkbox" checked={showArchived()} onChange={(e) => setShowArchived(e.currentTarget.checked)} />
                  <span>Archived</span>
                </label>
              </div>
              </details>
              <datalist id="chat-folder-options">
                <For each={folderNames()}>{(f) => <option value={f} />}</For>
              </datalist>
              <ul role="list">
                <For each={visibleConversations()} fallback={<li class="hint">{state.list.length ? "No chats in this view." : "No conversations yet."}</li>}>
                  {(c) => (
                    <li class="chat-list__item" data-current={c.id === currentId()} data-pinned={c.pinned}>
                      <button type="button" class="chat-list__open" onClick={() => setCurrentId(c.id)} aria-current={c.id === currentId() ? "true" : undefined}>
                        <Show when={c.pinned}>
                          <Icon name="pin" size={12} class="chat-list__pin" />
                        </Show>
                        <span>{c.title}</span>
                      </button>
                      <button
                        type="button"
                        class="icon-btn icon-btn--sm"
                        aria-label={c.pinned ? `Unpin ${c.title}` : `Pin ${c.title}`}
                        aria-pressed={c.pinned}
                        onClick={() => setPinned(c.id, !c.pinned)}
                      >
                        <Icon name="pin" size={14} />
                      </button>
                      <details class="chat-list__options">
                        <summary class="icon-btn icon-btn--sm" aria-label={`Chat options for ${c.title}`} title="Chat options">
                          <Icon name="sliders" size={14} />
                        </summary>
                        <div class="chat-list__options-body">
                          <label class="chat-list__field">
                            <span>Folder</span>
                            <input class="input" list="chat-folder-options" value={c.folder ?? ""} maxlength="60" placeholder="No folder" onChange={(e) => setFolder(c.id, e.currentTarget.value)} />
                          </label>
                          <label class="chat-list__field">
                            <span>Tags, separated by commas</span>
                            <input class="input" value={c.tags.join(", ")} maxlength="260" placeholder="No tags" onChange={(e) => setTags(c.id, e.currentTarget.value)} />
                          </label>
                          <label class="composer__check">
                            <input type="checkbox" checked={c.archived} onChange={(e) => setArchived(c.id, e.currentTarget.checked)} />
                            <span>Archived</span>
                          </label>
                        </div>
                      </details>
                      <button type="button" class="icon-btn icon-btn--sm" aria-label={`Delete ${c.title}`} onClick={() => remove(c.id)}>
                        <Icon name="trash" size={14} />
                      </button>
                    </li>
                  )}
                </For>
              </ul>
            </>
          }
        >
          <div class="chat-list__results-head">
            <span role="status">
              {searchResults().length} {searchResults().length === 1 ? "result" : "results"}
            </span>
            <button class="btn btn--tertiary btn--sm" type="button" onClick={() => setListQuery("")}>
              Clear
            </button>
          </div>
          <ul class="chat-list__results" role="list">
            <For each={searchResults()} fallback={<li class="hint">No chats match “{listQuery().trim()}”.</li>}>
              {(result) => (
                <li>
                  <button class="chat-list__result" type="button" onClick={() => jumpTo(result.conversationId, result.messageId)}>
                    <span class="chat-list__result-title">
                      <HighlightQuery text={result.conversationTitle} query={listQuery()} />
                    </span>
                    <span class="chat-list__result-where">{result.messageId === null ? "Chat title" : result.role === "user" ? "You" : "Reply"}</span>
                    <Show when={result.messageId !== null}>
                      <span class="chat-list__result-snippet">
                        <HighlightQuery text={result.snippet} query={listQuery()} />
                      </span>
                    </Show>
                  </button>
                </li>
              )}
            </For>
          </ul>
        </Show>
      </aside>

      <section class="chat-main" aria-label={view() === "arena" ? "Arena" : "Chat"} data-view={view()}>
        <header class="chat-head">
          <h1 class="chat-head__title sr-only">{view() === "chat" ? "Chat" : "Arena"}</h1>
          <div class="chat-tabs" role="group" aria-label="Chat view">
            <button class="chat-tabs__tab" type="button" aria-pressed={view() === "chat"} onClick={() => setView("chat")} data-testid="chat-tab">
              Chat
            </button>
            <button
              class="chat-tabs__tab"
              type="button"
              aria-pressed={view() === "arena"}
              onClick={() => {
                // A reply belongs to the conversation view; leaving it stops it rather than
                // streaming unseen behind the Arena.
                abort?.abort();
                setView("arena");
              }}
              data-testid="arena-tab"
            >
              Arena
            </button>
          </div>
          <label class="chat-head__model">
            <span class="sr-only">Model</span>
            <select value={modelId()} onChange={(e) => setModelId(e.currentTarget.value)} aria-label="Model">
              <Show when={cloudModels().length} fallback={<For each={models()}>{(m) => <option value={m.id}>{m.name}</option>}</For>}>
                <optgroup label="On this computer">
                  <For each={models()}>{(m) => <option value={m.id}>{m.name}</option>}</For>
                </optgroup>
                <optgroup label="Cloud (your API keys)">
                  <For each={cloudModels()}>{(m) => <option value={m.id}>{`${m.name} · ${m.provider}`}</option>}</For>
                </optgroup>
              </Show>
            </select>
          </label>
          <Show when={cloudOf()}>
            {(c) => (
              <span class="chat-head__cloud" title={`This chat is sent to ${c().provider} over the internet with your API key.`} data-testid="cloud-badge">
                <Icon name="online" size={14} /> Sent to {c().provider}
              </span>
            )}
          </Show>

          <span class="chat-head__live mono" aria-live="off" data-testid="live-readout">
            <Show when={live()}>
              {(l) => (
                <>
                  {l().tps.toFixed(1)} tok/s · first token {l().ttft} ms
                </>
              )}
            </Show>
          </span>
          {/* The web preview runs on example data; the desktop app's figures are measured. */}
          <Show when={!__DESKTOP__}>
            <IllustrativeFlag />
          </Show>
          <button class="icon-btn" type="button" aria-label="Voice mode" title="Voice mode: talk hands-free" onClick={() => setVoice(true)}>
            <Icon name="mic" size={18} />
          </button>
          <button
            class="icon-btn"
            type="button"
            aria-label="Chat settings and side panel"
            title="Settings, live figures, prompts and memory"
            aria-pressed={panel() !== "none"}
            onClick={() => setPanel(panel() === "none" ? "settings" : "none")}
            data-testid="panel-toggle"
          >
            <Icon name="sliders" size={18} />
          </button>
        </header>

        <div class="chat-view" hidden={view() !== "chat"}>
        <div class="chat-thread" ref={thread} role="log" aria-live="polite" aria-relevant="additions">
          <Show when={current()?.messages.length} fallback={<Empty onPick={(t) => send(t, [])} />}>
            {/* The transcript's own heading: replies' Markdown headings (h3, h4) nest under it. */}
            <h2 class="sr-only">Conversation</h2>
            <For each={current()!.messages}>
              {(m, i) => (
                <MessageView
                  m={m}
                  last={i() === current()!.messages.length - 1}
                  branch={branchPosition(current()!.tree, m.id)}
                  found={foundId() === m.id}
                  busy={streaming()}
                  onRegenerate={regenerate}
                  onMore={more}
                  onEdit={editMessage}
                  onBranch={switchMessageBranch}
                  onToolDecide={(id, d) => decideTool(currentId(), m.id, id, d)}
                  onToolAnswer={answerTool}
                  onOpenSource={openSource}
                />
              )}
            </For>
          </Show>
        </div>

        <Show when={voice()}>
          <VoiceMode onAsk={askAndWait} onClose={() => setVoice(false)} />
        </Show>
        <Dialog.Root open={helpOpen()} onOpenChange={(d) => setHelpOpen(d.open)} lazyMount unmountOnExit>
          <Portal>
            <Dialog.Backdrop class="guide-backdrop" />
            <Dialog.Positioner class="guide-positioner">
              <Dialog.Content class="guide card" data-testid="command-help">
                <header class="guide__head">
                  <div>
                    <Dialog.Title class="guide__title">Chat commands</Dialog.Title>
                    <Dialog.Description class="guide__desc">Type / at the start of a message. Commands run here and are never sent to the model. Start a message with // to send a slash as text.</Dialog.Description>
                  </div>
                  <Dialog.CloseTrigger class="icon-btn" aria-label="Close">
                    <Icon name="close" size={18} />
                  </Dialog.CloseTrigger>
                </header>
                <div class="guide__body">
                  <dl class="command-help">
                    <For each={COMMANDS}>
                      {(c) => (
                        <>
                          <dt class="mono">
                            /{c.name}
                            <Show when={c.args}>
                              <span class="command-menu__args"> {c.args}</span>
                            </Show>
                          </dt>
                          <dd>{c.description}</dd>
                        </>
                      )}
                    </For>
                  </dl>
                </div>
              </Dialog.Content>
            </Dialog.Positioner>
          </Portal>
        </Dialog.Root>
        <Composer
          levels={levels}
          mode={modeOf(prefs(), sampler())}
          onMode={(m) => pickMode(m, false)}
          prefs={prefs()}
          setPrefs={setPrefs}
          streaming={streaming()}
          notes={built().notes}
          onSend={send}
          onCommand={command}
          onStop={() => abort?.abort()}
          onTranscribe={(s) => service.transcribe(s)}
          onRecordStart={() => service.startRecording()}
          onLevel={() => service.recordingLevel()}
          onOpenPrompt={() => setPanel("prompt")}
          onOpenPrompts={() => setPanel("prompts")}
          docsOn={docsPrefs().on}
          onDocs={(on) => setDocsPrefs({ on })}
          docsScopes={docsScopes()}
          docsScope={docsPrefs().scope}
          onDocsScope={(scope) => setDocsPrefs({ scope })}
          docsNote={docsNote()}
          draft={draft()}
        />
        </div>

        <div class="chat-view" hidden={view() !== "arena"}>
          <Arena models={models()} active={view() === "arena"} />
        </div>
      </section>

      <Show when={view() === "chat" && panel() !== "none"}>
        <aside
          class="chat-side card"
          classList={{ "chat-side--dock": panel() === "dock" }}
          style={panel() === "dock" ? { width: `${dockWidth()}px` } : undefined}
          aria-label={panel() === "dock" ? "Live panel" : panelLabel()}
        >
          <Show when={panel() === "dock"}>
            <div class="chat-side__resize" role="separator" aria-orientation="vertical" aria-label="Resize the live panel" aria-valuemin={280} aria-valuemax={520} aria-valuenow={dockWidth()} tabindex="0" onPointerDown={startResize} onKeyDown={(e) => {
              if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
                e.preventDefault();
                setDockWidth((w) => Math.min(520, Math.max(280, w + (e.key === "ArrowLeft" ? 20 : -20))));
                writePref("cerebro-chat-dock-width", String(dockWidth()));
              }
            }} />
          </Show>
          <div class="chat-side__head">
            <div class="chat-side__tabs" role="group" aria-label="Side panel">
              <For each={[["settings", "Settings", "Chat settings"], ["dock", "Live", "Live panel (Ctrl .)"], ["prompts", "Prompts", "Prompt library"], ["memory", "Memory", "Memory"]] as const}>
                {([id, label, name]) => (
                  <button class="chat-side__tab" type="button" aria-label={name} aria-pressed={panel() === id || (id === "prompts" && panel() === "prompt")} onClick={() => setPanel(panel() === id ? "none" : id)} data-testid={`side-tab-${id}`}>
                    {label}
                  </button>
                )}
              </For>
            </div>
            <button class="icon-btn" type="button" aria-label="Close panel" onClick={() => setPanel("none")}>
              <Icon name="close" size={16} />
            </button>
          </div>
          <h2 class="chat-side__title">{panel() === "dock" ? "Live panel" : panelLabel()}</h2>
          <Show when={panel() === "settings"}>
            <div class="chat-side__body">
              <ChatSettings
                thinkingSupported={supportsNativeReasoning(`${model()?.family ?? ""} ${model()?.name ?? ""}`)}
                levels={levels}
                mode={modeOf(prefs(), sampler())}
                onMode={(m) => pickMode(m, false)}
                prefs={prefs()}
                setPrefs={setPrefs}
                notes={built().notes}
                onOpenPrompt={() => setPanel("prompt")}
                onOpenPrompts={() => setPanel("prompts")}
                docsOn={docsPrefs().on}
                onDocs={(on) => setDocsPrefs({ on })}
                docsScopes={docsScopes()}
                docsScope={docsPrefs().scope}
                onDocsScope={(scope) => setDocsPrefs({ scope })}
                docsNote={docsNote()}
              />
              <div class="chat-settings__sampler">
                <h3 class="chat-settings__label">Sampling</h3>
                <SamplerMenu sampler={sampler()} onChange={setSampler} inline />
              </div>
            </div>
          </Show>
          <Show when={panel() === "dock"}>
            <LiveDock
              mode={modeOf(prefs(), sampler())}
              onMode={(m) => pickMode(m, false)}
              sampler={sampler()}
              setSampler={setSampler}
              thinking={prefs().thinking !== "off"}
              setThinking={(on) => setPrefs({ thinking: on ? (prefs().thinking === "off" ? "low" : prefs().thinking) : "off" })}
              onOpenPrompt={() => setPanel("prompt")}
              hasPreset={presetIds().has(modelId())}
              onSavePreset={() => {
                savePreset(modelId(), prefs(), sampler());
                setPresetIds(new Set(Object.keys(loadPresets())));
                notify("Saved for this model", "These settings come back whenever you pick it.", "success");
              }}
              onClearPreset={() => {
                removePreset(modelId());
                setPresetIds(new Set(Object.keys(loadPresets())));
                notify("Forgotten", "This model uses your general settings again.");
              }}
              onUnload={() => {
                const id = modelId();
                void service.unloadModel(id).then(
                  () => notify("Model unloaded", "Its memory is free. It loads again with your next message."),
                  (e: unknown) => notify("Could not unload", e instanceof Error ? e.message : undefined, "error"),
                );
              }}
            />
          </Show>
          <Show when={panel() !== "dock" && panel() !== "settings"}>          <Show
            when={panel() === "memory"}
            fallback={
              <Show
                when={panel() === "prompt"}
                fallback={
                  <PromptLibrary
                    prompts={prompts()}
                    setPrompts={setPrompts}
                    onUse={(text) => {
                      setDraft({ text, at: Date.now() });
                      setPanel("none");
                      notify("Prompt ready", "Fill in any details, then send it.", "info");
                    }}
                  />
                }
              >
                <PromptPanel prefs={prefs()} setPrefs={setPrefs} preview={built().systemPrompt} native={built().native} />
              </Show>
            }
          >
            <MemoryPanel
              memories={memories()}
              setMemories={setMemories}
              globalOn={prefs().memoryOn}
              setGlobalOn={(v) => setPrefs({ memoryOn: v })}
              chatOff={current()?.noMemory ?? false}
              setChatOff={(v) => {
                const cid = ensureConversation();
                setState("list", (c) => c.id === cid, "noMemory", v);
                persist();
              }}
            />
          </Show>
        </Show>
        </aside>
      </Show>
    </div>
  );
}

function Empty(props: { onPick: (t: string) => void }): JSX.Element {
  const ideas = ["Why is my 30B model slower than the 8B?", "Write a Python function that counts lines in a huge file", "/imagine a lighthouse at dusk, oil painting"];
  return (
    <div class="chat-empty">
      <h2>What can I help with?</h2>
      <p class="chat-empty__sub">Everything runs on this computer.</p>
      <ul role="list">
        <For each={ideas}>
          {(t) => (
            <li>
              <button class="chat-empty__idea" type="button" onClick={() => props.onPick(t)}>
                {t}
              </button>
            </li>
          )}
        </For>
      </ul>
    </div>
  );
}

function SamplerMenu(props: { sampler: Sampler; onChange: (s: Sampler) => void; inline?: boolean }): JSX.Element {
  const field = (k: keyof Sampler, label: string, min: number, max: number, step: number) => (
    <label class="sampler__field">
      <span>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={props.sampler[k]}
        onChange={(e) => {
          const v = Number(e.currentTarget.value);
          if (Number.isFinite(v)) props.onChange({ ...props.sampler, [k]: Math.min(max, Math.max(min, v)) });
        }}
      />
    </label>
  );
  const reset = (
    <button class="btn btn--tertiary btn--sm" type="button" onClick={() => props.onChange({ ...DEFAULT_SAMPLER })}>
      Reset
    </button>
  );
  if (props.inline) {
    // In the side panel: the fields themselves, two to a row.
    return (
      <div class="sampler sampler--inline" aria-label="Sampler settings" role="group">
        {field("temperature", "Temperature", 0, 2, 0.05)}
        {field("topP", "Top P", 0, 1, 0.01)}
        {field("minP", "Min P", 0, 1, 0.01)}
        {field("topK", "Top K", 0, 200, 1)}
        {field("repeatPenalty", "Repeat penalty", 1, 2, 0.01)}
        {field("maxTokens", "Max tokens", 16, 32768, 16)}
        {reset}
      </div>
    );
  }
  return (
    <details class="sampler">
      <summary class="icon-btn" aria-label="Sampler settings">
        <Icon name="sliders" size={18} />
      </summary>
      <div class="sampler__pop card">
        {field("temperature", "Temperature", 0, 2, 0.05)}
        {field("topP", "Top P", 0, 1, 0.01)}
        {field("minP", "Min P", 0, 1, 0.01)}
        {field("topK", "Top K", 0, 200, 1)}
        {field("repeatPenalty", "Repeat penalty", 1, 2, 0.01)}
        {field("maxTokens", "Max tokens", 16, 32768, 16)}
        <button class="btn btn--tertiary btn--sm" type="button" onClick={() => props.onChange({ ...DEFAULT_SAMPLER })}>
          Reset
        </button>
      </div>
    </details>
  );
}

function MemoryPanel(props: {
  memories: Memory[];
  setMemories: (f: (m: Memory[]) => Memory[]) => void;
  globalOn: boolean;
  setGlobalOn: (v: boolean) => void;
  chatOff: boolean;
  setChatOff: (v: boolean) => void;
}): JSX.Element {
  const [draft, setDraft] = createSignal("");
  return (
    <div class="memory">
      <SettingSwitch id="memory-global" label="Memory" note="Use what Cerebro remembers in every chat. Stored only on this computer." checked={props.globalOn} onChange={props.setGlobalOn} />
      <SettingSwitch id="memory-chat" label="Don't remember this chat" note="Nothing from this conversation is added, and memory is not used in it." checked={props.chatOff} onChange={props.setChatOff} />
      <ul class="memory__list" role="list">
        <For each={props.memories} fallback={<li class="hint">Nothing remembered yet. Say "remember that …" in a chat, or add one below.</li>}>
          {(m) => (
            <li class="memory__item" data-scope={m.scope}>
              <span class="chip" data-tone="neutral">
                {m.scope === "you" ? "You" : "Project"}
              </span>
              <input
                class="memory__text"
                value={m.text}
                aria-label="Edit memory"
                maxlength="500"
                onChange={(e) => {
                  const t = e.currentTarget.value.trim();
                  props.setMemories((l) => (t ? l.map((x) => (x.id === m.id ? { ...x, text: t } : x)) : l.filter((x) => x.id !== m.id)));
                }}
              />
              <button class="icon-btn icon-btn--sm" type="button" aria-label={`Forget: ${m.text}`} onClick={() => props.setMemories((l) => l.filter((x) => x.id !== m.id))}>
                <Icon name="trash" size={14} />
              </button>
            </li>
          )}
        </For>
      </ul>
      <form
        class="memory__add"
        onSubmit={(e) => {
          e.preventDefault();
          const t = draft().trim();
          if (!t) return;
          props.setMemories((l) => [...l, { id: newId(), text: t.slice(0, 500), scope: "you" }]);
          setDraft("");
        }}
      >
        <input value={draft()} onInput={(e) => setDraft(e.currentTarget.value)} placeholder="Add something to remember" aria-label="New memory" maxlength="500" />
        <button class="btn btn--secondary btn--sm" type="submit" data-press>
          Add
        </button>
      </form>
    </div>
  );
}

function PromptPanel(props: { prefs: ChatPrefs; setPrefs: (p: Partial<ChatPrefs>) => void; preview: string; native: { reasoningEffort: string; thinkingBudget: number } | null }): JSX.Element {
  return (
    <div class="prompt-panel">
      <label class="prompt-panel__label" for="base-prompt">
        Your system prompt
      </label>
      <textarea id="base-prompt" class="prompt-panel__base" rows="4" maxlength="8000" value={props.prefs.base} onChange={(e) => props.setPrefs({ base: e.currentTarget.value })} />
      <div class="prompt-panel__row">
        <label class="composer__check">
          <input type="checkbox" checked={!props.prefs.addInstructions} onChange={(e) => props.setPrefs({ addInstructions: !e.currentTarget.checked })} />
          <span>No added prompt (send only the text above)</span>
        </label>
        <button class="btn btn--tertiary btn--sm" type="button" onClick={() => props.setPrefs({ base: DEFAULT_BASE_PROMPT })}>
          Reset
        </button>
      </div>
      <h3 class="prompt-panel__label">Exactly what the model receives</h3>
      <pre class="prompt-panel__preview" data-testid="prompt-preview">
        {props.preview}
      </pre>
      <p class="hint" data-testid="native-params">
        {props.native ? `Native reasoning: effort ${props.native.reasoningEffort}, thinking budget ${props.native.thinkingBudget.toLocaleString()} tokens.` : "No native reasoning parameters are sent."}
      </p>
    </div>
  );
}
