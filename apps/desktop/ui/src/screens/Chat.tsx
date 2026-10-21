import "../styles/chat.css";
import { batch, createEffect, createMemo, createResource, createSignal, For, on, onCleanup, Show, type JSX } from "solid-js";
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
import { chatToolRules, loadPolicy, mcpToolKey, savePolicy } from "../lib/permissions";
import { notify } from "../lib/toast";
import { useAppState } from "../state";
import { DEFAULT_SAMPLER, LEDGER_FILES, type ChatTools, type ChatTurn, type LedgerFileName, type Sampler, type ToolActivity } from "../service/chat-types";
import { useService } from "../service/context";
import type { LibraryModel } from "../service/types";
import { HighlightQuery } from "../components/ChatSearch";
import { Composer } from "./chat/Composer";
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
  const models = createMemo(() => (library() ?? []).filter((m) => m.kind === "text" || m.kind === "vision"));
  const [modelId, setModelId] = createSignal<string>("");
  createEffect(() => {
    if (!modelId() && models().length) setModelId(models()[0]!.id);
  });
  const model = () => models().find((m) => m.id === modelId());

  const [prefs, setPrefsSignal] = createSignal<ChatPrefs>(loadPrefs());
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
  const [panel, setPanel] = createSignal<"none" | "memory" | "prompt" | "prompts">("none");
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
  const pickMode = (m: Mode) => {
    const next = applyMode(m, prefs(), sampler());
    setPrefs(next.prefs);
    setSampler(next.sampler);
    // Dragging across levels shows only the one it stops on.
    clearTimeout(modeNotice);
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

  const run = async (cid: string, parentId: string | null) => {
    const conv = state.list.find((c) => c.id === cid);
    if (!conv || !modelId()) return;
    const tools = await toolsForRequest();
    const req = built();
    const p = prefs();
    const path = branchPath(treeOf(conv));
    const at = parentId === null ? -1 : path.findIndex((n) => n.id === parentId);
    const context = (at >= 0 ? path.slice(0, at + 1) : path).map((n) => n.message);
    const turns: ChatTurn[] = context
      .filter((m) => !m.streaming && !m.error)
      .map((m) => ({ role: m.role, text: m.text, ...(m.attachments.length ? { attachments: m.attachments.map(({ name, type, bytes, preview }) => ({ name, type, bytes, ...(preview ? { data: preview } : {}) })) } : {}) }));
    const mid = newId();
    const reply: ChatMessage = { id: mid, role: "assistant", text: "", thinking: "", checks: [], image: null, tools: [], ledger: null, attachments: [], stats: null, effort: p.effort, verbosity: p.verbosity, source: p.source, streaming: true, error: null };
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
        {
          modelId: modelId(),
          systemPrompt: req.systemPrompt,
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

  const panelLabel = () => (panel() === "memory" ? "Memory" : panel() === "prompt" ? "System prompt" : "Prompt library");

  return (
    <div class="screen chat" data-panel={panel() === "none" ? "closed" : "open"}>
      <aside class="chat-list" aria-label="Conversations">
        <button class="btn btn--primary chat-list__new" type="button" onClick={newChat} data-press>
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

      <section class="chat-main" aria-label="Chat">
        <header class="chat-head">
          <h1 class="chat-head__title">Chat</h1>
          <label class="chat-head__model">
            <span class="sr-only">Model</span>
            <select value={modelId()} onChange={(e) => setModelId(e.currentTarget.value)} aria-label="Model">
              <For each={models()}>{(m) => <option value={m.id}>{m.name}</option>}</For>
            </select>
          </label>
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
          <SamplerMenu sampler={sampler()} onChange={setSampler} />
          <button class="icon-btn" type="button" aria-label="Prompt library" aria-pressed={panel() === "prompts"} onClick={() => setPanel(panel() === "prompts" ? "none" : "prompts")}>
            <Icon name="book" size={18} />
          </button>
          <button class="icon-btn" type="button" aria-label="Memory" aria-pressed={panel() === "memory"} onClick={() => setPanel(panel() === "memory" ? "none" : "memory")}>
            <Icon name="memory" size={18} />
          </button>
        </header>

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
                />
              )}
            </For>
          </Show>
        </div>

        <Composer
          levels={levels}
          mode={modeOf(prefs(), sampler())}
          onMode={pickMode}
          prefs={prefs()}
          setPrefs={setPrefs}
          streaming={streaming()}
          notes={built().notes}
          onSend={send}
          onStop={() => abort?.abort()}
          onTranscribe={(s) => service.transcribe(s)}
          onRecordStart={() => service.startRecording()}
          onLevel={() => service.recordingLevel()}
          onOpenPrompt={() => setPanel("prompt")}
          onOpenPrompts={() => setPanel("prompts")}
          draft={draft()}
        />
      </section>

      <Show when={panel() !== "none"}>
        <aside class="chat-side card" aria-label={panelLabel()}>
          <div class="chat-side__head">
            <h2>{panelLabel()}</h2>
            <button class="icon-btn" type="button" aria-label="Close panel" onClick={() => setPanel("none")}>
              <Icon name="close" size={16} />
            </button>
          </div>
          <Show
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
        </aside>
      </Show>
    </div>
  );
}

function Empty(props: { onPick: (t: string) => void }): JSX.Element {
  const ideas = ["Why is my 30B model slower than the 8B?", "Write a Python function that counts lines in a huge file", "/imagine a lighthouse at dusk, oil painting"];
  return (
    <div class="chat-empty">
      <h2>Ask anything. It stays on this computer.</h2>
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

function SamplerMenu(props: { sampler: Sampler; onChange: (s: Sampler) => void }): JSX.Element {
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
