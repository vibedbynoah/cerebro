import "../styles/chat.css";
import { batch, createEffect, createMemo, createResource, createSignal, For, on, onCleanup, onMount, Show, type JSX } from "solid-js";
import { createStore, produce } from "solid-js/store";
import { Icon } from "../components/icons";
import { IllustrativeFlag } from "../components/Screen";
import { SettingSwitch } from "../components/SettingSwitch";
import { applyMode, loadSampler, MODE_PRESETS, modeOf, saveSampler, type Mode } from "../lib/chat/modes";
import { buildRequest, DEFAULT_BASE_PROMPT, effectiveLevels, loadLevelEdits, loadPrefs, savePrefs, supportsNativeReasoning, type ChatPrefs, type Memory } from "../lib/chat/prompt";
import {
  emptyConversation,
  loadConversations,
  loadMemories,
  memoryCandidate,
  newId,
  saveConversations,
  saveMemories,
  titleFrom,
  type Attachment,
  type ChatMessage,
  type Conversation,
  type LedgerState,
} from "../lib/chat/store";
import { chatToolRules, loadPolicy, mcpToolKey, savePolicy } from "../lib/permissions";
import { notify, notifyAction } from "../lib/toast";
import { COMMANDS, numberArg, orchestrateArg, speedArg, switchArg } from "../lib/chat/commands";
import { parseMarkdown, plainText } from "../lib/chat/markdown";
import { saveText } from "../lib/download";
import { Dialog } from "@ark-ui/solid/dialog";
import { Portal } from "solid-js/web";
import "../styles/guide.css";
import { useAppState } from "../state";
import { DEFAULT_SAMPLER, LEDGER_FILES, type ChatTools, type ChatTurn, type LedgerFileName, type Sampler, type ToolActivity } from "../service/chat-types";
import { useService } from "../service/context";
import type { LibraryModel } from "../service/types";
import { Composer } from "./chat/Composer";
import { LiveDock } from "./chat/LiveDock";
import { readPref, writePref } from "../lib/storage";
import { MessageView } from "./chat/Message";
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
  // The side panel: memory, the system prompt, or the live dock (remembered while it is the dock).
  const DOCK_KEY = "cerebro-chat-dock";
  const [panel, setPanelSignal] = createSignal<"none" | "memory" | "prompt" | "dock">(readPref(DOCK_KEY) === "open" ? "dock" : "none");
  const setPanel = (p: "none" | "memory" | "prompt" | "dock") => {
    setPanelSignal(p);
    writePref(DOCK_KEY, p === "dock" ? "open" : null);
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
  const [live, setLive] = createSignal<{ tps: number; ttft: number } | null>(null);

  let saveTimer: ReturnType<typeof setTimeout> | undefined;
  const persist = () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => saveConversations(state.list), 400);
  };
  onCleanup(() => {
    clearTimeout(saveTimer);
    saveConversations(state.list);
    abort?.abort();
  });
  createEffect(on(memories, (m) => saveMemories(m), { defer: true }));

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

  const updateMsg = (cid: string, mid: string, fn: (m: ChatMessage) => void) =>
    setState(
      "list",
      (c) => c.id === cid,
      "messages",
      (m) => m.id === mid,
      produce(fn),
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

  const run = async (cid: string) => {
    const conv = state.list.find((c) => c.id === cid);
    if (!conv || !modelId()) return;
    const tools = await toolsForRequest();
    const req = built();
    const p = prefs();
    const turns: ChatTurn[] = conv.messages
      .filter((m) => !m.streaming && !m.error)
      .map((m) => ({ role: m.role, text: m.text, ...(m.attachments.length ? { attachments: m.attachments.map(({ name, type, bytes, preview }) => ({ name, type, bytes, ...(preview ? { data: preview } : {}) })) } : {}) }));
    const mid = newId();
    const reply: ChatMessage = { id: mid, role: "assistant", text: "", thinking: "", checks: [], image: null, tools: [], ledger: null, attachments: [], stats: null, effort: p.effort, verbosity: p.verbosity, source: p.source, streaming: true, error: null };
    setState("list", (c) => c.id === cid, "messages", (l) => [...l, reply]);
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
    batch(() => {
      if (!conv.messages.length) setState("list", (c) => c.id === cid, "title", titleFrom(text || atts[0]?.name || ""));
      setState("list", (c) => c.id === cid, "messages", (l) => [
        ...l,
        { id: newId(), role: "user", text, thinking: "", checks: [], image: null, tools: [], ledger: null, attachments: atts, stats: null, effort: prefs().effort, verbosity: prefs().verbosity, source: prefs().source, streaming: false, error: null } satisfies ChatMessage,
      ]);
    });
    void run(cid);
  };

  const regenerate = () => {
    const c = current();
    if (!c || streaming()) return;
    const lastBot = [...c.messages].reverse().find((m) => m.role === "assistant");
    if (lastBot) setState("list", (x) => x.id === c.id, "messages", (l) => l.filter((m) => m.id !== lastBot.id));
    void run(c.id);
  };

  const more = () => send("Tell me more.", []);

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
        return regenerate();
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
        setPrefs(on ? { effort: prefs().effort === "low" ? "medium" : prefs().effort, source: prefs().source === "none" ? "native" : prefs().source } : { effort: "low" });
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
    }
  };
  const [helpOpen, setHelpOpen] = createSignal(false);

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
      { id: newId(), role: "user", text: `/image ${prompt}`, thinking: "", checks: [], image: null, tools: [], ledger: null, attachments: [], stats: null, effort: prefs().effort, verbosity: prefs().verbosity, source: prefs().source, streaming: false, error: null } satisfies ChatMessage,
      { id: mid, role: "assistant", text: "Making the image…", thinking: "", checks: [], image: null, tools: [], ledger: null, attachments: [], stats: null, effort: prefs().effort, verbosity: prefs().verbosity, source: prefs().source, streaming: true, error: null } satisfies ChatMessage,
    ]);
    try {
      const g = await service.generateImage({ modelId: im.id, prompt, size: 512, steps: 1, seed: null });
      updateMsg(cid, mid, (m) => {
        m.text = `Made with ${im.name} in ${g.seconds.toFixed(1)} s, seed ${g.seed}.`;
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
    persist();
  };

  const remove = (id: string) => {
    if (id === currentId()) abort?.abort();
    setState("list", (l) => l.filter((c) => c.id !== id));
    if (id === currentId()) setCurrentId(state.list[0]?.id ?? "");
    persist();
  };

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

  return (
    <div class="screen chat" data-panel={panel() === "none" ? "closed" : "open"}>
      <aside class="chat-list" aria-label="Conversations">
        <button class="btn btn--primary chat-list__new" type="button" onClick={newChat} data-press>
          <Icon name="plus" size={16} />
          <span class="btn__label">New chat</span>
        </button>
        <ul role="list">
          <For each={state.list} fallback={<li class="hint">No conversations yet.</li>}>
            {(c) => (
              <li class="chat-list__item" data-current={c.id === currentId()}>
                <button type="button" class="chat-list__open" onClick={() => setCurrentId(c.id)} aria-current={c.id === currentId() ? "true" : undefined}>
                  {c.title}
                </button>
                <button type="button" class="icon-btn icon-btn--sm" aria-label={`Delete ${c.title}`} onClick={() => remove(c.id)}>
                  <Icon name="trash" size={14} />
                </button>
              </li>
            )}
          </For>
        </ul>
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
          <button class="icon-btn" type="button" aria-label="Memory" aria-pressed={panel() === "memory"} onClick={() => setPanel(panel() === "memory" ? "none" : "memory")}>
            <Icon name="memory" size={18} />
          </button>
          <button class="icon-btn" type="button" aria-label="Live panel (Ctrl .)" title="Live panel (Ctrl .)" aria-pressed={panel() === "dock"} onClick={() => setPanel(panel() === "dock" ? "none" : "dock")}>
            <Icon name="activity" size={18} />
          </button>
        </header>

        <div class="chat-thread" ref={thread} role="log" aria-live="polite" aria-relevant="additions">
          <Show when={current()?.messages.length} fallback={<Empty onPick={(t) => send(t, [])} />}>
            {/* The transcript's own heading: replies' Markdown headings (h3, h4) nest under it. */}
            <h2 class="sr-only">Conversation</h2>
            <For each={current()!.messages}>
              {(m, i) => <MessageView m={m} last={i() === current()!.messages.length - 1} onRegenerate={regenerate} onMore={more} onToolDecide={(id, d) => decideTool(currentId(), m.id, id, d)} onToolAnswer={answerTool} />}
            </For>
          </Show>
        </div>

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
          onMode={pickMode}
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
        />
      </section>

      <Show when={panel() !== "none"}>
        <aside
          class="chat-side card"
          classList={{ "chat-side--dock": panel() === "dock" }}
          style={panel() === "dock" ? { width: `${dockWidth()}px` } : undefined}
          aria-label={panel() === "memory" ? "Memory" : panel() === "dock" ? "Live panel" : "System prompt"}
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
            <h2>{panel() === "memory" ? "Memory" : panel() === "dock" ? "Live panel" : "System prompt"}</h2>
            <button class="icon-btn" type="button" aria-label="Close panel" onClick={() => setPanel("none")}>
              <Icon name="close" size={16} />
            </button>
          </div>
          <Show when={panel() === "dock"}>
            <LiveDock
              mode={modeOf(prefs(), sampler())}
              onMode={pickMode}
              sampler={sampler()}
              setSampler={setSampler}
              thinking={prefs().effort !== "low" && prefs().source !== "none"}
              setThinking={(on) => setPrefs(on ? { effort: prefs().effort === "low" ? "medium" : prefs().effort, source: prefs().source === "none" ? "native" : prefs().source } : { effort: "low" })}
              onOpenPrompt={() => setPanel("prompt")}
              onUnload={() => {
                const id = modelId();
                void service.unloadModel(id).then(
                  () => notify("Model unloaded", "Its memory is free. It loads again with your next message."),
                  (e: unknown) => notify("Could not unload", e instanceof Error ? e.message : undefined, "error"),
                );
              }}
            />
          </Show>
          <Show when={panel() !== "dock"}>
          <Show when={panel() === "memory"} fallback={<PromptPanel prefs={prefs()} setPrefs={setPrefs} preview={built().systemPrompt} native={built().native} />}>
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
      <h2>Ask anything. The model runs here.</h2>
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
