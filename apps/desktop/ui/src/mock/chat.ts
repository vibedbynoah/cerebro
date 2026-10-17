import type { ChatEvent, ChatRequest, ChatStats, LedgerFileName } from "../service/chat-types";

/**
 * Illustrative chat engine for web mode and tests: streams a reply token by token at a believable
 * speed, with thinking and review passes when the request asks for them. Replies are canned and
 * labelled illustrative in the UI. `/echo <text>` replies with <text> verbatim (tests use it to feed
 * the renderer hostile Markdown); `/imagine <prompt>` returns an image card.
 */

export interface MockChatOptions {
  /** Playback speed multiplier (tests use 20). */
  speed?: number;
  /** Decode speed of the simulated model, tokens per second. */
  tokPerSec?: number;
  /** Playback speed multiplier for Orchestrate runs: slow enough to watch the steps advance. */
  orchestrateSpeed?: number;
}

const BRIEF = [
  "Short answer: yes. Keep the model's context at the length you actually use, because a bigger context costs memory and speed on every token.",
  "Use a 4-bit quantisation of the larger model rather than an 8-bit one of the smaller: on this machine it scores higher on coding at almost the same speed.",
  "The slowdown comes from experts running on the processor. Moving more of them into graphics memory is the biggest lever.",
];

const CODE = `Here is a version that streams the file instead of loading it whole:

\`\`\`python
def count_lines(path: str) -> int:
    """Count lines without reading the whole file into memory."""
    total = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            total += chunk.count(b"\\n")
    return total
\`\`\`

It reads one megabyte at a time, so memory stays flat even for very large files.`;

const MORE = `## In more detail

The speed of a mixture-of-experts model depends on **where its experts live**:

1. Experts in graphics memory run at full speed.
2. Experts in system memory run on the processor, in parallel with the graphics card.
3. Each token uses only a few experts, so a cache of the most-used ones covers most tokens.

> Cerebro measures this on your machine rather than guessing, and keeps the fastest stable setting.

A useful rule: *more graphics memory beats a faster graphics card* for this kind of model.`;

const THINKING = "The user wants the practical answer. Check what limits speed here: memory for the context, where the experts sit, and whether the quantisation fits. Verify the numbers before answering.";

function pick(req: ChatRequest): { text: string; image?: { prompt: string } } {
  const last = [...req.turns].reverse().find((t) => t.role === "user")?.text.trim() ?? "";
  if (last.startsWith("/echo ")) return { text: last.slice(6) };
  if (last.startsWith("/imagine ")) {
    const prompt = last.slice(9).trim().slice(0, 400);
    return { text: `Made an image for "${prompt}". It stayed on this computer.`, image: { prompt } };
  }
  if (/tell me more|more detail|explain/i.test(last) || req.verbosity === "detailed") return { text: MORE };
  if (/\b(code|function|script|python|rust|write)\b/i.test(last)) return { text: CODE };
  const base = BRIEF[last.length % BRIEF.length]!;
  return { text: req.verbosity === "normal" ? `${base}\n\nIf you tell me your context length and which model you run, I can give the exact setting.` : base };
}

/** Splits text into token-sized pieces (words with their following space, newlines kept). */
export function tokenize(text: string): string[] {
  return text.match(/\s*\S+\s*|\n/g) ?? [];
}

const wait = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    if (signal.aborted) return resolve();
    const t = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => (clearTimeout(t), resolve()), { once: true });
  });

// ---- Orchestrate mode (GVS5H): a scripted ledger run, the same shape the engine streams ---------

interface LedgerScript {
  role: string;
  round: number;
  /** The role's reply, as "thinking" events (the engine streams each role's reasoning this way). */
  thinking: string;
  /** A ledger file this role rewrites (its whole new text), as the engine's `file` events carry it. */
  file?: { name: LedgerFileName; text: string };
}

/** One planned run: plan, ideate, manage, two workers, a second manage, then finalize. The file
 *  texts are illustrative, exactly as the UI labels the web build. */
function ledgerScript(goal: string): { dir: string; steps: LedgerScript[]; answer: string } {
  const a = goal.length * 37 + 11;
  const b = goal.length * 19 + 7;
  const tasks = (done: number[]) =>
    JSON.stringify(
      [
        { id: 1, desc: "Work out what the question really asks", status: done.includes(1) ? "done" : "todo" },
        { id: 2, desc: `Work the numbers through from scratch (${a} and ${b})`, status: done.includes(2) ? "done" : "todo" },
        { id: 3, desc: "Verify the result a second, independent way", status: done.includes(3) ? "done" : "todo" },
      ],
      null,
      2,
    );
  const notes1 = "- **Ideation:** The plain computation is easy to get wrong in the head; write the steps out, then check the result by regrouping the factors.\n- **Approaches:** closed form first; repeated addition as the cross-check.\n";
  const notes2 = "- **Ideation:** The plain computation is easy to get wrong in the head; write the steps out, then check the result by regrouping the factors.\n- **Worker 1:** split the factors - " + a + " x " + b + " = " + a * 10 + " + " + a * (b - 10) + " = " + a * b + ".\n- **Worker 2:** regrouping as " + b + " x " + a + " gives the same " + a * b + ".\n";
  const answer = [
    `Here is the worked answer for "${goal.trim().slice(0, 80)}".\n\n`,
    `Split the product and add the parts: ${a} × ${b} = ${a} × 10 + ${a} × ${b - 10} = ${a * 10} + ${a * (b - 10)} = ${a * b}.\n`,
    `The same result comes from regrouping the other way, ${b} × ${a} = ${a * b}. Both workers agreed, so this is the answer.`,
  ].join("\n");
  return {
    // A plausible app-data path; the desktop build's ledger really lands on disk.
    dir: "C:\\Users\\me\\AppData\\Roaming\\Cerebro\\orchestrate\\r1illustrative",
    steps: [
      { role: "plan", round: 0, thinking: "Read the goal, then settle on a strategy and write it down before anyone computes anything.\n", file: { name: "plan.md", text: "Split the product into parts that can be checked by hand, then have two workers verify the result in different ways. Do not treat the first number that comes out as the final one.\n" } },
      { role: "ideate", round: 0, thinking: "First worker: find approaches that are genuinely different before picking one.\n", file: { name: "notes.md", text: notes1 } },
      { role: "manage", round: 1, thinking: "Two independent checks are worth more than one careful one, so keep both workers on this.\n", file: { name: "tasks.json", text: tasks([1]) } },
      { role: "worker 1", round: 1, thinking: "Work the computation through step by step and say how each step could go wrong.\n", file: { name: "notes.md", text: notes2 } },
      { role: "worker 2", round: 1, thinking: "Take the same problem from the other end, so a shared mistake cannot hide.\n", file: { name: "result.md", text: `Draft: ${a} x ${b} = ${a * b}. Both methods agree.\n` } },
      { role: "manage", round: 2, thinking: "The two checks agree; there is nothing left to split out, so finish.\n", file: { name: "tasks.json", text: tasks([1, 2, 3]) } },
      { role: "finalize", round: 3, thinking: "Write the answer for the user: complete, no mention of the team or the ledger.\n", file: { name: "result.md", text: answer } },
    ],
    answer,
  };
}

export function createMockChat(options: MockChatOptions = {}) {
  const speed = options.speed && options.speed > 0 ? options.speed : 1;
  const tps = options.tokPerSec ?? 24;
  const orchestrateSpeed = options.orchestrateSpeed && options.orchestrateSpeed > 0 ? options.orchestrateSpeed : speed;
  const pendingDecisions = new Map<string, (allow: boolean) => void>();

  async function orchestrate(req: ChatRequest, onEvent: (e: ChatEvent) => void, signal: AbortSignal, ttftMs: number): Promise<void> {
    const goal = [...req.turns].reverse().find((t) => t.role === "user")?.text.trim() ?? "";
    const script = ledgerScript(goal);
    const started = performance.now();
    let out = 0;
    let thinkingTokens = 0;
    // The engine's first event is the ledger folder, then a step event per role and a file event per
    // rewrite. A stop mid-run keeps whatever already landed, so the partial run stays readable.
    onEvent({ type: "ledger", step: null, round: 0, file: null, content: null, dir: script.dir });
    for (const step of script.steps) {
      if (signal.aborted) break;
      onEvent({ type: "ledger", step: step.role, round: step.round, file: null, content: null, dir: null });
      for (const w of tokenize(step.thinking)) {
        if (signal.aborted) break;
        onEvent({ type: "thinking", text: w });
        thinkingTokens++;
        await wait(1000 / tps / orchestrateSpeed, signal);
      }
      if (signal.aborted) break;
      await wait(900 / orchestrateSpeed, signal);
      if (step.file) {
        onEvent({ type: "ledger", step: null, round: 0, file: step.file.name, content: step.file.text, dir: null });
        await wait(300 / orchestrateSpeed, signal);
      }
    }
    // A stopped run keeps the partial ledger and produces no final answer; one that finishes streams
    // the finished result.md as plain tokens, as the engine does.
    if (!signal.aborted) {
      for (const w of tokenize(script.answer)) {
        if (signal.aborted) break;
        onEvent({ type: "token", text: w });
        out++;
        await wait(1000 / tps / orchestrateSpeed, signal);
      }
    }
    const totalMs = Math.round((performance.now() - started) * orchestrateSpeed);
    const stats: ChatStats = { outputTokens: out, thinkingTokens, tokPerSec: tps, ttftMs, totalMs, extraMs: 0, stopped: signal.aborted };
    onEvent({ type: "done", stats });
  }

  async function chat(req: ChatRequest, onEvent: (e: ChatEvent) => void, signal: AbortSignal): Promise<void> {
    if (!req || typeof req.systemPrompt !== "string" || !Array.isArray(req.turns) || req.turns.length === 0) throw new TypeError("Malformed chat request");
    const started = performance.now();
    const promptChars = req.systemPrompt.length + req.turns.reduce((n, t) => n + t.text.length, 0);
    const ttftMs = Math.round(180 + promptChars / 40);
    await wait(ttftMs / speed, signal);
    onEvent({ type: "start", ttftMs });

    if (req.orchestrate) return orchestrate(req, onEvent, signal, ttftMs);

    // A request with tools that asks about files: the model calls the mock files server's
    // list_directory first, following the rule the UI sent (allow / ask / deny), as the engine does.
    if (req.tools && /\b(files?|folder|directory)\b/i.test(req.turns[req.turns.length - 1]?.text ?? "") && !signal.aborted) {
      const id = `t${Date.now()}-0`;
      const base = { id, server: "files", name: "list_directory", arguments: { path: "C:/Users/me/Projects" } };
      const rule = req.tools.rules["files/list_directory"] ?? req.tools.defaultRule;
      let allowed = rule === "allow";
      if (rule === "ask") {
        onEvent({ type: "tool", ...base, state: "approval", result: null, error: null });
        allowed = await new Promise<boolean>((resolve) => {
          pendingDecisions.set(id, resolve);
          signal.addEventListener("abort", () => resolve(false), { once: true });
        });
        pendingDecisions.delete(id);
      }
      if (allowed) {
        onEvent({ type: "tool", ...base, state: "running", result: null, error: null });
        await wait(300 / speed, signal);
        onEvent({ type: "tool", ...base, state: "done", result: "[DIR] cerebro\n[DIR] notes\n[FILE] todo.md", error: null });
      } else onEvent({ type: "tool", ...base, state: "denied", result: null, error: null });
      if (signal.aborted) {
        onEvent({ type: "done", stats: { outputTokens: 0, thinkingTokens: 0, tokPerSec: tps, ttftMs, totalMs: 0, extraMs: 0, stopped: true } });
        return;
      }
      const text = allowed ? "Your Projects folder has two folders, cerebro and notes, and one file, todo.md." : "I could not look at the folder because the tool was not allowed.";
      for (const w of tokenize(text)) {
        if (signal.aborted) break;
        onEvent({ type: "token", text: w });
        await wait(1000 / tps / speed, signal);
      }
      onEvent({ type: "done", stats: { outputTokens: tokenize(text).length, thinkingTokens: 0, tokPerSec: tps, ttftMs, totalMs: Math.round(performance.now() - started), extraMs: 0, stopped: signal.aborted } });
      return;
    }

    const reply = pick(req);
    let thinkingTokens = 0;
    if (req.native && req.native.thinkingBudget > 0 && !signal.aborted) {
      const words = tokenize(THINKING).slice(0, Math.max(8, Math.min(60, Math.round(req.native.thinkingBudget / 64))));
      for (const w of words) {
        if (signal.aborted) break;
        onEvent({ type: "thinking", text: w });
        thinkingTokens++;
        await wait(1000 / tps / speed, signal);
      }
    }

    // Long replies are delivered a few tokens per tick so the UI repaints once per frame, not per token.
    const tokens = tokenize(reply.text);
    const perTick = tokens.length > 400 ? 4 : 1;
    let out = 0;
    for (let i = 0; i < tokens.length && !signal.aborted; i += perTick) {
      onEvent({ type: "token", text: tokens.slice(i, i + perTick).join("") });
      out += Math.min(perTick, tokens.length - i);
      await wait((1000 / tps / speed) * perTick, signal);
    }
    if (reply.image && !signal.aborted) {
      await wait(800 / speed, signal);
      onEvent({ type: "image", prompt: reply.image.prompt, seed: reply.image.prompt.length * 7919, width: 1024, height: 1024 });
    }
    for (let pass = 1; pass <= req.passes && !signal.aborted; pass++) {
      await wait(400 / speed, signal);
      onEvent({ type: "check", pass: pass as 1 | 2, text: pass === 1 ? "Reread the answer: the numbers and the advice hold." : "Independent check in a fresh context: no error found." });
      thinkingTokens += 40;
    }
    const totalMs = Math.round((performance.now() - started) * speed);
    const stats: ChatStats = {
      outputTokens: out,
      thinkingTokens,
      tokPerSec: tps,
      ttftMs,
      totalMs,
      extraMs: Math.round(((thinkingTokens * 1000) / tps) + req.passes * 400),
      stopped: signal.aborted,
    };
    onEvent({ type: "done", stats });
  }

  async function decide(id: string, allow: boolean): Promise<boolean> {
    const resolve = pendingDecisions.get(id);
    if (!resolve) return false;
    resolve(allow);
    return true;
  }

  async function transcribe(seconds: number): Promise<string> {
    if (typeof seconds !== "number" || !(seconds > 0) || seconds > 120) throw new TypeError("Recording length out of range");
    await new Promise((r) => setTimeout(r, 300 / speed));
    return "How much graphics memory do I need for a 30B mixture-of-experts model?";
  }

  return { chat, transcribe, decide };
}
