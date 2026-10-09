import type { ChatEvent, ChatRequest, ChatStats } from "../service/chat-types";

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

export function createMockChat(options: MockChatOptions = {}) {
  const speed = options.speed && options.speed > 0 ? options.speed : 1;
  const tps = options.tokPerSec ?? 24;

  async function chat(req: ChatRequest, onEvent: (e: ChatEvent) => void, signal: AbortSignal): Promise<void> {
    if (!req || typeof req.systemPrompt !== "string" || !Array.isArray(req.turns) || req.turns.length === 0) throw new TypeError("Malformed chat request");
    const started = performance.now();
    const promptChars = req.systemPrompt.length + req.turns.reduce((n, t) => n + t.text.length, 0);
    const ttftMs = Math.round(180 + promptChars / 40);
    await wait(ttftMs / speed, signal);
    onEvent({ type: "start", ttftMs });

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

  async function transcribe(seconds: number): Promise<string> {
    if (typeof seconds !== "number" || !(seconds > 0) || seconds > 120) throw new TypeError("Recording length out of range");
    await new Promise((r) => setTimeout(r, 300 / speed));
    return "How much graphics memory do I need for a 30B mixture-of-experts model?";
  }

  return { chat, transcribe };
}
