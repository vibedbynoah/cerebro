import type { Effort, ReasoningSource, Verbosity } from "../lib/chat/prompt";

/** One message as sent to the engine. Attachments are referenced by name and type; the bytes stay local. */
export interface ChatTurn {
  role: "user" | "assistant";
  text: string;
  attachments?: ReadonlyArray<{ name: string; type: string; bytes: number }>;
}

export interface Sampler {
  temperature: number;
  topP: number;
  minP: number;
  topK: number;
  repeatPenalty: number;
  maxTokens: number;
}

export const DEFAULT_SAMPLER: Sampler = { temperature: 0.7, topP: 0.95, minP: 0.05, topK: 40, repeatPenalty: 1.05, maxTokens: 4096 };

export interface ChatRequest {
  modelId: string;
  systemPrompt: string;
  turns: readonly ChatTurn[];
  sampler: Sampler;
  native: { reasoningEffort: "low" | "medium" | "high"; thinkingBudget: number } | null;
  passes: 0 | 1 | 2;
  /** For the reply footer only; the engine acts on systemPrompt/native/passes. */
  effort: Effort;
  verbosity: Verbosity;
  source: ReasoningSource;
}

export type ChatEvent =
  | { type: "start"; ttftMs: number }
  | { type: "thinking"; text: string }
  | { type: "token"; text: string }
  | { type: "check"; pass: 1 | 2; text: string }
  | { type: "image"; prompt: string; seed: number; width: number; height: number }
  | { type: "done"; stats: ChatStats }
  | { type: "error"; message: string };

export interface ChatStats {
  outputTokens: number;
  thinkingTokens: number;
  tokPerSec: number;
  ttftMs: number;
  totalMs: number;
  /** Extra time the effort level added over a Low answer of the same length. */
  extraMs: number;
  stopped: boolean;
}
