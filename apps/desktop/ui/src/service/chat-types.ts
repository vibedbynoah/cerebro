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
  /** MCP tools the model may call, with the user's rule per "server/tool"; null = no tools. */
  tools?: ChatTools | null;
  /** Orchestrate mode: several instances of the model through a shared ledger; null = off. */
  orchestrate?: OrchestrateRequest | null;
  /** Settings > Protections "Keep long chats ready": save this chat's cache to disk, and restore
   *  it next time. Absent means on. */
  keepCache?: boolean;
}

/** Workers per round (1-6) and rounds before the result is finalized (1-8). */
export interface OrchestrateRequest {
  workers: number;
  rounds: number;
}

/** The ledger files Orchestrate mode writes, in the order the panel shows them. */
export type LedgerFileName = "plan.md" | "notes.md" | "tasks.json" | "result.md";
export const LEDGER_FILES: readonly LedgerFileName[] = ["plan.md", "notes.md", "tasks.json", "result.md"];

export type ToolRuleNow = "allow" | "ask" | "deny";
export interface ChatTools {
  rules: Record<string, ToolRuleNow>;
  defaultRule: ToolRuleNow;
  /** Settings > Protections "Keep web fetches off this network" is off. */
  allowLocalFetch?: boolean;
  /** How far the built-in tools reach (Tools screen). */
  access?: "read-only" | "workspace" | "full";
  /** Settings > Protections "Hide secrets from tools" is off. */
  showSecrets?: boolean;
  /** Settings > Protections "Read before write" is off. */
  blindWrites?: boolean;
  /** Plan mode: the model investigates read-only, then presents a plan for approval. */
  planMode?: boolean;
  /** The user's Offline mode setting: built-in tools that touch the network must refuse. */
  offline: boolean;
}

/** "question" (ask_user) and "plan" (exit_plan_mode) wait for the user's answer in the chat. */
export type ToolState = "approval" | "question" | "plan" | "running" | "done" | "denied" | "error";

/** One tool call in a reply, as Chat shows and stores it. */
export interface ToolActivity {
  id: string;
  server: string;
  name: string;
  arguments: unknown;
  state: ToolState;
  result: string | null;
  error: string | null;
}

export type ChatEvent =
  | { type: "start"; ttftMs: number }
  | { type: "thinking"; text: string }
  | { type: "token"; text: string }
  | { type: "check"; pass: 1 | 2; text: string }
  | { type: "image"; prompt: string; seed: number; width: number; height: number }
  | { type: "tool"; id: string; server: string; name: string; arguments: unknown; state: ToolState; result: string | null; error: string | null }
  /** Orchestrate mode: a role started (`step`), or a ledger file changed (`file` + whole new text). */
  | { type: "ledger"; step: string | null; round: number; file: LedgerFileName | null; content: string | null; dir: string | null }
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
