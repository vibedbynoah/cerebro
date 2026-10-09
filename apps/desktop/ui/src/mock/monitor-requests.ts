import type { RequestSummary } from "../service/monitor-types";

/**
 * ILLUSTRATIVE REQUEST TRAFFIC for the Monitor's inspector: what a coding agent, a chat and a small
 * business app send to Cerebro's API in a working session. Bodies are written to read like real
 * captures; every number derived from them is invented and labelled illustrative on screen.
 */

export interface RequestTemplate {
  session: string;
  path: string;
  stream: boolean;
  responseFormat: RequestSummary["responseFormat"];
  tool: string | null;
  /** Prompt length in tokens, and the share the conversation cache already holds. */
  prompt: number;
  reusedShare: number;
  /** Tokens written: reasoning first, then the answer or tool call. */
  reasoning: number;
  output: number;
  maxTokens: number;
  user: string;
  reasoningText: string;
  outputText: string;
  /** Ends with an HTTP 400 before any token is written: the prompt is longer than the context. */
  error?: true;
  /** The client closed the stream after this share of the output. */
  cancelAt?: number;
}

const AGENT_SYSTEM =
  "You are a coding agent working in the user's repository. Use the tools to read and change files. Ask before running anything that changes state outside the repository.";

const AGENT_TOOLS = ["read_file", "edit_file", "run_tests", "search_code"];

export const TEMPLATES: readonly RequestTemplate[] = [
  {
    session: "Coding agent · cerebro-app",
    path: "/v1/chat/completions",
    stream: true,
    responseFormat: "text",
    tool: "read_file",
    prompt: 9_840,
    reusedShare: 0.91,
    reasoning: 96,
    output: 42,
    maxTokens: 4096,
    user: "The date parser test fails on 29 February. Find the cause.",
    reasoningText:
      "The failing case is a leap day, so the parser probably validates the day against a fixed month length table. Read the parser before guessing.",
    outputText: '{"name":"read_file","arguments":{"path":"src/parse/date.rs"}}',
  },
  {
    session: "Coding agent · cerebro-app",
    path: "/v1/chat/completions",
    stream: true,
    responseFormat: "text",
    tool: "edit_file",
    prompt: 11_920,
    reusedShare: 0.94,
    reasoning: 180,
    output: 310,
    maxTokens: 4096,
    user: "(tool result) src/parse/date.rs, 212 lines",
    reasoningText:
      "DAYS_IN_MONTH[1] is 28 and is_leap() is never consulted. The fix is to use days_in_month(year, month), which already exists in calendar.rs.",
    outputText:
      'The parser checks the day against a fixed table where February always has 28 days. I will use the existing `days_in_month(year, month)` helper instead.\n\n{"name":"edit_file","arguments":{"path":"src/parse/date.rs","find":"if day > DAYS_IN_MONTH[(month - 1) as usize]","replace":"if day > calendar::days_in_month(year, month)"}}',
  },
  {
    session: "Coding agent · cerebro-app",
    path: "/v1/chat/completions",
    stream: true,
    responseFormat: "text",
    tool: "run_tests",
    prompt: 12_480,
    reusedShare: 0.96,
    reasoning: 40,
    output: 38,
    maxTokens: 4096,
    user: "(tool result) edit applied: 1 line changed",
    reasoningText: "Run only the parser tests first; the full suite takes longer.",
    outputText: '{"name":"run_tests","arguments":{"command":"cargo test parse::date"}}',
  },
  {
    session: "Coding agent · cerebro-app",
    path: "/v1/chat/completions",
    stream: true,
    responseFormat: "text",
    tool: null,
    prompt: 12_910,
    reusedShare: 0.97,
    reasoning: 60,
    output: 164,
    maxTokens: 4096,
    user: "(tool result) test result: ok. 14 passed; 0 failed",
    reasoningText: "All 14 parser tests pass, including the leap-day case. Summarise the change.",
    outputText:
      "Fixed. The parser compared the day with a fixed table in which February always has 28 days, so 29 February 2028 was rejected. It now asks `calendar::days_in_month(year, month)`, which handles leap years. All 14 parser tests pass.",
  },
  {
    session: "Chat · Supplier email",
    path: "/v1/chat/completions",
    stream: true,
    responseFormat: "text",
    tool: null,
    prompt: 1_240,
    reusedShare: 0.62,
    reasoning: 0,
    output: 236,
    maxTokens: 2048,
    user: "Draft a short, polite reply asking Northwind to move our delivery from the 7th to the 14th.",
    reasoningText: "",
    outputText:
      "Hello Sam,\n\nThank you for confirming the order. Would it be possible to move the delivery from Tuesday the 7th to Tuesday the 14th? Our storage space frees up that week, and a later date would let us receive everything in one go.\n\nIf the 14th does not suit your schedule, any day that week works for us.\n\nBest regards,\nAlex",
  },
  {
    session: "API · invoice-extractor",
    path: "/v1/chat/completions",
    stream: false,
    responseFormat: "json_schema",
    tool: null,
    prompt: 2_310,
    reusedShare: 0.48,
    reasoning: 0,
    output: 88,
    maxTokens: 512,
    user: "Extract vendor, invoice number, total, currency and due date from this invoice text: NORTHWIND PAPER CO. Invoice INV-20931 ... Total due EUR 1,284.50 by 2 November 2026",
    reasoningText: "",
    outputText: '{"vendor":"Northwind Paper Co.","invoice_number":"INV-20931","total":1284.5,"currency":"EUR","due_date":"2026-11-02"}',
  },
  {
    session: "Coding agent · billing-service",
    path: "/v1/responses",
    stream: true,
    responseFormat: "text",
    tool: "search_code",
    prompt: 7_420,
    reusedShare: 0.88,
    reasoning: 120,
    output: 34,
    maxTokens: 4096,
    user: "Where is the retry backoff for failed card charges configured?",
    reasoningText: "Search for the backoff constant rather than reading every module.",
    outputText: '{"name":"search_code","arguments":{"query":"retry_backoff","glob":"src/**/*.py"}}',
  },
  {
    session: "Coding agent · billing-service",
    path: "/v1/messages",
    stream: true,
    responseFormat: "text",
    tool: "edit_file",
    prompt: 8_960,
    reusedShare: 0.93,
    reasoning: 140,
    output: 220,
    maxTokens: 4096,
    user: "(tool result) src/billing/retry.py:14 RETRY_BACKOFF_S = [1, 1, 1]",
    reasoningText: "Three one-second retries hammer the processor. Use exponential backoff with jitter, capped at 30 s.",
    outputText:
      'The retries are fixed at one second, three times. I will switch to exponential backoff (2, 4, 8 s) with jitter.\n\n{"name":"edit_file","arguments":{"path":"src/billing/retry.py","find":"RETRY_BACKOFF_S = [1, 1, 1]","replace":"RETRY_BACKOFF_S = [2, 4, 8]  # plus up to 25% jitter in schedule()"}}',
  },
  {
    session: "API · invoice-extractor",
    path: "/v1/chat/completions",
    stream: false,
    responseFormat: "json_schema",
    tool: null,
    prompt: 70_112,
    reusedShare: 0,
    reasoning: 0,
    output: 0,
    maxTokens: 512,
    user: "Extract vendor, invoice number, total, currency and due date from this 41-page statement: ...",
    reasoningText: "",
    outputText: "",
    error: true,
  },
  {
    session: "Chat · Supplier email",
    path: "/v1/chat/completions",
    stream: true,
    responseFormat: "text",
    tool: null,
    prompt: 1_610,
    reusedShare: 0.81,
    reasoning: 0,
    output: 260,
    maxTokens: 2048,
    user: "Make it a little more formal.",
    reasoningText: "",
    outputText:
      "Dear Mr Patel,\n\nThank you for confirming our order. May I ask whether the delivery could be rescheduled from Tuesday 7 October to Tuesday 14 October? Our storage will be available from that week, which would allow us to receive the full consignment at once.",
    cancelAt: 0.45,
  },
];

/** The request body a client sent, as the capture shows it. */
export function inputBody(t: RequestTemplate, model: string): string {
  const agent = t.session.startsWith("Coding agent");
  const tools = agent ? AGENT_TOOLS.map((name) => ({ type: "function", function: { name } })) : undefined;
  if (t.path === "/v1/messages") {
    return JSON.stringify({ model, max_tokens: t.maxTokens, stream: t.stream, system: AGENT_SYSTEM, tools: AGENT_TOOLS.map((name) => ({ name })), messages: [{ role: "user", content: t.user }] }, null, 2);
  }
  if (t.path === "/v1/responses") {
    return JSON.stringify({ model, stream: t.stream, instructions: AGENT_SYSTEM, tools, input: t.user, max_output_tokens: t.maxTokens }, null, 2);
  }
  const body: Record<string, unknown> = {
    model,
    stream: t.stream,
    max_tokens: t.maxTokens,
    messages: [...(agent ? [{ role: "system", content: AGENT_SYSTEM }] : []), { role: "user", content: t.user }],
  };
  if (tools) body["tools"] = tools;
  if (t.responseFormat === "json_schema") {
    body["response_format"] = {
      type: "json_schema",
      json_schema: { name: "invoice", schema: { type: "object", required: ["vendor", "invoice_number", "total", "currency", "due_date"] } },
    };
  }
  return JSON.stringify(body, null, 2);
}

/** The response body, or for a stream the last events as the client received them. */
export function contextError(promptTokens: number, contextMax: number): string {
  const f = (n: number) => n.toLocaleString("en-US");
  return `The prompt is ${f(promptTokens)} tokens; this model is loaded with a ${f(contextMax)}-token context. Shorten the input or load the model with a longer context.`;
}

export function responseBody(t: RequestTemplate, id: string, model: string, output: string, promptTokens: number, outputTokens: number, done: boolean, error: string | null): string {
  if (error !== null) return JSON.stringify({ error: { type: "invalid_request_error", code: "context_length_exceeded", message: error } }, null, 2);
  const usage = { prompt_tokens: promptTokens, completion_tokens: outputTokens, total_tokens: promptTokens + outputTokens };
  if (!t.stream) {
    return JSON.stringify({ id: `chatcmpl-${id}`, object: "chat.completion", model, choices: [{ index: 0, finish_reason: done ? "stop" : null, message: { role: "assistant", content: output } }], usage }, null, 2);
  }
  if (!done) return "";
  const reason = t.cancelAt !== undefined ? null : t.tool ? "tool_calls" : "stop";
  const last = { id: `chatcmpl-${id}`, object: "chat.completion.chunk", model, choices: [{ index: 0, delta: {}, finish_reason: reason }], usage };
  return `data: ${JSON.stringify(last)}\n\n${t.cancelAt !== undefined ? "(client closed the stream)" : "data: [DONE]"}`;
}
