## Part 0 — Substrate, vocabulary, and shared data model

**1. Single canonical internal tool schema.** Define one Rust type used everywhere, aliasing OpenAI's shape, so every adapter is a pure function over it:

```rust
#[derive(Clone, Serialize, Deserialize)]
struct ToolSpec {
    name: String,              // ^[a-zA-Z0-9_-]{1,64}$
    title: Option<String>,
    description: String,       // budget-capped, see item 12
    input_schema: serde_json::Value,  // JSON Schema Draft 2020-12 subset
    output_schema: Option<serde_json::Value>, // MCP 2025-06-18 structured output
    annotations: Option<ToolAnnotations>,     // MCP readOnlyHint/destructiveHint etc.
}
```
MCP's `Tool` uses `inputSchema`/`outputSchema` (camelCase) exactly; keep MCP as the source of truth and derive the OpenAI/Hermes `{"type":"function","function":{...}}` envelope on the fly.

**2. Three rendering layers, kept separate.** (a) *Spec advertisement* — how tools appear in the prompt; (b) *call emission* — how the model writes a call; (c) *result injection* — how the tool result re-enters context. These differ independently per model. Never collapse them into one function; bugs in tool calling almost always live in one layer while the other two are fine (e.g., Qwen3-Coder emitting correct XML but the app failing to wrap the response, HF discussion [Qwen/Qwen3-Coder-30B-A3B-Instruct#12](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/discussions/12)).

**3. Tool-call intermediate representation (IR).** After parsing any wire format, normalize to:

```rust
struct ToolCall { id: String, name: String, arguments: serde_json::Value, raw: String }
struct ToolResult { call_id: String, ok: bool, content: Vec<Content>, structured: Option<Value>,
                    error: Option<ErrorEnvelope>, truncated: Truncation }
```
`id` is *always minted by Cerebro*, never trusted from the model (Qwen3 emits no per-call ID; vLLM synthesizes `chatcmpl-tool-…`).

**4. Content blocks mirror MCP.** Support exactly the MCP content types so the same struct serves tool results and resources: `TextContent`, `ImageContent`, `AudioContent`, `ResourceLink`, `EmbeddedResource` (spec 2025-06-18, `server/tools`). Any richer app-native notion (file diffs, plots) becomes a `TextContent` with a MIME hint until Cerebro grows its own UI type.

**5. Capability descriptor.** A per-model struct that drives the whole pipeline:

```rust
enum ToolDialect { Hermes, Qwen3CoderXml, GemmaPythonic, GemmaFunctionTokens, Harmony }
struct ModelToolProfile {
    dialect: ToolDialect,
    supports_parallel_calls: bool,
    supports_named_tool_choice_forced: bool,
    reasoning_channel: ReasoningStyle, // think-tags | harmony-analysis | none
    stop_on_call_token: bool,
}
```

**6. Version the profile, not the model string.** Key profiles off the tokenizer/chat-template hash (e.g., SHA-256 of `chat_template.jinja` + special-token table), because fine-tunes and quantized conversions mutate behavior (the `grep`→`ggrep` corruption on a 4-bit MLX conversion is a documented example, [QwenLM/Qwen3-Coder#475](https://github.com/QwenLM/Qwen3-Coder/issues/475)).

**7. Template as executable truth.** Ship a small Jinja-ish renderer (or vendor each model's `chat_template.jinja` and implement the exact subset it uses: `tojson`, `items`, `is string`, `trim`). Rendering the template yourself guarantees byte-parity with training-time prompts, which is where tool-calling accuracy comes from (Qwen docs: "the chat template in `tokenizer_config.json` has already included support for the Hermes-style tool use", [qwen.readthedocs.io/en/latest/framework/function_call.html](https://qwen.readthedocs.io/en/latest/framework/function_call.html)).

**8. Golden byte-fixtures.** For each dialect, store input messages → exact expected prompt bytes and expected parse output. This is the unit-test surface that catches template regressions and is the cheapest defense against "the model suddenly stopped calling tools."

---

## Part 1 — Exact native tool-call formats

### 1a. Qwen3 dense — Hermes/JSON inside `<tool_call>`

**9. Envelope is ChatML.** Turn = `<|im_start|>{role}\n{body}<|im_end|>\n`. Special tokens (Qwen3-8B `added_tokens_decoder`, verified in the ompi writeup): `<tool_call>`=151657, `</tool_call>`=151658, `<tool_response>`=151665, `</tool_response>`=151666, ` thinking`=151667, ``=151668, `<|im_start|>`=151644, `<|im_end|>`=151645.

**10. System-turn advertisement (verbatim Qwen3 template).**
```
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":…,"description":…,"parameters":{…}}}
…
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```
Each tool is `tool | tojson` on its own line (HF `trl/chat_templates/qwen3.jinja`).

**11. Emission.** One block per call; `arguments` is a **nested JSON object**, not a string:
```
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA", "unit": "celsius"}}
</tool_call>
```
Multiple parallel calls = multiple consecutive blocks. The `arguments`-as-object distinction is the single most common integration bug: at the API boundary it becomes a **string**, on the wire it is an **object** (see item 16).

**12. Result injection.** Tool results are wrapped and placed in a **user** turn:
```
<|im_start|>user
<tool_response>
{"temperature": 26.1, "unit": "celsius"}
</tool_response><|im_end|>
```
Consecutive tool results share one `<|im_start|>user` turn (template groups them), each in its own `<tool_response>`.

**13. Reasoning split.** Thinking is `...`. With `enable_thinking:false` (no_think) the model skips it. vLLM pairing: `--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1`; SGLang: `--reasoning-parser qwen3`.

**14. Stop condition.** Stop generation on the first `</tool_call>` only if you must serialize; better is to allow the model to emit several calls then stop on `<|im_end|>`.

**15. Forced/required tool choice.** Qwen3's template has no native "tool_choice=required" token. Implement it by (a) injecting a system line "You must call exactly one tool," or (b) grammar-forcing the first tokens to `<tool_call>\n{"name": "` (item 88). Option (b) is deterministic and preferred for agent steps.

### 1b. Qwen3-Coder-30B-A3B — XML `<function=…><parameter=…>`

**16. Advertisement block (native template).**
```
<tools>
<function>
<name>write</name>
<description>…</description>
<parameters>
<parameter>
<name>path</name><type>string</type><description>…</description>
</parameter>
<parameter>
<name>content</name><type>string</type>
</parameter>
<required>path</required>
<required>content</required>
</parameters>
</function>
</tools>
```
`enum` is rendered as `<enum>a,b</enum>`; unknown JSON-Schema keys are normalized (`-`→`_`, `$` stripped) then emitted as tags.

**17. Emission (canonical).**
```
<tool_call>
<function=write>
<parameter=path>
/data/test.txt
</parameter>
<parameter=content>
hello world
</parameter>
</function>
</tool_call>
```

**18. Headline failure mode — missing opener.** The 30B model "frequently omits the initial `<tool_call>` tag, especially when the tool call is attempted after a textual response," producing `<function=…>…</function>` with no wrapper ([QwenLM/Qwen3-Coder#475](https://github.com/QwenLM/Qwen3-Coder/issues/475); [HF discussion 12](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/discussions/12)). Mitigation: (a) strengthen the template reminder text ("Do NOT omit the initial `<tool_call>` tag"); (b) make the parser tolerant: if `<function=` appears with no preceding `<tool_call>`, treat it as an implicit call (item 45).

**19. Second observed corruption — `grep`→`ggrep`.** A declared function named `grep` is emitted as `ggrep` deterministically at temp 0 on a 4-bit conversion; parser preserves the name verbatim. Mitigation: fuzzy name resolution against the declared tool set (item 46) + an app-side alias map. UNVERIFIED whether the base BF16 checkpoint reproduces this.

**20. Emission is spacing-sensitive.** Correct token is `<parameter=name>` terminated by `>` then a newline then value then `</parameter>`. Observed model slip: `<parameter=command=create …>` (double `=`) — `parameter=<name>` where the model writes `command=create`. The parser must tolerate a second `=` inside the parameter name and split on the **first** `=` (`WHOX qwen3_coder_parser`).

**21. Result injection identical to Qwen3.** `<tool_response>…</tool_response>` inside a user turn. The Coder template reuses Qwen3's tool_response wrapper.

**22. Reference parsers to mirror.** vLLM `qwen3_coder` parser (`vllm/entrypoints/openai/tool_parsers/qwen3_coder_tool_parser.py`) and the model's own `qwen3coder_tool_parser.py` in the HF repo. Both buffer from `<tool_call>`, emit a header delta (function name) then argument deltas as JSON fragments built from `<parameter>` blocks, converting values by schema type.

### 1c. Qwen3.8 hybrid (Gated DeltaNet) — derive, do not guess

**23. Status: UNVERIFIED as a named release.** Publicly documented Qwen hybrid is **Qwen3-Next** (80B-A3B), which mixes Gated DeltaNet linear-attention blocks with Gated Attention full-attention in a 3:1 ratio and uses a sparse MoE (512 routed + 1 shared, 10 active): Gated DeltaNet paper arXiv:2412.06464; Qwen3-Next blog ([alibabacloud.com](https://www.alibabacloud.com/blog/qwen3-next-...) ); NVIDIA writeup. "Qwen3.8" is treated here as a future derivative.

**24. Tool format expectation.** Qwen3-Next's lineage inherits the **Qwen3 Hermes** convention (same ChatML + `<tool_call>` JSON), because it reuses the Qwen tokenizer family and the same template school. UNVERIFIED until the actual `chat_template.jinja` is inspected. Do not hard-code; run item 7's extractor at load time.

**25. Derivation procedure (applies to any future Qwen build).** On model load: tokenize the template, grep for `tool_call`, `tool_response`, `<function=`, `<parameter=`; set `dialect` accordingly; extract the special-token IDs from `added_tokens_decoder`; snapshot the system preamble verbatim as the advertisement template. Cache the extracted profile keyed by template hash (item 6).

**26. Architecture implication — the recurrent state is not a KV cache.** Gated DeltaNet layers carry a fixed-size recurrent state updated by a delta rule (arXiv:2412.06464), not a growable K/V. Prefix caching (Part 8) must therefore checkpoint the recurrent state at tool-turn boundaries, not just KV blocks. Full-attention layers (25%) still use a normal KV cache. This asymmetry is the core engineering cost of the hybrid. UNVERIFIED: exact checkpoint layout for Cerebro until the layer stack is finalized.

### 1d. Gemma 3 / Gemma 4

**27. Gemma 3 has no native tool tokens in base IT; it is prompt-engineered.** Google documents two prompt recipes: Pythonic `[func_name1(params_name1=params_value1, …), func_name2(params)]`, and JSON `{"name": function name, "parameters": dictionary of argument name and its value}`. Turn format is `<start_of_turn>{role}\n…<end_of_turn>` (Simon Willison summary; [ai.google.dev Gemma function calling](https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices)).

**28. FunctionGemma (specialized Gemma 3 270M) defines real control tokens.** Pairs: `<start_function_declaration>…<end_function_declaration>`, `<start_function_call>…<end_function_call>`, `<start_function_response>…<end_function_response>`, plus an `<escape>` token used as the **string delimiter** inside structured blocks (so `{`, `}`, `,`, quotes inside strings are literal). Note: the docs render `<escape>` as a control token; treat it as tokenizer-special, not literal text.

**29. FunctionGemma call syntax.**
```
<start_function_call>call:get_current_weather{location:Tokyo, Japan<escape>…<end_function_call>
```
i.e. `call:{name}{{key:value,key:value}}` with `<escape>`-delimited strings and no quotes around keys. Declaration: `declaration:get_current_weather{description:<escape>…<escape>,parameters:{properties:{…},required:[…],type:<escape>OBJECT<escape>}}`.

**30. Mandatory developer trigger.** Gemma/FucntionGemma requires the developer turn `You are a model that can do function calling with the following functions` or it will answer in prose instead of calling. Bake this line into the advertisement layer.

**31. Cerebro policy for Gemma.** Use the FunctionGemma control-token format when the checkpoint exposes those tokens (checksum `added_tokens`); otherwise fall back to the Pythonic parser, which is what vLLM ships (`--tool-call-parser pythonic --chat-template tool_chat_template_gemma3_pythonic.jinja`, vLLM PR #17149). The Pythonic prompt fragments are documented there verbatim.

**32. Gemma 4 status: UNVERIFIED.** No confirmed public spec. Treat identically to item 25: inspect the shipped `chat_template.jinja`. Expected fallback is the FunctionGemma control-token scheme (its natural successor). Until confirmed, gate Gemma 4 tool calling behind a feature flag and log raw outputs to collect a corpus.

**33. Gemma string escaping rule.** Because `<escape>` delimits strings, the arguments parser for FunctionGemma must not JSON-unescape; it must split on the literal `<escape>` token and treat everything between as raw bytes, then coerce by schema type. This is a distinct code path from every JSON-based dialect.

### 1e. gpt-oss — Harmony

**34. Harmony is mandatory for gpt-oss.** The model "should not be used without using the harmony format, as it will not work correctly" ([OpenAI harmony cookbook](https://developers.openai.com/cookbook/articles/openai-harmony); [openai/harmony](https://github.com/openai/harmony)).

**35. Structure.** `<|start|>{header}<|message|>{content}<|end|>`; header carries role and optionally channel/recipient. Special tokens (o200k_harmony): `<|start|>`=200006, `<|end|>`=200007, `<|message|>`=200008, `<|channel|>`=200005, `<|constrain|>`=200003, `<|return|>`=200002, `<|call|>`=200012.

**36. Roles.** `system` (reasoning effort, knowledge cutoff, built-in tools), `developer` (instructions + function tools), `user`, `assistant`, `tool` (role name is the tool name). Channels: `analysis` (CoT, never shown to users), `commentary` (function calls + preambles), `final` (user-facing).

**37. Function definition uses TypeScript-like namespace, in the *developer* message.**
```
namespace functions {
// Gets the current weather in the provided location.
type get_current_weather = (_: {
// The city and state, e.g. San Francisco, CA
location: string,
format?: "celsius" | "fahrenheit",
}) => any;
} // namespace functions
```
Always return `any`; zero-arg functions are `type name = () => any`; comments become descriptions.

**38. Tool call.**
```
<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"SF"}<|call|>
```
Note the `to=` recipient may sit in the role **or** the channel section of the header; `<|constrain|>json` is optional but common.

**39. Tool result.** The calling name is the *role*:
```
<|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>{"sunny": true, "temperature": 20}<|end|>
```

**40. CoT must be replayed.** gpt-oss does tool calls inside its chain of thought, so the **raw** analysis-channel messages (not the summarized ones) must be fed back until a `final` response is reached. Returning only the summary degrades function calling (OpenAI "verifying implementations"). Store raw analysis channel text but never render it in the transcript.

**41. Stop tokens.** `<|call|>` (tool call) and `<|return|>` (done) both halt sampling. Cerebro's sampler must register both as stop tokens and, on `<|call|>`, take the agent-loop branch rather than ending the turn.

---

## Part 2 — Robust streaming parser for tool calls

**42. Two-phase design.** A byte-level *scanner* that classifies the stream into `Content` / `CallOpen` / `CallBody` / `CallClose` segments (dialect-specific delimiter table), and a *builder* that constructs IR (item 3) from segmented pieces. The scanner must be O(n) and allocation-light (a rolling `Vec<u8>` buffer with indices).

**43. Feed interface (incremental, resumable).**
```rust
trait ToolParser {
    fn feed(&mut self, delta: &[u8]) -> Vec<ParseEvent>; // Content | CallPartial | CallComplete | ContentComplete
    fn finish(&mut self) -> ParseOutcome;
    fn reset(&mut self, dialect: ToolDialect);
}
```
Model after vLLM's `extract_tool_calls_streaming(previous_text, current_text, delta_text, …)` contract (vLLM `hermes_tool_parser.py`, `qwen3_coder_tool_parser.py`).

**44. Partial-tag handling with a holdback buffer.** Never emit trailing bytes that could be a prefix of a start delimiter (`<tool_call>`, `<function=`, `to=functions.`, `<start_function_call>`, `<|channel|>`). Keep a `max_delim_len-1` byte tail in the buffer; only flush it once the next bytes prove it is plain content. This prevents the classic "user sees half of `<tool_call>` flash in the UI" bug.

**45. Implicit-call recovery for Qwen3-Coder.** If `<function=` is seen without a preceding `<tool_call>`, synthesize the opener (item 18). Record a metric `unwrapped_calls_total` — if it is non-zero you know the template guidance failed.

**46. Name resolution / fuzzy repair.** After extracting a call name, match against declared tool names with: exact → case-insensitive → Levenshtein distance ≤1 → known alias table. `ggrep`→`grep` is distance-1. Emit `tool_name_repaired_total{from,to}`.

**47. Malformed-JSON repair ladder (Hermes/Qwen3).** Attempt, in order: (1) `serde_json::from_str`; (2) trim trailing incomplete object to last balanced `}`; (3) LLM/hand JSON repair (`jsonrepair`-class: single→double quotes, trailing commas, unquoted keys, `True/False/None`→`true/false/null`); (4) regex field extraction as last resort. Log which rung succeeded; rung ≥3 means the model needs a format-tightening pass.

**48. Partial-argument streaming.** Emit argument deltas only when a complete value is buffered for a given parameter (Qwen3-Coder parses complete `<parameter>` blocks). For JSON dialects, expose partial arguments via a streaming JSON reader that can validate prefixes (`"loc" → "loca" → "locat"`) without committing invalid JSON. The UI needs partials for "typing" animation; the executor must only fire on `CallComplete`.

**49. Multiple parallel calls.** The scanner must maintain a call list with an `index`. Qwen3-Coder's parser keys off `current_tool_index` and finds each `<tool_call>` start offset independently. Store `Vec<CallBuilder>` and never assume one call per turn.

**50. Interleaved preamble text.** Harmony emits a `commentary`-channel preamble before calls; Qwen3 emits free text before `<tool_call>`. Scanner rule: bytes before the first open delimiter are `Content`; text between two calls (rare) is discarded or shown as a divider, never re-parsed as a call.

**51. Type coercion at parse time.** Use the tool's `input_schema` to coerce string parameter values into `number`/`boolean`/`array`/`object` (Qwen3-Coder parsers do this explicitly, e.g. `_convert_param_value`). If coercion fails against the schema, mark the call `invalid_arguments` and do **not** execute — feed a structured error back (item 63).

**52. `finish()` semantics.** On stream end without a `</tool_call>` (truncation, cancellation, token limit), attempt item 47 repair; if the object is unrecoverable, surface `ToolCallIncomplete` to the agent loop, which may retry (item 66) or abort.

**53. Determinism test harness.** Replay recorded token streams byte-by-byte into the parser and assert identical IR; fuzz with 1–4 byte split points around every delimiter to catch holdback bugs.

---

## Part 3 — The agent loop

**54. Loop skeleton.**
```
msgs = [system(tools)] + history + [user]
for step in 0..MAX_STEPS:
    out = engine.generate(msgs, stop=[EOS, CALL])
    msgs.push(out.assistant_message)          # includes raw CoT per item 40
    calls = parse(out)
    if calls.empty: return out.final_text
    approvals = classify(calls)               # item 58
    results = execute_parallel(calls, approvals) # items 59-61
    msgs.push(results)                         # wrapped per dialect, item 12/21/39
return "step limit reached"
```

**55. Limits.** Separate, independently configurable: `MAX_STEPS` (tool round-trips, default 8–16), `MAX_TOOL_CALLS` (default 32), `MAX_OUTPUT_TOKENS` per step, `MAX_WALL_CLOCK` for the whole turn, `MAX_PARALLEL_CALLS` (default 4). All enforced in the loop, none in the model prompt (prompts lie).

**56. Timeouts.** Per-tool execution timeout (default 30 s, MCP tools inherit the server's advertised behavior), per-generation timeout, and a global turn deadline propagated as a `tokio::time::Instant`. On timeout, produce a `ToolResult{ok:false, error:Timeout}` and *continue* the loop (give the model a chance to recover) unless the global deadline fired.

**57. Cancellation.** Thread a `CancellationToken` from the UI (SolidJS abort) through: the sampler (stop decoding), each tool future (`tokio::select!`), and the MCP client (send `notifications/cancelled`, item 82). On cancel, persist the partial transcript with a `cancelled` marker so the user can resume. MCP cancellation is cooperative; do not assume the server stops instantly.

**58. Per-tool approval policy.**
```rust
enum Approval { Always, Ask, Never }
struct ToolPolicy { approval: Approval, read_only: bool, max_calls_per_turn: u32, destructive: bool }
```
Derive defaults from MCP `annotations`: `readOnlyHint:true` → `Always` allowed; `destructiveHint:true` → `Ask`; `title`/`description` shown verbatim in the dialog but treated as **untrusted** (spec: "clients MUST consider tool annotations to be untrusted unless they come from trusted servers").

**59. Read-only vs write classification.** Do not trust annotations alone; Cerebro maintains a local allow/deny list keyed by `server::tool`. Local built-ins (`generate`, `embed`, `read_file`) are classified at compile time. MCP tools default to `Ask` until the user marks the server trusted.

**60. Parallel execution.** Group calls that are all `read_only` and approved; run with `futures::future::join_all` bounded by a semaphore. Serialize any `write`/`destructive` call relative to other writes to avoid state races. Preserve **call order** when assembling results (item 62), regardless of completion order.

**61. Result truncation strategies.** Tiered: (a) hard byte cap (e.g., 8 KiB per result, 32 KiB per turn) with head+tail preservation and a `…[truncated N bytes]…` marker; (b) type-aware (JSON → keep top-level keys, drop array tails; logs → keep first+last K lines); (c) summarize-via-model offload when over budget (optional, costly). Always set `ToolResult.truncated` and tell the model what was cut, so it can ask for narrower queries.

**62. Result correlation.** Never rely on order alone for models that ignore `tool_call_id` (Qwen3's template ignores the id). Deduplicate/serialize by the Cerebro-minted `id` (item 3) and render results in call order.

**63. Structured errors fed back.** On failure, synthesize an MCP-shaped result: JSON-RPC error object plus a `TextContent` human string, e.g. `{"isError":true,"content":[{"type":"text","text":"invalid_arguments: 'unit' must be one of [celsius,fahrenheit]"}]}`. MCP 2025-06-18 allows an `isError` flag on tool results; mirror it.

**64. Loop guard — no progress detection.** Hash each assistant message + result set; if two consecutive steps are byte-identical (model repeating a failed call), break and surface the failure. Cap consecutive identical calls at 2.

**65. Token/latency budget accounting.** Track per step: prompt tokens, generated tokens, TTFT, decode ms, tool ms; sum into the turn. This is the data that tells you whether prefix caching (Part 8) is paying off.

**66. Retry policy.** Distinguish (a) *parse failure* → repair (item 47) or one re-prompt with "your last output was malformed in this way: …"; (b) *tool execution failure* → feed error back (item 63), let the model retry with corrected args; (c) *transient network* (MCP) → exponential backoff with jitter, max 3, then surface.

---

## Part 4 — Grammar-constrained decoding of tool arguments

**67. Why constrain arguments.** Free-form JSON from an 8 GB model will occasionally be invalid; constrained decoding makes argument objects schema-valid by construction, and JSONSchemaBench (arXiv:2501.10868) shows constrained decoding improves downstream accuracy up to ~4% and can speed generation ~50% vs unconstrained. Caveat: the *engine* matters — a third-party comparison reports that `llguidance` and `xgrammar` can over-constrain enough to fall *below* the unconstrained baseline on some models, while the reference `dotjson` engine posted zero provable mask errors across 5,047 schemas ([dottxt blog](https://blog.dottxt.ai/comparing-the-quality-of-structured-generation-engines.html)). Treat token-mask correctness as a first-class correctness concern, not just a performance trick.

**68. Three implementation tiers for Cerebro's Rust sampler.**
1. **GBNF subset** (llama.cpp grammar language, `ggml-org/llama.cpp/grammars/README.md`): hand-written grammar per dialect; simplest to integrate with a token trie.
2. **JSON-Schema → grammar compiler** (llguidance, written in Rust, `guidance-ai/llguidance`, docs `json_schema.md`; supports Draft 2020-12 subset, `anyOf`/`oneOf`/`allOf`/`$ref`/`enum`/`const`, fixed property order).
3. **Token-mask engine** (xgrammar, arXiv:2411.15100, adaptive token-mask cache + PDA). Choose llguidance for a pure-Rust codebase; choose xgrammar via FFI only if you need its C++ speed after profiling.

**69. Where to plug the mask.** In the sampler, immediately before softmax: compute `allowed: &[bool]` over vocab, set `logits[i] = -inf` for disallowed, then argmax/sample as normal. For int8/f16 CUDA paths, apply the mask on the logits tensor after the final matmul and before the (fused) softmax/top-k; keep it as a device-side scatter so you do not round-trip logits to host.

**70. Token trie construction (one-time, per vocab).** Build a trie of all token byte-strings (or a DAWG). The grammar's current FSM/stack state filters which trie branches are live. Precompute once at model load; store in a compact arena (`Vec<u32>` children, byte-labeled).

**71. Token-mask computation algorithm (cheap, per step).** Given parser state S and token trie T: DFS T from root; at each node, advance S by the node's byte label; if S dies, prune; if S reaches a token-terminal node, mark that token allowed. Cache `(S, position) → mask bitmap` — xgrammar's "adaptive token mask cache" and llguidance's "slicer" are exactly this (llguidance: average mask < 50 µs on a 128k tokenizer; worst case ~1.5 ms).

**72. Mask bitmap representation.** `Vec<u64>` of `(vocab+63)/64` words; AND with the sampling distribution's support. For Cerebro's vocab (~150k for Qwen, ~256k for Gemma 3) that is ~2–4 KiB per step — allocate once, reuse.

**73. Backtrack-stack PDA for CFGs.** If you implement GBNF yourself, use a pushdown automaton with an explicit stack of (rule, position, return) frames, matching llama.cpp's cell-based stack (the `llama_grammar` cell model). On each accepted token, advance the PDA by the token's bytes; resolve partial rule matches at byte granularity.

**74. Fast-forward / forced tokens.** When the grammar admits exactly one continuation for the next k tokens, emit them without sampling (a "fast-forward"). This is worth it for the literal boilerplate in tool calls: every Hermes call starts with the byte-identical `<tool_call>\n{"name": "` — fast-forwarding that saves 6–8 sampling steps per call and removes a class of errors.

**75. Dialect grammars to ship.**
- **Hermes:** `<tool_call>\n{"name": <enum of tool names>, "arguments": <schema-object>}\n</tool_call>` repeated 0..n.
- **Qwen3-Coder:** `<tool_call>\n<function=<enum>>\n(<parameter=name>\nvalue\n</parameter>)*\n</function>\n</tool_call>`.
- **Harmony:** constrain the `commentary` message content to the JSON argument object; force `to=functions.<enum>` by pinning the recipient token sequence.
- **FunctionGemma:** `<start_function_call>call:<enum>{<key>:value,...}<end_function_call>` with `<escape>` string rules.

**76. Enforcing real constraints, not cosmetics.** Ensure `required`, `enum`, numeric ranges, `minItems/maxItems`, and `patternProperties` disjointness are enforced (llguidance documents departures: property order is fixed; unrecognized `format` errors by default). Document every divergence and test with the official JSON Schema Test Suite plus JSONSchemaBench.

**77. Reasoning interaction.** If you constrain from the first sampled token, you must include the ` thinking…` / `analysis` channel in the grammar, or the model cannot think. llguidance's worked example literally prefixes the JSON with a regex for `...(address|refusal)`. Otherwise, only enable the mask *after* the reasoning terminator is emitted.

**78. Cost estimate for sm_86 / i9-12900H.** Mask generation is CPU work; at ~50 µs/token with a 150k vocab (llguidance) it is negligible vs GPU decode (~10–20 ms/token for a 3B-active MoE). Prefer CPU mask generation on a worker thread overlapped with the previous token's GPU work. Measure `mask_us_per_token` separately from `decode_ms_per_token`.

**79. Failure modes.** (a) engine rejects the schema at compile time → fall back to unconstrained + post-hoc validation; (b) over-constraining forbids a valid value → detect via "model repeatedly requests an invalid tool / dead-ends" and disable masking for that turn; (c) grammar accepts invalid JSON (under-constraining) → always post-validate parsed arguments against the schema anyway (defense in depth).

---

## Part 5 — Cerebro as an MCP server

**80. Two transports, one tool registry.** Implement `stdio` (newline-delimited JSON-RPC 2.0 over a pipe) for local editor/agent clients, and **Streamable HTTP** (protocol 2025-06-18) for network clients: a single `/mcp` endpoint supporting POST and GET, optional SSE streaming, `Mcp-Session-Id` header minted at initialize, and `MCP-Protocol-Version: 2025-06-18` on subsequent requests (spec `basic/transports`). Server MAY require the session id; clients must send `validate 400` handling on missing session id.

**81. Initialize handshake.** Respond to `initialize` with `protocolVersion`, `capabilities` (declare `tools`, `resources` with `subscribe`/`listChanged`, `prompts`, `logging`, `completions` as supported), and `serverInfo`. Advertise only what you implement; clients negotiate.

**82. Tool surface to expose.**
- `generate` — args: `model`, `prompt` or `messages`, `max_tokens`, `temperature`, `stop[]`, optional `grammar`/`json_schema`. Result: text + token usage.
- `embed` — args: `model`, `input[]`; result: vectors (as base64 f32 or a resource link).
- `list_models` — result: id, params, quantization, context length, loaded status, VRAM footprint. Read-only.
- `load_model` / `unload_model` — args: model id, quant, `gpu_layers`, `ctx_len`. Destructive/expensive → annotate `destructiveHint:true`.
- `benchmark` — args: model, prompt set, n_tokens; result: TTFT, tok/s, memory. Long-running → emit progress (item 90).
- `list_tools`/`list_resources` serve the app's own introspection (meta).

**83. Structured output for every tool.** Give each tool an `outputSchema` and return `structuredContent` alongside a backward-compatible `TextContent` with the serialized JSON (spec `server/tools`: "a tool that returns structured content SHOULD also return the serialized JSON in a TextContent block"). Clients can then validate without parsing prose.

**84. Resources to expose.** `model://{id}/info` (static metadata), `model://{id}/template` (the chat template text — genuinely useful to external clients), `chat://{session}/transcript`, `engine://stats` (VRAM/RAM, KV occupancy), `gguf://{path}` (file-as-resource, with `resources/read` returning blobs). Support `resources/list`, `resources/read`, `resources/templates/list`, `resources/subscribe` for live stats.

**85. Prompts to expose.** `prompts/list` / `prompts/get` for reusable workflows: "summarize with model X", "code-review diff", "benchmark sweep". Prompts carry arguments and return `messages[]`, letting an editor client invoke Cerebro workflows without knowing the tool JSON.

**86. Auth and safety for the HTTP transport.** This endpoint starts a local inference server; bind to `127.0.0.1` by default. If exposed on a network, require an OAuth bearer token and implement MCP's protected-resource metadata (RFC 9728) flow. Log every `tools/call` with client identity (item 71/audit).

**87. Benchmark target.** "Expose the app itself as an MCP server" is satisfied when an external MCP client can: `initialize` → `tools/list` → `tools/call{"name":"generate"}` with a JSON-schema-constrained argument and receive `structuredContent`, and `resources/read` the model template. Write an automated MCP conformance test (item 115).

---

## Part 6 — MCP features beyond tools

**88. Structured tool output** (new in 2025-06-18, PR #371): `outputSchema` + `structuredContent`, with the requirement that structured results conform to the schema and clients SHOULD validate. Cerebro both *produces* (server side) and *validates* (client side) these.

**89. Resources.** `resources/list`, `resources/read` (text or base64 blob), `resources/templates/list` (RFC 6570 URI templates), `resources/subscribe` + `notifications/resources/updated`, and `notifications/resources/list_changed`. Use for exposing model files, transcripts, and live engine stats.

**90. Progress.** Clients pass a `progressToken` in request `_meta`; servers emit `notifications/progress{progressToken, progress, total, message}`. Wire this to `benchmark` and `load_model` so the UI shows a real progress bar; also mirror it to Cerebro's own transcript.

**91. Cancellation.** Send `notifications/cancelled{requestId, reason}` when the user aborts. Servers must treat it as best-effort. Cerebro's server side must actually check cancellation between batches and free the request.

**92. Sampling (server→client).** An MCP server can ask the *client* to run an LLM completion via `sampling/createMessage` with a `modelPreferences` hint. Cerebro-as-client must gate this behind explicit approval (it is a request for compute) and let the user pick which local model to serve; Cerebro-as-server can use it to delegate to a host model. Include `stopReason`.

**93. Elicitation** (new in 2025-06-18, PR #382): `elicitation/create{message, requestedSchema}` where the schema is a **flat object of primitives** (string/number/boolean, plus `enum`); response is a three-action model `accept` / `decline` / `cancel`. Rules: servers MUST NOT request sensitive data; clients SHOULD rate-limit, show which server is asking, and allow decline at any time. Render as a form; map `enum` to a select, boolean to a checkbox.

**94. Roots.** Client advertises `roots` capability and answers `roots/list` with filesystem/URI boundaries; emits `notifications/roots/list_changed` when the user adds/removes a workspace folder. Cerebro uses roots to sandbox file tools per chat session.

**95. Logging.** Client sends `logging/setLevel`; server emits `notifications/message{level, logger?, data}` (RFC 5424 severities: debug…emergency). Route these into the app's log viewer, tagged by server.

**96. Ping / liveness.** `ping` → `{}` for keepalive; use it in the Streamable HTTP transport and before long tool calls.

**97. Completions.** `completion/complete` for prompt/resource argument autocompletion. Cerebro can offer model-id and tool-name completions.

**98. No JSON-RPC batching as of 2025-06-18.** The spec removed JSON-RPC batching (PR #416); do not rely on batched arrays when talking to modern servers, and reject them on your server.

**99. Version negotiation is mandatory.** Client sends `initialize` with its `protocolVersion`; server responds with its own; if unsupported, the client may retry. All HTTP requests then carry `MCP-Protocol-Version`. A server that omits the header handling must assume `2025-03-26` for back-compat (spec `basic/transports`).

**100. Client capability gating.** Only send elicitation/sampling/roots requests if the client declared those capabilities during initialize; otherwise degrade gracefully (fall back to a plain error or a tool-argument request).

---

## Part 7 — UX: rendering, approvals, retries

**101. Transcript rendering model.** Each tool call is a **collapsible card**: header (icon, `server::tool`, status chip: pending/running/ok/error/cancelled), body showing pretty-printed arguments, expandable result with syntax highlighting and truncation badge, and a latency/step badge. Read-only calls auto-collapse; destructive ones stay expanded.

**102. Streaming call rendering.** Show partial arguments as they stream (item 48) with a shimmer; do not show raw delimiters (`<tool_call>`, `<|channel|>`) ever — strip control tokens in the streaming layer, not the renderer.

**103. CoT rendering.** gpt-oss analysis channel is "not subject to the same safety standards" and should not be shown; render a "thinking…" affordance only, with an opt-in expander that is clearly labeled as model internal monologue. Never put raw CoT in shareable/exported transcripts by default.

**104. Approval cards.** For `Ask` tools, present: tool name, human title, a diff-like preview of the arguments, and buttons **Approve / Approve always / Decline**. "Approve always" writes to the policy store (item 58). Include the *source* (which MCP server) and mark server-supplied descriptions as unverified text (item 58/spec security note).

**105. Elicitation UX.** Render `requestedSchema` as a native form (item 93); disable fields not in `required`; show the requesting server's identity; provide Decline and Cancel distinctly (they mean different things to the server).

**106. Error and retry UX.** On tool failure show the structured error (item 63) inline with a "Retry" button (re-runs the same call) and "Let model retry" (continues the loop). On parse failure, offer "Show raw output" so power users can see exactly what the model emitted — this is how format bugs get diagnosed in the wild.

**107. Empty/aborted turns.** If the user cancels mid-tool, render the partial card as `cancelled` and keep the transcript resumable (item 57). Do not silently drop the turn.

**108. Accessibility & keyboard.** Approvals must be reachable by keyboard; destructive approvals require a deliberate action (not the default-focused button); announce streaming completion via ARIA live region.

**109. Cost/latency transparency.** Append a per-turn footer: steps used, tools called, tokens generated, total time, cache-hit ratio (Part 8). Users tuning local models need this to understand why a turn was slow.

---

## Part 8 — KV-cache reuse across tool turns (prefix caching)

**110. The reuse opportunity.** In an agent loop, each step re-sends `system(tools) + history + [tool results]`. The system+tool-schema block and all prior turns are an unchanged prefix; only the newest result is appended. Reusing that prefix's KV is the single largest latency win for tool calling: SGLang's RadixAttention reuses KV across calls via a radix tree with LRU eviction (arXiv:2312.07104), and vLLM does hash-based block prefix caching where a block is keyed by its token contents plus all preceding tokens (vLLM `design/prefix_caching`).

**111. Data structure for Cerebro.** A **radix tree over token IDs** (`prefix_tokens → KV block range`). Each node stores: token span, refcount, last-used tick, and the KV block indices. Operations: longest-prefix match on new prompt, insert, touch, evict LRU leaf. Cache the *fully rendered* prompt tokens (template output), not the message list — because tool-result wrapping changes the exact bytes (item 12/21/39).

**112. Reuse algorithm per turn.**
```
new_tokens = render(msgs)                       # exact bytes incl. tool wrappers
n = radix.longest_prefix(new_tokens)
kv_keep = radix.blocks_for_prefix(n)
prefill(new_tokens[n..])                        # only the suffix
decode(...)
radix.insert(new_tokens, kv_range)
```
The engine must support starting prefill at `prefill_pos = n` while retaining the KV for `[0..n)`.

**113. Cache-key correctness.** Hash includes: token ids, model id, quantization, LoRA/adapter id, RoPE config, and any cache salt. vLLM recommends a reproducible hash (`sha256_cbor`) for cross-run determinism and offers `cache_salt` for multi-tenant isolation. Cerebro should salt per session if multiple chats share the model.

**114. Interaction with MoE offload and the hybrid architecture.** (a) MoE expert offload does not change KV layout, so prefix reuse is orthogonal — but re-prefill after a cache miss re-activates experts, so cache misses cost more than dense models; measure both. (b) For **Gated DeltaNet** layers the "cache" is recurrent state (fixed size), so a prefix checkpoint must store per-layer recurrent state snapshots at tool-turn boundaries (arXiv:2412.06464); only the 25% full-attention layers use growable KV (Qwen3-Next 3:1 block pattern). This makes checkpointing cost asymmetric: snapshot both, keyed together. UNVERIFIED: exact snapshot size/format until Cerebro's layer stack is frozen.

**115. Cache invalidation triggers.** Tool-schema change (user toggles a tool), model reload, quant change, system-prompt edit, template hash change, and `roots` change must all flush or re-key the cache. An "invalidated but reused" cache is a correctness bug, not just a perf bug.

**116. Measurement.** Log per turn: `prefix_tokens_reused / total_prompt_tokens` (cache hit rate), `ttft_ms` with and without cache, `prefill_tokens_per_s`, and `cache_evictions`. Benchmark with a fixed scripted agent transcript (system + 5 tool calls) and report the TTFT delta from caching. Compare against a "no-cache" build flag.

**117. Persistence (optional).** Like llama.cpp's slot save/restore (`/slots/{id}?action=save`), Cerebro can serialize a session's KV blocks to disk so a reopened chat skips prefill. Include a version/hash header; refuse to load on mismatch (item 113).

---

## Part 9 — Cross-cutting failure modes and how to measure each

**118. Format-drift metric suite.** Track, per model/dialect: `missing_opener_total` (item 18), `name_repaired_total` (item 46), `json_repair_rung_histogram` (item 47), `schema_violation_total` (item 51), `unwrapped_calls_total`, `empty_call_total`. These six numbers tell you whether a model/template combo is production-ready.

**119. Accuracy benchmarking.** Use Berkeley Function Calling Leaderboard (BFCL, `live_multiple` and multi-turn) for tool-selection and argument accuracy; JSONSchemaBench (arXiv:2501.10868) for schema coverage/compliance; MaskBench for token-mask correctness (item 67). Record unconstrained vs constrained deltas per model — over-constraining can regress (item 67).

**120. Latency benchmarking.** Define a standard agent trace; measure TTFT, decode tok/s, mask µs/token, tool wall time, and cache hit rate. Report first-turn (cold) vs steady-state (warm cache).

**121. MCP conformance testing.** Automate: initialize/version negotiation, capability gating, `tools/list`→`tools/call` round trip, `structuredContent` validation, `elicitation/create` accept/decline/cancel, `notifications/cancelled`, progress delivery, session-id/404 re-init, and `MCP-Protocol-Version` enforcement. Target both stdio and Streamable HTTP.

---

## Ranked top-20 (highest impact first)

1. **Byte-exact template rendering per model (items 7–8).** Tool-calling accuracy *is* prompt parity; everything else is downstream.
2. **Dialect-specific streaming parser with holdback + implicit-opener recovery (items 42–51).** Qwen3-Coder's missing `<tool_call>` is the #1 real-world breakage.
3. **MCP 2025-06-18 client with Streamable HTTP + stdio, correct session/version handshake (items 80–82, 99).**
4. **Per-tool approval policy (Always/Ask/Never) from untrusted-safe annotations (items 58–59).**
5. **Radix-tree prefix caching with `prefill_pos = n` (items 110–117).** Largest latency win in the loop.
6. **Structured tool output both directions: `outputSchema`/`structuredContent` (items 83, 88).**
7. **Function-call IR with Cerebro-minted IDs + strict schema validation before execution (items 3, 51).**
8. **Malformed-JSON repair ladder, instrumented by rung (item 47).**
9. **Grammar-constrained argument decoding via llguidance (Rust) or xgrammar (items 67–78).**
10. **Harmony support done right, including raw-CoT replay and `<|call|>`/`<|return|>` stops (items 34–41).**
11. **Result truncation with explicit markers and budget accounting (items 61, 65).**
12. **Parallel read-only calls, serialized writes, order-preserving result assembly (items 60, 62).**
13. **Cancellation threaded through sampler + tools + MCP notifications (items 57, 91).**
14. **Structured errors fed back so the model can self-correct (items 63, 66).**
15. **Gemma/FunctionGemma `<escape>`-delimited string parsing as a dedicated path (items 27–33).**
16. **Model profile extraction at load time keyed by template hash (items 6, 25).**
17. **Progress notifications wired to load/benchmark UI (items 90).**
18. **Elicitation rendered as native forms with accept/decline/cancel semantics (items 93, 105).**
19. **Elicitation/sampling/roots capability gating (item 100).**
20. **Format-drift + conformance + BFCL/JSONSchemaBench measurement suite gating releases (items 118–121).**

---

## Primary references

- Qwen function calling guide — https://qwen.readthedocs.io/en/latest/framework/function_call.html ; repo `QwenLM/Qwen3` `docs/source/framework/function_call.md`.
- Qwen3 chat template — `huggingface/trl` `trl/chat_templates/qwen3.jinja`.
- Qwen3-Coder XML format & native-template unreliability — HF discussion `Qwen/Qwen3-Coder-30B-A3B-Instruct/discussions/12`; issue `QwenLM/Qwen3-Coder#475`; vLLM parser docs `docs.vllm.ai` (`vllm.parser.qwen3`); model repo `qwen3coder_tool_parser.py`.
- Hermes convention & token IDs — `NousResearch/Hermes-Function-Calling`; vLLM `examples/tool_chat_template_hermes.jinja`, `vllm/tool_parsers/hermes_tool_parser.py`.
- gpt-oss / Harmony — https://developers.openai.com/cookbook/articles/openai-harmony ; `openai/harmony`; `openai/gpt-oss` (`gpt_oss/responses_api/types.py`).
- Gemma / FunctionGemma formatting — https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices ; vLLM PR #17149 (`pythonic_tool_parser`).
- MCP spec 2025-06-18 — https://modelcontextprotocol.io/specification/2025-06-18 (transports, server/tools, client/elicitation, changelog); `modelcontextprotocol/modelcontextprotocol` PRs #371, #382, #416, #548.
- XGrammar — arXiv:2411.15100. llguidance — `guidance-ai/llguidance` (README, `docs/json_schema.md`); https://guidance-ai.github.io/llguidance/llg-go-brrr.
- JSONSchemaBench — arXiv:2501.10868. Engine-quality comparison — https://blog.dottxt.ai/comparing-the-quality-of-structured-generation-engines.html.
- GBNF grammar language — `ggml-org/llama.cpp/grammars/README.md`.
- RadixAttention / SGLang — arXiv:2312.07104. vLLM prefix caching — https://docs.vllm.ai/en/latest/design/prefix_caching/ ; llama.cpp slot save/restore — `ggml-org/llama.cpp` discussion #8860.
- Gated DeltaNet — arXiv:2412.06464 ; Qwen3-Next blog (Alibaba Cloud) ; `huggingface/transformers` `models/qwen3_next/modular_qwen3_next.py`.

**Unverified items flagged in text:** items 19 (ggrep on base weights), 23/24/26 (Qwen3.8 naming and exact format), 32 (Gemma 4 spec), 68 (xgrammar Rust binding maturity), 114 (hybrid recurrent-state checkpoint format). All should be re-confirmed against the actual `chat_template.jinja` and tokenizer files of each checkpoint at integration time.
