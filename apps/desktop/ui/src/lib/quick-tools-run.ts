import { DEFAULT_SAMPLER, type ChatRequest, type Sampler } from "../service/chat-types";
import type { CerebroService } from "../service/service";
import { buildQuickToolPrompt, type QuickToolId, type QuickToolInputValue } from "./quick-tools";

/**
 * Running a quick tool: the tool's instruction becomes the system prompt, the user's text the one
 * turn, and the reply comes back token by token. The same path serves the cards in Documents and
 * the / commands in Chat, so both give the same answer.
 */

export interface QuickToolOptions {
  tool: QuickToolId;
  input: QuickToolInputValue;
  /** The model that answers. */
  modelId: string;
  sampler?: Sampler;
  signal: AbortSignal;
  /** Called with each piece of the reply as it arrives. */
  onToken?: (text: string) => void;
}

export async function runQuickTool(service: CerebroService, options: QuickToolOptions): Promise<string> {
  const prompt = buildQuickToolPrompt(options.tool, options.input);
  const attachment = options.input.attachment;
  const request: ChatRequest = {
    modelId: options.modelId,
    systemPrompt: prompt.system,
    turns: [
      {
        role: "user",
        text: prompt.user.trim() || "Do the task.",
        ...(attachment ? { attachments: [{ name: attachment.name, type: attachment.type, bytes: attachment.bytes, data: attachment.data }] } : {}),
      },
    ],
    sampler: options.sampler ?? DEFAULT_SAMPLER,
    native: null,
    passes: 0,
    effort: "low",
    verbosity: "normal",
    source: "none",
    tools: null,
    orchestrate: null,
    keepCache: false,
  };
  let out = "";
  let failure: string | null = null;
  await service.chat(request, (e) => {
    if (e.type === "token") {
      out += e.text;
      options.onToken?.(e.text);
      return;
    }
    if (e.type === "error") failure = e.message;
  }, options.signal);
  if (failure) throw new Error(failure);
  if (!out.trim()) throw new Error("The model did not answer.");
  return out;
}
