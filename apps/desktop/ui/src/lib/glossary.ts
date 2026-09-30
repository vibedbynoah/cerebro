/**
 * Plain-words definitions for every technical term the app shows, used by <Term>. Security entries
 * are written from the implementing code and docs/CLAIMS.md, including what a mechanism does NOT do.
 */
export const GLOSSARY = {
  llamacpp: {
    term: "llama.cpp",
    text: "llama.cpp is the open-source program that actually runs an AI model on your CPU or GPU. Cerebro configures and drives it; it does not fork or replace it.",
  },
  tokps: {
    term: "tok/s",
    text: "Tokens per second: how fast a model writes. A token is a word or a piece of a word.",
  },
  vram: {
    term: "VRAM",
    text: "Video memory: the memory on your graphics card. The more of a model that fits in it, the faster the model usually runs.",
  },
  backend: {
    term: "backend",
    text: "The program that runs the model. In Cerebro that is llama.cpp's server, which Cerebro starts, watches and stops.",
  },
  loopback: {
    term: "loopback",
    text: "127.0.0.1, an address only this computer can reach. Other devices on your network cannot connect to it.",
  },
  quant: {
    term: "quantization",
    text: "Storing a model's numbers with fewer bits (Q4_K_M is about 4 bits each) so it needs less memory, at a small cost in quality.",
  },
  moe: {
    term: "MoE",
    text: "Mixture of experts: a model that uses only a few of its parts for each token, so a large model can still run quickly.",
  },
  jobObject: {
    term: "Job Object",
    text: "A Windows feature that groups a process with everything it starts. When Cerebro stops, Windows ends the whole group. It does not limit which files or network addresses the model process can open.",
  },
  integrity: {
    term: "integrity check",
    text: "Before every start, Cerebro fingerprints each program file in the backend folder and compares it with the list it verified. A changed, extra or missing file stops the start.",
  },
  cleanEnv: {
    term: "clean environment",
    text: "The model process gets a short, fixed list of environment settings built from scratch, never a copy of Cerebro's own, so none of Cerebro's variables or secrets reach it.",
  },
  api: {
    term: "OpenAI-compatible API",
    text: "A local web address that speaks the same request format as OpenAI's API, so existing tools can use your local model instead.",
  },
  rag: {
    term: "RAG",
    text: "Retrieval-augmented generation: the model first looks up the most relevant passages in your documents, then answers using them.",
  },
  quality: {
    term: "quality",
    text: "How well a model answers, not how fast: its score on a task benchmark run on this machine. Cerebro always shows it beside speed, because a fast model that answers badly is not a win. A dash means no quality run has finished yet.",
  },
  tuning: {
    term: "tuning",
    text: "Cerebro runs real benchmarks on this machine with different llama.cpp settings and keeps the fastest configuration that stays within its safety checks.",
  },
} as const;

export type GlossaryKey = keyof typeof GLOSSARY;
