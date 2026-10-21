/**
 * Plain-words definitions for every technical term the app shows, used by <Term>. Security entries
 * are written from the implementing code and docs/CLAIMS.md, including what a mechanism does NOT do.
 */
export const GLOSSARY = {
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
    text: "The program that runs the model. In Cerebro that is Cerebro Engine, its own inference engine, running inside the app.",
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
  cpu: {
    term: "CPU",
    text: "The processor: the computer's main chip. It runs whatever part of a model does not fit on the graphics card, more slowly.",
  },
  gpu: {
    term: "GPU",
    text: "The graphics card. It does many small calculations at once, which is why a model runs much faster on it than on the processor alone.",
  },
  ram: {
    term: "RAM",
    text: "System memory: the computer's main working memory. Usually larger than video memory, but slower for running a model.",
  },
  gguf: {
    term: "GGUF",
    text: "The single-file format local models come in. One file holds the model's weights and the details needed to run it.",
  },
  tuning: {
    term: "tuning",
    text: "Cerebro runs real benchmarks on this machine with different engine settings and keeps the fastest configuration that stays within its safety checks.",
  },
  context: {
    term: "context length",
    text: "How many tokens the model can keep in view at once: your messages, its replies and any documents. More context needs more memory.",
  },
  offload: {
    term: "GPU layers",
    text: "A model is a stack of layers. Each layer placed on the graphics card runs fast; the rest run on the processor. All layers on the GPU is fastest when they fit.",
  },
  moeCpu: {
    term: "experts on CPU",
    text: "For a mixture-of-experts model, keeps the large expert weights of some layers in system memory while the rest stays on the graphics card. Often the fastest way to run a model bigger than video memory.",
  },
  kvCache: {
    term: "KV cache",
    text: "The model's short-term memory of the conversation so far. Storing it at lower precision (q8_0 or q4_0 instead of f16) makes it smaller, at a small cost in accuracy.",
  },
  flashAttention: {
    term: "flash attention",
    text: "A faster, leaner way to compute attention, the step where the model looks back over the context. It is required for a lower-precision V cache.",
  },
  threads: {
    term: "threads",
    text: "How many processor threads work on the parts of the model that run on the CPU. More is not always faster.",
  },
  batch: {
    term: "batch size",
    text: "How many prompt tokens are processed in one go. The micro-batch is the slice actually sent to the hardware at once; larger is faster to read a prompt but uses more memory.",
  },
  mmap: {
    term: "memory mapping",
    text: "Reads the model file straight from disk as it is needed instead of copying it all into memory first. Loads faster and shares memory with the file cache.",
  },
  speculative: {
    term: "speculative decoding",
    text: "A cheap guess of the next few tokens that the model then checks in one pass. It is distribution-preserving: the model verifies every guessed token, so a bad guess cannot steer the output, it can only waste time.",
  },
  ngram: {
    term: "n-gram speculation",
    text: "Guesses the next tokens by finding the same short run of text earlier in the conversation. Needs no second model and no extra video memory.",
  },
  prefill: {
    term: "prompt reading",
    text: "Also called prefill: how fast the model reads your message and any documents before it starts to reply, in tokens per second.",
  },
  decode: {
    term: "writing speed",
    text: "Also called decode: how fast the model writes its reply, in tokens per second. It slows as the context fills, so Cerebro measures it with an empty context and again with a working amount of text already in view.",
  },
  mmproj: {
    term: "projector",
    text: "A small companion file (often named mmproj) that turns an image into something the language model can read. A vision model needs its matching projector.",
  },
  rope: {
    term: "RoPE scaling",
    text: "How the model tracks word positions. Scaling it (YaRN or linear) stretches a model to more context than it was trained on, as its authors specify.",
  },
  chatTemplate: {
    term: "chat template",
    text: "The recipe that turns a conversation into the exact text format the model was trained on. A wrong template makes a good model answer badly.",
  },
  stopToken: {
    term: "stop token",
    text: "Text that ends a reply as soon as the model writes it, so it does not run on into a made-up next turn.",
  },
  sampler: {
    term: "sampler",
    text: "The rules that pick each next token from the model's options. Temperature sets how adventurous it is; top-k, top-p and min-p limit the choices.",
  },
  hardLink: {
    term: "hard link",
    text: "A second name for the same file on the same drive. Each app still finds the file where it expects it, but the disk stores it only once.",
  },
  sha256: {
    term: "SHA-256",
    text: "A fingerprint of a file's exact contents. Two files with the same fingerprint are the same file, byte for byte.",
  },
  ggufMeta: {
    term: "GGUF metadata",
    text: "Named settings stored inside a GGUF file, such as its context length or chat template. The engine reads them when it loads the model.",
  },
  soak: {
    term: "soak test",
    text: "A long, steady run under real load. Short bursts can look faster than a model really runs, and some settings only run out of memory after minutes, so Cerebro confirms the winner this way before keeping it.",
  },
  autofit: {
    term: "automatic settings",
    text: "What the engine does on its own: its planner puts as much of the model on the graphics card as memory allows. The most that fits is often not the fastest, which is why Cerebro measures against it.",
  },
  klDiv: {
    term: "KL divergence",
    text: "A measure of how far a setting moves the model's next-word probabilities away from a reference run. 0 means no change; the quality gate rejects anything above its stated limit.",
  },
  noiseMargin: {
    term: "noise margin",
    text: "No two runs of the same settings measure exactly alike. A candidate must beat the automatic settings by more than this (3%) to replace them, so a lucky run never changes your settings.",
  },
  depth: {
    term: "working depth",
    text: "How much text is already in the model's view while it writes, in tokens. Writing slows as this grows, so Cerebro measures at the depth your kind of work actually reaches.",
  },
  pcie: {
    term: "PCIe",
    text: "The link between the processor and the graphics card. When parts of a model live in system memory, their data crosses this link, so its traffic shows how much the card waits on the rest of the computer.",
  },
  expertCache: {
    term: "expert cache",
    text: "For a mixture-of-experts model whose experts mostly live in system memory: a set of the most-used experts kept in spare video memory. The hit rate is the share of expert lookups it answered.",
  },
  acceptance: {
    term: "draft acceptance",
    text: "The share of guessed tokens the model confirmed during speculative decoding. Each confirmed token is one the model did not have to write on its own.",
  },
  conversationCache: {
    term: "conversation cache",
    text: "Keeps the model's state for recent conversations, so a follow-up message only reads what is new instead of the whole history again. Agents that resend long histories benefit most.",
  },
  firstToken: {
    term: "first token",
    text: "Time to first token: from the moment a request arrives until the first word of the reply appears. It includes waiting in the queue and reading the prompt.",
  },
  throttling: {
    term: "thermal throttling",
    text: "When a graphics card gets too hot, its driver lowers the clock speed to protect it. The model keeps running, only slower, until the card cools.",
  },
  channel: {
    term: "release channel",
    text: "Stable features are on for everyone. Beta and Experimental features are newer optimisations you can switch on or off per feature; Experimental ones may change between versions.",
  },
} as const;

export type GlossaryKey = keyof typeof GLOSSARY;
