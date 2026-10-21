/**
 * What a model's publisher allows, in plain words, before a file is downloaded or run.
 *
 * The catalogue, the Hub and the library all use this to show one licence line per model. Names
 * are matched loosely (a repository id, a file name and a display name all work); the Hub's own
 * `license:` tag is the fallback when the name is not one this file knows. Keep every entry short
 * enough to read in a table, and never imply a licence grants more than it does.
 */

export type CommercialUse = "yes" | "no" | "conditions" | "unknown";

export interface ModelLicence {
  /** The licence's own name, e.g. "Apache-2.0". */
  name: string;
  /** Whether the licence allows commercial use, and whether conditions apply. */
  commercial: CommercialUse;
  /** One plain sentence: what the licence allows or requires. */
  summary: string;
}

const APACHE: ModelLicence = {
  name: "Apache-2.0",
  commercial: "yes",
  summary: "Commercial use is allowed. Keep the licence and notice if you share the model.",
};

const MIT: ModelLicence = {
  name: "MIT",
  commercial: "yes",
  summary: "Commercial use is allowed. Keep the copyright and licence notice if you share it.",
};

const GEMMA: ModelLicence = {
  name: "Gemma Terms of Use",
  commercial: "conditions",
  summary: "Commercial use is allowed under the Gemma Terms of Use, including their acceptable-use rules.",
};

const LLAMA: ModelLicence = {
  name: "Llama Community Licence",
  commercial: "conditions",
  summary: "Commercial use is allowed under the Llama Community Licence, its acceptable-use policy and its user limit.",
};

const FLUX_DEV: ModelLicence = {
  name: "FLUX.1 Non-Commercial Licence",
  commercial: "no",
  summary: "Not for commercial use. This licence covers non-commercial work only.",
};

const SD_COMMUNITY: ModelLicence = {
  name: "Stability AI Community Licence",
  commercial: "conditions",
  summary: "Commercial use is allowed for organisations under $1M annual revenue. Larger ones need a paid licence.",
};

const SD_TURBO: ModelLicence = {
  name: "Stability AI Non-Commercial Research Licence",
  commercial: "no",
  summary: "Not for commercial use. This is a non-commercial research licence.",
};

const OPENRAIL: ModelLicence = {
  name: "CreativeML OpenRAIL++",
  commercial: "conditions",
  summary: "Commercial use is allowed, with the use restrictions listed in the licence.",
};

const UNKNOWN: ModelLicence = {
  name: "Check the model card",
  commercial: "unknown",
  summary: "The licence is on the model's page. Read it before you download: it decides whether commercial use is allowed.",
};

/** Tags Hugging Face publishers use, mapped to the same plain answers. */
const TAGS: Readonly<Record<string, ModelLicence>> = {
  "apache-2.0": APACHE,
  mit: MIT,
  "gemma": GEMMA,
  "llama3": LLAMA,
  "llama3.1": LLAMA,
  "llama3.2": LLAMA,
  "llama3.3": LLAMA,
  "llama4": LLAMA,
  "cc-by-nc-4.0": { name: "CC BY-NC 4.0", commercial: "no", summary: "Not for commercial use. Non-commercial sharing needs attribution." },
  "cc-by-nc-sa-4.0": { name: "CC BY-NC-SA 4.0", commercial: "no", summary: "Not for commercial use. Non-commercial sharing needs attribution and the same licence." },
  "cc-by-4.0": { name: "CC BY 4.0", commercial: "yes", summary: "Commercial use is allowed with attribution." },
  "openrail++": OPENRAIL,
  other: UNKNOWN,
};

/** The licence a model name, repository id, family or Hub tag points at. */
export function licenceForModel(name: string, family?: string | null, tag?: string | null): ModelLicence {
  const text = `${name} ${family ?? ""}`.toLowerCase();
  // The publisher's own licence tag is the most reliable source when it names a known licence.
  const key = (tag ?? "").toLowerCase().replace(/^license:/, "").trim();
  if (key && key !== "other" && TAGS[key]) return TAGS[key];

  // Most specific first: FLUX dev and schnell, and the Stability image families, differ sharply.
  if (/flux[.\s_-]*1[.\s_-]*dev|flux1-dev/.test(text)) return FLUX_DEV;
  if (/flux[.\s_-]*1[.\s_-]*schnell|flux1-schnell/.test(text)) return APACHE;
  if (/sd[.\s_-]*turbo|sdxl[.\s_-]*turbo/.test(text)) return SD_TURBO;
  if (/stable[.\s_-]*diffusion[.\s_-]*2|sd[.\s_-]*2[.\s_-]*1?/.test(text)) return OPENRAIL;
  if (/sd[.\s_-]*3[.\s_-]*5|stable[.\s_-]*diffusion[.\s_-]*3[.\s_-]*5|sd3\.5/.test(text)) return SD_COMMUNITY;
  if (/sdxl|stable[.\s_-]*diffusion[.\s_-]*xl/.test(text)) return OPENRAIL;
  if (/qwen[.\s_-]*image/.test(text)) return APACHE;

  // Distilled models carry the distiller's licence even when the name contains the teacher's.
  if (/\bdeepseek/.test(text)) return MIT;
  if (/\bqwen/.test(text)) return APACHE;
  if (/\bgemma/.test(text)) return GEMMA;
  if (/\bllama/.test(text)) return LLAMA;
  if (/mistral[.\s_-]*(nemo|small)/.test(text)) return APACHE;
  if (/mistral[.\s_-]*medium/.test(text)) {
    return { name: "Mistral Medium Modified MIT", commercial: "yes", summary: "Commercial use is allowed. The licence adds conditions to the standard MIT terms." };
  }
  if (/\bphi-?4\b/.test(text)) return MIT;
  if (/gpt[.\s_-]*oss/.test(text)) return APACHE;
  if (/\bglm[.\s_-]*4/.test(text)) return MIT;
  if (/\bwhisper/.test(text)) return MIT;
  if (/outetts|\borpheus\b/.test(text)) return APACHE;
  if (/\blaguna\b/.test(text)) return UNKNOWN;

  return TAGS[key] ?? UNKNOWN;
}

/** Short badge text for a licence, e.g. "Commercial use" or "No commercial use". */
export function commercialLabel(use: CommercialUse): string {
  switch (use) {
    case "yes":
      return "Commercial use allowed";
    case "no":
      return "No commercial use";
    case "conditions":
      return "Commercial use with conditions";
    case "unknown":
      return "Licence on the model page";
  }
}
