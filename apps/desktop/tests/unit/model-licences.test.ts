import { describe, expect, it } from "vitest";
import { commercialLabel, licenceForModel } from "../../ui/src/lib/model-licences";

describe("model licence lookup", () => {
  it("maps the catalogue's model families to their publisher's licence", () => {
    const cases: Array<[string, string, string, string]> = [
      ["Qwen3-8B", "", "Apache-2.0", "yes"],
      ["Qwen2.5-VL-7B-Instruct", "", "Apache-2.0", "yes"],
      ["gemma-4-12b-it", "", "Gemma Terms of Use", "conditions"],
      ["Llama-3.1-8B-Instruct", "", "Llama Community Licence", "conditions"],
      ["Mistral-Nemo-Instruct-2407", "", "Apache-2.0", "yes"],
      ["Mistral-Small-3.2-24B-Instruct", "", "Apache-2.0", "yes"],
      ["Mistral-Medium-3.5-128B", "", "Mistral Medium Modified MIT", "yes"],
      ["phi-4-mini", "", "MIT", "yes"],
      ["DeepSeek-R1-Distill-Qwen-7B", "", "MIT", "yes"],
      ["gpt-oss-20b", "", "Apache-2.0", "yes"],
      ["GLM-4.5-Air", "", "MIT", "yes"],
      ["whisper-large-v3", "", "MIT", "yes"],
      ["OuteTTS-1.0-1B", "", "Apache-2.0", "yes"],
      ["Orpheus-3B", "", "Apache-2.0", "yes"],
      ["FLUX.1-dev", "", "FLUX.1 Non-Commercial Licence", "no"],
      ["FLUX.1-schnell", "", "Apache-2.0", "yes"],
      ["SD 3.5 Medium", "", "Stability AI Community Licence", "conditions"],
      ["SDXL", "", "CreativeML OpenRAIL++", "conditions"],
      ["SD-Turbo", "", "Stability AI Non-Commercial Research Licence", "no"],
      ["Laguna-X", "", "Check the model card", "unknown"],
    ];
    for (const [name, family, licence, commercial] of cases) {
      const found = licenceForModel(name, family);
      expect({ name, licence: found.name, commercial: found.commercial }).toEqual({ name, licence, commercial });
    }
  });

  it("uses a repository id or family when the display name is generic", () => {
    expect(licenceForModel("unsloth/gemma-3-12b-it-GGUF").name).toBe("Gemma Terms of Use");
    expect(licenceForModel("some-model", "Qwen3").name).toBe("Apache-2.0");
  });

  it("falls back to the Hub's licence tag", () => {
    expect(licenceForModel("some-model", null, "license:apache-2.0").name).toBe("Apache-2.0");
    expect(licenceForModel("some-model", null, "mit").name).toBe("MIT");
    expect(licenceForModel("some-model", null, "cc-by-nc-4.0").commercial).toBe("no");
    expect(licenceForModel("some-model", null, "other").name).toBe("Check the model card");
    expect(licenceForModel("some-model", null, null).name).toBe("Check the model card");
  });

  it("keeps a known family ahead of a generic licence tag", () => {
    expect(licenceForModel("Qwen3-8B", null, "other").name).toBe("Apache-2.0");
    expect(licenceForModel("Llama-3.1-8B", null, "apache-2.0").name).toBe("Llama Community Licence");
  });

  it("turns the commercial answer into a short badge", () => {
    expect(commercialLabel("yes")).toBe("Commercial use allowed");
    expect(commercialLabel("no")).toBe("No commercial use");
    expect(commercialLabel("conditions")).toBe("Commercial use with conditions");
    expect(commercialLabel("unknown")).toBe("Licence on the model page");
  });
});
