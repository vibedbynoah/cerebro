import { describe, expect, it } from "vitest";
import { abilitiesOf, familyOf, fitBadge, HUB_PICKS } from "../../ui/src/lib/hub-picks";

describe("hub picks", () => {
  it("reads families from repository names", () => {
    expect(familyOf("unsloth/Qwen3.5-4B-GGUF").id).toBe("qwen");
    expect(familyOf("bartowski/gemma-4-E4B-it-GGUF").id).toBe("gemma");
    expect(familyOf("someone/Mixtral-8x7B-GGUF").id).toBe("mistral");
    expect(familyOf("stabilityai/sd-turbo").id).toBe("stable-diffusion");
    const other = familyOf("acme/Zephyr-7B-GGUF");
    expect(other).toEqual({ id: "other", mark: "Ze", hue: null });
  });

  it("reads abilities from names and tags", () => {
    expect(abilitiesOf("x/Qwen3-Coder-30B-GGUF", [])).toEqual(["chat", "code"]);
    expect(abilitiesOf("x/Model-GGUF", ["image-text-to-text"])).toContain("vision");
    expect(abilitiesOf("x/Qwen3-ASR-0.6B-GGUF", [])).toEqual(["speech", "small"]);
    expect(abilitiesOf("x/Thing-Thinking-GGUF", [])).toContain("reasoning");
  });

  it("judges fit by graphics memory first, then with system memory", () => {
    expect(fitBadge(4.6, 8, 32).tone).toBe("good");
    expect(fitBadge(17.3, 8, 32).tone).toBe("ok");
    expect(fitBadge(17.3, 0, 32).tone).toBe("slow");
    expect(fitBadge(60, 8, 32).tone).toBe("no");
  });

  it("every pick is a search with a size and abilities", () => {
    for (const p of HUB_PICKS) {
      expect(p.query).toMatch(/GGUF$/);
      expect(p.sizeGiB).toBeGreaterThan(0);
      expect(p.abilities.length).toBeGreaterThan(0);
    }
  });
});
