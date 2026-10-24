import { describe, expect, it } from "vitest";
import { fitOf, isDynamicQuant, recommendQuant, type QuantFile } from "../../ui/src/lib/hub-quant";
import type { HardwareProfile } from "../../ui/src/service/types";

/** A machine: 8 GB of graphics memory and 32 GB of system memory unless a test says otherwise. */
const machine = (vramGiB = 8, ramGiB = 32): HardwareProfile => ({
  provenance: "measured",
  cpu: { name: "Test CPU", cores: 8, threads: 16 },
  gpu: vramGiB > 0 ? { name: "Test GPU", vramTotalGiB: vramGiB } : null,
  ramTotalGiB: ramGiB,
});

const file = (quant: string, sizeGiB: number, path = `model-${quant}.gguf`, parts = 1): QuantFile => ({ path, quant, sizeGiB, parts });

describe("the quant chooser", () => {
  it("prefers an Unsloth Dynamic quant when the repository has one", () => {
    const files = [file("Q4_K_M", 4.6), file("Q8_0", 8.7), file("UD-Q4_K_XL", 5)];
    const pick = recommendQuant(files, machine());
    expect(pick?.file.quant).toBe("UD-Q4_K_XL");
    expect(pick?.slow).toBe(false);
  });

  it("takes the largest dynamic quant that runs entirely on the graphics card", () => {
    const files = [file("UD-Q4_K_XL", 5), file("UD-Q8_0", 9), file("UD-Q2_K", 2)];
    const pick = recommendQuant(files, machine(8, 32));
    expect(pick?.file.quant).toBe("UD-Q4_K_XL");
    expect(pick?.why).toBe("UD-Q4_K_XL: the best quality that runs entirely on your 8 GB graphics card.");
  });

  it("never picks a full-precision file that only fits split with the processor", () => {
    const files = [file("f32", 29.9), file("f16", 15), file("Q8_0", 8), file("Q4_K_M", 4.6), file("Q6_K", 6.1)];
    expect(recommendQuant(files, machine(8, 32))?.file.quant).toBe("Q6_K");
  });

  it("takes a 4- or 5-bit file for a model too big for the card", () => {
    const files = [file("UD-Q8_K_XL", 29.3), file("UD-Q6_K_XL", 22), file("UD-Q4_K_XL", 16.5), file("UD-Q5_K_XL", 19), file("UD-IQ2_XXS", 8)];
    expect(recommendQuant(files, machine(8, 32))?.file.quant).toBe("UD-Q5_K_XL");
  });

  it("does not pick a 1- or 2-bit file just because it fits the card", () => {
    const files = [file("UD-IQ1_M", 6.3), file("UD-IQ2_XXS", 7.0), file("UD-Q4_K_XL", 16.5), file("UD-Q8_K_XL", 29)];
    expect(recommendQuant(files, machine(8, 32))?.file.quant).toBe("UD-Q4_K_XL");
  });

  it("allows 16-bit only when the repository has nothing smaller", () => {
    expect(recommendQuant([file("F32", 7.2), file("F16", 3.8)], machine(8, 32))?.file.quant).toBe("F16");
  });

  it("takes the largest file on the card when the repository has no dynamic quant", () => {
    const files = [file("Q4_K_M", 4.6), file("Q8_0", 12), file("IQ2_XXS", 2)];
    const pick = recommendQuant(files, machine(8, 32));
    expect(pick?.file.quant).toBe("Q4_K_M");
    expect(pick?.why).toMatch(/your 8 GB graphics card/);
  });

  it("recommends the smallest file when nothing fits, and says it will be slow", () => {
    const files = [file("Q8_0", 60), file("Q4_K_M", 45), file("IQ2_XXS", 40)];
    const pick = recommendQuant(files, machine(8, 32));
    expect(pick?.file.quant).toBe("IQ2_XXS");
    expect(pick?.slow).toBe(true);
    expect(pick?.why).toBe("IQ2_XXS: the smallest file here, so it will run slowly from the drive.");
  });

  it("uses a file that fits when the dynamic quants are all too big", () => {
    const files = [file("UD-Q8_0", 60), file("Q4_K_M", 6)];
    const pick = recommendQuant(files, machine(8, 32));
    expect(pick?.file.quant).toBe("Q4_K_M");
    expect(pick?.slow).toBe(false);
  });

  it("treats a split model as one file", () => {
    const files = [file("UD-Q4_K_XL", 18.2, "model-UD-Q4_K_XL-00001-of-00002.gguf", 2), file("Q4_K_M", 17.3)];
    const pick = recommendQuant(files, machine(8, 32));
    expect(pick?.file.quant).toBe("UD-Q4_K_XL");
    expect(pick?.file.parts).toBe(2);
  });

  it("says nothing when the machine or the repository is empty", () => {
    expect(recommendQuant([file("Q4_K_M", 4.6)], undefined)).toBeNull();
    expect(recommendQuant([], machine())).toBeNull();
  });

  it("leaves out files without a size", () => {
    expect(recommendQuant([{ path: "x.gguf", quant: "Q4_K_M", sizeGiB: 0, parts: 1 }], machine())).toBeNull();
  });

  it("knows a dynamic quant by its label or its file name", () => {
    expect(isDynamicQuant(file("UD-Q4_K_XL", 5))).toBe(true);
    expect(isDynamicQuant({ quant: "Q4_K_M", path: "model-UD-Q4_K_XL.gguf" })).toBe(true);
    expect(isDynamicQuant(file("Q4_K_M", 5))).toBe(false);
  });

  it("judges the fit of a file the same way the Hub's table does", () => {
    expect(fitOf(4, 8, 32).tone).toBe("good");
    expect(fitOf(20, 8, 32).tone).toBe("ok");
    expect(fitOf(60, 8, 32).tone).toBe("no");
    expect(fitOf(4, 0, 32).tone).toBe("slow");
  });
});
