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

  it("takes the largest dynamic quant that fits this computer", () => {
    const files = [file("UD-Q4_K_XL", 5), file("UD-Q8_0", 9), file("UD-Q2_K", 2)];
    const pick = recommendQuant(files, machine(8, 32));
    expect(pick?.file.quant).toBe("UD-Q8_0");
    expect(pick?.why).toBe("UD-Q8_0: the best quality that fits your 8 GB graphics card and 32 GB of memory.");
  });

  it("takes the largest file that fits when the repository has no dynamic quant", () => {
    const files = [file("Q4_K_M", 4.6), file("Q8_0", 12), file("IQ2_XXS", 2)];
    const pick = recommendQuant(files, machine(8, 32));
    expect(pick?.file.quant).toBe("Q8_0");
    expect(pick?.why).toMatch(/your 8 GB graphics card and 32 GB of memory/);
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
