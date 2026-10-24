import { describe, expect, it } from "vitest";
import { estimateSpeed, parseRepo, pickLatest } from "../../ui/src/lib/hub-latest";
import type { QuantFile } from "../../ui/src/lib/hub-quant";
import type { HardwareProfile, HubRepo } from "../../ui/src/service/types";

const laptop: HardwareProfile = { provenance: "measured", cpu: { name: "CPU", cores: 14, threads: 20 }, gpu: { name: "RTX 3070 Ti Laptop", vramTotalGiB: 8 }, ramTotalGiB: 32 };
const big: HardwareProfile = { provenance: "measured", cpu: { name: "CPU", cores: 32, threads: 64 }, gpu: { name: "RTX 6000", vramTotalGiB: 48 }, ramTotalGiB: 256 };
const repo = (id: string): HubRepo => ({ id, author: id.split("/")[0]!, downloads: 0, likes: 0, updated: "", license: null, tags: [] });
const f = (quant: string, sizeGiB: number): QuantFile => ({ path: `${quant}.gguf`, quant, sizeGiB, parts: 1 });

describe("the newest Qwen of each kind", () => {
  it("reads size and kind from the repository name", () => {
    expect(parseRepo("unsloth/Qwen3.6-35B-A3B-GGUF")).toEqual({ totalB: 35, activeB: 3, kind: "moe" });
    expect(parseRepo("Qwen/Qwen3-8B-GGUF")).toEqual({ totalB: 8, activeB: 8, kind: "dense" });
    expect(parseRepo("unsloth/Qwen3.8-Flash-Next-GGUF")?.kind).toBe("moe");
    expect(parseRepo("Qwen/Qwen3-Embedding-8B-GGUF")).toBeNull();
    expect(parseRepo("Qwen/Qwen3-ASR-0.6B-GGUF")).toBeNull();
  });

  it("estimates speed from where the weights are read, and flags swapping", () => {
    const onCard = estimateSpeed(4.7, 1, laptop);
    expect(onCard.swapping).toBe(false);
    expect(onCard.tps).toBeGreaterThan(40);
    const flash = estimateSpeed(76, 3 / 177, laptop);
    expect(flash.swapping).toBe(true);
    expect(flash.note).toMatch(/Heavy swapping/);
    expect(estimateSpeed(76, 3 / 177, big).swapping).toBe(false);
  });

  it("takes the newest model that runs here, skipping ones too big", async () => {
    const repos = [repo("unsloth/Qwen3.8-Flash-Next-GGUF"), repo("unsloth/Qwen3.6-35B-A3B-GGUF"), repo("unsloth/Qwen3.8-27B-GGUF"), repo("unsloth/Qwen3.5-4B-GGUF")];
    const files = async (id: string): Promise<QuantFile[]> =>
      id.includes("Flash") ? [f("UD-IQ3_XXS", 76), f("UD-Q4_K_XL", 100)] : id.includes("35B") ? [f("UD-Q4_K_XL", 20), f("Q8_0", 36)] : id.includes("27B") ? [f("UD-Q4_K_XL", 16.5), f("UD-Q8_K_XL", 29)] : [f("UD-Q4_K_XL", 2.6), f("UD-Q8_K_XL", 5.5)];
    const moe = await pickLatest("moe", repos, laptop, files);
    expect(moe?.repo.id).toBe("unsloth/Qwen3.6-35B-A3B-GGUF");
    const dense = await pickLatest("dense", repos, laptop, files);
    expect(dense?.repo.id).toBe("unsloth/Qwen3.8-27B-GGUF");
    expect((await pickLatest("moe", repos, big, files))?.repo.id).toBe("unsloth/Qwen3.8-Flash-Next-GGUF");
  });
});
