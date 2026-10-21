import { afterEach, describe, expect, it } from "vitest";
import {
  CHANNEL_KEY,
  channelIncludes,
  DEFAULT_LEVEL,
  EXPERIENCE_KEY,
  OPTIMISATIONS,
  otherLevel,
  parseChannel,
  parseLevel,
  readChannel,
  readLevel,
  writeChannel,
  writeLevel,
} from "../../ui/src/lib/experience";
import { availability, bestAtSize, fmtKl, qualityKept, qualityPerGiB } from "../../ui/src/lib/quants";
import { MOCK_CATALOG, MOCK_HARDWARE } from "../../ui/src/mock/fixtures";
import {
  autoConfig,
  changedFrom,
  configFlags,
  CONTEXT_STEPS,
  parseConfig,
  pickQuant,
  readConfig,
  sanitize,
  SETUP_CONFIG_KEY,
  writeConfig,
  type SetupConfig,
} from "../../ui/src/onboarding/autoconfig";
import type { CatalogModel, CatalogQuant, HardwareProfile } from "../../ui/src/service/types";

const model = (id: string): CatalogModel => {
  const m = MOCK_CATALOG.find((x) => x.id === id);
  if (!m) throw new Error(`no ${id}`);
  return m;
};

afterEach(() => window.localStorage.clear());

describe("experience level and channel", () => {
  it("defaults to Advanced and Stable, and reads back only known values", () => {
    expect(DEFAULT_LEVEL).toBe("advanced");
    expect(readLevel()).toBe("advanced");
    expect(readChannel()).toBe("stable");
    for (const bad of ["", "Beginner", "expert", "beginner ", "<script>", "__proto__"]) {
      expect(parseLevel(bad), bad).toBeNull();
      expect(parseChannel(bad), bad).toBeNull();
    }
    window.localStorage.setItem(EXPERIENCE_KEY, "root");
    window.localStorage.setItem(CHANNEL_KEY, "nightly");
    expect(readLevel()).toBe("advanced");
    expect(readChannel()).toBe("stable");
  });

  it("persists both choices", () => {
    writeLevel("beginner");
    writeChannel("experimental");
    expect(window.localStorage.getItem(EXPERIENCE_KEY)).toBe("beginner");
    expect(readLevel()).toBe("beginner");
    expect(readChannel()).toBe("experimental");
    expect(otherLevel("beginner")).toBe("advanced");
    expect(otherLevel("advanced")).toBe("beginner");
  });

  it("channels nest: Experimental includes Beta includes Stable", () => {
    expect(channelIncludes("stable", "stable")).toBe(true);
    expect(channelIncludes("stable", "beta")).toBe(false);
    expect(channelIncludes("beta", "stable")).toBe(true);
    expect(channelIncludes("experimental", "beta")).toBe(true);
    expect(channelIncludes("beta", "experimental")).toBe(false);
    // Each channel adds at least one optimisation.
    for (const c of ["stable", "beta", "experimental"] as const) expect(OPTIMISATIONS.some((o) => o.channel === c)).toBe(true);
  });
});

describe("catalogue for pros", () => {
  it("lists current models only, with every figure marked illustrative", () => {
    const names = MOCK_CATALOG.map((m) => m.name).join(" ");
    expect(names).not.toMatch(/Llama-3|Qwen2|Qwen3-8B|Mistral-Small|Phi-|DeepSeek-R1|Gemma-3|Mixtral/);
    for (const m of MOCK_CATALOG) {
      expect(m.provenance).toBe("illustrative");
      expect(m.quants.length).toBeGreaterThan(2);
      for (const q of m.quants) {
        expect(q.provenance, q.id).toBe("illustrative");
        for (const n of [q.bitsPerWeight, q.sizeGiB, q.klDiv, q.pplDeltaPct]) expect(Number.isFinite(n) && n >= 0, q.id).toBe(true);
        expect(q.decodeTps === null).toBe(q.unavailable !== null);
      }
      expect(new Set(m.quants.map((q) => q.id)).size).toBe(m.quants.length);
    }
  });

  it("compares today's quantisation families side by side", () => {
    const formats = new Set(MOCK_CATALOG.flatMap((m) => m.quants.map((q) => q.format)));
    for (const f of ["gsq-rco", "unsloth-ud", "imatrix-iq", "exl3", "awq", "mlx"] as const) expect(formats.has(f), f).toBe(true);
  });
});

describe("quality per GiB", () => {
  const gemma = model("gemma-4-12b-it");
  const by = (label: string): CatalogQuant => {
    const q = gemma.quants.find((x) => x.label === label);
    if (!q) throw new Error(label);
    return q;
  };

  it("says which files run here: GSQ-RCO needs Experimental, other engines never", () => {
    expect(availability(by("UD-Q4_K_XL"), "stable")).toBe("runs");
    expect(availability(by("GSQ-RCO 3.5"), "stable")).toBe("channel");
    expect(availability(by("GSQ-RCO 3.5"), "beta")).toBe("channel");
    expect(availability(by("GSQ-RCO 3.5"), "experimental")).toBe("runs");
    expect(availability(by("EXL3 4.0bpw"), "experimental")).toBe("elsewhere");
    expect(availability(by("MLX 4-bit"), "experimental")).toBe("elsewhere");
  });

  it("reads KL as quality kept and divides by size, never producing NaN", () => {
    expect(qualityKept({ ...by("Q6_K"), klDiv: 0 })).toBe(100);
    expect(qualityKept(by("Q6_K"))).toBeCloseTo(Math.exp(-0.006) * 100, 6);
    expect(qualityPerGiB({ ...by("Q6_K"), sizeGiB: 0 })).toBe(0);
    expect(qualityPerGiB({ ...by("Q6_K"), klDiv: Number.NaN })).toBe(0);
    expect(fmtKl(Number.NaN)).toBe("–");
    expect(fmtKl(0.0184)).toBe("0.018");
  });

  it("marks only files nothing else beats on both size and KL, among those that run here", () => {
    const stable = bestAtSize(gemma.quants, "stable");
    expect([...stable].sort()).toEqual([by("IQ4_XS").id, by("Q6_K").id, by("UD-Q4_K_XL").id].sort());
    const exp = bestAtSize(gemma.quants, "experimental");
    expect(exp.has(by("GSQ-RCO 3.5").id)).toBe(true);
    // IQ4_XS is larger and further from the reference than GSQ-RCO 3.5.
    expect(exp.has(by("IQ4_XS").id)).toBe(false);
  });
});

describe("Beginner's automatic plan", () => {
  it("fits a dense model whole on the card when a close file fits", () => {
    const cfg = autoConfig(model("gemma-4-12b-it"), MOCK_HARDWARE, "stable");
    expect(cfg).toMatchObject({ quantId: "gemma-4-12b-it--ud-q4-k-xl", contextLength: 32768, gpuLayers: 48, moeCpuLayers: 0, kvCache: "f16", flashAttention: "on" });
    expect(cfg).toMatchObject({ temperature: 1, topP: 0.95, topK: 64, minP: 0 });
  });

  it("keeps a MoE model's attention on the card and moves experts of some layers to system memory", () => {
    const cfg = autoConfig(model("gpt-oss-20b"), MOCK_HARDWARE, "stable");
    expect(cfg?.quantId).toBe("gpt-oss-20b--ud-q4-k-xl");
    expect(cfg?.gpuLayers).toBe(24);
    expect(cfg?.moeCpuLayers).toBe(12);
    expect(cfg?.contextLength).toBe(16384);
    expect(cfg?.kvCache).toBe("q8_0");
  });

  it("never picks a file this machine cannot run under the channel", () => {
    for (const m of MOCK_CATALOG) {
      for (const ch of ["stable", "experimental"] as const) {
        const q = pickQuant(m, MOCK_HARDWARE, ch);
        if (q) expect(availability(q, ch), `${m.id} ${ch}`).toBe("runs");
      }
    }
  });

  it("is deterministic and runs without a hardware reading or a graphics card", () => {
    const m = model("qwen3.8-27b");
    expect(autoConfig(m, MOCK_HARDWARE, "stable")).toEqual(autoConfig(m, MOCK_HARDWARE, "stable"));
    const noHw = autoConfig(m, undefined, "stable");
    expect(noHw?.gpuLayers).toBe(0);
    const cpuOnly: HardwareProfile = { ...MOCK_HARDWARE, gpu: null, ramTotalGiB: 64 };
    const cpu = autoConfig(m, cpuOnly, "stable");
    expect(cpu).toMatchObject({ gpuLayers: 0, contextLength: 4096, flashAttention: "auto" });
  });

  it("splits a dense model larger than the card by layers", () => {
    const cfg = autoConfig(model("qwen3.8-27b"), MOCK_HARDWARE, "stable");
    expect(cfg?.gpuLayers).toBeGreaterThan(0);
    expect(cfg?.gpuLayers).toBeLessThan(64);
    expect(cfg?.contextLength).toBe(8192);
  });
});

describe("one config, two views", () => {
  const m = model("gpt-oss-20b");
  const auto = autoConfig(m, MOCK_HARDWARE, "stable") as SetupConfig;

  it("clamps every edited value to what the model allows", () => {
    const wild = sanitize(m, { ...auto, quantId: "nope", contextLength: 12345, gpuLayers: 999, moeCpuLayers: -3, kvCache: "f32" as never, flashAttention: "maybe" as never, temperature: 9, topP: Number.NaN, topK: 1e9, minP: -1 });
    expect(wild.quantId).toBe(m.quants[0]?.id);
    expect(wild.contextLength).toBe(4096);
    expect(wild.gpuLayers).toBe(24);
    expect(wild.moeCpuLayers).toBe(0);
    expect(wild.kvCache).toBe("f16");
    expect(wild.flashAttention).toBe("auto");
    expect(wild.temperature).toBe(2);
    expect(wild.topP).toBe(0);
    expect(wild.topK).toBe(200);
    expect(wild.minP).toBe(0);
    expect(CONTEXT_STEPS.includes(sanitize(m, { ...auto, contextLength: 65536 }).contextLength as never)).toBe(true);
    // Dense models never carry a MoE offload value.
    const dense = model("gemma-4-12b-it");
    expect(sanitize(dense, { ...auto, modelId: dense.id, moeCpuLayers: 10 }).moeCpuLayers).toBe(0);
  });

  it("lists exactly the fields that differ from the automatic plan", () => {
    expect(changedFrom(auto, auto)).toEqual([]);
    expect(changedFrom(auto, { ...auto, contextLength: 8192, topK: 10 })).toEqual(["contextLength", "topK"]);
    expect(changedFrom(auto, { ...auto, modelId: "other" })).toEqual([]);
    expect(changedFrom(null, auto)).toEqual([]);
  });

  it("produces loopback-only backend flags that follow every value", () => {
    const flags = configFlags(m, auto).join("\n");
    expect(flags).toMatch(/^cerebro serve -m gpt-oss-20b-UD-Q4_K_XL\.gguf/);
    expect(flags).toContain("--n-cpu-moe 12");
    expect(flags).toContain("-ctk q8_0 -ctv q8_0");
    expect(flags).toContain("--host 127.0.0.1");
    expect(configFlags(m, { ...auto, moeCpuLayers: 0 }).join("\n")).not.toContain("--n-cpu-moe");
    const gsq = m.quants.find((q) => q.format === "gsq-rco");
    expect(configFlags(m, { ...auto, quantId: gsq?.id ?? "" })[0]).toMatch(/^strata-server /);
  });
});

describe("saved plan", () => {
  const auto = autoConfig(model("gemma-4-12b-it"), MOCK_HARDWARE, "stable") as SetupConfig;

  it("round-trips through storage", () => {
    writeConfig(auto);
    expect(readConfig()).toEqual(auto);
    writeConfig(null);
    expect(window.localStorage.getItem(SETUP_CONFIG_KEY)).toBeNull();
  });

  it("drops anything malformed, oversized or tampered with, and copies only known fields", () => {
    const bad = [
      null,
      "",
      "{",
      "[]",
      "null",
      JSON.stringify({ ...auto, modelId: "../../etc/passwd" }),
      JSON.stringify({ ...auto, quantId: "<img src=x onerror=alert(1)>" }),
      JSON.stringify({ ...auto, kvCache: "f32" }),
      JSON.stringify({ ...auto, flashAttention: true }),
      JSON.stringify({ ...auto, gpuLayers: "48" }),
      JSON.stringify({ ...auto, temperature: null }),
      JSON.stringify({ ...auto, pad: "x".repeat(2000) }),
    ];
    for (const raw of bad) expect(parseConfig(raw), String(raw).slice(0, 60)).toBeNull();
    const parsed = parseConfig(JSON.stringify({ ...auto, admin: true }).replace("{", '{"__proto__":{"polluted":true},'));
    expect(parsed).toEqual(auto);
    expect(Object.keys(parsed ?? {})).not.toContain("admin");
    expect(({} as Record<string, unknown>)["polluted"]).toBeUndefined();
  });
});
