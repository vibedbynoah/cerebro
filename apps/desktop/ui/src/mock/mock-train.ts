import type { TrainerStatus, TrainEvent, TrainInstalled, TrainJob } from "../service/train-types";

/**
 * Train mode in web mode and tests: an illustrative run with the real event sequence (setup lines,
 * stages, one log per step with a falling loss, one export per format, done). `stepMs` 0 runs it
 * without timers.
 */
export function createMockTrainer(opts: { stepMs?: number } = {}) {
  const stepMs = opts.stepMs ?? 120;
  let installed: TrainInstalled | null = null;
  let settingUp = false;
  let running = false;
  let stop = false;
  const wait = (ms: number) => (ms > 0 ? new Promise<void>((r) => setTimeout(r, ms)) : Promise.resolve());

  const status = (): TrainerStatus => ({ installed: installed ? { ...installed } : null, settingUp, running, root: "C:\\Users\\me\\AppData\\Roaming\\Cerebro\\trainer", stderrTail: [] });

  async function setup(onEvent: (e: TrainEvent) => void): Promise<TrainInstalled> {
    if (settingUp) throw new Error("Setup is already running.");
    settingUp = true;
    try {
      for (const text of ["Downloading uv 0.12.22", "Using CPython 3.12.10", "Installing PyTorch with CUDA (about 3 GB)", "Installing Unsloth", "Installed 142 packages"]) {
        onEvent({ type: "setup", text });
        await wait(stepMs);
      }
      installed = { uv: "0.12.22", python: "3.12.10", torch: "2.11.0+cu128", unsloth: "2026.10.6", cuda: true };
      return { ...installed };
    } finally {
      settingUp = false;
    }
  }

  async function start(job: TrainJob, onEvent: (e: TrainEvent) => void): Promise<void> {
    if (!installed) throw new Error("Set up training first.");
    if (running) throw new Error("A training job is already running.");
    running = true;
    stop = false;
    try {
      const t0 = Date.now();
      onEvent({ type: "status", stage: "loading_model", detail: job.model });
      await wait(stepMs);
      onEvent({ type: "status", stage: "preparing_data", detail: "64 examples (0 skipped)" });
      await wait(stepMs);
      onEvent({ type: "status", stage: "training", detail: "64 examples" });
      const max = job.maxSteps > 0 ? job.maxSteps : 24;
      let step = 0;
      for (step = 1; step <= max && !stop; step++) {
        const loss = 0.35 + 2.4 * Math.exp(-step / (max / 4)) + 0.05 * Math.sin(step * 1.7);
        onEvent({ type: "log", step, maxSteps: max, loss, evalLoss: null, lr: job.lr * Math.min(1, step / 2) * (1 - step / (max + 1)), gradNorm: 1.2 / Math.sqrt(step), epoch: step / max });
        await wait(stepMs);
      }
      onEvent({ type: "status", stage: "saving", detail: "LoRA adapter" });
      const models: string[] = [];
      if (!stop) {
        onEvent({ type: "status", stage: "exporting", detail: job.export.join(", ") });
        for (const q of job.export) {
          const path = `C:\\Users\\me\\AppData\\Local\\Cerebro\\models\\trained\\${job.name.trim().replace(/ /g, "-")}\\unsloth.${q.toUpperCase()}.gguf`;
          onEvent({ type: "export", quant: q, path });
          models.push(path);
          await wait(stepMs);
        }
      }
      onEvent({ type: "done", models, adapter: "C:\\Users\\me\\AppData\\Roaming\\Cerebro\\trainer\\jobs\\x\\out\\adapter", steps: step - 1, seconds: (Date.now() - t0) / 1000, stopped: stop });
    } finally {
      running = false;
    }
  }

  return {
    trainerStatus: () => Promise.resolve(status()),
    trainerSetup: setup,
    trainerStart: start,
    trainerCancel: () => {
      const was = running;
      stop = running;
      return Promise.resolve(was);
    },
  };
}
