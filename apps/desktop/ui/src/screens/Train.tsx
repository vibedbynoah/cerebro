import "../styles/train.css";
import { createMemo, createResource, createSignal, For, onMount, Show, type JSX } from "solid-js";
import { Icon } from "../components/icons";
import { IllustrativeFlag, ScreenHeader, Section } from "../components/Screen";
import { useService } from "../service/context";
import { TRAIN_QUANTS, type TrainerStatus, type TrainFormat, type TrainJob, type TrainMethod, type TrainQuant } from "../service/train-types";
import { applyTrainEvent, BASE_MODELS, DEFAULT_JOB, fileName, IDLE_RUN, jobErrors, lastLoss, normalizeJob, STAGE_LABELS, type JobField, type TrainRun } from "../lib/train";
import { notify } from "../lib/toast";
import { Sparkline } from "./monitor/Sparkline";

/**
 * Train: fine-tune a model on this computer with a LoRA adapter and add it to the model library as
 * GGUF. The first visit sets up the training environment (trainer.rs: pinned, checksummed tools in
 * Cerebro's own folder). A run keeps going when the screen is closed; its state lives here, at
 * module level, so coming back shows it.
 */

const [run, setRun] = createSignal<TrainRun>(IDLE_RUN);
const [setupLines, setSetupLines] = createSignal<string[]>([]);
const [job, setJob] = createSignal<TrainJob>({ ...DEFAULT_JOB });

const QUANT_LABELS: Readonly<Record<TrainQuant, string>> = { q4_k_m: "Q4_K_M (smallest, recommended)", q5_k_m: "Q5_K_M", q8_0: "Q8_0", f16: "F16 (full size)" };
const FORMATS: ReadonlyArray<{ id: TrainFormat; label: string }> = [
  { id: "auto", label: "Detect from the rows" },
  { id: "alpaca", label: "instruction / input / output" },
  { id: "chatml", label: "messages: role / content" },
  { id: "sharegpt", label: "conversations: from / value" },
];

export function Train(): JSX.Element {
  const service = useService();
  // Recommendations from this machine's graphics memory (the trainer runs on the card).
  const [trainHw] = createResource(() => service.detectHardware().catch(() => undefined));
  const vram = () => trainHw()?.gpu?.vramTotalGiB ?? 8;
  const recContext = () => (vram() >= 16 ? 4096 : 2048);
  const recBatch = () => (vram() >= 16 ? 4 : 2);
  const [status, setStatus] = createSignal<TrainerStatus | null>(null);
  const [setupError, setSetupError] = createSignal<string | null>(null);
  const [datasetKind, setDatasetKind] = createSignal<"file" | "hf">(job().datasetHf ? "hf" : "file");
  const [tried, setTried] = createSignal(false);

  const refresh = async () => {
    try {
      setStatus(await service.trainerStatus());
    } catch (e) {
      setSetupError(String(e));
    }
  };
  onMount(() => void refresh());

  const installed = () => status()?.installed ?? null;
  const settingUp = () => status()?.settingUp ?? false;
  const running = () => run().phase === "running" || (status()?.running ?? false);

  const setup = async () => {
    setSetupError(null);
    setSetupLines([]);
    setStatus((s) => (s ? { ...s, settingUp: true } : s));
    try {
      await service.trainerSetup((e) => {
        if (e.type === "setup") setSetupLines((l) => [...l.slice(-199), e.text]);
      });
      notify("Training is ready", "The training environment is set up on this computer.", "success");
    } catch (e) {
      setSetupError(String(e));
    }
    await refresh();
  };

  const patch = (p: Partial<TrainJob>) => setJob((j) => ({ ...j, ...p }));
  const sent = () => normalizeJob({ ...job(), datasetPath: datasetKind() === "file" ? job().datasetPath : null, datasetHf: datasetKind() === "hf" ? job().datasetHf : null });
  const errors = createMemo(() => jobErrors(sent()));
  const err = (f: JobField) => (tried() ? errors()[f] : undefined);
  const canStart = () => Object.keys(errors()).length === 0;

  const start = async (e: Event) => {
    e.preventDefault();
    setTried(true);
    if (!canStart() || running()) return;
    const j = sent();
    setRun({ ...IDLE_RUN, phase: "running", stage: "loading_model", detail: j.model });
    try {
      await service.trainerStart(j, (ev) => setRun((r) => applyTrainEvent(r, ev)));
      const r = run();
      if (r.phase !== "done") setRun({ ...r, phase: "done" });
      notify(r.stopped ? "Training stopped" : "Training finished", r.models.length ? `${j.name} is in the model library.` : "The adapter was saved.", r.stopped ? "warning" : "success");
    } catch (ex) {
      setRun((r) => ({ ...r, phase: "failed", error: String(ex).replace(/^Error: /, "") }));
      notify("Training failed", String(ex).replace(/^Error: /, ""), "error");
    }
    await refresh();
  };

  const stop = async () => {
    if (await service.trainerCancel()) notify("Stopping", "Training stops after the current step and keeps what it learned.", "info");
  };

  const num = (v: string) => (v.trim() === "" ? Number.NaN : Number(v));
  const progress = () => (run().maxSteps > 0 ? Math.min(1, run().step / run().maxSteps) : 0);

  return (
    <div class="screen train">
      <ScreenHeader route="train" actions={__DESKTOP__ ? undefined : <IllustrativeFlag />} />

      <Section
        id="train-setup"
        title="Training environment"
        description="Training runs in its own Python environment in Cerebro's folder: PyTorch with CUDA and a LoRA trainer. Every download is pinned and checked. It is about 6 GB and set up once; nothing is installed system-wide."
      >
        <Show
          when={installed()}
          fallback={
            <div class="train-setup card card--pad">
              <button class="btn btn--primary" type="button" disabled={settingUp()} onClick={() => void setup()}>
                <Icon name="download" size={16} />
                {settingUp() ? "Setting up…" : "Set up training"}
              </button>
              <Show when={setupLines().length}>
                <ol class="train-log mono" aria-live="polite" aria-label="Setup progress">
                  <For each={setupLines().slice(-8)}>{(l) => <li>{l}</li>}</For>
                </ol>
              </Show>
              <Show when={setupError()}>{(m) => <p class="train-error" role="alert">{m()}</p>}</Show>
            </div>
          }
        >
          {(i) => (
            <div class="train-ready">
              <span class="chip" data-tone="on">
                Ready
              </span>
              <span class="hint mono">
                Python {i().python} · PyTorch {i().torch} · trainer {i().unsloth}
              </span>
              <Show when={!i().cuda}>
                <span class="chip" data-tone="off">
                  No compatible graphics card: training will not start
                </span>
              </Show>
            </div>
          )}
        </Show>
      </Section>

      <Section id="train-job" title="New fine-tune" description="Teaches a model custom examples. The result is a new model in the library that runs like any other.">
        <form class="train-form" onSubmit={(e) => void start(e)} novalidate>
          <label class="train-field">
            <span>Name</span>
            <input class="input" value={job().name} placeholder="support-bot" maxLength={64} aria-invalid={err("name") ? "true" : undefined} onInput={(e) => patch({ name: e.currentTarget.value })} />
            <Show when={err("name")}>{(m) => <span class="train-field__err">{m()}</span>}</Show>
          </label>

          <label class="train-field">
            <span>Base model</span>
            <input class="input mono" list="train-models" value={job().model} aria-invalid={err("model") ? "true" : undefined} onInput={(e) => patch({ model: e.currentTarget.value })} />
            <datalist id="train-models">
              <For each={BASE_MODELS}>{(m) => <option value={m.id}>{m.note}</option>}</For>
            </datalist>
            <span class="hint">{BASE_MODELS.find((m) => m.id === job().model)?.note ?? "Any Hugging Face model id; it is downloaded on first use."}</span>
            <Show when={err("model")}>{(m) => <span class="train-field__err">{m()}</span>}</Show>
          </label>

          <fieldset class="train-field train-dataset">
            <legend>Dataset</legend>
            <div class="train-seg" role="radiogroup" aria-label="Dataset source">
              <label>
                <input type="radio" name="ds-kind" checked={datasetKind() === "file"} onChange={() => setDatasetKind("file")} /> A file on this computer
              </label>
              <label>
                <input type="radio" name="ds-kind" checked={datasetKind() === "hf"} onChange={() => setDatasetKind("hf")} /> A Hugging Face dataset
              </label>
            </div>
            <Show
              when={datasetKind() === "file"}
              fallback={<input class="input mono" aria-label="Hugging Face dataset id" placeholder="tatsu-lab/alpaca" value={job().datasetHf ?? ""} onInput={(e) => patch({ datasetHf: e.currentTarget.value })} />}
            >
              <div class="train-dataset__file">
                <input class="input mono" aria-label="Dataset file path" placeholder="C:\data\examples.jsonl" value={job().datasetPath ?? ""} onInput={(e) => patch({ datasetPath: e.currentTarget.value })} />
                <Show when={service.saveDataset}>
                  <label class="btn btn--secondary">
                    Choose a file…
                    <input
                      type="file"
                      accept=".jsonl,.json,.csv,.txt"
                      class="sr-only"
                      onChange={(e) => {
                        const f = e.currentTarget.files?.[0];
                        if (!f) return;
                        void f
                          .text()
                          .then((t) => service.saveDataset!(f.name, t))
                          .then((path) => patch({ datasetPath: path }))
                          .catch((err: unknown) => notify("The dataset could not be added", err instanceof Error ? err.message : String(err), "error"));
                      }}
                    />
                  </label>
                </Show>
              </div>
            </Show>
            <span class="select">
              <select class="input" aria-label="Dataset format" value={job().format} onChange={(e) => patch({ format: e.currentTarget.value as TrainFormat })}>
                <For each={FORMATS}>{(f) => <option value={f.id}>{f.label}</option>}</For>
              </select>
            </span>
            <span class="hint">One example per row: .jsonl, .json, .csv or .parquet.</span>
            <Show when={err("dataset")}>{(m) => <span class="train-field__err">{m()}</span>}</Show>
          </fieldset>

          <div class="train-row">
            <label class="train-field">
              <span>Method</span>
              <span class="select">
                <select class="input" value={job().method} onChange={(e) => patch({ method: e.currentTarget.value as TrainMethod })}>
                  <option value="qlora">QLoRA: 4-bit base, least memory</option>
                  <option value="lora">LoRA: 16-bit base, more memory</option>
                </select>
              </span>
            </label>
            <label class="train-field">
              <span>Steps</span>
              <input class="input mono" type="number" min="0" max="100000" value={job().maxSteps} aria-invalid={err("maxSteps") ? "true" : undefined} onInput={(e) => patch({ maxSteps: num(e.currentTarget.value) })} />
              <span class="hint">0 runs whole epochs instead.</span>
              <Show when={err("maxSteps")}>{(m) => <span class="train-field__err">{m()}</span>}</Show>
            </label>
          </div>

          <details class="train-advanced" open>
            <summary>More settings</summary>
            <div class="train-grid">
              <NumField label="Epochs" value={job().epochs} step="0.1" err={err("epochs")} onInput={(v) => patch({ epochs: v })} rec={1} why="Passes over the examples. 1 to 3: more follows them more closely but can start repeating them word for word." />
              <NumField label="Learning rate" value={job().lr} step="0.00001" err={err("lr")} onInput={(v) => patch({ lr: v })} rec={2e-4} why="How big each correction is. 0.0002 suits LoRA and QLoRA; try 0.00005 if the loss jumps around." />
              <NumField label="LoRA rank" value={job().rank} err={err("rank")} onInput={(v) => patch({ rank: v })} rec={16} why="How much the adapter can learn. 16 for most datasets; 32 to 64 for large ones, at more memory." />
              <NumField label="LoRA alpha" value={job().alpha} err={err("alpha")} onInput={(v) => patch({ alpha: v })} rec={job().rank} why="How strongly the adapter changes the model. Usually equal to the rank." />
              <NumField label="Context (sequence length)" value={job().maxSeqLen} err={err("maxSeqLen")} onInput={(v) => patch({ maxSeqLen: v })} rec={recContext()} why="The longest example, in tokens. Longer examples are cut; a longer context uses more graphics memory." />
              <NumField label="Batch size" value={job().batch} err={err("batch")} onInput={(v) => patch({ batch: v })} rec={recBatch()} why="Examples per step. Raise it when graphics memory allows; lower it if training runs out of memory." />
              <NumField label="Gradient accumulation" value={job().gradAccum} err={err("gradAccum")} onInput={(v) => patch({ gradAccum: v })} rec={4} why="Steps added together before each update: batch size times this is the effective batch (8 by default)." />
            </div>
          </details>

          <fieldset class="train-field">
            <legend>Save as</legend>
            <div class="train-quants">
              <For each={TRAIN_QUANTS}>
                {(q) => (
                  <label>
                    <input
                      type="checkbox"
                      checked={job().export.includes(q)}
                      onChange={(e) => patch({ export: e.currentTarget.checked ? [...job().export, q] : job().export.filter((x) => x !== q) })}
                    />{" "}
                    {QUANT_LABELS[q]}
                  </label>
                )}
              </For>
            </div>
            <span class="hint">None checked keeps only the adapter.</span>
          </fieldset>

          <div class="train-actions">
            <button class="btn btn--primary" type="submit" disabled={!installed() || running()} title={installed() ? undefined : "Set up training first"}>
              <Icon name="play" size={16} />
              Start training
            </button>
            <Show when={tried() && !canStart()}>
              <span class="train-field__err" role="alert">
                Fix the highlighted settings first.
              </span>
            </Show>
          </div>
        </form>
      </Section>

      <Show when={run().phase !== "idle"}>
        <Section id="train-run" title="Run">
          <div class="train-run card card--pad" aria-live="polite">
            <div class="train-run__head">
              <span class="chip" data-tone={run().phase === "failed" ? "off" : run().phase === "done" ? "on" : "active"}>
                {run().phase === "failed" ? "Failed" : run().phase === "done" ? (run().stopped ? "Stopped" : "Finished") : (STAGE_LABELS[run().stage ?? ""] ?? "Working")}
              </span>
              <Show when={run().phase === "running" && run().detail}>
                <span class="hint">{run().detail}</span>
              </Show>
              <Show when={run().phase === "running"}>
                <button class="btn btn--tertiary btn--sm train-run__stop" type="button" onClick={() => void stop()}>
                  <Icon name="stop" size={14} />
                  Stop
                </button>
              </Show>
            </div>

            <Show when={run().maxSteps > 0}>
              <div class="train-progress" role="progressbar" aria-label="Training steps" aria-valuemin="0" aria-valuemax={run().maxSteps} aria-valuenow={run().step}>
                <span style={{ transform: `scaleX(${progress()})` }} />
              </div>
              <dl class="train-figures">
                <div>
                  <dt>Step</dt>
                  <dd class="mono">
                    {run().step} / {run().maxSteps}
                  </dd>
                </div>
                <div>
                  <dt>Loss</dt>
                  <dd class="mono">{lastLoss(run())?.toFixed(3) ?? "–"}</dd>
                </div>
                <div>
                  <dt>Learning rate</dt>
                  <dd class="mono">{run().lr !== null ? run().lr!.toExponential(1) : "–"}</dd>
                </div>
              </dl>
              <Sparkline
                label="Training loss"
                lines={[{ key: "loss", label: "Loss", values: run().losses, tone: "accent", area: true, format: (v) => v.toFixed(3) }]}
                stepS={1}
                end={run().losses.length}
                rangeLabel={`steps 1 to ${run().losses.length}`}
              />
            </Show>

            <Show when={run().phase === "done"}>
              <Show when={run().models.length} fallback={<p class="hint">Only the adapter was saved.</p>}>
                <ul class="train-models" role="list">
                  <For each={run().models}>
                    {(m) => (
                      <li>
                        <Icon name="models" size={16} />
                        <span class="mono">{fileName(m)}</span>
                      </li>
                    )}
                  </For>
                </ul>
                <a class="btn btn--secondary btn--sm" href="#/models">
                  Open in Models
                </a>
              </Show>
              <Show when={run().seconds !== null}>
                <p class="hint">
                  {run().step} steps in {Math.round(run().seconds ?? 0)} s.
                </p>
              </Show>
            </Show>

            <Show when={run().phase === "failed"}>
              <p class="train-error" role="alert">
                {run().error}
              </p>
              <Show when={status()?.stderrTail.length}>
                <details class="train-advanced" open>
                  <summary>Details</summary>
                  <pre class="train-tail mono">{status()?.stderrTail.slice(-40).join("\n")}</pre>
                </details>
              </Show>
            </Show>
          </div>
        </Section>
      </Show>
    </div>
  );
}

/** A number setting with what it does and the value Cerebro recommends; the user decides. */
function NumField(props: { label: string; value: number; step?: string; err: string | undefined; onInput: (v: number) => void; rec?: number; why?: string }): JSX.Element {
  return (
    <label class="train-field">
      <span>{props.label}</span>
      <input
        class="input mono"
        type="number"
        step={props.step ?? "1"}
        value={Number.isFinite(props.value) ? props.value : ""}
        aria-invalid={props.err ? "true" : undefined}
        onInput={(e) => props.onInput(e.currentTarget.value.trim() === "" ? Number.NaN : Number(e.currentTarget.value))}
      />
      <Show when={props.err}>{(m) => <span class="train-field__err">{m()}</span>}</Show>
      <Show when={props.why}>
        <span class="train-why">
          {props.why}
          <Show when={props.rec !== undefined}>
            {" "}
            Recommended: <span class="mono">{props.rec}</span>.
            <Show when={props.rec !== props.value}>
              {" "}
              <button type="button" class="btn btn--tertiary btn--sm train-use" onClick={(e) => (e.preventDefault(), props.onInput(props.rec!))}>
                Use
              </button>
            </Show>
          </Show>
        </span>
      </Show>
    </label>
  );
}
