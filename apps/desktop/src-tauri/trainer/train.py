"""Cerebro's training sidecar: fine-tune a model with Unsloth and export it to GGUF.

Run by the app (src-tauri/src/trainer.rs) inside the venv it manages:
    python train.py JOB.json

The job file (written by the app) holds every setting. stdout carries ONLY JSON lines, one event
per line, which the app streams to the Train screen:
    {"type":"status","stage":"loading_model"|"preparing_data"|"training"|"saving"|"exporting","detail":...}
    {"type":"log","step":N,"max_steps":M,"loss":x,"lr":x,"grad_norm":x,"epoch":x}
    {"type":"export","quant":"q4_k_m","path":"..."}
    {"type":"done","gguf":[...],"adapter":"...","steps":N,"seconds":S}
    {"type":"error","message":"..."}
Everything else (library logs, warnings, progress bars) goes to stderr. Creating the job's stop_file stops
training at the next step; the adapter is still saved.
"""

import json
import os
import signal
import sys
import time
import traceback

# Library chatter must never reach stdout, which is the event channel.
_events = sys.stdout
sys.stdout = sys.stderr
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def emit(event: dict) -> None:
    _events.write(json.dumps(event, ensure_ascii=False) + "\n")
    _events.flush()


STOP = {"requested": False}


def _on_break(_signum, _frame):
    STOP["requested"] = True


for _sig in ("SIGBREAK", "SIGINT", "SIGTERM"):
    if hasattr(signal, _sig):
        signal.signal(getattr(signal, _sig), _on_break)


def load_job(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        job = json.load(f)
    required = ["model", "dataset", "output_dir"]
    missing = [k for k in required if not job.get(k)]
    if missing:
        raise ValueError(f"The job is missing {', '.join(missing)}.")
    defaults = {
        "format": "auto",
        "method": "qlora",
        "rank": 16,
        "alpha": 16,
        "lr": 2e-4,
        "epochs": 1,
        "max_steps": 0,
        "max_seq_len": 2048,
        "batch": 2,
        "grad_accum": 4,
        "seed": 3407,
        "export": ["q4_k_m"],
        "responses_only": True,
    }
    for k, v in defaults.items():
        job.setdefault(k, v)
    return job


def to_messages(row: dict, fmt: str):
    """One dataset row as chat messages, or None when the row does not fit the format."""
    if fmt in ("auto", "alpaca") and "instruction" in row and "output" in row:
        user = row["instruction"] if not row.get("input") else f"{row['instruction']}\n\n{row['input']}"
        return [{"role": "user", "content": str(user)}, {"role": "assistant", "content": str(row["output"])}]
    for key in ("messages", "conversations"):
        conv = row.get(key)
        if fmt in ("auto", "chatml", "sharegpt") and isinstance(conv, list) and conv:
            out = []
            for turn in conv:
                role = turn.get("role") or {"human": "user", "gpt": "assistant", "system": "system"}.get(turn.get("from"), turn.get("from"))
                content = turn.get("content", turn.get("value"))
                if role in ("system", "user", "assistant") and isinstance(content, str):
                    out.append({"role": role, "content": content})
            return out or None
    if fmt in ("auto",) and "prompt" in row and "completion" in row:
        return [{"role": "user", "content": str(row["prompt"])}, {"role": "assistant", "content": str(row["completion"])}]
    return None


def load_rows(spec: dict):
    from datasets import load_dataset

    if spec.get("path"):
        path = spec["path"]
        ext = os.path.splitext(path)[1].lower()
        kind = {".jsonl": "json", ".json": "json", ".csv": "csv", ".parquet": "parquet"}.get(ext)
        if kind is None:
            raise ValueError("Datasets must be .jsonl, .json, .csv or .parquet files.")
        return load_dataset(kind, data_files=path, split="train")
    if spec.get("hf"):
        return load_dataset(spec["hf"], split=spec.get("split", "train"))
    raise ValueError("The dataset needs a file path or a Hugging Face id.")


def main() -> int:
    if len(sys.argv) != 2:
        emit({"type": "error", "message": "usage: train.py JOB.json"})
        return 2
    started = time.time()
    job = load_job(sys.argv[1])
    os.makedirs(job["output_dir"], exist_ok=True)

    emit({"type": "status", "stage": "loading_model", "detail": job["model"]})
    from unsloth import FastLanguageModel  # imported first: it patches transformers/trl
    import torch
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=job["model"],
        max_seq_length=int(job["max_seq_len"]),
        load_in_4bit=job["method"] == "qlora",
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(job["rank"]),
        lora_alpha=int(job["alpha"]),
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(job["seed"]),
    )

    emit({"type": "status", "stage": "preparing_data", "detail": job["dataset"].get("path") or job["dataset"].get("hf")})
    rows = load_rows(job["dataset"])
    fmt = job["format"]
    texts, skipped = [], 0
    for row in rows:
        msgs = to_messages(row, fmt)
        if not msgs:
            skipped += 1
            continue
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
    if not texts:
        raise ValueError(f"No row of the dataset could be read as {fmt}; {skipped} rows skipped.")
    from datasets import Dataset

    data = Dataset.from_dict({"text": texts})
    emit({"type": "status", "stage": "preparing_data", "detail": f"{len(texts)} examples ({skipped} skipped)"})

    class Events(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" in logs or "eval_loss" in logs:
                emit({
                    "type": "log",
                    "step": state.global_step,
                    "max_steps": state.max_steps,
                    "loss": logs.get("loss"),
                    "eval_loss": logs.get("eval_loss"),
                    "lr": logs.get("learning_rate"),
                    "grad_norm": logs.get("grad_norm"),
                    "epoch": logs.get("epoch"),
                })

        def on_step_end(self, args, state, control, **kwargs):
            # The app asks for a stop by creating the job's stop file (signals are unreliable on Windows).
            if job.get("stop_file") and os.path.exists(job["stop_file"]):
                STOP["requested"] = True
            if STOP["requested"]:
                control.should_training_stop = True
            return control

    config = SFTConfig(
        dataset_text_field="text",
        max_length=int(job["max_seq_len"]),
        per_device_train_batch_size=int(job["batch"]),
        gradient_accumulation_steps=int(job["grad_accum"]),
        num_train_epochs=float(job["epochs"]),
        max_steps=int(job["max_steps"]) if int(job["max_steps"]) > 0 else -1,
        learning_rate=float(job["lr"]),
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=int(job["seed"]),
        output_dir=os.path.join(job["output_dir"], "checkpoints"),
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=data, args=config, callbacks=[Events()])
    if job.get("responses_only"):
        # Learn the answers, not the prompts, when the template marks where answers start.
        try:
            from unsloth.chat_templates import train_on_responses_only

            parts = {
                "<|im_start|>": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
                "<start_of_turn>": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
                "<|start_header_id|>": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
            }
            for marker, (inst, resp) in parts.items():
                if marker in texts[0]:
                    trainer = train_on_responses_only(trainer, instruction_part=inst, response_part=resp)
                    break
        except Exception as e:  # noqa: BLE001 - optional refinement; training still works without it
            print(f"responses-only training not applied: {e}", file=sys.stderr)

    emit({"type": "status", "stage": "training", "detail": f"{len(texts)} examples"})
    result = trainer.train()

    emit({"type": "status", "stage": "saving", "detail": "LoRA adapter"})
    adapter = os.path.join(job["output_dir"], "adapter")
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)

    gguf = []
    quants = [q for q in job["export"] if q]
    if quants and not STOP["requested"]:
        emit({"type": "status", "stage": "exporting", "detail": ", ".join(quants)})
        out = os.path.join(job["output_dir"], "gguf")
        model.save_pretrained_gguf(out, tokenizer, quantization_method=quants)
        for root, _dirs, files in os.walk(job["output_dir"]):
            for name in files:
                if name.lower().endswith(".gguf"):
                    path = os.path.join(root, name)
                    quant = next((q for q in quants if q.lower() in name.lower()), "gguf")
                    gguf.append(path)
                    emit({"type": "export", "quant": quant, "path": path})
        # The merged 16-bit weights were only the converter's input (8 GB for a 4B model).
        for root, _dirs, files in os.walk(out):
            for name in files:
                if name.endswith(".safetensors"):
                    os.remove(os.path.join(root, name))
    emit({
        "type": "done",
        "gguf": gguf,
        "adapter": adapter,
        "steps": int(getattr(result, "global_step", 0)),
        "seconds": round(time.time() - started, 1),
        "stopped": STOP["requested"],
    })
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 - every failure must reach the app as an event
        traceback.print_exc(file=sys.stderr)
        emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        sys.exit(1)
