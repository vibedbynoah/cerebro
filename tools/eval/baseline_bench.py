"""Measured throughput baselines, collected without any Claude agent.

Runs llama-bench over a small matrix of (build, model, flag set) and stores the raw JSON
plus a provenance record per run. This is the data the claims policy requires: every number
carries the build, model, quant, flags, context depth, and free VRAM at start.

Two builds are compared deliberately: the vendored b10751 that Cerebro currently spawns, and
the official b11026 release, because a self-built fork previously measured 3-5x slower than an
official release. Flag sets are the honest comparison Cerebro's thesis rests on: llama.cpp's
own defaults (--fit decides) versus the tuned placement the owner measured earlier.

Screening only: -r 4 with the first repetition discarded by the analysis, depths 0 and 4096.
Sustained soaks belong to the eval-suite crate, not here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\noahm\Projects\marrow")
OUT = REPO / "docs" / "research" / "2026-09" / "measurements"
LOG = OUT / "baseline.log"

BUILDS = {
    "b10751-vendored": Path(r"C:\Users\noahm\Projects\qwen-flash-next\llama.cpp-b10751"),
    "b11026-official": Path(r"C:\bench\llamacpp-official"),
}

MODELS = {
    "gemma-4-E4B-Q4_K_M": Path(r"C:\Users\noahm\.lmstudio\models\lmstudio-community\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-Q4_K_M.gguf"),
    # Re-scanned 2026-09-22: the Brian6145 27B is gone and several 35B-class MoEs have appeared,
    # so the Sep 12 inventory (and the eval plan's model matrix) is stale.
    "qwen3.8-27B-Q4_K_M": Path(r"C:\Users\noahm\.lmstudio\models\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf"),
    "qwen3.6-35B-A3B-Q4_K_M": Path(r"C:\Users\noahm\.lmstudio\models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf"),
    "gemma-4-26B-A4B-QAT-Q4_0": Path(r"C:\Users\noahm\.lmstudio\models\lmstudio-community\gemma-4-26B-A4B-it-QAT-GGUF\gemma-4-26B-A4B-it-QAT-Q4_0.gguf"),
    "qwen3-coder-30B-A3B-Q4_K_M": Path(r"C:\models\Qwen3-Coder-30B-A3B-Instruct-GGUF\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"),
}

# Flag sets. "defaults" leaves llama.cpp's own fitter in charge, which is the fair baseline to
# beat; "tuned" is the owner's measured placement (full offload, MoE experts to host RAM, q8_0 KV,
# P-cores only). -ncmoe is meaningless for a dense model, so it is only used for the MoE.
COMMON = ["-p", "512", "-n", "128", "-d", "0,4096", "-r", "4", "-o", "json"]
FLAGSETS = {
    "defaults": [],
    "tuned": ["-ngl", "999", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-t", "6", "-C", "555"],
    "tuned-moe": ["-ngl", "999", "-ncmoe", "32", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-t", "6", "-C", "555"],
}

MATRIX = [
    ("b10751-vendored", "gemma-4-E4B-Q4_K_M", "defaults"),
    ("b10751-vendored", "gemma-4-E4B-Q4_K_M", "tuned"),
    ("b11026-official", "gemma-4-E4B-Q4_K_M", "tuned"),
    ("b10751-vendored", "qwen3-coder-30B-A3B-Q4_K_M", "defaults"),
    ("b10751-vendored", "qwen3-coder-30B-A3B-Q4_K_M", "tuned-moe"),
    ("b11026-official", "qwen3-coder-30B-A3B-Q4_K_M", "tuned-moe"),
    ("b10751-vendored", "qwen3.8-27B-Q4_K_M", "defaults"),
    ("b10751-vendored", "qwen3.8-27B-Q4_K_M", "tuned"),
    ("b10751-vendored", "qwen3.6-35B-A3B-Q4_K_M", "defaults"),
    ("b10751-vendored", "qwen3.6-35B-A3B-Q4_K_M", "tuned-moe"),
    ("b10751-vendored", "gemma-4-26B-A4B-QAT-Q4_0", "tuned-moe"),
]

TIMEOUT_S = 45 * 60


QUIET = "--quiet-wait" in sys.argv
if QUIET:
    # Builders compiling Rust in three worktrees saturate the same P-cores llama-bench pins, so a
    # contended run measures the machine, not the configuration. Quiet runs are kept separately.
    OUT = OUT / "quiet"
    LOG = OUT / "baseline.log"
NOISY = ("cargo", "rustc", "llama-server", "llama-bench", "opencode")
QUIET_REQUIRED_S = 60
QUIET_MAX_WAIT_S = 6 * 60 * 60


def busy_processes() -> list[str]:
    try:
        out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    names = []
    for line in out.splitlines():
        name = line.split('","')[0].lstrip('"').lower()
        for n in NOISY:
            if name.startswith(n) and not name.startswith("llama-bench"):
                names.append(name)
    return names


def wait_for_quiet() -> bool:
    """Block until nothing heavy has run for QUIET_REQUIRED_S. False if the wait times out."""
    deadline = time.time() + QUIET_MAX_WAIT_S
    calm_since = None
    announced = False
    while time.time() < deadline:
        busy = busy_processes()
        if busy:
            if not announced:
                log(f"waiting for a quiet machine; busy: {sorted(set(busy))}")
                announced = True
            calm_since = None
        elif calm_since is None:
            calm_since = time.time()
        elif time.time() - calm_since >= QUIET_REQUIRED_S:
            return True
        time.sleep(15)
    log("quiet wait timed out; measuring anyway and marking the run contended")
    return False


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_vram_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def run_one(build: str, model: str, flagset: str) -> dict:
    stem = f"{build}__{model}__{flagset}"
    dest = OUT / f"{stem}.json"
    meta_path = OUT / f"{stem}.meta.json"
    if dest.exists() and meta_path.exists():
        log(f"have {stem}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    exe = BUILDS[build] / "llama-bench.exe"
    gguf = MODELS[model]
    if not exe.exists() or not gguf.exists():
        log(f"SKIP {stem}: missing {'binary' if not exe.exists() else 'model'}")
        return {"stem": stem, "status": "missing-input"}

    argv = [str(exe), "-m", str(gguf), *COMMON, *FLAGSETS[flagset]]
    quiet = wait_for_quiet() if QUIET else False
    before = free_vram_mib()
    log(f"run {stem}: free VRAM {before} MiB")
    started = time.time()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_S, cwd=str(BUILDS[build]))
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT {stem} after {TIMEOUT_S}s")
        return {"stem": stem, "status": "timeout"}
    elapsed = round(time.time() - started, 1)

    meta = {
        "stem": stem, "build": build, "model": model, "flagset": flagset,
        "argv": argv, "returncode": proc.returncode, "seconds": elapsed,
        "free_vram_mib_before": before, "free_vram_mib_after": free_vram_mib(),
        "machine_state": "quiet" if quiet else ("contended" if QUIET else "unchecked"),
        "stderr_tail": proc.stderr.strip().splitlines()[-8:],
    }
    start = proc.stdout.find("[")
    if proc.returncode == 0 and start >= 0:
        dest.write_text(proc.stdout[start:], encoding="utf-8")
        meta["status"] = "ok"
    else:
        meta["status"] = "failed"
        log(f"FAILED {stem} rc={proc.returncode}: {meta['stderr_tail'][-1] if meta['stderr_tail'] else 'no stderr'}")
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"done {stem} status={meta['status']} in {elapsed}s")
    return meta


def summarize() -> None:
    rows = []
    for meta_file in sorted(OUT.glob("*.meta.json")):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        data_file = OUT / (meta["stem"] + ".json")
        if meta.get("status") != "ok" or not data_file.exists():
            rows.append((meta["stem"], meta.get("status"), "", "", ""))
            continue
        records = json.loads(data_file.read_text(encoding="utf-8"))
        pp = tg0 = tgd = ""
        for r in records:
            samples = r.get("samples_ts") or []
            kept = samples[1:] or samples  # first repetition is always discarded
            if not kept:
                continue
            median = sorted(kept)[len(kept) // 2]
            if r.get("n_prompt", 0) > 0:
                pp = f"{median:.1f}"
            elif r.get("n_depth", 0) > 0:
                tgd = f"{median:.1f}"
            else:
                tg0 = f"{median:.1f}"
        rows.append((meta["stem"], "ok", pp, tg0, tgd))
    width = max((len(r[0]) for r in rows), default=20)
    lines = [f"{'run'.ljust(width)}  status    pp tok/s  tg tok/s  tg@4096"]
    for stem, status, pp, tg0, tgd in rows:
        lines.append(f"{stem.ljust(width)}  {str(status).ljust(8)}  {pp.rjust(7)}  {tg0.rjust(8)}  {tgd.rjust(7)}")
    text = "\n".join(lines)
    (OUT / "SUMMARY.txt").write_text(text + "\n", encoding="utf-8")
    log("summary:\n" + text)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"start; C: free {shutil.disk_usage('C:')[2] / 2**30:.0f} GB")
    for build, model, flagset in MATRIX:
        run_one(build, model, flagset)
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
