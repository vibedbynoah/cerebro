"""Run Cerebro research subtasks on free OpenCode Zen models, three at a time.

Each subtask runs `opencode run` in its own work directory and must produce
notes.md and items.json there. Valid outputs are copied into
docs/research/2026-09/sub/. Already-validated subtasks are skipped, so the
script is safe to re-run after an interruption.

After all subtasks, per-track outputs are merged into
docs/research/2026-09/<track>.md (only when no earlier notes file exists) and
docs/research/2026-09/state/research__<track>.json, with every source URL
link-checked so adjudicators can see which citations actually resolve.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RDIR = REPO / "docs" / "research" / "2026-09b"
SUB = RDIR / "sub"
STATE = RDIR / "state"
WORK = Path(os.environ["LOCALAPPDATA"]) / "cerebro-research-b"
OPENCODE = Path(os.environ["APPDATA"]) / "npm" / "node_modules" / "opencode-ai" / "bin" / "opencode.exe"

MODELS = os.environ.get("CEREBRO_MODELS", "").split(",") if os.environ.get("CEREBRO_MODELS") else [
    "opencode/muse-spark-1.3-contributor-free",
    "opencode/mimo-v2.5-free",
]
if len(MODELS) == 1:
    MODELS.append(MODELS[0])
CONCURRENCY = 3
ATTEMPT_TIMEOUT_S = 25 * 60
MIN_ITEMS = 5

FEATURE_FIELDS = """Each item object has exactly these string fields (use "unknown" when unverified, never guess):
- "name": technique or product name
- "status": one of "production" (shipping in a maintained runtime or product), "emerging" (in a fork, PR or experimental flag), "research-only" (paper or prototype)
- "what": what it is, one or two sentences
- "where_implemented": exact runtime / flag / PR / repo
- "reported_gain": measured gain WITH conditions (hardware, batch size, model)
- "quality_cost": effect on output quality
- "fit_reference_machine": fit for the 8 GB Ampere laptop + 32 GB RAM, batch 1 and a small office
- "harness_integration": how a Rust harness that spawns llama.cpp could implement, expose or approximate it
and "sources": an array of URLs you actually opened."""

DESIGN_FIELDS = FEATURE_FIELDS.replace(
    "- \"reported_gain\"", "- (for this design/stack topic: status = maturity; where_implemented = exemplar products or libraries; reported_gain = measured numbers such as startup ms, MB, KB, scores)\n- \"reported_gain\""
)

EVAL_FIELDS = FEATURE_FIELDS.replace(
    "- \"reported_gain\"", "- (for this evaluation topic: where_implemented = where the benchmark or tool lives; reported_gain = runtime cost and known scores of open models; fit_reference_machine = feasibility on this machine; harness_integration = how Cerebro would run it)\n- \"reported_gain\""
)

COMPETITOR_FIELDS = """Each item object is one product, with exactly these fields (use "unknown" when unverified, never guess):
- "name", "url" (official site), "license"
- "category": runner / workspace / enthusiast / vendor / coding-agent
- "active_2026": evidence of maintenance in 2026 (latest release and date)
- "windows_install": portable zip / installer / Docker / pip / npm, with official asset names
- "backend": inference engine(s) used
- "features": chat, RAG, agents, tools/MCP, coding, voice, vision, multi-user, API compatibility
- "perf_tuning": offload control, speculative decoding exposure, KV cache options, autotuning
- "security_record": CVEs, default network exposure, authentication
- "ux_reputation": what users praise and complain about, with where it was said
- "same_model_testable": true or false (boolean) - can it load an arbitrary local GGUF file
and "sources": an array of URLs you actually opened."""

log_lock = threading.Lock()
LOG = RDIR / "sub" / "run.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with log_lock:
        print(line, flush=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def build_prompt(task: dict, facts: str) -> str:
    kind = task["kind"]
    fields = {
        "features": FEATURE_FIELDS,
        "convert": FEATURE_FIELDS,
        "design": DESIGN_FIELDS,
        "eval": EVAL_FIELDS,
        "competitors": COMPETITOR_FIELDS,
    }[kind]
    aim = "one item per product listed in the scope, plus any product you add" if kind == "competitors" else "8-20 items"
    return f"""You are a research agent for Cerebro Engine.

{facts}

TASK ID: {task['id']}
TRACK: {task['track_title']}
SUBTOPIC: {task['title']}
SCOPE (cover all of it, and add anything cutting-edge from 2025-2026 that is missing, including research-only work):
{task['scope']}

METHOD
- Use your websearch and webfetch tools heavily: at least 10 searches, and open and read primary sources (arXiv papers, GitHub repositories, pull requests, release notes, official documentation). Do not rely on memory for facts, versions, numbers or dates.
- Every factual claim needs a source URL that you actually opened. Never invent a URL. If you could not verify something, write "unverified".
- Be skeptical: datacenter or large-batch numbers do not transfer to a batch-1 laptop; say so when relevant.
- Do not install software. Do not create or modify files outside the current directory.

OUTPUT: create exactly two files in the current directory.
1. notes.md: detailed markdown notes with inline source URLs, at most about 20 KB.
2. items.json: valid JSON (no comments, no trailing commas) with the shape
   {{"headline_findings": ["..."], "items": [ ... ]}}
{fields}
Aim for {aim}. Keep each string under about 400 characters.

When both files are written, reply with exactly: DONE {task['id']}
"""


def validate(workdir: Path, kind: str) -> tuple[bool, str]:
    notes = workdir / "notes.md"
    items_path = workdir / "items.json"
    if not notes.exists() or notes.stat().st_size < 1500:
        return False, "notes.md missing or too small"
    if not items_path.exists():
        return False, "items.json missing"
    raw = items_path.read_text(encoding="utf-8", errors="replace").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"items.json invalid JSON: {e}"
    if isinstance(data, list):
        data = {"headline_findings": [], "items": data}
    items = data.get("items")
    if not isinstance(items, list) or len(items) < (3 if kind == "competitors" else MIN_ITEMS):
        return False, f"too few items ({0 if not isinstance(items, list) else len(items)})"
    for i, it in enumerate(items):
        if not isinstance(it, dict) or not it.get("name"):
            return False, f"item {i} has no name"
        if not isinstance(it.get("sources"), list) or not it["sources"]:
            return False, f"item {i} ({it.get('name')}) has no sources"
    items_path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return True, f"{len(items)} items"


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


# The free tier rate-limits per account, across every free model at once. A limited
# request either fails fast with this message or hangs with no output until the
# attempt timeout, so both count as rate limiting and trigger a shared cooldown.
RATE_LIMIT_MARKERS = ("rate limit exceeded", "too many requests", "429")
STALL_S = 8 * 60
MAX_RATE_LIMIT_RETRIES = 10
_cooldown_lock = threading.Lock()
_cooldown_until = 0.0
_cooldown_step = 0


def _wait_for_cooldown() -> None:
    while True:
        with _cooldown_lock:
            remaining = _cooldown_until - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 30))


def _register_rate_limit() -> None:
    global _cooldown_until, _cooldown_step
    with _cooldown_lock:
        if time.time() < _cooldown_until:
            return  # another worker already started this cooldown
        delay = min(5 * 60 * 2 ** _cooldown_step, 60 * 60)
        _cooldown_step += 1
        _cooldown_until = time.time() + delay
    log(f"RATE LIMIT: all workers pause {delay // 60} min")


def _register_success() -> None:
    global _cooldown_step
    with _cooldown_lock:
        _cooldown_step = 0


def run_opencode(cmd: list[str], workdir: Path, label: str) -> str:
    """Run one opencode attempt. Returns "exit:<rc>", "timeout", "stalled" or "rate-limited"."""
    _wait_for_cooldown()
    log_path = workdir / "opencode.log"
    env = dict(os.environ, OPENCODE_ENABLE_EXA="1")
    started = last_growth = time.time()
    last_size = -1
    with log_path.open("w", encoding="utf-8", errors="replace") as out:
        proc = subprocess.Popen(cmd, cwd=workdir, stdout=out, stderr=subprocess.STDOUT, env=env,
                                stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        outcome = None
        while outcome is None:
            try:
                rc = proc.wait(timeout=15)
                outcome = f"exit:{rc}"
            except subprocess.TimeoutExpired:
                now = time.time()
                size = log_path.stat().st_size
                if size != last_size:
                    last_size, last_growth = size, now
                if now - started > ATTEMPT_TIMEOUT_S:
                    kill_tree(proc.pid)
                    outcome = "timeout"
                elif now - last_growth > STALL_S:
                    kill_tree(proc.pid)
                    outcome = "stalled"
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    if any(m in text for m in RATE_LIMIT_MARKERS) or (outcome == "stalled" and last_size < 400):
        # The caller registers the cooldown only if the output is also invalid: a successful run can
        # mention a 429 it recovered from.
        return "rate-limited"
    return outcome


def run_task(task: dict, facts: str) -> dict:
    tid = task["id"]
    done_md = SUB / f"{tid}.md"
    done_items = SUB / f"{tid}.items.json"
    if done_md.exists() and done_items.exists():
        return {"id": tid, "status": "cached"}

    prompt = build_prompt(task, facts)
    attempt = 0
    limited = 0
    while attempt < 2:
        model = MODELS[attempt]
        # A fresh directory per try: a killed opencode can briefly hold files open, so reusing and
        # deleting the previous directory races with it.
        workdir = WORK / tid / f"attempt{attempt + 1}-try{limited}-{int(time.time())}"
        workdir.mkdir(parents=True, exist_ok=True)
        cmd = [str(OPENCODE), "run", "--pure", "-m", model, "--dir", str(workdir), "--title", f"cerebro-{tid}"]
        if task["kind"] == "convert":
            shutil.copy(RDIR / "test-time-reasoning.md", workdir / "test-time-reasoning.md")
            cmd += ["-f", str(workdir / "test-time-reasoning.md")]
        # "--" ends yargs option parsing; without it -f is greedy and swallows the prompt as a file.
        cmd += ["--", prompt]
        started = time.time()
        log(f"START {tid} attempt {attempt + 1} model {model}")
        outcome = run_opencode(cmd, workdir, tid)
        elapsed = int(time.time() - started)
        ok, why = validate(workdir, task["kind"])
        log(f"END   {tid} attempt {attempt + 1} {outcome} {elapsed}s valid={ok} ({why})")
        if ok:
            _register_success()
            SUB.mkdir(parents=True, exist_ok=True)
            shutil.copy(workdir / "notes.md", done_md)
            shutil.copy(workdir / "items.json", done_items)
            return {"id": tid, "status": "ok", "model": model, "attempt": attempt + 1, "seconds": elapsed, "detail": why}
        if outcome == "rate-limited" and limited < MAX_RATE_LIMIT_RETRIES:
            _register_rate_limit()
            limited += 1
            continue  # a rate limit says nothing about the model, so it does not use up an attempt
        attempt += 1
    return {"id": tid, "status": "failed"}


def check_url(url: str) -> str:
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return "invalid"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (cerebro-research link check)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return "ok" if r.status < 400 else f"http{r.status}"
    except urllib.error.HTTPError as e:
        return "blocked" if e.code in (401, 403, 429) else f"http{e.code}"
    except Exception:
        return "unreachable"


def merge_tracks(tasks: list[dict]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    tracks: dict[str, list[dict]] = {}
    for t in tasks:
        tracks.setdefault(t["track"], []).append(t)

    all_urls: set[str] = set()
    loaded: dict[str, list[tuple[dict, dict, str]]] = {}
    for track, ts in tracks.items():
        for t in ts:
            p = SUB / f"{t['id']}.items.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            notes = (SUB / f"{t['id']}.md").read_text(encoding="utf-8", errors="replace")
            loaded.setdefault(track, []).append((t, data, notes))
            for it in data.get("items", []):
                all_urls.update(u for u in it.get("sources", []) if isinstance(u, str))

    log(f"link-checking {len(all_urls)} URLs")
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        results = dict(zip(all_urls, ex.map(check_url, all_urls)))

    for track, parts in loaded.items():
        items, headlines, md = [], [], []
        seen: set[str] = set()
        for t, data, notes in parts:
            headlines += [f"[{t['title']}] {h}" for h in data.get("headline_findings", [])]
            md.append(f"# {t['title']}\n\n_Subtask {t['id']}_\n\n{notes}\n")
            for it in data.get("items", []):
                key = re.sub(r"[^a-z0-9]", "", str(it.get("name", "")).lower())
                if key in seen:
                    continue
                seen.add(key)
                srcs = [u for u in it.get("sources", []) if isinstance(u, str)]
                checks = [results.get(u, "unchecked") for u in srcs]
                good = sum(c in ("ok", "blocked") for c in checks)
                it["source_check"] = f"{good}/{len(srcs)} resolve" + ("" if good == len(srcs) else f" (dead: {', '.join(u for u, c in zip(srcs, checks) if c not in ('ok', 'blocked'))})")
                it["subtopic"] = t["title"]
                items.append(it)
        track_md = RDIR / f"{track}.md"
        if track == "test-time-reasoning" and track_md.exists():
            file_written = str(track_md)
        else:
            track_md.write_text("\n\n".join(md), encoding="utf-8")
            file_written = str(track_md)
        missing = [t["id"] for t in tracks[track] if not (SUB / f"{t['id']}.items.json").exists()]
        state = {"track": track, "file_written": file_written, "headline_findings": headlines,
                 "items": items, "missing_subtasks": missing, "produced_by": "opencode free models"}
        (STATE / f"research__{track}.json").write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
        dead = sum("dead" in it["source_check"] for it in items)
        log(f"merged {track}: {len(items)} items, {dead} with dead links, missing subtasks: {missing or 'none'}")


def main() -> int:
    if not OPENCODE.exists():
        print(f"opencode not found at {OPENCODE}", file=sys.stderr)
        return 2
    SUB.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    facts = (HERE / "facts.md").read_text(encoding="utf-8")
    tasks = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))
    only = set(sys.argv[1:])
    if only:
        tasks_to_run = [t for t in tasks if t["id"] in only]
    else:
        tasks_to_run = tasks
    log(f"running {len(tasks_to_run)} subtasks, concurrency {CONCURRENCY}")
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        results = list(ex.map(lambda t: run_task(t, facts), tasks_to_run))
    (SUB / "status.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    failed = [r["id"] for r in results if r["status"] == "failed"]
    log(f"subtasks done: {sum(r['status'] in ('ok', 'cached') for r in results)} ok, {len(failed)} failed {failed}")
    merge_tracks(tasks)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
