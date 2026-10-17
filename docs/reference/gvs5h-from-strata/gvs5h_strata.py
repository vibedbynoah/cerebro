"""
GVS5H ledger-based self-orchestration, ported to run against a local Strata server.

Ported from slee-persis/GVS5H (MIT), codebase/v2-current/escalation/multiagent.py.
The original targets stdin/stdout competition problems via LiveCodeBench; this
version targets real engineering tasks and speaks to Strata's Anthropic-compatible
/v1/messages endpoint on 127.0.0.1:8081.

Method (unchanged in spirit): every role is the SAME model in a fresh context.
They coordinate only through files in a shared workspace -- plan.md, notes.md,
tasks.json, and the evolving artifact. Nothing is passed in memory between agents,
which is what makes it training-free and what the ledger reconstructs.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("STRATA_URL", "http://127.0.0.1:8081").rstrip("/")
MODEL = os.environ.get("STRATA_MODEL", "qwen3.8-flash-next-ud-iq4_xs")
TOKEN = os.environ.get("STRATA_TOKEN", "strata-local")

# --- DSH worker engine -------------------------------------------------------
# DSH (DeepSeek Harness) is the executor: it runs the task with real tools in the
# working directory. The manager/ideation roles stay on direct chat because they
# must return parsable "### SECTION" blocks, which an agent run does not.
ENGINE = os.environ.get("GVS5H_ENGINE", "dsh")           # "dsh" or "direct"
DSH_NODE = os.environ.get("DSH_NODE", r"C:\Program Files\nodejs\node.exe")
DSH_BIN = os.environ.get(
    "DSH_BIN",
    r"C:\Program Files\DSH Desktop\resources\app\node_modules\@deepseek-ai\dsh\lib\bin.js")
DSH_HOME = os.environ.get("DSH_HOME", r"C:\Users\noahm\AppData\Roaming\dsh-desktop\harness")
DSH_PROFILE = os.environ.get("DSH_PROFILE", "headless")
DSH_PATCH = os.environ.get("DSH_PATCH", "")              # optional overlay YAML
DSH_TIMEOUT = int(os.environ.get("DSH_TIMEOUT", "7200"))
WORKDIR = os.environ.get("GVS5H_WORKDIR", os.getcwd())

MAX_ITERS = int(os.environ.get("GVS5H_MAX_ITERS", "8"))
REQUEST_TIMEOUT = int(os.environ.get("STRATA_TIMEOUT", "3600"))
MAX_TOKENS = int(os.environ.get("STRATA_MAX_TOKENS", "16000"))
WS_ROOT = os.environ.get("GVS5H_WS", os.path.abspath(os.path.join(os.path.dirname(__file__), "ws")))
STRICT = os.environ.get("GVS5H_STRICT", "auto")


def log(msg):
    print(msg, flush=True)


# --- transport --------------------------------------------------------------

def chat(system, user, temperature=0.2, max_tokens=None, thinking=False):
    """One Strata call. Returns (text, meta)."""
    body = {
        "model": MODEL,
        "max_tokens": max_tokens or MAX_TOKENS,
        "messages": [{"role": "user", "content": f"{system}\n\n---\n\n{user}"}],
    }
    if thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": 2048}
    else:
        body["temperature"] = temperature
    req = urllib.request.Request(
        f"{BASE_URL}/v1/messages",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
    )
    meta = {}
    last = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                out = json.loads(r.read())
            text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
            think = "".join(b.get("thinking", "") for b in out.get("content", []) if b.get("type") == "thinking")
            meta["finish_reason"] = out.get("stop_reason")
            meta["in_tokens"] = (out.get("usage") or {}).get("input_tokens")
            meta["out_tokens"] = (out.get("usage") or {}).get("output_tokens")
            meta["thinking"] = think
            if text.strip():
                return text, meta
            last = text
            log(f"    empty reply (stop={out.get('stop_reason')}), retry {attempt + 1}")
        except urllib.error.HTTPError as e:
            body_txt = (e.read().decode() or "")[:300]
            log(f"    HTTP {e.code}: {body_txt}")
            if e.code in (400, 401, 403, 404, 422):
                meta["infra_exhausted"] = True
                return last, meta
        except Exception as e:  # noqa: BLE001
            log(f"    {type(e).__name__}: {str(e)[:200]}")
            meta.setdefault("infra_exhausted", True)
            return last, meta
        time.sleep(3 * (attempt + 1))
    meta["infra_exhausted"] = True
    return last, meta


# --- workspace ledger -------------------------------------------------------

def wpath(ws, name):
    return os.path.join(ws, name)


def _read(ws, name):
    p = wpath(ws, name)
    if not os.path.exists(p):
        return ""
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


def _write(ws, name, content):
    with open(wpath(ws, name), "w", encoding="utf-8") as f:
        f.write(content)


def _append(ws, name, content):
    with open(wpath(ws, name), "a", encoding="utf-8") as f:
        f.write(content)


def record(ws, role, messages, resp, meta):
    with open(wpath(ws, "transcript.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "t": time.time(), "role": role, "model": MODEL,
            "request": messages, "response": resp,
            "finish_reason": meta.get("finish_reason"),
            "in_tokens": meta.get("in_tokens"), "out_tokens": meta.get("out_tokens"),
        }) + "\n")


def called(ws, role, system, user, **kw):
    meta = {}
    resp, meta = chat(system, user, **kw)
    record(ws, role, [{"system": system, "user": user}], resp, meta)
    return resp, meta


def run_dsh(task_text, workdir, log):
    """Run one task through DSH headless. Returns (stdout, rc, seconds).

    DSH has no --resume, so every call is a fresh conversation. That is not a
    problem here: the shared ledger on disk (plan.md / notes.md / result.md) is
    the only channel between agents by design, so the executor is expected to
    start cold and read those files.
    """
    env = dict(os.environ)
    env["DSH_HOME"] = DSH_HOME
    env["DSH_LOCAL_API_KEY"] = env.get("DSH_LOCAL_API_KEY", "local-no-key-needed")
    # Route DSH's model at the same local server this harness uses.
    env["DSH_PATCH"] = DSH_PATCH or env.get("DSH_PATCH", "")

    cmd = [DSH_NODE, DSH_BIN, "--profile", DSH_PROFILE]
    if DSH_PATCH:
        cmd += ["--patch", DSH_PATCH]
    cmd.append(task_text)

    os.makedirs(workdir, exist_ok=True)
    t0 = time.time()
    log(f"    [dsh] launching worker in {workdir}")
    try:
        proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True,
                              text=True, timeout=DSH_TIMEOUT)
        out, err, rc = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = f"DSH TIMEOUT after {DSH_TIMEOUT}s"
        rc = -1
    except FileNotFoundError as e:
        return f"DSH launch failed: {e}", -1, time.time() - t0

    secs = time.time() - t0
    with open(wpath(ws, "dsh-last.log"), "w", encoding="utf-8") as f:
        f.write(f"rc={rc} seconds={secs:.0f}\n\n--- stdout ---\n{out}\n\n--- stderr ---\n{err}")
    log(f"    [dsh] finished rc={rc} in {secs:.0f}s ({len(out)} chars)")
    return out, rc, secs


# --- parsing ----------------------------------------------------------------

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
HDR_RE = re.compile(r"^#{1,3}\s*([A-Za-z_]+)\s*$")


def strip_think(text):
    return THINK_RE.sub("", text or "").strip()


def sections(text):
    out, cur, buf = {}, None, []
    for line in strip_think(text).splitlines():
        s = line.strip()
        m = HDR_RE.match(s) or re.match(r"^\*\*([A-Za-z_]+)\*\*\s*:?\s*$", s) or re.match(r"^([A-Z_]{3,})\s*:\s*$", s)
        if m:
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1).upper(), []
        else:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


def bullets(text):
    items = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.*\S)", line)
        if m:
            items.append(m.group(1).strip())
    return items


def fences(text, lang=None):
    pat = re.compile(rf"```{lang}\s*\n(.*?)```", re.DOTALL) if lang else re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)
    return [m.group(1).rstrip() for m in pat.finditer(strip_think(text))]


def parse_tasks(text):
    out = []
    for b in bullets(text):
        m = re.match(r"\[\s*([a-z_ ]+?)\s*\]\s*(.*)", b, re.I)
        if m:
            raw = m.group(1).lower().replace(" ", "").replace("_", "")
            desc, status = m.group(2).strip(), ("done" if raw == "done" else ("in_progress" if raw in ("wip", "inprogress") else "pending"))
        else:
            desc, status = b, "pending"
        if desc:
            out.append({"id": len(out) + 1, "desc": desc, "status": status})
    return out


def merge_tasks(tasks, descs, cap=12):
    have = {t["desc"].lower() for t in tasks}
    nid = max([t["id"] for t in tasks], default=0)
    for d in descs:
        d = d.strip()
        if d and d.lower() not in have and len(tasks) < cap:
            nid += 1
            tasks.append({"id": nid, "desc": d, "status": "pending"})
            have.add(d.lower())


# --- roles ------------------------------------------------------------------

SWARM_SYSTEM = (
    "You are the PRIMARY orchestrator of a small team of senior engineers. You own the "
    "task list and decide when the goal is met. Workers coordinate with you ONLY through "
    "files in a shared workspace, so state what they must know explicitly."
)

WORKER_SYSTEM = (
    "You are a WORKER engineer on a small team. You share a workspace with other workers "
    "and see only the files listed in your prompt. Build on the existing work where it helps, "
    "but if your task is to try a different approach, write a fresh version rather than "
    "patching a stuck one. Do not rewrite files you were not asked to touch."
)


def phase_plan(ws, goal, context, log_fn):
    sys_p = (
        SWARM_SYSTEM + " Given a goal, produce a short strategy and a task list the workers "
        "can pick up. Respond with EXACTLY these sections:\n"
        "### PLAN\n<3-6 sentence strategy>\n"
        "### TASKS\n<3-6 bullets, each a concrete unit of work>"
    )
    reply, _ = called(ws, "primary_plan", sys_p, f"GOAL:\n{goal}\n\nCONTEXT:\n{context}", temperature=0.3)
    sec = sections(reply)
    _write(ws, "plan.md", (sec.get("PLAN") or reply.strip())[:6000])
    tasks = []
    merge_tasks(tasks, bullets(sec.get("TASKS", "")))
    log_fn(f"    [primary] plan + {len(tasks)} tasks")
    return tasks


def phase_ideate(ws, goal, context, log_fn):
    sys_p = (
        "You are the FIRST WORKER. Do NOT implement anything yet. Think about the goal, "
        "identify the real difficulty, and list SEVERAL DISTINCT approaches (genuinely "
        "different designs or strategies, not variations of one idea), with pitfalls for "
        "each. Describe them in prose, no code. Respond with EXACTLY:\n"
        "### NOTES\n<your analysis>\n### NEXT\n<bullet list of distinct approaches to try>"
    )
    reply, _ = called(ws, "ideation", sys_p, f"GOAL:\n{goal}\n\nPLAN:\n{_read(ws, 'plan.md')}", temperature=0.4)
    sec = sections(reply)
    _append(ws, "notes.md", f"\n## ideation\n{strip_think(sec.get('NOTES') or reply.strip())[:8000]}\n")
    proposals = bullets(sec.get("NEXT", ""))
    log_fn(f"    [worker:ideate] {len(proposals)} approaches")
    return proposals


def phase_manage(ws, goal, context, tasks, proposals, last_summary, log_fn):
    task_lines = "\n".join(f"- [{'done' if t['status'] == 'done' else 'todo'}] {t['desc']}" for t in tasks) or "(none)"
    prop_lines = "\n".join(f"- {p}" for p in proposals) or "(none)"
    current = _read(ws, "result.md")
    sys_p = (
        SWARM_SYSTEM + " Review progress and the latest worker result, then:\n"
        "- Treat any VERIFIED / TEST OUTPUT in the latest result as ground truth: only set "
        "STATUS 'done' if it passed. If it failed you MUST set 'continue'.\n"
        "- If the current work satisfies the goal, set STATUS to 'done'.\n"
        "- Otherwise curate the task list: merge duplicates, mark completed [done], drop "
        "irrelevant items, and fold in ONLY genuinely new sub-tasks.\n"
        "- If the last worker made no real progress, do NOT keep polishing the same idea. "
        "Switch to a DIFFERENT approach, or ask for a new one.\n"
        "Respond with EXACTLY these sections:\n"
        "### STATUS\n<done|continue>\n### NEXT\n<the ONE task to do next; omit if done>\n"
        "### TASKS\n<curated list, '- [done] ...' or '- [todo] ...'>"
    )
    reply, _ = called(ws, "primary_manage", sys_p, (
        f"GOAL:\n{goal}\n\nCONTEXT:\n{context}\n\n"
        f"CURRENT RESULT:\n{current[:20000] or '(none yet)'}\n\nNOTES:\n{_read(ws, 'notes.md')[:20000]}\n\n"
        f"TASK LIST:\n{task_lines}\n\nLATEST WORKER RESULT: {last_summary}\n\nPROPOSALS:\n{prop_lines}"
    ), temperature=0.2)
    sec = sections(reply)
    curated = parse_tasks(sec.get("TASKS", "")) or tasks
    status = "done" if sec.get("STATUS", "").strip().lower().startswith("done") else "continue"
    if status == "done" and not current.strip():
        status = "continue"
    nxt = (bullets(sec.get("NEXT", ""))[:1] or [sec.get("NEXT", "").strip()])[0]
    nxt = re.sub(r"^[-*]\s*", "", nxt).strip()
    if status == "continue" and not nxt:
        todo = [t for t in curated if t["status"] != "done"]
        nxt = todo[0]["desc"] if todo else ""
    if not current.strip() and not nxt:
        status, nxt = "continue", "Implement the change described by the goal."
    _write(ws, "tasks.json", json.dumps(curated, indent=2))
    log_fn(f"    [manage] {status}" + (f" -> {nxt[:70]}" if status == "continue" else ""))
    return status, nxt, curated


def _phase_worker_dsh(ws, goal, context, task, log_fn, goal_line):
    """Worker executed by DSH instead of a bare model call.

    DSH gets the task plus the whole ledger, does the work with real tools in the
    working directory, and reports back. Its final message is asked for in the
    same ### sections the manager parses; if the model ignores that format the raw
    output still lands in result.md so nothing is lost.
    """
    current = _read(ws, "result.md")
    prompt = (
        "You are a WORKER engineer on a small team, completing ONE task.\n"
        "You share a working directory with other workers who cannot see your\n"
        "reasoning. All shared state lives in files on disk.\n\n"
        f"GOAL:\n{goal}\n\n"
        + (f"CONTEXT:\n{context}\n\n" if context else "")
        + f"PLAN:\n{_read(ws, 'plan.md')}\n\n"
        f"NOTES:\n{_read(ws, 'notes.md')}\n\n"
        f"CURRENT WORK:\n{current[:20000] or '(none yet)'}\n\n"
        f"YOUR TASK:\n{goal_line}\n\n"
        "Do the task with your tools in the current directory. Build on existing\n"
        "work where it helps; if your task is to try a different approach, write a\n"
        "fresh version rather than patching a stuck one. Do not touch files you\n"
        "were not asked to change.\n\n"
        "Finish your reply with EXACTLY these sections and nothing after them:\n"
        "### RESULT\n<what you changed: files written, and the substance of the change>\n"
        "### NOTES\n<the COMPLETE replacement notes file, under 800 words, as\n"
        "- **Topic:** ... bullets. No # headings, no bold-only lines, no ALLCAPS.>\n"
        "### NEXT\n<bulleted remaining steps, or 'none'>\n"
        "### STATUS\n<done|continue>"
    )

    out, rc, secs = run_dsh(prompt, WORKDIR, log_fn)
    record(ws, "finalize" if task.get("finalize") else f"worker:{task['id']}",
           [{"dsh_task": prompt}], out[-20000:], {"rc": rc, "seconds": round(secs), "engine": "dsh"})

    if not out.strip():
        log(f"    [worker:{task['id']}] DSH produced no output (rc={rc}); see dsh-last.log")
        return "continue", [], (f"worker {task['id']} produced no output (rc={rc}). "
                               f"Check {wpath(ws, 'dsh-last.log')}."), False

    sec = sections(out)
    result = (sec.get("RESULT") or "").strip()
    if result:
        _write(ws, "result.md", result)
    elif not _read(ws, "result.md").strip():
        # No parsable section: keep the whole reply so the manager still has it.
        _write(ws, "result.md", out.strip()[-20000:])
    if sec.get("NOTES"):
        _write(ws, "notes.md", sec["NOTES"].strip()[:8000] + "\n")

    nexts = [b for b in bullets(sec.get("NEXT", "")) if b.lower() != "none"]
    status = sec.get("STATUS", "continue").strip().lower()
    if rc != 0:
        log(f"    [worker:{task['id']}] DSH exit rc={rc}; treating as continue")
        status = "continue"
    log_fn(f"    [worker:{task['id']}] {status or 'continue'} via dsh (+{len(nexts)} next)")
    return status, nexts, (f"worker reported '{status or 'continue'}' via DSH on: "
                           f"{task['desc'][:90]}"), bool(result or out)


def phase_worker(ws, goal, context, task, log_fn, thinking=True, verify_cmd=None):
    current = _read(ws, "result.md")
    goal_line = "Produce the DEFINITIVE final deliverable now, using all notes and current work." if task.get("finalize") else f"Complete this task: {task['desc']}"
    if verify_cmd:
        goal_line += f"\nYou MAY verify your work by running: {verify_cmd}"

    if ENGINE == "dsh":
        return _phase_worker_dsh(ws, goal, context, task, log_fn, goal_line)

    out_fmt = (
        "### RESULT\n<the current full content of the deliverable: prose, code, "
        "diff or file contents as appropriate>\n"
        "### NOTES\n<the COMPLETE notes file, rewritten. Fold your findings in, keep what "
        "still matters, DELETE anything superseded or disproven. This REPLACES the file. "
        "Use '- **Topic:** ...' bullets, under ~800 words. Do NOT use markdown headings "
        "(#, ##), bold-only lines or ALLCAPS -- the reply is split on those and your notes "
        "would be truncated.>\n"
        "### NEXT\n<bulleted remaining steps, or 'none'>\n### STATUS\n<done|continue>"
    )
    sys_p = WORKER_SYSTEM + f"\nRespond with EXACTLY these sections:\n{out_fmt}"
    reply, meta = called(ws, "finalize" if task.get("finalize") else f"worker:{task['id']}", sys_p, (
        f"GOAL:\n{goal}\n\nCONTEXT:\n{context}\n\nPLAN:\n{_read(ws, 'plan.md')}\n\n"
        f"NOTES:\n{_read(ws, 'notes.md')}\n\nCURRENT RESULT:\n{current or '(none yet)'}\n\n"
        f"YOUR TASK:\n{goal_line}"
    ), temperature=0.2, thinking=thinking)
    sec = sections(reply)
    result = (sec.get("RESULT") or "").strip()
    if result:
        _write(ws, "result.md", result)
    if sec.get("NOTES") and meta.get("finish_reason") != "length":
        _write(ws, "notes.md", sec["NOTES"].strip()[:8000] + "\n")
    nexts = [b for b in bullets(sec.get("NEXT", "")) if b.lower() != "none"]
    status = sec.get("STATUS", "continue").strip().lower()
    if meta.get("finish_reason") == "length":
        log_fn(f"    [worker:{task['id']}] CUT OFF at token limit")
        return "continue", nexts, f"worker {task['id']} hit the token limit before finishing; prefer a simpler or different approach.", bool(result)
    log_fn(f"    [worker:{task['id']}] {status or 'continue'} (+{len(nexts)} next)")
    return status, nexts, f"worker reported '{status or 'continue'}' on: {task['desc'][:90]}", bool(result)


# --- driver -----------------------------------------------------------------

def run(goal, context, max_iters, verify_cmd=None, thinking=True):
    ws = os.path.join(WS_ROOT, "current")
    os.makedirs(ws, exist_ok=True)
    for f in ("notes.md", "transcript.jsonl", "plan.md", "result.md", "tasks.json"):
        _write(ws, f, "")
    _write(ws, "task.md", goal)
    record(ws, "_meta", [{"goal": goal}], "", {"model": MODEL, "max_iters": max_iters})

    t0 = time.time()
    tasks = phase_plan(ws, goal, context, log)
    proposals = phase_ideate(ws, goal, context, log)
    status, nxt, tasks = phase_manage(ws, goal, context, tasks, proposals, "ideation complete", log)

    iters, prev = 0, None
    while status == "continue" and nxt and iters < max_iters:
        if prev is not None and nxt.strip().lower() == prev.strip().lower():
            log("    [primary] reissued the same task, no progress; stopping")
            break
        prev = nxt
        iters += 1
        _, nexts, summary, wrote = phase_worker(ws, goal, context, {"id": iters, "desc": nxt}, log, thinking=thinking, verify_cmd=verify_cmd)
        status, nxt, tasks = phase_manage(ws, goal, context, tasks, nexts, summary, log)

    if status != "done" or not _read(ws, "result.md").strip():
        log("    [finalize] forcing a definitive deliverable")
        phase_worker(ws, goal, context, {"id": 0, "desc": "finalize", "finalize": True}, log, thinking=thinking, verify_cmd=verify_cmd)

    elapsed = time.time() - t0
    result = _read(ws, "result.md").strip()
    n_calls = sum(1 for _ in open(wpath(ws, "transcript.jsonl"), encoding="utf-8")) - 1
    log(f"\n=== done in {elapsed / 60:.1f} min, {iters} iterations, {n_calls} model calls ===")
    log(f"=== workspace: {ws} ===")
    return result


def main():
    global MODEL, ENGINE, WORKDIR, DSH_PATCH
    ap = argparse.ArgumentParser(description="GVS5H ledger self-orchestration against a local Strata server")
    ap.add_argument("goal", help="what the swarm should accomplish")
    ap.add_argument("--context", default="", help="extra context (paths, constraints, repo notes)")
    ap.add_argument("--iters", type=int, default=MAX_ITERS, help="max worker iterations")
    ap.add_argument("--verify", default=None, help="command a worker may run to verify its work")
    ap.add_argument("--no-thinking", action="store_true", help="disable extended thinking on workers")
    ap.add_argument("--engine", choices=["dsh", "direct"], default=ENGINE,
                    help="dsh: workers run through DeepSeek Harness with tools; direct: raw model calls")
    ap.add_argument("--workdir", default=WORKDIR, help="directory DSH works in")
    ap.add_argument("--patch", default=DSH_PATCH, help="DSH overlay YAML (model/provider routing)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    MODEL = args.model
    ENGINE = args.engine
    WORKDIR = args.workdir
    DSH_PATCH = args.patch
    run(args.goal, args.context, args.iters, args.verify, thinking=not args.no_thinking)


if __name__ == "__main__":
    main()