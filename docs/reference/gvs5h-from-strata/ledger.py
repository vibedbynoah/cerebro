"""Ledger-based self-orchestration: one model, many fresh instances, one shared workspace.

Every call below starts a fresh conversation. Nothing is carried in a message history; the
team's state lives on disk in the workspace, and each role reads only what it needs:

    <ws>/task.md      the problem statement
    <ws>/plan.md      the manager's overarching plan
    <ws>/notes.md     the shared findings file, rewritten by every worker
    <ws>/tasks.json   the curated task list the manager owns
    <ws>/solution.py  the current code artifact (kind=code) or answer.md (kind=math)
    <ws>/transcript.jsonl   every call: request, reply, tokens, wall time, finish reason

The loop is PRIMARY (manager) and WORKER roles over one model: plan, propose approaches,
then repeatedly pick one task, run a worker that writes the full artifact and folds its
findings into notes.md, run the public sample tests when code is involved, and let the
manager decide done or continue -- including forcing a switch to a different approach when
the current one keeps failing the samples.

Ported from the GVS5H harness (Persis Capital Inc., Toronto; MIT-licensed,
github.com/slee-persis/GVS5H, codebase/v2-current/escalation/multiagent.py), which measured
up to +25.6 points pass@1 on LiveCodeBench hard by coordinating five fresh instances of one
model this way. Measured on this box: Strata decodes one request at a time unless the model
config sets "parallel": N, so the loop is deliberately sequential; coder-IQ1_M runs ~7.4
tok/s at 32k context and IQ3_XXS ~2.2 tok/s at 128k, so prompt sizes are capped below.

Standard library only. Run a problem standalone:

    python -m serve.ledger --problem-file task.md --kind code
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass

MAX_PLAN_CHARS = 4000          # plan.md is read by every role: keep it small
MAX_NOTES_CHARS = 8000         # notes.md is rewritten, not appended: the whole file is read
MAX_ARTIFACT_CHARS = 12000     # the current solution is quoted into every manager prompt
MAX_TASKS = 12                 # live task list cap, as in the source harness
SAMPLE_TIMEOUT_S = 10          # one public sample test gets this long, then it counts as failed

CODE_SPEC = {
    "kind": "code",
    "solver_system": (
        "You are an elite competitive programmer. Solve the given problem in Python. "
        "Think carefully about algorithmic complexity and edge cases. "
        "Output EXACTLY ONE complete, self-contained Python program inside a single "
        "```python ...``` fenced block, and nothing else after it."
    ),
}

MATH_SPEC = {
    "kind": "math",
    "solver_system": (
        "You are an elite mathematician. Solve the problem. Reason step by step, then on "
        "the FINAL line output exactly 'ANSWER: <final answer>'."
    ),
}

_FORMAT_RULE = (
    "\n\nTHE FORMAT IS MANDATORY AND YOUR REPLY IS PARSED BY A PROGRAM. "
    "Begin your reply with the literal line '### {first}', then the literal line "
    "'### {second}'{rest}. Write nothing before the first header and nothing outside "
    "these sections. Do NOT solve the problem and do NOT write code -- a later worker "
    "does that. A reply without these exact header lines is discarded."
)


def _cap(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n...[truncated to fit the context]..."


# --- section and list parsing -------------------------------------------------


def _strip_think(text: str) -> str:
    return re.sub(r".*?</think>", "", text or "", flags=re.DOTALL).strip()


def _strip_code(text: str) -> str:
    return re.sub(r"```.*?```", "[code omitted -- approach only]", text or "", flags=re.DOTALL).strip()


def sections(text: str) -> dict[str, str]:
    """Split a reply on header lines. '### NOTES', '**NOTES**' and 'NOTES:' all count."""
    out, cur, buf = {}, None, []
    for line in _strip_think(text).splitlines():
        s = line.strip()
        m = (re.match(r"^#{1,3}\s*([A-Za-z_]+)\s*$", s)
             or re.match(r"^\*\*([A-Za-z_]+)\*\*\s*:?\s*$", s)
             or re.match(r"^([A-Z_]{3,})\s*:\s*$", s))
        if m:
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1).upper(), []
        else:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


def bullets(text: str) -> list[str]:
    items = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.*\S)", line)
        if m:
            items.append(m.group(1).strip())
    return items


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", _strip_think(text), re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_tasks(text: str) -> list[dict]:
    """'- [done] x' / '- [wip] y' / bare '- z' bullets become the task list."""
    out = []
    for b in bullets(text):
        m = re.match(r"\[\s*([a-z_ ]+?)\s*\]\s*(.*)", b, re.I)
        if m:
            raw = m.group(1).lower().replace(" ", "").replace("_", "")
            desc = m.group(2).strip()
            status = "done" if raw == "done" else ("in_progress" if raw in ("wip", "inprogress") else "pending")
        else:
            desc, status = b, "pending"
        if desc:
            out.append({"id": len(out) + 1, "desc": desc, "status": status, "result": ""})
    return out[:MAX_TASKS]


_ANS_RE = re.compile(r"ANSWER:\s*\S", re.I)


# --- the workspace ------------------------------------------------------------


class Workspace:
    """The ledger: the files the team coordinates through, and the call transcript."""

    def __init__(self, root: str, slug: str):
        self.dir = os.path.join(root, slug)
        os.makedirs(self.dir, exist_ok=True)
        for name in ("task.md", "plan.md", "notes.md", "tasks.json",
                     "solution.py", "answer.md", "transcript.jsonl"):
            self.write(name, "")

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def read(self, name: str) -> str:
        try:
            with open(self.path(name), encoding="utf-8", errors="replace") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def write(self, name: str, content: str) -> None:
        with open(self.path(name), "w", encoding="utf-8") as f:
            f.write(content)

    def append_transcript(self, rec: dict) -> None:
        with open(self.path("transcript.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


# --- the solver ---------------------------------------------------------------


@dataclass
class SolveStatus:
    """What one solve cost and how it ended; the measurement half of the mechanism."""

    n_calls: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    wall_s: float = 0.0
    truncated_calls: int = 0
    finish_reason: str | None = None
    ws: str = ""
    iters: int = 0


class LedgerSolver:
    """Manager + workers over one model, coordinating only through the workspace."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", model: str = "strata",
                 max_iters: int = 10, max_tokens: int = 2048, timeout: int = 1800,
                 ws_root: str = "runs/ws", log=print, transport=None,
                 strict_format: bool | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_iters = max_iters
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.ws_root = ws_root
        self.log = log or (lambda *a, **k: None)
        self.transport = transport      # (messages, temperature, max_tokens) -> (text, meta); tests inject one
        self.strict_format = strict_format

    # --- one call -----------------------------------------------------------

    def chat(self, role: str, ws: Workspace, messages: list[dict], temperature: float) -> tuple[str, dict]:
        meta: dict = {"role": role}
        t0 = time.perf_counter()
        if self.transport is not None:
            text, extra = self.transport(messages, temperature, self.max_tokens)
            meta.update(extra or {})
        else:
            text = self._post(messages, temperature, meta)
        meta["wall_s"] = round(time.perf_counter() - t0, 2)
        ws.append_transcript({"t": time.time(), "role": role, "request": messages,
                              "response": text, **meta})
        return text, meta

    def _post(self, messages: list[dict], temperature: float, meta: dict) -> str:
        body = {"model": self.model, "messages": messages, "stream": False,
                "temperature": temperature, "max_tokens": self.max_tokens,
                "reasoning_effort": "none"}
        req = urllib.request.Request(self.base_url + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.load(r)
        choice = (out.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or msg.get("reasoning_content") or ""
        usage = out.get("usage") or {}
        meta["finish_reason"] = choice.get("finish_reason")
        meta["completion_tokens"] = usage.get("completion_tokens")
        meta["prompt_tokens"] = usage.get("prompt_tokens")
        return text

    # --- roles ---------------------------------------------------------------

    def _fmt_rule(self, *headers: str) -> str:
        if not self.strict_format:
            return ""
        first, second = headers[0], headers[1] if len(headers) > 1 else headers[0]
        rest = "".join(f", then the literal line '### {h}'" for h in headers[2:])
        return _FORMAT_RULE.format(first=first, second=second, rest=rest)

    def primary_plan(self, ws: Workspace, problem: str, kind: str) -> list[dict]:
        sys = ("You are the PRIMARY orchestrator (manager) of a small team of workers, all "
               f"expert at {'competitive programming' if kind == 'code' else 'olympiad mathematics'}. "
               "Given a problem, produce a short overarching plan to solve it, then a task list "
               "the workers can pick up. Respond with EXACTLY these sections:\n"
               "### PLAN\n<3-6 sentence strategy>\n"
               "### TASKS\n<3-6 bullet tasks, each a concrete unit of work>"
               + self._fmt_rule("PLAN", "TASKS"))
        reply, _ = self.chat("primary_plan", ws, [{"role": "system", "content": sys},
                                                  {"role": "user", "content": problem}], 0.3)
        sec = sections(reply)
        ws.write("plan.md", _cap(sec.get("PLAN") or reply.strip(), MAX_PLAN_CHARS))
        tasks = []
        have = set()
        for d in bullets(sec.get("TASKS", "")):
            d = d.strip()
            if d and d.lower() not in have and len(tasks) < MAX_TASKS:
                tasks.append({"id": len(tasks) + 1, "desc": d, "status": "pending", "result": ""})
                have.add(d.lower())
        self.log(f"    [primary] plan + {len(tasks)} initial tasks")
        return tasks

    def ideation_worker(self, ws: Workspace, problem: str) -> list[str]:
        sys = ("You are the FIRST WORKER. Do NOT solve the problem and do NOT write any code. "
               "Just think about it: identify the core difficulty, then list SEVERAL DISTINCT "
               "candidate approaches (genuinely different algorithms / data structures / problem "
               "reductions, not variations of one idea), and note pitfalls for each. Describe each "
               "approach in prose only -- absolutely no code blocks; a later worker will implement. "
               "Respond with EXACTLY:\n### NOTES\n<your analysis>\n"
               "### NEXT\n<bullet list of distinct approaches to try next>"
               + self._fmt_rule("NOTES", "NEXT"))
        reply, _ = self.chat("ideation", ws, [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"PROBLEM:\n{problem}\n\nPLAN:\n{ws.read('plan.md')}"}], 0.4)
        sec = sections(reply)
        ws.write("notes.md", _cap(
            (ws.read("notes.md") + f"\n## ideation\n{_strip_code(sec.get('NOTES') or reply.strip())}\n").strip(),
            MAX_NOTES_CHARS))
        proposals = [_strip_code(p) for p in bullets(sec.get("NEXT", ""))]
        self.log(f"    [worker:ideate] proposed {len(proposals)} approaches")
        return proposals

    def primary_manage(self, ws: Workspace, problem: str, kind: str, tasks: list[dict],
                       proposals: list[str], last_summary: str) -> tuple[str, str, list[dict]]:
        artifact = "solution.py" if kind == "code" else "answer.md"
        cur = ws.read(artifact).strip()
        task_lines = "\n".join(f"- [{'done' if t['status'] == 'done' else 'todo'}] {t['desc']}"
                               for t in tasks) or "(none)"
        prop_lines = "\n".join(f"- {p}" for p in proposals) or "(none)"
        sys = ("You are the PRIMARY orchestrator and manager. You OWN the task list and decide "
               "when the problem is solved. Review the current progress and the latest worker's "
               "result, then:\n"
               "- The LATEST WORKER RESULT may include a SAMPLE TESTS verdict from actually running "
               "the code. Treat it as ground truth: only set STATUS 'done' if the solution PASSED the "
               "sample tests; if it FAILED, you MUST set STATUS 'continue' and choose a task that "
               "fixes the failing case or switches to a different approach.\n"
               "- If the current solution/answer is complete and correct, set STATUS to 'done'.\n"
               "- Otherwise CURATE the task list: merge duplicates, drop finished or irrelevant "
               "items, mark completed ones [done], and fold in ONLY genuinely new sub-tasks from "
               "the proposals. Then choose the single most valuable next task.\n"
               "- IMPORTANT: if the current solution keeps failing, or the last worker made no real "
               "progress, do NOT keep refining the same idea. Switch to a DIFFERENT approach "
               "(a different algorithm / data structure / reduction) from the notes, or ask for a "
               "new one. You have many rounds -- use them to try distinct approaches, not to polish "
               "a stuck one.\n"
               "Respond with EXACTLY these sections:\n"
               "### STATUS\n<done|continue>\n"
               "### NEXT\n<exact text of the ONE task to do next; omit if done>\n"
               "### TASKS\n<curated list, one per line, each '- [done] ...' or '- [todo] ...'>"
               + self._fmt_rule("STATUS", "NEXT", "TASKS"))
        reply, _ = self.chat("primary_manage", ws, [
            {"role": "system", "content": sys},
            {"role": "user", "content": (
                f"PROBLEM:\n{problem}\n\nCURRENT {'SOLUTION' if kind == 'code' else 'ANSWER'}:\n"
                f"{_cap(cur, MAX_ARTIFACT_CHARS) or '(none yet)'}\n\nNOTES:\n{_cap(ws.read('notes.md'), MAX_NOTES_CHARS)}\n\n"
                f"CURRENT TASK LIST:\n{task_lines}\n\nLATEST WORKER RESULT: {last_summary}\n\n"
                f"PROPOSED NEW STEPS:\n{prop_lines}")}], 0.2)
        sec = sections(reply)
        curated = parse_tasks(sec.get("TASKS", "")) or tasks
        status = "done" if sec.get("STATUS", "").strip().lower().startswith("done") else "continue"
        if status == "done" and not cur:
            status = "continue"
        nxt = (bullets(sec.get("NEXT", ""))[:1] or [sec.get("NEXT", "").strip()])[0]
        nxt = re.sub(r"^[-*]\s*", "", nxt).strip()
        if status == "continue" and not nxt:
            todo = [t for t in curated if t["status"] != "done"]
            nxt = todo[0]["desc"] if todo else ""
        if not cur and not nxt:
            status, nxt = "continue", "Implement the full working solution for the most promising approach in the notes."
        self.log(f"    [primary-manage] {status}" + (f", next: {nxt[:60]}" if status == "continue" else ""))
        return status, nxt, curated

    def worker(self, ws: Workspace, problem: str, kind: str, task: dict,
               finalize: bool = False) -> tuple[str, list[str], str, bool]:
        artifact = "solution.py" if kind == "code" else "answer.md"
        cur = _cap(ws.read(artifact), MAX_ARTIFACT_CHARS)
        solver_system = (CODE_SPEC if kind == "code" else MATH_SPEC)["solver_system"]
        notes_fmt = (
            "### NOTES\n<the COMPLETE notes file, rewritten. You are shown the current NOTES "
            "above: fold your findings into them, keep what still matters, and DELETE anything "
            "superseded, disproven, or now obvious. This REPLACES the file, so whatever you omit "
            "is gone. Organise it as '- **Topic:** ...' bullets, under ~800 words. Do NOT use "
            "markdown headings (#, ##), bold-only lines, or ALLCAPS: lines anywhere inside this "
            "section -- the reply is split on those, so they would truncate your notes.>\n")
        out_fmt = ((
            "### CODE\n```python\n<the FULL updated self-contained program>\n```\n" + notes_fmt +
            "### NEXT\n<bullet list of remaining steps, or 'none'>\n### STATUS\n<solved|continue>")
            if kind == "code" else
            ("### ANSWER\n<full worked solution, ending with the exact final "
             "ANSWER: line your instructions require>\n" + notes_fmt +
             "### NEXT\n<bullet list of remaining checks, or 'none'>\n### STATUS\n<solved|continue>"))
        goal = ("Produce the DEFINITIVE final solution now, using all notes and current work."
                if finalize else f"Complete this task: {task['desc']}")
        sys = (f"You are a WORKER subagent, {solver_system} "
               "You share a workspace with the team. Build on the current work and notes where "
               "useful -- but if your task is to try a different approach, write a FRESH solution "
               "for that approach instead of patching the stuck one. "
               f"Respond with EXACTLY these sections:\n{out_fmt}")
        reply, meta = self.chat("finalize" if finalize else f"worker:{task['id']}", ws, [
            {"role": "system", "content": sys},
            {"role": "user", "content": (
                f"PROBLEM:\n{problem}\n\nPLAN:\n{ws.read('plan.md')}\n\n"
                f"NOTES:\n{ws.read('notes.md')}\n\nCURRENT WORK:\n{cur or '(none yet)'}\n\n"
                f"YOUR TASK: {goal}")}], 0.2)
        sec = sections(reply)
        wrote = False
        if kind == "code":
            code = extract_code(sec.get("CODE", "")) or extract_code(reply)
            if code:
                ws.write("solution.py", code)
                wrote = True
        else:
            ans = sec.get("ANSWER", "").strip()
            if not ans and len(reply) < 20000:
                ans = _strip_think(reply).strip()
            if ans and (_ANS_RE.search(ans) or not ws.read("answer.md").strip()):
                ws.write("answer.md", ans)
                wrote = True
        if sec.get("NOTES") and meta.get("finish_reason") != "length":
            ws.write("notes.md", _cap(sec["NOTES"].strip(), MAX_NOTES_CHARS) + "\n")
        nexts = [b for b in bullets(sec.get("NEXT", "")) if b.lower() != "none"]
        status = sec.get("STATUS", "continue").strip().lower()
        if not finalize:
            self.log(f"    [worker] task {task['id']} -> {status}, +{len(nexts)} next steps")
        return status, nexts, f"worker reported '{status}' on: {task['desc'][:80]}.", wrote

    # --- public sample tests (code only): the manager's ground truth ---------

    @staticmethod
    def run_samples(ws: Workspace, tests) -> dict:
        """Run the problem's public stdin/stdout samples against solution.py."""
        stdin_tests = [t for t in (tests or []) if "stdin" in str(getattr(t, "testtype", "")).lower()]
        code = ws.read("solution.py")
        if not code.strip() or not stdin_tests:
            return {"ran": False}
        sol = ws.path("solution.py")
        passed, fail = 0, None
        for t in stdin_tests:
            inp = getattr(t, "input", "") or ""
            exp = (getattr(t, "output", "") or "").strip()
            try:
                r = subprocess.run([sys.executable, sol], input=inp, capture_output=True,
                                   text=True, timeout=SAMPLE_TIMEOUT_S)
                got = (r.stdout or "").strip()
                if r.returncode != 0 and not got:
                    got = f"<runtime error: {(r.stderr or '')[:200]}>"
            except subprocess.TimeoutExpired:
                got = f"<timed out (>{SAMPLE_TIMEOUT_S}s)>"
            except Exception as e:  # noqa: BLE001 - a broken sample is data, not a crash
                got = f"<error: {e}>"
            if got == exp:
                passed += 1
            elif fail is None:
                fail = {"input": inp[:600], "expected": exp[:400], "got": got[:400]}
        return {"ran": True, "passed": passed, "total": len(stdin_tests), "fail": fail}

    @staticmethod
    def sample_feedback(res: dict) -> str:
        if not res or not res.get("ran"):
            return ""
        if res["passed"] == res["total"]:
            return f"[SAMPLE TESTS: PASSED all {res['total']} public samples -- the solution looks correct.] "
        f = res.get("fail") or {}
        return ("[SAMPLE TESTS: FAILED -- passed {p}/{t}. The current solution is WRONG. First failing "
                "case: input={i!r} expected={e!r} got={g!r}. Fix the bug or, if this approach keeps "
                "failing, switch to a DIFFERENT approach.] ").format(
                    p=res["passed"], t=res["total"], i=f.get("input", ""),
                    e=f.get("expected", ""), g=f.get("got", ""))

    # --- the loop -------------------------------------------------------------

    def solve(self, problem: str, spec: dict = CODE_SPEC, tests=None,
              status_out: SolveStatus | None = None) -> str:
        kind = spec["kind"]
        st = status_out or SolveStatus()
        ws = Workspace(self.ws_root, hashlib.md5(problem.encode()).hexdigest()[:12])
        st.ws = ws.dir
        ws.write("task.md", problem)

        tasks = self.primary_plan(ws, problem, kind)
        proposals = self.ideation_worker(ws, problem)
        status, next_desc, tasks = self.primary_manage(ws, problem, kind, tasks, proposals,
                                                       "ideation complete")
        prev_desc, iters = None, 0
        while status == "continue" and next_desc and iters < self.max_iters:
            if prev_desc is not None and next_desc.strip().lower() == prev_desc.strip().lower():
                self.log(f"    [primary] reissued the same task; no progress, stopping after {iters} iters")
                break
            prev_desc, iters = next_desc, iters + 1
            task = {"id": iters, "desc": next_desc, "status": "in_progress", "result": ""}
            _, nexts, summary, wrote = self.worker(ws, problem, kind, task)
            res = self.run_samples(ws, tests) if (kind == "code" and tests and wrote) else None
            if res and res.get("ran"):
                summary = self.sample_feedback(res) + summary
                self.log(f"    [samples] {res['passed']}/{res['total']} public tests passed")
            status, next_desc, tasks = self.primary_manage(ws, problem, kind, tasks, nexts, summary)
            if res and res.get("ran") and res["passed"] < res["total"] and status == "done":
                status = "continue"
                if not next_desc:
                    next_desc = "The solution fails the public sample tests; fix it or try a different approach."
                self.log("    [primary-manage] overriding 'done' -- sample tests still failing")
            ws.write("tasks.json", json.dumps(tasks, indent=2))

        st.iters = iters
        artifact = "solution.py" if kind == "code" else "answer.md"
        if not (status == "done" and ws.read(artifact).strip()):
            self.worker(ws, problem, kind, {"id": 0, "desc": "finalize"}, finalize=True)
        with open(ws.path("transcript.jsonl"), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                st.n_calls += 1
                st.completion_tokens += rec.get("completion_tokens") or 0
                st.prompt_tokens += rec.get("prompt_tokens") or 0
                st.wall_s += rec.get("wall_s") or 0.0
                if rec.get("finish_reason") == "length":
                    st.truncated_calls += 1
                if rec.get("finish_reason"):
                    st.finish_reason = rec["finish_reason"]
        self.log(f"    [ledger] {st.n_calls} calls, {st.completion_tokens} completion tokens, "
                 f"{st.wall_s:.0f}s wall, {st.iters} worker iterations")
        if kind == "code" and ws.read("solution.py").strip():
            return f"```python\n{ws.read('solution.py')}\n```"
        return ws.read(artifact)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem-file", required=True, help="file holding the problem statement")
    ap.add_argument("--kind", choices=["code", "math"], default="code")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--iters", type=int, default=10, help="worker iterations per problem")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--ws-root", default="runs/ws")
    a = ap.parse_args()
    with open(a.problem_file, encoding="utf-8") as f:
        problem = f.read()
    solver = LedgerSolver(base_url=a.base_url, model=a.model, max_iters=a.iters,
                          max_tokens=a.max_tokens, timeout=a.timeout, ws_root=a.ws_root)
    status = SolveStatus()
    answer = solver.solve(problem, CODE_SPEC if a.kind == "code" else MATH_SPEC, status_out=status)
    print(f"\nworkspace: {status.ws}")
    return 0 if answer.strip() else 1


if __name__ == "__main__":
    raise SystemExit(main())
