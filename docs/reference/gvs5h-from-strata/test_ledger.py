"""Ledger self-orchestration: parsing, workspace, sample-test feedback, and the full loop.

No GPU and no server: the calls go through an injected transport that replays scripted
replies, exactly the sequence a real run produces for a two-iteration problem. The live
transport's request shape is checked against a mocked urlopen.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve.ledger import (CODE_SPEC, MATH_SPEC, LedgerSolver, SolveStatus, Workspace,
                          bullets, extract_code, parse_tasks, sections)


class ParsingTest(unittest.TestCase):
    def test_sections_accept_all_three_header_styles(self):
        text = "### PLAN\ndo it\n**TASKS**\n- a\nSTATUS:\ndone"
        self.assertEqual(sections(text), {"PLAN": "do it", "TASKS": "- a", "STATUS": "done"})

    def test_sections_strips_think_blocks(self):
        text = "### NOTES\nreal"
        self.assertEqual(sections(text), {"NOTES": "real"})

    def test_bullets_and_task_status(self):
        tasks = parse_tasks("- [done] write code\n- [wip] test it\n- plain one")
        self.assertEqual([t["status"] for t in tasks], ["done", "in_progress", "pending"])
        self.assertEqual([t["id"] for t in tasks], [1, 2, 3])

    def test_bullets_ignore_prose_lines(self):
        self.assertEqual(bullets("not a bullet\n- yes"), ["yes"])

    def test_extract_code_prefers_python_fence(self):
        self.assertEqual(extract_code("text\n```python\nprint(1)\n```\n"), "print(1)")
        self.assertEqual(extract_code("no fence here"), "")


class WorkspaceTest(unittest.TestCase):
    def test_starts_empty_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp, "slug")
            for name in ("task.md", "plan.md", "notes.md", "tasks.json",
                         "solution.py", "answer.md", "transcript.jsonl"):
                self.assertEqual(ws.read(name), "")
                self.assertTrue(Path(ws.path(name)).exists())
            ws.write("solution.py", "x = 1")
            self.assertEqual(ws.read("solution.py"), "x = 1")
            ws.append_transcript({"a": 1})
            ws.append_transcript({"a": 2})
            with open(ws.path("transcript.jsonl"), encoding="utf-8") as f:
                lines = [json.loads(l) for l in f]
            self.assertEqual([r["a"] for r in lines], [1, 2])


class SampleTestTest(unittest.TestCase):
    ECHO = "import sys\nsys.stdout.write(sys.stdin.read().upper())"

    def _ws(self, tmp, code):
        ws = Workspace(tmp, "s")
        ws.write("solution.py", code)
        return ws

    def _test(self, testtype, inp, out):
        return SimpleNamespace(testtype=testtype, input=inp, output=out)

    def test_all_samples_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = LedgerSolver.run_samples(self._ws(tmp, self.ECHO),
                                           [self._test("stdin", "ab", "AB")])
            self.assertTrue(res["ran"])
            self.assertEqual((res["passed"], res["total"]), (1, 1))

    def test_failing_sample_reports_first_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = LedgerSolver.run_samples(self._ws(tmp, self.ECHO),
                                           [self._test("stdin", "ab", "AB"),
                                            self._test("stdin", "cd", "XX")])
            self.assertEqual((res["passed"], res["total"]), (1, 2))
            self.assertEqual(res["fail"]["expected"], "XX")
            self.assertEqual(res["fail"]["got"], "CD")

    def test_runtime_error_is_data_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = LedgerSolver.run_samples(self._ws(tmp, "raise SystemExit(1)\n"),
                                           [self._test("stdin", "", "anything")])
            self.assertEqual(res["passed"], 0)
            self.assertIn("runtime error", res["fail"]["got"])

    def test_timeout_counts_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = "import time\ntime.sleep(30)\n"
            with mock.patch("serve.ledger.SAMPLE_TIMEOUT_S", 1):
                res = LedgerSolver.run_samples(self._ws(tmp, code), [self._test("stdin", "", "")])
            self.assertEqual(res["passed"], 0)
            self.assertIn("timed out", res["fail"]["got"])

    def test_no_stdin_tests_means_no_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = LedgerSolver.run_samples(self._ws(tmp, self.ECHO), [self._test("functional", "", "")])
            self.assertEqual(res, {"ran": False})

    def test_feedback_strings(self):
        self.assertIn("PASSED all", LedgerSolver.sample_feedback({"ran": True, "passed": 2, "total": 2}))
        bad = LedgerSolver.sample_feedback({"ran": True, "passed": 0, "total": 2,
                                            "fail": {"input": "i", "expected": "e", "got": "g"}})
        self.assertIn("FAILED", bad)
        self.assertIn("DIFFERENT approach", bad)
        self.assertEqual(LedgerSolver.sample_feedback({"ran": False}), "")


class _Scripted:
    """The transport a real run would produce, in order, for one two-iteration problem."""

    def __init__(self):
        self.calls = []

    def __call__(self, messages, temperature, max_tokens):
        self.calls.append((messages, temperature))
        n = len(self.calls)
        if n == 1:      # primary_plan
            return "### PLAN\nbinary search\n### TASKS\n- [ ] write the solution\n- [ ] test it", {}
        if n == 2:      # ideation_worker
            return ("### NOTES\ncore difficulty is the tie-break\n"
                    "### NEXT\n- sort and scan\n- binary search"), {}
        if n == 3:      # primary_manage #1
            return "### STATUS\ncontinue\n### NEXT\nwrite the solution\n### TASKS\n- [todo] write the solution", {}
        if n == 4:      # worker 1: wrong answer on the samples
            return ("### CODE\n```python\nprint('nope')\n```\n"
                    "### NOTES\n- **Scan:** off-by-one on the last group\n"
                    "### NEXT\n- fix the group split\n### STATUS\ncontinue"), {}
        if n == 5:      # primary_manage #2: says done although the samples failed
            return "### STATUS\ndone\n### NEXT\n\n### TASKS\n- [done] write the solution", {}
        if n == 6:      # worker 2 after the override
            return ("### CODE\n```python\nprint('yes')\n```\n"
                    "### NOTES\n- **Fix:** the split needs the separator\n"
                    "### NEXT\n- none\n### STATUS\nsolved"), {}
        # primary_manage #3: the samples pass, done for real
        return "### STATUS\ndone\n### NEXT\n\n### TASKS\n- [done] write the solution", {}


class SolveLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.transport = _Scripted()
        self.solver = LedgerSolver(ws_root=self.tmp.name, transport=self.transport, log=lambda *a, **k: None)
        self.tests = [SimpleNamespace(testtype="stdin", input="x", output="x")]

    def test_full_loop(self):
        # solution.py must echo the sample for the feedback to pass on worker 2
        orig = LedgerSolver.run_samples

        def run_samples(ws, tests):
            code = ws.read("solution.py")
            if "yes" in code:
                return {"ran": True, "passed": 1, "total": 1}
            return orig(ws, tests)

        with mock.patch.object(LedgerSolver, "run_samples", staticmethod(run_samples)):
            st = SolveStatus()
            out = self.solver.solve("echo the input", CODE_SPEC, tests=self.tests, status_out=st)
        self.assertIn("print('yes')", out)
        self.assertEqual(st.iters, 2)
        self.assertEqual(st.n_calls, 7)
        self.assertEqual(st.truncated_calls, 0)
        self.assertTrue(st.finish_reason is None or st.finish_reason in ("stop", "length"))

        ws_dir = Path(st.ws)
        self.assertIn("binary search", (ws_dir / "plan.md").read_text())
        # notes are REPLACED by each worker, not appended: only the last rewrite survives
        self.assertIn("separator", (ws_dir / "notes.md").read_text())
        self.assertNotIn("core difficulty", (ws_dir / "notes.md").read_text())
        transcript = [json.loads(l) for l in open(ws_dir / "transcript.jsonl")]
        self.assertEqual([r["role"] for r in transcript],
                         ["primary_plan", "ideation", "primary_manage",
                          "worker:1", "primary_manage", "worker:2", "primary_manage"])
        self.assertIn("wall_s", transcript[0])

    def test_manager_sees_sample_feedback_as_ground_truth(self):
        with mock.patch.object(LedgerSolver, "run_samples",
                               return_value={"ran": True, "passed": 0, "total": 1,
                                             "fail": {"input": "x", "expected": "x", "got": "nope"}}):
            self.solver.solve("p", CODE_SPEC, tests=self.tests)
        manage_calls = [m for m, t in self.transport.calls if "manager" in m[0]["content"]]
        self.assertTrue(manage_calls)
        self.assertIn("SAMPLE TESTS: FAILED", manage_calls[-1][1]["content"])
        # the manager said 'done' while the samples failed: the loop must have overridden it
        self.assertGreater(len(self.transport.calls), 5)

    def test_repeated_same_task_stops_the_loop(self):
        stuck = _Scripted()

        def repeat(messages, temperature, max_tokens):
            stuck.calls.append((messages, temperature))
            n = len(stuck.calls)
            if n <= 3:
                return _Scripted.__call__(stuck, messages, temperature, max_tokens)
            return "### STATUS\ncontinue\n### NEXT\nwrite the solution\n### TASKS\n- [todo] write the solution", {}

        stuck.__call__ = repeat
        solver = LedgerSolver(ws_root=self.tmp.name, transport=repeat, max_iters=10,
                              log=lambda *a, **k: None)
        with mock.patch.object(LedgerSolver, "run_samples",
                               return_value={"ran": True, "passed": 1, "total": 1}):
            st = SolveStatus()
            solver.solve("p", CODE_SPEC, tests=self.tests, status_out=st)
        self.assertEqual(st.iters, 1)      # second identical task broke the loop

    def test_math_kind_uses_answer_artifact(self):
        solver = LedgerSolver(ws_root=self.tmp.name, transport=lambda m, t, mt: (
            "### ANSWER\nblah\nANSWER: 42\n### NOTES\n- n\n### NEXT\n- none\n### STATUS\nsolved", {}),
            log=lambda *a, **k: None)
        with mock.patch.object(LedgerSolver, "run_samples", return_value={"ran": False}):
            out = solver.solve("what is 6*7?", MATH_SPEC, tests=None)
        self.assertIn("ANSWER: 42", out)

    def test_transport_token_counts_land_in_status(self):
        counted = _Scripted()

        def with_usage(messages, temperature, max_tokens):
            text, extra = counted(messages, temperature, max_tokens)
            return text, {"completion_tokens": 11, "prompt_tokens": 22, "finish_reason": "stop"}

        solver = LedgerSolver(ws_root=self.tmp.name, transport=with_usage, log=lambda *a, **k: None)
        with mock.patch.object(LedgerSolver, "run_samples",
                               return_value={"ran": True, "passed": 1, "total": 1}):
            st = SolveStatus()
            solver.solve("p", CODE_SPEC, tests=self.tests, status_out=st)
        self.assertEqual(st.n_calls, 5)            # samples pass: no override, no finalize call
        self.assertEqual(st.completion_tokens, 5 * 11)
        self.assertEqual(st.prompt_tokens, 5 * 22)


class LiveTransportTest(unittest.TestCase):
    def test_request_shape(self):
        captured = {}

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "hi"},
                                                "finish_reason": "stop"}],
                                   "usage": {"completion_tokens": 3, "prompt_tokens": 5}}).encode()

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode())
            captured["timeout"] = timeout
            return FakeResp()

        solver = LedgerSolver(base_url="http://127.0.0.1:9999", model="strata",
                              ws_root=tempfile.mkdtemp(), log=lambda *a, **k: None)
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp, "s")
            with mock.patch("urllib.request.urlopen", fake_urlopen):
                text, meta = solver.chat("worker:1", ws,
                                         [{"role": "user", "content": "hello"}], 0.2)
            with open(ws.path("transcript.jsonl"), encoding="utf-8") as f:
                transcript = [json.loads(l) for l in f]
        self.assertEqual(text, "hi")
        self.assertEqual(captured["url"], "http://127.0.0.1:9999/v1/chat/completions")
        self.assertEqual(captured["body"]["model"], "strata")
        self.assertEqual(captured["body"]["messages"][0]["content"], "hello")
        self.assertEqual(captured["body"]["temperature"], 0.2)
        self.assertEqual(captured["body"]["reasoning_effort"], "none")
        self.assertEqual(meta["completion_tokens"], 3)
        self.assertEqual(captured["timeout"], 1800)
        self.assertEqual(len(transcript), 1)
        self.assertEqual(transcript[0]["role"], "worker:1")


if __name__ == "__main__":
    unittest.main()
