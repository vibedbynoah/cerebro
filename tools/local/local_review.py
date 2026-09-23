"""Free first-pass review on the local Gemma-4-E4B model ("starting rating").

Builders run this before handing work to the paid Opus critic, so obvious problems are fixed
without spending Claude usage. It is a filter, not a judge: a small model misses subtle bugs
and sometimes flags non-issues, so builders must verify every point before acting on it.

  Code:   python local_review.py --repo C:/path/to/worktree [--focus "what matters"]
  Visual: python local_review.py --images shot1.png shot2.png [--focus "screen and state"]

Prints one JSON object: {"score": 0-10, "issues": [...], "summary": "..."}.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_llm  # noqa: E402

MAX_DIFF_CHARS = 36_000  # ~10k tokens, inside the 12k context with room for the answer
SKIP_SUFFIXES = (".lock", ".json", ".png", ".jpg", ".svg", ".gguf", ".bin")

CODE_RUBRIC = """You are a strict senior reviewer giving a FIRST-PASS rating of a code change for Cerebro, a security-focused Windows desktop app (Rust core, llama.cpp backend spawned via argv only, TypeScript UI).
Rate 0-10. Report only concrete problems you can point to in the diff: compile or logic errors, missing error handling, unwrap/expect/panic outside tests, security issues (shell use, path or argument injection, unchecked input, secrets in code), untested behavior, tests that cannot fail, missing docs on public items, dead code.
Do not invent issues. If the diff looks correct, say so and score high.
Answer with JSON only: {"score": <0-10>, "issues": [{"severity": "critical|major|minor", "file": "...", "problem": "...", "fix": "..."}], "summary": "..."}"""

VISUAL_RUBRIC = """You are a strict product designer giving a FIRST-PASS rating of UI screenshots for Cerebro, a desktop app that must look as polished as Linear, Raycast, Claude desktop and Cursor.
Rate 0-10 on: alignment and spacing consistency, typography hierarchy, contrast and legibility (light and dark), visual noise, clipped or overlapping elements, empty or broken states, and whether it looks professional.
Report only problems you can see. Answer with JSON only: {"score": <0-10>, "issues": [{"severity": "critical|major|minor", "file": "<screenshot name>", "problem": "...", "fix": "..."}], "summary": "..."}"""


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def collect_diff(repo: Path) -> str:
    parts = [git(repo, "diff", "HEAD", "--", ".", ":(exclude)*.lock", ":(exclude)docs/research/*")]
    for rel in git(repo, "ls-files", "--others", "--exclude-standard").splitlines():
        if rel.startswith("docs/research/") or rel.endswith(SKIP_SUFFIXES):
            continue
        p = repo / rel
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parts.append(f"\n--- new file: {rel}\n{text}")
    diff = "".join(parts)
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + f"\n\n[TRUNCATED: diff is {len(diff)} chars; only the first {MAX_DIFF_CHARS} were reviewed]"
    return diff


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0) if m else text)
    except (json.JSONDecodeError, AttributeError):
        return {"score": None, "issues": [], "summary": "local model returned unparseable output", "raw": text[:2000]}
    data.setdefault("issues", [])
    data.setdefault("summary", "")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--focus", default="")
    args = ap.parse_args()
    focus = f"\nFOCUS: {args.focus}\n" if args.focus else ""
    if args.images:
        imgs = [Path(p) for p in args.images]
        names = ", ".join(p.name for p in imgs)
        out = local_llm.ask("gemma", f"{VISUAL_RUBRIC}{focus}\nScreenshots, in order: {names}", imgs, 1536, True)
    elif args.repo:
        diff = collect_diff(Path(args.repo))
        if not diff.strip():
            print(json.dumps({"score": None, "issues": [], "summary": "no changes to review"}))
            return 0
        out = local_llm.ask("gemma", f"{CODE_RUBRIC}{focus}\n\nCHANGE:\n{diff}", None, 1536, True)
    else:
        ap.error("give --repo or --images")
    result = parse(out)
    result["reviewer"] = "local gemma-4-E4B (first pass only; verify every point)"
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
