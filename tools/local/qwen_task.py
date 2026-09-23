"""Delegate a small, low-stakes coding chore to the local Qwen3-Coder-30B MoE (free).

For boilerplate only: test fixtures, repetitive test tables, doc tables, data files, simple
glue. The caller owns the result: read every line, fix it, and run the tests. Never use this for
security-sensitive code, unsafe blocks, or anything the brief calls load-bearing.

  python qwen_task.py --task "write X" [--context FILE ...] --out FILE

Writes the model's answer (code fences stripped) to --out and prints the path.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_llm  # noqa: E402

MAX_CONTEXT_CHARS = 40_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--context", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    ctx = []
    for c in args.context:
        p = Path(c)
        ctx.append(f"--- {p.name}\n{p.read_text(encoding='utf-8', errors='replace')}")
    context = "\n\n".join(ctx)[:MAX_CONTEXT_CHARS]
    prompt = (f"You are a careful programmer. Do exactly this task and output ONLY the file content, "
              f"with no explanation.\nTASK: {args.task}\n\nCONTEXT FILES:\n{context}")
    text = local_llm.ask("qwen", prompt, None, args.max_tokens, False, 0.2)
    text = re.sub(r"^\s*```[\w-]*\n|\n```\s*$", "", text.strip())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
