"""Rows for the "Cerebro vs llama.cpp" dashboard: every (model, placement, measure) both engines ran,
from the newest clean sweep that has it, plus quiet re-tests named on the command line.

Usage: python -I dash_rows.py OUT.json SWEEP_DIR [SWEEP_DIR ...]   (newest first)
Writing pairs llama.cpp at depth 512 where the sweep ran it (the work Cerebro's bench does); older
sweeps only have depth 0, and those rows say so.
"""
import json
import os
import sys

PLACE = {"cpu": "Processor", "gpu": "Graphics card", "cpu+gpu": "Card + processor (split)", "igpu": "Integrated GPU", "igpu+cpu": "Integrated GPU + processor"}


def name(f):
    b = os.path.basename(f).replace(".gguf", "")
    return b.replace("-Q", " Q").replace("Qwen3-", "Qwen3 ").replace("Qwen3.8-", "Qwen3.8 ").replace("-", " ")


def load(d):
    run = os.path.basename(d.rstrip("/\\"))
    cb, ll, depth512 = {}, {}, False
    why = {}
    for l in open(os.path.join(d, "cerebro.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        if r["label"] not in PLACE:
            continue
        if r.get("failed"):
            # A run Cerebro did not finish counts as a loss where llama.cpp has a number.
            for met in ("Prompt reading", "Writing"):
                cb.setdefault((r["model"], r["label"], met), None)
                why[(r["model"], r["label"], met)] = r.get("why", "did not run")[:120]
            continue
        cb[(r["model"], r["label"], "Prompt reading")] = r["prefill_tps_median"]
        cb[(r["model"], r["label"], "Writing")] = r["decode_tps_median"]
    rows = [json.loads(l) for l in open(os.path.join(d, "llamacpp.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if not r.get("failed")]
    depth512 = any(r.get("n_depth", 0) == 512 for r in rows)
    for r in rows:
        m = os.path.basename(str(r["model_filename"]))
        lab = r["label"].split(" (")[0]
        met = "Prompt reading" if r["n_prompt"] else "Writing"
        want = 512 if (met == "Writing" and depth512) else 0
        if r.get("n_depth", 0) != want:
            continue
        k = (m, lab, met)
        ll[k] = max(ll.get(k, 0), r["avg_ts"])
    out = {}
    for k, c in cb.items():
        if k in ll:
            m, lab, met = k
            note = "" if met == "Prompt reading" or depth512 else "llama.cpp wrote from an empty context, Cerebro after the 512-token prompt"
            if c is None:
                out[k] = {"model": name(m), "placement": PLACE[lab], "metric": met, "cerebro": None, "llama": round(ll[k], 1), "ratio": None, "run": run, "note": "Cerebro did not run: " + why[k]}
            else:
                out[k] = {"model": name(m), "placement": PLACE[lab], "metric": met, "cerebro": round(c, 1), "llama": round(ll[k], 1), "ratio": round(c / ll[k], 3), "run": run, "note": note}
    return out


def main():
    dst, dirs = sys.argv[1], sys.argv[2:]
    merged = {}
    for d in dirs:
        for k, v in load(d).items():
            merged.setdefault(k, v)
    rows = []
    for v in merged.values():
        v["id"] = f'{v["model"]}|{v["placement"]}|{v["metric"]}'
        rows.append(v)
    json.dump(rows, open(dst, "w"), indent=1)
    print(f"{len(rows)} rows, {sum((r['ratio'] or 0) > 1 for r in rows)} won")


main()
