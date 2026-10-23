"""The stop condition of the improvement loop: Cerebro's engine ahead of llama.cpp on every metric in
two sweeps in a row.

Reads the two newest sweep folders under docs/measurements/engine (or the folders given), pairs
every (model, placement, metric) that both engines measured, and lists each one Cerebro did not win
in both sweeps. Exit code 0 only when there is none.

Usage: python -I verdict.py [SWEEP_DIR_OLDER SWEEP_DIR_NEWER]
"""
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "measurements", "engine")
# Cerebro placement label -> llama.cpp label (the placements llama.cpp also has).
EQUIV = {"cpu": "cpu", "gpu": "gpu", "cpu+gpu": "cpu+gpu", "igpu": "igpu", "igpu+cpu": "igpu+cpu"}


def rows(path):
    out = []
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    out.append(json.loads(l))
                except json.JSONDecodeError:
                    pass
    return out


def scores(d):
    """{(model, placement, metric): (cerebro, llama)} for one sweep."""
    cb = {}
    for r in rows(os.path.join(d, "cerebro.jsonl")):
        if r.get("failed"):
            continue
        m = os.path.basename(r.get("model", "")).lower()
        cb[(m, r["label"], "read")] = r.get("prefill_tps_median")
        cb[(m, r["label"], "write")] = r.get("decode_tps_median")
    ll = {}
    lrows = [r for r in rows(os.path.join(d, "llamacpp.jsonl")) if not r.get("failed")]
    # Same work as Cerebro's bench: reading from an empty context, writing after a 512-token prompt
    # (n_depth 512 where the sweep ran it; older sweeps only have depth 0).
    has_depth = any(r.get("n_depth", 0) == 512 for r in lrows)
    for r in lrows:
        m = os.path.basename(str(r.get("model_filename", ""))).lower()
        lab = r.get("label", "").split(" (")[0]
        metric = "read" if r.get("n_prompt", 0) > 0 else "write"
        if r.get("n_depth", 0) != (512 if metric == "write" and has_depth else 0):
            continue
        key = (m, lab, metric)
        ll[key] = max(ll.get(key, 0.0), r.get("avg_ts", 0.0))
    out = {}
    for (m, lab, metric), v in cb.items():
        eq = EQUIV.get(lab)
        if eq and (m, eq, metric) in ll and v is not None:
            out[(m, lab, metric)] = (v, ll[(m, eq, metric)])
    return out


def main():
    dirs = sys.argv[1:3] if len(sys.argv) >= 3 else sorted(glob.glob(os.path.join(ROOT, "*sweep*")), key=os.path.getmtime)[-2:]
    if len(dirs) < 2:
        print("Need two sweeps.")
        return 2
    a, b = scores(dirs[0]), scores(dirs[1])
    keys = sorted(set(a) | set(b))
    lost = []
    for k in keys:
        ra, rb = a.get(k), b.get(k)
        ok = all(r is not None and r[0] > r[1] for r in (ra, rb))
        if not ok:
            lost.append((k, ra, rb))
    print(f"Sweeps: {os.path.basename(dirs[0])} then {os.path.basename(dirs[1])}")
    print(f"{len(keys) - len(lost)} of {len(keys)} metrics won in both.")
    for (m, lab, metric), ra, rb in lost:
        f = lambda r: "missing" if r is None else f"{r[0]:.1f} vs {r[1]:.1f}"
        print(f"  not yet: {m} {lab} {metric}: {f(ra)}; {f(rb)}")
    return 0 if not lost else 1


sys.exit(main())
