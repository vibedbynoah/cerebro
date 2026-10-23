"""Builds the head-to-head report from a sweep's raw results (tools/eval/sweep.sh).

For every model and placement: Cerebro's prompt-reading and writing speed next to llama.cpp's on
the same model, machine and prompt, and which is faster. llama.cpp gets its best thread count;
placements llama.cpp has no equivalent for are marked as such. Every number comes from the jsonl
files; nothing is estimated.

Usage: python -I report.py SWEEP_DIR > REPORT.md
"""
import json
import os
import sys
from collections import defaultdict

d = sys.argv[1]


def lines(name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p, encoding="utf-8"):
        l = l.strip()
        if l:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def model_key(name):
    return os.path.basename(str(name)).lower()


cb = defaultdict(dict)  # model -> label -> row
for r in lines("cerebro.jsonl"):
    cb[model_key(r.get("model", ""))][r["label"]] = r

ll = defaultdict(lambda: defaultdict(dict))  # model -> label -> {"pp": best, "tg": best, "t": threads}
for r in lines("llamacpp.jsonl"):
    m = model_key(r.get("model_filename", ""))
    lab = r.get("label", "")
    if r.get("failed"):
        ll[m][lab]["failed"] = True
        continue
    key = "pp" if r.get("n_prompt", 0) > 0 else "tg"
    # Same work as Cerebro's bench: prompts read from an empty context, writing after a 512-token
    # prompt (n_depth 512; sweeps from before depth runs only have depth 0).
    want = 0 if key == "pp" else 512
    depths = {x.get("n_depth", 0) for x in lines("llamacpp.jsonl") if model_key(x.get("model_filename", "")) == m and x.get("label") == lab and not x.get("failed") and (x.get("n_prompt", 0) > 0) == (key == "pp")}
    if r.get("n_depth", 0) != (want if want in depths else 0):
        continue
    v = r.get("avg_ts", 0.0)
    if v > ll[m][lab].get(key, 0):
        ll[m][lab][key] = v
        ll[m][lab][key + "_t"] = r.get("n_threads")
        ll[m][lab]["ngl"] = r.get("n_gpu_layers")

ORDER = ["cpu", "gpu", "ssd", "cpu+gpu", "cpu+ssd", "gpu+ssd", "cpu+gpu+ssd", "igpu", "igpu+cpu"]
NAMES = {
    "cpu": "Processor (RAM)",
    "gpu": "Graphics card",
    "ssd": "Drive (SSD)",
    "cpu+gpu": "Processor + graphics card",
    "cpu+ssd": "Processor + drive",
    "gpu+ssd": "Graphics card + drive",
    "cpu+gpu+ssd": "All three",
    "igpu": "Integrated GPU",
    "igpu+cpu": "Integrated GPU + processor",
}
# Cerebro labels map to llama.cpp labels where an equivalent exists.
LL_EQUIV = {"cpu": "cpu", "gpu": "gpu", "cpu+gpu": "cpu+gpu", "igpu": "igpu", "igpu+cpu": "igpu+cpu"}


def fmt(v, digits=1):
    return "-" if v is None else f"{v:,.{digits}f}"


def verdict(a, b):
    if a is None or b is None or b == 0:
        return ""
    r = a / b
    return f"**{r:.2f}x**" if r >= 1.0 else f"{r:.2f}x (slower)"


print("# Placement sweep: Cerebro vs llama.cpp\n")
print("Same machine, models and prompt (512 tokens read, then greedy writing). Speeds in tokens per second;")
print("the ratio is Cerebro / llama.cpp, above 1 = Cerebro faster. llama.cpp has no drive tier, so the")
print("drive rows have no comparison. `-` = not run or failed (see the raw files).\n")
wins = losses = 0
for m in sorted(cb, key=lambda k: min((r.get("weight_bytes_per_token") or 0) for r in cb[k].values()) if cb[k] else 0):
    rows = cb[m]
    print(f"## {m}\n")
    print("| Placement | GPU layers | Drive layers | Read: Cerebro | Read: llama.cpp | Read ratio | Write: Cerebro | Write: llama.cpp | Write ratio |")
    print("|---|---:|---:|---:|---:|---|---:|---:|---|")
    labels = [l for l in ORDER if l in rows] + sorted(l for l in rows if l not in ORDER)
    for lab in labels:
        r = rows[lab]
        if r.get("failed"):
            print(f"| {NAMES.get(lab, lab)} | | | failed: {str(r.get('why', ''))[:80]} | | | | | |")
            continue
        pp, tg = r.get("prefill_tps_median"), r.get("decode_tps_median")
        eq = LL_EQUIV.get(lab)
        lpp = ltg = None
        if eq:
            lr = ll.get(m, {}).get(eq) or next((v for k, v in ll.get(m, {}).items() if k.startswith(eq + " (")), None)
            if lr and not lr.get("failed"):
                lpp, ltg = lr.get("pp"), lr.get("tg")
        for a, b in ((pp, lpp), (tg, ltg)):
            if a is not None and b:
                if a >= b:
                    wins += 1
                else:
                    losses += 1
        print(f"| {NAMES.get(lab, lab)} | {r.get('gpu_layers', '')} | {r.get('drive_layers', '')} | {fmt(pp)} | {fmt(lpp)} | {verdict(pp, lpp)} | {fmt(tg, 2)} | {fmt(ltg, 2)} | {verdict(tg, ltg)} |")
    print()
print(f"**Head-to-head: Cerebro faster in {wins}, slower in {losses} of {wins + losses} comparisons.**")
