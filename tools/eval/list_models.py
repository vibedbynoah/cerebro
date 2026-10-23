"""Text models for the sweep, one per line as "<bytes>\t<path>", smallest first.

Default: every GGUF in the folders the app scans (one entry per model, the first part of a split
model, no projectors / draft heads / vocab / imatrix files, identical copies once), between MIN_GB
and MAX_GB. With --paths "a;b;c" exactly those files, in that order.
Usage: python -I list_models.py MAX_GB MIN_GB [--paths "p1;p2"]
"""
import os
import re
import sys

SPLIT = re.compile(r"-00001-of-(\d{5})\.gguf$")


def size_of(p):
    m = SPLIT.search(p)
    if not m:
        return os.path.getsize(p)
    n = int(m.group(1))
    base = p[: -len("00001-of-%s.gguf" % m.group(1))]
    return sum(os.path.getsize(base + "%05d-of-%s.gguf" % (i, m.group(1))) for i in range(1, n + 1) if os.path.exists(base + "%05d-of-%s.gguf" % (i, m.group(1))))


def main():
    max_gb, min_gb = float(sys.argv[1]), float(sys.argv[2])
    if "--paths" in sys.argv:
        paths = [p for p in sys.argv[sys.argv.index("--paths") + 1].split(";") if p]
        for p in paths:
            print(f"{size_of(p)}\t{p}")
        return
    home, local = os.environ["USERPROFILE"], os.environ["LOCALAPPDATA"]
    roots = [os.path.join(local, "Cerebro", "models"), os.path.join(home, ".lmstudio", "models"), os.path.join(home, ".cache", "lm-studio", "models"), r"C:\models"]
    seen, out = set(), []
    for r in roots:
        for d, _, fs in os.walk(r):
            for f in fs:
                if not f.endswith(".gguf") or re.search(r"mmproj|^mtp|vocab|imatrix|-0000[2-9]-of-", f, re.I):
                    continue
                p = os.path.join(d, f)
                size = size_of(p)
                if size / 1e9 > max_gb or size / 1e9 < min_gb or (f, size) in seen:
                    continue
                seen.add((f, size))
                out.append((size, p))
    for s, p in sorted(out):
        print(f"{s}\t{p}")


main()
