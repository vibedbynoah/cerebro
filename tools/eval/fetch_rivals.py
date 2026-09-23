"""Download the competitor harnesses and the MoE model named in COMPETITORS.md.

Owner approved installs on 2026-09-17. Everything lands under C:\\bench (software) and
C:\\models (weights), so nothing touches the Cerebro repository and cleanup is a folder
delete. Downloads resume (curl -C -), so re-running continues an interrupted fetch and
skips what is already complete.

Portable payloads are unpacked here; GUI installers are only downloaded, and are listed at
the end for a separate, deliberate install step.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BENCH = Path(r"C:\bench")
MODELS = Path(r"C:\models")
LOG = Path(r"C:\Users\noahm\Projects\marrow\docs\research\2026-09\rivals-fetch.log")
UA = {"User-Agent": "cerebro-eval-fetch"}

# repo -> (asset name regex, destination folder, unpack?, tag regex or None)
# llama.cpp's "latest" release is a versioned tag (v0.4.1); the CUDA Windows zips live on the
# rolling bNNNN build tags, so those need a scan of recent releases instead.
GITHUB = [
    ("ggml-org/llama.cpp", r"^llama-b\d+-bin-win-cuda-12\.\d+-x64\.zip$", "llamacpp-official", True, r"^b\d+$"),
    ("ggml-org/llama.cpp", r"^cudart-llama-bin-win-cuda-12\.\d+-x64\.zip$", "llamacpp-official", True, r"^b\d+$"),
    ("ollama/ollama", r"^ollama-windows-amd64\.zip$", "ollama", True, None),
    ("LostRuins/koboldcpp", r"^koboldcpp\.exe$", "koboldcpp", False, None),
    ("janhq/jan", r"^Jan_[\d.]+_x64-setup\.exe$", "installers", False, None),
    ("Mintplex-Labs/anything-llm", r"^AnythingLLMDesktop\.exe$", "installers", False, None),
    ("aaif-goose/goose", r"^Goose-win32-x64-cuda\.zip$", "goose", True, None),
    ("aaif-goose/goose", r"^goose-x86_64-pc-windows-msvc\.zip$", "goose", True, None),
]

# Hugging Face repo -> quant substring to match in the file list
HF_MODELS = [("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Q4_K_M")]


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def curl(url: str, dest: Path, expect: int | None = None) -> bool:
    """Resumable download. Returns True when the file is complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expect and dest.exists() and dest.stat().st_size == expect:
        log(f"have {dest.name} ({expect / 1e6:.0f} MB)")
        return True
    log(f"fetch {dest.name} <- {url}")
    rc = subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "10",
                         "--connect-timeout", "30", "-o", str(dest), url]).returncode
    size = dest.stat().st_size if dest.exists() else 0
    if rc != 0 or (expect and size != expect):
        log(f"INCOMPLETE {dest.name} rc={rc} size={size} expected={expect}")
        return False
    log(f"done {dest.name} ({size / 1e6:.0f} MB)")
    return True


def unpack(archive: Path, target: Path) -> None:
    marker = target / (".unpacked-" + archive.name)
    if marker.exists():
        return
    log(f"unpack {archive.name} -> {target}")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(target)
    marker.write_text(str(archive.stat().st_size), encoding="utf-8")


def releases(repo: str, tag_re: str | None) -> list[dict]:
    if not tag_re:
        return [get_json(f"https://api.github.com/repos/{repo}/releases/latest")]
    rels = get_json(f"https://api.github.com/repos/{repo}/releases?per_page=30")
    return [r for r in rels if re.match(tag_re, r.get("tag_name", ""))]


def fetch_github() -> list[Path]:
    installers: list[Path] = []
    for repo, pattern, folder, do_unpack, tag_re in GITHUB:
        try:
            rels = releases(repo, tag_re)
        except Exception as e:  # a rate-limited or renamed repo must not kill the rest
            log(f"SKIP {repo}: release lookup failed: {e}")
            continue
        rel, assets = None, []
        for r in rels:  # newest first; take the newest release that actually has the asset
            assets = [a for a in r.get("assets", []) if re.match(pattern, a["name"])]
            if assets:
                rel = r
                break
        if not assets:
            log(f"SKIP {repo}: no asset matching {pattern} in {[r.get('tag_name') for r in rels][:5]}")
            continue
        a = assets[0]
        dest = BENCH / "downloads" / a["name"]
        if not curl(a["browser_download_url"], dest, a.get("size")):
            continue
        log(f"{repo} {rel.get('tag_name')}: {a['name']}")
        if do_unpack:
            unpack(dest, BENCH / folder)
        elif folder == "installers":
            installers.append(dest)
        else:
            (BENCH / folder).mkdir(parents=True, exist_ok=True)
            shutil.copy(dest, BENCH / folder / a["name"])
    return installers


def fetch_models() -> None:
    for repo, quant in HF_MODELS:
        try:
            info = get_json(f"https://huggingface.co/api/models/{repo}?blobs=true")
        except Exception as e:
            log(f"SKIP {repo}: {e}")
            continue
        files = [s for s in info.get("siblings", []) if quant in s["rfilename"] and s["rfilename"].endswith(".gguf")]
        if not files:
            log(f"SKIP {repo}: no {quant} gguf; available: {[s['rfilename'] for s in info.get('siblings', [])][:20]}")
            continue
        target = MODELS / repo.split("/")[-1]
        for s in sorted(files, key=lambda s: s["rfilename"]):
            name = s["rfilename"]
            url = f"https://huggingface.co/{repo}/resolve/main/{name}"
            curl(url, target / Path(name).name, s.get("size"))


def main() -> int:
    free_gb = shutil.disk_usage("C:\\").free / 2**30
    log(f"start; C: free {free_gb:.0f} GB")
    if free_gb < 60:
        log("ABORT: under 60 GB free on C:")
        return 2
    installers = fetch_github()
    fetch_models()
    log("pip/npm rivals are installed by the eval harness step, not here")
    if installers:
        log("GUI installers downloaded, run deliberately: " + ", ".join(str(p) for p in installers))
    log(f"finished; C: free {shutil.disk_usage('C:\\').free / 2**30:.0f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
