"""On-demand local LLM service for cheap, free work (owner rule 2026-09-22).

Two roles, one resident at a time because both want the 8 GB GPU:
  gemma  Gemma-4-E4B + vision projector: first-pass code rating and screenshot judging.
  qwen   Qwen3-Coder-30B-A3B MoE: small, low-stakes coding (fixtures, boilerplate, doc tables).

`ensure(role)` starts a detached daemon that runs the official llama-server on loopback with an
API key, and stops it after IDLE_S without requests, so the machine goes quiet again for the
throughput benchmark. Asking for the other role stops the resident one first.

CLI:
  python local_llm.py ask ROLE --prompt-file P [--image IMG ...] [--max-tokens N] [--json]
  python local_llm.py stop
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path(os.environ["LOCALAPPDATA"]) / "cerebro-local"
KEY_FILE = HOME / "api.key"
LLAMA_SERVER = Path(r"C:\bench\llamacpp-official\llama-server.exe")
IDLE_S = 15 * 60
LMS = Path(r"C:\Users\noahm\.lmstudio\models")

ROLES = {
    "gemma": {
        "port": 8190,
        "model": LMS / r"lmstudio-community\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-Q4_K_M.gguf",
        "mmproj": LMS / r"lmstudio-community\gemma-4-E4B-it-GGUF\mmproj-gemma-4-E4B-it-BF16.gguf",
        # Measured 2026-09-22: llama.cpp's own fit beat hand-tuned flags on this model, so only
        # the context size is pinned and placement is left to --fit.
        "args": ["-c", "12288"],
    },
    "qwen": {
        "port": 8191,
        "model": Path(r"C:\models\Qwen3-Coder-30B-A3B-Instruct-GGUF\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"),
        "mmproj": None,
        # Measured: experts in host RAM (-ncmoe) is the placement that makes this MoE usable.
        "args": ["-c", "16384", "-ngl", "999", "-ncmoe", "32", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on"],
    },
}


def _key() -> str:
    HOME.mkdir(parents=True, exist_ok=True)
    if not KEY_FILE.exists():
        KEY_FILE.write_text(secrets.token_hex(32), encoding="ascii")
    return KEY_FILE.read_text(encoding="ascii").strip()


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _healthy(role: str) -> bool:
    port = ROLES[role]["port"]
    if not _port_open(port):
        return False
    req = urllib.request.Request(f"http://127.0.0.1:{port}/health", headers={"Authorization": f"Bearer {_key()}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _pid_file(role: str) -> Path:
    return HOME / f"{role}.daemon.pid"


def _heartbeat(role: str) -> Path:
    return HOME / f"{role}.heartbeat"


def stop(role: str | None = None) -> None:
    for r in ([role] if role else list(ROLES)):
        pf = _pid_file(r)
        if pf.exists():
            pid = pf.read_text().strip()
            subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)
            pf.unlink(missing_ok=True)


def ensure(role: str, timeout_s: int = 240) -> str:
    """Make sure `role` is serving; returns its base URL."""
    cfg = ROLES[role]
    _heartbeat(role).parent.mkdir(parents=True, exist_ok=True)
    _heartbeat(role).touch()
    if _healthy(role):
        return f"http://127.0.0.1:{cfg['port']}"
    for other in ROLES:
        if other != role:
            stop(other)
    if not _port_open(cfg["port"]):
        creation = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "daemon", role],
                         creationflags=creation, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _healthy(role):
            return f"http://127.0.0.1:{cfg['port']}"
        time.sleep(2)
    raise RuntimeError(f"local {role} server did not become healthy in {timeout_s}s; see {HOME / (role + '.server.log')}")


def daemon(role: str) -> None:
    cfg = ROLES[role]
    _pid_file(role).write_text(str(os.getpid()))
    argv = [str(LLAMA_SERVER), "-m", str(cfg["model"]), "--host", "127.0.0.1", "--port", str(cfg["port"]),
            "--api-key-file", str(KEY_FILE), "--no-webui", *cfg["args"]]
    if cfg["mmproj"]:
        argv += ["--mmproj", str(cfg["mmproj"])]
    _key()
    log = open(HOME / f"{role}.server.log", "w", encoding="utf-8", errors="replace")
    server = subprocess.Popen(argv, cwd=str(LLAMA_SERVER.parent), stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        while server.poll() is None:
            idle = time.time() - _heartbeat(role).stat().st_mtime
            if idle > IDLE_S:
                break
            time.sleep(20)
    finally:
        subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], capture_output=True)
        _pid_file(role).unlink(missing_ok=True)


def ask(role: str, prompt: str, images: list[Path] | None = None, max_tokens: int = 2048,
        json_mode: bool = False, temperature: float = 0.2) -> str:
    base = ensure(role)
    content: list[dict] | str
    if images:
        content = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = base64.b64encode(img.read_bytes()).decode("ascii")
            mime = "image/png" if img.suffix.lower() == ".png" else "image/jpeg"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    else:
        content = prompt
    # Thinking is switched off: these are short rating and drafting jobs, and a reasoning model can
    # otherwise spend the whole token budget in its reasoning channel and return an empty answer.
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": temperature, "min_p": 0.05,
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_format": "none"}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {_key()}"})
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.load(r)
    _heartbeat(role).touch()
    msg = data["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content") or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("role", choices=list(ROLES))
    a.add_argument("--prompt-file", required=True)
    a.add_argument("--image", action="append", default=[])
    a.add_argument("--max-tokens", type=int, default=2048)
    a.add_argument("--json", action="store_true")
    d = sub.add_parser("daemon")
    d.add_argument("role", choices=list(ROLES))
    sub.add_parser("stop")
    args = ap.parse_args()
    if args.cmd == "daemon":
        daemon(args.role)
    elif args.cmd == "stop":
        stop()
    else:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
        print(ask(args.role, text, [Path(p) for p in args.image], args.max_tokens, args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
