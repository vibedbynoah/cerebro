"""Send a task to a local OpenAI-compatible server (default: Strata serving Qwen3.8-Flash-Next at
127.0.0.1:8081) and print or save the answer. Used to draft boilerplate locally instead of spending
cloud credits. Usage: python ask_local.py [--out FILE] [--max N] [--url URL] [--model ID] < prompt.txt"""
import json, os, sys, time, urllib.request

args = sys.argv[1:]
out = args[args.index("--out") + 1] if "--out" in args else None
mx = int(args[args.index("--max") + 1]) if "--max" in args else 4096
url = args[args.index("--url") + 1] if "--url" in args else os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:8081")
model = args[args.index("--model") + 1] if "--model" in args else os.environ.get("LOCAL_LLM_MODEL", "qwen3.8-flash-next-ud-iq4_xs")
prompt = sys.stdin.read()
body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": mx, "temperature": 0.2}
req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t = time.time()
r = json.load(urllib.request.urlopen(req, timeout=7200))
text = r["choices"][0]["message"].get("content") or ""
u = r.get("usage", {})
info = f"[{u.get('completion_tokens')} tokens in {time.time()-t:.0f}s]"
if out:
    open(out, "w", encoding="utf-8").write(text)
    print(info, "->", out)
else:
    print(text); print(info)
