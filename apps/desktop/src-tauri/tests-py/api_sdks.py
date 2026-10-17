"""The local API against the official OpenAI and Anthropic Python SDKs, with a real model loaded.

Run by the ignored Rust test `local_api_with_real_sdks` (commands.rs):
    python api_sdks.py BASE_URL KEY
Prints one line per check and exits non-zero on the first failure.
"""

import json
import sys

import anthropic
import openai

base, key = sys.argv[1], sys.argv[2]
WEATHER = {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        sys.exit(1)


oa = openai.OpenAI(base_url=base, api_key=key)
models = [m.id for m in oa.models.list()]
check("openai models.list", len(models) > 0, ", ".join(models[:3]))

r = oa.chat.completions.create(model="local", max_tokens=60, temperature=0, messages=[{"role": "system", "content": "Answer in one word."}, {"role": "user", "content": "What is the capital of France?"}])
text = r.choices[0].message.content or ""
check("openai chat", "paris" in text.lower(), repr(text))
check("openai usage", r.usage.prompt_tokens > 0 and r.usage.completion_tokens > 0, str(r.usage))

parts, finish = [], None
for chunk in oa.chat.completions.create(model="local", max_tokens=60, temperature=0, stream=True, messages=[{"role": "user", "content": "Count from 1 to 5, separated by spaces."}]):
    if chunk.choices:
        parts.append(chunk.choices[0].delta.content or "")
        finish = chunk.choices[0].finish_reason or finish
streamed = "".join(parts)
check("openai stream", "1" in streamed and "5" in streamed and finish == "stop", repr(streamed))

msgs = [{"role": "user", "content": "What is the weather in Oslo? Use the tool."}]
r = oa.chat.completions.create(model="local", max_tokens=200, temperature=0, tools=[{"type": "function", "function": WEATHER}], messages=msgs)
calls = r.choices[0].message.tool_calls or []
check("openai tool call", r.choices[0].finish_reason == "tool_calls" and calls and calls[0].function.name == "get_weather", str(calls))
args = json.loads(calls[0].function.arguments)
check("openai tool arguments", "oslo" in str(args.get("city", "")).lower(), str(args))
msgs += [r.choices[0].message.model_dump(exclude_none=True), {"role": "tool", "tool_call_id": calls[0].id, "content": "Sunny, 21 degrees Celsius."}]
r = oa.chat.completions.create(model="local", max_tokens=120, temperature=0, tools=[{"type": "function", "function": WEATHER}], messages=msgs)
final = r.choices[0].message.content or ""
check("openai tool result used", "21" in final, repr(final))

an = anthropic.Anthropic(base_url=base.removesuffix("/v1"), api_key=key)
m = an.messages.create(model="local", extra_body={"temperature": 0}, max_tokens=60, system="Answer in one word.", messages=[{"role": "user", "content": "What is the capital of France?"}])
text = "".join(b.text for b in m.content if b.type == "text")
check("anthropic messages", "paris" in text.lower() and m.stop_reason == "end_turn", repr(text))

with an.messages.stream(model="local", extra_body={"temperature": 0}, max_tokens=60, messages=[{"role": "user", "content": "Count from 1 to 5, separated by spaces."}]) as s:
    streamed = "".join(s.text_stream)
    final_msg = s.get_final_message()
check("anthropic stream", "5" in streamed and final_msg.usage.output_tokens > 0, repr(streamed))

atool = {"name": WEATHER["name"], "description": WEATHER["description"], "input_schema": WEATHER["parameters"]}
amsgs = [{"role": "user", "content": "What is the weather in Oslo? Use the tool."}]
m = an.messages.create(model="local", extra_body={"temperature": 0}, max_tokens=200, tools=[atool], messages=amsgs)
uses = [b for b in m.content if b.type == "tool_use"]
check("anthropic tool_use", m.stop_reason == "tool_use" and uses and uses[0].name == "get_weather", str(m.content))
amsgs += [{"role": "assistant", "content": [b.model_dump() for b in m.content]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0].id, "content": "Rain, 9 degrees Celsius."}]}]
with an.messages.stream(model="local", extra_body={"temperature": 0}, max_tokens=120, tools=[atool], messages=amsgs) as s:
    final = "".join(s.text_stream)
check("anthropic tool result used (streamed)", "9" in final, repr(final))

with an.messages.stream(model="local", extra_body={"temperature": 0}, max_tokens=30, stop_sequences=["4"], messages=[{"role": "user", "content": "Count from 1 to 9, separated by spaces."}]) as s:
    cut = "".join(s.text_stream)
    fm = s.get_final_message()
check("anthropic stop sequence", "4" not in cut and fm.stop_reason == "stop_sequence", repr(cut))
print("ALL PASS")
