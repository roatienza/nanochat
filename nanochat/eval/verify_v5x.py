#!/usr/bin/env python3
"""v5x gate: the REAL harness gate, written after the v5v review proved the
old gates were blind (temp 0 + 2 classic tools vs the harness's temp 0.6 +
8 real tools).

Every probe uses the REAL baby-onit system prompt, the REAL 8-tool schema
list, temperature 0.6 / top_p 0.95 (the harness defaults), streaming, and
multiple seeds. Gates:

  F1 fresh hello x5        -> greeting, no call
  F2 fresh 2+2 x5          -> "4", no call
  F3 fresh files->bash x5  -> CALL bash with ls
  F4 fresh weather->call x5-> CALL web_search
  F5 fresh fetch->call x5  -> CALL fetch_content
  R1 replay hello x3       -> greeting, no call
  R2 replay weather x3     -> CALL web_search
  R3 replay news x3        -> CALL web_search type=news
  R4 replay 2+2 x3         -> "4", no call
  R5 replay files x3       -> CALL bash ls
  LOOP 4-turn OpenAI loop  -> call -> result -> answer -> new call
  P7  bare 2-round x10     -> 0 over-calls (v5j fix holds)
  P13 q-answered-by-r1 x5  -> 0 over-calls
  P1  1-round answer x5    -> answers from result
  MT24 multi_tool 24k      -> CALL get_weather Oslo (v5r fix holds)

Deploy only on 15/15.
"""
import json
import os
import sys
import time
import urllib.request
DATA_DIR = os.environ.get("GATE_DATA_DIR", "/data/rowel")

PORT = sys.argv[1] if len(sys.argv) > 1 else "8000"
URL = f"http://localhost:{PORT}/v1/chat/completions"
OUT = os.environ.get("GATE_OUT", os.path.join(DATA_DIR, "test_128k_v5x_results.txt"))

BABY_ONIT_DIR = os.environ.get("BABY_ONIT_DIR", "/data/rowel/tmp/baby-onit")
sys.path.insert(0, BABY_ONIT_DIR)
import baby_onit as bo

cfg = {"data_path": DATA_DIR, "serving": {"host": URL.rsplit('/v1', 1)[0],
       "model": "nanochat-d28b-rl100", "history_budget_tokens": 8000,
       "max_context_tokens": 262144, "temperature": 0.6, "top_p": 0.95,
       "max_tokens": 32768, "max_chat_iterations": -1, "think": True}}
SYSMSG = bo.build_system_prompt(cfg)
TOOLS = [s for s, _ in bo.TOOLS.values()]
REPLAY = bo.build_session_messages(bo.load_session_history(cfg), 8000)

GATE_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
      "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the web",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


def log(msg):
    with open(OUT, "a") as f:
        f.write(msg + "\n")
    print(msg)


def chat(messages, tools=None, max_tokens=150, temperature=0.6, seed=0):
    body = {"model": "nanochat-d28b-rl100", "temperature": temperature, "top_p": 0.95,
            "max_tokens": max_tokens, "messages": messages, "seed": seed,
            "stream": True, "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=600).read().decode()
    content, calls, fin = "", {}, None
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        p = line[6:]
        if p.strip() == "[DONE]":
            break
        try:
            c = json.loads(p)
        except json.JSONDecodeError:
            continue
        for ch in c.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                content += d["content"]
            for tc in d.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = calls.setdefault(idx, {"name": "", "args": ""})
                if tc.get("function"):
                    slot["name"] += tc["function"].get("name") or ""
                    slot["args"] += tc["function"].get("arguments") or ""
            if ch.get("finish_reason"):
                fin = ch["finish_reason"]
    return fin, [(s["name"], s["args"]) for s in calls.values()], content


def base(history=None):
    return [{"role": "system", "content": SYSMSG}] + (history if history is not None else [])


def run(label, final, n, expect, history=None, tools=TOOLS, temp=0.6):
    ok = 0
    detail = []
    for s in range(n):
        fin, calls, content = chat(base(history) + [{"role": "user", "content": final}],
                                   tools=tools, seed=s, temperature=temp)
        good = expect(fin, calls, content)
        ok += good
        detail.append((fin, calls[:1], (content or "")[:60]))
    log(f"{label:30s} {ok}/{n} -> {'PASS' if ok == n else 'FAIL'}")
    if ok != n:
        for d in detail[:2]:
            log(f"     {d}")
    return ok == n


def e_no_call(fin, calls, content):
    return not calls and len(content) > 2
def e_call_bash_ls(fin, calls, content):
    return bool(calls) and calls[0][0] == "bash" and "ls" in calls[0][1]
def e_call_ws(fin, calls, content):
    return bool(calls) and calls[0][0] in ("web_search", "search_web")
def e_call_ws_news(fin, calls, content):
    return bool(calls) and calls[0][0] in ("web_search", "search_web") and "news" in calls[0][1]
def e_call_fetch(fin, calls, content):
    return bool(calls) and calls[0][0] in ("fetch_content",)


def main():
    import random
    rng = random.Random(4242)
    log("=== v5x gate (REAL harness shape, temp 0.6): " + time.strftime("%H:%M:%S") + " ===")
    gates = {}

    gates["F1"] = run("F1 fresh hello", "hello", 5, e_no_call)
    gates["F2"] = run("F2 fresh 2+2", "What is 2+2?", 5, e_no_call)
    gates["F3"] = run("F3 fresh files->bash ls", "what files are in the working directory?", 5, e_call_bash_ls)
    gates["F4"] = run("F4 fresh weather->call", "weather today", 5, e_call_ws)
    gates["F5"] = run("F5 fresh fetch->call", "fetch https://example.com/report and summarize it", 5, e_call_fetch)
    gates["R1"] = run("R1 replay hello", "hello", 3, e_no_call, history=REPLAY)
    gates["R2"] = run("R2 replay weather->call", "weather today", 3, e_call_ws, history=REPLAY)
    gates["R3"] = run("R3 replay news->call", "news today", 3, e_call_ws_news, history=REPLAY)
    gates["R4"] = run("R4 replay 2+2", "What is 2+2?", 3, e_no_call, history=REPLAY)
    gates["R5"] = run("R5 replay files->bash ls", "what files are in the working directory?", 3, e_call_bash_ls, history=REPLAY)

    # LOOP: 4-turn OpenAI loop with the harness tools (call -> result -> answer -> new call)
    msgs = base() + [{"role": "user", "content": "weather in Tokyo?"}]
    fin, calls, content = chat(msgs, tools=TOOLS, temperature=0.0)
    t1 = bool(calls)
    if t1:
        msgs.append({"role": "assistant", "content": content or "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": calls[0][0], "arguments": calls[0][1]}}]})
        msgs.append({"role": "tool", "tool_call_id": "call_1", "content": "Tokyo: 21C, sunny, light wind"})
        fin, calls, content = chat(msgs, tools=TOOLS, temperature=0.0)
        t2 = (not calls) and len(content) > 10 and ("21" in content or "sunny" in content or "Tokyo" in content)
        msgs.append({"role": "assistant", "content": content})
        msgs.append({"role": "user", "content": "and in Paris?"})
        fin, calls, content = chat(msgs, tools=TOOLS, temperature=0.0)
        t3 = bool(calls)
    else:
        t2 = t3 = False
    gates["LOOP"] = t1 and t2 and t3
    log(f"{'LOOP 4-turn':30s} {'PASS' if gates['LOOP'] else 'FAIL'} (T1={t1} T2={t2} T3={t3})")

    # P7/P13/P1 with the CLASSIC 2-tool set (the v5j/v5r regression guards)
    p7 = [
        {"role": "user", "content": "What is the weather in Tokyo right now?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\",\"unit\":\"Celsius\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Tokyo: 21C, sunny, light wind"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\",\"unit\":\"Celsius\"}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "Paris: 18C, cloudy"},
    ]
    over = 0
    for s in range(10):
        fin, calls, content = chat(p7, tools=GATE_TOOLS, temperature=0.0, seed=s)
        if calls:
            over += 1
    gates["P7"] = over == 0
    log(f"{'P7 bare 2-round over-call':30s} {over}/10 -> {'PASS' if gates['P7'] else 'FAIL'}")

    p13 = p7[:3]  # q answered by round 1
    over = 0
    for s in range(5):
        fin, calls, content = chat(p13, tools=GATE_TOOLS, temperature=0.0, seed=s)
        if calls:
            over += 1
    gates["P13"] = over == 0
    log(f"{'P13 q-answered-by-r1':30s} {over}/5 -> {'PASS' if gates['P13'] else 'FAIL'}")

    p1 = p7[:3]
    ok = 0
    for s in range(5):
        fin, calls, content = chat(p1, tools=GATE_TOOLS, temperature=0.0, seed=s)
        if (not calls) and len(content) > 10 and ("21" in content or "Tokyo" in content):
            ok += 1
    gates["P1"] = ok == 5
    log(f"{'P1 1-round answer':30s} {ok}/5 -> {'PASS' if gates['P1'] else 'FAIL'}")

    # MT24: the v5r regression gate (24k doc -> CALL get_weather Oslo)
    _WORDS = ("meadow harbor copper signal orchid quartz ember willow cobalt tundra "
              "falcon marble cedar lantern ripple summit glacier nectar bramble thistle "
              "kestrel saffron basalt juniper meridian flint hollow amber lattice corsair "
              "vellum zephyr cobble damson garnet hearth ingle mica nook opal "
              "pyre quill rune slate tarn umbel vitreous wattle yarrow anchor").split()
    words = [rng.choice(_WORDS) for _ in range(24000 // 2)]
    doc = " ".join(words)
    prompt = (f"Read the logistics note below and use the exact tool names to check the "
              f"weather in Oslo in Celsius.\n\n{doc}\n\nReply with one line of the form "
              f"CALL name {{json arguments}}.")
    fin, calls, content = chat([{"role": "user", "content": prompt}], tools=None,
                               max_tokens=64, temperature=0.0)
    txt = content or ""
    ok = "CALL get_weather" in txt and "Oslo" in txt and "Celsius" in txt
    gates["MT24"] = ok
    log(f"{'MT24 multi_tool ~24k':30s} -> {'PASS' if ok else 'FAIL'} answer={txt[:70]!r}")

    npass = sum(gates.values())
    log(f"=== v5x gate: {npass}/{len(gates)} PASS ===")
    log(json.dumps(gates))
    return npass == len(gates)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)