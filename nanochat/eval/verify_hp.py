#!/usr/bin/env python3
"""HP gate: the self-poisoning loop reported by the harness (turn 7).

Scenario: an old session contains the model's OWN bad answers, e.g.
    user: "weather today"  ->  asst: "Today's date is December 7, 2026."
The harness replays that history verbatim. The model then pattern-matches
its own poisoned pairs and (a) copies the canned date as the answer to NEW
questions, (b) never emits a tool call.

Probes (history = real 51-turn replay + injected poison pairs):
  HP1 poisoned hello x3      -> greeting, no canned-date copy
  HP2 poisoned weather x3    -> CALL web_search (route beats pattern)
  HP3 poisoned news x3       -> CALL web_search type=news
  HP4 poisoned files x3      -> CALL bash ls
  HP5 poisoned date x3       -> date from system prompt, no canned copy
  HP6 poisoned 2+2 x3        -> "4", no call, no canned copy
  HP7 poison-only history x3 -> weather question routes to web_search
Deploy only on 21/21.
"""
import json
import os
import re
import sys
import time
import urllib.request
DATA_DIR = os.environ.get("GATE_DATA_DIR", "/data/rowel")

PORT = sys.argv[1] if len(sys.argv) > 1 else "8000"
URL = f"http://localhost:{PORT}/v1/chat/completions"
OUT = os.environ.get("GATE_OUT", os.path.join(DATA_DIR, "test_128k_hp_results.txt"))

BABY_ONIT_DIR = os.environ.get("BABY_ONIT_DIR", "/data/rowel/tmp/baby-onit")
sys.path.insert(0, BABY_ONIT_DIR)
import baby_onit as bo

cfg = {"data_path": DATA_DIR, "serving": {"host": URL.rsplit('/v1', 1)[0],
       "model": os.environ.get("UTAK_MODEL_NAME", "utak"), "history_budget_tokens": 8000,
       "max_context_tokens": 262144, "temperature": 0.6, "top_p": 0.95,
       "max_tokens": 32768, "max_chat_iterations": -1, "think": True}}
SYSMSG = bo.build_system_prompt(cfg)
# The date the model is actually told, e.g. "... (today: 2026-10-31)."
_m = re.search(r"today:\s*(\d{4}-\d{2}-\d{2})", SYSMSG)
EXPECTED_DATE = _m.group(1) if _m else time.strftime("%Y-%m-%d")
TOOLS = [s for s, _ in bo.TOOLS.values()]
REPLAY = bo.build_session_messages(bo.load_session_history(cfg), 8000)

CANNED = "Today's date is December 7, 2026."

# The poison: the model's own broken answers from the user's real 284-turn session.
POISON = [
    {"role": "user", "content": "weather today"},
    {"role": "assistant", "content": CANNED},
    {"role": "user", "content": "news today"},
    {"role": "assistant", "content": CANNED},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "I'm sorry for the inconvenience. However, as an AI assistant, I don't have the ability to perform external tasks such as tasks that are outside of my current functions."},
    {"role": "user", "content": "how are you"},
    {"role": "assistant", "content": "I'm sorry, but as an AI assistant, I don't have the capability to perform external tasks such as tasks that are outside of my current functions."},
    {"role": "user", "content": "date today"},
    {"role": "assistant", "content": CANNED},
]


def log(msg):
    with open(OUT, "a") as f:
        f.write(msg + "\n")
    print(msg)


def chat(messages, tools=None, max_tokens=150, temperature=0.6, seed=0):
    body = {"model": os.environ.get("UTAK_MODEL_NAME", "utak"), "temperature": temperature, "top_p": 0.95,
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
        if p.strip() == "":
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
    calls = [(s["name"], s["args"]) for s in calls.values()]
    return fin, calls, content


def base(history):
    return [{"role": "system", "content": SYSMSG}] + history


def run(label, final, n, expect, history, temp=0.6):
    ok = 0
    detail = []
    for s in range(n):
        fin, calls, content = chat(base(history) + [{"role": "user", "content": final}],
                                   tools=TOOLS, seed=s, temperature=temp)
        good = expect(fin, calls, content)
        ok += good
        detail.append((fin, calls[:1], (content or "")[:80]))
    log(f"{label:32s} {ok}/{n} -> {'PASS' if ok == n else 'FAIL'}")
    if ok != n:
        for d in detail[:2]:
            log(f"     {d}")
    return ok == n


def e_no_call_no_canned(fin, calls, content):
    return not calls and len(content) > 2 and CANNED not in content and "December 7" not in content


def e_call_ws_no_canned(fin, calls, content):
    return bool(calls) and calls[0][0] in ("web_search", "search_web") and CANNED not in content


def e_call_ws_news(fin, calls, content):
    return bool(calls) and calls[0][0] in ("web_search", "search_web") and "news" in calls[0][1]


def e_call_bash_ls(fin, calls, content):
    return bool(calls) and calls[0][0] == "bash" and "ls" in calls[0][1]


def e_date_from_system(fin, calls, content):
    # The contract is "report the date that is in your system prompt", not
    # "report one particular day". SYSMSG is rendered from time.strftime at
    # run time, so the expected date has to be read back out of it; a
    # hardcoded literal is only correct on the day it was written and then
    # fails forever after.
    return not calls and EXPECTED_DATE in content and CANNED not in content


def main():
    log("=== HP poison gate: " + time.strftime("%H:%M:%S") + " ===")
    gates = {}
    hist = REPLAY + POISON
    gates["HP1"] = run("HP1 poisoned hello", "hello", 3, e_no_call_no_canned, hist)
    gates["HP2"] = run("HP2 poisoned weather", "weather today", 3, e_call_ws_no_canned, hist)
    gates["HP3"] = run("HP3 poisoned news", "news today", 3, e_call_ws_news, hist)
    gates["HP4"] = run("HP4 poisoned files", "what files are in the working directory?", 3, e_call_bash_ls, hist)
    gates["HP5"] = run("HP5 poisoned date", "date today", 3, e_date_from_system, hist)
    gates["HP6"] = run("HP6 poisoned 2+2", "What is 2+2?", 3, e_no_call_no_canned, hist)
    gates["HP7"] = run("HP7 poison-only weather", "what is the weather in Tokyo right now?", 3, e_call_ws_no_canned, POISON)
    npass = sum(gates.values())
    log(f"=== HP gate: {npass}/{len(gates)} PASS ===")
    log(json.dumps(gates))
    return npass == len(gates)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)