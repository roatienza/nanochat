#!/usr/bin/env python3
"""Hypothesis test: scrub degenerate assistant turns from the REQUEST history.

If the model then answers cleanly and fast, the server-side fix is
'scrub the prompt', not just 'abort the output'.
"""
import json
import re
import os
import sys
import time
import urllib.request

sys.path.insert(0, "/home/rowel/github/sibyl/baby-onit")
import baby_onit as B  # noqa: E402

CFG = {"data_path": "/home/rowel/baby-sandbox",
       "serving": {"history_budget_tokens": 8000}}

RUN48 = re.compile(r"(.)\1{47,}")


def cycle_run(s: str, min_reps: int = 8) -> int:
    """Length of the trailing short-period cycle, 0 if none.
    p in 2..8, requires min_reps repetitions."""
    w = s[-256:]
    for p in range(2, 9):
        need = p * min_reps
        if len(w) >= need and w[-need:] == w[-p:] * min_reps:
            return need
    return 0


def is_degenerate(s: str) -> bool:
    if not s:
        return False
    if RUN48.search(s):
        return True
    if cycle_run(s):
        return True
    if len(s) >= 200 and re.fullmatch(r"[\sm]{200,}", s[-200:]):
        return True
    return False


def trim_tail(s: str) -> str:
    """Cut a string at the start of its trailing degenerate cycle."""
    c = cycle_run(s)
    if c:
        return s[: len(s) - c]
    m = RUN48.search(s)
    if m:
        return s[: m.start()]
    return s


def call(base, msgs, max_tokens=1024, temperature=0.6):
    body = {"model": "nanochat-d28b-rl100", "messages": msgs,
            "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.95}
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=280) as r:
        j = json.loads(r.read())
    dt = time.monotonic() - t0
    m = (j.get("choices") or [{}])[0].get("message") or {}
    return dt, (m.get("content") or ""), m.get("tool_calls")


def show(tag, dt, c, tcs=None):
    print(f"--- {tag}")
    print(f"    wall={dt:.1f}s len={len(c)} realNL={c.count(chr(10))} "
          f"litBSn={c.count(chr(92)+'n')} tools={len(tcs or [])}")
    print(f"    {c[:160]!r}")


base = "http://127.0.0.1:8010"
hist = B.load_session_history(CFG)
sysmsg = {"role": "system", "content": B.build_system_prompt(CFG)}

poisoned = [sysmsg] + B.build_session_messages(hist, 8000)
poisoned.append({"role": "user", "content": "date today"})

# --- variant A: replace degenerate assistant turns with a marker
scrubbed = []
n_scrub = 0
for m in poisoned:
    if m.get("role") == "assistant" and is_degenerate(m.get("content") or ""):
        n_scrub += 1
        scrubbed.append({"role": "assistant", "content": "[earlier answer truncated]"})
    else:
        scrubbed.append(m)
print(f"scrubbed {n_scrub} degenerate assistant turns")
dt, c, tcs = call(base, scrubbed)
show("A: poisoned history, degenerate turns replaced by marker", dt, c, tcs)

# --- variant B: trim the degenerate tail but keep the clean prefix
trimmed = []
n_trim = 0
for m in poisoned:
    if m.get("role") == "assistant" and is_degenerate(m.get("content") or ""):
        n_trim += 1
        trimmed.append({"role": "assistant", "content": trim_tail(m.get("content") or "")})
    else:
        trimmed.append(m)
dt, c, tcs = call(base, trimmed)
show("B: poisoned history, tails trimmed (prefix kept)", dt, c, tcs)
