#!/usr/bin/env python3
"""msx gate: multi-step, long-context reasoning and constraint following.

Why this file exists
--------------------
`verify_v5x.py` is 15 checks of ROUTING: every one sends a single user turn and
grades the tool NAME. Nothing in it needs two dependent calls, combines two
facts from one document, or tests whether the model honours an explicit
constraint in the instruction. So the suite reported 15/15 while the capability
the user actually notices -- context understanding and multi-step task
completion -- was untested.

This gate closes that hole. It is deliberately independent of verify_v5x:

  MS1  chained calls   grep -> read_file, and the second call's argument is the
                       first call's OUTPUT (the file name is not in the prompt)
  MS2  list -> read    bash ls, pick by a stated rule, read it
  MS3  edit -> verify  read, edit, read back; the answer is post-edit state
  MS4  three reads     sum across three files; no single call yields it
  MS5  search -> read  local_search ranks, read the winner
  LC1  doc arithmetic  answer is a SUM of line items inside a 24k document
  LC2  doc join        combine two facts from DIFFERENT sections
  LC3  doc contradict  two values, higher revision governs; position randomised
  LC4  doc track       apply deltas in section order; final state
  CF1  constraint      an explicit "do not use a tool" must be honoured when
                       the tools ARE present and the request looks toolish
  CF2  constraint      an explicit output format must be honoured verbatim

MS* run the REAL tool loop against tasks.multistep.Sandbox, the same object the
training rows are built from, so the gate cannot drift from the data. LC* ask
for a short answer and grade it exactly. CF* grade the literal output.

Usage:  python3 eval/verify_msx.py <port> [--n 3] [--only MS1,LC3]
Exit 0 only when every check passes.
"""
import argparse
import json
import os
import re
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DATA_DIR = os.environ.get("GATE_DATA_DIR", "/data/rowel")
BABY_ONIT_DIR = os.environ.get("BABY_ONIT_DIR", "/data/rowel/tmp/baby-onit")
sys.path.insert(0, BABY_ONIT_DIR)
import baby_onit as bo  # noqa: E402

from tasks.multistep import Sandbox, build_episode, MultiStep  # noqa: E402
from tasks.longctx_sft import GENERATORS  # noqa: E402

cfg = {"data_path": DATA_DIR, "serving": {"host": "http://localhost:8000", "model": "utak",
       "history_budget_tokens": 8000, "max_context_tokens": 262144, "temperature": 0.6,
       "top_p": 0.95, "max_tokens": 32768, "max_chat_iterations": -1, "think": True}}
SYSMSG = bo.build_system_prompt(cfg)
TOOLS = [s for s, _ in bo.TOOLS.values()]
MODEL = os.environ.get("UTAK_MODEL_NAME", "utak")

FAILS = []
RESULTS = []


def log(msg):
    print(msg)
    out = os.environ.get("GATE_OUT")
    if out:
        with open(out, "a") as f:
            f.write(msg + "\n")


def chat(url, messages, tools=None, max_tokens=200, temperature=0.6, seed=0):
    body = {"model": MODEL, "temperature": temperature, "top_p": 0.95,
            "max_tokens": max_tokens, "messages": messages, "seed": seed,
            "stream": True, "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=900).read().decode()
    content, calls = "", {}
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
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                if tc.get("function"):
                    slot["name"] += tc["function"].get("name") or ""
                    slot["args"] += tc["function"].get("arguments") or ""
    return [(s["name"], s["args"]) for s in calls.values()], content


def parse_args(raw):
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}


def tool_loop(url, question, sandbox, max_steps=6, seed=0, temperature=0.6):
    """Run the deployed loop: call -> result -> call ... -> prose answer."""
    messages = [{"role": "system", "content": SYSMSG},
                {"role": "user", "content": question}]
    trace = []
    for _ in range(max_steps):
        calls, content = chat(url, messages, tools=TOOLS, max_tokens=300,
                              temperature=temperature, seed=seed)
        if not calls:
            return content, trace
        # the deployed harness executes ONE call per turn
        name, raw = calls[0]
        args = parse_args(raw)
        result = sandbox.call(name, args)
        trace.append((name, args))
        messages.append({"role": "assistant", "content": content or f"CALL {name} {raw}"})
        messages.append({"role": "user", "content": f"Tool result: {result}"})
    return "", trace


def num(s):
    """Extract the first integer from a model answer (tolerates commas)."""
    m = re.search(r"-?\d[\d,]*", s or "")
    return int(m.group(0).replace(",", "")) if m else None


# ---------------------------------------------------------------------------
# MS checks: multi-step episodes, graded by sandbox state / final answer
# ---------------------------------------------------------------------------

def ms_check(family, index, n_seeds=3, temp=0.6):
    """Run `family` episodes and require the answer to be right."""
    ok = 0
    detail = []
    for s in range(n_seeds):
        ds = MultiStep(families=(family,), n_rows=index + 1, seed=7000 + s)
        fam, question, files, answer, min_calls = ds.gold(index)
        sb = Sandbox(files)
        got, trace = tool_loop(URL, question, sb, seed=s, temperature=temp)
        if family == "edit_then_verify":
            good = answer.lower() in (got or "").lower()
        else:
            good = num(got) is not None and num(got) == num(answer)
        # a single call cannot be multi-step, whatever the answer says
        if len(trace) < 2:
            good = False
        ok += bool(good)
        detail.append((s, len(trace), (got or "")[:50], answer))
    return ok, n_seeds, detail


# ---------------------------------------------------------------------------
# LC checks: long-context reasoning families, exact short answer
# ---------------------------------------------------------------------------

def lc_check(family, level, n_seeds=3):
    import random
    ok = 0
    detail = []
    for s in range(n_seeds):
        rng = random.Random(f"gate:{family}:{level}:{s}")
        prompt, answer = GENERATORS[family](rng, level, s)
        calls, got = chat(URL, [{"role": "system", "content": SYSMSG},
                                {"role": "user", "content": prompt}],
                          tools=None, max_tokens=120, seed=s)
        if num(answer) is not None:
            good = num(got) is not None and num(got) == num(answer)
        else:
            good = answer.strip().lower() in (got or "").strip().lower()
        ok += bool(good)
        detail.append((s, (got or "")[:50], answer[:50]))
    return ok, n_seeds, detail


# ---------------------------------------------------------------------------
# CF checks: explicit constraints, with the tools PRESENT
# ---------------------------------------------------------------------------

def cf_no_tool(n_seeds=3):
    """A toolish-looking request that explicitly forbids tools."""
    ok = 0
    detail = []
    for s in range(n_seeds):
        q = ("Do NOT use any tool for this. From memory only: in one sentence, "
             "what is the difference between a list and a dict in Python?")
        calls, got = chat(URL, [{"role": "system", "content": SYSMSG},
                                {"role": "user", "content": q}],
                          tools=TOOLS, max_tokens=160, seed=s)
        good = not calls and len((got or "").strip()) > 10
        ok += bool(good)
        detail.append((s, "CALL" if calls else "text", (got or "")[:50]))
    return ok, n_seeds, detail


def cf_format(n_seeds=3):
    """An explicit output format must be honoured verbatim."""
    ok = 0
    detail = []
    for s in range(n_seeds):
        q = ("Reply with exactly this and nothing else, no tool call: "
             "STATUS=READY;COUNT=3")
        calls, got = chat(URL, [{"role": "system", "content": SYSMSG},
                                {"role": "user", "content": q}],
                          tools=TOOLS, max_tokens=60, seed=s)
        good = (not calls) and "STATUS=READY;COUNT=3" in (got or "")
        ok += bool(good)
        detail.append((s, "CALL" if calls else "text", (got or "")[:50]))
    return ok, n_seeds, detail


def run(name, fn):
    try:
        ok, n, detail = fn()
    except Exception as e:
        log(f"{name:24s} ERROR {type(e).__name__}: {e}")
        FAILS.append(name)
        RESULTS.append((name, 0, 0))
        return
    verdict = "PASS" if ok == n else "FAIL"
    log(f"{name:24s} {ok}/{n} -> {verdict}")
    if ok != n:
        for d in detail:
            log(f"    seed {d[0]}: {d[1:]}")
        FAILS.append(name)
    RESULTS.append((name, ok, n))


CHECKS = [
    ("MS1 grep_then_read", lambda: ms_check("grep_then_read", 3)),
    ("MS2 list_then_read", lambda: ms_check("list_then_read", 3)),
    ("MS3 edit_then_verify", lambda: ms_check("edit_then_verify", 3)),
    ("MS4 sum_three", lambda: ms_check("sum_three", 3)),
    ("MS5 search_then_read", lambda: ms_check("search_then_read", 3)),
    ("LC1 doc_arith", lambda: lc_check("doc_arith", 1)),
    ("LC2 doc_join", lambda: lc_check("doc_join", 1)),
    ("LC3 doc_contradict", lambda: lc_check("doc_contradict", 1)),
    ("LC4 doc_track", lambda: lc_check("doc_track", 1)),
    ("LC5 doc_qa", lambda: lc_check("doc_qa", 1)),
    ("CF1 no-tool constraint", cf_no_tool),
    ("CF2 format constraint", cf_format),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="8000")
    ap.add_argument("--only", default="", help="comma-separated check prefixes")
    a = ap.parse_args()
    URL = f"http://localhost:{a.port}/v1/chat/completions"
    only = [x.strip() for x in a.only.split(",") if x.strip()]
    log(f"msx gate on port {a.port} ({URL})")
    for name, fn in CHECKS:
        if only and not any(name.startswith(o) for o in only):
            continue
        run(name, fn)
    total = sum(1 for _ in RESULTS)
    passed = total - len(FAILS)
    log(f"\nmsx: {passed}/{total} checks passed")
    if FAILS:
        log("failed: " + ", ".join(FAILS))
    sys.exit(0 if not FAILS else 1)
