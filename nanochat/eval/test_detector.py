#!/usr/bin/env python3
"""Unit-test the guard's degeneracy detector against the REAL poisoned data."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "serving"))
import guard_proxy as G  # noqa: E402

# Use the POISONED backup: the live session history has been sanitized, so it
# no longer contains the degenerate answers this test needs to check.
# The fixture is checked in at eval/fixtures/session_history_poisoned.jsonl;
# override with POISONED_HISTORY to point at a live backup.
import glob  # noqa: E402
_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURE = os.path.join(_HERE, "fixtures", "session_history_poisoned.jsonl")
_cands = sorted(glob.glob(os.path.join(_HERE, "..", "serving", "session_history.jsonl.bak-*")))
if os.environ.get("POISONED_HISTORY"):
    BAK = os.environ["POISONED_HISTORY"]
elif os.path.exists(_FIXTURE):
    BAK = _FIXTURE
elif _cands:
    BAK = _cands[-1]
else:
    raise SystemExit("no poisoned history found: expected %s" % _FIXTURE)
rows = [json.loads(l) for l in open(BAK) if l.strip()]
print(f"(poisoned history: {BAK})")

print("=== is_degenerate on each stored response ===")
for i, r in enumerate(rows):
    s = r.get("response") or ""
    print(f"  row {i:2d} len={len(s):5d} degenerate={G.is_degenerate(s)!s:5} "
          f"cycle_run={G.cycle_run_len(s):4d}  {repr(s[:45])}")

print()
print("=== false-positive check: CLEAN text must NOT be flagged ===")
clean = [
    "Today's date is 2026-10-20, per the system header.",
    "Hello! How can I help you today?",
    "Hi! What are we working on?",
    "2+2 is 4.",
    "The capital of France is Paris.",
    "## Status\n\nThe run is healthy. Steps are landing at the expected rate.\n\n"
    "| | |\n|---|---|\n| step | 12/40 |\n| loss | 0.3300 |\n",
    "I checked the path end to end. The numbers:\n\n- baseline: 0.33\n- with the fix: 0.29\n",
    # a legitimate long answer with a repeated token but no cycle
    "aaaa bbbb aaaa bbbb aaaa bbbb aaaa bbbb aaaa bbbb aaaa bbbb",
    # code with escaped newlines (legit!)
    "print('line1\\nline2\\nline3')",
]
bad = 0
for t in clean:
    d = G.is_degenerate(t)
    if d:
        bad += 1
    print(f"  degenerate={d!s:5}  {repr(t[:60])}")
print(f"  false positives: {bad}/{len(clean)}")

print()
print("=== the exact poisoned answers MUST be flagged ===")
for i in (7, 8, 9, 11):
    s = rows[i]["response"]
    assert G.is_degenerate(s), f"row {i} NOT flagged!"
    print(f"  row {i}: flagged OK")

print()
print("=== strip_degenerate_tail ===")
for i in (7, 9, 11):
    s = rows[i]["response"]
    t = G.strip_degenerate_tail(s)
    print(f"  row {i}: {len(s)} -> {len(t)} chars; tail now {t[-40:]!r}")

print()
print("=== scrub_messages on the real harness message list ===")
sys.path.insert(0, "/home/rowel/github/sibyl/baby-onit")
import baby_onit as B  # noqa: E402
CFG = {"data_path": "/home/rowel/baby-sandbox",
       "serving": {"history_budget_tokens": 8000}}
hist = B.load_session_history(CFG)
msgs = [{"role": "system", "content": B.build_system_prompt(CFG)}]
msgs += B.build_session_messages(hist, 8000)
msgs.append({"role": "user", "content": "date today"})
new, n = G.scrub_messages(msgs)
print(f"  messages {len(msgs)} -> {len(new)}, scrubbed {n}")
for m in new:
    if m.get("role") == "assistant":
        print(f"    assistant len={len(m.get('content') or ''):5d} "
              f"degen={G.is_degenerate(m.get('content') or '')!s:5} "
              f"{repr((m.get('content') or '')[:40])}")
