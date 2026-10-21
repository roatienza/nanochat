#!/usr/bin/env python3
"""Regression: the guard must NOT damage legitimate tool calls.

Checks that a real CALL (with JSON args) passes the guard's degeneracy
tests untouched, and that only genuine garbage is dropped.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "serving"))
import guard_proxy as G  # noqa: E402

LEGIT_ARGS = [
    '{"city":"Oslo","unit":"Celsius"}',
    '{"path":"summary.md","content":"# Summary\\n\\n(done in the answer)"}',
    '{"url":"https://example.com/report"}',
    '{"pattern":"def .*","path":"/data/rowel/x.py"}',
    '{"query":"literal backslash n \\\\n in a search"}',
    '{"content":"line1\\nline2\\nline3"}',
    '{}',
    '{"n1":240,"n2":467,"f1":0.3312}',
]
GARBAGE_ARGS = [
    "mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm",
    '{"city":"Oslo","note":"' + "\\\\n" * 30 + '"}',
    '{"x":"' + "m" * 200 + '"}',
    "not json at all",
]

print("=== legitimate tool args must SURVIVE ===")
bad = 0
for a in LEGIT_ARGS:
    slots = {0: {"id": "t1", "name": "get_weather", "args": a}}
    out = G._clean_tool_slots(slots)
    ok = len(out) == 1
    if not ok:
        bad += 1
    print(f"  kept={ok!s:5} {a[:66]!r}")

print()
print("=== garbage tool args must be DROPPED ===")
for a in GARBAGE_ARGS:
    slots = {0: {"id": "t1", "name": "get_weather", "args": a}}
    out = G._clean_tool_slots(slots)
    dropped = len(out) == 0
    if not dropped:
        bad += 1
    print(f"  dropped={dropped!s:5} {a[:60]!r}")

print()
print("=== scrub_messages must keep a NORMAL assistant turn ===")
msgs = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Hello! How can I help you today?"},
    {"role": "user", "content": "date today"},
]
new, n = G.scrub_messages(msgs)
if n != 0 or len(new) != 4:
    bad += 1
print(f"  scrubbed={n} (expect 0), messages {len(msgs)}->{len(new)} (expect 4)")

print()
print("=== a legit long answer with repeated table rows must SURVIVE ===")
legit = ("I checked the path end to end. The numbers:\n\n"
         "- baseline: 0.3312\n- with the fix: 0.2990\n- p95 latency: 10999 ms\n\n"
         "The loader bug is gone. I re-ran the gate three times to be sure it "
         "isn't noise, and the third run matches the first two exactly.")
if G.is_degenerate(legit):
    bad += 1
print(f"  is_degenerate={G.is_degenerate(legit)} (expect False)")

print()
print("FAILURES:", bad)
print("PASS" if bad == 0 else "FAIL")
sys.exit(0 if bad == 0 else 1)
