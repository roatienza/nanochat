#!/usr/bin/env python3
"""Test the segment-repetition + low-entropy detectors.

Includes the REAL failing 'weather today' shape captured from the live run.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "serving"))
import guard_proxy as G  # noqa: E402

# The real shape: "## 11<C>" + literal \n repeated with VARYING gaps.
weather = ("Here's the updated weather forecast for October 2026:\n\n"
           "### October\n\n## 11\\u00b0C\\n\\n\\n\\n\\n## 11\\u00b0C\\n\\n\\n\\n\\n\\n\\n"
           "## 11\\u00b0C\\n\\n\\n\\n## 11\\u00b0C\\n\\n\\n\\n\\n\\n\\n## 11\\u00b0C")

print("=== the real failing shape must be FLAGGED ===")
for name, txt in [("weather-segment-loop", weather)]:
    seg = G.repeated_segment_run(txt)
    print(f"  {name}: repeated_segment_run={seg} "
          f"is_degenerate={G.is_degenerate(txt)}")
    assert G.is_degenerate(txt), f"{name} NOT flagged!"

print()
print("=== false-positive check: normal prose / code / tables ===")
clean = [
    "Today's date is 2026-10-20, per the system header.",
    "Hello! How can I help you today?",
    "## Status\n\nThe run is healthy.\n\n| | |\n|---|---|\n| step | 12/40 |\n",
    "The weather today is sunny with some clouds scattered around the horizon.",
    "I checked the path end to end. The numbers:\n\n- baseline: 0.33\n- fix: 0.29\n",
    "Here is the plan:\n\n1. pack the checkpoint\n2. gate it\n3. deploy it\n",
    "The steps are:\n- one\n- two\n- three\n- four\n- five\n- six\n",
    "def solve(rows):\n    return '\\n'.join(out)",
    "## November\n\n## December\n\n## January\n\n## February\n",
    "Today is 2026-10-20. Tomorrow is 2026-10-21. Yesterday was 2026-10-19.",
]
bad = 0
for t in clean:
    d = G.is_degenerate(t)
    if d:
        bad += 1
    print(f"  degenerate={d!s:5} seg_run={G.repeated_segment_run(t):4d} "
          f"{t[:62]!r}")
print(f"  false positives: {bad}/{len(clean)}")

print()
print("=== strip_degenerate_tail on the failing shape ===")
t = G.strip_degenerate_tail(weather)
print(f"  {len(weather)} -> {len(t)} chars")
print(f"  tail now: {t[-60:]!r}")

print()
print("=== a REAL long weather answer with varied temps must SURVIVE ===")
varied = ("Weekly forecast:\n\n## Mon\n18\\u00b0C\n\n## Tue\n19\\u00b0C\n\n"
          "## Wed\n17\\u00b0C\n\n## Thu\n16\\u00b0C\n\n## Fri\n20\\u00b0C\n")
print(f"  is_degenerate={G.is_degenerate(varied)} (expect False)")

print()
print("FAILURES:", bad)
print("PASS" if bad == 0 and G.is_degenerate(weather) else "FAIL")
sys.exit(0 if bad == 0 and G.is_degenerate(weather) else 1)
