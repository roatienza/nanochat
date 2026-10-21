#!/usr/bin/env python3
"""Measure the slow/incomplete rate on the user's EXACT reported sequence.

Runs the real baby-onit agent_loop over:
  hello -> who are you -> date today -> weather today
against the served endpoint, N times, and reports:
  - per-turn latency
  - how many turns were slow (> 30s) or incomplete (degenerate)
"""
import asyncio
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

BABY_ONIT_DIR = os.environ.get("BABY_ONIT_DIR", "/data/rowel/tmp/baby-onit")
sys.path.insert(0, BABY_ONIT_DIR)
import baby_onit as B  # noqa: E402

HOST = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
TURNS = ["hello", "who are you", "date today", "weather today"]
BS = chr(92) + "n"

slow = 0
degen = 0
total = 0
worst = 0.0


async def main():
    global slow, degen, total, worst
    for i in range(N):
        tmp = Path(tempfile.mkdtemp(prefix="babyonit-rate-"))
        cfg = {
            "data_path": str(tmp),
            "serving": {"provider": "vllm", "host": HOST,
                        "model": os.environ.get("UTAK_MODEL_NAME", "utak"), "think": True,
                        "max_tokens": 32768, "max_chat_iterations": -1,
                        "max_context_tokens": 262144,
                        "history_budget_tokens": 8000,
                        "temperature": 0.6, "top_p": 0.95},
            "enforce_jail": True, "block_dangerous": True,
        }
        prov = B.Provider(cfg)
        h = []
        try:
            for turn in TURNS:
                t0 = time.monotonic()
                ans = await B.agent_loop(turn, cfg, prov, session_history=h,
                                         verbose=False)
                dt = time.monotonic() - t0
                h.append({"task": turn, "response": ans})
                total += 1
                worst = max(worst, dt)
                bad = (ans.count(BS) > 3 or len(ans) > 2000
                       or bool(re.search(r"(.)\1{47,}", ans)))
                if bad:
                    degen += 1
                if dt > 30:
                    slow += 1
                if dt > 30 or bad:
                    print(f"  run {i} {turn!r}: {dt:.1f}s len={len(ans)} "
                          f"bad={bad}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print()
    print(f"turns={total}  slow(>30s)={slow}  degenerate={degen}  "
          f"worst={worst:.1f}s")
    print("PASS" if slow == 0 and degen == 0 else "ISSUES REMAIN")


asyncio.run(main())
