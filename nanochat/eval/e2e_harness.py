#!/usr/bin/env python3
"""End-to-end harness simulation: the exact turn sequence the user reported.

Runs the REAL baby-onit agent loop (its own code) against the served
endpoint, in a THROWAWAY data dir, and reports per-turn latency + whether
the answer is degenerate. This is the acceptance test for the hang.
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

PORT = os.environ.get("PORT", "8000")
HOST = os.environ.get("HOST", f"http://127.0.0.1:{PORT}")
TURNS = ["hello", "who are you", "date today", "weather today", "hello",
         "what is the weather today", "current data path", "date today"]

BS = chr(92) + "n"


async def main():
    tmp = Path(tempfile.mkdtemp(prefix="babyonit-e2e-"))
    cfg = {
        "data_path": str(tmp),
        "serving": {"provider": "vllm", "host": HOST, "model": "nanochat-d28b-rl100",
                    "think": True, "max_tokens": 32768, "max_chat_iterations": -1,
                    "max_context_tokens": 262144, "history_budget_tokens": 8000,
                    "temperature": 0.6, "top_p": 0.95},
        "enforce_jail": True, "block_dangerous": True,
    }
    prov = B.Provider(cfg)
    history = []
    bad = 0
    try:
        for turn in TURNS:
            t0 = time.monotonic()
            answer = await B.agent_loop(turn, cfg, prov,
                                        session_history=history, verbose=False)
            dt = time.monotonic() - t0
            degen = (answer.count(BS) > 3 or len(answer) > 2000
                     or bool(re.search(r"(.)\1{47,}", answer)))
            if degen:
                bad += 1
            history.append({"task": turn, "response": answer})
            flag = "DEGENERATE" if degen else "ok"
            print(f"  {dt:6.1f}s  len={len(answer):6d}  {flag:11s} "
                  f"| {turn!r} -> {answer[:66]!r}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    print(f"degenerate turns: {bad}/{len(TURNS)}")
    print("PASS" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


sys.exit(asyncio.run(main()))
