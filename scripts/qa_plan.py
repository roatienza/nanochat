"""QA: verify every factual claim in PLAN.md against the real nanochat repo.

Catches the failure mode where a plan drifts from the code it describes.
Run from the reasoner-model repo root, with _repos/nanochat cloned alongside.
"""

import re
import subprocess
import sys
from pathlib import Path

PLAN = Path(__file__).resolve().parent.parent / "PLAN.md"
NANOCHAT = Path(__file__).resolve().parents[2] / "_repos" / "nanochat"

fails = []
oks = []


def check(name, cond, detail=""):
    (oks if cond else fails).append((name, detail))
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))


def read(p):
    try:
        return Path(p).read_text()
    except Exception:
        return ""


def has(path, pattern):
    return re.search(pattern, read(path), re.I | re.M) is not None


print("=" * 72)
print("nanochat repo:", NANOCHAT, "(present)" if NANOCHAT.exists() else "(MISSING)")
print("=" * 72)

if not NANOCHAT.exists():
    print("cannot verify against source; clone karpathy/nanochat first")
    sys.exit(2)

# --- README claims quoted in the plan ---
r = read(NANOCHAT / "README.md")
check("README says code runs on Ampere 8xA100", "Ampere 8XA100" in r)
check("README documents <80GB device-batch-size guidance",
      "less than 80GB" in r and "--device-batch-size" in r)
check("README auto-selects bfloat16 on SM 80+",
      "SM 80+" in r and "bfloat16" in r)
check("speedrun is configured for an 8-GPU node", "8XH100" in r)

# --- FP8 is Hopper-only ---
bt = read(NANOCHAT / "scripts" / "base_train.py")
check("base_train --fp8 is gated on H100+",
      re.search(r'--fp8.*requires H100\+', bt) is not None,
      "flag help text no longer states H100+ requirement")
check("speedrun.sh passes --fp8 (so it must be dropped on A100)",
      "--fp8" in read(NANOCHAT / "runs" / "speedrun.sh"))

# --- d24 does not fit: recompute independently of scripts/budget.py ---
sys.path.insert(0, str(PLAN.parent / "scripts"))
import budget  # noqa: E402

for depth, should_fit in ((12, True), (20, True), (22, True), (24, False), (26, False)):
    fits = budget.max_fitting_batch(depth) is not None
    check(f"d{depth} fit={should_fit}", fits == should_fit,
          f"computed fit={fits}, peak@bs1={budget.peak_gib(depth, 1):.1f} GiB")

check("d20 params ~2.15B", abs(budget.n_params(20)["params"] / 1e9 - 2.155) < 0.02)
check("d24 params ~3.20B", abs(budget.n_params(24)["params"] / 1e9 - 3.196) < 0.02)
check("d20 peak memory under 38 GiB at bs=32", budget.peak_gib(20, 32) < 38,
      f"{budget.peak_gib(20, 32):.1f} GiB")

# --- SmolTalk has no function calling (the Stage 4 justification) ---
check("chat_sft imports SmolTalk from tasks.smoltalk",
      "from tasks.smoltalk import SmolTalk" in read(NANOCHAT / "scripts" / "chat_sft.py"))
st = read(NANOCHAT / "tasks" / "smoltalk.py")
check("SmolTalk task loads smol-smoltalk", "smol-smoltalk" in st)
check("SFT mixture is SmolTalk + MMLU + GSM8K",
      all(x in read(NANOCHAT / "scripts" / "chat_sft.py")
          for x in ("SmolTalk(split=", "MMLU(subset=", "GSM8K(subset=")))

# --- RL and sandbox exist ---
check("chat_rl.py exists (reasoning RL)", (NANOCHAT / "scripts" / "chat_rl.py").exists())
rl = read(NANOCHAT / "scripts" / "chat_rl.py")
check("chat_rl self-describes as GRPO in quotes",
      '"GRPO"' in rl or "'GRPO'" in rl or "GRPO in quotes" in rl)
check("chat_rl imports the GSM8K task", "tasks.gsm8k" in rl)
ex = read(NANOCHAT / "nanochat" / "execution.py")
check("execution.py provides a sandbox", "subprocess" in ex and "rlimit" in ex.lower())
check("execution.py warns it is not a security boundary",
      "not a true security sandbox" in ex.lower() or "no kernel-level" in ex.lower())

# --- architecture defaults quoted in the plan ---
check("default aspect-ratio is 64",
      re.search(r"--aspect-ratio.*default=64", bt) is not None)
check("default max-seq-len is 2048",
      re.search(r"--max-seq-len.*default=2048", bt) is not None)
check("default head-dim is 128",
      re.search(r"--head-dim.*default=128", bt) is not None)
check("tokenizer vocab is 2**15 = 32768",
      "32768" in read(NANOCHAT / "runs" / "speedrun.sh"))

# --- every source URL in the plan is well-formed and cited ---
plan = read(PLAN)
urls = re.findall(r"https?://[^\s\)\]]+", plan)
check("plan cites >= 10 sources", len(urls) >= 10, f"found {len(urls)}")
bad = [u for u in urls if " " in u]
check("no malformed URLs", not bad, str(bad[:3]))

print()
print("=" * 72)
print(f"{len(oks)} passed, {len(fails)} failed")
print("=" * 72)
for n, d in fails:
    print(f"  FAILED: {n}   {d}")
sys.exit(1 if fails else 0)
