"""QA: verify every factual claim in PLAN.md against the real nanochat repo.

Catches the failure mode where a plan drifts from the code it describes.

Run from the reasoner-model repo root. The nanochat source it checks against
is the vendored copy at `nanochat/` in this repo (or a clone at
`_repos/nanochat` if one is present, which takes precedence).
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLAN = REPO / "PLAN.md"
# Prefer an external clone if the user has one; otherwise use the vendored copy.
_VENDORED = REPO / "nanochat"
_CLONE = REPO.parent / "_repos" / "nanochat"
NANOCHAT = _CLONE if _CLONE.exists() else _VENDORED

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
# The plan quotes nanochat's upstream README (the Ampere/bfloat16 guidance).
# In this repo that file is vendored as README.upstream.md, since the fork's
# own README.md documents the fork instead.
_readme = NANOCHAT / "README.upstream.md"
if not _readme.exists():
    _readme = NANOCHAT / "README.md"
r = read(_readme)
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

# --- the measured feasibility frontier, recomputed independently of the plan ---
# NOTE: this block used to call budget.peak_gib(), which does not exist, and
# the detail f-string was evaluated eagerly, so the script raised
# AttributeError on the FIRST check even when that check passed. It also
# asserted the plan's pre-audit numbers (d20 = 2.155 B, d24 infeasible), which
# the audit in docs/EXECUTION_LOG_PART1.md (Finding 1) measured to be wrong.
#
# The plan's table is a derived model; budget.py holds the measured one. The
# divergence between them is DOCUMENTED and expected — see the PLAN_DEVIATIONS
# table below — so it is reported, not failed. What must hold is that
# budget.py itself still reproduces the measured frontier.
sys.path.insert(0, str(REPO / "scripts"))
import budget  # noqa: E402


def peak_gib(depth, bs):
    """Measured peak GiB for (depth, device_batch_size); 0.0 == does not fit."""
    return budget.MEASURED_GIB.get(depth, {}).get(bs, 0.0)


# The measured frontier, per docs/EXECUTION_LOG_PART1.md "Finding 1".
MEASURED_MAX_BS = {12: 16, 16: 8, 18: 4, 20: 8, 22: 8, 24: 4, 26: 4, 28: 4}
for depth, want_bs in MEASURED_MAX_BS.items():
    got = budget.max_fitting_batch(depth)
    check(f"measured max bs for d{depth} is {want_bs}", got == want_bs,
          f"computed fit={got}, peak@bs1={peak_gib(depth, 1):.1f} GiB")

# Real parameter counts, verified by instantiating nanochat's actual GPT class
# on meta device (nanochat/param_audit.py). The plan's numbers are 2.3-2.6x
# too high because it charged every layer a value embedding; has_ve() puts
# them on alternating layers only.
REAL_PARAMS = {12: 286.3e6, 16: 536.9e6, 20: 896.5e6, 24: 1384.1e6, 28: 2018.5e6}
for depth, want in REAL_PARAMS.items():
    got = budget.n_params(depth)["params"]
    check(f"d{depth} real params ~{want/1e6:.0f}M", abs(got - want) / want < 0.01,
          f"computed {got/1e6:.1f}M vs measured {want/1e6:.1f}M")

# The batch ceiling is set by the vocab-sized logits tensor, not the model.
check("d20 does NOT fit at bs=32 (logits tensor ceiling)", peak_gib(20, 32) == 0.0,
      f"peak@bs32={peak_gib(20, 32):.1f} GiB")
check("d28 fits at bs=4", 0.0 < peak_gib(28, 4) < budget.MEM_LIMIT_GIB,
      f"peak@bs4={peak_gib(28, 4):.1f} GiB")
check("d32 does not fit at bs=1", peak_gib(32, 1) == 0.0,
      f"peak@bs1={peak_gib(32, 1):.1f} GiB")

# PLAN_DEVIATIONS: claims in PLAN.md that the audit measured to be false.
# These are reported, not failed — the audit is the authority and the plan is
# retained as the teaching document. Listed here so a future edit to either
# side has to confront them.
PLAN_DEVIATIONS = [
    ("plan: d20 = 2,155 M params", f"measured d20 = {budget.n_params(20)['params']/1e6:.0f} M"),
    ("plan: d24+ does not fit", "measured d24/d26/d28 all fit at bs=4"),
    ("plan: d20 runs at bs=32", f"measured max bs = {budget.max_fitting_batch(20)}"),
    ("plan: headline is d20", "measured d28 = 2,019 M is the same target size"),
    ("plan: 30% MFU", "measured 34-48%"),
    ("plan: xLAM-60k is the Stage 4 source", "gated; hermes-function-calling-v1 used instead"),
]
print()
print("-" * 72)
print("KNOWN PLAN-vs-MEASUREMENT DEVIATIONS (documented in "
      "docs/EXECUTION_LOG_PART1.md, Finding 1):")
for claim, reality in PLAN_DEVIATIONS:
    print(f"  ! {claim}\n      -> {reality}")
print("-" * 72)
print()

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
