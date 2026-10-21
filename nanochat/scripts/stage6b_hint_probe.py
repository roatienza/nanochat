"""
Stage 6b hint probe — measure whether train-time hints create reward variance
on the 5 frontier blocks for the pilot's final checkpoint (model_000177).

Why: the pilot scored 0/256 samples per step on every frontier block (all-zero
sample groups -> zero advantage -> no gradient). Stage 6b adds train-only hints.
This probe answers, BEFORE committing ~31.5 GPU-hours: does the hint move the
frontier blocks from 0/16 to a mixed pass rate at the training sampling params
(temp 1.0, top_k 50, 16 samples/instance)?

Protocol per frontier block:
  - 5 train-side instances (indices 0..4 of the block's train range; val tail untouched)
  - 16 samples per instance at temp 1.0, top_k 50, max 512 new tokens
    (matches training generation params)
  - pass@16 with and without the hint, plus per-instance detail

Usage (one GPU):
  cd /data/rowel/nanochat && NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2 \
    CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m scripts.stage6b_hint_probe \
    --model-step 177 --out /data/rowel/nanochat_store/v2/stage6b_hint_probe.json
"""
import argparse, os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nanochat.common import print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.agentic import AgenticSandbox
from tasks.common import TaskSequence

parser = argparse.ArgumentParser(description="Stage 6b hint probe (frontier blocks, hint vs no-hint)")
parser.add_argument("--model-tag", type=str, default="d28b-rl100")
parser.add_argument("--model-step", type=int, default=177)
parser.add_argument("--agentic-instances-per-level", type=int, default=200)
parser.add_argument("--agentic-levels", type=str, default="1,2,3")
parser.add_argument("--agentic-eval-split-frac", type=float, default=0.05)
parser.add_argument("--instances-per-block", type=int, default=5)
parser.add_argument("--num-samples", type=int, default=16)
parser.add_argument("--device-batch-size", type=int, default=2, help="samples per generate_batch call (OOM floor at db=2)")
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--out", type=str, default=None, help="optional JSON output path")
args = parser.parse_args()

# Frontier blocks + hints — MUST match scripts/chat_rl_curriculum.py
FRONTIER = [("count_chars", 3), ("format_rows", 3), ("list_ops", 2), ("list_ops", 3), ("math_fns", 3)]
HINTS = {
    ("list_ops", 2):
        "Hint: to rotate a list xs right by k positions cyclically, use "
        "xs[-k % len(xs):] + xs[:-k % len(xs)] — the modulo handles k >= len(xs).",
    ("count_chars", 3):
        "Hint: count each character with s.count(c), then sort by (-count, char) "
        "and take the first k characters of that ordering.",
    ("format_rows", 3):
        "Hint: build the header line first, then one ','.join((name, score, tag)) "
        "style line per tuple — no spaces after commas — and join all lines with "
        "newline characters.",
    ("math_fns", 3):
        "Hint: the number of tilings of a 2 x n board with 2x1 dominoes follows "
        "the Fibonacci recurrence: ways(n) = ways(n-1) + ways(n-2), with ways(1) = 1 "
        "and ways(2) = 2. Iterate, don't recurse.",
    ("list_ops", 3):
        "Hint: walk the list once, extending the current strictly-increasing run "
        "while x is greater than the previous element; otherwise start a new run at "
        "x. Keep the best (longest) run seen; on ties keep the first.",
}

levels = tuple(int(x) for x in args.agentic_levels.split(","))
ipl = args.agentic_instances_per_level
holdout = max(1, int(ipl * args.agentic_eval_split_frac))
train_ipl = ipl - holdout

# Sequence-index -> block layout (matches chat_rl_curriculum.BLOCKS)
probe = AgenticSandbox(levels=levels, instances_per_level=ipl)
BLOCKS = [(probe.family_names[f], levels[ls]) for f in range(len(probe.family_names)) for ls in range(len(levels))]

def block_examples(block, n):
    """First n train-side sequence indices of this block."""
    b = BLOCKS.index(block)
    return [b * train_ipl + i for i in range(n)]

def gen_and_eval(task, conversation, num_samples):
    """Generate num_samples completions, return list of 0/1 correctness."""
    tokens = tokenizer.render_for_completion(conversation)
    prefix_length = len(tokens)
    outcomes = []
    steps = (num_samples + args.device_batch_size - 1) // args.device_batch_size
    for s in range(steps):
        k = min(args.device_batch_size, num_samples - s * args.device_batch_size)
        seqs, _ = engine.generate_batch(
            tokens, num_samples=k, max_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k)
        for st in seqs:
            text = tokenizer.decode(st[prefix_length:])
            outcomes.append(task.evaluate(conversation, text))
    return outcomes

model, tokenizer, meta = load_model("rl", torch.device("cuda"), phase="eval",
                                    model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)
train_task = TaskSequence([
    AgenticSandbox(levels=levels, instances_per_level=ipl,
                   start=f * len(levels) * ipl + ls * ipl,
                   stop=f * len(levels) * ipl + ls * ipl + train_ipl)
    for f in range(len(probe.family_names)) for ls in range(len(levels))
])

results = {}
for block in FRONTIER:
    fam, lvl = block
    print0(f"=== {fam}@L{lvl} ===")
    rec = {"no_hint": [], "hint": []}
    for i in range(args.instances_per_block):
        idx = block_examples(block, args.instances_per_block)[i]
        conv = train_task[idx]
        # no-hint
        oh = gen_and_eval(train_task, conv, args.num_samples)
        rec["no_hint"].append({"idx": idx, "pass16": sum(oh) / len(oh)})
        # hint
        hinted = dict(conv)
        hinted["messages"] = [dict(m) for m in conv["messages"]]
        hinted["messages"][0]["content"] = hinted["messages"][0]["content"] + "\n\n" + HINTS[block]
        oh2 = gen_and_eval(train_task, hinted, args.num_samples)
        rec["hint"].append({"idx": idx, "pass16": sum(oh2) / len(oh2)})
        print0(f"  inst {idx}: no-hint {sum(oh)}/{len(oh)} | hint {sum(oh2)}/{len(oh2)}")
    nh = sum(r["pass16"] for r in rec["no_hint"]) / len(rec["no_hint"])
    h = sum(r["pass16"] for r in rec["hint"]) / len(rec["hint"])
    rec["mean_no_hint"] = nh
    rec["mean_hint"] = h
    rec["lift"] = h - nh
    print0(f"  MEAN {fam}@L{lvl}: no-hint {nh:.3f} | hint {h:.3f} | lift {h-nh:+.3f}")
    results[f"{fam}@L{lvl}"] = rec

print0(json.dumps({k: {"mean_no_hint": v["mean_no_hint"], "mean_hint": v["mean_hint"], "lift": v["lift"]}
                   for k, v in results.items()}, indent=2))
if args.out:
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print0(f"Wrote {args.out}")