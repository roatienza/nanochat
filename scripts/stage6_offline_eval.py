"""
Stage 6 offline eval — evaluate saved RL checkpoints on the held-out agentic val
split, WITHOUT training. Mirrors the in-run eval exactly:
  - same val_task construction (tail 5% of each (family, level) block, 150 instances)
  - same generation params as the in-run eval: temperature=1.0, top_k=50,
    num_samples=device_batch_size, max_completion_tokens=256
  - same pass@k reduction: pass@k = fraction of instances with >=1 correct in
    first k samples; per-family/level Pass@1 breakdown printed identically.

Usage (one GPU):
  cd /data/rowel/nanochat && NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2 \
    CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m scripts.stage6_offline_eval \
    --steps 120,150,177 --run stage6-offline-eval
"""
import argparse, os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, DummyWandb
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.agentic import AgenticSandbox
from tasks.common import TaskSequence

parser = argparse.ArgumentParser(description="Stage 6 offline checkpoint eval (no training)")
parser.add_argument("--steps", type=str, required=True, help="comma-separated checkpoint steps, e.g. 120,150,177")
parser.add_argument("--model-tag", type=str, default="d28b-rl100")
parser.add_argument("--agentic-instances-per-level", type=int, default=200)
parser.add_argument("--agentic-levels", type=str, default="1,2,3")
parser.add_argument("--agentic-eval-split-frac", type=float, default=0.05)
parser.add_argument("--eval-examples", type=int, default=150)
parser.add_argument("--device-batch-size", type=int, default=2, help="num samples per instance for pass@k")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--out", type=str, default=None, help="optional JSON output path")
args = parser.parse_args()

def make_val_task():
    levels = tuple(int(x) for x in args.agentic_levels.split(","))
    ipl = args.agentic_instances_per_level
    holdout = max(1, int(ipl * args.agentic_eval_split_frac))
    train_ipl = ipl - holdout
    task = AgenticSandbox(levels=levels, instances_per_level=ipl)
    slices = []
    n_fam = len(task.family_names)
    for f in range(n_fam):
        for ls in range(len(levels)):
            base = f * len(levels) * ipl + ls * ipl
            slices.append(AgenticSandbox(levels=levels, instances_per_level=ipl,
                                         start=base + train_ipl, stop=base + ipl))
    return TaskSequence(slices)

def run_agentic_eval(task, tokenizer, engine, max_examples, num_samples,
                     max_completion_tokens, temperature, top_k):
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(max_examples):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        generated_token_sequences, masks = engine.generate_batch(
            tokens, num_samples=num_samples, max_tokens=max_completion_tokens,
            temperature=temperature, top_k=top_k)
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})
        record = {"idx": idx, "outcomes": outcomes,
                  "family": conversation.get("family", "?"),
                  "level": conversation.get("level", "?")}
        yield record

val_task = make_val_task()
steps = [int(s) for s in args.steps.split(",")]
all_results = {}

for step in steps:
    print0(f"=== Evaluating checkpoint step {step} ===")
    model, tokenizer, meta = load_model("rl", torch.device("cuda"), phase="eval",
                                        model_tag=args.model_tag, step=step)
    engine = Engine(model, tokenizer)
    records = list(run_agentic_eval(val_task, tokenizer, engine,
                                    max_examples=args.eval_examples,
                                    num_samples=args.device_batch_size,
                                    max_completion_tokens=args.max_new_tokens,
                                    temperature=args.temperature, top_k=args.top_k))
    n = len(records)
    passk = {}
    for k in range(1, args.device_batch_size + 1):
        passk[k] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records) / n
    print_passk = ", ".join(f"Pass@{k}: {passk[k]:.4f}" for k in sorted(passk))
    print0(f"Step {step} | {print_passk}")
    fam_stats = {}
    for r in records:
        key = (r.get("family", "?"), r.get("level", "?"))
        s = fam_stats.setdefault(key, [0, 0])
        s[1] += 1
        s[0] += any(o["is_correct"] for o in r["outcomes"][:1])
    fam_str = " | ".join(f"{f}@L{l}:{c/n2:.2f}" for (f, l), (c, n2) in sorted(fam_stats.items()))
    print0(f"Step {step} | per-family/level Pass@1: {fam_str}")
    all_results[str(step)] = {"passk": {str(k): v for k, v in passk.items()},
                              "per_family": {f"{f}@L{l}": c / n2 for (f, l), (c, n2) in sorted(fam_stats.items())}}
    # free GPU memory before next checkpoint load
    del model, engine
    torch.cuda.empty_cache()

print0(json.dumps(all_results, indent=2))
if args.out:
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print0(f"Wrote {args.out}")
