"""
WIDE acceptance probe for Stage 4 v2 (plan option (1)).

Why this exists
---------------
Every probe so far ran on ToolCalling(split="val") with n=54. That is the ENTIRE
hermes val split (train=1036, val=54 -- val_fraction=0.05 of 1090). So n=54 is not
a sample, it is the whole population, and 2/54 = 3.7% was 2 examples. Nothing below
~8% is resolvable at that n, which is why I kept having to retract numbers.

Two questions the narrow probe could not answer:
  Q1. Is exact match really 0.0%, or is that noise from a 54-example population?
  Q2. Termination was cured (94.4%/98.1% stop before cap). But the model then
      emits 5.81 calls against a 2.20 reference (hermes) -- OVER-calling. At
      budget=512 a call is truncated mid-JSON and extract_predicted_calls() parses
      the fragment as a bogus extra call. So "over-call" and "budget too small"
      are still confounded: maybe with budget=1024 the extra calls stop being
      truncated garbage and become genuine over-calls, or maybe exact match rises.

Methodology corrections carried over from probe_term_real.py:
  * Termination is measured as "generation returned before the token cap", NOT by
    testing for <|assistant_end|> in the output -- generate_batch() strips terminal
    tokens by construction (engine.py:290-294), so that test can never be true.
  * Comparison is on (name, args) via _norm_call, never on formatting.

Run:
  .venv/bin/python probe_wide.py --task hermes --n 54 --budgets 512,1024 --shard 0 --num-shards 1
"""
import argparse
import json
import time
from collections import Counter

import torch

from nanochat.common import autodetect_device_type, compute_init, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.toolcalling import ToolCalling, _norm_call, extract_predicted_calls
from tasks.toolace import ToolACE

p = argparse.ArgumentParser()
p.add_argument("--source", default="sft")
p.add_argument("--model-tag", default="d28b-v2")
p.add_argument("--step", type=int, default=15548)
p.add_argument("--task", default="hermes", choices=["hermes", "toolace"])
p.add_argument("--n", type=int, default=54)
p.add_argument("--budgets", default="512,1024")
p.add_argument("--shard", type=int, default=0)
p.add_argument("--num-shards", type=int, default=1)
p.add_argument("--out", default=None)
p.add_argument("--dump", type=int, default=0, help="dump N raw completions per budget")
a = p.parse_args()

ddp, rank, local_rank, world_size, device = compute_init(autodetect_device_type())
master = rank == 0
print0(f"task={a.task} tag={a.model_tag} step={a.step} budgets={a.budgets} shard {a.shard+1}/{a.num_shards}")

model, tokenizer, meta = load_model(a.source, device, phase="eval",
                                    model_tag=a.model_tag, step=a.step)
engine = Engine(model, tokenizer)
if master:
    print0(f"checkpoint val_bpb={meta.get('val_bpb')}")

task_obj = ToolCalling(split="val") if a.task == "hermes" else ToolACE(split="val")

budgets = [int(b) for b in a.budgets.split(",")]

# Build (prompt, reference) pairs once; shard deterministically so the shards
# partition the population exactly with no overlap.
items = []
for i in range(min(a.n, len(task_obj))):
    conv = task_obj[i]
    ref = [_norm_call(p["text"]) for m in conv["messages"]
           if m["role"] == "assistant" and isinstance(m["content"], list)
           for p in m["content"] if p.get("type") == "python"]
    pm = []
    for m in conv["messages"]:
        pm.append(m)
        if m["role"] == "assistant":
            break
    prompt = tokenizer.render_for_completion({"messages": pm})
    items.append((prompt, ref))

items = items[a.shard::a.num_shards]

records = []
t_start = time.time()
for budget in budgets:
    for idx, (prompt, ref) in enumerate(items):
        out, _ = engine.generate_batch(prompt, max_tokens=budget,
                                       temperature=0.0, top_k=0)
        gen = out[0][len(prompt):]
        comp = tokenizer.decode(gen)
        pred = extract_predicted_calls(comp)
        hit_cap = len(gen) >= budget
        rec = {
            "i": idx,
            "budget": budget,
            "exact": int(pred == ref and len(ref) > 0),
            "stopped": int(not hit_cap),
            "n_pred": len(pred),
            "n_ref": len(ref),
            "fn": sum(1 for r, q in zip(ref, pred) if q[0] == r[0]) / max(1, len(ref)),
            "prefix": 0,
            "first_bad": None,
        }
        # longest common prefix of the normalized (name,args) sequence
        k = 0
        while k < len(ref) and k < len(pred) and ref[k] == pred[k]:
            k += 1
        rec["prefix"] = k / max(1, len(ref))
        if rec["first_bad"] is None and k < len(ref):
            rec["first_bad"] = {"ref": ref[k], "pred": pred[k] if k < len(pred) else None}
        records.append(rec)

if master and a.dump:
    print0("")

dt = time.time() - t_start
if a.out:
    with open(a.out, "w") as f:
        json.dump(records, f)
    print0(f"wrote {len(records)} records -> {a.out}  ({dt:.0f}s)")

# Aggregate. Rank 0 only sees its own shard, so aggregate across shards if files
# were written by a launcher.
if master:
    print0("=" * 74)
    for budget in budgets:
        rs = [r for r in records if r["budget"] == budget]
        n = len(rs)
        if n == 0:
            continue
        exact = sum(r["exact"] for r in rs)
        stopped = sum(r["stopped"] for r in rs)
        fn = sum(r["fn"] for r in rs) / n
        prefix = sum(r["prefix"] for r in rs) / n
        over = sum(1 for r in rs if r["n_pred"] > r["n_ref"])
        cnt_ok = sum(1 for r in rs if r["n_pred"] == r["n_ref"])
        npred = sum(r["n_pred"] for r in rs) / n
        nref = sum(r["n_ref"] for r in rs) / n
        cap = sum(1 for r in rs if not r["stopped"])
        print0(f"[{a.task}] budget={budget}  n={n}")
        print0(f"  exact match        : {exact}/{n} = {exact/n:6.1%}")
        print0(f"  count_ok           : {cnt_ok}/{n} = {cnt_ok/n:6.1%}")
        print0(f"  STOPPED before cap : {stopped}/{n} = {stopped/n:6.1%}")
        print0(f"  hit cap            : {cap}/{n} = {cap/n:6.1%}")
        print0(f"  fn acc             : {fn:6.1%}")
        print0(f"  call-prefix acc    : {prefix:6.1%}")
        print0(f"  mean pred {npred:.2f} vs ref {nref:.2f}   over-call {over}/{n}")
