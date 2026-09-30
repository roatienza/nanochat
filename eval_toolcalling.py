"""
End-to-end check of Stage 4: does the SFT model actually emit tool calls?

This is the acceptance test for the stage. The masking verification proves the
data is correct; this proves the model learned from it. Held-out val examples
only (the split in tasks/toolcalling.py keeps them disjoint from training).

Reports:
  - exact-match accuracy on (function name, arguments)
  - a qualitative dump, because exact-match on 54 examples is noisy and a
    student needs to SEE the failure modes, not just the average

Run: .venv/bin/python eval_toolcalling.py
"""
import argparse
import torch

from nanochat.common import autodetect_device_type, compute_init, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.toolcalling import ToolCalling, _norm_call, extract_predicted_calls

p = argparse.ArgumentParser()
p.add_argument("--model-tag", default="d12")
p.add_argument("--source", default="sft", choices=["base", "sft", "rl"])
p.add_argument("--step", type=int, default=None)
p.add_argument("--n", type=int, default=54)
p.add_argument("--show", type=int, default=5)
p.add_argument("--max-new-tokens", type=int, default=256)
a = p.parse_args()

device_type = autodetect_device_type()
ddp, rank, local_rank, world_size, device = compute_init(device_type)
master = rank == 0

model, tokenizer, meta = load_model(a.source, device, phase="eval",
                                   model_tag=a.model_tag, step=a.step)
engine = Engine(model, tokenizer)
task = ToolCalling(split="val")
n = min(a.n, len(task))
print0(f"Evaluating tool calling on {n} held-out examples\n")

correct = 0
shown = 0
agg = {"exact": 0, "fn": 0.0, "count": 0, "n_ref": 0}
for i in range(n):
    conv = task[i]
    # build the prompt exactly as training did: everything up to and including the
    # first assistant turn, then render_for_completion pops that turn and appends
    # the assistant-start token
    prompt_msgs = []
    for m in conv["messages"]:
        prompt_msgs.append(m)
        if m["role"] == "assistant":
            break
    prompt = tokenizer.render_for_completion({"messages": prompt_msgs})
    try:
        out, _ = engine.generate_batch(prompt, max_tokens=a.max_new_tokens,
                                       temperature=0.0, top_k=0)
        # generate_batch returns prompt+continuation; keep only the continuation
        completion = tokenizer.decode(out[0][len(prompt):])
    except Exception as e:
        print0(f"[{i}] generation failed: {type(e).__name__}: {e}")
        continue

    s = task.score(conv, completion)
    correct += s["exact"]
    for k in ("exact", "count"):
        agg[k] += s[k]
    agg["fn"] += s["fn"]
    agg["n_ref"] += s["n_ref"]

    if shown < a.show:
        shown += 1
        ref = [pp["text"] for m in conv["messages"] if m["role"] == "assistant"
               and isinstance(m["content"], list) for pp in m["content"]
               if pp.get("type") == "python"]
        pred = extract_predicted_calls(completion)
        print0(f"--- example {i} --- {'CORRECT' if s['exact'] else 'WRONG'}")
        print0(f"  ref:  {ref[:2]}")
        print0(f"  pred: {[f'{nm} {ar}' for nm, ar in pred[:2]] or completion[:180]!r}")
        print0("")

if master:
    print0("=" * 64)
    print0(f"held-out tool-calling examples: {n}")
    print0(f"  exact match (name+args) : {agg['exact']}/{n} = {agg['exact']/n:.1%}")
    print0(f"  right function name     : {agg['fn']/max(1,agg['n_ref']):.1%} of {agg['n_ref']} calls")
    print0(f"  right number of calls   : {agg['count']}/{n} = {agg['count']/n:.1%}")
    print0("=" * 64)
