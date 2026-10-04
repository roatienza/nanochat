"""
Follow-up to probe_toolcall_tf.py: TF exact match is 100%, greedy is 0%. Why?

The probe showed teacher-forced exact match 54/54 while greedy scored 0/54. Two
candidate explanations, both testable, and they imply different fixes:

  (A) CONTEXT BUDGET. The acceptance test generates with max_new_tokens=256. Long
      calls (a 13-day thermostat schedule) get cut off mid-JSON, and
      extract_predicted_calls() happily parses the truncated fragment as a bogus
      call -> the list never equals the reference. This is a HARNESS bug, and it
      would make a perfectly good model look broken.
  (B) CALL-COUNT COLLAPSE. The model emits the right call 1, then repeats it
      instead of advancing to call 2 (examples 0 and 3: gold ' record' -> pred
      ' get', gold ' set' -> pred ' activate').

Measure both: the distribution of len(pred) vs len(ref), and greedy exact match
as a function of the token budget.

Run: .venv/bin/python probe_toolcall_budget.py --step 15560
"""
import argparse
from collections import Counter

import torch

from nanochat.common import autodetect_device_type, compute_init, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.toolcalling import ToolCalling, _norm_call, extract_predicted_calls

p = argparse.ArgumentParser()
p.add_argument("--source", default="sft")
p.add_argument("--model-tag", default="d28b-v2")
p.add_argument("--step", type=int, default=None)
p.add_argument("--n", type=int, default=54)
p.add_argument("--budgets", default="256,384,512,768")
a = p.parse_args()

device_type = autodetect_device_type()
ddp, rank, local_rank, world_size, device = compute_init(device_type)
master = rank == 0

model, tokenizer, meta = load_model(a.source, device, phase="eval",
                                   model_tag=a.model_tag, step=a.step)
engine = Engine(model, tokenizer)
task = ToolCalling(split="val")
budgets = [int(b) for b in a.budgets.split(",")]

refs, prompts = [], []
for i in range(min(a.n, len(task))):
    conv = task[i]
    refs.append([_norm_call(pp["text"]) for m in conv["messages"]
                 if m["role"] == "assistant" and isinstance(m["content"], list)
                 for pp in m["content"] if pp.get("type") == "python"])
    pm = []
    for m in conv["messages"]:
        pm.append(m)
        if m["role"] == "assistant":
            break
    prompts.append(tokenizer.render_for_completion({"messages": pm}))

# how long is the gold answer, in tokens?
gold_len = []
for conv in list(task)[:len(refs)]:
    ids, _ = tokenizer.render_conversation(conv)
    gold_len.append(len(ids))
if master:
    print0(f"gold continuation lengths: min {min(gold_len)}, "
           f"median {sorted(gold_len)[len(gold_len)//2]}, max {max(gold_len)}")
    print0(f"ref call counts: {Counter(len(r) for r in refs).most_common()}")
    print0("")

for budget in budgets:
    exact = 0
    count_ok = 0
    fn = 0.0
    truncated = 0
    pred_counts = Counter()
    multi = 0
    for prompt, ref in zip(prompts, refs):
        out, _ = engine.generate_batch(prompt, max_tokens=budget,
                                       temperature=0.0, top_k=0)
        comp = tokenizer.decode(out[0][len(prompt):])
        pred = extract_predicted_calls(comp)
        pred_counts[len(pred)] += 1
        if len(pred) > len(ref):
            multi += 1
        # did generation hit the cap?
        if len(out[0]) - len(prompt) >= budget:
            truncated += 1
        exact += int(pred == ref)
        count_ok += int(len(pred) == len(ref))
        fn += sum(1 for r, q in zip(ref, pred) if q[0] == r[0]) / max(1, len(ref))
    N = len(refs)
    if master:
        print0(f"budget {budget:4d}  exact {exact}/{N} = {exact/N:5.1%}  "
               f"count_ok {count_ok}/{N} = {count_ok/N:5.1%}  "
               f"fn {fn/N:5.1%}  hit_cap {truncated}/{N}  "
               f"more_pred_than_ref {multi}/{N}")
        print0(f"             pred call-count histogram: {pred_counts.most_common()}")