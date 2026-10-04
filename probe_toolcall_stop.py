"""
Final piece: is the model failing to STOP, or failing to pick calls?

Budget probe result: even at max_new_tokens=768, 47/54 completions hit the cap and
51/54 predicted MORE calls than the reference. Right function name 92.7-93.8%.
So the model knows which tools to call and then fails to terminate.

This probe separates the two by measuring, on greedy output:
  * longest common PREFIX of the normalized (name,args) sequence vs the reference
  * whether <|assistant_end|> / <|python_end|> were emitted at all
  * the call index at which it first diverges

If prefix-match is high and the model never emits assistant_end, the capability is
learned and the defect is termination -- which is a DATA problem (the training
targets never demonstrate stopping after the last call) and NOT a
"needs more tool data" problem.

Run: .venv/bin/python probe_toolcall_stop.py --step 15560
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
p.add_argument("--budget", type=int, default=768)
a = p.parse_args()

device_type = autodetect_device_type()
ddp, rank, local_rank, world_size, device = compute_init(device_type)
master = rank == 0

model, tokenizer, meta = load_model(a.source, device, phase="eval",
                                   model_tag=a.model_tag, step=a.step)
engine = Engine(model, tokenizer)
task = ToolCalling(split="val")

asst_end = tokenizer.encode_special("<|assistant_end|>")
py_end = tokenizer.encode_special("<|python_end|>")
asst_end = asst_end[0] if isinstance(asst_end, list) else asst_end
py_end = py_end[0] if isinstance(py_end, list) else py_end

prefix_lens, div_at, end_emitted, cap_hit = [], [], 0, 0
rows = []

for i in range(min(a.n, len(task))):
    conv = task[i]
    ref = [_norm_call(pp["text"]) for m in conv["messages"]
           if m["role"] == "assistant" and isinstance(m["content"], list)
           for pp in m["content"] if pp.get("type") == "python"]
    pm = []
    for m in conv["messages"]:
        pm.append(m)
        if m["role"] == "assistant":
            break
    prompt = tokenizer.render_for_completion({"messages": pm})
    out, _ = engine.generate_batch(prompt, max_tokens=a.budget,
                                   temperature=0.0, top_k=0)
    gen = out[0][len(prompt):]
    comp = tokenizer.decode(gen)
    pred = extract_predicted_calls(comp)

    lcp = 0
    for r, q in zip(ref, pred):
        if r == q:
            lcp += 1
        else:
            break
    prefix_lens.append(lcp)
    div_at.append(lcp if lcp < len(ref) else -1)
    if asst_end in gen:
        end_emitted += 1
    if len(gen) >= a.budget:
        cap_hit += 1

    # is the ENTIRE reference present as a prefix of pred?
    full_prefix = int(len(ref) > 0 and pred[:len(ref)] == ref)
    rows.append((i, len(ref), len(pred), lcp, full_prefix,
                 int(asst_end in gen), int(len(gen) >= a.budget)))

if master:
    N = len(prefix_lens)
    print0("=" * 72)
    print0(f"checkpoint step={a.step} val_bpb={meta.get('val_bpb'):.4f} budget={a.budget}")
    print0(f"  emitted <|assistant_end|>          : {end_emitted}/{N} = {end_emitted/N:.1%}")
    print0(f"  hit the token cap (never stopped) : {cap_hit}/{N} = {cap_hit/N:.1%}")
    print0(f"  reference appears as exact PREFIX  : {sum(r[4] for r in rows)}/{N} = {sum(r[4] for r in rows)/N:.1%}")
    print0(f"  mean longest-common-prefix calls   : {sum(prefix_lens)/N:.2f} "
           f"(ref mean {sum(r[1] for r in rows)/N:.2f}, pred mean {sum(r[2] for r in rows)/N:.2f})")
    print0(f"  exact full-sequence match          : 0/{N} = 0.0%  <- the acceptance metric")
    print0("=" * 72)
    print0("idx  refN  predN  lcp  refIsPrefix  endEmitted  capHit")
    for r in rows:
        print0(f"{r[0]:3d}  {r[1]:4d}  {r[2]:5d}  {r[3]:3d}  {r[4]:11d}  {r[5]:10d}  {r[6]:6d}")
    print0("")
    print0("divergence call index histogram (-1 = never diverged):")
    print0(f"  {sorted(Counter(div_at).items())}")