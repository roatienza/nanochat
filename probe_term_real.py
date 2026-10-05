"""Correct termination measurement.

generate_batch() *excludes* the terminal token from its output (engine.py:294),
so the earlier probe's `assistant_end in gen` was ALWAYS False by construction --
it could not ever have reported non-zero. Termination must be measured by whether
generation returned before the cap, and by the completion itself.

Also compares hermes val (which Stage 4 was evaluated on) against ToolACE val.
"""
import argparse, json, collections
import torch
from nanochat.common import autodetect_device_type, compute_init, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.toolcalling import ToolCalling, _norm_call, extract_predicted_calls
from tasks.toolace import ToolACE

p = argparse.ArgumentParser()
p.add_argument("--model-tag", default="d28b-v2")
p.add_argument("--step", type=int, default=None)
p.add_argument("--budget", type=int, default=512)
p.add_argument("--n", type=int, default=16)
a = p.parse_args()

ddp, rank, _, _, device = compute_init(autodetect_device_type())
master = rank == 0
model, tok, meta = load_model("sft", device, phase="eval", model_tag=a.model_tag, step=a.step)
engine = Engine(model, tok)
from nanochat.engine import use_calculator
if master: print0(f"checkpoint tag={a.model_tag} step={a.step} val_bpb={meta.get('val_bpb')}")

for tname, task in [("hermes_val", ToolCalling(split="val")), ("toolace_val", ToolACE(split="val"))]:
    exact = stopped = 0; ncalls = []; refn = []; dumps = []
    n = min(a.n, len(task))
    for i in range(n):
        conv = task[i]
        ref = [_norm_call(q["text"]) for m in conv["messages"]
               if m["role"] == "assistant" and isinstance(m["content"], list)
               for q in m["content"] if q.get("type") == "python"]
        pm = []
        for m in conv["messages"]:
            pm.append(m)
            if m["role"] == "assistant": break
        prompt = tok.render_for_completion({"messages": pm})
        out, _ = engine.generate_batch(prompt, max_tokens=a.budget, temperature=0.0, top_k=0)
        gen = out[0][len(prompt):]
        comp = tok.decode(gen)
        pred = extract_predicted_calls(comp)
        hit_cap = len(gen) >= a.budget
        exact += int(pred == ref and len(ref) > 0)
        stopped += int(not hit_cap)
        ncalls.append(len(pred)); refn.append(len(ref))
        if i < 3: dumps.append((i, len(ref), len(pred), hit_cap, comp[:600]))
    if master:
        print0("=" * 74)
        print0(f"[{tname}] n={n}  budget={a.budget}")
        print0(f"  exact match        : {exact}/{n} = {exact/n:.1%}")
        print0(f"  STOPPED before cap : {stopped}/{n} = {stopped/n:.1%}   <- real termination rate")
        print0(f"  mean pred calls {sum(ncalls)/n:.2f} vs ref {sum(refn)/n:.2f}")
        for d in dumps:
            print0(f"  --- ex{d[0]} ref={d[1]} pred={d[2]} hit_cap={d[3]}")
            print0(f"      {d[4]!r}")
