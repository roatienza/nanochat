"""Is the repetition loop a DECODING pathology or a capability gap?

If the reference call sequence already appears verbatim as a prefix of the
generation, the model KNOWS the answer and greedy decoding is simply falling
into a loop -> a decoding fix (repetition penalty) is enough, no retrain.
If the prefix is wrong, it is a capability/data problem -> retrain.
"""
import argparse
import torch
from nanochat.common import autodetect_device_type, compute_init, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.toolcalling import ToolCalling, _norm_call, extract_predicted_calls

p = argparse.ArgumentParser()
p.add_argument("--model-tag", default="d28b-v2"); p.add_argument("--step", type=int, default=15560)
p.add_argument("--budget", type=int, default=512); p.add_argument("--n", type=int, default=16)
a = p.parse_args()
ddp, rank, _, _, device = compute_init(autodetect_device_type())
model, tok, meta = load_model("sft", device, phase="eval", model_tag=a.model_tag, step=a.step)
engine = Engine(model, tok)

def run(gen_kwargs):
    exact = prefix = 0; lcp = []; refn = []; predn = []
    task = ToolCalling(split="val"); n = min(a.n, len(task))
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
        out, _ = engine.generate_batch(prompt, max_tokens=a.budget, **gen_kwargs)
        pred = extract_predicted_calls(tok.decode(out[0][len(prompt):]))
        k = 0
        for r, q in zip(ref, pred):
            if r == q: k += 1
            else: break
        lcp.append(k); refn.append(len(ref)); predn.append(len(pred))
        exact += int(pred == ref and ref); prefix += int(len(ref) > 0 and pred[:len(ref)] == ref)
    return exact/n, prefix/n, sum(lcp)/n, sum(refn)/n, sum(predn)/n

print0("=" * 74)
print0("decoding mode                exact   refIsPrefix  meanLCP  refN  predN")
for label, kw in [("greedy (t=0, top_k=0)", dict(temperature=0.0, top_k=0)),
                  ("greedy + top_k=1",       dict(temperature=0.0, top_k=1))]:
    e, pf, l, rn, pn = run(kw)
    print0(f"  {label:26s} {e:5.1%}   {pf:9.1%}   {l:6.2f}  {rn:4.2f}  {pn:5.2f}")
