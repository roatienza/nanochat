"""
Stage 4 diagnosis: is the 0.0% exact-match a DATA problem, a TRAINING problem,
or an INFERENCE problem?

The question "data or training?" has a third answer that everyone forgets, and it
is the cheapest to test:

  * teacher-forced accuracy on the gold call tokens is HIGH  => the model DID learn
    from the data. The data is fine, the training worked, and the fault is at
    inference (decoding, attention, prompt surface, stopping).
  * teacher-forced accuracy is LOW => the model never learned the mapping. That is
    either data (wrong/unlearnable targets, too few examples) or training (not
    enough steps, LR, or masking that discarded the signal).

Greedy exact-match is 0.0% for BOTH the Stage 3 and Stage 4 checkpoints, and 8.6x
more tool data moved it by zero. That pattern is much more consistent with an
inference-side fault than with "the data is bad", but it must be measured, not
argued.

Method
------
For each held-out val example we render the conversation exactly as training does
(render_conversation -> ids, mask), teacher-force the whole thing through the
model in one forward pass, and read off:
  * accuracy over ALL supervised tokens (mask == 1)
  * accuracy over the CALL tokens specifically (the python part) -- this is the
    capability the acceptance test measures
  * accuracy over argument *value* tokens -- where the observed failures live
    (dropped array items, "08:00" -> 8)

Then we greedy-decode the same examples with the same prompt the acceptance test
uses and print the two side by side. If TF is high and greedy is 0, the fault is
inference. If TF is also low, it is data/training.

Run: .venv/bin/python probe_toolcall_tf.py --source sft --step 15560
"""
import argparse
import json

import torch

from nanochat.common import autodetect_device_type, compute_init, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.tokenizer import get_tokenizer
from tasks.toolcalling import ToolCalling, _norm_call, extract_predicted_calls

p = argparse.ArgumentParser()
p.add_argument("--source", default="sft", choices=["base", "sft", "rl"])
p.add_argument("--model-tag", default="d28b-v2")
p.add_argument("--step", type=int, default=None)
p.add_argument("--n", type=int, default=54)
p.add_argument("--show", type=int, default=4)
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

# Which token ids delimit a CALL? python_start / python_end.
py_start = tokenizer.encode_special("<|python_start|>")[0] \
    if isinstance(tokenizer.encode_special("<|python_start|>"), list) \
    else tokenizer.encode_special("<|python_start|>")
py_end = tokenizer.encode_special("<|python_end|>")[0] \
    if isinstance(tokenizer.encode_special("<|python_end|>"), list) \
    else tokenizer.encode_special("<|python_end|>")


def in_call(ids, i):
    """Is token i inside a <|python_start|> ... <|python_end|> span?"""
    j = i - 1
    while j >= 0:
        if ids[j] == py_start:
            return True
        if ids[j] == py_end:
            return False
        j -= 1
    return False


agg = {"sup_tok": 0, "sup_ok": 0, "call_tok": 0, "call_ok": 0,
       "n": 0, "tf_exact": 0, "greedy_exact": 0, "tf_fn": 0.0}
per_ex = []

for i in range(n):
    conv = task[i]
    ids, mask = tokenizer.render_conversation(conv)

    # ---- teacher forcing: one forward pass over the gold sequence -------------
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
    pred = logits[0].argmax(-1)  # pred[t] predicts token t+1
    gold_next = torch.tensor(ids[1:], device=device)
    pred_next = pred[:-1]
    correct = (pred_next == gold_next)

    sup_tok = sup_ok = call_tok = call_ok = 0
    first_bad = None
    for t in range(len(ids) - 1):
        if mask[t + 1] != 1:
            continue  # token t+1 is supervised => predicted from position t
        sup_tok += 1
        ok = bool(correct[t])
        sup_ok += ok
        if in_call(ids, t + 1):
            call_tok += 1
            call_ok += ok
            if not ok and first_bad is None:
                first_bad = (tokenizer.decode([ids[t + 1]]),
                             tokenizer.decode([int(pred_next[t])]))

    # TF "exact": does the argmax continuation reproduce every CALL token?
    # NOTE: must decode the PREDICTED ids, not the gold ids -- decoding gold ids
    # just restates the reference and would trivially score 100%.
    spans, cur, open_ = [], [], False
    for t in range(len(ids)):
        if ids[t] == py_start:
            cur, open_ = [], True
        elif ids[t] == py_end and open_:
            spans.append(cur)
            open_ = False
        elif open_:
            cur.append(t)  # POSITION, not token id -- we index pred_next by it
    tf_calls = []
    for sp in spans:
        # token at index t is predicted from position t-1, since pred_next[i]
        # is the argmax that produced token i+1
        tf_calls.append(_norm_call(tokenizer.decode(
            [int(pred_next[t - 1]) for t in sp if t >= 1])))

    ref = [_norm_call(pp["text"]) for m in conv["messages"]
           if m["role"] == "assistant" and isinstance(m["content"], list)
           for pp in m["content"] if pp.get("type") == "python"]

    # ---- greedy decode, identical prompt to the acceptance test ---------------
    prompt_msgs = []
    for m in conv["messages"]:
        prompt_msgs.append(m)
        if m["role"] == "assistant":
            break
    prompt = tokenizer.render_for_completion({"messages": prompt_msgs})
    out, _ = engine.generate_batch(prompt, max_tokens=a.max_new_tokens,
                                   temperature=0.0, top_k=0)
    completion = tokenizer.decode(out[0][len(prompt):])
    greedy = extract_predicted_calls(completion)

    agg["sup_tok"] += sup_tok
    agg["sup_ok"] += sup_ok
    agg["call_tok"] += call_tok
    agg["call_ok"] += call_ok
    agg["n"] += 1
    tf_exact = int(tf_calls == ref)
    g_exact = int(greedy == ref)
    agg["tf_exact"] += tf_exact
    agg["greedy_exact"] += g_exact
    agg["tf_fn"] += sum(1 for r, q in zip(ref, greedy) if q[0] == r[0]) / max(1, len(ref))

    per_ex.append((i, sup_tok, sup_ok / max(1, sup_tok), call_tok,
                   call_ok / max(1, call_tok), tf_exact, g_exact, first_bad))
    if master and i < a.show:
        print0(f"--- example {i} --- TF={tf_exact} GREEDY={g_exact}")
        print0(f"  ref   : {ref[:2]}")
        print0(f"  tf    : {tf_calls[:2]}")
        print0(f"  greedy: {[f'{nm} {ar}' for nm, ar in greedy[:2]]}")
        if first_bad:
            print0(f"  first CALL token error: gold={first_bad[0]!r} pred={first_bad[1]!r}")
        print0("")

if master:
    N = max(1, agg["n"])
    print0("=" * 70)
    print0(f"checkpoint: source={a.source} tag={a.model_tag} step={a.step} "
           f"val_bpb={meta.get('val_bpb')}")
    print0(f"examples: {agg['n']}")
    print0(f"  TEACHER-FORCED token acc, all supervised : "
           f"{agg['sup_ok']}/{agg['sup_tok']} = {agg['sup_ok']/max(1,agg['sup_tok']):.1%}")
    print0(f"  TEACHER-FORCED token acc, CALL tokens   : "
           f"{agg['call_ok']}/{agg['call_tok']} = {agg['call_ok']/max(1,agg['call_tok']):.1%}")
    print0(f"  TEACHER-FORCED exact match (all calls)  : {agg['tf_exact']}/{N} = {agg['tf_exact']/N:.1%}")
    print0(f"  GREEDY exact match (acceptance test)     : {agg['greedy_exact']}/{N} = {agg['greedy_exact']/N:.1%}")
    print0(f"  GREEDY right fn name                     : {agg['tf_fn']/N:.1%}")
    print0("=" * 70)
    # per-example table
    print0("idx  supTok  supAcc  callTok  callAcc  TF  GREEDY")
    for (i, st, sa, ct, ca, te, ge, fb) in per_ex:
        print0(f"{i:3d}  {st:6d}  {sa:6.3f}  {ct:7d}  {ca:7.3f}  {te}   {ge}")