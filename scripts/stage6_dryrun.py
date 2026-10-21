"""Pre-launch dry run of the Stage 6 RL path on GPU (fast) — one rollout reward.

Proves: load_model("sft", d28b-rl100) -> Engine -> AgenticSandbox instance ->
generate -> reward. No optimizer step. ~2 min on GPU vs ~10+ min on CPU.

Usage: run via launch wrapper; writes nothing outside logs/.
"""
import sys
sys.path.insert(0, "/data/rowel/nanochat")

import torch
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.agentic import AgenticSandbox
from tasks.common import TaskSequence

levels = (1, 2, 3)
ipl = 200
train_ipl = 190


def _make_task(train=True):
    task = AgenticSandbox(levels=levels, instances_per_level=ipl)
    slices = []
    for f in range(len(task.family_names)):
        for ls in range(len(levels)):
            base = f * len(levels) * ipl + ls * ipl
            if train:
                slices.append(AgenticSandbox(levels=levels, instances_per_level=ipl,
                                             start=base, stop=base + train_ipl))
            else:
                slices.append(AgenticSandbox(levels=levels, instances_per_level=ipl,
                                             start=base + train_ipl, stop=base + ipl))
    return TaskSequence(slices)


device = torch.device("cuda")
model, tok, meta = load_model("sft", device, phase="eval", model_tag="d28b-rl100", step=100)
engine = Engine(model, tok)
train_task = _make_task(True)

# one instance per family, one greedy sample, report reward
for fam_idx, fam in enumerate(sorted(set(c["family"] for c in
                                         (train_task[i] for i in range(0, len(train_task), 190))))):
    pass
# simpler: walk the 15 (family, level) blocks, take instance 0 of each
n_fam = len(train_task.tasks[0].family_names) if hasattr(train_task, "tasks") else 5
results = []
for f in range(5):
    for ls in range(3):
        idx = f * 3 * train_ipl + ls * train_ipl + 0
        conv = train_task[idx]
        tokens = tok.render_for_completion(conv)
        out, masks = engine.generate_batch(tokens, num_samples=1, max_tokens=400,
                                           temperature=0.0, top_k=50)
        text = tok.decode(out[0][len(tokens):])
        r = train_task.reward(conv, text)
        results.append((conv["family"], conv["level"], r))
        print(f"{conv['family']} L{conv['level']}: reward={r}", flush=True)

n_pass = sum(r for _, _, r in results)
print(f"TOTAL: {n_pass}/15 instances solved at temperature 0 (greedy)")
print("Per-family:", {f: sum(r for ff, l, r in results if ff == f) for f in sorted(set(f for f, _, _ in results))})