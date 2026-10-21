"""
Probe: what does the d28b-rl100 base model actually emit on count_chars L3?
Diagnoses the step-24+ all-zero reward wall in the Stage 6 pilot.

Renders the exact training prompt (render_for_completion), samples like the RL
loop does (temperature 1.0, 8 passes), then reports per-pass:
  - decoded completion (truncated display)
  - extract_code result
  - discovered fn name
  - sandbox verdict + error info from run_hidden_tests
No training, no checkpoint writes. Read-only on the store.
"""
import sys
sys.path.insert(0, "/data/rowel/nanochat")

import torch
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.agentic import (AgenticSandbox, extract_code, discover_fn_name,
                           run_hidden_tests)

device = torch.device("cuda")
model, tokenizer, meta = load_model("sft", device, phase="eval",
                                    model_tag="d28b-rl100", step=100)
engine = Engine(model, tokenizer)

task = AgenticSandbox(levels=(3,), instances_per_level=200, seed=1234)
# count_chars is family_idx 0 in sorted(FAMILIES) -> physical indices 0..199
# (Task.__getitem__ applies start; we want the train-side instances, not the
# 5% tail holdout, so take indices 0..7 of the L3 block)
probe_indices = [0, 1, 2, 3, 4, 5, 6, 7]

for idx in probe_indices:
    conv = task[idx]
    fam, lvl = conv["family"], conv["level"]
    tokens = tokenizer.render_for_completion(conv)
    prefix_length = len(tokens)
    generated, masks = engine.generate_batch(
        tokens, num_samples=8, max_tokens=512, temperature=1.0, top_k=None)
    print(f"\n=== idx {idx} | {fam}@L{lvl} ===")
    print(f"prompt: {conv['messages'][0]['content'][:160]}")
    for p, sample_tokens in enumerate(generated):
        gen = sample_tokens[prefix_length:]
        text = tokenizer.decode(gen)
        code = extract_code(text)
        fn = discover_fn_name(code) if code else None
        if code and "def " in code:
            reward, info = run_hidden_tests(code, conv["tests"])
        else:
            reward, info = 0, {"fn": None, "error": "no code extracted"}
        hit_end = (gen[-1] == tokenizer.encode_special("<|assistant_end|>")
                   if len(gen) > 0 else False)
        print(f"  pass {p}: reward={reward} fn={fn} len={len(gen)} "
              f"end_tok={hit_end} err={info.get('error')!r} "
              f"timeout={info.get('timeout')}")
        if p == 0:
            disp = text.replace("\n", "\\n")[:300]
            print(f"    completion[0]: {disp}")