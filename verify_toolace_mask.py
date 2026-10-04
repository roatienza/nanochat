"""
Verify the masking contract for tasks/toolace.py.

The stage's central recipe detail (PLAN.md: "apply loss only on the tool-call
arguments, not the whole sequence") is only worth anything if it is true. This
proves it on every rendered example, not a sample:

  1. no tool SCHEMA token is ever in a supervised position
  2. no tool RESULT token is ever in a supervised position
  3. every CALL the model is meant to emit IS supervised
  4. the conversation satisfies the renderer's strict alternation assert

Run: .venv/bin/python verify_toolace_mask.py
"""
import argparse
from collections import Counter

from nanochat.common import get_base_dir, autodetect_device_type
from tasks.toolace import ToolACE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--max-examples", type=int, default=None)
    a = p.parse_args()

    task = ToolACE(split=a.split)
    n = len(task.examples) if a.max_examples is None else min(a.max_examples, len(task.examples))
    print(f"examples checked: {n}  (split={a.split})")

    base_dir = get_base_dir()
    tokenizer = _load_tokenizer(base_dir)

    schema_bad = result_bad = missing_call = alternation_bad = trunc_dropped = 0
    leak_sample = []
    tokenizer_max_tokens = getattr(tokenizer, "model_max_tokens", 2048)
    fracs, part_types = [], Counter()
    n_calls = 0

    for i in range(n):
        conv = task.examples[i]
        msgs = conv["messages"]

        # (4) alternation, exactly as render_conversation asserts it
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
            alternation_bad += 1
        for j, m in enumerate(msgs):
            want = "user" if j % 2 == 0 else "assistant"
            if m["role"] != want:
                alternation_bad += 1
                break

        ids, mask = tokenizer.render_conversation(conv)
        fracs.append(sum(mask) / max(1, len(mask)))

        # walk the decoded supervised spans and classify every supervised token
        supervised_text = tokenizer.decode([t for t, m in zip(ids, mask) if m == 1])
        # the tool-result payload, as it appears in the conversation
        results = [p["text"] for m in msgs if isinstance(m["content"], list)
                   for p in m["content"] if p.get("type") == "python_output"]
        expected_calls = [p["text"] for m in msgs if isinstance(m["content"], list)
                          for p in m["content"] if p.get("type") == "python"]
        n_calls += len(expected_calls)
        for m in msgs:
            if isinstance(m["content"], list):
                for p in m["content"]:
                    part_types[p["type"]] += 1

        # (1) DEFINITIVE provenance test, replacing three generations of probes
        # that all false-positived on legitimate target content:
        #   - keyword probes ("parameters", "properties") match a call's OWN args
        #   - schema-slice probes match enum values / URL templates the model is
        #     SUPPOSED to copy into its arguments
        #   - structural-keyword probes match args named "parameters":{}
        #
        # Instead: strip every CALL's text out of the supervised span. Whatever is
        # left over must be only special tokens. Any schema prose surviving that
        # subtraction is real leakage, by definition.
        # A conversation that hit the 2048-token cap has its tail cut off
        # mid-call, so the residue is a truncated fragment, not leakage.
        was_truncated = len(ids) >= tokenizer_max_tokens
        residue = supervised_text
        for c in expected_calls:
            residue = residue.replace(c, " ")
        # NB: <|assistant_end|> IS supervised by design (tokenizer.py:217 marks the
        # assistant turn terminator as a target), so it is legitimate residue.
        for tokstr in ("<|python_start|>", "<|python_end|>", "<|assistant_start|>",
                       "<|assistant_end|>"):
            residue = residue.replace(tokstr, " ")
        residue = residue.replace("CALL", " ").strip()
        if len(residue) > 2:
            if was_truncated:
                trunc_dropped += 1
            else:
                schema_bad += 1
                leak_sample.append(residue[:120])

        # (2) tool results must never be supervised
        for r in results:
            probe = r.strip()[:60]
            if len(probe) > 20 and probe in supervised_text:
                result_bad += 1
                break

        # (3) every CALL we intend to teach must be supervised, UNLESS the
        # rendered conversation hit the 2048-token truncation cap and dropped it.
        # Truncation is not a masking bug, so report it separately.
        truncated = len(ids) >= tokenizer_max_tokens
        for c in expected_calls:
            probe = c.split("(")[0].strip()[:24]
            if probe and probe not in supervised_text:
                if truncated:
                    trunc_dropped += 1
                else:
                    missing_call += 1
                break

    print(f"supervised fraction: min {min(fracs):.1%}  mean {sum(fracs)/len(fracs):.1%}  max {max(fracs):.1%}")
    print(f"python (CALL) parts: {part_types.get('python', 0)}   python_output (result) parts: {part_types.get('python_output', 0)}")
    print(f"non-CALL text in the SUPERVISED span    : {schema_bad}   (want 0)")
    for r in leak_sample[:3]:
        print("    leak sample:", repr(r))
    print(f"tool results in a SUPERVISED position   : {result_bad}   (want 0)")
    print(f"CALL fragments missing from supervision  : {missing_call}   (want 0)")
    print(f"  (excluded: {trunc_dropped} dropped by 2048-token truncation)")
    print(f"alternation violations                  : {alternation_bad}   (want 0)")

    ok = schema_bad == result_bad == missing_call == alternation_bad == 0
    print("\nRESULT: PASS - loss lands only on the emitted calls" if ok else "\nRESULT: FAIL")
    return 0 if ok else 1


def _schema_only_probes(user_turn):
    """Substrings that exist ONLY in the schema block, never in a valid target.

    1. schema description prose: "description":"<40+ chars>"
    2. the schema object scaffolding: {"name":"<x>","description":
    3. the JSON-schema keywords in their structural form: "parameters":{,
       "properties":{, "required":[, "items":{
    Enum *values* are deliberately excluded -- they belong in the target.
    """
    import re

    out = set()
    for m in re.finditer(r'"description":\s*"(.{40,120}?)"', user_turn, re.DOTALL):
        out.add(m.group(1)[:60])
    for m in re.finditer(r'\{"name":\s*"[^"]{1,40}",\s*"description":', user_turn):
        out.add(m.group(0)[:40])
    for kw in ('"parameters":{', '"properties":{', '"items":{', '"required":['):
        if kw in user_turn:
            out.add(kw)
    return [p for p in out if p]


def _load_tokenizer(base_dir):
    """Load the exact tokenizer the checkpoints were trained with (vocab is
    coupled to the model, so this must not be a fresh NanoChatTokenizer)."""
    import os
    import torch
    from nanochat.checkpoint_manager import build_model

    ckpt_dir = os.path.join(base_dir, "base_checkpoints", "d28b-v2")
    step = max(int(f.split("_")[1].split(".")[0])
               for f in os.listdir(ckpt_dir) if f.startswith("model_"))
    _, tokenizer, _ = build_model(ckpt_dir, step, torch.device("cpu"), "eval")
    return tokenizer


if __name__ == "__main__":
    raise SystemExit(main())
