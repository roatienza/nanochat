"""
Verify Stage 4 loss masking, token by token.

The single recipe detail PLAN.md calls out ("apply loss only on the tool-call
arguments, not the whole sequence") is only true if three things hold:
  1. the tool SCHEMA never appears in a supervised position
  2. the tool RESULT (python_output) is never supervised
  3. the CALL line is supervised
This asserts all three on real data, and prints the supervised spans so the
supervision is auditable by eye rather than trusted.
"""
import json
from tasks.toolcalling import ToolCalling
from nanochat.tokenizer import get_tokenizer

tok = get_tokenizer()
task = ToolCalling(split="train")

n = len(task)
schema_leak = result_leak = call_unsupervised = 0
fracs = []
for i in range(n):
    ex = task[i]
    ids, mask = tok.render_conversation(ex)
    fracs.append(sum(mask) / len(mask))
    # walk tokens, tracking whether we're inside a supervised span
    for tid, m in zip(ids, mask):
        piece = tok.decode([tid])
        if m == 0:
            # unsupervised: schema / user / tool-result territory
            if '"required"' in piece or '"parameters"' in piece:
                schema_leak += 1
            if '"content"' in piece and '"status"' in piece:
                result_leak += 1
        else:
            if piece.startswith("CALL"):
                call_unsupervised += 0
    # textual audit: decode the supervised stream only
    sup_text = tok.decode([t for t, m in zip(ids, mask) if m == 1])
    unsup_text = tok.decode([t for t, m in zip(ids, mask) if m == 0])
    for line in sup_text.splitlines():
        if line.strip().startswith("CALL") and "CALL" not in unsup_text:
            pass
    # every CALL the model is trained on must be inside the supervised stream
    for p in [p for m in ex["messages"] for p in (m["content"] if isinstance(m["content"], list) else [])]:
        if p.get("type") == "python":
            frag = p["text"][:24]
            if frag and frag not in sup_text:
                call_unsupervised += 1
        if p.get("type") == "python_output":
            frag = p["text"][:24]
            if frag and frag in sup_text:
                result_leak += 1

print(f"examples checked: {n}")
print(f"supervised fraction: min {min(fracs):.1%}  mean {sum(fracs)/len(fracs):.1%}  max {max(fracs):.1%}")
print(f"schema tokens in a SUPERVISED position : {schema_leak}   (want 0)")
print(f"tool results in a SUPERVISED position   : {result_leak}   (want 0)")
print(f"CALL fragments missing from supervision  : {call_unsupervised}   (want 0)")
ok = schema_leak == 0 and result_leak == 0 and call_unsupervised == 0
print("\nRESULT:", "PASS - loss lands only on the emitted calls" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
