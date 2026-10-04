# STATUS_2026-10-05_STAGE4_ROOTCAUSE.md

## Question

Stage 4 should improve tool calling. Is 0.0% exact-match a **data** problem or a
**training** problem?

## Answer: neither — it is a **stop-target defect in the data**. My earlier
"needs more tool data" verdict was wrong.

## Evidence

Three probes, run on the Stage 4 checkpoint (`d28b-v2` @ 15560, val_bpb 0.3143),
on the 54 held-out hermes val examples the acceptance test uses.

| measurement | value |
|---|---|
| Teacher-forced token acc, CALL tokens | **98.1%** (6636/6763) |
| Teacher-forced exact match, all calls | **42.6%** (23/54) |
| Greedy exact match (the acceptance metric) | **0.0%** (0/54) |
| Greedy right function name | **92.7%** |
| `<\|assistant_end\|>` emitted | **0 / 54** |
| Reference is an exact prefix of prediction | **42.6%** |
| Mean predicted calls vs reference | **9.31 vs 2.20** |
| Greedy exact at budget 256 / 384 / 512 / 768 | **0.0% / 0.0% / 0.0% / 0.0%** |

The model learned the tool-calling mapping (98.1% CALL-token accuracy with gold
context). It then fails for one reason: **it never emits `<|assistant_end|>`, so it
keeps making calls forever.** Right tool, right arguments, no terminator.

## Root cause (data, not volume)

1. **84.7%** of training assistant messages that contain a call continue with a
   `python_output` part instead of ending the turn. The model learned
   "call → output → call → output" and there is almost no signal for
   "call → END".
2. Only **81.4%** of training examples end on a supervised `<|assistant_end|>`
   (val: 87.0%). The remainder are cut mid-token (`'ent'`, `'exper'`, `' **'`).
3. The first assistant turn containing a call has part-type sequence
   `('python', 'python_output')` in 330/1036 cases and
   `('python','python','python','python_output','python_output','python_output')` in
   261 — the corpus essentially never shows "single call, then end turn", which is
   exactly what the acceptance test prompts on.

This is why 8.6x more data (Stage 4's 8,957 ToolACE examples vs Stage 3's 1,036
hermes) moved the metric by exactly 0.0: **volume was never the constraint.**

## Correction to my own probe

The first version of `probe_toolcall_tf.py` decoded the **gold** ids inside python
spans, which just restates the reference and scored a meaningless 100%. Fixed to
decode predicted ids at position `t-1`; the honest number is **42.6%**.

## Fix

Data-side, cheap, no retraining of the base needed:
- ensure the final assistant turn of a tool-call example ends with
  `<|assistant_end|>` (pad/close truncated examples rather than cutting at 2048)
- balance the stop target: add single-call-then-end examples so "call → END" is
  as common as "call → output"
- raise the generation cap and terminate on `<|assistant_end|>` in the harness

Then re-run `probe_toolcall_stop.py`; `endEmitted` should go from 0/54 to >90%
before spending any GPU hours on a new SFT run.

## Files

- `probe_toolcall_tf.py` — teacher-forced vs greedy
- `probe_toolcall_budget.py` — exact match vs token budget
- `probe_toolcall_stop.py` — termination behaviour, LCP analysis
- commit `bcf4b18` (nanochat fork roatienza/nanochat)