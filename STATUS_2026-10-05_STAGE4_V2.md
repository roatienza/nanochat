# STATUS_2026-10-05_STAGE4_V2.md

## Decision

Stage 4 v2 launched with a data fix. **But the diagnosis I gave you last
message was wrong twice, and the second probe that seemed to confirm it was
itself broken.**

## Retraction 1: "the model never emits assistant_end (0/54)"

That number was an artifact. `Engine.generate_batch` (engine.py:294) *excludes*
terminal tokens from the sequence it returns, so the probe's
`assistant_end in gen` test could never fire. It read 0/54 by construction, for
any model, including a perfect one.

## Retraction 2: "ToolACE lacks the call -> END stop signal"

I measured the corpus directly:

| | hermes | ToolACE |
|---|---|---|
| ends on `<\|assistant_end\|>` | 81.4% | **99.8%** |
| final turn is call-only | 0.0% | **93.2%** |

ToolACE already had the stop signal in overwhelming majority. There was no
missing stop token to add. The fix I proposed (close truncated turns) would have
changed almost nothing.

## What the defect actually is: a repetition loop

Corrected measurement on `d28b-v2` @ 15560, greedy, held-out hermes:

| decoding | exact | ref is exact prefix | mean pred calls |
|---|---|---|---|
| greedy, top_k=0 | 0.0% | 43.8% | 11.12 (ref 1.88) |
| greedy, top_k=1 | 0.0% | 43.8% | 11.12 |

top_k=1 is byte-identical to top_k=0, so this is not a sampling pathology. The
model reproduces the reference calls **verbatim** and then keeps emitting calls
(`get_camera_live_feed {"camera_id":"front_garden"...}` x4). Real termination
rate is 62.5% on hermes and 93.8% on ToolACE — it does stop, just after ~11
calls instead of ~2.

## Root cause

ToolACE trains `call -> python_output -> call -> python_output`. At inference the
environment supplies results, and `eval_toolcalling.py` never shows them, but it
scores *every* python part in the reference. The model is trained to expect a
result after each call, gets none, and re-issues the call.

## The fix (`truncate_to_terminal_call`)

Collapse the final assistant turn to its `python` parts only, so every training
target ends `... call -> <|assistant_end|>`.

Verified on all 8,957 examples (`verify_toolace_mask.py`, PASS):

| | before | after |
|---|---|---|
| final turn call-only | 93.2% | **100.0%** |
| ends on `<\|assistant_end\|>` | 99.8% | 99.8% |
| calls per example | 1.82 | **1.95** |
| schema tokens supervised | 0 | 0 |
| tool results supervised | 0 | 0 |

**Count is preserved, deliberately.** My first implementation truncated to the
*first* call and would have capped the metric at 35.2% — the share of val
references with exactly one call. Caught it before launching and corrected it.

## Run

| | |
|---|---|
| GPUs | **3,4,5,7** (2 and 6 hold other users' jobs) |
| batch | 32,768 total, device-batch 2, grad accum 2 — identical to Stage 3 v4 |
| throughput | 66.5k tok/s, 490 ms/step, 39.3% MFU |
| mixture | 818,702 rows (MMLU x3, GSM8K x4, ToolCalling x2, ToolACE x3) |
| wandb | run `stage4-toolace-d28-v2-termcall` |

Five GPUs (1,3,4,5,7) was tried first and failed at launch: 5 x 2 x 2048 =
20,480 does not divide 32,768. Four GPUs gives 16,384 and grad accum 2.

## Caveats

- The acceptance test has not run yet. 43.8% prefix-match is the realistic
  ceiling for this fix, and it only becomes the metric if the model stops.
- 54 val examples is small; a few examples move the number several points.
- Top-1 decoding cannot fix a repetition loop — confirmed identical. If v2
  still loops, the next lever is the data shape again, not sampling.

## Files

- `tasks/toolace.py` — `truncate_to_terminal_call()` (+ `truncate_terminal_call=False` flag)
- `probe_term_real.py` — corrected termination measurement
- `probe_term_fix.py` — prefix-match vs exact-match discriminator
- commit `e404ea0` on `roatienza/nanochat`