# Stage 4 v2 — step-5,000 termination probe (2026-10-05)

Mid-run read on the acceptance metric, using a **corrected** probe.

## Why the old probe was wrong

`Engine.generate_batch` (nanochat/engine.py:294) strips terminal tokens from
what it returns, so the earlier `assistant_end in gen` check was False by
construction — it could never have reported non-zero for *any* model. I had
reported that artifact as a model defect twice; retracting it.

Termination is now measured as "generation returned before the token cap"
(`probe_term_real.py`, committed in nanochat `e404ea0`).

## Result — the acceptance metric moved off zero for the first time

Held-out hermes val, n=54, budget=768, greedy:

| metric | Stage 4 v1 @15560 | **Stage 4 v2 @5000** |
|---|---|---|
| **exact match** | **0.0%** | **3.7%** (2/54) |
| stopped before cap | 13.0% | **61.1%** |
| mean pred calls | 9.31 | **7.33** (ref 2.20) |

ToolACE val, same budget:

| metric | value |
|---|---|
| exact match | 0.0% (0/54) |
| stopped before cap | 53.7% |
| mean pred calls | 18.39 (ref 1.91) |

## Verdict on the fix

`truncate_to_terminal_call` did what it was supposed to: the exact-match
metric is non-zero for the first time in this project, and real termination
rose 13.0% -> 61.1%. But the repetition loop is **not** cured — mean predicted
calls is still 3.3x the reference, and the dumps show the model emitting the
same `CALL <fn> <same args>` verbatim in a loop until the cap.

Note this is step 5,000 of ~15,533 — roughly a third of the way through, and
before warmdown (`warmdown_ratio=0.5`) begins. The v1 curve was still improving
at 15,560, so this is an early read, not a final verdict.

## Measured defects still open

1. **Repetition loop persists.** ex0: `get_recorded_stream {"camera_id":
   "front_garden"}` x4+. ex1 (toolace): `Get Zip Code Information` x6+.
2. **Token-level degeneration**, not just call repetition: `connect_to_all_
   devices_ through_interface_interface_interface...` (ex1), `c commodity`
   inside a JSON string (toolace ex0). JSON string copying is the weak spot.
3. ToolACE is **harder** than hermes (0.0% vs 3.7%) despite being 8.6x larger,
   consistent with Stage 4 v1 where ToolACE also failed to move the metric.

## Caveats

- n=54; 3.7% is 2 examples. Not statistically separable from 0%.
- The probe ran on GPU 1, which the training run does not use; GPUs 3,4,5,7
  were unaffected (492 ms/step maintained, 39.3% MFU throughout).
