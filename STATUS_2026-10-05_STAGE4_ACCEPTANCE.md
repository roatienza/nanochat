# Stage 4 acceptance test — FAILED, and Stage 4 is not the cause

Date: 2026-10-05. Stage 4 (`stage4-toolace-d28-v2`) finished cleanly at 05:39.

## What ran

| | |
|---|---|
| wandb | [bajdpvbs](https://wandb.ai/rowel/nanochat-sft/runs/bajdpvbs) |
| GPUs | 3,4,5,7 |
| wall clock | 163.1 min |
| final step | 15,560 (data-driven stop) |
| min val_bpb | **0.314287** @ 15,560 |
| exit | clean — no signal, no traceback |

val_bpb by checkpoint (`meta_*.json`):

| step | 5,000 | 10,000 | 15,000 | 15,560 |
|---|---|---|---|---|
| val_bpb | 0.434312 | 0.379203 | 0.317277 | 0.314287 |

## The acceptance test: still 0.0% exact match

`eval_toolcalling.py`, 54 held-out hermes val examples.

| checkpoint | exact match | right fn name | right #calls |
|---|---|---|---|
| Stage 3 @ 15,028 (hermes only, 2 ep) | **0.0%** | 41.5% | 3.7% |
| Stage 4 @ 15,560 (+8,957 ToolACE, 3 ep) | **0.0%** | 42.0% | 7.4% |

Stage 4 moved the function-name rate +0.5pt and the call-count rate 3.7%→7.4%.
Both are noise on n=54. **The headline capability did not change.**

## This is not a scoring bug — I checked

Before reporting 0.0% I tried to falsify my own harness:

1. `_norm_call` key-order handling — reproduced examples 1 and 5 by hand; they
   normalise to *identical* tuples. Key order is not the problem.
2. Reference/multi-turn mismatch — `score()` reads all assistant turns, the prompt
   stops at the first. Measured across all 54: **0/54 mismatched**. Not the cause.
3. Renderer round-trip — **119/119** reference calls start with `CALL`. Not it.
4. Truncation — `max_new_tokens=256` vs p90 ref length 757 chars. Only **3.7%**
   (2/54) could plausibly truncate. Not the cause.

So 0.0% is a real measurement, and the failure mode is visible in the dumps:
the model picks the **right functions** but cannot copy JSON arguments verbatim
under multi-call context — it drops items from arrays, misreads `"08:00"` as
`"8:00"` and `"07:00"` as `"temperature":8`, and repeats the first call instead
of advancing to the second.

## Why Stage 4 could not have fixed this

Stage 4 is a **fresh SFT pass from the pretrained base** (d28b-v2 @ 277,760), not a
continuation of Stage 3 — so it could not have decayed anything Stage 3 learned.
It also *added* 8,957 ToolACE examples, 8.6× the hermes set. It simply does not
move exact match.

The constraint is **data shape, not data volume or compute**. hermes/ToolACE
teach "which function, roughly what arguments". Exact match needs the model to
reproduce a long JSON object token-for-token, and only 1,036+8,957 examples ever
supervise that. Stage 3's val_bpb of 0.308864 is better than Stage 4's 0.314287 —
**Stage 4 is slightly worse on general quality** while identical on tool calls.

## Recommended next step (not yet executed)

Do **not** advance to Stage 5 RL with tool calling broken. The acceptance test is
the stage's definition of done, and it is not done. Options, in order of expected
value per GPU-hour:

1. **Verbatim-copy drill** — build a synthetic task that supervises *only* long
   JSON emission (echo the user's list verbatim into `{"items":[...]}`), mixed into
   the SFT mixture. This targets the observed failure directly.
2. **More ToolACE epochs / higher weight** — cheap to try, but the failure is
   structural; unlikely to move exact match much.
3. **Teacher-forcing eval** — measure teacher-forced next-token accuracy on the
   call tokens. If that is high (>90%) while greedy decode is 0%, the problem is
   decoding/attention, not knowledge, and option 1 is wrong.

Option 3 is the cheapest next measurement and should run before any retraining.