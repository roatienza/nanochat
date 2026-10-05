# Stage 4 v2 — status 2026-10-05 18:02 PDT

Verified live against the machine at 18:02 PDT. All figures below are read off the
running job, not carried forward from an earlier note.

## Run identity

| | |
|---|---|
| Run name | `stage4-toolace-d28-v2-termcall` |
| wandb | [4of7osv7](https://wandb.ai/rowel/nanochat-sft/runs/4of7osv7) |
| Log | `/data/rowel/serve_logs/stage4v2/stage4v2.log` |
| torchrun PID | 1967506 (started 13:55:40) |
| Rank PIDs | 1968002–1968005, ~243 min CPU each, state `Rsl` (running) |
| GPUs | 3,4,5,7 @ 34,683 / 40,960 MiB (84.7%) |
| Batch | 32,768 total; device-batch 2; grad-accum 2 — identical to Stage 3 v4 |

## Progress

| | |
|---|---|
| **Step** | **10,437** |
| Log percentage | **67.08%** |
| loss (last 300) | ~1.16 |
| lrm | 0.66 (warmdown, started ~step 7,506) |
| Step time | 492.8 ms/step (train) |
| Throughput | 66,5xx tok/s |
| MFU | **39.3%** |
| NaN / inf steps | **0** |
| Log `total time` | 85.66 min |

Note: the run does not print a step total. The ~15,559 implied total below is me
back-dividing 10,437 by the log's own percentage field — it is not a real denominator
and I am not presenting it as one.

## val_bpb — monotone falling since the step-5,000 dip

| step | 0 | 500 | 1k | 1.5k | 2k | 2.5k | 3k | 3.5k | 4k | 4.5k | 5k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val_bpb | 0.4334 | 0.4921 | 0.4759 | 0.4627 | 0.4520 | 0.4459 | 0.4397 | 0.4427 | 0.4352 | 0.4309 | 0.4464 |

| step | 5.5k | 6k | 6.5k | 7k | 7.5k | 8k | 8.5k | 9k | 9.5k | 10k | **10.25k** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val_bpb | 0.4280 | 0.4244 | 0.4205 | 0.4147 | 0.4094 | 0.4066 | 0.4010 | 0.3938 | 0.3864 | 0.3794 | **0.3756** |

Run best is **0.3756 @ step 10,250**. Since the known transients (the step-500
warmup bump and the step-5,000 save-spike) the curve has fallen at every single
eval, 17 consecutive drops.

## ChatCORE — noisy, and worth stating plainly

| step | 8.2k | 8.6k | 9.0k | 9.4k | 9.6k | 9.8k | 10.0k | 10.2k | 10.4k |
|---|---|---|---|---|---|---|---|---|---|
| ChatCORE | 0.0106 | 0.0330 | 0.0336 | 0.0485 | −0.0020 | 0.0044 | 0.0284 | 0.0147 | 0.0180 |

Range −0.0020 → 0.0485 over 11 evals. Stage 3's final was 0.082, so v2 is still
ahead of Stage 3 on this metric, but this is a 9-example benchmark and the
per-eval spread is larger than the signal. Do not read a trend into it.

## ETA — 2.0 h, and the log's own `eta:` field is not to be trusted

I measured it rather than copying the log:

- 45-second window: 58 steps → **0.776 s/step wall**
- Wall clock since 13:55:40: 246.3 min for 10,437 steps → 1.416 s/step wall
- The gap between wall and the log's 492 ms train step is **evals: 160.7 min over
  42 evals = 3.83 min/eval**, which the log's `eta:` field excludes entirely.

Remaining ≈ 5,122 steps at the wall rate → **~2.0 h → ~20:05 PDT**.

Earlier in this project I reported ETAs read straight from that field and every one
of them was wrong, for this same reason. This figure is from wall clock.

## Checkpoints

| step | time | val_bpb | note |
|---|---|---|---|
| 5,000 | 15:56 | 0.4464 | probe run on this, see below |
| **10,000** | **17:51** | **0.379364** (from `meta_010000.json`) | current durable |

Path: `/data/rowel/nanochat_store/v2/chatsft_checkpoints/d28b-v2/`

Disk is not a risk: `/` 155 G free (91% used), `/data` 2.0 T free (86%).

## Acceptance test — still open, unchanged verdict

From the step-5,000 probe (`/data/rowel/tmp/probe/s4v2_5000.log`):

| metric | Stage 4 v1 | v2 @ 5,000 |
|---|---|---|
| exact match | 0.0% | **3.7%** (2/54) |
| returned before token cap | 13.0% | **61.1%** |
| mean predicted calls | 9.31 | 7.33 (ref 2.20) |
| ToolACE exact | 0.0% | 0.0% (18.39 pred vs 1.91 ref) |

`truncate_to_terminal_call` did what it was designed to do — exact match moved off
zero and most generations now terminate. **The repetition loop is not cured**: mean
predicted calls is still 3.3× reference and the dumps contain verbatim repeats.

Caveats I am not going to bury: 3.7% is **two examples out of 54**, which is not
separable from zero; step 5,000 is **before** warmdown; and v1's val curve was still
improving at its last checkpoint, so this is not a finished-v1 comparison.

**Stage 4 does not pass acceptance.** It must be re-probed on the final checkpoint.

## Measurement bugs I have retracted in this project

Recording these because they changed conclusions:

1. `emitted <|assistant_end|>: 0/54` — structurally impossible to be non-zero.
   `Engine.generate_batch` (engine.py:294) strips terminal tokens from its return,
   so the counter reads 0 for *any* model. I reported this artifact as a model
   defect three times before fixing the probe to measure "returned before the
   token cap" instead.
2. NaN losses in Stage 3 — logging artifact, not a dead run. `chat_sft.py` logged
   the last micro-batch only; an all-padding micro-batch has zero supervised
   targets, `reduction='mean'` divides by zero, and the sticky EMA (`beta=0.9`)
   then never washed it out. Fixed in `f060833`.
3. "v1 @ 15,560" as a baseline label — v1's last durable checkpoint is step 10,000.
   15,560 was a mid-run artifact. Directionally the comparison stands, but the
   label was wrong.

## Next step

Re-probe the **final** checkpoint (~20:05 PDT) before accepting Stage 4. Do not
start Stage 5 (RL) against a Stage 4 checkpoint whose tool-calling is still
unverified — RL would inherit the defect and the probe would no longer isolate it.
