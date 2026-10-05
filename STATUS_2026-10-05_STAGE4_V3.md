# STATUS — Stage 4 v3 launched, 2026-10-05 21:15 PDT

## What the wide probe settled (option 1, done)

`probe_wide.py` on `model_015548.pt`, full val populations, GPUs 1/3/4/5/7, ~11 min.

| | hermes val (n=54, whole pop) | ToolACE val (n=182, whole pop) |
|---|---|---|
| exact @512 | 0/54 = 0.0% | 1/182 = 0.5% |
| exact @1024 | 0/54 = 0.0% | 1/182 = 0.5% |
| count_ok | 7.4% | 28.0% |
| stopped before cap | 94.4% | 97.8% |
| fn acc | **94.6%** | 58.2% |
| call-prefix acc | 48.1% | 10.7% |
| mean pred vs ref | 5.81 vs 2.20 | 2.46 vs 1.97 |

**Budget 512 -> 1024 changed nothing.** The long-standing "calls are truncated by
the token cap" explanation is falsified: only 5.6% / 2.2% hit the cap, and exact
match did not move at all. The earlier n=54 "3.7%" was 2 examples.

## Root cause found for the hermes half (option 2 basis)

Measured on all 1,036 hermes train conversations: **100% end their call-bearing
assistant turn in `python_output`**, because `_convert` attaches each tool result
to the turn that made the call. hermes therefore teaches *call -> tool result*.
At inference the environment supplies results the eval never shows — exactly the
transition the model fails to make. ToolACE got `truncate_to_terminal_call()` in
v2 (`e404ea0`); **hermes never did**, because it is 1,036 of 818,702 mixture rows.

Symptom match is exact: 23/54 hermes val examples match the ENTIRE reference as a
correct prefix then emit +4.35 more calls, with 94.6% fn accuracy.

## Fix applied — nanochat `8ef9092`

`_truncate_terminal_call()` added to `tasks/toolcalling.py`, handling both hermes
shapes (159 end with the call turn; 877 end with a synthetic "Continue." + prose
summary turn, calls in an earlier turn).

Verified on all 1,036 train examples:
- final turn call-only **0.0% -> 100.0%**
- ends on `<|assistant_end|>` **99.9%** (1035/1036)
- calls per conversation **2.00 -> 2.00** (count preserved; capping at one would
  cap exact match at 35.2%)
- `python_output` in final turn 100% -> 0%
- 121,885 supervised tokens of 840,736 (14.5%); **0 tool-result tokens supervised**

**Val deliberately NOT truncated** (`truncate_terminal_call` defaults to
`split=="train"`), verified identical 54/54 with truncation off. Rewriting eval
targets to match training would be grading our own homework.

## Reweighted mixture (option 2)

Tool rows were only **3.5%** of the old mixture (1,036x2 + 8,957x3 of 818,702).

| | rows/epoch | tool row share |
|---|---|---|
| v2 (old) | 818,702 | 3.5% |
| **v3 (new)** | **625,542** | **9.3%** |

`--mmlu-epochs 1 --gsm8k-epochs 1 --toolcall-epochs 4 --toolace-epochs 6`
(MMLU x3->x1, GSM8K x4->x1, ToolCalling x2->x4, ToolACE x3->x6.)

## Run status

Init from the pretrained base `base_checkpoints/d28b-v2/model_277760.pt`
(val_bpb 0.715437) — the same init the v2 run used, **not** the Stage 3 SFT
checkpoint. Stage 4 is a from-base tool-calling stage by design.

| | |
|---|---|
| GPUs | 3,4,5,7 @ 99%, 34,683 MiB each |
| throughput | 66.5k tok/s, 492.9 ms/step, **39.3% MFU** |
| step 0 val_bpb | 0.4314 |
| NaN steps | 0 |
| wandb | [qm00mjt0](https://wandb.ai/rowel/nanochat-sft/runs/qm00mjt0) |
| ETA | **~3.3 h** → ~00:30 PDT (wall-clock measured, 1.111 steps/s incl. evals) |

First checkpoint (step 5,000) lands ~23:00 PDT. Re-probe required before Stage 4
can be accepted — the fix is a hypothesis until the acceptance test moves.

## Repos

- nanochat `8ef9092` pushed to fork; `karpathy/nanochat` untouched.
- reasoner-model: this file.
