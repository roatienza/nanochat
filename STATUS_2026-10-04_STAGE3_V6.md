# Stage 3 (SFT) — COMPLETE

**Date:** 2026-10-04, 17:05 PDT
**Status:** FINISHED. Clean exit, no error, no signal, no NaN.
**wandb:** https://wandb.ai/rowel/nanochat-sft/runs/tpkzb8yr
**Checkpoint:** `chatsft_checkpoints/d28b-v2/model_015028.pt` (val_bpb **0.308864**)

---

## 1. Final state

| | |
|---|---|
| **Step** | **15,028** — dataset-driven stop |
| **val_bpb** | **0.308864** (run best, set on the last step) |
| loss (final) | 0.9417 (last-5 mean 0.9165) |
| lrm | 0.00 (warmdown, `warmdown_ratio=0.5`) |
| Throughput | 66,585 tok/s, 492 ms/step, **39.30% BF16 MFU** |
| Epoch | 1 |
| Wall clock | **123.5 m ≈ 2.06 h** (12:50 → 16:21) |
| GPUs | 1, 3, 5, 7 |
| **NaN steps** | **0** |

Exit was clean: `grep -iE "signal|Traceback|Error|KeyboardInterrupt|out of memory"`
returns nothing on the v4 log. wandb summary block printed normally and the final
checkpoint wrote successfully.

---

## 2. The step count is data-driven, not a preset total — I was wrong to quote a total

I told you repeatedly that the run was `X / 15,012`. That number was never real.
`meta_015028.json` records `"num_iterations": -1`, and `chat_sft.py:465`:

```python
# Stopping condition to respect num_iterations, if given
if 0 < args.num_iterations <= it and split == "train":
```

With `num_iterations = -1` that condition is **never true**, so the run never stops
on step count. It stops when the dataloader exhausts the data and sets the global
`last_step = True` (`chat_sft.py:508` comment: *"SFT doesn't always know
num_iterations in advance (dataset-driven stopping)"*).

Consequences:

- There is no meaningful "total steps" figure to report a percentage against.
- My earlier derivations of 15,060 / 15,026 / 15,019 / **15,012** by dividing
  step ÷ percent were all artifacts of `approx_progress`, which is an *estimate*
  (`chat_sft.py:379` "will go from 0 to 1 over the course of the epoch"), not a
  true total. They converged toward 15,012 only because the epoch was nearly done.
- The actual final step was **15,028**.

Reporting "52.31% at step 7,853" was therefore meaningless on both sides — the
percent came from a guess and the denominator did not exist.

---

## 3. Val curve — monotone after the transient, exactly as predicted

| step | 0 | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k |
|---|---|---|---|---|---|---|---|---|---|
| val_bpb | 0.4314 | 0.4682 | 0.4441 | 0.4331 | 0.4270 | 0.4269 | 0.4125 | 0.4044 | 0.3939 |

| step | 9k | 10k | 11k | 12k | 13k | 14k | 15k |
|---|---|---|---|---|---|---|---|
| val_bpb | 0.3808 | 0.3652 | 0.3528 | 0.3391 | 0.3267 | 0.3158 | **0.3089** |

The step-500 transient I flagged at launch (peak 0.4842, vs 0.4905 in the
discarded v3 run) behaved as described: peak, then **monotone decrease for the
remaining ~14,500 steps**. No overfitting, no turnaround.

---

## 4. Comparison to the dead runs

| run | GPUs | final val_bpb | notes |
|---|---|---|---|
| 1-GPU (first attempt) | 2 | 0.4464 | died on disk-full kill |
| v3 | 1+2 | 0.4012 | SIGINT at step 8,164; NaN was logging-only |
| **v4** | **1,3,5,7** | **0.308864** | **complete, clean** |

---

## 5. Corrections to my earlier reports

1. **"15,012 total steps" / all percentage figures were fabricated denominators.**
   There is no preset total (see §2). Every "% complete" I reported for Stage 3
   was computed against an estimate. Retracted.
2. **`f7a3352` in reasoner-model** — a commit I did not write, with my message
   and identity but content byte-identical to `10b2290`. Still unexplained. The
   `nanochat` origin is `karpathy/nanochat.git`, which denies write access, so
   this is not a plain `git push`.
3. **nanochat fixes remain local only** — `f060833` (NaN logging) and `68c19b0`
   (optimizer re-shard + periodic checkpoints) are **5 commits ahead of
   `origin/master`, unpushed** because `origin` is upstream karpathy and rejects
   our writes. Unbacked-up.

---

## 6. Next

Stage 3 is done. Stage 4 (tool calling) was already built and verified in an
earlier session (hermes-v1, masking token-by-token). Ready to start on your go.
