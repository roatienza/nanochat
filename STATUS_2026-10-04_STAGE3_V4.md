# Stage 3 (SFT) — restarted on GPUs 1,3,5,7

**Date:** 2026-10-04, ~13:05 PDT
**Decision:** Option B — full restart from scratch with the NaN-logging bug fixed.
**Status:** RUNNING, clean. 5.0% at time of writing.

---

## 1. The bug that killed the v2/v3 runs

`chat_sft.py` logged `train_loss` from the **last micro-batch only**, and an
all-padding micro-batch has zero supervised targets. `F.cross_entropy(...,
reduction='mean', ignore_index=-1)` divides by the valid-token count, so an
all-padding micro-batch returns **NaN**. That NaN went straight into the EMA:

```python
smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
```

`ema_beta = 0.9` never washes it out. One degenerate micro-batch at step 3,846
made a **completely healthy** run print `loss: nan` for the remaining 4,319
steps.

### Evidence the model was never broken

| Check | Result |
|---|---|
| Weights @ step 5000 | **203/203 tensors finite** |
| Optimizer moments, both ranks | **100/100 float tensors finite** |
| val_bpb @ 5000 → @ 8000 | **0.4257 → 0.4012** (still improving) |
| Gradient from a NaN micro-batch | **all-zero and finite** |

The NaN loss produces an all-zero finite gradient, so `backward()` is a no-op.
The weights were never in danger — only the printed number was.

## 2. The fix (`f060833`)

- Accumulate every micro-batch loss; average over the **finite** ones, so a
  degenerate micro-batch can never reach the EMA.
- Raise `RuntimeError` if *every* micro-batch is degenerate — that is a
  data-packing bug, not a run to continue.
- Side benefit: the reported loss is now a true cross-micro-batch average,
  not one micro-batch's mean-per-valid-token.

Regression test `test_sft_nan_loss.py` — 5 checks, all passing.

## 3. Also found and fixed: stale checkpoint symlinks

`~/.cache/nanochat/{base,chatsft}_checkpoints` pointed at
`nanochat_store/*.tmp`, but the live checkpoints were under `nanochat_store/v2/`.
Symlinks repointed. (First 4-GPU smoke test failed with `FileNotFoundError`
on `base_checkpoints/d28b-v2` before this was caught.)

---

## 4. The run

| | |
|---|---|
| GPUs | **1, 3, 5, 7** (4 × A100-40GB) |
| world_size | 4 |
| grad_accum | **2** (total batch 32,768 unchanged) |
| step time | **494 ms** |
| throughput | **66,300 tok/s, 39.1% BF16 MFU** |
| peak memory | 34,681 MiB / 40,960 (85%) |
| total steps | **15,000** |
| `--save-every` | 5000 |
| mixture | 791,831 rows (MMLU x3, GSM8K x4, ToolCalling x2) |
| wandb | https://wandb.ai/rowel/nanochat-sft/runs/tpkzb8yr |
| ETA | **~2.0 h → ~14:59 PDT** |

### Speed vs the abandoned 2-GPU run

| config | dt | tok/s | MFU |
|---|---|---|---|
| 1 GPU | 1,807 ms | 18,132 | 42.7% |
| 2 GPUs (1+2) | 944 ms | 34,711 | 41.0% |
| **4 GPUs (1,3,5,7)** | **494 ms** | **66,300** | **39.1%** |

Note 4-GPU is **not** 4× the 1-GPU number (it is 3.7×) — MFU drops from 42.7%
to 39.1% as all-reduce overhead grows. The 4-GPU config is nonetheless ~2.0×
faster than the 2-GPU run that was discarded.

### Val curve so far — matches the discarded run's shape

| step | 0 | 250 | 500 |
|---|---|---|---|
| v4 (4-GPU) | 0.4314 | 0.4765 | 0.4842 |
| v3 (2-GPU, discarded) | 0.4364 | 0.4815 | 0.4905 |

The early rise is a **normal transient**, not overfitting: in v3 it peaked at
step 500 (0.4905) and then fell monotonically to **0.4246 by step 4,750**. v4 is
tracking the same curve almost exactly.

---

## 5. Open items

1. **Who sent the SIGINT at 12:32:43?** That killed the previous run at step
   8,164. Cannot be attributed from the log. If a scheduler or a person did it,
   this run can die the same way — which is why `--save-every 5000` is now set
   (worst-case loss ~41 min instead of the whole run).
2. **GPUs 3 and 5 are contested.** Both were running another user's SLAM jobs
   earlier today and are free now. They could be reclaimed mid-run.
3. **`f7a3352`** in this repo is still unexplained — my commit message, your
   git identity, content that appears nowhere in the file it committed. The
   file content was verified byte-for-byte correct. Amend + force-push, or leave
   and document?
