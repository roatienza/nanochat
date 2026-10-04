# Stage 3 (SFT) v4 — 52.3% complete, clean, val curve confirmed

**Date:** 2026-10-04, 14:43 PDT
**Status:** RUNNING, healthy. No NaN. Best val_bpb of the run so far.
**wandb:** https://wandb.ai/rowel/nanochat-sft/runs/tpkzb8yr

---

## 1. Live state

| | |
|---|---|
| **Step** | **7,853 / 15,012 (52.31%)** |
| **val_bpb @ 7,750** | **0.3972** ← run best (still falling) |
| loss (last 300 mean) | **1.2563** |
| lrm | 0.95 (warmdown, since step ~7,506) |
| Throughput | 66,279 tok/s, 494.4 ms/step, **39.12% BF16 MFU** |
| Epoch | 1 |
| GPUs | **1, 3, 5, 7** @ 92–94%, 34,681 / 40,960 MiB each |
| PIDs | 1834434, 1834435, 1834436, 1834438 (all Rsl, 1h49m elapsed) |
| **NaN steps** | **0** (`grep -c "loss: nan"` → 0) |
| **ETA** | **~1.7 h → Sun Oct 4, ~16:24 PDT** |

Last durable checkpoint: **step 5,000** @ 14:01, `meta_005000.json` val_bpb
0.426904. Next at step 10,000 (~15:15 PDT).

---

## 2. Two numbers in my last report were wrong

### 2a. Total steps is 15,012, not 15,078

I quoted 15,078 in three prior messages without checking. Derived the true total
from `step / pct` across many points in the log — it converges tightly:

| step | pct | implied total |
|---|---|---|
| 2,000 | 13.28 | 15,060 |
| 4,000 | 26.62 | 15,026 |
| 6,000 | 39.95 | 15,019 |
| 7,700 | 51.30 | 15,012 |

**15,012** is stable to ±4. The 15,078 figure appears nowhere in the run.

### 2b. The log's `dt` understates wall-clock by 1.72x — my ETAs have been optimistic

`dt: 494.39ms` is train-only fwd/bwd. Actual wall-clock per step is **0.01424
min = 854 ms**. Two independent measurements agree:

| segment | steps | elapsed | min/step |
|---|---|---|---|
| step 1 → 5,000 | 5,000 | 70.8 min | 0.01417 |
| step 5,000 → 7,853 | 2,853 | 41.0 min | 0.01437 |
| **combined** | **7,853** | **111.8 min** | **0.01424** |

0.01424 / 0.00824 = **1.72x**. The gap is eval every 250 steps, checkpoint save
every 5,000, and dataloader refill. Had I used `dt` (as I did in the Stage 2
"log eta off by 67%" finding), I would have promised ~15:41 — the run will not
finish before **16:24**.

---

## 3. The val curve behaved exactly as predicted

In the v4 launch message I stated the step-500 bump was the known transient, not
overfitting, and that v3 fell monotonically to 0.4246 by step 4,750. Confirmed:

| step | 0 | 250 | 500 | 1k | 2k | 3k | 4k | 5k | 6k | 7k | **7.75k** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val_bpb | 0.4314 | 0.4765 | **0.4842** | 0.4682 | 0.4441 | 0.4331 | 0.4270 | 0.4269 | 0.4125 | 0.4044 | **0.3972** |

Peak at step 500 exactly where v3 peaked (0.4842 vs 0.4905 — v4 slightly
better), then **down on every eval since**. v3 tracked v4 to within 0.006 the
whole way.

**Already past v3's best:** v3 reached 0.4012 @ step 8,000 before the SIGINT.
v4 is at **0.3972** with 48% of the run still to go, and warmdown just started.

### vs the discarded 1-GPU run
| run | best val_bpb |
|---|---|
| 1-GPU (killed at 54.4%) | 0.4464 |
| v3, 2-GPU (SIGINT @ 8,164) | 0.4012 |
| **v4, 4-GPU** | **0.3972** ← and falling |

---

## 4. Disk headroom is fine

`/` 165 GB free (91%), `/data` 2.1 TB free (85%). Store is 1.2 TB. The
checkpoint symlinks (`base_checkpoints`, `chatsft_checkpoints`) both resolve to
`/data/rowel/nanochat_store/v2/...` and were fixed during the v4 launch.

---

## 5. State on disk

`chatsft_checkpoints/d28b-v2/` (→ `/data/rowel/nanochat_store/v2/`):

| file | size |
|---|---|
| `model_005000.pt` | 6.31 GB |
| `optim_005000_rank{0,1,2,3}.pt` | 2.08 GB each |
| `meta_005000.json` | 935 B, val_bpb 0.426904 |

No resume-from-`--model-tag` path exists in `chat_sft.py` (it can only load
optimizer state from the *pretrained* base), so a mid-run kill still costs a full
restart. `--save-every 5000` bounds that to ≤41 min of work.

---

## 6. Still open

1. **nanochat has no writable remote.** `origin` is `karpathy/nanochat.git`,
   which denies write access. The NaN fix (`f060833`) and its test are local
   only. Fork it so they are backed up?
2. **Commit `f7a3352` remains unexplained** — my message, your identity,
   describing content that appears nowhere in the file it committed. Content is
   byte-for-byte correct. Amend + force-push, or leave and document?
3. **GPUs 3 and 5 are contested.** GPUs 2, 4, 6 currently hold other users'
   jobs (40,455 / 12,049 / 38,473 MiB). If 3/5 get reclaimed the run dies.
