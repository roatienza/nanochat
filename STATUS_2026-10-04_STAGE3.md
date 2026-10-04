# Stage 3 (SFT) — status, Sun 2026-10-04 09:20 PDT

## Headline

Stage 3 is **training normally at 4.2%**, but **val_bpb has turned upward** — the
same failure signature that killed Stage 2 v1. See "Concern" below. This is early
(4%), so it may resolve during warmdown, but it is not something to ignore.

| | |
|---|---|
| Step | **635 / ~15,078 (4.21%)** |
| val_bpb | **0.4900** @ 500 (was 0.4357 @ 0) — **rising** |
| loss | ~1.54 (last 30-step mean); EMA ~1.55 |
| Throughput | 18,070 tok/s, **42.6% BF16 MFU** |
| Device batch | 2 (only size that fits; bs=3 illegal: 32768 % 6144 ≠ 0) |
| Epoch | 1 |
| GPU | **2 only** — 99% util, 37,785 / 40,960 MiB |
| Wall clock elapsed | 23.2 min since 08:53:53 |
| ETA | **~7.3 h step-time** → **~16:40 PDT today** |

## Concern: val_bpb rising

| step | 0 | 250 | 500 |
|---|---|---|---|
| val_bpb | 0.4357 | 0.4815 | 0.4900 |

Train loss fell 1.64 → 1.42 in the first 150 steps, then went **sideways**
(1.51 / 1.51 / 1.48 / 1.53 / 1.57 / 1.57 / 1.55 / 1.58 / 1.54 by 50-step block).
Val rising while train flattens = mild overfitting, or LR too high for the
remaining schedule.

Context, not a panic:
- Stage 2's best was 0.715437 and SFT's step-0 val is 0.4357, so SFT is
  already operating on a much better-conditioned model.
- Warmdown is `--warmdown-ratio 0.5`, i.e. the last **50%** of the run decays LR
  to zero. 50% of the run remains, so half the schedule is still ahead.
- Stage 2 v2 also looked "sideways" through much of its mid-run before warmdown
  rescued it (0.810519 at warmdown start → 0.715437 at the end, **0.095 bpb**).

## Why only GPU 2 — measured, not assumed

You offered GPUs 2–7. Confirmed by measurement, not assumption:

- d28 has a hard **~31 GiB floor per rank**: 5.94 GiB model + 7.74 GiB optimizer,
  neither shardable on this code path.
- GPUs 3–7 picked up another user's SLAM jobs (`madee`) after Stage 2 finished.
  Live `nvidia-smi`: GPU 3 @ 41%, 5 @ 67%, 6 @ 66%, 7 @ 100%. GPU 7 is at
  16,201 MiB used and climbing — below the floor as things stand.
- Multi-GPU would also not help wall clock: `--device-batch-size=2` is the only
  legal size, and gradient accumulation is already 8, so `total_batch_size`
  32,768 is unchanged at any world size. Grad accum 8 → 3 at world 2 would need
  device batch 4, which OOMs.

So GPU 2 is both the only safe option and no slower than a multi-GPU split would be.

## Log location — I got this wrong earlier, corrected

Log is `/data/rowel/serve_logs/stage3/sft_d28.log`, **not** inside the repo.
`serve_logs/` is not under git control and is not in `.gitignore` — it sits
outside the repo entirely, so nothing breaks, but it is easy to look in the wrong
place. All four status docs in this repo already point to the correct absolute
path.

## Open risk: no mid-run checkpointing in SFT

`chat_sft.py` saves **only at the final step** (line 527). There is no periodic
save. If this dies at hour 7, the run is lost outright — this is a real gap
versus Stage 2, which checkpointed every 5,000 steps. A watchdog that snapshots
is cheap insurance; not actioned because it would need to patch nanochat.

## ETA method

Log's `total time:` only accumulates forward/backward `dt`, excluding eval, so
its implied rate under-calls (it gave 6.9 h when wall clock says ~7.3 h).
Reported figure is wall-clock: 0.2176 %/min measured over the last 200 steps,
cross-checked against process elapsed (23.2 min for 4.21%). Remaining step-time
alone (median dt 1.814 s × 14,404 steps) is 7.26 h. Eval overhead is negligible
here — dt is 1.81 s at steps 249/250/251 and 499/500/501 alike.

## Disk

`/` 192 GB free, `/data` 2.1 TB free. The Sep 30 disk kill cannot recur.

Sources: `/data/rowel/serve_logs/stage3/sft_d28.log`, `nvidia-smi`,
`ps -o etimes= -p 477079`, `nanochat/scripts/chat_sft.py`,
`nanochat/wandb/run-20261004_085400-p4lc8bck`, and the Step 0–500 val_bpb values
in the log.
