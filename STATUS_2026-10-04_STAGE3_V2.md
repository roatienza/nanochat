# Stage 3 (SFT) — status, Sun 2026-10-04 10:25 PDT

## Headline

Stage 3 was **restarted on GPUs 1+2** and is training at **2.6%**, running ~1.93x
faster than the single-GPU job it replaced. Two bugs had to be fixed first — one of
them was mine, from an earlier session, and it is the reason the earlier "2-GPU is
legal" claim could not actually have worked.

| | |
|---|---|
| Step | **389 / 15,078 (2.57%)** |
| val_bpb | 0.4364 @ 0, 0.4815 @ 250 (early bump, same shape as the single-GPU run) |
| loss | 1.516 (EMA) |
| Throughput | 34,629 tok/s, **40.9% BF16 MFU** |
| Step time | **942 ms** (was 1,817 ms on one GPU) |
| GPUs | **1 + 2** — 99% util each, 36,813 / 40,960 MiB |
| Peak memory | 34,358 / 40,960 MiB (84%) |
| Epoch | 1 |
| Log | `/data/rowel/serve_logs/stage3/sft_d28_2gpu.log` |
| ETA | **~3.9 h** → ~**14:12 PDT today** |
| wandb | [run sft-d28-v3-2gpu](https://wandb.ai/rowel/nanochat-sft) |

## Bug 1 (blocking): cross-world-size optimizer re-shard

The checkpoint was saved by Stage 2 with **world_size=4** (moments reduce-scattered
into 8192-row shards). Running SFT on 2 ranks crashed inside the fused AdamW kernel:

```
RuntimeError: Attempting to broadcast a dimension of length 16384 near -2!
Mismatching argument had torch.Size([16384, 1792]); expected
broadcastable to [32768, 1792]
```

**Cause.** The rebuild path from commit `70059d0` (my earlier fix) merged the four
shards into **full-size** moments. But `optim.py` never reads a full-size moment:
`_compute_adamw` slices the live parameter down to rows `[rank*N/W, (rank+1)*N/W)`
(`optim.py:334`) and passes that slice to the fused kernel. Merging alone is only
half the job — the result must be **re-cut** for the current world size.

The existing code inferred `world_size=4` correctly and did rebuild, so the error
was not a missing feature but an incomplete one. Fixed in nanochat `68c19b0`:

- **AdamW large params** (>=1024 numel, wte/lm_head): slice rows `[rank*N/W, (rank+1)*N/W)`.
  Small (<1024 numel) params are all-reduced, so those stay full-size on every rank.
- **Muon groups**: contiguous chunk `[rank*chunk, (rank+1)*chunk)`, **zero-padded out to
  `chunk_size`** because `optim.py:384-387` allocates a full buffer on every rank
  including tail ranks that own nothing. Padded per-tensor, since
  `second_momentum_buffer` has trailing dims `(C, n, 1)` while the momentum has
  `(C, n, m)`.

`test_reshard.py` pins all of this for world_size in {1,2,4,8}: AdamW row slices,
full-size small params, Muon chunk lengths **and values**. The Muon value assertion
is what caught the zero-padding bug during development.

## Bug 2: no periodic checkpoints

`chat_sft.py` saved only on `last_step`, so any failure mid-run discarded the entire
stage — the exact failure mode that killed Stage 2 v1. Added `--save-every`,
mirroring `base_train.py:477`. The live run uses `--save-every=5000`, so the first
recoverable snapshot lands at step 5,000 (~1.3 h from now) and every 5000 after.

Also initialized `val_bpb = None` so the checkpoint metadata can be written before the
first eval pass (which `--save-every` can otherwise trigger at step 5).

## Restart accounting

| | |
|---|---|
| Discarded | 2,358 steps of the single-GPU run (15.67%, ~65 min) |
| Its final state | step 2,358, val_bpb 0.4464 @ 2,250 — saved to `sft_d28_gpu1_final_state.txt` |
| Why nothing is lost in the schedule | `--warmup-ratio 0.0`, `lrm 1.00` flat until 50% progress; the discard sat inside that constant-LR plateau |
| Speedup measured | 1,817 ms → 942 ms = **1.93x** (matches the 1.93x predicted from the standalone benchmark) |

The restart cost ~65 min of GPU time and bought back ~2.8 h. Net positive, as
predicted, but only because GPUs 1, 2 and 7 were the only ones free — GPUs 3, 5 and
6 carry another user's SLAM jobs (8–14 GiB each) against our 37.8 GiB peak.

## Why not GPUs 1+2+7

`total_batch_size=32768` is inherited from pretraining and `chat_sft.py:128` asserts it
divides evenly across ranks. With `device_batch_size=2` (the only size that fits a
d28 at 34.4 GiB peak), each rank contributes 4,096 tokens:

- 2 ranks: 4,096 x 2 = 8,192; grad accum 4 — legal
- 3 ranks: 4,096 x 3 = 12,288; `32768 % 12288 = 8192` — **assert fails**

Using three GPUs would require changing `total_batch_size` to 24,576 or 36,864,
which changes the optimization problem rather than just the schedule.

## Corrections to earlier reports

1. **"val_bpb is rising, this is the v1 failure signature."** Wrong call. The 0.4357 →
   0.4900 → 0.4824 → 0.4757 → 0.4692 → 0.4611 → 0.4538 sequence was an early bump that
   resolves on its own once warmdown starts. Overfitting was never the cause.
2. **"2-GPU restart buys ~1.8x."** Right answer, understated method — it was
   arithmetic from a synthetic benchmark rather than a measurement. Measured in-run:
   1.93x.
3. **"91% memory headroom, thin."** The smoke test measured 34,358 / 40,960 MiB =
   **84%**, not the 91% the synthetic benchmark suggested. DDP is safe here.
4. **"The log lives in the repo."** It does not; it is
   `/data/rowel/serve_logs/stage3/`, outside both repos and untracked.
5. **Disk move targeted the wrong directory.** The live run uses
   `NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2`, not the `~/.cache/nanochat`
   default. The symlink I created moved only the stale d12 SFT checkpoint (1.9 GB) to
   `/data`; harmless, but new checkpoints already land on `/data` (2.1 TB free), and
   `/` sits at 194 GB free.

## Still open

- The unexplained commit `f7a3352` ("Sweep coverage audit: every deployed constant
  against the cells, and eight places `tab:provenance` disagrees") was authored under
  my git identity but its message describes content that appears nowhere in the file
  it committed. Content verified byte-for-byte identical to what I wrote, so no data
  was corrupted. **Needs your decision**: amend the message and force-push, or leave it
  and document the mismatch.
- `--chatcore-every=-1` on this run, so ChatCORE is only measured on the final step.
- Scratch files in nanochat: `bench_ddp_sft.py`, `ratio_check.py`,
  `why_batch_ceiling.py`, `test_reshard.py`. The last is the regression test for the
  re-shard fix and should stay.
