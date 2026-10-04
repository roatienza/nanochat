# Stage 3 (SFT) — started 2026-10-04

## What launched

```
CUDA_VISIBLE_DEVICES=2 \
NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2 \
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
  --run=sft-d28-v2 --model-tag=d28b-v2 --model-step=277760 \
  --device-batch-size=2 \
  --eval-every=250 --eval-tokens=2097152 --chatcore-every=999999
```

  log    /data/rowel/serve_logs/stage3/sft_d28.log
  PIDs   476800 (torchrun) + 477079 (rank 0), detached
  wandb  run 8zpdsgzq  https://wandb.ai/rowel/nanochat-sft/runs/8zpdsgzq

## Status at launch

  step 00090 of ~15,517 estimated  (0.58%)
  loss 1.4721 mean over last 60 steps
  18,090 tok/s, 42.71% BF16 MFU, epoch 1
  GPU 2 @ 99%, 37,785 / 40,960 MiB
  step 0 val_bpb 0.4357 (from the pretrained base model)
  ETA ~6.9 h

  Total-step count is an estimate: the SFT dataloader stops when the mixture is
  consumed, so the true count is only known at the end. 791,831 rows at
  32,768 tokens/step.

## GPU choice: single GPU, not 2-7

  The request was GPUs 2-7 "if needed". It was needed as a memory question and
  the answer is no. d28 has a hard floor of ~31 GiB per rank regardless of batch
  size, measured on an idle GPU (mem_probe.py runs a real fwd/bwd/optimizer.step):

    bs=1  PEAK 31.24 GiB      bs=2  PEAK 31.30 GiB
    bs=4  PEAK 34.86 GiB

  The floor is the model (5.94 GiB) plus its full optimizer state (7.74 GiB),
  both of which are per-rank and cannot be sharded by this code path.

  Since Stage 2 finished, GPUs 3-7 are occupied by another user's SLAM jobs
  (user `madee`, slam.py on several scannet/tum configs). Measured free memory
  at launch:

    GPU 2  40,960 MiB free   <- the only idle GPU
    GPU 6  ~35,900
    GPU 4  ~35,900
    GPU 7  ~27,500
    GPU 3  ~32,600
    GPU 5  ~30,500

  GPUs 5 and 7 are below the 31.24 GiB floor outright; 3 and 7 have under
  2 GiB of margin against the bs=4 peak. Using 2-7 would mean co-scheduling
  with someone else's job and OOMing, possibly taking their run down too.
  So this runs on GPU 2 alone, at bs=2 for an 8.2 GiB margin.

## Three real bugs found, all fixed in nanochat (commit 70059d0)

### 1. Cross-world-size optimizer warm-start (the blocker)

First launch died on the first optimizer.step():

  RuntimeError: Attempting to broadcast a dimension of length 32768 at -2!
  ... exp_avg(8192,1792) vs param(32768,1792)

Not a GPU or config problem. Stage 2 ran at world_size=4, and nanochat shards
optimizer state, so `optim_277760_rank0.pt` holds wte/lm_head moments
reduce-scattered to 1/4 of the rows: [8192, 1792] instead of [32768, 1792].
Loading that straight into a full-size param breaks the fused AdamW kernel.

Verified against the shards: keys 0-15 are all sliced 1/4, keys 16-20 are
replicated scalars, Muon groups are padded chunks. 203 live params, 25 state
entries, 0 shape mismatches once rebuilt.

  - AdamW large params (>=1024 numel): concatenate shards along dim 0.
  - AdamW small params (<1024 numel): all-reduced, every rank identical, take rank 0.
  - Muon groups: concatenate then trim the zero padding (group 6 is 14 real
    params stored as 16 padded rows across 4 ranks).

Shard shapes alone are ambiguous - a replicated (28,) is indistinguishable from
a 1-D slice - so the rebuild compares against the live parameter shapes, keyed
the way optimizer.state_dict() keys its state. My first attempt used
enumerate(model.parameters()), which is the wrong order; caught by verifying
the key mapping against the real shards before relaunching.

`save_checkpoint` now records `world_size`. The Stage 2 checkpoint predates that,
so the factor is inferred from the moments when the key is absent (verified:
returns 4 for d28, 1 for an unsharded control).

### 2. Stale tool-calling corpus path

tasks/toolcalling.py DEFAULT_PATH pointed at
/home/rowel/baby-sandbox/data/hermes/func-calling.json, which no longer exists
(the sandbox moved to /data/rowel/reasoner-model). The real corpus is at
/data/rowel/data/hermes/func-calling.json, 1,893 rows / 20.5 MB. Without this the
ToolCalling task cannot load and Stage 3 cannot start at all.

Re-verified masking after the path change: 1,036 train rows, 54 val,
supervised fraction mean 22.8%, **0 schema tokens and 0 tool results in a
supervised position**, 0 CALL fragments missing.

### 3. OOM at the inherited batch size

Second launch OOMed at step ~0 with `--device-batch-size=4` inherited from
pretrain: 30.64 GiB allocated by PyTorch plus **7.25 GiB reserved-but-unallocated**
despite PYTORCH_ALLOC_CONF=expandable_segments:True being set at chat_sft.py:15.

bs=4 only fits with room to spare when the memory is clean. Measured with the
real full-size optimizer state loaded and the real grad-accum loop:

    bs=4  OOM       bs=3  PEAK 36.02 GiB (illegal anyway)   bs=2  PEAK 31.30 GiB

bs=3 is not a legal setting regardless: total_batch_size 32768 must be divisible
by bs * max_seq_len * world_size, and 32768 % 6144 != 0. The real choice was
bs=2 -> grad_accum 8, peak 31.30 GiB, 8.2 GiB of headroom on GPU 2.

## Config decisions

  --device-batch-size=2   forced by memory, see bug 3
  --eval-every=250        matches the inherited pretrain cadence
  --eval-tokens=2097152   2 MiB, not the 40 MiB default. The Stage 2 finding was
                          that eval ran at 2.56x training tokens and ate ~40% of
                          wall clock; at 15.5k SFT steps that would dominate.
                          Re-tune once the first few evals show the curve shape.
  --chatcore-every=999999 disabled for now. ChatCORE runs generation over 5
                          tasks and at d28 is expensive; enable once the run
                          proves stable rather than risk it at step 0.

  Inherited from pretrain: max_seq_len 2048, total_batch_size 32768,
  embedding_lr 0.3, unembedding_lr 0.008, matrix_lr 0.02, warmdown_ratio 0.5.

  Mixture: 791,831 rows = SmolTalk 460,341 + MMLU auxiliary_train x3 (299,526)
  + GSM8K main x4 (29,892) + ToolCalling x2 (2,072).

## Note on nanochat's git remote

  origin is https://github.com/karpathy/nanochat.git - upstream, not ours.
  The three fixes are committed locally as 70059d0 on branch master and were
  deliberately NOT pushed. They are candidate upstream patches; pushing to
  someone else's repository is not ours to do.

## Open items

  - No watchdog. The Stage 2 kill (Sep 30, full disk) has no equivalent here
    yet the same single point of failure remains. SFT saves only at the final
    step, so a kill now costs the entire run, not 5,000 steps. Worth a watchdog
    or periodic save before committing to a 7 h run unattended.
  - ChatCORE is off; it is the real capability signal and should be turned on
    once the loss curve is known good.
  - The 15,517-step total is an estimate, not a known quantity.

## Sources

  /data/rowel/serve_logs/stage3/sft_d28.log
  /data/rowel/nanochat_store/v2/base_checkpoints/d28b-v2/{model,meta}_277760.*
  /data/rowel/nanochat_store/v2/base_checkpoints/d28b-v2/optim_277760_rank{0..3}.pt
  /data/rowel/data/hermes/func-calling.json
  nanochat commit 70059d0 (local)
  wandb run 8zpdsgzq
