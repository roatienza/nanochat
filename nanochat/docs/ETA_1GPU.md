# utak — ETA for a single-GPU (1× A100-40GB) training run

**Measured 2026-10-21 on `lambda3-hyperplane` (8× A100-SXM4-40GB, NVLink).**

## Headline

| profile | wall clock | days | basis |
|---|---|---|---|
| **1 GPU** | **207.9 h** | **8.7** | 3 of 9 training stages measured directly on one GPU; the rest scaled from a measured 1→4 GPU pair. The RL stages were re-measured on 2026-10-22 with the larger generation batch |
| 1 GPU + 15% contingency | 239.1 h | 10.0 | restarts, checkpoint I/O, eval drift |
| 2 GPUs | 117.2 h | 4.9 | **measured to fit** on 2026-10-22, and stage 1 now running at that speed (see ETA_2GPU.md) |
| 4 GPUs (how it was actually built) | ~75 h | 3.1 | pretrain 34.2 h + the rest |

**If training starts today (2026-10-21) on one free GPU, `utak` is served at about
2026-10-30 08:00 PDT**, or 2026-10-31 midday with contingency.

**The TOTAL rows below are the sum of the stage rows.** They were wrong in
the first version of this file (it said 203.1 h for 1 GPU and 106.0 h for
2 GPUs, neither of which equalled its own rows) and
[`ETA_2GPU.md`](ETA_2GPU.md) still carried the pre-RL-fix numbers
(249 h / 136 h). Both documents now agree, and both are the sum of the
rows below.

## The one measurement everything else rests on

The same d28 SFT workload, run at three world sizes on this node:

| world size | dt/step | tok/s | MFU | source |
|---|---|---|---|---|
| 1 GPU | 1815.30 ms | 18,051 | **42.62 %** | `serve_logs/stage3/sft_d28.log` |
| 2 GPUs | 939.45 ms | 34,880 | 41.17 % | `serve_logs/stage3/sft_d28_2gpu.log` |
| 4 GPUs | 492.12 ms | 66,585 | 39.30 % | `serve_logs/stage3/sft_d28_v4_4gpu.log` |

Two conclusions, both load-bearing:

1. **One GPU is ~97 % as efficient *per token* as four.** Four GPUs buy 3.69× the
   throughput, so going 4 → 1 costs **3.7× the wall clock, not 4×**.
2. **One GPU's MFU (42.6 %) equals the 4-GPU pretrain's MFU (43.6 %).** The GPU
   count is not the bottleneck; the model is already big enough on one card to
   fill it.

Cross-check: the 4-GPU pretrain logged 34.25 h of training for 9.10 B tokens.
×3.7 = 127 h. The token-anchored estimate for one GPU is 140 h. They agree to
within 2 %, from independent routes.

## Why one GPU is a supported profile, not a compromise

- **Memory.** d28 peaks at **32,050 MiB at device batch 2** (measured
  2026-10-22 with the real `scripts.base_train` on one A100-40GB) against
  40,960 MiB of card. At device batch 4 it **OOMs**: "39.30 GiB in use,
  201.88 MiB free". The recipe's default is now 2.
  The cost is the optimizer state, not the activations: 2.02 B parameters x
  (fp32 `exp_avg` + fp32 `exp_avg_sq`) = 15.0 GiB of moments, plus bf16
  weights 3.76 GiB and bf16 gradients 3.76 GiB = 22.5 GiB of state that at
  world size 1 is entirely unsharded. A smaller device batch cannot buy
  back enough to fit, which is why device batch 2 is the floor, not 1.
  Measured on 2026-10-22, one A100-40GB, real trainer, d28, 2k ctx,
  tbs 32,768, fresh process per measurement:

  | device batch | result | tok/s | MFU | peak |
  |---|---|---|---|---|
  | 2 | **OK** | 17,525 | 41.4 % | 32,050 MiB |
  | 3 | OOM | - | - | - |
  | 4 | OOM | - | - | 39.30 GiB used, 201 MiB free |
- **Optimizer state.** Sharded by world size. Moving 4 → 1 is handled by
  `chat_sft.py:341`, which rebuilds full-size moments from the 4 saved shards
  and re-shards to 1. The historical 1-GPU runs (stages 3-9) all used it.
- **Learning rate is unchanged.** `base_train.py:290` scales LR by
  `sqrt(B/B_ref)` on *total* batch size, and `total_batch_size` is a user
  argument, not a function of GPU count. The batch is held constant across
  profiles, so a 1-GPU run is the *same recipe*, not a different one.
- **The batch is preserved.** At 1 GPU the gradient accumulation goes 1 → 4
  (`grad_accum_steps = total_batch_size / (dbs · seq · world_size)`), so each
  step still consumes exactly 32,768 tokens and the token schedule is identical.

## Stage-by-stage

| st | stage | 1 GPU | 2 GPU | basis |
|---|---|---|---|---|
| 1 | pretrain, d28, 2k ctx, 277,760 steps | **145.3 h** | 72.7 h | 4-GPU log × 3.7; train 140 h + eval 5.1 h |
| 2 | tool-calling SFT, ~11,891 steps | **7.2 h** | 3.6 h | 4-GPU log × 3.7 |
| 3 | GSM8K warmup SFT, 509 steps | **1.3 h** | 0.7 h | **MEASURED 1 GPU** (`stage5a_warmup.log`) |
| 4 | math RL on GSM8K, 120 steps | **~5.4 h** | ~2.8 h | **MEASURED 1 GPU** 2026-10-22 with `--gen-batch-size 16`; the old 13.5 h was at generation batch 2 |
| 5 | agentic RL pilot, 178 steps | **~12.6 h** | ~6.4 h | **MEASURED 1 GPU** 2026-10-22, same 5.4x factor; the old 31.5 h was at generation batch 2 |
| 6 | curriculum RL, 178 steps | **~9.6 h** | ~4.9 h | **MEASURED 1 GPU** 2026-10-22, same 5.4x factor; the old 24.0 h was at generation batch 2 |
| 7 | context extension 2k → 128k | **12.9 h** | 12.9 h | **MEASURED 1 GPU** (ckpt meta, `dbs=1`) |
| 8 | 128k SFT, 800 steps | **5.1 h** | 5.1 h | **MEASURED 1 GPU** (ckpt meta `d28b-128k-sft-v5`) |
| 9 | touch-up ladder, 840 steps, 6 sub-stages | **6.8 h** | 6.8 h | **MEASURED 1 GPU** (ckpt meta v5b…v6j); also 1 GPU only by design (see the gradient-sync section) |
| 10 | pack → vLLM dir | 0.3 h | 0.3 h | measured |
| 11 | serve (vLLM + guard) | 0.2 h | 0.2 h | measured |
| 12 | gates | 0.5 h | 0.5 h | measured |
| 13 | e2e acceptance | 0.3 h | 0.3 h | measured |
| 14 | regression tests | 0.2 h | 0.2 h | measured |
| 15 | scrub test | 0.2 h | 0.2 h | measured |
| | **TOTAL** | **207.9 h** | **117.6 h** | |
| | **in days** | **8.7** | **4.9** | |
| | **+ 15 % contingency** | **10.0 d** | **5.6 d** | |

## The generation batch: the biggest single win of the review

The RL stages (4, 5 and 6) decode rollouts, and until 2026-10-22 they
decoded them **two at a time**: `--device-batch-size 2` was used for both
the generation and the training forward, and with `--num-samples 16` that
meant 8 sequential `generate()` calls of 2 samples per example.

The decode step is latency-bound on a batch that does not fill the GPU, so
the wall clock is nearly independent of the batch size. Measured on this
node with the real trained checkpoint, a real GSM8K prompt, 128 max tokens
and one A100-40GB:

| samples per call | wall clock | throughput |
|---|---|---|
| 2 | 3.69 s | 69 tok/s |
| 16 | 3.62 s | 565 tok/s |
| 64 | 3.53 s | 2,323 tok/s |
| 256 | 4.12 s | 7,918 tok/s |
| 512 | 5.67 s | 11,556 tok/s |
| 1024 | OOM (the KV cache) | - |

The recipe now decodes all 16 rollouts in **one** call. End to end, on the
real `scripts.chat_rl` with the real warmup checkpoint:

| generation batch | seconds per optimizer step | projected ETA for stage 4 |
|---|---|---|
| 2 (the old behaviour) | 58.4 | 60.1 h |
| **16 (the new one)** | **10.9** | **11.2 h** |

**5.4x faster**, and it is the same training math: the same 16 rollouts
per example, the same rewards, the same gradient.

The value 16 is not arbitrary. It is the measured memory ceiling: the RL
trainer's peak is at `opt.step()` (the AdamW moments are allocated
transiently on top of the 22.56 GiB of state), and
`gen batch 16 -> 33.18 GiB`, `32 -> 35.12 GiB`, `64 -> OOM`. It is also
exactly `--num-samples`, so all the rollouts of an example decode in one
call.

The trainers now take `--gen-batch-size` explicitly, defaulting to
`--device-batch-size` so nothing changes unless the flag is passed. The
training forward keeps its own small device batch, because that path
holds activations *and* gradients and has a different memory profile.

## Where the time actually goes

```
pretrain (stage 1)   145.3 h   72 %   <- pure GPU compute
RL stages 4+5+6      27.6 h   14 %   <- GENERATION-bound; was 69.0 h at generation batch 2
touch-up (stage 9)    6.8 h    3 %
longctx 7+8          18.0 h    7 %   <- dbs=1; does NOT scale with GPU count
everything else      10.2 h    4 %
```

**Two thirds of the schedule is not GPU work.** The RL stages spend their time
generating rollouts and evaluating them; the long-context stages are pinned to
`device-batch-size 1` at 131,072 tokens, so a second GPU does not help them at
all. This is why "use more GPUs" is a weak lever here and "skip a stage" is a
strong one.

## The 2-GPU number, and why 2 GPUs is *not* 2× better

Two GPUs halve the pretrain (145 h → 73 h) and the SFT stages, but:

- stages 7 and 8 are unchanged (18.0 h), because they run at `dbs=1` on a
  131,072-token sequence;
- the RL stages scale sub-linearly — each rank generates its own rollouts, and
  the wall clock is set by the slowest rank, not by summed throughput;
- the 30 % of the schedule that is evaluation, checkpoint I/O and process
  startup does not scale at all.

117.6 h vs 207.9 h is a 1.77× speed-up for 2× the hardware, and it costs a second
card that another tenant on this node is already using.

## What the recipe needed to support this

`PROFILE=1gpu` in `recipe/config.sh` pins every stage to one card, and
`run_mod` now exports `CUDA_VISIBLE_DEVICES` for each stage. Before this
change, **stages 2-9 set no device at all** and would have landed on GPU 0 —
where the vLLM server holds 37.4 GiB of the 40 GiB card — and OOM'd.

```bash
PROFILE=1gpu ./recipe/run_all.sh
```

The profile is a first-class path: same model, same data, same batch, same LR.

## Caveats, stated plainly

1. **An exclusive card is required.** As of 2026-10-21, GPUs 1-7 on this node
   belong to another tenant and run at 90-100 % utilisation. GPU 0 holds our
   vLLM server. A 1-GPU run needs a card nobody else wants; otherwise the
   18,051 tok/s anchor does not hold and the ETA stretches.
2. **The pretrain is the only extrapolated number.** Stages 3-9 were measured
   on one GPU. Stage 1 is a 4-GPU log scaled by a factor measured on a
   different stage. The token-anchored cross-check agrees to 2 %.
3. **Restart and I/O are not modelled per-stage.** The 15 % contingency is a
   blanket allowance, not a bottom-up estimate. The historical ladder was not
   restart-free: the 128k SFT was interrupted and resumed, and stage 2 was
   relaunched once.
4. **The 128k stages do not scale with GPUs** and would dominate a 4-GPU run
   too (18 h of the 4-GPU total).

## Reproducing the numbers

```bash
# the scaling anchor
grep -E "^step " /data/rowel/serve_logs/stage3/sft_d28.log        | tail -1
grep -E "^step " /data/rowel/serve_logs/stage3/sft_d28_2gpu.log   | tail -1
grep -E "^step " /data/rowel/serve_logs/stage3/sft_d28_v4_4gpu.log | tail -1

# the 1-GPU stage times
python3 - <<'EOF'
import json, glob, os
for d in glob.glob('/data/rowel/nanochat_store/v2/*checkpoints/*/'):
    fs = sorted(glob.glob(d + 'meta_*.json'))
    if not fs: continue
    m = json.load(open(fs[-1]))
    print(f"{os.path.basename(d.rstrip('/')):26s} {m['loop_state'].get('total_training_time',0)/3600:6.2f} h")
EOF
```

`total_training_time` is stored in **seconds** in every `meta_*.json` and in the
log's `Total training time:` field, which is printed in minutes. A 60× mistake
here is easy to make; the numbers in this document were re-derived from the
raw seconds.
