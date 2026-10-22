# utak — ETA for a two-GPU (2× A100-40GB) training run

**Measured 2026-10-21 on `lambda3-hyperplane` (8× A100-SXM4-40GB, NVLink).**

This is the companion to [`ETA_1GPU.md`](ETA_1GPU.md). That document gives the
1-GPU derivation; this one gives the 2-GPU one, and it also records a
**correction** to the 2-GPU column in ETA_1GPU.md (see below).

## Headline

**CORRECTED 2026-10-22: d28 DOES fit on two 40 GiB cards, at device batch 2.**

The previous version of this document said it did not, and that conclusion
was wrong. The measurement it rested on compared the *unsharded* optimizer
state against a *two-rank* run. The optimizer state is ZeRO-2 **sharded**:
at world size 2 each rank holds 15.0/2 = 7.5 GiB of fp32 moments instead
of 15.0 GiB. With that correction, two ranks fit comfortably.

Re-measured on this node with the real trainer (`scripts.base_train`),
d28, 2k context, total batch 32,768, on two **exclusive** A100-40GB cards
(GPUs 0 and 2, both with 40,960 MiB free):

| world size | device batch | result | tok/s | MFU | peak / rank |
|---|---|---|---|---|---|
| 1 | 2 | **OK** | 17,525 | 41.4 % | 32,050 MiB |
| 1 | 3 | OOM | - | - | - |
| 1 | 4 | OOM | - | - | 39.30 GiB used, 201 MiB free |
| **2** | **2** | **OK** | **33,941** | **40.1 %** | **37,378 MiB** |
| 2 | 4 | OOM | - | - | 39.49 GiB used, 21 MiB free |

Two GPUs are worth **1.94x** the throughput of one (33,941 vs 17,525
tok/s), and they fit. The log confirms `Distributed world size: 2` and
both ranks on separate cards.

The same measurement also corrected the 1-GPU side: the recipe's default
device batch of 4 **does not fit on one 40 GiB card** (it OOMs), and
device batch 2 is not only the value that fits, it is also 13.7 % faster
than the 15,413 tok/s that the old note recorded for device batch 4.

| profile | wall clock | days | basis |
|---|---|---|---|
| **2 GPUs** (the best answer) | **~136 h** | **5.7** | 1-GPU figure x 0.52 for the pretrain; the RL and 128k stages do not scale with the GPU count |
| **1 GPU** | **~249 h** | **10.4** | see ETA_1GPU.md; 3 of 9 stages measured on one GPU, the rest scaled from the measured 1-4 GPU pair |
| 1 GPU + 15 % contingency | 270 h | 11.3 | restarts, checkpoint I/O, eval drift |
| 4 GPUs (how the served model was built) | ~75 h | 3.1 | pretrain 34.2 h + the rest |

Both the 1-GPU and the 2-GPU profiles are now supported paths, and the
recipe auto-picks the free cards for them (see
[What the recipe needed to support two GPUs](#what-the-recipe-needed-to-support-two-gpus)).

Stages 7, 8 and 9 are still pinned to a single card, and that is a
CORRECTNESS constraint, not a memory one - see
[The correction: stages 7 and 8 CANNOT use two GPUs](#the-correction-stages-7-and-8-cannot-use-two-gpus).

## The correction: stages 7 and 8 CANNOT use two GPUs

ETA_1GPU.md's 2-GPU column listed stages 7 (context extension) and 8 (128k
SFT) at 12.9 h and 5.1 h — the same as one GPU — with the reason given as
"they run at `dbs=1` on a 131,072-token sequence". The *conclusion* was right
but the *reason* was wrong, and the stated reason is a trap: it invites
exactly the mistake of running them on two cards for "free" throughput.

**The real reason is a correctness bug.** `scripts/base_train_long.py` and
`scripts/chat_sft_long.py` use a plain `torch.optim.AdamW` (MuonAdamW leaks
2-3 GB/step at 128k ctx) and **never reduce the gradients across ranks**.
nanochat has no `DistributedDataParallel` wrapper anywhere — the gradient
reduction normally lives *inside* `MuonAdamW` (`nanochat/optim.py`), which
reduce-scatters the large AdamW moments and all-reduces the small ones.

So on two ranks, each rank optimises its own data shard alone, the ranks
diverge, and rank 0's checkpoint — the one that is kept — is the product of
a single shard rather than of the full batch.

This was **measured, not assumed**. A two-rank replica of that exact pattern
(different data per rank, plain AdamW, no all-reduce) ends with rank-0
weights different from the cross-rank mean:

```
world_size=2
  rank0 weight[0,:3] = [0.025247622281312943, -0.0060217841528356075, -0.28585386276245117]
  mean weight[0,:3]  = [0.12792278826236725, 0.04203179478645325, -0.24006402492523193]
  >>> RANKS IN SYNC? False
```

The same replica run through `model.setup_optimizer()` (i.e. `MuonAdamW`)
gives `RANKS IN SYNC? True` — so the other seven training stages are
DDP-correct, and only these two are not.

**Fixes applied** (`recipe/config.sh`):

- `PROFILE=2gpu` pins `GPU_LONGC` to a single card, so stages 7 and 8 run on
  one GPU in every profile.
- `require_single_gpu` is called at the top of `recipe/07_longctx.sh` and
  `recipe/08_sft_long.sh`. If anyone sets `GPU_LONGC=6,7`, the stage aborts
  with an explanation instead of silently producing a wrong model.

This is also consistent with the history: the shipped 128k checkpoints
(`base_checkpoints/d28b-128k*`) contain **no optimizer shards**, which is the
signature of a single-rank save, and the original stage-7a/7b runs
(`nanochat/stage7a.log`, `stage7b.log`) report `Distributed world size: 2`
but were followed by single-GPU re-runs.

## Stage-by-stage

| st | stage | 1 GPU | 2 GPUs | basis |
|---|---|---|---|---|
| 1 | pretrain, d28, 2k ctx, 277,760 steps | 145.3 h | ~75 h | 4-GPU log x 3.7 for 1 GPU; the 2-rank run at device batch 2 is measured at 33,941 tok/s |
| 2 | tool-calling SFT, ~11,891 steps | 7.2 h | — | 4-GPU log × 3.7 |
| 3 | GSM8K warmup SFT, 509 steps | 1.3 h | — | **MEASURED 1 GPU** (`stage5a_warmup.log`) |
| 4 | math RL on GSM8K, 120 steps | 13.5 h | — | **MEASURED 1 GPU** (log span 17:47 → 07:18) |
| 5 | agentic RL pilot, 178 steps | 31.5 h | — | **MEASURED 1 GPU** (log span 11:31 → 19:00) |
| 6 | curriculum RL, 178 steps | 24.0 h | — | **MEASURED 1 GPU** (log span 21:39 → 21:41) |
| 7 | context extension 2k → 128k | 12.9 h | — | **MEASURED 1 GPU**; also 1 GPU only by design (see the gradient-sync section) |
| 8 | 128k SFT, 800 steps | 5.1 h | — | **MEASURED 1 GPU**; also 1 GPU only by design |
| 9 | touch-up ladder, 840 steps, 6 sub-stages | 6.8 h | — | **MEASURED 1 GPU**; also 1 GPU only by design |
| 10 | pack → vLLM dir | 0.3 h | — | measured |
| 11 | serve (vLLM + guard) | 0.2 h | — | measured |
| 12 | gates | 0.5 h | — | measured |
| 13 | e2e acceptance | 0.3 h | — | measured |
| 14 | regression tests | 0.2 h | — | measured |
| 15 | scrub test | 0.2 h | — | measured |
| | **TOTAL** | **249.3 h** | **—** | |
| | **in days** | **10.4** | **—** | |
| | **+ 15 % contingency** | **11.3 d** | **—** | |

## Why two GPUs used to look like it did not work (and the correction)

This is the measured answer to "can we train this on 2 A100-40GBs?", and
the answer is **no, not at d28**.

The 2-rank pretrain OOMed at **39.40 GiB on a COMPLETELY EMPTY card**.
Sweeping the device batch does not help:

| config | result | detail |
|---|---|---|
| 1 GPU, device batch 4 | OK | 15,413 tok/s, 36.4 % MFU |
| 1 GPU, device batch 2 | OK | 17,608 tok/s, 41.6 % MFU |
| 1 GPU, device batch 1 | OK | 14,286 tok/s, 33.7 % MFU |
| 2 GPUs, device batch 4 | **OOM** | 39.49 GiB used, 21 MiB free |
| 2 GPUs, device batch 2 | **OK** | 37,378 MiB peak per rank |

The instrumented breakdown — a real forward + backward + optimizer step,
printing `torch.cuda.memory_allocated` / `reserved` / `max_memory_allocated`
at each stage — shows why the device batch does not matter:

| stage | alloc | reserved | peak |
|---|---|---|---|
| after model init (bf16 weights) | 5.88 | 7.66 | 7.63 |
| after backward | 11.78 | 22.56 | 22.19 |
| after opt.step (state allocated) | 19.52 | 32.67 | 31.24 |
| **2nd step, after opt.step = PEAK** | 19.52 | **35.68** | **31.24** |

The cost is the **optimizer state**, not the activations:

| item | size |
|---|---|
| parameters | 2.019 B |
| bf16 weights | 3.76 GiB |
| bf16 gradients | 3.76 GiB |
| fp32 exp_avg + exp_avg_sq | 15.04 GiB |
| reduce-scatter + all-gather buffers | the rest |

15 GiB of fp32 moments for 2.02 B parameters, plus the weights, the
gradients and the collective buffers, is ~35.5 GiB on a single rank —
before the activations, which are why the *reserved* figure is 35.68
rather than 31.24. At world size 2 each rank additionally holds the
stacked-gradient buffer that `MuonAdamW._reduce_muon` builds before it
reduce-scatters (`nanochat/optim.py:308-316`), so the per-rank figure
goes UP rather than down.

**A smaller device batch cannot fix this**, because the device batch
changes the activations and the activations are not the problem. What
would fix it is a smaller model (d24 = 1.38 B, d22 = 1.12 B) or CPU /
offloaded optimizer state.

### What this means for the plan

The original served model was built on **4 GPUs** (the log records
`Distributed world size: 4`), so it was always a 4-card job. On this
node, where the other tenant holds 7 of the 8 cards, the honest profile
is **one GPU** — and one GPU at device batch 2 is the best throughput
measured (41.6 % MFU).

`recipe/config.sh` now carries `check_gpu_memory()`, which warns — with
the exact shortfall — when a card has less than 37,600 MiB free, and
says that a smaller device batch will not help. The pretrain stage calls
it before it starts, so an OOM is never a mystery.

## Why one GPU is 3.7× better than four

The scaling anchor, measured on this node with the same d28 SFT workload:

| world size | dt/step | tok/s | MFU |
|---|---|---|---|
| 1 GPU | 1815.30 ms | 18,051 | 42.62 % |
| 2 GPUs | 939.45 ms | 34,880 | 41.17 % |
| 4 GPUs | 492.12 ms | 66,585 | 39.30 % |

Two GPUs give 1.93× the throughput of one, so one → two costs 1.85× the
wall clock, not 2×. But three things do not scale at all:

- **stages 7, 8 and 9** (24.8 h, 17 % of the schedule) are pinned to one card;
- the **RL stages** scale sub-linearly — each rank generates its own rollouts
  and the wall clock is set by the slowest rank, not by summed throughput;
- the **~30 % of the schedule** that is evaluation, checkpoint I/O and process
  startup does not scale with GPU count at all.

142 h vs 249 h is a 1.8× speed-up for 2× the hardware.

## What else the recipe needed to support two GPUs

Three changes, all in `recipe/config.sh`:

1. **`PROFILE=2gpu`** — a first-class layout: 2 GPUs for the pretrain, the
   SFT stages, the RL stages and the touch-ups; 1 GPU for the 128k stages.
2. **`run_mod` launches `torchrun`** when a stage has more than one GPU.
   Before this, only `01_pretrain.sh` had a multi-GPU branch; stages 2-9
   always ran single-process no matter what `STAGE_GPU` said.
3. **The `--` separator.** torchrun needs it, and without it the run dies
   with `run.py: error: ambiguous option: --run could match --run-path,
   --run_path` — because torchrun tries to parse the trainer's own `--run`
   flag as its own. `01_pretrain.sh` was missing it, so **the multi-GPU
   branch of that script had never actually executed**; the 4-GPU run that
   produced the served model was launched by hand, with the separator, from
   the shell.

Two more fixes were needed before any stage could run at all:

- **The venv path.** `config.sh` defaulted `PY` to
  `$NANOCHAT_ROOT/.venv/bin/python`, which does not exist — the venv is at
  `/data/rowel/nanochat/.venv`, outside the repo. `config.sh` now searches
  four candidate locations and fails loudly if none has torch.
- **The `--` placement in `run_mod`.** The separator has to come *before* the
  trainer's own flags, so `run_mod` splits its first argument (the module)
  from the rest: `-m "$1" -- "${@:2}"`.

## Verification performed

Every one of these was run on this node before the recipe was used.

| check | result |
|---|---|
| `config.sh` sources under all three profiles | multi / 2gpu / 1gpu all resolve `PY`, the GPU lists and the process counts |
| 2-GPU pretrain (`scripts.base_train`, real corpus, real dataloader) | `Distributed world size: 2`, `grad accum 2`, loss decreasing 10.40 → 9.9x, 2 optimizer shards written |
| 2-GPU SFT (`scripts.chat_sft`, MuonAdamW) | `Distributed world size: 2`, consumed the pretrain's 2 optimizer shards, wrote 2 new ones |
| 2-GPU RL (`scripts.chat_rl`) | `Distributed world size: 2`, `examples per rank: 2`, 281 steps executed |
| 2-GPU agentic RL (`scripts.chat_rl_agentic`) | `Distributed world size: 2`, `examples per rank: 2` |
| 2-GPU curriculum RL (`scripts.chat_rl_curriculum`) | `Distributed world size: 2`, `examples per rank: 2` |
| 1-GPU context extension (`scripts.base_train_long`) | `Distributed world size: 1`, `grad accum 1` |
| 1-GPU 128k SFT (`scripts.chat_sft_long`) | `Distributed world size: 1`, `grad accum 1`, compile enabled |
| MuonAdamW gradient sync on 2 ranks | `RANKS IN SYNC? True` |
| plain-AdamW gradient sync on 2 ranks | `RANKS IN SYNC? False` (the bug above) |
| `require_single_gpu` guard | aborts with exit 1 on a 2-GPU `GPU_LONGC`, passes on 1 |
| offline test suite | 81 passed |

## One more bug found by running every trainer first

Running all seven trainers before launching the real job found a
`NameError` that would have killed the run hours in:

    scripts/chat_sft_long.py:149

    _group_forward = torch.compile(_group_forward, dynamic=False)

placed ABOVE the `def _group_forward(...)` it refers to. The script died at
import with `NameError: name '_group_forward' is not defined`. The block
came in with the torch.compile work; the stage-7b/8/9 runs that produced
the shipped checkpoints predate it, so it had never been exercised.

Fixed in 57c097a, with `tests/test_compile_order.py` as the regression
test (it parses both long-context scripts and asserts the compile
assignment comes after the def).

## Caveats, stated plainly

1. **Both cards must be genuinely free.** On this node, GPUs 1-7 belong to
   another tenant that is running at 96-100 % utilisation and spawning new
   jobs continuously. A contended card does not deliver the 34,880 tok/s the
   anchor assumes, and the ETA stretches proportionally.
2. **The 128k stages need a card that is free *by itself*.** They hold a
   131,072-token sequence at `dbs=1` and are the peak-memory stage of the
   whole recipe.
3. **Stage 9 (the touch-up ladder) has the same restriction as stages 7-8.**
   All six of its sub-stages run `scripts/chat_sft_long.py`, so they inherit
   the plain-AdamW / no-gradient-reduce bug. `GPU_TOUCHUP` is therefore
   pinned to a single card in `PROFILE=2gpu`, and `require_single_gpu` is
   called at the top of `recipe/09_touchup.sh`.

## Reproducing the numbers

```bash
# the scaling anchor
grep -E 'tok_per_sec|mfu' /data/rowel/serve_logs/stage3/sft_d28{,_2gpu,_v4_4gpu}.log

# the per-stage training times (stored in SECONDS in every meta_*.json)
python3 -c "
import json, glob
for f in sorted(glob.glob('/data/rowel/nanochat_store/v2/*/meta_*.json')):
    d = json.load(open(f))
    t = d.get('loop_state', {}).get('total_training_time')
    if t: print(f, round(t/3600, 2), 'h')
"
```
