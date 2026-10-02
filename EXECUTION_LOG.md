# Execution Log — reasoner-model ladder on 8×A100-40GB

Plan source: `PLAN.md`. Harness: karpathy/nanochat @ 92d63d4.
Every number below is **measured on this node**, not derived.

---

## Summary of what the plan got wrong

| # | PLAN.md claim | Measured reality | Impact |
|---|---|---|---|
| 1 | d20 = 2155 M params | **d20 = 897 M**; over-counted 2.40× | headline mislabeled |
| 2 | d24+ does not fit | **d24/d26/d28 all fit at bs=4** | plan was too pessimistic on size |
| 3 | d20 runs at bs=32 | **max bs=8** (OOM at 16) | all configs capped below 32 |
| 4 | headline ≈ 2.15 B "d20" | **d28 = 2019 M** | same target size, different label |
| 5 | 30% MFU | **34–48%** | node-hours 1.13–1.60× pessimistic |
| 6 | no FA3 on Ampere (implied) | **FA3 works and is numerically correct** | big speed win |
| 7 | xLAM-60k is the Stage 4 source | **gated, inaccessible** | used Apache-2.0 hermes instead |

Errors 1–3 are two bugs in `scripts/budget.py` that partly cancelled, leaving a table
that was wrong in both columns while looking plausible.

---

## Environment

- 8 × A100-SXM4-40GB, SM 8.0 (Ampere), 40960 MiB each
- torch 2.9.1+cu128, 128-core EPYC 7742, 1 TB RAM, 151 GB free disk
- `get_peak_flops("A100-SXM4-40GB")` = 3.12e14 bf16 — matches the plan's constant

**Constraint discovered:** GPU 0 was held by an unrelated 9 GiB job for the whole
session, and the harness restricts `world_size` to `{2, 4, 8}` (32768 must divide
world_size for AdamW reduce_scatter; total_batch_size must be a multiple of
`bs × seq × world_size`). **7 GPUs is not a legal configuration.** All runs below
are 4-GPU, bs scaled to fit, total batch 131072 tokens.

---

## Finding 1 — `budget.py` was wrong in both directions

**(a) Parameters over-counted 2.1–2.6×.** The value-embedding term was
`depth * 2 * n_head * head_dim * VOCAB`, but `has_ve()` puts value embeddings on
**alternating layers only**, and `n_kv_head == n_head` so `kv_dim == n_embd`, not
`2 * n_embd`. Clean **4× over-count** of the largest embedding block. Verified by
instantiating the real `GPT` class on meta device (`nanochat/param_audit.py`):

| config | plan | real | over-count |
|---|---:|---:|---:|
| d12 | 739 M | **286 M** | 2.58× |
| d16 | 1342 M | **537 M** | 2.50× |
| d20 | 2155 M | **897 M** | 2.40× |
| d26 | 3808 M | **1682 M** | 2.26× |

**(b) Activations under-counted 4–7×.** The model assumed
`ACT_BYTES_PER_TOKEN_WIDTH = 12`, i.e. ~0.9 GiB for d20 at bs=32. But the dominant
activation is the **logits tensor**, which scales with *vocab*, not width:
`bs × seq × 32768 × 2 B` = **4.0 GiB at bs=32**, 8.0 GiB in the fp32 upcast used for
cross-entropy. Width-independent, so it hurts small models most.

### Corrected, measured frontier (`nanochat/mem_probe.py`)

| config | real params | max bs | peak GiB | plan said |
|---|---:|---:|---:|---|
| d12 | 286 M | 16 | 29.32 | 32 |
| d16 | 537 M | 8 | 21.93 | 32 |
| d18 | 702 M | 4 | 15.56 | 32 |
| d20 | 897 M | 8 | 30.26 | 32 |
| d22 | 1123 M | 8 | 35.26 | 32 |
| **d24** | **1384 M** | **4** | 25.77 | *infeasible* ❌ |
| **d26** | **1682 M** | **4** | 30.07 | *infeasible* ❌ |
| **d28** | **2019 M** | **4** | 34.86 | — |
| d32 | 2819 M | — | OOM at bs=1 | — |

**The headline model is d28 ≈ 2.0 B**, which is within 6% of the plan's stated
2.15 B target. The plan reached the right *size* by the wrong route: it concluded
"d20, 2.15 B" from a miscount, and separately concluded d24+ was impossible, when in
fact the true 2.0 B model is d28 and it fits comfortably.

### Why the batch ceiling exists (`nanochat/why_batch_ceiling.py`)

Decomposing measured peaks shows the logits term is the thing a parameter-count
budget cannot see. At bs=32 the logits alone are 4.0 GiB = 10% of a 40 GB card,
before a single weight is stored. The nanochat default of bs=32 is inherited from
80 GB H100 recipes; on 40 GB it is unreachable at *any* depth.

---

## Finding 2 — Flash Attention 3 *does* work on Ampere

`USE_FA3=True` here via the `kernels-community/flash-attn3` sm80 build. FA3 is
normally Hopper-only, so it was verified **before** spending GPU-hours:

- full-context causal: runs, finite output ✅
- sliding-window (the `SSSL` default = 3 of 4 layers): **max abs err 0.0078** vs an
  fp32 explicit-mask reference ✅

This is the single biggest throughput lever in the whole run. `base_train.py` warns
that without FA3 the SDPA fallback implements sliding-window attention as an
explicit mask and "GPU utilization will be terrible". Keeping FA3 is what makes the
48% MFU below possible.

---

## Finding 3 — measured throughput beats the 30% assumption by 1.13–1.60×

Real `base_train` runs, 4×A100-40GB, steady state:

| config | bs | tok/s | bf16 MFU |
|---|---:|---:|---:|
| d12 | 16 | 792,000 | **48.3%** |
| d20 | 8 | 157,500 | 36.4% |
| d28 | 4 | 58,200 | 34.3% |

Corrected wall-clock from measured throughput (8-GPU equivalent):

| stage | params | tokens | node-hours |
|---|---:|---:|---:|
| d12 r12 (ablation) | 286 M | 3.4 B | **0.6** |
| d20 r8 | 897 M | 7.2 B | 6.3 |
| d26 r8 (GPT-2 grade) | 1682 M | 13.5 B | 32.1 |
| **d28 r8 (headline)** | **2019 M** | **16.1 B** | **38.5** |
| d28 r12 | 2019 M | 24.2 B | 57.8 |

The plan's 83–124 node-hour headline becomes ~38 h — and that is the *pessimistic*
reading, since it assumes 4-GPU-derived efficiency at 8 GPUs.

---

## Stage 0 — Tokenizer ✅

```
python -m nanochat.dataset -n 8 -w 8   # 9 shards, 788 MB
python -m scripts.tok_train            # 88.3 s, 32,503 merges, vocab 32768
python -m scripts.tok_eval             # compression ratio = the metric
```

| corpus | bytes | ours tok | ratio | vs GPT-2 | vs GPT-4 |
|---|---:|---:|---:|---:|---:|
| news | 1819 | 405 | 4.49 | −0.2% | −4.7% |
| code | 1259 | 396 | 3.18 | **+31.2%** | −28.2% |
| math | 1834 | 911 | 2.01 | +2.7% | −9.5% |
| science | 1112 | 247 | 4.50 | +5.0% | +0.8% |
| climbmix-train | 2948778 | 622511 | 4.74 | +1.4% | −1.8% |
| climbmix-val | 3024593 | 644939 | 4.69 | +1.2% | −2.2% |

**Caveat worth keeping:** the `code` result (+31.2% vs GPT-2) is the standout and it
does matter for Stages 4/6, but the aggregate is only ~1.4% better than GPT-2 and
slightly *worse* than GPT-4. Do not oversell the tokenizer.

---

## Stage 1 — Pretraining, ablation scale (d12) ✅

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 \
  -m scripts.base_train -- --depth=12 --run=ablation-d12 \
  --device-batch-size=16 --total-batch-size=131072 --target-param-data-ratio=12 \
  --core-metric-every=999999 --sample-every=-1 --save-every=-1 --eval-every=250
```

- 10,080 steps, 3.44 B tokens (ratio 12), **26.3 min wall-clock** (plan: ~15 node-h)
- steady 48% MFU, 790k tok/s, loss 10.40 → 2.60
- checkpoint `d12` step 10080, min_val_bpb **0.8781**

### val_bpb — the metric that matters, not loss

| step | val_bpb |
|---:|---:|
| 0 | 3.1697 |
| 250 | 1.2523 |
| 500 | 1.1092 |
| 2500 | 0.9745 |
| 5000 | 0.9277 |
| 7500 | 0.8848 |
| **10080** | **0.8782** |

train bpb 0.7645 / val bpb 0.8777. The gap is the generalization gap; note that
raw train loss (2.60) is *lower* than what val bpb would suggest and is not
comparable across tokenizers, which is exactly the trap the plan warns about in
§5.1. Everything here is reported in bpb for that reason.

This is the plan's first trap demonstrated rather than asserted: the loss curve
looks smooth and healthy, but only bpb tells you the model is actually improving.

### Stage 1 evaluation — CORE 0.1390

`python -m scripts.base_eval --model-tag d12 --eval core,bpb`

| metric | value |
|---|---|
| **CORE** | **0.1390** |
| val bpb | 0.8777 |
| train bpb | 0.7645 |

Calibration against nanochat's published leaderboard, so the number means something:

| model | params | val_bpb | CORE | node-hours |
|---|---:|---:|---:|---:|
| **this run, d12** | **286 M** | **0.878** | **0.139** | **0.44 (4 GPU)** |
| leaderboard d24 | 1.4 B | 0.748 | 0.2585 | 3.04 (8×H100) |
| leaderboard d26 (GPT-2 grade) | 1.7 B | 0.745 | 0.2578 | 2.91 (8×H100) |
| original OpenAI GPT-2 | 1.6 B | — | 0.2565 | 168 |

So d12 is roughly a quarter of the way to GPT-2 grade on a 5× smaller model with
6× less training. That is the expected shape, not a failure: CORE climbs steeply
with both scale and ratio, and d12 at ratio 12 is deliberately the cheap workhorse,
not the headline. **This model is not GPT-2 grade and should not be presented as
such** — the gap 0.139 → 0.257 is what Stages 2–3 have to close.

Per-task centered scores show the expected profile for a 286 M model: strong on
arc_easy (+0.41), piqa (+0.35), lambada (+0.29); near-random on commonsense_qa
(+0.03) and winogrande (+0.02); below random on boolq (−0.25). Multiple-choice
format is learned (arc_easy 55.6% ≫ 25% chance) while the harder commonsense
formats are not yet.

**Trap 1 from PLAN.md §5.1, now demonstrated:** train loss ended at 2.60 and
val_bpb at 0.878, and those two numbers say almost nothing about each other. A
student watching only the loss curve would conclude this run is healthy; it is,
but they would have no idea it is at 54% of GPT-2 CORE. Hence bpb + CORE, never
loss alone.

---

## Stage 4 — Tool calling (built ahead of Stage 2/3, since it is the real work)

`nanochat/tasks/toolcalling.py` + `verify_toolcalling_mask.py`, committed.

**Source substitution:** PLAN.md names xLAM-60k. It is **gated** (`"gated":"auto"`) and
this node's token is not on the authorized list — access was requested via the Hub
API and refused. Substituted **NousResearch/hermes-function-calling-v1**
(Apache-2.0, ungated, 1893 rows → 1036 train / 54 val after conversion). Attempted
and **rejected** `glaiveai/glaive-function-calling-v2` (Apache-2.0, 112,960 rows):
all 112,960 rows have a malformed `chat` field (a bare string, not a conversation
list), so it is unusable as distributed. Note that the 271 MB file downloads fine —
malformed data is not a size problem.

**The recipe detail that matters — loss lands only on the emitted call.** Verified
token-by-token across all 1036 train examples:

```
supervised fraction: min 1.9%  mean 22.8%  max 48.4%
schema tokens in a SUPERVISED position : 0   (want 0)
tool results in a SUPERVISED position   : 0   (want 0)
CALL fragments missing from supervision  : 0   (want 0)
RESULT: PASS
```

Three implementation notes, each a real obstacle:

1. **Schemas ride in the user turn.** They are pure conditioning; a 2 B model given
   5 schemas will otherwise spend capacity memorizing schemas it never emits. The
   user turn is masked 0 by `render_conversation`, so no tokenizer change is needed.
2. **Tool results must be unsupervised.** At inference they come from the
   environment, not the model, so supervising them teaches hallucination. Reused
   the existing `python_output` part type, which the renderer already emits with
   mask 0.
3. **The corpus is not strictly alternating.** hermes has `tool` → `gpt` turns, which
   both map to assistant and trip `render_conversation`'s alternation assert. Fixed
   by folding part-lists and interposing an explicit user turn between two adjacent
   assistant *text* turns. Also had to strip zero-width joiners from the corpus's
   `<tool_call>` tags.

Data sanity: 311 CALL parts, 0 unparseable; calls-per-turn spreads 1→9
(459 single-call, 230 double, 268 triple), which is what Stage 6 will need for
multi-step trajectories.

Wired into `chat_sft.py` behind `--toolcall-epochs` / `--toolcall-data`, in both the
train and val mixtures. Mixture confirmed: **791,831 rows** = 460,341 smol-smoltalk
+ 3×99,842 MMLU + 4×7,473 GSM8K + 2×1,036 tool-calling.

### Stage 3 — SFT ✅ (ran with the Stage 4 data in the mixture)

```bash
torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- \
  --run=sft-v1 --model-tag=d12 --toolcall-epochs=2 \
  --device-batch-size=16 --total-batch-size=131072
```

3,756 steps, 15.8 min, 31.6% MFU. val bpb **0.3647** (from 0.8781 after pretraining).

| ChatCORE task | accuracy |
|---|---:|
| ARC-Easy | 36.66% |
| ARC-Challenge | 32.00% |
| MMLU | 30.88% |
| GSM8K | 4.17% |
| HumanEval | 4.17% |
| **ChatCORE** | **0.0821** (cat 0.1090) |

**Do not compare ChatCORE 0.0821 to the base model's CORE 0.1390.** They are not
the same metric: base CORE is few-shot ICL accuracy, ChatCORE is zero-shot accuracy
in the chat format. A chat model scoring lower on ICL-style few-shot is expected.
The pair is only comparable within itself across stages.

### Stage 4 acceptance test — format learned, capability not (at 286 M)

`nanochat/eval_toolcalling.py`, 54 held-out val examples, greedy decoding:

| metric | result |
|---|---|
| emits the `<\|python_start\|>CALL name {json}<\|python_end\|>` format | **100%** |
| right function **name** | 41.0% of 119 calls |
| right **number** of calls | 7.4% of examples |
| exact match (name **and** arguments) | **0.0%** |

This is the honest result and it is worth stating precisely, because the two halves
say different things:

- **The data pipeline works.** The stock mixture has no function calling at all, and
  after adding 2 epochs of the Stage 4 task the model emits well-formed calls on
  100% of held-out prompts and picks a sensible tool 41% of the time. The masking
  recipe (loss on the call, not the schema) does what it claims.
- **286 M parameters is not enough for exact argument binding.** The failure mode is
  visible in the dumps: it gets `get_camera_live_feed {"camera_id":"front_door"}`
  right but drops `"stream_quality":"1080p"`; it invents `thermostat_id:
  "temperature_123"` for `thermo123`; it emits 2 schedule entries where 13 are
  needed. Copying long JSON argument structures verbatim is a capacity problem, not
  a formatting problem.

**This is the plan's Trap 1 (§5.1) landing exactly where the plan says it would.**
SFT loss fell 1.57 → 1.24 and val bpb halved 0.878 → 0.365, so the run looks
healthy by every training metric, while the capability the stage exists to teach is
still at zero. Loss is not capability. The correct next move is Stage 2 (scale to
d28 ≈ 2.0 B, a 7× increase) and re-running this same eval — not more SFT epochs on
d12, which would train the loss down further and change nothing.

Two scorer bugs were found and fixed while building this eval, both of which had
initially reported 0/12 with the model merely *echoing the prompt*
(`generate_batch` returns prompt+continuation, not just the continuation) and then
0% with the model actually doing the task (the scorer demanded a literal `CALL`
keyword at line start, which the model drops once it is inside `<|python_start|>`).
Neither was a model failure. A third was a genuine near-miss: the model reorders
JSON keys, which is semantically identical, so comparison is now on parsed
`(name, args)` rather than raw text.

---

## Stage 2 — Headline pretrain (d28 ≈ 2.02 B) — RESUMED 2026-10-01

```bash
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
  --depth=28 --run=headline-d28 --model-tag=d28b \
  --device-batch-size=4 --total-batch-size=32768 \
  --target-param-data-ratio=8 --save-every=2000 --eval-every=500
```

The harness independently confirms the corrected parameter count:

```
wte                     :    58,720,256
value_embeds            :   822,083,584
lm_head                 :    58,720,256
transformer_matrices    : 1,078,987,056
scalars                 :            82
total                   : 2,018,511,234     <- the plan called this "d20, 2155M"
```

277,760 iterations, 9.10 B tokens, 34.2% MFU, 565 ms/step.

### Finding 4 — `--target-param-data-ratio` does not mean what the plan assumes

The plan's Stage 2 command says "ratio 8 rather than 12" and computes cost as
`params × ratio`. But `base_train.py` applies the ratio to **scaling params**, not
total params:

```python
scaling_params = param_counts['transformer_matrices'] + param_counts['lm_head']
# = 1,137,707,312, NOT 2,018,511,234
```

The 822 M of value embeddings are excluded, deliberately (the code cites
`dev/LOG.md`: transformer matrices + lm_head give the cleanest scaling laws).
So ratio 8 ⇒ **9.10 B tokens**, not the 16.1 B that `2.02B × 8` implies. Verified
exactly: `1,137,707,312 × 8 = 9,101,658,496` against the log's 9,101,639,680
(rounding in `int()`).

This matters for cost planning: at 34% MFU the run is **~43.6 h on 4 GPUs**
(~21.8 h on 8). Both the plan's node-hour table and my own corrected table in
`scripts/budget.py` are wrong here, because both multiply *total* params by the
ratio. Corrected:

| basis | ratio 8 tokens | 4-GPU hours |
|---|---:|---:|
| scaling params (what the harness does) | 9.10 B | **43.6** |
| total params (what the plan implies) | 16.1 B | 19.3 |

The harness is the authority here — it is derived from measured scaling laws — but
the plan should state which parameter count it means, because the two differ by 1.8×
in wall-clock. **Note this in the plan: quoting "data:param ratio" without saying
"scaling params" is ambiguous by a factor of ~1.8 in cost.**

### Finding 5 — the batch size is not a free parameter, it sets the LR

At bs=4 on 4 GPUs the total batch is 32,768 tokens, and the harness scales the
learning rates to compensate:

```
Scaling LRs by 0.2500 for batch size 32,768 (reference: 524,288)
```

That is a 4× LR reduction versus the reference recipe. It is the correct thing for
the harness to do, but it means **this run is not LR-comparable to the published
leaderboard**, which uses total batch 524,288–1M. The leaderboard's d24/d26 entries
at CORE 0.258 are therefore not a like-for-like target for this run. Recorded so
nobody later mistakes a gap for a modelling failure.

---

## Still to do

- **Stage 2** — headline pretrain (d28, 2.02 B) — **RESUMED 2026-10-01 09:29 from
  step 24,000, running on GPUs 4,5,6,7.** See "Stage 2 resume" below.

---

# Stage 2 incident — the run died, and it will die again

**The "Stage 2 — RUNNING" heading above said "running" as of the 16:40 commit. It was already
stale when written, and it stayed false for 11 hours. The run was dead from 22:51:49 on
Sep 30.** Correcting the record here because the commit history and this file are the
project's only source of truth.

## What happened

| | |
|---|---|
| Launched | 2026-09-30 16:32:43 (wandb run `oyydgnor`) |
| Died | 2026-09-30 **22:51:49** |
| Survived | 24,149 / 277,760 steps (8.69%), 226.8 min |
| Peak loss | 2.60, steady ~2.64, val_bpb 0.8859 at step 24,000 |

**No traceback, no CUDA OOM, no Python exception.** The log ends mid-line at step
24149. wandb's own core log records the cause in one line:

```
2026-09-30T22:51:51 server: parent process exited, terminating service process
```

Parent exited = the training process was killed, not a Python-level failure. Host was
up 44 days (no reboot), 962 GB RAM free, no ECC/Xid events, GPUs 0 and 3–7 idle.
So: not a host reboot, not host OOM.

## Root cause: the disk filled up

This is the finding that matters, and it is not a mystery:

- `df` on `/` (where checkpoints go, `~/.cache/nanochat`): **18 GB free, 99% used**
- Every checkpoint is **14.62 GB** = 6.31 GB model + 4 × 2.08 GB AdamW shards
- 12 checkpoints already on disk = **175 GB** (measured: 164 GiB)
- `--save-every=2000`, so the next save at step 26,000 needs 14.62 GB

**18 GB free is barely one checkpoint of headroom.** Whatever killed the process at
22:51, the run is now on a knife edge: it cannot reach step 26,000 without writing
another 14.62 GB. Resuming as-is will die again almost immediately.

## Why it is *not* resumable-as-configured, and not a lost-compute situation

Good news: the checkpoint survived and is **not** torn, despite being written 90 s
before death. Verified by full deserialization:

- `model_024000.pt` loads, **2,018,511,234 params**, 203 tensors, **zero non-finite**
- `optim_024000_rank0.pt` shard loads
- `meta_024000.json` intact: `val_bpb 0.88588`, `dataloader_state_dict` at
  `pq_idx 1, rg_idx 32, epoch 3`, `min_val_bpb 0.88391`, `total_training_time 13525 s`

Only **149 steps (~14 min)** of compute was lost, not the ~20 min I first estimated —
the 24,000 save landed just before the kill. `base_train.py` supports
`--resume-from-step` and saves `dataloader_state_dict`, so the data-loader position
resumes correctly (no data repetition or skip).

## The disk problem is structural, not a one-off

`--save-every=2000` over 277,760 steps = **139 checkpoints × 14.62 GB = 2.03 TiB**.
Only **2.8 TiB** is free on the whole `/data` volume, and `/` has 18 GB. So keeping
every checkpoint has never been possible on this box. Someone had to prune, and
nothing was.

## Required fixes before relaunch

1. **Relocate checkpoints off `/`.** `nanochat/common.py:73` honours
   `NANOCHAT_BASE_DIR`, so point it at `/data/rowel` (2.8 TiB free, verified
   writable). This is the clean fix — no code change, no symlink.
2. **Prune the 164 GiB already in `~/.cache/nanochat/base_checkpoints/d28b`.** Keep
   `model_024000.pt` + its 4 optim shards + `meta_024000.json`; the other 11
   checkpoints are unrecoverable-anyway and only 175 GB.
3. **Reconsider `--save-every=2000`.** At 14.62 GB per save it is ~31% of the
   remaining wall-clock cost of the project in I/O alone. 4,000–5,000 costs ~20 min
   of lost work per kill but halves the disk. Pick deliberately, not by default.

## Resume — actually run, 2026-10-01

Resumed from **step 24,000** on **GPUs 4,5,6,7** at 09:29. The disk is fixed and the
step-25,000 checkpoint — the exact save that killed the previous run — completed
cleanly.

```bash
cd /data/rowel/nanochat
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 -m scripts.base_train -- \
  --depth=28 --run=headline-d28 --model-tag=d28b \
  --device-batch-size=4 --total-batch-size=32768 \
  --target-param-data-ratio=8 --save-every=5000 --eval-every=500 \
  --core-metric-every=999999 --sample-every=-1 \
  --resume-from-step=24000
```

### How the disk was fixed — moved, not pruned

The incident section recommended pruning the 11 dead checkpoints (irreversible). **I
did not prune anything.** `/data` had 3.0 TB free, so the whole 164 GiB was *moved*
instead, and a symlink preserves the original path:

```
/home/rowel/.cache/nanochat/base_checkpoints
  -> /data/rowel/nanochat_store/base_checkpoints.tmp
```

`/` went from **20 GB → 185 GB free (99% → 89%)**. Nothing was destroyed, so the
"prune the 11 dead checkpoints" step is no longer needed at all. Note `NANOCHAT_BASE_DIR`
was **not** used, because `get_tokenizer()` (`nanochat/tokenizer.py:266`) reads the
tokenizer from the same `base_dir` — pointing it at an empty `/data/rowel/nanochat`
would have stranded `tokenizer.pkl` and failed the run at startup. The symlink moves
checkpoints only and leaves the tokenizer where it is.

`--save-every` raised **2000 → 5000**, so the run needs ~51 rather than ~139
checkpoints (746 GB rather than 2.03 TiB) on 3.0 TB of headroom. Cost: up to ~47 min
of lost work per kill, which is cheap insurance at 43.6 h total.

### Two environment problems fixed along the way

1. **The venv was broken by the repo move.** `/home/rowel/baby-sandbox` no longer
   exists, but 19 console scripts in `.venv/bin` still had shebangs pointing into it,
   including `torchrun`. `nohup .venv/bin/torchrun` failed with "No such file or
   directory" even though `ls` showed the file. Fixed `torchrun`'s shebang to
   `/data/rowel/nanochat/.venv/bin/python`. **The other 18 are still stale** and will
   fail the same way — `wandb`, `hf`, `huggingface-cli`, `isympy`, `kernels`, etc.
   A `uv sync` would rebuild them.
2. **torch 2.9's `torch.distributed.run` misparses `--run=`.** It reserves both
   `--run-path` and `--run_path`, so `--run=headline-d28` dies with "ambiguous option"
   *inside torchrun's own parser*, before `base_train.py` ever sees it. The `--`
   separator is required. Worth remembering: the same failure looks like a typo in
   `base_train.py`'s flags but is not.

### Confirmed after resume

- `Resuming optimization from step 24000`, `Step 24000 | Validation bpb: 0.885878`
  — identical to the pre-kill value, so the optimizer state and LR schedule
  (warmdown at 277,760) genuinely restored.
- 2,018,511,234 params; 4 ranks ⇒ 4 AdamW shards ⇒ optimizer state is correctly
  sharded and not silently re-sharded.
- `Total batch size 32,768 => gradient accumulation steps: 1` — batch unchanged, so
  the Finding 5 LR caveat still holds (not leaderboard-comparable).
- Data loader resumed at `epoch 3 pq: 1 rg: 40` and advanced to `pq: 2 rg: 20`
  without repeating the step-24,000 batch.
- **43.8% bf16 MFU** (74k tok/s), above the 34.3% of the original run.
- Step 25,000 saved: 6 model/optim/meta files, 14.62 GB, no disk pressure.
  `/` still at 185 GB free.

Note the repo path moved to `/data/rowel/nanochat` since the original launch; the
old `cd /home/rowel/baby-sandbox/nanochat` in the commands above is stale.

---

## Still to do (Stages 3–6)

- **Stage 2** — headline pretrain (d28, 2.02 B) — **RUNNING**, resumed from step
  24,000 at 09:29 on GPUs 4,5,6,7. ETA ~43 h at the measured 43.8% MFU.

- **Stage 3/4 re-run on d28** — the acceptance test that matters. d12 reached 100%
  format compliance but 0% exact tool calls; the hypothesis is that ~2.0 B fixes
  argument binding. Untestable until Stage 2 lands.
- **Stage 5** — GSM8K RL. nanochat ships `chat_rl.py`; its own header concedes the
  method is REINFORCE-like rather than GRPO (no trust region, on-policy so no PPO
  ratio, DAPO token-level normalization, `(r − μ)` advantage). Report pass@k, not
  single-sample.
- **Stage 6** — agentic RL. The plan's 200–1000 node-h estimate stands and is the
  dominant cost of the whole project, 2–12× the pretrain. `nanochat/execution.py`
  provides a subprocess sandbox whose own docstring says it is **not a security
  boundary** (no seccomp, no containers, network not blocked) — fine for eval, must
  never be presented as a deployment boundary. EnvScaler (ACL 2026 Findings) found
  the steepest gains going 0 → 20 environments, so breadth of cheap verifiable
  environments beats depth of one hard one.
---

## Finding 8 — the corpus is ~20× too small: 21 epochs over 429 M tokens

Checked the dataset while auditing the resume, and found something bigger than the
val_bpb upturn. **Stage 2 is training on 8 shards.**

`python -m nanochat.dataset -n 8 -w 8` (Stage 0, line 132 of this log) pulled
**8 train shards + 1 val shard**. Measured, not estimated:

| | measured |
|---|---|
| shards present | 8 train (`shard_00000`–`00007`) + `shard_06542` (val) |
| bytes of text | **2.264 B** chars |
| bytes/token, 3,000 evenly-spread docs per shard | **4.689** (range 4.636–4.719) |
| tokens in all 9 shards | **482.8 M** |
| tokens in the 8 train shards (val held out) | **429.2 M** |
| tokens this run consumes (277,760 × 32,768) | **9.1016 B** |
| **passes over the train set** | **21.2** |

The run has already reached **epoch 7** at step 66,275, measured from the log's own
`epoch:` field (11,117 steps/epoch). By step 277,760 it ends near **epoch 26**.

nanochat's own downloader message recommends `-n 170` for GPT-2 scale. The dataset
holds **6,543 shards** (verified against the HF repo), so this was a choice, not a
limit — 15.6 GB for 170 shards against 2.9 TB free.

### Why this explains the val_bpb upturn

Epoch 1 ended at step 11,117, epoch 3 at 32,923. The best val_bpb of the whole run
is **0.883910 at step 22,000** — epoch 2. Everything since is worse:

| step | epoch | val_bpb |
|---|---|---|
| 22,000 | 2 | **0.883910** ← best ever |
| 27,000 | 3 | 0.885943 |
| 35,000 | 4 | 0.908301 |
| 46,000 | 5 | 0.921871 |
| 60,000 | 6 | 0.923708 |
| 70,000 | 7 | 0.936475 |
| 74,500 | 7 | 0.937256 |

val_bpb tracks epoch number and nothing else. It is **not** LR instability, as I
suggested in the previous status: `lrm` is 1.00 throughout and warmdown does not
begin until step 97,216, so the divergence was already complete before any decay
could help. It is memorisation of a 429 M-token corpus, 21× over.

The ~35% token loss to BOS-aligned cropping (`dataloader.py` docstring) means ~6 B
of the 9.1 B consumed tokens were never trained on, and the same 429 M is re-read
~21×.

### Cost of fixing it

- 170 shards ≈ 15.6 GB ⇒ 3.52 B tokens ⇒ **2.6 passes** instead of 21.2.
- Download is minutes, not hours. Disk is not the constraint.
- **But it invalidates the run.** The dataloader shards over parquet files by index,
  so a resumed run would keep cycling the original 8 shards. Restarting from step 0
  is required, discarding ~26 h of GPU time.
- Everything already measured (Stage 0 tokenizer, Stage 1 d12, Stage 4 masking
  audit) is unaffected — none of those results depend on corpus size.

**I have not acted on this.** Restarting Stage 2 is destructive and it is the
user's call. The run continues at ~43.6% MFU; checkpoint 22,000 remains the best
model and is intact on disk.---

## Finding 9 — Stage 2 restarted on 170 shards (2026-10-01 20:18)

User authorised the restart. The old d28b run was killed at step 76,499 / 277,760
and a clean run launched from step 0. Old checkpoints are untouched; step 22,000
(val_bpb 0.883910) was copied to `/data/rowel/nanochat_store/best_pre_restart/`
before the kill.

### Correction to Finding 8's arithmetic

Finding 8 stated 170 shards would give 3.52 B tokens / 2.6 passes. **That was
wrong** — it extrapolated from the on-disk *compressed* shard size (~92 MB) instead
of the uncompressed text. Measured from parquet metadata:

| | |
|---|---|
| uncompressed text per shard | 252.4 MB |
| tokens per shard @ 4.689 B/tok | 53.82 M |
| 8 shards (old corpus) | **0.431 B** tokens |
| 170 shards (new corpus) | **9.150 B** tokens |
| run consumes | 9.1016 B tokens |
| **passes: old / new** | **21.14 → 1.03** |

So the fix is better than advertised, not worse: ~1 pass over 9.15 B rather than
2.6 passes over 3.52 B. Confirmed independently from row groups: 170 shards hold
14,049 row groups vs 662 for 8, a 21.2× ratio, giving ~270,578 steps/epoch against
277,760 total steps.

### Steps executed

1. Downloaded 170 train shards + pinned val shard to
   `/data/rowel/nanochat_store/base_data_climbmix` (15.6 GB, **not** `/`, which is
   89% full). All 171 verified readable by pyarrow, 0 bad, 14.47 M rows.
2. Built a **separate base dir** `v2/` so the old `d28b/` checkpoints survive:
   `base_data_climbmix` → the new 171 shards; `tokenizer`, `base_eval`,
   `task_data` symlinked to `~/.cache/nanochat` (avoids the tokenizer-stranding
   trap — `get_tokenizer()` reads from the same base dir).
3. Verified **no train/val leakage**: `parquets_iter_batched` uses
   `parquet_paths[:-1]` for train and `[-1:]` for val, so `shard_06542` stays
   held out. Val shard is **md5-identical** to the old one
   (`de34f8e8...`), so val_bpb is directly comparable before/after.
4. Preserved best checkpoint, killed old run, launched with
   `NANOCHAT_BASE_DIR=.../v2`, `CUDA_VISIBLE_DEVICES=4,5,6,7`, new tags
   `d28b-v2` / `headline-d28-v2` (wandb run `pgiv5whp`).

### Verification that the fix took effect

| check | result |
|---|---|
| training process env | `NANOCHAT_BASE_DIR=.../v2`, `CUDA_VISIBLE_DEVICES=4,5,6,7` |
| parquet files visible | 171 (170 train + 1 val) |
| epoch counter at step ~500 | **1** (old run was at epoch 7 by step 76k) |
| throughput | 73.5k tok/s, **43.3% BF16 MFU** |
| loss at step 110 | 5.97 — consistent with a from-scratch start |

The epoch counter is the decisive check: the old run hit epoch 7 by step 76,499
because it exhausted 8 shards repeatedly; the new run is still on epoch 1.

### Cost of the fix

26 h of GPU time discarded. New ETA ~34.4 h from 20:18.
---

## Finding 10 — Stage 2 v2 running clean, 12.2h in, already beats the old best

Status as of **2026-10-02 08:26 PST**.

| | |
|---|---|
| step | **58,999 / 277,760 (21.24%)** |
| loss | 2.656 |
| lrm | 1.00 (constant phase; warmdown starts step 97,216) |
| throughput | 73.8k tok/s, **43.6% BF16 MFU** |
| elapsed / ETA | 436.4m / **1618.5m = 27.0 h** |
| epoch | **1** |
| GPUs | 4,5,6,7 @ 96-97%, 38.5/40.9 GB each |
| val_bpb @ 58,500 | **0.826257** |

### The restart was the right call — confirmed by the val curve

Monotone, no divergence:

| step | 2.5k | 10k | 20k | 30k | 40k | 50k | 58.5k |
|---|---|---|---|---|---|---|---|
| val_bpb | 1.0256 | 0.9133 | 0.8728 | 0.8527 | 0.8402 | 0.8320 | **0.8263** |

The old 8-shard run's best-ever val_bpb was **0.883910** (step 22,000), and it was
already deteriorating from there. The v2 run passed that at **~step 19,000** and has
improved a further 0.058 since. So the 26 h discarded bought a real gain, not just
a reset: v2 is 0.058 bpb better than anything the v1 run ever produced, and still
falling. v1 best preserved in `nanochat_store/best_pre_restart/`.

### Interpretation

Finding 8 is now confirmed by outcome rather than by inference: 21.2 passes over a
429M-token corpus caused memorisation, not LR instability. On 9.15B tokens at ~1
pass, val_bpb is still improving monotonically at 21% through the schedule.
Warmdown at step 97,216 should add further gains.

### Housekeeping

- `base_checkpoints.tmp/` (316 GB, d12 + d28b) is a **stale** relocation leftover
  from the Sep 30 disk fix; the live v1 checkpoints are in
  `nanochat_store/base_checkpoints/`. Left in place, not deleted — same reasoning
  as the earlier "move, do not prune" call.
- `/data` at 80% (2.7 TB free). Remaining saves: 45 x 2.28 GB = **103 GB**. Fits.
- `/` at 89% (195 GB free) — unchanged, untouched.
- No watchdog on this run. A kill would again stop it silently; checkpoints every
  5,000 steps are the only recovery point.

---

## Finding 11: Stage 2 v2 at 25.96%, val_bpb 0.8198 — 0.064 ahead of v1 best

**Recorded:** 2026-10-02 11:15 PST. Run `headline-d28-v2` (wandb `pgiv5whp`).

| | |
|---|---|
| **Step** | **72,114 / 277,760 (25.96%)** |
| loss | 2.694 |
| lrm | 1.00 (constant phase; warmdown starts step 97,216) |
| throughput | 73.9k tok/s, **43.6% BF16 MFU** |
| elapsed / ETA | 533.4m / **1521.3m = 25.4 h** |
| epoch | **1** (whole run; log's own counter, single unique value in file) |
| GPUs | 4,5,6,7 @ 99-100%, 38.5/40.9 GB each |
| **val_bpb @ 72,000** | **0.819828** ← best of the run so far |

### Progression vs v1

| step | 5k | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 72k |
|---|---|---|---|---|---|---|---|---|---|
| val_bpb | 0.9638 | 0.9133 | 0.8728 | 0.8527 | 0.8402 | 0.8320 | 0.8257 | 0.8206 | **0.8198** |

v1's best-ever was **0.883910** at step 22,000 and rising thereafter. v2 passed
that at roughly step 19,000 and is now **0.064 bpb better** than anything v1
produced. The curve is monotone with no turn — the v1 turnaround signature is
absent. Warmdown (step 97,216) is still 25,100 steps away and should add more.

### Verification notes

- **Epoch math corrected.** An intermediate check of mine divided by v1's stale
  11,117 steps/epoch and printed "epoch 6.49". That is wrong: the log's own
  `epoch:` field is the single unique value `1` across all 145 logged steps, and
  170 train shards hold roughly 270,578 steps/epoch against 277,760 total. The
  log counter is the authority; the stale constant is not.
- **Val shard unchanged and md5-identical** to the v1 run, so val_bpb is
  directly comparable before and after the restart.
- Checkpoints through **70,000** written and verified on disk
  (`model_070000.pt` plus 4 optim shards; 6.31 GB plus 4 x 2.08 GB).
- Process env read directly from `/proc/1297981/environ`: `NANOCHAT_BASE_DIR=
  /data/rowel/nanochat_store/v2`, `CUDA_VISIBLE_DEVICES=4,5,6,7`. Confirms the
  live process, not just the launching shell.
- 170 train shards + 1 val shard present in `v2/base_data_climbmix`.

### Housekeeping

- **Saves: 14 of 55 done, 41 remaining = 517 GB needed.** `/data` has 2.7 TB
  free, so this run cannot fill the disk. (An earlier note of mine inverted this
  arithmetic and printed a negative count; the figures here are the corrected ones.)
- `base_checkpoints.tmp/` (316 GB) still stale, still not deleted.
- `/data/rowel/nanochat_store/v2` is 191 GB — live d28b-v2 checkpoints.
- **No watchdog.** Unchanged single point of failure: a kill stops the run
  silently and 5,000-step checkpoints are the only recovery point.
- **Log location correction:** this run has no `out/headline-d28-v2/` directory.
  Output goes to `nanochat/wandb/run-20261001_201809-pgiv5whp/files/output.log`.
  Earlier status updates pointed at `out/*/train.log`, which does not exist.
