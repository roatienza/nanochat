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

## Still to do

- **Stage 2** — headline pretrain. Corrected target: **d28 (2.02 B) at ratio 8**,
  ~38.5 node-h, bs=4. Blocked in practice on GPU 0 being occupied.
- **Stage 3** — SFT on the d12 checkpoint, with `--toolcall-epochs=2`.
- **Stage 5** — GSM8K RL (nanochat ships it; note the author's own header concedes
  it is REINFORCE-like, not GRPO).
- **Stage 6** — agentic RL. Unbounded; the plan's own 200–1000 node-h estimate
  stands and is the dominant cost of the project.
