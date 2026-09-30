# Execution Log — reasoner-model ladder on 8×A100-40GB

Plan source: `PLAN.md`. Harness: karpathy/nanochat @ 92d63d4.
Started 2026-09-30. Every number below is **measured on this node**, not derived.

## Pre-flight verification

| Claim in PLAN.md | Check | Result |
|---|---|---|
| 8× A100 40GB (Ampere) | `nvidia-smi` | ✅ 8× A100-SXM4-40GB, 40960 MiB, SM 8.0 |
| No FP8 (Ampere) | `get_device_capability` | ✅ (8,0); bf16 native; `--fp8` never used |
| 30% MFU assumption | measured `bf16_mfu` | ⚠️ **real 34–48%** — plan is pessimistic |
| d20 = 2155 M params | `param_audit.py` | ❌ **real 897 M** (plan over-counts 2.40×) |
| d20 fits at bs=32 | `mem_probe.py` | ❌ **OOM at bs=16**; max is bs=8 |
| d24+ infeasible | `mem_probe.py` | ❌ **d24/d26/d28 all fit at bs=4** |

Environment: torch 2.9.1+cu128, 128-core EPYC 7742, 1 TB RAM, 162 GB free disk.

### Finding 1 — two errors in `scripts/budget.py`, in opposite directions

**(a) Parameters over-counted 2.1–2.6×.** The value-embedding term is
`depth * 2 * n_head * head_dim * VOCAB`, but nanochat's `has_ve()` puts value
embeddings on **alternating layers only** (`layer_idx % 2 == (n_layer-1) % 2`), and
`n_kv_head == n_head` so `kv_dim == n_embd`, not `2*n_embd`. That is a clean 4× over-count
of the largest embedding block. Verified by instantiating the real `GPT` class on meta
device and counting parameters — matches the analytic fix to 0.1%:

| config | plan | real | over-count |
|---|---:|---:|---:|
| d12 | 739 M | **286 M** | 2.58× |
| d16 | 1342 M | **537 M** | 2.50× |
| d20 | 2155 M | **897 M** | 2.40× |
| d26 (GPT-2 grade) | 3808 M | **1682 M** | 2.26× |

**(b) Activations under-counted ~4–7×.** The model assumes
`ACT_BYTES_PER_TOKEN_WIDTH = 12`, i.e. ~0.9 GiB for d20 at bs=32. But the dominant
activation is the **logits tensor**, which scales with *vocab*, not width:
`bs × seq × 32768 × 2 B` = **4.0 GiB at bs=32** (8.0 GiB in the fp32 upcast used for
cross-entropy). This is width-independent, which is why the error is worse for small d.

Net effect: the plan's table is wrong in both columns, and wrong in the direction that
*hides the actual constraint*. Corrected frontier, measured
(`nanochat/mem_probe.py`, real fwd+bwd+step, 7 clean A100s):

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

**Consequence for the headline model.** d28 at aspect-64 is **2.02 B params** — within
5% of the plan's stated 2.15 B d20 target — and it *fits*, at bs=4. The plan's
conclusion ("d20, 2.15 B, is the headline") is right in spirit and wrong in every
label. The corrected headline is d28 ≈ 2.0 B, which is the plan's target size.

### Finding 2 — FA3 *does* work on Ampere, contrary to expectation

`USE_FA3=True` via the `kernels-community/flash-attn3` sm80 build. FA3 is normally
Hopper-only, so this was verified before spending GPU-hours:
- full-context causal: runs, finite ✅
- sliding-window (the `SSSL` default, 3 of 4 layers): max abs err **0.0078** vs an fp32
  explicit-mask reference ✅

This matters. Without FA3 the SDPA fallback implements sliding-window attention via an
explicit mask, which `base_train.py` itself warns will make "GPU utilization terrible".
Keeping FA3 is the difference between 48% MFU and a crippled run.

### Finding 3 — world_size is restricted to powers of two dividing vocab

Two hard constraints, both from the harness, not the hardware:
- `total_batch_size % (bs × seq × world_size) == 0`
- `AdamW reduce_scatter requires shape[0] (32768) divisible by world_size`

⇒ world_size ∈ {2, 4, 8}. **7 GPUs is not a legal configuration.** An unrelated job
(9 GiB) holds GPU 0, so this run is on **4 GPUs**, with bs and total batch adjusted
(micro-batch halved, total batch kept at ~131k tokens ≈ the auto-computed 524288
scaled by the 4/8 GPU ratio). Throughput is reported per-node below.

## Stage 0 — Tokenizer ✅

```
python -m scripts.tok_train    # 88.3 s, 32,503 merges, vocab 32768
python -m scripts.tok_eval     # compression ratio = the metric
```

| corpus | bytes | ours tok | ratio | vs GPT-2 | vs GPT-4 |
|---|---:|---:|---:|---:|---:|
| news | 1819 | 405 | 4.49 | −0.2% | −4.7% |
| code | 1259 | 396 | 3.18 | **+31.2%** | −28.2% |
| math | 1834 | 911 | 2.01 | +2.7% | −9.5% |
| science | 1112 | 247 | 4.50 | +5.0% | +0.8% |
| climbmix-train | 2948778 | 622511 | 4.74 | +1.4% | −1.8% |
| climbmix-val | 3024593 | 644939 | 4.69 | +1.2% | −2.2% |

Data: 9 ClimbMix shards (8 train + 1 val) = 788 MB.

**Note:** the `code` win (+31.2% vs GPT-2) is the largest single-domain result and matters
for Stage 4/6, but the aggregate is only ~1.4% better than GPT-2. Do not oversell the
tokenizer; the model is the story.

## Measured throughput (4×A100-40GB, real `base_train`)

| config | bs | tok/s | bf16 MFU | note |
|---|---:|---:|---:|---|
| d12 | 16 | 792,000 | **48.3%** | fastest: small model, logits-bound |
| d20 | 8 | 157,000 | 36.4% | |
| d28 | 4 | 58,200 | 34.3% | ≈2.0 B params |

**The plan's 30% MFU is 1.15–1.6× pessimistic.** The published node-hour table
(15 / 83 / 124 h) is therefore an overestimate by that factor. Corrected estimate for
the d28 headline at ratio 8: **~20 node-hours on 4 GPUs** (≈5 h on 8), not 83–124 h.
This is the single most consequential correction in this log: it converts a
multi-day commitment into an afternoon.
