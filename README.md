# reasoner-model

Teaching plan: train one model end-to-end from language modeling to
long-horizon agentic task completion, on a single **8 × A100 40 GB** node.

## Documentation

The two main documents are the ones in the repository root:

- **[README.md](README.md)** — this file: the plan in brief, the headline
  numbers, and how to reproduce them.
- **[PLAN.md](PLAN.md)** — the full curriculum, memory budgets, and stage
  timings.

Supporting evidence lives in `docs/`:

- **[docs/EXECUTION_LOG.md](docs/EXECUTION_LOG.md)** — the dated, append-only
  record of every stage: what was measured, what was decided, and why.
  Covers 2026-10-03 onward.
- **[docs/EXECUTION_LOG_PART1.md](docs/EXECUTION_LOG_PART1.md)** — the recovered
  first half (2026-09-30 → 2026-10-01): the pre-flight verification, the
  feasibility audit, and Stages 0–4. See "A lost half of the log" below.
- **[docs/CORRECTION_2026-10-06_CHATCORE.md](docs/CORRECTION_2026-10-06_CHATCORE.md)**
  — the retraction of the "ChatCORE regression" claim (it was a phantom, from
  comparing against a d12 ablation).

The **implementation** of Stages 4–6 lives in `nanochat/` — see
[Implementation](#implementation) below and
[nanochat/docs/REPRODUCE.md](nanochat/docs/REPRODUCE.md).

## TL;DR

- **Model: d28 (≈2.02 B params, bf16) — the headline.** Not d20: the plan's
  "d20 = 2.15 B" was a parameter-count bug (2.40× over-counted), and the real
  2.0 B model is d28, which fits at device-batch 4. The plan reached the right
  *size* by the wrong route.
- **No FP8.** A100 is Ampere; FP8 is Hopper-only.
- **Harness: (https://github.com/karpathy/nanochat)**, which covers
  stages 0–5 end to end in ~6k lines and explicitly supports Ampere.
- **Agentic RL is the expensive part**, not pretraining: 200–1,000+ node-hours
  versus ~124 for the headline pretrain.
- **Stage 4 is the one stage you write.** nanochat's stock SFT mixture excludes
  function-calling data by design, so tool calling needs a real data task.

### The plan's table is a derived model; the measured one is the authority

`PLAN.md` §1 carries the September feasibility table. The audit in
[docs/EXECUTION_LOG_PART1.md](docs/EXECUTION_LOG_PART1.md) (Finding 1) measured
it on the node and found it wrong in both columns. `scripts/budget.py` holds the
**measured** numbers; `scripts/qa_plan.py` re-derives them and prints the
divergence.

| # | PLAN.md claim | Measured reality |
|---|---|---|
| 1 | d20 = 2,155 M params | **d20 = 897 M**; over-counted 2.40× |
| 2 | d24+ does not fit | **d24/d26/d28 all fit at bs=4** |
| 3 | d20 runs at bs=32 | **max bs=8** (OOM at 16) |
| 4 | headline ≈ 2.15 B "d20" | **d28 = 2,019 M** — same size, different label |
| 5 | 30% MFU | **34–48%** (node-hours 1.13–1.60× pessimistic) |
| 6 | no FA3 on Ampere (implied) | **FA3 works and is numerically correct** |
| 7 | xLAM-60k is the Stage 4 source | **gated**; hermes-function-calling-v1 used instead |

`PLAN.md` is kept as the teaching document. The deviations are reported by the
QA script rather than silently patched, so the plan's reasoning stays visible
next to the measurement that corrected it.

## Compute

| Stage | Node-hours (8× A100-40GB) |
|---|---:|
| Pretrain d12 (ablations) | ~15 |
| Pretrain d20 (headline) | ~124 |
| SFT + tool calling | ~20 |
| Reasoning RL (GSM8K) | ~20 |
| Agentic RL | 200–1,000+ |

Assumes 30% MFU, calibrated against SmolLM3's reported 14,000 tok/s/GPU on H100.
(The measured MFU was 34–48%, so these are pessimistic; the measured table is
in `scripts/budget.py` and `nanochat/budget_real.py`.)

## A lost half of the log

`docs/EXECUTION_LOG.md` starts on **2026-10-03**. Everything before that — the
pre-flight verification, the feasibility audit (Finding 1), the FA3 and
throughput findings, and Stages 0, 1, 3, and 4 — was **truncated out of the
working tree** in commit `ddb6176` (878 lines → 56).

It survives only in git history, at `65a102a` and `5e860e5` (identical). It has
been restored to
**[docs/EXECUTION_LOG_PART1.md](docs/EXECUTION_LOG_PART1.md)**, rebuilt
verbatim from that commit (0 source lines dropped) with a provenance header.

This was **not** caused by the 2026-10-21 `STATUS_*.md` consolidation — that
preserved what was already in the tree. The September entries were gone before
it, and the consolidation is what made the gap hard to see, because the log
looked complete.

## Implementation

`nanochat/` is the training + serving code. It is the **working tree of the
fork**, not a git archive — the changes that matter (the `no_train` tokenizer
flag, the harness-shaped data tasks, the literal-`\n` fix) were never
committed upstream, and an archive of `HEAD` would not reproduce the served
model.

It contains the PLAN's own Stages 4–6, implemented:

| PLAN stage | Where it lives | Status |
|---|---|---|
| Stage 4 — tool calling | `nanochat/tasks/toolcalling.py`, `toolace.py`, `glaive_multiround.py`, `toolfollowup.py`, `toolstop.py` | done |
| Stage 5 — reasoning RL | `nanochat/scripts/chat_rl.py` (stock) + `nanochat/tasks/gsm8k.py` | done, **starved** (see log) |
| Stage 6 — agentic RL | `nanochat/tasks/agentic.py`, `nanochat/scripts/chat_rl_agentic.py` | done, 0.667 offline |

Plus the serving stack and the gates that verify the deployed model:

```
nanochat/
  nanochat/            model, tokenizer, dataloader, optim, engine
  scripts/             training entry points (one per stage)
  tasks/               the SFT/RL data tasks
  run_v6*.sh           the exact touch-up training commands
  chat_template*.jinja the serving chat templates (v7 is the served one)
  serving/             vLLM stack + the guard proxy
  eval/                the gates and end-to-end checks
  docs/                REPRODUCE.md, MODEL_CARD.md, TECHNICAL_REPORT.md,
                       EXECUTION_LOG.md, checkpoint_meta/
```

### The served checkpoint

```
model   nanochat-d28b-rl100        root nanochat-d28b-rl100-128k-v6j
params  2,018,511,234              28L · 14H · n_embd 1792 · vocab 32768
ctx     131072                     window SSSL/512 · rope 1e5 · FA3
gates   v5x 15/15 · HP 6/7 (HP5 fails; cosmetic, pre-existing)
```

The weights are not in git (8 GB). **[nanochat/docs/REPRODUCE.md](nanochat/docs/REPRODUCE.md)**
is the guide to regenerating them.

### Two execution logs, not one

Both trees are named `EXECUTION_LOG.md` and they are **different documents**:

- `docs/EXECUTION_LOG.md` — the **stage ladder** for this plan (Stage 2
  pretrain → Stage 6 agentic RL), the decisions and retractions.
- `nanochat/docs/EXECUTION_LOG.md` — the **checkpoint lineage** for the
  `d28b` model, v5a → v6j, the touch-up loop that produced the served export.

## Layout

```
README.md                   the plan in brief  (main doc)
PLAN.md                     the curriculum     (main doc)
docs/EXECUTION_LOG.md       dated evidence, 2026-10-03 onward
docs/EXECUTION_LOG_PART1.md the recovered first half, 2026-09-30 to 10-01
docs/CORRECTION_*.md       retractions and corrections
scripts/budget.py           memory + wall-clock calculator (the MEASURED numbers)
scripts/launch_stage*.sh    stage launchers (5a warmup, 5 RL, 6 pilot, 6b curriculum)
scripts/stage6_*.py         Stage 6 dry-run, sandbox smoke, offline eval
scripts/probe_l3.py         level-3 capability probe
scripts/qa_plan.py         checks PLAN.md's claims against the code
logs/                       small evidence logs (<=10 KB) from the stage runs
nanochat/                   the training + serving implementation
```

## Reproduce the numbers

```bash
python3 scripts/budget.py
```

Writes the feasibility table and the node-hour estimates from the same
architecture assumptions nanochat uses (aspect-ratio 64, head_dim 128, vocab
32768, Muon on matrices / Adam on embeddings).

The ground truth behind that table, both runnable:

```bash
cd nanochat
.venv/bin/python param_audit.py   # real GPT class on meta device vs budget.py
.venv/bin/python mem_probe.py     # real fwd+bwd+step peak memory
```

`param_audit.py` agrees with `budget.py` to a ratio of **1.00** at every depth
(it is the same model, counted two ways), which is what makes the measured
table trustworthy.

## Verify the plan against the code

```bash
python3 scripts/qa_plan.py
```

Checks every factual claim in `PLAN.md` against the nanochat source — the
`--fp8` H100 gate, the measured feasibility frontier, the SmolTalk-mixture
assertion behind Stage 4, and the sandbox guarantees behind Stage 6 — and
prints the known plan-vs-measurement deviations. **36 passed, 0 failed.**
