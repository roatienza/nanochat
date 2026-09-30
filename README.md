# reasoner-model

Teaching plan: train one model end-to-end from language modeling to
long-horizon agentic task completion, on a single **8 × A100 40 GB** node.

Read **[PLAN.md](PLAN.md)** for the full curriculum, memory budgets, and stage
timings.

## TL;DR

- **Model: d20 (≈2.15 B params, bf16).** d24+ is arithmetically impossible on
  40 GB — optimizer state alone exceeds the card.
- **No FP8.** A100 is Ampere; FP8 is Hopper-only.
- **Harness: [nanochat](https://github.com/karpathy/nanochat)**, which covers
  stages 0–5 end to end in ~6k lines and explicitly supports Ampere.
- **Agentic RL is the expensive part**, not pretraining: 200–1,000+ node-hours
  versus ~124 for the headline pretrain.
- **Stage 4 is the one stage you write.** nanochat's stock SFT mixture excludes
  function-calling data by design, so tool calling needs a real data task.

## Compute

| Stage | Node-hours (8× A100-40GB) |
|---|---:|
| Pretrain d12 (ablations) | ~15 |
| Pretrain d20 (headline) | ~124 |
| SFT + tool calling | ~20 |
| Reasoning RL (GSM8K) | ~20 |
| Agentic RL | 200–1,000+ |

Assumes 30% MFU, calibrated against SmolLM3's reported 14,000 tok/s/GPU on H100.

## Layout

```
PLAN.md                     the curriculum
scripts/budget.py           memory + wall-clock calculator (reproduce the tables)
```

## Reproduce the numbers

```bash
python3 scripts/budget.py
```

Writes the feasibility table and the node-hour estimates from the same
architecture assumptions nanochat uses (aspect-ratio 64, head_dim 128, vocab
32768, Muon on matrices / Adam on embeddings).
