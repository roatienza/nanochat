# CORRECTION — the "ChatCORE regression" was a phantom (2026-10-06)

## The missing number, measured

Ran `scripts/chat_eval.py` on the **Stage 3 final** checkpoint
(`d28b-v2/model_015028.pt`, val_bpb 0.308864) on GPU 1, mirroring the
in-training ChatCORE loop exactly (categorical tasks = full test set,
generative tasks = 24 samples, batch 2, temp 0.0, top_k 50, single Engine):

| task | Stage 3 final (measured now) |
|---|---|
| ARC-Easy (2376) | 24.62% |
| ARC-Challenge (1172) | 26.54% |
| MMLU (14,042) | 24.65% |
| GSM8K (24) | 0.00% |
| HumanEval (24) | 12.50% |
| **ChatCORE** | **0.0272** |
| ChatCORE_cat | 0.0036 |

## Why the number was missing

The Stage 3 final run (`tpkzb8yr`) was launched with `--chatcore-every=-1`,
so ChatCORE was **never measured in-training**. Its wandb summary shows
`chatcore_metric: N/A`. The number above is the first real measurement of
that checkpoint.

## The "0.082" I had been citing was the wrong run

Every status file from Oct 5–6 repeated "Stage 3's final was 0.082" and
claimed it was "verified from the Stage 3 log." It is **not** in any Stage 3
log (all three logs have zero `ChatCORE:` lines). The 0.0821 comes from wandb
run **`9w53jmjw`** (Sept 30), whose config is:

```
--run=sft-v1 --model-tag=d12 --toolcall-epochs=2 --device-batch-size=16
--total-batch-size=131072 --chatcore-every=1000000 --eval-every=100
```

That is a **d12 (0.286 B) ablation** SFT run, batch 131,072, at step 414 —
a different model, different batch, different scale. It was never the Stage 3
final (d28, 2.02 B). Comparing Stage 4 v2/v3 (d28) against it was
apples-to-oranges.

## The corrected comparison — no regression

| run (final) | ChatCORE | note |
|---|---|---|
| Stage 3 final, `model_015028.pt` | **0.0272** | measured now |
| Stage 4 v2, `model_015548.pt` | 0.1367 | |
| Stage 4 v3, `model_013630.pt` | 0.1360 | |

Stage 4 v2/v3 are **~0.11 ABOVE** the Stage 3 final on ChatCORE, not below.
There is no regression. The reweight (MMLU ×3→×1, GSM8K ×4→×1) did not trade
general chat for tool ability. v3 is statistically identical to v2 on
ChatCORE (0.1360 vs 0.1367) and better on val_bpb (0.3022 vs 0.3148).

## What this means for Stage 5 (RL)

The one open question — "does Stage 5 start from this checkpoint or a blend?"
— is answered: **start from `model_013630.pt` (v3).** It is the best on
val_bpb, statistically tied on ChatCORE, and has 40.7% hermes exact-match
vs v2's 0%. No blend needed.

## Files corrected

The per-run `STATUS_*.md` notes have since been consolidated into
`EXECUTION_LOG.md` and removed from the working tree. The five listed below
remain readable in git history at commit `4279619`
(`git show 4279619:STATUS_<name>.md`). The two later notes — the Stage 5
warmup gate (2026-10-07) and the Stage 6 pilot final (2026-10-10) — were
never committed, but their full content was folded into `EXECUTION_LOG.md`
before removal.

- `STATUS_2026-10-06_STAGE4_V3_FINAL.md` — ChatCORE line + residual #2
- `STATUS_2026-10-05_STAGE4_V2_FINAL.md` — "below Stage 3's 0.082" line
- `STATUS_2026-10-05_STAGE4_V2_PROBE2.md` — "0.082 verified from log" claim
- `STATUS_2026-10-05_STAGE4_V2_67PCT.md`, `..._91PCT.md` — 0.082 references
- `EXECUTION_LOG.md` — "regressed 0.082 -> 0.136" line
