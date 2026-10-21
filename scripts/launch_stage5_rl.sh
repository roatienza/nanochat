#!/usr/bin/env bash
# Stage 5 — Reasoning RL on GSM8K (stock nanochat scripts/chat_rl.py, "GRPO" = REINFORCE + DAPO-style token-level norm)
# Base: Stage 5a GSM8K warmup checkpoint (d28b-warmup/model_000509.pt, from Stage 4 v3 model_013630.pt).
#   Warmup = GSM8K x10 + ToolCalling x4 SFT (no SmolTalk/MMLU/ToolACE), purpose: nonzero
#   reward signal for RL (v1 launch from raw v3 died at step 7: reward 0.0, Pass@1 0.5%,
#   model did arithmetic in-head, never emitted <calc> format).
#   Gate before launch: chatcore_eval_driver.py on d28b-warmup step 509 must hold near
#   v3's ChatCORE 0.1360 (log: scripts/chatcore_warmup.log).
# GPUs: 5 (re-measured 2026-10-07 17:2x; ~38.2 GB free, foreign procs 1.8 GB).
#   GPU 1 also viable (~36.7 GB free) but carries 2.6-3.3 GB foreign load.
#   NOTE: GPU 2 is rowel's live vLLM server (DeepSeek-Coder-V2-Lite, port 8016, 38.4GB)
#   — not killed. GPU 0 is rowel's active LoRA job. GPU 6 ran the chatcore gate eval.
#
# Hyperparameters (deliberate deviations from stock defaults, each measured):
#   --device-batch-size 2   SFT measured floor ~31-34 GiB at db=2 on 40 GB A100; db=4 OOMs.
#                           RL generation KV adds ~0.4 GiB at 512 tokens — db=2 is the safe max.
#   --max-new-tokens 512    stock is 256; GSM8K CoT chains from a 2B model can exceed 256.
#                           Cost measured: ~1-1.5 s per 512-token gen call (batch 2), not the bottleneck.
#   --eval-every 20         stock 60; run is only ~467 steps, want the pass@k curve resolved.
#   --save-every 100        stock 60; checkpoints are model-only (~4 GiB), 100 is enough granularity.
#   --num-epochs 1          467 steps x 16 examples x 16 samples = 119,552 rollouts.
#
# Expected: 467 steps. MEASURED 2026-10-07 (this run): step-0 eval ~25 min (400 ex x 2
#   samples = 800 gens), steady state ~5.6 min/step (step 0 -> 1), reward 0.89, seq len ~183.
#   Total ETA ~53 h (43.6 h steps + 23 evals x ~25 min). Stock 6.3 min/step estimate held.
#   Stop-if-flat horizon: check pass@k/reward after ~120 steps (~13 h + evals).
#   Watch first eval at step 20: Pass@1 must be clearly above v1's 0.5% — that is the
#   signal the warmup took. If not, stop and rebalance the warmup mixture, not RL steps.
set -euo pipefail
cd /data/rowel/nanochat
export NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2
export CUDA_VISIBLE_DEVICES=5
# NOTE: '--' is REQUIRED — torchrun's argparse greedily prefix-matches '--run'
# against its own '--run-path' and errors without the separator.
exec .venv/bin/torchrun --standalone --nproc_per_node=1 -m scripts.chat_rl -- \
  --run=stage5-rl-gsm8k-warmup \
  --model-tag=d28b-warmup \
  --model-step=509 \
  --device-batch-size=2 \
  --examples-per-step=16 \
  --num-samples=16 \
  --num-epochs=1 \
  --max-new-tokens=512 \
  --eval-every=20 \
  --eval-examples=400 \
  --save-every=100
