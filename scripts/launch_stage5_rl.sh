#!/usr/bin/env bash
# Stage 5 — Reasoning RL on GSM8K (stock nanochat scripts/chat_rl.py, "GRPO" = REINFORCE + DAPO-style token-level norm)
# Base: Stage 4 v3 final checkpoint (model_013630.pt, val_bpb 0.3022, hermes exact 40.7%)
# GPUs: 5 (measured free 2026-10-06 10:4x; 40GB free, 12% util = lowest contention)
#   NOTE: original A' target GPU 2 is now occupied by rowel's live vLLM server
#   (DeepSeek-Coder-V2-Lite, port 8016, 38.4GB) — not killed. GPU 0 is rowel's
#   active LoRA job at 92%. Substituted GPU 5.
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
# Expected: 467 steps, ~6.3 min/step (measured rollout-dominated), ~49 h wall clock.
#   Stop-if-flat horizon: check pass@k/reward after ~120 steps (~13 h).
set -euo pipefail
cd /data/rowel/nanochat
export NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2
export CUDA_VISIBLE_DEVICES=5
# NOTE: '--' is REQUIRED — torchrun's argparse greedily prefix-matches '--run'
# against its own '--run-path' and errors without the separator.
exec .venv/bin/torchrun --standalone --nproc_per_node=1 -m scripts.chat_rl -- \
  --run=stage5-rl-gsm8k-d28b-v3 \
  --model-tag=d28b-v2 \
  --model-step=13630 \
  --device-batch-size=2 \
  --examples-per-step=16 \
  --num-samples=16 \
  --num-epochs=1 \
  --max-new-tokens=512 \
  --eval-every=20 \
  --eval-examples=400 \
  --save-every=100
