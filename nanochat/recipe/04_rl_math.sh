#!/usr/bin/env bash
# utak — Stage 4: math RL on GSM8K.
#
# The upstream algorithm is "GRPO in quotes": REINFORCE with four
# deliberate simplifications.
#
#   1. No trust region -- no KL regularization against a reference model.
#   2. On-policy, so there is no PPO probability ratio and no clipping.
#   3. DAPO-style TOKEN-level normalization, not sequence-level.
#   4. Advantage is (r - mu) -- no z-score division by sigma.
#
# Measured behaviour of this stage: test-split Pass@1 0.1350 (step 0) ->
# 0.1775 (step 40, the peak) -> 0.1575 (step 120, where it was stopped).
# The run is stopped at the flat point on purpose: train reward saturates
# near 0.95 (exactly 1.0 at some steps), so saturated groups carry
# near-zero advantage and the gradient dies. The fix is not more steps --
# it is the fresh-reward agentic tasks in stages 5 and 6.
#
# One caveat, measured and recorded: the rollouts are on the GSM8K TRAIN
# split, the same problems the warmup SFT'd for 10 epochs, so the TRAIN
# reward is inflated by memorization. The honest metric is eval Pass@1 on
# the TEST split.
#
# chat_rl.py has no --output-tag: it reads chatsft_checkpoints/<model-tag>
# and writes chatrl_checkpoints/<model-tag>.
#
# Usage:  ./recipe/04_rl_math.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 4: math RL (GSM8K) -> $TAG_RL_WARMUP ==="
log "    source: sft/$TAG_WARMUP @ 509"

if [ ! -d "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_RL_WARMUP" ]; then
  log "staging chatsft/$TAG_WARMUP -> chatsft/$TAG_RL_WARMUP"
  cp -al "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_WARMUP" \
          "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_RL_WARMUP"
fi

run_mod "$LOG_DIR/stage4_rl_math.log" scripts.chat_rl \
  --run "utak-rl-gsm8k" \
  --model-tag "$TAG_RL_WARMUP" --model-step 509 \
  --num-epochs 1 \
  --device-batch-size 2 --examples-per-step 16 --num-samples 16 \
  --max-new-tokens 512 --temperature 1.0 --top-k 50 \
  --embedding-lr 0.2 --unembedding-lr 0.004 --matrix-lr 0.02 \
  --weight-decay 0.0 --init-lr-frac 0.05 \
  --eval-every "$RL_EVAL_EVERY" --eval-examples "$RL_EVAL_EXAMPLES_04" --save-every 100 \
|| die "stage 4 failed"

log "Stage 4 complete. Checkpoint: chatrl_checkpoints/$TAG_RL_WARMUP @ 100"
