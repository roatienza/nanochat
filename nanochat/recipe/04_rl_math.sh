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
# WHY --max-steps 100
# -------------------
# The task set has 7,473 GSM8K train problems, so chat_rl.py computes 467
# optimizer steps for it. The original run of this stage was stopped BY
# HAND at step 120, and its last periodic save -- the one the next stage
# reads -- was model_000100.pt. The recipe's docstring above says the stop
# is deliberate: the reward saturates and the gradient dies.
#
# Without a cap the stage would run all 467 steps: about four times the
# intended compute, and all of it past the point where the recipe says the
# gradient is dead. --max-steps 100 stops it at the same place, and
# because the learning rate still ramps down over the full 467-step
# schedule, the trajectory is the same as the original run's.
#
# Usage:  ./recipe/04_rl_math.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 4: math RL (GSM8K) -> $TAG_RL_WARMUP ==="
STAGE_GPU="${GPU_SFT}"

# The step budget: the original run's stop point.
RL_MATH_MAX_STEPS="${RL_MATH_MAX_STEPS:-100}"

if [ ! -d "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_RL_WARMUP" ]; then
  log "staging chatsft/$TAG_WARMUP -> chatsft/$TAG_RL_WARMUP"
  cp -al "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_WARMUP" \
          "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_RL_WARMUP"
fi

# A DYNAMIC link: the step comes from stage 3's handoff record.
STEP_WARMUP="$(stage_read_step "$TAG_WARMUP" \
  "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_RL_WARMUP" \
  "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_WARMUP" \
  "$TAG_WARMUP_FINAL_STEP" dynamic)" \
  || die "cannot establish the step to read from stage 3"
log "    source: sft/$TAG_WARMUP @ $STEP_WARMUP"

run_mod "$LOG_DIR/stage4_rl_math.log" scripts.chat_rl \
  --run "utak-rl-gsm8k" \
  --model-tag "$TAG_RL_WARMUP" --model-step "$STEP_WARMUP" \
  --num-epochs 1 \
  --max-steps "$RL_MATH_MAX_STEPS" \
  --device-batch-size 2 --gen-batch-size 16 \
  --examples-per-step 16 --num-samples 16 \
  --max-new-tokens 512 --temperature 1.0 --top-k 50 \
  --embedding-lr 0.2 --unembedding-lr 0.004 --matrix-lr 0.02 \
  --weight-decay 0.0 --init-lr-frac 0.05 \
  --eval-every "$RL_EVAL_EVERY" --eval-examples "$RL_EVAL_EXAMPLES_04" --save-every 100 \
|| die "stage 4 failed"

# The handoff record: the step this stage finished at, for stage 5 to
# read. Written only after the trainer exited 0.
record_stage_handoff "$TAG_RL_WARMUP" \
  "$NANOCHAT_BASE_DIR/chatrl_checkpoints/$TAG_RL_WARMUP" \
  || die "could not record the stage 4 handoff"

log "Stage 4 complete. Checkpoint: chatrl_checkpoints/$TAG_RL_WARMUP @ $(handoff_read "$TAG_RL_WARMUP")"
