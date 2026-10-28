#!/usr/bin/env bash
# utak — Stage 5: agentic RL on sandbox-verified environments.
#
# scripts/chat_rl_agentic.py. The optimization core is BYTE-IDENTICAL to
# chat_rl.py; only the reward source changes. tasks/agentic.py generates
# fresh instances per index, so the rollout set can never be memorized --
# that is the fix for the stage-4 failure mode (train reward saturating,
# advantage going to zero, gradients dying at ~step 40).
#
# The reward is 0/1, verified by the nanochat/execution.py sandbox. The
# reference solutions were validated at 15/15 and 900 wrong-solution trials
# leaked 0.
#
# Env holdout: the tail 5% of each (family, level) block is reserved for
# eval and never trained on.
#
# Measured: eval Pass@1 0.0667 (step 0) -> 0.2667 (step 20). Fresh
# verifiable reward moves the needle about 5x faster than the memorized
# GSM8K mixture did.
#
# Usage:  ./recipe/05_rl_agentic.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 5: agentic RL -> $TAG_RL_PILOT ==="
STAGE_GPU="${GPU_SFT}"

# The pilot's base is the math-RL checkpoint. chat_rl_agentic.py reads
# chatsft_checkpoints/<model-tag>, so it is staged there. STAGING FIRST:
# the step resolution below links the checkpoint set into the read
# directory, and on 2026-10-28 it ran before this block, so it linked
# into a directory that did not exist yet and the stage died 15 seconds
# in on a missing meta file. Stage 04 already had the order right.
if [ ! -d "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_RL_PILOT" ]; then
  log "staging chatrl/$TAG_RL_WARMUP -> chatsft/$TAG_RL_PILOT"
  cp -al "$NANOCHAT_BASE_DIR/chatrl_checkpoints/$TAG_RL_WARMUP" \
          "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_RL_PILOT"
fi

# A DYNAMIC link: the step comes from stage 4's handoff record.
STEP_RL_WARMUP="$(stage_read_step "$TAG_RL_WARMUP" \
  "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_RL_PILOT" \
  "$NANOCHAT_BASE_DIR/chatrl_checkpoints/$TAG_RL_WARMUP" \
  "$TAG_RL_WARMUP_FINAL_STEP" dynamic)" \
  || die "cannot establish the step to read from stage 4"
log "    source: rl/$TAG_RL_WARMUP @ $STEP_RL_WARMUP"

run_mod "$LOG_DIR/stage5_rl_agentic.log" scripts.chat_rl_agentic \
  --run "utak-rl-agentic" \
  --model-tag "$TAG_RL_PILOT" --model-step "$STEP_RL_WARMUP" \
  --num-epochs 1 \
  --device-batch-size 2 --gen-batch-size 16 \
  --examples-per-step 16 --num-samples 16 \
  --max-new-tokens 512 --temperature 1.0 --top-k 50 \
  --embedding-lr 0.2 --unembedding-lr 0.004 --matrix-lr 0.02 \
  --weight-decay 0.0 --init-lr-frac 0.05 \
  --eval-every "$RL_EVAL_EVERY" --eval-examples "$RL_EVAL_EXAMPLES_05" --save-every 50 \
  --agentic-instances-per-level 200 --agentic-levels 1,2,3 \
  --agentic-eval-split-frac 0.05 \
|| die "stage 5 failed"

# The handoff record: the step this stage finished at, for stage 6 to
# read. Written only after the trainer exited 0.
record_stage_handoff "$TAG_RL_PILOT" \
  "$NANOCHAT_BASE_DIR/chatrl_checkpoints/$TAG_RL_PILOT" \
  || die "could not record the stage 5 handoff"

log "Stage 5 complete. Checkpoint: chatrl_checkpoints/$TAG_RL_PILOT @ $(handoff_read "$TAG_RL_PILOT")"
