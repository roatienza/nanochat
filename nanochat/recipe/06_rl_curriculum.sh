#!/usr/bin/env bash
# utak — Stage 6: agentic RL with a weighted block curriculum.
#
# scripts/chat_rl_curriculum.py. The optimization core and the eval are
# untouched, so the numbers are directly comparable to stage 5.
#
# TWO changes, each with the evidence that motivated it:
#
# 1. WEIGHTED BLOCK POOL instead of a sequential walk. The pilot walked
#    blocks in order, so 42/178 steps (24%) landed entirely inside
#    uncracked blocks where all 16 samples of every example scored 0 ->
#    advantage 0 -> zero gradient. Meanwhile data_fix@L1, trained once at
#    steps 36-47 and never revisited, regressed from 1.00 to 0.00 on eval.
#    The fix: sample example indices from a weighted pool every step, so
#    replayed blocks keep their capability and frontier blocks get maximal
#    exposure.
#
# 2. TRAIN-ONLY EXEMPLAR SCAFFOLD for the 5 frontier blocks. The pilot
#    never scored a single success on them (0/256 samples per step), and
#    with a 0/1 reward and advantage = r - mu, an all-zero sample group
#    contributes nothing no matter how often it is drawn. A worked example
#    (instance 0 of the block, with its verified solution) is prepended to
#    the user prompt at TRAIN time only, giving the policy a path to its
#    first successes. It is annealed off at --exemplar-anneal-step, and the
#    eval never sees exemplars, so the eval curve measures whether the
#    capability was internalized rather than parroted.
#
# The probe that chose "exemplar" over "hint": prose hints lifted NOTHING
# on the frontier (0/16 with and without, all 5 blocks), while worked
# examples moved count_chars@L3 0.000 -> 1.000 and list_ops@L2
# 0.000 -> 1.000.
#
# THIS STAGE PRODUCES THE CHECKPOINT THE LONG-CONTEXT STAGE BUILDS ON
# (chatrl_checkpoints/<tag> step 177, Pass@1 0.667).
#
# Usage:  ./recipe/06_rl_curriculum.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 6: curriculum RL -> $TAG_RL @ 177 ==="
STAGE_GPU="${GPU_SFT}"

# The curriculum run continues from the pilot's final checkpoint. It reads
# chatsft_checkpoints/<model-tag> and writes chatrl_checkpoints/<model-tag>,
# so the pilot's output is staged into the chatsft tree under the final tag.
# STAGING FIRST: the step resolution below links the checkpoint set into
# the read directory, and on 2026-10-28 it ran before this block, so it
# linked into a directory that did not exist yet and the stage would have
# died 15 seconds in on a missing meta file. Stage 04 already had the
# order right.
if [ ! -d "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_RL" ]; then
  log "staging chatrl/$TAG_RL_PILOT -> chatsft/$TAG_RL"
  cp -al "$NANOCHAT_BASE_DIR/chatrl_checkpoints/$TAG_RL_PILOT" \
          "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_RL"
fi

# A DYNAMIC link: the step comes from stage 5's handoff record.
STEP_RL_PILOT="$(stage_read_step "$TAG_RL_PILOT" \
  "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_RL" \
  "$NANOCHAT_BASE_DIR/chatrl_checkpoints/$TAG_RL_PILOT" \
  "$TAG_RL_PILOT_FINAL_STEP" dynamic)" \
  || die "cannot establish the step to read from stage 5"
log "    source: rl/$TAG_RL_PILOT @ $STEP_RL_PILOT"

run_mod "$LOG_DIR/stage6_rl_curriculum.log" scripts.chat_rl_curriculum \
  --run "utak-rl-curriculum" \
  --model-tag "$TAG_RL" --model-step "$STEP_RL_PILOT" \
  --num-epochs 1 \
  --device-batch-size 2 --gen-batch-size 16 \
  --examples-per-step 16 --num-samples 16 \
  --max-new-tokens 512 --temperature 1.0 --top-k 50 \
  --embedding-lr 0.2 --unembedding-lr 0.004 --matrix-lr 0.02 \
  --weight-decay 0.0 --init-lr-frac 0.05 \
  --eval-every "$RL_EVAL_EVERY" --eval-examples "$RL_EVAL_EXAMPLES_06" --save-every 25 \
  --agentic-instances-per-level 200 --agentic-levels 1,2,3 \
  --agentic-eval-split-frac 0.05 \
  --frontier-frac 0.5 \
  --exemplar-scaffold 1 --exemplar-anneal-step 89 \
  --curriculum-seed 7 \
|| die "stage 6 failed"

# The handoff record: the step this stage finished at, for stage 7 to
# read. Written only after the trainer exited 0.
record_stage_handoff "$TAG_RL" \
  "$NANOCHAT_BASE_DIR/chatrl_checkpoints/$TAG_RL" \
  || die "could not record the stage 6 handoff"

log "Stage 6 complete. Checkpoint: chatrl_checkpoints/$TAG_RL @ $(handoff_read "$TAG_RL")"
log "This is the long-context source for stage 7."
