#!/usr/bin/env bash
# utak — Stage 3: the GSM8K warmup SFT.
#
# A short, narrow mixture on a FRESH optimizer: GSM8K x10 + ToolCalling x4,
# no SmolTalk, no MMLU, no ToolACE.
#
# Why it exists: the first math-RL attempt, launched straight from the
# tool-calling SFT checkpoint, died at step 7 with reward 0.0 and Pass@1
# 0.5%. The model did the arithmetic in-head and never emitted the <calc>
# format, so REINFORCE had no gradient signal at all. This warmup gives it
# one: GSM8K Pass@1 goes from 0.00% to 25.00% on the warmup gate.
#
# The cost is real and was measured: ARC-Easy -8.7, ARC-Challenge -2.8,
# MMLU -1.9 and HumanEval -16.7 against the previous checkpoint, for a
# ChatCORE change of -0.019 (inside the noise band). The narrow mixture
# trades some breadth for the ability to be rewarded.
#
# THE STEP READ FROM STAGE 2 IS RESOLVED, NOT HARDCODED
# ------------------------------------------------------
# This stage used to read model_013630.pt, a constant taken from the
# original run. The corpus was re-rendered on 2026-10-22, so stage 2 now
# ends at 13,650, and 13,630 does not exist. The stage died on the
# missing file -- twice, each time after the watcher had spent 10.6 hours
# re-running stage 2, because the resume logic asked the same question
# and concluded stage 2 was incomplete.
#
# The step now comes from stage 2's handoff record: the step stage 2
# actually finished at. See recipe/config.sh, "THE STAGE HANDOFF".
#
# Usage:  ./recipe/03_warmup.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 3: GSM8K warmup SFT -> $TAG_WARMUP ==="
STAGE_GPU="${GPU_SFT}"

# chat_sft.py reads base_checkpoints/<model-tag>. The warmup's input is the
# stage-2 SFT output, so stage it into the base tree under the warmup tag.
# The staging happens BEFORE the step is resolved, so that the staged tree
# is a complete snapshot of what stage 2 produced.
if [ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_WARMUP" ]; then
  log "staging chatsft/$TAG_SFT_TOOLS -> base_checkpoints/$TAG_WARMUP"
  cp -al "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_SFT_TOOLS" \
          "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_WARMUP"
fi

# A DYNAMIC link: the step comes from stage 2's handoff record.
STEP_SFT_TOOLS="$(stage_read_step "$TAG_SFT_TOOLS" \
  "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_WARMUP" \
  "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_SFT_TOOLS" \
  "$TAG_SFT_TOOLS_FINAL_STEP" dynamic)" \
  || die "cannot establish the step to read from stage 2"
log "    source: sft/$TAG_SFT_TOOLS @ $STEP_SFT_TOOLS"

run_mod "$LOG_DIR/stage3_warmup.log" scripts.chat_sft \
  --run "utak-warmup-gsm8k" \
  --model-tag "$TAG_WARMUP" --model-step "$STEP_SFT_TOOLS" \
  --load-optimizer 0 \
  --gsm8k-epochs 10 --toolcall-epochs 4 \
  --smoltalk-epochs 0 --mmlu-epochs 0 --toolace-epochs 0 \
  --device-batch-size 2 --total-batch-size 32768 \
  --init-lr-frac 0.8 \
  --warmup-ratio 0.0 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 50 --eval-tokens 2097152 --save-every 100 \
|| die "stage 3 failed"

# The handoff record: the step this stage finished at, for stage 4 to
# read. Written only after the trainer exited 0.
record_stage_handoff "$TAG_WARMUP" \
  "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_WARMUP" \
  || die "could not record the stage 3 handoff"

log "Stage 3 complete. Checkpoint: chatsft_checkpoints/$TAG_WARMUP @ $(handoff_read "$TAG_WARMUP")"
