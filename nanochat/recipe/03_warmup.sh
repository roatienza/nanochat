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
# Usage:  ./recipe/03_warmup.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 3: GSM8K warmup SFT -> $TAG_WARMUP ==="
STAGE_GPU="${GPU_SFT}"
log "    source: sft/$TAG_SFT_TOOLS @ 13630"

# chat_sft.py reads base_checkpoints/<model-tag>. The warmup's input is the
# stage-2 SFT output, so stage it into the base tree under the warmup tag.
if [ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_WARMUP" ]; then
  log "staging chatsft/$TAG_SFT_TOOLS -> base_checkpoints/$TAG_WARMUP"
  cp -al "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_SFT_TOOLS" \
          "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_WARMUP"
fi

run_mod "$LOG_DIR/stage3_warmup.log" scripts.chat_sft \
  --run "utak-warmup-gsm8k" \
  --model-tag "$TAG_WARMUP" --model-step 13630 \
  --load-optimizer 0 \
  --gsm8k-epochs 10 --toolcall-epochs 4 \
  --smoltalk-epochs 0 --mmlu-epochs 0 --toolace-epochs 0 \
  --device-batch-size 2 --total-batch-size 32768 \
  --init-lr-frac 0.8 \
  --warmup-ratio 0.0 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 50 --eval-tokens 2097152 --save-every 100 \
|| die "stage 3 failed"

log "Stage 3 complete. Checkpoint: chatsft_checkpoints/$TAG_WARMUP @ 509"
