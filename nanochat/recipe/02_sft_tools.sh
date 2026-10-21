#!/usr/bin/env bash
# utak — Stage 2: tool-calling SFT (still at 2k context).
#
# This is the "stage 4" run in the original numbering: the first SFT on top
# of the base model, teaching tool calling and multi-turn behaviour.
#
# The mixture, read from the checkpoint metadata of the original run:
#   toolace x6, toolcall x4, gsm8k x1, mmlu x1, smoltalk x1
#
# It runs with --num-iterations -1 (one epoch over the mixture) and saves
# into chatsft_checkpoints/<model-tag>, which is why the tag it saves under
# is the SAME tag it was given as its input: the original lineage used the
# tag "d28b-v2" for both, i.e. the base checkpoint and this SFT shared a
# name in different directories. This recipe keeps them distinct.
#
# Usage:  ./recipe/02_sft_tools.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 2: tool-calling SFT -> $TAG_SFT_TOOLS ==="
log "    source: base/$TAG_BASE @ 277760"

# chat_sft.py reads base_checkpoints/<model-tag> and writes
# chatsft_checkpoints/<model-tag>. The original lineage gave both the same
# tag (the base and the SFT shared a name in different directories); here
# the names differ, so the base is hard-linked into place first. Hard links
# cost no disk and no copy time.
if [ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_SFT_TOOLS" ]; then
  log "hard-linking base_checkpoints/$TAG_BASE -> $TAG_SFT_TOOLS"
  cp -al "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_BASE" \
          "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_SFT_TOOLS"
fi

run_mod "$LOG_DIR/stage2_sft_tools.log" scripts.chat_sft \
  --run "utak-sft-tools" \
  --model-tag "$TAG_SFT_TOOLS" --model-step 277760 \
  --load-optimizer 1 \
  --toolace-epochs 6 --toolcall-epochs 4 --gsm8k-epochs 1 --mmlu-epochs 1 \
  --smoltalk-epochs 1 \
  --device-batch-size 2 --total-batch-size 32768 \
  --init-lr-frac 0.8 \
  --warmup-ratio 0.0 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every "$SFT_EVAL_EVERY" --eval-tokens "$SFT_EVAL_TOKENS" --save-every 5000 \
  --chatcore-every 200 --chatcore-max-cat -1 --chatcore-max-sample 24 \
|| die "stage 2 failed"

log "Stage 2 complete. Checkpoint: chatsft_checkpoints/$TAG_SFT_TOOLS"
