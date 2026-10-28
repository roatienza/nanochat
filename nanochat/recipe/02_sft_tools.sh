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
# THE STEP THIS STAGE REACHES IS NOT A CONSTANT
# ---------------------------------------------
# --num-iterations -1 means "stop when the dataloader has consumed the
# mixture", so the final step follows from the SIZE OF THE CORPUS, not from
# the recipe. On the corpus rendered on 2026-10-22 this stage ends at step
# 13,650; the original run's corpus ended at 13,630.
#
# So this stage WRITES its own handoff record (the step it finished at)
# and stage 3 reads that record rather than a constant. The old constant
# cost two 10.6-hour cycles of both GPUs: stage 3 asked for the
# non-existent model_013630.pt, and the resume logic -- asking the same
# question in the other direction -- concluded that stage 2 was
# incomplete and restarted it. See recipe/config.sh, "THE STAGE HANDOFF".
#
# Usage:  ./recipe/02_sft_tools.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 2: tool-calling SFT -> $TAG_SFT_TOOLS ==="
STAGE_GPU="${GPU_SFT}"

# chat_sft.py reads base_checkpoints/<model-tag> and writes
# chatsft_checkpoints/<model-tag>. The original lineage gave both the same
# tag (the base and the SFT shared a name in different directories); here
# the names differ, so the base is hard-linked into place first. Hard links
# cost no disk and no copy time.
#
# The staging happens BEFORE the step is resolved, because the staged tree
# has to hold the whole checkpoint set -- this stage resumes the optimizer
# state (--load-optimizer 1), so it needs the optimizer shards as well as
# the model file.
if [ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_SFT_TOOLS" ]; then
  log "hard-linking base_checkpoints/$TAG_BASE -> $TAG_SFT_TOOLS"
  cp -al "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_BASE" \
          "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_SFT_TOOLS"
fi

# The pretrain's final step is a FIXED link: it follows from the model
# size (--target-param-data-ratio 8.0), not from the corpus, so it does
# not move between runs.
STEP_BASE="$(stage_read_step "$TAG_BASE" \
  "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_SFT_TOOLS" \
  "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_BASE" \
  "$TAG_BASE_FINAL_STEP" fixed)" \
  || die "cannot establish the pretrain's final step"
log "    source: base/$TAG_BASE @ $STEP_BASE"

run_mod "$LOG_DIR/stage2_sft_tools.log" scripts.chat_sft \
  --run "utak-sft-tools" \
  --model-tag "$TAG_SFT_TOOLS" --model-step "$STEP_BASE" \
  --load-optimizer 1 \
  --toolace-epochs 6 --toolcall-epochs 4 --gsm8k-epochs 1 --mmlu-epochs 1 \
  --smoltalk-epochs 1 \
  --device-batch-size 2 --total-batch-size 32768 \
  --init-lr-frac 0.8 \
  --warmup-ratio 0.0 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every "$SFT_EVAL_EVERY" --eval-tokens "$SFT_EVAL_TOKENS" --save-every 5000 \
  --chatcore-every 200 --chatcore-max-cat -1 --chatcore-max-sample 24 \
|| die "stage 2 failed"

# The handoff record. Written only now, after the trainer exited 0, so its
# presence is proof that this stage FINISHED -- not merely that a
# checkpoint exists.
record_stage_handoff "$TAG_SFT_TOOLS" \
  "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_SFT_TOOLS" \
  || die "could not record the stage 2 handoff"

log "Stage 2 complete. Checkpoint: chatsft_checkpoints/$TAG_SFT_TOOLS @ $(handoff_read "$TAG_SFT_TOOLS")"
