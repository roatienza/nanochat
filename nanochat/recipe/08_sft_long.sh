#!/usr/bin/env bash
# utak — Stage 8: the long-context SFT (the "v5" run).
#
# Produces $TAG_SFT from the long-context base checkpoint.
#
# The original run is "stage7b-v5" (and "stage7b-v5-resume" for its second
# half). It differs from the earlier "stage7b-sft-128k" run in three ways,
# and they matter:
#
#   longctx_epochs       20   (not 2) -- the long-context data is weighted
#                               heavily, because the model is being taught
#                               to actually USE a 128k window rather than
#                               merely tolerate it
#   levels               0,1,2,3 (not 0,1)
#   instances_per_level  2000,1000,500,250 (not 40) -- four orders of
#                               magnitude more long-context data
#   total_batch_size     131072 (not 262144)
#   toolcall/toolace     2 each (not 1 each)
#
# It runs for 800 iterations in two parts: the first 100, then a resume
# from step 100 for the remaining 700. The resume is a plain
# --resume-step 100 continuation of the same output tag.
#
# NOTE ON SCRIPT CHOICE: scripts/chat_sft.py is the upstream 2k-context SFT
# and cannot reach 128k. scripts/chat_sft_long.py is the fork's
# long-context SFT and is the one that matches this stage -- it renders
# with the full tokenizer conversation format and packs at the full
# max_seq_len.
#
# Usage:  ./recipe/08_sft_long.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 8: long-context SFT -> $TAG_SFT ==="
STAGE_GPU="${GPU_LONGC}"
# The long-context trainers do not reduce gradients across ranks, so
# they MUST run on a single card. See recipe/config.sh.
require_single_gpu "08_sft_long"
# 8 reads 7b's output at step 250. That step is FIXED: 7b runs for
# exactly 250 iterations, so its final step is 250 by construction. It
# is still resolved through the same helper, so the file's existence is
# checked before the stage starts rather than twenty minutes into it.
STEP_LONGC_FINAL="$(stage_read_step "$TAG_LONGC_FINAL" \
  "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_LONGC_FINAL" \
  "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_LONGC_FINAL" \
  "$TAG_LONGC_FINAL_STEP" fixed)" \
  || die "cannot establish the step to read from stage 7"
log "    source: base/$TAG_LONGC_FINAL @ $STEP_LONGC_FINAL"

# --- the first 100 iterations --------------------------------------------
# The step budget: 8a stops at step 100, which is the step 8b resumes
# from.
#
# WITHOUT the budget, 8a runs the whole 800-step schedule and then 8b
# runs 101..800 again, so steps 101..800 are trained TWICE -- 1,500 steps
# where the original did 800. At the original's own step time (~23.4s,
# from its meta timestamps) the duplicate 700 steps are ~4.6 hours of a
# GPU.
#
# The original's own checkpoints say what it did: in
# chatsft_checkpoints/d28b-128k-sft-v5, meta_000100.json has no
# resume_step -- that run was INTERRUPTED at 100 -- and meta_000200 ..
# meta_000800 all have resume_step=100. So the original trained 0..100
# once, then 101..800 once. This reproduces that.
#
# --max-steps caps the COMPUTE and leaves the SCHEDULE alone: the
# learning rate still follows the full 800-step warmdown, so the
# trajectory at step 100 is the one the original had. Using
# --num-iterations 100 instead would be a DIFFERENT run: the warmdown is
# computed from --num-iterations, so the learning rate would decay to
# zero over 100 steps instead of being near its peak -- and 8b would
# then resume from a model trained on a decayed schedule.
run_mod "$LOG_DIR/stage8a_sft_long.log" scripts.chat_sft_long \
  --run "utak-sft-long" \
  --source-phase base --model-tag "$TAG_LONGC_FINAL" --model-step "$STEP_LONGC_FINAL" \
  --output-tag "$TAG_SFT" \
  --max-seq-len 131072 --short-window 512 \
  --device-batch-size 1 --total-batch-size 131072 \
  --num-iterations 800 --max-steps 100 \
  --longctx-epochs 20 --levels 0,1,2,3 --instances-per-level 2000,1000,500,250 \
  --toolcall-epochs 2 --toolace-epochs 2 \
  --smoltalk-epochs 1 \
  --embedding-lr 0.072 --unembedding-lr 0.0019 \
  --matrix-lr 0.0048 --scalar-lr 0.12 \
  --weight-decay 0.028 \
  --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 50 --eval-tokens 262144 --save-every 100 \
  --ce-chunk 4096 \
|| die "stage 8a failed"

# --- the remaining 700, resumed from step 100 ---------------------------
# The original run was interrupted and resumed. The resume is a plain
# continuation of the same output tag from its own step-100 checkpoint.
run_mod "$LOG_DIR/stage8b_sft_long_resume.log" scripts.chat_sft_long \
  --run "utak-sft-long-resume" \
  --source-phase base --model-tag "$TAG_LONGC_FINAL" --model-step "$STEP_LONGC_FINAL" \
  --output-tag "$TAG_SFT" --resume-step 100 \
  --max-seq-len 131072 --short-window 512 \
  --device-batch-size 1 --total-batch-size 131072 \
  --num-iterations 800 \
  --longctx-epochs 20 --levels 0,1,2,3 --instances-per-level 2000,1000,500,250 \
  --toolcall-epochs 2 --toolace-epochs 2 \
  --smoltalk-epochs 1 \
  --embedding-lr 0.072 --unembedding-lr 0.0019 \
  --matrix-lr 0.0048 --scalar-lr 0.12 \
  --weight-decay 0.028 \
  --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 50 --eval-tokens 262144 --save-every 100 \
  --ce-chunk 4096 \
|| die "stage 8b failed"

# The handoff record: the step this stage finished at, for stage 9 to
# read. Written only after the trainer exited 0.
record_stage_handoff "$TAG_SFT" \
  "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG_SFT" \
  || die "could not record the stage 8 handoff"

log "Stage 8 complete. Checkpoint: chatsft_checkpoints/$TAG_SFT @ $(handoff_read "$TAG_SFT")"
