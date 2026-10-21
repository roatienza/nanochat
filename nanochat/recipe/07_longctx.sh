#!/usr/bin/env bash
# utak — Stage 7: context extension (2k -> 128k), without YaRN.
#
# The long-context model is rebuilt at sequence_len=131072 with the SSSL
# sliding-window pattern and rope_base=100000, then trained in two passes:
#
#   7a. -> $TAG_LONGC        1750 iterations, warmdown 0.5
#   7b. -> $TAG_LONGC_FINAL   250 iterations, warmdown 0.9 (the anneal)
#
# The source is the RL checkpoint at 2,048 context -- the RL chain runs
# BEFORE the context extension, not after. That ordering is the opposite of
# the one implied by the stage numbering, and it is load-bearing: the
# long-context model is initialized from the RL weights.
#
# The val bpb rises from 0.840 to 0.893 across this boundary because the
# validation mixture changed -- it is now evaluated at 128k on long packed
# rows, a strictly harder distribution. The two numbers are not comparable.
#
# Usage:  ./recipe/07_longctx.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 7: long-context extension -> $TAG_LONGC -> $TAG_LONGC_FINAL ==="
STAGE_GPU="${GPU_LONGC}"
# The long-context trainers do not reduce gradients across ranks, so
# they MUST run on a single card. See recipe/config.sh.
require_single_gpu "07_longctx"
log "    source: rl/$TAG_RL @ step 177  ->  131072 ctx"

# ---- 7a: the main long-context pass -------------------------------------
run_mod "$LOG_DIR/stage7a_longctx.log" scripts.base_train_long \
  --run "utak-longctx" \
  --source-phase rl --model-tag "$TAG_RL" --model-step 177 \
  --output-tag "$TAG_LONGC" \
  --max-seq-len 131072 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 2000 \
  --embedding-lr 0.09 --unembedding-lr 0.0024 \
  --matrix-lr 0.006 --scalar-lr 0.15 \
  --weight-decay 0.028 \
  --warmup-steps 40 --warmdown-ratio 0.5 --final-lr-frac 0.05 \
  --eval-every 100 --eval-tokens 262144 --save-every 250 \
  --ce-chunk 4096 \
|| die "stage 7a failed"

# ---- 7b: the anneal ------------------------------------------------------
# Continues from 7a's own checkpoint and lowers the LR toward zero.
run_mod "$LOG_DIR/stage7b_longctx_anneal.log" scripts.base_train_long \
  --run "utak-longctx-final" \
  --source-phase base --model-tag "$TAG_LONGC" --model-step 1750 \
  --output-tag "$TAG_LONGC_FINAL" \
  --max-seq-len 131072 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 250 \
  --embedding-lr 0.09 --unembedding-lr 0.0024 \
  --matrix-lr 0.006 --scalar-lr 0.15 \
  --weight-decay 0.028 \
  --warmup-steps 40 --warmdown-ratio 0.9 --final-lr-frac 0.05 \
  --eval-every 50 --eval-tokens 262144 --save-every 250 \
  --ce-chunk 4096 \
|| die "stage 7b failed"

log "Stage 7 complete. Checkpoint: base_checkpoints/$TAG_LONGC_FINAL @ 250"
