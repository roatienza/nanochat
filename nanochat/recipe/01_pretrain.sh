#!/usr/bin/env bash
# utak — Stage 1: pre-training (base model, 2k context).
#
# Produces the base checkpoint at 2,048 tokens. This is the from-scratch
# pretraining run; nothing here is inherited from a previous stage.
#
# The original run (tag "d28b-v2", run name "headline-d28-v2") trained to
# STEP 277,760 with --num-iterations -1 and --target-param-data-ratio 8.0,
# which is how the iteration count was derived. The iteration count is
# therefore NOT pinned here -- the ratio is what sets the schedule. Pinning
# a step count would silently change the compute budget, because the corpus
# is what it is.
#
# Corpus: karpathy/climbmix-400b-shuffle (NVIDIA Nemotron ClimbMix).
# 171 shards / 15 GB on disk, symlinked to base_data_climbmix/. The LAST
# shard is pinned as validation; all others are train. See recipe/00_data.sh.
#
# Usage:  ./recipe/01_pretrain.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 1: pretrain -> $TAG_BASE ==="
log "    GPUs=$GPU_PRETRAIN  base dir=$NANOCHAT_BASE_DIR"

NGPUS=$(awk -F, '{print NF}' <<< "$GPU_PRETRAIN")

run_pretrain() {
  ( cd "$NANOCHAT_ROOT" \
    && NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" \
       CUDA_VISIBLE_DEVICES="$GPU_PRETRAIN" \
       PYTHONPATH="$NANOCHAT_ROOT:${PYTHONPATH:-}" \
       "$@" \
         --run "utak-pretrain" \
         --depth 28 --aspect-ratio 64 --head-dim 128 \
         --max-seq-len 2048 --window-pattern SSSL \
         --device-batch-size 4 --total-batch-size 32768 \
         --embedding-lr 0.3 --unembedding-lr 0.008 \
         --matrix-lr 0.02 --scalar-lr 0.5 \
         --weight-decay 0.28 \
         --warmup-steps 40 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
         --num-iterations -1 --target-param-data-ratio 8.0 \
         --eval-every 500 --eval-tokens 41943040 --save-every 5000 \
         --model-tag "$TAG_BASE" ) > "$LOG_DIR/stage1_pretrain.log" 2>&1
}

if [ "$NGPUS" -gt 1 ]; then
  # Multi-GPU: the dataloader shards the corpus; DDP averages the gradients.
  run_pretrain "$PY" -m torch.distributed.run --standalone \
               --nproc_per_node="$NGPUS" -m scripts.base_train
else
  run_pretrain "$PY" -m scripts.base_train
fi
rc=$?
[ $rc -eq 0 ] || { tail -40 "$LOG_DIR/stage1_pretrain.log"; die "stage 1 failed (exit $rc)"; }

log "Stage 1 complete. Checkpoint: base_checkpoints/$TAG_BASE"
