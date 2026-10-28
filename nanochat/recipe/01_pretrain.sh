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
# EVALUATION BUDGET
#   The original run evaluated 41,943,040 tokens every 500 steps. That cost
#   147 s per eval x 556 evals = 25.4 h -- MORE than the 34.3 h of actual
#   training, and 40% of the 57.35 h wall clock. The evaluation was also
#   more precise than it needed to be: the standard deviation of a single
#   estimate at that size was 0.000065 bpb, while the real improvement over
#   500 steps at the end of training was 0.00021 bpb.
#
#   PRETRAIN_EVAL_TOKENS (4,194,304) cuts the cost ~10x while keeping the
#   noise (0.000204) at the same order as the signal. Set
#   PRETRAIN_EVAL_TOKENS=41943040 PRETRAIN_EVAL_EVERY=500 to reproduce the
#   original run exactly. See recipe/config.sh.
#
#   --core-metric-every must be set explicitly: base_train.py's DEFAULT is
#   2000, which would fire 138 times over this run (~4.3 h). The original
#   run passed 999999. The CORE metric is run once at the end by stage 12.
#
# RESUME
#   This is the longest stage in the recipe (~73 h on 2 GPUs, ~145 h on
#   1). It saves every PRETRAIN_SAVE_EVERY steps, and it can pick up
#   from the last save automatically: if the checkpoint directory already
#   holds a model_*.pt, the newest one is used as the resume point.
#
#   That matters because a multi-day run on a shared node WILL be
#   interrupted -- a card gets taken, the node reboots, the queue is
#   preempted. Without resume, every such event costs the whole stage.
#
#   Set PRETRAIN_RESUME=0 to force a clean run from scratch.
#
# Usage:  ./recipe/01_pretrain.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 1: pretrain -> $TAG_BASE ==="
log "    GPUs=$GPU_PRETRAIN  base dir=$NANOCHAT_BASE_DIR"

NGPUS=$(awk -F, '{print NF}' <<< "$GPU_PRETRAIN")

# ---------------------------------------------------------------------------
# The device batch size: how many sequences ONE RANK holds at once.
#
# The TOTAL batch is 32,768 tokens and it must stay that way: base_train.py
# scales the learning rate by sqrt(B / B_ref) on the TOTAL batch, so
# changing the total batch would change the recipe. The device batch only
# changes HOW the work is split between the ranks and the micro-steps.
#
# MEASURED on this node, real scripts.base_train, d28, 2k ctx, tbs 32,768
# (2026-10-22, fresh process per measurement, A100-SXM4-40GB):
#
#   world size 1, device batch 2    OK   1,868 ms  17,525 tok/s  41.4% MFU
#                                       peak 32,050 MiB
#   world size 1, device batch 3    OOM
#   world size 1, device batch 4    OOM  39.30 GiB in use, 201 MiB free
#   world size 2, device batch 2    OK     964 ms  33,941 tok/s  40.1% MFU
#                                       peak 37,378 MiB PER RANK
#   world size 2, device batch 4    OOM  39.49 GiB in use, 21 MiB free
#
# Two things fall out of that table.
#
# 1. DEVICE BATCH 4 DOES NOT FIT ON ONE 40 GiB CARD. The old version of
#    this comment claimed "world size 1, device batch 4 OK 15,413 tok/s";
#    that is wrong, and the default was set to 4 anyway. The run died with
#    "CUDA out of memory. Tried to allocate 686.00 MiB ... 201.88 MiB is
#    free" on 2026-10-21. The default is now 2, which is the only value
#    that fits on one card -- and it is ALSO 13.7% FASTER than the 4
#    (17,525 vs 15,413 tok/s), so the old default was both broken and
#    slower.
#
#    Why the memory works out this way: the cost is the OPTIMIZER STATE,
#    not the activations. 2.02 B parameters x (fp32 exp_avg + fp32
#    exp_avg_sq) = 15.0 GiB of moments, plus bf16 weights 3.76 GiB and
#    bf16 gradients 3.76 GiB = 22.5 GiB of state that at world size 1 is
#    entirely unsharded. The activations are the small part, which is why
#    a smaller device batch does not buy back enough to fit.
#
# 2. TWO GPUs DO FIT, at device batch 2, and they are worth 1.94x. This
#    overturns the earlier "2 GPUs is impossible" conclusion (commit
#    0d4aafd), which compared the UNSHARDED 35.34 GiB against a two-rank
#    run. The optimizer state is ZeRO-2 SHARDED, so at world size 2 each
#    rank holds 15.0/2 = 7.5 GiB of moments instead of 15.0 GiB.
#
# The TOTAL batch stays 32,768 tokens: base_train.py scales the learning
# rate by sqrt(B / B_ref) on the TOTAL batch, so changing the total batch
# would change the recipe. The device batch only changes how the work is
# split between the ranks and the micro-steps.
#
# On 4 GPUs the original run used device batch 4 and worked, because the
# optimizer state is sharded 4 ways (3.75 GiB of moments per rank). Keep 4
# there so the multi-GPU profile reproduces the original recipe exactly.
# -----------------------------------------------------------------------
if [ "${PRETRAIN_DEVICE_BATCH:-}" = "" ]; then
  if [ "$NGPUS" -ge 4 ]; then
    PRETRAIN_DEVICE_BATCH=4
  else
    PRETRAIN_DEVICE_BATCH=2
  fi
fi
log "    device batch=$PRETRAIN_DEVICE_BATCH (world size $NGPUS, total batch 32768)"
check_gpu_memory "1 (pretrain)" "$GPU_PRETRAIN" "$PRETRAIN_MEM_NEEDED_MIB"

# ---------------------------------------------------------------------------
# Where would we resume from?
#
# base_train.py takes --resume-from-step N and loads
# base_checkpoints/$TAG_BASE/model_<N>.pt plus its optimizer shards and its
# dataloader state, so the data stream continues where it left off. It does
# NOT know how to pick the newest checkpoint by itself, so we do it here.
# ---------------------------------------------------------------------------
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_BASE"
RESUME_STEP=-1
if [ "${PRETRAIN_RESUME:-1}" = "1" ] && [ -d "$CKPT_DIR" ]; then
  # The newest model_<step>.pt, by the step number in the name.
  newest=$(ls -1 "$CKPT_DIR"/model_*.pt 2>/dev/null \
            | sed -n 's/.*model_\([0-9]\{6\}\)\.pt/\1/p' | sort -n | tail -1)
  if [ -n "$newest" ]; then
    RESUME_STEP=$((10#$newest))
    log "    resuming from step $RESUME_STEP ($CKPT_DIR)"
  fi
fi

run_pretrain() {
  ( cd "$NANOCHAT_ROOT" \
    && NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" \
       CUDA_VISIBLE_DEVICES="$GPU_PRETRAIN" \
       PYTHONPATH="$NANOCHAT_ROOT:${PYTHONPATH:-}" \
       "$@" \
         --run "utak-pretrain" \
         --depth 28 --aspect-ratio 64 --head-dim 128 \
         --max-seq-len 2048 --window-pattern SSSL \
         --device-batch-size "$PRETRAIN_DEVICE_BATCH" --total-batch-size 32768 \
         --embedding-lr 0.3 --unembedding-lr 0.008 \
         --matrix-lr 0.02 --scalar-lr 0.5 \
         --weight-decay 0.28 \
         --warmup-steps 40 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
         --num-iterations -1 --target-param-data-ratio 8.0 \
         --eval-every "$PRETRAIN_EVAL_EVERY" --eval-tokens "$PRETRAIN_EVAL_TOKENS" \
         --save-every 5000 \
         --core-metric-every "$CORE_METRIC_EVERY" \
         --resume-from-step "$RESUME_STEP" \
         --model-tag "$TAG_BASE" ) > "$LOG_DIR/stage1_pretrain.log" 2>&1
}

if [ "$NGPUS" -gt 1 ]; then
  # Multi-GPU: the dataloader shards the corpus; DDP averages the gradients.
  #
  # The "--" is REQUIRED. Without it, torchrun tries to parse the trainer's
  # own flags as its own, and "--run" is ambiguous against torchrun's
  # "--run-path"/"--run_path": the run dies with
  #   run.py: error: ambiguous option: --run could match --run-path, --run_path
  # This was a latent bug in this script -- the multi-GPU branch had never
  # actually executed. The 4-GPU run that produced the served model was
  # launched by hand, with the separator, from the shell.
  #
  # GPU HEALTH GATE
  # ---------------
  # If the NVIDIA driver is in a fault state, torch.cuda.is_available() is
  # False and the trainer SILENTLY FALLS BACK TO THE CPU. On 2026-10-24
  # that turned "the driver died" into "the run is on the CPU, making no
  # progress, for 15.6 hours". Check the driver before starting.
  if ! gpus_healthy; then
    log "FATAL: the NVIDIA driver is not healthy -- refusing to start stage 1."
    log "       nvidia-smi -L reports 'Unknown Error' or 'No devices were found'."
    log "       This is a HOST-level fault and needs a driver reload or node reset."
    die "no usable GPU"
  fi
  run_pretrain "$PY" -m torch.distributed.run --standalone \
               --nproc_per_node="$NGPUS" -m scripts.base_train --
else
  # Same gate for the single-GPU path.
  if ! gpus_healthy; then
    log "FATAL: the NVIDIA driver is not healthy -- refusing to start stage 1."
    log "       nvidia-smi -L reports 'Unknown Error' or 'No devices were found'."
    log "       This is a HOST-level fault and needs a driver reload or node reset."
    die "no usable GPU"
  fi
  run_pretrain "$PY" -m scripts.base_train
fi
rc=$?
[ $rc -eq 0 ] || { tail -40 "$LOG_DIR/stage1_pretrain.log"; die "stage 1 failed (exit $rc)"; }

# The handoff record: the step the pretrain finished at, for stage 2 to
# read. Written only after the trainer exited 0.
record_stage_handoff "$TAG_BASE" \
  "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG_BASE" \
  || die "could not record the stage 1 handoff"

log "Stage 1 complete. Checkpoint: base_checkpoints/$TAG_BASE @ $(handoff_read "$TAG_BASE")"
