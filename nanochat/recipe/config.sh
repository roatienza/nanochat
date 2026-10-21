#!/usr/bin/env bash
# utak — the single source of truth for the training recipe.
#
# Every stage script sources this file. Nothing else hardcodes a path, a
# model name, or a hyperparameter. Change it here and the whole pipeline
# follows.
#
# Model name: "utak". The served OpenAI-compatible model id is "utak";
# the vLLM model directory is "utak-128k"; the training checkpoint tags
# are "utak-<stage>".
set -u

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-utak}"                       # served model id
MODEL_DIR_NAME="${MODEL_DIR_NAME:-utak-128k}"         # vLLM model dir
REPO_TAG="${REPO_TAG:-v6j}"                           # the touch-up revision

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NANOCHAT_ROOT="${NANOCHAT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/data/rowel/nanochat_store/v2}"
VLLM_DIR="${VLLM_DIR:-$NANOCHAT_BASE_DIR/vllm_d28b}"
VLLM_BIN="${VLLM_BIN:-$VLLM_DIR/venv-vllm/bin/vllm}"
VLLM_PY="${VLLM_PY:-$VLLM_DIR/venv-vllm/bin/python}"
PY="${PY:-$NANOCHAT_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="${PYTHON:-python3}"

# Where the packed model lands and where the guard/vLLM read it from.
MODEL_DIR="${MODEL_DIR:-$VLLM_DIR/$MODEL_DIR_NAME}"

# ---------------------------------------------------------------------------
# GPUs. Every stage that needs a GPU pins one; the stages that do not (pack,
# serve) inherit it.
# ---------------------------------------------------------------------------
GPU_PRETRAIN="${GPU_PRETRAIN:-4,5,6,7}"
GPU_SFT="${GPU_SFT:-4}"
GPU_LONGC="${GPU_LONGC:-5}"
GPU_TOUCHUP="${GPU_TOUCHUP:-7}"
GPU_SERVE="${GPU_SERVE:-0}"

# ---------------------------------------------------------------------------
# Ports. The guard owns the public port; vLLM is internal.
# ---------------------------------------------------------------------------
PUBLIC_PORT="${PUBLIC_PORT:-8000}"
UPSTREAM_PORT="${UPSTREAM_PORT:-8010}"

# ---------------------------------------------------------------------------
# Checkpoint tags. These are the names the stages read and write.
# ---------------------------------------------------------------------------
TAG_BASE="${TAG_BASE:-utak-base}"                      # stage 1 output
TAG_SFT_TOOLS="${TAG_SFT_TOOLS:-utak-sft-tools}"        # stage 2 output
TAG_WARMUP="${TAG_WARMUP:-utak-warmup}"                # stage 3 output
TAG_RL_WARMUP="${TAG_RL_WARMUP:-utak-rl-warmup}"       # stage 4 output
TAG_RL_PILOT="${TAG_RL_PILOT:-utak-rl-pilot}"          # stage 5 output
TAG_RL="${TAG_RL:-utak-rl}"                            # stage 6 output (rl dir)
TAG_LONGC="${TAG_LONGC:-utak-128k}"                    # stage 7 output
TAG_LONGC_FINAL="${TAG_LONGC_FINAL:-utak-128k-final}"  # stage 7b output
TAG_SFT="${TAG_SFT:-utak-sft}"                         # stage 8 output
TAG_TOUCH_A="${TAG_TOUCH_A:-utak-sft-a}"                # the v6a-equivalent anchor
TAG_TOUCH_B="${TAG_TOUCH_B:-utak-sft-b}"                # the v6a-equivalent: harness basics
TAG_TOUCH_C="${TAG_TOUCH_C:-utak-sft-c}"                # the SERVED checkpoint (v6j)

# ---------------------------------------------------------------------------
# Final checkpoint that gets exported and served.
# ---------------------------------------------------------------------------
FINAL_TAG="${FINAL_TAG:-$TAG_TOUCH_C}"
FINAL_STEP="${FINAL_STEP:-240}"

# ---------------------------------------------------------------------------
# Performance.
#
# These are the knobs that decide how fast the pipeline runs. They are here,
# not in the stage scripts, because the cost of a wrong value is silent: the
# run is just slow, and the model is the same.
#
# THE MEASURED BASELINE (the d28 pretrain, 4x A100-40GB, the run that
# produced the served model -- serve_logs/stage2/restart170.log):
#
#   step                445 ms          73,655 tok/s     43.5% BF16 MFU
#   peak memory         34,071 / 40,960 MiB  (83% of the card)
#   wall clock          57.35 h   but only 34.33 h of it was training
#   => 23.0 h (40.1%) was OVERHEAD, and it was almost all evaluation
#
# The GPU was never the bottleneck. 43.5% MFU is respectable for a 2B model
# on A100. The waste was in the evaluation schedule, the dataloader, and
# five trainers that never called torch.compile.
#
# ---------------------------------------------------------------------------
# How much validation to run. THIS IS THE BIGGEST SINGLE LEVER.
#
# Each eval is a full forward-only pass over --eval-tokens tokens. The
# measured cost of the original setting (41,943,040 tokens every 500 steps)
# was 147 s per eval, 556 evals, 25.4 h -- more than the training itself.
#
# How much precision do we actually need? From the same log, the standard
# deviation of a single bpb estimate at 41.9M tokens is 0.000065, while the
# real improvement in the tail of training is 0.00021 bpb per 500 steps.
# The original eval was therefore ~3x more precise than the signal it was
# trying to resolve. These defaults keep the noise comfortably BELOW the
# signal (so the curve still shows the true shape) while cutting the cost
# by ~10x:
#
#   eval tokens   sd of estimate   vs signal per 500 steps
#   41,943,040       0.000065          0.00021      (original: 3.2x signal)
#    4,194,304       0.000204          0.00021      (1.0x signal -- default)
#    2,097,152       0.000289          0.00021      (1.4x signal)
#
# Override EVAL_TOKENS=41943040 EVAL_EVERY=500 to reproduce the original run
# exactly. The default below is the fast setting.
PRETRAIN_EVAL_TOKENS="${PRETRAIN_EVAL_TOKENS:-4194304}"
PRETRAIN_EVAL_EVERY="${PRETRAIN_EVAL_EVERY:-1000}"
SFT_EVAL_TOKENS="${SFT_EVAL_TOKENS:-2097152}"
SFT_EVAL_EVERY="${SFT_EVAL_EVERY:-500}"

# The RL stages evaluate by GENERATING, not by a forward pass: 400 problems
# x device_batch_size samples, each up to 512 new tokens. The measured cost
# of one eval is ~25 minutes, and a training step is ~5.6 min (GSM8K) to
# ~11 min (agentic). So the eval is 4-5x the cost of a step, and it is the
# single biggest item in the RL budget.
#
# The eval size is NOT reduced: the measured confidence interval on pass@1
# is +/- 3.7 pt at n=400, and the effect the RL stage is chasing is of that
# same order (the pilot moved 6.7 pts, the curriculum 53 pts). Halving n
# would widen the CI to ~5 pt and hide exactly the signal being looked for.
#
# What is reduced is HOW OFTEN it runs, and that is set per stage below.
RL_EVAL_EVERY="${RL_EVAL_EVERY:-20}"
# Per-stage example counts. The GSM8K stage uses the full 400 (its
# confidence interval is +/- 3.7 pt); the two agentic stages use 150
# because each example runs a sandbox rollout and costs far more.
RL_EVAL_EXAMPLES_04="${RL_EVAL_EXAMPLES_04:-400}"
RL_EVAL_EXAMPLES_05="${RL_EVAL_EXAMPLES_05:-150}"
RL_EVAL_EXAMPLES_06="${RL_EVAL_EXAMPLES_06:-150}"

# The CORE metric (the few-shot benchmark) costs ~111 s per pass and is
# the only thing that produced a headline number in the original run, so it
# stays off during training and runs ONCE at the end.
#
# NOTE: base_train.py's condition is
#     core_metric_every > 0 and (last_step or step % every == 0)
# so the last step ALWAYS evaluates it, whatever this is set to. Setting
# it to 999999 means "only at the end" -- which is what the original run
# did (it passed --core-metric-every=999999). base_train.py's DEFAULT is
# 2000, which would fire 138 times over a 277k-step run (~4.3 h), so the
# recipe must always set this explicitly.
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-999999}"

# torch.compile. Two of the seven trainers had it (base_train, chat_sft);
# the other five did not. This is now passed explicitly to the ones that
# support it, and the long-context ones compile their inner block instead
# (they use per-block activation checkpointing, which full-graph compile
# cannot fuse -- see the note in 07_longctx.sh).
COMPILE_BASE="${COMPILE_BASE:-1}"

# Threading and interconnect. Without OMP_NUM_THREADS=1, torchrun sets it
# anyway but only after the process has already forked; with the tokenizer
# also using threads, the ranks fight over the same cores and the
# dataloader stalls. The original launch script set this by hand.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# The allocator hint. PYTORCH_ALLOC_CONF is the current name;
# PYTORCH_CUDA_ALLOC_CONF is the deprecated one (torch still honours it but
# prints a warning).
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# NCCL: NVLink-connected A100s on this node. These are the settings that
# avoid the per-iteration NCCL teardown that shows up as jitter in dt.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"

# ---------------------------------------------------------------------------
# Client / harness. The gates import the real baby-onit package from here.
# ---------------------------------------------------------------------------
BABY_ONIT_DIR="${BABY_ONIT_DIR:-/data/rowel/baby-onit}"
GATE_DATA_DIR="${GATE_DATA_DIR:-/data/rowel}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR="${LOG_DIR:-$NANOCHAT_ROOT/recipe/logs}"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { printf '[utak][%s] %s\n' "$(date '+%F %T')" "$*"; }

die() { printf '[utak][%s] FATAL: %s\n' "$(date '+%F %T')" "$*" >&2; exit 1; }

# Run a module from the nanochat checkout with the base dir exported.
# The perf env vars (OMP_NUM_THREADS, PYTORCH_ALLOC_CONF, NCCL_*) are
# exported by this file, so every stage inherits them automatically.
run_mod() {
  local logfile="$1"; shift
  log "RUN: $*  (log: $logfile)"
  ( cd "$NANOCHAT_ROOT" \
    && NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" \
       PYTHONPATH="$NANOCHAT_ROOT:${PYTHONPATH:-}" \
       "$PY" -m "$@" ) > "$logfile" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || { log "FAILED (exit $rc): $*"; tail -30 "$logfile"; return $rc; }
  log "OK: $*"
  return 0
}

export MODEL_NAME MODEL_DIR_NAME REPO_TAG NANOCHAT_ROOT NANOCHAT_BASE_DIR \
       VLLM_DIR VLLM_BIN VLLM_PY MODEL_DIR \
       GPU_PRETRAIN GPU_SFT GPU_LONGC GPU_TOUCHUP GPU_SERVE \
       PUBLIC_PORT UPSTREAM_PORT \
       TAG_BASE TAG_SFT TAG_WARMUP TAG_RL_WARMUP TAG_RL_PILOT TAG_RL \
       TAG_LONGC TAG_LONGC_FINAL \
       TAG_TOUCH_A TAG_TOUCH_B TAG_TOUCH_C FINAL_TAG FINAL_STEP \
       BABY_ONIT_DIR GATE_DATA_DIR LOG_DIR \
       PRETRAIN_EVAL_TOKENS PRETRAIN_EVAL_EVERY \
       SFT_EVAL_TOKENS SFT_EVAL_EVERY CORE_METRIC_EVERY COMPILE_BASE \
       RL_EVAL_EVERY RL_EVAL_EXAMPLES_04 RL_EVAL_EXAMPLES_05 RL_EVAL_EXAMPLES_06

# The gates ask the server for the model by name. Keep that in one place.
export UTAK_MODEL_NAME="$MODEL_NAME"
export UTAK_MODEL_DIR_NAME="$MODEL_DIR_NAME"
