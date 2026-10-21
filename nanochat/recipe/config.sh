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
       BABY_ONIT_DIR GATE_DATA_DIR LOG_DIR

# The gates ask the server for the model by name. Keep that in one place.
export UTAK_MODEL_NAME="$MODEL_NAME"
export UTAK_MODEL_DIR_NAME="$MODEL_DIR_NAME"
