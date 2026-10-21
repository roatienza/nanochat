#!/usr/bin/env bash
# utak — the full recipe, end to end.
#
#   ./recipe/run_all.sh              # everything, from scratch
#   ./recipe/run_all.sh --from 7     # resume at the long-context stage
#   ./recipe/run_all.sh --only 9,10  # run only the listed stages
#   ./recipe/run_all.sh --list       # list the stages and exit
#
# THE STAGES, IN THE ORDER THEY MUST RUN:
#
#    0  data          the tokenizer + the pre-training corpus
#    1  pretrain      base model, 2k context                -> $TAG_BASE
#    2  sft_tools     tool-calling SFT, 2k context          -> $TAG_SFT_TOOLS
#    3  warmup        GSM8K warmup SFT (fresh optimizer)     -> $TAG_WARMUP
#    4  rl_math       math RL on GSM8K                       -> $TAG_RL_WARMUP
#    5  rl_agentic    agentic RL on sandbox environments     -> $TAG_RL_PILOT
#    6  rl_curriculum weighted-block + exemplar curriculum  -> $TAG_RL
#    7  longctx       context extension 2k -> 128k           -> $TAG_LONGC_FINAL
#    8  sft_long      supervised fine-tuning at 128k         -> $TAG_SFT
#    9  touchup       the touch-up ladder                   -> $TAG_TOUCH_B
#   10  pack          checkpoint -> vLLM model dir
#   11  serve         guard on :8000 -> vLLM on :8010
#   12  gate          the v5x and HP gates
#   13  e2e           the end-to-end acceptance test
#   14  regression    the offline unit tests
#   15  scrub_test    the live-server hypothesis test (the guard's history scrub)
#
# Note the ordering: the RL chain (4-6) runs BEFORE the context
# extension (7). The 131,072-context model is initialized from the RL
# weights, so running 7 before 4-6 would train a different model.
#
# Stage 11 is a long-running foreground process (it supervises the guard
# and vLLM). This script starts it in the background so the gates can
# run, and leaves it running -- that is the deployment.
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

FROM=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --list) ls -1 "$NANOCHAT_ROOT/recipe"/*.sh | xargs -n1 basename; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

should_run() {
  local n="$1"
  [ -n "$ONLY" ] && { [[ ",$ONLY," == *",$n,"* ]] && return 0 || return 1; }
  [ "$n" -ge "$FROM" ] && return 0 || return 1
}

run_stage() {
  local n="$1"; shift
  should_run "$n" || { log "skip stage $n"; return 0; }
  log "########## STAGE $n ##########"
  "$@"
}

run_stage 0  bash "$NANOCHAT_ROOT/recipe/00_data.sh"
run_stage 1  bash "$NANOCHAT_ROOT/recipe/01_pretrain.sh"
run_stage 2  bash "$NANOCHAT_ROOT/recipe/02_sft_tools.sh"
run_stage 3  bash "$NANOCHAT_ROOT/recipe/03_warmup.sh"
run_stage 4  bash "$NANOCHAT_ROOT/recipe/04_rl_math.sh"
run_stage 5  bash "$NANOCHAT_ROOT/recipe/05_rl_agentic.sh"
run_stage 6  bash "$NANOCHAT_ROOT/recipe/06_rl_curriculum.sh"
run_stage 7  bash "$NANOCHAT_ROOT/recipe/07_longctx.sh"
run_stage 8  bash "$NANOCHAT_ROOT/recipe/08_sft_long.sh"
run_stage 9  bash "$NANOCHAT_ROOT/recipe/09_touchup.sh"
run_stage 10 bash "$NANOCHAT_ROOT/recipe/10_pack.sh"

if should_run 11; then
  log "########## STAGE 11 (serve, background) ##########"
  # Restart-safe: the serve script frees the ports first.
  nohup bash "$NANOCHAT_ROOT/recipe/11_serve.sh" > "$LOG_DIR/serve.log" 2>&1 &
  SERVE_PID=$!
  echo "$SERVE_PID" > "$LOG_DIR/serve.pid"
  log "serve started (pid $SERVE_PID); waiting for it to answer..."
  ready=0
  for i in $(seq 1 180); do
    curl -s -m 3 "http://127.0.0.1:$PUBLIC_PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    sleep 2
  done
  [ "$ready" = 1 ] || { tail -30 "$LOG_DIR/serve.log"; die "the server never came up"; }
  log "the server is answering on :$PUBLIC_PORT"
fi

run_stage 12 bash "$NANOCHAT_ROOT/recipe/12_gate.sh"
run_stage 13 bash "$NANOCHAT_ROOT/recipe/13_e2e.sh"
run_stage 14 bash "$NANOCHAT_ROOT/recipe/14_regression.sh"
run_stage 15 bash "$NANOCHAT_ROOT/recipe/15_scrub_test.sh"

log ""
log "=========================================================="
log " utak is trained, packed, served and verified."
log "   model id : $MODEL_NAME"
log "   endpoint : http://<host>:$PUBLIC_PORT/v1"
log "   upstream : http://127.0.0.1:$UPSTREAM_PORT (vLLM, internal)"
log "=========================================================="
