#!/usr/bin/env bash
# Launch the utak pipeline, detached, with the 2-GPU profile.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# The pipeline must survive a NODE REBOOT. On 2026-10-25 15:48 the node
# rebooted and took the run and the watcher with it: the pretrain was
# sitting at step 205,000 of 277,760, and nothing restarted it. The
# watcher cannot detect a reboot, because a reboot kills the watcher too.
#
# So the launcher is a separate, tiny thing that:
#   * picks the resume point from the STAGE CONTRACT, not from "does a
#     checkpoint exist" (recipe/resume_point.sh);
#   * exports the GPU layout with plain shell exports, so it does not
#     depend on `env`;
#   * refuses to start on a faulted driver, and on a CPU;
#   * writes the pid where the watcher looks for it.
#
# It is safe to run twice: the second invocation sees a live pipeline and
# exits.
#
# Usage:  bash /data/rowel/utak_smoke/start_pipeline.sh [profile] [cards]
set -u

PROFILE="${1:-2gpu}"
CARDS="${2:-0,2}"
NANOCHAT=/data/rowel/reasoner-model/nanochat
WORK=/data/rowel/utak_smoke
LOGDIR="/data/rowel/serve_logs/utak-$PROFILE"
RUNLOG="$LOGDIR/run_all.log"

FIRST="${CARDS%%,*}"
mkdir -p "$LOGDIR" "$WORK"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# ---- is the pipeline already running? -------------------------------------
if [ -f "$WORK/run_all.pid" ]; then
  OLD=$(cat "$WORK/run_all.pid" 2>/dev/null)
  if [ -n "$OLD" ] && [ -r "/proc/$OLD/cmdline" ] \
     && tr '\0' ' ' < "/proc/$OLD/cmdline" 2>/dev/null | grep -q 'run_all.sh'; then
    log "a pipeline is already running (pid $OLD) -- not starting another"
    exit 0
  fi
fi

# ---- is the driver alive? -------------------------------------------------
if ! nvidia-smi -L 2>&1 | grep -q 'GPU 0:'; then
  log "FATAL: the NVIDIA driver is not healthy (nvidia-smi -L has no GPU 0)."
  log "       This is a HOST-level fault: it needs a driver reload or a node"
  log "       reset. Refusing to start -- a restart now would run on the CPU."
  exit 70
fi

# ---- any card we want already busy with OUR OWN process? -------------------
if ! bash "$WORK/kill_stale.sh" --quiet >/dev/null 2>&1; then
  log "warning: kill_stale.sh reported a problem"
fi
sleep 3

# ---- where do we resume? --------------------------------------------------
cd "$NANOCHAT" || { log "FATAL: no $NANOCHAT"; exit 1; }
RESUME=$(bash "$NANOCHAT/recipe/resume_point.sh" 2>/dev/null | tail -1)
case "$RESUME" in
  ''|*[!0-9]*) RESUME=1 ;;
esac
log "the stage contract says the first incomplete stage is $RESUME"

# ---- go --------------------------------------------------------------------
export PROFILE
export GPU_PRETRAIN="$CARDS"
export GPU_SFT="$CARDS"
export GPU_LONGC="$FIRST"
export GPU_TOUCHUP="$FIRST"
export GPU_SERVE="$FIRST"

log "=== utak pipeline: PROFILE=$PROFILE cards=$CARDS from stage $RESUME ==="
log "    base dir: /data/rowel/nanochat_store/v2"
log "    run log:  $RUNLOG"

nohup bash "$NANOCHAT/recipe/run_all.sh" --from "$RESUME" \
      >> "$RUNLOG" 2>&1 &
RPID=$!
echo "$RPID" > "$WORK/run_all.pid"
log "  launched, pid $RPID"
log "  detached: it will keep running after this script exits"
