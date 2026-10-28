#!/usr/bin/env bash
# Watch the utak run and react to a failure.
#
# WHAT IT DOES
# ------------
# The utak run takes days. The failure modes that actually happen on this
# node are:
#
#   1. A stage OOMs because a parallel session's job (robot-decisions VLA
#      benchmarks) grabbed one of the two cards. The stage dies, and the
#      corpse keeps its ~35 GiB, so even a restart would die at model
#      load. kill_stale.sh clears that.
#   2. The run finishes. This script then leaves the server up (stage 11
#      is the deployment) and stops.
#
# So: check every 10 minutes. If the pipeline pid is gone AND the last
# stage log shows a failure, clean the cards and RESTART from the stage
# after the last successful one. The recipe's own resume logic picks up
# from the newest checkpoint, so a restart is cheap.
#
# Usage: ./watch_utak.sh          # run in the foreground
#        nohup ./watch_utak.sh &   # detached
set -u
LOG=/data/rowel/serve_logs/utak-2gpu/run_all.log
STAGE_DIR=/data/rowel/reasoner-model/nanochat/recipe/logs
WORK=/data/rowel/utak_smoke
NANOCHAT=/data/rowel/reasoner-model/nanochat
INTERVAL=600

# The checkpoint tags, so the disk guard knows which directories to prune.
# config.sh is the single source of truth for them.
# shellcheck disable=SC1091
. "$NANOCHAT/recipe/config.sh" >/dev/null 2>&1 || true

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# ---- THE NO-PROGRESS GUARD ----------------------------------------------
# A restart that lands on the SAME stage and fails the same way is not
# progress; it is a loop, and on this recipe it costs 10.6 hours of both
# GPUs per cycle.
#
# That is exactly what happened on 2026-10-27: stage 3 died on a missing
# model_013630.pt, the resume logic said "stage 2 is unfinished", and the
# watcher restarted stage 2 -- twice, for 21 hours of both GPUs, to
# produce a checkpoint that stage 3 could not read either.
#
# So the watcher remembers where it last restarted from. If the pipeline
# dies again and the resume point is the SAME stage, that is the second
# failure of the same thing; after a few of those it stops restarting and
# says so loudly, because at that point the problem is a broken stage,
# and burning the rest of the schedule on retries helps nobody.
# ---------------------------------------------------------------------------
MAX_SAME_STAGE_RESTARTS="${MAX_SAME_STAGE_RESTARTS:-3}"
GUARD_STATE="$WORK/restart_guard.state"

note_restart() {
  # note_restart <stage>
  local stage="$1" last count
  last=""
  count=0
  if [ -f "$GUARD_STATE" ]; then
    read -r last count < "$GUARD_STATE" 2>/dev/null || true
  fi
  if [ "$last" = "$stage" ]; then
    count=$((count + 1))
  else
    count=1
  fi
  printf '%s %s\n' "$stage" "$count" > "$GUARD_STATE"
  printf '%s\n' "$count"
}

log "=== watching the utak run (every ${INTERVAL}s) ==="

while true; do
  sleep "$INTERVAL"

  if [ ! -f "$WORK/run_all.pid" ]; then
    log "no pid file; nothing to watch"
    exit 1
  fi
  PID=$(cat "$WORK/run_all.pid")
  if kill -0 "$PID" 2>/dev/null; then
    # still running
    step=$(grep -oE "^step [0-9]+/[0-9]+" "$STAGE_DIR/stage1_pretrain.log" 2>/dev/null | tail -1)
    [ -n "$step" ] && log "running: $step"

    # ---- GPU HEALTH GATE ------------------------------------------------
    # A live pid does NOT mean a healthy run. On 2026-10-24 the node's
    # driver faulted; the trainer fell back to the CPU and the pipeline
    # pid stayed alive for 15.6 hours while nothing progressed. This
    # watcher only asked "is the pid alive?", so it saw a healthy run.
    #
    # Ask the driver directly.
    if ! gpus_healthy; then
      log "FATAL: the NVIDIA driver is NOT healthy (nvidia-smi -L reports"
      log "       'Unknown Error' or 'No devices were found'). The run is"
      log "       either dead or has silently fallen back to the CPU."
      log "       This is a HOST-level fault: it needs a driver reload or a"
      log "       node reset, which is an ops action. Restarting is pointless"
      log "       until that is done."
      log "       Not restarting. Will keep checking every ${INTERVAL}s."
      sleep "$INTERVAL"
      continue
    fi

    # ---- STALL DETECTOR --------------------------------------------------
    # The training log is the heartbeat. If NO stage log has been written
    # for STALL_SECONDS, the run is making no progress even though the pid
    # is alive. A healthy stage 1 writes a step line roughly every second
    # and an eval line every few minutes, so 30 minutes without a write
    # means something is very wrong.
    #
    # IMPORTANT: look at the NEWEST stage log, not at each one in turn.
    # Stage 1's log stops being written the moment stage 2 starts, so
    # checking "is stage1_pretrain.log fresh?" would fire on a perfectly
    # healthy run as soon as the pipeline moved on to stage 2. The
    # question is "has ANY stage made progress recently?".
    STALL_SECONDS=${STALL_SECONDS:-1800}
    NOW=$(date +%s)
    NEWEST=0
    NEWEST_FILE=""
    for lf in "$STAGE_DIR"/stage*.log; do
      [ -f "$lf" ] || continue
      MT=$(stat -c %Y "$lf" 2>/dev/null || echo 0)
      if [ "$MT" -gt "$NEWEST" ]; then NEWEST=$MT; NEWEST_FILE="$lf"; fi
    done
    if [ "$NEWEST" -gt 0 ]; then
      AGE=$((NOW - NEWEST))
      if [ "$AGE" -gt "$STALL_SECONDS" ]; then
        log "STALL: the newest stage log ($(basename "$NEWEST_FILE")) has not been"
        log "       written for $((AGE/60)) min (threshold ${STALL_SECONDS}s)."
        log "       The pipeline pid is alive but the stage is not progressing."
        log "       Last 5 lines:"
        tail -5 "$NEWEST_FILE" 2>/dev/null | sed 's/^/           /'
        # Do NOT restart into the same wall if the driver is the problem;
        # the GPU health gate above already handled that. If the driver is
        # fine and the stage is still stuck, a restart is the right move.
        log "       The driver is healthy, so this is a stuck stage. Restarting."
        kill -TERM "$PID" 2>/dev/null
        sleep 30
        kill -KILL "$PID" 2>/dev/null
        log "       killed the stuck pipeline (pid $PID); the next cycle restarts it."
        break
      fi
    fi

    # ---- DISK GUARD ----------------------------------------------------
    # One checkpoint set is 14.63 GB (6.31 GB model + 2 x 4.16 GB optimizer
    # shards). Stage 1 saves one every 5,000 steps and is 277,760 steps
    # long, so it alone writes 819 GB; the whole pipeline projects to
    # 1.68 TB on a filesystem with 4.0 TB free that another tenant is
    # also using. The ORIGINAL run pruned -- base_checkpoints/d28b-v2
    # holds 56 metas but ONE model file -- so this run must prune too or
    # it dies of ENOSPC three days from now.
    #
    # The pruner keeps the newest 3 sets (one is the resume point) and
    # every meta file, and it will not touch a file younger than 30
    # minutes, so it can never delete a checkpoint that is being written.
    avail=$(df -Pk /data | awk 'NR==2 {print $4}')
    avail_gb=$((avail / 1024 / 1024))
    if [ "$avail_gb" -lt 1200 ]; then
      log "disk: ${avail_gb} GB free -- pruning old checkpoints"
      for d in base_checkpoints/$TAG_BASE chatsft_checkpoints/$TAG_SFT_TOOLS \
               chatsft_checkpoints/$TAG_WARMUP chatrl_checkpoints/$TAG_RL_WARMUP \
               chatrl_checkpoints/$TAG_RL_PILOT chatrl_checkpoints/$TAG_RL \
               base_checkpoints/$TAG_LONGC_FINAL chatsft_checkpoints/$TAG_SFT \
               chatsft_checkpoints/$FINAL_TAG; do
        out=$(bash "$NANOCHAT/recipe/prune_checkpoints.sh" \
                "$NANOCHAT_BASE_DIR/$d" 3 1800 2>/dev/null)
        case "$out" in
          pruned*) log "    $out" ;;
        esac
      done
      avail=$(df -Pk /data | awk 'NR==2 {print $4}')
      log "disk after pruning: $((avail / 1024 / 1024)) GB free"
    fi
    continue
  fi

  # the pipeline is gone. Did it finish, or did it die?
  if grep -q "utak is trained, packed, served and verified" "$LOG" 2>/dev/null; then
    log "the run COMPLETED successfully"
    exit 0
  fi

  log "the pipeline is GONE and did not complete. Last lines:"
  tail -12 "$LOG" 2>/dev/null | sed 's/^/    /'

  # ---- GPU HEALTH GATE, BEFORE THE RESTART ----------------------------
  # This is the path that caused the 15.6-hour loss on 2026-10-24: the
  # driver had faulted, the pipeline died, the watcher restarted it, and
  # the trainer came back up on the CPU. Restarting into a dead driver is
  # worse than not restarting, because the restart looks like progress.
  #
  # So: check the driver first. If it is faulted, wait for an operator to
  # reload it. The recipe's own gpus_healthy() is the single source of
  # truth, and it is the same check the stages now use.
  if ! gpus_healthy; then
    log "FATAL: the NVIDIA driver is NOT healthy (nvidia-smi -L reports"
    log "       'Unknown Error' or 'No devices were found' for every card)."
    log "       The pipeline died because of this, and restarting would just"
    log "       start it on the CPU again. Not restarting."
    log "       This needs an ops action (driver reload or node reset)."
    log "       Waiting ${INTERVAL}s and re-checking."
    sleep "$INTERVAL"
    continue
  fi
  log "the driver is healthy; proceeding with the restart."

  # ---- NO-PROGRESS GUARD ------------------------------------------------
  # Where would this restart land? If it lands on the same stage that
  # just failed, and it has done that several times, the stage is broken
  # and retrying only burns the schedule.
  resume_guess="$(bash "$NANOCHAT/recipe/resume_point.sh" 2>/dev/null | tail -1)"
  case "$resume_guess" in
    ''|*[!0-9]*) resume_guess="" ;;
  esac
  if [ -n "$resume_guess" ]; then
    tries="$(note_restart "$resume_guess")"
    if [ "$tries" -gt "$MAX_SAME_STAGE_RESTARTS" ]; then
      log "FATAL: the pipeline has failed and been restarted ${tries} times"
      log "       on the SAME stage (${resume_guess}). That is not a flaky"
      log "       failure; the stage is broken. NOT restarting again --"
      log "       each attempt costs hours of both GPUs and cannot succeed."
      log "       The last 30 lines of the log:"
      tail -30 "$LOG" 2>/dev/null | sed 's/^/    /'
      log "       Fix the stage, then delete $GUARD_STATE to re-arm the guard."
      sleep "$INTERVAL"
      continue
    fi
    log "the no-progress guard: this is attempt ${tries} on stage ${resume_guess}"
  fi

  log "clearing any stale GPU holders"
  bash "$WORK/kill_stale.sh" 2>/dev/null | sed 's/^/    /' || true
  sleep 20

  # Work out where to resume.
  #
  # This used to be "the last stage whose directory contains any
  # model_*.pt". That is WRONG, and wrong in the most expensive way
  # possible: stage 1 saves every 5,000 steps, so a pretrain that died at
  # step 27,299 leaves model_025000.pt and model_027000.pt behind. The old
  # test called that "stage 1 done" and restarted at stage 2, which then
  # failed for want of model_277760.pt -- and the watcher restarted at
  # stage 2 again, for ever. The pretrain, 74 of the ~120 hours, would
  # never finish.
  #
  # recipe/resume_point.sh asks the right question: has each stage produced
  # the FINAL checkpoint, at the exact step the next stage reads?
  resume=$(bash /data/rowel/reasoner-model/nanochat/recipe/resume_point.sh 2>/dev/null | tail -1)
  case "$resume" in
    ''|*[!0-9]*) resume=1 ;;
  esac
  log "the contract says the first incomplete stage is $resume:"
  bash /data/rowel/reasoner-model/nanochat/recipe/resume_point.sh --why 2>/dev/null \
    | sed 's/^/    /'

  # Disk. Stage 1 alone writes 14.6 GB per checkpoint (6.3 GB model + 2 x
  # 4.2 GB optimizer shards) and it saves every 5,000 steps -- 56 saves,
  # about 800 GB if nothing is pruned. The original run kept only the last
  # model file; this run keeps them all, so the disk has to be watched.
  avail=$(df -Pk /data | awk 'NR==2 {print $4}')
  avail_gb=$((avail / 1024 / 1024))
  log "disk: ${avail_gb} GB free on /data"
  if [ "$avail_gb" -lt 400 ]; then
    log "WARNING: less than 400 GB free. Stage 1 needs about 14.6 GB per"
    log "         checkpoint and saves every 5,000 steps. Old checkpoints"
    log "         may need pruning before the run can finish."
  fi

  # Before restarting, check that the stages ahead still work. A broken
  # stage would otherwise be retried forever, and each attempt would
  # burn the same time.
  log "running the pre-flight (the stage contract, the pack, the tests)..."
  if ! bash /data/rowel/reasoner-model/nanochat/recipe/preflight.sh \
       > "$WORK/preflight.log" 2>&1; then
    log "THE PRE-FLIGHT FAILED -- not restarting. The log:"
    tail -25 "$WORK/preflight.log" | sed 's/^/    /'
    log "waiting 30 minutes and trying again"
    sleep 1800
    continue
  fi
  log "the pre-flight passed"

  cd /data/rowel/reasoner-model/nanochat || exit 1
  nohup env PROFILE=2gpu GPU_PRETRAIN=0,2 GPU_SFT=0,2 GPU_LONGC=0 \
        GPU_TOUCHUP=0 GPU_SERVE=0 \
        bash ./recipe/run_all.sh --from "$resume" \
        >> "$LOG" 2>&1 &
  echo $! > "$WORK/run_all.pid"
  log "restarted, pid $!, from stage $resume"
done
