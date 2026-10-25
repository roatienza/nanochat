#!/usr/bin/env bash
# Bring the utak run back after a NODE REBOOT.
#
# WHY THIS EXISTS
# ---------------
# On 2026-10-25 15:48 the node rebooted. The pretrain was at step 205,000
# of 277,760 and the watcher was killed with it, so nothing restarted the
# run: 7 GPUs sat idle for 20 minutes until it was noticed by hand.
#
# A watcher cannot survive a reboot -- a reboot kills the watcher too. So
# the thing that must survive the reboot is NOT a process, it is a piece
# of configuration. On this node:
#
#   * systemd --user is NOT available (no user bus; Linger=no)
#   * /etc/cron.d is NOT writable
#   * the user crontab IS available and cron IS running
#
# so the @reboot cron entry is the mechanism. This script is what that
# entry calls.
#
# It is safe to run at any time: it refuses to start a second pipeline.
set -u
WORK=/data/rowel/utak_smoke
LOG=/data/rowel/serve_logs/utak-watch/reboot_resume.log
mkdir -p "$(dirname "$LOG")"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

log "=== post-reboot resume for the utak run ==="

# 1. Is the driver alive? If it is not, a restart would run on the CPU.
if ! nvidia-smi -L 2>&1 | grep -q 'GPU 0:'; then
  log "the NVIDIA driver is not healthy -- leaving the run down."
  log "this needs an ops action (driver reload or node reset)."
  exit 70
fi
log "the driver is healthy"

# 2. Is the pipeline already running? (a manual start may have beaten us)
if [ -f "$WORK/run_all.pid" ]; then
  OLD=$(cat "$WORK/run_all.pid" 2>/dev/null)
  if [ -n "$OLD" ] && [ -r "/proc/$OLD/cmdline" ] \
     && tr '\0' ' ' < "/proc/$OLD/cmdline" 2>/dev/null | grep -q 'run_all.sh'; then
    log "a pipeline is already running (pid $OLD) -- nothing to do"
    exit 0
  fi
fi

# 3. Give the other tenant's jobs a chance to come up first, so we do not
#    fight them for the CPU. They were restarted by their own @reboot.
sleep 60

# 4. Start the pipeline. It resumes from the stage contract.
log "starting the pipeline"
bash "$WORK/start_pipeline.sh" 2gpu 0,2 >> "$LOG" 2>&1 || log "start_pipeline.sh failed"

# 5. Start the watcher, which handles failures WITHIN a boot.
if [ -f "$WORK/watch.pid" ]; then
  W=$(cat "$WORK/watch.pid" 2>/dev/null)
  if [ -z "$W" ] || [ ! -r "/proc/$W/cmdline" ]; then
    log "starting the watcher"
    nohup bash "$WORK/watch_utak.sh" >> /data/rowel/serve_logs/utak-watch/stdout.log 2>&1 &
    echo $! > "$WORK/watch.pid"
  fi
fi
log "done"
