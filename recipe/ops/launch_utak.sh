#!/usr/bin/env bash
# Launch the whole utak pipeline, detached, with the 2-GPU profile.
#
# WHY THE 2-GPU PROFILE
# ---------------------
# Measured on 2026-10-22, d28, 2k ctx, real scripts.base_train:
#   1 GPU  device batch 2:  1,868 ms/step  17,525 tok/s  41.4% MFU
#   2 GPUs device batch 2:    964 ms/step  33,941 tok/s  40.1% MFU
#   (device batch 4 OOMs on both)
# So two GPUs are worth 1.94x the throughput and they fit. The card
# indices are pinned here because the auto-pick inside config.sh can
# only see the cards that are free AT THAT MOMENT, and this run must
# keep both cards for days.
#
# The stages that need a single card (7, 8, 9) are pinned to GPU 0 by
# the profile itself, because base_train_long.py / chat_sft_long.py use
# a plain AdamW and never reduce gradients across ranks: running them
# on two cards would silently diverge the ranks.
#
# Usage:
#   ./launch_utak.sh 2gpu 0,2     # 2 GPUs, cards 0 and 2
#   ./launch_utak.sh 1gpu 0       # 1 GPU, card 0
set -u
PROFILE="${1:-2gpu}"
CARDS="${2:-0,2}"
NANOCHAT=/data/rowel/reasoner-model/nanochat
WORK=/data/rowel/utak_smoke
LOGDIR=/data/rowel/serve_logs/utak-$PROFILE
mkdir -p "$LOGDIR" "$WORK"

# One card for the single-GPU stages, taken from the list.
FIRST="${CARDS%%,*}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

log "=== utak pipeline: PROFILE=$PROFILE cards=$CARDS ==="
log "    nanochat: $NANOCHAT"
log "    base dir: /data/rowel/nanochat_store/v2"
log "    log dir:  $LOGDIR"

# Free the cards of anything of ours that is still holding them.
bash "$WORK/kill_stale.sh" --quiet 2>/dev/null || true
sleep 3

# Make sure no vLLM server is holding a card or a port.
for port in 8000 8010; do
  pids=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
  for p in $pids; do
    log "    freeing port $port (pid $p)"
    kill "$p" 2>/dev/null || true
  done
done
sleep 2

RUNLOG="$LOGDIR/run_all.log"
: > "$RUNLOG"

cd "$NANOCHAT" || { log "FATAL: no $NANOCHAT"; exit 1; }

nohup env PROFILE="$PROFILE" \
  GPU_PRETRAIN="$CARDS" \
  GPU_SFT="$CARDS" \
  GPU_LONGC="$FIRST" \
  GPU_TOUCHUP="$FIRST" \
  GPU_SERVE="$FIRST" \
  bash "$NANOCHAT/recipe/run_all.sh" --from 0 \
  >> "$RUNLOG" 2>&1 &

RPID=$!
echo "$RPID" > "$WORK/run_all.pid"
log "  launched, pid $RPID"
log "  run log: $RUNLOG"
log "  the run is detached and will continue after this script exits"
