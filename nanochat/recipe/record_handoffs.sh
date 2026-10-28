#!/usr/bin/env bash
# utak — check the stage handoff records against the disk.
#
# WHAT A HANDOFF RECORD IS
# ------------------------
# A claim that a stage FINISHED, at a given step. It is written by the
# stage itself, after its training command exits 0 (see
# record_stage_handoff in recipe/config.sh).
#
# WHY NOTHING MAY WRITE ONE AUTOMATICALLY
# ----------------------------------------
# Because a CHECKPOINT does not mean a stage finished. Stage 1 saves
# every 5,000 steps and stage 3 every 100, so a stage that died at step
# 27,299 leaves checkpoints behind that look exactly like a finished
# stage's output.
#
# Recording one of those would make the resume logic skip the stage and
# the next stage would train on a half-trained model. That is the very
# bug the records were introduced to kill -- on 2026-10-27 the resume
# logic called a stage unfinished because it looked for a step that did
# not exist, and the run looped for 32 hours.
#
# So this script NEVER writes a record just because it sees a checkpoint.
# It only ever writes one when an operator says which stage finished
# (--backfill), and even then it refuses if that stage's trainer is
# running right now.
#
# Usage:
#   ./recipe/record_handoffs.sh              # verify only (the default)
#   ./recipe/record_handoffs.sh --verify     # the same thing, explicitly
#   ./recipe/record_handoffs.sh --backfill TAG   # record TAG's final step
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

MODE="verify"
BACKFILL_TAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --verify) MODE="verify" ;;
    --backfill) MODE="backfill"; shift; BACKFILL_TAG="${1:-}" ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# Is the trainer that PRODUCES this tag running right now?
#
# The match is on the tag as the trainer's OWN --model-tag or
# --output-tag. A stage that merely READS a checkpoint with that tag in
# it is a different stage, and must not count.
stage_is_running() {
  local tag="$1" p cmd
  for p in /proc/[0-9]*; do
    # A process can vanish between the glob and the read; that is normal
    # on a busy node and must not print anything.
    cmd=$( { tr '\0' ' ' < "$p/cmdline"; } 2>/dev/null ) || continue
    [ -n "$cmd" ] || continue
    case "$cmd" in
      *"--model-tag $tag "*|*"--output-tag $tag "*) return 0 ;;
    esac
  done
  return 1
}

# The checkpoint steps a directory holds, newest first.
steps_in() {
  ls -1 "$1"/model_*.pt 2>/dev/null \
    | sed -n 's/.*model_\([0-9]\{6\}\)\.pt/\1/p' | sort -rn
}

# tag : the directory the stage writes its checkpoint into
STAGES=(
  "$TAG_BASE|base_checkpoints/$TAG_BASE"
  "$TAG_SFT_TOOLS|chatsft_checkpoints/$TAG_SFT_TOOLS"
  "$TAG_WARMUP|chatsft_checkpoints/$TAG_WARMUP"
  "$TAG_RL_WARMUP|chatrl_checkpoints/$TAG_RL_WARMUP"
  "$TAG_RL_PILOT|chatrl_checkpoints/$TAG_RL_PILOT"
  "$TAG_RL|chatrl_checkpoints/$TAG_RL"
  "$TAG_LONGC_FINAL|base_checkpoints/$TAG_LONGC_FINAL"
  "$TAG_SFT|chatsft_checkpoints/$TAG_SFT"
  "$FINAL_TAG|chatsft_checkpoints/$FINAL_TAG"
)

if [ "$MODE" = "backfill" ]; then
  [ -n "$BACKFILL_TAG" ] || { echo "--backfill needs a tag" >&2; exit 2; }
  row=""
  for r in "${STAGES[@]}"; do
    [ "${r%%|*}" = "$BACKFILL_TAG" ] && row="$r"
  done
  [ -n "$row" ] || { echo "unknown tag: $BACKFILL_TAG" >&2; exit 2; }
  dir="${row#*|}"
  full="$NANOCHAT_BASE_DIR/$dir"
  if stage_is_running "$BACKFILL_TAG"; then
    log "REFUSING to backfill $BACKFILL_TAG: its trainer is running right now."
    log "  A stage that is still training has not finished, and its newest"
    log "  checkpoint is a periodic save, not the final one."
    exit 1
  fi
  step=$(stage_dir_final_step "$full") \
    || { log "no checkpoints in $full; nothing to backfill"; exit 1; }
  log "backfilling $BACKFILL_TAG -> $step (from $full)"
  handoff_write "$BACKFILL_TAG" "$step" || exit 1
  exit 0
fi

# ---- verify --------------------------------------------------------------
FAILED=0
for row in "${STAGES[@]}"; do
  tag="${row%%|*}"; dir="${row#*|}"
  full="$NANOCHAT_BASE_DIR/$dir"
  [ -d "$full" ] || continue
  have=$(steps_in "$full")
  [ -n "$have" ] || continue
  newest=$(printf '%s\n' "$have" | head -1)
  rec=$(handoff_read "$tag" 2>/dev/null || echo "")

  if [ -z "$rec" ]; then
    if stage_is_running "$tag"; then
      # Expected: a stage that is running has not written its record yet.
      log "  $tag: still running (newest checkpoint $newest); no record yet -- expected"
    else
      # Either it died, or it finished before the records existed. Either
      # way the resume logic will redo it, which is SAFE. It is worth
      # saying out loud, but it is not a broken record.
      log "  WARNING $tag: has checkpoints (newest $newest) but no handoff record."
      log "           The resume logic will redo this stage, which is safe."
      log "           If it really finished, backfill it with:"
      log "             recipe/record_handoffs.sh --backfill $tag"
    fi
    continue
  fi

  # A record exists. It must name a checkpoint that is really there.
  if [ ! -f "$full/model_$(printf '%06d' "$rec").pt" ]; then
    log "  FAIL $tag: the record names step $rec, but $full has no such checkpoint"
    FAILED=$((FAILED + 1))
    continue
  fi

  # And, if the stage is not running, it must name the FINAL checkpoint.
  # A stale record would make the next stage read an older model.
  if ! stage_is_running "$tag" && [ "$rec" != "$((10#$newest))" ]; then
    log "  FAIL $tag: the record names step $rec, but the final checkpoint is $newest"
    FAILED=$((FAILED + 1))
    continue
  fi

  log "  $tag: record $rec is current"
done

if [ "$FAILED" != 0 ]; then
  log "VERIFY FAILED: $FAILED record(s) disagree with the disk."
  exit 1
fi
exit 0
