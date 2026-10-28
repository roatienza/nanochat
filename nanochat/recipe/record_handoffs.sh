#!/usr/bin/env bash
# utak — one-shot: record the handoff for a stage that finished under the
# OLD code, before the handoff mechanism existed.
#
# WHY THIS EXISTS
# ---------------
# The handoff records were introduced on 2026-10-28, after the 2-GPU run
# had already completed stages 1 and 2. Those two stages ran the OLD
# scripts, which did not write a record, so the records were backfilled
# by hand from what was on the disk.
#
# This script re-derives them, so the records are always the truth about
# the checkpoints rather than something somebody typed. It is safe to run
# at any time: it only writes a record for a stage whose final checkpoint
# is on disk, and it never deletes one.
#
# It is also what makes a stage that is running under the old code safe:
# when it finishes, running this refreshes its record.
#
# Usage:  ./recipe/record_handoffs.sh            # refresh what is there
#         ./recipe/record_handoffs.sh --verify   # check, do not write
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

VERIFY=0
[ "${1:-}" = "--verify" ] && VERIFY=1

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

RC=0
for row in "${STAGES[@]}"; do
  tag="${row%%|*}"; dir="${row#*|}"
  full="$NANOCHAT_BASE_DIR/$dir"
  [ -d "$full" ] || continue
  if ! step=$(stage_dir_final_step "$full"); then
    continue  # no checkpoints yet: the stage has not run
  fi
  have=$(handoff_read "$tag" 2>/dev/null || echo "")
  if [ "$have" = "$step" ]; then
    log "  $tag: record $step is current"
    continue
  fi
  if [ "$have" != "" ]; then
    log "  $tag: record says $have but the final checkpoint is $step -- fixing"
  else
    log "  $tag: no record; the final checkpoint is $step -- recording"
  fi
  if [ "$VERIFY" = 1 ]; then
    RC=1
    continue
  fi
  handoff_write "$tag" "$step" || RC=1
done

if [ "$VERIFY" = 1 ] && [ "$RC" != 0 ]; then
  log "VERIFY FAILED: at least one record does not match the disk."
  exit 1
fi
exit 0
