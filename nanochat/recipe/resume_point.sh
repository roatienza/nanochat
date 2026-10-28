#!/usr/bin/env bash
# Work out the first stage to run, given what is already on disk.
#
# WHY THIS EXISTS
# ---------------
# The run takes days. If a stage dies, the pipeline has to be restarted, and
# the restart must resume at the right place.
#
# The obvious way to do that -- "if the stage's checkpoint directory has a
# model_*.pt in it, that stage is done" -- is WRONG, and wrong in the most
# expensive way possible.
#
# Stage 1 saves every 5,000 steps. If it dies at step 27,299, the directory
# base_checkpoints/utak-base holds model_025000.pt and model_027000.pt. A
# "does any checkpoint exist?" test says stage 1 is DONE, so the restart
# begins at stage 2 -- which reads the pretrain output at step 277,760, a
# file that does not exist, and dies in seconds. The restart then loops:
# stage 2 fails, the watcher restarts from stage 2 again, for ever. The
# pretrain, which is 74 of the ~120 hours, never finishes.
#
# So a stage is only done when it has produced its FINAL checkpoint, at the
# exact step the next stage reads. Those steps are the stage contract.
#
# WHY THIS FILE ASKED A SECOND, SEPARATE QUESTION
# -----------------------------------------------
# This used to look for a checkpoint at a HARDCODED step: the recipe's
# constant for that link, taken from the original run. The consuming stage
# scripts looked for the same constant, so at least they agreed with each
# other -- but they both disagreed with the DISK whenever the run's real
# step differed from the original run's.
#
# It did. Stage 2's final step is 13,650 on the corpus rendered on
# 2026-10-22; the original run's corpus gave 13,630. So:
#
#   - stage 3 asked for model_013630.pt, which does not exist, and died;
#   - THIS file asked the same question, found no model_013630.pt, and
#     concluded that stage 2 was INCOMPLETE;
#   - the watcher therefore restarted the pipeline AT STAGE 2, which ran
#     for another 10.6 hours to produce model_013650.pt again;
#   - and stage 3 died the same way. That loop ran twice, on 2026-10-27 at
#     05:37 and 16:21, before it was found.
#
# The fix is to ask ONE question, in ONE place, and have both the resume
# logic and the consuming stages use it. A stage that finishes now writes
# a HANDOFF RECORD -- the step it reached -- into $HANDOFF_DIR/<tag>.step,
# and the next stage reads that record. See recipe/config.sh, "THE STAGE
# HANDOFF". The record is written only after the stage's training command
# exits 0, so its presence also proves the stage FINISHED.
#
# This file and the stage scripts now both resolve the step through
# stage_resolved_step(), so they cannot disagree again.
#
# Usage:
#   ./recipe/resume_point.sh          # prints e.g. 4
#   ./recipe/resume_point.sh --why    # prints the table too
#   ./recipe/resume_point.sh --why TAG_BASE=d28b-v2 ...   # override the tags
set -u
cd "$(dirname "$(readlink -f "$0")")/.."

# Tag overrides, so the same table can be pointed at another run's
# checkpoints (the original run's d28b-* tags) to test the logic.
WHY=0
OVERRIDES=()
for arg in "$@"; do
  case "$arg" in
    --why) WHY=1 ;;
    *=*)  OVERRIDES+=("$arg") ;;
    *)    echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

. ./recipe/config.sh

# config.sh sets the tags with ${VAR:-default}, so an override has to be in
# the environment before it is sourced.
for ov in ${OVERRIDES+"${OVERRIDES[@]}"}; do
  name="${ov%%=*}"; value="${ov#*=}"
  case "$name" in
    TAG_*|FINAL_*|NANOCHAT_BASE_DIR) printf -v "$name" '%s' "$value" ;;
    *) echo "refusing to override $name" >&2; exit 2 ;;
  esac
done

BASE="${NANOCHAT_BASE_DIR}"

# stage : the tag that PRODUCES the checkpoint : the checkpoint directory :
#         the step the next stage reads : how the step is decided
#
# "dynamic" means the step comes from the producer's handoff record -- it
# is the step the producer really reached, and it can only be known after
# the producer has run. "fixed" means the step is a property of the
# recipe (a --num-iterations, or a deliberate mid-run pick) and the
# constant is authoritative.
#
# The tags and the steps both come from config.sh, so this table follows the
# recipe if either is renamed.
CONTRACT=(
  "1|$TAG_BASE|base_checkpoints/$TAG_BASE|$TAG_BASE_FINAL_STEP|fixed"
  "2|$TAG_SFT_TOOLS|chatsft_checkpoints/$TAG_SFT_TOOLS|$TAG_SFT_TOOLS_FINAL_STEP|dynamic"
  "3|$TAG_WARMUP|chatsft_checkpoints/$TAG_WARMUP|$TAG_WARMUP_FINAL_STEP|dynamic"
  "4|$TAG_RL_WARMUP|chatrl_checkpoints/$TAG_RL_WARMUP|$TAG_RL_WARMUP_FINAL_STEP|dynamic"
  "5|$TAG_RL_PILOT|chatrl_checkpoints/$TAG_RL_PILOT|$TAG_RL_PILOT_FINAL_STEP|dynamic"
  "6|$TAG_RL|chatrl_checkpoints/$TAG_RL|$TAG_RL_FINAL_STEP|dynamic"
  "7|$TAG_LONGC_FINAL|base_checkpoints/$TAG_LONGC_FINAL|$TAG_LONGC_FINAL_STEP|fixed"
  "8|$TAG_SFT|chatsft_checkpoints/$TAG_SFT|$TAG_SFT_FINAL_STEP|fixed"
  "9|$FINAL_TAG|chatsft_checkpoints/$FINAL_TAG|$FINAL_STEP|fixed"
)

# The step each stage's checkpoint is at, resolved the same way the
# consuming stage scripts resolve it.
resolve() {
  local tag="$1" dir="$2" constant="$3" mode="$4" step
  step="$(stage_resolved_step "$tag" "$BASE/$dir" "$constant")"
  if [ "$mode" = "dynamic" ] && ! handoff_read "$tag" >/dev/null 2>&1; then
    # No handoff record: the producer has not finished, so there is no
    # step to read and the stage is not done -- whatever the constant
    # says.
    printf 'none\n'
    return 0
  fi
  printf '%s\n' "$step"
}

if [ "$WHY" = 1 ]; then
  printf 'st  tag                      directory                                       step    present\n'
  for row in "${CONTRACT[@]}"; do
    IFS='|' read -r st tag dir constant mode <<< "$row"
    step="$(resolve "$tag" "$dir" "$constant" "$mode")"
    if [ "$step" = "none" ]; then
      mark="no handoff"
    elif [ -f "$BASE/$dir/model_$(printf '%06d' "$step").pt" ]; then
      mark="yes"
    else
      mark="NO"
    fi
    printf '%-3s %-23s %-47s %-7s %s\n' "$st" "$tag" "$dir" "$step" "$mark"
  done
  echo
fi

# The first stage that is NOT complete. Stages must be complete IN ORDER:
# if stage 3 is missing but stage 4 exists, the run still has to redo
# stage 3, because stage 4's input was produced by a stage 3 that never
# finished.
first=1
for row in "${CONTRACT[@]}"; do
  IFS='|' read -r st tag dir constant mode <<< "$row"
  step="$(resolve "$tag" "$dir" "$constant" "$mode")"
  if [ "$step" != "none" ] && [ -f "$BASE/$dir/model_$(printf '%06d' "$step").pt" ]; then
    first=$((st + 1))
  else
    break
  fi
done

# All nine stages done: the run is finished, and there is no stage 10 to
# resume into.
if [ "$first" -gt 9 ]; then
  echo "all stages complete"
  exit 0
fi

echo "$first"
