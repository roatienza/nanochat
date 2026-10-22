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
# exact step the next stage reads. Those steps are the stage contract, and
# they live in tests/test_stage_contract.py and in config.sh.
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

# stage : checkpoint directory : the step the NEXT stage reads.
#
# The tags and the steps both come from config.sh, so this table follows the
# recipe if either is renamed. The steps are the contract; see
# tests/test_stage_contract.py.
CONTRACT=(
  "1|base_checkpoints/$TAG_BASE|$TAG_BASE_FINAL_STEP"
  "2|chatsft_checkpoints/$TAG_SFT_TOOLS|$TAG_SFT_TOOLS_FINAL_STEP"
  "3|chatsft_checkpoints/$TAG_WARMUP|$TAG_WARMUP_FINAL_STEP"
  "4|chatrl_checkpoints/$TAG_RL_WARMUP|$TAG_RL_WARMUP_FINAL_STEP"
  "5|chatrl_checkpoints/$TAG_RL_PILOT|$TAG_RL_PILOT_FINAL_STEP"
  "6|chatrl_checkpoints/$TAG_RL|$TAG_RL_FINAL_STEP"
  "7|base_checkpoints/$TAG_LONGC_FINAL|$TAG_LONGC_FINAL_STEP"
  "8|chatsft_checkpoints/$TAG_SFT|$TAG_SFT_FINAL_STEP"
  "9|chatsft_checkpoints/$FINAL_TAG|$FINAL_STEP"
)

if [ "$WHY" = 1 ]; then
  printf 'st  directory                                       final step  present\n'
  for row in "${CONTRACT[@]}"; do
    st="${row%%|*}"; rest="${row#*|}"
    dir="${rest%%|*}"; step="${rest#*|}"
    if [ -f "$BASE/$dir/model_$(printf '%06d' "$step").pt" ]; then
      mark="yes"
    else
      mark="NO"
    fi
    printf '%-3s %-47s %-11s %s\n' "$st" "$dir" "$step" "$mark"
  done
  echo
fi

# The first stage that is NOT complete. Stages must be complete IN ORDER:
# if stage 3 is missing but stage 4 exists, the run still has to redo
# stage 3, because stage 4's input was produced by a stage 3 that never
# finished.
first=1
for row in "${CONTRACT[@]}"; do
  st="${row%%|*}"; rest="${row#*|}"
  dir="${rest%%|*}"; step="${rest#*|}"
  if [ -f "$BASE/$dir/model_$(printf '%06d' "$step").pt" ]; then
    first=$((st + 1))
  else
    break
  fi
done

# All nine training stages are done: the rest (pack, serve, gates, tests)
# is cheap and must run.
if [ "$first" -gt 9 ]; then
  first=10
fi

echo "$first"
