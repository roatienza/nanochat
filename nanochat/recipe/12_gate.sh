#!/usr/bin/env bash
# utak — Stage 12: the deployment gates.
#
# Two gates, both run against the LIVE server (so they measure the model
# exactly as a client sees it; there is no proxy in the path).
#
#   v5x gate (eval/verify_v5x.py)  15 checks, must be 15/15
#     F1 fresh hello x5      -> greeting, no call
#     F2 fresh 2+2 x5        -> "4", no call
#     F3 fresh files x5      -> CALL bash (ls)
#     F4 fresh weather x5    -> CALL web_search
#     F5 fresh fetch x5      -> CALL fetch_content
#     R1..R5 replay checks   -> the same, but with a poisoned history replayed
#     LOOP                   -> a 4-turn OpenAI tool loop
#
#   HP gate (eval/verify_hp.py)   7 checks, must be 7/7
#     The self-poisoning loop: the harness replays the model's own bad
#     answers and the model must not copy them.
#
# The known-good result for this checkpoint is v5x 15/15 and HP 6/7 -- HP5
# ("poisoned date") fails. That failure is cosmetic and pre-existing; the
# gate reports it rather than hiding it.
#
# Usage:  ./recipe/12_gate.sh [port]
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

PORT="${1:-$PUBLIC_PORT}"
export BABY_ONIT_DIR GATE_DATA_DIR

# ---------------------------------------------------------------------------
# THE RESULTS FILES MUST BELONG TO THIS RUN.
#
# The three gate scripts APPEND to their results file -- verify_v5x.py:61,
# verify_hp.py:65 and verify_msx.py:69 all open it with mode "a" -- and
# nothing in this stage ever emptied them. The verdicts below are then read
# back with
#
#     grep -oE '[0-9]+/15' "$LOG_DIR/gate_v5x_results.txt" | tail -1
#
# which takes the last match IN THE WHOLE FILE, including matches written by
# earlier runs.
#
# That is not cosmetic. If a gate dies before reaching its own verdict line
# -- the server goes away mid-run, one check throws, a request times out --
# the file still ends with the PREVIOUS run's score, and this stage reads
# that as this run's result. On 2026-10-31 the file on disk ended in
# "15/15" from a 09:44 run that graded a DIFFERENT MODEL (v6j, not the
# checkpoint being deployed). A v5x gate that crashed after that would have
# been read as 15/15, and this stage would have printed "the model is
# deployable" for a model it never graded. Replayed to confirm: appending
# nothing but a header line to a copy of that file still yielded 15/15.
#
# So each gate starts from an empty file, and the previous one is archived
# rather than deleted -- the history is what let us date the stale 15/15 in
# the first place.
archive_gate_results() {
  local f="$1"
  [ -f "$f" ] || return 0
  mkdir -p "$LOG_DIR/gate_archive"
  mv -f "$f" "$LOG_DIR/gate_archive/$(basename "$f" .txt).$(date '+%Y%m%d-%H%M%S').txt"
}

# A gate that wrote no verdict of its own must FAIL the stage. Silence is
# not a pass: the empty string never compares equal to "15/15", and saying
# "unknown" here is what stops a crash from being read as the old score.
require_verdict() {
  local what="$1" got="$2" f="$3"
  [ -n "$got" ] && return 0
  log "$what wrote no verdict -- it did not finish. Last lines:"
  tail -12 "$f" 2>/dev/null | sed 's/^/    /'
  die "$what did not report a result -- refusing to deploy on a gate that did not run"
}

log "=== Stage 12: gates against :$PORT (model id '$MODEL_NAME') ==="

# --- the v5x gate ---------------------------------------------------------
archive_gate_results "$LOG_DIR/gate_v5x_results.txt"
export GATE_OUT="$LOG_DIR/gate_v5x_results.txt"
log "running the v5x gate (15 checks)..."
( cd "$NANOCHAT_ROOT" && "$PY" eval/verify_v5x.py "$PORT" ) \
  > "$LOG_DIR/gate_v5x.log" 2>&1
tail -25 "$LOG_DIR/gate_v5x.log"

# --- the HP gate ----------------------------------------------------------
archive_gate_results "$LOG_DIR/gate_hp_results.txt"
export GATE_OUT="$LOG_DIR/gate_hp_results.txt"
log "running the HP gate (7 checks)..."
( cd "$NANOCHAT_ROOT" && "$PY" eval/verify_hp.py "$PORT" ) \
  > "$LOG_DIR/gate_hp.log" 2>&1
tail -25 "$LOG_DIR/gate_hp.log"

# --- verdict --------------------------------------------------------------
v5x_pass=$(grep -oE '[0-9]+/15' "$LOG_DIR/gate_v5x_results.txt" 2>/dev/null | tail -1)
hp_pass=$(grep -oE '[0-9]+/7'  "$LOG_DIR/gate_hp_results.txt"  2>/dev/null | tail -1)
require_verdict "the v5x gate" "$v5x_pass" "$LOG_DIR/gate_v5x.log"
require_verdict "the HP gate"  "$hp_pass"  "$LOG_DIR/gate_hp.log"
log "RESULT: v5x=$v5x_pass  HP=$hp_pass"

if [ "$v5x_pass" = "15/15" ]; then
  log "v5x gate: PASS (15/15)"
else
  die "v5x gate FAILED (${v5x_pass:-unknown}) -- do not deploy"
fi

if [ "$hp_pass" = "7/7" ]; then
  log "HP gate: PASS (7/7)"
elif [ "$hp_pass" = "6/7" ]; then
  log "HP gate: 6/7 -- the known HP5 failure. Cosmetic and pre-existing;"
  log "          the shipped checkpoint behaves identically. Deployable."
else
  die "HP gate FAILED (${hp_pass:-unknown}) -- regression, investigate"
fi

log "Gates passed. The model is deployable."

# --- the msx gate: multi-step, long-context reasoning, constraints ---------
#
# v5x and HP grade ROUTING: single user turns, tool-name match. Neither can see
# the capability the user actually notices. msx adds chained tool calls, document
# arithmetic/joins/contradictions/state tracking, and explicit constraints.
# It is reported BEFORE it is enforced so one full ladder pass records a
# baseline on the new checks without blocking the deploy on an uncalibrated bar.
archive_gate_results "$LOG_DIR/gate_msx_results.txt"
export GATE_OUT="$LOG_DIR/gate_msx_results.txt"
log "running the msx gate (12 checks: multi-step, long-context, constraints)..."
( cd "$NANOCHAT_ROOT" && "$PY" eval/verify_msx.py "$PORT" ) \
  > "$LOG_DIR/gate_msx.log" 2>&1
tail -20 "$LOG_DIR/gate_msx.log"

msx_pass=$(grep -oE '[0-9]+/12 checks passed' "$LOG_DIR/gate_msx_results.txt" 2>/dev/null | tail -1)
msx_n="${msx_pass%%/*}"
log "msx gate: ${msx_pass:-unknown}"
if [ "${MSX_ENFORCE:-0}" = "1" ]; then
  # Enforced: a gate that wrote no verdict is a failure, not a pass. Without
  # this the empty string falls into the else branch below and reads as
  # "12 != 12 -> regression", which happens to fail safe -- but it reports a
  # regression that did not happen. Say what actually happened.
  require_verdict "the msx gate" "$msx_pass" "$LOG_DIR/gate_msx.log"
  if [ "$msx_n" = "12" ]; then
    log "msx gate: PASS (12/12)"
  else
    die "msx gate FAILED (${msx_pass:-unknown}) -- regression in multi-step/context"
  fi
else
  log "msx gate: reported, NOT enforced (MSX_ENFORCE=0)."
  log "          Set MSX_ENFORCE=1 to make 12/12 a deploy condition."
fi
