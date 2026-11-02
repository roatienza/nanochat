#!/usr/bin/env bash
# utak — Stage 13: the end-to-end acceptance test.
#
# This is the test for the original symptom ("it hangs"). It runs the REAL
# baby-onit agent loop -- its own code, not a mock -- against the served
# endpoint, in a throwaway data dir, and reports per-turn latency and
# whether any answer is degenerate.
#
# It then replays the poisoned-history scenario, which is the case that
# used to produce 4,096 tokens of literal "\n" and a 124 s stall.
#
# Usage:  ./recipe/13_e2e.sh [port]
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

PORT="${1:-$PUBLIC_PORT}"
export BABY_ONIT_DIR PORT

log "=== Stage 13: end-to-end acceptance test on :$PORT ==="

( cd "$NANOCHAT_ROOT" && "$PY" eval/e2e_harness.py ) \
  > "$LOG_DIR/e2e_harness.log" 2>&1
rc=$?

cat "$LOG_DIR/e2e_harness.log"

if [ $rc -ne 0 ]; then
  die "e2e acceptance test failed (exit $rc)"
fi

# The acceptance criteria, straight from the original bug report:
#   - no turn may take more than a few seconds
#   - no answer may contain a literal backslash-n run
#   - the tool loop must reach a final answer
# NOTE: this used to grep for the word "degenerate" anywhere in the log. The
# harness ALWAYS prints a summary line "degenerate turns: N/8" (and "PASS" or
# "FAIL" right after it), so that grep matched on a perfectly clean run and
# stage 13 died with "found a degenerate answer" while the log said
# "degenerate turns: 0/8". The harness already exits non-zero on a real
# failure (checked above); test the summary line, not the word.
deg_summary=$(grep -oE 'degenerate turns: [0-9]+/[0-9]+' "$LOG_DIR/e2e_harness.log" | tail -1)
if [ -n "$deg_summary" ]; then
  deg_bad=${deg_summary#degenerate turns: }
  deg_bad=${deg_bad%%/*}
  if [ "$deg_bad" != "0" ]; then
    die "e2e found $deg_summary -- see $LOG_DIR/e2e_harness.log"
  fi
elif grep -qi 'DEGENERATE' "$LOG_DIR/e2e_harness.log"; then
  die "e2e found a degenerate answer -- see $LOG_DIR/e2e_harness.log"
fi

log "e2e acceptance test PASSED."
log "Next: ./recipe/14_regression.sh  (the offline unit tests)"
