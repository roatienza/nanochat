#!/usr/bin/env bash
# utak — Stage 15: the live-server hypothesis test.
#
# eval/test_scrub.py proves the specific claim behind the guard's history
# scrub: that SCRUBBING THE REQUEST HISTORY -- not merely aborting the
# output -- is what makes the model answer cleanly and fast.
#
# It is a hypothesis test, not a unit test: it needs a running server, and
# it talks to the RAW vLLM endpoint (:8010) rather than the guard, so that
# the guard's own scrub is not what makes the difference.
#
# The three conditions it compares, on identical prompts:
#   raw model + poisoned history  -> slow, degenerate
#   raw model + scrubbed history  -> fast, clean
#
# Usage:  ./recipe/15_scrub_test.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

export BABY_ONIT_DIR GATE_DATA_DIR
export UTAK_MODEL_NAME="$MODEL_NAME"
export UTAK_RAW_URL="${UTAK_RAW_URL:-http://127.0.0.1:$UPSTREAM_PORT}"

log "=== Stage 15: the live-server scrub hypothesis test ==="
log "    raw endpoint: $UTAK_RAW_URL"

# The raw vLLM port must be up for this test to mean anything.
curl -s -m 3 "$UTAK_RAW_URL/v1/models" >/dev/null 2>&1 \
  || die "the raw vLLM endpoint is not answering on $UTAK_RAW_URL"

( cd "$NANOCHAT_ROOT" && "$PY" eval/test_scrub.py ) \
  > "$LOG_DIR/test_scrub.log" 2>&1
rc=$?

cat "$LOG_DIR/test_scrub.log"

[ $rc -eq 0 ] || die "the scrub hypothesis test failed (exit $rc)"

log "The scrub hypothesis test PASSED."
