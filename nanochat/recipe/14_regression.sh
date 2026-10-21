#!/usr/bin/env bash
# utak — Stage 14: the offline regression tests.
#
# These need no GPU and no running server. They are the fast check that
# the code itself is healthy, and they include the regression test for the
# literal-backslash-n training-data bug that caused the original hang.
#
# NOTE ON HOW THEY RUN: every one of these is a STANDALONE SCRIPT that
# reports through its exit code -- they are not pytest modules. They are
# invoked directly, each with the interpreter it needs:
#
#   tasks/test_no_literal_newline.py  the training interpreter
#   eval/test_detector.py             the vLLM interpreter (needs httpx)
#   eval/test_segment_detector.py     the vLLM interpreter
#   eval/test_guard_regression.py      the vLLM interpreter
#
# The guard tests import serving/guard_proxy.py, which imports httpx. httpx
# lives in the vLLM virtualenv, not the training one.
#
# eval/test_scrub.py is NOT here. It is a live-server hypothesis test and
# runs in stage 15.
#
# Usage:  ./recipe/14_regression.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 14: offline regression tests ==="

fail=0

run_check() {
  local name="$1"; shift
  local interp="$1"; shift
  log "  $name"
  ( cd "$NANOCHAT_ROOT" && "$interp" "$@" ) \
    > "$LOG_DIR/$name.log" 2>&1 \
    || { fail=1; log "    FAILED"; tail -15 "$LOG_DIR/$name.log"; return 1; }
  tail -2 "$LOG_DIR/$name.log"
}

# --- the v6j data bug: no literal backslash-n in any training target -----
run_check test_no_literal_newline "$PY" tasks/test_no_literal_newline.py

# --- the guard: detector, segment detector, regression -------------------
run_check test_detector          "$VLLM_PY" eval/test_detector.py
run_check test_segment_detector  "$VLLM_PY" eval/test_segment_detector.py
run_check test_guard_regression  "$VLLM_PY" eval/test_guard_regression.py

# --- the library: engine, optim, tokenizer, execution, tasks -------------
# These ARE pytest modules, so they are run with pytest.
log "  test_library (pytest)"
( cd "$NANOCHAT_ROOT" && "$PY" -m pytest tests/ -q ) \
  > "$LOG_DIR/test_library.log" 2>&1 \
  || { fail=1; log "    FAILED"; tail -25 "$LOG_DIR/test_library.log"; }
tail -3 "$LOG_DIR/test_library.log"

[ $fail -eq 0 ] || die "regression tests FAILED"

log "All offline regression tests passed."
log "Next: ./recipe/15_scrub_test.sh  (the live-server hypothesis test)"
