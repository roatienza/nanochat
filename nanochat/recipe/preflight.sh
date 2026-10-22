#!/usr/bin/env bash
# utak — pre-flight checks for the stages that have not run yet.
#
# WHY THIS EXISTS
# ---------------
# The pipeline takes ~4.4 days. The pretrain is the first stage; the
# other fifteen have not run since the recipe changed. A break in any of
# them surfaces DAYS from now, when the expensive part is already done
# and there is no time to react.
#
# This script checks, without a GPU and without disturbing the run:
#
#   1. THE STEP CONTRACT. Every stage reads the previous stage's output
#      at a hardcoded step (stage 3 reads stage 2's output at step 13630,
#      stage 7 reads stage 6's at 177, ...). The steps are a property of
#      the RECIPE, not of the weights, so they can be verified against the
#      checkpoints the ORIGINAL run produced -- today, not in three days.
#
#   2. STAGE 10 (pack). It needs no GPU, so it can be run for real: the
#      original run's final checkpoint is exported to a scratch dir and
#      the result is verified (parameter count, key shapes, the config,
#      and the aux files vLLM needs).
#
#   3. THE OFFLINE TESTS that stage 14 runs, so a failure there is not
#      discovered at the end of the run.
#
# Usage:
#   ./recipe/preflight.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

FAILED=0
note() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf '[%s] FAIL: %s\n' "$(date '+%F %T')" "$*"; FAILED=$((FAILED+1)); }

# ---------------------------------------------------------------------------
# 1. The step contract
# ---------------------------------------------------------------------------
note "=== 1. the step contract ==="
if "$PY" "$NANOCHAT_ROOT/tests/test_stage_contract.py"; then
  note "  the contract holds"
else
  fail "the step contract is broken"
fi

# ---------------------------------------------------------------------------
# 2. Stage 10 (pack), for real, into a scratch dir
# ---------------------------------------------------------------------------
note "=== 2. stage 10 (pack) ==="
SCRATCH="${PREFLIGHT_DIR:-/tmp/utak-preflight}"
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH/model"
# The original run's final checkpoint. The utak run uses the same recipe,
# so the pack path is the same.
PACK_TAG="${PREFLIGHT_PACK_TAG:-d28b-128k-sft-v6j}"
PACK_STEP="${PREFLIGHT_PACK_STEP:-240}"
if ( cd "$NANOCHAT_ROOT/serving" \
     && NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" \
        "$PY" export_hf_128k.py --phase sft --base-dir "$NANOCHAT_BASE_DIR" \
          --tag "$PACK_TAG" --step "$PACK_STEP" --out "$SCRATCH/model" \
     ) > "$SCRATCH/pack.log" 2>&1; then
  note "  the export worked"
  if ( cd "$NANOCHAT_ROOT/serving" \
       && "$PY" patch_config_131072.py "$SCRATCH/model/config.json" \
     ) >> "$SCRATCH/pack.log" 2>&1; then
    note "  the config patch worked"
  else
    fail "the config patch failed"
  fi
  for f in tokenizer.json tokenizer_config.json configuration_nanochat.py \
           modeling_nanochat_d28.py generation_config.json; do
    cp -p "$NANOCHAT_ROOT/serving/model_assets/$f" "$SCRATCH/model/$f" \
      || fail "missing aux file: $f"
  done
  cp -p "$NANOCHAT_ROOT/chat_template_v7.jinja" "$SCRATCH/model/chat_template.jinja"
  if "$PY" "$NANOCHAT_ROOT/tests/test_packed_model.py" --dir "$SCRATCH/model"; then
    note "  the packed model dir verifies"
  else
    fail "the packed model dir does not verify"
  fi
  rm -rf "$SCRATCH/model"
else
  fail "the export failed"
  tail -20 "$SCRATCH/pack.log" | sed 's/^/    /'
fi

# ---------------------------------------------------------------------------
# 3. The offline tests that stage 14 runs
# ---------------------------------------------------------------------------
note "=== 3. the offline regression tests ==="
# These need no GPU (they are excluded when there is none).
if CUDA_VISIBLE_DEVICES="" "$PY" -m pytest tests/ -q > "$SCRATCH/pytest.log" 2>&1; then
  note "  the test suite passes ($(tail -1 "$SCRATCH/pytest.log"))"
else
  fail "the test suite fails"
  tail -20 "$SCRATCH/pytest.log" | sed 's/^/    /'
fi

if [ -x "$VLLM_PY" ]; then
  for t in eval/test_detector.py eval/test_segment_detector.py eval/test_guard_regression.py; do
    if "$VLLM_PY" "$NANOCHAT_ROOT/$t" > "$SCRATCH/$(basename "$t").log" 2>&1; then
      note "  $t passes"
    else
      fail "$t fails"
    fi
  done
else
  note "  (no vLLM interpreter at $VLLM_PY; skipping the guard tests)"
fi

rm -rf "$SCRATCH"

note ""
if [ "$FAILED" -eq 0 ]; then
  note "PRE-FLIGHT PASSED. The stages that have not run yet should work."
  exit 0
else
  note "PRE-FLIGHT FAILED ($FAILED problem(s)). Fix them before the run reaches"
  note "those stages."
  exit 1
fi
