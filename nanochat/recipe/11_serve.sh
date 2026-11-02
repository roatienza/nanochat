#!/usr/bin/env bash
# utak — Stage 11: serve.
#
# Topology:
#
#   client (baby-onit, OpenAI-compatible)
#      |
#      v
#   :8000  vLLM              <- the model itself, served directly
#
# There is NO proxy in front of the model. Until 2026-11-01 there was a
# guard_proxy.py on :8000 forwarding to vLLM on :8010. It was removed on
# purpose: it truncated every answer at 2000 characters (MAX_ANSWER_CHARS)
# and still reported finish_reason="stop", so a cut answer arrived at the
# client looking complete. The model answers in full now, and max_tokens is
# whatever the client asks for.
#
# serving/guard_proxy.py is still on disk -- the stage-14 unit tests import
# its pure functions (is_degenerate, first_degeneracy_pos, _clean_tool_slots)
# -- but nothing in the serving path runs it.
#
# Usage:  ./recipe/11_serve.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 11: serve $MODEL_NAME ==="
log "    serving directly on :$PUBLIC_PORT (no proxy)"

[ -d "$MODEL_DIR" ] || die "model dir not found: $MODEL_DIR (run 10_pack.sh first)"
[ -x "$VLLM_BIN" ] || die "vllm binary not found: $VLLM_BIN"

# ---------------------------------------------------------------------------
# Pick the card to serve on.
#
# vLLM takes 94% of whatever card it is given, so it cannot share with a
# training process. By the time this stage runs, the training has finished
# and its cards are free -- but the OTHER tenant on this node may have
# taken one of them in the meantime.
#
# So: use GPU_SERVE if it is free, otherwise fall back to the first card
# with no compute process on it. Fail loudly if there is none.
# ---------------------------------------------------------------------------
if gpu_is_free "$GPU_SERVE"; then
  log "    serving on GPU $GPU_SERVE (free)"
else
  for g in 0 1 2 3 4 5 6 7; do
    if gpu_is_free "$g"; then
      log "    GPU $GPU_SERVE is busy -- serving on GPU $g instead (free)"
      GPU_SERVE="$g"
      break
    fi
  done
  if ! gpu_is_free "$GPU_SERVE"; then
    die "no free GPU to serve on: every card has a process on it"
  fi
fi

# Free the port if a previous instance is holding it.
for port in $PUBLIC_PORT $UPSTREAM_PORT; do
  pids=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
  for p in $pids; do kill "$p" 2>/dev/null || true; done
done
sleep 2

cleanup() {
  [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES="$GPU_SERVE" "$VLLM_BIN" serve \
  "$MODEL_DIR" \
  --port "$PUBLIC_PORT" --trust-remote-code --enforce-eager \
  --gpu-memory-utilization 0.94 \
  --max-model-len 131072 --max-num-batched-tokens 131072 --max-num-seqs 8 \
  --enable-prefix-caching --no-enable-chunked-prefill \
  --served-model-name "$MODEL_NAME" \
  --enable-auto-tool-choice --tool-call-parser nanochat \
  --tool-parser-plugin "$NANOCHAT_ROOT/serving/nanochat_tool_parser.py" \
  > "$LOG_DIR/vllm_serve_${PUBLIC_PORT}.log" 2>&1 &
VLLM_PID=$!

log "waiting for vLLM on :$PUBLIC_PORT ..."
up=0
for i in $(seq 1 150); do
  curl -s -m 2 "http://127.0.0.1:$PUBLIC_PORT/v1/models" >/dev/null 2>&1 && { up=1; break; }
  sleep 2
done
if [ "$up" != 1 ]; then
  log "vLLM failed to start:"
  tail -25 "$LOG_DIR/vllm_serve_${PUBLIC_PORT}.log"
  exit 1
fi
log "vLLM is up (pid $VLLM_PID)"

log "serving model id '$MODEL_NAME' on :$PUBLIC_PORT (no proxy)"
log "Next: ./recipe/12_gate.sh"

wait $VLLM_PID
echo "EXIT=$?"
