#!/usr/bin/env bash
# utak — Stage 11: serve.
#
# Topology (the inversion matters):
#
#   client (baby-onit, OpenAI-compatible)
#      |
#      v
#   :8000  guard_proxy.py     <- the PUBLIC endpoint; owns the safety policy
#      |      clamp max_tokens, scrub poisoned history, always stream,
#      |      abort degenerate output, 45 s wall-clock cap
#      v
#   :8010  vLLM              <- the model itself
#
# vLLM has no server-side completion cap that a client cannot override, so
# the cap has to live in front of it. The public port must therefore belong
# to the proxy, not to vLLM.
#
# The guard is not optional. It is the load-bearing half of the fix for the
# "\n" degeneration: the retrain removes the source, the guard's history
# scrub stops an already-poisoned history from re-infecting the model.
# Measured on identical prompts:
#   raw model + poisoned history -> 15.7 s of garbage
#   raw model + scrubbed history -> 0.6 s, clean
#   via the guard + poisoned history -> 0.6 s, clean
#
# Usage:  ./recipe/11_serve.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 11: serve $MODEL_NAME ==="
log "    public :$PUBLIC_PORT (guard)  ->  internal :$UPSTREAM_PORT (vLLM)"

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

# Free the ports if a previous instance is holding them.
for port in $PUBLIC_PORT $UPSTREAM_PORT; do
  pids=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
  for p in $pids; do kill "$p" 2>/dev/null || true; done
done
sleep 2

cleanup() {
  [ -n "${GUARD_PID:-}" ] && kill "$GUARD_PID" 2>/dev/null
  [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES="$GPU_SERVE" "$VLLM_BIN" serve \
  "$MODEL_DIR" \
  --port "$UPSTREAM_PORT" --trust-remote-code --enforce-eager \
  --gpu-memory-utilization 0.94 \
  --max-model-len 131072 --max-num-batched-tokens 131072 --max-num-seqs 8 \
  --enable-prefix-caching --no-enable-chunked-prefill \
  --served-model-name "$MODEL_NAME" \
  --enable-auto-tool-choice --tool-call-parser nanochat \
  --tool-parser-plugin "$NANOCHAT_ROOT/serving/nanochat_tool_parser.py" \
  > "$LOG_DIR/vllm_serve_${UPSTREAM_PORT}.log" 2>&1 &
VLLM_PID=$!

log "waiting for vLLM on :$UPSTREAM_PORT ..."
up=0
for i in $(seq 1 150); do
  curl -s -m 2 "http://127.0.0.1:$UPSTREAM_PORT/v1/models" >/dev/null 2>&1 && { up=1; break; }
  sleep 2
done
if [ "$up" != 1 ]; then
  log "vLLM failed to start:"
  tail -25 "$LOG_DIR/vllm_serve_${UPSTREAM_PORT}.log"
  exit 1
fi
log "vLLM is up (pid $VLLM_PID)"

# The guard is the public endpoint. Respawn it if it dies.
(
  while true; do
    UTAK_MODEL_NAME="$MODEL_NAME" \
    GUARD_UPSTREAM="http://127.0.0.1:$UPSTREAM_PORT" \
    GUARD_PORT="$PUBLIC_PORT" \
    GUARD_MAX_TOKENS="${GUARD_MAX_TOKENS:-4096}" \
    GUARD_MAX_SECONDS="${GUARD_MAX_SECONDS:-45}" \
      "$VLLM_PY" "$NANOCHAT_ROOT/serving/guard_proxy.py" \
      >> "$LOG_DIR/guard_proxy_${PUBLIC_PORT}.log" 2>&1
    echo "$(date '+%F %T') guard exited ($?), respawning" >> "$LOG_DIR/guard_proxy_${PUBLIC_PORT}.log"
    sleep 2
  done
) &
GUARD_PID=$!

log "guard is up (pid $GUARD_PID) on :$PUBLIC_PORT, serving model id '$MODEL_NAME'"
log "Next: ./recipe/12_gate.sh"

wait $VLLM_PID
echo "EXIT=$?"
