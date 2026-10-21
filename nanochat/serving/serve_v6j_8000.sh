#!/usr/bin/env bash
# v6j: serve on port 8000 (public) with template v7 + the nanochat tool parser.
#
# v6j = the literal-backslash-n training-data fix: v5x gate 15/15 (v6h was
# 14/15, F1 fresh-hello now passes), HP gate 6/7 (HP5 still fails, same as
# v6h). Final val bpb 0.393419.
#
# TOPOLOGY
#   :8000  guard_proxy.py     public endpoint, owns the safety policy
#   :8010  vLLM              the actual model, internal
# The guard is the LOAD-BEARING fix for the '\n' spam: it scrubs degenerate
# assistant turns out of the replayed history, which removes the in-context
# attractor (proven: raw v6j + poisoned history = 15.7s of garbage; + scrub
# = 0.6s clean; via the guard = 0.6s clean).
#
# Restart-safe: kills stale listeners on 8000/8010 first, respawns the
# guard if it dies, and traps EXIT so vLLM never outlives this script.
#
# Usage:
#   MODEL_DIR=/path/to/nanochat-d28b-rl100-128k-v6j \
#   VLLM_BIN=/path/to/venv/bin/vllm \
#   ./serve_v6j_8000.sh
set -u

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

MODEL_DIR="${MODEL_DIR:-/data/rowel/nanochat_store/v2/vllm_d28b/nanochat-d28b-rl100-128k-v6j}"
VLLM_BIN="${VLLM_BIN:-/data/rowel/nanochat_store/v2/vllm_d28b/venv-vllm/bin/vllm}"
PY="${PY:-/data/rowel/nanochat_store/v2/vllm_d28b/venv-vllm/bin/python}"
LOGDIR="${LOGDIR:-$HERE}"
GPU="${GPU:-0}"
PUBLIC_PORT="${PUBLIC_PORT:-8000}"
UPSTREAM_PORT="${UPSTREAM_PORT:-8010}"
SERVED_NAME="${SERVED_NAME:-nanochat-d28b-rl100}"
MODEL_ID="$(basename "$MODEL_DIR")"

cd "$HERE"

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

VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=$GPU "$VLLM_BIN" serve \
  "$MODEL_DIR" \
  --port $UPSTREAM_PORT --trust-remote-code --enforce-eager --gpu-memory-utilization 0.94 \
  --max-model-len 131072 --max-num-batched-tokens 131072 --max-num-seqs 8 \
  --enable-prefix-caching --no-enable-chunked-prefill \
  --served-model-name "$SERVED_NAME" \
  --enable-auto-tool-choice --tool-call-parser nanochat \
  --tool-parser-plugin "$HERE/nanochat_tool_parser.py" \
  > "$LOGDIR/vllm_serve_${UPSTREAM_PORT}_128k_v6j.log" 2>&1 &
VLLM_PID=$!

up=0
for i in $(seq 1 150); do
  curl -s -m 2 "http://127.0.0.1:$UPSTREAM_PORT/v1/models" >/dev/null 2>&1 && { up=1; break; }
  sleep 2
done
if [ "$up" != 1 ]; then
  echo "vllm failed to come up on $UPSTREAM_PORT"
  tail -20 "$LOGDIR/vllm_serve_${UPSTREAM_PORT}_128k_v6j.log"; exit 1
fi

while true; do
  GUARD_UPSTREAM="http://127.0.0.1:$UPSTREAM_PORT" GUARD_PORT=$PUBLIC_PORT \
    "$PY" "$HERE/guard_proxy.py" \
    >> "$LOGDIR/guard_proxy_${PUBLIC_PORT}.log" 2>&1
  echo "$(date '+%F %T') guard exited ($?), respawning" >> "$LOGDIR/guard_proxy_${PUBLIC_PORT}.log"
  sleep 2
done &
GUARD_PID=$!
echo "v6j: vllm pid=$VLLM_PID ($UPSTREAM_PORT), guard supervisor pid=$GUARD_PID ($PUBLIC_PORT)"
wait $VLLM_PID
echo EXIT=$?
