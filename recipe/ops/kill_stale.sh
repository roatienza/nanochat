#!/usr/bin/env bash
# Kill any lingering nanochat training/serving processes that hold a GPU.
#
# WHY THIS IS NEEDED
# ------------------
# When a nanochat stage hits a CUDA OOM inside a multi-threaded torch op
# (e.g. torch.load), the process can be left spinning with its CUDA
# context intact: state R, ~89 threads, parent dead. It keeps its ~35 GiB
# of GPU memory, and the NEXT run then OOMs at model load with
# "Process N has 34.87 GiB memory in use" -- which looks like a memory
# bug in the new code but is actually a corpse from the previous attempt.
#
# This kills anything of ours that holds a CUDA context and belongs to
# the nanochat tree, then waits for the cards to drain.
#
# Usage:
#   ./kill_stale.sh          # kill and report
#   ./kill_stale.sh --quiet  # kill, no report
set -u
QUIET=0
[ "${1:-}" = "--quiet" ] && QUIET=1

KILLED=0
for round in 1 2 3; do
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ "$pid" = "$$" ] && continue
    cmd=$(tr "\0" " " < "$p/cmdline" 2>/dev/null) 2>/dev/null || continue
    case "$cmd" in
      *scripts.chat_rl*|*scripts.base_train*|*scripts.chat_sft*|*scripts.base_train_long*|*scripts.chat_sft_long*|*scripts.chat_rl_agentic*|*scripts.chat_rl_curriculum*|*torch.distributed.run*)
        # Only if it actually holds GPU memory.
        used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
               | awk -F', *' -v p="$pid" '$1==p {print $2}')
        [ -z "$used" ] && continue
        [ "$used" -lt 500 ] && continue
        [ "$QUIET" = 1 ] || echo "  killing $pid (${used} MiB) : $(echo "$cmd" | cut -c1-90)"
        kill -9 "$pid" 2>/dev/null && KILLED=$((KILLED + 1))
        ;;
    esac
  done
  sleep 5
done

# Wait for the cards to drain.
for i in 1 2 3 4 5 6; do
  busy=0
  for g in 0 1 2 3 4 5 6 7; do
    u=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ "${u:-0}" -gt 1000 ] && busy=$((busy + 1))
  done
  [ "$busy" = 0 ] && break
  sleep 5
done

if [ "$QUIET" = 0 ]; then
  echo "  killed $KILLED process(es)"
  for g in 0 1 2 3 4 5 6 7; do
    u=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    t=$(nvidia-smi -i "$g" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
    echo "  GPU $g: $((t - u)) MiB free"
  done
fi
