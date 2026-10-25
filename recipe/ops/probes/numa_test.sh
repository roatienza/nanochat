#!/usr/bin/env bash
# Test the NUMA/PCIe hypothesis for the 2-GPU slowdown.
#
# THE FACTS
# ----------
#   Oct 22 reference, 2 GPUs (0,2), dbs=2, grad_accum=4 :   965 ms/step
#   NOW, 2 GPUs (0,2), dbs=2, grad_accum=4, RESUMED      : 1,200 ms/step  (1.24x)
#   Oct 22 reference, 1 GPU, dbs=2                        : 1,870 ms/step
#   NOW, 1 GPU (1), dbs=2, FRESH                          : 1,854 ms/step  (1.00x)
#
# So ONE GPU is exactly at the reference speed, and TWO GPUs are 24%
# slow. 1 GPU has no reduce_scatter / all_gather at all. 2 GPUs do
# 16 large-param reduce_scatter + 16 all_gather per step, over PCIe
# (GPU0<->GPU2 is NODE, not NV#).
#
# GPUs 0,1,2,3 sit behind host bridge 0 (NUMA node 0, cores 0-63).
# Another tenant's jobs are on GPUs 1 and 3, so they push their own
# traffic through the same bridge, and they also eat the node-0 CPUs
# that feed it.
#
# THE TEST
# --------
# Run the SAME 2-GPU config twice:
#   (a) unpinned            -- the default, everything on all 128 cores
#   (b) pinned to node 0    -- taskset -c 0-63, next to the GPUs
#
# If (b) is faster, the cause is CPU/PCIe locality, and the fix is to
# pin the training ranks to the NUMA node that owns their GPUs.
set -u
NANOCHAT=/data/rowel/reasoner-model/nanochat
VENV=/data/rowel/nanochat/.venv/bin/python
CARDS="${1:-0,2}"
ITERS="${2:-12}"

COMMON="--depth 28 --aspect-ratio 64 --head-dim 128 --max-seq-len 2048 \
--window-pattern SSSL --device-batch-size 2 --total-batch-size 32768 \
--num-iterations $ITERS --eval-every -1 --core-metric-every -1 --save-every 0 \
--model-tag numaprobe"

steady() {
  tr '\r' '\n' < "$1" | grep -oE 'dt: [0-9.]+ms' | tail -6 \
    | sed 's/dt: //;s/ms//' \
    | awk '{s+=$1; n++} END{if(n) printf "%.0f ms/step over %d steps\n", s/n, n}'
}

run() {
  local label="$1"; shift
  echo "### $label"
  CUDA_VISIBLE_DEVICES="$CARDS" NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2 \
  PYTHONPATH="$NANOCHAT" OMP_NUM_THREADS=1 \
  timeout 600 "$@" -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m scripts.base_train -- --run "numa-$label" $COMMON > "/tmp/numa_$label.log" 2>&1
  echo "   rc=$?"
  tr '\r' '\n' < "/tmp/numa_$label.log" | grep -oE '^step [0-9]+/.*dt: [0-9.]+ms' | tail -2 | sed 's/^/     /'
  echo -n "   STEADY: "; steady "/tmp/numa_$label.log"
}

run unpinned "$VENV" -u
run pinned-node0 taskset -c 0-63 "$VENV" -u
run pinned-node1 taskset -c 64-127 "$VENV" -u
