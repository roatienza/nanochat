#!/usr/bin/env bash
# CLEAN 2-GPU A/B at depth 20 (fits on a contended card), fresh vs resumed.
#
# The one comparison I have not managed cleanly: is the 2-GPU slowdown
# (1,200 ms now vs 965 ms on Oct 22) caused by the RESUME, or by the
# node (another tenant's jobs on the neighbouring GPUs, which share
# host bridge 0 and the node-0 CPUs)?
#
# A depth-20 model is used so both probes fit on cards that another
# tenant is also using. The RATIO is what matters, not the absolute.
set -u
NANOCHAT=/data/rowel/reasoner-model/nanochat
VENV=/data/rowel/nanochat/.venv/bin/python
CARDS="${1:-5,7}"
DEPTH="${2:-20}"
TAG="ab20"

COMMON="--depth $DEPTH --aspect-ratio 64 --head-dim 128 --max-seq-len 2048 \
--window-pattern SSSL --device-batch-size 2 --total-batch-size 32768 \
--eval-every -1 --core-metric-every -1"

run() {
  local name="$1"; shift
  CUDA_VISIBLE_DEVICES="$CARDS" NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2 \
  PYTHONPATH="$NANOCHAT" OMP_NUM_THREADS=1 \
  timeout 900 "$VENV" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m scripts.base_train -- \
    --run "ab20-$name" --num-iterations "$1" $COMMON --model-tag "$TAG" "${@:2}" \
    > "/tmp/ab20_$name.log" 2>&1
  echo "   rc=$?"
}

steady() {
  tr '\r' '\n' < "/tmp/ab20_$1.log" | grep -oE 'dt: [0-9.]+ms' | tail -6 \
    | sed 's/dt: //;s/ms//' \
    | awk '{s+=$1; n++} END{if(n) printf "   STEADY: %.0f ms/step over %d steps\n", s/n, n}'
}

echo "### A: 2 GPUs ($CARDS) depth $DEPTH FRESH, 12 steps, saving a 2-rank checkpoint"
run fresh 12 --save-every 12
tr '\r' '\n' < /tmp/ab20_fresh.log | grep -oE '^step [0-9]+/.*dt: [0-9.]+ms' | tail -2 | sed 's/^/     /'
steady fresh

echo
echo "### B: 2 GPUs ($CARDS) depth $DEPTH RESUMED from it, 20 steps"
run resumed 20 --resume-from-step 12 --save-every 0
tr '\r' '\n' < /tmp/ab20_resumed.log | grep -oE '^step [0-9]+/.*dt: [0-9.]+ms' | tail -2 | sed 's/^/     /'
steady resumed
