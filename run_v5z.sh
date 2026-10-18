#!/usr/bin/env bash
# v5z: round 3 — LONGER run (240 iters). The 60-step touch-ups (v5x 5/15,
# v5y 2/15) thrash the call boundary because the mixture is ~107k rows and a
# 60-step run sees only ~15% of one epoch — every run re-rolls the boundary.
# 240 steps lets the new routing/no-call data actually be learned while the
# LR schedule (warmdown to 0) re-settles the old boundaries.
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=4
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v5z-harnessbasics \
  --source-phase sft --model-tag d28b-128k-sft-v5e --model-step 60 \
  --output-tag d28b-128k-sft-v5z \
  --max-seq-len 32768 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 240 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 \
  --glaive-mr-epochs 2 --toolstop-epochs 6 \
  --harness-epochs 2 --harness-basics-epochs 3 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 40 --eval-tokens 524288 --save-every 60 --ce-chunk 4096