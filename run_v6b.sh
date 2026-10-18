#!/usr/bin/env bash
# v6b: v6a + diversified P7 anchors. v6a final gate 7/15 with the replay
# routing FIXED (R2 3/3) but P7 re-flipped 10/10: the anchor rows were all
# harness8+system while the failing probe is classic-tools+no-system.
# Anchors now: 50% harness8+sys, 25% classic+no-sys, 25% classic+sys, with
# the CALL matching the toolset (get_weather for classic).
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=4
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v6b-harnessbasics \
  --source-phase sft --model-tag d28b-128k-sft-v5e --model-step 60 \
  --output-tag d28b-128k-sft-v6b \
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