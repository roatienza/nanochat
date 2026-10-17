#!/usr/bin/env bash
# v5q: ROOT-CAUSE fix for the multi_tool 24k regression. Every touch-up since
# v5f trained at max-seq-len 8192, which TRUNCATES all level-1 (24k) longctx
# rows (target cut off -> skipped) — the touch-ups never saw a 24k doc->CALL
# row and the behavior drifted to the search_web attractor. v5e trained at
# 131072 and saw them in full (hence v5e passed 24k).
# Fix: same v5j recipe at max-seq-len 24576 (fits 24k rows), dbs 1, GPU 0.
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=0
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v5q-longctx24k \
  --source-phase sft --model-tag d28b-128k-sft-v5e --model-step 60 \
  --output-tag d28b-128k-sft-v5q \
  --max-seq-len 24576 --short-window 512 \
  --device-batch-size 1 --total-batch-size 245760 \
  --num-iterations 60 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 \
  --glaive-mr-epochs 2 --toolstop-epochs 4 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 20 --eval-tokens 524288 --save-every 30 --ce-chunk 4096
