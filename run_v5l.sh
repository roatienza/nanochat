#!/usr/bin/env bash
# v5l: multi_tool re-anchor. v5j dist (C35/C'25/D30) + NEW E doc-call rows (10%).
# Fixes v5j's multi_tool 24k regression (search_web instead of get_weather)
# WITHOUT result->CALL rows (flips P1) or longctx x3 (re-flips P7).
# From v5e ckpt 60. GPU 3 (shares with an idle 13.7GB process -> dbs 2).
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=3
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v5l-mtanchor \
  --source-phase sft --model-tag d28b-128k-sft-v5e --model-step 60 \
  --output-tag d28b-128k-sft-v5l \
  --max-seq-len 8192 --short-window 512 \
  --device-batch-size 2 --total-batch-size 262144 \
  --num-iterations 60 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 \
  --glaive-mr-epochs 2 --toolstop-epochs 4 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 20 --eval-tokens 524288 --save-every 30 --ce-chunk 4096
