#!/usr/bin/env bash
# v6h: the spurious-second-call fix. v6g gates: v5x 12/15, HP 6/7 — P1/P13/
# LOOP-T2 still fail with a hard "get_weather Toronto" re-call attractor.
# Census traced it to the harness_basics P7-anchor rows (402 x3 epochs): each
# teaches [get_weather result -> ANOTHER get_weather call for city2] as a
# SUPERVISED target (the spurious round the final prose must ignore). The P1
# probe is exactly the anchor's mid-row state, so the model completes it with
# the anchor's second call. v6h: the anchor's round-2 call rides as CONTEXT
# ONLY (no_train) — visible in history (P7 gate needs that) but never a target.
# Everything else identical to v6g (desc toolsets, B3b loop rows, header dates).
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=4
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v6h-anchorstop \
  --source-phase sft --model-tag d28b-128k-sft-v6a --model-step 240 \
  --output-tag d28b-128k-sft-v6h \
  --max-seq-len 32768 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 240 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 \
  --glaive-mr-epochs 2 --toolstop-epochs 6 \
  --harness-epochs 2 --harness-basics-epochs 3 --harness-poison-epochs 3 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 40 --eval-tokens 524288 --save-every 60 --ce-chunk 4096