#!/usr/bin/env bash
# v5o: weather-forced re-anchor. Exact v5j ToolStop dist
# (C35/C'35/D30 — E rows re-flipped P7 in v5l/v5m; v5n count-boost
# was not enough: model still picks the memorized search_web branch) +
# weather-branch-only oversample (--mt-extra-epochs 2, seed 4243, ipl 40).
# From v5e ckpt 60. GPU 3 (shares with an idle 13.7GB process -> dbs 2).
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=3
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v5o-mtweather \
  --source-phase sft --model-tag d28b-128k-sft-v5e --model-step 60 \
  --output-tag d28b-128k-sft-v5o \
  --max-seq-len 8192 --short-window 512 \
  --device-batch-size 2 --total-batch-size 262144 \
  --num-iterations 60 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --mt-extra-epochs 2 \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 \
  --glaive-mr-epochs 2 --toolstop-epochs 4 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 20 --eval-tokens 524288 --save-every 30 --ce-chunk 4096
