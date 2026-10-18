#!/usr/bin/env bash
# v5u: harness-replay fix, third pass. v5s recipe (no P7-anchor rows — v5t
# showed they regress P1) + schema-driven H3 mapping (v5s's fixed
# weather->web_search leaked into MT24 as check_weather) + H4 date share 10->12%
# + toolstop-epochs 4->6 to hold the call boundary.
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=0
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v5w-harnessreplay5 \
  --source-phase sft --model-tag d28b-128k-sft-v5e --model-step 60 \
  --output-tag d28b-128k-sft-v5w \
  --max-seq-len 32768 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 60 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 \
  --glaive-mr-epochs 2 --toolstop-epochs 6 \
  --harness-epochs 2 --mt-extra-epochs 2 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 20 --eval-tokens 524288 --save-every 30 --ce-chunk 4096