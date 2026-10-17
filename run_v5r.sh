#!/usr/bin/env bash
# v5r: TRUE fix — level-1 longctx rows are 31.2k tokens (not 24k; the
# 1.3 tok/word estimate is off), so max-seq-len must be 32768. v5q's 24576
# still truncated them. All v5f..v5q touch-ups at 8192 never saw any 24k row.
# v5e trained at 131072 and saw them in full (hence v5e passed 24k).

# Fix: v5j recipe at max-seq-len 32768, dbs 1, GPU 0.
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=0
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v5r-longctx32k \
  --source-phase sft --model-tag d28b-128k-sft-v5e --model-step 60 \
  --output-tag d28b-128k-sft-v5r \
  --max-seq-len 32768 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 60 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 \
  --glaive-mr-epochs 2 --toolstop-epochs 4 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 20 --eval-tokens 524288 --save-every 30 --ce-chunk 4096
