#!/usr/bin/env bash
# v5y: the fresh-session harness fix. v5v review (Oct 18) quantified the gap:
#   fresh+tools->CALL 9993 rows (ToolCalling+ToolACE) vs
#   fresh+tools->prose 281 rows (HarnessReplay)  => 35:1 toward spurious calls
#   routing never taught (files->web_search instead of bash)
#   temp-0.6 (the harness's default) hallucinates where temp-0 gates passed
# HarnessBasics adds ~8600 fresh+prose rows (x2 epochs) with correct routing
# and diverse answers. Everything else matches the v5v recipe that held
# P7/P13/P1/LOOP.
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=4
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v5y-harnessbasics \
  --source-phase sft --model-tag d28b-128k-sft-v5e --model-step 60 \
  --output-tag d28b-128k-sft-v5y \
  --max-seq-len 32768 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 60 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 \
  --glaive-mr-epochs 2 --toolstop-epochs 6 \
  --harness-epochs 2 --harness-basics-epochs 3 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 20 --eval-tokens 524288 --save-every 30 --ce-chunk 4096