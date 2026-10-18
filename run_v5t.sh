#!/usr/bin/env bash
# v5s: harness-replay fix — the model must answer the NEXT turn after its own
# long technical answers are replayed as history (baby-onit resumed-session
# shape). Root cause + ablations: probe_harness_repro.py / probe_harness_ablation.py.
#
# Recipe: v5r's exactly (multi_tool 24k PASS, P7 0/10) + HarnessReplay x3.
# Source: v5e step 60 (same as v5r) so both fixes are learned from the same base.
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=0
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v5t-harnessreplay2 \
  --source-phase sft --model-tag d28b-128k-sft-v5e --model-step 60 \
  --output-tag d28b-128k-sft-v5t \
  --max-seq-len 32768 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 60 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 \
  --glaive-mr-epochs 2 --toolstop-epochs 4 \
  --harness-epochs 3 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 20 --eval-tokens 524288 --save-every 30 --ce-chunk 4096