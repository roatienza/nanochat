#!/usr/bin/env bash
# v6f: the train-vs-serve placement fix. v6e gates measured the failure:
# P1/P13/LOOP-T2 (result->prose re-calls) + HP2/HP3/HP5 (poison redo).
#
# Root cause (census + live render check, Oct 19): template v6 injects the
# tools block onto the LAST user turn. In a live tool loop the last user turn
# IS the "Tool result:" turn when the model is about to answer, so the block
# landed right before the prose — a shape ZERO training rows teach. ~36k
# corpus rows (ToolStop D 1845, ToolFollowUp 881, GlaiveMR 33484) end
# result->prose with the block on the QUESTION turn. v6e therefore re-calls
# (get_weather Oslo attractor) instead of answering.
#
# v6f: template v7 = v5 placement (block on FIRST user turn) + harness rows
# moved back to first-turn placement + HP-adjacency rows (poison pair for the
# redo question appended immediately before the final turn in 50% of redo
# rows; v6e had only 36/6000).
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=4
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v6f-placement \
  --source-phase sft --model-tag d28b-128k-sft-v6a --model-step 240 \
  --output-tag d28b-128k-sft-v6f \
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