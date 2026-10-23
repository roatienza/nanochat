#!/usr/bin/env bash
# v6g: descriptions + date-header fix. v6f gates: HP 6/7 (HP5 pool-date leak),
# v5x 11/15 with P1/P13/LOOP-T2 still failing. Census showed ZERO rows teach
# [get_weather CALL -> Tool result -> PROSE] with tool DESCRIPTIONS (the gate
# and real-harness shape): ToolStop D rows had no descriptions, ToolFollowUp
# has no get_weather. v6g: ToolStop._mk_tools uses _TOOLS_DESC on 50% of rows;
# harness_basics B3b (18% of rows) = full call->result->prose loop under the
# harness/classic toolsets; date answers in all three harness tasks now cite
# the system header explicitly (HP5 fix).
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=4
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v6g-descstop \
  --source-phase sft --model-tag d28b-128k-sft-v6a --model-step 240 \
  --output-tag d28b-128k-sft-v6g \
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