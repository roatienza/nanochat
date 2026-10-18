#!/usr/bin/env bash
# v6a: template v6 + v6-placement training rows. The v5z review isolated the
# replay+toolish failures as POSITIONAL: with the tools block on the FIRST
# user turn and 60+ turns of history after it, the 2B model loses the
# block->action association and hallucinates tool names (get_weather/web_news/
# get_file_path). With the block on the LAST user turn (the acting turn) the
# same history+tools+question renders correctly (verified live: web_search
# {"query":"weather today"}). v6a trains the harness rows in the v6 shape and
# deploys with chat_template_v6.jinja.
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=4
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v6a-harnessbasics \
  --source-phase sft --model-tag d28b-128k-sft-v5e --model-step 60 \
  --output-tag d28b-128k-sft-v6a \
  --max-seq-len 32768 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 240 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 \
  --glaive-mr-epochs 2 --toolstop-epochs 6 \
  --harness-epochs 2 --harness-basics-epochs 3 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 40 --eval-tokens 524288 --save-every 60 --ce-chunk 4096