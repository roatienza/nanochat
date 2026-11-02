#!/usr/bin/env bash
# v6m: Tier-2 touch-up on top of v6j — the msx-gate failures (2/12).
#
# What changed vs run_v5r.sh:
#   - source is v6j (the currently served checkpoint), not v5e
#   - integration longctx families (doc_arith, doc_join, doc_track,
#     doc_contradict) at level 1 — the LC1/LC2/LC4 gate failures
#   - multistep episodes x2 (grep_then_read, list_then_read, edit_then_verify,
#     sum_three, search_then_read) — the MS1..MS5 gate failures
#   - longctx base families kept (niah, doc_qa, multi_tool) so the v5r
#     24k-doc->CALL fix does not regress
#   - P1.4 invariant now in chat_sft_long.py: any row that renders with
#     zero supervised tokens CRASHES the run loudly instead of being skipped
#
# Same hyperparameters as v5r (60 it, tbs 262144, msl 32768, dbs 1).
# GPU: set CUDA_VISIBLE_DEVICES before invoking (default 0).
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v6m-msx \
  --source-phase sft --model-tag d28b-128k-sft-v6j --model-step 240 \
  --output-tag d28b-128k-sft-v6m \
  --max-seq-len 32768 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 60 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa,doc_arith,doc_join,doc_track,doc_contradict \
  --multistep-epochs 2 --multistep-rows 3000 --multistep-seed 2027 \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 \
  --glaive-mr-epochs 2 --toolstop-epochs 4 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 20 --eval-tokens 524288 --save-every 30 --ce-chunk 4096