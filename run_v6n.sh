#!/usr/bin/env bash
# v6n: the FULL-DOSE Tier-2 run — v6m's lesson is that 60 it with 2x3000
# multistep rows teaches the loop shape (MS5 3/3, MS1 2/3 on the fixed gate)
# but is underdosed and interferes with doc_qa (3/3 -> 1/3).
#
# Changes vs run_v6m.sh:
#   - 240 iterations (full warmdown, same as v6j's own schedule)
#   - multistep x4 (12k rows), integration families x4 (doc_arith, doc_join,
#     doc_track, doc_contradict at levels 0,1, ipl 40 each)
#   - base longctx families kept x2 (niah, doc_qa, multi_tool) so the v5r
#     24k doc->CALL fix holds
#   - source stays v6j@240 (v6m is a probe, not a base)
#
# ~2.5 h on one A100 (240 it x ~26 s/step at msl 32768, dbs 1, tbs 262144).
set -u
cd /data/rowel/nanochat
CVG="$(if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo 3; else echo "$CUDA_VISIBLE_DEVICES"; fi)"
CUDA_VISIBLE_DEVICES="$CVG" exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v6n-msx-full \
  --source-phase sft --model-tag d28b-128k-sft-v6j --model-step 240 \
  --output-tag d28b-128k-sft-v6n \
  --max-seq-len 32768 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 240 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa,doc_arith,doc_join,doc_track,doc_contradict \
  --multistep-epochs 4 --multistep-rows 3000 --multistep-seed 2027 \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 \
  --glaive-mr-epochs 2 --toolstop-epochs 4 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 40 --eval-tokens 524288 --save-every 60 --ce-chunk 4096