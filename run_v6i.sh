#!/usr/bin/env bash
# v6i: the long-history date fix. v6h gates: v5x 14/15 (F1 4/5), HP 6/7 (HP5).
# Bisection showed BOTH residuals share one root cause: a LONG dense assistant
# turn at the end of replayed history (the real session's 2479-char 'run A'
# answer) breaks the system-header date association — the model samples a
# _DATES pool date instead. Filler text of the same length does NOT trigger it
# => register/content effect, not length. Training rows only taught
# date-from-header with SHORT histories.
# v6i: all three harness tasks now end ~30-35% of histories on a LONG dense
# assistant turn (800-2600 chars, _LONG_ANSWER_PADS register-matched to the
# real session), so date-from-header + no-call boundaries are taught UNDER
# that shape. Everything else identical to v6h (anchor-stop, desc toolsets,
# B3b loop rows, header dates).
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=3
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v6i-longdate \
  --source-phase sft --model-tag d28b-128k-sft-v6a --model-step 240 \
  --output-tag d28b-128k-sft-v6i \
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