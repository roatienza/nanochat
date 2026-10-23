#!/usr/bin/env bash
# v6j: fix the literal-backslash-n training-data bug.
#
# THE BUG (found 2026-10-20, live repro on the real baby-onit harness):
#   tasks/harness_basics.py _LONG_ANSWER_PADS[1] used "\\n\\n" (an escaped
#   BACKSLASH followed by n) instead of a real newline. The supervised
#   TARGET therefore contained the two characters '\' 'n'. The model learned
#   to emit a literal backslash-n, which at inference degenerates into a
#   period-2 byte-BPE cycle: "\n\n\n\n..." for the full token cap.
#   Live symptom: "date today" -> 120.4s, 4096 tokens of literal \n, saved
#   into session_history.jsonl, then replayed forever as an in-context
#   attractor that poisoned every later turn.
#
# THE FIX: tasks/harness_basics.py now uses a real newline; plus a
#   regression test (tasks/test_no_literal_newline.py) that fails if any
#   prose target ever contains a literal backslash-n again.
#
# Everything else is identical to v6h (the anchor-stop fix that passed the
# real-harness gate 14/15 and the poison gate 6/7).
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=7
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v6j-literalnl \
  --source-phase sft --model-tag d28b-128k-sft-v6a --model-step 240 \
  --output-tag d28b-128k-sft-v6j \
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
