#!/usr/bin/env bash
# v6c: the self-poisoning-loop fix. Harness feedback (Oct 19): the model
# hallucinated a date, the harness saved it to session history, and the
# replayed 284-turn context then kept the canned string in front of the model,
# which copy-pasted it as the answer to every later question (tool-calling code
# never ran). Live repro on deployed v6a: verify_hp.py 2/7 — with poison in
# history the model copies the canned date and never routes.
#
# Root cause: no training row shows a BROKEN assistant answer in history (all
# history answers are competent), so in-context pattern matching on the model's
# own bad answers beats routing. And broken answers cannot be put in rows as-is
# — render_conversation supervises every assistant turn, which would TRAIN the
# canned strings. v6c adds the no_train message flag (context-only turns, mask
# 0) + tasks/harness_poison.py (poison pairs mask-0 in history, correct final
# turn supervised: redo-poison 30% / social 20% / routed CALL 20% / date-from-
# system 10% / easy 10% / knowledge 10%).
#
# Source = deployed v6a@240 (holds R1-R5/LOOP/P13/P1/MT24 + F1/F2); the
# diversified P7 anchors now in harness_basics re-fight P7 (10/10 on v6a).
set -u
cd /data/rowel/nanochat
export CUDA_VISIBLE_DEVICES=1
exec .venv/bin/python -m scripts.chat_sft_long \
  --run stage7b-v6c-harnesspoison \
  --source-phase sft --model-tag d28b-128k-sft-v6a --model-step 240 \
  --output-tag d28b-128k-sft-v6c \
  --max-seq-len 32768 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 240 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 \
  --glaive-mr-epochs 2 --toolstop-epochs 6 \
  --harness-epochs 2 --harness-basics-epochs 3 --harness-poison-epochs 2 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 40 --eval-tokens 524288 --save-every 60 --ce-chunk 4096