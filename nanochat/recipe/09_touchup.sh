#!/usr/bin/env bash
# utak — Stage 9: the touch-up ladder (the served model's path).
#
# This is the chain that actually produces the deployed checkpoint. It is a
# series of short, cheap SFT runs, each fixing one specific defect found by
# the evaluation gates.
#
#   $TAG_SFT          (stage 8)         baseline long-context SFT
#   -> followup       tool follow-up: continue after a tool result
#   -> glaive         multi-round tool-use fluency
#   -> glaive-heavy   more multi-round weight
#   -> $TAG_TOUCH_A   re-anchor: bring long-context ability back after the
#                     8k-context touch-ups, and move to the v6 shape (the
#                     tools block on the FIRST user turn)
#   -> $TAG_TOUCH_B   harness basics + harness poison + the
#                     literal-backslash-n data fix.
#                     THIS IS THE SERVED CHECKPOINT.
#
# The v6 shape matters. With the tools block on the LAST user turn and 60+
# turns of history after it, the 2B model loses the block->action association
# and hallucinates tool names (get_weather / web_news / get_file_path).
# With the block on the FIRST user turn, the same history+tools+question
# renders correctly. The served chat template (v7) places it first.
#
# The final rung is the one that fixed the original bug: tasks/harness_basics.py
# had used "\\n\\n" -- an escaped BACKSLASH followed by n -- where a real
# newline was intended, so the supervised TARGET contained the two
# characters '\' and 'n'. The model learned to emit a literal backslash-n,
# which at inference degenerates into a period-2 byte-BPE cycle. The fix
# uses a real newline, and tasks/test_no_literal_newline.py fails if any
# prose target ever contains a literal backslash-n again.
#
# Usage:  ./recipe/09_touchup.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 9: touch-up ladder -> $TAG_TOUCH_B ==="
STAGE_GPU="${GPU_TOUCHUP}"
# The touch-ups use scripts/chat_sft_long.py, which uses a plain
# torch.optim.AdamW and does NOT reduce gradients across ranks. On
# more than one GPU the ranks would silently diverge. One card only.
# See recipe/config.sh.
require_single_gpu "09_touchup"

# ---------------------------------------------------------------------------
# 9a: tool follow-up. Teaches the model to continue after a tool result.
# ---------------------------------------------------------------------------
run_mod "$LOG_DIR/stage9a_toolfollowup.log" scripts.chat_sft_long \
  --run "utak-touch-toolfollowup" \
  --source-phase sft --model-tag "$TAG_SFT" --model-step 800 \
  --output-tag "${TAG_SFT}-followup" \
  --max-seq-len 8192 --short-window 512 \
  --device-batch-size 4 --total-batch-size 262144 \
  --num-iterations 120 \
  --longctx-epochs 1 --levels 0 --instances-per-level 40 \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 3 \
  --smoltalk-epochs 1 \
  --embedding-lr 0.0144 --unembedding-lr 0.00038 \
  --matrix-lr 0.00096 --scalar-lr 0.024 \
  --weight-decay 0.028 \
  --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 30 --eval-tokens 524288 --save-every 60 --ce-chunk 4096 \
|| die "stage 9a failed"

# ---------------------------------------------------------------------------
# 9b: glaive multi-round. Multi-turn tool-use fluency.
# ---------------------------------------------------------------------------
run_mod "$LOG_DIR/stage9b_glaive.log" scripts.chat_sft_long \
  --run "utak-touch-glaive" \
  --source-phase sft --model-tag "${TAG_SFT}-followup" --model-step 120 \
  --output-tag "${TAG_SFT}-glaive" \
  --max-seq-len 8192 --short-window 512 \
  --device-batch-size 4 --total-batch-size 262144 \
  --num-iterations 80 \
  --longctx-epochs 1 --levels 0 --instances-per-level 40 \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 2 --glaive-mr-epochs 3 \
  --smoltalk-epochs 1 \
  --embedding-lr 0.0072 --unembedding-lr 0.00019 \
  --matrix-lr 0.00048 --scalar-lr 0.012 \
  --weight-decay 0.028 \
  --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 20 --eval-tokens 524288 --save-every 40 --ce-chunk 4096 \
|| die "stage 9b failed"

# ---------------------------------------------------------------------------
# 9c: glaive heavy. More multi-round weight.
# ---------------------------------------------------------------------------
run_mod "$LOG_DIR/stage9c_glaive_heavy.log" scripts.chat_sft_long \
  --run "utak-touch-glaive-heavy" \
  --source-phase sft --model-tag "${TAG_SFT}-glaive" --model-step 80 \
  --output-tag "${TAG_SFT}-glaive-heavy" \
  --max-seq-len 8192 --short-window 512 \
  --device-batch-size 4 --total-batch-size 262144 \
  --num-iterations 100 \
  --longctx-epochs 1 --levels 0 --instances-per-level 40 \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 2 --glaive-mr-epochs 4 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0072 --unembedding-lr 0.00019 \
  --matrix-lr 0.00048 --scalar-lr 0.012 \
  --weight-decay 0.028 \
  --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 25 --eval-tokens 524288 --save-every 50 --ce-chunk 4096 \
|| die "stage 9c failed"

# ---------------------------------------------------------------------------
# 9d: re-anchor. Brings long-context ability back after the 8k-context
#     touch-ups, and moves to the v6 shape.
# ---------------------------------------------------------------------------
run_mod "$LOG_DIR/stage9d_anchor.log" scripts.chat_sft_long \
  --run "utak-touch-anchor" \
  --source-phase sft --model-tag "${TAG_SFT}-glaive-heavy" --model-step 100 \
  --output-tag "$TAG_TOUCH_A" \
  --max-seq-len 8192 --short-window 512 \
  --device-batch-size 4 --total-batch-size 262144 \
  --num-iterations 60 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 --glaive-mr-epochs 2 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 \
  --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 \
  --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 20 --eval-tokens 524288 --save-every 30 --ce-chunk 4096 \
|| die "stage 9d failed"

# ---------------------------------------------------------------------------
# 9e: harness basics. THE KEY RUNG.
#
#     Teaches the three things the general tool-calling corpora do not:
#       B1 social      -> "hello" is a greeting, NO tool call
#       B2 knowledge   -> "What is 2+2?" is answered, NO tool call
#       B3 toolish     -> route to the CORRECT tool
#       B4 date        -> read the date from the system prompt, never invent
#       B5 toolish+KB  -> answer from knowledge, NO tool call
#
#     Without this the model's default is "tools present -> call web_search".
#     The census that found it: fresh+tools -> CALL was 9,993 rows against
#     281 prose rows -- a 35:1 imbalance toward calling.
#
#     Runs at 32,768 context (not 8k) because the harness rows carry long
#     histories.
# ---------------------------------------------------------------------------
run_mod "$LOG_DIR/stage9e_harness_basics.log" scripts.chat_sft_long \
  --run "utak-touch-harness-basics" \
  --source-phase sft --model-tag "$TAG_TOUCH_A" --model-step 60 \
  --output-tag "$TAG_TOUCH_B" \
  --max-seq-len 32768 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 240 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 --glaive-mr-epochs 2 \
  --toolstop-epochs 6 \
  --harness-epochs 2 --harness-basics-epochs 3 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 \
  --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 \
  --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 40 --eval-tokens 524288 --save-every 60 --ce-chunk 4096 \
|| die "stage 9e failed"

# ---------------------------------------------------------------------------
# 9f: harness poison + the literal-backslash-n data fix. THE SERVED RUNG.
#
#     The ONLY difference from 9e is --harness-poison-epochs 3. Everything
#     else -- the learning rates, the mixture, the 240 iterations, the
#     32,768 context -- is byte-identical. That is deliberate: it isolates
#     the change, so the measured difference belongs to the change.
#
#     The bug it fixes: tasks/harness_basics.py had used "\\n\\n" -- an
#     escaped BACKSLASH followed by n -- where a real newline was
#     intended, so the supervised TARGET contained the two characters '\'
#     and 'n'. The model faithfully learned to emit a literal backslash-n,
#     which at inference degenerates into a period-2 byte-BPE cycle:
#     "\n\n\n\n..." for the full token cap, 4,096 tokens, 120 s per answer.
#     The harness then SAVED that answer into session_history.jsonl and
#     replayed it forever, so the poison became an in-context attractor.
#
#     The fix uses a real newline, and tasks/test_no_literal_newline.py
#     fails if any prose target ever contains a literal backslash-n again.
#
#     The harness-poison rows additionally teach the model to keep its
#     composure when its OWN degenerate answers are replayed in the history.
# ---------------------------------------------------------------------------
run_mod "$LOG_DIR/stage9f_harness_poison.log" scripts.chat_sft_long \
  --run "utak-touch-harness-poison" \
  --source-phase sft --model-tag "$TAG_TOUCH_B" --model-step 240 \
  --output-tag "$TAG_TOUCH_C" \
  --max-seq-len 32768 --short-window 512 \
  --device-batch-size 1 --total-batch-size 262144 \
  --num-iterations 240 \
  --longctx-epochs 2 --levels 0,1 --instances-per-level 40 \
  --longctx-families multi_tool,niah,doc_qa \
  --toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1 --glaive-mr-epochs 2 \
  --toolstop-epochs 6 \
  --harness-epochs 2 --harness-basics-epochs 3 --harness-poison-epochs 3 \
  --smoltalk-epochs 1 --smoltalk-cap 20000 \
  --embedding-lr 0.0036 --unembedding-lr 0.0001 \
  --matrix-lr 0.00024 --scalar-lr 0.006 \
  --weight-decay 0.028 \
  --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 40 --eval-tokens 524288 --save-every 60 --ce-chunk 4096 \
|| die "stage 9f failed"

log "Stage 9 complete. SERVED CHECKPOINT: chatsft_checkpoints/$FINAL_TAG @ $FINAL_STEP"
log "Expected: v5x gate 15/15, HP gate 6/7 (HP5 is a known, cosmetic, pre-existing failure)."
