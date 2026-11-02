#!/usr/bin/env bash
# utak — Stage 10: pack the trained checkpoint into a vLLM model directory.
#
# Converts a nanochat checkpoint (.pt + meta) into a HuggingFace-style
# directory that vLLM can load with --trust-remote-code:
#
#   1. export the weights + config      (serving/export_hf_128k.py)
#   2. patch sequence_len to 131072     (serving/patch_config_131072.py)
#   3. copy the aux files vLLM needs    (serving/model_assets/)
#   4. install the served chat template (chat_template_v7.jinja)
#
# The aux files live in the repo, so this pack is reproducible from the
# tree alone -- no network and no external asset directory.
#
# Usage:  ./recipe/10_pack.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

log "=== Stage 10: pack $FINAL_TAG @ $FINAL_STEP -> $MODEL_DIR ==="

[ -d "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$FINAL_TAG" ] \
  || die "no such checkpoint: chatsft_checkpoints/$FINAL_TAG"

# The step to pack. It is the step stage 9's handoff record names -- the
# step the served checkpoint actually reached. The FINAL_STEP constant is
# the fallback for output that predates the handoff records.
#
# This must be "dynamic", not "fixed". Every other consumer in the recipe
# (02..09) asks for "dynamic"; this one asked for "fixed", which tells
# stage_read_step to take the FINAL_STEP constant LITERALLY and never look
# at the handoff record -- the exact opposite of what the comment above
# promises, and the exact failure the handoff records were introduced to
# end.
#
# It bit for real. FINAL_STEP was left at 240 when the served tag moved
# from 9f (240 iterations, utak-sft-c) to 9g (120 iterations,
# utak-sft-d). Stage 9 wrote a correct handoff naming step 120, and this
# call ignored it and demanded model_000240.pt -- so stage 10 would have
# died after all nine rungs, with the training done and nothing packable.
#
# "dynamic" also survives the case that actually happens on a restart: the
# pipeline shell exported FINAL_STEP at launch, so a config.sh edit does
# not reach a stage started from that shell. The handoff record always
# names what the trainer really wrote.
FINAL_STEP_RESOLVED="$(stage_read_step "$FINAL_TAG" \
  "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$FINAL_TAG" \
  "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$FINAL_TAG" \
  "$FINAL_STEP" dynamic)" \
  || die "cannot establish the step to pack"
log "    packing $FINAL_TAG @ $FINAL_STEP_RESOLVED"

mkdir -p "$MODEL_DIR"

log "exporting weights..."
( cd "$NANOCHAT_ROOT/serving" \
  && NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" \
     "$PY" export_hf_128k.py \
       --phase sft \
       --base-dir "$NANOCHAT_BASE_DIR" \
       --tag "$FINAL_TAG" \
       --step "$FINAL_STEP_RESOLVED" \
       --out "$MODEL_DIR" ) > "$LOG_DIR/pack_export.log" 2>&1 \
  || { tail -25 "$LOG_DIR/pack_export.log"; die "export failed"; }

log "patching config to 131072 context..."
( cd "$NANOCHAT_ROOT/serving" && "$PY" patch_config_131072.py "$MODEL_DIR/config.json" ) \
  || die "config patch failed"

log "installing aux files + served chat template..."
for f in tokenizer.json tokenizer_config.json configuration_nanochat.py \
         modeling_nanochat_d28.py generation_config.json; do
  cp -p "$NANOCHAT_ROOT/serving/model_assets/$f" "$MODEL_DIR/$f" \
    || die "missing aux file: $f"
done
cp -p "$NANOCHAT_ROOT/chat_template_v7.jinja" "$MODEL_DIR/chat_template.jinja" \
  || die "missing chat_template_v7.jinja"

log "packed: $MODEL_DIR"
log "  model dir name : $MODEL_DIR_NAME"
log "  served name    : $MODEL_NAME"
log "Next: ./recipe/11_serve.sh"
