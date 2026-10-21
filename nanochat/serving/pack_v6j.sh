#!/usr/bin/env bash
# v6j: pack the trained checkpoint -> a vLLM model dir, then patch the config
# to 131072 ctx and copy the aux files + the served chat template (v7).
#
# v6j = the literal-backslash-n training-data fix (see ../run_v6j.sh for the
# full story).
#
# This script is self-contained: it needs the nanochat checkout that contains
# it (../), a NANOCHAT_BASE_DIR holding the checkpoints, and a writable
# output dir. Everything else (aux files, generation defaults) comes from
# serving/model_assets/ in this repo, so the pack is reproducible from this
# tree alone.
#
# Usage:
#   NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2 \
#   OUT=/data/rowel/nanochat_store/v2/vllm_d28b/nanochat-d28b-rl100-128k-v6j \
#   ./pack_v6j.sh
set -u

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT="$(dirname "$HERE")"                     # the nanochat/ checkout root
ASSETS="$HERE/model_assets"

BASE_DIR="${NANOCHAT_BASE_DIR:-/data/rowel/nanochat_store/v2}"
OUT="${OUT:-$BASE_DIR/vllm_d28b/nanochat-d28b-rl100-128k-v6j}"
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="${PYTHON:-python3}"

TAG="${TAG:-d28b-128k-sft-v6j}"
STEP="${STEP:-240}"

cd "$HERE"
mkdir -p "$OUT"

echo "export: $TAG @ $STEP  ->  $OUT"
if "$PY" export_hf_128k.py --phase sft --base-dir "$BASE_DIR" \
     --tag "$TAG" --step "$STEP" --out "$OUT" > "$OUT/export_v6j.log" 2>&1; then
  echo "export DONE"
else
  echo "export FAILED"; tail -10 "$OUT/export_v6j.log"; exit 1
fi

# config.json: sequence_len/max_position_embeddings -> 131072
"$PY" patch_config_131072.py "$OUT/config.json"

# aux files the vLLM plugin needs (architecture + tokenizer + generation defaults)
for f in tokenizer.json tokenizer_config.json configuration_nanochat.py \
         modeling_nanochat_d28.py generation_config.json; do
  cp -p "$ASSETS/$f" "$OUT/$f"
done

# the SERVED chat template (v7 = tools block on the FIRST user turn)
cp -p "$ROOT/chat_template_v7.jinja" "$OUT/chat_template.jinja"

echo "packed: $OUT"
