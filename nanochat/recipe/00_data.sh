#!/usr/bin/env bash
# utak — Stage 0: data preparation.
#
# Two things have to be in place before any training starts: the
# tokenizer, and the pre-training corpus.
#
# 0a. THE TOKENIZER. A 32,768-token byte-level BPE, trained on the
#     ClimbMix corpus with a 10,000-character document cap. It is a
#     PRE-TRAINING artifact: every later stage (SFT, RL, serving) reads
#     it from the base directory, so it must be trained once and kept.
#
# 0b. THE CORPUS. karpathy/climbmix-400b-shuffle (NVIDIA Nemotron
#     ClimbMix), 6,543 shards upstream. The original run used 171 shards
#     (~15 GB on disk). The downloader always pins the LAST shard
#     (shard_06542) as the validation set and uses everything else for
#     training.
#
#     The shard count is not a free parameter. The pretrain schedule is
#     derived from --target-param-data-ratio 8.0 against the data that is
#     actually present, so changing the shard count changes the number of
#     iterations and therefore the compute budget.
#
# 0c. THE TASK CORPORA. The SFT and RL stages pull their datasets from
#     the base directory too: GSM8K, MMLU, ARC, HumanEval and SmolTalk.
#
# Usage:  ./recipe/00_data.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

CORPUS_SHARDS="${CORPUS_SHARDS:-170}"
TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-2000000000}"
TOKENIZER_VOCAB="${TOKENIZER_VOCAB:-32768}"
TOKENIZER_DOC_CAP="${TOKENIZER_DOC_CAP:-10000}"

# --- 0a: the tokenizer ----------------------------------------------------
if [ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
  log "tokenizer already present -- skipping"
else
  log "=== Stage 0a: training the tokenizer (vocab $TOKENIZER_VOCAB) ==="
  ( cd "$NANOCHAT_ROOT" \
    && NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" \
       "$PY" -m scripts.tok_train \
         --max-chars "$TOKENIZER_MAX_CHARS" \
         --doc-cap "$TOKENIZER_DOC_CAP" \
         --vocab-size "$TOKENIZER_VOCAB" ) \
    > "$LOG_DIR/stage0a_tokenizer.log" 2>&1 \
    || { tail -25 "$LOG_DIR/stage0a_tokenizer.log"; die "tokenizer training failed"; }
  log "tokenizer trained"
fi

# --- 0b: the pre-training corpus -----------------------------------------
have=$(find "$NANOCHAT_BASE_DIR/base_data_climbmix" -name 'shard_*.parquet' 2>/dev/null | wc -l)
if [ "$have" -ge $((CORPUS_SHARDS + 1)) ]; then
  log "corpus already present ($have shards) -- skipping"
else
  log "=== Stage 0b: downloading $CORPUS_SHARDS train shards (+1 validation) ==="
  ( cd "$NANOCHAT_ROOT" \
    && NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" \
       "$PY" -m nanochat.dataset -n "$CORPUS_SHARDS" -w 8 ) \
    > "$LOG_DIR/stage0b_corpus.log" 2>&1 \
    || { tail -25 "$LOG_DIR/stage0b_corpus.log"; die "corpus download failed"; }
  log "corpus downloaded"
fi

# --- 0c: the task corpora -------------------------------------------------
# The SFT and RL stages fetch their own datasets on first use, through
# nanochat/dataset.py. This is just a report of what landed.
log "=== Stage 0c: task corpora ==="
if [ -d "$NANOCHAT_BASE_DIR/task_data" ]; then
  for d in "$NANOCHAT_BASE_DIR/task_data"/*/; do
    [ -d "$d" ] && log "    $(basename "$d")"
  done
else
  log "    task_data not present yet -- the SFT/RL stages will fetch on first use"
fi

log "Stage 0 complete. Next: ./recipe/01_pretrain.sh"
