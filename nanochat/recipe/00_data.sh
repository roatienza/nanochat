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
#     The shard count does NOT change the compute budget: the pretrain
#     schedule is derived from --target-param-data-ratio against the
#     MODEL's parameter count, not against the size of the corpus. It only
#     changes how many times the data is repeated. 500 shards means the
#     9.10 G token budget is 0.40 epochs -- no repetition at all.
#
# 0c. THE TASK CORPORA. The SFT and RL stages pull their datasets from
#     the base directory too: GSM8K, MMLU, ARC, HumanEval and SmolTalk.
#
# Usage:  ./recipe/00_data.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
. ./recipe/config.sh

# 500 train shards, not the original 170.
#
# The original run had 171 files (170 train + the validation shard):
# ~12.1 G source tokens, ~7.9 G trainable after the 35% best-fit
# cropping loss, and it trained on 9.10 G tokens -- about 1.15 epochs,
# so roughly 15% of the data was seen twice. The checkpoint metadata
# agrees (epoch 2, pq_idx 30).
#
# 500 train shards is ~35.3 G source tokens, ~23.0 G trainable. At the
# same 9.10 G token budget that is 0.40 epochs: the same tokens, seen
# once instead of a seventh time. That is a real quality gain for the
# same wall clock, and it costs only disk.
#
# The download is resumable (it skips files that exist) and takes about
# 11 minutes at the measured 67.4 MB/s with 8 workers.
CORPUS_SHARDS="${CORPUS_SHARDS:-500}"
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
