#!/usr/bin/env bash
# utak — the single source of truth for the training recipe.
#
# Every stage script sources this file. Nothing else hardcodes a path, a
# model name, or a hyperparameter. Change it here and the whole pipeline
# follows.
#
# Model name: "utak". The served OpenAI-compatible model id is "utak";
# the vLLM model directory is "utak-128k"; the training checkpoint tags
# are "utak-<stage>".
set -u

# log writes to STDERR on purpose: several helpers are called inside a
# command substitution (e.g. GPU_PRETRAIN="$(pick_free_gpus 1)"), and a log
# line on stdout would be captured into the variable. Stage logs are
# redirected with 2>&1 anyway, so nothing is lost.
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }

die() { printf '[utak][%s] FATAL: %s\n' "$(date '+%F %T')" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-utak}"                       # served model id
MODEL_DIR_NAME="${MODEL_DIR_NAME:-utak-128k}"         # vLLM model dir
REPO_TAG="${REPO_TAG:-v6j}"                           # the touch-up revision

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NANOCHAT_ROOT="${NANOCHAT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# The repo lives at <parent>/reasoner-model, and the training venv lives at
# <parent>/nanochat/.venv. Derive that path instead of hardcoding it, so the
# recipe still works if the tree is moved.
NANOCHAT_PARENT_DIR="${NANOCHAT_PARENT_DIR:-$(cd "$NANOCHAT_ROOT/.." && pwd)}"
NANOCHAT_GRANDPARENT_DIR="${NANOCHAT_GRANDPARENT_DIR:-$(cd "$NANOCHAT_PARENT_DIR/.." && pwd)}"
# Candidate venv locations, most specific first. The real one on this node is
# /data/rowel/nanochat/.venv, i.e. a SIBLING of the repo directory
# (reasoner-model/nanochat), not of the repo itself.
NANOCHAT_SIBLING_VENV="${NANOCHAT_SIBLING_VENV:-$NANOCHAT_GRANDPARENT_DIR/nanochat/.venv}"
NANOCHAT_PARENT_VENV="${NANOCHAT_PARENT_VENV:-$NANOCHAT_GRANDPARENT_DIR/.venv}"
NANOCHAT_REPO_VENV="${NANOCHAT_REPO_VENV:-$NANOCHAT_PARENT_DIR/nanochat/.venv}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/data/rowel/nanochat_store/v2}"
VLLM_DIR="${VLLM_DIR:-$NANOCHAT_BASE_DIR/vllm_d28b}"
VLLM_BIN="${VLLM_BIN:-$VLLM_DIR/venv-vllm/bin/vllm}"
VLLM_PY="${VLLM_PY:-$VLLM_DIR/venv-vllm/bin/python}"
# The training interpreter. It must have torch.
#
# The venv is not tracked (it is 6 GB of wheels), so it usually lives
# OUTSIDE the repo. Search for it in this order:
#
#   1. $PY                       (explicit override -- always wins)
#   2. $NANOCHAT_ROOT/.venv      (an in-repo venv, if someone made one)
#   3. $NANOCHAT_SIBLING_VENV    (the venv beside the repo; this is where
#                                  it actually is on this node)
#   4. $PYTHON / python3
#
# FAIL LOUDLY if the chosen interpreter has no torch. A silent fallthrough
# to a system python turns a 10-day run into an ImportError 40 minutes in.
PY="${PY:-}"
if [ -z "$PY" ]; then
  for cand in \
      "$NANOCHAT_ROOT/.venv/bin/python" \
      "$NANOCHAT_REPO_VENV/bin/python" \
      "$NANOCHAT_SIBLING_VENV/bin/python" \
      "$NANOCHAT_PARENT_VENV/bin/python"
  do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
  done
fi
if [ -z "$PY" ]; then PY="${PYTHON:-python3}"; fi
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  printf ' no usable python found (no torch).\n' >&2
  printf '  Tried: %s/.venv, %s, %s, %s, and %s.\n' \
    "$NANOCHAT_ROOT" "${NANOCHAT_REPO_VENV:-<unset>}" \
    "${NANOCHAT_SIBLING_VENV:-<unset>}" \
    "${NANOCHAT_PARENT_VENV:-<unset>}" "$PY" >&2
  printf '  Fix:  cd %s && uv venv && uv pip install -r pyproject.toml\n' "$NANOCHAT_ROOT" >&2
  printf '  Or:   export PY=/path/to/python-with-torch\n' >&2
  exit 1
fi

# Where the packed model lands and where the guard/vLLM read it from.
MODEL_DIR="${MODEL_DIR:-$VLLM_DIR/$MODEL_DIR_NAME}"

# ---------------------------------------------------------------------------
# GPUs. Every stage that needs a GPU pins one; the stages that do not (pack,
# serve) inherit it.
#
# PROFILE: "multi" (default) or "1gpu". Set PROFILE=1gpu to run the whole
# recipe on a single A100-40GB. It is a supported path, not a degraded one:
#
#   * MEASURED on this node, same d28 workload, 2k ctx, tbs 32,768:
#       1 GPU  18,051 tok/s  MFU 42.62
#       2 GPUs 34,880 tok/s  MFU 41.17
#       4 GPUs 66,585 tok/s  MFU 39.30
#     One GPU is ~97% as efficient PER TOKEN as four. 4 GPUs buy 3.69x
#     throughput, so one GPU costs ~3.7x the wall clock, not 4x.
#   * d28 fits one card: measured peak 34.86 GiB at device-batch 4
#     (nanochat/mem_probe.py), against 40 GiB of card.
#   * The optimizer shards by world_size. Going 4 -> 1 is handled by
#     chat_sft.py:341 (rebuild full-size moments from the 4 saved shards,
#     then re-shard to 1). Going 1 -> 4 is the same path in reverse.
#   * LR does NOT depend on world size. base_train.py scales LR by
#     sqrt(B/B_ref) on TOTAL batch size, and total_batch_size is a user
#     argument that is unchanged by the GPU count. The batch is held
#     constant across profiles, so a 1-GPU run is not a different recipe.
#
# 1-GPU wall clock for the full 16-stage recipe: ~249 h (~10.4 days),
# or ~270 h (~11.3 days) with a 15% restart/IO contingency.
# See docs/ETA_1GPU.md for the derivation.
# ---------------------------------------------------------------------------
# How much free memory does a card have, in MiB?
# ---------------------------------------------------------------------------
gpu_free_mib() {
  local idx="$1" used total
  used=$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  total=$(nvidia-smi -i "$idx" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
  if [ -z "$used" ] || [ -z "$total" ]; then echo 0; return; fi
  echo $((total - used))
}

# ---------------------------------------------------------------------------
# Pick N free cards automatically.
#
# WHY THIS EXISTS
# --------------
# The profiles below used to hardcode card indices (GPU_PRETRAIN=6, etc.).
# That is wrong on a shared node: the occupancy of a card changes by the
# hour. The 2026-10-21 21:25 launch picked GPU 6 because the profile said
# so; GPU 6 was at 100% utilisation with another tenant's job, and the run
# OOMed. Worse, a hardcoded card can be BUSY or, just as bad, a card that
# is free but that the other tenant is about to fill.
#
# A card qualifies as free when it has no compute process on it that is not
# ours. "Free" is deliberately about PROCESSES, not about free memory: a
# card with a small leftover allocation but no process is still usable, and
# a card with 30 GiB used by someone else is not.
#
# Usage:
#   GPU_PRETRAIN="$(pick_free_gpus 1)"     # one free card
#   GPU_PRETRAIN="$(pick_free_gpus 2)"     # two free cards
#
# If fewer than N cards are free it returns the ones it found and logs a
# warning; the caller decides whether to continue.
# ---------------------------------------------------------------------------
gpu_is_free() {
  local idx="$1" pid owner
  for pid in $(nvidia-smi -i "$idx" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ -z "$pid" ] && continue
    owner=$(stat -c %U "/proc/$pid" 2>/dev/null) || owner="?"
    [ "$owner" = "rowel" ] || return 1
  done
  return 0
}

pick_free_gpus() {
  local want="${1:-1}" out="" g
  for g in 0 1 2 3 4 5 6 7; do
    if gpu_is_free "$g"; then
      out="${out:+$out,}$g"
      [ "$(awk -F, '{print NF}' <<< "$out")" -ge "$want" ] && break
    fi
  done
  if [ -z "$out" ]; then
    log "WARNING: no free GPU found (every card has a foreign process on it)."
    return 1
  fi
  local got
  got="$(awk -F, '{print NF}' <<< "$out")"
  if [ "$got" -lt "$want" ]; then
    log "WARNING: asked for $want free GPU(s), found $got ($out)."
  else
    log "auto-picked $got free GPU(s): $out"
  fi
  echo "$out"
}

# ---------------------------------------------------------------------------
# PROFILE selects the GPU layout. Three are supported:
#
#   PROFILE=multi   4 GPUs for the pretrain, 1 for the rest  (the default;
#                    the layout the original run used)
#   PROFILE=2gpu    2 GPUs for every GPU stage
#   PROFILE=1gpu    1 GPU for every GPU stage
#
# Set PROFILE=2gpu to run the whole recipe on two A100-40GB cards. It is a
# supported path, not a degraded one:
#
#   * MEASURED on this node, same d28 workload, 2k ctx, tbs 32,768:
#       1 GPU  18,051 tok/s  MFU 42.62
#       2 GPUs 34,880 tok/s  MFU 41.17
#       4 GPUs 66,585 tok/s  MFU 39.30
#     One GPU is ~97% as efficient PER TOKEN as four. 4 GPUs buy 3.69x
#     throughput, so one GPU costs ~3.7x the wall clock, not 4x.
#   * d28 fits one card: measured peak 34.86 GiB at device-batch 4
#     (nanochat/mem_probe.py), against 40 GiB of card.
#   * The optimizer shards by world_size. Going 4 -> 1 is handled by
#     chat_sft.py:341 (rebuild full-size moments from the 4 saved shards,
#     then re-shard to 1). Going 1 -> 4 is the same path in reverse.
#   * LR does NOT depend on world size. base_train.py scales LR by
#     sqrt(B/B_ref) on TOTAL batch size, and total_batch_size is a user
#     argument that is unchanged by the GPU count. The batch is held
#     constant across profiles, so a 1-GPU run is not a different recipe.
#
# 1-GPU wall clock for the full 16-stage recipe: ~249 h (~10.4 days),
# or ~270 h (~11.3 days) with a 15% restart/IO contingency.
# See docs/ETA_1GPU.md for the derivation.
#
# GRADIENT SYNC. nanochat has no DistributedDataParallel wrapper: the
# gradient reduction lives INSIDE MuonAdamW (nanochat/optim.py), which
# reduce-scatters the large AdamW moments and all-reduces the small ones.
# So every stage that uses model.setup_optimizer() is DDP-correct.
#
# The two long-context scripts are the exception: base_train_long.py and
# chat_sft_long.py use a plain torch.optim.AdamW (MuonAdamW leaks 2-3
# GB/step at 128k ctx) and they never reduce the gradients themselves.
# Running them on more than one GPU therefore SILENTLY DIVERGES the ranks:
# each rank optimizes its own data shard alone, and rank 0's checkpoint --
# the one that is kept -- is the product of a single shard.
# This was measured, not assumed: a two-rank replica of that exact
# pattern (different data per rank, plain AdamW, no all-reduce) ends with
# rank-0 weights different from the cross-rank mean.
#
# => The 128k stages (7 and 8) and the touch-ups (9) are pinned to ONE GPU
#    in every profile. Stage 9 runs chat_sft_long.py too, so it inherits
#    the same restriction. See recipe/07_longctx.sh, 08_sft_long.sh and
#    09_touchup.sh, each of which calls require_single_gpu.
#
#    The original stage-7a/7b runs were also done on 1-2 GPUs, and the
#    shipped checkpoints (base_checkpoints/d28b-128k*) carry no optimizer
#    shards, which is the signature of a single-rank save.
#
# 2-GPU wall clock: the pretrain is 3.69x/2 = 1.85x the 4-GPU time, and
# every other stage is unchanged (the RL and 128k stages are not
# compute-bound on the GPU count). See docs/ETA_2GPU.md.
PROFILE="${PROFILE:-multi}"
case "$PROFILE" in
  1gpu)
    # One card for everything except the server.
    #
    # The card is AUTO-PICKED (a free one), not hardcoded: card occupancy on
    # this node changes by the hour, and a hardcoded index is how the
    # 2026-10-21 launch died (it wanted GPU 6; GPU 6 was at 100% with
    # another tenant's job). GPU_PRETRAIN etc. still override it.
    #
    # THE DEVICE BATCH IS 2, NOT 4. Measured with the real trainer on one
    # A100-40GB (2026-10-22):
    #     dbs 2 -> 1,868 ms/step, 17,525 tok/s, 41.4% MFU, peak 32,050 MiB
    #     dbs 3 -> OOM
    #     dbs 4 -> OOM (39.30 GiB in use, 201 MiB free)
    # dbs=2 is both the only one that fits AND 13.7% faster than the
    # dbs=4 figure the old note recorded (15,413 tok/s). The reason it
    # fits: the memory is dominated by the OPTIMIZER STATE, which at world
    # size 1 is entirely unsharded -- 15.0 GiB of fp32 moments for 2.02 B
    # parameters -- plus 3.76 GiB of bf16 weights and 3.76 GiB of bf16
    # gradients. The activations are the small part.
    if [ -z "${GPU_PRETRAIN:-}" ]; then GPU_PRETRAIN="$(pick_free_gpus 1)" || die "no free GPU for the 1gpu profile"; fi
    if [ -z "${GPU_SFT:-}" ];       then GPU_SFT="$GPU_PRETRAIN"; fi
    if [ -z "${GPU_LONGC:-}" ];     then GPU_LONGC="$GPU_PRETRAIN"; fi
    if [ -z "${GPU_TOUCHUP:-}" ];   then GPU_TOUCHUP="$GPU_PRETRAIN"; fi
    ;;
  2gpu)
    # Two cards. The 128k stages (7, 8) and the touch-ups (9) stay on ONE
    # card: they use base_train_long.py / chat_sft_long.py, which use a
    # plain AdamW and do not reduce gradients across ranks.
    #
    # THE DEVICE BATCH IS 2, NOT 4. Measured with the real trainer on two
    # A100-40GB (2026-10-22, GPUs 0+2, both exclusive):
    #     dbs 2 -> 964 ms/step, 33,941 tok/s, 40.1% MFU, peak 37,378 MiB/rank
    #     dbs 4 -> OOM (39.49 GiB in use, 21 MiB free)
    #
    # THIS OVERTurnS the earlier "2 GPUs is impossible" conclusion
    # (commit 0d4aafd). The old note compared the UNSHARDED 35.34 GiB
    # against a two-rank run, but the optimizer state is ZeRO-2 SHARDED, so
    # at world size 2 each rank holds 15.0/2 = 7.5 GiB of moments instead
    # of 15.0 GiB. Two ranks at dbs=2 fit, and give 1.94x the throughput
    # of one rank (33,941 vs 17,525 tok/s).
    if [ -z "${GPU_PRETRAIN:-}" ]; then GPU_PRETRAIN="$(pick_free_gpus 2)" || die "need 2 free GPUs for the 2gpu profile"; fi
    if [ -z "${GPU_SFT:-}" ];       then GPU_SFT="$GPU_PRETRAIN"; fi
    if [ -z "${GPU_LONGC:-}" ];     then GPU_LONGC="${GPU_LONGC:-$(echo "$GPU_PRETRAIN" | cut -d, -f1)}"; fi
    if [ -z "${GPU_TOUCHUP:-}" ];   then GPU_TOUCHUP="${GPU_TOUCHUP:-$(echo "$GPU_PRETRAIN" | cut -d, -f1)}"; fi
    ;;
  multi)
    GPU_PRETRAIN="${GPU_PRETRAIN:-4,5,6,7}"
    GPU_SFT="${GPU_SFT:-4}"
    GPU_LONGC="${GPU_LONGC:-5}"
    GPU_TOUCHUP="${GPU_TOUCHUP:-7}"
    ;;
  *)
    die "unknown PROFILE '$PROFILE' (want: multi, 2gpu, 1gpu)"
    ;;
esac
# The card the SERVED model runs on. It must NOT be one of the training
# cards: vLLM takes 94% of the card it is given (--gpu-memory-utilization
# 0.94), so it cannot share with a training process.
#
# In the default profile the training uses 4,5,6,7 and the server uses 0.
# Under PROFILE=2gpu the training uses two cards and the server needs a
# third; by the time stage 11 runs, the training has finished and its
# cards are free again, so the server can take one of them. Stage 11
# picks the first GPU that is not currently busy, preferring GPU 0.
GPU_SERVE="${GPU_SERVE:-0}"

# ---------------------------------------------------------------------------
# Ports. The guard owns the public port; vLLM is internal.
# ---------------------------------------------------------------------------
PUBLIC_PORT="${PUBLIC_PORT:-8000}"
UPSTREAM_PORT="${UPSTREAM_PORT:-8010}"

# ---------------------------------------------------------------------------
# THE TOKEN BUDGET.
#
# --target-param-data-ratio 8.0 is KEPT, and the schedule is not extended
# to Chinchilla's 20. The reason is the repo's own scaling laws, recorded
# in nanochat/dev/LOG.md over ~320 experiments:
#
#   * line 542: the optimal tokens:params ratio for THIS architecture is
#     ~10.5 in the Kaplan style (the most consistent of the three
#     parameter conventions, and the one that gives stable ~0.5
#     exponents for both params and tokens). The Chinchilla-style ratio
#     for the same architecture is 3.0-4.0, because the architecture is
#     parameter-bloated: "only about a quarter of parameters are now
#     weight projections and the vast majority is embedding tables".
#
#   * line 621: with the value embeddings in place the ratio "has HALVED
#     FROM 8 TO 4", and "any attempt to reduce the capacity of value
#     embeddings (param sharing, low rank, projections) fail. The model
#     wants many of them, and with all the capacity".
#
# Chinchilla's 20 is a different architecture's number and does not
# transfer here. Going from 8.0 to 20.0 would be 2.5x the tokens
# (694,401 steps, 186 h of pretrain on two GPUs) for a budget the repo's
# own sweeps argue is already past the optimum.
#
# What the corpus expansion DOES buy is the other half of the problem.
# The original run had 170 train shards: ~7.9 G trainable tokens for
# 9.10 G of training, i.e. ~1.15 epochs, so roughly 15% of the data was
# seen twice. With 500 train shards (~23.0 G trainable) the same 9.10 G
# budget is 0.40 epochs -- the same tokens, seen once instead of a
# seventh time. That is a real quality gain for the same wall clock, and
# it costs only disk (43 GB, ~11 min to download at 67.4 MB/s).
#
# Whether a longer run would help is a MEASUREMENT, not a guess: the
# recipe evaluates every 1,000 steps, so the shape of the val bpb curve
# in the log answers it.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Checkpoint tags. These are the names the stages read and write.
# ---------------------------------------------------------------------------
TAG_BASE="${TAG_BASE:-utak-base}"                      # stage 1 output
TAG_SFT_TOOLS="${TAG_SFT_TOOLS:-utak-sft-tools}"        # stage 2 output
TAG_WARMUP="${TAG_WARMUP:-utak-warmup}"                # stage 3 output
TAG_RL_WARMUP="${TAG_RL_WARMUP:-utak-rl-warmup}"       # stage 4 output
TAG_RL_PILOT="${TAG_RL_PILOT:-utak-rl-pilot}"          # stage 5 output
TAG_RL="${TAG_RL:-utak-rl}"                            # stage 6 output (rl dir)
TAG_LONGC="${TAG_LONGC:-utak-128k}"                    # stage 7 output
TAG_LONGC_FINAL="${TAG_LONGC_FINAL:-utak-128k-final}"  # stage 7b output
TAG_SFT="${TAG_SFT:-utak-sft}"                         # stage 8 output
TAG_TOUCH_A="${TAG_TOUCH_A:-utak-sft-a}"                # the v6a-equivalent anchor
TAG_TOUCH_B="${TAG_TOUCH_B:-utak-sft-b}"                # the v6a-equivalent: harness basics
TAG_TOUCH_C="${TAG_TOUCH_C:-utak-sft-c}"                # the SERVED checkpoint (v6j)

# ---------------------------------------------------------------------------
# Final checkpoint that gets exported and served.
# ---------------------------------------------------------------------------
FINAL_TAG="${FINAL_TAG:-$TAG_TOUCH_C}"
FINAL_STEP="${FINAL_STEP:-240}"

# ---------------------------------------------------------------------------
# Performance.
#
# These are the knobs that decide how fast the pipeline runs. They are here,
# not in the stage scripts, because the cost of a wrong value is silent: the
# run is just slow, and the model is the same.
#
# THE MEASURED BASELINE (the d28 pretrain, 4x A100-40GB, the run that
# produced the served model -- serve_logs/stage2/restart170.log):
#
#   step                445 ms          73,655 tok/s     43.5% BF16 MFU
#   peak memory         34,071 / 40,960 MiB  (83% of the card)
#   wall clock          57.35 h   but only 34.33 h of it was training
#   => 23.0 h (40.1%) was OVERHEAD, and it was almost all evaluation
#
# The GPU was never the bottleneck. 43.5% MFU is respectable for a 2B model
# on A100. The waste was in the evaluation schedule, the dataloader, and
# five trainers that never called torch.compile.
#
# ---------------------------------------------------------------------------
# How much validation to run. THIS IS THE BIGGEST SINGLE LEVER.
#
# Each eval is a full forward-only pass over --eval-tokens tokens. The
# measured cost of the original setting (41,943,040 tokens every 500 steps)
# was 147 s per eval, 556 evals, 25.4 h -- more than the training itself.
#
# How much precision do we actually need? From the same log, the standard
# deviation of a single bpb estimate at 41.9M tokens is 0.000065, while the
# real improvement in the tail of training is 0.00021 bpb per 500 steps.
# The original eval was therefore ~3x more precise than the signal it was
# trying to resolve. These defaults keep the noise comfortably BELOW the
# signal (so the curve still shows the true shape) while cutting the cost
# by ~10x:
#
#   eval tokens   sd of estimate   vs signal per 500 steps
#   41,943,040       0.000065          0.00021      (original: 3.2x signal)
#    4,194,304       0.000204          0.00021      (1.0x signal -- default)
#    2,097,152       0.000289          0.00021      (1.4x signal)
#
# Override EVAL_TOKENS=41943040 EVAL_EVERY=500 to reproduce the original run
# exactly. The default below is the fast setting.
PRETRAIN_EVAL_TOKENS="${PRETRAIN_EVAL_TOKENS:-4194304}"
PRETRAIN_EVAL_EVERY="${PRETRAIN_EVAL_EVERY:-1000}"
SFT_EVAL_TOKENS="${SFT_EVAL_TOKENS:-2097152}"
SFT_EVAL_EVERY="${SFT_EVAL_EVERY:-500}"

# The RL stages evaluate by GENERATING, not by a forward pass: 400 problems
# x device_batch_size samples, each up to 512 new tokens. The measured cost
# of one eval is ~25 minutes, and a training step is ~5.6 min (GSM8K) to
# ~11 min (agentic). So the eval is 4-5x the cost of a step, and it is the
# single biggest item in the RL budget.
#
# The eval size is NOT reduced: the measured confidence interval on pass@1
# is +/- 3.7 pt at n=400, and the effect the RL stage is chasing is of that
# same order (the pilot moved 6.7 pts, the curriculum 53 pts). Halving n
# would widen the CI to ~5 pt and hide exactly the signal being looked for.
#
# What is reduced is HOW OFTEN it runs, and that is set per stage below.
RL_EVAL_EVERY="${RL_EVAL_EVERY:-20}"
# Per-stage example counts. The GSM8K stage uses the full 400 (its
# confidence interval is +/- 3.7 pt); the two agentic stages use 150
# because each example runs a sandbox rollout and costs far more.
RL_EVAL_EXAMPLES_04="${RL_EVAL_EXAMPLES_04:-400}"
RL_EVAL_EXAMPLES_05="${RL_EVAL_EXAMPLES_05:-150}"
RL_EVAL_EXAMPLES_06="${RL_EVAL_EXAMPLES_06:-150}"

# The CORE metric (the few-shot benchmark) costs ~111 s per pass and is
# the only thing that produced a headline number in the original run, so it
# stays off during training and runs ONCE at the end.
#
# NOTE: base_train.py's condition is
#     core_metric_every > 0 and (last_step or step % every == 0)
# so the last step ALWAYS evaluates it, whatever this is set to. Setting
# it to 999999 means "only at the end" -- which is what the original run
# did (it passed --core-metric-every=999999). base_train.py's DEFAULT is
# 2000, which would fire 138 times over a 277k-step run (~4.3 h), so the
# recipe must always set this explicitly.
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-999999}"

# torch.compile. Two of the seven trainers had it (base_train, chat_sft);
# the other five did not. This is now passed explicitly to the ones that
# support it, and the long-context ones compile their inner block instead
# (they use per-block activation checkpointing, which full-graph compile
# cannot fuse -- see the note in 07_longctx.sh).
COMPILE_BASE="${COMPILE_BASE:-1}"

# Threading and interconnect. Without OMP_NUM_THREADS=1, torchrun sets it
# anyway but only after the process has already forked; with the tokenizer
# also using threads, the ranks fight over the same cores and the
# dataloader stalls. The original launch script set this by hand.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# The allocator hint. PYTORCH_ALLOC_CONF is the current name;
# PYTORCH_CUDA_ALLOC_CONF is the deprecated one (torch still honours it but
# prints a warning).
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# NCCL: NVLink-connected A100s on this node. These are the settings that
# avoid the per-iteration NCCL teardown that shows up as jitter in dt.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"

# ---------------------------------------------------------------------------
# Client / harness. The gates import the real baby-onit package from here.
# ---------------------------------------------------------------------------
BABY_ONIT_DIR="${BABY_ONIT_DIR:-/data/rowel/baby-onit}"
GATE_DATA_DIR="${GATE_DATA_DIR:-/data/rowel}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR="${LOG_DIR:-$NANOCHAT_ROOT/recipe/logs}"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Run a module from the nanochat checkout with the base dir exported.
# The perf env vars (OMP_NUM_THREADS, PYTORCH_ALLOC_CONF, NCCL_*) are
# exported by this file, so every stage inherits them automatically.
#
# STAGE_GPU: the GPU(s) this stage runs on. Every stage script sets it
# before calling run_mod, because a stage that does not pin its device
# lands on GPU 0 -- which is where the vLLM server lives (37.4 GiB of
# 40 GiB) and where a training run OOMs. A single-GPU profile is
# therefore a first-class path, not an accident.
# Refuse to run a stage on more than one GPU when that would be WRONG.
#
# The two long-context trainers (base_train_long.py, chat_sft_long.py) use a
# plain torch.optim.AdamW and never reduce the gradients across ranks, so
# on >1 GPU each rank optimises its own data shard alone and the ranks
# diverge silently. Rank 0's checkpoint -- the one that is kept -- is then
# the product of a single shard, not of the full batch.
#
# The trainers that DO use model.setup_optimizer() are fine: the gradient
# reduction lives inside MuonAdamW (nanochat/optim.py).
require_single_gpu() {
  local n
  n="$(stage_gpu_count)"
  if [ "$n" -gt 1 ]; then
    die "stage '$1' must run on ONE GPU, but STAGE_GPU=${STAGE_GPU:-unset} gives $n. \
The long-context trainers do not reduce gradients across ranks (plain AdamW, \
no DistributedDataParallel), so a multi-GPU run would silently diverge. \
Set GPU_LONGC to a single card."
  fi
}

# ---------------------------------------------------------------------------
# Warn if a card does not have enough room for the model.
#
# MEASURED on this node, d28, 2k ctx, real scripts.base_train
# (2026-10-22, A100-SXM4-40GB, device batch 2):
#
#   world size 1   peak 32,050 MiB   (37,378 MiB at world size 2 per rank)
#   world size 2   peak 37,378 MiB PER RANK
#
# The cost is the optimizer state, not the activations: 2.02 B parameters
# x (fp32 exp_avg + fp32 exp_avg_sq) = 15.0 GiB of moments, plus the bf16
# weights and gradients (3.8 GiB each), plus the reduce-scatter and
# all-gather buffers. The moments are ZeRO-2 SHARDED, so a two-rank run
# holds 7.5 GiB of moments per rank instead of 15.0 GiB -- which is why
# two ranks fit where the old note (which compared unsharded numbers)
# said they could not.
#
# A card shared with another tenant's jobs cannot hold d28 at world size 1
# (32 GiB of the 40 GiB card is needed and the tenant's processes are not
# ours to evict), so the profiles auto-pick a card with no foreign
# process on it.
# ---------------------------------------------------------------------------
PRETRAIN_MEM_NEEDED_MIB="${PRETRAIN_MEM_NEEDED_MIB:-38000}"

check_gpu_memory() {
  local stage="$1" gpus="$2" need="$3" g free
  for g in ${gpus//,/ }; do
    free=$(gpu_free_mib "$g")
    if [ "$free" -lt "$need" ]; then
      log "WARNING: stage $stage: GPU $g has $free MiB free but d28 needs about"
      log "WARNING:   $need MiB. The run will probably OOM at the first optimizer"
      log "WARNING:   step (the memory is optimizer state, not activations, so a"
      log "WARNING:   smaller device batch will NOT help)."
    fi
  done
}

# How many GPUs does this stage get? STAGE_GPU is a comma-separated list.
stage_gpu_count() {
  local g="${STAGE_GPU:-0}"
  [ -z "$g" ] && g=0
  awk -F, '{print NF}' <<< "$g"
}

# Run a module from the nanochat checkout with the base dir exported.
#
# If the stage has MORE THAN ONE GPU, the module is launched under
# torchrun with --nproc_per_node=<count>, which is what gives the stage a
# DDP process group. Every trainer in this repo is DDP-aware:
# compute_init() reads RANK/WORLD_SIZE and initialises NCCL, the
# dataloader shards the corpus by rank, and the gradient reduction
# happens inside MuonAdamW.
#
# The "--" separator is REQUIRED: torchrun treats everything after it as
# the script's own arguments rather than its own.
run_mod() {
  local logfile="$1"; shift
  local ngpus
  ngpus="$(stage_gpu_count)"
  log "RUN: $*  (log: $logfile, gpu: ${STAGE_GPU:-unset}, procs: $ngpus)"
  if [ "$ngpus" -gt 1 ]; then
    ( cd "$NANOCHAT_ROOT" \
      && NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" \
         CUDA_VISIBLE_DEVICES="${STAGE_GPU:-0}" \
         PYTHONPATH="$NANOCHAT_ROOT:${PYTHONPATH:-}" \
         "$PY" -m torch.distributed.run --standalone \
           --nproc_per_node="$ngpus" -m "$1" -- "${@:2}" ) > "$logfile" 2>&1
  else
    ( cd "$NANOCHAT_ROOT" \
      && NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" \
         CUDA_VISIBLE_DEVICES="${STAGE_GPU:-0}" \
         PYTHONPATH="$NANOCHAT_ROOT:${PYTHONPATH:-}" \
         "$PY" -m "$@" ) > "$logfile" 2>&1
  fi
  local rc=$?
  [ $rc -eq 0 ] || { log "FAILED (exit $rc): $*"; tail -30 "$logfile"; return $rc; }
  log "OK: $*"
  return 0
}

export MODEL_NAME MODEL_DIR_NAME REPO_TAG NANOCHAT_ROOT NANOCHAT_BASE_DIR \
       NANOCHAT_SIBLING_VENV NANOCHAT_PARENT_VENV PROFILE PY \
       VLLM_DIR VLLM_BIN VLLM_PY MODEL_DIR \
       GPU_PRETRAIN GPU_SFT GPU_LONGC GPU_TOUCHUP GPU_SERVE \
       PUBLIC_PORT UPSTREAM_PORT \
       TAG_BASE TAG_SFT TAG_WARMUP TAG_RL_WARMUP TAG_RL_PILOT TAG_RL \
       TAG_LONGC TAG_LONGC_FINAL \
       TAG_TOUCH_A TAG_TOUCH_B TAG_TOUCH_C FINAL_TAG FINAL_STEP \
       BABY_ONIT_DIR GATE_DATA_DIR LOG_DIR \
       PRETRAIN_EVAL_TOKENS PRETRAIN_EVAL_EVERY \
       SFT_EVAL_TOKENS SFT_EVAL_EVERY CORE_METRIC_EVERY COMPILE_BASE \
       RL_EVAL_EVERY RL_EVAL_EXAMPLES_04 RL_EVAL_EXAMPLES_05 RL_EVAL_EXAMPLES_06

# The gates ask the server for the model by name. Keep that in one place.
export UTAK_MODEL_NAME="$MODEL_NAME"
export UTAK_MODEL_DIR_NAME="$MODEL_DIR_NAME"
