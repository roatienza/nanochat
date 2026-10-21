#!/usr/bin/env bash
# Stage 6b — Agentic RL with weighted block curriculum + exemplar scaffold
# (scripts/chat_rl_curriculum.py), continuing from the stage-6 pilot's final
# checkpoint (chatrl_checkpoints/d28b-rl100/model_000177.pt, Pass@1 0.600).
#
# WHY THIS RUN (evidence):
# 1) Pilot plateau 120=140=150=160=177 at 0.600/0.667 was a capability wall, not
#    a sampling problem: 42/178 steps (24%) landed inside uncracked blocks where
#    all 16 samples of every example scored 0 -> advantage 0 -> zero gradient.
# 2) data_fix@L1 regressed 1.00 -> 0.00 because the sequential walk trained it
#    once (steps 36-47) and never revisited it. Replay fixes this.
# 3) Prose hints lift NOTHING on the frontier (stage6b_hint_probe: 0/16 with and
#    without, all 5 blocks). Worked examples DO (stage6b_exemplar_probe,
#    model_000177, training params): count_chars@L3 0.000 -> 1.000,
#    list_ops@L2 0.000 -> 1.000, math_fns@L3 0.000 -> 0.500; format_rows@L3 and
#    list_ops@L3 stay 0. So the scaffold is an exemplar, not a hint.
#
# DESIGN:
# - Weighted pool: 50% of examples from the 5 frontier blocks, 50% replayed
#   across the 10 mastered/regressed blocks (anti-forgetting + data_fix@L1
#   recovery). Reshuffled every epoch.
# - Exemplar scaffold: instance 0 of each frontier block (solution verified
#   against its own hidden tests) prepended to the user message, train time
#   only, annealed off at step 89 (half of 178). Eval never sees exemplars, so
#   the eval curve measures whether the pattern was internalized, not parroted.
# - Eval identical to the pilot: same val split, eval-every 20, 150 instances,
#   greedy pass@1/pass@2 + per-block breakdown -> directly comparable.
# - 178 steps (same compute as the pilot, ~31.5 h on one GPU).
#
# EXPECTED OUTCOMES (honest):
# - data_fix@L1 recovery -> 10/15 blocks (Pass@1 0.667) — near-certain.
# - count_chars@L3 / list_ops@L2 / math_fns@L3 transfer after anneal -> up to
#   12-13/15 (0.800-0.867) — the actual question this run answers.
# - format_rows@L3 / list_ops@L3 stay 0 (exemplar doesn't lift them).
#
# GPU: picked dynamically = most-free-memory, excluding GPU 0 (rowel LoRA) and
#   GPUs 0-4 when the 4-rank vLLM job is resident (the sort below handles it:
#   it picks the least-loaded GPU that is not in the exclude list).
set -euo pipefail
cd /data/rowel/nanochat
export NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2

# Pick the most-free GPU, excluding GPU 0 (LoRA job) and GPU 2 (vLLM server
# precedent — keep the exclusion even though it is currently free).
GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F, '$1 != 0 && $1 != 2' \
  | sort -t, -k2 -n | head -1 | awk -F, '{print $1}' | tr -d ' ')
echo "Selected GPU: $GPU"
export CUDA_VISIBLE_DEVICES=$GPU

# NOTE: '--' is REQUIRED — torchrun's argparse greedily prefix-matches '--run'
# against its own '--run-path' and errors without the separator.
exec .venv/bin/torchrun --standalone --nproc_per_node=1 -m scripts.chat_rl_curriculum -- \
  --run=stage6b-rl-curriculum \
  --model-tag=d28b-rl100 \
  --model-step=177 \
  --device-batch-size=2 \
  --examples-per-step=16 \
  --num-samples=16 \
  --num-epochs=1 \
  --max-new-tokens=512 \
  --temperature=1.0 \
  --eval-every=20 \
  --eval-examples=150 \
  --save-every=25 \
  --agentic-instances-per-level=200 \
  --agentic-levels=1,2,3 \
  --agentic-eval-split-frac=0.05 \
  --frontier-frac=0.5 \
  --exemplar-scaffold=1 \
  --exemplar-anneal-step=89 \
  --curriculum-seed=7