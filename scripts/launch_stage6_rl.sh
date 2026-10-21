#!/usr/bin/env bash
# Stage 6 — Agentic RL pilot on sandbox-verified environments (scripts/chat_rl_agentic.py).
#
# Base: Stage 5 RL best saved checkpoint (chatrl_checkpoints/d28b-warmup/model_000100.pt,
#   Pass@1 0.1600 test) hard-linked into chatsft_checkpoints/d28b-rl100/ so
#   load_model("sft", ...) resolves it (same hardlink precedent as v3 -> base_checkpoints).
#
# WHY a pilot, not the full stage: PLAN.md budgets agentic RL at 200–1,000+ node-h
# (2–12x the pretrain). One GPU cannot do that. This pilot answers the only question
# that matters before committing more: does the fresh-instance reward signal actually
# move pass@1? ~150 steps x 16 examples x 16 samples = 38,400 rollouts.
#
# Reward source: tasks/agentic.py — 5 families x 3 levels, instances generated fresh
# per index (never memorizable — the Stage 5 failure mode), verified by
# nanochat/execution.py sandbox (0/1 reward). Validated: 15/15 reference solutions
# pass, 900 wrong-solution trials leak 0.
#
# Env holdout: tail 5% of each (family, level) block reserved for eval, never trained.
#
# Hyperparameters (Stage 5 measured values carried over):
#   --device-batch-size 2    SFT/RL measured floor ~31-34 GiB at db=2 on 40 GB A100; db=4 OOMs.
#   --max-new-tokens 512     agentic solutions are longer than GSM8K chains; 512 measured OK.
#   --eval-every 20          resolve the pass@k curve; pilot is short.
#   --save-every 50          FINER than Stage 5's 100 — the step-40 peak (0.1775) was
#                            lost forever at save-every 100. 50 x ~4 GiB is affordable.
#   --num-epochs 1           2850 train instances / 16 per step = 178 steps; ~150-step
#                            stop-if-flat horizon lands inside epoch 1.
#   --eval-examples 150      the whole val split (150 instances) — full coverage, n large
#                            enough that ±3.7pt noise band shrinks to ~±8pt at n=150...
#                            actually n=150 CI is ±8pt; acceptable for a pilot, note it.
#
# Stop-if-flat: if Pass@1 is flat from step 20 to step ~60 (3 evals), stop — the env
#   mix or level weighting needs work, not more steps. Watch step 20 for the
#   per-family/level breakdown line (which families reward at all).
#
# GPU: picked dynamically = most-free-memory (map shifts fast). DO NOT pick GPU 2
#   (rowel's live vLLM server) or GPU 0 (rowel's LoRA job) — the sort below can
#   pick GPU 0 if it ever shows 0 MiB; the exclude list guards that.
set -euo pipefail
cd /data/rowel/nanochat
export NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2

# Pick the most-free GPU, excluding GPU 2 (vLLM server) and GPU 0 (LoRA job)
GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F, '$1 != 0 && $1 != 2' \
  | sort -t, -k2 -n | head -1 | awk -F, '{print $1}' | tr -d ' ')
echo "Selected GPU: $GPU"
export CUDA_VISIBLE_DEVICES=$GPU

# NOTE: '--' is REQUIRED — torchrun's argparse greedily prefix-matches '--run'
# against its own '--run-path' and errors without the separator.
exec .venv/bin/torchrun --standalone --nproc_per_node=1 -m scripts.chat_rl_agentic -- \
  --run=stage6-rl-agentic-pilot \
  --model-tag=d28b-rl100 \
  --model-step=100 \
  --device-batch-size=2 \
  --examples-per-step=16 \
  --num-samples=16 \
  --num-epochs=1 \
  --max-new-tokens=512 \
  --temperature=1.0 \
  --eval-every=20 \
  --eval-examples=150 \
  --save-every=50 \
  --agentic-instances-per-level=200 \
  --agentic-levels=1,2,3 \
  --agentic-eval-split-frac=0.05