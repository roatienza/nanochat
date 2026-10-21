#!/usr/bin/env bash
# Stage 5a — GSM8K SFT warm-up (pre-RL). Goal: lift the RL baseline (pass@1 0.5%)
# so GRPO has denser signal (more questions partially-correct => non-zero advantage).
#
# WHY: inspect_gen.log shows v3 does GSM8K arithmetic IN ITS HEAD (plain text),
# never uses the <|python_start|> calculator format, and makes arithmetic errors
# (Q0: "twice as many as Mimi" -> "2 x 2 = 4" instead of "2 x 24 = 48"). GSM8K SFT
# teaches step-by-step solutions that SIMULATE the calculator (exact intermediates).
#
# BASE: v3 SFT checkpoint (model_013630.pt, val_bpb 0.3022, hermes exact 40.7%).
#   NOTE: the pipeline is NOT cumulative — every SFT run loads from base_checkpoints
#   (the pretrain base). chat_sft.py:104 hardcodes load_model("base", ...). So the v3
#   SFT checkpoint (in chatsft_checkpoints/d28b-v2/) is hard-linked into
#   base_checkpoints/d28b-warmup/model_013630.pt so chat_sft.py can load it as its base.
#
# MIXTURE: GSM8K x10 (74,730 rows, 13.5M tok) + ToolCalling x4 (4,144 rows, 3.2M tok)
#   = 78,874 rows ~ 16.7M tok ~ 510 steps (32,768 tok/step). NO SmolTalk/MMLU/ToolACE
#   so GSM8K dominates. ToolCalling x4 (matches v3's tool emphasis) preserves the
#   tool-call format (safety net so we don't destroy v3's 40.7% hermes). Requires the
#   --smoltalk-epochs code change.
#
# OPTIMIZER: fresh (load-optimizer=0). v3's optimizer momentum was tuned for the v3
#   mixture (MMLU x1, GSM8K x1, ToolCalling x4, ToolACE x6); a fresh optimizer is
#   cleaner for this different GSM8K-dominant mixture.
#
# GPU: picked dynamically = most-free-memory (map shifts fast). db=2 floor ~31-34 GiB
#   on 40 GB A100. Single GPU: 461 steps x ~1.8 s ~ 15 min + ~9 evals ~ 20-25 min total.
set -euo pipefail
cd /data/rowel/nanochat
export NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2

# Pick the most-free GPU (lowest memory.used, tie-break lowest util)
GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | sort -t, -k2 -n | head -1 | awk -F, '{print $1}' | tr -d ' ')
echo "Selected GPU: $GPU"
export CUDA_VISIBLE_DEVICES=$GPU

# NOTE: '--' is REQUIRED (torchrun argparse prefix-matches '--run' against '--run-path').
exec .venv/bin/torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
  --run=stage5a-gsm8k-warmup-d28b-v3 \
  --model-tag=d28b-warmup \
  --model-step=13630 \
  --load-optimizer=0 \
  --smoltalk-epochs=0 \
  --mmlu-epochs=0 \
  --gsm8k-epochs=10 \
  --toolcall-epochs=4 \
  --toolace-epochs=0 \
  --num-iterations=-1 \
  --device-batch-size=2 \
  --total-batch-size=32768 \
  --init-lr-frac=0.8 \
  --warmup-ratio=0 \
  --warmdown-ratio=0.5 \
  --final-lr-frac=0 \
  --eval-every=50 \
  --eval-tokens=2097152 \
  --save-every=100 \
  --chatcore-every=-1
