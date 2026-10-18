#!/usr/bin/env bash
# v5z finalize: wait for the training run to finish, pack step 240, serve on
# 8000, run the real gate. Runs detached; logs to v5z_finalize.log.
set -u
VLLMD=/data/rowel/nanochat_store/v2/vllm_d28b
LOG=/data/rowel/nanochat/v5z_finalize.log
{
# 1. wait for the v5z step-240 checkpoint (training writes it last)
for i in $(seq 1 240); do
  if [ -f /home/rowel/.cache/nanochat/chatsft_checkpoints/d28b-128k-sft-v5z/meta_000240.json ]; then
    echo "ckpt 240 present at $(date +%H:%M:%S)"; break
  fi
  sleep 30
done
if [ ! -f /home/rowel/.cache/nanochat/chatsft_checkpoints/d28b-128k-sft-v5z/meta_000240.json ]; then
  echo "TIMED OUT waiting for ckpt 240"; exit 1
fi
# 2. stop the current v5v/v5y server on 8000 if any (it holds the port)
#    (the harness swaps by same served name; nothing else to do)
# 3. pack
bash $VLLMD/pack_v5z.sh >> $LOG 2>&1 || { echo "pack FAILED"; exit 1; }
echo "packed at $(date +%H:%M:%S)"
# 4. serve on 8000
nohup bash $VLLMD/serve_v5z.sh > /dev/null 2>&1 &
for i in $(seq 1 60); do
  sleep 15
  if curl -s --max-time 5 http://localhost:8000/v1/models | grep -q v5z; then
    echo "serving v5z at $(date +%H:%M:%S)"; break
  fi
done
# 5. gate
python3 $VLLMD/verify_v5x.py 8000 >> $LOG 2>&1
echo "gate done at $(date +%H:%M:%S)"
} >> $LOG 2>&1