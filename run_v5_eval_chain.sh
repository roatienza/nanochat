#!/usr/bin/env bash
# v5 export -> serve -> conformant-eval chain. Waits for stage7b-v5 step 800.
# Serve flags = the exact 128k config from vllm_serve_8000_128k.log (v4-era):
#   0.94 mem, max_model_len 131072, max_num_batched_tokens 131072, max_num_seqs 1,
#   no prefix caching, no chunked prefill, eager, nanochat tool parser,
#   VLLM_USE_FLASHINFER_SAMPLER=0 (REQUIRED: flashinfer sampler JIT needs ninja).
set -u
BASE=/data/rowel/nanochat_store/v2
VLLMD=$BASE/vllm_d28b
OUT=$VLLMD/nanochat-d28b-rl100-128k-v5
LOG=/data/rowel/nanochat/v5_eval_chain.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== v5 eval chain start: waiting for step 800 ==="
while true; do
  if ls $BASE/chatsft_checkpoints/d28b-128k-sft-v5/model_000800.pt >/dev/null 2>&1; then
    log "step-800 checkpoint found"; break
  fi
  if ! pgrep -f "chat_sft_long.*stage7b-v5-resume" >/dev/null; then
    log "WARNING: trainer process gone; step-800 ckpt present? $(ls $BASE/chatsft_checkpoints/d28b-128k-sft-v5/model_000800.pt 2>/dev/null || echo NO)"
    ls $BASE/chatsft_checkpoints/d28b-128k-sft-v5/model_*.pt 2>/dev/null | tail -3 >> "$LOG"
    if ls $BASE/chatsft_checkpoints/d28b-128k-sft-v5/model_000800.pt >/dev/null 2>&1; then break; fi
    log "ABORT: trainer died without step 800"; exit 1
  fi
  sleep 120
done

log "--- export ---"
cd $VLLMD
if /data/rowel/nanochat/.venv/bin/python export_hf_128k.py --phase sft \
     --tag d28b-128k-sft-v5 --step 800 --out $OUT > $VLLMD/export_v5.log 2>&1; then
  log "export DONE: $(grep -E 'tensors:|total params' $VLLMD/export_v5.log | tail -2)"
else
  log "export FAILED"; tail -10 $VLLMD/export_v5.log >> "$LOG"; exit 1
fi
# aux files (tokenizer + template + remote-code modules) from the v4 dir
for f in tokenizer.json tokenizer_config.json configuration_nanochat.py modeling_nanochat_d28.py; do
  cp -p $VLLMD/nanochat-d28b-rl100-128k-v4/$f $OUT/$f
done
# v4 template: tool results as user turns (toolfollowup-trained). Master copy
# lives in the repo; the v5 dir's template was patched in place for v5b.
cp -p /data/rowel/nanochat/chat_template.jinja $OUT/chat_template.jinja
ls -la $OUT >> "$LOG"

log "--- serve on GPU 5, port 8000 ---"
cd $VLLMD
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=5 ./venv-vllm/bin/vllm serve \
  nanochat-d28b-rl100-128k-v5 \
  --port 8000 --trust-remote-code --enforce-eager --gpu-memory-utilization 0.94 \
  --max-model-len 131072 --max-num-batched-tokens 131072 --max-num-seqs 1 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --served-model-name nanochat-d28b-rl100 \
  --enable-auto-tool-choice --tool-call-parser nanochat \
  --tool-parser-plugin $VLLMD/nanochat_tool_parser.py \
  > $VLLMD/vllm_serve_8000_128k_v5.log 2>&1 &
SRV=$!
log "vllm pid $SRV; waiting for /health"
ok=""
for i in $(seq 1 60); do
  sleep 10
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then ok=yes; break; fi
  if ! kill -0 $SRV 2>/dev/null; then log "vllm process died"; break; fi
done
if [ -z "$ok" ]; then log "SERVER FAILED to become healthy"; tail -20 $VLLMD/vllm_serve_8000_128k_v5.log >> "$LOG"; exit 1; fi
log "server healthy after ~$((i*10))s"

log "--- conformant eval ---"
rm -f /data/rowel/test_128k_v5_results.txt
if /data/rowel/nanochat/.venv/bin/python $VLLMD/test_128k_v5_conformant.py \
     > $VLLMD/test_128k_v5_conformant.log 2>&1; then
  log "conformant eval DONE"; cat /data/rowel/test_128k_v5_results.txt >> "$LOG"
else
  log "conformant eval FAILED"; tail -10 $VLLMD/test_128k_v5_conformant.log >> "$LOG"
fi
log "=== v5 eval chain end (server left running on port 8000) ==="