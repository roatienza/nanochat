"""Memory probe: per-block activation checkpointing + chunked CE at seq 131072.

Replicates GPT.forward trunk with:
- ve recomputed INSIDE the checkpoint (saves 6.6 GB of stored ve tensors)
- per-block checkpointing (stores only block inputs)
- chunked lm_head + cross-entropy (8192-token chunks)

Measures actual peak CUDA memory for fwd+bwd on one A100-40GB.
"""
import sys, torch
sys.path.insert(0, '/data/rowel/nanochat')
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.common import COMPUTE_DTYPE

SEQ = 131072
cfg = GPTConfig(sequence_len=SEQ, vocab_size=32768, n_layer=28, n_head=14,
                n_kv_head=14, n_embd=1792, window_pattern="SSSL")
torch.cuda.set_device(0)
with torch.device("meta"):
    m = GPT(cfg)
m.to_empty(device="cuda")
m.init_weights()
m.train()

x = torch.randint(0, 32768, (1, SEQ), device="cuda")
y = torch.randint(0, 32768, (1, SEQ), device="cuda")

def trunk(m, idx, cos_sin):
    x = m.transformer.wte(idx).to(COMPUTE_DTYPE)
    x = norm(x)
    gate = m.smear_lambda.to(x.dtype) * torch.sigmoid(m.smear_gate(x[:, 1:, :24]))
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    x0 = x
    x_backout = None
    for i, block in enumerate(m.transformer.h):
        rl = m.resid_lambdas[i]
        xl = m.x0_lambdas[i]
        x = rl * x + xl * x0
        ws = m.window_sizes[i]
        def blk_fn(x_, idx_, i=i, ws=ws):
            ve = m.value_embeds[str(i)](idx_).to(x_.dtype) if str(i) in m.value_embeds else None
            return m.transformer.h[i](x_, ve, cos_sin, ws, None)
        x = cp.checkpoint(blk_fn, x, idx, use_reentrant=True)
        if i % 4 == 0:
            print(f"  layer {i}: mem {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
        if i == m.config.n_layer // 2:
            x_backout = x
    if x_backout is not None:
        x = x - m.backout_lambda.to(x.dtype) * x_backout
    return norm(x)

cos_sin = (m.cos[:, :SEQ], m.sin[:, :SEQ])
h = trunk(m, x, cos_sin)
print("trunk done. mem:", f"{torch.cuda.memory_allocated()/1e9:.2f}",
      "peak:", f"{torch.cuda.max_memory_allocated()/1e9:.2f}")

softcap = 15
total = 0.0
ntok = (y >= 0).sum()
CH = 8192
for s in range(0, SEQ, CH):
    e = min(s + CH, SEQ)
    logits = m.lm_head(h[:, s:e])[..., :32768].float()
    logits = softcap * torch.tanh(logits / softcap)
    lc = F.cross_entropy(logits.view(-1, 32768), y[:, s:e].reshape(-1),
                         ignore_index=-1, reduction='sum')
    total = total + lc
loss = total / ntok
print("loss", f"{loss.item():.4f}")
print("after fwd: mem", f"{torch.cuda.memory_allocated()/1e9:.2f}",
      "peak", f"{torch.cuda.max_memory_allocated()/1e9:.2f}")
loss.backward()
print("after bwd: mem", f"{torch.cuda.memory_allocated()/1e9:.2f}",
      "peak", f"{torch.cuda.max_memory_allocated()/1e9:.2f}")