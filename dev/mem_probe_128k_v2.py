"""Memory probe v2: checkpoint every K blocks (lambda folded inside), chunked CE.

Measures actual peak CUDA memory for fwd+bwd at seq 131072 on one A100-40GB.
"""
import sys, torch
sys.path.insert(0, '/data/rowel/nanochat')
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.common import COMPUTE_DTYPE

SEQ = 131072
K = int(sys.argv[1]) if len(sys.argv) > 1 else 2  # checkpoint interval
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

def run_group(m, idx, cos_sin, x_in, lo, hi):
    """Run blocks [lo, hi) with lambdas, no checkpoint inside."""
    x = x_in
    x0 = None  # x0 handled outside; here we get pre-lambda input
    for i in range(lo, hi):
        rl = m.resid_lambdas[i]
        xl = m.x0_lambdas[i]
        # x0 blending uses the ORIGINAL embedding — must be passed in
        x = rl * x + xl * x_in_orig[0]
        ws = m.window_sizes[i]
        ve = m.value_embeds[str(i)](idx).to(x.dtype) if str(i) in m.value_embeds else None
        x = m.transformer.h[i](x, ve, cos_sin, ws, None)
    return x

# simpler: x0 passed via closure
x0_global = None

def group_fn(m, idx, cos_sin, x_in, lo, hi):
    x = x_in
    for i in range(lo, hi):
        x = m.resid_lambdas[i] * x + m.x0_lambdas[i] * x0_global
        ws = m.window_sizes[i]
        ve = m.value_embeds[str(i)](idx).to(x.dtype) if str(i) in m.value_embeds else None
        x = m.transformer.h[i](x, ve, cos_sin, ws, None)
    return x

def trunk(m, idx, cos_sin):
    global x0_global
    x = m.transformer.wte(idx).to(COMPUTE_DTYPE)
    x = norm(x)
    gate = m.smear_lambda.to(x.dtype) * torch.sigmoid(m.smear_gate(x[:, 1:, :24]))
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    x0_global = x
    x_backout = None
    n = m.config.n_layer
    for lo in range(0, n, K):
        hi = min(lo + K, n)
        x = cp.checkpoint(group_fn, m, idx, cos_sin, x, lo, hi, use_reentrant=False)
        if lo <= n // 2 < hi:
            x_backout = x
    if x_backout is not None:
        x = x - m.backout_lambda.to(x.dtype) * x_backout
    return norm(x)

cos_sin = (m.cos[:, :SEQ], m.sin[:, :SEQ])
h = trunk(m, x, cos_sin)
print(f"K={K} trunk done. mem:", f"{torch.cuda.memory_allocated()/1e9:.2f}",
      "peak:", f"{torch.cuda.max_memory_allocated()/1e9:.2f}")

softcap = 15
total = 0.0
ntok = (y >= 0).sum()
CH = 2048

def chunk_loss_fn(hc, yc):
    logits = m.lm_head(hc)[..., :32768].float()
    logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits.view(-1, 32768), yc.reshape(-1),
                           ignore_index=-1, reduction='sum')

for s in range(0, SEQ, CH):
    e = min(s + CH, SEQ)
    # checkpoint the chunk: logits recomputed in backward, only loss kept
    lc = cp.checkpoint(chunk_loss_fn, h[:, s:e], y[:, s:e], use_reentrant=False)
    total = total + lc
loss = total / ntok
print("loss", f"{loss.item():.4f}")
print("after fwd: mem", f"{torch.cuda.memory_allocated()/1e9:.2f}",
      "peak", f"{torch.cuda.max_memory_allocated()/1e9:.2f}")
loss.backward()
print("after bwd: mem", f"{torch.cuda.memory_allocated()/1e9:.2f}",
      "peak", f"{torch.cuda.max_memory_allocated()/1e9:.2f}")