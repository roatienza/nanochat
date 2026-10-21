"""Throughput probe: fwd+bwd time at seq 131072, K=1 checkpointing, on one A100."""
import sys, time, torch
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
x0_holder = [None]

def group_fn(m, idx, cos_sin, x_in, lo, hi):
    x = x_in
    for i in range(lo, hi):
        x = m.resid_lambdas[i] * x + m.x0_lambdas[i] * x0_holder[0]
        ws = m.window_sizes[i]
        ve = m.value_embeds[str(i)](idx).to(x.dtype) if str(i) in m.value_embeds else None
        x = m.transformer.h[i](x, ve, cos_sin, ws, None)
    return x

def trunk(m, idx, cos_sin):
    x = m.transformer.wte(idx).to(COMPUTE_DTYPE)
    x = norm(x)
    gate = m.smear_lambda.to(x.dtype) * torch.sigmoid(m.smear_gate(x[:, 1:, :24]))
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    x0_holder[0] = x
    x_backout = None
    n = m.config.n_layer
    for lo in range(0, n, 1):
        hi = min(lo + 1, n)
        x = cp.checkpoint(group_fn, m, idx, cos_sin, x, lo, hi, use_reentrant=False)
        if lo <= n // 2 < hi:
            x_backout = x
    if x_backout is not None:
        x = x - m.backout_lambda.to(x.dtype) * x_backout
    return norm(x)

cos_sin = (m.cos[:, :SEQ], m.sin[:, :SEQ])
softcap = 15
CH = 8192

def chunk_loss_fn(hc, yc):
    logits = m.lm_head(hc)[..., :32768].float()
    logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits.view(-1, 32768), yc.reshape(-1),
                           ignore_index=-1, reduction='sum')

def step():
    h = trunk(m, x, cos_sin)
    total = 0.0
    ntok = (y >= 0).sum()
    for s in range(0, SEQ, CH):
        e = min(s + CH, SEQ)
        lc = cp.checkpoint(chunk_loss_fn, h[:, s:e], y[:, s:e], use_reentrant=False)
        total = total + lc
    loss = total / ntok
    loss.backward()
    m.zero_grad(set_to_none=True)
    return loss.item()

# warmup
step()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

t0 = time.time()
N = 3
for _ in range(N):
    step()
torch.cuda.synchronize()
dt = (time.time() - t0) / N
print(f"fwd+bwd time: {dt:.2f} s per step (1 x {SEQ} tokens)")
print(f"throughput: {SEQ/dt:.0f} tokens/s per GPU")
print(f"peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")