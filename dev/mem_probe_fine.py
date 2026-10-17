"""Fine-grained memory probe at seq 131072: where do the extra GB go?"""
import sys, torch
sys.path.insert(0, '/data/rowel/nanochat')
import torch.utils.checkpoint as cp
from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.common import COMPUTE_DTYPE

SEQ = 131072
cfg = GPTConfig(sequence_len=SEQ, vocab_size=32768, n_layer=6, n_head=14,
                n_kv_head=14, n_embd=1792, window_pattern="SSSL")
torch.cuda.set_device(0)
with torch.device("meta"):
    m = GPT(cfg)
m.to_empty(device="cuda")
m.init_weights()
m.train()

x = torch.randint(0, 32768, (1, SEQ), device="cuda")
y = torch.randint(0, 32768, (1, SEQ), device="cuda")

def gb():
    return torch.cuda.memory_allocated() / 1e9

print(f"start: {gb():.2f}")
x_e = m.transformer.wte(x).to(COMPUTE_DTYPE)
x_e = norm(x_e)
print(f"after embed: {gb():.2f}")
gate = m.smear_lambda.to(x_e.dtype) * torch.sigmoid(m.smear_gate(x_e[:, 1:, :24]))
xs = torch.cat([x_e[:, :1], x_e[:, 1:] + gate * x_e[:, :-1]], dim=1)
print(f"after smear: {gb():.2f}")
del gate
x0 = xs
cur = xs
cos_sin = (m.cos[:, :SEQ], m.sin[:, :SEQ])

for i, block in enumerate(m.transformer.h):
    m0 = gb()
    rl = m.resid_lambdas[i]
    xl = m.x0_lambdas[i]
    cur = rl * cur + xl * x0
    m1 = gb()
    ws = m.window_sizes[i]
    def blk_fn(x_, idx_, i=i, ws=ws):
        ve = m.value_embeds[str(i)](idx_).to(x_.dtype) if str(i) in m.value_embeds else None
        return m.transformer.h[i](x_, ve, cos_sin, ws, None)
    out = cp.checkpoint(blk_fn, cur, x, use_reentrant=True)
    m2 = gb()
    cur = out
    print(f"layer {i}: pre {m0:.2f} after-lambda {m1:.2f} after-ckpt {m2:.2f} (delta {m2-m0:.2f})")

print(f"after 6 layers: {gb():.2f}")