"""Find what holds GPU memory after an optimizer step at seq 131072.

Runs 1 training step of the longctx path, then dumps the largest live CUDA
blocks via torch.cuda.memory_snapshot() with allocation size histograms.
"""
import sys, torch
sys.path.insert(0, '/data/rowel/nanochat')
sys.path.insert(0, '/data/rowel/nanochat/scripts')
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.common import COMPUTE_DTYPE, get_base_dir
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.tokenizer import get_tokenizer
import os

SEQ = 131072
os.environ.setdefault("NANOCHAT_BASE_DIR", "/data/rowel/nanochat_store/v2")
torch.cuda.set_device(0)

base_dir = get_base_dir()
src_dir = os.path.join(base_dir, "chatrl_checkpoints", "d28b-rl100")
model_data, _, meta_data = load_checkpoint(src_dir, 177, torch.device("cuda"), load_optimizer=False)
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
mc = dict(meta_data["model_config"])
mc["sequence_len"] = SEQ
mc["short_window"] = 512
mc.setdefault("rope_base", 100000)
cfg = GPTConfig(**mc)
with torch.device("meta"):
    model = GPT(cfg)
model.to_empty(device="cuda")
model.init_weights()
model.load_state_dict(model_data, strict=True, assign=True)
model.train()
del model_data

x0_holder = [None]

def _group_forward(model, idx, cos_sin, x_in, lo, hi):
    x = x_in
    for i in range(lo, hi):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0_holder[0]
        ws = model.window_sizes[i]
        ve = model.value_embeds[str(i)](idx).to(x.dtype) if str(i) in model.value_embeds else None
        blk = model.transformer.h[i]
        x = x + blk.attn(norm(x), ve, cos_sin, ws, None)
        xn = norm(x)
        out = torch.empty_like(xn)
        M = 16384
        for s in range(0, xn.size(1), M):
            e = min(s + M, xn.size(1))
            h = blk.mlp.c_fc(xn[:, s:e])
            h = F.relu(h).square()
            out[:, s:e] = blk.mlp.c_proj(h)
        x = x + out
    return x

def fwd_loss(model, idx, targets):
    cos_sin = model.cos[:, :SEQ], model.sin[:, :SEQ]
    x = model.transformer.wte(idx).to(COMPUTE_DTYPE)
    x = norm(x)
    gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(model.smear_gate(x[:, 1:, :24]))
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    x0_holder[0] = x
    x_backout = None
    n = model.config.n_layer
    for lo in range(n):
        x = cp.checkpoint(_group_forward, model, idx, cos_sin, x, lo, lo + 1, use_reentrant=False)
        if lo == n // 2:
            x_backout = x
    x = x - model.backout_lambda.to(x.dtype) * x_backout
    h = norm(x)
    total = 0.0
    ntok = (targets >= 0).sum()
    for s in range(0, SEQ, 4096):
        e = min(s + 4096, SEQ)
        def closs(hc, yc):
            logits = model.lm_head(hc)[..., :32768].float()
            logits = 15 * torch.tanh(logits / 15)
            return F.cross_entropy(logits.view(-1, 32768), yc.reshape(-1), ignore_index=-1, reduction='sum')
        total = total + cp.checkpoint(closs, h[:, s:e], targets[:, s:e], use_reentrant=False)
    return total / ntok

optimizer = model.setup_optimizer(unembedding_lr=0.0024, embedding_lr=0.09,
                                  scalar_lr=0.15, matrix_lr=0.006, weight_decay=0.028)

tokenizer = get_tokenizer()
loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, 1, SEQ, split="train", device="cuda")
x, y, _ = next(loader)

def gb():
    return torch.cuda.memory_allocated() / 1e9

print(f"weights: {gb():.2f}")
loss = fwd_loss(model, x, y)
print(f"after fwd: {gb():.2f}")
loss.backward()
print(f"after bwd: {gb():.2f} peak {torch.cuda.max_memory_allocated()/1e9:.2f}")
optimizer.step()
print(f"after step: {gb():.2f}")
model.zero_grad(set_to_none=True)
print(f"after zero_grad: {gb():.2f}")
optimizer.step()
print(f"after step: {gb():.2f}")
model.zero_grad(set_to_none=True)
print(f"after zero_grad: {gb():.2f}")

# stable offload (same approach as the training script)
_opt_stable = {}
def _iter():
    for k, v in optimizer.state.items():
        if torch.is_tensor(v):
            yield optimizer.state, k, v
        elif isinstance(v, dict):
            for kk, vv in list(v.items()):
                if torch.is_tensor(vv):
                    yield v, kk, vv

def offload():
    for sd, kk, vv in _iter():
        e = _opt_stable.get((id(sd), kk))
        if e is None:
            cpu = torch.empty_like(vv, device="cpu", pin_memory=True)
            cpu.copy_(vv, non_blocking=True)
            _opt_stable[(id(sd), kk)] = {"gpu": vv, "cpu": cpu}
            sd[kk] = cpu
        else:
            e["cpu"].copy_(e["gpu"], non_blocking=True)
            sd[kk] = e["cpu"]

def reload():
    for sd, kk, vv in _iter():
        e = _opt_stable.get((id(sd), kk))
        if e is not None and vv.device.type == "cpu":
            e["gpu"].copy_(e["cpu"], non_blocking=True)
            sd[kk] = e["gpu"]

offload()
print(f"after offload: {gb():.2f}")
reload()
print(f"after reload: {gb():.2f}")
optimizer.step()
offload()
print(f"after step2+offload: {gb():.2f}")
model.zero_grad(set_to_none=True)
print(f"final: {gb():.2f}")

# dump largest live blocks
snap = torch.cuda.memory_snapshot()
blocks = []
for seg in snap:
    for b in seg.get("blocks", []):
        if b.get("state") == "active_allocated" and b["size"] > 1e8:
            blocks.append(b["size"])
blocks.sort(reverse=True)
from collections import Counter
c = Counter(blocks)
print("largest active blocks (size -> count):")
for size, cnt in c.most_common(15):
    print(f"  {size/1e9:.3f} GB x {cnt}")
print(f"sum of >100MB blocks: {sum(blocks)/1e9:.2f} GB")