"""Measure REAL peak GPU memory for nanochat at (depth, device_batch_size).

Runs a single forward+backward, after one warmup step, and reports
torch.cuda.max_memory_allocated. Uses one GPU.

Run: .venv/bin/python mem_probe.py 12 32
"""
import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse, torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import MuonAdamW  # noqa

HEAD_DIM, VOCAB, ASPECT, SEQ = 128, 32768, 64, 2048

p = argparse.ArgumentParser()
p.add_argument("--depth", type=int, required=True)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--aspect-ratio", type=int, default=ASPECT)
p.add_argument("--gpu", type=int, default=0)
a = p.parse_args()

dev = "cuda"
torch.cuda.set_device(a.gpu)
torch.cuda.reset_peak_memory_stats()

base_dim = a.depth * a.aspect_ratio
n_embd = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
n_head = n_embd // HEAD_DIM
cfg = GPTConfig(sequence_len=SEQ, vocab_size=VOCAB, n_layer=a.depth,
                n_head=n_head, n_kv_head=n_head, n_embd=n_embd)
with torch.device("meta"):
    model = GPT(cfg)
model.to_empty(device=dev)
model.init_weights()
model.train()

params = sum(q.numel() for q in model.parameters())
emb = 2 * n_embd * VOCAB
mat = params - emb

# use nanochat's own optimizer to get the true state footprint
opt = model.setup_optimizer(
    unembedding_lr=0.008, embedding_lr=0.3, scalar_lr=0.5,
    matrix_lr=0.02, weight_decay=0.28,
)

x = torch.randint(0, VOCAB, (a.batch_size, SEQ), device=dev)
y = torch.randint(0, VOCAB, (a.batch_size, SEQ), device=dev)

for i in range(2):  # warmup then measure
    loss = model(x, y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()
loss = model(x, y)
loss.backward()
opt.step()
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() / 2**30

logits_gib = a.batch_size * SEQ * VOCAB * 2 / 2**30
print(f"d{a.depth} aspect{a.aspect_ratio} bs={a.batch_size}  params={params/1e6:.1f}M  "
      f"PEAK={peak:.2f} GiB   (logits bf16 alone = {logits_gib:.2f} GiB)  "
      f"{'FITS' if peak < 38.0 else 'OOM'}")
