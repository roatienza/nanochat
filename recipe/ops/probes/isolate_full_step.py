#!/usr/bin/env python
"""Reproduce base_train's step EXACTLY, fresh vs resumed, in ONE process.

Same ORDER as base_train.py:
    line 163  model.load_state_dict(ckpt, strict=True, assign=True)
    line 248  model = torch.compile(model, dynamic=False)
    line 310  optimizer = model.setup_optimizer(...)
    line 321  optimizer.load_state_dict(optim_ckpt)

A SMALL model (d20 by default) so it fits on a card that another tenant
is also using -- the mechanism does not depend on the size.

Why this matters: every part measured EQUAL between fresh and resumed
(model fwd+bwd 156.5 vs 156.5 ms, optimizer.step 298.44 vs 298.43 ms,
dataloader 0.8 vs 0.6 ms, same 10 param groups) yet the full step was
2,296 ms fresh vs 4,050 ms resumed. The parts add up to the same number
in isolation, so the difference must be in how they COMPOSE -- which is
what this script measures.
"""
import time
import torch
import sys
import gc
import argparse

sys.path.insert(0, "/data/rowel/reasoner-model/nanochat")

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import COMPUTE_DTYPE

ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=int, default=20)
ap.add_argument("--dbs", type=int, default=2)
ap.add_argument("--total-batch", type=int, default=32768)
ap.add_argument("--steps", type=int, default=4)
args = ap.parse_args()

DEV = torch.device("cuda")
CKPT = "/data/rowel/nanochat_store/v2/base_checkpoints/speedprobe-src"
STEP = 3
T = 2048
HEAD_DIM = 128

model_dim = ((args.depth * 64 + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
n_head = model_dim // HEAD_DIM
GRAD_ACCUM = args.total_batch // (args.dbs * T)

cfg = GPTConfig(
    sequence_len=T, vocab_size=32768, n_layer=args.depth, n_head=n_head,
    n_kv_head=n_head, n_embd=model_dim, window_pattern="SSSL", ce_chunk=1024,
)

print(f"COMPUTE_DTYPE = {COMPUTE_DTYPE}")
print(f"depth={args.depth} n_embd={model_dim} dbs={args.dbs} grad_accum={GRAD_ACCUM}")
print()


def build(resume: bool, use_copy: bool = False):
    m = GPT(cfg)
    m.to_empty(device=DEV)
    m.init_weights()
    if resume:
        md = torch.load(f"{CKPT}/model_{STEP:06d}.pt", map_location=DEV, weights_only=False)
        if use_copy:
            # keep the ORIGINAL tensor objects; copy only the values
            with torch.no_grad():
                named = dict(m.named_parameters())
                for n, p in named.items():
                    if n in md:
                        p.copy_(md[n])
        else:
            m.load_state_dict(md, strict=True, assign=True)
    comp = torch.compile(m, dynamic=False)
    opt = m.setup_optimizer(
        unembedding_lr=0.008, embedding_lr=0.3, matrix_lr=0.02,
        scalar_lr=0.5, weight_decay=0.28,
    )
    if resume:
        od = torch.load(f"{CKPT}/optim_{STEP:06d}_rank0.pt", map_location=DEV, weights_only=False)
        # only the moments for the params that exist at this depth
        opt.load_state_dict(od)
    return m, comp, opt


def full_step(comp, opt, batches):
    t0 = time.time()
    for x, y in batches:
        loss = comp(x, y)
        loss = loss / GRAD_ACCUM
        loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return time.time() - t0


def make_batches(n):
    return [(torch.randint(0, 32768, (args.dbs, T), device=DEV),
             torch.randint(0, 32768, (args.dbs, T), device=DEV)) for _ in range(n)]


def run(label, resume, use_copy=False):
    m, comp, opt = build(resume, use_copy)
    batches = make_batches(GRAD_ACCUM)
    ts = []
    for i in range(args.steps):
        dt = full_step(comp, opt, batches)
        if i >= 1:
            ts.append(dt)
    med = sorted(ts)[len(ts) // 2]
    print(f"  {label:36s} median {med*1000:8.1f} ms   ("
          + ", ".join(f"{t*1000:.0f}" for t in ts) + ")")
    del m, comp, opt, batches
    gc.collect()
    torch.cuda.empty_cache()
    return med


print("=== the FULL base_train step, same process, same device ===")
t_fresh = run("FRESH (no resume)", resume=False)
t_res = run("RESUMED (assign=True)", resume=True)
t_copy = run("RESUMED (copy_ values, keep tensors)", resume=True, use_copy=True)

print("\n=== SUMMARY ===")
print(f"  fresh                          {t_fresh*1000:8.1f} ms")
print(f"  resumed (assign=True)          {t_res*1000:8.1f} ms  "
      f"({(t_res/t_fresh-1)*100:+.1f}%)")
print(f"  resumed (copy_ values)         {t_copy*1000:8.1f} ms  "
      f"({(t_copy/t_fresh-1)*100:+.1f}%)")
print(f"\n  the fix is worth "
      f"{((t_res/t_copy-1)*100):+.1f}% on the resumed path"
      if t_copy else "")
