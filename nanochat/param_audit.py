"""Ground-truth parameter counts from nanochat's REAL GPT class (meta device),
compared against reasoner-model/scripts/budget.py's model.

Run: .venv/bin/python param_audit.py
"""
import sys, importlib.util
from pathlib import Path
import torch
sys.argv = ["x"]  # silence argparse inside nanochat modules if any

from nanochat.gpt import GPT, GPTConfig, has_ve

# budget.py lives in the parent repo's scripts/ dir: <repo>/scripts/budget.py
# (this file is at <repo>/nanochat/param_audit.py). Previously hardcoded to an
# absolute path that only existed on one machine.
BUDGET_PY = Path(__file__).resolve().parent.parent / "scripts" / "budget.py"
spec = importlib.util.spec_from_file_location("budget", BUDGET_PY)
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)

HEAD_DIM, VOCAB, ASPECT = 128, 32768, 64

def real_params(depth, aspect=ASPECT):
    """Instantiate the actual model on meta device and count parameters."""
    base_dim = depth * aspect
    n_embd = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    n_head = n_embd // HEAD_DIM
    cfg = GPTConfig(sequence_len=2048, vocab_size=VOCAB, n_layer=depth,
                    n_head=n_head, n_kv_head=n_head, n_embd=n_embd)
    with torch.device("meta"):
        m = GPT(cfg)
    return sum(p.numel() for p in m.parameters()), n_embd

def budget_params(depth, aspect=ASPECT):
    return budget.n_params(depth, aspect)["params"]

print("=" * 88)
print("PARAMETER COUNT: nanochat's real GPT vs budget.py's analytic model")
print("=" * 88)
print(f"{'config':<8}{'n_embd':>8}{'REAL (M)':>12}{'budget.py (M)':>15}{'ratio':>9}   verdict")
print("-" * 88)
for d in (6, 12, 16, 18, 20, 22, 24, 26, 28, 32):
    r, ne = real_params(d)
    b = budget_params(d)
    ratio = b / r
    verdict = "OK" if abs(ratio - 1) < 0.05 else f"WRONG x{ratio:.2f}"
    print(f"d{d:<7}{ne:>8}{r/1e6:>12.1f}{b/1e6:>15.1f}{ratio:>9.2f}   {verdict}")

print()
print("=" * 88)
print("WHERE budget.py GOES WRONG")
print("=" * 88)
# decompose the budget.py value-embed term
for d in (12, 20, 26):
    r, ne = real_params(d)
    per_layer = 12 * ne**2
    n_ve_layers = sum(1 for i in range(d) if has_ve(i, d))
    ve_real = n_ve_layers * ne * VOCAB            # actual: half the layers, kv_dim == n_embd
    ve_budget = d * 2 * (ne // HEAD_DIM) * HEAD_DIM * VOCAB  # budget.py's formula
    emb = 2 * ne * VOCAB
    print(f"\nd{d}: n_embd={ne}, value-embed layers={n_ve_layers}/{d}")
    print(f"  matrices (12*n_embd^2 * depth) = {d*per_layer/1e6:9.1f}M   [both agree]")
    print(f"  token embed + lm_head (2*n_embd*V) = {emb/1e6:9.1f}M   [both agree]")
    print(f"  value embeds  REAL = {ve_real/1e6:9.1f}M")
    print(f"  value embeds budget = {ve_budget/1e6:9.1f}M   <-- over-counted "
          f"x{ve_budget/ve_real:.1f}")
    print(f"  sum real = {(d*per_layer + ve_real + emb)/1e6:.1f}M vs measured {r/1e6:.1f}M")

print()
print("=" * 88)
print("THE REAL MEMORY DRIVER: the logits tensor (vocab-proportional, width-independent)")
print("=" * 88)
for bs, seq in ((32, 2048), (16, 2048), (8, 2048), (4, 2048), (1, 2048)):
    logits_bf16 = bs * seq * VOCAB * 2 / 2**30
    logits_fp32 = bs * seq * VOCAB * 4 / 2**30
    print(f"  bs={bs:>3} seq={seq}: logits bf16 = {logits_bf16:6.2f} GiB, "
          f"fp32 upcast = {logits_fp32:6.2f} GiB")
print()
print("budget.py ACT_BYTES_PER_TOKEN_WIDTH = 12 bytes per (sample*token*width)")
for d in (12, 20):
    _, ne = real_params(d)
    print(f"  d{d}: activation model says {12*32*2048*ne/2**30:5.2f} GiB, "
          f"but logits alone are {32*2048*VOCAB*2/2**30:.2f} GiB (bf16) "
          f"-> under-counted ~{(32*2048*VOCAB*2)/(12*32*2048*ne):.1f}x")
