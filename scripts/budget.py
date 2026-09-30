"""Feasibility + wall-clock budget for the reasoner-model ladder on 8 x A100 40GB.

Reproduces the tables in PLAN.md. Architecture assumptions are nanochat's own
defaults: aspect-ratio 64, head_dim 128, vocab 32768, Muon on matrices and
Adam on embeddings/unembeddings.

Run: python3 scripts/budget.py
"""

VOCAB = 32768
HEAD_DIM = 128

# A100 40GB bf16 dense tensor-core peak (no sparse doubling).
A100_BF16_TFLOPS = 312e12
GPUS = 8

# 30% MFU, calibrated from SmolLM3's reported ~14,000 tok/s/GPU on H100 at 29.4%
# MFU. A100 bf16 peak is ~0.32x H100 bf16 peak.
MFU = 0.30

# Optimizer bytes per parameter, no fp32 master copy needed beyond this:
#   Muon on matrices: fp32 master 4 + bf16 grad 2 + momentum 4
#   Adam on embeddings: fp32 master 4 + m 4 + v 4  (+ bf16 grad 2 counted once)
MATRIX_BYTES = 10
EMBED_BYTES = 12
GRAD_BYTES = 2
WEIGHT_BYTES = 2

# Empirical activation cost for nanochat's SDPA/flash path, in bytes per
# (sample * token * width). Coarse but sufficient to rank configs.
ACT_BYTES_PER_TOKEN_WIDTH = 12

MEM_HEADROOM_GIB = 38.0


def n_params(depth, aspect_ratio=64):
    base_dim = depth * aspect_ratio
    n_embd = -(-base_dim // HEAD_DIM) * HEAD_DIM  # round up to head_dim
    n_head = n_embd // HEAD_DIM
    # 12 * n_embd^2 per layer: attn (q, k, v, o) + mlp (c_fc, c_proj)
    per_layer = 12 * n_embd**2
    # per-layer value/unembedding embeddings
    value_embeds = depth * 2 * n_head * HEAD_DIM * VOCAB
    embed = 2 * n_embd * VOCAB
    return dict(
        depth=depth,
        aspect_ratio=aspect_ratio,
        n_embd=n_embd,
        n_head=n_head,
        params=depth * per_layer + value_embeds + embed,
    )


def peak_gib(depth, device_batch_size, max_seq_len=2048, aspect_ratio=64):
    m = n_params(depth, aspect_ratio)
    p = m["params"]
    embed = 2 * m["n_embd"] * VOCAB
    matrices = p - embed
    opt = matrices * MATRIX_BYTES + embed * EMBED_BYTES
    grads = p * GRAD_BYTES
    weights = p * WEIGHT_BYTES
    acts = ACT_BYTES_PER_TOKEN_WIDTH * device_batch_size * max_seq_len * m["n_embd"]
    return (opt + grads + weights + acts) / 2**30


def max_fitting_batch(depth, max_seq_len=2048, aspect_ratio=64):
    for bs in (32, 16, 8, 4, 2, 1):
        if peak_gib(depth, bs, max_seq_len, aspect_ratio) < MEM_HEADROOM_GIB:
            return bs
    return None


def node_hours(depth, ratio=12, aspect_ratio=64):
    p = n_params(depth, aspect_ratio)["params"]
    tokens = p * ratio
    flops = 6 * p * tokens
    return flops / (GPUS * A100_BF16_TFLOPS * MFU) / 3600


def main():
    print("=" * 72)
    print("Feasibility: nanochat defaults (aspect 64, head_dim 128, vocab 32768)")
    print(f"budget: {MEM_HEADROOM_GIB:.0f} GiB per A100 40GB, MFU {MFU:.0%}")
    print("=" * 72)
    print(f"{'config':<10}{'params':>10}{'best batch':>13}{'peak GiB':>11}   fits")
    for depth in (12, 16, 18, 20, 22, 24, 26, 28):
        p = n_params(depth)["params"]
        bs = max_fitting_batch(depth)
        if bs is None:
            print(f"d{depth:<9}{p/1e6:>8.0f}M{'--':>13}{'--':>11}   NO")
        else:
            g = peak_gib(depth, bs)
            print(f"d{depth:<9}{p/1e6:>8.0f}M{bs:>13}{g:>11.1f}   yes")

    print()
    print("=" * 72)
    print("Wall-clock on 8 x A100 40GB")
    print("=" * 72)
    print(f"{'stage':<30}{'params':>9}{'tokens':>10}{'node-hours':>12}")
    stages = [
        ("pretrain d12 (ablation)", 12, 12),
        ("pretrain d16", 16, 12),
        ("pretrain d18", 18, 12),
        ("pretrain d20 ratio-12", 20, 12),
        ("pretrain d20 ratio-8 (HEADLINE)", 20, 8),
        ("pretrain d20 aspect-32", 20, 12),
    ]
    for name, depth, ratio in stages:
        aspect = 32 if "aspect-32" in name else 64
        p = n_params(depth, aspect)["params"]
        tok = p * ratio
        print(f"{name:<30}{p/1e6:>7.0f}M{tok/1e9:>9.1f}B{node_hours(depth, ratio, aspect):>11.1f}")

    print()
    print("Cost of the headline run at published 8xA100 cloud rates:")
    h = node_hours(20, 8)
    for rate in (1.10, 1.50, 2.00):
        print(f"  ${rate:.2f}/GPU-hr (node ${rate*8:.2f}/hr) -> ${h*rate*8:,.0f}")


if __name__ == "__main__":
    main()
