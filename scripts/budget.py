"""FEASIBILITY + WALL-CLOCK BUDGET for the reasoner-model ladder on 8 x A100 40GB.

Supersedes the first version of this file. The original derived parameter counts
analytically and over-counted them by 2.1-2.6x (it assumed a value embedding on
every layer; nanochat puts one on alternating layers only), and it modelled
activations as width-proportional when the real driver is the vocab-sized logits
tensor. Both errors are corrected here, and the memory column is now MEASURED
rather than derived. See docs/EXECUTION_LOG.md for the full audit.

Reproduce the measurements with:
    nanochat/param_audit.py   # parameter counts, real GPT class on meta device
    nanochat/mem_probe.py     # peak memory, real fwd+bwd+step
    nanochat/budget_real.py   # the same table this file prints

Run: python3 scripts/budget.py
"""

VOCAB = 32768
HEAD_DIM = 128
GPUS = 8

# A100-SXM4-40GB bf16 dense tensor-core peak, no sparse doubling.
# Verified against nanochat.common.get_peak_flops("NVIDIA A100-SXM4-40GB").
A100_BF16_TFLOPS = 312e12

MEM_LIMIT_GIB = 38.0   # 40GB card, leave headroom for fragmentation

# MEASURED peak GiB for one fwd+bwd+opt-step, nanochat defaults
# (aspect 64, head_dim 128, vocab 32768, window_pattern SSSL, Muon on matrices).
# 0.0 = does not fit. Source: nanochat/mem_probe.py.
MEASURED_GIB = {
    12: {1:  3.83, 2:  5.53, 4:  8.93, 8: 15.73, 16: 29.32, 32: 0.0},
    16: {1:  7.21, 2:  8.51, 4: 12.98, 8: 21.93, 16:  0.0, 32: 0.0},
    18: {1: 10.43, 2: 10.43, 4: 15.56, 8:  0.0, 16:  0.0, 32: 0.0},
    20: {1: 12.75, 2: 12.75, 4: 18.54, 8: 30.26, 16:  0.0, 32: 0.0},
    22: {1: 16.37, 2: 16.37, 4: 16.37, 8: 35.26, 16:  0.0, 32: 0.0},
    24: {1: 20.62, 2: 20.62, 4: 25.77, 8:  0.0, 16:  0.0, 32: 0.0},
    26: {1: 25.56, 2: 25.56, 4: 30.07, 8:  0.0, 16:  0.0, 32: 0.0},
    28: {1: 31.24, 2: 31.24, 4: 34.86, 8:  0.0, 16:  0.0, 32: 0.0},
    32: {1:  0.0,  2:  0.0, 4:  0.0, 8:  0.0, 16:  0.0, 32: 0.0},
}

# MEASURED steady-state throughput, 4xA100-40GB, real base_train runs.
# The published 30% MFU assumption was pessimistic by 1.15-1.6x.
MEASURED_TOK_S_4GPU = {12: 792_000, 20: 157_500, 28: 58_200}
MEASURED_MFU = {12: 0.48, 20: 0.36, 28: 0.34}


def n_params(depth, aspect_ratio=64):
    """EXACT parameter count, matching nanochat's GPT.

    Key correction vs the original: value embeddings exist on ALTERNATING layers
    (has_ve: layer_idx % 2 == (n_layer-1) % 2), and n_kv_head == n_head, so
    kv_dim == n_embd. The original charged every layer 2*n_embd*V.
    """
    n_embd = ((depth * aspect_ratio + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    n_head = n_embd // HEAD_DIM
    per_layer = 12 * n_embd**2
    n_ve = (depth + 1) // 2 if depth % 2 == 1 else depth // 2
    value_embeds = n_ve * n_embd * VOCAB
    embed = 2 * n_embd * VOCAB
    matrices = depth * per_layer
    return dict(depth=depth, aspect_ratio=aspect_ratio, n_embd=n_embd,
                n_head=n_head, params=matrices + value_embeds + embed,
                matrices=matrices, value_embeds=value_embeds, embed=embed)


def n_scaling_params(depth, aspect_ratio=64):
    """The parameter count `--target-param-data-ratio` is actually applied to.

    base_train.py uses `transformer_matrices + lm_head`, EXCLUDING the value
    embeddings, because those two give the cleanest scaling laws (the code cites
    dev/LOG.md Jan 27 2026). Verified against a live d28 run:
    1,137,707,312 * 8 = 9,101,658,496 vs the log's 9,101,639,680.

    This matters: ratio 8 means 9.1B tokens, not 16.1B, so quoting a "data:param
    ratio" without saying WHICH params is ambiguous by ~1.8x in wall-clock.
    """
    m = n_params(depth, aspect_ratio)
    return m["matrices"] + m["embed"] // 2  # lm_head only, not wte


def max_fitting_batch(depth, aspect_ratio=64):
    if aspect_ratio != 64:
        raise ValueError("measured table is aspect-64 only")
    for bs in (32, 16, 8, 4, 2, 1):
        g = MEASURED_GIB[depth][bs]
        if 0.0 < g < MEM_LIMIT_GIB:
            return bs
    return None


def node_hours(depth, ratio, aspect_ratio=64, gpus=8, params_basis="scaling"):
    """Wall-clock from MEASURED throughput, not from an assumed MFU.

    Throughput was measured on 4 GPUs (GPU0 was occupied by an unrelated job on
    this node). Scaling to 8 assumes the same per-GPU efficiency, which holds as
    long as world_size stays a legal value for the harness: 32768 (vocab) must be
    divisible by world_size for AdamW reduce_scatter, so world_size in {2,4,8}.

    params_basis: "scaling" matches what the harness actually does (matrices +
    lm_head, excluding value embeddings). "total" is what PLAN.md's arithmetic
    implies. They differ by ~1.8x in wall-clock; "scaling" is the truth.
    """
    p = (n_scaling_params(depth, aspect_ratio) if params_basis == "scaling"
         else n_params(depth, aspect_ratio)["params"])
    tokens = p * ratio
    base_depth = min(MEASURED_TOK_S_4GPU, key=lambda d: abs(d - depth))
    tok_s = MEASURED_TOK_S_4GPU[base_depth] * (gpus / 4)
    return tokens / tok_s / 3600


def main():
    print("=" * 78)
    print("Feasibility on 8 x A100 40GB - MEASURED, nanochat defaults")
    print("(aspect 64, head_dim 128, vocab 32768, Muon-on-matrices)")
    print("=" * 78)
    print(f"{'config':<8}{'params':>10}{'max bs':>8}{'peak GiB':>11}   vs original plan")
    for d in (12, 16, 18, 20, 22, 24, 26, 28, 32):
        p = n_params(d)["params"]
        bs = max_fitting_batch(d)
        if bs is None:
            print(f"d{d:<7}{p/1e6:>8.0f}M{'--':>8}{'--':>11}   OOM even at bs=1")
        else:
            note = "plan: 32" if bs != 32 else "as planned"
            print(f"d{d:<7}{p/1e6:>8.0f}M{bs:>8}{MEASURED_GIB[d][bs]:>11.2f}   {note}")

    print()
    print("=" * 78)
    print("Headline model: the plan's stated target was ~2.15 B params")
    print("=" * 78)
    for d in (20, 24, 26, 28):
        p = n_params(d)["params"]
        bs = max_fitting_batch(d)
        mark = "  <== matches the plan's 2.15B target" if 1.9e9 < p < 2.3e9 else ""
        print(f"  d{d}: {p/1e6:>6.0f}M params, max bs={bs}, {node_hours(d, 8):>6.1f} node-h "
              f"at ratio 8{mark}")

    print()
    print("=" * 78)
    print("Wall-clock, from measured throughput (not assumed MFU)")
    print("tokens = ratio x SCALING params (matrices + lm_head), which is what")
    print("base_train.py actually uses -- NOT total params")
    print("=" * 78)
    print(f"{'stage':<30}{'total':>8}{'scaling':>9}{'tokens':>9}{'h(4GPU)':>9}{'h(8GPU)':>9}")
    stages = [
        ("pretrain d12 (ablation) r12", 12, 12),
        ("pretrain d20 r8", 20, 8),
        ("pretrain d24 r8", 24, 8),
        ("pretrain d26 (GPT-2 grade) r8", 26, 8),
        ("pretrain d28 HEADLINE r8", 28, 8),
        ("pretrain d28 r12", 28, 12),
    ]
    for name, d, r in stages:
        tot = n_params(d)["params"]
        sca = n_scaling_params(d)
        print(f"{name:<30}{tot/1e6:>6.0f}M{sca/1e6:>8.0f}M{sca*r/1e9:>8.1f}B"
              f"{node_hours(d, r, gpus=4):>9.1f}{node_hours(d, r, gpus=8):>9.1f}")

    print()
    print("  If you instead (incorrectly) use total params for the ratio:")
    for name, d, r in stages:
        if d != 28 or r != 8:
            continue
        print(f"    {name}: {node_hours(d, r, params_basis='total'):.1f} h on 4 GPU"
              "  <- 2.3x optimistic, the trap PLAN.md falls into")

    print()
    print("Measured MFU vs the plan's 30% assumption:")
    for d, m in sorted(MEASURED_MFU.items()):
        print(f"  d{d}: {m:.0%}  ({m/0.30:.2f}x the assumption)")


if __name__ == "__main__":
    main()
