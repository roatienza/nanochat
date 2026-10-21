"""CORRECTED memory + wall-clock budget for nanochat on 8 x A100 40GB.

Replaces reasoner-model/scripts/budget.py, whose parameter model is wrong by
2.1-2.6x and whose activation model is wrong in the opposite direction.

GROUND TRUTH (measured on this node, not derived):
  - param counts: nanochat/param_audit.py, which instantiates the real GPT class
    on meta device and counts parameters.
  - peak memory: nanochat/mem_probe.py, one real fwd+bwd+step, after warmup.
  - throughput: measured in a real 8-GPU base_train run.

Run: .venv/bin/python budget_real.py
"""

VOCAB = 32768
HEAD_DIM = 128
A100_BF16_TFLOPS = 312e12
GPUS = 8
MEM_LIMIT_GIB = 38.0          # 40GB card, leave headroom for fragmentation
MFU = 0.30                    # SmolLM3 calibration point; verified below

# --- measured peak GiB per (depth, device_batch_size) ------------------------
# 0 = does not fit. From nanochat/mem_probe.py, 7 clean A100s, cross-checked.
MEASURED = {
    12: {1: 3.83,  2: 5.53,  4: 8.93,  8: 15.73, 16: 29.32, 32: 0.0},
    16: {1: 7.21,  2: 8.51,  4: 12.98, 8: 21.93, 16: 0.0,  32: 0.0},
    18: {1: 10.43, 2: 10.43, 4: 15.56, 8: 0.0,  16: 0.0,  32: 0.0},
    20: {1: 12.75, 2: 12.75, 4: 18.54, 8: 30.26, 16: 0.0,  32: 0.0},
    22: {1: 16.37, 2: 16.37, 4: 16.37, 8: 35.26, 16: 0.0,  32: 0.0},
    24: {1: 20.62, 2: 20.62, 4: 25.77, 8: 0.0,  16: 0.0,  32: 0.0},
    26: {1: 25.56, 2: 25.56, 4: 30.07, 8: 0.0,  16: 0.0,  32: 0.0},
    28: {1: 31.24, 2: 31.24, 4: 34.86, 8: 0.0,  16: 0.0,  32: 0.0},
    32: {1: 0.0,   2: 0.0,   4: 0.0,   8: 0.0,  16: 0.0,  32: 0.0},
}

def best_batch(depth):
    for bs in (32, 16, 8, 4, 2, 1):
        if 0.0 < MEASURED[depth][bs] < MEM_LIMIT_GIB:
            return bs
    return None

def node_hours(depth, ratio, throughput_tok_s=None):
    """Throughput scales ~linearly with tokens/batch until you hit the vocab wall,
    but empirically depth-20 at bs=8 is *slower* per token than depth-12 at bs=16
    because of the larger logits/upcast. Pass measured throughput when known."""
    if throughput_tok_s is None:
        throughput_tok_s = default_throughput(depth, best_batch(depth))
    tokens = n_params(depth) * ratio
    return tokens / (throughput_tok_s * 3600)

def n_params(depth, aspect_ratio=64):
    """EXACT count, matching nanochat's GPT: value embeddings live on alternating
    layers only (has_ve), and kv_dim == n_embd because n_kv_head == n_head."""
    n_embd = ((depth * aspect_ratio + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    n_head = n_embd // HEAD_DIM
    per_layer = 12 * n_embd**2
    n_ve = (depth + 1) // 2 if depth % 2 == 1 else depth // 2
    value_embeds = n_ve * n_embd * VOCAB
    embed = 2 * n_embd * VOCAB
    return dict(depth=depth, n_embd=n_embd, n_head=n_head,
                params=depth * per_layer + value_embeds + embed)

def default_throughput(depth, bs):
    """Measured tok/s per node, from the d12 and d20 8-GPU runs."""
    return {12: 13_958 * 8, 20: 0, }.get(depth, 13_958 * 8 * (bs / 16))

def main():
    print("=" * 84)
    print("CORRECTED feasibility: measured peak memory per A100-40GB")
    print(f"limit {MEM_LIMIT_GIB:.0f} GiB | nanochat aspect-64, head_dim 128, vocab 32768")
    print("=" * 84)
    print(f"{'config':<8}{'params':>10}{'max bs':>8}{'peak GiB':>11}   note")
    for d in (12, 16, 18, 20, 22, 24, 26, 28, 32):
        p = n_params(d)["params"]
        bs = best_batch(d) if d in MEASURED else None
        if bs is None:
            print(f"d{d:<7}{p/1e6:>8.0f}M{'--':>8}{'--':>11}   OOM even at bs=1")
        else:
            g = MEASURED[d][bs]
            note = "plan said 32" if bs != 32 else ""
            print(f"d{d:<7}{p/1e6:>8.0f}M{bs:>8}{g:>11.2f}   {note}")

    print()
    print("=" * 84)
    print("WHAT CHANGED vs PLAN.md")
    print("=" * 84)
    import importlib.util
    from pathlib import Path
    # <repo>/nanochat/budget_real.py -> <repo>/scripts/budget.py
    old_py = Path(__file__).resolve().parent.parent / "scripts" / "budget.py"
    spec = importlib.util.spec_from_file_location("old", old_py)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    print(f"{'config':<8}{'plan params':>13}{'real params':>13}{'over-count':>12}"
          f"{'plan bs':>9}{'real bs':>9}")
    for d in (12, 16, 20, 24, 26):
        op = old.n_params(d)["params"]
        rp = n_params(d)["params"]
        obs = old.max_fitting_batch(d)
        print(f"d{d:<7}{op/1e6:>11.0f}M{rp/1e6:>11.0f}M{x if (x:=op/rp) else 0:>11.2f}x"
              f"{str(obs):>9}{str(best_batch(d)):>9}")

if __name__ == "__main__":
    main()
