"""Unit test: _reshard_optimizer_state must produce exactly what optim.py indexes."""
import sys, torch
sys.path.insert(0, "/data/rowel/nanochat")

src = open("/data/rowel/nanochat/scripts/chat_sft.py").read()
ns = {"torch": torch, "os": __import__("os")}
start = src.index("def _infer_saved_world_size")
end = src.index("if args.load_optimizer:")
exec(src[start:end], ns)
_reshard = ns["_reshard_optimizer_state"]

E = 1792
big_rows = 32768
expected_shapes = {}
i = 0
for _ in range(16):
    expected_shapes[i] = (big_rows, E); i += 1
for s in [(28,), (28,), (1, 24), (1,), (1,)]:
    expected_shapes[i] = s; i += 1
muon_sizes = [14, 112, 28, 28]
muon_keys = []
for n in muon_sizes:
    muon_keys.append(i); expected_shapes[i] = (n, 128, 128); i += 1

full = {}
for k in range(16):
    full[k] = {"step": 100, "exp_avg": torch.zeros(big_rows, E), "exp_avg_sq": torch.zeros(big_rows, E)}
for k in range(16, 21):
    full[k] = {"step": 100, "exp_avg": torch.zeros(*expected_shapes[k]), "exp_avg_sq": torch.zeros(*expected_shapes[k])}
for k in muon_keys:
    n = expected_shapes[k][0]
    full[k] = {"momentum_buffer": torch.zeros(n, 128, 128),
               "second_momentum_buffer": torch.zeros(n, 128, 1)}

for world_size in (1, 2, 4, 8):
    for rank in range(world_size):
        got = _reshard(full, rank, world_size, expected_shapes)
        rows = big_rows // world_size
        for k in range(16):
            assert tuple(got[k]["exp_avg"].shape) == (rows, E), (world_size, rank, k, got[k]["exp_avg"].shape)
        for k in range(16, 21):
            assert tuple(got[k]["exp_avg"].shape) == expected_shapes[k], ("small shrunk!", k)
        for k, n in zip(muon_keys, muon_sizes):
            chunk = (n + world_size - 1) // world_size
            # optim.py:384-387 allocates chunk_size rows on EVERY rank, even the tail
            # ranks that own nothing (they read num_owned=0 but the buffer must exist).
            assert tuple(got[k]["momentum_buffer"].shape) == (chunk, 128, 128), (
                "muon", world_size, rank, k, got[k]["momentum_buffer"].shape, "want", chunk)

got = _reshard(full, 1, 2, expected_shapes)
lo = big_rows // 2
for k in range(16):
    assert torch.equal(got[k]["exp_avg"], full[k]["exp_avg"][lo:lo + big_rows // 2]), k

# Muon value check: rank r must get rows [r*chunk, (r+1)*chunk) of the group.
for k, n in zip(muon_keys, muon_sizes):
    ref = torch.arange(n, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    src = {"momentum_buffer": ref, "second_momentum_buffer": ref}
    for world_size in (2, 4):
        chunk = (n + world_size - 1) // world_size
        for rank in range(world_size):
            got = _reshard({k: src}, rank, world_size, expected_shapes)
            start = rank * chunk
            keep = min(chunk, max(0, n - start))
            # real rows first, then zeros out to chunk_size (optim.py:384-387 zero-fills)
            want = torch.cat([ref[start:start + keep],
                              torch.zeros(chunk - keep, 1, 1)]) if keep < chunk else ref[start:start + keep]
            assert torch.equal(got[k]["momentum_buffer"], want), (
                "muon values", k, world_size, rank, got[k]["momentum_buffer"][:, 0, 0].tolist())

print("PASS: shapes + value-slicing correct for world_size in {1,2,4,8}")
